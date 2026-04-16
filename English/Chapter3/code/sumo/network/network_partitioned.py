"""Create per-zone SUMO network files after network partitioning."""

import os
import requests
import subprocess
import pandas as pd
import folium
import osmnx as ox
import xml.etree.ElementTree as ET
import geopandas as gpd


def getData(bbox):
    """Download OSM road data within the given bounding box via Overpass."""
    # Overpass APIエンドポイント
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # クエリ：特定の範囲のデータを取得
    # way["highway"~"trunk|primary|secondary|tertiary|trunk_link|primary_link|secondary_link|tertiary_link|unclassified"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    overpass_query = f"""
    [out:xml];
    (
    way["highway"~"trunk|primary|secondary|tertiary|trunk_link|primary_link|secondary_link|tertiary_link|unclassified"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    node(w);
    relation["type"="restriction"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """
    
    # APIリクエストを送信
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    return response.text

# 現在のファイルのあるディレクトリにcd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ゾーニングのファイル
zone_file = '../../network_partitioning/zoning.csv'
# ゾーニングのデータを読み込む
df_zone = pd.read_csv(zone_file)
# ゾーン数
num_zones = len(df_zone.columns)
# 全ネットワークのgeojsonファイル
all_geojson_file = 'data/output.geojson'
# geojsonを読み込み、"properties"の"id"のedge idが各ゾーンに属しているかを判定
all_geojson = gpd.read_file(all_geojson_file)

for i in range(num_zones):
    output_dir = os.path.join('data', f'zone{i}')
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # OSMファイルのパス
    osm_file = os.path.join(output_dir, 'map.osm')
    # 出力するSUMOネットワークファイルのパス
    net_file = os.path.join(output_dir, 'output.net.xml')

    '''
    GDALサポート付きのバージョンを入れるには、仮想環境に入って以下を実行
    $ pip install -f https://sumo.dlr.de/daily/wheels/ eclipse-sumo
    その後、パスを設定（例）
    $ export SUMO_HOME = "/Users/masudasatoki/Desktop/MFD_evac/.venv/lib/python3.11/site-packages/sumo"
    $ export PROJ_LIB=/Users/masudasatoki/Desktop/MFD_evac/.venv/lib/python3.11/site-packages/pyproj/proj_dir/share/proj

    # HomebrewでインストールしたSUMOを使う場合
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo" # SUMOのインストールパスを指定
    '''
    # SUMOの環境変数を設定
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")
    else:
        print(f"SUMO_HOME: {sumo_home}")
        
    # ゾーンiのedge id 
    zone_edges = df_zone.iloc[:,i].values
    # ゾーンiのgeojsonを抽出
    zone_geojson = all_geojson[all_geojson['id'].isin(zone_edges)]
    # 各edgeのgeometryを取得し、bboxを計算
    bbox = zone_geojson.total_bounds
    # bboxを結合
    #bbox = pd.DataFrame(bbox.reshape(1, 4), columns=['minx', 'miny', 'maxx', 'maxy'])
    # ゾーンiのbbox # southern-most latitude, western-most longitude, northern-most latitude, eastern-most longitude
    zone_bbox = (bbox[1]-0.001, bbox[0]-0.001, bbox[3]+0.001, bbox[2]+0.001)

    data = getData(zone_bbox)

    # OSMデータをファイルに保存
    with open(osm_file, 'w') as f:
        f.write(data)

    print(f"OSM data saved to {osm_file}")

    # XMLツリーをパース
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # 地図を作成
    # 初期表示の中心座標を設定
    latitude, longitude = (zone_bbox[0]+zone_bbox[2])/2, (zone_bbox[1]+zone_bbox[3])/2

    mymap = folium.Map(location=[latitude, longitude], zoom_start=14)

    # 道路種別ごとの色設定
    road_colors = {
        "motorway": "red",
        "trunk": "orange",
        "primary": "blue",
        "secondary": "green",
        "tertiary": "purple",
        "residential": "gray"
    }

    # ノードを収集
    nodes = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lat, lon)
        # ノードを地図に追加 (サイズを大きく設定)
        folium.CircleMarker(location=(lat, lon), radius=0.5, opacity = 0.8, color='black', fill=True).add_to(mymap)


    # ウェイを地図に追加
    for way in root.findall('way'):
        nds = way.findall('nd')
        latlons = []
        for nd in nds:
            ref = nd.get('ref')
            if ref in nodes:
                latlons.append(nodes[ref])

        # 道路種別を取得
        highway_tag = None
        for tag in way.findall('tag'):
            if tag.get('k') == 'highway':
                highway_tag = tag.get('v')
                break

        # 色を設定
        color = road_colors.get(highway_tag, "black")  # 未指定の場合は黒

        # ウェイを地図に追加
        if latlons:
            folium.PolyLine(latlons, color=color).add_to(mymap)

    # 地図を保存して表示
    mymap.save(os.path.join(output_dir, 'osm_map.html'))
    print("Map has been saved as osm_map.html")

    # osmファイルをnet.xmlに変換するコマンド
    # `simple-projection`オプションを使用
    netconvert_cmd = [
        os.path.join(sumo_home, 'bin', 'netconvert'),
        '--osm-files', osm_file,
        '--output-file', net_file,
        '--speed-in-kmh', # 速度をkm/hで指定
        '--junctions.join', # 近接している交差点を結合する
        #'--junctions.join-dist', '20',  # 交差点を結合する最大距離を20メートルに設定. default is 10
        '--tls.guess', # OSM上に存在しない交差点もルールベースで追加する
        '--tls.guess.threshold', '39.0', # 交差点を設置する道路の制限速度の最小値
        '--tls.guess-signals', # OSM上に存在しない信号もルールベースで追加する
        '--tls.discard-simple', # 単純な交差点（単一の接続道路が1つだけの交差点）に交通信号を設置しない
        '--tls.join', # 近接している信号機を一体の構成とみなす
        '--lefthand', # 左側通行の設定
        '--proj.utm',  # ここでUTM投影法を指定
        '--geometry.remove', # 余分なジオメトリを削除
        '--ramps.guess', # Acceleration/Deceleration lanes are often not included in OSM data. This option identifies roads that likely have these additional lanes and adds them
        '--ignore-errors',  # エラーを無視
        #'--keep-edges.min-speed', '5',  # 最小速度のエッジを保持
        '--no-turnarounds'  # Uターンを防止
    ]

    # コマンドを実行
    subprocess.run(netconvert_cmd, check=True)
    print(f"SUMO network file created at {net_file}")

    '''
    # GeoJsonファイルを作成
    geojson_file = os.path.join(output_dir, 'output.geojson')
    net2geojson_cmd = [
        os.path.join(sumo_home, 'tools', 'net', 'net2geojson.py'),
        '-n', net_file,
        '-o', geojson_file,
        # '--lanes', # write lane geometries
        # '--internal' # write junction-internal edges or lanes。うまく接続しない問題
        # '--junctions', # Export junction geometries。junctionsというポリゴンが出力される
        '--junction-coordinates' # 交差点で逆走側のレーンと接続されてしまうが、、もっとも悪くないチョイス
    ]
    # コマンドを実行
    subprocess.run(net2geojson_cmd, check=True)
    print(f"GeoJson file created at {geojson_file}")
    '''
