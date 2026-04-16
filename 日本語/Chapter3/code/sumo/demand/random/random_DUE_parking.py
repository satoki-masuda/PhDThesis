# ネットワーク上でランダムに需要を生成し、その時点での最短距離のルートを計算する
# https://sumo.dlr.de/docs/Tools/Trip.html#randomtripspy
# https://sumo.dlr.de/docs/duarouter.html
import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import sumolib
from tqdm import tqdm
from geopy.distance import geodesic
import argparse


os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(prog="random_DUE_parking.py",
                                 description="Generate random trips and calculate shortest paths for evacuation with parking")
parser.add_argument('-z', '--zone', type=str, default="all", help='Zone ID')
args = parser.parse_args()

vehicle_generation_end_time = 3600 * 2#7200 # 車両の生成を終了する時間
vehicle_generation_period = 1 # 車両が生成される間隔。小さくすると多くの車両が生成される
binomial_param = 3  # 1ステップに最大5台の車両を生成。大きくすると多くの車両が生成される
seed = 42 # 乱数のシード
vehicle_generation_start_time = 0
#capacity_scaling_dict = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5}
capacity_scaling_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}

if args.zone == "all":
    capacity_scaling = 1.0
    output_dir = f'zone_all_shelter{capacity_scaling}'
    network_dir = '../../network/data'
else:
    capacity_scaling = capacity_scaling_dict[int(args.zone)]
    output_dir = f'zone{args.zone}_shelter{capacity_scaling}'
    network_dir = f'../../network/data/zone{args.zone}'
    if args.zone == "2":
        vehicle_generation_period = 0.75

net_file = os.path.join(network_dir, 'output.net.xml')
parking_dir = '../../evac_shelter'
parking_csv_path = os.path.join(parking_dir, 'evac_flood_koto_depth.csv') # 駐車場ファイルのパス
df_parking = pd.read_csv(parking_csv_path)

# SUMOの環境変数を設定
if not 'SUMO_HOME' in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo" # SUMOのインストールパスを指定
# SUMOのホームディレクトリを設定
sumo_home = os.environ.get('SUMO_HOME')
if not sumo_home:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

# 出力するディレクトリとファイルのパス
trips_file = os.path.join(output_dir, 'trips.trips.xml') # 出力するトリップファイルのパス
routes_file = os.path.join(output_dir, 'routes.rou.xml') # 出力するルートファイルのパス
parking_output_path = os.path.join(output_dir, 'shelter_list.csv') # 出力する駐車場ファイルのパス

# ディレクトリが存在しない場合は作成　存在すれば既存のファイルを削除
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    # 全削除
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

# ランダムなトリップを生成するためのコマンド
random_trips_cmd = [
    'python', os.path.join(sumo_home, 'tools', 'randomTrips.py'),
    '-n', net_file, # ネットワークファイルを指定
    '-o', trips_file, # トリップファイルの出力先を指定
    '-e', str(vehicle_generation_end_time),  # 車両の生成を終了する時間
    '--period', str(vehicle_generation_period),  # 車両が生成される間隔
    '--binomial', str(binomial_param),  # バイノミアル分布のパラメータ
    '--random', # ランダムなトリップを生成
    '--seed', str(seed),  # 乱数シード
    '--validate', # 結果を検証
]

# コマンドを実行
subprocess.run(random_trips_cmd, check=True)
print(f"Trips file created at {trips_file}")

# 最短経路を計算するためのコマンド
duarouter_cmd = [
    os.path.join(sumo_home, 'bin', 'duarouter'),
    '-n', net_file,  # ネットワークファイルを指定
    '--route-files', trips_file,  # トリップファイルを指定
    '-o', routes_file,  # ルートファイルの出力先を指定
    '--ignore-errors',  # エラーを無視
    '--begin', '0',  # シミュレーションの開始時間
    '--end', str(vehicle_generation_end_time)  # シミュレーションの終了時間
]

# コマンドを実行
subprocess.run(duarouter_cmd, check=True)
print(f"Routes file created at {routes_file}")

net = sumolib.net.readNet(net_file)
def find_nearest_edges(lat, lon, net, radius=250):
    x, y = net.convertLonLat2XY(lon, lat)
    edges = net.getNeighboringEdges(x, y, radius)  # 250メートル以内のエッジを取得
    nearest_edges = [edge[0].getID() for edge in sorted(edges, key=lambda e: e[1])]
    if edges:
        return nearest_edges
    else:
        return None

def get_lat_lon_from_edge(edge_id, net):
    edge = net.getEdge(edge_id)
    shape = edge.getShape()
    lon, lat = net.convertXY2LonLat(*shape[-1])  # エッジの終点の緯度経度を取得
    return lat, lon

def is_edge_reachable(net, start_edge, end_edge):
    try:
        route = net.getShortestPath(net.getEdge(start_edge), net.getEdge(end_edge))
        return route[0] is not None
    except Exception:
        return False


# routes.xmlファイルの読み込み
routes_tree = ET.parse(routes_file)
routes_root = routes_tree.getroot()

start_edges = list(set([route.find('route').get('edges').split(' ')[0] for route in routes_root.findall('vehicle')]))[:10] # 適当に10台分の出発エッジを取得し、そこから各避難所へ到達可能かを確認する
df_parking['edge_id'] = None
parking_set = set()
for index, row in tqdm(df_parking.iterrows(), desc="Processing parking areas", total=len(df_parking)):
    target_edges = find_nearest_edges(row['lat'], row['lon'], net, radius=250)
    if target_edges is None:
        #print(f"Failed to find edge for parking area {row['id']}")
        continue
    for edge_id in target_edges:
        if all(is_edge_reachable(net, start_edge, edge_id) for start_edge in start_edges):
            if not edge_id in parking_set:
                parking_set.add(edge_id)
                df_parking.at[index, 'edge_id'] = edge_id
                break
    if df_parking.at[index, 'edge_id'] is None:
        target_edges = find_nearest_edges(row['lat'], row['lon'], net, radius=350)
        for edge_id in target_edges:
            if all(is_edge_reachable(net, start_edge, edge_id) for start_edge in start_edges):
                if not edge_id in parking_set:
                    parking_set.add(edge_id)
                    df_parking.at[index, 'edge_id'] = edge_id
                    break
        if df_parking.at[index, 'edge_id'] is None:
            #print(f"Failed to find edge for parking area {row['id']}")
            pass
df_parking = df_parking.dropna(subset=['edge_id'])
df_parking = df_parking.loc[df_parking['capacity'] > 0]
df_parking['capacity'] = df_parking['capacity'].apply(lambda x: int(x * capacity_scaling))
df_parking.to_csv(parking_output_path, index=False, encoding='utf-8-sig')


routes_dict = {}
for vehicle in tqdm(routes_root.findall('vehicle'), desc="Updating route destinations"):
    # ルート情報を辞書に保存
    vehicle.set('id', f"{vehicle.get('id')}_evac")
    vehicle_id = vehicle.get('id')
    route = vehicle.find('route')
    edges = route.get('edges').split()
    # 最も近い駐車場リンクに目的地を置き換え
    lat, lon = get_lat_lon_from_edge(edges[-1], net)
    nearest_parking_id = df_parking.loc[df_parking.apply(lambda row: geodesic((lat, lon), (row['lat'], row['lon'])).meters, axis=1).idxmin(), 'edge_id']
    edges[-1] = nearest_parking_id
    routes_dict[vehicle_id] = (edges[0], nearest_parking_id)
    route.set('edges', ' '.join(edges))
    stop = ET.SubElement(vehicle, 'stop', parkingArea=nearest_parking_id, duration=f'{48*3600}')
routes_tree.write(routes_file, encoding='utf-8', xml_declaration=True)
print(f"Routes file created at {routes_file}")

# trips.xmlのトリップを更新
trips_tree = ET.parse(trips_file)
trips_root = trips_tree.getroot()
for trip in trips_root.findall('trip'):
    trip.set('id', f"{trip.get('id')}_evac")
    trip_id = trip.get('id')
    if trip_id in routes_dict:
        from_edge, to_edge = routes_dict[trip_id]
        trip.set('from', from_edge)
        trip.set('to', to_edge)
# 修正したトリップファイルを保存
trips_tree.write(trips_file, encoding='utf-8', xml_declaration=True)
print(f"Updated trips file saved at {trips_file}")


# 経路を計算するためのコマンド
duarouter_cmd_evac = [
    os.path.join(sumo_home, 'bin', 'duarouter'),
    '-n', net_file,
    '--route-files', trips_file,
    '-o', routes_file,
    '--ignore-errors',
    '--begin', str(vehicle_generation_start_time),
    '--end', str(vehicle_generation_end_time),
    '--repair',
    #'--repair.from',
    #'--repair.to',
]
# コマンドを実行
subprocess.run(duarouter_cmd_evac, check=True)

# route.xmlにstopを追加
routes_tree = ET.parse(routes_file)
routes_root = routes_tree.getroot()
for vehicle in routes_root.findall('vehicle'):
    stop = ET.SubElement(vehicle, 'stop', parkingArea=vehicle.find('route').get('edges').split()[-1], duration=f'{48*3600}')
routes_tree.write(routes_file, encoding='utf-8', xml_declaration=True)
print(f"Updated routes file saved at {routes_file}")





# ルートファイルを解析
tree = ET.parse(routes_file)
root = tree.getroot()
# 車両数をカウント
vehicle_count = len(root.findall('vehicle'))
print(f"Number of vehicles generated: {vehicle_count}")
print(f"Capacity of shelters in the zone: {df_parking['capacity'].sum()}")

output_dir_name = f'{output_dir}_veh{vehicle_count}_end{vehicle_generation_end_time}_period{vehicle_generation_period}'
os.rename(output_dir, output_dir_name)

# 各時間帯に生成された車両数をカウント
vehicle_count_by_time = {}
for vehicle in root.iter('vehicle'):
    time = float(vehicle.get('depart'))
    if time in vehicle_count_by_time:
        vehicle_count_by_time[time] += 1
    else:
        vehicle_count_by_time[time] = 1

df_vehicle_count_by_time = pd.DataFrame(vehicle_count_by_time.items(), columns=['time', 'vehicle_count'])
df_vehicle_count_by_time = df_vehicle_count_by_time.sort_values('time')
df_vehicle_count_by_time.to_csv(os.path.join(output_dir_name, 'vehicle_count_by_time.csv'), index=False)

plt.figure(figsize=(10, 6))
df_vehicle_count_by_time['time'].hist(bins=range(0, vehicle_generation_end_time, 600))
plt.title('All traffic')
plt.savefig(os.path.join(output_dir_name, 'vehicle_count_by_time.png'), bbox_inches='tight')

