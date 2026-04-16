"""Run SUMO parking simulations with time-varying contraflow controls."""

# 設定
period_average = 300 # MFDを計算するための間隔
cui = True

import os
import sys
import subprocess
from collections import defaultdict
import pandas as pd
import pandas_read_xml as pdx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import lxml.etree as ET # lxmlを使用してXMLファイルを解析する
import traci
from traci.exceptions import TraCIException
# If you decide to switch to libsumo:
#import libsumo as traci
import sumolib
import platform
import distro
import copy
import argparse
import shutil
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(prog="simulation_parking_contraflow.py",
                                 description="Run SUMO simulation with parking")
parser.add_argument('-f', '--folder', type=str, required=True,
                    help='Setting folder in sumo/demand/od_koto')
parser.add_argument('-d', '--demand-scenario', type=str,required=True,
                    help='Demand scenario folder in sumo/demand/od_koto/{folder}')
parser.add_argument('--mode', choices=['baseline', 'control'], required=True,
                    help='baseline: contraflow無し / control: contraflow有り')
parser.add_argument('--optim-obj', choices=['normal', 'evac', 'noreconf', 'worst'], default=None,
                    help='Optimization objective for contraflow planning')

args = parser.parse_args()
setting = args.folder
demand_scenario = args.demand_scenario
optim_obj = args.optim_obj
mode = args.mode
demand_dir = f'demand/od_koto/{setting}/{demand_scenario}'
if optim_obj is None:
    output_dir = f'output/{setting}_{mode}_{demand_scenario}'
else:
    output_dir = f'output/{setting}_{mode}_{optim_obj}_{demand_scenario}'
network_dir = 'network/data'

def detect_os():
    """Return a coarse OS label used for SUMO environment setup."""
    os_name = platform.system()
    if os_name == 'Linux':
        if 'Ubuntu' in distro.id():
            return "Ubuntu"
        else:
            return "Other Linux"
    elif os_name == 'Darwin':
        return "macOS"
    else:
        return "Other OS"

os_type = detect_os()

if os_type == "Ubuntu":
    # SUMOのホームディレクトリを設定
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")
else:
    # SUMOのホームディレクトリを設定
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

# インプットファイルのパス
net_file = os.path.join(network_dir, 'output_contraflow.net.xml')
trips_file = os.path.join(demand_dir, 'trips.trips.xml') # トリップファイルのパス
routes_file = os.path.join(demand_dir, 'routes.rou.xml') # ルートファイルのパス
#routes_file = os.path.join(demand_dir, 'trips.trips.xml')
parking_csv = os.path.join(demand_dir, 'shelter_list.csv') # 駐車場ファイルのパス
zone_edge_file = os.path.join('../network_partitioning/zoning.csv') # ゾーンとエッジの対応表

# outputファイルのパス
additional_file = 'additional.add.xml' # 車両タイプの定義, エッジデータの出力ファイル
edge_output_file = os.path.join(output_dir, 'edge_data.xml')
parking_output_file = os.path.join(output_dir, 'parking_output.csv')
stat_output_file = os.path.join(output_dir, 'stat_data.xml')
trip_info_output_file = os.path.join(output_dir, 'tripinfo_data.xml')
vehroute_output_file = os.path.join(output_dir, 'vehroute_data.xml')

def get_simulation_start_time(xml_file):
    """Read the earliest departure time from a SUMO trip XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    min_depart = 1e10
    for trip in root.findall('trip'):
        depart_time = float(trip.get('depart', 0))
        if depart_time < min_depart:
            min_depart = depart_time
    return int(min_depart)

# シミュレーションの終了時間を取得する関数
def get_simulation_end_time(xml_file, buffer_time=1800):
    """Read the latest departure time and add a simulation buffer."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    max_depart = 0
    for trip in root.findall('trip'):
        depart_time = float(trip.get('depart', 0))
        if depart_time > max_depart:
            max_depart = depart_time
    # バッファ時間を追加
    return int(max_depart + buffer_time)  # バッファとして追加

simulation_start_time = 6 * 3600 # get_simulation_start_time(trips_file)
simulation_end_time = 15 * 3600 # 18 # get_simulation_end_time(trips_file)
print(f"Simulation time: {simulation_start_time} - {simulation_end_time}")

########################################## contraflowの読み込み #############################################
num_zones = 9
if os.path.exists("contraflow_simulation/adj_matrix.csv"):
    adj_matrix = np.loadtxt("contraflow_simulation/adj_matrix.csv", delimiter=',')
else:
    raise FileNotFoundError("adj_matrix.csv not found in contraflow_simulation/")
link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if adj_matrix[i, j] == 1]
cf_roads = pd.read_csv(os.path.join(network_dir, 'contraflow.csv'))
# key: (min_zone, max_zone) -> list of rows
cf_roads_pair = defaultdict(list)
for _, r in cf_roads.iterrows():
    z1, z2 = int(r.zone1), int(r.zone2)
    key = (min(z1, z2), max(z1, z2))
    cf_roads_pair[key].append(r)

if mode == 'control':
    contraflow_seq =  np.loadtxt(f"contraflow_simulation/{setting}_{demand_scenario}/{optim_obj}/best_sequence.csv", delimiter=',')
    T, G = contraflow_seq.shape
    assert G == len(link_indices), f"best cols={G}, links={len(link_indices)}"
    # directed link index lookup
    idx = {link: k for k, link in enumerate(link_indices)}
    # key: undirected pair -> time series desired direction (fav_from, fav_to) or None for baseline
    desired = [dict() for _ in range(T)]
    for i in range(num_zones):
        for j in range(i+1, num_zones):
            if adj_matrix[i, j] != 1:
                continue
            kij = idx[(i, j)]
            kji = idx[(j, i)]
            for t in range(T):
                a = contraflow_seq[t, kij] == 1.0
                b = contraflow_seq[t, kji] == 1.0
                if a and not b:
                    desired[t][(i, j)] = (i, j)   # i->j を優先
                elif b and not a:
                    desired[t][(i, j)] = (j, i)   # j->i を優先
                else:
                    desired[t][(i, j)] = None     # baseline

    control_dt = 3600  # best_sequenceの1行あたりの時間間隔（秒）
    control_t0 = 6 * 3600  # いつからbest_sequenceを適用するか
    begin_events = defaultdict(list)  # sim_time_sec -> [(pair, fav_from, fav_to), ...]
    end_events   = defaultdict(list)  # sim_time_sec -> [pair, ...]
    # 各無向ペアごとに状態変化点を拾う
    pairs = sorted({p for d in desired for p in d.keys()})
    for p in pairs:
        prev = None
        for t in range(T):
            cur = desired[t][p]
            if cur != prev:
                sim_time = int(control_t0 + (t + 1) * control_dt)
                # prevを終了
                if prev is not None:
                    end_events[sim_time].append(p)
                # curを開始
                if cur is not None:
                    begin_events[sim_time].append((p, cur[0], cur[1]))
                prev = cur
        # 最後がONなら終了を入れる（必要なら）
        #if prev is not None:
        #    sim_time = int(control_t0 + T * control_dt)
        #    end_events[sim_time].append(p)

lane_perm_backup = {}
def backup_lane(lid):
    """Cache the current lane permissions before modifying them."""
    if lid in lane_perm_backup:
        return
    lane_perm_backup[lid] = (traci.lane.getAllowed(lid), traci.lane.getDisallowed(lid))

def close_lane(lid):
    """Close a lane to ordinary traffic by restricting allowed vehicle classes."""
    traci.lane.setAllowed(lid, ["authority"])  # 実質閉鎖

def open_lane(lid):
    """Restore a lane to its original open state."""
    backup_lane(lid)
    restore_lane(lid) # 全許可と同義

def restore_lane(lid):
    """Restore cached lane permissions for a previously modified lane."""
    allowed, disallowed = lane_perm_backup[lid]
    traci.lane.setAllowed(lid, list(allowed))
    traci.lane.setDisallowed(lid, list(disallowed))

def lane_ids(edge_id, orig_n):
    """Return all lane ids and the extra lanes created for contraflow use."""
    edge_id = str(edge_id)
    orig_n = int(orig_n)
    all_lanes = [f"{edge_id}_{i}" for i in range(2*orig_n)]
    extra = [f"{edge_id}_{i}" for i in range(orig_n, 2*orig_n)]
    #try:
    #    nlanes = traci.edge.getLaneNumber(edge_id)
    #except TraCIException:
    #    return [], []
    #all_lanes = [f"{edge_id}_{i}" for i in range(nlanes)]
    #extra = [f"{edge_id}_{i}" for i in range(min(int(orig_n), nlanes), nlanes)]
    return all_lanes, extra

def set_baseline_pair(pair):
    """Reset one zone-pair to the baseline layout with added lanes closed."""
    # pairは (min_zone, max_zone) の形で入ってくる
    if pair not in cf_roads_pair:
        return
    for r in cf_roads_pair[pair]:
        # 追加レーンだけ閉じる（両方向）
        pos_all, pos_extra = lane_ids(r.edge_pos, r.orig_lanes_pos)
        neg_all, neg_extra = lane_ids(r.edge_neg, r.orig_lanes_neg)
        for lid in pos_all + neg_all:
            backup_lane(lid)
        for lid in pos_extra + neg_extra:
            close_lane(lid)

def apply_contraflow_pair(pair, fav_from, fav_to):
    """Open favored-direction lanes and mark the opposite direction for closure."""
    opened = set()
    closed = set()
    # fav_from->fav_to を優先（容量2倍相当）、逆を容量0
    if pair not in cf_roads_pair:
        return opened, closed
    for r in cf_roads_pair[pair]:
        z1, z2 = int(r.zone1), int(r.zone2)
        # CSVの edge_pos は z1->z2 方向
        if (fav_from, fav_to) == (z1, z2):
            fav_edge, fav_orig = str(r.edge_pos), int(r.orig_lanes_pos) # 増強
            opp_edge, opp_orig = str(r.edge_neg), int(r.orig_lanes_neg) # 閉鎖
        elif (fav_from, fav_to) == (z2, z1):
            fav_edge, fav_orig = str(r.edge_neg), int(r.orig_lanes_neg) # 増強
            opp_edge, opp_orig = str(r.edge_pos), int(r.orig_lanes_pos) # 閉鎖
        else:
            continue
        fav_all, _ = lane_ids(fav_edge, fav_orig) # 増強
        # 逆方向（opp）はsimulation中で「空なら閉じる／空じゃなければ保留」
        # 優先方向（fav）は即時全開放（追加含む）
        for lid in fav_all:
            backup_lane(lid)
            open_lane(lid)
        opened.add(fav_edge)
        closed.add(opp_edge)
        
    return opened, closed # 開放、閉鎖したレーンIDリスト

def end_contraflow_pair(pair):
    """End contraflow control for one pair and restore baseline permissions."""
    released = set()
    # 一旦復元して baseline に戻す
    if pair not in cf_roads_pair:
        return released
    touched = set()
    for r in cf_roads_pair[pair]:
        pos_all, _ = lane_ids(r.edge_pos, r.orig_lanes_pos)
        neg_all, _ = lane_ids(r.edge_neg, r.orig_lanes_neg)
        touched.update(pos_all + neg_all)
        released.add(str(r.edge_pos))
        released.add(str(r.edge_neg))
    for lid in touched:
        restore_lane(lid)
    set_baseline_pair(pair)
    return released

def dump_edge(edge_id: str, target_next: str | None = None):
    """Print detailed lane and downstream-edge state for debugging one edge."""
    n = traci.edge.getLaneNumber(edge_id)
    print(f"[dump] edge={edge_id} nlanes={n}")
    for i in range(n):
        lid = f"{edge_id}_{i}"
        allowed = traci.lane.getAllowed(lid)
        disallowed = traci.lane.getDisallowed(lid)
        links = traci.lane.getLinks(lid)  # (toLane, via, priority, open, ... )
        next_edges = []
        for lk in links:
            toLane = lk[0]
            toEdge = traci.lane.getEdgeID(toLane)
            next_edges.append(toEdge)
        ok_next = (target_next in next_edges) if target_next else None
        print(f"  {lid} allow={allowed} dis={disallowed} next={set(next_edges)} ok_next={ok_next}")

########################################## Simulation setup #############################################
def getNeighboringEdges(edge_id, net, radius=0.1):
    """Find nearby non-internal edges around both ends of an edge."""
    neighboring_edges = net.getEdge(edge_id).getShape()
    x_0, y_0 = neighboring_edges[0]
    x_last, y_last = neighboring_edges[-1]
    edges = net.getNeighboringEdges(x_0, y_0, r=radius) + net.getNeighboringEdges(x_last, y_last, r=radius)
    edge_ids = []
    for e, _ in edges:
        eid = e.getID()
        if eid and (not str(eid).startswith(":")):
            edge_ids.append(eid)
    return edge_ids
# net_fileのエッジを読み込み
tree = ET.parse(net_file)
root = tree.getroot()
xml_edges = []
# edgeタグを確認
for edge in root.findall('.//edge'):
    xml_edges.append(edge.get('id'))
# ゾーンとエッジのCSVを読み込む
zone_edges_df = pd.read_csv(zone_edge_file, header=0)

# ゾーンごとにエッジリストを作成
zone2edge = {}
for i, col in enumerate(zone_edges_df.columns):
    # xml_edgesに存在するエッジのみを抽出
    zone2edge[i] = zone_edges_df[col].apply(lambda x: x if x in xml_edges else np.nan).dropna().tolist()
# エッジとゾーンの対応
edge2zone = {}
for zone_id, edges in zone2edge.items():
    for edge_id in edges:
        edge2zone[edge_id] = zone_id

# additional.add.xml ファイルの内容を作成
df_park = pd.read_csv(parking_csv, encoding='utf-8', usecols=['name', 'address', 'lat', 'lon', 'capacity', 'edge_id'])
net = sumolib.net.readNet(net_file)
with open(additional_file, 'w') as f:
    f.write(f"""
<additional>
    <vType id="container" vClass="passenger" length="10.0" maxSpeed="20.0" color="1,0,0"/>
    <vType id="cable_car" vClass="passenger" length="15.0" maxSpeed="15.0" color="0,1,0"/>
    <vType id="subway" vClass="passenger" length="25.0" maxSpeed="30.0" color="0,0,1"/>
    <vType id="aircraft" vClass="passenger" length="50.0" maxSpeed="100.0" color="1,1,0"/>
    <vType id="wheelchair" vClass="passenger" length="1.0" maxSpeed="5.0" color="0,1,1"/>
    <vType id="scooter" vClass="passenger" length="2.0" maxSpeed="10.0" color="1,0,1"/>
    <vType id="drone" vClass="passenger" length="5.0" maxSpeed="50.0" color="0.5,0.5,0.5"/>
    <edgeData id="edgeData" file="{edge_output_file}" begin="{simulation_start_time}" end="{simulation_end_time}" freq="{period_average}"/>
""")
    
    parking_edge_ids = list()
    for index, row in df_park.iterrows():
        edge_id = row['edge_id']
        lane_id = edge_id + '_0'
        if edge_id not in parking_edge_ids:
            parking_edge_ids.append(edge_id)
            f.write(f"""
                    <parkingArea id="{edge_id}" name="{row['name']}" lane="{lane_id}" roadsideCapacity="{row['capacity']}" roadSide="right" angle="270" length="8"/>
                    """)
    
    for edge_id in parking_edge_ids:
        next_edges = getNeighboringEdges(edge_id, net, radius=20)
        next_next_edges = list()
        for next_id in next_edges:
            next_next_edges.extend(getNeighboringEdges(next_id, net, radius=20))
        connected_edges = list(set([edge_id] + next_edges + next_next_edges))
        f.write(f"""
            <rerouter id="Rerouter_{edge_id}" edges="{" ".join(connected_edges)}">
                <interval begin="{simulation_start_time}" end="{simulation_end_time}">
        """)
        for alt_edge_id in parking_edge_ids:
            if alt_edge_id != edge_id:
                f.write(f"""
                        <parkingAreaReroute id="{alt_edge_id}"/>
                """)
            else:
                f.write(f"""
                        <parkingAreaReroute id="{alt_edge_id}" visible="True"/>
                """)
        f.write(f"""
                </interval>
            </rerouter>
        """)
    f.write("""
</additional>
""")
print("Additional XML file created.")

# シミュレーションの設定ファイル（.sumocfg）の作成
sumo_config_file = f"{setting}.sumocfg"
with open(sumo_config_file, 'w') as f:
    f.write(f"""
    <configuration>
        <input>
            <net-file value="{net_file}"/>
            <route-files value="{routes_file}"/>
            <additional-files value="{additional_file}"/>
        </input>
        <time>
            <begin value="{simulation_start_time}"/>
            <end value="{simulation_end_time}"/>
            <step-length value="1.0"/>  <!-- ステップ長は1.0秒 -->
        </time>
        <processing>
            <time-to-teleport value="1800"/> <!-- 正の値を設定すると、その時間経過後にテレポートする。負の値ならテレポートを禁止。 -->
        </processing>
    </configuration>
    """)

if cui:
    def safe_rmtree(path: str, must_contain: str):
        p = Path(path).resolve()
        # 「必ず output_dir / setting 配下」であることを保証
        if must_contain not in str(p):
            raise ValueError(f"Refuse to delete suspicious path: {p}")
        if p.exists():
            shutil.rmtree(p)
    # 出力ディレクトリを作成
    base_out = Path(output_dir)
    base_out_str = str(base_out.resolve())

    # 既存を安全に削除して作り直し
    safe_rmtree(str(base_out), must_contain=str(Path(output_dir).resolve()))
    (base_out / "edge").mkdir(parents=True, exist_ok=True)
    (base_out / "mfd").mkdir(parents=True, exist_ok=True)
    (base_out / "parking").mkdir(parents=True, exist_ok=True)
    (base_out / "simulation_summary").mkdir(parents=True, exist_ok=True)
    summary_folder = str(base_out / "simulation_summary")
    
    # SUMOシミュレーションを実行するためのコマンド
    sumo_cmd = [
        os.path.join(sumo_home, 'bin', 'sumo'),
        '-c', sumo_config_file,
        '--start',  # シミュレーションを自動的に開始するオプション
        '--statistic-output', stat_output_file,
        '--tripinfo-output', trip_info_output_file,
        '--tripinfo-output.write-unfinished',
        '--vehroute-output', vehroute_output_file,
        '--vehroute-output.write-unfinished',
        '--vehroute-output.last-route',
        '--routing-algorithm', 'astar',
        "--device.rerouting.probability", "0.1",
        "--device.rerouting.period", "3600",
        "--ignore-route-errors", "true",
        "--error-log", os.path.join(output_dir, "sumo.err.log"),
        "--message-log", os.path.join(output_dir, "sumo.msg.log"),
    ]

    # SUMOシミュレーションを実行
    #subprocess.run(sumo_cmd, check=True)
    
    ########################################## Simulation 実行 #############################################  
    unit_time = 600 # 10分ごとに駐車場の状態を取得
    sample_time = 60 # MPCのMFD Dyanmicsを計算するときの1ステップの時間間隔 = 1分。駐車成功率のスケールを合わせるため
    # Traciを使って駐車場の情報を取得
    traci.start(sumo_cmd)
    #dump_edge("264448295#3", target_next="31090134") # デバッグ用
    
    initial_dest = {}
    parking_vehicles = {}
    ex_parking_vehicles = set()
    searching_vehicles = {}
    for zone_id, edges in zone2edge.items():
        df_park.loc[df_park['edge_id'].isin(edges), 'zone_id'] = zone_id
    for pair in cf_roads_pair.keys():
        set_baseline_pair(pair)
    # dump_edge("264448295#3", target_next="31090134") # デバッグ用
    
    dest_neighbors_cache = {}
    def get_dest_neighbors(dest_edge_id, r=100):
        if dest_edge_id in dest_neighbors_cache:
            return dest_neighbors_cache[dest_edge_id]
        x, y = net.getEdge(dest_edge_id).getShape()[0]
        neighbors = {e[0].getID() for e in net.getNeighboringEdges(x, y, r=r)}
        dest_neighbors_cache[dest_edge_id] = neighbors
        return neighbors
    
    pending_close_edges = set()   # まだ閉じてないが閉じたい edge
    active_closed_edges = set()   # 既に閉じた edge
    def edge_vehicle_count(edge_id: str) -> int:
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except TraCIException:
            return 0

    def try_close_edge_now(edge_id: str):
        """そのエッジ上に車が居なければ全レーン閉鎖。居れば保留。"""
        if edge_vehicle_count(edge_id) > 0:
            pending_close_edges.add(edge_id)
            return False
        # 全レーン閉鎖
        try:
            nlanes = traci.edge.getLaneNumber(edge_id)
            for i in range(nlanes):
                lid = f"{edge_id}_{i}"
                backup_lane(lid)
                close_lane(lid)
            active_closed_edges.add(edge_id)
            pending_close_edges.discard(edge_id)
            return True
        except TraCIException:
            return False

    def process_pending_closures():
        """毎step呼んで、空になったら閉じる。"""
        for eid in list(pending_close_edges):
            try_close_edge_now(eid)
            
    def safe_reroute(vid: str) -> bool:
        try:
            traci.vehicle.rerouteTraveltime(vid)
            return True
        except TraCIException:
            # フォールバック：findRoute -> setRoute
            try:
                route = traci.vehicle.getRoute(vid)
                idx = traci.vehicle.getRouteIndex(vid)
                if not route or idx < 0:
                    return False
                from_edge = route[idx]
                to_edge = route[-1]
                vtype = traci.vehicle.getTypeID(vid)
                r = traci.simulation.findRoute(from_edge, to_edge, vType=vtype)
                if r and r.edges:
                    traci.vehicle.setRoute(vid, r.edges)
                    return True
            except TraCIException:
                return False
        return False
    
    for step in range(simulation_start_time, simulation_end_time):
        # シミュレーションを1ステップ進める
        traci.simulationStep()
        process_pending_closures()
        
        if mode == 'control':
            # contraflowの適用・解除
            if step in end_events:
                for pair in end_events[step]:
                    print(f"Step {step}: End contraflow {pair}")
                    released = end_contraflow_pair(pair)
                    active_closed_edges -= released  # 解除
                    pending_close_edges -= released  # 解除 (保留中も)
            
            if step in begin_events:
                # 1) contraflow適用（oppは pending に入る）
                newly_closed_targets = set()
                for (pair, fav_from, fav_to) in begin_events[step]:
                    print(f"Step {step}: Apply contraflow {pair} zone {fav_from}->{fav_to}")
                    opened, closed = apply_contraflow_pair(pair, fav_from, fav_to)
                    print(f"  Opened edges: {opened}")
                    print(f"  Closed edges (pending): {closed}")
                    # closed は opp_edge の集合（予約対象）
                    newly_closed_targets |= closed
                    pending_close_edges |= closed
                # 2) このstepで「空なら」閉鎖を実行
                for e in list(newly_closed_targets):
                    try_close_edge_now(e)
                # 3) 影響車両だけ reroute（残りルートで判定）
                vids = traci.vehicle.getIDList()
                for vid in vids:
                    try:
                        route = traci.vehicle.getRoute(vid)
                        idx = traci.vehicle.getRouteIndex(vid)
                        if not route or idx < 0:
                            continue
                        remaining = route[idx:]  # 残り
                        if any(e in active_closed_edges for e in remaining):
                            safe_reroute(vid)
                    except TraCIException:
                        continue
        
        for veh_id in set(set(traci.vehicle.getIDList()) - ex_parking_vehicles): # 現在駐車している車が探索することはない
            if veh_id not in initial_dest:
                initial_dest[veh_id] = traci.vehicle.getRoute(veh_id)[-1]
            vehicle_position = traci.vehicle.getRoadID(veh_id)
            if not vehicle_position in edge2zone:
                continue
            
            destination_edge_id = traci.vehicle.getRoute(veh_id)[-1]
            # destination_edge_idの半径100m以内のエッジを取得
            neighboring_edges = get_dest_neighbors(destination_edge_id, r=100)
            if traci.vehicle.isStoppedParking(veh_id):
                parking_vehicles[veh_id] = edge2zone[vehicle_position]
            elif ((vehicle_position in neighboring_edges) or (initial_dest[veh_id] != destination_edge_id)):
                if veh_id not in searching_vehicles:
                    searching_vehicles[veh_id] = set()
                searching_vehicles[veh_id].add(edge2zone[vehicle_position])
        
        if step % (60*30) == 0:
            print(f"Step {step}")
        if (step+1) % unit_time == 0:
            parking_success_vehicles = set(list(parking_vehicles.keys())) - ex_parking_vehicles # 新たに駐車に成功した車両
            # 駐車場の状態
            parking_occupancies = {}
            parking_occupancies = {parking_id: traci.parkingarea.getVehicleCount(parking_id) for parking_id in traci.parkingarea.getIDList()}
            df_park['occupancy'] = df_park['edge_id'].map(parking_occupancies)
            df_park['vacancy'] = df_park['capacity'] - df_park['occupancy']
            
            # ゾーンごとの統計情報を集計
            zone_vacancy = {zone_id: df_park.loc[df_park['zone_id']==zone_id, 'vacancy'].sum() for zone_id in zone2edge.keys()}
            zone_occupancy = {zone_id: df_park.loc[df_park['zone_id']==zone_id, 'occupancy'].sum() for zone_id in zone2edge.keys()}
            zone_searching_rate = {zone_id: sum([zone_id in searching_zone for searching_zone in searching_vehicles.values()]) * (sample_time / unit_time) for zone_id in zone2edge.keys()}
            zone_parking_success_rate = {zone_id: sum([parking_vehicles[v] == zone_id for v in parking_success_vehicles]) * (sample_time / unit_time) for zone_id in zone2edge.keys()}
            zone_sum_speed = {zone_id: [traci.edge.getLastStepMeanSpeed(edge) * 3.6 for edge in edges if traci.edge.getLastStepMeanSpeed(edge) is not None] for zone_id, edges in zone2edge.items()} # m/s -> km/h
            zone_avg_speed = {zone_id: sum(zone_sum_speed[zone_id]) / len(zone_sum_speed[zone_id]) if len(zone_sum_speed[zone_id]) > 0 else 0 for zone_id in zone2edge.keys()}
            
            # ゾーンごとの統計情報をCSVに保存
            zone_statistics = []
            for zone_id in zone2edge.keys():
                zone_statistics.append({
                    'zone_id': zone_id,
                    'time': step,
                    'occupancy': zone_occupancy[zone_id],
                    'vacancy': zone_vacancy[zone_id],
                    'mean_speed': zone_avg_speed[zone_id],
                    'searching_vehicles': zone_searching_rate[zone_id],
                    'parking_success_rate': zone_parking_success_rate[zone_id]
                })
            zone_statistics_df = pd.DataFrame(zone_statistics)
            zone_statistics_df.to_csv(f"{summary_folder}/zone_statistics_summary_{step+1}.csv", index=False, encoding='utf-8-sig')
            
            # ゾーンごとの駐車場情報を集計
            zone_parking_data = []
            for zone_id in zone2edge.keys():
                zone_df_park = df_park.loc[df_park['zone_id']==zone_id]
                if not zone_df_park.empty:
                    zone_parking_data.append(zone_df_park)
            # 全てのゾーンのデータを一つにまとめてCSV出力
            zone_parking_df = pd.concat(zone_parking_data)
            zone_parking_df.to_csv(f"{output_dir}/parking/zone_parking_summary_{step+1}.csv", index=False, encoding='utf-8-sig')

            ex_parking_vehicles = copy.copy(set(list(parking_vehicles.keys())))
            searching_vehicles.clear()
            
        
    traci.close()


    def process_chunk(chunk):
        chunk = pdx.flatten(chunk)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.rename({'@begin': 'begin', '@end': 'end',
                            'edge|@id': 'edge_id',
                            'edge|@sampledSeconds': 'sampledSeconds', 'edge|@density': 'density',
                            'edge|@laneDensity': 'laneDensity', 'edge|@speed': 'speed'}, axis=1)
        chunk = chunk.iloc[:, 1:]
        

        chunk['begin'] = chunk['begin'].astype(float)
        chunk['end'] = chunk['end'].astype(float)
        chunk["sampledSeconds"] = chunk["sampledSeconds"].astype(float)
        chunk["density"] = chunk["density"].astype(float)
        chunk["laneDensity"] = chunk["laneDensity"].astype(float)
        chunk["speed"] = chunk["speed"].astype(float)
        chunk = chunk.replace(np.NaN, 0)
        chunk['begin'] = chunk['begin'].astype(int)

        return chunk

    def plot_mfd(MD, MS, MF, ACCUMULATION, PRODUCTION):
        # Build a csv file
        Macro_Features = {'time': [time_interval * t/60 for t in range(len(MD))],
                        'density': MD,
                        'speed': MS,
                        'flow': MF,
                        'accumulation': ACCUMULATION,
                        'production': PRODUCTION
                        }
        mfd_edge = pd.DataFrame(Macro_Features)
        mfd_edge.to_csv(f"{output_dir}/mfd/MFD_edge.csv", index=False)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='density', y='speed', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Density (#veh/km)")
        plt.ylabel("Speed (Km/hr)")
        plt.title("V-K MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/v-k.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='accumulation', y='flow', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Accumulation (Veh)")
        plt.ylabel("Flow (Veh/hr)")
        plt.title("Q-K MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/q-k.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='accumulation', y='production', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Accumulation (Veh)")
        plt.ylabel(r"Production (Veh $\cdot$ km/hr)")
        plt.title("MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/mfd.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='speed', y='flow', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Speed (Km/hr)")
        plt.ylabel("Flow (Veh/hr)")
        plt.title("Q-V MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/q-v.png")
        plt.close()

    # calculating meandensity,meanflow,meanspeed (density=density)
    MD = []
    MS = []
    MF = []
    ACCUMULATION = []
    PRODUCTION = []
    i = 0
    context = ET.iterparse(edge_output_file, events=("end",), tag="interval")

    for event, elem in context:
        if elem.tag == 'interval':
            chunk = ET.tostring(elem, encoding='utf8').decode('utf8')
            df_chunk = pdx.read_xml(chunk, ['interval'])
            df = process_chunk(df_chunk)
            
            # calculation time interval
            bft = df.begin.iloc[0]
            eft = df.end.iloc[0]
            time_interval = int(eft - bft)

            # calculating total length of network
            # 密度0のリンクはMFDの計算に含めないので、リンク長を0とする
            length = df['sampledSeconds'] / (df['end']-df['begin']) / df['density'] # km
            df['Length'] = length.replace(np.NaN, 0).replace(np.inf, 0) # km
            _net = sum(df.Length)

            # Edie's definition
            # sampledSeconds (sum_k {tau_k}) = sum of the time spent by vehicles k within the time-space window, defined by link i and time interval [t, t + Δt]
            # link_accumulation (=l_i * K_i) : sampledSeconds / time_interval = average number of vehicles on link i during time interval [t, t + Δt]
            link_accumulation = (1/time_interval)*(df.sampledSeconds) # #veh
            # numofveh (sum_i {l_i * K_i}) : number of vehicles on the whole network during time interval [t, t + Δt]
            numofveh = sum(link_accumulation) # #veh
            # sampledSeconds * speed (sum_k {d_k}) = sum of the total distance traveled by vehicles k within the time-space window, defined by link i and time interval [t, t + Δt]
            # link_weighted_flow (=l_i * Q_i) : sampledSeconds * speed / time_interval = average flow on link i during time interval [t, t + Δt]
            link_weighted_flow = (1/time_interval) * (df.sampledSeconds * df.speed) * 3.6 # #veh*km/hr (3.6 = 3600/1000)
            # speedznumofveh (=sum_i {l_i * Q_i}) : total weighted flow on the whole network during time interval [t, t + Δt]
            speedznumofveh = sum(link_weighted_flow) # #veh*km/hr
            if numofveh > 0:
                meanspeed_ = speedznumofveh/numofveh # km/hr
            else:
                meanspeed_ = 0
            meandensity_ = numofveh / _net # #veh/km
            meanflow_ = speedznumofveh / _net # #veh/hr
            MD.append(meandensity_)
            MS.append(meanspeed_)
            MF.append(meanflow_)
            ACCUMULATION.append(numofveh)
            PRODUCTION.append(speedznumofveh)
            
            # edge data
            if i % (time_interval) == 0:
                edge_data = pd.DataFrame({'edge_id': df.edge_id,
                                        'density': (1/df.Length) * link_accumulation, # #veh/km
                                        'speed': df.speed * 3.6, # km/hr
                                        'flow': (1/df.Length) * link_weighted_flow, # #veh/hr
                                        'accumulation': link_accumulation, # #veh
                                        'link_weighted_flow': link_weighted_flow, # #veh*km/hr
                                        'length': df.Length # km
                                        })
                edge_data.to_csv(f"{output_dir}/edge/edge_data_{i}.csv", index=False)
            
        i += time_interval
        
        # Clear the processed element from memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        
    plot_mfd(MD, MS, MF, ACCUMULATION, PRODUCTION)
else:
    sumo_gui_cmd = [
        os.path.join(sumo_home, 'bin', 'sumo-gui'),
        '-c', sumo_config_file
    ]
    subprocess.run(sumo_gui_cmd, check=True)
