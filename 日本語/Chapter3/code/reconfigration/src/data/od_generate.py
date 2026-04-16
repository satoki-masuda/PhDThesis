"""Create processed OD demand tables from SUMO trip and TAZ XML files."""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import random

# 確率丸めを適用
def probabilistic_round(value):
    """Round a float to an integer while preserving its expectation in aggregate."""
    lower = np.floor(value)
    upper = np.ceil(value)
    prob = value - lower
    return np.random.choice([lower, upper], p=[1 - prob, prob])

def load_clusters(cluster_file):
    """Map each edge id to a zone id using the zoning CSV."""
    cluster_df = pd.read_csv(cluster_file)
    edge_to_zone = {}

    for zone_id, column in enumerate(cluster_df.columns):
        for edge in cluster_df[column].dropna():
            edge_to_zone[edge] = zone_id
    
    return edge_to_zone

def parse_normal_trip_file(trip_file, edge_to_zone):
    """Convert normal-period SUMO trip XML into zone-level trip records."""
    tree = ET.parse(trip_file)
    root = tree.getroot()

    data = {
        'origin': [],
        'destination': [],
        'time': [],
        'value': []
    }

    for trip in root.findall('trip'):
        from_edge = trip.get('from')
        to_edge = trip.get('to')
        depart_time = float(trip.get('depart'))

        origin_zone = edge_to_zone.get(from_edge, None)
        destination_zone = edge_to_zone.get(to_edge, None)

        if origin_zone is not None and destination_zone is not None:
            data['origin'].append(origin_zone)
            data['destination'].append(destination_zone)
            data['time'].append(depart_time)
            data['value'].append(1)

    return pd.DataFrame(data)

def parse_od_file(od_file):
    """Read evacuation OD alternatives from SUMO's ``odtrips.xml`` format."""
    tree = ET.parse(od_file)
    root = tree.getroot()
    
    data = {
        'from_taz': [],
        'to_taz': [],
        'dest_alt1': [],
        'dest_alt2': [],
        'dest_alt3': []
    }
    
    for od in root.findall('.//tazRelation'):
        from_taz = od.get('from')
        to_taz = od.get('to')
        dest_alt1 = od.get('toAlt1')
        dest_alt2 = od.get('toAlt2')
        dest_alt3 = od.get('toAlt3')
        
        data['from_taz'].append(from_taz)
        data['to_taz'].append(to_taz)
        data['dest_alt1'].append(dest_alt1)
        data['dest_alt2'].append(dest_alt2)
        data['dest_alt3'].append(dest_alt3)
        
    return pd.DataFrame(data)

def parse_taz_file(taz_file):
    """Build a mapping from TAZ ids to candidate destination edges."""
    tree = ET.parse(taz_file)
    root = tree.getroot()
    taz_to_edge = {}
    for taz in root.findall('taz'):
        taz_id = taz.get('id')
        edges = taz.get('edges').split(" ")
        taz_to_edge[taz_id] = edges
    
    taz_to_edge['999999'] = ['999999']
    
    return taz_to_edge

def parse_evac_trip_file(trip_file, od_df, taz_to_edge, edge_to_zone):
    """Convert evacuation trips into zone OD records with alternative destinations."""
    tree = ET.parse(trip_file)
    root = tree.getroot()

    data = {
        'origin': [],
        'destination': [],
        'time': [],
        'dest_alt1': [],
        'dest_alt2': [],
        'dest_alt3': [],
        'value': []
    }

    for trip in root.findall('trip'):
        from_edge = trip.get('from')
        to_edge = trip.get('to')
        depart_time = float(trip.get('depart'))
        from_taz = trip.get('fromTaz')
        to_taz = trip.get('toTaz')
        if len(od_df.loc[(od_df['from_taz'] == from_taz) & (od_df['to_taz'] == to_taz)]) > 0:
            dest_alt1, dest_alt2, dest_alt3 = od_df.loc[(od_df['from_taz'] == from_taz) & (od_df['to_taz'] == to_taz), ['dest_alt1', 'dest_alt2', 'dest_alt3']].values[0]
        else:
            raise ValueError(f"OD not found: from_taz={from_taz}, to_taz={to_taz}, depart_time={depart_time}")

        origin_zone = edge_to_zone.get(from_edge, None)
        destination_zone = edge_to_zone.get(to_edge, None)
        alt_list = [destination_zone]
        
        for dest_alt in [dest_alt1, dest_alt2, dest_alt3]:
            itr = 0
            while itr < 10:
                alt = edge_to_zone.get(random.choice(taz_to_edge[dest_alt]), '999999')
                if alt not in alt_list:
                    break
                itr += 1
            alt_list.append(alt)
        
        alt1_zone = alt_list[1]
        alt2_zone = alt_list[2]
        alt3_zone = alt_list[3]

        if origin_zone is not None and destination_zone is not None:
            data['origin'].append(origin_zone)
            data['destination'].append(destination_zone)
            data['time'].append(depart_time)
            data['dest_alt1'].append(alt1_zone)
            data['dest_alt2'].append(alt2_zone)
            data['dest_alt3'].append(alt3_zone)
            data['value'].append(1)

    return pd.DataFrame(data)

def create_evac_od_matrix(df):
    """Aggregate evacuation trip records into the processed OD CSV layout."""
    # 30分の枠で需要を均等に配分（丸め誤差に対応するため）
    for ts in range(0, 48 * 3600, 30 * 60):
        condition = (df['time'] >= ts) & (df['time'] < ts + 30 * 60)
        df.loc[condition, 'time'] = np.random.uniform(ts, ts + 30 * 60, condition.sum())
    
    # はじめの12時間は避難需要0とする
    condition = df['time'] < 12 * 3600
    choice_sets = df.loc[(df['time'] >= 12 * 3600) & (df['time'] < 24 * 3600), 'time'].values
    #df.loc[condition, 'time'] = np.random.uniform(12 * 3600, 24 * 3600, condition.sum())
    df.loc[condition, 'time'] = np.random.choice(choice_sets, condition.sum())
    
    df = df.groupby(['time', 'origin', 'destination', 'dest_alt1', 'dest_alt2', 'dest_alt3'], as_index=False)['value'].sum()
    # origin, destination, time, valueの順番に列を並び替え
    df = df[['origin', 'destination', 'time', 'dest_alt1', 'dest_alt2', 'dest_alt3', 'value']]
    return df

def create_normal_od_matrix(df):
    """Aggregate normal-period trip records into the processed OD CSV layout."""
    # 30分の枠で需要を均等に配分（丸め誤差に対応するため）
    for ts in range(0, 48 * 3600, 30 * 60):
        condition = (df['time'] >= ts) & (df['time'] < ts + 30 * 60)
        df.loc[condition, 'time'] = np.random.uniform(ts, ts + 30 * 60, condition.sum())
        
    df = df.groupby(['time', 'origin', 'destination'], as_index=False)['value'].sum()
    # origin, destination, time, valueの順番に列を並び替え
    df = df[['origin', 'destination', 'time', 'value']]
    return df

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cluster_file = '../../data/raw/zoning.csv'
    evac_trip_dir = '../../../sumo/demand/od_koto/0.0normal_1.0evac'
    normal_trip_dir = '../../../sumo/demand/od_koto/1.0normal_0.0evac'
    output_dir = '../../data/processed/demand_36h'
    evac_folders = [f for f in os.listdir(evac_trip_dir) if os.path.isdir(os.path.join(evac_trip_dir, f)) and f not in [exists.split('_')[0] for exists in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, exists))]]
    synthetic_population = pd.read_csv('../../../sumo/demand/od_koto/pop_synthesis/synthetic_population.csv')
    population = len(synthetic_population)
    taz_file = '../../../sumo/demand/od_koto/input/districts.taz.xml'
    taz_to_edge = parse_taz_file(taz_file)
    
    edge_to_zone = load_clusters(cluster_file)
    # 平常時のODファイルを読み込み、OD行列を作成
    normal_trip_file = os.path.join(normal_trip_dir, 'trips_normal.trips.xml')
    normal_trip_df = parse_normal_trip_file(normal_trip_file, edge_to_zone)
    normal_od = create_normal_od_matrix(normal_trip_df)
    normal_od.to_csv('../../data/processed/normal_od_all.csv', index=False)
    normal_od_original = normal_od.copy()
    
    # 災害時のODファイル
    for evac_folder in evac_folders:
        normal_od = normal_od_original.copy()
        if not os.path.exists(os.path.join(output_dir, evac_folder)):
            os.makedirs(os.path.join(output_dir, evac_folder))
        evac_trip_file = os.path.join(evac_trip_dir, evac_folder, 'trips_evac.trips.xml')
        evac_od_file = os.path.join(evac_trip_dir, evac_folder, 'od_matrix_evac.odtrips.xml')
        od_df = parse_od_file(evac_od_file)
        evac_trip_df = parse_evac_trip_file(evac_trip_file, od_df, taz_to_edge, edge_to_zone)
        evac_od = create_evac_od_matrix(evac_trip_df)
        evac_od.to_csv(os.path.join(output_dir, evac_folder, 'evac_od.csv'), index=False)
        # 災害時のODが増加するのと同じ割合で、平叙のODを減少させる
        time_interval_cum = 60
        for ts in range(0, 48 * 3600, time_interval_cum):  # 0時から48時までtime_interval刻み
            evac_sum = evac_od.loc[(evac_od['time'] < ts + time_interval_cum), 'value'].sum()
            condition = (normal_od['time'] >= ts) & (normal_od['time'] < ts + time_interval_cum)
            adjustment_factor = evac_sum / population
            normal_od.loc[condition, 'value'] = normal_od.loc[condition, 'value'].apply(
                lambda x: probabilistic_round(x * (1 - adjustment_factor)).astype(int))
            
        normal_od.to_csv(os.path.join(output_dir, evac_folder, 'normal_od.csv'), index=False)
        # 累積図を作成
        evac_count = np.histogram(evac_od['time'], bins=range(0, 48*3600+1, time_interval_cum), weights=evac_od['value'])[0]
        normal_count = np.histogram(normal_od['time'], bins=range(0, 48*3600+1, time_interval_cum), weights=normal_od['value'])[0]
        evac_cum = np.cumsum(evac_count)
        normal_cum = np.cumsum(normal_count)
        
        # グラフ作成
        fig, ax1 = plt.subplots(figsize=(12, 8))
        # ヒストグラムを左軸にプロット
        time_interval_hist = 30*60
        ax1.hist(evac_od['time'], bins=range(0, 48*3600+1, time_interval_hist), alpha=0.5, color='red', label='Evacuation (Histogram)', weights=evac_od['value'])
        ax1.hist(normal_od['time'], bins=range(0, 48*3600+1, time_interval_hist), alpha=0.5, color='blue', label='Normal (Histogram)', weights=normal_od['value'])
        ax1.set_ylim(0, 10000)
        ax1.set_xlabel('Time', fontsize=14)
        ax1.set_ylabel('Histogram Count', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='black')
        # 累積曲線を右軸にプロット
        ax2 = ax1.twinx()
        time_range = np.arange(0, 48*3600, time_interval_cum)
        line1 = ax2.plot(time_range, evac_cum, label='Evacuation (Cumulative)', color='red', linewidth=2, linestyle='--')
        line2 = ax2.plot(time_range, normal_cum, label='Normal (Cumulative)', color='blue', linewidth=2, linestyle='--')
        # 右軸のラベルと設定
        ax2.set_ylabel('Cumulative Count', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='black')
        #ax2.set_ylim(0, 1.1 * max(max(evac_cum), max(normal_cum)))
        ax2.set_ylim(0, 300000)
        fig.legend(loc="upper center", fontsize=12, ncol=2)
        plt.xticks(np.arange(0, 48*3600+1, 6*3600), np.arange(0, 48*3600+1, 6*3600) // 3600)
        plt.grid(False)
        output_path = os.path.join(output_dir, evac_folder, 'cumulative_od.png')
        plt.savefig(output_path)
        plt.close()
