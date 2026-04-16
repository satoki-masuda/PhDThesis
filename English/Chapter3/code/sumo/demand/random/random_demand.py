"""Generate random SUMO trips directly on the network for toy experiments."""

vehicle_generation_end_time = 7200 # 車両の生成を終了する時間
vehicle_generation_period = 0.4 # 車両が生成される間隔。小さくすると多くの車両が生成される
binomial_param = 50  # 1ステップに最大5台の車両を生成。大きくすると多くの車両が生成される
seed = 42 # 乱数のシード

import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# SUMOの環境変数を設定
if not 'SUMO_HOME' in os.environ:
    os.environ['SUMO_HOME'] = "/opt/homebrew/opt/sumo/share/sumo" # SUMOのインストールパスを指定
# SUMOのホームディレクトリを設定
sumo_home = os.environ.get('SUMO_HOME')
if not sumo_home:
    raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

# ネットワークファイルのパス
network_dir = '../../network/data'
net_file = os.path.join(network_dir, 'output.net.xml')

# 出力するディレクトリとファイルのパス
output_dir = 'random'
trips_file = os.path.join(output_dir, 'trips.trips.xml') # 出力するトリップファイルのパス
routes_file = os.path.join(output_dir, 'routes.rou.xml') # 出力するルートファイルのパス

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
    '--route-file', routes_file, # ルートファイルの出力先を指定
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
print(f"Routes file created at {routes_file}")

# ルートファイルを解析
tree = ET.parse(routes_file)
root = tree.getroot()
# 車両数をカウント
vehicle_count = len(root.findall('vehicle'))
print(f"Number of vehicles generated: {vehicle_count}")

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
