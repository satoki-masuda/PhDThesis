import random
from tqdm import tqdm
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import contextlib
import os
from reconf_horizon import ReconfigZDD, ReconfigBFS, ReconfigZDD_MPC, ReconfigBFS_MPC
from parameters_ndp import Parameters
import time
import psutil
import threading

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray


#algorithms = [ReconfigZDD, ReconfigBFS]
algorithms = [ReconfigBFS_MPC]
depth = [2,3,4,5,6,7]
depth = [2,3]
repetitions = 5
num_cpus = 2#repetitions + 2
timeout_seconds = 300  # 1h
output_path = "../../output/memory_calc_horizon"
if os.path.exists(f'{output_path}'):
    shutil.rmtree(f'{output_path}')
os.makedirs(f'{output_path}')

# Rayの初期化時に最大CPU数を指定
ray.init(num_cpus=num_cpus)

params = Parameters()
link_indices = [(i, j) for i in range(params.num_zones) for j in range(params.num_zones) if params.adj_matrix[i, j] == 1]
num_links = len(link_indices)

class MemoryExceededError(Exception):
    pass

# Function to generate a random graph of specific format
def generate_random_graph(constraints, max_attempts=10000):
    """
    制約を満たすランダムグラフを生成する
    max_attempts: 最大試行回数
    """
    for _ in range(max_attempts):
        # ランダムにグラフを生成
        #random_graph = [random.choice([0, 1]) for _ in range(len(link_indices))]
        random_graph = random.choices([0, 1], k=len(link_indices), weights=[0.9, 0.1])
        graph_link_indices = [(i, j) for idx, (i, j) in enumerate(link_indices) if random_graph[idx] == 1]
        
        # 制約チェック
        is_valid = True
        if constraints:
            for constraint in constraints:
                if all(edge in graph_link_indices for edge in constraint):
                    is_valid = False
                    break
        
        if is_valid:
            return random_graph
            
    raise ValueError(f"Could not generate valid graph after {max_attempts} attempts")

# Function to create a graph at a specific Hamming distance from another graph
def create_graph_with_hamming_distance(base_graph, distance, constraints):
    target_graph = base_graph[:]
    flip_indices = random.sample(range(len(base_graph)), distance)
    for idx in flip_indices:
        target_graph[idx] = 1 - target_graph[idx]
    graph_link_indices = [(i, j) for idx, (i, j) in enumerate(link_indices) if target_graph[idx] == 1]
    if constraints:
        for constraint in constraints:
            if all(edge in graph_link_indices for edge in constraint):
                return create_graph_with_hamming_distance(base_graph, distance, constraints) # Recursion. 制約を満たすまで繰り返す
    return target_graph

# タスクごとに使用するCPU数を指定
@ray.remote(num_cpus=1)
def evaluate_task_with_timeout(algorithm, params, constraints, k):
    df = {"Depth": [], "Algorithm": [], "Time (s)": [], "Memory (GB)": [], "memo": []}
    initial_graph = generate_random_graph(constraints)
    if "MPC" not in algorithm.__name__:
        target_graph = create_graph_with_hamming_distance(initial_graph, 2, constraints)
    
    gc.collect()
    if "MPC" in algorithm.__name__:
        reconf = algorithm(params, initial_graph, k, constraints_csv=None, output_path=None)
    else:
        reconf = algorithm(params, initial_graph, target_graph, k, constraints_csv=None, output_path=None)
    
    stop_event = threading.Event()
    memory_log = []
    
    def monitor_memory():
        process = psutil.Process(os.getpid())
        while not stop_event.is_set():
            try:
                memory_info = process.memory_info().rss / (1024 ** 3)  # メモリ使用量 (GB)
                memory_log.append(memory_info)  # ログに追加
            except psutil.NoSuchProcess:
                break
            if algorithm.__name__ == "ReconfigZDD" or k <= 2:
                time.sleep(0.001)
            else:
                time.sleep(1)

    monitor_thread = threading.Thread(
        target=monitor_memory,daemon=True
    )
    monitor_thread.start()
    start_time = time.time()

    try:
        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull):
                reconf.reconfiguration()
        calculation_time = time.time() - start_time
        max_memory_usage = max(memory_log) if memory_log else None

        df["Depth"].append(k)
        df["Algorithm"].append(algorithm.__name__)
        df["Time (s)"].append(calculation_time)
        df["memo"].append("")
        df["Memory (GB)"].append(max_memory_usage)
    except Exception as e:
        print(f"Task failed: {e}")
        df["Depth"].append(k)
        df["Algorithm"].append(algorithm.__name__)
        df["Time (s)"].append(None)
        df["memo"].append("MemoryExceeded")
        df["Memory (GB)"].append(None)
    finally:
        stop_event.set()
        monitor_thread.join()

    return df

if __name__ == "__main__":
    # Generate constraints
    initial_graph = generate_random_graph(constraints=None)
    target_graph = create_graph_with_hamming_distance(initial_graph, 1, constraints=None)
    reconf = ReconfigBFS(params, initial_graph, target_graph, 2, constraints_csv=None, output_path=None)
    reconf.make_constraints()
    constraints = reconf.constraints
    results = []
    for algorithm in algorithms:
        for k in depth:
            print(f"Running {algorithm.__name__} with Depth = {k}")
            tasks = [evaluate_task_with_timeout.remote(algorithm, params, constraints, k) for _ in range(repetitions)]
        
            gc.collect()
            finished, tasks = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_seconds)
            print(f"Finished {len(finished)} tasks")
            if len(finished) < len(tasks):
                for _ in range(len(tasks) - len(finished)):
                    results.append([{"Depth": k, "Algorithm": algorithm.__name__, "Time (s)": None, "Memory (GB)": None, "memo": "Timeout"}])
            else:
                results.extend(ray.get(finished))
            gc.collect()
            
            #メモリオーバーフローを防ぐため一定時間後にrayプロセスを終了
            ray.shutdown()
            ray.init(num_cpus=num_cpus)
        
    # 結果を統合
    final_results = pd.concat([pd.DataFrame(r) for r in results])

    color_dict = {"ReconfigZDD": "red", "ReconfigBFS": "grey", "ReconfigZDD_MPC": "red", "ReconfigBFS_MPC": "grey"}

    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        subset = final_results[final_results["Algorithm"] == algorithm.__name__]
        grouped = subset.groupby("Depth")["Memory (GB)"]
        means = grouped.median()
        errors = grouped.quantile([0.0, 1.0]).unstack() # 最大値と最小値をエラーバーにする
        
        plt.plot(means.index, means.values, label=algorithm.__name__, marker='o', linestyle='-', color=color_dict[algorithm.__name__])
        plt.errorbar(means.index, means.values, yerr=[means - errors[0.0], errors[1.0] - means], fmt='o', capsize=5, alpha=0.6, color=color_dict[algorithm.__name__])

    plt.xticks(depth)
    plt.xlabel("Search depth")
    plt.ylabel("Memory usage (GB)")
    plt.title("Memory usage vs Search depth")
    plt.legend()
    plt.savefig(f"{output_path}/memory_usage.png")
    plt.show()

    # Plot Calculation Time with mean line and box-like visualization
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        subset = final_results[final_results["Algorithm"] == algorithm.__name__]
        grouped = subset.groupby("Depth")["Time (s)"]
        means = grouped.median()
        errors = grouped.quantile([0.0, 1.0]).unstack()
        
        plt.plot(means.index, means.values, label=algorithm.__name__, marker='o', linestyle='-', color=color_dict[algorithm.__name__])
        plt.errorbar(means.index, means.values, yerr=[means - errors[0.0], errors[1.0] - means], fmt='o', capsize=5, alpha=0.6, color=color_dict[algorithm.__name__])

    plt.xticks(depth)
    plt.xlabel("Search depth")
    plt.ylabel("Calculation time (s)")
    plt.title("Calculation time vs Search depth")
    plt.legend()
    plt.savefig(f"{output_path}/calculation_time.png")
    plt.show()
    
    
    # タイムアウトしていない（"Time (s)" が None でない）行をフィルタリング
    successful_results = final_results[final_results["Time (s)"].notnull()]
    # アルゴリズムごとに成功したタスク数を集計
    successful_counts = successful_results["Algorithm"].value_counts()
    # アルゴリズムごとの成功タスク数を表示
    print("\n=== Successful Task Counts by Algorithm ===")
    for algo_name, count in successful_counts.items():
        print(f"{algo_name}: {count} successful tasks")
    
    # final_resultsをCSVに保存
    final_results.to_csv(f"{output_path}/final_results.csv", index=False)