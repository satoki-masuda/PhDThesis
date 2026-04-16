import random
from tqdm import tqdm
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import contextlib
import os
from reconf_shortest import ReconfigZDD, ReconfigBFS, ReconfigAstar, ReconfigDFS
from parameters_ndp import Parameters
import time
import psutil
import threading

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray


algorithms = [ReconfigZDD, ReconfigBFS, ReconfigAstar, ReconfigDFS]
#algorithms = [ReconfigZDD]
hamming_distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
repetitions = 10
num_cpus = repetitions + 2
timeout_seconds = 3600  # 1h
threshold_gb = 2.0  # メモリ使用量の閾値（GB）
weight_initial = pd.read_csv("../../output/NDP_normal_all/criticality/df_normal.csv")
weight_target = pd.read_csv("../../output/NDP_evac_ttt/criticality/df_evac.csv")
constraints_csv = pd.read_csv("../../output/constraint_all/constraints.csv")
output_path = "../../output/memory_calc"
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
def generate_random_graph(constraints):
    random_graph = [random.choice([0, 1]) for _ in range(num_links)]
    graph_link_indices = [(i, j) for idx, (i, j) in enumerate(link_indices) if random_graph[idx] == 1]
    if constraints:
        for constraint in constraints:
            if all(edge in graph_link_indices for edge in constraint):
                return generate_random_graph(constraints) # Recursion. 制約を満たすまで繰り返す
    return random_graph

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
def evaluate_task_with_timeout(algorithm, n, params, weight_initial, weight_target, constraints, threshold_gb):
    df = {"Hamming Distance": [], "Algorithm": [], "Time (s)": [], "Memory (GB)": [], "memo": []}
    initial_graph = generate_random_graph(constraints)
    target_graph = create_graph_with_hamming_distance(initial_graph, n, constraints)
    
    gc.collect()
    reconf = algorithm(params, initial_graph, target_graph, weight_initial, weight_target, constraints_csv, output_path=None)
    
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
            if algorithm.__name__ == "ReconfigZDD" or n <= 3:
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

        df["Hamming Distance"].append(n)
        df["Algorithm"].append(algorithm.__name__)
        df["Time (s)"].append(calculation_time)
        df["memo"].append("")
        df["Memory (GB)"].append(max_memory_usage)
    except Exception as e:
        print(f"Task failed: {e}")
        df["Hamming Distance"].append(n)
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
    reconf = ReconfigBFS(params, initial_graph, target_graph, weight_initial, weight_target, constraints_csv, output_path=None)
    reconf.make_constraints()
    constraints = reconf.constraints
    results = []
    for algorithm in algorithms:
        for n in hamming_distances:
            print(f"Running {algorithm.__name__} with Hamming Distance = {n}")
            tasks = [evaluate_task_with_timeout.remote(algorithm, n, params, weight_initial, weight_target, constraints, threshold_gb) for _ in range(repetitions)]
        
            gc.collect()
            finished, tasks = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_seconds)
            print(f"Finished {len(finished)} tasks")
            if len(finished) < len(tasks):
                for _ in range(len(tasks) - len(finished)):
                    results.append([{"Hamming Distance": n, "Algorithm": algorithm.__name__, "Time (s)": None, "Memory (GB)": None, "memo": "Timeout"}])
            else:
                results.extend(ray.get(finished))
            gc.collect()
            
            #メモリオーバーフローを防ぐため一定時間後にrayプロセスを終了
            ray.shutdown()
            ray.init(num_cpus=num_cpus)
        
    # 結果を統合
    final_results = pd.concat([pd.DataFrame(r) for r in results])

    color_dict = {"ReconfigZDD": "red", "ReconfigBFS": "orange", "ReconfigAstar": "green", "ReconfigDFS": "blue"}

    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        subset = final_results[final_results["Algorithm"] == algorithm.__name__]
        grouped = subset.groupby("Hamming Distance")["Memory (GB)"]
        means = grouped.median()
        errors = grouped.quantile([0.0, 1.0]).unstack() # 最大値と最小値をエラーバーにする
        
        plt.plot(means.index, means.values, label=algorithm.__name__, marker='o', linestyle='-', color=color_dict[algorithm.__name__])
        plt.errorbar(means.index, means.values, yerr=[means - errors[0.0], errors[1.0] - means], fmt='o', capsize=5, alpha=0.6, color=color_dict[algorithm.__name__])

    plt.xticks(hamming_distances)
    plt.xlabel("Hamming Distance")
    plt.ylabel("Memory Usage (GB)")
    plt.title("Memory Usage vs Hamming Distance (Mean and IQR)")
    plt.legend()
    plt.savefig(f"{output_path}/memory_usage.png")
    plt.show()

    # Plot Calculation Time with mean line and box-like visualization
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        subset = final_results[final_results["Algorithm"] == algorithm.__name__]
        grouped = subset.groupby("Hamming Distance")["Time (s)"]
        means = grouped.median()
        errors = grouped.quantile([0.0, 1.0]).unstack()
        
        plt.plot(means.index, means.values, label=algorithm.__name__, marker='o', linestyle='-', color=color_dict[algorithm.__name__])
        plt.errorbar(means.index, means.values, yerr=[means - errors[0.0], errors[1.0] - means], fmt='o', capsize=5, alpha=0.6, color=color_dict[algorithm.__name__])

    plt.xticks(hamming_distances)
    plt.xlabel("Hamming Distance")
    plt.ylabel("Calculation Time (s)")
    plt.title("Calculation Time vs Hamming Distance (Mean and IQR)")
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