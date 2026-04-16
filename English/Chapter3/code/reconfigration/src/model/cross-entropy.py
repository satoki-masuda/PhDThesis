"""Cross-entropy optimization for dynamic reconfiguration sequences."""

import copy
import time
from pandas._libs.writers import word_len
import ray
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from graphillion import setset, GraphSet
import networkx as nx
import matplotlib.pyplot as plt
import random
from mfd_dynamics import MFD_Dynamics
from parameters_ndp import Parameters
from reconf_horizon import ReconfigZDD

class CrossEntropy():
    """
        クロスエントロピー法 (CEM) により、各リンク*時間ステップの離散制御変数（0/1）の最適な遷移パターンを求める。

        Args:
            params: Parametersクラスのインスタンス
            initial_graph: 初期グラフ
            target_graph: 遷移先のグラフ
            time_steps: 制御ホライゾンの時間ステップ数
            population_size: 1世代あたりの候補数
            elite_frac: エリートサンプルとして採用する割合
            
            Returns:
                best_control_sequence: 最適と判定された制御変数の配列 (shape: [time_steps, num_links])
                best_cost: 対応するコスト
    """
    def __init__(self, params, initial_graph, target_graph, time_steps, seq_interval, reconf_start_time, reconf_end_time, objective_func, step_change_limit=1, output_path=None):
        """Initialize the optimizer, simulator, and ZDD feasibility constraints."""
        self.params = params
        self.alpha = 0.7
        self.epsilon = 1e-2
        self.sim = MFD_Dynamics(params)
        self.initial_graph = initial_graph
        self.target_graph = target_graph
        self.time_steps = time_steps
        self.seq_interval = seq_interval
        self.generation = 0
        self.original_capacity = copy.deepcopy(self.params.max_boundary_capacity)
        self.reconf_start_time = reconf_start_time
        self.reconf_end_time = reconf_end_time
        self.objective_func = objective_func
        self.step_change_limit = max(1, int(step_change_limit))
        if objective_func not in ['evac', 'normal', 'multi']:
            raise ValueError("objective_func must be one of 'evac', 'normal', or 'multi'")
        
        num_zones = self.params.num_zones
        self.link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if self.params.adj_matrix[i, j] == 1]
        self.num_links = len(self.link_indices)
        self.initial_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        self.make_graph()
        if output_path is None:
            self.output_path = f'../../output/mfd_dynamics/green_{self.params.green_split}/{self.params.demand}/{self.params.demand_variation}/back{self.params.background_ratio}/{int(self.params.simulation_start_time/60)}_{int(self.params.simulation_end_time/60)}_control{self.time_steps}_{objective_func}'
        else:
            self.output_path = str(output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(f'{self.output_path}/p_distribution'):
            os.makedirs(f'{self.output_path}/p_distribution')
        else:
            shutil.rmtree(f'{self.output_path}/p_distribution')
            os.makedirs(f'{self.output_path}/p_distribution')
        
        # 組合せ遷移の遷移過程の制約条件を表すZDDを生成する
        self.reconf = ReconfigZDD(
            self.params,
            self.initial_graph,
            self.target_graph,
            k=self.time_steps,
            constraints_csv=None,
            output_path=self.output_path+"/reconf",
            step_change_limit=self.step_change_limit,
        )
        self.zdd_constraints = self.reconf.reconfiguration()

    def make_graph(self):
        """Build the Graphillion universe shared by all control-sequence states."""
        self.G = nx.Graph()
        self.G_edges = set()
        # 各エッジを入ノードと出ノードに分けて追加
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
        # エッジリストからグラフを作成
        GraphSet.set_universe(self.G_edges)

    def array_to_zdd(self, array):
        """
        1, 2次元配列 (row: graph数, col: リンクの0/1)をZDDに変換する関数。
        """
        if len(array.shape) == 1:
            assert array.shape[0] == self.num_links
            gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[idx] == 1]]
        else:
            assert array.shape[1] == self.num_links
            gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[row, idx] == 1] for row in range(array.shape[0])]
        return GraphSet(gs)

    def zdd_to_array(self, zdd):
        """
        ZDDを2次元配列 (row: graph数, col: リンクの0/1)に変換する関数。
        """
        array = [[1 if (f"{i}_out", f"{j}_in") in g or (f"{j}_in", f"{i}_out") in g else 0 for i, j in self.link_indices] for g in list(zdd)]
        return array

    def compute_cost(sim, reconf_end_time, objective_func):
        """
        シミュレーション結果からコストを計算する関数。
        避難時のthroughput最大化
        """
        #cost = np.array(sim.throughput_list[reconf_end_time:]).sum() + np.array(sim.throughput_background_list[reconf_end_time:]).sum()
        #cost = -sim.cost_func_evac.risk_people(sim.x, sim.non_evac_Q.sum(axis=1))
        #cost = np.array(sim.throughput_background_list[:reconf_end_time]).sum()
        
        if objective_func == 'evac':
            # 災害時のTTT最小化
            cost = - sim.tet_list[-1] / 1e5
        elif objective_func == 'normal':
            # 平時のTTT最小化 → ATTだと混雑を起こしてN_pを減らすことによりATTを増やせるのでダメ
            cost = - sim.ttt_list[reconf_end_time-sim.params.simulation_start_time-1] / 1e5
        elif objective_func == 'multi':
            # evac を少し軽くして normal を強めに見る中間目的
            idx = reconf_end_time - sim.params.simulation_start_time - 1
            cost = -(
                0.15 * sim.tet_list[-1] + 0.85 * sim.ttt_list[idx]
            ) / 1e5
        
        return cost

    def initial_sampling_probability(self):
        """Create the initial Bernoulli parameter vector for sequence sampling."""
        dim = self.num_links * (self.time_steps - 1)
        p_init = np.array(
            [
                np.full(self.num_links, sum(self.target_graph) * (t + 1) / self.time_steps / self.num_links)
                for t in range(self.time_steps - 1)
            ]
        ).flatten()
        p0_idx = [t * self.num_links + i for t in range(self.time_steps - 1) for i in self.const_edge_idx()]
        p_init = np.array([0.0 if i in p0_idx else p_init[i] for i in range(dim)], dtype=float)
        return p_init

    def sample_population_from_distribution(self, population_size, p=None, batch_size=1000):
        """Sample feasible control sequences from the current Bernoulli distribution."""
        if p is None:
            p = self.initial_sampling_probability()
        p = np.asarray(p, dtype=float)

        population = np.zeros((population_size, self.num_links * (self.time_steps - 1)))
        concat_list = []
        n_sample = 0

        while n_sample < population_size:
            tmp = (np.random.rand(batch_size, self.num_links) < p[: self.num_links]).astype(int)
            tmp = self.array_to_zdd(tmp) & self.constraint_at(1)
            if len(tmp) > 0:
                concat_list.extend(self.zdd_to_array(tmp))
                n_sample += len(tmp)
        population[:, : self.num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(
            population_size, self.num_links
        )

        for t in range(1, self.time_steps - 1):
            invalid_idx = []
            group, indices = np.unique(
                population[:, ((t - 1) * self.num_links) : (t * self.num_links)],
                axis=0,
                return_inverse=True,
            )
            for idx, ind in enumerate(group):
                arr = []
                n_sample = 0
                target_count = np.sum(indices == idx)
                count = 0
                while n_sample < target_count:
                    if count > 10**2 and n_sample > 0:
                        break
                    elif count > 10**3:
                        invalid_idx.append(idx)
                        break
                    tmp = (
                        np.random.rand(batch_size, self.num_links)
                        < p[(t * self.num_links) : ((t + 1) * self.num_links)]
                    ).astype(int)
                    zdd_tmp = self.array_to_zdd(tmp) & self.constraint_at(t + 1) & self.transition_candidates(self.array_to_zdd(ind))
                    n = min(len(zdd_tmp), target_count - n_sample)
                    if n > 0:
                        arr.extend(self.zdd_to_array(zdd_tmp)[:n])
                        n_sample += n
                    count += 1

                if n_sample == target_count:
                    population[indices == idx, (t * self.num_links) : ((t + 1) * self.num_links)] = np.concatenate(
                        np.array(arr), axis=0
                    ).reshape(target_count, self.num_links)
                elif n_sample > 0:
                    arr.extend([arr[-1] for _ in range(target_count - n_sample)])
                    population[indices == idx, (t * self.num_links) : ((t + 1) * self.num_links)] = np.concatenate(
                        np.array(arr), axis=0
                    ).reshape(target_count, self.num_links)

            if len(invalid_idx) > 0:
                valid_indices = np.setdiff1d(np.arange(len(group)), invalid_idx)
                for idx in invalid_idx:
                    population[indices == idx, :] = np.tile(
                        population[indices == random.choice(valid_indices), :][0, :],
                        (np.sum(indices == idx), 1),
                    )

        population = np.hstack((population, np.tile(self.target_graph, (population_size, 1))))
        return population
    
    def const_edge_idx(self):
        const_edges = [(i, j) for i, j in self.link_indices if i in [0, 3, 7, 8] and j in [0, 3, 7, 8]]
        #const_edges += [(0, 4), (4, 0), (5, 6), (6, 5), (5, 8), (8, 5)]
        const_edge_idx = [self.link_indices.index(e) for e in const_edges]
        
        return const_edge_idx

    def constraint_at(self, level: int):
        if not isinstance(self.zdd_constraints, list) or len(self.zdd_constraints) <= level:
            raise IndexError(
                f"zdd_constraints does not contain level {level}; "
                f"available length={len(self.zdd_constraints) if isinstance(self.zdd_constraints, list) else 'N/A'}"
            )
        return self.zdd_constraints[level]

    def transition_candidates(self, state_zdd):
        """
        Feasible transition set from the current state.

        For the standard case (step_change_limit == 1), this is the strict
        one-step successor set. For larger limits, explicitly use the
        multi-step reachability closure instead of overloading ``transition``.
        """
        if self.step_change_limit <= 1:
            return self.reconf.transition(state_zdd, self.reconf.constraints)
        return self.reconf.reachable_within_k(
            state_zdd,
            self.reconf.constraints,
            step_limit=self.step_change_limit,
            include_self=True,
        )

    def cumulative_evac_demand(self):
        cumsum = self.params.Q[:, :, self.params.simulation_start_time:self.params.simulation_end_time].sum(axis=(0, 1)).cumsum()
        return float(cumsum[-1]) if len(cumsum) > 0 else 0.0

    def cumulative_normal_demand_at_reconf_end(self):
        cumsum = self.params.Q_background[:, :, self.params.simulation_start_time:self.params.simulation_end_time].sum(axis=(0, 1)).cumsum()
        idx = self.reconf_end_time - self.params.simulation_start_time - 1
        if len(cumsum) == 0 or idx < 0:
            return 0.0
        return float(cumsum[idx])

    def summarize_policy_simulation(self, sim):
        idx = self.reconf_end_time - self.params.simulation_start_time - 1
        evac_demand = self.cumulative_evac_demand()
        normal_demand = self.cumulative_normal_demand_at_reconf_end()
        tet_total = float(sim.tet_list[-1]) if len(sim.tet_list) > 0 else 0.0
        ttt_total = float(sim.ttt_list[idx]) if len(sim.ttt_list) > idx else (float(sim.ttt_list[-1]) if len(sim.ttt_list) > 0 else 0.0)
        metrics = {
            "tet_total": tet_total,
            "ttt_total": ttt_total,
            "att_evac": tet_total / evac_demand if evac_demand > 0 else np.nan,
            "att_normal": ttt_total / normal_demand if normal_demand > 0 else np.nan,
            "evac_demand": evac_demand,
            "normal_demand": normal_demand,
        }
        if self.objective_func == "evac":
            metrics["objective_total"] = tet_total
        elif self.objective_func == "normal":
            metrics["objective_total"] = ttt_total
        elif self.objective_func == "multi":
            metrics["objective_total"] = 0.15 * tet_total + 0.85 * ttt_total
        else:
            metrics["objective_total"] = np.nan
        return metrics
    
    # 評価処理を並列実行するためのヘルパー関数
    @ray.remote(num_cpus=1)
    def evaluate_candidate(individual, params, original_capacity, seq_interval, link_indices, reconf_start_time, reconf_end_time, time_steps, objective_func):
        # 各プロセスで独自のシミュレーションインスタンスを生成
        params.max_boundary_capacity = copy.deepcopy(original_capacity)
        sim = MFD_Dynamics(copy.deepcopy(params))
        control_sequence = individual.reshape((time_steps, len(link_indices)))
        sim.reset()
        for _ in range(sim.sim_start_step, sim.sim_end_step):
            if (sim.step % seq_interval == 0) and (reconf_start_time < sim.step <= reconf_end_time):
                sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
                contraflow_idx = control_sequence[(sim.step // seq_interval) - (reconf_start_time // seq_interval) - 1]
                for i, j in [link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                    sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
            sim.step_simulation()
            
        cost = CrossEntropy.compute_cost(sim, reconf_end_time, objective_func)        
        return cost
    
    def plot_p(self, p):
        # pを図示
        plt.bar(range(len(p)), p)
        # p==1のところは赤線
        plt.bar([i for i, x in enumerate(p) if x == 1], [1 for x in p if x == 1], color='r')
        plt.ylim(0, 1)
        plt.xlabel("Control variable")
        plt.ylabel("Probability") 
        plt.title(f"Generation {self.generation}")
        plt.savefig(f'{self.output_path}/p_distribution/{self.generation}.png')
        plt.close()    
    
    def cross_entropy(self, population_size, elite_frac, kappa, n_iter, max_cpu=1):
        dim = self.num_links * (self.time_steps-1)
        
        # 各制御変数について、1になる確率の初期値を0.5に設定
        p = np.full(dim, 0.1)
        p0_idx = [t * self.num_links + i for t in range(self.time_steps-1) for i in self.const_edge_idx()]
        p = [0.0 if i in p0_idx else p[i] for i in range(dim)]
        self.plot_p(p)
        p_list = [p]

        converged = False
        best_individual = None
        best_cost = -np.inf
        fitness_record = []
        elite_count = int(np.ceil(elite_frac * population_size))
        # 計算時間記録用の配列
        computation_times = []  # 各世代の計算時間を記録
        
        while converged == False:
            population = np.zeros((population_size, self.num_links*(self.time_steps-1)))
            costs = []
            gen_times = []
            start_time = time.time()
            concat_list = []
            n_sample = 0
            # 1段階目のサンプリング：各候補は dim 次元のバイナリベクトル
            count = 0
            while n_sample < population_size:
                # ランダムな制御変数を生成
                tmp = (np.random.rand(1000, self.num_links) < p[:self.num_links]).astype(int)
                # ZDDの制約条件を満たすかどうかを確認
                tmp = self.array_to_zdd(tmp) & self.constraint_at(1)
                if len(tmp) > 0:
                    concat_list.extend(self.zdd_to_array(tmp))
                    n_sample += len(tmp)
                count += 1            
            #population[:,:self.num_links] = np.vstack(concat_list)[:population_size]
            population[:,:self.num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0)#.reshape(population_size, self.num_links)
            gen_times.append(time.time() - start_time)
            
            for t in range(1, self.time_steps-1):
                start_time = time.time()
                # 同じ個体は一気に処理する
                group, indices = np.unique(population[:,((t-1)*self.num_links):(t*self.num_links)], axis=0, return_inverse=True)
                for idx, ind in enumerate(group):
                    arr = []
                    n_sample = 0
                    target_count = np.sum(indices == idx)
                    count = 0
                    while n_sample < target_count:
                        if count > 10:
                            print("Too many iterations and forced to break")
                            arr.append(np.array([np.zeros_like(ind) for _ in range(target_count - n_sample)]))
                            break
                        # ランダムな制御変数を生成
                        tmp = (np.random.rand(10**4, self.num_links) < p[(t*self.num_links):((t+1)*self.num_links)]).astype(int)
                        # ZDDの制約条件を満たす、かつ、現在状態から遷移可能なもののみを選択
                        zdd_tmp = self.array_to_zdd(tmp) & self.constraint_at(t + 1) & self.transition_candidates(self.array_to_zdd(ind))
                        n = min(len(zdd_tmp), target_count - n_sample)
                        if n > 0:
                            arr.extend(self.zdd_to_array(zdd_tmp)[:n])
                            n_sample += n
                        count += 1
                    population[indices == idx, (t*self.num_links):((t+1)*self.num_links)] = np.concatenate(arr, axis=0)#[:target_count]
                gen_times.append(time.time() - start_time)
            # 最終段階では、目標コントラフローに到達
            population = np.hstack((population, np.tile(self.target_graph, (population.shape[0], 1))))
            computation_times.append(gen_times)
            
            # 各候補解について評価
            costs = []
            # rayを使って並列計算を実行
            ray.init(num_cpus=max_cpu)
            tasks = [
                self.evaluate_candidate.remote(
                    individual, self.params, self.original_capacity, self.seq_interval, 
                    self.link_indices, self.reconf_start_time, self.reconf_end_time, self.time_steps, self.objective_func
                ) for individual in population
            ]
            costs = ray.get(tasks)
            ray.shutdown()
            
            gen_best_cost = np.max(costs)
            fitness_record.append(gen_best_cost)
            print(f"Generation {self.generation}: Best cost = {gen_best_cost}")
            
            if gen_best_cost > best_cost:
                best_cost = gen_best_cost
                best_individual = population[np.argmax(costs)]
                print(np.array(best_individual).reshape((self.time_steps, self.num_links)))
            
            # エリートサンプルの選定（コストが低い順に上位 elite_count を選ぶ）
            elite_indices = np.argsort(costs)[-elite_count:]
            elite_population = [population[i] for i in elite_indices]
            # 確率分布の更新：エリートサンプルにおける各制御変数が1である割合で更新
            p_new = np.mean(elite_population, axis=0)[:-self.num_links]
            p = self.alpha * p_new + (1-self.alpha) * np.array(p)
            p = [0.0 if i in p0_idx else p[i] for i in range(dim)]
            p_list.append(p)
            self.generation += 1
            self.plot_p(p)
            
            if (self.generation > kappa and np.all([np.abs(np.array(p_list[-i]) - np.array(p_list[-i-1])).max() < self.epsilon for i in range(1, kappa)])) or self.generation > n_iter:
                converged = True
        
        # 収束状況のプロット
        plt.plot(fitness_record, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Objective function")
        plt.title("CEM Optimization Progress for Control Sequence")
        plt.savefig(f'{self.output_path}/cem_progress.png')
        plt.close()
        
        # 計算時間のグラフ作成
        plt.figure(figsize=(10, 6))
        computation_times = np.array(computation_times)
        
        # 各タイムステップの計算時間をプロット
        for t in range(self.time_steps-1):
            plt.scatter(range(self.generation), computation_times[:, t], alpha=0.3, 
                    label=f'Time step {t}' if t == 0 else None)
        
        # 平均計算時間をプロット
        mean_times = computation_times.mean(axis=1)
        plt.plot(range(self.generation), mean_times, 'r-', linewidth=2, label='Mean computation time')
        
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Computation time (seconds)', fontsize=14)
        plt.title('Computation time of constraints per generation', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_path}/computation_time.png')
        plt.close()
        
        # 計算時間の統計をファイルに保存
        if self.output_path is not None:
            # 世代ごとの詳細データ
            detailed_data = pd.DataFrame(
                computation_times,
                columns=[f'timestep_{t}' for t in range(self.time_steps-1)],
                index=[f'generation_{g}' for g in range(self.generation)]
            )
            detailed_data['mean'] = detailed_data.mean(axis=1)
            detailed_data.to_csv(f'{self.output_path}/computation_time.csv')
        
        best_control_sequence = best_individual.reshape((self.time_steps, self.num_links))
        return best_control_sequence, best_cost
    
    def fully_adaptive_cross_entropy(
        self,
        elite_count,
        Nmin,
        Nmax,
        Nunit,
        d,
        max_cpu=1,
        max_sim_evals=None,
        disable_early_stop=False,
    ):
        """Run the main adaptive CEM loop and return the best sequence found."""
        assert elite_count > d, "Elite count must be greater than d"
        population_size = Nmin
        dim = self.num_links * (self.time_steps-1)
        
        # 各制御変数について、1になる確率の初期値を設定
        p_init = self.initial_sampling_probability()
        p0_idx = [t * self.num_links + i for t in range(self.time_steps-1) for i in self.const_edge_idx()]
        p = p_init.copy()
        self.plot_p(p)
        p_list = [p]

        converged = False
        best_individual = None
        fitness_record = []
        gamma_record = []   
        best_cost = -np.inf
        # 計算時間記録用の配列
        computation_times = []  # 各世代の計算時間を記録
        population_size_list = []
        history_records = []
        cumulative_candidate_evaluations = 0
        cumulative_simulation_evaluations = 0
        run_start_time = time.time()
        
        while converged == False:
            costs = []
            gen_times = []
            start_time = time.time()
            print("Sampling CEM population")
            population = self.sample_population_from_distribution(population_size, p=p, batch_size=10**3)
            gen_times.append(time.time() - start_time)
            if population_size == Nmin:
                computation_times.append(gen_times)
            
            # 各候補解について評価
            costs = []
            # rayを使って並列計算を実行
            ray.init(num_cpus=max_cpu)
            tasks = [
                self.evaluate_candidate.remote(
                    individual, self.params, self.original_capacity, self.seq_interval, 
                    self.link_indices, self.reconf_start_time, self.reconf_end_time, self.time_steps, self.objective_func
                ) for individual in population
            ]
            costs = ray.get(tasks)
            ray.shutdown()
            cumulative_candidate_evaluations += population_size
            cumulative_simulation_evaluations += population_size
            
            gen_best_cost = np.max(costs)
            gamma_hat = np.argsort(costs)[-elite_count]
            print(f"Generation {self.generation}, population {population_size}: Best cost = {gen_best_cost}")
            if gen_best_cost > best_cost:
                best_cost = gen_best_cost
                best_individual = population[np.argmax(costs)]
                print(np.array(best_individual).reshape((self.time_steps, self.num_links)))

            history_records.append(
                {
                    "generation": self.generation,
                    "population_size": population_size,
                    "candidate_evaluations": population_size,
                    "simulation_evaluations": population_size,
                    "cumulative_candidate_evaluations": cumulative_candidate_evaluations,
                    "cumulative_simulation_evaluations": cumulative_simulation_evaluations,
                    "best_generation_cost": gen_best_cost,
                    "best_so_far_cost": best_cost,
                    "elapsed_time_sec": time.time() - run_start_time,
                    "gamma_hat_index": int(gamma_hat),
                    "mean_cost": float(np.mean(costs)),
                    "std_cost": float(np.std(costs)),
                    "feasible_ratio": 1.0,
                }
            )

            if max_sim_evals is not None and cumulative_simulation_evaluations >= max_sim_evals:
                print(f"Reached max_sim_evals={max_sim_evals}; stopping CEM optimization.")
                break
            
            # エリートサンプルの選定（コストが低い順に上位 elite_count を選ぶ）
            elite_indices = np.argsort(costs)[-elite_count:]
            elite_costs = [costs[i] for i in elite_indices]
            elite_population = [population[i] for i in elite_indices]
            
            if (self.generation < d) or (gen_best_cost > fitness_record[-1] and gamma_hat > gamma_record[-1]):
                pass
            else:
                if gen_best_cost == fitness_record[-1] and np.all([elite_costs[-i] == elite_costs[-i-1] for i in range(1, d)]):
                    if not disable_early_stop:
                        print("reliable results")
                        break
                else:
                    if population_size == Nmax:
                        if np.all([population_size_list[-i]==population_size_list[-i-1] for i in range(1, d)]):
                            if not disable_early_stop:
                                print("Unreliable results")
                                break
                    else:
                        population_size += Nunit      
                        continue   
            
            # 確率分布の更新：エリートサンプルにおける各制御変数が1である割合で更新
            p_new = np.mean(elite_population, axis=0)[:-self.num_links]
            p = self.alpha * p_new + (1-self.alpha) * np.array(p)
            p = [0.0 if i in p0_idx else p[i] for i in range(dim)]
            p_list.append(p)
            self.generation += 1
            self.plot_p(p)
            gamma_record.append(gamma_hat)
            fitness_record.append(gen_best_cost)
            population_size_list.append(population_size)
            population_size = Nmin
        
        # 収束状況のプロット
        plt.plot(fitness_record, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Objective function")
        plt.title("CEM Optimization Progress for Control Sequence")
        plt.savefig(f'{self.output_path}/cem_progress.png')
        plt.close()

        if self.output_path is not None:
            pd.DataFrame(history_records).to_csv(f'{self.output_path}/optimization_history.csv', index=False)

        try:
            computation_df = pd.DataFrame(computation_times)
            if not computation_df.empty:
                computation_df.columns = [f'timestep_{t}' for t in range(computation_df.shape[1])]
            # 計算時間のグラフ作成
            plt.figure(figsize=(10, 6))
            # 各タイムステップの計算時間をプロット
            for t in range(computation_df.shape[1]):
                plt.scatter(
                    range(computation_df.shape[0]),
                    computation_df.iloc[:, t],
                    alpha=0.3,
                    label=f'Time step {t}',
                )
            # 平均計算時間をプロット
            mean_times = computation_df.mean(axis=1)
            plt.plot(range(computation_df.shape[0]), mean_times, 'r-', linewidth=2, label='Mean computation time')
            
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Computation time (seconds)', fontsize=14)
            plt.title('Computation time of constraints per generation', fontsize=16)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{self.output_path}/computation_time.png')
            plt.close()     
        except:
            print("Error in plotting computation time")
        
        try:
            # 計算時間の統計をファイルに保存
            if self.output_path is not None:
                detailed_data = pd.DataFrame(computation_times)
                if not detailed_data.empty:
                    detailed_data.columns = [f'timestep_{t}' for t in range(detailed_data.shape[1])]
                    detailed_data.index = [f'generation_{g}' for g in range(detailed_data.shape[0])]
                detailed_data['mean'] = detailed_data.mean(axis=1)
                detailed_data.to_csv(f'{self.output_path}/computation_time.csv')
        except:
            print("Error in saving computation time data")
        
        best_control_sequence = best_individual.reshape((self.time_steps, self.num_links))
        return best_control_sequence, best_cost


    def sim_best_policy(self, best_control_sequence, plot=True):
        """Replay the best sequence in the simulator and summarize the results."""
        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        sim = MFD_Dynamics(self.params, output_path=self.output_path if plot else None)
        for _ in range(sim.sim_start_step, sim.sim_end_step):
            if ((sim.step % self.seq_interval) == 0) and (self.reconf_start_time < sim.step <= self.reconf_end_time):
                # キャパシティを元に戻す
                sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                # コントラフローのリンクを取得
                contraflow_idx = best_control_sequence[(sim.step // self.seq_interval) - (self.reconf_start_time // self.seq_interval) - 1]
                #print(f"Step: {sim.step}: capacity {contraflow_idx}")
                # 個体(individual)に基づいてリンクキャパシティを調整
                for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                    # 対向車線にする
                    sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
                
            sim.step_simulation()
        if plot:
            sim.plot_accumulation()
            sim.plot_mfd()
            sim.plot_mfd_animation()
            #sim.plot_time()
            sim.plot_throughput()
            sim.plot_jam("../../data/processed/zone_polygon.geojson")
        
        best_control_sequence_graph = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if best_control_sequence[step][idx] == 1] for step in range(len(best_control_sequence))]
        metrics = self.summarize_policy_simulation(sim)
        print("TET evac with best policy: ", metrics["tet_total"])
        print("TTT normal with best policy: ", metrics["ttt_total"])
        print(f"ATT evac with best policy: ", metrics["att_evac"])
        print("ATT normal with best policy: ", metrics["att_normal"])
        if plot:
            self.reconf.draw_sequence(best_control_sequence_graph)
        return metrics
        
    def sim_no_policy(self, plot=True):
        """Evaluate the baseline case with no reconfiguration control."""
        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        output_path = f'../../output/mfd_dynamics/green_{self.params.green_split}/{self.params.demand}/{self.params.demand_variation}/back{self.params.background_ratio}/{int(self.params.simulation_start_time/60)}_{int(self.params.simulation_end_time/60)}'
        sim = MFD_Dynamics(self.params, output_path=output_path if plot else None)
        sim.run_simulation()
        if plot:
            sim.plot_accumulation()
            sim.plot_mfd()
            sim.plot_mfd_animation()
            #sim.plot_time()
            sim.plot_throughput()
            sim.plot_jam("../../data/processed/zone_polygon.geojson")
        metrics = self.summarize_policy_simulation(sim)
        print("TET evac with no policy: ", metrics["tet_total"])
        print("TTT normal with no policy: ", metrics["ttt_total"])
        print(f"ATT evac with no policy: ", metrics["att_evac"])
        print("ATT normal with no policy: ", metrics["att_normal"])
        return metrics
        
    def sim_no_reconfig(self, target_graph, plot=True):
        """Evaluate a static target-graph policy without intermediate changes."""
        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        output_path = f'../../output/mfd_dynamics/green_{self.params.green_split}/{self.params.demand}/{self.params.demand_variation}/back{self.params.background_ratio}/{int(self.params.simulation_start_time/60)}_{int(self.params.simulation_end_time/60)}_contra{self.params.contra_ratio}'
        sim = MFD_Dynamics(self.params, output_path=output_path if plot else None)
        for _ in range(sim.sim_start_step, sim.sim_end_step):
            if sim.step == self.reconf_end_time:
                print(f"Step: {sim.step}")
                sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                # 個体(individual)に基づいてリンクキャパシティを調整
                for i, j in [self.link_indices[idx] for idx in np.where(np.array(target_graph) == 1)[0]]:
                    # 対向車線にする
                    sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
                
            sim.step_simulation()
        
        if plot:
            sim.plot_accumulation()
            sim.plot_mfd()
            sim.plot_mfd_animation()
            #sim.plot_time()
            sim.plot_throughput()
            sim.plot_jam("../../data/processed/zone_polygon.geojson")
        metrics = self.summarize_policy_simulation(sim)
        print("TET evac with no reconfig: ", metrics["tet_total"])
        print("TTT normal with no reconfig: ", metrics["ttt_total"])
        print(f"ATT evac with no reconfig: ", metrics["att_evac"])
        print("ATT normal with no reconfig: ", metrics["att_normal"])
        return metrics
    
    def sim_random_policy(self, seed=42, population_size=100, plot=True, save_outputs=True):
        """Evaluate random feasible policies as a loose baseline for comparison."""
        np.random.seed(seed)
        dim = self.num_links * (self.time_steps-1)
        p_init = np.full(dim, 0.1)
        p0_idx = [t * self.num_links + i for t in range(self.time_steps-1) for i in self.const_edge_idx()]
        p_init = [0.0 if i in p0_idx else p_init[i] for i in range(dim)]
        p = p_init.copy()
        
        population = np.zeros((population_size, self.num_links*(self.time_steps-1)))
        concat_list = []
        n_sample = 0
        while n_sample < population_size:
            # ランダムな制御変数を生成
            tmp = (np.random.rand(100, self.num_links) < p[:self.num_links]).astype(int)
            # ZDDの制約条件を満たすかどうかを確認
            tmp = self.array_to_zdd(tmp) & self.constraint_at(1)
            if len(tmp) > 0:
                concat_list.extend(self.zdd_to_array(tmp))
                n_sample += len(tmp)
        population[:,:self.num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(population_size, self.num_links)
        
        for t in tqdm(range(1, self.time_steps-1)):
            group, indices = np.unique(population[:,((t-1)*self.num_links):(t*self.num_links)], axis=0, return_inverse=True)
            for idx, ind in enumerate(group):
                arr = []
                n_sample = 0
                target_count = np.sum(indices == idx)
                while n_sample < target_count:
                    tmp = (np.random.rand(10**2, self.num_links) < p[(t*self.num_links):((t+1)*self.num_links)]).astype(int)                                                
                    # ZDDの制約条件を満たす、かつ、現在状態から遷移可能なもののみを選択
                    zdd_tmp = self.array_to_zdd(tmp) & self.constraint_at(t + 1) & self.transition_candidates(self.array_to_zdd(ind))
                    n = min(len(zdd_tmp), target_count - n_sample)
                    if n > 0:
                        arr.extend(self.zdd_to_array(zdd_tmp)[:n])
                        n_sample += n

                population[indices == idx, (t*self.num_links):((t+1)*self.num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, self.num_links)
        
        # 最終段階では、目標コントラフローに到達
        population = np.hstack((population, np.tile(self.target_graph, (population_size, 1))))
        
        att_evac = []
        att_normal = []
        tet_totals = []
        ttt_totals = []
        evac_demand = self.cumulative_evac_demand()
        normal_demand = self.cumulative_normal_demand_at_reconf_end()
        worst_att = -np.inf
        for pop in tqdm(range(population_size)):
            random_sequence = population[pop,:].reshape((self.time_steps, self.num_links))
            self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
            sim = MFD_Dynamics(self.params, output_path=self.output_path if plot else None)
            for _ in range(sim.sim_start_step, sim.sim_end_step):
                if ((sim.step % self.seq_interval) == 0) and (self.reconf_start_time < sim.step <= self.reconf_end_time):
                    # キャパシティを元に戻す
                    sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                    # コントラフローのリンクを取得
                    contraflow_idx = random_sequence[(sim.step // self.seq_interval) - (self.reconf_start_time // self.seq_interval) - 1]
                    for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                        # 対向車線にする
                        sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                        sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
                sim.step_simulation()

            tet_total = float(sim.tet_list[-1])
            ttt_total = float(sim.ttt_list[self.reconf_end_time-self.params.simulation_start_time-1])
            tet_totals.append(tet_total)
            ttt_totals.append(ttt_total)
            att_evac.append(tet_total / evac_demand if evac_demand > 0 else np.nan)
            att_normal.append(ttt_total / normal_demand if normal_demand > 0 else np.nan)
            if self.objective_func == "evac":
                score = tet_total
            elif self.objective_func == "normal":
                score = ttt_total
            elif self.objective_func == "multi":
                score = 0.15 * tet_total + 0.85 * ttt_total
            else:
                raise ValueError(f"Unsupported objective_func: {self.objective_func}")

            if score > worst_att:
                worst_att = score
                worst_sequence = random_sequence.copy()
        
        print(f"ATT evac with random policy: ", f"Mean {np.array(att_evac).mean()}, Std {np.array(att_evac).std()}, Min {np.array(att_evac).min()}, Max {np.array(att_evac).max()}")
        print("ATT normal with random policy: ", f"Mean {np.array(att_normal).mean()}, Std {np.array(att_normal).std()}, Min {np.array(att_normal).min()}, Max {np.array(att_normal).max()}")
        if plot:
            worst_control_sequence_graph = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if worst_sequence[step][idx] == 1] for step in range(len(worst_sequence))]
            self.reconf.draw_sequence(worst_control_sequence_graph)
        if save_outputs:
            # csvで保存
            pd.DataFrame(worst_sequence).to_csv(f'{self.output_path}/worst_random_sequence.csv', index=False, header=False)
        return {
            "tet_total": float(np.mean(tet_totals)),
            "ttt_total": float(np.mean(ttt_totals)),
            "att_evac": float(np.nanmean(att_evac)),
            "att_normal": float(np.nanmean(att_normal)),
            "att_evac_std": float(np.nanstd(att_evac)),
            "att_normal_std": float(np.nanstd(att_normal)),
            "att_evac_min": float(np.nanmin(att_evac)),
            "att_evac_max": float(np.nanmax(att_evac)),
            "att_normal_min": float(np.nanmin(att_normal)),
            "att_normal_max": float(np.nanmax(att_normal)),
            "n_samples": population_size,
        }
        
        
if __name__ == '__main__':
    params = Parameters()
    reconf_start_time = 6 * 60
    reconf_end_time = 12 * 60
    seq_interval = 60 # min. 遷移のステップ幅
    time_steps = int((reconf_end_time - reconf_start_time) // seq_interval)
    initial_graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    ###################################### ここを設定 ######################################
    objective_func = "evac"
    target_graph =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0] # 93max_1.0
    #target_graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 93max_0.9
    #target_graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 93max_0.8
    #target_graph =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 53_0.95_1.0
    ##########################################################################################
    
    ce = CrossEntropy(params, initial_graph, target_graph, time_steps, seq_interval, reconf_start_time, reconf_end_time, objective_func, step_change_limit=1)
    '''
    #ce.sim_no_policy()
    start = time.time()
    best_sequence, best_cost = ce.fully_adaptive_cross_entropy(elite_count=10, Nmin=400, Nmax=1000, Nunit=200, d=4, max_cpu=25)
    #best_sequence, best_cost = ce.fully_adaptive_cross_entropy(elite_count=5, Nmin=10, Nmax=40, Nunit=30, d=2, max_cpu=2)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    print("Optimal Control Sequence (each row corresponds to a time step, each column to a link):")
    print(best_sequence)
    pd.DataFrame(best_sequence).to_csv(f'{ce.output_path}/best_sequence.csv', index=False, header=False)
    print(f"Best cost: {best_cost}")
    '''
    best_sequence = pd.read_csv(f'/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back1.0/0_24_control6_{objective_func}/best_sequence.csv', header=None).values
    #best_sequence = pd.read_csv(f'/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back0.9/0_24_control6_{objective_func}/best_sequence.csv', header=None).values
    #best_sequence = pd.read_csv(f'/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back0.8/0_24_control6_{objective_func}/best_sequence.csv', header=None).values
    #best_sequence = pd.read_csv(f'/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/53_0.95/back1.0/0_24_control6_{objective_func}/best_sequence.csv', header=None).values
    
    ce.sim_best_policy(best_sequence)
    
    #ce.sim_no_reconfig(target_graph)
    #ce.sim_random_policy(seed=40)
    
