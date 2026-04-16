'''
交通状態*タイムステップ数を状態として、価値反復法を用いて最適なコントラフロー戦略を求める。
逐次的な制御リンクの追加を扱う場合、状態変数の履歴も状態に追加する必要があるため、状態空間が膨大になる。
これだったら、深層強化学習を使った方が良いかもしれない。
'''

import numpy as np
import copy
from mfd_dynamics import MFD_Dynamics
from parameters_ndp import Parameters

class ContraflowValueIteration:
    def __init__(self, params, time_horizon, gamma=0.95, epsilon=1e-4):
        self.params = params
        self.time_horizon = time_horizon  # 最適制御の時間範囲
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # 収束閾値

        self.num_zones = self.params.num_zones
        self.adj_matrix = self.params.adj_matrix
        self.link_indices = [(i, j) for i in range(self.num_zones) for j in range(self.num_zones) if self.adj_matrix[i, j] == 1]
        self.num_links = len(self.link_indices)
        self.control_space = [0, 1]  # コントラフロー 0/1 の離散制御
        self.state_bins = [0.2, 0.4, 0.6, 0.8, 1.0]  # 渋滞度ベースの区間
        self.state_bins = [0.5, 1.0]
        self.num_state_bins = len(self.state_bins)  # 状態カテゴリ

        # 価値関数: 時間 × ゾーンの状態
        self.V = np.zeros((self.time_horizon, self.num_state_bins**self.num_zones))
        self.policy = np.zeros((self.time_horizon, self.num_state_bins**self.num_zones), dtype=int)

        # 初期キャパシティを保存
        self.original_capacity = copy.deepcopy(self.params.max_boundary_capacity)

    def state_to_index(self, t, accumulations):
        """ 累積台数を離散化し、状態インデックスに変換 """
        ratio = accumulations / self.params.N_jam
        state_indices = np.digitize(ratio, self.state_bins)
        zone_state_index = sum(self.num_state_bins ** i * state_indices[i] for i in range(self.num_zones))
        return t * (self.num_state_bins ** self.num_zones) + zone_state_index

    def index_to_state(self, index):
        """インデックスを (t, zone_states) に変換"""
        t = index // (self.num_state_bins ** self.num_zones)
        state = []
        index = index % (self.num_state_bins ** self.num_zones)
        for i in range(self.num_zones):
            state.append(index % self.num_state_bins)
            index //= self.num_state_bins
        
        return t, state

    def cost_function(self, accumulations, contraflow_state):
        """ MFDシミュレーションを実行し、コスト（総旅行時間 + 渋滞ペナルティ）を計算 """
        """
        sim = MFD_Dynamics(self.params)

        # コントラフローの適用
        for idx, (i, j) in enumerate(self.link_indices):
            if contraflow_state[idx] == 1:
                sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= 2 - self.params.contra_ratio

        sim.run_simulation()
        
        # コスト = 総旅行時間 + 渋滞ペナルティ
        cost = sim.ttt + sim.tet + np.sum(np.maximum(sim.N_all - self.params.N_jam, 0))

        # キャパシティを元に戻す
        sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        """
        cost = -sum(accumulations)
        return cost

    def value_iteration(self):
        """ 価値反復法の実行 """
        iteration = 0
        while True:
            V_prev = self.V.copy()
            delta = 0

            for t in range(self.time_horizon - 1, -1, -1):  # 時間を逆向きにループ
                for state_index in range(self.num_state_bins ** self.num_zones):
                    print(state_index)
                    _, state = self.index_to_state(state_index)

                    min_cost = float('inf')
                    best_action = 0
                    
                    for u in range(2**self.num_links):  # コントラフローの設定
                        contraflow_state = [(u >> i) & 1 for i in range(self.num_links)] # 整数uを各リンクのコントラフローのオン・オフ (0/1)に変換
                        
                        # シミュレーションを実行して遷移先の状態を取得
                        new_accumulations = np.random.rand(self.num_zones) * self.params.N_jam  # (仮) 実際はシミュレーションで取得
                        next_state_index = self.state_to_index(t + 1, new_accumulations) if t < self.time_horizon - 1 else state_index
                        
                        cost = self.cost_function(new_accumulations, contraflow_state)

                        # 価値関数の更新
                        total_cost = cost + self.gamma * V_prev[t + 1, next_state_index] if t < self.time_horizon - 1 else cost

                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_action = u

                    self.V[t, state_index] = min_cost
                    self.policy[t, state_index] = best_action
                    delta = max(delta, abs(V_prev[t, state_index] - self.V[t, state_index]))
            print(f"Iteration {iteration}, Delta: {delta}")
            
            iteration += 1
            if delta < self.epsilon:
                break

        print(f"Value Iteration Converged in {iteration} iterations")

    def get_optimal_policy(self):
        """ 最適なコントラフロー戦略を取得 """
        optimal_policy = []
        accumulations = np.zeros(self.num_zones)  # 初期状態
        
        for t in range(self.time_horizon):
            state_index = self.state_to_index(t, accumulations)
            best_action = self.policy[t, state_index]  # 最適な制御を取得

            # コントラフローの適用
            contraflow_state = [(best_action >> i) & 1 for i in range(self.num_links)]
            optimal_policy.append(contraflow_state.copy())

            # 新しい累積台数 (シミュレーションで取得)
            accumulations = np.random.rand(self.num_zones) * self.params.N_jam

        return optimal_policy

if __name__ == '__main__':
    params = Parameters()
    controller = ContraflowValueIteration(params, time_horizon=2)
    controller.value_iteration()
    
    optimal_policy = controller.get_optimal_policy()
    print("Optimal Contraflow Policy:")
    for t, policy in enumerate(optimal_policy):
        print(f"Time {t}: {policy}")
