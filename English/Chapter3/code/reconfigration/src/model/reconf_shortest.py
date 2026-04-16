"""Shortest-path reconfiguration search on the aggregated zone graph.

This module compares several exact / heuristic search strategies, centered on a
ZDD representation of feasible network states.
"""
import shutil
import psutil
import heapq
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc

from graphillion import setset, GraphSet
from parameters_ndp import Parameters

class ReconfigZDD():
    """Search shortest reconfiguration sequences with Graphillion/ZDD states."""

    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        """Prepare the reconfiguration graph, weights, and optional output folder."""
        self.params = params
        self.output_path = output_path
        num_zones = params.num_zones
        if output_path is not None:
            # 既にフォルダが存在する場合は中身を削除
            if os.path.exists(f'{self.output_path}'):
                shutil.rmtree(f'{self.output_path}')
            os.makedirs(f'{self.output_path}')
            os.makedirs(f'{self.output_path}/seq')
                
        self.pos = {0: (1,5.5), 1: (3,2), 2: (5.5,2), 3:(8,4), 4:(2.5,5), 5:(6,4.5), 6:(3.5, 4.25), 7:(3, 7.5), 8:(6,7)}
        
        # リンクの総数を隣接行列に基づいて計算
        self.link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if params.adj_matrix[i, j] == 1]
        self.initial_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        
        self.weight_initial = weight_initial
        self.constraints_csv = constraints_csv
        self.make_graph()
        
    def make_graph(self):
        """Construct the bipartite edge-state graph used by Graphillion."""
        self.G = nx.Graph()
        self.G_edges = set()
        # 各エッジを入ノードと出ノードに分けて追加
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
        # エッジリストからグラフを作成
        GraphSet.set_universe(self.G_edges)

        weight = {}
        if self.weight_initial is not None:
            for idx, data in self.weight_initial.iterrows():
                u, v = data['link'].split('-')
                weight[(f"{u}_out", f"{v}_in")] = data['overall']
        else:
            for u, v in self.link_indices:
                weight[(f"{u}_out", f"{v}_in")] = 1
        
        if self.output_path is not None:
            plt.figure(figsize=(10, 10))
            plt.hist(list(weight.values()), bins=100)
            plt.title("Weight distribution")
            plt.savefig(f"{self.output_path}/weight_distribution.png")
            #plt.show()
            plt.close()
        # G.edgesの重みをweightに基づいて設定
        nx.set_edge_attributes(self.G, weight, 'weight')
        self.weight = {edge: weight.get(edge, 0) if edge in weight.keys() else weight.get((edge[1], edge[0]), 0) for edge in self.G_edges}

    def make_constraints(self):
        """Build the feasible-state set by removing prohibited edge combinations."""
        #GraphSet.set_universe(self.G.edges())
        constraint = GraphSet({})
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(f"{i}_out", f"{j}_in"), (f"{j}_out", f"{i}_in")] for i, j in self.link_indices]
        for edges in opposite_edges:
            without_opposite_graph = ~GraphSet({}).including(edges)
            constraint = constraint.intersection(without_opposite_graph)
        
        # 制約条件のcsvファイルから制約条件をみたさないグラフを読み込む
        #for _, data in self.constraints_csv.iterrows():
        if self.constraints_csv is not None:
            constraints_gs = GraphSet([[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if data[f'link{idx}'] == 1] for _, data in self.constraints_csv.iterrows()])
            constraint = constraint.difference(constraints_gs) # 制約条件を満たさないグラフを削除
        
        self.constraint = constraint
        print(f"Number of search space: {len(constraint)}")
        print(f"Number of constraints: {len(GraphSet({})) - len(constraint)}")
        
    def transition(self, model, search_space, sz):
        """Enumerate successor states under the chosen reconfiguration move set."""
        if model == 'tj':
                if isinstance(search_space, GraphSet):
                    # 終端グラフszから1本の辺を削除し，1本の辺を追加して得られるグラフの集合がnext_ss
                    next_ss = sz.remove_add_some_edges()
                else:
                    next_ss = sz.remove_add_some_elements()
        elif model == 'add_remove_tj':
            if isinstance(search_space, GraphSet):
                next_ad = sz.add_some_edge()
                next_rm = sz.remove_some_edge()
                next_tj = sz.remove_add_some_edges()
                next_ss = next_ad | next_rm | next_tj
            else:
                next_ad = sz.add_some_elements()
                next_rm = sz.remove_some_elements()
                next_tj = sz.remove_add_some_elements()
                next_ss = next_ad | next_rm | next_tj
        elif model == 'add_remove':
            if isinstance(search_space, GraphSet):
                next_ad = sz.add_some_edge()
                next_rm = sz.remove_some_edge()
                next_ss = next_ad | next_rm
            else:
                next_ad = sz.add_some_elements()
                next_rm = sz.remove_some_elements()
                next_ss = next_ad | next_rm
        return next_ss

    def _get_seq(self, setset_seq, s, t, search_space, model, k, weight):
        '''
        終端状態から初期状態まで逆向きに辿って，遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_seq = [t]
        current_set = t
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            #print(i)
            if isinstance(search_space, GraphSet):
                sz = GraphSet([current_set])
            else:
                sz = setset([current_set])
            next_ss = self.transition(model, search_space, sz)
                    
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            # & は共通部分を取る演算子
            # 2step先の遷移グラフのうち、重みが最大のものを選択
            if i == 0:
                for min_graph in (setset_seq[i] & next_ss).min_iter(weight):
                    current_set = min_graph #制約グラフ集合の中を探索
                    break
            else:
                next_candidate = setset_seq[i] & next_ss
                next_next_ss = self.transition(model, search_space, next_candidate)
                next_next_candidate = setset_seq[i-1] & next_next_ss
                
                for next_next_min_graph in next_next_candidate.min_iter(weight):
                    for min_graph in (next_candidate & self.transition(model, search_space, GraphSet([next_next_min_graph]))).min_iter(weight):
                        current_set = min_graph
                        break
                    break
            reconf_seq.append(current_set)
        return reconf_seq[::-1]


    def get_reconf_seq(self, s, t, search_space, model = 'add_remove', k = 1, weight = None):
        '''
        目標グラフまでの遷移過程を取得するための関数 (初期グラフから前向きに探索)
        - s: 初期グラフ
        - t: 目標グラフ
        - search_space: 探索空間
        - model: モデル (今のところtoken jumping modelのみ対応)
        - k: 何ステップ先まで探索するか (使ってない)
        - weight: エッジの重み
        '''
        if s == t:
            return [s]
        
        if model == 'tj':
            setset_seq = [] # 初期状態から遷移できるグラフの集合
            if isinstance(search_space, GraphSet): # GraphSetオブジェクトの場合
                setset_seq.append(GraphSet([s])) # 初期状態を追加
            elif isinstance(search_space, setset): # setsetオブジェクトの場合
                setset_seq.append(setset([s]))
            else:
                raise TypeError

            # token-jumpingで同じグラフが連続して出現する (例外処理)，または終端状態が現れるまで (やりたいこと)遷移を続ける
            while len(setset_seq) <= 1 or setset_seq[-2] != setset_seq[-1]: 
                if isinstance(search_space, GraphSet):
                    # グラフから1本の辺を削除し，1本の辺を追加して得られるグラフの集合
                    next_ss = setset_seq[-1].remove_add_some_edges() 
                else:
                    next_ss = setset_seq[-1].remove_add_some_elements()

                setset_seq.append(next_ss) # next_ssを追加
                if t in next_ss: # next_ssにtが含まれていれば，_get_seqで遷移過程を取得して終了
                    return self._get_seq(setset_seq, s, t, search_space, model, k, weight)
            
        elif model == 'add_remove_tj':
            setset_seq = [] # 初期状態から遷移できるグラフの集合
            if isinstance(search_space, GraphSet): # GraphSetオブジェクトの場合
                setset_seq.append(GraphSet([s])) # 初期状態を追加
            elif isinstance(search_space, setset): # setsetオブジェクトの場合
                setset_seq.append(setset([s]))
            else:
                raise TypeError

            # 同じグラフが連続して出現する (例外処理)，または終端状態が現れるまで (やりたいこと)遷移を続ける
            while len(setset_seq) <= 1 or setset_seq[-2] != setset_seq[-1]: 
                #print(len(setset_seq), setset_seq)
                if isinstance(search_space, GraphSet):
                    # グラフから1本の辺を削除し，1本の辺を追加して得られるグラフの集合
                    next_ad = setset_seq[-1].add_some_edge()
                    next_rm = setset_seq[-1].remove_some_edge()
                    next_tj = setset_seq[-1].remove_add_some_edges()
                    next_ss = next_ad | next_rm | next_tj & search_space
                else:
                    next_ad = setset_seq[-1].add_some_edge()
                    next_rm = setset_seq[-1].remove_some_edge()
                    next_tj = setset_seq[-1].remove_add_some_edges()
                    next_ss = next_ad | next_rm | next_tj & search_space
                setset_seq.append(next_ss) # next_ssを追加
                print(len(setset_seq), len(next_ss))
                if t in next_ss: # next_ssにtが含まれていれば，_get_seqで遷移過程を取得して終了
                    print("遷移系列あり")
                    return self._get_seq(setset_seq, s, t, search_space, model, k, weight)
        
        elif model == 'add_remove':
            setset_seq = [] # 初期状態から遷移できるグラフの集合
            if isinstance(search_space, GraphSet): # GraphSetオブジェクトの場合
                setset_seq.append(GraphSet([s])) # 初期状態を追加
            elif isinstance(search_space, setset): # setsetオブジェクトの場合
                setset_seq.append(setset([s]))
            else:
                raise TypeError

            # 同じグラフが連続して出現する (例外処理)，または終端状態が現れるまで (やりたいこと)遷移を続ける
            while len(setset_seq) <= 1 or setset_seq[-2] != setset_seq[-1]: 
                #print(len(setset_seq), setset_seq)
                if isinstance(search_space, GraphSet):
                    # グラフから1本の辺を削除し，1本の辺を追加して得られるグラフの集合
                    next_ad = setset_seq[-1].add_some_edge()
                    next_rm = setset_seq[-1].remove_some_edge()
                    next_ss = next_ad | next_rm & search_space
                else:
                    next_ad = setset_seq[-1].add_some_edge()
                    next_rm = setset_seq[-1].remove_some_edge()
                    next_ss = next_ad | next_rm & search_space
                setset_seq.append(next_ss) # next_ssを追加
                print(len(setset_seq), len(next_ss))
                if t in next_ss: # next_ssにtが含まれていれば，_get_seqで遷移過程を取得して終了
                    print("遷移系列あり")
                    return self._get_seq(setset_seq, s, t, search_space, model, k, weight)
            
        return []
    
    def draw_sequence(self, reconf_seq):
        all_edges = []
        edge_labels = {}
        for edge in self.G.edges():
            u, v = edge
            in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
            out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
            all_edges.append((out_node, in_node))
            all_edges.append((in_node, out_node))
            #edge_labels[(out_node, in_node)] = round(self.weight[(u, v)], 2) if (u, v) in self.weight else round(self.weight[(v, u)], 2)
            edge_labels[(out_node, in_node)] = self.weight[(u, v)] if (u, v) in self.weight else self.weight[(v, u)]

            
        def normalize_edge_widths(edge_labels, target_average=1):
            # edge_labels の重みを二乗
            edge_labels = {edge: abs(weight ** 2) for edge, weight in edge_labels.items()}
            mean_weight = sum(edge_labels.values()) / len(edge_labels)
            scale_factor = target_average / mean_weight
            return {edge: max(1.0, min(3, weight * scale_factor)) for edge, weight in edge_labels.items()}

        # 正規化されたエッジ幅
        normalized_edge_widths = normalize_edge_widths(edge_labels)

        for i, g in enumerate(reconf_seq):
            #print(i, g)
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i in range(self.params.num_zones)])
            pre_edges = []
            post_edges = []
            
            for e in g:
                # inとついている方がinノード
                u, v = e
                in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
                out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
                if (u,v) in self.initial_link_indices or (v,u) in self.initial_link_indices:
                    pre_edges.append((out_node, in_node))
                if (u,v) in self.target_link_indices or (v,u) in self.target_link_indices:
                    post_edges.append((out_node, in_node))
            
            nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
            
            # 全エッジをweightに基づいて描画
            for edge in all_edges:
                width = normalized_edge_widths[edge] if edge in normalized_edge_widths else 1  # エッジが `edge_labels` にない場合のデフォルト太さ
                nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="lightgrey", width=width,
                                    arrowstyle="->", connectionstyle="arc3,rad=0.1", style="-")
                if edge in pre_edges:
                    nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="blue", width=width,
                                        arrowstyle="->", connectionstyle="arc3,rad=0.1")
                elif edge in post_edges:
                    nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="red", width=width,
                                        arrowstyle="->", connectionstyle="arc3,rad=0.1")
            # ノードとラベルの描画
            nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
            #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.0), label_pos=0.65)
            plt.title(f"Step {i+1}")
            # 描画表示
            plt.axis("off")
            if self.output_path is not None:
                plt.savefig(f"{self.output_path}/seq/step_{i+1}.png")
            plt.close()
    
    def draw_initial_target_graphs(self):
        all_edges = []
        for edge in self.G.edges():
            u, v = edge
            in_node = int(u.split("_")[0])
            out_node = int(v.split("_")[0])
            all_edges.append((out_node, in_node))
            all_edges.append((in_node, out_node))

        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(self.params.num_zones)])
        pre_edges = []
        post_edges = []

        for e in self.G.edges():
            # inとついている方がinノード
            u, v = e
            in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
            out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
            if (u,v) in self.initial_link_indices or (v,u) in self.initial_link_indices:
                pre_edges.append((out_node, in_node))
            if (u,v) in self.target_link_indices or (v,u) in self.target_link_indices:
                post_edges.append((out_node, in_node))
            
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        # 全エッジのデフォルトの色と太さで描画
        nx.draw_networkx_edges(graph, self.pos, edgelist=all_edges, edge_color="black", width=1,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # 選択したエッジだけ色と太さを変更
        nx.draw_networkx_edges(graph, self.pos, edgelist=pre_edges, edge_color="blue", width=3,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # ノードとラベルの描画
        nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
        plt.title(f"Normal time")
        # 描画表示
        plt.axis("off")
        if self.output_path is not None:
            plt.savefig(f"{self.output_path}/initial_graph.png")
        plt.close()

            
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        # 全エッジのデフォルトの色と太さで描画
        nx.draw_networkx_edges(graph, self.pos, edgelist=all_edges, edge_color="black", width=1,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # 選択したエッジだけ色と太さを変更
        nx.draw_networkx_edges(graph, self.pos, edgelist=post_edges, edge_color="red", width=3,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # ノードとラベルの描画
        nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
        plt.title(f"Emergency time")
        # 描画表示
        plt.axis("off")
        if self.output_path is not None:
            plt.savefig(f"{self.output_path}/target_graph.png")
        plt.close()
    
    def reconfiguration(self):
        """Run the configured search and return a representative sequence."""
        self.make_constraints()
        reconf_seq = self.get_reconf_seq(self.initial_link_indices, self.target_link_indices, self.constraint, model = 'add_remove', k = 1, weight = self.weight)
        return reconf_seq


class ReconfigZDD_1StepLookahead(ReconfigZDD):
    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
    
    def _get_seq(self, setset_seq, s, t, search_space, model, k, weight):
        '''
        終端状態から初期状態まで逆向きに辿って，遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_seq = [t]
        current_set = t
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            #print(i)
            if isinstance(search_space, GraphSet):
                sz = GraphSet([current_set])
            else:
                sz = setset([current_set])
            next_ss = self.transition(model, search_space, sz)
                    
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            # & は共通部分を取る演算子
            for min_graph in (setset_seq[i] & next_ss).max_iter(weight):
                current_set = min_graph #制約グラフ集合の中を探索
                break
            reconf_seq.append(current_set)
        return reconf_seq[::-1]

class ReconfigZDD_MaxIter(ReconfigZDD):
    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
    
    def _get_seq(self, setset_seq, s, t, search_space, model, k, weight):
        '''
        終端状態から初期状態まで逆向きに辿って，遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_seq = [t]
        current_set = t
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            #print(i)
            if isinstance(search_space, GraphSet):
                sz = GraphSet([current_set])
            else:
                sz = setset([current_set])
            next_ss = self.transition(model, search_space, sz)
                    
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            # & は共通部分を取る演算子
            # 2step先の遷移グラフのうち、重みが最大のものを選択
            if i == 0:
                for max_graph in (setset_seq[i] & next_ss).max_iter(weight):
                    current_set = max_graph #制約グラフ集合の中を探索
                    break
            else:
                next_candidate = setset_seq[i] & next_ss
                next_next_ss = self.transition(model, search_space, next_candidate)
                next_next_candidate = setset_seq[i-1] & next_next_ss
                
                for next_next_max_graph in next_next_candidate.max_iter(weight):
                    for max_graph in (next_candidate & self.transition(model, search_space, GraphSet([next_next_max_graph]))).max_iter(weight):
                        current_set = max_graph
                        break
                    break
            reconf_seq.append(current_set)
        return reconf_seq[::-1]

class ReconfigZDD_RandIter(ReconfigZDD):
    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
    
    def _get_seq(self, setset_seq, s, t, search_space, model, k, weight):
        '''
        終端状態から初期状態まで逆向きに辿って，遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_seq = [t]
        current_set = t
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            #print(i)
            if isinstance(search_space, GraphSet):
                sz = GraphSet([current_set])
            else:
                sz = setset([current_set])
            next_ss = self.transition(model, search_space, sz)
                    
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            # & は共通部分を取る演算子
            for rand_graph in (setset_seq[i] & next_ss).rand_iter():
                current_set = rand_graph #制約グラフ集合の中を探索
                break
            reconf_seq.append(current_set)
        return reconf_seq[::-1]
    

class ReconfigBFS(ReconfigZDD):
    """Breadth-first baseline on explicit networkx graph states."""

    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
        self.initial_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        #self.make_graph()
        
    def make_graph(self):
        # 入ノードと出ノードに分けずにエッジを追加
        self.G = nx.DiGraph()
        for u, v in self.link_indices:
            self.G.add_edge(u, v)
        weight = {}
        for idx, data in self.weight_initial.iterrows():
            u, v = data['link'].split('-')
            weight[(int(u), int(v))] = data['overall']

        if self.output_path is not None:
            plt.figure(figsize=(10, 10))
            plt.hist(list(weight.values()), bins=100)
            plt.title("Weight distribution")
            plt.savefig(f"{self.output_path}/weight_distribution.png")
            #plt.show()
            plt.close()
        # G.edgesの重みをweightに基づいて設定
        nx.set_edge_attributes(self.G, weight, 'weight')
        self.weight = {edge: weight.get(edge, 0) for edge in self.G.edges}
        
    def make_constraints(self):
        self.constraints = []
        for u, v in self.link_indices:
            self.constraints.append([(u, v), (v, u)])
        
        # 制約条件のcsvファイルから制約条件を読み込む
        self.constraints.extend([[(i, j) for idx, (i, j) in enumerate(self.link_indices) if data[f'link{idx}'] == 1] for _, data in self.constraints_csv.iterrows()])
        
    def transition(self, graph, edge_set):
        new_states = []
        for edge in edge_set:
            if edge in graph.edges:
                modified_graph = graph.copy()
                modified_graph.remove_edge(*edge)
                if self._satisfies_constraints(modified_graph):
                    new_states.append(modified_graph)
            else:
                modified_graph = graph.copy()
                modified_graph.add_edge(*edge)
                if self._satisfies_constraints(modified_graph):
                    new_states.append(modified_graph)
        return new_states

    def _satisfies_constraints(self, graph):
        for constraint in self.constraints:
            if all(edge in graph.edges for edge in constraint):
                return False
        return True

    def get_reconf_seq(self, initial_graph, target_graph):
        if set(initial_graph.edges) == set(target_graph.edges):
            return [initial_graph]

        visited = set()
        queue = [(initial_graph, [initial_graph])]
        
        level = 0
        branches = 0
        level_list = [0]
        while queue:
            current_graph, path = queue.pop(0)
            visited.add(str(nx.to_dict_of_dicts(current_graph)))
            next_graphs = self.transition(current_graph, self.G.edges)
            branches += len(next_graphs)
            if level == level_list[-1]:
                print(f"Level {len(level_list)}: {level_list[-1]+1}")
                level_list.append(level_list[-1] + branches)
                branches = 0
            for next_graph in next_graphs:
                if str(nx.to_dict_of_dicts(next_graph)) not in visited:
                    if set(next_graph.edges) == set(target_graph.edges):
                        print(f"Level {len(level_list)}: {level_list[-1]+branches+1}")
                        return path + [next_graph]
                    queue.append((next_graph, path + [next_graph]))
                    visited.add(str(nx.to_dict_of_dicts(next_graph))) 
            level += 1
        return []

    def draw_sequence(self, reconf_seq):
        all_edges = []
        edge_labels = {}
        for edge in self.G.edges():
            u, v = edge
            all_edges.append((u, v))
            #edge_labels[(u, v)] = round(self.weight[(u, v)], 2) if (u, v) in self.weight else round(self.weight[(v, u)], 2)
            edge_labels[(u, v)] = self.weight[(u, v)] if (u, v) in self.weight else self.weight[(v, u)]
        
        def normalize_edge_widths(edge_labels, target_average=1):
            # edge_labels の重みを二乗
            edge_labels = {edge: abs(weight ** 2) for edge, weight in edge_labels.items()}
            mean_weight = sum(edge_labels.values()) / len(edge_labels)
            scale_factor = target_average / mean_weight
            return {edge: max(1.0, min(3, weight * scale_factor)) for edge, weight in edge_labels.items()}

        # 正規化されたエッジ幅
        normalized_edge_widths = normalize_edge_widths(edge_labels)

        for i, g in enumerate(reconf_seq):
            #print(i, g)
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i in range(self.params.num_zones)])
            pre_edges = []
            post_edges = []
            
            for e in g.edges():
                u, v = e
                if (u,v) in self.initial_link_indices:
                    pre_edges.append((u, v))
                post_edges.append((u, v))
            
            nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
            
            # 全エッジをweightに基づいて描画
            for edge in all_edges:
                width = normalized_edge_widths[edge] if edge in normalized_edge_widths else 1  # エッジが `edge_labels` にない場合のデフォルト太さ
                nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="lightgrey", width=width,
                                    arrowstyle="->", connectionstyle="arc3,rad=0.1", style="-")
                if edge in pre_edges:
                    nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="blue", width=width,
                                        arrowstyle="->", connectionstyle="arc3,rad=0.1")
                elif edge in post_edges:
                    nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="red", width=width,
                                        arrowstyle="->", connectionstyle="arc3,rad=0.1")
            # ノードとラベルの描画
            nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
            #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.0), label_pos=0.65)
            plt.title(f"Step {i+1}")
            # 描画表示
            plt.axis("off")
            if self.output_path is not None:
                plt.savefig(f"{self.output_path}/seq/step_{i+1}.png")
            plt.close()
            
    def make_initial_target_graphs(self):
        initial_graph = nx.DiGraph()
        target_graph = nx.DiGraph()

        for e in self.G.edges():
            # inとついている方がinノード
            u, v = e
            if (u,v) in self.initial_link_indices:
                initial_graph.add_edge(u, v)
            if (u,v) in self.target_link_indices:
                target_graph.add_edge(u, v)
        
        return initial_graph, target_graph

    def reconfiguration(self):
        self.make_constraints()
        initial_graph, target_graph = self.make_initial_target_graphs()
        reconf_seq = self.get_reconf_seq(initial_graph, target_graph)
        return reconf_seq


class ReconfigAstar(ReconfigBFS):
    """A* baseline that adds a simple graph-difference heuristic."""

    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
        self.initial_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        
    def heuristic(self, current_graph, target_graph):
        # 目標グラフとの距離
        #return (len(set(current_graph.edges)-set(target_graph.edges)) + len(set(target_graph.edges)-set(current_graph.edges)))/2
        return max(len(set(current_graph.edges)-set(target_graph.edges)), len(set(target_graph.edges)-set(current_graph.edges)))
    
    def get_reconf_seq(self, initial_graph, target_graph):
        if set(initial_graph.edges) == set(target_graph.edges):
            return [initial_graph]

        open_set = []
        heapq.heappush(open_set, (self.heuristic(initial_graph, target_graph), id(initial_graph), initial_graph, [initial_graph]))
        closed_set = set()
        
        while open_set:
            _, _, current_graph, path = heapq.heappop(open_set)
            if str(nx.to_dict_of_dicts(current_graph)) in closed_set:
                continue
            closed_set.add(str(nx.to_dict_of_dicts(current_graph)))
            if set(current_graph.edges) == set(target_graph.edges):
                return path
            for next_graph in self.transition(current_graph, self.G.edges):
                if str(nx.to_dict_of_dicts(next_graph)) not in closed_set:
                    heapq.heappush(open_set, (len(path) + 1 + self.heuristic(next_graph, target_graph), id(next_graph), next_graph, path + [next_graph]))
        return []


class ReconfigDFS(ReconfigBFS):
    """Depth-first baseline for comparison with BFS and ZDD search."""

    def __init__(self, params, initial_graph, target_graph, weight_initial, constraints_csv, output_path=None):
        super().__init__(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path)
        self.initial_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        self.found_path = None
        
    def get_reconf_seq(self, initial_graph, target_graph):
        if set(initial_graph.edges) == set(target_graph.edges):
            return [initial_graph]
        
        def dls(graph, target_graph, path, depth):
            """
            深さ制限探索（Depth-Limited Search）
            """
            if set(graph.edges) == set(target_graph.edges):
                return path
            if set(graph.edges) != set(target_graph.edges) and depth == 0:
                return None
            
            for next_graph in self.transition(graph, self.G.edges):
                result = dls(next_graph, target_graph, path + [next_graph], depth - 1)
                if result is not None:
                    return result
            return None
        
        depth = 1
        while True:
            print(f"Depth: {depth}")
            #visited = set()
            self.found_path = dls(initial_graph, target_graph, [initial_graph], depth)
            if self.found_path is not None:
                return self.found_path
            depth += 1


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    
    initial_graph = [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    target_graph =  [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
    #target_graph = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    print(f"Hammimg distance: {sum([1 for i, j in zip(initial_graph, target_graph) if i != j])}")
    
    params = Parameters()    
    #weight_initial = pd.read_csv(f"../../output/NDP/{params.demand}/normal/criticality/df_normal.csv")
    #constraints_csv = pd.read_csv("../../output/constraint_all/constraints.csv")
    
    # ZDD
    reconf = ReconfigZDD(params, initial_graph, target_graph, weight_initial=None, constraints_csv=None, output_path="../../output/reconfiguration")
    # BFS
    # reconf = ReconfigBFS(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path="../../output/reconfiguration")
    # A star
    #reconf = ReconfigAstar(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path="../../output/reconfiguration")
    # DFS
    #reconf = ReconfigDFS(params, initial_graph, target_graph, weight_initial, constraints_csv, output_path="../../output/reconfiguration")
    
    reconf_seq = reconf.reconfiguration()
    reconf.draw_sequence(reconf_seq)
    #reconf.draw_initial_target_graphs()
    
