"""Legacy genetic-search baseline for static contraflow design."""

import numpy as np
import math
import matplotlib.pyplot as plt
from parameters_ndp import Parameters
from mfd_dynamics import MFD_Dynamics
import os
import copy
import time
import networkx as nx
import pandas as pd
import shutil
import sys
import logging
from logger_writer import LoggerWriter

class NDP:
    """Optimize a fixed contraflow pattern with a simple genetic algorithm."""

    def __init__(self, params, normal: bool, output_path):
        """Initialize the baseline optimizer and compute the no-policy reference."""
        self.normal = normal
        self.contra_ratio = params.contra_ratio
        self.params = params
        self.original_capacity = copy.deepcopy(self.params.max_boundary_capacity)
        self.pos = {0: (1,5.5), 1: (3,2), 2: (5.5,2), 3:(8,4), 4:(2.5,5), 5:(6,4.5), 6:(3.5, 4.25), 7:(3, 7.5), 8:(6,7)}
        self.output_path = output_path
        self.cost = 0
        self.sim = MFD_Dynamics(params, output_path=self.output_path)
        self.no_policy_cost = self.calc_no_policy()
        print(f"Cost without contraflow: {self.no_policy_cost}")
        
        # 出力フォルダの作成
        if not os.path.exists(f'{self.output_path}') and self.output_path is not None:
            os.makedirs(f'{self.output_path}')
    
    def calc_no_policy(self):
        """Evaluate the baseline objective value without any contraflow control."""
        # 無政策時のコストを計算
        self.sim.run_simulation()
        if self.normal:
            #cost = self.sim.ttt
            cost = np.array(self.sim.throughput_background_list).sum()
        else:
            #cost = self.sim.ttt + self.sim.tet
            cost = np.array(self.sim.throughput_list).sum() + np.array(self.sim.throughput_background_list).sum()
            #cost = -self.sim.risk_people
        return cost
        
    def contraflow_ndp(self):
        """Run the genetic search and return the best static contraflow pattern."""
        # max_boundary_capacityは対称なので、1方向のリンクの要領を1.5倍、対応するリンクの要領を0.5倍とすると、全体の道路容量が保たれる
        # simulationを繰り返し実行し、total travel timeを最小にする、contraflow linkの組み合わせを求める
        # contraflow linkの組み合わせは、self.params.max_boundary_capacityを変化させることで表現し、遺伝的アルゴリズムで計算
        
        num_zones = self.params.num_zones
        adjacent_matrix = self.params.adj_matrix
        population_size = 20
        generations = 30
        self.crossover_rate = 0.8
        self.copy_rate = 0.1
        self.mutation_size = 0.2
        tournament_size = 5

        # リンクの総数を隣接行列に基づいて計算
        link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if adjacent_matrix[i, j] == 1]
        num_links = len(link_indices)

        def fitness_function(individual):

            # 個体(individual)に基づいてリンクキャパシティを調整
            for idx, (i, j) in enumerate(link_indices):
                if individual[idx] == 1:
                    # 対向車線にする
                    self.sim.params.max_boundary_capacity[i, j] *= self.contra_ratio
                    self.sim.params.max_boundary_capacity[j, i] *= 2 - self.contra_ratio
            # シミュレーション実行
            self.sim.run_simulation()

            # キャパシティを元に戻す
            self.sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)

            if self.normal:
                #self.cost = self.sim.ttt
                self.cost = np.array(self.sim.throughput_background_list).sum()
            else:
                #self.cost = self.sim.ttt + self.sim.tet
                self.cost = np.array(self.sim.throughput_list).sum() + np.array(self.sim.throughput_background_list).sum()
                #self.cost = -self.sim.risk_people

            return self.cost

        def initialize_population():
            population = []
            while len(population) < population_size:
                individual = np.random.randint(2, size=num_links)
                if not violates_constraints(individual):
                    population.append(individual)
            return population

        def crossover(parent1, parent2):
            # 一様交叉
            #point = np.random.randint(num_links)
            #child = np.concatenate((parent1[:point], parent2[point:]))
            point = np.random.rand(num_links) < 0.5
            child1 = np.where(point, parent1, parent2)
            child2 = np.where(point, parent2, parent1)
            if violates_constraints(child1):
                child1 = correct_violations(child1)
            if violates_constraints(child2):
                child2 = correct_violations(child2)
            return child1, child2

        def mutate(individual):
            indices = np.random.choice(num_links, int(self.mutation_size * num_links), replace=False)
            for idx in indices:
                individual[idx] = 1 - individual[idx]
            if violates_constraints(individual):
                individual = correct_violations(individual)
            return individual
        
        def violates_constraints(individual):
            for idx, (i, j) in enumerate(link_indices):
                if individual[idx] == 1 and individual[link_indices.index((j, i))] == 1:
                    return True
            return False

        def correct_violations(individual):
            for idx, (i, j) in enumerate(link_indices):
                if individual[idx] == 1 and individual[link_indices.index((j, i))] == 1:
                    if np.random.rand() < 0.5:
                        individual[idx] = 0
                    else:
                        individual[link_indices.index((j, i))] = 0
            return individual

        # 初期集団の生成
        population = initialize_population()
        # 適応度の記録
        fitness_record = []

        for generation in range(generations):
            print(f"Generation {generation}")
            # 適応度計算
            fitness_scores = [fitness_function(ind) for ind in population]
            print(fitness_scores)
            
            # fitness_scoreが最大かつsum(individual)が最小のindividualを保存
            candidate = []
            for x in range(len(population)):
                if fitness_scores[x] == max(fitness_scores):
                    candidate.append(population[x])
            best_individual = min(candidate, key=lambda x: sum(x))
            print(f"Generation {generation}: Best fitness = {max(fitness_scores)}")
            print(f"Best population: {best_individual}")
            fitness_record.append(max(fitness_scores))
            
            # エリート選択
            #sort_index = np.argsort(fitness_scores)
            #sorted_population = [population[i] for i in sort_index]

            # 次世代の生成
            next_population = []
            while len(next_population) < population_size:
                rand = np.random.rand()
                if rand < self.crossover_rate:
                    # 交叉
                    # トーナメント選択
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    parent1 = population[tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]]
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    parent2 = population[tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]]
                    child1, child2 = crossover(parent1, parent2)
                    next_population.append(child1)
                    next_population.append(child2)
                    
                elif rand < self.crossover_rate + self.copy_rate:
                    # コピー
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    child = population[tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]]
                    next_population.append(child)
                    
                else:
                    # 突然変異
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    child = population[tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]]
                    next_population.append(mutate(child))
                    
                # 重複を削除
                next_population = [list(x) for x in set(tuple(x) for x in next_population)]
                
            population = next_population
        
        # 適応度計算
        fitness_scores = [fitness_function(ind) for ind in population]
        print(fitness_scores)
        candidate = []
        for x in range(len(population)):
            if fitness_scores[x] == max(fitness_scores):
                candidate.append(population[x])
        best_individual = min(candidate, key=lambda x: sum(x))
        best_fitness = max(fitness_scores)
        
        reduce_list = []
        for idx, tf in enumerate(best_individual):
            if tf == 1:
                modified = copy.copy(best_individual)
                modified[idx] = 0
                fitness = fitness_function(modified)
                if fitness == best_fitness:
                    reduce_list.append(True)
                elif fitness < best_fitness:
                    reduce_list.append(False)
                else:
                    raise ValueError(f"Error: contraflow {modified} is better than best_individual {best_individual}")
            else:
                reduce_list.append(False)
                    
        best_individual = np.where(reduce_list, 0, best_individual).tolist()
        assert fitness_function(best_individual) == best_fitness
        print(f"Best fitness = {best_fitness}")
        print(f"Best population: {best_individual}")
        fitness_record.append(max(fitness_scores))

        # fitness record　のプロット
        plt.plot(fitness_record)
        plt.xlabel('Generation')
        plt.ylabel('Objective function')
        plt.xticks([i for i in range(generations+1) if i % 10 == 0])
        plt.text(generations, max(fitness_record), f'{round(max(fitness_record))}', ha='right', va='bottom', color='blue')        
        plt.axhline(y=self.no_policy_cost, color='r', linestyle='--', label='No contraflow')
        plt.text(generations, self.no_policy_cost, f'{round(self.no_policy_cost)}', ha='right', va='bottom', color='red')
        plt.legend()
        # y軸の上下限を最大値と最小値に
        plt.ylim(min(self.no_policy_cost, min(fitness_record))-25, max(self.no_policy_cost, max(fitness_record))+25)
        
        if self.output_path is not None:
            if self.normal:
                plt.savefig(f'{self.output_path}/fitness_record_normal.png')
            else:
                plt.savefig(f'{self.output_path}/fitness_record_evac.png')
            plt.close()
        else:
            plt.show()
        
        return best_individual
        

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    params = Parameters()
    normal = False
    output_path = f'../../output/NDP/green_{params.green_split}/{params.demand}/{params.demand_variation}/back{params.background_ratio}/{int(params.simulation_start_time/60)}_{int(params.simulation_end_time/60)}_contra{params.contra_ratio}'
    
    ndp = NDP(params, normal = normal, output_path = output_path)
    
    # logging の設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'{output_path}/log.txt',
        filemode='a'
    )
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    
    # 1. optimization
    best_individual = ndp.contraflow_ndp()
    
    num_zones = params.num_zones
    optimal_link = np.zeros((num_zones, num_zones))
    link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if params.adj_matrix[i, j] == 1]
    
    sim = MFD_Dynamics(params, output_path=output_path.replace('NDP', 'mfd_dynamics').replace(f'_contra{params.contra_ratio}', ''))
    sim.run_simulation()
    sim.plot_accumulation()
    sim.plot_mfd()
    sim.plot_mfd_animation()
    sim.plot_time()
    sim.plot_throughput()
    sim.plot_jam("../../data/processed/zone_polygon.geojson")
    
    sim = MFD_Dynamics(params, output_path=output_path.replace('NDP', 'mfd_dynamics')) 
    for idx, (i, j) in enumerate(link_indices):
        if best_individual[idx] == 1:
            # 対向車線にする
            sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
            sim.params.max_boundary_capacity[j, i] *= 2 - params.contra_ratio
            optimal_link[i, j] = 1
    print(optimal_link)
    
    start = time.time()
    sim.run_simulation()
    print(f"elapsed_time: {time.time() - start}")
    print(f"average evacuation time: {sim.tet / sim.Q[:,:,:].sum()}")
    print(f"people in risk areas: {sim.risk_people}")
    print(f"average normal time: {sim.ttt / sim.Q_background[:,:,:].sum()}")
    print(f"people not started: {sim.not_started_background.sum()}")
    sim.plot_accumulation()
    sim.plot_mfd()
    sim.plot_mfd_animation()
    sim.plot_time()
    sim.plot_throughput()
    sim.plot_jam("../../data/processed/zone_polygon.geojson")
    
