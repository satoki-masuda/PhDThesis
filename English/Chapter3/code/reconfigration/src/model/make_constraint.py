import numpy as np
import math
import matplotlib.pyplot as plt
from parameters_ndp import Parameters
from mfd_dynamics import MFD_Dynamics_numpy
from cost_function import CostFunction_normal, CostFunction_evacuation
from contraflow_ndp import NDP
import os
import copy
import time
import networkx as nx
import pandas as pd

class MakeConstraint(NDP):
    def __init__(self, params, normal, output_path):
        super().__init__(params, normal, output_path)
        
    def make_constraints(self):
        # max_boundary_capacityは対称なので、1方向のリンクの要領を1.5倍、対応するリンクの要領を0.5倍とすると、全体の道路容量が保たれる
        # simulationを繰り返し実行し、total travel timeを最小にする、contraflow linkの組み合わせを求める
        # contraflow linkの組み合わせは、self.params.max_boundary_capacityを変化させることで表現し、遺伝的アルゴリズムで計算
        
        num_zones = self.params.num_zones
        adjacent_matrix = self.params.adj_matrix
        population_size = 100
        generations = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.3
        self.copy_rate = 0.2
        tournament_size = 5

        # リンクの総数を隣接行列に基づいて計算
        link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if adjacent_matrix[i, j] == 1]
        num_links = len(link_indices)

        def fitness_function(individual):
            # 現在のリンクキャパシティを保存
            original_capacity = copy.deepcopy(self.params.max_boundary_capacity)

            # 個体(individual)に基づいてリンクキャパシティを調整
            for idx, (i, j) in enumerate(link_indices):
                if individual[idx] == 1:
                    # 対向車線にする
                    self.params.max_boundary_capacity[i, j] *= 1.5
                    self.params.max_boundary_capacity[j, i] *= 0.5
            # シミュレーション実行
            self.simulation()

            # キャパシティを元に戻す
            self.params.max_boundary_capacity = original_capacity

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
            indices = np.random.choice(num_links, int(0.2 * num_links), replace=False)
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

        # 無政策時の目的関数
        self.simulation()
        cost_without_contraflow = self.cost
        constraints = []
        
        # 初期集団の生成
        population = initialize_population()
        # リストのリストに変換
        population = [x.tolist() for x in population]
        # 適応度の記録
        fitness_record = []

        for generation in range(generations):
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
            fitness_record.append(max(fitness_scores))
            # 無政策時の目的関数より大きいindividualを保存
            for i, ind in enumerate(population):
                if ind not in constraints and fitness_scores[i] > cost_without_contraflow:
                    constraints.append(ind)
            print(f"number of constraints: {len(constraints)}")
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
                    parent1 = population[tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]]
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    parent2 = population[tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]]
                    child1, child2 = crossover(parent1, parent2)
                    next_population.append(child1)
                    next_population.append(child2)
                    
                elif rand < self.crossover_rate + self.copy_rate:
                    # コピー
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    child = population[tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]]
                    next_population.append(child)
                    
                else:
                    # 突然変異
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    child = population[tournament_indices[np.argmin([fitness_scores[i] for i in tournament_indices])]]
                    next_population.append(mutate(child))
                    
                # 重複を削除
                next_population = [list(x) for x in set(tuple(x) for x in next_population)]
                
            population = next_population
            df = pd.DataFrame(constraints, columns=[f'link{i}' for i in range(len(constraints[0]))])
            df.to_csv(f'{self.output_path}/constraints.csv', index=False)
        
        # 適応度計算
        fitness_scores = [fitness_function(ind) for ind in population]
        print(fitness_scores)
        candidate = []
        for x in range(len(population)):
            if fitness_scores[x] == max(fitness_scores):
                candidate.append(population[x])
        best_individual = min(candidate, key=lambda x: sum(x))
        print(f"Generation {generation+1}: Best fitness = {max(fitness_scores)}")
        fitness_record.append(max(fitness_scores))
        # 無政策時の目的関数より大きいindividualを保存
        for i, ind in enumerate(population):
            if ind not in constraints and fitness_scores[i] > cost_without_contraflow:
                constraints.append(ind)
        print(f"number of constraints: {len(constraints)}")
        
        # fitness record　のプロット
        plt.plot(fitness_record)
        plt.xlabel('Generation')
        plt.ylabel('Objective function')
        plt.text(generations, min(fitness_record), f'{round(min(fitness_record))}', ha='right', va='bottom', color='blue')
        # 無政策時の目的関数を線で引く
        self.simulation()
        plt.axhline(y=self.cost, color='r', linestyle='--', label='No contraflow')
        plt.text(generations, self.cost, f'{round(self.cost)}', ha='right', va='bottom', color='red')
        plt.legend()
        # y軸の上下限を最大値と最小値に
        plt.ylim(min(self.cost, min(fitness_record))-25, max(self.cost, max(fitness_record))+25)
        
        if self.normal:
            plt.savefig(f'{self.output_path}/fitness_record_normal.png')
        else:
            plt.savefig(f'{self.output_path}/fitness_record_evac.png')
        plt.show()
        
        return best_individual, constraints
        
        
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    params = Parameters()
    ndp = MakeConstraint(params, normal=True, output_path='../../output/constraint_half')
    
    # 2. optimization
    best_individual, constraints = ndp.make_constraints()
    # constraintsをdataframeとして保存
    df = pd.DataFrame(constraints, columns=[f'link{i}' for i in range(len(constraints[0]))])
    df.to_csv(f'{ndp.output_path}/constraints.csv', index=False)
    
    num_zones = ndp.params.num_zones
    optimal_link = np.zeros((num_zones, num_zones))
    adjacent_matrix = ndp.params.adj_matrix
    link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if adjacent_matrix[i, j] == 1]
    for idx, (i, j) in enumerate(link_indices):
        if best_individual[idx] == 1:
            optimal_link[i, j] = 1
    print(optimal_link)
