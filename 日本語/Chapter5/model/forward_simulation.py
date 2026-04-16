"""推定済みモデルから将来系列を生成する forward simulation。"""

import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.payoff_structure import compute_payoff_features
import statsmodels.api as sm
from utils.config_manager import ConfigManager

# 設定の読み込みと初期化
config = ConfigManager("config.yaml")
# 設定値の展開（後方互換性のため変数としても利用可能）
data_start_year = config.start_year
num_features_gov = config.num_features_gov
num_features_dev = config.num_features_dev
scale_pop_dense = config.scale_pop_dense
scale_pop = config.scale_pop
scale_los = config.scale_los
scale_dev_res = config.scale_dev_res
scale_dev_shop = config.scale_dev_shop
scale_price = config.scale_price
scale_land_demand = config.scale_land_demand
move_ratio = config.move_ratio

class ForwardSimulation:
    """
    将来の人口・投資・開発・地価系列を forward simulation で生成する。

    Parameters:
        N0 (float): Initial population
        T (int): Simulation horizon
        gov_model: Trained model for government policy
        dev_model: Trained model for developer action
        state_model: Trained model for population transition
    """
    def __init__(
        self, 
        N0, I0, P0, development_lists, 
        zone, start_year, data_final_year, T, discount_factor, 
        gov_model, dev_model, base_gov_model, base_dev_model, 
        transition_model, features, df, zone_convert_dict, area_list, master_seed, transition_type="deterministic"):
        self.N0 = N0
        self.I0 = I0
        self.P0 = P0
        self.initial_development_lists = copy.deepcopy(development_lists)
        self.zone = zone
        self.start_year = start_year
        self.data_final_year = data_final_year
        self.T = T
        self.discount_factor = discount_factor
        self.gov_model = gov_model
        self.dev_model = dev_model
        self.base_gov_model = base_gov_model
        self.base_dev_model = base_dev_model
        self.transition_model = transition_model
        self.features = features
        self.df = df
        self.zone_convert_dict = zone_convert_dict
        self.area_list = area_list
        self.n_zone = len(N0)
        self.transition_type = transition_type
        self.master_seed = master_seed
        self.generate_shocks(self.master_seed)
        self.generated_samples = None
    
    def generate_shocks(self, seed):
        rng = np.random.default_rng(seed)
        self.nu_gov = rng.normal(0, 1, size=(self.T, self.n_zone))
        self.nu_dev = rng.normal(0, 1, size=(self.T, self.n_zone))
    
    def simulate(self):
        W_gov = np.zeros((self.T, num_features_gov))  # To store features at each time step
        W_dev = np.zeros((self.T, num_features_dev))  # To store features at each time step
        self._simulate_transition()
        distCBD = self.features[1][0, self.zone]
        
        for t in range(self.T):
            gov, res = compute_payoff_features(
                self.population_list[t, self.zone],
                self.investment_list[t, self.zone],
                self.development_list[t, self.zone],
                self.house_price_list_D[t, self.zone], self.land_demand[t, self.zone], distCBD
                )
            W_gov[t, :] = copy.deepcopy(gov)
            W_dev[t, :] = copy.deepcopy(res)
        
        discount_factors = self.discount_factor ** np.arange(self.T)
        W_discounted_gov = np.sum(W_gov.T * discount_factors, axis=1)
        W_discounted_dev = np.sum(W_dev.T * discount_factors, axis=1)

        return {
            "W_gov": W_discounted_gov, # n_feature \times 1
            "W_dev": W_discounted_dev,
            "population_path": self.population_list,
            "investment_path": self.investment_list,
            "development_path": self.development_list,
            "price_path": self.land_price_list_D
        }

    def _simulate_transition(self):
        self.population_list = np.zeros((self.T, self.n_zone))
        self.investment_list = self.population_list.copy()
        self.house_price_list_D = self.population_list.copy()
        self.land_price_list_D = self.population_list.copy()
        self.land_demand = self.population_list.copy()
        self.development_list = self.population_list.copy()
        
        population = copy.deepcopy(self.N0)
        investment = copy.deepcopy(self.I0)
        land_price_D = copy.deepcopy(self.P0)
        development_lists = copy.deepcopy(self.initial_development_lists)
        choice_zone = self.transition_model.zoning
        # 不動産価格の初期値
        if self.start_year <= self.data_final_year:
            house_price_R = choice_zone[[f"UnitPrice_Attached_{self.start_year}", f"UnitPrice_Condo_{self.start_year}"]].mean(axis=1, skipna=True).to_list()
        else:
            house_price_R = choice_zone[[f"UnitPrice_Attached_{self.data_final_year}", f"UnitPrice_Condo_{self.data_final_year}"]].mean(axis=1, skipna=True).to_list()
        house_price_D = [np.mean(np.array(house_price_R), where=np.array(self.zone_convert_dict[Lzone])>0) for Lzone in range(self.n_zone)]
        
        for t in range(self.T):
            year = self.start_year + t + 1
            year_index = year-self.start_year if year <= self.data_final_year else self.data_final_year-self.start_year
            
            gov_data, dev_data = self._build_arrays(population, investment, development_lists[-1], year, year_index, land_price_D)
            
            investment = self.base_gov_model.predict(gov_data.copy(), self.nu_gov[t,:])
            development = self.base_dev_model.predict(dev_data.copy(), self.nu_dev[t,:])
            investment[self.zone] = self.gov_model.predict(gov_data.iloc[[self.zone], :].copy(), [self.nu_gov[t,self.zone]], t=t, actor="gov", zones=[self.zone])[0]
            development[self.zone] = self.dev_model.predict(dev_data.iloc[[self.zone], :].copy(), [self.nu_dev[t,self.zone]], t=t, actor="dev", zones=[self.zone])[0]
            investment = np.clip(investment, 1.0, None).tolist()  # Ensure non-negative investment
            development = np.clip(development, 1.0, None).tolist()  # Ensure non-negative development
            
            development_lists.pop(0)
            development_lists.append(copy.deepcopy(development))
            if self.transition_type == "deterministic":
                new_house_price_D, population, land_demand = self.transition_model.predict_aggregate_equilibrium_population(population, development_lists, investment, house_price_D, year=year, move_ratio=move_ratio)
            elif self.transition_type == "stochastic":
                new_house_price_D, population, land_demand = self.transition_model.predict_equilibrium_population(self.df, population, development_lists, investment, house_price_D, year=year, move_ratio=move_ratio)
            # 地価は住宅価格の変動と同じ割合で変動すると仮定
            land_price_D = (np.array(land_price_D) * np.array(new_house_price_D) / np.array(house_price_D)).tolist()
            house_price_D = copy.deepcopy(new_house_price_D)
            
            self.population_list[t, :] = copy.deepcopy(population)
            self.investment_list[t, :] = copy.deepcopy(investment)
            self.house_price_list_D[t, :] = copy.deepcopy(house_price_D)
            self.land_price_list_D[t, :] = copy.deepcopy(land_price_D)
            self.land_demand[t, :] = copy.deepcopy(land_demand)
            self.development_list[t, :] = copy.deepcopy(development)
        
            
    
    def _build_arrays(self, population, investment, development, year, year_index, land_price_D):
        pop_dense = np.array(population) / np.array(self.area_list) / scale_pop_dense
        population = np.array(population) / scale_pop
        dev_res   = np.array(development) / scale_dev_res
        dev_shop  = self.features[0][year_index, :]
        dist      = self.features[1][year_index, :]
        risk      = self.features[2][year_index, :]
        los      = np.array(investment) / scale_los
        land_price  = np.array(land_price_D) / scale_price
        # year_fe_data: data_start_year to self.data_final_yearの列について，yearに対応する列が1，その他が0のone-hotベクトル
        year_fe_data = np.zeros((self.n_zone, self.data_final_year - data_start_year + 1))
        if data_start_year <= year <= self.data_final_year:
            year_fe_data[:, year - data_start_year] = 1
        elif year > self.data_final_year:
            year_fe_data[:, -1] = 1
        
        gov_arr = np.concatenate(
            [np.column_stack([pop_dense, population, dev_res, dev_shop, dist, risk]), np.eye(self.n_zone), year_fe_data, np.ones((self.n_zone, 1))],
            axis=1
        )
        dev_arr = np.concatenate(
            [np.column_stack([pop_dense, population, los, dev_shop, land_price, dist, risk]), np.eye(self.n_zone), year_fe_data, np.ones((self.n_zone, 1))],
            axis=1
        )
        gov_col_names = ["pop_dense_t", "population_t", "develop_res_t", "develop_shop_t", "distance_CBD", "risk"] + [f"zone_{z}" for z in range(self.n_zone)] + [f"year_{y}" for y in range(data_start_year, self.data_final_year + 1)] + ["const"]
        dev_col_names = ["pop_dense_t", "population_t", "los_t", "develop_shop_t", "land_price_t", "distance_CBD", "risk"] + [f"zone_{z}" for z in range(self.n_zone)] + [f"year_{y}" for y in range(data_start_year, self.data_final_year + 1)] + ["const"]
        gov_arr = pd.DataFrame(gov_arr, columns=gov_col_names)
        dev_arr = pd.DataFrame(dev_arr, columns=dev_col_names)
        
        return gov_arr, dev_arr
    
    def generate_samples(self, n_samples):
        W_gov_list = []
        W_dev_list = []
        population_list = []
        development_list = []
        investment_list = []
        land_price_list = []
        
        for sim_id in range(n_samples):
            self.generate_shocks(self.master_seed + sim_id)
            out = self.simulate()
            
            W_gov_list.append(out["W_gov"])
            W_dev_list.append(out["W_dev"])
            population_list.append(out["population_path"])
            development_list.append(out["development_path"])
            investment_list.append(out["investment_path"])
            land_price_list.append(out["price_path"])
        
        self.generated_samples = {
            "W_gov": np.array(W_gov_list),
            "W_dev": np.array(W_dev_list),
            "population_path": np.array(population_list),
            "development_path": np.array(development_list),
            "investment_path": np.array(investment_list),
            "price_path": np.array(land_price_list)
        }
        
        return self.generated_samples
    
    def visualize_paths(self, output_dir, observed_data=None, name_observed:str="Observed Data"):
        if self.generate_samples is None:
            raise ValueError("No simulation data available. Please run generate_samples() method first.")
        if observed_data is not None:
            observation_time_steps = len(observed_data["population_path"])
            if observation_time_steps < self.T + 1:
                for key in observed_data:
                    observed_data[key] = np.concatenate(
                        (observed_data[key], 
                         np.full(self.T + 1 - observation_time_steps, np.nan)),
                        axis=0
                    )
        
        os.makedirs(output_dir, exist_ok=True)
        time_steps = np.arange(self.T+1)
        # プロットも保存
        # 各種パスごとに個別にプロットを保存するよう修正
        n_forward = self.generated_samples["population_path"].shape[0]
        path_dict = {
            "population": np.concatenate((np.tile(np.array(self.N0), (n_forward, 1))[:,None,:], self.generated_samples["population_path"]), axis=1),
            "investment": np.concatenate((np.tile(np.array(self.I0), (n_forward, 1))[:,None,:], self.generated_samples["investment_path"]), axis=1),
            "development": np.concatenate((np.tile(np.array(self.initial_development_lists), (n_forward, 1))[:,None,:], self.generated_samples["development_path"]), axis=1),
            "price": np.concatenate((np.tile(np.array(self.P0), (n_forward, 1))[:,None,:], self.generated_samples["price_path"]), axis=1)
        }
        
        mae_dict = {}
        mape_dict = {}
        for key, item in path_dict.items():
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, item[:, :, self.zone].T, color='lightgray', alpha=0.5)
            avg_path = np.mean(item[:, :, self.zone], axis=0)
            upper_limits = np.max(avg_path) * 1.1
            plt.plot(time_steps, avg_path, color='blue', linewidth=2, label='Average Path')
            if observed_data is not None:
                plt.plot(time_steps, observed_data[f"{key}_path"], color='red', linewidth=2, label=name_observed)
                upper_limits = max(float(np.nanmax(observed_data[f"{key}_path"])) * 1.1, upper_limits)
            
            # 年平均誤差（年度ごとの平均のずれ）を計算・表示（観測データがある場合のみ）
            mean_annual_error = None
            mean_annual_perc_error = None
            if observed_data is not None:
                # 観測パスがnanの位置は無視して一致部分のみ評価
                sim_path = avg_path
                obs_path = observed_data.get(f"{key}_path")
                # 同じ長さを仮定（前処理で補間 or pad済み）
                if obs_path is not None:
                    obs_path = np.asarray(obs_path)
                    sim_path = np.asarray(sim_path)
                    mask = ~np.isnan(obs_path)
                    if np.any(mask):
                        sim_on_obs = sim_path[mask]
                        obs_on_obs = obs_path[mask]
                        annual_diff = np.abs(sim_on_obs - obs_on_obs)  # 各年のずれ
                        mean_annual_error = np.mean(annual_diff)  # 年平均のずれ
                        nonzero_idx = obs_on_obs != 0
                        if np.any(nonzero_idx):
                            mean_annual_perc_error = np.mean(annual_diff[nonzero_idx] / obs_on_obs[nonzero_idx]) * 100  # 年平均相対ずれ(%)
                        else:
                            mean_annual_perc_error = None
                        # 注釈 (annotation)
                        mae_text = f"MAE (anuual)={mean_annual_error:.2f}"
                        mape_text = (
                            f"MAPE (annual)={mean_annual_perc_error:.1f}%"
                            if mean_annual_perc_error is not None
                            else "MAPE (annual)=N/A"
                        )
                        plt.gca().text(
                            0.98,
                            0.02,
                            f"{mae_text}\n{mape_text}",
                            transform=plt.gca().transAxes,
                            va="bottom",
                            ha="right",
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                        )
            
            plt.xlabel("Time")
            plt.ylabel(key.capitalize())
            plt.ylim(0, upper_limits)
            plt.title(f"Zone {self.zone}", fontsize=20)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{key}_path_zone_{self.zone}.png"), dpi=400)
            plt.close()
            
            mae_dict[key] = mean_annual_error
            mape_dict[key] = mean_annual_perc_error
            
        return mae_dict, mape_dict
        
    def in_sample_validation(self, output_dir, observed_data):
        if self.generate_samples is None:
            raise ValueError("No simulation data available. Please run generate_samples() method first.")
        
        os.makedirs(output_dir, exist_ok=True)
        time_steps = np.arange(self.T+1)
        n_forward = self.generated_samples["population_path"].shape[0]
        path_dict = {
            "population": np.concatenate((np.tile(np.array(self.N0), (n_forward, 1))[:,None,:], self.generated_samples["population_path"]), axis=1),
            "investment": np.concatenate((np.tile(np.array(self.I0), (n_forward, 1))[:,None,:], self.generated_samples["investment_path"]), axis=1),
            "development": np.concatenate((np.tile(np.array(self.initial_development_lists), (n_forward, 1))[:,None,:], self.generated_samples["development_path"]), axis=1),
            "price": np.concatenate((np.tile(np.array(self.P0), (n_forward, 1))[:,None,:], self.generated_samples["price_path"]), axis=1)
        }
        
        for key, item in path_dict.items():
            # 観測と予測の散布図
            plt.figure(figsize=(8, 8))
            avg_path = np.mean(item[:, 1:-1, self.zone], axis=0)
            obs_path = observed_data.get(f"{key}_path")[1:]
            plt.scatter(obs_path, avg_path, color='blue', alpha=0.7)
            max_val = max(np.nanmax(obs_path), np.nanmax(avg_path))
            min_val = min(np.nanmin(obs_path), np.nanmin(avg_path))
            plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
            plt.xlabel("Observed")
            plt.ylabel("Simulated Average")
            plt.title(f"Zone {self.zone}", fontsize=20)
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            # 回帰係数を乗せる
            obs_path = np.asarray(obs_path)
            sim_path = np.asarray(avg_path)
            mask = ~np.isnan(obs_path)
            if np.any(mask):
                sim_on_obs = sim_path[mask]
                obs_on_obs = obs_path[mask]
                X = sm.add_constant(obs_on_obs)
                model = sm.OLS(sim_on_obs, X).fit()
                intercept, slope = model.params
                plt.gca().text(
                    0.05,
                    0.95,
                    f"y = {slope:.2f}x + {intercept:.2f}",
                    transform=plt.gca().transAxes,
                    va="top",
                    ha="left",
                    fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
                    
            plt.savefig(os.path.join(output_dir, f"{key}_scatter_zone_{self.zone}.png"), dpi=400)
            plt.close()
        
        
