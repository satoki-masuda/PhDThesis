"""Chapter 5 のメイン推定スクリプト。

このファイルは Chapter 5 の本流処理の入口で、データ読込、政策関数推定、
居住地選択モデル推定、forward simulation、構造パラメータ推定を順に実行する。
"""

import random
import copy
import os

import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import ray

from model.policy_estimation import PerturbedModel, HeteroMLEPolicyFunction
from model.forward_simulation import ForwardSimulation
from utils.helpers import timeit
from utils.config_manager import ConfigManager
from utils.data_loader import load_simulation_data
from model.bbl_objective_function import Q

# 設定の読み込みと初期化
config = ConfigManager("config.yaml")

# 設定値の展開（後方互換性のため変数としても利用可能）
start_year = config.start_year
end_year = config.end_year
ref_year = config.ref_year
dev_zone = config.dev_zone
res_zone = config.res_zone
n_forward = config.n_forward
n_pertub = config.n_perturb
seed = config.seed
horizon = config.horizon
discount_factor = config.discount_factor
num_features_gov = config.num_features_gov
num_features_dev = config.num_features_dev
bootstrap = config.bootstrap
n_bootstrap = config.n_bootstrap
scale_pop_dense = config.scale_pop_dense
scale_pop = config.scale_pop
scale_los = config.scale_los
scale_dev_res = config.scale_dev_res
scale_dev_shop = config.scale_dev_shop
scale_price = config.scale_price
scale_land_demand = config.scale_land_demand

# Ray の初期化
ray.init(**config.ray_config)

# -------------------------------
# Step 0: データ読み込み
with timeit("データ読み込み"):
    data_bundle = load_simulation_data(config)
    transition_model = data_bundle.transition_model
    df = data_bundle.df_pt
    data = data_bundle.data
    Dzone_list = data_bundle.Dzone_list
    Rzone_list = data_bundle.Rzone_list
    Darea_list = data_bundle.Darea_list
    columns_zone_fe = data_bundle.columns_zone_fe
    columns_year_fe = data_bundle.columns_year_fe
    los_std = data_bundle.los_std
    develop_std = data_bundle.develop_std
    features = data_bundle.features

    zone_code_in = transition_model.zone_code_in
    Dzone_code_all = transition_model.Dzone_code_all
    Dzoning = transition_model.Dzoning
    zone_conversion = transition_model.zone_conversion
    
# -------------------------------
# Step 1: モデル推定
with timeit("方策関数と遷移関数の推定"):
    # 平均に使う説明変数
    #gov_mean_X = data[['population_t', 'distance_CBD'] + columns_zone_fe + columns_year_fe] #  
    #gov_var_X = data[['const', 'population_t', 'distance_CBD']]
    # 2体用 (govは状態変数に影響されない)
    gov_mean_X = data[columns_zone_fe + columns_year_fe] #  
    gov_var_X = data[['const']]
    
    dev_mean_X = data[['pop_dense_t', 'los_t', 'land_price_t', 'distance_CBD'] + columns_zone_fe + columns_year_fe] # + columns_zone_fe 
    dev_var_X = data[['const', 'pop_dense_t', 'los_t', 'land_price_t', 'distance_CBD']]
    
    # multiplicative heteroskedasticity model
    gov_model = HeteroMLEPolicyFunction()
    gov_model.estimate_policy_function_mle(
        gov_mean_X,
        gov_var_X,
        data['los_t_raw'].values,
        y_scale=scale_los,
        output_path="output/estimates/gov_model_mle.joblib"
    )

    dev_model = HeteroMLEPolicyFunction()
    dev_model.estimate_policy_function_mle(
        dev_mean_X,
        dev_var_X,
        data['develop_res_t_raw'].values,
        y_scale=scale_dev_res,
        output_path="output/estimates/dev_model_mle.joblib"
    )

    # transition_model
    if os.path.exists("output/estimates/mxl_step1_results.json") and os.path.exists("output/estimates/mxl_step2_results.json"):
        print("遷移モデルの推定結果が既に存在します。再推定をスキップします。")
        transition_model.load_estimates("output/estimates/mxl_step1_results.json", "output/estimates/mxl_step2_results.json", model="MXL_2step")
    else:
        print("遷移モデルの推定を行います。")
        X1_hetero, X1_mean, X2, Z, W, y, obs_share, relocation_years = transition_model.make_estimation_data_2step()
        transition_model.estimate_choice_model(y, method="MXL_2step", X1_hetero=X1_hetero, X1_mean=X1_mean, X2=X2, Z=Z, W=W, obs_share=obs_share, relocation_years=relocation_years)

# -------------------------------
# Step 2: forward simulation による特徴量 W の生成
@ray.remote
def simulate_W(market, data, features, df, zone_conversion, 
               gov_model, dev_model, base_gov_model, base_dev_model,
               transition_model, Darea_list, Dzone_list, Rzone_list, 
               start_year, ref_year, horizon, discount_factor, n_forward, boot_id=0):
    """1つの市場 `(zone, year)` について discounted payoff feature W を平均化する。"""
    gov_model = copy.deepcopy(gov_model)
    dev_model = copy.deepcopy(dev_model)
    base_gov_model = copy.deepcopy(base_gov_model)
    base_dev_model = copy.deepcopy(base_dev_model)
    W_gov_list = []
    W_dev_list = []
    zone, year = market
    assert year >= start_year + ref_year - 1, "Year must be greater than or equal to start_year + ref_year - 1"
    # dfが全ゾーンの全人口の合成人口を持っている場合の初期人口生成
    #df_start = df[df["年"]==year].copy()
    #N0_szone = [df_start.loc[df_start["居住地_前_ゾーン"]==szone, "拡大係数"].sum() for szone in Rzone_list]
    #idx_map = {z: i for i, z in enumerate(Dzone_list)}
    #N0 = [float(np.array(N0_szone) @ np.array(zone_conversion[idx_map[z], :])) for z in Dzone_list]
    # 初期人口を住民基本台帳データから取得する場合
    N0 = data.loc[(data['year'] == year) & data['zone'].isin(Rzone_list), 'population_t_raw'].to_list()
    
    I0 = data.loc[(data['year'] == year), 'los_t_raw'].to_list()
    P0 = data.loc[(data['year'] == year), 'land_price_t_raw'].to_list()
    development_lists = [data.loc[(data['year'] == year-i), 'develop_res_t_raw'].to_list() for i in range(ref_year)]
    
    for sim_id in range(n_forward):
        seed = np.uint32(abs(hash((zone, year, sim_id, boot_id))) % (2**32))
        forward_simulation = ForwardSimulation(N0, I0, P0, development_lists, zone, start_year=year, data_final_year=end_year, T=horizon, discount_factor=discount_factor,
                                            gov_model=gov_model,
                                            dev_model=dev_model,
                                            base_gov_model=base_gov_model,
                                            base_dev_model=base_dev_model,
                                            transition_model = transition_model,
                                            features=features,
                                            df=df,
                                            zone_convert_dict=zone_conversion,
                                            area_list=Darea_list,
                                            master_seed = seed,
                                            transition_type="deterministic")
        result_forward = forward_simulation.simulate()
        W_gov, W_dev = result_forward["W_gov"], result_forward["W_dev"]
        W_gov_list.append(W_gov)
        W_dev_list.append(W_dev)
        
    return np.mean(np.array(W_gov_list), axis=0), np.mean(np.array(W_dev_list), axis=0)

data_id = ray.put(data)
df_id = ray.put(df)
Darea_list_id = ray.put(Darea_list)
Dzone_list_id = ray.put(Dzone_list)
Rzone_list_id = ray.put(Rzone_list)
gov_model_id = ray.put(gov_model)
dev_model_id = ray.put(dev_model)
zone_conversion_id = ray.put(zone_conversion)
transition_model_id = ray.put(transition_model)

features_id = ray.put(features)

# 全市場． zone, yearの組 (2年に1回．．連続した年を入れると独立性が損なわれるため)
market_list = [(zone, year) for zone in Dzone_list for year in range(start_year+ref_year-1, end_year+1, 3)]
n_sample = len(market_list)

# ブートストラップサンプリング, サブサンプル法
boot_thetas = []
for boot_id in tqdm(range(n_bootstrap), desc="bootstrap sampling"):
    #boot_sample = random.choices(market_list, k=n_sample) # 復元抽出, ブートストラップ
    boot_sample = random.sample(market_list, k=int(n_sample * 0.5)) # 非復元抽出，サブサンプル法
    n_boot_sample = len(boot_sample)
    
    with timeit("並列forward simulation"):
        futures = [simulate_W.remote(mrkt, data_id, features_id, df_id, zone_conversion_id, gov_model_id, dev_model_id, gov_model_id, dev_model_id, transition_model_id, Darea_list_id, Dzone_list_id, Rzone_list_id, start_year, ref_year, horizon, discount_factor, n_forward, boot_id) for mrkt in boot_sample]
        print(f"Simulating {len(futures)} markets...")
        results = ray.get(futures)
        #W_gov = np.array([W_gov for W_gov, _ in results])
        W_dev = np.array([W_dev for _, W_dev in results])
    
    with timeit("並列forward simulation (代替モデル)"):
        #W_gov_perturb = np.zeros((n_boot_sample, W_gov.shape[1], n_pertub))
        W_dev_perturb = np.zeros((n_boot_sample, W_dev.shape[1], n_pertub))
        sigma_scale = [0.7, 0.85, 1.15, 1.3]
        mean_shift_scale = [0.05, 0.10, 0.15]
        for per in tqdm(range(n_pertub)):
            """ 2体推定ではいらない
            results = []
            futures_alt_gov = []
            for mrkt in boot_sample:
                alt_gov = PerturbedModel(gov_model, mean_shift=los_std[Dzone_list.index(mrkt[0])] * random.choice(mean_shift_scale), sigma_scale= random.choice(sigma_scale))
                futures_alt_gov.append(simulate_W.remote(mrkt, data_id, features_id, df_id, zone_conversion_id, alt_gov, dev_model_id, gov_model_id, dev_model_id, transition_model_id, Darea_list_id, Dzone_list_id, Rzone_list_id, start_year, ref_year, horizon, discount_factor, n_forward, boot_id))
            results = ray.get(futures_alt_gov)
            W_gov_perturb[:,:,per] = np.array([W_gov for W_gov, _ in results])
            """
            
            results = []
            futures_alt_dev = []
            for mrkt in boot_sample:
                alt_dev = PerturbedModel(dev_model, mean_shift=develop_std[Dzone_list.index(mrkt[0])] * random.choice(mean_shift_scale), sigma_scale= random.choice(sigma_scale))
                futures_alt_dev.append(simulate_W.remote(mrkt, data_id, features_id, df_id, zone_conversion_id, gov_model_id, alt_dev, gov_model_id, dev_model_id, transition_model_id, Darea_list_id, Dzone_list_id, Rzone_list_id, start_year, ref_year, horizon, discount_factor, n_forward, boot_id))
            results = ray.get(futures_alt_dev)
            W_dev_perturb[:,:,per] = np.array([W_dev for _, W_dev in results])
    
    with timeit("パラメータ推定"):
        theta_est = []
        #for W, W_perturb, num_features in [(W_gov, W_gov_perturb, num_features_gov), (W_dev, W_dev_perturb, num_features_dev)]:
        if True:
            W = W_dev.copy()
            W_perturb = W_dev_perturb.copy()
            num_features = num_features_dev
            theta0 = np.zeros(num_features-1)  # 初期値
            print("初期損失関数値:", Q(theta0, W, W_perturb))
            res = minimize(Q, theta0,
                        args=(W, W_perturb),
                        method='Nelder-Mead')  # Nelder-Mead
            iterations = 0
            while (not res.success) and (iterations < 10):
                theta0 = res.x
                res = minimize(Q, theta0,
                            args=(W, W_perturb),
                            method='Nelder-Mead') # Nelder-Mead
                iterations += 1
            
            print(f"最適化結果: {res.fun}")
            theta = res.x
            theta_est.extend(theta)

        boot_thetas.append(theta_est)
        #print(f"gov: {np.round(theta_est[:num_features_gov-1], 3)}")
        #print(f"dev: {np.round(theta_est[num_features_gov-1:], 3)}")
        print(f"dev: {np.round(theta_est, 3)}")

if bootstrap:
    print("ブートストラップ推定完了")
    boot_thetas = np.array(boot_thetas)
    boot_thetas_mean = np.mean(boot_thetas, axis=0)
    boot_thetas_std = np.std(boot_thetas, axis=0, ddof=1) # 不偏標準偏差
    boot_theta_ci = np.percentile(boot_thetas, [2.5, 97.5], axis=0)
    print("ブートストラップ推定結果")
    print(f"平均: {boot_thetas_mean}")
    print(f"標準偏差: {boot_thetas_std}")
    print(f"95% 信頼区間: {boot_theta_ci}")
    print(f"95% 信頼区間: {[(l, u) for l, u in zip(np.round(boot_theta_ci[0], 3), np.round(boot_theta_ci[1], 3))]}")
else:
    print("推定完了")
# Ray のシャットダウン
ray.shutdown()
