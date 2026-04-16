"""Counterfactual analysis script for Chapter 5.

This script loads the estimated model, applies a policy restriction to the
target zone, recomputes a local equilibrium path, and compares the simulated
series with the observed data.
"""

import os
import numpy as np
import ray
from model.policy_estimation import HeteroMLEPolicyFunction
from model.forward_simulation import ForwardSimulation
from utils.helpers import timeit
from simulation.mpe_local_driver import solve_local_mpe_for_market
from simulation.visualization import plot_simulation_paths, plot_all_zones_lines, plot_all_zones_observed_lines
from utils.config_manager import ConfigManager
from utils.data_loader import load_simulation_data
from simulation.policy_override import PolicyLocationControl

# Load configuration.
config = ConfigManager("config.yaml")

# Unpack config values for backward-compatible script usage.
theta = {
    "gov": np.array(config.theta_gov),
    "dev": np.array(config.theta_dev)
}
target_zone = config.target_zone
sim_start_year = config.sim_start_year
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
scale_pop_dense = config.scale_pop_dense
scale_pop = config.scale_pop
scale_los = config.scale_los
scale_dev_res = config.scale_dev_res
scale_dev_shop = config.scale_dev_shop
scale_price = config.scale_price
scale_land_demand = config.scale_land_demand
# Initialize Ray.
ray.init(**config.ray_config)

with timeit("Load data"):
    data_bundle = load_simulation_data(config)
    transition_model = data_bundle.transition_model
    df = data_bundle.df_pt
    data = data_bundle.data
    Dzone_list = data_bundle.Dzone_list
    Rzone_list = data_bundle.Rzone_list
    Darea_list = data_bundle.Darea_list
    features = data_bundle.features
    zone_conversion = transition_model.zone_conversion

    gov_model = HeteroMLEPolicyFunction()
    gov_model.load_model("output/estimates/gov_model_mle.joblib")
    dev_model = HeteroMLEPolicyFunction()
    dev_model.load_model("output/estimates/dev_model_mle.joblib")
    # Transition model.
    if os.path.exists("output/estimates/mxl_step1_results.json") and os.path.exists("output/estimates/mxl_step2_results.json"):
        print("Transition-model estimates already exist. Skipping re-estimation.")
        transition_model.load_estimates("output/estimates/mxl_step1_results.json", "output/estimates/mxl_step2_results.json", model="MXL_2step")
    else:
        print("Transition-model estimates are missing. Run estimation first.")
    
bounds_gov = (1e-3, np.max(data["los_t_raw"]) * 2.0)
bounds_dev = (1e-3, np.max(data["develop_res_t_raw"]) * 2.0)

# Apply a location cap as the counterfactual policy.
upper_cap_dev = data.loc[data["zone"]==target_zone, "develop_res_t_raw"].mean() * 0.1
dev_wrap = PolicyLocationControl(dev_model)
dev_wrap.set_location_cap(target_zone, upper_cap_dev)

bounds_dev_cf = (1e-3, upper_cap_dev)
bounds_gov_cf = bounds_gov

# Alternative initialization if df stores a fully synthetic population.
#df_start = df[df["年"]==sim_start_year].copy()
#N0_szone = [df_start.loc[df_start["居住地_前_ゾーン"]==szone, "拡大係数"].sum() for szone in Rzone_list]
#idx_map = {z: i for i, z in enumerate(Dzone_list)}
#N0 = [float(np.array(N0_szone) @ np.array(zone_conversion[idx_map[z], :])) for z in Dzone_list]

# Current initialization based on resident-registry population data.
N0 = data.loc[(data['year'] == sim_start_year) & data['zone'].isin(Rzone_list), 'population_t_raw'].to_list()

I0 = data.loc[(data['year'] == sim_start_year), 'los_t_raw'].to_list()
P0 = data.loc[(data['year'] == sim_start_year), 'land_price_t_raw'].to_list()
development_lists = [data.loc[(data['year'] == sim_start_year-i-1), 'develop_res_t_raw'].to_list() for i in range(ref_year)]

# First generate one path under the observed policy as an initialization.
init_forward = ForwardSimulation(
    N0=N0, I0=I0, P0=P0, development_lists=development_lists,
    zone=target_zone, start_year=sim_start_year, data_final_year=end_year, T=end_year - sim_start_year +1,
    discount_factor=discount_factor,
    gov_model=gov_model, dev_model=dev_model,
    base_gov_model=gov_model, base_dev_model=dev_model,
    transition_model=transition_model, features=features, df=df,
    zone_convert_dict=zone_conversion, area_list=Darea_list,
    master_seed=seed,
    transition_type="deterministic"
).simulate()

init_paths = {
    "population_path": init_forward["population_path"],
    "investment_path": init_forward["investment_path"],
    "development_path": init_forward["development_path"],
    "price_path": init_forward["price_path"]
}

models = {
    "gov": gov_model,
    "dev": dev_model,
    "base_gov": gov_model,
    "base_dev": dev_model,
    "trans": transition_model
}
sol = solve_local_mpe_for_market(
    market=(target_zone, sim_start_year),
    init_paths=init_paths,
    ref_year=ref_year, H=horizon, n_forward=n_forward, beta=discount_factor,
    models=models, features=features, df=df, zone_convert=zone_conversion,
    area_list=Darea_list, bounds_gov=bounds_gov, bounds_dev=bounds_dev,
    seed=seed, theta=theta
)
init_paths = {k: sol["result"][k] for k in ["population_path","investment_path","development_path","price_path"]}

# This local procedure is intended for policy-function overrides only.
# If the structural parameters themselves change, a fuller equilibrium method
# such as a stochastic algorithm or deep RL would be more appropriate.
models = {
    "gov": gov_model,
    "dev": dev_wrap,
    "base_gov": gov_model,
    "base_dev": dev_wrap,
    "trans": transition_model
}
sol = solve_local_mpe_for_market(
    market=(target_zone, sim_start_year),
    init_paths=init_paths,
    ref_year=ref_year, H=horizon, n_forward=n_forward, beta=discount_factor,
    models=models, features=features, df=df, zone_convert=zone_conversion,
    area_list=Darea_list, bounds_gov=bounds_gov_cf, bounds_dev=bounds_dev_cf,
    seed=seed, theta=theta
)

# Equilibrium paths.
Wg, Wd = sol["result"]["W_gov"], sol["result"]["W_dev"]
Vg = np.dot(Wg, theta["gov"].reshape(-1,1))
Vd = np.dot(Wd, theta["dev"].reshape(-1,1))
paths = {k: sol["result"][k] for k in ["population_path","investment_path","development_path","price_path"]}

# Plotting.
observed_cols = {
    "investment": "los_t_raw",
    "development": "develop_res_t_raw",
    "price": "land_price_t_raw",
    "population": "population_t_raw",
}
zone_labels = Dzone_list  # Zone labels shown in the legend.

plot_simulation_paths(
    paths=paths,
    target_zone=target_zone,
    start_year=sim_start_year,
    init_paths=init_paths,          # Overlay the initial path.
    data=data,                      # Overlay observed data.
    observed_cols=observed_cols,
    savepath_prefix="output/simulation/opt_path/market_z{}_{}".format(target_zone, sim_start_year)
)

plot_all_zones_lines(
    paths=paths,
    start_year=sim_start_year,
    zone_labels=zone_labels,
    legend_cols=7,
    savepath_prefix=f"output/simulation/opt_path/allzones_{sim_start_year}"
)

plot_all_zones_observed_lines(
    data=data,
    zone_col="zone",
    year_col="year",
    observed_cols=observed_cols,
    legend_cols=7,
    zone_order=Dzone_list,
    savepath_prefix=f"output/simulation/opt_path/allzones_observed"
)
