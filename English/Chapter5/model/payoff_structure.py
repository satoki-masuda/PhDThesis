import numpy as np
from utils.config_manager import ConfigManager

# 設定の読み込みと初期化
config = ConfigManager("config.yaml")
# 設定値の展開（後方互換性のため変数としても利用可能）
scale_pop_dense = config.scale_pop_dense
scale_pop = config.scale_pop
scale_los = config.scale_los
scale_dev_res = config.scale_dev_res
scale_dev_shop = config.scale_dev_shop
scale_price = config.scale_price
scale_land_demand = config.scale_land_demand

def compute_payoff_features(population, investment, development, price, demand, distCBD, eta=0.3):
    """
    Return a feature vector Psi(s, a) for use in linear payoff function:
    π = θ' * Psi(s, a)

    Parameters:
        population (float): population at time t
        investment (float): government infrastructure investment
        development (float): developer's land use expansion

    Returns:
        list[float]: feature vector [population, investment, development, population]
    """
    #population /= scale_pop
    effective_pop = population * (investment ** eta) / scale_pop
    investment /= scale_los
    development /= scale_dev_res
    price /= scale_price
    demand /= scale_land_demand
    #print("p:", population/investment, -investment, development)
    #print("d:", price * demand, -development, investment)

    return np.array([effective_pop, -investment, -investment*distCBD, development]), np.array([price * demand, -development, -development*distCBD, investment])
