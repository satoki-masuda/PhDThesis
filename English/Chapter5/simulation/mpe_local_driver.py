"""単一 market 上で局所 MPE を組み立てるドライバ。"""

import numpy as np
from copy import deepcopy
from simulation.best_response_mpe import best_response_local
from model.forward_simulation import ForwardSimulation
from simulation.policy_override import PolicyOverride

def solve_local_mpe_for_market(
    market, init_paths, ref_year, H, n_forward, beta,
    models, features, df, zone_convert, area_list,
    bounds_gov, bounds_dev, seed=12345, theta=None
):
    """観測パスを初期値にして各期の局所 best response を順に解く。"""
    z, year = market
    T = init_paths["population_path"].shape[0]
    N_path = deepcopy(init_paths["population_path"])
    I_path = deepcopy(init_paths["investment_path"])
    D_path = deepcopy(init_paths["development_path"])
    P_path = deepcopy(init_paths["price_path"])

    def get_hist(t):
        hist = []
        for lag in range(ref_year):
            idx = max(t-1-lag, 0)
            hist.append(D_path[idx,:].tolist())
        hist.reverse()
        return hist
    # 1期から順に最適応答を計算し、パスを更新
    for t in range(T):
        print(f"  Time {t+1}")
        snap = dict(
            N=N_path[t,:].tolist(),
            I=I_path[t,:].tolist() if t>0 else I_path[0,:].tolist(),
            D_hist=get_hist(t),
            P=P_path[t,:].tolist(),
            zone=z, year=year+t, data_final_year=models["trans"].end_year
        )
        # gov
        init_a = float(I_path[t,z])
        # 現在の状態における最適応答を計算 (他のプレーヤーや将来の行動は観測均衡に固定)
        a_star, _ = best_response_local("gov", bounds_gov, init_a, snap,
                                        H=min(H, T-t), n_forward=n_forward, beta=beta,
                                        models=models, features=features, df=df,
                                        zone_convert=zone_convert, area_list=area_list,
                                        seed=seed+101*t, theta=theta)
        I_path[t,z] = a_star
        # dev
        # 現在の状態における最適応答を計算 (他のプレーヤーや将来の行動は観測均衡に固定)
        init_d = float(D_path[t,z])
        a_star, _ = best_response_local("dev", bounds_dev, init_d, snap,
                                        H=min(H, T-t), n_forward=n_forward, beta=beta,
                                        models=models, features=features, df=df,
                                        zone_convert=zone_convert, area_list=area_list,
                                        seed=seed+202*t, theta=theta)
        D_path[t,z] = a_star

    #  T 期フル評価
    gov_wrap = PolicyOverride(models["gov"]); dev_wrap = PolicyOverride(models["dev"])
    for t in range(T):
        gov_wrap.set_override(t, z, "gov", I_path[t,z])
        dev_wrap.set_override(t, z, "dev", D_path[t,z])

    sim = ForwardSimulation(
        N0=N_path[0,:].tolist(), I0=I_path[0,:].tolist(), P0=P_path[0,:].tolist(),
        development_lists=[row.tolist() for row in D_path[:ref_year]],
        zone=z, start_year=year, data_final_year=models["trans"].end_year,
        T=T, discount_factor=beta,
        gov_model=gov_wrap, dev_model=dev_wrap,
        base_gov_model=models["base_gov"], base_dev_model=models["base_dev"],
        transition_model=models["trans"], features=features, df=df,
        zone_convert_dict=zone_convert, area_list=area_list,
        master_seed=seed,
        transition_type="deterministic"
    )
    out = sim.simulate()
    return dict(I_path=I_path, D_path=D_path, result=out)
