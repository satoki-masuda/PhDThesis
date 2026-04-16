"""局所均衡計算で使う best response 計算ルーチン。"""

import numpy as np
from copy import deepcopy
import ray
from scipy.optimize import minimize
from model.forward_simulation import ForwardSimulation
from simulation.policy_override import PolicyOverride


@ray.remote
def forward_simulation_remote(**kwargs):
    """ForwardSimulation を Ray ワーカー上で実行するための薄いラッパー。"""
    sim = ForwardSimulation(**kwargs)
    out = sim.simulate()
    return out

def evaluate_actor_value(actor, a_val, snap, H, n_forward, beta, models, features, df, zone_convert, area_list, seed, theta):
    """現在状態で actor が所与の行動を選んだときの期待価値を計算する。"""
    gov_wrap = PolicyOverride(models["gov"])
    dev_wrap = PolicyOverride(models["dev"])
    t0, z = 0, snap["zone"]
    if actor=="gov":
        gov_wrap.set_override(t0, z, "gov", a_val)
    else:
        dev_wrap.set_override(t0, z, "dev", a_val)

    futures = [forward_simulation_remote.remote(N0=snap["N"], I0=snap["I"], P0=snap["P"], development_lists=deepcopy(snap["D_hist"]),
                                                zone=z, start_year=snap["year"], data_final_year=snap["data_final_year"],
                                                T=H, discount_factor=beta,
                                                gov_model=gov_wrap, dev_model=dev_wrap,
                                                base_gov_model=models["base_gov"], base_dev_model=models["base_dev"],
                                                transition_model=models["trans"], features=features, df=df,
                                                zone_convert_dict=zone_convert, area_list=area_list,
                                                master_seed=seed + sim_id,
                                                transition_type="deterministic"
                                                ) for sim_id in range(n_forward)]
    W_list = ray.get(futures)
    if actor=="gov":
        W_array = np.array([out["W_gov"] for out in W_list])
    else:
        W_array = np.array([out["W_dev"] for out in W_list])
    return np.dot(np.mean(W_array, axis=0), theta[actor].reshape(-1,1))

def best_response_local(actor, bounds, init_a, snap, H, n_forward, beta, models, features, df, zone_convert, area_list, seed, theta):
    """1主体だけ行動を変えたときの局所的な best response を数値最適化で求める。"""
    #scale = scale_los if actor=="gov" else scale_dev_res
    def neg_obj(a_scalar):
        a = float(a_scalar[0]) #* scale
        v = evaluate_actor_value(actor, a, snap, H, n_forward, beta, models, features, df, zone_convert, area_list, seed, theta)
        return -float(v)

    res = minimize(neg_obj, x0=np.array([init_a]), method="L-BFGS-B", bounds=[bounds], options={"maxiter":50})
    #res = minimize(neg_obj, x0=np.zeros_like(init_a), method="Nelder-Mead", options={"maxiter":100})
    '''
    if not res.success:
        from scipy.optimize import minimize as _min
        def neg_obj_clip(a):
            a = np.clip(a[0], bounds[0], bounds[1])
            return neg_obj([a])
        res = _min(neg_obj_clip, x0=np.array([init_a]), method="Nelder-Mead", options={"maxiter":120})
    '''
    a_star = float(np.clip(res.x[0], bounds[0], bounds[1])) #* scale
    print(f"      Best response for {actor}: a*={a_star:.4f}, value={-res.fun:.4f}")
    return a_star, float(res.fun)
