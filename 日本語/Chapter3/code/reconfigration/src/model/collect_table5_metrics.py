"""Batch evaluation helpers for reproducing the Table 5-style comparisons."""

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))


SCENARIOS = {
    "93_max_1.0": {
        "demand_variation": "93_max",
        "background_ratio": 1.0,
        "target_graph": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    },
    "93_max_0.9": {
        "demand_variation": "93_max",
        "background_ratio": 0.9,
        "target_graph": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "93_max_0.8": {
        "demand_variation": "93_max",
        "background_ratio": 0.8,
        "target_graph": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "53_0.95_1.0": {
        "demand_variation": "53_0.95",
        "background_ratio": 1.0,
        "target_graph": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}


def parse_args():
    """Parse command-line arguments for scenario-wise policy evaluation."""
    parser = argparse.ArgumentParser(description="Compute Table 5 metrics for multiple scenarios and policies.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--output-root", type=Path, default=(SCRIPT_DIR / "../../output/table5_batch").resolve())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--max-cpu", type=int, default=25)
    parser.add_argument("--max-sim-evals", type=int, default=1000)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--nmin", type=int, default=100)
    parser.add_argument("--nmax", type=int, default=300)
    parser.add_argument("--nunit", type=int, default=100)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--reconf-start-time", type=int, default=6 * 60)
    parser.add_argument("--reconf-end-time", type=int, default=12 * 60)
    parser.add_argument("--seq-interval", type=int, default=60)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def load_cross_entropy_module():
    """Load ``cross-entropy.py`` as a module despite the hyphenated filename."""
    module_path = SCRIPT_DIR / "cross-entropy.py"
    spec = importlib.util.spec_from_file_location("cross_entropy_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_output_dir(output_root: Path, scenario: str) -> Path:
    """Return the output directory dedicated to one scenario."""
    return output_root / scenario


def make_cross_entropy(module, params, target_graph, objective: str, args, output_dir: Path):
    """Instantiate the Chapter 3 cross-entropy optimizer with shared settings."""
    CrossEntropy = module.CrossEntropy
    initial_graph = [0] * len(target_graph)
    return CrossEntropy(
        params=copy.deepcopy(params),
        initial_graph=initial_graph,
        target_graph=target_graph,
        time_steps=int((args.reconf_end_time - args.reconf_start_time) // args.seq_interval),
        seq_interval=args.seq_interval,
        reconf_start_time=args.reconf_start_time,
        reconf_end_time=args.reconf_end_time,
        objective_func=objective,
        output_path=output_dir,
    )


@ray.remote(num_cpus=1)
def simulate_random_sequence_worker(
    params_dict,
    target_graph,
    seq_interval,
    reconf_start_time,
    reconf_end_time,
    objective_func,
    original_capacity,
    link_indices,
    control_sequence,
):
    from mfd_dynamics import MFD_Dynamics

    params = copy.deepcopy(params_dict)
    params.max_boundary_capacity = copy.deepcopy(original_capacity)
    sim = MFD_Dynamics(params, output_path=None)
    for _ in range(sim.sim_start_step, sim.sim_end_step):
        if (sim.step % seq_interval == 0) and (reconf_start_time < sim.step <= reconf_end_time):
            sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
            step_idx = (sim.step // seq_interval) - (reconf_start_time // seq_interval) - 1
            contraflow_idx = control_sequence[step_idx]
            for i, j in [link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
        sim.step_simulation()

    idx = reconf_end_time - params.simulation_start_time - 1
    evac_cumsum = params.Q[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    normal_cumsum = params.Q_background[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    evac_demand = float(evac_cumsum[-1]) if len(evac_cumsum) > 0 else 0.0
    normal_demand = float(normal_cumsum[idx]) if len(normal_cumsum) > idx and idx >= 0 else 0.0
    tet_total = float(sim.tet_list[-1])
    ttt_total = float(sim.ttt_list[idx])
    if objective_func == "evac":
        score = tet_total
    elif objective_func == "normal":
        score = ttt_total
    elif objective_func == "multi":
        score = 0.15 * tet_total + 0.85 * ttt_total
    else:
        raise ValueError(f"Unsupported objective_func: {objective_func}")
    return {
        "tet_total": tet_total,
        "ttt_total": ttt_total,
        "att_evac": tet_total / evac_demand if evac_demand > 0 else np.nan,
        "att_normal": ttt_total / normal_demand if normal_demand > 0 else np.nan,
        "objective_total": score,
    }


def run_optimal_policy(module, params, target_graph, objective: str, args, output_dir: Path):
    """Optimize and evaluate one objective-specific reconfiguration policy."""
    ce = make_cross_entropy(module, params, target_graph, objective, args, output_dir)
    start = time.time()
    best_sequence, best_cost = ce.fully_adaptive_cross_entropy(
        elite_count=args.elite_count,
        Nmin=args.nmin,
        Nmax=args.nmax,
        Nunit=args.nunit,
        d=args.d,
        max_cpu=args.max_cpu,
        max_sim_evals=None if args.max_sim_evals <= 0 else args.max_sim_evals,
        disable_early_stop=args.disable_early_stop,
    )
    optimization_time = time.time() - start
    metrics = ce.sim_best_policy(best_sequence, plot=False)
    metrics.update(
        {
            "policy": f"optimal_{objective}",
            "objective": objective,
            "best_cost": float(best_cost),
            "optimization_time_sec": optimization_time,
        }
    )
    return metrics, best_sequence


def run_fixed_policy(module, params, target_graph, policy: str, args, output_dir: Path):
    """Evaluate deterministic baseline policies such as no-policy or static target."""
    ce = make_cross_entropy(module, params, target_graph, "normal", args, output_dir)
    start = time.time()
    if policy == "no_policy":
        metrics = ce.sim_no_policy(plot=False)
    elif policy == "no_reconfig":
        metrics = ce.sim_no_reconfig(target_graph, plot=False)
    elif policy == "random_policy":
        random_metrics = run_random_policy_parallel(
            params=params,
            target_graph=target_graph,
            args=args,
            output_dir=output_dir,
            module=module,
        )
        metrics = random_metrics
    else:
        raise ValueError(f"Unsupported fixed policy: {policy}")
    metrics.update(
        {
            "policy": policy,
            "objective": "",
            "best_cost": np.nan,
            "optimization_time_sec": time.time() - start,
        }
    )
    return metrics


def run_random_policy_parallel(module, params, target_graph, args, output_dir: Path):
    """Sample and evaluate many random feasible policies in parallel with Ray."""
    ce = make_cross_entropy(module, params, target_graph, "normal", args, output_dir)
    dim = ce.num_links * (ce.time_steps - 1)
    p_init = np.full(dim, 0.1)
    p0_idx = [t * ce.num_links + i for t in range(ce.time_steps - 1) for i in ce.const_edge_idx()]
    p_init = np.array([0.0 if i in p0_idx else p_init[i] for i in range(dim)], dtype=float)

    def constraint_at(level: int):
        if not isinstance(ce.zdd_constraints, list) or len(ce.zdd_constraints) <= level:
            raise IndexError(
                f"zdd_constraints does not contain level {level}; "
                f"available length={len(ce.zdd_constraints) if isinstance(ce.zdd_constraints, list) else 'N/A'}"
            )
        return ce.zdd_constraints[level]

    population = np.zeros((args.random_samples, ce.num_links * (ce.time_steps - 1)))
    concat_list = []
    n_sample = 0
    while n_sample < args.random_samples:
        tmp = (np.random.rand(100, ce.num_links) < p_init[:ce.num_links]).astype(int)
        tmp = ce.array_to_zdd(tmp) & constraint_at(1)
        if len(tmp) > 0:
            concat_list.extend(ce.zdd_to_array(tmp))
            n_sample += len(tmp)
    population[:, : ce.num_links] = np.concatenate(np.array(concat_list)[: args.random_samples], axis=0).reshape(
        args.random_samples, ce.num_links
    )

    for t in range(1, ce.time_steps - 1):
        group, indices = np.unique(population[:, ((t - 1) * ce.num_links) : (t * ce.num_links)], axis=0, return_inverse=True)
        for idx, ind in enumerate(group):
            arr = []
            n_sample = 0
            target_count = int(np.sum(indices == idx))
            while n_sample < target_count:
                tmp = (np.random.rand(100, ce.num_links) < p_init[(t * ce.num_links) : ((t + 1) * ce.num_links)]).astype(int)
                zdd_tmp = (
                    ce.array_to_zdd(tmp)
                    & constraint_at(t + 1)
                    & ce.reconf.transition(ce.array_to_zdd(ind), ce.reconf.constraints)
                )
                n = min(len(zdd_tmp), target_count - n_sample)
                if n > 0:
                    arr.extend(ce.zdd_to_array(zdd_tmp)[:n])
                    n_sample += n
            population[indices == idx, (t * ce.num_links) : ((t + 1) * ce.num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, ce.num_links)

    population = np.hstack((population, np.tile(ce.target_graph, (args.random_samples, 1))))
    random_sequences = [population[i].reshape((ce.time_steps, ce.num_links)).astype(int) for i in range(args.random_samples)]

    ray.init(num_cpus=args.max_cpu, ignore_reinit_error=True)
    try:
        tasks = [
            simulate_random_sequence_worker.remote(
                params,
                target_graph,
                args.seq_interval,
                args.reconf_start_time,
                args.reconf_end_time,
                "normal",
                ce.original_capacity,
                ce.link_indices,
                seq,
            )
            for seq in random_sequences
        ]
        results = ray.get(tasks)
    finally:
        ray.shutdown()

    tet_totals = np.array([res["tet_total"] for res in results], dtype=float)
    ttt_totals = np.array([res["ttt_total"] for res in results], dtype=float)
    att_evac = np.array([res["att_evac"] for res in results], dtype=float)
    att_normal = np.array([res["att_normal"] for res in results], dtype=float)
    objective_totals = np.array([res["objective_total"] for res in results], dtype=float)
    return {
        "tet_total": float(np.mean(tet_totals)),
        "ttt_total": float(np.mean(ttt_totals)),
        "att_evac": float(np.nanmean(att_evac)),
        "att_normal": float(np.nanmean(att_normal)),
        "att_evac_std": float(np.nanstd(att_evac)),
        "att_normal_std": float(np.nanstd(att_normal)),
        "att_evac_min": float(np.nanmin(att_evac)),
        "att_evac_max": float(np.nanmax(att_evac)),
        "att_normal_min": float(np.nanmin(att_normal)),
        "att_normal_max": float(np.nanmax(att_normal)),
        "n_samples": args.random_samples,
        "objective_total": float(np.mean(objective_totals)),
    }


def main():
    """Run all requested policy evaluations and save a combined summary CSV."""
    args = parse_args()
    scenario = SCENARIOS[args.scenario]

    module = load_cross_entropy_module()
    from parameters_ndp import Parameters

    params = Parameters(
        demand_variation=scenario["demand_variation"],
        background_ratio=scenario["background_ratio"],
    )
    target_graph = np.array(scenario["target_graph"], dtype=int)

    output_dir = scenario_output_dir(args.output_root, args.scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or (args.output_root / "table5_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    '''
    fixed_policies = ["no_policy", "no_reconfig", "random_policy"]
    for policy in fixed_policies:
        policy_dir = output_dir / policy
        policy_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {args.scenario} / {policy}")
        metrics = run_fixed_policy(module, params, target_graph, policy, args, policy_dir)
        metrics.update(
            {
                "scenario": args.scenario,
                "demand_variation": scenario["demand_variation"],
                "background_ratio": scenario["background_ratio"],
                "seed": args.seed,
                "random_samples": args.random_samples if policy == "random_policy" else np.nan,
                "max_sim_evals": args.max_sim_evals,
            }
        )
        rows.append(metrics)
    '''
    #for objective in ["normal", "evac", "multi"]:
    for objective in ["multi"]:
        policy_dir = output_dir / f"optimal_{objective}"
        policy_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {args.scenario} / optimal_{objective}")
        metrics, best_sequence = run_optimal_policy(module, params, target_graph, objective, args, policy_dir)
        pd.DataFrame(best_sequence).to_csv(policy_dir / "best_sequence.csv", index=False, header=False)
        metrics.update(
            {
                "scenario": args.scenario,
                "demand_variation": scenario["demand_variation"],
                "background_ratio": scenario["background_ratio"],
                "seed": args.seed,
                "random_samples": np.nan,
                "max_sim_evals": args.max_sim_evals,
            }
        )
        rows.append(metrics)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        combined = pd.concat([existing, result_df], ignore_index=True)
        combined.to_csv(summary_csv, index=False)
    else:
        result_df.to_csv(summary_csv, index=False)

    print(f"Saved scenario summary to {output_dir / 'scenario_summary.csv'}")
    print(f"Appended combined summary to {summary_csv}")


if __name__ == "__main__":
    main()
