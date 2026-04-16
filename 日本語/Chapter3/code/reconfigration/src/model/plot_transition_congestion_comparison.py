import argparse
import copy
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare transition-period congestion indicators across reconfiguration policies."
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.0,
        help="Background traffic ratio used in the macro model.",
    )
    parser.add_argument(
        "--best-evac-sequence",
        type=Path,
        default=None,
        help="CSV path for the best evacuation-oriented sequence.",
    )
    parser.add_argument(
        "--best-normal-sequence",
        type=Path,
        default=None,
        help="CSV path for the best normal-traffic-oriented sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and summary tables.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=100,
        help="Number of feasible random sequences used to compute the random-policy mean.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random feasible sequence generation.",
    )
    parser.add_argument(
        "--worst-objective",
        choices=["evac", "normal", "multi"],
        default="normal",
        help="Objective used to pick the worst random sequence, following cross-entropy.py.",
    )
    parser.add_argument(
        "--reconf-start-time",
        type=int,
        default=6 * 60,
        help="Reconfiguration start time in minutes.",
    )
    parser.add_argument(
        "--reconf-end-time",
        type=int,
        default=12 * 60,
        help="Reconfiguration end time in minutes.",
    )
    parser.add_argument(
        "--seq-interval",
        type=int,
        default=60,
        help="Control update interval in minutes.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="best_evac,no_reconfig,worst_random,best_normal,random_mean",
        help=(
            "Comma-separated list of policies to compute and plot. "
            "Available: no_policy,best_evac,no_reconfig,worst_random,best_normal,random_mean"
        ),
    )
    return parser.parse_args()


VALID_POLICIES = {
    "no_policy",
    "best_evac",
    "no_reconfig",
    "worst_random",
    "best_normal",
    "random_mean",
}


def parse_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [policy for policy in policies if policy not in VALID_POLICIES]
    if invalid:
        raise ValueError(
            f"Unknown policies: {', '.join(invalid)}. "
            f"Available: {', '.join(sorted(VALID_POLICIES))}"
        )
    if not policies:
        raise ValueError("At least one policy must be selected.")
    return policies


def load_sequence(sequence_csv: Path) -> np.ndarray:
    return pd.read_csv(sequence_csv, header=None).to_numpy(dtype=int)


def infer_target_graph(best_evac_sequence: np.ndarray, best_normal_sequence: np.ndarray) -> np.ndarray:
    target_evac = best_evac_sequence[-1].astype(int)
    target_normal = best_normal_sequence[-1].astype(int)
    if not np.array_equal(target_evac, target_normal):
        raise ValueError("Best evac and best normal sequences end at different target graphs.")
    return target_evac


def build_cross_entropy_helper(params, target_graph: np.ndarray, time_steps: int, seq_interval: int, reconf_start_time: int, reconf_end_time: int, objective_func: str):
    module_path = SCRIPT_DIR / "cross-entropy.py"
    spec = importlib.util.spec_from_file_location("cross_entropy_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load CrossEntropy from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CrossEntropy = module.CrossEntropy

    initial_graph = np.zeros_like(target_graph, dtype=int)
    return CrossEntropy(
        params=params,
        initial_graph=initial_graph.tolist(),
        target_graph=target_graph.tolist(),
        time_steps=time_steps,
        seq_interval=seq_interval,
        reconf_start_time=reconf_start_time,
        reconf_end_time=reconf_end_time,
        objective_func=objective_func,
    )


def default_sequence_path(background_ratio: float, objective: str) -> Path:
    return (
        SCRIPT_DIR
        / "../../output/mfd_dynamics/route_update_60min/demand_36h/93_max"
        / f"back{background_ratio}"
        / f"0_24_control6_{objective}"
        / "best_sequence.csv"
    ).resolve()


def default_output_dir(background_ratio: float) -> Path:
    return (SCRIPT_DIR / f"../../output/validation/transition_congestion/back{background_ratio}").resolve()


def sample_random_sequences(ce, population_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = ce.num_links * (ce.time_steps - 1)
    p_init = np.full(dim, 0.1)
    p0_idx = [
        t * ce.num_links + i
        for t in range(ce.time_steps - 1)
        for i in ce.const_edge_idx()
    ]
    p = np.array([0.0 if i in p0_idx else p_init[i] for i in range(dim)])

    def constraint_at(level: int):
        """Return the ZDD constraint for a given level."""
        if not isinstance(ce.zdd_constraints, list) or len(ce.zdd_constraints) <= level:
            raise IndexError(
                f"zdd_constraints does not contain level {level}; "
                f"available length={len(ce.zdd_constraints) if isinstance(ce.zdd_constraints, list) else 'N/A'}"
            )
        return ce.zdd_constraints[level]

    population = np.zeros((population_size, ce.num_links * (ce.time_steps - 1)), dtype=int)
    concat_list = []
    n_sample = 0
    while n_sample < population_size:
        tmp = (rng.random((100, ce.num_links)) < p[:ce.num_links]).astype(int)
        tmp = ce.array_to_zdd(tmp) & constraint_at(1)
        if len(tmp) > 0:
            arr = ce.zdd_to_array(tmp)
            concat_list.extend(arr)
            n_sample += len(arr)
    population[:, : ce.num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(population_size, ce.num_links)

    for t in range(1, ce.time_steps - 1):
        start = (t - 1) * ce.num_links
        end = t * ce.num_links
        group, indices = np.unique(population[:, start:end], axis=0, return_inverse=True)
        for idx, ind in enumerate(group):
            arr = []
            n_sample = 0
            target_count = int(np.sum(indices == idx))
            while n_sample < target_count:
                tmp = (rng.random((100, ce.num_links)) < p[t * ce.num_links : (t + 1) * ce.num_links]).astype(int)
                zdd_tmp = (
                    ce.array_to_zdd(tmp)
                    & constraint_at(t + 1)
                    & ce.reconf.transition(ce.array_to_zdd(ind), ce.reconf.constraints)
                )
                n = min(len(zdd_tmp), target_count - n_sample)
                if n > 0:
                    arr.extend(ce.zdd_to_array(zdd_tmp)[:n])
                    n_sample += n
            population[indices == idx, t * ce.num_links : (t + 1) * ce.num_links] = np.concatenate(np.array(arr), axis=0).reshape(target_count, ce.num_links)

    population = np.hstack((population, np.tile(np.array(ce.target_graph, dtype=int), (population_size, 1))))
    return [population[i].reshape((ce.time_steps, ce.num_links)).astype(int) for i in range(population_size)]


def apply_progressive_sequence(sim, params, original_capacity, link_indices, control_sequence: np.ndarray, reconf_start_time: int, reconf_end_time: int, seq_interval: int):
    for _ in range(sim.sim_start_step, sim.sim_end_step):
        if (sim.step % seq_interval == 0) and (reconf_start_time < sim.step <= reconf_end_time):
            sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
            step_idx = (sim.step // seq_interval) - (reconf_start_time // seq_interval) - 1
            contraflow_idx = control_sequence[step_idx]
            for i, j in [link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
        sim.step_simulation()


def apply_no_reconfig(sim, params, original_capacity, link_indices, target_graph: np.ndarray, reconf_end_time: int):
    for _ in range(sim.sim_start_step, sim.sim_end_step):
        if sim.step == reconf_end_time:
            sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
            for i, j in [link_indices[idx] for idx in np.where(target_graph == 1)[0]]:
                sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
        sim.step_simulation()


def simulate_policy(policy_name: str, params, original_capacity, link_indices, target_graph: np.ndarray, reconf_start_time: int, reconf_end_time: int, seq_interval: int, control_sequence: np.ndarray | None):
    from mfd_dynamics import MFD_Dynamics

    params_copy = copy.deepcopy(params)
    params_copy.max_boundary_capacity = copy.deepcopy(original_capacity)
    sim = MFD_Dynamics(params_copy, output_path=None)

    if policy_name == "no_policy":
        sim.run_simulation()
    elif policy_name == "no_reconfig":
        apply_no_reconfig(sim, params_copy, original_capacity, link_indices, target_graph, reconf_end_time)
    else:
        apply_progressive_sequence(sim, params_copy, original_capacity, link_indices, control_sequence, reconf_start_time, reconf_end_time, seq_interval)

    return sim


def sim_to_timeseries(sim, params, policy_name: str) -> pd.DataFrame:
    tgrid = np.arange(params.simulation_start_time, params.simulation_end_time, params.sampling_time)
    n_check = np.array(sim.n_check, dtype=float)
    throughput_evac = np.array([np.sum(x) for x in sim.throughput_list], dtype=float)
    throughput_normal = np.array([np.sum(x) for x in sim.throughput_background_list], dtype=float)
    evac_cumsum = params.Q[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    normal_cumsum = params.Q_background[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    total_cumsum = evac_cumsum + normal_cumsum
    evac_avg_tt = np.divide(
        np.array(sim.tet_list, dtype=float),
        evac_cumsum,
        out=np.full_like(np.array(sim.tet_list, dtype=float), np.nan),
        where=evac_cumsum > 0,
    )
    normal_avg_tt = np.divide(
        np.array(sim.ttt_list, dtype=float),
        normal_cumsum,
        out=np.full_like(np.array(sim.ttt_list, dtype=float), np.nan),
        where=normal_cumsum > 0,
    )
    total_avg_tt = np.divide(
        np.array(sim.tet_list, dtype=float) + np.array(sim.ttt_list, dtype=float),
        total_cumsum,
        out=np.full_like(np.array(sim.tet_list, dtype=float), np.nan),
        where=total_cumsum > 0,
    )

    df = pd.DataFrame(
        {
            "policy": policy_name,
            "time_min": tgrid,
            "time_hour": tgrid / 60.0,
            "evac_avg_tt": evac_avg_tt,
            "normal_avg_tt": normal_avg_tt,
            "total_avg_tt": total_avg_tt,
            "n_over_njam_max": np.max(n_check, axis=1),
            "n_over_njam_mean": np.mean(n_check, axis=1),
            "throughput_evac_total": throughput_evac,
            "throughput_normal_total": throughput_normal,
            "throughput_all_total": throughput_evac + throughput_normal,
        }
    )
    return df


def objective_value_from_sim(sim, reconf_end_time: int, simulation_start_time: int, objective_func: str) -> float:
    if objective_func == "evac":
        return float(sim.tet_list[-1])
    if objective_func == "multi":
        idx = reconf_end_time - simulation_start_time - 1
        return float(0.15 * sim.tet_list[-1] + 0.85 * sim.ttt_list[idx])
    idx = reconf_end_time - simulation_start_time - 1
    return float(sim.ttt_list[idx])


def summarize_final_metrics(sim, params, reconf_end_time: int) -> dict:
    idx = reconf_end_time - params.simulation_start_time - 1
    normal_cumsum = params.Q_background[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    evac_cumsum = params.Q[:, :, params.simulation_start_time : params.simulation_end_time].sum(axis=(0, 1)).cumsum()
    normal_att = float(sim.ttt_list[idx] / normal_cumsum[idx]) if len(normal_cumsum) > idx and normal_cumsum[idx] > 0 else np.nan
    evac_att = float(sim.tet_list[-1] / evac_cumsum[-1]) if len(evac_cumsum) > 0 and evac_cumsum[-1] > 0 else np.nan

    return {
        "final_evac_tet": float(sim.tet_list[-1]),
        "final_normal_ttt_at_reconf_end": float(sim.ttt_list[idx]),
        "final_total_travel_time": float(sim.tet_list[-1] + sim.ttt_list[-1]),
        "final_evac_att": evac_att,
        "final_normal_att_at_reconf_end": normal_att,
        "peak_n_over_njam_max": float(np.max(np.array(sim.n_check, dtype=float))),
        "mean_n_over_njam_max": float(np.mean(np.max(np.array(sim.n_check, dtype=float), axis=1))),
        "peak_throughput_all": float(np.max(np.array([np.sum(x) + np.sum(y) for x, y in zip(sim.throughput_list, sim.throughput_background_list)], dtype=float))),
    }


def plot_series(timeseries_df: pd.DataFrame, output_path: Path, reconf_start_time: int, reconf_end_time: int, selected_policies: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default_policy_order = [
        "best_evac",
        "best_normal",
        "no_reconfig",
        "no_policy",
        "random_mean",
        "worst_random",
    ]
    policy_order = [
        policy
        for policy in default_policy_order
        if policy in selected_policies and policy in set(timeseries_df["policy"])
    ]
    colors = {
        "best_evac": "tab:red",
        "best_normal": "tab:green",
        "no_reconfig": "tab:blue",
        "no_policy": "tab:gray",
        "random_mean": "tab:orange",
        "worst_random": "tab:purple",
    }
    line_alpha = 0.75
    line_width = 1.5

    def add_reconf_window(ax):
        ax.axvspan(reconf_start_time / 60.0, reconf_end_time / 60.0, color="lightgray", alpha=0.3)
        ax.grid(True, alpha=0.25)

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    for policy in policy_order:
        df = timeseries_df[timeseries_df["policy"] == policy]
        if df.empty:
            continue
        ax.plot(df["time_hour"], df["total_avg_tt"], label=policy, color=colors[policy], linewidth=line_width, alpha=line_alpha)
    ax.set_ylabel("Overall average travel time (min)")
    ax.set_xlabel("Time (hour)")
    ax.legend(ncol=3)
    add_reconf_window(ax)
    ax.set_title("Travel-time indicator during and after transition")
    plt.tight_layout()
    plt.savefig(output_path / "travel_time_comparison.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for policy in policy_order:
        df = timeseries_df[timeseries_df["policy"] == policy]
        if df.empty:
            continue
        axes[0].plot(df["time_hour"], df["n_over_njam_max"], label=policy, color=colors[policy], linewidth=line_width, alpha=line_alpha)
        axes[1].plot(df["time_hour"], df["n_over_njam_mean"], label=policy, color=colors[policy], linewidth=line_width, alpha=line_alpha)
    axes[0].set_ylabel("Max N/N_jam across zones")
    axes[1].set_ylabel("Mean N/N_jam across zones")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Time (hour)")
    axes[0].legend(ncol=3)
    add_reconf_window(axes[0])
    add_reconf_window(axes[1])
    fig.suptitle("Congestion indicators during and after transition", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path / "n_over_njam_comparison.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    metric_specs = [
        ("throughput_evac_total", "Evacuation throughput (veh/min)"),
        ("throughput_normal_total", "Normal throughput (veh/min)"),
        ("throughput_all_total", "Total throughput (veh/min)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for policy in policy_order:
            df = timeseries_df[timeseries_df["policy"] == policy]
            if df.empty:
                continue
            ax.plot(df["time_hour"], df[metric], label=policy, color=colors[policy], linewidth=line_width, alpha=line_alpha)
        ax.set_ylabel(ylabel)
        add_reconf_window(ax)
    axes[-1].set_xlabel("Time (hour)")
    axes[0].legend(ncol=3)
    fig.suptitle("Throughput indicators during and after transition", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path / "throughput_comparison.png", dpi=300)
    plt.close(fig)


def write_readme(output_dir: Path, args, worst_sequence_path: Path):
    text = (
        "Transition congestion comparison\n"
        "- Scenarios: best_evac, best_normal, no_reconfig, no_policy, random_mean, worst_random.\n"
        "- no_reconfig switches directly to the common target graph at the end of the transition horizon.\n"
        "- no_policy keeps the initial no-contraflow network throughout the full simulation horizon.\n"
        f"- Random sequences: {args.random_samples} feasible sequences sampled with seed {args.random_seed}.\n"
        f"- Worst random sequence is selected by the {args.worst_objective} objective, following cross-entropy.py.\n"
        f"- Reconfiguration window: {args.reconf_start_time/60:.1f}h to {args.reconf_end_time/60:.1f}h.\n"
        f"- Saved worst random sequence: {worst_sequence_path}\n"
        f"- Selected policies: {args.policies}\n"
    )
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    selected_policies = parse_policies(args.policies)
    if args.best_evac_sequence is None:
        args.best_evac_sequence = default_sequence_path(args.background_ratio, "evac")
    if args.best_normal_sequence is None:
        args.best_normal_sequence = default_sequence_path(args.background_ratio, "normal")
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.background_ratio)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from parameters_ndp import Parameters

    params = Parameters(background_ratio=args.background_ratio)

    best_evac_sequence = load_sequence(args.best_evac_sequence.resolve())
    best_normal_sequence = load_sequence(args.best_normal_sequence.resolve())
    target_graph = infer_target_graph(best_evac_sequence, best_normal_sequence)
    time_steps = int((args.reconf_end_time - args.reconf_start_time) // args.seq_interval)

    ce_helper = build_cross_entropy_helper(
        params=params,
        target_graph=target_graph,
        time_steps=time_steps,
        seq_interval=args.seq_interval,
        reconf_start_time=args.reconf_start_time,
        reconf_end_time=args.reconf_end_time,
        objective_func=args.worst_objective,
    )
    original_capacity = copy.deepcopy(ce_helper.original_capacity)
    link_indices = ce_helper.link_indices

    scenario_sims = {}
    if "best_evac" in selected_policies:
        scenario_sims["best_evac"] = simulate_policy(
            "best_evac",
            params,
            original_capacity,
            link_indices,
            target_graph,
            args.reconf_start_time,
            args.reconf_end_time,
            args.seq_interval,
            best_evac_sequence,
        )
    if "best_normal" in selected_policies:
        scenario_sims["best_normal"] = simulate_policy(
            "best_normal",
            params,
            original_capacity,
            link_indices,
            target_graph,
            args.reconf_start_time,
            args.reconf_end_time,
            args.seq_interval,
            best_normal_sequence,
        )
    if "no_reconfig" in selected_policies:
        scenario_sims["no_reconfig"] = simulate_policy(
            "no_reconfig",
            params,
            original_capacity,
            link_indices,
            target_graph,
            args.reconf_start_time,
            args.reconf_end_time,
            args.seq_interval,
            None,
        )
    if "no_policy" in selected_policies:
        scenario_sims["no_policy"] = simulate_policy(
            "no_policy",
            params,
            original_capacity,
            link_indices,
            target_graph,
            args.reconf_start_time,
            args.reconf_end_time,
            args.seq_interval,
            None,
        )

    worst_sequence_path = output_dir / "worst_random_sequence.csv"
    if "worst_random" in selected_policies or "random_mean" in selected_policies:
        random_sequences = sample_random_sequences(ce_helper, args.random_samples, args.random_seed)
        random_series = []
        worst_sequence = None
        worst_score = -np.inf
        for idx, sequence in enumerate(random_sequences):
            sim = simulate_policy(
                f"random_{idx}",
                params,
                original_capacity,
                link_indices,
                target_graph,
                args.reconf_start_time,
                args.reconf_end_time,
                args.seq_interval,
                sequence,
            )
            if "random_mean" in selected_policies:
                random_series.append(sim_to_timeseries(sim, params, f"random_{idx}"))
            score = objective_value_from_sim(
                sim,
                reconf_end_time=args.reconf_end_time,
                simulation_start_time=params.simulation_start_time,
                objective_func=args.worst_objective,
            )
            if score > worst_score:
                worst_score = score
                worst_sequence = sequence.copy()
                if "worst_random" in selected_policies:
                    scenario_sims["worst_random"] = sim

        if "random_mean" in selected_policies:
            random_stack = pd.concat(random_series, ignore_index=True)
            random_mean = (
                random_stack.groupby("time_min", as_index=False)[
                    [
                        "time_hour",
                        "evac_avg_tt",
                        "normal_avg_tt",
                        "total_avg_tt",
                        "n_over_njam_max",
                        "n_over_njam_mean",
                        "throughput_evac_total",
                        "throughput_normal_total",
                        "throughput_all_total",
                    ]
                ]
                .mean()
            )
            random_mean["policy"] = "random_mean"
        else:
            random_mean = None

        if worst_sequence is not None:
            pd.DataFrame(worst_sequence).to_csv(worst_sequence_path, index=False, header=False)
    else:
        random_mean = None

    timeseries_frames = [sim_to_timeseries(sim, params, policy) for policy, sim in scenario_sims.items()]
    if random_mean is not None:
        timeseries_frames.append(random_mean)
    timeseries_df = pd.concat(timeseries_frames, ignore_index=True)
    timeseries_df.to_csv(output_dir / "policy_timeseries.csv", index=False)

    summary_rows = []
    for policy, sim in scenario_sims.items():
        row = {"policy": policy}
        row.update(summarize_final_metrics(sim, params, args.reconf_end_time))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("policy")
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    plot_series(timeseries_df, output_dir, args.reconf_start_time, args.reconf_end_time, selected_policies)
    write_readme(output_dir, args, worst_sequence_path)

    print(f"Saved transition congestion comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
