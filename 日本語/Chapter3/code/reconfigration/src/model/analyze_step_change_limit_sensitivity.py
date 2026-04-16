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
import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))


DEFAULT_TARGET_GRAPH = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the maximum number of link changes allowed per transition step."
    )
    parser.add_argument("--objective", choices=["evac", "normal", "multi"], default="normal")
    parser.add_argument("--background-ratio", type=float, default=0.8)
    parser.add_argument("--target-graph", type=str, default=",".join(map(str, DEFAULT_TARGET_GRAPH)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--change-limits", type=str, default="1,2,3")
    parser.add_argument("--reconf-start-time", type=int, default=6 * 60)
    parser.add_argument("--reconf-end-time", type=int, default=12 * 60)
    parser.add_argument("--seq-interval", type=int, default=60)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--nmin", type=int, default=100)
    parser.add_argument("--nmax", type=int, default=300)
    parser.add_argument("--nunit", type=int, default=100)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--max-cpu", type=int, default=1)
    parser.add_argument("--max-sim-evals", type=int, default=1000)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(SCRIPT_DIR / "../../output/validation/step_change_limit_sensitivity").resolve(),
    )
    return parser.parse_args()


def parse_binary_vector(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if any(v not in (0, 1) for v in values):
        raise ValueError("target_graph must contain only 0 or 1.")
    return values


def parse_limits(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if any(v < 1 for v in values):
        raise ValueError("change limits must be positive integers.")
    return values


def load_cross_entropy_module():
    module_path = SCRIPT_DIR / "cross-entropy.py"
    spec = importlib.util.spec_from_file_location("cross_entropy_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def simulate_sequence(best_control_sequence, ce):
    module = load_cross_entropy_module()
    MFD_Dynamics = module.MFD_Dynamics

    params_copy = copy.deepcopy(ce.params)
    original_capacity = copy.deepcopy(ce.original_capacity)
    params_copy.max_boundary_capacity = copy.deepcopy(original_capacity)
    sim = MFD_Dynamics(params_copy, output_path=None)

    for _ in range(sim.sim_start_step, sim.sim_end_step):
        if (sim.step % ce.seq_interval == 0) and (ce.reconf_start_time < sim.step <= ce.reconf_end_time):
            sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
            step_idx = (sim.step // ce.seq_interval) - (ce.reconf_start_time // ce.seq_interval) - 1
            contraflow_idx = best_control_sequence[step_idx]
            for i, j in [ce.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                sim.params.max_boundary_capacity[i, j] *= ce.params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= (2 - ce.params.contra_ratio)
        sim.step_simulation()
    return sim


def summarize_metrics(sim, params, reconf_end_time: int, simulation_start_time: int, objective: str):
    idx = reconf_end_time - simulation_start_time - 1
    final_ttt = float(sim.ttt_list[idx])
    final_tet = float(sim.tet_list[-1])
    peak_congestion = float(np.max(np.array(sim.n_check, dtype=float)))
    mean_peak_congestion = float(np.mean(np.max(np.array(sim.n_check, dtype=float), axis=1)))

    objective_value = final_tet if objective == "evac" else final_ttt
    return {
        "objective_metric": objective_value,
        "final_normal_ttt_at_reconf_end": final_ttt,
        "final_evac_tet": final_tet,
        "peak_n_over_njam_max": peak_congestion,
        "mean_n_over_njam_max": mean_peak_congestion,
    }


def bytes_to_gb(value: int) -> float:
    return float(value) / (1024 ** 3)


def plot_summary(summary_df: pd.DataFrame, output_dir: Path, objective: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = "TET" if objective == "evac" else "TTT"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    axes[0].plot(summary_df["step_change_limit"], summary_df["objective_metric"], marker="o", color="tab:red")
    axes[0].set_xlabel("Max link changes per step")
    axes[0].set_ylabel(f"Final {metric_label} (veh min)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(summary_df["step_change_limit"], summary_df["peak_n_over_njam_max"], marker="o", color="tab:blue")
    axes[1].set_xlabel("Max link changes per step")
    axes[1].set_ylabel("Peak N/N_jam")
    axes[1].set_ylim(0, max(1.0, summary_df["peak_n_over_njam_max"].max() * 1.05))
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(summary_df["step_change_limit"], summary_df["elapsed_time_sec"], marker="o", color="tab:green")
    axes[2].set_xlabel("Max link changes per step")
    axes[2].set_ylabel("Computation time (s)")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(
        summary_df["step_change_limit"],
        summary_df["rss_after_optimization_gb"],
        marker="o",
        color="tab:purple",
        label="RSS after optimization",
    )
    axes[3].plot(
        summary_df["step_change_limit"],
        summary_df["rss_after_constraints_gb"],
        marker="s",
        color="tab:brown",
        label="RSS after constraints",
    )
    axes[3].set_xlabel("Max link changes per step")
    axes[3].set_ylabel("Memory usage (GB)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    fig.suptitle("Sensitivity to the maximum number of link changes per step", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "step_change_limit_summary.png", dpi=300)
    plt.close(fig)


def build_table(summary_df: pd.DataFrame, objective: str) -> pd.DataFrame:
    metric_col = "Final TET" if objective == "evac" else "Final TTT"
    base_row = summary_df.sort_values("step_change_limit").iloc[0]
    base_objective = float(base_row["objective_metric"])
    base_mem = float(base_row["rss_after_constraints_gb"])
    base_total_mem = float(base_row["rss_after_optimization_gb"])
    base_time = float(base_row["elapsed_time_sec"])
    base_peak = float(base_row["peak_n_over_njam_max"])

    rows = []
    for _, row in summary_df.sort_values("step_change_limit").iterrows():
        objective_value = float(row["objective_metric"])
        constraint_mem = float(row["rss_after_constraints_gb"])
        total_mem = float(row["rss_after_optimization_gb"])
        elapsed = float(row["elapsed_time_sec"])
        peak = float(row["peak_n_over_njam_max"])
        rows.append(
            {
                "k": int(row["step_change_limit"]),
                f"{metric_col} (veh min)": objective_value,
                f"{metric_col} change vs k=1 (%)": 100.0 * (objective_value - base_objective) / base_objective if base_objective else np.nan,
                "Peak N/N_jam": peak,
                "Peak N/N_jam change vs k=1 (%)": 100.0 * (peak - base_peak) / base_peak if base_peak else np.nan,
                "Constraint memory (GB)": constraint_mem,
                "Constraint memory change vs k=1 (%)": 100.0 * (constraint_mem - base_mem) / base_mem if base_mem else np.nan,
                "Total memory (GB)": total_mem,
                "Total memory change vs k=1 (%)": 100.0 * (total_mem - base_total_mem) / base_total_mem if base_total_mem else np.nan,
                "Runtime (s)": elapsed,
                "Runtime change vs k=1 (%)": 100.0 * (elapsed - base_time) / base_time if base_time else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_table(summary_df: pd.DataFrame, output_dir: Path, objective: str):
    table_df = build_table(summary_df, objective)
    table_df.to_csv(output_dir / "step_change_limit_table.csv", index=False)
    latex = table_df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        caption="Sensitivity of reconfiguration performance to the maximum number of link changes per step.",
        label="tab:step_change_limit_sensitivity",
        column_format="lrrrrrrrrr",
    )
    (output_dir / "step_change_limit_table.tex").write_text(latex, encoding="utf-8")
    md_path = output_dir / "step_change_limit_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(table_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    module = load_cross_entropy_module()
    CrossEntropy = module.CrossEntropy
    from parameters_ndp import Parameters

    params = Parameters(background_ratio=args.background_ratio)
    target_graph = parse_binary_vector(args.target_graph)
    initial_graph = [0] * len(target_graph)
    time_steps = int((args.reconf_end_time - args.reconf_start_time) // args.seq_interval)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    limits = parse_limits(args.change_limits)

    for step_change_limit in limits:
        scenario_dir = output_dir / f"k_{step_change_limit}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss

        ce = CrossEntropy(
            params=copy.deepcopy(params),
            initial_graph=initial_graph,
            target_graph=target_graph,
            time_steps=time_steps,
            seq_interval=args.seq_interval,
            reconf_start_time=args.reconf_start_time,
            reconf_end_time=args.reconf_end_time,
            objective_func=args.objective,
            step_change_limit=step_change_limit,
        )
        ce.output_path = str(scenario_dir)
        Path(ce.output_path).mkdir(parents=True, exist_ok=True)
        Path(ce.output_path, "p_distribution").mkdir(parents=True, exist_ok=True)

        start_const = time.time()
        ce.zdd_constraints = ce.reconf.reconfiguration()
        constraint_build_time = time.time() - start_const
        rss_after_constraints = process.memory_info().rss
        search_space_size = int(len(ce.reconf.constraints))
        feasible_state_counts = [int(len(state_set)) for state_set in ce.zdd_constraints]

        run_start = time.time()
        best_sequence, best_cost = ce.fully_adaptive_cross_entropy(
            elite_count=args.elite_count,
            Nmin=args.nmin,
            Nmax=args.nmax,
            Nunit=args.nunit,
            d=args.d,
            max_cpu=args.max_cpu,
            max_sim_evals=(None if args.max_sim_evals <= 0 else args.max_sim_evals),
            disable_early_stop=args.disable_early_stop,
        )
        elapsed_time = time.time() - run_start
        rss_after_optimization = process.memory_info().rss

        pd.DataFrame(best_sequence).to_csv(scenario_dir / "best_sequence.csv", index=False, header=False)
        sim = simulate_sequence(best_sequence, ce)
        metric_summary = summarize_metrics(
            sim=sim,
            params=ce.params,
            reconf_end_time=args.reconf_end_time,
            simulation_start_time=ce.params.simulation_start_time,
            objective=args.objective,
        )

        record = {
            "step_change_limit": step_change_limit,
            "best_cost": float(best_cost),
            "search_space_size": search_space_size,
            "num_reconf_levels": len(feasible_state_counts),
            "level_0_states": feasible_state_counts[0] if len(feasible_state_counts) > 0 else np.nan,
            "level_1_states": feasible_state_counts[1] if len(feasible_state_counts) > 1 else np.nan,
            "level_2_states": feasible_state_counts[2] if len(feasible_state_counts) > 2 else np.nan,
            "level_3_states": feasible_state_counts[3] if len(feasible_state_counts) > 3 else np.nan,
            "level_4_states": feasible_state_counts[4] if len(feasible_state_counts) > 4 else np.nan,
            "level_5_states": feasible_state_counts[5] if len(feasible_state_counts) > 5 else np.nan,
            "level_6_states": feasible_state_counts[6] if len(feasible_state_counts) > 6 else np.nan,
            "constraint_build_time_sec": constraint_build_time,
            "elapsed_time_sec": elapsed_time,
            "rss_before_gb": bytes_to_gb(rss_before),
            "rss_after_constraints_gb": bytes_to_gb(rss_after_constraints),
            "rss_after_optimization_gb": bytes_to_gb(rss_after_optimization),
            "delta_constraints_gb": bytes_to_gb(rss_after_constraints - rss_before),
            "delta_optimization_gb": bytes_to_gb(rss_after_optimization - rss_after_constraints),
        }
        record.update(metric_summary)
        records.append(record)

        (scenario_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "objective": args.objective,
                    "background_ratio": args.background_ratio,
                    "seed": args.seed,
                    "step_change_limit": step_change_limit,
                    "elite_count": args.elite_count,
                    "nmin": args.nmin,
                    "nmax": args.nmax,
                    "nunit": args.nunit,
                    "d": args.d,
                    "max_cpu": args.max_cpu,
                    "max_sim_evals": args.max_sim_evals,
                    "disable_early_stop": args.disable_early_stop,
                    "search_space_size": search_space_size,
                    "feasible_state_counts": feasible_state_counts,
                    "rss_before_gb": bytes_to_gb(rss_before),
                    "rss_after_constraints_gb": bytes_to_gb(rss_after_constraints),
                    "rss_after_optimization_gb": bytes_to_gb(rss_after_optimization),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"k={step_change_limit}: objective={record['objective_metric']:.3f}, "
            f"peak N/N_jam={record['peak_n_over_njam_max']:.3f}, "
            f"space={search_space_size}, mem={record['rss_after_optimization_gb']:.2f}GB, elapsed={elapsed_time:.1f}s"
        )

    summary_df = pd.DataFrame(records).sort_values("step_change_limit")
    summary_df.to_csv(output_dir / "step_change_limit_summary.csv", index=False)
    write_table(summary_df, output_dir, args.objective)
    plot_summary(summary_df, output_dir, args.objective)
    print(f"Saved step-change sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    main()
