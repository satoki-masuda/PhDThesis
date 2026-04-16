import argparse
import importlib.util
import json
import os
import random
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


DEFAULT_TARGET_GRAPH = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CEM+ZDD for transition-sequence optimization.")
    parser.add_argument("--objective", choices=["evac", "normal", "multi"], default="normal")
    parser.add_argument("--background-ratio", type=float, default=0.8)
    parser.add_argument("--target-graph", type=str, default=",".join(map(str, DEFAULT_TARGET_GRAPH)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reconf-start-time", type=int, default=6 * 60)
    parser.add_argument("--reconf-end-time", type=int, default=12 * 60)
    parser.add_argument("--seq-interval", type=int, default=60)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--nmin", type=int, default=400)
    parser.add_argument("--nmax", type=int, default=1000)
    parser.add_argument("--nunit", type=int, default=200)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--max-cpu", type=int, default=1)
    parser.add_argument("--max-sim-evals", type=int, default=0)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def parse_binary_vector(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if any(v not in (0, 1) for v in values):
        raise ValueError("target_graph must contain only 0 or 1.")
    return values


def load_cross_entropy_module():
    module_path = SCRIPT_DIR / "cross-entropy.py"
    spec = importlib.util.spec_from_file_location("cross_entropy_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_output_dir(params, objective: str, seed: int) -> Path:
    return (
        Path("../../output/optimizer_baselines")
        / "cem_zdd"
        / f"green_{params.green_split}"
        / params.demand
        / params.demand_variation
        / f"back{params.background_ratio}"
        / f"{int(params.simulation_start_time/60)}_{int(params.simulation_end_time/60)}_control6_{objective}"
        / f"seed_{seed}"
    )


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

    if args.output_dir is None:
        output_dir = default_output_dir(params, args.objective, args.seed).resolve()
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    ce = CrossEntropy(
        params=params,
        initial_graph=initial_graph,
        target_graph=target_graph,
        time_steps=time_steps,
        seq_interval=args.seq_interval,
        reconf_start_time=args.reconf_start_time,
        reconf_end_time=args.reconf_end_time,
        objective_func=args.objective,
    )
    if output_dir != Path(ce.output_path).resolve():
        ce.output_path = str(output_dir)
        Path(ce.output_path).mkdir(parents=True, exist_ok=True)
        Path(ce.output_path, "p_distribution").mkdir(parents=True, exist_ok=True)

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

    pd.DataFrame(best_sequence).to_csv(output_dir / "best_sequence.csv", index=False, header=False)
    metadata = {
        "objective": args.objective,
        "background_ratio": args.background_ratio,
        "seed": args.seed,
        "elite_count": args.elite_count,
        "Nmin": args.nmin,
        "Nmax": args.nmax,
        "Nunit": args.nunit,
        "d": args.d,
        "max_cpu": args.max_cpu,
        "max_sim_evals": args.max_sim_evals,
        "disable_early_stop": args.disable_early_stop,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved CEM outputs to {output_dir}")
    print(f"Best cost: {best_cost}")


if __name__ == "__main__":
    main()
