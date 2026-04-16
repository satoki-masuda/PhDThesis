import argparse
import copy
import importlib.util
import os
import random
import sys
import tempfile
import time
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

GA_ZDD_CONFIGS = {
    "balanced": {
        "crossover_rate": 0.8,
        "copy_rate": 0.1,
        "mutation_size": 0.2,
    },
    "explore": {
        "crossover_rate": 0.5,
        "copy_rate": 0.1,
        "mutation_size": 0.3,
    },
    "exploit": {
        "crossover_rate": 0.9,
        "copy_rate": 0.05,
        "mutation_size": 0.1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare feasible-sequence sampling efficiency.")
    parser.add_argument("--background-ratio", type=float, default=0.8)
    parser.add_argument("--target-graph", type=str, default=",".join(map(str, DEFAULT_TARGET_GRAPH)))
    parser.add_argument("--objective", choices=["evac", "normal", "multi"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reconf-start-time", type=int, default=6 * 60)
    parser.add_argument("--reconf-end-time", type=int, default=12 * 60)
    parser.add_argument("--seq-interval", type=int, default=60)
    parser.add_argument("--target-feasible-samples", type=int, default=100)
    parser.add_argument("--plain-batch-size", type=int, default=200)
    parser.add_argument("--cem-batch-size", type=int, default=1000)
    parser.add_argument("--ga-zdd-batch-size", type=int, default=200)
    parser.add_argument("--ga-zdd-parent-pool", type=int, default=20)
    parser.add_argument(
        "--ga-zdd-configs",
        type=str,
        default="balanced,explore,exploit",
        help="Comma-separated GA+ZDD sampling configs to compare.",
    )
    parser.add_argument(
        "--max-raw-candidates",
        type=int,
        default=0,
        help="Optional cap on raw candidates. Use 0 to keep sampling until the target feasible count is reached.",
    )
    parser.add_argument(
        "--max-wall-time-sec",
        type=float,
        default=0.0,
        help="Optional cap on wall-clock time per method. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/validation/sampling_efficiency"),
    )
    return parser.parse_args()


def parse_binary_vector(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if any(v not in (0, 1) for v in values):
        raise ValueError("target_graph must contain only 0 or 1.")
    return values


def parse_config_names(raw: str):
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in GA_ZDD_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown GA+ZDD config(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(GA_ZDD_CONFIGS))}"
        )
    if not names:
        raise ValueError("At least one GA+ZDD config must be provided.")
    return names


def load_cross_entropy_module():
    module_path = SCRIPT_DIR / "cross-entropy.py"
    spec = importlib.util.spec_from_file_location("cross_entropy_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplingBenchmark:
    def __init__(self, params, target_graph, objective, reconf_start_time, reconf_end_time, seq_interval, seed):
        module = load_cross_entropy_module()
        CrossEntropy = module.CrossEntropy
        self.params = params
        self.target_graph = np.array(target_graph, dtype=int)
        self.objective = objective
        self.reconf_start_time = reconf_start_time
        self.reconf_end_time = reconf_end_time
        self.seq_interval = seq_interval
        self.time_steps = int((reconf_end_time - reconf_start_time) // seq_interval)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        initial_graph = np.zeros_like(self.target_graph, dtype=int)
        self.ce = CrossEntropy(
            params=self.params,
            initial_graph=initial_graph.tolist(),
            target_graph=self.target_graph.tolist(),
            time_steps=self.time_steps,
            seq_interval=self.seq_interval,
            reconf_start_time=self.reconf_start_time,
            reconf_end_time=self.reconf_end_time,
            objective_func=self.objective,
        )
        self.num_links = self.ce.num_links
        self.const_edge_idx = self.ce.const_edge_idx()

    def initial_sampling_probability(self):
        p = np.array(
            [
                np.full(self.num_links, self.target_graph.sum() * (t + 1) / self.time_steps / self.num_links)
                for t in range(self.time_steps - 1)
            ]
        ).flatten()
        for idx in self.const_edge_idx:
            for t in range(self.time_steps - 1):
                p[t * self.num_links + idx] = 0.0
        return p

    def enforce_fixed_entries(self, genome: np.ndarray) -> np.ndarray:
        genome = genome.astype(int).copy()
        for t in range(self.time_steps - 1):
            offset = t * self.num_links
            genome[offset + np.array(self.const_edge_idx, dtype=int)] = 0
        return genome

    def reshape_genome(self, genome: np.ndarray) -> np.ndarray:
        variable = self.enforce_fixed_entries(genome).reshape((self.time_steps - 1, self.num_links))
        return np.vstack([variable, self.target_graph])

    def flatten_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return sequence[:-1].reshape(-1).astype(int)

    def is_feasible_sequence(self, sequence: np.ndarray) -> bool:
        prev_state = None
        for t in range(self.time_steps):
            state = sequence[t].astype(int)
            state_zdd = self.ce.array_to_zdd(state)
            if len(state_zdd & self.ce.zdd_constraints[t + 1]) == 0:
                return False
            if prev_state is not None:
                reachable = self.ce.reconf.transition(self.ce.array_to_zdd(prev_state), self.ce.reconf.constraints)
                if len(state_zdd & reachable) == 0:
                    return False
            prev_state = state
        return np.array_equal(sequence[-1], self.target_graph)

    def sample_one_cem_sequence(self, batch_size: int):
        p = self.initial_sampling_probability()
        sequence = np.zeros((self.time_steps, self.num_links), dtype=int)
        raw_candidates = 0
        feasible_hits = 0

        while True:
            raw_candidates += batch_size
            tmp = (self.rng.random((batch_size, self.num_links)) < p[: self.num_links]).astype(int)
            feasible = self.ce.array_to_zdd(tmp) & self.ce.zdd_constraints[1]
            if len(feasible) > 0:
                arr = self.ce.zdd_to_array(feasible)
                sequence[0] = np.array(arr[0], dtype=int)
                feasible_hits += len(arr)
                break

        for t in range(1, self.time_steps - 1):
            prev_state = sequence[t - 1]
            while True:
                raw_candidates += batch_size
                tmp = (self.rng.random((batch_size, self.num_links)) < p[t * self.num_links : (t + 1) * self.num_links]).astype(int)
                feasible = (
                    self.ce.array_to_zdd(tmp)
                    & self.ce.zdd_constraints[t + 1]
                    & self.ce.reconf.transition(self.ce.array_to_zdd(prev_state), self.ce.reconf.constraints)
                )
                if len(feasible) > 0:
                    arr = self.ce.zdd_to_array(feasible)
                    sequence[t] = np.array(arr[0], dtype=int)
                    feasible_hits += len(arr)
                    break

        sequence[-1] = self.target_graph.copy()
        return sequence, raw_candidates, feasible_hits

    def sample_plain_random_sequence_batch(self, batch_size: int):
        genomes = (self.rng.random((batch_size, self.num_links * (self.time_steps - 1))) < 0.15).astype(int)
        genomes = np.array([self.enforce_fixed_entries(genome) for genome in genomes], dtype=int)
        sequences = np.array([self.reshape_genome(genome) for genome in genomes], dtype=int)
        feasible_mask = np.array([self.is_feasible_sequence(seq) for seq in sequences], dtype=bool)
        return sequences, feasible_mask

    def make_initial_parent_pool(self, pool_size: int, cem_batch_size: int):
        parents = []
        while len(parents) < pool_size:
            seq, _, _ = self.sample_one_cem_sequence(cem_batch_size)
            parents.append(self.flatten_sequence(seq))
        return parents

    def ga_style_children(
        self,
        parents: list[np.ndarray],
        batch_size: int,
        mutation_size: float = 0.2,
        crossover_rate: float = 0.8,
        copy_rate: float = 0.1,
    ):
        children = []
        tournament_size = 3
        while len(children) < batch_size:
            idxs = self.rng.choice(len(parents), size=min(tournament_size, len(parents)), replace=False)
            parent1 = parents[int(idxs[0])].copy()
            parent2 = parents[int(idxs[-1])].copy()
            rand = self.rng.random()
            if rand < crossover_rate:
                mask = self.rng.random(parent1.shape[0]) < 0.5
                child = np.where(mask, parent1, parent2)
            elif rand < crossover_rate + copy_rate:
                child = parent1.copy()
            else:
                child = parent1.copy()
                n_flip = max(1, int(mutation_size * self.num_links))
                for t in range(self.time_steps - 1):
                    valid_indices = [idx for idx in range(self.num_links) if idx not in self.const_edge_idx]
                    chosen = self.rng.choice(valid_indices, size=min(n_flip, len(valid_indices)), replace=False)
                    offset = t * self.num_links
                    child[offset + chosen] = 1 - child[offset + chosen]
            children.append(self.enforce_fixed_entries(child))
        return np.array(children, dtype=int)

    def filter_feasible_sequences_zdd(self, genomes: np.ndarray) -> np.ndarray:
        sequences = np.array([self.reshape_genome(genome) for genome in genomes], dtype=int)
        alive = np.ones(len(sequences), dtype=bool)

        state_maps = []
        states = sequences[:, 0, :]
        unique_states, inverse = np.unique(states, axis=0, return_inverse=True)
        feasible_zdd = self.ce.array_to_zdd(unique_states) & self.ce.zdd_constraints[1]
        feasible_states = {tuple(row.tolist()) for row in np.array(self.ce.zdd_to_array(feasible_zdd), dtype=int)}
        alive &= np.array([tuple(state.tolist()) in feasible_states for state in states], dtype=bool)

        for t in range(1, self.time_steps):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            grouped: dict[tuple, list[int]] = {}
            for idx in alive_indices:
                prev_key = tuple(sequences[idx, t - 1, :].tolist())
                grouped.setdefault(prev_key, []).append(idx)
            for prev_key, idx_list in grouped.items():
                prev_state = np.array(prev_key, dtype=int)
                current_states = sequences[idx_list, t, :]
                unique_current = np.unique(current_states, axis=0)
                feasible_zdd = (
                    self.ce.array_to_zdd(unique_current)
                    & self.ce.zdd_constraints[t + 1]
                    & self.ce.reconf.transition(self.ce.array_to_zdd(prev_state), self.ce.reconf.constraints)
                )
                feasible_states = {tuple(row.tolist()) for row in np.array(self.ce.zdd_to_array(feasible_zdd), dtype=int)}
                for idx in idx_list:
                    if tuple(sequences[idx, t, :].tolist()) not in feasible_states:
                        alive[idx] = False
        return alive


def record_curve(label: str, cumulative_raw: int, cumulative_feasible: int, elapsed: float):
    return {
        "label": label,
        "cumulative_raw_candidates": cumulative_raw,
        "cumulative_feasible_samples": cumulative_feasible,
        "elapsed_time_sec": elapsed,
    }


def reached_limit(cumulative_raw: int, elapsed: float, args) -> bool:
    raw_limit_hit = args.max_raw_candidates > 0 and cumulative_raw >= args.max_raw_candidates
    time_limit_hit = args.max_wall_time_sec > 0 and elapsed >= args.max_wall_time_sec
    return raw_limit_hit or time_limit_hit


def run_benchmarks(args):
    from parameters_ndp import Parameters

    params = Parameters(background_ratio=args.background_ratio)
    target_graph = parse_binary_vector(args.target_graph)
    benchmark = SamplingBenchmark(
        params=params,
        target_graph=target_graph,
        objective=args.objective,
        reconf_start_time=args.reconf_start_time,
        reconf_end_time=args.reconf_end_time,
        seq_interval=args.seq_interval,
        seed=args.seed,
    )

    records = []

    start = time.time()
    cumulative_raw = 0
    cumulative_feasible = 0
    print("Sampling with CEM+ZDD...")
    while cumulative_feasible < args.target_feasible_samples:
        _, raw_count, _ = benchmark.sample_one_cem_sequence(args.cem_batch_size)
        cumulative_raw += raw_count
        cumulative_feasible += 1
        records.append(record_curve("CEM+ZDD", cumulative_raw, cumulative_feasible, time.time() - start))

    start = time.time()
    cumulative_raw = 0
    cumulative_feasible = 0
    print("Sampling with naive random sequence generation...")
    while cumulative_feasible < args.target_feasible_samples:
        sequences, feasible_mask = benchmark.sample_plain_random_sequence_batch(args.plain_batch_size)
        cumulative_raw += len(sequences)
        feasible_count = int(feasible_mask.sum())
        if feasible_count > 0:
            cumulative_feasible += 1
        elapsed = time.time() - start
        records.append(record_curve("Naive random sampling", cumulative_raw, cumulative_feasible, elapsed))
        if reached_limit(cumulative_raw, elapsed, args):
            print("  Naive random sampling stopped before reaching the target feasible count due to the specified limit.")
            break

    for config_name in parse_config_names(args.ga_zdd_configs):
        config = GA_ZDD_CONFIGS[config_name]
        label = f"GA+ZDD ({config_name})"
        start = time.time()
        cumulative_raw = 0
        cumulative_feasible = 0
        print(f"Preparing feasible parent pool for {label}...")
        parents = []
        while len(parents) < args.ga_zdd_parent_pool:
            seq, raw_count, _ = benchmark.sample_one_cem_sequence(args.cem_batch_size)
            cumulative_raw += raw_count
            parents.append(benchmark.flatten_sequence(seq))
        print(
            f"Sampling offspring with {label}: "
            f"copy={config['copy_rate']}, crossover={config['crossover_rate']}, mutation_size={config['mutation_size']}"
        )
        seen_sequences = {tuple(parent.tolist()) for parent in parents}
        while cumulative_feasible < args.target_feasible_samples:
            children = benchmark.ga_style_children(
                parents,
                args.ga_zdd_batch_size,
                mutation_size=config["mutation_size"],
                crossover_rate=config["crossover_rate"],
                copy_rate=config["copy_rate"],
            )
            alive = benchmark.filter_feasible_sequences_zdd(children)
            feasible_children = []
            for child in children[alive]:
                key = tuple(child.tolist())
                if key not in seen_sequences:
                    seen_sequences.add(key)
                    feasible_children.append(child.copy())
            cumulative_raw += len(children)
            if feasible_children:
                cumulative_feasible += 1
                parents = feasible_children[: args.ga_zdd_parent_pool]
                while len(parents) < args.ga_zdd_parent_pool:
                    seq, raw_count, _ = benchmark.sample_one_cem_sequence(args.cem_batch_size)
                    cumulative_raw += raw_count
                    genome = benchmark.flatten_sequence(seq)
                    key = tuple(genome.tolist())
                    if key not in seen_sequences:
                        seen_sequences.add(key)
                        parents.append(genome)
            elapsed = time.time() - start
            records.append(record_curve(label, cumulative_raw, cumulative_feasible, elapsed))
            if reached_limit(cumulative_raw, elapsed, args):
                print(f"  {label} stopped before reaching the target feasible count due to the specified limit.")
                break

    return pd.DataFrame(records)


def plot_curves(curve_df: pd.DataFrame, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "CEM+ZDD": "tab:red",
        "Naive random sampling": "tab:blue",
        "GA+ZDD (balanced)": "tab:green",
        "GA+ZDD (explore)": "tab:orange",
        "GA+ZDD (exploit)": "tab:purple",
    }

    def plot_subset(labels, output_path, title):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for label in labels:
            df = curve_df[curve_df["label"] == label].sort_values("cumulative_feasible_samples")
            if df.empty:
                continue
            axes[0].plot(df["cumulative_feasible_samples"], df["cumulative_raw_candidates"], label=label, color=colors[label], linewidth=2)
            axes[1].plot(df["cumulative_feasible_samples"], df["elapsed_time_sec"], label=label, color=colors[label], linewidth=2)
            if df["cumulative_feasible_samples"].iloc[-1] < curve_df["cumulative_feasible_samples"].max():
                axes[0].scatter(df["cumulative_feasible_samples"].iloc[-1], df["cumulative_raw_candidates"].iloc[-1], color=colors[label], s=30, zorder=3)
                axes[1].scatter(df["cumulative_feasible_samples"].iloc[-1], df["elapsed_time_sec"].iloc[-1], color=colors[label], s=30, zorder=3)
        axes[0].set_xlabel("Feasible sequences collected")
        axes[0].set_ylabel("Required raw candidates")
        axes[1].set_xlabel("Feasible sequences collected")
        axes[1].set_ylabel("Required wall-clock time (s)")
        axes[0].grid(True, alpha=0.25)
        axes[1].grid(True, alpha=0.25)
        axes[0].legend()
        fig.suptitle(title, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

    ga_zdd_labels = [label for label in curve_df["label"].unique() if label.startswith("GA+ZDD (")]
    plot_subset(
        ["CEM+ZDD", "Naive random sampling"],
        output_dir / "main_sampling_efficiency.png",
        "Main figure: CEM+ZDD vs naive random sampling",
    )
    plot_subset(
        ["CEM+ZDD"] + sorted(ga_zdd_labels),
        output_dir / "appendix_sampling_efficiency.png",
        "Appendix: CEM+ZDD vs GA+ZDD parameter settings",
    )


def write_readme(output_dir: Path, args):
    text = (
        "Sampling efficiency comparison\n"
        "- Main figure compares CEM+ZDD against naive random sequence generation.\n"
        "- Appendix sampling figure compares CEM+ZDD against GA+ZDD batch offspring filtering with multiple copy/crossover/mutation settings.\n"
        "- GA+ZDD starts from a feasible parent pool generated before timing and then filters crossover/mutation offspring with ZDD intersections.\n"
        f"- Target feasible samples: {args.target_feasible_samples}\n"
        f"- Plain GA batch size: {args.plain_batch_size}\n"
        f"- CEM batch size: {args.cem_batch_size}\n"
        f"- GA+ZDD batch size: {args.ga_zdd_batch_size}\n"
        f"- GA+ZDD configs: {args.ga_zdd_configs}\n"
        f"- Raw-candidate cap: {'disabled' if args.max_raw_candidates == 0 else args.max_raw_candidates}\n"
        f"- Wall-clock cap: {'disabled' if args.max_wall_time_sec == 0 else args.max_wall_time_sec}\n"
    )
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_df = run_benchmarks(args)
    curve_df.to_csv(output_dir / "sampling_efficiency_curve.csv", index=False)
    plot_curves(curve_df, output_dir)
    write_readme(output_dir, args)
    print(f"Saved sampling-efficiency outputs to {output_dir}")


if __name__ == "__main__":
    main()
