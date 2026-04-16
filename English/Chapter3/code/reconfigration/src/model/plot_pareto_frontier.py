import argparse
import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)


#DEFAULT_TARGET_GRAPH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 93max_0.8
DEFAULT_TARGET_GRAPH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0] # 93max_1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample feasible reconfiguration sequences and plot a Pareto frontier."
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of feasible sequences to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--reconf-start", type=int, default=6 * 60, help="Reconfiguration start time in minutes.")
    parser.add_argument("--reconf-end", type=int, default=12 * 60, help="Reconfiguration end time in minutes.")
    parser.add_argument("--seq-interval", type=int, default=60, help="Transition interval in minutes.")
    parser.add_argument(
        "--target-graph",
        type=str,
        default=",".join(map(str, DEFAULT_TARGET_GRAPH)),
        help="Comma-separated 0/1 vector for the target contraflow graph.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1000,
        help="Bernoulli candidates generated per trial when sampling each transition step.",
    )
    return parser.parse_args()


def parse_binary_vector(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if any(v not in (0, 1) for v in values):
        raise ValueError("target_graph must contain only 0 or 1.")
    return values


def build_output_dir(params, reconf_start_time: int, reconf_end_time: int, time_steps: int):
    output_dir = (
        Path("../../output/pareto_frontier")
        / f"green_{params.green_split}"
        / params.demand
        / params.demand_variation
        / f"back{params.background_ratio}"
        / f"{int(params.simulation_start_time/60)}_{int(params.simulation_end_time/60)}_control{time_steps}"
        / f"{int(reconf_start_time/60)}_{int(reconf_end_time/60)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequences").mkdir(exist_ok=True)
    return output_dir


class ParetoFrontierSampler:
    def __init__(
        self,
        params,
        initial_graph,
        target_graph,
        reconf_start_time: int,
        reconf_end_time: int,
        seq_interval: int,
        output_dir: Path,
        sample_batch_size: int = 1000,
    ):
        self.params = params
        self.initial_graph = np.array(initial_graph, dtype=int)
        self.target_graph = np.array(target_graph, dtype=int)
        self.reconf_start_time = reconf_start_time
        self.reconf_end_time = reconf_end_time
        self.seq_interval = seq_interval
        self.time_steps = int((reconf_end_time - reconf_start_time) // seq_interval)
        self.output_dir = output_dir
        self.sample_batch_size = sample_batch_size

        self.link_indices = [
            (i, j)
            for i in range(self.params.num_zones)
            for j in range(self.params.num_zones)
            if self.params.adj_matrix[i, j] == 1
        ]
        self.num_links = len(self.link_indices)
        if len(self.target_graph) != self.num_links:
            raise ValueError(
                f"target_graph length must be {self.num_links}, but got {len(self.target_graph)}."
            )

        self.original_capacity = copy.deepcopy(self.params.max_boundary_capacity)
        from reconf_horizon import ReconfigZDD

        self.reconf = ReconfigZDD(
            self.params,
            self.initial_graph.tolist(),
            self.target_graph.tolist(),
            k=self.time_steps,
            constraints_csv=None,
            output_path=str(self.output_dir / "reconf"),
        )
        self.zdd_constraints = self.reconf.reconfiguration()

    def array_to_zdd(self, array: np.ndarray):
        from graphillion import GraphSet

        if array.ndim == 1:
            graphs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[idx] == 1]]
        else:
            graphs = [
                [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[row, idx] == 1]
                for row in range(array.shape[0])
            ]
        return GraphSet(graphs)

    def zdd_to_array(self, zdd):
        return np.array(
            [
                [
                    1 if (f"{i}_out", f"{j}_in") in g or (f"{j}_in", f"{i}_out") in g else 0
                    for i, j in self.link_indices
                ]
                for g in list(zdd)
            ],
            dtype=int,
        )

    def const_edge_idx(self):
        const_edges = [
            (i, j)
            for i, j in self.link_indices
            if i in [0, 3, 7, 8] and j in [0, 3, 7, 8]
        ]
        return [self.link_indices.index(edge) for edge in const_edges]

    def initial_sampling_probability(self):
        base = np.array(
            [
                np.full(self.num_links, self.target_graph.sum() * (t + 1) / self.time_steps / self.num_links)
                for t in range(self.time_steps - 1)
            ]
        ).flatten()
        for idx in self.const_edge_idx():
            for t in range(self.time_steps - 1):
                base[t * self.num_links + idx] = 0.0
        return base

    def sample_sequences(self, population_size: int, rng: np.random.Generator):
        p = self.initial_sampling_probability()
        population = np.zeros((population_size, self.num_links * (self.time_steps - 1)), dtype=int)

        concat_list = []
        n_sample = 0
        while n_sample < population_size:
            tmp = (rng.random((self.sample_batch_size, self.num_links)) < p[: self.num_links]).astype(int)
            feasible = self.array_to_zdd(tmp) & self.zdd_constraints[1]
            if len(feasible) > 0:
                arr = self.zdd_to_array(feasible)
                concat_list.extend(arr.tolist())
                n_sample += len(arr)
        population[:, : self.num_links] = np.array(concat_list[:population_size], dtype=int)

        for t in range(1, self.time_steps - 1):
            groups, indices = np.unique(
                population[:, (t - 1) * self.num_links : t * self.num_links],
                axis=0,
                return_inverse=True,
            )
            for idx, state in enumerate(groups):
                rows = []
                needed = int(np.sum(indices == idx))
                sampled = 0
                attempts = 0
                while sampled < needed:
                    if attempts > 1000:
                        raise RuntimeError(
                            f"Failed to sample enough feasible states at time step {t} after 1000 attempts. Collected {sampled} states out of {needed} needed."
                        )
                    tmp = (
                        rng.random((self.sample_batch_size, self.num_links))
                        < p[t * self.num_links : (t + 1) * self.num_links]
                    ).astype(int)
                    feasible = (
                        self.array_to_zdd(tmp)
                        & self.zdd_constraints[t + 1]
                        & self.reconf.transition(self.array_to_zdd(state), self.reconf.constraints)
                    )
                    if len(feasible) > 0:
                        arr = self.zdd_to_array(feasible)
                        take = min(len(arr), needed - sampled)
                        rows.extend(arr[:take].tolist())
                        sampled += take
                    attempts += 1
                population[indices == idx, t * self.num_links : (t + 1) * self.num_links] = np.array(rows, dtype=int)

        population = np.hstack((population, np.tile(self.target_graph, (population_size, 1))))
        return population.reshape((population_size, self.time_steps, self.num_links))

    def evaluate_sequence(self, sequence: np.ndarray):
        from mfd_dynamics import MFD_Dynamics

        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        sim = MFD_Dynamics(copy.deepcopy(self.params), output_path=None)

        for _ in range(sim.sim_start_step, sim.sim_end_step):
            if (sim.step % self.seq_interval == 0) and (self.reconf_start_time < sim.step <= self.reconf_end_time):
                sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                step_idx = (sim.step // self.seq_interval) - (self.reconf_start_time // self.seq_interval) - 1
                contraflow_idx = sequence[step_idx]
                for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                    sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
            sim.step_simulation()

        reconf_end_idx = self.reconf_end_time - self.params.simulation_start_time - 1
        evac_total = float(sim.tet_list[-1])
        normal_transition_total = float(sim.ttt_list[reconf_end_idx])
        evac_cumsum = self.params.Q[:, :, self.params.simulation_start_time : self.params.simulation_end_time].sum(axis=(0, 1)).cumsum()
        normal_cumsum = self.params.Q_background[:, :, self.params.simulation_start_time : self.params.simulation_end_time].sum(axis=(0, 1)).cumsum()
        evac_avg = float(sim.tet_list[-1] / evac_cumsum[-1]) if len(evac_cumsum) > 0 and evac_cumsum[-1] > 0 else np.nan
        normal_avg = float(sim.ttt_list[reconf_end_idx] / normal_cumsum[reconf_end_idx]) if len(normal_cumsum) > reconf_end_idx and normal_cumsum[reconf_end_idx] > 0 else np.nan

        return {
            "evac_total_tet": evac_total,
            "normal_transition_ttt": normal_transition_total,
            "evacuation_avg_tt": evac_avg,
            "normal_avg_tt": normal_avg,
        }

    @staticmethod
    def pareto_mask(df: pd.DataFrame):
        values = df[["normal_transition_ttt", "evac_total_tet"]].to_numpy()
        mask = np.ones(len(values), dtype=bool)
        for i in range(len(values)):
            dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
            dominated[i] = False
            if np.any(dominated):
                mask[i] = False
        return mask

    def save_sequences(self, sequences: np.ndarray, results: pd.DataFrame):
        flat = sequences.reshape((sequences.shape[0], -1))
        sequence_columns = [
            f"step_{step}_link_{link}"
            for step in range(sequences.shape[1])
            for link in range(sequences.shape[2])
        ]
        df_sequences = pd.DataFrame(flat, columns=sequence_columns)
        df_sequences.insert(0, "sample_id", np.arange(len(df_sequences)))
        merged = results.merge(df_sequences, on="sample_id")
        merged.to_csv(self.output_dir / "pareto_samples.csv", index=False)

        frontier = merged[merged["is_pareto"]].copy().sort_values(
            ["normal_transition_ttt", "evac_total_tet"]
        )
        frontier.to_csv(self.output_dir / "pareto_frontier.csv", index=False)

        representative = {}
        representative["min_evac"] = int(frontier["evac_total_tet"].idxmin())
        representative["min_normal"] = int(frontier["normal_transition_ttt"].idxmin())
        normalized = (
            (frontier["evac_total_tet"] - frontier["evac_total_tet"].min())
            / max(frontier["evac_total_tet"].max() - frontier["evac_total_tet"].min(), 1e-9)
        ) + (
            (frontier["normal_transition_ttt"] - frontier["normal_transition_ttt"].min())
            / max(frontier["normal_transition_ttt"].max() - frontier["normal_transition_ttt"].min(), 1e-9)
        )
        representative["balanced"] = int(normalized.idxmin())

        with open(self.output_dir / "representative_samples.txt", "w", encoding="utf-8") as fh:
            for name, row_idx in representative.items():
                sample_id = int(merged.loc[row_idx, "sample_id"])
                fh.write(f"{name}: sample_id={sample_id}\n")
                sequence = sequences[sample_id]
                pd.DataFrame(sequence).to_csv(
                    self.output_dir / "sequences" / f"{name}_sequence.csv",
                    index=False,
                    header=False,
                )

    def plot_frontier(self, results: pd.DataFrame):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
        import matplotlib.pyplot as plt

        frontier = results[results["is_pareto"]].copy().sort_values(
            ["normal_transition_ttt", "evac_total_tet"]
        )

        # pareto frontierあり
        plt.figure(figsize=(10, 7))
        plt.scatter(
            results["normal_transition_ttt"],
            results["evac_total_tet"],
            color="lightgray",
            alpha=0.7,
            label="Feasible samples",
        )
        plt.plot(
            frontier["normal_transition_ttt"],
            frontier["evac_total_tet"],
            color="crimson",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Pareto frontier",
        )
        plt.xlabel("Normal-traffic TTT during transition (veh min)")
        plt.ylabel("Evacuation TET over full horizon (veh min)")
        plt.title("Pareto frontier of feasible reconfiguration sequences")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "pareto_frontier.png", dpi=300)
        plt.close()
        
        # pareto frontierなし
        plt.figure(figsize=(10, 7))
        plt.scatter(
            results["normal_transition_ttt"],
            results["evac_total_tet"],
            color="lightgray",
            alpha=0.7,
            label="All samples",
        )
        plt.xlabel("Normal-traffic TTT during transition (veh min)")
        plt.ylabel("Evacuation TET over full horizon (veh min)")
        plt.title("Feasible reconfiguration sequences")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "pareto_frontier_none.png", dpi=300)
        plt.close()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    target_graph = parse_binary_vector(args.target_graph)
    initial_graph = [0] * len(target_graph)

    from parameters_ndp import Parameters

    params = Parameters()
    time_steps = int((args.reconf_end - args.reconf_start) // args.seq_interval)
    output_dir = build_output_dir(params, args.reconf_start, args.reconf_end, time_steps)

    sampler = ParetoFrontierSampler(
        params=params,
        initial_graph=initial_graph,
        target_graph=target_graph,
        reconf_start_time=args.reconf_start,
        reconf_end_time=args.reconf_end,
        seq_interval=args.seq_interval,
        output_dir=output_dir,
        sample_batch_size=args.sample_batch_size,
    )

    print(f"Sampling {args.samples} feasible sequences...")
    sequences = sampler.sample_sequences(args.samples, rng)

    print("Evaluating sampled sequences...")
    rows = []
    for sample_id, sequence in enumerate(sequences):
        metrics = sampler.evaluate_sequence(sequence)
        metrics["sample_id"] = sample_id
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(
        ["evac_total_tet", "normal_transition_ttt"]
    ).reset_index(drop=True)
    results["is_pareto"] = sampler.pareto_mask(results)
    results.to_csv(output_dir / "pareto_objectives.csv", index=False)

    sampler.save_sequences(sequences, results)
    sampler.plot_frontier(results)

    print(f"Saved Pareto analysis to {output_dir.resolve()}")
    print(f"Pareto-optimal samples: {int(results['is_pareto'].sum())} / {len(results)}")


if __name__ == "__main__":
    main()
