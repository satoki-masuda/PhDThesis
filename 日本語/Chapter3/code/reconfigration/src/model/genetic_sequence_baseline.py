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

GA_CONFIGS = {
    "balanced": {
        "population_size": 40,
        "generations": 30,
        "crossover_rate": 0.8,
        "copy_rate": 0.1,
        "mutation_size": 0.2,
        "tournament_size": 3,
    },
    "explore": {
        "population_size": 40,
        "generations": 30,
        "crossover_rate": 0.5,
        "copy_rate": 0.1,
        "mutation_size": 0.3,
        "tournament_size": 3,
    },
    "exploit": {
        "population_size": 40,
        "generations": 30,
        "crossover_rate": 0.9,
        "copy_rate": 0.05,
        "mutation_size": 0.1,
        "tournament_size": 5,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm baseline for transition-sequence optimization."
    )
    parser.add_argument("--objective", choices=["evac", "normal", "multi"], default="normal")
    parser.add_argument("--background-ratio", type=float, default=0.8)
    parser.add_argument("--target-graph", type=str, default=",".join(map(str, DEFAULT_TARGET_GRAPH)))
    parser.add_argument("--config", choices=sorted(GA_CONFIGS), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-zdd", action="store_true", help="Restrict GA offspring to feasible sequences using ZDD checks.")
    parser.add_argument("--reconf-start-time", type=int, default=6 * 60)
    parser.add_argument("--reconf-end-time", type=int, default=12 * 60)
    parser.add_argument("--seq-interval", type=int, default=60)
    parser.add_argument("--max-cpu", type=int, default=1)
    parser.add_argument("--max-sim-evals", type=int, default=0)
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory.",
    )
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


class GeneticSequenceOptimizer:
    def __init__(
        self,
        params,
        target_graph,
        objective_func: str,
        reconf_start_time: int,
        reconf_end_time: int,
        seq_interval: int,
        config_name: str,
        ga_config: dict,
        seed: int,
        use_zdd: bool,
        output_dir: Path,
    ):
        module = load_cross_entropy_module()
        self.CrossEntropy = module.CrossEntropy
        self.params = params
        self.target_graph = np.array(target_graph, dtype=int)
        self.objective_func = objective_func
        self.reconf_start_time = reconf_start_time
        self.reconf_end_time = reconf_end_time
        self.seq_interval = seq_interval
        self.time_steps = int((reconf_end_time - reconf_start_time) // seq_interval)
        self.config_name = config_name
        self.ga_config = ga_config
        self.seed = seed
        self.use_zdd = use_zdd
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        initial_graph = np.zeros_like(self.target_graph, dtype=int)
        self.ce_helper = self.CrossEntropy(
            params=self.params,
            initial_graph=initial_graph.tolist(),
            target_graph=self.target_graph.tolist(),
            time_steps=self.time_steps,
            seq_interval=self.seq_interval,
            reconf_start_time=self.reconf_start_time,
            reconf_end_time=self.reconf_end_time,
            objective_func=self.objective_func,
        )
        self.link_indices = self.ce_helper.link_indices
        self.num_links = self.ce_helper.num_links
        self.original_capacity = copy.deepcopy(self.ce_helper.original_capacity)
        self.const_edge_idx = self.ce_helper.const_edge_idx()
        self.population_size = ga_config["population_size"]
        self.generations = ga_config["generations"]
        self.crossover_rate = ga_config["crossover_rate"]
        self.copy_rate = ga_config["copy_rate"]
        self.mutation_size = ga_config["mutation_size"]
        self.tournament_size = ga_config["tournament_size"]
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def variable_length(self):
        return self.num_links * (self.time_steps - 1)

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

    def enforce_local_graph_constraints(self, sequence: np.ndarray) -> np.ndarray:
        sequence = sequence.astype(int).copy()
        incoming_selected = {t: {} for t in range(self.time_steps)}
        for t in range(self.time_steps):
            for idx, (i, j) in enumerate(self.link_indices):
                if sequence[t, idx] == 0:
                    continue
                opposite = (j, i)
                if opposite in self.link_indices:
                    opp_idx = self.link_indices.index(opposite)
                    if sequence[t, opp_idx] == 1:
                        if idx < opp_idx:
                            sequence[t, opp_idx] = 0
                        else:
                            sequence[t, idx] = 0
                            continue
                if j in incoming_selected[t]:
                    sequence[t, idx] = 0
                else:
                    incoming_selected[t][j] = idx
        sequence[-1] = self.target_graph.copy()
        return sequence

    def is_feasible_sequence(self, sequence: np.ndarray) -> bool:
        prev_state = None
        for t in range(self.time_steps):
            state = sequence[t].astype(int)
            state_zdd = self.ce_helper.array_to_zdd(state)
            if len(state_zdd & self.ce_helper.zdd_constraints[t + 1]) == 0:
                return False
            if prev_state is not None:
                reachable = self.ce_helper.reconf.transition(self.ce_helper.array_to_zdd(prev_state), self.ce_helper.reconf.constraints)
                if len(state_zdd & reachable) == 0:
                    return False
            prev_state = state
        return np.array_equal(sequence[-1], self.target_graph)

    def sample_feasible_sequence(self) -> np.ndarray:
        p = np.array(
            [
                np.full(self.num_links, self.target_graph.sum() * (t + 1) / self.time_steps / self.num_links)
                for t in range(self.time_steps - 1)
            ]
        ).flatten()
        for idx in self.const_edge_idx:
            for t in range(self.time_steps - 1):
                p[t * self.num_links + idx] = 0.0

        population = np.zeros((1, self.variable_length()), dtype=int)
        concat_list = []
        n_sample = 0
        while n_sample < 1:
            tmp = (self.rng.random((1000, self.num_links)) < p[: self.num_links]).astype(int)
            feasible = self.ce_helper.array_to_zdd(tmp) & self.ce_helper.zdd_constraints[1]
            if len(feasible) > 0:
                arr = self.ce_helper.zdd_to_array(feasible)
                concat_list.extend(arr)
                n_sample += len(arr)
        population[:, : self.num_links] = np.array(concat_list[:1], dtype=int)

        for t in range(1, self.time_steps - 1):
            prev_state = population[0, (t - 1) * self.num_links : t * self.num_links]
            sampled = False
            attempts = 0
            while not sampled:
                tmp = (self.rng.random((1000, self.num_links)) < p[t * self.num_links : (t + 1) * self.num_links]).astype(int)
                feasible = (
                    self.ce_helper.array_to_zdd(tmp)
                    & self.ce_helper.zdd_constraints[t + 1]
                    & self.ce_helper.reconf.transition(self.ce_helper.array_to_zdd(prev_state), self.ce_helper.reconf.constraints)
                )
                if len(feasible) > 0:
                    arr = self.ce_helper.zdd_to_array(feasible)
                    population[0, t * self.num_links : (t + 1) * self.num_links] = np.array(arr[0], dtype=int)
                    sampled = True
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError(f"Failed to sample a feasible sequence for time step {t}.")

        return self.reshape_genome(population[0])

    def sample_initial_population_cem_style(self) -> list[np.ndarray]:
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            population = self.ce_helper.sample_population_from_distribution(
                population_size=self.population_size,
                p=self.ce_helper.initial_sampling_probability(),
                batch_size=10**3,
            )
        finally:
            np.random.set_state(state)
        return [population[i, :-self.num_links].astype(int).copy() for i in range(population.shape[0])]

    def initialize_population(self) -> tuple[list[np.ndarray], int]:
        print(f"Initializing feasible population: target size = {self.population_size}")
        if self.use_zdd:
            population = self.sample_initial_population_cem_style()
            for count in range(max(1, self.population_size // 5), self.population_size + 1, max(1, self.population_size // 5)):
                print(
                    f"  feasible individuals: {min(count, self.population_size)}/{self.population_size} "
                    f"(initialized with the same CEM-style generation-0 sampler)"
                )
            return population, self.population_size

        population = []
        raw_candidate_count = 0
        last_report = 0
        while len(population) < self.population_size:
            raw_candidate_count += 1
            genome = (self.rng.random(self.variable_length()) < 0.15).astype(int)
            genome = self.flatten_sequence(
                self.enforce_local_graph_constraints(self.reshape_genome(self.enforce_fixed_entries(genome)))
            )
            if self.is_feasible_sequence(self.reshape_genome(genome)):
                population.append(genome)
                if len(population) == self.population_size or len(population) - last_report >= max(1, self.population_size // 5):
                    print(
                        f"  feasible individuals: {len(population)}/{self.population_size} "
                        f"(raw candidates tried: {raw_candidate_count})"
                    )
                    last_report = len(population)
        return population, raw_candidate_count

    def tournament_select(self, population: list[np.ndarray], fitness_scores: np.ndarray) -> np.ndarray:
        indices = self.rng.choice(len(population), self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness_scores[indices])]
        return population[best_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self.rng.random(self.variable_length()) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return self.enforce_fixed_entries(child1), self.enforce_fixed_entries(child2)

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        genome = genome.copy()
        n_flip = max(1, int(self.mutation_size * self.num_links))
        for t in range(self.time_steps - 1):
            valid_indices = [idx for idx in range(self.num_links) if idx not in self.const_edge_idx]
            chosen = self.rng.choice(valid_indices, size=min(n_flip, len(valid_indices)), replace=False)
            offset = t * self.num_links
            genome[offset + chosen] = 1 - genome[offset + chosen]
        return self.enforce_fixed_entries(genome)

    def make_child(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        rand = self.rng.random()
        if rand < self.crossover_rate:
            child1, child2 = self.crossover(parent1, parent2)
            child = child1 if self.rng.random() < 0.5 else child2
        elif rand < self.crossover_rate + self.copy_rate:
            child = parent1.copy()
        else:
            child = self.mutate(parent1)
        child = self.enforce_fixed_entries(child)
        if not self.use_zdd:
            child = self.flatten_sequence(
                self.enforce_local_graph_constraints(self.reshape_genome(child))
            )
        return child

    def repair_or_resample(self, child: np.ndarray) -> np.ndarray:
        if not self.use_zdd:
            return self.flatten_sequence(
                self.enforce_local_graph_constraints(self.reshape_genome(child))
            )
        sequence = self.reshape_genome(child)
        if self.is_feasible_sequence(sequence):
            return child
        for _ in range(200):
            parent_like = child.copy()
            mutated = self.mutate(parent_like)
            if self.is_feasible_sequence(self.reshape_genome(mutated)):
                return mutated
        return self.flatten_sequence(self.sample_feasible_sequence())

    def generate_feasible_child(self, parent1: np.ndarray, parent2: np.ndarray, max_attempts: int = 500) -> tuple[np.ndarray, int]:
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            child = self.make_child(parent1, parent2)
            child = self.repair_or_resample(child)
            if self.is_feasible_sequence(self.reshape_genome(child)):
                if attempts >= 50:
                    print(f"    child accepted after {attempts} attempts")
                return child, attempts

        while True:
            attempts += 1
            if self.use_zdd:
                fallback = self.flatten_sequence(self.sample_feasible_sequence())
            else:
                fallback = (self.rng.random(self.variable_length()) < 0.15).astype(int)
                fallback = self.flatten_sequence(
                    self.enforce_local_graph_constraints(self.reshape_genome(self.enforce_fixed_entries(fallback)))
                )
            if self.is_feasible_sequence(self.reshape_genome(fallback)):
                print(f"    fallback feasible child accepted after {attempts} attempts")
                return fallback, attempts

    def evaluate_population(self, population: list[np.ndarray], max_cpu: int):
        tasks = []

        if max_cpu > 1:
            import ray

            if not ray.is_initialized():
                print(f"  Starting parallel evaluation with ray (num_cpus={max_cpu})")
                ray.init(num_cpus=max_cpu)
        else:
            ray = None

        for genome in population:
            sequence = self.reshape_genome(genome)
            if not self.is_feasible_sequence(sequence):
                raise ValueError("Population contains an infeasible sequence during evaluation.")
            if ray is not None:
                tasks.append(
                    self.CrossEntropy.evaluate_candidate.remote(
                        sequence.reshape(-1),
                        self.params,
                        self.original_capacity,
                        self.seq_interval,
                        self.link_indices,
                        self.reconf_start_time,
                        self.reconf_end_time,
                        self.time_steps,
                        self.objective_func,
                    )
                )

        feasible_costs = []
        if ray is not None and tasks:
            print(f"  Evaluating {len(tasks)} feasible individuals in parallel")
            feasible_costs = ray.get(tasks)
            ray.shutdown()
        else:
            feasible_costs = []

        if max_cpu <= 1:
            from mfd_dynamics import MFD_Dynamics

            print(f"  Evaluating {len(population)} feasible individuals serially")
            for idx, genome in enumerate(population, start=1):
                sequence = self.reshape_genome(genome)
                params_copy = copy.deepcopy(self.params)
                params_copy.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                sim = MFD_Dynamics(params_copy, output_path=None)
                for _ in range(sim.sim_start_step, sim.sim_end_step):
                    if (sim.step % self.seq_interval == 0) and (self.reconf_start_time < sim.step <= self.reconf_end_time):
                        sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                        step_idx = (sim.step // self.seq_interval) - (self.reconf_start_time // self.seq_interval) - 1
                        contraflow_idx = sequence[step_idx]
                        for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                            sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                            sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
                    sim.step_simulation()
                feasible_costs.append(self.CrossEntropy.compute_cost(sim, self.reconf_end_time, self.objective_func))
                if idx == len(population) or idx % max(1, len(population) // 5) == 0:
                    print(f"    completed evaluations: {idx}/{len(population)}")

        fitness_scores = np.array(feasible_costs, dtype=float)
        return fitness_scores

    def run(self, max_cpu: int = 1, max_sim_evals: int | None = None):
        population, init_raw_candidates = self.initialize_population()
        best_sequence = None
        best_cost = -np.inf
        history_records = []
        cumulative_candidate_generations = init_raw_candidates
        cumulative_simulation_evaluations = 0
        run_start_time = time.time()

        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            fitness_scores = self.evaluate_population(population, max_cpu=max_cpu)
            feasible_count = len(population)
            cumulative_simulation_evaluations += feasible_count
            generation_best = float(np.max(fitness_scores))
            if generation_best > best_cost:
                best_cost = generation_best
                best_sequence = self.reshape_genome(population[int(np.argmax(fitness_scores))])
                print(f"  new best cost: {best_cost}")

            history_records.append(
                {
                    "generation": generation,
                    "population_size": len(population),
                    "candidate_evaluations": cumulative_candidate_generations if generation == 0 else generation_candidate_generations,
                    "simulation_evaluations": feasible_count,
                    "cumulative_candidate_evaluations": cumulative_candidate_generations,
                    "cumulative_simulation_evaluations": cumulative_simulation_evaluations,
                    "best_generation_cost": generation_best,
                    "best_so_far_cost": best_cost,
                    "elapsed_time_sec": time.time() - run_start_time,
                    "mean_cost": float(np.mean(fitness_scores)),
                    "std_cost": float(np.std(fitness_scores)),
                    "feasible_ratio": float(feasible_count / max(cumulative_candidate_generations if generation == 0 else generation_candidate_generations, 1)),
                    "use_zdd": bool(self.use_zdd),
                    "config": self.config_name,
                    "seed": self.seed,
                }
            )

            if max_sim_evals is not None and cumulative_simulation_evaluations >= max_sim_evals:
                print(f"Reached max_sim_evals={max_sim_evals}; stopping GA optimization.")
                break

            elite_idx = int(np.argmax(fitness_scores))
            next_population = [population[elite_idx].copy()]
            generation_candidate_generations = 0
            print("  Generating next feasible population...")
            while len(next_population) < self.population_size:
                parent1 = self.tournament_select(population, fitness_scores)
                parent2 = self.tournament_select(population, fitness_scores)
                child, attempts = self.generate_feasible_child(parent1, parent2)
                generation_candidate_generations += attempts
                next_population.append(child)
                if len(next_population) == self.population_size or len(next_population) % max(1, self.population_size // 5) == 0:
                    print(
                        f"    next population: {len(next_population)}/{self.population_size} "
                        f"(raw child attempts this generation: {generation_candidate_generations})"
                    )
            cumulative_candidate_generations += generation_candidate_generations
            population = next_population[: self.population_size]
            print(
                f"  generation summary: best={generation_best}, "
                f"sim_evals={feasible_count}, raw_candidates={generation_candidate_generations}, "
                f"elapsed={time.time() - run_start_time:.1f}s"
            )

        pd.DataFrame(history_records).to_csv(self.output_dir / "optimization_history.csv", index=False)
        if best_sequence is not None:
            pd.DataFrame(best_sequence).to_csv(self.output_dir / "best_sequence.csv", index=False, header=False)

        metadata = {
            "objective": self.objective_func,
            "background_ratio": self.params.background_ratio,
            "seed": self.seed,
            "use_zdd": self.use_zdd,
            "ga_config": self.ga_config,
            "reconf_start_time": self.reconf_start_time,
            "reconf_end_time": self.reconf_end_time,
            "seq_interval": self.seq_interval,
            "max_sim_evals": max_sim_evals,
        }
        (self.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return best_sequence, best_cost, pd.DataFrame(history_records)


def default_output_dir(params, objective: str, config_name: str, use_zdd: bool, seed: int) -> Path:
    method_name = "ga_zdd" if use_zdd else "ga"
    return (
        Path("../../output/optimizer_baselines")
        / method_name
        / f"green_{params.green_split}"
        / params.demand
        / params.demand_variation
        / f"back{params.background_ratio}"
        / f"{int(params.simulation_start_time/60)}_{int(params.simulation_end_time/60)}_control6_{objective}"
        / config_name
        / f"seed_{seed}"
    )


def main():
    args = parse_args()
    from parameters_ndp import Parameters

    params = Parameters(background_ratio=args.background_ratio)
    target_graph = parse_binary_vector(args.target_graph)
    if args.output_dir is None:
        output_dir = default_output_dir(params, args.objective, args.config, args.use_zdd, args.seed)
    else:
        output_dir = args.output_dir.resolve()

    ga_config = copy.deepcopy(GA_CONFIGS[args.config])
    if args.generations > 0:
        ga_config["generations"] = args.generations
    if args.population_size > 0:
        ga_config["population_size"] = args.population_size

    optimizer = GeneticSequenceOptimizer(
        params=params,
        target_graph=target_graph,
        objective_func=args.objective,
        reconf_start_time=args.reconf_start_time,
        reconf_end_time=args.reconf_end_time,
        seq_interval=args.seq_interval,
        ga_config=ga_config,
        config_name=args.config,
        seed=args.seed,
        use_zdd=args.use_zdd,
        output_dir=output_dir,
    )
    best_sequence, best_cost, _ = optimizer.run(
        max_cpu=args.max_cpu,
        max_sim_evals=(None if args.max_sim_evals <= 0 else args.max_sim_evals),
    )
    print(f"Saved GA baseline outputs to {output_dir}")
    print(f"Best cost: {best_cost}")
    if best_sequence is not None:
        print(best_sequence)


if __name__ == "__main__":
    main()
