# Chapter 3

Chapter 3 studies evacuation traffic control using a zoned road network, MFD-based aggregate dynamics, and reconfiguration or contraflow policies.

## Where To Start

If you are new to this chapter, start with these files:

- `code/network_partitioning/network_partitioning.py`
  The main implementation for creating zone partitions from the SUMO network and estimating per-cluster MFDs.
- `code/sumo/simulation_parking.py`
  The main SUMO baseline simulation entry point.
- `code/sumo/simulation_parking_contraflow.py`
  Replays time-varying contraflow policies in SUMO.
- `code/reconfigration/src/model/collect_table5_metrics.py`
  Aggregates multiple scenarios and policies in the format used for the paper tables.
- `code/reconfigration/src/model/run_table5_batch.sh`
  Batch launcher for the Table 5 style experiments.

`reconfigration/` intentionally keeps the original directory name, even though it contains a typo, because the existing code and relative paths already depend on it.

## Directory Structure

```text
Chapter3/
├── code/
│   ├── network_partitioning/
│   ├── sumo/
│   └── reconfigration/
├── pyproject.toml
└── poetry.lock
```

## `network_partitioning/`

- `network_partitioning.py`
  Loads the SUMO network, constructs the line graph, computes link similarity, runs recursive bipartitioning, visualizes clusters, and saves MFD and regression outputs.
- `sigma_i.py`
  Small sensitivity script for comparing partitions under different `sigma_i` values.
- `zoning.csv`
  Saved mapping from edge IDs to clusters.
- `log_norm/`
  Generated MFD, regression, and visualization outputs for different partition settings.
- `network_partitioning.ipynb`
  Supporting notebook for exploratory checks. The main reproducible workflow is the `.py` code.

## `sumo/`

`code/sumo/` contains the micro-simulation layer used before or alongside the aggregate model.

### `network/`

- `network.py`
  Builds the base SUMO network from OSM data and exports `output.net.xml` and related files.
- `network_partitioned.py`
  Builds zone-specific subnetworks using `network_partitioning/zoning.csv`.
- `network/data/`
  SUMO network files, GeoJSON exports, contraflow definitions, and related generated assets.

### `demand/`

- `demand/od_koto/od_demand.py`
  Generates normal-demand SUMO trips and routes from OD data.
- `demand/od_koto/od_demand_parking.py`
  Builds the main evacuation-demand scenario with shelter and parking information.
- `demand/random/random_demand.py`, `random_DUE.py`
  Small random-demand test scripts for validation and sensitivity experiments.

### `evac_shelter/`

- `FLOOD_API.py`
  Utility helpers for querying flood depth and duration from geographic coordinates.

### Simulation Entry Points

- `simulation_parking.py`
  Runs the baseline SUMO simulation and exports edge data, tripinfo, and route traces.
- `simulation_parking_contraflow.py`
  Reads `contraflow_simulation/*/best_sequence.csv` and applies time-varying lane control in SUMO.
- `contraflow_simulation/`
  Stores aggregate-model policy outputs that are replayed on the SUMO side.

## `reconfigration/`

### `data/`

- `data/raw/`
  Raw input data such as networks, shelter lists, and zoning definitions.
- `data/processed/`
  Preprocessed OD tables, boundary capacities, average trip lengths, route dictionaries, and zone polygons.

### `src/data/`

- `mfd_zoning.py`
  Utilities for loading zone partitions, building adjacency relationships, enumerating routes, plotting zones, and preparing MFD inputs.
- `od_generate.py`
  Converts SUMO trip and OD XML files into aggregate OD CSV inputs for normal and evacuation demand.

### `src/model/`

Core model files:

- `parameters_ndp.py`
  Central configuration and data-loading module for MFD parameters, OD data, route probabilities, and capacity constraints.
- `mfd_dynamics.py`
  The aggregate traffic simulation engine.
- `reconf_shortest.py`
  Reconfiguration logic for shortest-step transitions between network states.
- `reconf_horizon.py`
  Reconfiguration logic for horizon-constrained transitions and related sensitivity analysis.
- `cross-entropy.py`
  Cross-entropy optimization for searching reconfiguration sequences.
- `discrete_mpc.py`
  MPC-style sequential policy comparison.
- `value_iteration.py`
  Value-iteration experiments over the discrete state space.
- `contraflow_ndp.py`
  Older static contraflow optimization baseline.

Comparison, aggregation, and plotting scripts:

- `collect_table5_metrics.py`
- `run_table5_batch.sh`
- `run_optimizer_comparison.sh`
- `plot_optimizer_comparison.py`
- `plot_transition_congestion_comparison.py`
- `plot_pareto_frontier.py`
- `compare_macro_micro_validation.py`
- `compare_sampling_efficiency.py`
- `analyze_step_change_limit_sensitivity.py`
- `format_step_change_limit_table.py`

Support scripts:

- `make_constraint.py`
- `memory_calculation_shortest.py`
- `memory_calculation_horizon.py`
- `run_cem_sequence_baseline.py`
- `genetic_sequence_baseline.py`
- `logger_writer.py`
- `cost_function.py`

### `output/`

This directory mainly stores generated outputs such as MFD results, reconfiguration sequences, plots, tables, and animation frames.

Be careful with `output/ScaledMFD/fitted_scale_factors.csv` and `output/ParkingSuccessRate/zone_speed_penalty/estimated_params.csv`: the current code still reads them as inputs via `parameters_ndp.py`, so they are not yet disposable outputs.

## Suggested Reading Order

1. `code/sumo/network/network.py`
2. `code/network_partitioning/network_partitioning.py`
3. `code/sumo/demand/od_koto/od_demand_parking.py`
4. `code/sumo/simulation_parking.py` or `simulation_parking_contraflow.py`
5. `code/reconfigration/src/data/od_generate.py`
6. `code/reconfigration/src/data/mfd_zoning.py`
7. `code/reconfigration/src/model/parameters_ndp.py`
8. `code/reconfigration/src/model/mfd_dynamics.py`
9. `code/reconfigration/src/model/reconf_shortest.py`, `reconf_horizon.py`, or `cross-entropy.py`
10. `code/reconfigration/src/model/collect_table5_metrics.py`
