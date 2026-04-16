# Dissertation Codebase

This repository collects the analysis and experiment code used across multiple dissertation chapters.
The top-level structure is organized by chapter so that each chapter can be shared, cleaned, and documented independently for GitHub release.

## Directory Overview

```text
.
├── Chapter1/   # Early validation and small experiments
├── Chapter3/   # Evacuation traffic, network partitioning, and reconfiguration
├── Chapter4/   # Chapter 4 materials
└── Chapter5/   # Dynamic land-use model, estimation, simulation, and counterfactuals
```

## Chapter 3

`Chapter3/` contains the Chapter 3 workflow for:

- partitioning a road network into aggregated zones,
- estimating MFD-based aggregate traffic dynamics,
- comparing contraflow and network reconfiguration policies under evacuation scenarios,
- linking the aggregate model back to SUMO micro-simulation.

Main entry points:

- `Chapter3/code/network_partitioning/network_partitioning.py`
- `Chapter3/code/sumo/simulation_parking.py`
- `Chapter3/code/reconfigration/src/model/collect_table5_metrics.py`
- `Chapter3/code/reconfigration/src/model/run_table5_batch.sh`


## Chapter 5

`Chapter5/` contains the Chapter 5 workflow for:

- reading and preprocessing zoning, population, LOS, and housing data,
- estimating policy functions and a residential location-choice model,
- running forward simulations from the estimated model,
- solving local equilibrium paths for counterfactual policy analysis.

Main entry points:

- `Chapter5/main.py`
- `Chapter5/counterfactual.py`
- `Chapter5/demand_estimation.py`
