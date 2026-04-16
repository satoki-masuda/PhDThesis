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

See [Chapter3/README.md](/Users/masudasatoki/Desktop/D論コードまとめ/English/Chapter3/README.md) for the full Chapter 3 structure.

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

See [Chapter5/README.md](/Users/masudasatoki/Desktop/D論コードまとめ/English/Chapter5/README.md) for details.

## Notes For Public Release

- `Chapter5/data/raw/` still contains original source datasets. Redistribution permissions should be checked before publication.
- Many outputs have already been removed or are intended to be regenerated locally.
- `Chapter5/notebook/` is a supporting analysis notebook and does not perfectly match the streamlined main code path.
- `Chapter3/` contains many generated artifacts under `output/` and SUMO-related directories; those are generally better treated as reproducible outputs than as core source files.
