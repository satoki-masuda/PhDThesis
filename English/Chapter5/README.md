# Chapter 5

Chapter 5 implements a dynamic model of land use, residential location choice, forward simulation, and counterfactual policy analysis.

The workflow is organized into four main parts:

- data loading and preprocessing,
- model estimation,
- forward simulation and local equilibrium calculation,
- counterfactual analysis.

## What You Can Do Here

- estimate policy functions and the residential location-choice model,
- simulate future population, investment, development, and land prices,
- run counterfactual policy changes for selected zones,
- save and visualize estimated and simulated outputs.

## Main Entry Points

- `main.py`
  Main estimation script for Chapter 5. It loads the shared configuration, prepares the common datasets, estimates policy functions, estimates the transition model, generates forward-simulation features, and runs the structural estimation step.
- `counterfactual.py`
  Loads the estimated model, applies a policy intervention to the target zone, recomputes a local equilibrium path, and saves comparison plots.
- `demand_estimation.py`
  Small standalone script for estimating only the residential location-choice model.
- `config.yaml`
  Shared configuration for time periods, zoning schemes, discounting, forward-simulation settings, and counterfactual settings.

## Directory Structure

```text
Chapter5/
├── main.py
├── counterfactual.py
├── demand_estimation.py
├── config.yaml
├── pyproject.toml
├── model/
├── simulation/
├── utils/
├── data/
│   ├── raw/
│   └── processed/
└── notebook/
```

## Subdirectories

### `model/`

- `transition_model.py`
  Core class for residential choice and population transitions. It manages zone definitions, distance matrices, spatial weights, and estimation data assembly.
- `data_reading.py`
  Reads raw data on population, building activity, LOS, PT, and zone mappings, then converts them into dictionaries and tables used by the model.
- `policy_estimation.py`
  Estimation, serialization, and loading of policy functions for government investment and development.
- `mnl.py`, `mxl.py`
  Residential choice model estimators. `MXL` is the main specification used in the current workflow.
- `forward_simulation.py`
  Simulates future paths of population, investment, development, and prices from the estimated policy and transition models.
- `payoff_structure.py`
  Defines the payoff features used in the structural estimation step.
- `bbl_objective_function.py`
  Computes the BBL-style objective function.

### `simulation/`

- `mpe_local_driver.py`
  Solves a local MPE path market by market using sequential best responses.
- `best_response_mpe.py`
  Repeatedly runs forward simulation and numerically solves best responses for each player.
- `policy_override.py`
  Wrapper classes for injecting zone-specific policy overrides and caps into an existing estimated model.
- `visualization.py`
  Plotting utilities for comparing simulated and observed series.

### `utils/`

- `data_loader.py`
  Shared data-preparation entry point used by both `main.py` and `counterfactual.py`.
- `config_manager.py`
  Centralized configuration loader with defaults and validation.
- `helpers.py`
  Small helper functions for JSON I/O, directory creation, and timing.

### `data/`

- `data/raw/`
  Raw source datasets for population, building records, PT, zoning definitions, and related supporting data.
- `data/processed/`
  Zone-level CSV, JSON, and NumPy files used directly during estimation and simulation.

### `notebook/`

- `notebook/data_analysis.ipynb`
  Supporting analysis notebook used for preprocessing and exploratory work. It is helpful for context, but the main reproducible pipeline is the Python code above.

## Execution Flow

1. Check the settings in `config.yaml`.
2. Run `main.py` for estimation and structural parameter fitting.
3. Run `counterfactual.py` using the saved estimates.
4. Review the generated outputs and figures under `output/`.

## Raw Data Folder Names

In this English copy, the major raw-data directories under `data/raw/` have been renamed to English:

- `real_estate_prices/`
- `population_by_district_age/`
- `district_social_mobility/`
- `interdistrict_population_migration/`
- `facility_locations/`
- `disaster_risk/`
- `population_by_chome/`

Code references in the streamlined Chapter 5 workflow now follow these English names.
