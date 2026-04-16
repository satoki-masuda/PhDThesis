#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$REPO_ROOT"

OBJECTIVE="${OBJECTIVE:-normal}"
BACKGROUND_RATIO="${BACKGROUND_RATIO:-0.8}"
MAX_CPU="${MAX_CPU:-25}"
SEEDS_STR="${SEEDS:-1 2 3}"
GA_ZDD_CONFIGS_STR="${GA_ZDD_CONFIGS:-balanced}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/code/reconfigration/output/validation/optimizer_comparison}"
MAX_SIM_EVALS="${MAX_SIM_EVALS:-0}"
GA_GENERATIONS="${GA_GENERATIONS:-1000}"
GA_POPULATION_SIZE="${GA_POPULATION_SIZE:-100}"
CEM_DISABLE_EARLY_STOP="${CEM_DISABLE_EARLY_STOP:-1}"
CEM_NMIN="${CEM_NMIN:-100}"
CEM_NMAX="${CEM_NMAX:-300}"
CEM_NUNIT="${CEM_NUNIT:-100}"
CEM_ELITE_COUNT="${CEM_ELITE_COUNT:-10}"

read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a GA_CONFIGS_ARR <<< "$GA_ZDD_CONFIGS_STR"

print_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

build_appendix_labels() {
  local labels="CEM+ZDD"
  local config
  for config in "${GA_CONFIGS_ARR[@]}"; do
    labels="${labels},GA+ZDD-${config}"
  done
  printf '%s\n' "$labels"
}

cem_extra_args=()
if [[ "$CEM_DISABLE_EARLY_STOP" == "1" ]]; then
  cem_extra_args+=(--disable-early-stop)
fi

print_section "Optimizer comparison setup"
echo "repo root        : $REPO_ROOT"
echo "objective        : $OBJECTIVE"
echo "background ratio : $BACKGROUND_RATIO"
echo "max cpu          : $MAX_CPU"
echo "max sim evals    : $MAX_SIM_EVALS"
echo "seeds            : $SEEDS_STR"
echo "GA+ZDD configs   : $GA_ZDD_CONFIGS_STR"
echo "GA generations   : $GA_GENERATIONS"
echo "GA population    : $GA_POPULATION_SIZE"
echo "CEM nmin/nmax    : $CEM_NMIN / $CEM_NMAX"
echo "CEM nunit        : $CEM_NUNIT"
echo "CEM elite count  : $CEM_ELITE_COUNT"
echo "disable CEM stop : $CEM_DISABLE_EARLY_STOP"
echo "output dir       : $OUTPUT_DIR"

print_section "Running CEM+ZDD baselines"
for seed in "${SEEDS_ARR[@]}"; do
  echo "[CEM+ZDD] seed=$seed"
  poetry run python3 code/reconfigration/src/model/run_cem_sequence_baseline.py \
    --objective "$OBJECTIVE" \
    --background-ratio "$BACKGROUND_RATIO" \
    --seed "$seed" \
    --elite-count "$CEM_ELITE_COUNT" \
    --nmin "$CEM_NMIN" \
    --nmax "$CEM_NMAX" \
    --nunit "$CEM_NUNIT" \
    --max-cpu "$MAX_CPU" \
    --max-sim-evals "$MAX_SIM_EVALS" \
    "${cem_extra_args[@]}"
done

print_section "Running GA+ZDD baselines"
for config in "${GA_CONFIGS_ARR[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    echo "[GA+ZDD] config=$config seed=$seed"
    poetry run python3 code/reconfigration/src/model/genetic_sequence_baseline.py \
      --objective "$OBJECTIVE" \
      --background-ratio "$BACKGROUND_RATIO" \
      --config "$config" \
      --seed "$seed" \
      --use-zdd \
      --max-cpu "$MAX_CPU" \
      --max-sim-evals "$MAX_SIM_EVALS" \
      --generations "$GA_GENERATIONS" \
      --population-size "$GA_POPULATION_SIZE"
  done
done

print_section "Plotting optimizer comparison"
SERIES_ARGS=(
  --series "CEM+ZDD=$REPO_ROOT/code/reconfigration/output/optimizer_baselines/cem_zdd/**/optimization_history.csv"
)

for config in "${GA_CONFIGS_ARR[@]}"; do
  SERIES_ARGS+=(
    --series "GA+ZDD-${config}=$REPO_ROOT/code/reconfigration/output/optimizer_baselines/ga_zdd/**/${config}/**/optimization_history.csv"
  )
done

APPENDIX_LABELS="$(build_appendix_labels)"

poetry run python3 code/reconfigration/src/model/plot_optimizer_comparison.py \
  "${SERIES_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --objective "$OBJECTIVE" \
  --main-labels "" \
  --appendix-labels "$APPENDIX_LABELS"

print_section "Done"
echo "Saved comparison plots to $OUTPUT_DIR"
