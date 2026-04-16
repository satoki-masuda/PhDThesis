#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$REPO_ROOT"

PYTHON=(poetry run python3)

OUTPUT_ROOT="${TABLE5_OUTPUT_ROOT:-$REPO_ROOT/code/reconfigration/output/table5_batch}"
SEED="${TABLE5_SEED:-42}"
RANDOM_SAMPLES="${TABLE5_RANDOM_SAMPLES:-100}"
MAX_CPU="${TABLE5_MAX_CPU:-50}"
MAX_SIM_EVALS="${TABLE5_MAX_SIM_EVALS:-0}"
ELITE_COUNT="${TABLE5_ELITE_COUNT:-10}"
NMIN="${TABLE5_CEM_NMIN:-400}"
NMAX="${TABLE5_CEM_NMAX:-800}"
NUNIT="${TABLE5_CEM_NUNIT:-200}"
D="${TABLE5_CEM_D:-4}"
DISABLE_EARLY_STOP_FLAG=""
if [[ "${TABLE5_DISABLE_EARLY_STOP:-1}" == "1" ]]; then
  DISABLE_EARLY_STOP_FLAG="--disable-early-stop"
fi

SUMMARY_CSV="$OUTPUT_ROOT/table5_summary.csv"
rm -f "$SUMMARY_CSV"
mkdir -p "$OUTPUT_ROOT"

SCENARIOS=(
  "93_max_1.0"
  "93_max_0.9"
  "93_max_0.8"
  "53_0.95_1.0"
)

echo "Table 5 batch configuration"
echo "  output root     : $OUTPUT_ROOT"
echo "  seed            : $SEED"
echo "  random samples  : $RANDOM_SAMPLES"
echo "  max cpu         : $MAX_CPU"
echo "  max sim evals   : ${MAX_SIM_EVALS:-0} (0 means unlimited)"
echo "  CEM elite count : $ELITE_COUNT"
echo "  CEM nmin/nmax   : $NMIN / $NMAX"
echo "  CEM nunit       : $NUNIT"
echo "  CEM d           : $D"
echo "  early stop off  : ${TABLE5_DISABLE_EARLY_STOP:-1}"

for scenario in "${SCENARIOS[@]}"; do
  echo
  echo "============================================================"
  echo "Running scenario: $scenario"
  echo "============================================================"
  "${PYTHON[@]}" code/reconfigration/src/model/collect_table5_metrics.py \
    --scenario "$scenario" \
    --output-root "$OUTPUT_ROOT" \
    --summary-csv "$SUMMARY_CSV" \
    --seed "$SEED" \
    --random-samples "$RANDOM_SAMPLES" \
    --max-cpu "$MAX_CPU" \
    --max-sim-evals "$MAX_SIM_EVALS" \
    --elite-count "$ELITE_COUNT" \
    --nmin "$NMIN" \
    --nmax "$NMAX" \
    --nunit "$NUNIT" \
    --d "$D" \
    $DISABLE_EARLY_STOP_FLAG
done

echo
echo "Done."
echo "Combined summary: $SUMMARY_CSV"
