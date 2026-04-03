#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_aggregate.sh — Aggregate metrics across multiple prediction runs
#
# Computes run-level mean / variance / std for all metrics, producing a summary
# TSV suitable for reporting cross-run stability in a paper.
#
# Usage:
#   bash run_aggregate.sh                      # uses defaults below
#   RUN_ROOTS="out1 out2 out3" bash run_aggregate.sh
# ==============================================================================

SCRIPT_PATH="${SCRIPT_PATH:-$(dirname "$0")/aggregate_runs.py}"

# Space-separated list of prediction output root directories (one per run).
# Each root must follow the structure: <root>/<system>/<lang>/*.metrics.json
RUN_ROOTS="${RUN_ROOTS:-prediction_output prediction_output-2 prediction_output-3}"

OUT="${OUT:-summary_metrics_3runs_mean_var.tsv}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "FATAL: aggregate_runs.py not found: $SCRIPT_PATH" >&2; exit 2
fi

# shellcheck disable=SC2086
python "$SCRIPT_PATH" \
  --run_roots $RUN_ROOTS \
  --out "$OUT"

echo "Saved: $OUT"
