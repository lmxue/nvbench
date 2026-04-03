#!/bin/bash

# ==============================================================================
# Run statistics_and_summary.py
# Scans the current directory for eval_results_*.json files and prints
# per-system mean scores and variance for all metrics.
# ==============================================================================

cd "$(dirname "$0")"

echo "Running statistics and summary..."
echo ""

python3 statistics_and_summary.py

echo ""
echo "Done."
