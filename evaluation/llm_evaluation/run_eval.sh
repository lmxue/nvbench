#!/bin/bash

# ==============================================================================
# Run llm_eval.py — LLM-as-Judge evaluation for TTS / speech generation
# ==============================================================================

# ==============================================================================
# Proxy (optional)
# Uncomment and set if your environment requires a proxy to reach Google APIs.
# ==============================================================================
# export USE_PROXY=1
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"
# export HTTP_PROXY="$http_proxy"
# export HTTPS_PROXY="$https_proxy"

# ==============================================================================
# API configuration
# ==============================================================================
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
# Optional: override model (default: models/gemini-2.5-pro)
# export GEMINI_MODEL="models/gemini-2.5-pro"

# ==============================================================================
# Data directory
# Set DATA_DIR to the folder containing your samples_*.json and manifest_*.json.
# Defaults to ./example_data (for a quick format check without real audio).
# ==============================================================================
export DATA_DIR="./example_data"

# ==============================================================================
# Evaluation parameters
# ==============================================================================
export MAX_WORKERS=64           # Number of concurrent API threads
export JUDGE_TEMPERATURE=0.2    # Sampling temperature (lower = more consistent)
export GLOBAL_SEED=1234         # Random seed for reproducibility

# ==============================================================================
# GROUP_COMPARE mode (recommended)
# Evaluates all systems for the same sample together with anonymized labels,
# simulating a real listening test. Set GROUP_COMPARE=0 to evaluate per-system.
# ==============================================================================
export GROUP_COMPARE=1
export GROUP_SIZE=0             # 0 = compare all systems at once (recommended)
export ANCHOR_SYSTEM=""         # Optional fixed anchor system key; leave empty to disable

# ==============================================================================
# Rater simulation
# Simulates multiple independent raters with different calibration profiles.
# ==============================================================================
export N_RATERS=4               # Number of simulated raters
export MIN_RATERS_PER_SAMPLE=1  # Minimum raters per sample
export RATER_COVERAGE=0.25      # Fraction of extra raters per sample (1.0 = all raters score all samples)

# ==============================================================================
# Task selection
# Available: zh-prompt, zh-tag, en-prompt, en-tag
# Leave TASKS empty to run all tasks.
# ==============================================================================
export TASKS="en-prompt"        # Example: only run English prompt-based task

# ==============================================================================
# Run
# ==============================================================================
echo "Starting LLM-as-Judge evaluation..."
echo "  GROUP_COMPARE : $GROUP_COMPARE"
echo "  N_RATERS      : $N_RATERS"
echo "  RATER_COVERAGE: $RATER_COVERAGE"
echo "  TASKS         : ${TASKS:-all}"
echo "  MAX_WORKERS   : $MAX_WORKERS"
echo "  DATA_DIR      : $DATA_DIR"
echo ""

python llm_eval.py

echo ""
echo "Evaluation complete."
