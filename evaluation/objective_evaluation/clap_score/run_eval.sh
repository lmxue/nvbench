#!/bin/bash
# ==============================================================================
# run_eval.sh — CLAP score evaluation for TTS / speech generation
#
# Usage:
#   bash run_eval.sh
#
# Configure the variables below, then run this script.
# ==============================================================================

# Path to directory containing generated audio files (.wav / .mp3)
AUDIOS_DIR="/path/to/your/audio/outputs"

# Path to JSON file with reference texts (e.g., samples_en.json)
# Must be a JSON array where each object has "id" and a text field.
TEXTS_JSON="/path/to/samples_en.json"

# Text field to use as CLAP query:
#   "caption_with_nvb"  — natural language caption (recommended for prompt-based tasks)
#   "text"              — plain transcript
#   "text_with_mark"    — text with NVC tags
CLAP_TEXT_KEY="caption_with_nvb"

# Output CSV file
OUT_CSV="clap_results.csv"

# Comma-separated GPU IDs (e.g. "0,1"). Leave empty to use all available GPUs.
GPU_IDS="0"

# ==============================================================================
# Run
# ==============================================================================
echo "Running CLAP evaluation..."
echo "  AUDIOS_DIR  : $AUDIOS_DIR"
echo "  TEXTS_JSON  : $TEXTS_JSON"
echo "  TEXT_KEY    : $CLAP_TEXT_KEY"
echo "  OUT_CSV     : $OUT_CSV"
echo ""

python3 eval_clap.py \
    --audios-dir "$AUDIOS_DIR" \
    --clap-texts-json "$TEXTS_JSON" \
    --clap-text-key "$CLAP_TEXT_KEY" \
    --out "$OUT_CSV" \
    --device cuda \
    --gpu-ids "$GPU_IDS" \
    --run-clap \
    --checkpoint-every 50

echo ""
echo "Done. Results saved to: $OUT_CSV"
