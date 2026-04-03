#!/bin/bash
# ==============================================================================
# eval_dnsmos.sh — DNSMOS P.835 evaluation for TTS / speech generation
#
# Usage:
#   bash eval_dnsmos.sh <audios_dir> [output_csv]
#
# Arguments:
#   audios_dir   Directory containing generated audio files (.wav and/or .mp3)
#   output_csv   Path for per-file CSV results (default: <audios_dir>/dnsmos.csv)
#
# The script runs DNSMOS P.835 on all audio files in <audios_dir> and prints
# the mean OVRL / SIG / BAK scores.
#
# Note: This script must be run from the directory containing dnsmos_mp3.py
#       (the DNSMOS/ and pDNSMOS/ model folders must be present alongside).
# ==============================================================================

set -e

audios_dir=${1:?Usage: bash eval_dnsmos.sh <audios_dir> [output_csv]}
output_csv=${2:-"${audios_dir}/dnsmos.csv"}
result_log="${audios_dir}/dnsmos_summary.log"

echo "Running DNSMOS P.835 evaluation..."
echo "  audios_dir : $audios_dir"
echo "  output_csv : $output_csv"
echo ""

# Step 1: Run DNSMOS — supports both .wav and .mp3
python3 dnsmos_mp3.py -t "$audios_dir" -o "$output_csv"

# Step 2: Print mean scores
echo ""
echo "Mean scores:"
python3 avg_mos.py "$output_csv" | tee -a "$result_log"

echo ""
echo "Done. Per-file results: $output_csv"
echo "      Summary log:      $result_log"
