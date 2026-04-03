#!/bin/bash
# ==============================================================================
# eval_wer.sh — WER/CER evaluation for TTS / speech generation
#
# Usage:
#   bash eval_wer.sh <text_json> <audios_dir> <lang> [num_gpus]
#
# Arguments:
#   text_json   Path to JSON file with reference texts (samples_en.json or samples_zh.json)
#   audios_dir  Directory containing generated audio files (.wav / .mp3)
#   lang        Language: "en" (uses Whisper-large-v3) or "zh" (uses paraformer-zh)
#   num_gpus    Number of GPUs to use for parallel ASR (default: 1)
#
# Output:
#   <audios_dir>/wav_res_ref_text.wer   — final WER summary
# ==============================================================================

set -e

text_json=$1
audios_dir=$2
lang=$3
num_gpus=${4:-1}

if [ -z "$text_json" ] || [ -z "$audios_dir" ] || [ -z "$lang" ]; then
    echo "Usage: bash eval_wer.sh <text_json> <audios_dir> <lang> [num_gpus]"
    echo "  lang: en or zh"
    exit 1
fi

wav_wav_text=$audios_dir/wav_res_ref_text
score_file=$audios_dir/wav_res_ref_text.wer

# Step 1: Build audio-text pairing file
python3 get_wavs_text_from_json.py \
    --text-json "$text_json" \
    --audios-dir "$audios_dir" \
    --out-file "$wav_wav_text" \
    --auto-detect-id-suffixes

# Step 2: Pre-download / verify ASR checkpoints
python3 prepare_ckpt.py

# Step 3: Split work across GPUs and run ASR + WER in parallel
timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_${timestamp}/
mkdir -p "$thread_dir"

num=$(wc -l < "$wav_wav_text")
num_per_thread=$(( num / num_gpus + 1 ))
split -l "$num_per_thread" --additional-suffix=.lst -d "$wav_wav_text" "$thread_dir/thread-"

out_dir="$thread_dir/results/"
mkdir -p "$out_dir"

for rank in $(seq 0 $(( num_gpus - 1 ))); do
    sub_score_file="$out_dir/thread-$(printf '%02d' $rank).wer.out"
    CUDA_VISIBLE_DEVICES=$rank python3 run_wer.py \
        "$thread_dir/thread-$(printf '%02d' $rank).lst" \
        "$sub_score_file" \
        "$lang" &
done
wait

# Step 4: Merge results and compute mean WER
rm -f "$out_dir/merge.out"
cat "$out_dir"/thread-*.wer.out >> "$out_dir/merge.out"
python3 average_wer.py "$out_dir/merge.out" "$score_file"

echo "Done. Results written to: $score_file"
