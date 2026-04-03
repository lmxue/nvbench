#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
# run_predict.sh — NVC detection: Gemini-based prediction
#
# For each TTS system audio directory, calls predict_nvc.py to verify whether
# each NVC tag is present in the synthesized audio and localize its position.
# Outputs: <system>_<lang>_<model>.json  (predictions)
#          <system>_<lang>_<model>.raw.json (audit trail)
#          <system>_<lang>_<model>.skipped.json (failures)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0) Proxy (optional)
# Uncomment if your environment requires a proxy to reach Google APIs.
# ------------------------------------------------------------------------------
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="${http_proxy}"
# export HTTP_PROXY="${http_proxy}"
# export HTTPS_PROXY="${http_proxy}"
# unset ALL_PROXY

# ------------------------------------------------------------------------------
# 1) API key (required)
# Export GEMINI_API_KEY in your shell before running, or set it here.
# ------------------------------------------------------------------------------
export GEMINI_API_KEY="${GEMINI_API_KEY:?Please export GEMINI_API_KEY before running}"

# ------------------------------------------------------------------------------
# 2) Concurrency — tune for your quota and hardware
# ------------------------------------------------------------------------------
export MAX_WORKERS="${MAX_WORKERS:-16}"
export UPLOAD_CONCURRENCY="${UPLOAD_CONCURRENCY:-4}"
export UPLOAD_RETRY="${UPLOAD_RETRY:-6}"
export UPLOAD_BACKOFF="${UPLOAD_BACKOFF:-0.8}"
export UPLOAD_BACKOFF_MAX="${UPLOAD_BACKOFF_MAX:-12}"

# ------------------------------------------------------------------------------
# 3) Ground-truth JSON paths
# Each file is a JSON array of objects with fields: id, text, text_with_mark,
# non_verbal_events, etc. See README for the exact schema.
# ------------------------------------------------------------------------------
GT_EN="${GT_EN:?Please set GT_EN to the path of your English ground-truth JSON}"
GT_ZH="${GT_ZH:-}"   # Optional; leave empty if you only evaluate English

# ------------------------------------------------------------------------------
# 4) Model and I/O settings
# ------------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-models/gemini-2.5-pro}"
INPUT_MODE="${INPUT_MODE:-upload}"    # upload | bytes
FLUSH_EVERY="${FLUSH_EVERY:-50}"
FLUSH_SECS="${FLUSH_SECS:-10}"
DELETE_UPLOADED="${DELETE_UPLOADED:-1}"
FALLBACK_TO_BYTES="${FALLBACK_TO_BYTES:-1}"
FAIL_FAST="${FAIL_FAST:-0}"           # 0=continue on errors (recommended)

OUT_ROOT="${OUT_ROOT:-./prediction_output}"   # fixed dir enables resume
LOG_ROOT="${LOG_ROOT:-./logs}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# Auto-evaluate after each prediction run (set to 0 to skip)
AUTO_EVAL="${AUTO_EVAL:-0}"

# Evaluation parameters (used only when AUTO_EVAL=1)
DELTA_EN="${DELTA_EN:-2}"   # word-level collar for EN
DELTA_ZH="${DELTA_ZH:-5}"  # char-level collar for ZH

SCRIPT="$(dirname "$0")/predict_nvc.py"

# ------------------------------------------------------------------------------
# 5) Audio directories
# List all directories containing synthesized audio, one per TTS system × language.
# The script infers language from the path (nvb_taxonomy_en / nvb_taxonomy_zh).
# Replace the examples below with your actual paths.
# ------------------------------------------------------------------------------
AUDIO_DIRS=(
  "/path/to/your/system_a/nvb_taxonomy_en"
  "/path/to/your/system_b/nvb_taxonomy_en"
  # "/path/to/your/system_a/nvb_taxonomy_zh"
  # "/path/to/your/system_b/nvb_taxonomy_zh"
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
timestamp() { date +"%Y%m%d_%H%M%S"; }

infer_lang() {
  local p="$1"
  if [[ "$p" == *"/nvb_taxonomy_zh"* ]]; then echo "zh"; return; fi
  if [[ "$p" == *"/nvb_taxonomy_en"* ]]; then echo "en"; return; fi
  [[ "$p" == *"_zh"* ]] && echo "zh" || echo "en"
}

infer_system() {
  local p="$1"
  local base
  base="$(basename "$(dirname "$(dirname "$p")")")"
  echo "${base%%_nve_samples}"
}

sanitize() {
  echo "$1" | sed 's#[^a-zA-Z0-9_-]#_#g'
}

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
echo "MODEL_NAME=${MODEL_NAME}"
echo "MAX_WORKERS=${MAX_WORKERS}  UPLOAD_CONCURRENCY=${UPLOAD_CONCURRENCY}  INPUT_MODE=${INPUT_MODE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo ""

for AUDIO_DIR in "${AUDIO_DIRS[@]}"; do
  if [[ ! -d "${AUDIO_DIR}" ]]; then
    echo "[WARN] skip (not found): ${AUDIO_DIR}"
    continue
  fi

  LANG="$(infer_lang "${AUDIO_DIR}")"
  SYS="$(infer_system "${AUDIO_DIR}")"

  if [[ "${LANG}" == "zh" ]]; then
    if [[ -z "${GT_ZH}" ]]; then
      echo "[WARN] GT_ZH not set; skipping ZH dir: ${AUDIO_DIR}"
      continue
    fi
    GT="${GT_ZH}"
    POS_UNIT="char"
    DELTA="${DELTA_ZH}"
  else
    GT="${GT_EN}"
    POS_UNIT="word"
    DELTA="${DELTA_EN}"
  fi

  OUT_DIR="${OUT_ROOT}/${SYS}/${LANG}"
  mkdir -p "${OUT_DIR}"

  MODEL_TAG="$(sanitize "${MODEL_NAME#models/}")"
  RUN_NAME="$(sanitize "${SYS}_${LANG}_${MODEL_TAG}")"
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}_$(timestamp).log"

  echo "============================================================"
  echo "[RUN] SYS=${SYS} LANG=${LANG}"
  echo "      AUDIO_DIR=${AUDIO_DIR}"
  echo "      GT=${GT}"
  echo "      OUT_DIR=${OUT_DIR}"
  echo "      RUN_NAME=${RUN_NAME}"
  echo "============================================================"

  set +e
  python "${SCRIPT}" \
    "${AUDIO_DIR}" \
    --gt_json "${GT}" \
    --output_dir "${OUT_DIR}" \
    --system_name "${RUN_NAME}" \
    --model_name "${MODEL_NAME}" \
    --input_mode "${INPUT_MODE}" \
    --flush_every "${FLUSH_EVERY}" \
    --flush_secs "${FLUSH_SECS}" \
    --delete_uploaded "${DELETE_UPLOADED}" \
    --fallback_to_bytes "${FALLBACK_TO_BYTES}" \
    --pos_unit "${POS_UNIT}" \
    --max_workers "${MAX_WORKERS}" \
    --upload_concurrency "${UPLOAD_CONCURRENCY}" \
    --upload_retry "${UPLOAD_RETRY}" \
    --upload_backoff "${UPLOAD_BACKOFF}" \
    --upload_backoff_max "${UPLOAD_BACKOFF_MAX}" \
    --fail_fast "${FAIL_FAST}" \
    2>&1 | tee "${LOG_FILE}"
  RET=${PIPESTATUS[0]}
  set -e

  echo "[INFO] exit_code=${RET} for ${RUN_NAME}"

  if [[ "${AUTO_EVAL}" == "1" ]]; then
    PRED_JSON="${OUT_DIR}/${RUN_NAME}.json"
    if [[ -f "${PRED_JSON}" ]]; then
      echo "[EVAL] Computing metrics for ${RUN_NAME} ..."
      python "$(dirname "$0")/eval_metrics.py" \
        --gt_json "${GT}" \
        --pred_json "${PRED_JSON}" \
        --audio_dir "${AUDIO_DIR}" \
        --delta "${DELTA}" \
        --pos_unit "${POS_UNIT}" \
        --per_tag 1 \
        --save_json "${OUT_DIR}/${RUN_NAME}.metrics.json" \
        2>&1 | tee -a "${LOG_FILE}"
    else
      echo "[EVAL][WARN] pred_json not found: ${PRED_JSON}"
    fi
  fi

  echo ""
done

echo "[DONE] All runs finished."
