#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_eval.sh — Compute NVC evaluation metrics from prediction JSON files
#
# Scans PRED_ROOT for *.json prediction files, computes Coverage / Precision /
# Recall / F1 / CA-F1 / TPD / NTD for each (system, language) pair, and writes
# per-file *.metrics.json + a combined TSV summary.
# ==============================================================================

# ------------------------------------------------------------------------------
# Config — set these before running
# ------------------------------------------------------------------------------
PRED_ROOT="${PRED_ROOT:-./prediction_output}"

SCRIPT_PATH="${SCRIPT_PATH:-$(dirname "$0")/eval_metrics.py}"

# Path to ground-truth JSON files (required).
# Set GT_EN (and optionally GT_ZH) before running, or export them in your shell.
GT_EN="${GT_EN:?Please set GT_EN to the path of your English ground-truth JSON}"
GT_ZH="${GT_ZH:-}"   # Optional; leave empty if evaluating English only

# Root directory containing TTS system audio (used to resolve audio_dir per system).
# Expected structure: BM_ROOT/<system_name>_nve_samples/version1/nvb_taxonomy_{en,zh}/
# Leave empty to skip audio_dir resolution (metrics will be computed without file existence check).
BM_ROOT="${BM_ROOT:-}"

# Set FORCE=1 to recompute and overwrite existing *.metrics.json files.
FORCE="${FORCE:-0}"

LOG="${LOG:-eval_metrics_batch.log}"
ERR_LOG="${ERR_LOG:-eval_metrics_batch.errors.log}"
SUMMARY_TSV="${SUMMARY_TSV:-summary_metrics.tsv}"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
ts() { date +"%F %T"; }

resolve_audio_dir() {
  local sys="$1"
  local lang="$2"

  if [[ -z "${BM_ROOT}" ]]; then
    echo ""
    return 0
  fi

  local sys_u="${sys//-/_}"
  local sys_d="${sys//_/-}"

  local candidates=(
    "${BM_ROOT}/${sys}_nve_samples"
    "${BM_ROOT}/${sys_u}_nve_samples"
    "${BM_ROOT}/${sys_d}_nve_samples"
  )

  local base=""
  for c in "${candidates[@]}"; do
    if [[ -d "$c" ]]; then
      base="$c"
      break
    fi
  done

  if [[ -z "$base" ]]; then
    base="$(find "$BM_ROOT" -maxdepth 1 -type d \( \
              -iname "${sys}_nve_samples" -o \
              -iname "${sys_u}_nve_samples" -o \
              -iname "${sys_d}_nve_samples" \
            \) -print -quit 2>/dev/null || true)"
  fi

  if [[ -z "$base" ]]; then
    echo ""
    return 0
  fi

  local adir="${base}/version1/nvb_taxonomy_${lang}"
  [[ -d "$adir" ]] && echo "$adir" || echo ""
}

run_one() {
  local pred_json="$1"

  local rel="${pred_json#${PRED_ROOT}/}"
  local sys="${rel%%/*}"
  local rest="${rel#*/}"
  local lang="${rest%%/*}"

  if [[ "$lang" != "en" && "$lang" != "zh" ]]; then
    echo "[$(ts)] [SKIP] Unknown lang in path: $pred_json" | tee -a "$LOG"
    return 0
  fi

  local gt delta pos_unit
  if [[ "$lang" == "en" ]]; then
    gt="$GT_EN"
    delta=2
    pos_unit="word"
  else
    if [[ -z "${GT_ZH}" ]]; then
      echo "[$(ts)] [SKIP] GT_ZH not set; skipping: $pred_json" | tee -a "$LOG"
      return 0
    fi
    gt="$GT_ZH"
    delta=5
    pos_unit="char"
  fi

  local save_json="${pred_json%.json}.metrics.json"
  if [[ -f "$save_json" && "$FORCE" != "1" ]]; then
    echo "[$(ts)] [SKIP] exists: $save_json" | tee -a "$LOG"
    return 0
  fi

  # Resolve audio dir (optional)
  local audio_dir
  audio_dir="$(resolve_audio_dir "$sys" "$lang")"
  local audio_arg=""
  if [[ -n "$audio_dir" ]]; then
    audio_arg="--audio_dir $audio_dir"
  fi

  echo "[$(ts)] [RUN] sys=$sys lang=$lang delta=$delta pos_unit=$pos_unit" | tee -a "$LOG"
  echo "       pred_json=$pred_json" | tee -a "$LOG"
  echo "       save_json=$save_json" | tee -a "$LOG"

  python "$SCRIPT_PATH" \
    --gt_json "$gt" \
    --pred_json "$pred_json" \
    ${audio_arg} \
    --delta "$delta" \
    --pos_unit "$pos_unit" \
    --per_tag 1 \
    --save_json "$save_json" \
    >>"$LOG" 2>>"$ERR_LOG"
}

print_summary_aligned() {
  PRED_ROOT="$PRED_ROOT" SUMMARY_TSV="$SUMMARY_TSV" python - <<'PY'
import os, json, glob

PRED_ROOT   = os.environ.get("PRED_ROOT", "prediction_output")
SUMMARY_TSV = os.environ.get("SUMMARY_TSV", "summary_metrics.tsv")

PREFERRED = [
  "alpha", "ca_f1", "count_hallu_as_fp", "coverage", "delta", "pos_unit",
  "paired_size_I", "predicted_size_P", "reference_size_R",
  "tp", "fp", "fn",
  "precision_paired", "recall_paired", "f1_paired", "u_match",
  "tpd", "tpd_mean", "tpd_std", "tpd_var", "tpd_n",
  "ntd", "ntd_mean", "ntd_std", "ntd_var", "ntd_n",
]

def is_number(x):
  return isinstance(x, (int, float)) and not isinstance(x, bool)

def extract_metrics(d):
  for k in ("summary", "overall", "metrics"):
    if isinstance(d.get(k), dict):
      return {kk: vv for kk, vv in d[k].items() if is_number(vv) or isinstance(vv, (str, bool))}
  return {kk: vv for kk, vv in d.items() if is_number(vv) or isinstance(vv, (str, bool))}

def fmt(v):
  if isinstance(v, float): return f"{v:.6f}"
  if v is True: return "True"
  if v is False: return "False"
  if v is None: return ""
  return str(v)

def is_numeric_like(s):
  try: float(s); return True
  except: return False

def print_aligned(items, cols, title):
  print("\n" + title)
  if not items:
    print("(no rows)\n"); return
  table = [{c: fmt(r.get(c, "")) for c in cols} for r in items]
  widths = {c: max(len(c), max((len(row[c]) for row in table), default=0)) for c in cols}
  num_cols = {c for c in cols if (vals := [row[c] for row in table if row[c]]) and
              sum(is_numeric_like(v) for v in vals) / len(vals) >= 0.8}
  cell = lambda c, s: s.rjust(widths[c]) if c in num_cols else s.ljust(widths[c])
  print("  ".join(cell(c, c) for c in cols))
  print("  ".join("-" * widths[c] for c in cols))
  for row in table:
    print("  ".join(cell(c, row[c]) for c in cols))
  print()

files = sorted(glob.glob(os.path.join(PRED_ROOT, "*", "*", "*.metrics.json")))
rows, all_keys = [], set()
for fp in files:
  rel = os.path.relpath(fp, PRED_ROOT)
  parts = rel.split(os.sep)
  if len(parts) < 3: continue
  system, lang = parts[0], parts[1]
  if lang not in ("en", "zh"): continue
  try:
    with open(fp, "r", encoding="utf-8") as f:
      data = json.load(f)
    met = extract_metrics(data)
  except Exception as e:
    met = {"__error__": str(e)}
  row = {"lang": lang, "system": system, **met, "__file__": fp}
  rows.append(row)
  all_keys.update(row.keys())

base_cols = ["lang", "system"]
preferred = [k for k in PREFERRED if k in all_keys]
others = sorted(k for k in all_keys if k not in set(base_cols + preferred) and k != "__file__")
cols = base_cols + preferred + others

os.makedirs(os.path.dirname(SUMMARY_TSV) or ".", exist_ok=True)
with open(SUMMARY_TSV, "w", encoding="utf-8") as f:
  f.write("\t".join(cols) + "\n")
  for r in rows:
    f.write("\t".join(fmt(r.get(c, "")) for c in cols) + "\n")

for lang in ("en", "zh"):
  items = sorted([r for r in rows if r.get("lang") == lang], key=lambda x: x.get("system", ""))
  view_cols = ["system"] + [c for c in cols if c not in ("lang", "system")]
  print_aligned(items, view_cols, f"{lang.upper()} (System × Metrics)")

print(f"Saved summary TSV: {SUMMARY_TSV}")
PY
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "FATAL: eval_metrics.py not found: $SCRIPT_PATH" >&2; exit 2
fi
if [[ ! -d "$PRED_ROOT" ]]; then
  echo "FATAL: PRED_ROOT not found: $PRED_ROOT" >&2; exit 2
fi

mapfile -t files < <(find "$PRED_ROOT" -type f -name "*.json" \
  ! -name "*.raw.json" ! -name "*.skipped.json" ! -name "*.metrics.json" | sort)

echo "[$(ts)] Found ${#files[@]} prediction JSON files under $PRED_ROOT" | tee -a "$LOG"

n_ok=0; n_fail=0
for f in "${files[@]}"; do
  if run_one "$f"; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
done

echo "[$(ts)] DONE. ok=$n_ok fail=$n_fail" | tee -a "$LOG"
print_summary_aligned
