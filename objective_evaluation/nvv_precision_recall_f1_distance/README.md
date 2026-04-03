# NVC Detection: Objective Evaluation

This module evaluates TTS systems on their ability to synthesize **Non-Verbal Communication (NVC)** events (e.g., laughter, sighs, breath sounds) in the correct position.

Metrics computed:
- **Coverage** — fraction of GT samples that received a prediction
- **Precision / Recall / F1** — NVC detection accuracy (paired subset)
- **CA-F1** — Coverage-Adjusted F1: `HM(F1, Coverage)`
- **TPD** — Tag Position Distance (mean absolute position error in tokens)
- **NTD** — Normalized Tag Distance (TPD normalized by utterance length)

---

## Pipeline Overview

```
STAGE 1: Prediction (needs Gemini API)
  run_predict.sh  →  predict_nvc.py
    For each audio file: ask Gemini whether the target NVC tag is present
    and where it is inserted. Outputs *.json per (system, lang).

STAGE 2: Evaluation (no API needed)
  run_eval.sh  →  eval_metrics.py
    Compare predictions to ground truth. Outputs *.metrics.json + TSV summary.

STAGE 3: Aggregation (optional, for multi-run studies)
  run_aggregate.sh  →  aggregate_runs.py
    Aggregate metrics across N independent runs (mean / variance / std).
```

---

## Requirements

```bash
pip install google-generativeai tqdm
```

Python 3.9+ recommended.

---

## Quick Start

### Step 1 — Prepare ground-truth JSON

The ground-truth file is a JSON array, one object per sample:

```json
[
  {
    "id": "en_1",
    "text": "Wait a second... of course.",
    "text_with_mark": "Wait a second... <ah> of course.",
    "non_verbal_events": ["ah"]
  }
]
```

| Field | Description |
|---|---|
| `id` | Unique sample ID matching the audio filename (e.g. `en_1.wav` → `en_1`) |
| `text` | Plain reference text (no NVC tags) |
| `text_with_mark` | Reference text with one `<tag>` inserted at the correct position |
| `non_verbal_events` | List of NVC event types present in this sample |

### Step 2 — Configure and run prediction

Edit `run_predict.sh`, set the required variables, then:

```bash
export GEMINI_API_KEY="your_api_key_here"
export GT_EN="/path/to/ground_truth_en.json"

# Edit AUDIO_DIRS in run_predict.sh to point to your TTS output directories.
# Each directory should contain audio files named en_<id>.wav or zh_<id>.wav.

bash run_predict.sh
```

Prediction outputs are written to `./prediction_output/<system>/<lang>/`.

**Resume:** If interrupted, re-running the script resumes from where it left off (completed items are skipped).

### Step 3 — Compute metrics

```bash
export GT_EN="/path/to/ground_truth_en.json"
bash run_eval.sh
```

This scans `./prediction_output/` for `*.json` files and writes:
- `*.metrics.json` next to each prediction file
- `summary_metrics.tsv` — all systems × metrics in one table

### Step 4 — Aggregate across runs (optional)

If you ran prediction multiple times (for variance estimation):

```bash
RUN_ROOTS="prediction_output prediction_output-2 prediction_output-3" \
  bash run_aggregate.sh
```

Output: `summary_metrics_3runs_mean_var.tsv`

---

## Audio Directory Structure

`run_predict.sh` infers the language and system name from the audio directory path. The expected convention is:

```
<root>/<system_name>_nve_samples/version1/nvb_taxonomy_{en,zh}/
    en_1.wav
    en_2.wav
    ...
```

The system name is extracted from the parent directory name (e.g. `ChatTTS_nve_samples` → `ChatTTS`).

You can use any flat directory containing `en_*.wav` / `zh_*.wav` files — just set `--pos_unit word` for English and `--pos_unit char` for Chinese in the prediction script.

---

## Configuration Reference

### `run_predict.sh`

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GT_EN` | *(required)* | Path to English ground-truth JSON |
| `GT_ZH` | *(optional)* | Path to Chinese ground-truth JSON |
| `MODEL_NAME` | `models/gemini-2.5-pro` | Gemini model |
| `MAX_WORKERS` | `16` | Parallel processing threads |
| `UPLOAD_CONCURRENCY` | `4` | Max concurrent file uploads (reduce if SSL errors) |
| `INPUT_MODE` | `upload` | `upload` or `bytes` |
| `FALLBACK_TO_BYTES` | `1` | Fall back to bytes if upload fails |
| `FAIL_FAST` | `0` | `0` = continue on errors; `1` = stop immediately |
| `OUT_ROOT` | `./prediction_output` | Output root directory |
| `AUTO_EVAL` | `0` | `1` = run eval automatically after each prediction |
| `DELTA_EN` | `2` | Word-level collar for EN (match if distance ≤ delta) |
| `DELTA_ZH` | `5` | Char-level collar for ZH |

### `run_eval.sh`

| Variable | Default | Description |
|---|---|---|
| `GT_EN` | *(required)* | Path to English ground-truth JSON |
| `GT_ZH` | *(optional)* | Path to Chinese ground-truth JSON |
| `PRED_ROOT` | `./prediction_output` | Root of prediction output directories |
| `BM_ROOT` | *(empty)* | Root of TTS audio dirs (for audio existence check; optional) |
| `FORCE` | `0` | `1` = recompute even if `.metrics.json` exists |
| `SUMMARY_TSV` | `summary_metrics.tsv` | Output TSV path |

---

## Metrics Reference

| Metric | Direction | Formula | Notes |
|---|---|---|---|
| **Coverage** | ↑ | `|R ∩ P| / |R|` | Fraction of GT samples predicted |
| **Precision** | ↑ | `TP / (TP + FP)` | On paired subset R ∩ P |
| **Recall** | ↑ | `TP / (TP + FN)` | On paired subset R ∩ P |
| **F1** | ↑ | `HM(Precision, Recall)` | Harmonic mean |
| **CA-F1** | ↑ | `HM(F1, Coverage)` | Coverage-Adjusted F1 |
| **TPD** | ↓ | `mean |p_i - g_i|` | Mean absolute position error (tokens) |
| **NTD** | ↓ | `mean (|p_i - g_i| / L_i)` | TPD normalized by utterance length |

**Token collar (δ):** A prediction counts as a match (TP) if it says the NVC is present **and** the position is within δ tokens of the GT position. Default: δ=2 words (EN), δ=5 chars (ZH).

**Hallucination:** Extra NVC tags generated by the system (not in GT) are counted as FP by default (`--count_hallu 1`).

---

## Output Files

### Prediction output (`*.json`)

```json
{
  "/abs/path/to/audio.wav": {
    "id": "en_1",
    "text": "Wait a second... of course.",
    "text_with_mark": "Wait a second... <ah> of course.",
    "target_tag": "<ah>",
    "present": true,
    "pos_unit": "word",
    "pred_pos": { "index": 3, "n_units": 5, "norm": 0.6 },
    "gt_pos":   { "index": 3, "n_units": 5, "norm": 0.6 },
    "hallucinated": false,
    "hallucinated_events": []
  }
}
```

### Metrics output (`*.metrics.json`)

```json
{
  "reference_size_R": 100,
  "predicted_size_P": 98,
  "paired_size_I": 98,
  "coverage": 0.98,
  "tp": 82, "fp": 16, "fn": 18,
  "precision_paired": 0.8367,
  "recall_paired": 0.8200,
  "f1_paired": 0.8283,
  "ca_f1":  0.9027,
  "tpd": 0.09, "tpd_mean": 0.09, "tpd_std": 0.21,
  "ntd": 0.01, "ntd_mean": 0.01, "ntd_std": 0.02
}
```

---

## Citation

If you use this evaluation toolkit in your research, please cite:

```bibtex
@inproceedings{nvbench2026,
  title     = {NVBench: A Benchmark for Non-Verbal Communication in Speech Synthesis},
  booktitle = {Interspeech},
  year      = {2026}
}
```
