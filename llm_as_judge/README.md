# LLM-as-Judge for Speech Generation Evaluation

This toolkit uses a large language model (Google Gemini) as an automatic judge to evaluate speech synthesis systems on multiple perceptual metrics, focusing on **Non-Verbal Communication (NVC)** in synthesized speech. It is designed to closely simulate a real Mean Opinion Score (MOS) listening test.

---

## Overview

The evaluator operates in two modes:

| Mode | Task type | Metrics evaluated |
|---|---|---|
| **prompt-based** | Caption/description → audio | Overall Naturalness, Overall Quality, Caption–Audio Match (CAM), NVC Instruction Following (IF), NVC Perceptual Effect (PE) |
| **tag-based** | Text with NVC tags → audio | Overall Naturalness, Overall Quality, Overall Expression, NVC Accuracy, NVC Perceptual Effect (PE) |

**Key features:**
- **GROUP_COMPARE mode**: all systems are compared side-by-side on the same sample using anonymized labels (A, B, C, …), just like a real listening test. This greatly stabilizes ranking.
- **Multi-rater simulation**: multiple virtual raters with different calibration profiles score each sample.
- **Hard-cap post-processing**: consistency rules prevent contradictory outputs (e.g., Quality=5 when an artifact is mentioned in the reason).
- **Structured output via JSON Schema**: Gemini's `response_schema` parameter enforces all required metric fields.
- **Resumable evaluation**: results are flushed to disk after each sample; interrupted runs can be resumed safely.

---

## Requirements

```bash
pip install google-generativeai tqdm
```

Python 3.9+ is recommended.

---

## Quick Start

**Step 1 — Prepare your data**

Create a data directory with the following files (see [Data Format](#data-format) below):

```
your_data/
├── sampled_en.json         # source texts for English tasks
├── sampled_zh.json         # source texts for Chinese tasks (optional)
├── manifest_en-prompt.json # audio paths per system for en-prompt task
├── manifest_en-tag.json    # audio paths per system for en-tag task
├── manifest_zh-prompt.json # audio paths per system for zh-prompt task (optional)
└── manifest_zh-tag.json    # audio paths per system for zh-tag task (optional)
```

See `example_data/` for the exact JSON schema.

**Step 2 — Configure `run_eval.sh`**

```bash
# Required
export GEMINI_API_KEY="your_api_key_here"
export DATA_DIR="/path/to/your_data"

# Select tasks to run (space-separated subset of: en-prompt en-tag zh-prompt zh-tag)
export TASKS="en-prompt"
```

**Step 3 — Run**

```bash
bash run_eval.sh
```

Results are saved to `eval_results_{task}__GROUPCOMPARE.json` in the current directory.

To print a summary table of mean scores per system:

```bash
bash run_statistics.sh
```

---

## Data Format

### `samples_en.json` / `samples_zh.json`

A JSON array of source sample objects. Each object must have:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique sample identifier |
| `text` | string | Plain speech text |
| `text_with_mark` | string | Text with NVC tags, e.g. `<ah>`, `<laugh>` (for tag-based tasks) |
| `caption_with_nvb` | string | Natural language caption describing voice, emotion, and NVC (for prompt-based tasks) |
| `non_verbal_events` | list[str] | NVC event types present in this sample |

See `example_data/sampled_en.json` for a minimal example.

### `manifest_{lang}-{mode}.json`

Describes which audio files exist for each system, organized by NVC tag and sample ID.

Top-level structure:

```json
{
  "summary": {
    "test_id": "en-prompt",
    "lang": "en",
    "mode": "prompt",
    "systems": [
      { "key": "system_a", "name": "System A", "folder": "system_a_outputs" }
    ]
  },
  "by_tag": {
    "<nvc_tag>": {
      "sample_ids": ["en_1", "en_2"],
      "systems": {
        "system_a": {
          "name": "System A",
          "supported": true,
          "paths": {
            "en_1": "/absolute/path/to/system_a/en_1.wav",
            "en_2": "/absolute/path/to/system_a/en_2.wav"
          }
        }
      }
    }
  }
}
```

- `paths` values must be absolute paths to `.wav` (or other audio) files that Gemini's File API can process.
- If a system did not generate a sample, set its path to `null` or omit it; the script skips missing entries.
- At least **2 systems** are required per sample when running in GROUP_COMPARE mode.

See `example_data/manifest_en-prompt.json` for a minimal example.

---

## Configuration

All parameters are set via environment variables (in `run_eval.sh` or exported before calling `python llm_eval.py`).

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL` | `models/gemini-2.5-pro` | Gemini model to use |
| `DATA_DIR` | `./example_data` | Directory containing `samples_*.json` and `manifest_*.json` |
| `TASKS` | *(all)* | Space-separated task list, e.g. `"en-prompt en-tag"` |
| `GROUP_COMPARE` | `1` | `1` = compare all systems together (recommended); `0` = evaluate each system separately |
| `GROUP_SIZE` | `0` | Max systems per comparison set; `0` = all at once |
| `ANCHOR_SYSTEM` | *(empty)* | System key always included in every set when splitting; leave empty to disable |
| `N_RATERS` | `4` | Number of simulated raters |
| `MIN_RATERS_PER_SAMPLE` | `1` | Minimum raters that must score each sample |
| `RATER_COVERAGE` | `0.25` | Fraction of additional raters per sample (`1.0` = all raters score all samples) |
| `MAX_WORKERS` | `4` | Concurrent API threads (increase for faster evaluation) |
| `JUDGE_TEMPERATURE` | `0.2` | Sampling temperature for the judge model |
| `GLOBAL_SEED` | `1234` | Random seed (controls rater assignment and shuffling) |
| `USE_PROXY` | `0` | Set to `1` to enable proxy (also set `http_proxy` / `https_proxy`) |
| `ERROR_LOG_FILE` | `eval_errors.log` | File where API errors are appended |

---

## Output Format

Each evaluation produces a JSON file `eval_results_{task}__GROUPCOMPARE.json` with the structure:

```json
{
  "__meta__": { "task_name": "en-prompt", "model": "...", "n_raters": 4, ... },
  "items": {
    "<group_key>::g0::r0": {
      "status": "ok",
      "ground_truth": {
        "label_to_system": { "A": "system_a", "B": "system_b" },
        "source_text": "...",
        "tag": "ah",
        "sample_id": "en_1"
      },
      "prediction": {
        "results": {
          "A": {
            "heard_summary": "...",
            "issues": [],
            "overall_naturalness_score": 4,
            "overall_naturalness_reason": "...",
            "overall_quality_score": 4,
            "overall_quality_reason": "...",
            "cam_score": 4,
            "cam_reason": "...",
            "nvc_if_score": 3,
            "nvc_if_reason": "...",
            "nvc_pe_score": 3,
            "nvc_pe_reason": "..."
          },
          "B": { "..." }
        },
        "ranking": {
          "overall_naturalness": ["A", "B"],
          "overall_quality": ["B", "A"],
          "...": "..."
        },
        "notes": "..."
      },
      "prediction_capped": { "..." }
    }
  }
}
```

`prediction_capped` applies post-processing hard-cap rules for Quality and Naturalness consistency. Use `prediction_capped` for final analysis.

Run `bash run_statistics.sh` to compute per-system mean ± variance for all metrics across all result files in the current directory.

---

## Metrics Reference

**Prompt-based tasks** (input: natural language caption):

| Metric | Key | Scale | Description |
|---|---|---|---|
| Overall Naturalness | `overall_naturalness_score` | 1–5 | Human-likeness of speech (prosody, pronunciation, flow) |
| Overall Quality | `overall_quality_score` | 1–5 | Signal fidelity: noise, distortion, codec artifacts |
| Caption–Audio Match | `cam_score` | 1–5 | How well the audio matches the caption (voice, emotion, scene) |
| NVC Instruction Following | `nvc_if_score` | 0–5 | Whether NVC events match the instructions in the caption; 0 if no NVC |
| NVC Perceptual Effect | `nvc_pe_score` | 0–5 | Naturalness and expressiveness of the NVC events; 0 if no NVC |

**Tag-based tasks** (input: text with `<tag>` markers):

| Metric | Key | Scale | Description |
|---|---|---|---|
| Overall Naturalness | `overall_naturalness_score` | 1–5 | Human-likeness |
| Overall Quality | `overall_quality_score` | 1–5 | Signal fidelity |
| Overall Expression | `overall_expression_score` | 1–5 | Expressive effect of speech + NVC combined |
| NVC Accuracy | `nvc_accuracy_score` | 0–5 | Whether generated NVC matches the input tags; 0 if no NVC |
| NVC Perceptual Effect | `nvc_pe_score` | 0–5 | Naturalness and expressiveness of the NVC events; 0 if no NVC |

---

<!-- ## Citation

If you use this toolkit in your research, please cite:

```bibtex
@inproceedings{nvbench2026,
  title     = {NVBench: A Benchmark for Non-Verbal Communication in Speech Synthesis},
  booktitle = {Interspeech},
  year      = {2026}
}
``` -->
