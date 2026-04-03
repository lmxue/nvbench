# CLAP Score Evaluation

This toolkit computes the **CLAP (Contrastive Language-Audio Pretraining) score** between synthesized audio and natural language descriptions, measuring audio-text semantic alignment for speech generation systems.

The CLAP score uses the [`msclap`](https://github.com/microsoft/CLAP) library (CLAP 2023 model). A higher score indicates better alignment between the generated audio and the text description.

---

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended.

---

## Quick Start

**Step 1 — Configure `run_eval.sh`:**

```bash
AUDIOS_DIR="/path/to/your/audio/outputs"   # directory with .wav/.mp3 files
TEXTS_JSON="/path/to/samples_en.json"       # JSON with reference texts
CLAP_TEXT_KEY="caption_with_nvb"            # text field to use as CLAP query
OUT_CSV="clap_results.csv"                  # output file
GPU_IDS="0"                                 # GPU(s) to use
```

**Step 2 — Run:**

```bash
bash run_eval.sh
```

Results are saved to `clap_results.csv`.

---

## Input Format

### `--audios-dir`

A flat directory of `.wav` or `.mp3` files. Audio files must be named with the sample ID as a prefix (e.g., `en_001.wav`, `en_001_v2.wav`).

### `--clap-texts-json`

A JSON array where each object has an `id` field and a text field. For example:

```json
[
  {
    "id": "en_001",
    "text": "Hello world.",
    "caption_with_nvb": "A calm male voice says hello, followed by a soft laugh."
  }
]
```

The `id` is matched to audio filenames. Supported `--clap-text-key` values:
- `caption_with_nvb` — natural language caption (recommended for prompt-based tasks)
- `text` — plain transcript
- `text_with_mark` — text with NVC tags

---

## Output Format

`clap_results.csv` — CSV with columns:

```
audio_path, text, clap_pair_score, utmosv2
...
AVERAGE, "", <mean_clap>, <mean_utmos>
```

The final `AVERAGE` row reports the mean CLAP score across all samples. The `utmosv2` column will be empty if `--run-utmos` is not passed.

---

## Command-Line Reference

```bash
python3 eval_clap.py \
    --audios-dir /path/to/audios \
    --clap-texts-json /path/to/samples_en.json \
    --clap-text-key caption_with_nvb \
    --out clap_results.csv \
    --device cuda \
    --gpu-ids 0 \
    --run-clap \
    --checkpoint-every 50
```

Key flags:

| Flag | Description |
|---|---|
| `--run-clap` | Compute CLAP scores |
| `--run-utmos` | Also compute UTMOSv2 MOS scores (requires `utmosv2` package) |
| `--clap-text-key` | Which JSON field to use as the text query (`text`, `text_with_mark`, `caption_with_nvb`) |
| `--device` | `cuda` or `cpu` |
| `--gpu-ids` | Comma-separated GPU IDs, e.g. `0,1` |
| `--checkpoint-every N` | Write intermediate results every N samples (safe resume on interruption) |
| `--overwrite` | Recompute all scores, ignoring any existing output file |
| `--auto-detect-id-suffixes` | Auto-detect filename suffixes to strip before ID matching |
