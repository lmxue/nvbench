# WER / CER Evaluation

This toolkit computes **Word Error Rate (WER)** for English and **Character Error Rate (CER)** for Chinese using automatic speech recognition (ASR) models, enabling objective intelligibility evaluation of synthesized speech.

| Language | ASR Model | Metric |
|---|---|---|
| English (`en`) | `openai/whisper-large-v3` | WER |
| Chinese (`zh`) | `paraformer-zh` (FunASR) | CER |

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is required for efficient inference.

---

## Quick Start

```bash
bash eval_wer.sh <text_json> <audios_dir> <lang> [num_gpus]
```

**Arguments:**

| Argument | Description |
|---|---|
| `text_json` | Path to JSON file with reference texts (e.g., `samples_en.json`) |
| `audios_dir` | Directory containing generated `.wav` / `.mp3` audio files |
| `lang` | `en` for English, `zh` for Chinese |
| `num_gpus` | Number of GPUs for parallel ASR (default: `1`) |

**Example:**

```bash
bash eval_wer.sh /path/to/samples_en.json /path/to/system_outputs en 4
```

This produces `/path/to/system_outputs/wav_res_ref_text.wer` containing per-file WER and the overall mean WER.

---

## Input Format

### `text_json`

A JSON array where each object has at minimum:

```json
[
  { "id": "en_001", "text": "Hello world." },
  { "id": "en_002", "text": "Good morning." }
]
```

The `id` field is used to match audio files in `audios_dir`. Audio files must be named with the sample ID as a prefix (e.g., `en_001.wav`, `en_001_v1.wav`).

### `audios_dir`

A flat directory of `.wav` or `.mp3` files. The script auto-detects common filename suffixes (e.g., `_gemini`, `_v1`) to strip before matching against JSON `id` fields.

---

## Output Format

`wav_res_ref_text.wer` — tab-separated file with:

```
utt    wav_res    res_wer    text_ref    text_res    res_wer_ins    res_wer_del    res_wer_sub
...
WER: XX.XXX%
```

The final line reports the mean WER (%) across all samples.

---

## Individual Scripts

| Script | Description |
|---|---|
| `get_wavs_text_from_json.py` | Matches audio files to reference texts from JSON; outputs `audio_path\|text` lines |
| `prepare_ckpt.py` | Pre-downloads Whisper and paraformer-zh checkpoints |
| `run_wer.py` | Runs ASR on a list of audio files and writes per-file WER |
| `average_wer.py` | Aggregates per-file WER results and reports mean WER |

You can run each step independently. For example, to prepare checkpoints ahead of time:

```bash
python3 prepare_ckpt.py
```

To run ASR on a single GPU without the shell wrapper:

```bash
# Step 1: build pairing file
python3 get_wavs_text_from_json.py \
    --text-json samples_en.json \
    --audios-dir /path/to/audios \
    --out-file /tmp/wav_text.lst \
    --auto-detect-id-suffixes

# Step 2: run ASR + WER
python3 run_wer.py /tmp/wav_text.lst /tmp/results.wer en

# Step 3: summarize
python3 average_wer.py /tmp/results.wer /tmp/summary.wer
```
