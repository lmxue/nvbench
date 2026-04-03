# DNSMOS P.835 Evaluation

This toolkit computes **DNSMOS P.835**, a non-intrusive perceptual speech quality metric that predicts human Mean Opinion Scores (MOS) without requiring a clean reference signal. It evaluates three dimensions:

| Metric | Description |
|---|---|
| **OVRL** | Overall speech quality (ITU-T P.835 scale) |
| **SIG** | Speech signal quality (naturalness, distortion) |
| **BAK** | Background noise quality (noise intrusiveness) |

DNSMOS is particularly useful for evaluating TTS and speech generation systems where reference audio is unavailable.

> This implementation uses local ONNX models and does not require internet access or API keys.

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
bash eval_dnsmos.sh /path/to/your/audio/outputs
```

This scans the directory for `.wav` and `.mp3` files, runs DNSMOS P.835, and prints mean OVRL / SIG / BAK scores.

**Output files:**
- `<audios_dir>/dnsmos.csv` — per-file scores (filename, OVRL, SIG, BAK, raw scores)
- `<audios_dir>/dnsmos_summary.log` — mean scores summary

---

## Command-Line Reference

```bash
# Standard DNSMOS P.835 (regular MOS)
python3 dnsmos_mp3.py -t /path/to/audios -o results.csv

# Personalized DNSMOS (penalizes interfering speakers)
python3 dnsmos_mp3.py -t /path/to/audios -o results.csv -p

# Print mean scores from CSV
python3 avg_mos.py results.csv
```

**Arguments for `dnsmos_mp3.py`:**

| Flag | Description |
|---|---|
| `-t` / `--testset_dir` | Directory with `.wav` / `.mp3` audio files |
| `-o` / `--csv_path` | Output CSV path; prints summary to stdout if omitted |
| `-p` / `--personalized_MOS` | Use personalized MOS model (pDNSMOS) |

---

## Model Files

The ONNX models are included in this directory:

| Path | Used for |
|---|---|
| `DNSMOS/sig_bak_ovr.onnx` | Regular DNSMOS P.835 (default) |
| `pDNSMOS/sig_bak_ovr.onnx` | Personalized DNSMOS (flag `-p`) |

> **Important:** `eval_dnsmos.sh` and `dnsmos_mp3.py` must be run from the directory containing these model folders.

---

## Citation

If you use DNSMOS P.835 in your research, please cite:

```bibtex
@inproceedings{reddy2022dnsmos,
  title     = {DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors},
  author    = {Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross},
  booktitle = {ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2022},
  organization = {IEEE}
}
```
