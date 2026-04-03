#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Coverage / CA-F1 / TPD / NTD for "verify + sentence-position" NV predictions.

Designed to consume outputs from:
- predict_nvb_tag_pos_gt_verify_fast.py (dict keyed by abs audio path)
- predict_nvb_tag_pos_gt_verify.py / similar variants (list or dict)

Metrics (as in your paper snippet):
- Instance Coverage (↑): |R ∩ P| / |R|
- Coverage-Adjusted F1 (CA-F1, ↑): HM(F1_paired, Coverage)
- Tag Position Distance (TPD, ↓): mean |p_i - g_i| over matched pairs
- Normalized Tag Distance (NTD, ↓): mean_u (TPD_u / L_u) over utterances with ≥1 match

This version additionally reports **mean + variance (and std)**:
- For TPD/NTD *within a run*: mean/std/var over matched pairs (TPD) and matched utterances (NTD).
- If you pass multiple --pred_json files (e.g., 3 synthesis runs), it also reports
  **run-level mean/std/var across runs** for key metrics.

Notes:
- This script assumes *point* localization (insertion point index in word/char units), so matching
  uses a token "collar" threshold δ (integer units). IoU is for spans, not points.
- If your dataset has exactly one target tag per utterance (common in taxonomy sets), then each
  utterance contributes at most one reference event and one predicted event.
- "Hallucinated events" (extra NV tags) are counted as FP by default (toggle with --count_hallu).

Example:
  python eval_metrics_from_pos_json.py \
    --gt_json nvb_taxomomy_en_with_caption_clean.json \
    --pred_json run1.json run2.json run3.json \
    --audio_dir /path/to/audio_dir \
    --delta 2 \
    --pos_unit word \
    --per_tag 1 \
    --save_json out/metrics.json
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

ID_RE = re.compile(r"\b(?:en|zh)_[0-9]+\b", re.IGNORECASE)
ANGLE_TAG_RE = re.compile(r"<[^>]+>")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_id(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = ID_RE.search(s)
    return m.group(0) if m else None


def normalize_strip_tags(s: str) -> str:
    """Remove <tag> tokens and normalize whitespace."""
    if not isinstance(s, str):
        return ""
    s2 = ANGLE_TAG_RE.sub("", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def tokenize_words(s: str) -> List[str]:
    # conservative tokenization for EN; keep simple and stable
    s = re.sub(r"\s+", " ", s.strip())
    if not s:
        return []
    return s.split(" ")


def count_units(s: str, unit: str) -> int:
    if unit == "char":
        return len(s)
    # word
    return len(tokenize_words(s))


def compute_insert_position(ref_text: str, text_with_mark: str, tag_token: str, unit: str) -> Optional[Dict[str, Any]]:
    """Compute insertion-point index for the FIRST occurrence of tag_token.

    index = number of units BEFORE the FIRST occurrence of tag_token in text_with_mark.
    tag_token should be like "<ah>".
    """
    if not (isinstance(ref_text, str) and isinstance(text_with_mark, str) and isinstance(tag_token, str)):
        return None
    pos = text_with_mark.find(tag_token)
    if pos < 0:
        return None
    prefix = text_with_mark[:pos]
    idx = count_units(prefix, unit)
    L = count_units(ref_text, unit)
    norm = (idx / L) if L > 0 else None
    return {"index": idx, "n_units": L, "norm": norm}


def infer_gt_tag(rec: Dict[str, Any]) -> Optional[str]:
    nve = rec.get("non_verbal_events")
    if isinstance(nve, list) and nve:
        return str(nve[0]).strip()
    if isinstance(nve, str) and nve.strip():
        return nve.strip()
    # fallback: parse <tag> from text_with_mark
    twm = rec.get("text_with_mark", "")
    m = re.search(r"<([^>]+)>", twm)
    return m.group(1).strip() if m else None


def choose_unit(pos_unit: str, gt_id: str) -> str:
    if pos_unit in ("word", "char"):
        return pos_unit
    # auto by language id prefix
    if isinstance(gt_id, str) and gt_id.lower().startswith("zh_"):
        return "char"
    return "word"


def load_predictions(pred_json_path: str) -> List[Dict[str, Any]]:
    """Normalize predictions to a list of per-item records."""
    obj = load_json(pred_json_path)
    records: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for r in obj:
            if isinstance(r, dict):
                records.append(r)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                r = dict(v)
                r.setdefault("_src_key", k)
                records.append(r)
    else:
        raise ValueError(f"Unsupported pred json type: {type(obj)}")
    return records


def gather_audio_ids(audio_dir: str) -> Set[str]:
    ids: Set[str] = set()
    for root, _, files in os.walk(audio_dir):
        for fn in files:
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
                sid = extract_id(fn) or extract_id(os.path.join(root, fn))
                if sid:
                    ids.add(sid)
    return ids


def hm_alpha(x: float, y: float, alpha: float) -> float:
    # HM_α(x,y) = ( α/x + (1-α)/y )^{-1}
    if x <= 0 or y <= 0:
        return 0.0
    alpha = max(0.0, min(1.0, alpha))
    return 1.0 / (alpha / x + (1.0 - alpha) / y)


def safe_div(a: float, b: float) -> float:
    return (a / b) if b != 0 else 0.0


def mean_var(vals: List[float]) -> Dict[str, Optional[float]]:
    """Return mean/var/std for a list. Uses sample variance (ddof=1) when n>=2."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "var": None, "std": None}
    mu = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": mu, "var": 0.0, "std": 0.0}
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    std = var ** 0.5
    return {"n": n, "mean": mu, "var": var, "std": std}


def fmt_mvs(mvs: Dict[str, Optional[float]], *, digits: int = 6) -> str:
    mu, sd, var, n = mvs.get("mean"), mvs.get("std"), mvs.get("var"), mvs.get("n")
    if mu is None:
        return "NA"
    if sd is None or var is None:
        return f"{mu:.{digits}f}"
    return f"{mu:.{digits}f} ± {sd:.{digits}f} (var={var:.{digits}f}, n={n})"


def compute_metrics_for_predictions(
    gt_by_id: Dict[str, Dict[str, Any]],
    R: Set[str],
    pred_recs: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Compute metrics for a single prediction run."""

    # Build P: predicted ids (successfully produced a record with id)
    pred_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pred_recs:
        sid = r.get("id") or extract_id(r.get("_src_key", "")) or extract_id(r.get("audio_file", ""))
        if not sid:
            continue
        pred_by_id[sid].append(r)

    P = set(pred_by_id.keys())
    I = R.intersection(P)
    coverage = safe_div(len(I), len(R))

    def pick_one(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return recs[-1]

    # Per-tag reference counts on R and covered counts on (R ∩ P)
    ref_tag_counts: Dict[str, int] = defaultdict(int)
    cov_tag_counts: Dict[str, int] = defaultdict(int)
    if args.per_tag:
        for sid in R:
            gt_rec = gt_by_id.get(sid)
            if not gt_rec:
                continue
            gt_tag = infer_gt_tag(gt_rec)
            if not gt_tag:
                continue
            tok = f"<{gt_tag}>"
            ref_tag_counts[tok] += 1
            if sid in P:
                cov_tag_counts[tok] += 1

    # Accumulators (paired subset I)
    TP = FP = FN = 0
    match_distances: List[float] = []  # for TPD distribution
    per_u_ntd: List[float] = []        # for NTD distribution (per matched utterance)
    u_match = 0

    tag_stats = defaultdict(lambda: {
        "R": 0, "I": 0,
        "TP": 0, "FP": 0, "FN": 0,
        "match_d": [],
        "ntd_u": [],
        "u_match": 0,
    })
    if args.per_tag:
        for tok, n in ref_tag_counts.items():
            tag_stats[tok]["R"] = n
            tag_stats[tok]["I"] = cov_tag_counts.get(tok, 0)

    for sid in sorted(I):
        gt_rec = gt_by_id[sid]
        pred_rec = pick_one(pred_by_id[sid])

        unit = choose_unit(args.pos_unit, sid)

        gt_tag = infer_gt_tag(gt_rec)
        if not gt_tag:
            continue
        gt_tag_token = f"<{gt_tag}>"

        # recompute GT position from GT
        ref_text = gt_rec.get("text", "")
        gt_twm = gt_rec.get("text_with_mark", ref_text)
        gt_pos = compute_insert_position(ref_text, gt_twm, gt_tag_token, unit)

        present = bool(pred_rec.get("present", False))
        pred_twm = pred_rec.get("text_with_mark", ref_text)

        pred_pos = pred_rec.get("pred_pos")
        if not (isinstance(pred_pos, dict) and "index" in pred_pos):
            pred_target_tag = pred_rec.get("target_tag")
            if isinstance(pred_target_tag, str) and pred_target_tag.startswith("<") and pred_target_tag.endswith(">"):
                token = pred_target_tag
            else:
                token = gt_tag_token
            pred_pos = compute_insert_position(ref_text, pred_twm, token, unit) if present else None

        hallu_n = 0
        if args.count_hallu:
            hallu = pred_rec.get("hallucinated_events", [])
            if isinstance(hallu, list):
                hallu_n = len(hallu)

        ref_events = 1 if gt_pos is not None else 0
        pred_events = 1 if present else 0

        matched = False
        d: Optional[float] = None
        if ref_events and pred_events and gt_pos is not None and pred_pos is not None:
            d = float(abs(int(pred_pos["index"]) - int(gt_pos["index"])))
            if args.delta >= 0:
                matched = (d <= args.delta)
            else:
                matched = True

        tp = 1 if matched else 0
        fp = (pred_events - tp) + hallu_n
        fn = (ref_events - tp)

        TP += tp
        FP += fp
        FN += fn

        if matched and d is not None:
            match_distances.append(float(d))
            L_u = int(gt_pos["n_units"]) if gt_pos and gt_pos.get("n_units") else 0
            if not L_u and pred_pos and pred_pos.get("n_units"):
                L_u = int(pred_pos.get("n_units", 0))
            if L_u > 0:
                per_u_ntd.append(float(d) / float(L_u))
                u_match += 1

        if args.per_tag:
            ts = tag_stats[gt_tag_token]
            ts["TP"] += tp
            ts["FP"] += fp
            ts["FN"] += fn
            if matched and d is not None:
                ts["match_d"].append(float(d))
                L_u = int(gt_pos["n_units"]) if gt_pos and gt_pos.get("n_units") else 0
                if not L_u and pred_pos and pred_pos.get("n_units"):
                    L_u = int(pred_pos.get("n_units", 0))
                if L_u > 0:
                    ts["ntd_u"].append(float(d) / float(L_u))
                    ts["u_match"] += 1

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1_paired = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    ca_f1 = hm_alpha(f1_paired, coverage, args.alpha)

    tpd_mvs = mean_var(match_distances)
    ntd_mvs = mean_var(per_u_ntd)

    result = {
        "reference_size_R": len(R),
        "predicted_size_P": len(P),
        "paired_size_I": len(I),
        "coverage": coverage,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision_paired": precision,
        "recall_paired": recall,
        "f1_paired": f1_paired,
        "ca_f1": ca_f1,
        "alpha": args.alpha,
        "delta": args.delta,
        "pos_unit": args.pos_unit,
        "count_hallu_as_fp": bool(args.count_hallu),
        # within-run distributions
        "tpd": tpd_mvs["mean"],
        "tpd_mean": tpd_mvs["mean"],
        "tpd_std": tpd_mvs["std"],
        "tpd_var": tpd_mvs["var"],
        "tpd_n": tpd_mvs["n"],
        "ntd": ntd_mvs["mean"],
        "ntd_mean": ntd_mvs["mean"],
        "ntd_std": ntd_mvs["std"],
        "ntd_var": ntd_mvs["var"],
        "ntd_n": ntd_mvs["n"],
        "u_match": u_match,
    }

    return result, tag_stats


def main():
    ap = argparse.ArgumentParser(description="Compute Coverage / CA-F1 / TPD / NTD from position-based verify outputs.")
    ap.add_argument("--gt_json", required=True, help="GT json (e.g., nvb_taxomomy_en_with_caption_clean.json)")
    ap.add_argument("--pred_json", required=True, nargs="+", help="Prediction json file(s). Pass multiple for multiple runs.")
    ap.add_argument("--audio_dir", default="", help="Optional: audio dir to define reference set R as 'GT ids that exist in audio_dir'.")
    ap.add_argument("--delta", type=int, default=2, help="Token collar δ for a match (abs(pred_idx-gt_idx)<=δ). Use -1 to disable collar (match by presence only).")
    ap.add_argument("--pos_unit", choices=["auto", "word", "char"], default="auto", help="Unit for positions. Should match how pred_pos/gt_pos were computed.")
    ap.add_argument("--count_hallu", type=int, default=1, help="Count hallucinated_events as FP (1) or ignore (0).")
    ap.add_argument("--alpha", type=float, default=0.5, help="Weighted HM alpha for CA-F1 (0.5 = standard harmonic mean).")
    ap.add_argument("--per_tag", type=int, default=0, help="Also print per-tag coverage/F1/TPD/NTD summary.")
    ap.add_argument("--save_json", default="", help="Optional: save metrics json to this path.")
    args = ap.parse_args()

    gt_list = load_json(args.gt_json)
    if not isinstance(gt_list, list):
        raise ValueError("GT json must be a list of records.")

    gt_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in gt_list:
        if not isinstance(rec, dict):
            continue
        sid = rec.get("id")
        if isinstance(sid, str) and sid:
            gt_by_id[sid] = rec

    # Define reference set R
    if args.audio_dir:
        audio_ids = gather_audio_ids(args.audio_dir)
        R = set([sid for sid in audio_ids if sid in gt_by_id])
    else:
        R = set(gt_by_id.keys())

    if not R:
        raise RuntimeError("Reference set R is empty. Check --audio_dir or --gt_json ids.")

    per_run_results: List[Dict[str, Any]] = []
    per_run_tag_stats: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []

    for ppath in args.pred_json:
        recs = load_predictions(ppath)
        run_result, tag_stats = compute_metrics_for_predictions(gt_by_id, R, recs, args)
        run_result["pred_json"] = ppath
        per_run_results.append(run_result)
        if args.per_tag:
            per_run_tag_stats.append((ppath, tag_stats))

        print("=" * 80)
        print(f"EVAL SUMMARY | RUN: {ppath}")
        for k in [
            "reference_size_R", "predicted_size_P", "paired_size_I", "coverage",
            "tp", "fp", "fn", "precision_paired", "recall_paired", "f1_paired", "ca_f1",
            "tpd_mean", "tpd_std", "tpd_var", "tpd_n",
            "ntd_mean", "ntd_std", "ntd_var", "ntd_n",
            "u_match",
        ]:
            print(f"{k:>20}: {run_result.get(k)}")
        print("=" * 80)

        if args.per_tag:
            print("\nPER-TAG SUMMARY (F1 on paired subset; Coverage on R; TPD/NTD mean±std/var)")
            header = ["tag", "R", "I", "TP", "FP", "FN", "P", "Rcall", "F1", "Coverage", "CA-F1", "TPD", "NTD", "u_match"]
            print("\t".join(header))
            for tag, ts in sorted(tag_stats.items(), key=lambda kv: kv[0]):
                tp, fp, fn = ts.get("TP", 0), ts.get("FP", 0), ts.get("FN", 0)
                p = safe_div(tp, tp + fp)
                r = safe_div(tp, tp + fn)
                f1 = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
                cov_k = safe_div(ts.get("I", 0), ts.get("R", 0))
                ca = hm_alpha(f1, cov_k, args.alpha)

                tpd_mvs = mean_var([float(x) for x in ts.get("match_d", [])])
                ntd_mvs = mean_var([float(x) for x in ts.get("ntd_u", [])])

                row = [
                    tag,
                    str(ts.get("R", 0)),
                    str(ts.get("I", 0)),
                    str(tp), str(fp), str(fn),
                    f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}",
                    f"{cov_k:.4f}", f"{ca:.4f}",
                    fmt_mvs(tpd_mvs, digits=4),
                    fmt_mvs(ntd_mvs, digits=6),
                    str(ts.get("u_match", 0)),
                ]
                print("\t".join(row))

    # Aggregate across runs (mean/variance over run-level results)
    if len(per_run_results) > 1:
        print("\n" + "#" * 80)
        print("AGGREGATE ACROSS RUNS (mean ± std, sample variance)")
        keys = [
            "coverage", "precision_paired", "recall_paired", "f1_paired", "ca_f1",
            "tpd_mean", "ntd_mean",
        ]
        for k in keys:
            vals_f = [float(r[k]) for r in per_run_results if isinstance(r.get(k), (int, float))]
            print(f"{k:>20}: {fmt_mvs(mean_var(vals_f), digits=6)}")
        print("#" * 80)

    if args.save_json:
        out: Dict[str, Any] = {
            "gt_json": args.gt_json,
            "audio_dir": args.audio_dir,
            "reference_size_R": len(R),
            "runs": per_run_results,
        }
        if len(per_run_results) > 1:
            out["aggregate"] = {
                k: mean_var([float(r[k]) for r in per_run_results if isinstance(r.get(k), (int, float))])
                for k in [
                    "coverage", "precision_paired", "recall_paired", "f1_paired", "ca_f1",
                    "tpd_mean", "ntd_mean",
                ]
            }

        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics json to: {args.save_json}")


if __name__ == "__main__":
    main()
