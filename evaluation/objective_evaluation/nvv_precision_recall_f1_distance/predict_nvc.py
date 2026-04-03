#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_nvb_tag_pos_gt_verify_fast.py (faster + more robust upload)

Verify-based NVB tagging with Gemini using GT reference text, and position-based localization.

This version is optimized for speed AND robustness:
- Parallel processing with ThreadPoolExecutor (--max_workers / env MAX_WORKERS)
- Periodic flush to disk (flush_every / flush_secs)
- Input audio via:
    (a) Gemini file upload (default)  -> smaller request bodies
    (b) raw bytes                    -> no upload step, often more stable under proxies/SSL issues
- NEW: robust upload with:
    - upload_semaphore to limit concurrent uploads (avoid SSL EOF under high concurrency)
    - exponential-backoff retries for upload_file
    - optional fallback_to_bytes if upload keeps failing
- NEW: per-thread GenerativeModel (thread-local) to avoid potential thread-safety issues.

Outputs (compatible with previous versions):
- <run_name>.json        : minimal per-audio records (dict keyed by abs audio path)
- <run_name>.raw.json    : prompt/response audit (dict keyed by abs audio path)
- <run_name>.skipped.json: skipped items (dict keyed by abs audio path)

NOTE:
- Uses ANGLE tag format: <tag_name>.
"""

import os
import re
import json
import time
import hashlib
import traceback
import argparse
import threading
import random
import ssl
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import google.generativeai as genai

PROMPT_VERSION = "verify_pos_by_gt_v3_fast_robust_upload_2026-02-18"
RAW_TRUNC = 14000

# Proxy: leave empty by default. Set via --proxy flag or export http_proxy/https_proxy
# in your shell before running if your environment requires a proxy.
DEFAULT_PROXY = ""

# ---------- Tag patterns ----------
ANGLE_TAG_RE = re.compile(r"<\s*([a-zA-Z0-9_]+)\s*>")

# ---------- ID extraction ----------
# ID_RE = re.compile(r"\b((?:en|zh)_[0-9]+)\b", re.IGNORECASE)
ID_RE = re.compile(r"((?:en|zh)_[0-9]+)", re.IGNORECASE)

# =========================
# Utils
# =========================
def set_default_proxy(default_proxy: str = DEFAULT_PROXY) -> None:
    if not default_proxy:
        return  # no proxy configured; skip
    proxy = default_proxy
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, proxy)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")


def now_ms() -> int:
    return int(time.time() * 1000)


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def trunc(s: str, n: int = RAW_TRUNC) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...<truncated {len(s)-n} chars>"


def safe_dump_json(path: str, obj: Any) -> None:
    """Atomic write: tmp + replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def fail_fast(msg: str, exc: Optional[BaseException] = None) -> None:
    print("\n" + "=" * 80)
    print("FATAL:", msg)
    if exc is not None:
        print("Exception:", repr(exc))
        print(traceback.format_exc())
    print("=" * 80 + "\n")
    raise SystemExit(2)


def guess_mime_type(file_path: str) -> str:
    p = file_path.lower()
    if p.endswith(".wav"):
        return "audio/wav"
    if p.endswith(".mp3"):
        return "audio/mpeg"
    if p.endswith(".m4a"):
        return "audio/mp4"
    if p.endswith(".flac"):
        return "audio/flac"
    return "application/octet-stream"


def parse_json_response(resp_text: str) -> Dict[str, Any]:
    """Strict JSON parse; otherwise extract first {...} block and parse; else {}."""
    if not resp_text:
        return {}
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", resp_text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def extract_id_from_path(p: str) -> Optional[str]:
    base = os.path.basename(p)
    m = ID_RE.search(base) or ID_RE.search(p)
    if not m:
        return None
    return m.group(1).lower()


def tokenize_words(text: str) -> List[str]:
    """Word tokens for position counting: count only word-like units (ignore punctuation)."""
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)


def count_units(text: str, unit: str) -> int:
    if unit == "word":
        return len(tokenize_words(text))
    # char: count non-space characters (stable for zh/en)
    return sum(1 for ch in (text or "") if not ch.isspace())


def compute_insert_position(ref_text: str, text_with_mark: str, tag: str, unit: str) -> Optional[Dict[str, Any]]:
    """
    Compute insertion point index in ref_text units based on the FIRST occurrence of <tag> in text_with_mark.
    index = number of units BEFORE the tag token.
    """
    if not (isinstance(ref_text, str) and isinstance(text_with_mark, str) and isinstance(tag, str)):
        return None
    tag_token = f"<{tag}>"
    pos = text_with_mark.find(tag_token)
    if pos < 0:
        return None
    prefix = text_with_mark[:pos]
    idx = count_units(prefix, unit)
    L = count_units(ref_text, unit)
    norm = (idx / L) if L > 0 else None
    return {"index": idx, "n_units": L, "norm": norm}


def normalize_strip_tags(s: str) -> str:
    """Remove angle tags and normalize spaces for equality check."""
    if not isinstance(s, str):
        return ""
    s2 = ANGLE_TAG_RE.sub("", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def infer_tags_from_gt_record(rec: Dict[str, Any]) -> List[str]:
    """
    GT 'non_verbal_events' may be list[str] or str.
    Prefer that. If missing, fall back to parsing <tag> from text_with_mark.
    """
    nve = rec.get("non_verbal_events")
    tags: List[str] = []
    if isinstance(nve, list):
        tags = [t for t in nve if isinstance(t, str) and t.strip()]
    elif isinstance(nve, str) and nve.strip():
        tags = [nve.strip()]
    if tags:
        return tags
    twm = rec.get("text_with_mark", "")
    tags2 = [m.group(1) for m in ANGLE_TAG_RE.finditer(twm or "")]
    seen = set()
    out = []
    for t in tags2:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_verify_prompt(ref_text: str, target_tag: str, allowed_tags: List[str]) -> str:
    """Force Gemini to not paraphrase. We validate by stripping tags and comparing to ref_text."""
    allowed = ", ".join([f"<{t}>" for t in allowed_tags])
    return f"""
You are an expert evaluator for non-verbal vocalization (NV) in TTS audio.

You are given:
- A reference transcript TEXT (must be kept verbatim).
- A TARGET NV TAG to verify (one of the allowed tags).
- An audio sample generated by a TTS system.

Your job is NOT to guess any NV type freely.
Instead, verify whether the TARGET NV is actually synthesized in the audio.

IMPORTANT CONSTRAINTS (must follow):
1) DO NOT paraphrase, rewrite, or correct the reference TEXT.
2) If the target NV is present, indicate its sentence position by inserting the EXACT tag token <{target_tag}>
   into the reference TEXT at the correct location. This is the ONLY allowed modification.
3) Insert <{target_tag}> AT MOST ONCE.
4) If the target NV is NOT present, output text_with_mark exactly equal to the reference TEXT (no tags).
5) Additionally, check if there are extra NV events NOT requested (hallucinations). If yes, list up to 2
   hallucinated tags from the allowed list and provide a short location_hint (a few words around where it happens).
6) Only use audible evidence (do not infer from semantics).

ALLOWED TAGS (exact tokens): {allowed}

Return JSON ONLY in this schema:
{{
  "present": true/false,
  "text_with_mark": "REFERENCE TEXT with optional <{target_tag}> inserted once",
  "confidence": 0.0-1.0,
  "evidence": ["short audible reasons, 1-3 bullets"],
  "hallucinated": true/false,
  "hallucinated_events": [
    {{"tag": "<sniff>", "location_hint": "a few surrounding words"}}
  ]
}}

REFERENCE TEXT:
{ref_text}
""".strip()


def build_response_schema() -> Dict[str, Any]:
    """A permissive JSON schema that still forces valid JSON + key fields."""
    return {
        "type": "object",
        "properties": {
            "present": {"type": "boolean"},
            "text_with_mark": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "hallucinated": {"type": "boolean"},
            "hallucinated_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "location_hint": {"type": "string"},
                    },
                    "required": ["tag", "location_hint"],
                },
            },
        },
        "required": ["present", "text_with_mark", "hallucinated", "hallucinated_events"],
    }


def call_with_retry(parts, gen_config, max_retries: int, model: genai.GenerativeModel):
    """Retry on transient exceptions during generate_content."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return model.generate_content(contents=parts, generation_config=gen_config)
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 12.0)


def upload_file_with_retry(
    abs_path: str,
    mime: str,
    max_tries: int = 5,
    base_sleep: float = 0.8,
    max_sleep: float = 12.0,
) -> Any:
    """
    Robust wrapper for genai.upload_file. Common failure under high concurrency/proxy:
    ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING.
    """
    last_exc = None
    for i in range(max_tries):
        try:
            return genai.upload_file(path=abs_path, mime_type=mime)
        except Exception as e:
            last_exc = e
            # retry only if not last attempt
            if i >= max_tries - 1:
                break
            # exponential backoff + jitter
            sleep = min(max_sleep, base_sleep * (2 ** i))
            sleep = sleep * (0.65 + random.random() * 0.7)
            time.sleep(sleep)
    raise last_exc


def normalize_hallucinated_events(hallu_events: Any) -> List[Dict[str, Any]]:
    hallu_norm: List[Dict[str, Any]] = []
    if isinstance(hallu_events, list):
        for ev in hallu_events[:2]:
            if not isinstance(ev, dict):
                continue
            tag_tok = ev.get("tag", "")
            tag_name = ""
            if isinstance(tag_tok, str):
                m = ANGLE_TAG_RE.search(tag_tok)
                tag_name = (m.group(1) if m else tag_tok.strip().strip("<>").strip())
            if not tag_name:
                continue
            loc = ev.get("location_hint", "")
            hallu_norm.append({
                "tag": f"<{tag_name}>",
                "location_hint": (loc if isinstance(loc, str) else ""),
            })
    return hallu_norm


def process_one(
    abs_path: str,
    args,
    get_model_fn,
    schema: Dict[str, Any],
    gt_by_id: Dict[str, Dict[str, Any]],
    allowed_tags: List[str],
    allowed_tags_set: set,
    upload_sem: threading.Semaphore,
) -> Tuple[str, str, Any]:
    """
    Return (status, key, payload):
      - ("ok", abs_path, (record, raw_record))
      - ("skip", abs_path, skip_info_dict)
    """
    sid = extract_id_from_path(abs_path)
    if (not sid) or (sid not in gt_by_id):
        return ("skip", abs_path, {"reason": "no_gt_id_mapping", "id": sid})

    rec = gt_by_id[sid]
    ref_text = rec.get("text", "")
    gt_twm = rec.get("text_with_mark", "")
    tags = infer_tags_from_gt_record(rec)

    if not isinstance(ref_text, str) or not ref_text.strip():
        return ("skip", abs_path, {"reason": "gt_missing_text", "id": sid})
    if not tags:
        return ("skip", abs_path, {"reason": "gt_missing_tag", "id": sid})

    target = tags[0].strip()
    if target not in allowed_tags_set:
        return ("skip", abs_path, {"reason": "target_not_in_allowed", "id": sid, "target": target})

    # Decide pos unit
    if args.pos_unit == "auto":
        unit = "word" if sid.startswith("en_") else "char"
    else:
        unit = args.pos_unit

    gt_pos = compute_insert_position(ref_text, gt_twm, target, unit)

    prompt = build_verify_prompt(ref_text, target, allowed_tags)
    mime = guess_mime_type(abs_path)

    # Prepare audio part
    uploaded_file = None
    audio_part = None
    upload_error = None
    t_upload0 = now_ms()

    try:
        if args.input_mode == "upload":
            # limit concurrent uploads to avoid proxy/SSL EOF
            with upload_sem:
                try:
                    uploaded_file = upload_file_with_retry(
                        abs_path=abs_path,
                        mime=mime,
                        max_tries=args.upload_retry,
                        base_sleep=args.upload_backoff,
                        max_sleep=args.upload_backoff_max,
                    )
                    audio_part = uploaded_file
                except Exception as e:
                    upload_error = repr(e)
                    if args.fallback_to_bytes:
                        audio_bytes = open(abs_path, "rb").read()
                        audio_part = {"mime_type": mime, "data": audio_bytes}
                    else:
                        raise
        else:
            audio_bytes = open(abs_path, "rb").read()
            audio_part = {"mime_type": mime, "data": audio_bytes}
    except Exception as e:
        if args.fail_fast:
            fail_fast(f"Audio upload/prepare failed for: {abs_path}", e)
        return ("skip", abs_path, {"reason": "upload_error", "id": sid, "error": repr(e)})

    upload_ms = now_ms() - t_upload0

    gen_config = {
        "temperature": args.temperature,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }

    resp_raw = ""
    parsed: Dict[str, Any] = {}
    attempts = 0
    latency = None

    # We do our own light validation and ask for reformat if violated (cheap; usually 0-1 retries).
    prompt_for_retry = prompt
    max_retries = max(0, int(args.retry))

    model = get_model_fn()

    try:
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            t0 = now_ms()
            resp = call_with_retry(
                parts=[prompt_for_retry, audio_part],
                gen_config=gen_config,
                max_retries=args.net_retry,
                model=model,
            )
            latency = now_ms() - t0
            resp_raw = getattr(resp, "text", "") or ""
            parsed = parse_json_response(resp_raw)

            present = bool(parsed.get("present", False))
            twm = parsed.get("text_with_mark", "")
            if not isinstance(twm, str) or not twm.strip():
                twm = ref_text

            # Constraint: must preserve ref_text (after stripping tags)
            if normalize_strip_tags(twm) != normalize_strip_tags(ref_text):
                prompt_for_retry = prompt_for_retry + "\n\nFORMAT ERROR: text_with_mark must be EXACT reference TEXT except inserting the target tag token."
                continue

            tag_token = f"<{target}>"
            cnt = twm.count(tag_token)

            if present and cnt != 1:
                prompt_for_retry = prompt_for_retry + f"\n\nFORMAT ERROR: present=true requires exactly ONE occurrence of {tag_token} in text_with_mark."
                continue
            if (not present) and cnt != 0:
                prompt_for_retry = prompt_for_retry + f"\n\nFORMAT ERROR: present=false requires NO occurrence of {tag_token} in text_with_mark."
                continue

            # Accept
            break

    except Exception as e:
        if args.fail_fast:
            fail_fast(f"Gemini call failed for: {abs_path}", e)
        return ("skip", abs_path, {"reason": "gemini_error", "id": sid, "error": repr(e)})

    # Normalize outputs
    present = bool(parsed.get("present", False))
    twm = parsed.get("text_with_mark", ref_text)
    if not isinstance(twm, str) or not twm.strip():
        twm = ref_text

    hallucinated = bool(parsed.get("hallucinated", False))
    hallu_events = parsed.get("hallucinated_events", [])
    hallu_norm = normalize_hallucinated_events(hallu_events)
    if hallu_norm:
        hallucinated = True

    pred_pos = compute_insert_position(ref_text, twm, target, unit) if present else None

    record = {
        "id": sid,
        "text": ref_text,
        "text_with_mark": twm,
        "target_tag": f"<{target}>",
        "present": present,
        "pos_unit": unit,
        "pred_pos": pred_pos,
        "gt_pos": gt_pos,
        "hallucinated": hallucinated,
        "hallucinated_events": hallu_norm,
    }

    raw_record = {
        "prompt_version": PROMPT_VERSION,
        "audio_file": os.path.basename(abs_path),
        "id": sid,
        "target_tag": target,
        "pos_unit": unit,
        "model": args.model_name,
        "temperature": args.temperature,
        "attempts": attempts,
        "latency_ms": latency,
        "upload_ms": upload_ms,
        "upload_error": upload_error,
        "prompt_hash": sha1(prompt),
        **({"prompt": prompt} if args.save_prompts else {}),
        "raw_full": trunc(resp_raw, RAW_TRUNC),
        "parsed": parsed,
    }

    # Optional rate limit
    if args.min_sleep > 0:
        time.sleep(args.min_sleep)

    # cleanup uploaded file (optionally deferred)
    if uploaded_file is not None and args.delete_uploaded:
        try:
            genai.delete_file(uploaded_file.name)
        except Exception:
            pass

    return ("ok", abs_path, (record, raw_record))


def main():
    ap = argparse.ArgumentParser(description="Verify target NV tag + sentence-position localization using GT reference text (fast, robust upload).")
    ap.add_argument("audio_dir", help="Directory containing audio files")
    ap.add_argument("--gt_json", required=True, help="GT json (e.g., nvb_taxomomy_en_with_caption_clean.json)")
    ap.add_argument("--output_dir", default="./prediction_output", help="Output directory for json files")
    ap.add_argument("--model_name", default=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-pro"))
    ap.add_argument("--temperature", type=float, default=0.0, help="Gemini temperature (keep low for verification)")
    ap.add_argument("--min_sleep", type=float, default=0.0, help="Optional per-sample sleep (seconds) to reduce rate-limit errors")
    ap.add_argument("--retry", type=int, default=1, help="Format-retry rounds (prompt correction) per sample")
    ap.add_argument("--net_retry", type=int, default=2, help="Network retry count inside a single attempt")
    ap.add_argument("--pos_unit", choices=["auto", "word", "char"], default="auto")
    ap.add_argument("--save_prompts", action="store_true")
    ap.add_argument("--system_name", default="", help="Override system name for run_name. Default inferred from audio_dir path.")
    ap.add_argument("--max_files", type=int, default=0, help="For quick debug: process at most N files (0 means all).")
    ap.add_argument("--max_workers", type=int, default=int(os.environ.get("MAX_WORKERS", "16")), help="Parallel workers (env MAX_WORKERS)")
    ap.add_argument("--flush_every", type=int, default=25, help="Flush JSON to disk every N newly processed samples")
    ap.add_argument("--flush_secs", type=float, default=12.0, help="Also flush if this many seconds elapsed")

    ap.add_argument("--input_mode", choices=["upload", "bytes"], default="upload",
                    help="How to send audio to Gemini. bytes avoids upload step and is often more stable under proxies.")
    ap.add_argument("--delete_uploaded", type=int, default=1, help="1=delete uploaded files after each sample (safer for quota), 0=faster")
    ap.add_argument("--fallback_to_bytes", type=int, default=1,
                    help="If upload fails after retries, fall back to bytes mode for that sample (recommended).")

    ap.add_argument("--upload_concurrency", type=int, default=int(os.environ.get("UPLOAD_CONCURRENCY", "8")),
                    help="Max concurrent upload_file() calls (limits SSL EOF under high concurrency).")
    ap.add_argument("--upload_retry", type=int, default=int(os.environ.get("UPLOAD_RETRY", "5")),
                    help="Retries for upload_file on transient errors.")
    ap.add_argument("--upload_backoff", type=float, default=float(os.environ.get("UPLOAD_BACKOFF", "0.8")),
                    help="Base sleep seconds for upload retry backoff.")
    ap.add_argument("--upload_backoff_max", type=float, default=float(os.environ.get("UPLOAD_BACKOFF_MAX", "12.0")),
                    help="Max sleep seconds for upload retry backoff.")

    ap.add_argument("--fail_fast", type=int, default=1, help="1=exit on Gemini exception, 0=record skip and continue")
    ap.add_argument("--proxy", default=DEFAULT_PROXY, help="Proxy URL to set if env not present")

    args = ap.parse_args()
    args.delete_uploaded = bool(int(args.delete_uploaded))
    args.fallback_to_bytes = bool(int(args.fallback_to_bytes))
    args.fail_fast = bool(int(args.fail_fast))

    # Proxy & API
    set_default_proxy(args.proxy)

    api_key = (os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if not api_key:
        fail_fast("GEMINI_API_KEY is empty. Export it before running.")

    genai.configure(api_key=api_key)

    # Thread-local model (avoid sharing a model instance across threads)
    tls = threading.local()

    def get_model():
        m = getattr(tls, "model", None)
        if m is None:
            tls.model = genai.GenerativeModel(args.model_name)
            m = tls.model
        return m

    schema = build_response_schema()

    # ---- Load GT ----
    try:
        gt_list = json.load(open(args.gt_json, "r", encoding="utf-8"))
        if not isinstance(gt_list, list):
            raise ValueError("GT JSON must be a list of records.")
    except Exception as e:
        fail_fast(f"Failed to load gt_json: {args.gt_json}", e)

    gt_by_id: Dict[str, Dict[str, Any]] = {}
    allowed_tags_set = set()
    for rec in gt_list:
        if not isinstance(rec, dict):
            continue
        sid = rec.get("id")
        if isinstance(sid, str) and sid.strip():
            sid2 = sid.strip().lower()
            gt_by_id[sid2] = rec
            for t in infer_tags_from_gt_record(rec):
                allowed_tags_set.add(t)

    allowed_tags = sorted(list(allowed_tags_set))
    if not allowed_tags:
        fail_fast("No allowed tags inferred from GT. Check gt_json format.")

    # ---- Run name ----
    audio_dir = os.path.abspath(args.audio_dir)
    parts = audio_dir.rstrip("/").split("/")
    inferred = "__".join(parts[-3:]) if len(parts) >= 3 else parts[-1]
    run_name = args.system_name.strip() if args.system_name.strip() else inferred
    run_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", run_name)

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, f"{run_name}.json")
    raw_json = os.path.join(args.output_dir, f"{run_name}.raw.json")
    skipped_json = os.path.join(args.output_dir, f"{run_name}.skipped.json")

    # ---- Resume ----
    results: Dict[str, Any] = {}
    raw_db: Dict[str, Any] = {}
    skipped: Dict[str, Any] = {}

    if os.path.exists(output_json):
        try:
            results = json.load(open(output_json, "r", encoding="utf-8"))
            if not isinstance(results, dict):
                results = {}
        except Exception:
            results = {}

    if os.path.exists(raw_json):
        try:
            raw_db = json.load(open(raw_json, "r", encoding="utf-8"))
            if not isinstance(raw_db, dict):
                raw_db = {}
        except Exception:
            raw_db = {}

    if os.path.exists(skipped_json):
        try:
            skipped = json.load(open(skipped_json, "r", encoding="utf-8"))
            if not isinstance(skipped, dict):
                skipped = {}
        except Exception:
            skipped = {}

    # ---- Enumerate audio files ----
    exts = (".wav", ".mp3", ".m4a", ".flac")
    files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(exts)]
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    # Filter already done
    pending = []
    for fp in files:
        afp = os.path.abspath(fp)
        if afp in results or afp in skipped:
            continue
        pending.append(afp)

    # upload semaphore: never exceed max_workers
    upload_conc = max(1, min(int(args.upload_concurrency), int(args.max_workers)))
    upload_sem = threading.Semaphore(upload_conc)

    print(f"GT records: {len(gt_by_id)} | Allowed tags inferred: {len(allowed_tags)}")
    print(f"Audio files total: {len(files)} | Pending: {len(pending)}")
    print(f"Model: {args.model_name} | input_mode={args.input_mode} | max_workers={args.max_workers}")
    print(f"Upload concurrency: {upload_conc} | upload_retry={args.upload_retry} | fallback_to_bytes={args.fallback_to_bytes}")
    print(f"Output: {output_json}")

    # ---- Parallel run ----
    lock = threading.Lock()
    last_flush_t = time.time()
    new_done = 0

    def maybe_flush(force: bool = False):
        nonlocal last_flush_t, new_done
        t = time.time()
        if (not force) and (new_done < args.flush_every) and ((t - last_flush_t) < args.flush_secs):
            return
        with lock:
            safe_dump_json(output_json, results)
            safe_dump_json(raw_json, raw_db)
            safe_dump_json(skipped_json, skipped)
            new_done = 0
            last_flush_t = t

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = []
        for afp in pending:
            futures.append(ex.submit(
                process_one,
                afp, args, get_model, schema,
                gt_by_id, allowed_tags, allowed_tags_set, upload_sem
            ))

        for fu in tqdm(as_completed(futures), total=len(futures), desc="verify-pos-fast"):
            try:
                status, key, payload = fu.result()
            except Exception as e:
                # Catch unexpected worker crash so the whole run doesn't die.
                status, key, payload = ("skip", "UNKNOWN", {"reason": "worker_crash", "error": repr(e)})
            with lock:
                if status == "ok":
                    record, raw_record = payload
                    results[key] = record
                    raw_db[key] = raw_record
                    new_done += 1
                else:
                    skipped[key] = payload
                    new_done += 1
            maybe_flush(force=False)

    maybe_flush(force=True)

    print("🎉 Done.")
    print(f"Processed pending: {len(pending)}")
    print(f"Output (minimal): {output_json}")
    print(f"Output (raw):     {raw_json}")
    print(f"Skipped list:     {skipped_json}")


if __name__ == "__main__":
    main()
