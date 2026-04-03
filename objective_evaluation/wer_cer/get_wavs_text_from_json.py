#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import time
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from collections import Counter  # used by _index_existing

# ========================= 日志与计时 =========================

START = time.perf_counter()

def log(msg: str):
    """统一带时间戳的日志输出。"""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    elapsed = time.perf_counter() - START
    print(f"[{now} | +{elapsed:8.3f}s] {msg}", flush=True)

@contextmanager
def timed(title: str):
    """记录一个步骤的起止耗时。"""
    log(f"▶ {title} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {title} done in {dt:.3f}s")

# -------------------------- utils --------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def find_audio_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files, key=lambda p: natural_key(p.name))

def strip_suffixes(stem: str, suffixes: Tuple[str, ...]) -> str:
    s = stem
    for suf in suffixes:
        if suf and s.endswith(suf):
            s = s[: -len(suf)]
    return s

def load_clap_json(json_path: Path, text_key: str, id_key: str = "id") -> Dict[str, str]:
    with timed(f"Load JSON: {json_path.name}"):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} 顶层应为数组。")
    mapping: Dict[str, str] = {}
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"{json_path} 第 {i} 个元素不是对象。")
        if id_key not in obj:
            raise KeyError(f"{json_path} 第 {i} 个元素缺少字段 '{id_key}'。")
        if text_key not in obj:
            raise KeyError(f"{json_path} 第 {i} 个元素缺少字段 '{text_key}'。")
        mapping[str(obj[id_key])] = str(obj[text_key])
    return mapping

def auto_detect_suffixes(audios: List[Path]) -> Tuple[str, ...]:
    suffixes: Set[str] = set()
    for p in audios:
        stem = p.stem
        if "_" not in stem:
            continue
        last = stem.rsplit("_", 1)[-1]
        if re.search(r"[A-Za-z]", last):
            suffixes.add("_" + last)
    detected = tuple(sorted(suffixes, key=lambda x: x.lower()))
    if detected:
        log(f"🔎 Auto-detected ID suffixes: {', '.join(detected)}")
    else:
        log("🔎 Auto-detected ID suffixes: (none)")
    return detected

# ---------------------- resume 相关工具 ----------------------

def keep_until_second_underscore(stem: str) -> str:
    """
    只保留文件名中第二个下划线之前的部分：
    a_b_c_d -> a_b
    a_b     -> a_b
    a       -> a
    """
    parts = stem.split("_")
    if len(parts) <= 2:
        return stem
    return "_".join(parts[:2])


def _safe_float(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _read_existing_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取历史 CSV，返回 {audio_path: {text, clap, utmos}}"""
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ap = row.get("audio_path", "")
            if not ap or ap == "AVERAGE":
                continue
            out[ap] = {
                "text": row.get("text", "") or "",
                "clap": _safe_float(row.get("clap_pair_score", "")),
                "utmos": _safe_float(row.get("utmosv2", "")),
            }
    return out

def _index_existing(existing: Dict[str, Dict[str, Any]]):
    """构建按完整路径与按文件名的索引（仅唯一文件名参与后者）。"""
    by_path = existing
    name_cnt = Counter(Path(k).name for k in existing.keys())
    unique_names = {n for n, c in name_cnt.items() if c == 1}
    by_name: Dict[str, Dict[str, Any]] = {}
    for k, v in existing.items():
        name = Path(k).name
        if name in unique_names:
            by_name[name] = v
    return by_path, by_name

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-json", type=str, required=True)
    ap.add_argument("--text-key", default="text",
                    choices=["text", "text_with_mark", "caption_with_nvb"])
    ap.add_argument("--id-key", default="id")
    ap.add_argument("--out-file", type=str, required=True)
    ap.add_argument("--audios-dir", required=True)
    ap.add_argument("--extensions", default=".wav")
    ap.add_argument("--id-strip-suffixes", default="_gemini",
                    help="逗号分隔的后缀列表，例如: _gemini,_v11")
    ap.add_argument("--auto-detect-id-suffixes", action="store_true")
    args = ap.parse_args()

    # 读取 JSON (修正参数名)
    text_map = load_clap_json(
        Path(args.text_json),
        text_key=args.text_key,
        id_key=args.id_key,
    )

    # 扩展名
    exts = tuple(e.strip() for e in args.extensions.split(",") if e.strip()) or (".wav",".mp3")
    exts = (".wav",".mp3")

    # 收集音频
    with timed(f"Search audio files in {args.audios_dir}"):
        audios = find_audio_files(Path(args.audios_dir), exts)
    if not audios:
        raise RuntimeError(f"未在 {args.audios_dir} 中找到音频文件，扩展名：{exts}")
    log(f"Found {len(audios)} files")

    # 准备后缀表（手动 + 自动），总是合并去重
    manual_suffixes = tuple(s.strip() for s in getattr(args, "id_strip_suffixes").split(",") if s.strip())
    detected_suffixes: Tuple[str, ...] = tuple()
    if args.auto_detect_id_suffixes:
        with timed("Auto-detect ID suffixes"):
            detected_suffixes = auto_detect_suffixes(audios)
    else:
        log("🔎 Auto-detected ID suffixes: (skipped)")
    # 合并并去重
    strip_suffixes_tuple = tuple(dict.fromkeys(manual_suffixes + detected_suffixes))

    # 输出
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f_w:
        missing = 0
        for a in audios:
            # stem_clean = strip_suffixes(a.stem, strip_suffixes_tuple)
            
            stem_clean = keep_until_second_underscore(a.stem)
            if stem_clean not in text_map:
                missing += 1
                raise KeyError(
                    f"缺少 {stem_clean} 的文本描述（来自 --text-json={args.text_json}，键：{args.id_key}；"
                    f"原文件名 stem={a.stem}；剥离后缀={strip_suffixes_tuple}）"
                )
            text = text_map[stem_clean]
            out_line = "|".join([str(a), text])
            f_w.write(out_line + "\n")
    log(f"✅ Wrote {len(audios)} lines to {out_path}")

if __name__ == "__main__":
    main()
