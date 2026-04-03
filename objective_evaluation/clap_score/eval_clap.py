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
from collections import Counter  # === resume 相关 ===

import torch
import torch.nn.functional as F
from msclap import CLAP
import utmosv2
import multiprocessing as mp

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
    files = []
    for ext in exts:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files, key=lambda p: natural_key(p.name))

def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except Exception:
        pass
    return torch.tensor(x)

def strip_suffixes(stem: str, suffixes: Tuple[str, ...]) -> str:
    s = stem
    for suf in suffixes:
        if suf and s.endswith(suf):
            s = s[: -len(suf)]
    return s

def load_clap_json(json_path: Path, text_key: str, id_key: str = "id") -> Dict[str, str]:
    with timed(f"Load CLAP JSON: {json_path.name}"):
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

def chunk_indices(indices: List[int], chunk_size: int) -> List[List[int]]:
    if chunk_size is None or chunk_size <= 0:
        return [indices]
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

def norm_cosine_diag(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> List[float]:
    ae = F.normalize(audio_emb, dim=1)
    te = F.normalize(text_emb, dim=1)
    return (ae * te).sum(dim=1).tolist()

def setup_visible_gpus(gpu_ids_arg: str) -> List[int]:
    if gpu_ids_arg is None:
        gpu_ids_arg = ""
    s = gpu_ids_arg.strip().lower()
    if s == "" or s == "all" or s == "auto":
        env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if env:
            visible = [int(t) for t in env.split(",") if t.strip()]
            log(f"🟢 Using GPUs from existing CUDA_VISIBLE_DEVICES={env} "
                f"(logical->physical: {list(enumerate(visible))})")
            return visible
        if torch.cuda.is_available():
            try:
                count = torch.cuda.device_count()
            except Exception:
                count = 0
            logical2physical = list(range(count))
            log(f"🟢 Using all available GPUs (no mask). "
                f"logical->physical: {list(enumerate(logical2physical))}")
            return logical2physical
        log("⚠️ CUDA 不可用，退回 CPU。")
        return []
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        ids = []
        for p in parts:
            if not p.isdigit():
                raise ValueError(f"--gpu-ids 需为逗号分隔的整数，如 '0,2,3'；收到: {gpu_ids_arg}")
            ids.append(int(p))
        uniq = []
        for i in ids:
            if i not in uniq:
                uniq.append(i)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in uniq)
        log(f"🟢 Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"(logical->physical: {list(enumerate(uniq))})")
        return uniq

# ---------------------- UTMOS CPU pool ----------------------

_UTMOS_MODEL = None
_UTMOS_DEVICE_STR = "cpu"

def _utmos_worker_init(device_str: str):
    global _UTMOS_MODEL, _UTMOS_DEVICE_STR
    _UTMOS_DEVICE_STR = device_str
    torch.set_grad_enabled(False)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _UTMOS_MODEL = utmosv2.create_model(pretrained=True)

def _utmos_worker_predict(args_tuple):
    idx, a = args_tuple
    try:
        out = _UTMOS_MODEL.predict(
            input_path=str(a),
            device=_UTMOS_DEVICE_STR,
            num_workers=0,
            batch_size=1,
            num_repetitions=1,
            verbose=False
        )
        if isinstance(out, dict) and "mos" in out:
            out = out["mos"]
        return idx, float(out)
    except Exception as e:
        log(f"[UTMOS-CPU] failed on {a}: {e}")
        return idx, None

# ---------------------- resume 相关工具 ----------------------

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

def _assemble_rows(audios: List[Path], texts: List[str],
                   clap_vals: List[Optional[float]], utmos_vals: List[Optional[float]]):
    rows = []
    for i, a in enumerate(audios):
        text = texts[i] if (texts and i < len(texts)) else ""
        clap_sc = clap_vals[i] if clap_vals[i] is not None else ""
        utmos_sc = utmos_vals[i] if utmos_vals[i] is not None else ""
        rows.append([str(a), text, clap_sc, utmos_sc])
    # 统计平均
    cvals = [float(x[2]) for x in rows if x[2] != ""]
    mvals = [float(x[3]) for x in rows if x[3] != ""]
    avg_clap = (sum(cvals) / len(cvals)) if cvals else 0.0
    avg_mos = (sum(mvals) / len(mvals)) if mvals else 0.0
    return rows, avg_clap, avg_mos

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


def _write_csv_all(out_csv: Path, audios: List[Path], texts: List[str],
                   clap_vals: List[Optional[float]], utmos_vals: List[Optional[float]]):
    rows, avg_clap, avg_mos = _assemble_rows(audios, texts, clap_vals, utmos_vals)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp.csv")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "text", "clap_pair_score", "utmosv2"])
        w.writerows(rows)
        w.writerow(["AVERAGE", "", avg_clap, avg_mos])
    tmp.replace(out_csv)
    return avg_clap, avg_mos

# -------------------------- main --------------------------

def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--audios-dir", required=True)
    ap.add_argument("--extensions", default=".wav")
    ap.add_argument("--out", default="metrics.csv")

    ap.add_argument("--run-clap", action="store_true")
    ap.add_argument("--run-utmos", action="store_true")

    # CLAP
    ap.add_argument("--clap-texts-json", type=str)
    ap.add_argument("--clap-text-key", default="caption_with_nvb",
                    choices=["text", "text_with_mark", "caption_with_nvb"])
    ap.add_argument("--clap-id-key", default="id")
    ap.add_argument("--id-strip-suffixes", default="_gemini")
    ap.add_argument("--auto-detect-id-suffixes", action="store_true")
    ap.add_argument("--clap-batch", type=int, default=64)

    # 设备
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpu-ids", default="")

    # UTMOS 策略
    ap.add_argument("--utmos_device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto: 单卡+跑CLAP时用cpu, 否则cuda；cuda/cpu 强制指定")
    ap.add_argument("--utmos_cpu_workers", type=int, default=8,
                    help="当 UTMOS 用 CPU 时的并行进程数（每进程一份模型）")
    ap.add_argument("--utmos_chunksize", type=int, default=16)
    ap.add_argument("--utmos_log_every", type=int, default=50, help="UTMOS-GPU 逐条进度打印间隔")

    # === resume 相关 ===
    ap.add_argument("--overwrite", action="store_true",
                    help="忽略现有 --out 文件，全部重算。默认会自动断点续跑。")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help=">=1 时，每新完成这么多条就写一次 CSV 以便崩溃后能续跑。")

    args = ap.parse_args()

    # GPU 可见性（仅影响 CLAP 阶段 & utmos_device=auto 时的判定）
    if args.device == "cuda":
        setup_visible_gpus(args.gpu_ids)
        if not torch.cuda.is_available():
            log("⚠️ CUDA 不可用，切换到 CPU。")
            args.device = "cpu"

    # 扩展名
    exts = tuple(e.strip() for e in args.extensions.split(",") if e.strip()) or (".wav",".mp3")
    exts = (".wav", ".mp3")
    print("拓展名", exts)

    # 收集音频
    with timed(f"Search audio files in {args.audios_dir}"):
        audios = find_audio_files(Path(args.audios_dir), exts)
    if not audios:
        raise RuntimeError(f"未在 {args.audios_dir} 中找到音频文件，扩展名：{exts}")
    log(f"Found {len(audios)} files")

    # 准备后缀表（手动 + 自动）
    manual_suffixes = tuple(s.strip() for s in getattr(args, "id_strip_suffixes").split(",") if s.strip())
    if args.auto_detect_id_suffixes and not manual_suffixes:
        with timed("Auto-detect ID suffixes"):
            detected_suffixes: Tuple[str, ...] = auto_detect_suffixes(audios)
    else:
        detected_suffixes = tuple()
        log("🔎 Auto-detected ID suffixes: (skipped)")
    strip_suffixes_tuple = manual_suffixes + tuple(s for s in detected_suffixes if s not in manual_suffixes)

    # === 断点续跑：读取历史 CSV
    out_csv = Path(args.out)
    existing = _read_existing_csv(out_csv) if (out_csv.exists() and not args.overwrite) else {}
    if existing:
        log(f"🔁 Resume: loaded {len(existing)} rows from {out_csv}")
    ex_by_path, ex_by_name = _index_existing(existing)

    # 统一缓存
    N = len(audios)
    texts: List[str] = [""] * N
    clap_vals: List[Optional[float]] = [None] * N
    utmos_vals: List[Optional[float]] = [None] * N

    # 先用历史结果填充（路径优先匹配，其次唯一文件名匹配）
    filled_from_existing = 0
    for i, a in enumerate(audios):
        key_exact = str(a)
        key_res = str(a.resolve())
        ex = ex_by_path.get(key_exact) or ex_by_path.get(key_res) or ex_by_name.get(a.name)
        if ex:
            texts[i] = ex.get("text", "") or ""
            clap_vals[i] = ex.get("clap", None)
            utmos_vals[i] = ex.get("utmos", None)
            filled_from_existing += 1
    if filled_from_existing:
        log(f"🔁 Resume: matched {filled_from_existing} / {N} current files with existing CSV")

    # ===== 准备 CLAP 文本（仅为缺失者补齐） =====
    # 如果 run_clap 且某些条目需要计算，但文本为空，才去加载 JSON 进行补齐
    need_clap_indices_initial = []
    if args.run_clap:
        need_clap_indices_initial = [i for i in range(N) if clap_vals[i] is None]
        missing_text_needed = [i for i in need_clap_indices_initial if str(texts[i]).strip() == ""]
        if missing_text_needed:
            if not args.clap_texts_json:
                raise ValueError(
                    f"--run-clap 需要文本，而现有 CSV 中这些条目没有文本，且未提供 --clap-texts-json。"
                    f" 缺失文本的数量: {len(missing_text_needed)}"
                )
            text_map = load_clap_json(Path(args.clap_texts_json),
                                      text_key=args.clap_text_key,
                                      id_key=args.clap_id_key)
            with timed("Align audio IDs with CLAP texts"):
                for i in missing_text_needed:
                    a = audios[i]
                    # stem_clean = strip_suffixes(a.stem, strip_suffixes_tuple)
                    stem_clean = keep_until_second_underscore(a.stem)
                    if stem_clean not in text_map:
                        raise KeyError(
                            f"缺少 {stem_clean} 的文本描述（来自 {args.clap_texts_json}，键：{args.clap_id_key}；"
                            f"原文件名 stem={a.stem}；剥离后缀={strip_suffixes_tuple}）"
                        )
                    texts[i] = text_map[stem_clean]

    # ===== 阶段 A：UTMOS =====
    processed_since_ckpt = 0
    ckpt_every = max(0, int(args.checkpoint_every))

    def maybe_checkpoint(tag: str):
        nonlocal processed_since_ckpt
        if ckpt_every >= 1 and processed_since_ckpt >= ckpt_every:
            with timed(f"Checkpoint write CSV ({tag})"):
                avg_clap, avg_mos = _write_csv_all(out_csv, audios, texts, clap_vals, utmos_vals)
            log(f"💾 checkpoint saved. mean CLAP={avg_clap:.4f} | mean UTMOS={avg_mos:.4f}")
            processed_since_ckpt = 0

    if args.run_utmos:
        # 仅处理缺失者
        utmos_todo = [i for i in range(N) if utmos_vals[i] is None]
        log(f"🧪 UTMOS to compute: {len(utmos_todo)} / {N}")

        vis_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if args.utmos_device == "auto":
            if args.run_clap and vis_gpus <= 1:
                utmos_dev = "cpu"
            else:
                utmos_dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            utmos_dev = args.utmos_device

        log(f"🧪 UTMOS stage on {utmos_dev}")
        if utmos_dev == "cpu":
            if utmos_todo:
                with timed(f"UTMOS on CPU (workers={args.utmos_cpu_workers}, chunksize={args.utmos_chunksize})"):
                    n_workers = max(1, int(args.utmos_cpu_workers))
                    tasks = [(i, str(audios[i])) for i in utmos_todo]
                    with mp.Pool(processes=n_workers, initializer=_utmos_worker_init, initargs=("cpu",)) as pool:
                        total = len(tasks)
                        done = 0
                        for idx, mos in pool.imap_unordered(
                                _utmos_worker_predict,
                                tasks,
                                chunksize=max(1, int(args.utmos_chunksize))):
                            utmos_vals[idx] = mos
                            done += 1
                            processed_since_ckpt += 1
                            if done % 50 == 0 or done == total:
                                log(f"[UTMOS-CPU] progress: {done}/{total}")
                            maybe_checkpoint("UTMOS-CPU")
        else:
            if utmos_todo:
                with timed("UTMOS on GPU (sequential)"):
                    torch.set_grad_enabled(False)
                    model_t0 = time.perf_counter()
                    model = utmosv2.create_model(pretrained=True)
                    log(f"[UTMOS-GPU] model created in {time.perf_counter()-model_t0:.3f}s")
                    device_str = "cuda:0"
                    try:
                        torch.cuda.set_device(0)
                    except Exception:
                        device_str = "cuda"
                    total = len(utmos_todo)
                    log_every = max(1, int(args.utmos_log_every))
                    t_batch0 = time.perf_counter()
                    for j, i in enumerate(utmos_todo):
                        if (j % log_every == 0) or (j + 1 == total):
                            dt = time.perf_counter() - t_batch0
                            log(f"[UTMOS-GPU] progress {j}/{total} (last {log_every} ~ {dt:.3f}s)")
                            t_batch0 = time.perf_counter()
                        a = audios[i]
                        try:
                            t0 = time.perf_counter()
                            out = model.predict(
                                input_path=str(a),
                                device=device_str,
                                num_workers=0,
                                batch_size=1,
                                num_repetitions=1,
                                verbose=False
                            )
                            if isinstance(out, dict) and "mos" in out:
                                out = out["mos"]
                            utmos_vals[i] = float(out)
                            processed_since_ckpt += 1
                            log(f"[UTMOS-GPU] #{i:05d} {Path(a).name} -> {utmos_vals[i]:.4f} in {time.perf_counter()-t0:.3f}s")
                        except Exception as e:
                            log(f"[UTMOS-GPU] failed on {a}: {e}")
                            utmos_vals[i] = None
                        maybe_checkpoint("UTMOS-GPU")

    # ===== 阶段 B：CLAP =====
    if args.run_clap:
        # 仅对缺失者计算（文本非空）
        valid = [i for i in range(N) if (clap_vals[i] is None and str(texts[i]).strip() != "")]
        log(f"[CLAP] to compute: {len(valid)} / {N} | batch={args.clap_batch}")

        if valid:
            with timed("CLAP load model"):
                use_cuda = (args.device == "cuda") and torch.cuda.is_available()
                clap = CLAP(version="2023", use_cuda=use_cuda)

            with timed("CLAP embedding + similarity"):
                with torch.no_grad():
                    # 可选：按长度排序，降低批失败率（字符长度近似）
                    valid.sort(key=lambda i: len(texts[i]))
                    batch_size = max(1, int(args.clap_batch))
                    for bi, idx_chunk in enumerate(chunk_indices(valid, batch_size)):
                        t0 = time.perf_counter()
                        texts_chunk = [texts[i] for i in idx_chunk]
                        paths_chunk = [str(audios[i]) for i in idx_chunk]

                        # 文本批次尝试；失败则回退逐条
                        try:
                            te = clap.get_text_embeddings(texts_chunk)    # (B, D)
                            te = ensure_tensor(te)
                            if te.ndim == 1:
                                te = te.unsqueeze(0)
                        except Exception as e:
                            log(f"[CLAP] text batch collation failed on chunk size={len(texts_chunk)}: {e}")
                            te_list = []
                            for i in idx_chunk:
                                te_i = clap.get_text_embeddings([texts[i]])   # (1, D)
                                te_i = ensure_tensor(te_i)
                                if te_i.ndim == 1:
                                    te_i = te_i.unsqueeze(0)
                                te_list.append(te_i)
                            te = torch.cat(te_list, dim=0)

                        # 音频保持批量
                        ae = clap.get_audio_embeddings(paths_chunk)   # (B, D)
                        ae = ensure_tensor(ae)
                        if ae.ndim == 1:
                            ae = ae.unsqueeze(0)

                        sims = (F.normalize(ae, dim=1) * F.normalize(te, dim=1)).sum(dim=1).tolist()
                        for rel, i in enumerate(idx_chunk):
                            clap_vals[i] = float(sims[rel])
                            processed_since_ckpt += 1

                        log(f"[CLAP] chunk {bi:04d} size={len(idx_chunk)} done in {time.perf_counter()-t0:.3f}s")
                        maybe_checkpoint("CLAP")

    # ===== 合并并写出 =====
    with timed("Write CSV"):
        avg_clap, avg_mos = _write_csv_all(out_csv, audios, texts, clap_vals, utmos_vals)

    log(f"✅ Saved results to {out_csv}")
    if args.run_utmos and isinstance(avg_mos, float):
        log(f"UTMOS mean: {avg_mos:.4f}")
    if args.run_clap and isinstance(avg_clap, float):
        log(f"CLAP mean: {avg_clap:.4f}")

if __name__ == "__main__":
    main()
