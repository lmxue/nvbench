#!/usr/bin/env python3
import os, json, glob, argparse, math
from collections import defaultdict, OrderedDict

PREFERRED_NUMERIC = [
    "coverage",
    "precision_paired",
    "recall_paired",
    "f1_paired",
    "ca_f1",
    "tp", "fp", "fn",
    "paired_size_I", "predicted_size_P", "reference_size_R",
    "u_match",
    # 距离相关（如果你的 metrics.json 里有这些就会被聚合）
    "tpd", "ntd",
    "tpd_mean", "tpd_std", "tpd_var", "tpd_n",
    "ntd_mean", "ntd_std", "ntd_var", "ntd_n",
    "alpha", "delta",
]
PREFERRED_STRING = ["pos_unit"]

# 终端默认打印的列（基于“聚合输出列名”，即 *_mean / *_var / *_std）
DEFAULT_PRINT_COLS = [
    "system",
    "n_runs",
    "coverage_mean", "coverage_var",
    "precision_paired_mean", "precision_paired_var",
    "recall_paired_mean", "recall_paired_var",
    "f1_paired_mean", "f1_paired_var",
    "ca_f1_mean", "ca_f1_var",

    # 优先打印原始key=tpd/ntd聚合出来的列
    "tpd_mean", "tpd_var",
    "ntd_mean", "ntd_var",

    # 兼容原始key=tpd_mean/ntd_mean聚合出来的列（有的话也会显示）
    "tpd_mean_mean", "tpd_mean_var",
    "ntd_mean_mean", "ntd_mean_var",
]

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def pick_metrics_dict(obj: dict) -> dict:
    # v2: {"runs":[{...}], ...} -> 用 runs[0]
    if isinstance(obj.get("runs"), list) and obj["runs"]:
        if isinstance(obj["runs"][0], dict):
            return obj["runs"][0]
    # 其他常见结构
    for k in ("summary", "overall", "metrics", "aggregate"):
        if isinstance(obj.get(k), dict):
            return obj[k]
    return obj

def mean_var(vals, sample=True):
    n = len(vals)
    if n == 0:
        return None, None
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    denom = (n - 1) if sample else n
    var = sum((x - mu) ** 2 for x in vals) / denom
    return mu, var

def fmt(x):
    if x is None:
        return ""
    if x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)

def is_numeric_like(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def print_aligned(title, rows, cols):
    print("\n" + title)
    if not rows:
        print("(no rows)\n")
        return

    # stringify
    table = []
    for r in rows:
        table.append({c: fmt(r.get(c, "")) for c in cols})

    widths = {c: len(c) for c in cols}
    for row in table:
        for c in cols:
            widths[c] = max(widths[c], len(row[c]))

    # numeric align (>=80% numeric-like)
    num_cols = set()
    for c in cols:
        vals = [row[c] for row in table if row[c] != ""]
        if vals:
            if sum(is_numeric_like(v) for v in vals) / len(vals) >= 0.8:
                num_cols.add(c)

    def cell(c, s):
        w = widths[c]
        return s.rjust(w) if c in num_cols else s.ljust(w)

    print("  ".join(cell(c, c) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for row in table:
        print("  ".join(cell(c, row[c]) for c in cols))
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_roots", nargs="+", required=True,
                    help="三次运行根目录，例如 prediction_output_pos_verify prediction_output_pos_verify-2 prediction_output_pos_verify-3")
    ap.add_argument("--out", default="summary_metrics_3runs_mean_var.tsv")
    ap.add_argument("--population_var", action="store_true",
                    help="用总体方差(除以n)。默认样本方差(除以n-1)")
    ap.add_argument("--print_all", action="store_true",
                    help="终端打印全量列（非常宽，可能看不全）。默认只打印关键列")
    args = ap.parse_args()

    run_roots = args.run_roots
    sample = not args.population_var

    # data[(lang, system)][metric] -> [v_run1, v_run2, v_run3...]
    numeric_values = defaultdict(lambda: defaultdict(list))
    string_values  = defaultdict(lambda: defaultdict(list))
    all_numeric_keys = set()
    all_string_keys = set()

    for root in run_roots:
        files = sorted(glob.glob(os.path.join(root, "*", "*", "*.metrics.json")))
        if not files:
            raise RuntimeError(f"No *.metrics.json found under: {root}")

        for fp in files:
            rel = os.path.relpath(fp, root)
            parts = rel.split(os.sep)
            if len(parts) < 3:
                continue
            system, lang = parts[0], parts[1]
            if lang not in ("en", "zh"):
                continue

            try:
                obj = json.load(open(fp, "r", encoding="utf-8"))
            except Exception:
                continue

            m = pick_metrics_dict(obj)
            key = (lang, system)

            for k, v in m.items():
                if is_number(v):
                    numeric_values[key][k].append(float(v))
                    all_numeric_keys.add(k)

            for k in PREFERRED_STRING:
                if k in m and isinstance(m[k], str):
                    string_values[key][k].append(m[k])
                    all_string_keys.add(k)

    preferred = [k for k in PREFERRED_NUMERIC if k in all_numeric_keys]
    others = sorted([k for k in all_numeric_keys if k not in set(preferred)])
    numeric_order = preferred + others

    string_order = [k for k in PREFERRED_STRING if k in all_string_keys] + \
                   sorted([k for k in all_string_keys if k not in set(PREFERRED_STRING)])

    cols = ["lang", "system", "n_runs"] + string_order
    for k in numeric_order:
        cols += [f"{k}_mean", f"{k}_var", f"{k}_std"]

    rows = []
    for (lang, system) in sorted(numeric_values.keys()):
        row = OrderedDict()
        row["lang"] = lang
        row["system"] = system

        # n_runs: 取任意一个指标的长度
        some_metric = next(iter(numeric_values[(lang, system)].keys()))
        n_runs = len(numeric_values[(lang, system)][some_metric])
        row["n_runs"] = n_runs

        for k in string_order:
            vs = string_values[(lang, system)].get(k, [])
            if not vs:
                row[k] = ""
            elif all(v == vs[0] for v in vs):
                row[k] = vs[0]
            else:
                row[k] = "|".join(vs)

        for k in numeric_order:
            vals = numeric_values[(lang, system)].get(k, [])
            mu, var = mean_var(vals, sample=sample)
            std = None if var is None else math.sqrt(var)
            row[f"{k}_mean"] = mu
            row[f"{k}_var"]  = var
            row[f"{k}_std"]  = std

        rows.append(row)

    # 写 TSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(c, "")) for c in cols) + "\n")

    # 终端打印
    if args.print_all:
        # 全量打印（很宽）
        for lang in ("en", "zh"):
            sub = [r for r in rows if r["lang"] == lang]
            sub.sort(key=lambda x: x["system"])
            print_cols = ["system", "n_runs"] + [c for c in cols if c not in ("lang", "system", "n_runs")]
            print_aligned(f"{lang.upper()} (3 runs, ALL metrics mean/var/std)", sub, print_cols)
    else:
        # 关键列打印
        for lang in ("en", "zh"):
            sub = [r for r in rows if r["lang"] == lang]
            sub.sort(key=lambda x: x["system"])
            # 只打印存在的列
            print_cols = [c for c in DEFAULT_PRINT_COLS if c in sub[0]] if sub else DEFAULT_PRINT_COLS
            print_aligned(f"{lang.upper()} (3 runs, key metrics mean/var)", sub, print_cols)

    print(f"Saved TSV: {args.out}")
    print(f"Rows: {len(rows)} (lang×system)")

if __name__ == "__main__":
    main()