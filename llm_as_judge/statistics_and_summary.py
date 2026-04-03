#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计和汇总脚本
处理目录下的JSON评估结果文件，统计每个系统每个指标的平均值和方差
"""

import os
import json
import glob
from collections import defaultdict
import statistics


def calculate_variance(values):
    """计算方差"""
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def process_groupcompare_file(file_path):
    """
    处理GROUPCOMPARE模式的JSON文件
    返回: {
        "task_name": str,
        "systems": {
            system_name: {
                metric_key: [scores...]
            }
        }
    }
    """
    print(f"\n处理文件: {os.path.basename(file_path)}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return None
    
    meta = data.get("__meta__", {})
    task_name = meta.get("task_name", "unknown")
    items = data.get("items", {})
    
    if not isinstance(items, dict):
        print(f"  ❌ 无效的items结构")
        return None
    
    # 收集所有指标键
    all_metric_keys = set()
    for rec in items.values():
        if rec.get("status") != "ok":
            continue
        pred = rec.get("prediction_capped") or rec.get("prediction") or {}
        results = pred.get("results", {}) if isinstance(pred, dict) else {}
        if isinstance(results, dict):
            for lb, one in results.items():
                if isinstance(one, dict):
                    for k in one.keys():
                        if k.endswith("_score"):
                            all_metric_keys.add(k)
        if all_metric_keys:
            break  # 只需要看一个记录就能知道所有指标
    
    if not all_metric_keys:
        print(f"  ⚠️  未找到任何指标分数")
        return None
    
    # 按系统和指标收集分数
    system_scores = defaultdict(lambda: defaultdict(list))
    
    ok_count = 0
    error_count = 0
    
    for rec in items.values():
        if rec.get("status") != "ok":
            error_count += 1
            continue
        
        ok_count += 1
        gt = rec.get("ground_truth", {}) or {}
        label_to_system = gt.get("label_to_system", {}) or {}
        pred = rec.get("prediction_capped") or rec.get("prediction") or {}
        results = pred.get("results", {}) if isinstance(pred, dict) else {}
        
        if not isinstance(results, dict):
            continue
        
        # 提取每个标签对应的系统分数
        for lb, one in results.items():
            sys = label_to_system.get(lb)
            if not sys or not isinstance(one, dict):
                continue
            
            for metric_key in all_metric_keys:
                score_val = one.get(metric_key)
                if isinstance(score_val, (int, float)):
                    system_scores[sys][metric_key].append(float(score_val))
    
    print(f"  ✓ 成功: {ok_count}, 错误: {error_count}, 系统数: {len(system_scores)}")
    
    return {
        "task_name": task_name,
        "file_path": file_path,
        "systems": dict(system_scores),
        "all_metrics": sorted(all_metric_keys)
    }


def process_single_system_file(file_path):
    """
    处理单系统评估的JSON文件（非GROUPCOMPARE模式）
    返回类似结构
    """
    print(f"\n处理文件: {os.path.basename(file_path)}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return None
    
    meta = data.get("__meta__", {})
    task_name = meta.get("task_name", "unknown")
    system_key = meta.get("system_key", "unknown")
    items = data.get("items", {})
    
    if not isinstance(items, dict):
        print(f"  ❌ 无效的items结构")
        return None
    
    # 收集所有指标键
    all_metric_keys = set()
    for rec in items.values():
        if rec.get("status") != "ok":
            continue
        pred = rec.get("prediction_capped") or rec.get("prediction") or {}
        if isinstance(pred, dict):
            for k in pred.keys():
                if k.endswith("_score"):
                    all_metric_keys.add(k)
        if all_metric_keys:
            break
    
    if not all_metric_keys:
        print(f"  ⚠️  未找到任何指标分数")
        return None
    
    # 收集分数
    scores_by_metric = defaultdict(list)
    
    ok_count = 0
    error_count = 0
    
    for rec in items.values():
        if rec.get("status") != "ok":
            error_count += 1
            continue
        
        ok_count += 1
        pred = rec.get("prediction_capped") or rec.get("prediction") or {}
        
        for metric_key in all_metric_keys:
            score_val = pred.get(metric_key)
            if isinstance(score_val, (int, float)):
                scores_by_metric[metric_key].append(float(score_val))
    
    print(f"  ✓ 成功: {ok_count}, 错误: {error_count}")
    
    return {
        "task_name": task_name,
        "file_path": file_path,
        "systems": {system_key: dict(scores_by_metric)},
        "all_metrics": sorted(all_metric_keys)
    }


def print_statistics(results_data, show_variance=True, show_count=False):
    """打印统计结果（矩阵形式，默认显示均值±方差）"""
    if not results_data:
        return
    
    task_name = results_data["task_name"]
    systems = results_data["systems"]
    all_metrics = results_data["all_metrics"]
    
    print(f"\n{'='*100}")
    print(f"任务: {task_name}")
    if "file_path" in results_data:
        print(f"文件: {os.path.basename(results_data['file_path'])}")
    print(f"{'='*100}")
    
    if not systems:
        print("  无有效数据")
        return
    
    # 准备矩阵数据
    system_names = sorted(systems.keys())
    
    # 格式化指标名称（更易读）
    metric_displays = []
    for metric_key in all_metrics:
        metric_display = metric_key.replace("_score", "").replace("_", " ").title()
        metric_displays.append((metric_key, metric_display))
    
    # 计算列宽（考虑"均值±方差"格式，需要更宽的列）
    max_system_name_len = max([len(s) for s in system_names] + [10])
    max_metric_name_len = max([len(m[1]) for m in metric_displays] + [15])
    # "均值±方差"格式大约需要15-18个字符，取较大值
    col_width = max(18, max_metric_name_len, 15)
    
    # 打印表头（指标名称左对齐）
    header = f"{'系统':<{max_system_name_len}}"
    for metric_key, metric_display in metric_displays:
        header += f" {metric_display:<{col_width}}"
    print(header)
    print("-" * len(header))
    
    # 打印每行数据（每个系统，数值右对齐）
    for system_name in system_names:
        system_data = systems[system_name]
        row = f"{system_name:<{max_system_name_len}}"
        
        for metric_key, metric_display in metric_displays:
            scores = system_data.get(metric_key, [])
            if scores:
                mean_val = statistics.mean(scores)
                if show_variance:
                    var_val = calculate_variance(scores)
                    cell = f"{mean_val:.3f}±{var_val:.3f}"
                elif show_count:
                    n = len(scores)
                    cell = f"{mean_val:.3f}(n={n})"
                else:
                    cell = f"{mean_val:.3f}"
            else:
                cell = "N/A"
            row += f" {cell:>{col_width}}"  # 右对齐数值
        
        print(row)


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"扫描目录: {script_dir}")
    print(f"查找JSON评估结果文件...")
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(script_dir, "*.json"))
    
    if not json_files:
        print("未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    all_results = []
    
    # 处理每个文件
    for json_file in sorted(json_files):
        # 判断是GROUPCOMPARE模式还是单系统模式
        if "__GROUPCOMPARE" in os.path.basename(json_file):
            result = process_groupcompare_file(json_file)
        else:
            result = process_single_system_file(json_file)
        
        if result:
            all_results.append(result)
    
    # 打印所有统计结果（不进行跨文件汇总）
    print(f"\n\n{'#'*80}")
    print(f"统计汇总报告")
    print(f"{'#'*80}")
    
    for result in all_results:
        print_statistics(result)


if __name__ == "__main__":
    main()

