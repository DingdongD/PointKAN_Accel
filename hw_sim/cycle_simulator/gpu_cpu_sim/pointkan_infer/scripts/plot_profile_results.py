#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 pointkan_infer/outputs 下的 gpu_*/cpu_* 各任务 summary.json 与采样 CSV 聚合数据，
绘制三任务的 GPU/CPU Energy、Latency 对比与功耗/利用率随时间曲线。

用法：
  python scripts/plot_profile_results.py
  python scripts/plot_profile_results.py --out-dir outputs --out outputs/profile_curves.png
"""

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT / "outputs"

TASK_NAMES = {
    "modelnet40": "ModelNet40",
    "scanobjectnn": "ScanObjectNN",
    "part_seg": "Part Segmentation",
}


def find_latest_run_dir(base: Path, prefix: str) -> Path | None:
    """在 base 下找 prefix 开头的子目录，再取其中最新时间戳子目录。"""
    task_dir = base / prefix
    if not task_dir.is_dir():
        return None
    subdirs = [d for d in task_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[:8].isdigit()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)


def load_summary(run_dir: Path) -> dict | None:
    p = run_dir / "summary.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_gpu_samples(run_dir: Path, run_idx: int = 0) -> list[dict] | None:
    p = run_dir / f"run_{run_idx}_gpu_samples.csv"
    if not p.is_file():
        return None
    import csv
    rows = []
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["t_rel_s"] = float(row.get("t_rel_s", 0))
                row["power_w"] = float(row.get("power_w", 0)) if row.get("power_w") else None
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows if rows else None


def load_cpu_samples(run_dir: Path, run_idx: int = 0) -> list[dict] | None:
    p = run_dir / f"run_{run_idx}_cpu_samples.csv"
    if not p.is_file():
        return None
    import csv
    rows = []
    with open(p, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["t_rel_s"] = float(row.get("t_rel_s", 0))
                row["cpu_percent"] = float(row.get("cpu_percent", 0)) if row.get("cpu_percent") else None
                row["rss_mb"] = float(row.get("rss_mb", 0)) if row.get("rss_mb") else None
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows if rows else None


def collect_aggregates(outputs_dir: Path, prefixes: list[str]) -> dict:
    """汇总各任务最新 run 的 aggregate 指标。"""
    data = {}
    for prefix in prefixes:
        task = prefix.replace("gpu_", "").replace("cpu_", "")
        run_dir = find_latest_run_dir(outputs_dir, prefix)
        if run_dir is None:
            data[task] = None
            continue
        summary = load_summary(run_dir)
        if summary is None or "aggregate" not in summary:
            data[task] = None
            continue
        agg = summary["aggregate"]
        data[task] = {
            "run_dir": run_dir,
            "latency_s_mean": agg.get("latency_s_mean"),
            "inference_only_s_mean": agg.get("inference_only_s_mean"),
            "inference_only_per_forward_s_mean": agg.get("inference_only_per_forward_s_mean"),
            "latency_s_min": agg.get("latency_s_min"),
            "latency_s_max": agg.get("latency_s_max"),
            "gpu_energy_j_mean": agg.get("gpu_energy_j_mean"),
            "gpu_avg_power_w_mean": agg.get("gpu_avg_power_w_mean"),
            "gpu_peak_power_w_max": agg.get("gpu_peak_power_w_max"),
            "cpu_energy_pkg_j_mean": agg.get("cpu_energy_pkg_j_mean"),
            "cpu_energy_ram_j_mean": agg.get("cpu_energy_ram_j_mean"),
            "cpu_energy_total_j_mean": agg.get("cpu_energy_total_j_mean"),
            "cpu_energy_inference_pkg_j_mean": agg.get("cpu_energy_inference_pkg_j_mean"),
            "cpu_energy_inference_ram_j_mean": agg.get("cpu_energy_inference_ram_j_mean"),
            "cpu_energy_inference_total_j_mean": agg.get("cpu_energy_inference_total_j_mean"),
            "cpu_avg_percent_mean": agg.get("cpu_avg_percent_mean"),
            "rss_peak_mb_max": agg.get("rss_peak_mb_max"),
            "num_runs": agg.get("num_runs"),
        }
    return data


def plot_summary_bars(data: dict, out_path: Path, mode: str = "gpu") -> None:
    """绘制三任务 Latency、GPU Energy、CPU Energy 三组柱状图（各一子图，量纲独立）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skip bar plot")
        return

    tasks = [TASK_NAMES.get(t, t) for t in ("modelnet40", "scanobjectnn", "part_seg")]
    latency = []
    gpu_j = []
    cpu_j = []
    for task_key in ("modelnet40", "scanobjectnn", "part_seg"):
        d = data.get(task_key)
        if d is None:
            latency.append(0)
            gpu_j.append(0)
            cpu_j.append(0)
            continue
        # 仅 model 推理时间
        latency.append(d.get("inference_only_per_forward_s_mean") or d.get("inference_only_s_mean") or 0)
        gpu_j.append(d["gpu_energy_j_mean"] or 0)
        # 仅 model 推理阶段 CPU 能耗
        cpu_j.append(
            d.get("cpu_energy_inference_total_j_mean")
            or d.get("cpu_energy_inference_pkg_j_mean")
            or 0
        )

    x = np.arange(len(tasks))
    width = 0.6
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2ecc71", "#3498db", "#9b59b6"]

    axes[0].bar(x, latency, width=width, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=15, ha="right")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Single forward (batch=1)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, gpu_j, width=width, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=15, ha="right")
    axes[1].set_ylabel("Energy (J)")
    axes[1].set_title("GPU Energy")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, cpu_j, width=width, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tasks, rotation=15, ha="right")
    axes[2].set_ylabel("Energy (J)")
    axes[2].set_title("CPU Energy (RAPL)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"PointKAN Inference: Three Tasks ({mode.upper()} mode)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_curves(data: dict, out_path: Path) -> None:
    """绘制各任务 run_0 的 GPU 功耗、CPU 利用率随时间曲线（子图）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip curves")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes[0, 0].set_title("GPU Power (W) vs Time")
    axes[0, 1].set_title("CPU Util (%) vs Time")

    for row, task_key in enumerate(("modelnet40", "scanobjectnn", "part_seg")):
        d = data.get(task_key)
        if d is None or "run_dir" not in d:
            continue
        run_dir = d["run_dir"]
        gpu_rows = load_gpu_samples(run_dir, 0)
        cpu_rows = load_cpu_samples(run_dir, 0)
        name = TASK_NAMES.get(task_key, task_key)
        if gpu_rows:
            t = [r["t_rel_s"] for r in gpu_rows]
            p = [r.get("power_w") for r in gpu_rows]
            p = [x if x is not None else 0 for x in p]
            axes[row, 0].plot(t, p, label=name)
        if cpu_rows:
            t = [r["t_rel_s"] for r in cpu_rows]
            u = [r.get("cpu_percent") for r in cpu_rows]
            u = [x if x is not None else 0 for x in u]
            axes[row, 1].plot(t, u, label=name)

    for ax in axes.flat:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (s)")
    fig.suptitle("PointKAN Inference: GPU Power & CPU Util over Time (run_0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot PointKAN profile: latency, GPU/CPU energy, power curves")
    ap.add_argument("--out-dir", type=str, default=str(OUTPUTS_DIR), help="outputs 根目录")
    ap.add_argument("--out", type=str, default=None, help="汇总柱状图输出路径，默认 out-dir/profile_summary.png")
    ap.add_argument("--curves", type=str, default=None, help="功耗/利用率曲线图路径，默认 out-dir/profile_curves.png")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = ["gpu_modelnet40", "gpu_scanobjectnn", "gpu_part_seg"]
    data = collect_aggregates(out_dir, prefixes)
    if all(v is None for v in data.values()):
        print("No summary.json found under gpu_* subdirs. Run profile_gpu.sh first.")
        return

    out_bar = Path(args.out) if args.out else out_dir / "profile_summary.png"
    out_curves = Path(args.curves) if args.curves else out_dir / "profile_curves.png"
    plot_summary_bars(data, out_bar, mode="gpu")
    plot_curves(data, out_curves)


if __name__ == "__main__":
    main()
