#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 outputs 下 GPU/CPU 各任务 summary 聚合，绘制「CPU vs GPU」对比图。
仅绘制 model(x) 的推理时间与能耗，不使用进程级 latency_s / wall time。
- Latency：inference_only_per_forward_s_mean（单次前向）或 inference_only_s_mean
- Energy：GPU 为运行期间采样，CPU 为 pyRAPL 仅包推理循环
支持 checkpoint 跑的结果（gpu_*/cpu_*）或随机数据跑的结果（random_gpu_*/random_cpu_*）。

用法：
  python scripts/plot_cpu_gpu_compare.py
  python scripts/plot_cpu_gpu_compare.py --out-dir outputs --source random
  python scripts/plot_cpu_gpu_compare.py --out outputs/cpu_gpu_compare.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT / "outputs"

TASK_NAMES = {
    "modelnet40": "ModelNet40",
    "scanobjectnn": "ScanObjectNN",
    "part_seg": "Part Segmentation",
}
TASK_ORDER = ("modelnet40", "scanobjectnn", "part_seg")


def find_latest_run_dir(base: Path, prefix: str) -> Optional[Path]:
    task_dir = base / prefix
    if not task_dir.is_dir():
        return None
    subdirs = [d for d in task_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[:8].isdigit()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)


def load_aggregate(run_dir: Path) -> Optional[dict]:
    p = run_dir / "summary.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    return s.get("aggregate")


def _infer_lat(agg) -> float:
    """仅 model(x) 推理时间，绝不使用进程 latency_s。"""
    v = (agg.get("inference_only_per_forward_s_mean") or agg.get("inference_only_s_mean")) if agg else None
    return float(v) if v is not None else 0.0


def collect_cpu_gpu_pairs(outputs_dir: Path, prefix_gpu: str, prefix_cpu: str) -> dict:
    """
    对每个 task 分别取 GPU 目录与 CPU 目录下最新 run 的 aggregate，
    只使用 model(x) 推理时间与能耗，不使用进程级 latency_s。
    返回 { task: { "latency_gpu", "latency_cpu", "energy_gpu", "energy_cpu", ... } }
    """
    result = {}
    for task in TASK_ORDER:
        gpu_dir = find_latest_run_dir(outputs_dir, prefix_gpu + task)
        cpu_dir = find_latest_run_dir(outputs_dir, prefix_cpu + task)
        agg_gpu = load_aggregate(gpu_dir) if gpu_dir else None
        agg_cpu = load_aggregate(cpu_dir) if cpu_dir else None
        latency_gpu = _infer_lat(agg_gpu)
        latency_cpu = _infer_lat(agg_cpu)
        energy_gpu = (agg_gpu.get("gpu_energy_j_mean") if agg_gpu else None) or 0.0
        # 仅 model 推理阶段 CPU 能耗（pyRAPL），不使用进程级 RAPL
        energy_cpu = None
        if agg_cpu:
            energy_cpu = (
                agg_cpu.get("cpu_energy_inference_total_j_mean")
                or agg_cpu.get("cpu_energy_inference_pkg_j_mean")
            )
        energy_cpu = energy_cpu if energy_cpu is not None else 0.0
        result[task] = {
            "latency_gpu": latency_gpu,
            "latency_cpu": latency_cpu,
            "energy_gpu": energy_gpu,
            "energy_cpu": energy_cpu,
        }
    return result


def plot_cpu_gpu_compare(data: dict, out_path: Path, title_suffix: str = "") -> None:
    """绘制一张图：左 Latency (CPU vs GPU)，右 Energy (CPU vs GPU)，三任务并列。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    tasks = [TASK_NAMES.get(t, t) for t in TASK_ORDER]
    x = np.arange(len(tasks))
    width = 0.35

    latency_cpu = [data[t]["latency_cpu"] for t in TASK_ORDER]
    latency_gpu = [data[t]["latency_gpu"] for t in TASK_ORDER]
    energy_cpu = [data[t]["energy_cpu"] for t in TASK_ORDER]
    energy_gpu = [data[t]["energy_gpu"] for t in TASK_ORDER]

    fig, (ax_lat, ax_eng) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars1 = ax_lat.bar(x - width / 2, latency_cpu, width, label="CPU", color="#3498db")
    bars2 = ax_lat.bar(x + width / 2, latency_gpu, width, label="GPU", color="#e74c3c")
    ax_lat.set_ylabel("Time (s)")
    ax_lat.set_title("model(x) only, single forward (batch=1): CPU vs GPU")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(tasks, rotation=15, ha="right")
    ax_lat.legend()
    ax_lat.grid(axis="y", alpha=0.3)

    bars3 = ax_eng.bar(x - width / 2, energy_cpu, width, label="CPU", color="#3498db")
    bars4 = ax_eng.bar(x + width / 2, energy_gpu, width, label="GPU", color="#e74c3c")
    ax_eng.set_ylabel("Energy (J)")
    ax_eng.set_title("Energy: CPU vs GPU")
    ax_eng.set_xticks(x)
    ax_eng.set_xticklabels(tasks, rotation=15, ha="right")
    ax_eng.legend()
    ax_eng.grid(axis="y", alpha=0.3)

    suptitle = "PointKAN: model(x) inference time & Energy (CPU vs GPU)"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot CPU vs GPU latency and energy comparison")
    ap.add_argument("--out-dir", type=str, default=str(OUTPUTS_DIR), help="outputs 根目录")
    ap.add_argument(
        "--source",
        type=str,
        choices=["random", "checkpoint"],
        default="random",
        help="数据来源：random=random_gpu_*/random_cpu_*，checkpoint=gpu_*/cpu_*",
    )
    ap.add_argument("--out", type=str, default=None, help="输出图片路径，默认 out-dir/cpu_gpu_compare.png")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "random":
        prefix_gpu, prefix_cpu = "random_gpu_", "random_cpu_"
        title_suffix = "random data"
    else:
        prefix_gpu, prefix_cpu = "gpu_", "cpu_"
        title_suffix = "checkpoint"

    data = collect_cpu_gpu_pairs(out_dir, prefix_gpu, prefix_cpu)
    if all(data[t]["latency_gpu"] == 0 and data[t]["latency_cpu"] == 0 for t in TASK_ORDER):
        print(f"No summary.json found under {prefix_gpu}*/ and {prefix_cpu}*/. Run profile_random_all.sh or profile_all.sh first.")
        return

    out_path = Path(args.out) if args.out else out_dir / "cpu_gpu_compare.png"
    plot_cpu_gpu_compare(data, out_path, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
