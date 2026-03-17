#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 SRAM size sweep 各子目录的 report.json 汇总能耗，输出 CSV 并绘制 DRAM 能耗占比随 SRAM 大小的散点图。
支持 feature（层间 SRAM）与 systolic（脉动阵列 SRAM）两种扫描类型。

用法（在 PointKAN-pytorch 根目录）:
  python scripts/aggregate_sram_sweep_and_plot.py
  python scripts/aggregate_sram_sweep_and_plot.py --sweep_dir sim_results_systolic_sram_sweep --sweep_type systolic
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_sram_bytes_from_dirname(dir_name: str, sweep_type: str) -> int | None:
    """从目录名解析出字节数。feature: feature_sram_524288B，systolic: systolic_sram_32768B。"""
    if sweep_type == "systolic":
        m = re.match(r"systolic_sram_(\d+)B", dir_name)
    else:
        m = re.match(r"feature_sram_(\d+)B", dir_name)
    return int(m.group(1)) if m else None


def aggregate_one_report(report_path: Path) -> dict | None:
    """从单个 report.json 汇总 total/dram/sram 能耗及占比，返回一行数据字典。"""
    if not report_path.is_file():
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    end_to_end = report.get("end_to_end") or {}
    total_energy_nj = float(end_to_end.get("total_energy_nj") or 0.0)
    latency_ns = float(end_to_end.get("latency_ns") or 0.0)
    latency_ms = latency_ns / 1e6

    dram_energy_nj = 0.0
    sram_energy_nj = 0.0
    for item in report.get("operator_sequence") or []:
        mod = (item.get("circuit_module") or "").strip().lower()
        en = float(item.get("energy_nj") or 0.0)
        if mod == "dram":
            dram_energy_nj += en
        elif mod in ("inter_layer_buffer", "sram_ii", "layer_buffer"):
            sram_energy_nj += en

    if total_energy_nj <= 0:
        dram_energy_pct = 0.0
        sram_energy_pct = 0.0
    else:
        dram_energy_pct = 100.0 * dram_energy_nj / total_energy_nj
        sram_energy_pct = 100.0 * sram_energy_nj / total_energy_nj

    return {
        "total_energy_nj": total_energy_nj,
        "dram_energy_nj": dram_energy_nj,
        "sram_energy_nj": sram_energy_nj,
        "dram_energy_pct": dram_energy_pct,
        "sram_energy_pct": sram_energy_pct,
        "latency_ns": latency_ns,
        "latency_ms": latency_ms,
    }


def run(sweep_dir: Path, out_csv: Path, out_plot: Path | None, sweep_type: str = "feature") -> None:
    sweep_dir = Path(sweep_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if sweep_type not in ("feature", "systolic"):
        raise ValueError("sweep_type 必须是 feature 或 systolic")

    subdir_pattern = "systolic_sram_*B" if sweep_type == "systolic" else "feature_sram_*B"
    rows = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        sram_bytes = _parse_sram_bytes_from_dirname(sub.name, sweep_type)
        if sram_bytes is None:
            continue
        report_path = sub / "report.json"
        agg = aggregate_one_report(report_path)
        if agg is None:
            continue
        rows.append({
            "sram_size_bytes": sram_bytes,
            "sram_size_kb": sram_bytes / 1024.0,
            **agg,
        })

    if not rows:
        print(f"未在 {sweep_dir} 下找到可用的 {subdir_pattern}/report.json，请先运行 run_sram_size_sweep.sh（SWEEP_TARGET={sweep_type}）")
        return

    rows.sort(key=lambda x: x["sram_size_kb"])

    # 写 CSV
    import csv
    fieldnames = [
        "sram_size_bytes", "sram_size_kb",
        "dram_energy_nj", "sram_energy_nj", "total_energy_nj",
        "dram_energy_pct", "sram_energy_pct",
        "latency_ns", "latency_ms",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"已写入: {out_csv}（共 {len(rows)} 条）")

    # 散点图：SRAM size (KB) vs DRAM 能耗占比 (%)
    xlabel = "Systolic SRAM size (KB)" if sweep_type == "systolic" else "Feature SRAM size (KB)"
    if out_plot is not None:
        out_plot = Path(out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("未安装 matplotlib，跳过绘图。")
            return

        x = [r["sram_size_kb"] for r in rows]
        y_dram_pct = [r["dram_energy_pct"] for r in rows]
        y_sram_pct = [r["sram_energy_pct"] for r in rows]
        y_total = [r["total_energy_nj"] for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax1 = axes[0]
        ax1.scatter(x, y_dram_pct, s=80, c="C0", label="DRAM energy %", zorder=2)
        ax1.scatter(x, y_sram_pct, s=80, c="C1", label="SRAM energy %", zorder=2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Energy share (%)")
        ax1.set_title(f"SRAM size sweep ({sweep_type}): DRAM/SRAM energy share")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        ax2 = axes[1]
        ax2.scatter(x, y_total, s=80, c="green", alpha=0.7, label="Total energy (nJ)", zorder=2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Total energy (nJ)")
        ax2.set_title(f"SRAM size sweep ({sweep_type}): total energy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x)

        fig.tight_layout()
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"散点图已保存: {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="汇总 SRAM sweep 报告并绘制 DRAM 能耗占比散点图")
    parser.add_argument(
        "--sweep_dir",
        type=str,
        default="sim_results_sram_size_sweep",
        help="sweep 结果根目录（含 feature_sram_*B 或 systolic_sram_*B 子目录）",
    )
    parser.add_argument(
        "--sweep_type",
        type=str,
        choices=("feature", "systolic"),
        default="feature",
        help="扫描类型：feature=层间 SRAM，systolic=脉动阵列 SRAM",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出 CSV 路径；默认根据 sweep_type 与 sweep_dir 自动设置",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default=None,
        help="输出散点图路径；默认根据 sweep_type 与 sweep_dir 自动设置；设为空则不绘图",
    )
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)
    if args.out_csv is None:
        args.out_csv = str(sweep_dir / "sram_size_energy_sweep.csv")
    if args.out_plot is None:
        args.out_plot = str(sweep_dir / "sram_size_dram_pct_scatter.png")
    out_plot = args.out_plot if args.out_plot else None
    run(sweep_dir, Path(args.out_csv), Path(out_plot) if out_plot else None, sweep_type=args.sweep_type)


if __name__ == "__main__":
    main()
