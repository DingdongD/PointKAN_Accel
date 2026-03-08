"""
PPA 报告：端到端 latency/energy、按层与按算子类型的 breakdown，输出 JSON 与图表。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 算子类型在 MNSIM 中的键：conv / fc
OP_TYPE_CONV = "conv"
OP_TYPE_FC = "fc"


def build_ppa_report(
    eval_result: Any,
    clock_ns: float,
    netstruct: Optional[List[Any]] = None,
    conv_linear_order: Optional[List[Tuple[str, int]]] = None,
    mnsim_latency_ns: Optional[List[float]] = None,
    mnsim_energy_nj: Optional[List[float]] = None,
    full_operator_breakdown: Optional[Dict[str, Dict[str, float]]] = None,
    checkpoint: str = "",
    mnsim_full: bool = False,
    rram_tile_stats: Optional[Dict[str, Any]] = None,
    operator_sequence: Optional[List[Dict[str, Any]]] = None,
    latency_per_sample_ns: Optional[float] = None,
    total_area_um2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    构建包含端到端 PPA、按层与按算子类型 breakdown 的报告字典。
    full_operator_breakdown: 仿真事件按算子类型聚合的 cycles/energy_nj/latency_ns（含 FPS/KNN/BN/Conv/Linear 等）。
    rram_tile_stats: 可选，来自 MNSIM TCG 的 RRAM 阵列/PE/Tile 数量与尺寸统计，写入 report["rram_tile_stats"]。
    latency_per_sample_ns: 若提供则作为端到端每样本延迟(ns)（各模块 duration×自身 clock_period_ns 累加）；未传则用 cycles_per_sample*clock_ns。
    """
    total_cycles = getattr(eval_result, "total_cycles", 0) or 0
    total_energy_nj = getattr(eval_result, "total_energy_nj", 0.0) or 0.0
    num_samples = getattr(eval_result, "num_samples", 0) or 1

    cycles_per_sample = total_cycles / num_samples if num_samples else total_cycles
    energy_per_sample_nj = total_energy_nj / num_samples if num_samples else total_energy_nj
    if latency_per_sample_ns is not None and latency_per_sample_ns > 0:
        end_to_end_latency_ns = latency_per_sample_ns
    else:
        end_to_end_latency_ns = cycles_per_sample * clock_ns
    end_to_end_latency_ms = end_to_end_latency_ns / 1e6

    report: Dict[str, Any] = {
        "num_samples": num_samples,
        "accuracy": getattr(eval_result, "accuracy", 0.0),
        "balanced_accuracy": getattr(eval_result, "balanced_accuracy", 0.0),
        "test_loss": getattr(eval_result, "test_loss", 0.0),
        "end_to_end": {
            "total_cycles": total_cycles,
            "cycles_per_sample": cycles_per_sample,
            "total_energy_nj": total_energy_nj,
            "energy_per_sample_nj": energy_per_sample_nj,
            "latency_ns": end_to_end_latency_ns,
            "latency_ms": end_to_end_latency_ms,
            "clock_period_ns": clock_ns,
            "total_area_um2": float(total_area_um2) if total_area_um2 is not None else None,
        },
        "checkpoint": checkpoint,
        "mnsim_full": mnsim_full,
    }

    # RRAM 阵列 / PE / Tile 统计（来自 MNSIM TCG）
    if rram_tile_stats:
        report["rram_tile_stats"] = rram_tile_stats

    # 按执行顺序的完整硬件算子序列（cycle 排序）
    if operator_sequence is not None:
        report["operator_sequence"] = operator_sequence

    # 全算子 breakdown（FPS/KNN/BN/Conv/Linear/ReLU 等），用于饼图；含 latency_ns、area_um2 时一并写入
    if full_operator_breakdown:
        total_cy = sum(v.get("cycles", 0) for v in full_operator_breakdown.values())
        total_en = sum(v.get("energy_nj", 0) for v in full_operator_breakdown.values())
        total_lat_ns = sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values())
        total_area = sum(v.get("area_um2", 0) for v in full_operator_breakdown.values())
        full_list: List[Dict[str, Any]] = []
        for op_name, v in sorted(full_operator_breakdown.items(), key=lambda x: -x[1].get("latency_ns", x[1].get("cycles", 0))):
            cy = v.get("cycles", 0)
            en = v.get("energy_nj", 0)
            lat_ns = v.get("latency_ns", 0)
            area_u = v.get("area_um2", 0)
            full_list.append({
                "operator": op_name,
                "cycles": round(cy, 2),
                "energy_nj": round(en, 2),
                "latency_ns": round(lat_ns, 2),
                "area_um2": round(area_u, 2),
                "cycles_pct": round(cy / total_cy * 100, 2) if total_cy > 0 else 0,
                "energy_pct": round(en / total_en * 100, 2) if total_en > 0 else 0,
                "latency_pct": round(lat_ns / total_lat_ns * 100, 2) if total_lat_ns > 0 else 0,
                "area_pct": round(area_u / total_area * 100, 2) if total_area > 0 else 0,
            })
        report["full_operator_breakdown"] = full_list
        report["full_operator_totals"] = {
            "total_cycles": total_cy,
            "total_energy_nj": total_en,
            "total_latency_ns": total_lat_ns,
            "total_area_um2": total_area,
        }

    return report


def plot_ppa_breakdown(report: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    根据 report 绘制全算子 latency/energy 饼图，保存到 output_dir。
    不合并为 Other，所有算子（FPS/KNN/BN/LayerNorm/Conv/Linear 等）均单独显示。
    返回生成的文件路径列表。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return []

    saved: List[Path] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 全算子 Latency 饼图（cycles），所有算子单独显示，不合并
    if report.get("full_operator_breakdown"):
        full = report["full_operator_breakdown"]
        main_cycles = [x["cycles"] for x in full if x["cycles"] > 0]
        main_labels = [x["operator"] for x in full if x["cycles"] > 0]
        total_cy = sum(main_cycles)
        if total_cy > 0 and main_labels:
            cmap = cm.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(len(main_labels))]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.pie(main_cycles, labels=main_labels, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title("Latency Breakdown by Operator (FPS, KNN, BN, LN, Conv, Linear, ...)")
            fig.tight_layout()
            p = output_dir / "latency_breakdown_by_operator.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

    # 2) 全算子 Energy 饼图，所有 energy_nj>0 的算子单独显示
    if report.get("full_operator_breakdown"):
        full = report["full_operator_breakdown"]
        main_en = [x["energy_nj"] for x in full if x["energy_nj"] > 0]
        main_labels = [x["operator"] for x in full if x["energy_nj"] > 0]
        total_en = sum(main_en)
        if total_en > 0 and main_labels:
            cmap = cm.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(len(main_labels))]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.pie(main_en, labels=main_labels, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title("Energy Breakdown by Operator (FPS, KNN, BN, LN, Conv, Linear, ...)")
            fig.tight_layout()
            p = output_dir / "energy_breakdown_by_operator.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

    return saved


def write_ppa_report_and_plots(
    report: Dict[str, Any],
    output_dir: str | Path,
    report_filename: str = "report.json",
    log_info_fn: Optional[Any] = None,
) -> Tuple[Path, List[Path]]:
    """
    将 report 写入 output_dir/report_filename，并生成 breakdown 图。返回 (report_path, figure_paths)。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if log_info_fn:
        log_info_fn(f"PPA 报告已写入: {report_path}")

    figure_paths = plot_ppa_breakdown(report, output_dir)
    if log_info_fn and figure_paths:
        log_info_fn(f"PPA 图表已写入: {output_dir} ({len(figure_paths)} 张)")
    return report_path, figure_paths
