"""
PPA 报告：端到端 latency/energy、按层/算子/模块的 breakdown，输出 JSON 与图表。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 算子类型在 CIM 层描述中的键：conv / fc
OP_TYPE_CONV = "conv"
OP_TYPE_FC = "fc"


def _json_safe_activation_shapes(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    safe: List[Dict[str, Any]] = []
    for item in items or []:
        if item is None:
            continue
        if isinstance(item, dict):
            safe.append(item)
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            safe.append(to_dict())
            continue
        safe.append({
            "repr": str(item),
        })
    return safe


def build_ppa_report(
    eval_result: Any,
    clock_ns: float,
    netstruct: Optional[List[Any]] = None,
    conv_linear_order: Optional[List[Tuple[str, int]]] = None,
    cim_layer_latency_ns: Optional[List[float]] = None,
    cim_layer_energy_nj: Optional[List[float]] = None,
    full_operator_breakdown: Optional[Dict[str, Dict[str, float]]] = None,
    checkpoint: str = "",
    neurosim_full_network: bool = False,
    rram_tile_stats: Optional[Dict[str, Any]] = None,
    operator_sequence: Optional[List[Dict[str, Any]]] = None,
    latency_per_sample_ns: Optional[float] = None,
    total_area_um2: Optional[float] = None,
    event_breakdown: Optional[Dict[str, Any]] = None,
    activation_shapes: Optional[List[Dict[str, Any]]] = None,
    neurosim_summary: Optional[Dict[str, Any]] = None,
    neurosim_layer_breakdown: Optional[List[Dict[str, Any]]] = None,
    pointcim_distance_summary: Optional[Dict[str, Any]] = None,
    dist_chain_analysis: Optional[Dict[str, Any]] = None,
    critical_path_latency_per_sample_ns: Optional[float] = None,
) -> Dict[str, Any]:
    """
    构建包含端到端 PPA、按层/算子/模块 breakdown 的报告字典。
    full_operator_breakdown: 仿真事件按算子类型聚合的 cycles/energy_nj/latency_ns（含 FPS/KNN/BN/Conv/Linear 等）。
    rram_tile_stats: 可选，来自 NeuroSim（或兼容字段）的 RRAM 阵列/PE/Tile 数量与尺寸统计，写入 report["rram_tile_stats"]。
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
    if critical_path_latency_per_sample_ns is not None and critical_path_latency_per_sample_ns > 0:
        critical_path_latency_ns = critical_path_latency_per_sample_ns
    else:
        critical_path_latency_ns = cycles_per_sample * clock_ns
    critical_path_latency_ms = critical_path_latency_ns / 1e6
    busy_time_sum_ns = 0.0
    if full_operator_breakdown:
        busy_time_sum_ns = float(sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values()) or 0.0)
    unattributed_gap_stall_ns = max(0.0, float(critical_path_latency_ns) - float(busy_time_sum_ns))
    busy_time_sum_ms = busy_time_sum_ns / 1e6
    unattributed_gap_stall_ms = unattributed_gap_stall_ns / 1e6

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
            "latency_sum_ns": end_to_end_latency_ns,
            "latency_sum_ms": end_to_end_latency_ms,
            "latency_critical_path_ns": critical_path_latency_ns,
            "latency_critical_path_ms": critical_path_latency_ms,
            "busy_time_sum_ns": busy_time_sum_ns,
            "busy_time_sum_ms": busy_time_sum_ms,
            "unattributed_gap_stall_ns": unattributed_gap_stall_ns,
            "unattributed_gap_stall_ms": unattributed_gap_stall_ms,
            "clock_period_ns": clock_ns,
            "total_area_um2": float(total_area_um2) if total_area_um2 is not None else None,
            "modeling_note": (
                "latency_sum_* is the sum of event duration * event clock period; "
                "latency_critical_path_* is scheduler makespan (cycles_per_sample * base clock); "
                "busy_time_sum_* is the sum of merged per-operator busy intervals; "
                "unattributed_gap_stall_* = critical_path - busy_time_sum."
            ),
        },
        "checkpoint": checkpoint,
        "neurosim_full_network": neurosim_full_network,
    }

    # RRAM 阵列 / PE / Tile 统计
    if rram_tile_stats:
        report["rram_tile_stats"] = rram_tile_stats

    if neurosim_summary:
        report["neurosim_summary"] = neurosim_summary
    if neurosim_layer_breakdown:
        report["neurosim_layer_breakdown"] = neurosim_layer_breakdown
    if pointcim_distance_summary:
        report["pointcim_distance_summary"] = pointcim_distance_summary
    if dist_chain_analysis:
        report["dist_chain_analysis"] = dist_chain_analysis

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
            display_name = "SRAM" if op_name == "LayerBuffer" else op_name
            cy = v.get("cycles", 0)
            en = v.get("energy_nj", 0)
            lat_ns = v.get("latency_ns", 0)
            area_u = v.get("area_um2", 0)
            full_list.append({
                "operator": display_name,
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
            "total_busy_time_ns": total_lat_ns,
            "critical_path_latency_ns": critical_path_latency_ns,
            "unattributed_gap_stall_ns": max(0.0, float(critical_path_latency_ns) - float(total_lat_ns)),
            "total_area_um2": total_area,
        }

    if event_breakdown:
        report["breakdown"] = event_breakdown
        by_module = event_breakdown.get("by_module") or {}
        if by_module:
            total_cy = sum(v.get("cycles", 0) for v in by_module.values())
            total_en = sum(v.get("energy_nj", 0) for v in by_module.values())
            total_area = sum(v.get("area_um2", 0) for v in by_module.values())
            full_module_list: List[Dict[str, Any]] = []
            for module_name, v in sorted(by_module.items(), key=lambda x: -x[1].get("cycles", 0)):
                cy = v.get("cycles", 0)
                en = v.get("energy_nj", 0)
                area_u = v.get("area_um2", 0)
                full_module_list.append({
                    "module": module_name,
                    "cycles": round(cy, 2),
                    "energy_nj": round(en, 2),
                    "area_um2": round(area_u, 2),
                    "cycles_pct": round(cy / total_cy * 100, 2) if total_cy > 0 else 0,
                    "energy_pct": round(en / total_en * 100, 2) if total_en > 0 else 0,
                    "area_pct": round(area_u / total_area * 100, 2) if total_area > 0 else 0,
                    "event_count": int(v.get("event_count", 0) or 0),
                })
            report["full_module_breakdown"] = full_module_list
            report["full_module_totals"] = {
                "total_cycles": total_cy,
                "total_energy_nj": total_en,
                "total_area_um2": total_area,
            }
    if activation_shapes:
        report["activation_shapes"] = _json_safe_activation_shapes(activation_shapes)

    if cim_layer_latency_ns is not None:
        report["cim_layer_latency_ns"] = cim_layer_latency_ns
    if cim_layer_energy_nj is not None:
        report["cim_layer_energy_nj"] = cim_layer_energy_nj

    return report


def plot_ppa_breakdown(report: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    根据 report 绘制全算子与全模块 latency/energy 饼图，保存到 output_dir。
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

    def _pie_with_right_legend(
        values: List[float],
        labels: List[str],
        title: str,
        value_key: str,
        out_name: str,
    ) -> Optional[Path]:
        total = float(sum(values))
        if total <= 0 or not labels:
            return None
        cmap = cm.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(labels))]
        fig, ax = plt.subplots(figsize=(11, 6))
        wedges, _ = ax.pie(
            values,
            labels=None,
            startangle=90,
            colors=colors,
        )
        legend_labels = [
            f"{label} ({(value / total) * 100:.2f}%)"
            for label, value in zip(labels, values)
        ]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            title=value_key,
        )
        ax.set_title(title)
        fig.tight_layout()
        out_path = output_dir / out_name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # 1) 全算子 Latency 饼图（latency_ns），所有算子单独显示，不合并
    if report.get("full_operator_breakdown"):
        full = report["full_operator_breakdown"]
        main_lat_ns = [x["latency_ns"] for x in full if x.get("latency_ns", 0) > 0]
        main_labels = [x["operator"] for x in full if x.get("latency_ns", 0) > 0]
        p = _pie_with_right_legend(
            main_lat_ns,
            main_labels,
            "Latency Breakdown by Operator",
            "Latency share",
            "latency_breakdown_by_operator.png",
        )
        if p is not None:
            saved.append(p)

    # 2) 全算子 Energy 饼图，所有 energy_nj>0 的算子单独显示
    if report.get("full_operator_breakdown"):
        full = report["full_operator_breakdown"]
        main_en = [x["energy_nj"] for x in full if x["energy_nj"] > 0]
        main_labels = [x["operator"] for x in full if x["energy_nj"] > 0]
        p = _pie_with_right_legend(
            main_en,
            main_labels,
            "Energy Breakdown by Operator",
            "Energy share",
            "energy_breakdown_by_operator.png",
        )
        if p is not None:
            saved.append(p)

    # 3) 全模块 Latency 饼图（cycles）
    if report.get("full_module_breakdown"):
        full = report["full_module_breakdown"]
        main_cycles = [x["cycles"] for x in full if x.get("cycles", 0) > 0]
        main_labels = [x["module"] for x in full if x.get("cycles", 0) > 0]
        p = _pie_with_right_legend(
            main_cycles,
            main_labels,
            "Latency Breakdown by Module",
            "Latency share",
            "latency_breakdown_by_module.png",
        )
        if p is not None:
            saved.append(p)

    # 4) 全模块 Energy 饼图
    if report.get("full_module_breakdown"):
        full = report["full_module_breakdown"]
        main_en = [x["energy_nj"] for x in full if x.get("energy_nj", 0) > 0]
        main_labels = [x["module"] for x in full if x.get("energy_nj", 0) > 0]
        p = _pie_with_right_legend(
            main_en,
            main_labels,
            "Energy Breakdown by Module",
            "Energy share",
            "energy_breakdown_by_module.png",
        )
        if p is not None:
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
