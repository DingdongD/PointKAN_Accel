"""
PointKAN CIM PPA 仿真器 - 主入口
================================

在 CIM（存算一体）硬件仿真器上对 PointKAN / PointKAN-Elite 进行
端到端 PPA（性能-功耗-面积）评估，支持 baseline / acc 等加速器配置。

使用方式（建议在工程根目录 PointKAN-pytorch 下执行）:
    python -m pointkan_sim.main --help
    python -m pointkan_sim.main --demo
    python -m pointkan_sim.main --network pointkan --accelerator baseline
    python -m pointkan_sim.main --network pointkan-elite --accelerator acc
    python -m pointkan_sim.main --output report.json

仿真日志、报告、时间线 CSV 等统一写入 --sim_log_dir 目录（默认 ./sim_log）。
使用真实 checkpoint 与 ModelNet40 配置评估：python -m pointkan_sim.run_sim_with_real_eval --run_real_inference --output report.json
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 仿真日志与报告统一输出目录（相对当前工作目录）
SIM_LOG_DIR = "sim_log"

# -----------------------------------------------------------------------------
# 路径修正（仅当被直接以脚本方式运行且未作为包导入时执行）
# 目的：将上级目录加入 sys.path，使 import pointkan_sim 能成功。
# 若使用 python -m pointkan_sim.main 启动则 __package__ 非空，此处不会执行。
# -----------------------------------------------------------------------------
if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from pointkan_sim.configs.hw_config import (
    get_accelerator_config,
    list_accelerator_configs,
    FeatureComputingConfig,
    HardwareConfig,
    set_digital_clock_freq_hz,
    set_logic_parallelism,
)
from pointkan_sim.network.pointkan_net import (
    LayerDef,
    OperatorRef,
    create_pointkan_classification,
)
from pointkan_sim.network.pointkan_elite_net import (
    create_pointkan_elite_classification,
    create_pointkan_elite_scanobjectnn_classification,
    create_pointkan_elite_partseg,
)
from pointkan_sim.network.pointnet2_net import (
    create_pointnet2_cls_ssg,
    create_pointnet2_partseg_ssg,
    create_pointnet2_sem_ssg,
)
from pointkan_sim.network.activation_shape import (
    compute_activation_shapes,
    ActivationShapeInfo,
    format_activation_shape,
)
from pointkan_sim.operators.dispatcher import (
    dispatch_layer,
    list_supported_op_types,
)
from pointkan_sim.operators.layer_buffer_ops import (
    gen_first_layer_input_load_events,
    gen_layer_buffer_read_events,
    gen_layer_buffer_write_events,
)
from pointkan_sim.integration.cim_full_context import (
    ACC_CIM_KEY_PRELOADED,
    CIM_LAYER_INDEX,
    build_cim_full_context_from_neurosim_lists,
)
from pointkan_sim.simulator.scheduler import Scheduler
from pointkan_sim.report.reporter import Reporter, create_ppa_result
from pointkan_sim.analysis.breakdown import analyze_simulation_events_breakdown
from pointkan_sim.analysis.timeline import (
    build_module_activity_timeline,
    build_module_activity_segments,
    build_block_intervals,
    block_intervals_to_json_rows,
    plot_timeline,
    timeline_to_csv_rows,
)
from pointkan_sim.memory.transactions import (
    extract_memory_transactions,
    summarize_memory_transactions,
)
from pointkan_sim.utils.logger import (
    setup_logger,
    log_info,
    log_section,
    log_separator,
    LogLevel,
)
from pointkan_sim.utils.shared_helpers import (
    sanitize_log_tag as _sanitize_log_tag,
    resolve_neurosim_sar_column_power_w,
)
from pointkan_sim.constants import (
    KEY_IDX,
    KEY_CYCLE,
    KEY_END_CYCLE,
    KEY_DURATION,
    KEY_OPERATOR,
    KEY_STAGE,
    KEY_OPERATION,
    KEY_LAYER_NAME,
    KEY_ENERGY_NJ,
    KEY_CIRCUIT_MODULE,
)

POINTCIM_DISTANCE_OPS = {
    "xyz_refill",
    "xyz_quantize",
    "key_xyz_buffer_write",
    "key_xyz_buffer_read",
    "rram_program_keyset",
    "rram_self_k2_init",
    "self_distance_write",
    "xyz_buffer_query_read",
    "query_xyz_buffer_write",
    "query_xyz_buffer_read",
    "query_self_q",
    "rram_cross_term_distance",
    "self_distance_read",
    "distance_merge_accumulate",
    "distance_write",
}


def _display_operator_name(event: Any) -> str:
    """Map an event to the operator label used in reports and operator detail tables."""
    stage = str(getattr(event, "stage", "") or "").strip()
    if stage == "quant":
        return "Quant"
    if stage == "dequant":
        return "Dequant"
    if stage in ("dist_reuse_dram_read", "dist_matrix_dram_write", "ns_cycle_dist_reuse_read", "ns_cycle_centroid_load"):
        return "DRAM"
    op_name = str(getattr(event, "operator", "") or getattr(event, "operation", "") or "other").strip() or "other"
    if op_name == "LayerBuffer":
        return "SRAM"
    return op_name


def _summarize_operator_timeline_stats(events: List[Any]) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Aggregate operator stats using merged non-overlapped intervals per (operator, layer_name)
    instead of sum(duration).

    This avoids double-counting parallel PointCIM array events inside a single query window while
    also avoiding over-counting idle gaps between disjoint executions of the same operator/layer.
    Energy still sums across all events, while cycles/latency reflect merged busy time on
    non-overlapped events.
    """
    operator_breakdown: Dict[str, Dict[str, float]] = {}
    op_layer_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    op_layer_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    op_layer_clock_ns: Dict[Tuple[str, str], float] = {}

    metadata_keys = (
        "num_points",
        "in_features",
        "out_features",
        "activation_input_shape",
        "activation_output_shape",
        "im2col_input_read_bytes",
        "im2col_patch_write_bytes",
        "im2col_matrix_read_bytes",
        "im2col_ifmap_source_tensor_bytes",
        "im2col_ifmap_materialized_bytes",
        "im2col_total_cycles",
        "scalesim_ifmap_effective_bytes",
    )

    for event in events:
        op_name = _display_operator_name(event)
        clk_ns = float(getattr(event, "clock_period_ns", 1.0) or 1.0)
        meta = getattr(event, "metadata", None) or {}
        layer_name = str(meta.get("layer_name", "") or "")
        key = (op_name, layer_name)

        bucket = operator_breakdown.setdefault(
            op_name,
            {"cycles": 0.0, "energy_nj": 0.0, "latency_ns": 0.0, "area_um2": 0.0},
        )
        bucket["energy_nj"] += float(getattr(event, "energy", 0.0) or 0.0)

        area_um2 = meta.get("area_um2")
        if isinstance(area_um2, (int, float)):
            bucket["area_um2"] += float(area_um2)

        row = op_layer_rows.setdefault(
            key,
            {
                "operator": op_name,
                "layer_name": layer_name,
                "cycles": 0,
                "duration": 0,
                "latency_ns": 0.0,
                "energy_nj": 0.0,
                "area_um2": 0.0,
                "event_count": 0,
                "overlapped": False,
            },
        )
        row["energy_nj"] += float(getattr(event, "energy", 0.0) or 0.0)
        row["event_count"] += 1
        if isinstance(area_um2, (int, float)):
            row["area_um2"] += float(area_um2)
        for mk in metadata_keys:
            if row.get(mk) in (None, "", 0) and meta.get(mk) is not None:
                row[mk] = meta.get(mk)

        if not getattr(event, "overlapped", False):
            ec = int(getattr(event, "cycle", 0) or 0)
            ee = int(getattr(event, "end_cycle", ec + int(getattr(event, "duration", 1) or 1) - 1))
            op_layer_intervals.setdefault(key, []).append((ec, ee))
            op_layer_clock_ns.setdefault(key, clk_ns)

    for key, intervals in op_layer_intervals.items():
        op_name, _layer_name = key
        clk_ns = float(op_layer_clock_ns.get(key, 1.0) or 1.0)
        merged: List[List[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        cycles = int(sum(end - start + 1 for start, end in merged))
        latency_ns = cycles * float(clk_ns)
        operator_breakdown[op_name]["cycles"] += cycles
        operator_breakdown[op_name]["latency_ns"] += latency_ns
        op_layer_rows[key]["cycles"] = cycles
        op_layer_rows[key]["duration"] = cycles
        op_layer_rows[key]["latency_ns"] = latency_ns

    operator_cycle_detail = sorted(
        op_layer_rows.values(),
        key=lambda row: (-int(row.get("cycles", 0) or 0), str(row.get("operator", "")), str(row.get("layer_name", ""))),
    )
    return operator_breakdown, operator_cycle_detail


def _summarize_pointcim_distance_events(events: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Summarise PointCIM RRAM distance-chain events.

    - **Energy** sums every distance-chain op (including overlapped / background work).
    - **Latency** reports two views:
        * ``non_overlapped_*``: serial critical-path style (excludes overlapped=True).
        * ``timeline_span_*``: wall span min(start_time)→max(end_time) in ns using each
          event's ``cycle`` × ``clock_period_ns`` (single global timeline assumption).
    """
    all_chain = [
        e for e in events
        if str(getattr(e, "operation", "") or "") in POINTCIM_DISTANCE_OPS
    ]
    if not all_chain:
        return None

    serial = [e for e in all_chain if not getattr(e, "overlapped", False)]
    if not serial:
        serial = list(all_chain)

    total_energy_nj = sum(float(getattr(e, "energy", 0.0) or 0.0) for e in all_chain)

    overall_start = min(int(e.cycle) for e in serial)
    overall_end = max(int(e.end_cycle) for e in serial)
    overall_clk_ns = max(float(getattr(e, "clock_period_ns", 1.0) or 1.0) for e in serial)

    def _start_ns(evt: Any) -> float:
        return int(evt.cycle) * float(getattr(evt, "clock_period_ns", 1.0) or 1.0)

    def _end_ns(evt: Any) -> float:
        return int(evt.end_cycle) * float(getattr(evt, "clock_period_ns", 1.0) or 1.0)

    timeline_span_ns = max(_end_ns(e) for e in all_chain) - min(_start_ns(e) for e in all_chain)

    relevant = serial

    layers: Dict[str, Dict[str, Any]] = {}
    for e in relevant:
        meta = getattr(e, "metadata", None) or {}
        layer_name = str(meta.get("layer_name", "") or "")
        layer_key = layer_name or "__global__"
        op_name = str(getattr(e, "operation", "") or "")
        clk_ns = float(getattr(e, "clock_period_ns", 1.0) or 1.0)
        bucket = layers.setdefault(
            layer_key,
            {
                "layer_name": layer_name,
                "start_cycle": int(e.cycle),
                "end_cycle": int(e.end_cycle),
                "clock_period_ns": clk_ns,
                "energy_nj": 0.0,
                "event_count": 0,
                "query_indices": set(),
                "array_indices": set(),
                "operations": {},
            },
        )
        bucket["start_cycle"] = min(bucket["start_cycle"], int(e.cycle))
        bucket["end_cycle"] = max(bucket["end_cycle"], int(e.end_cycle))
        bucket["energy_nj"] += float(getattr(e, "energy", 0.0) or 0.0)
        bucket["event_count"] += 1
        if meta.get("query_index") is not None:
            bucket["query_indices"].add(int(meta["query_index"]))
        if meta.get("array_idx") is not None:
            bucket["array_indices"].add(int(meta["array_idx"]))
        op_bucket = bucket["operations"].setdefault(
            op_name,
            {
                "start_cycle": int(e.cycle),
                "end_cycle": int(e.end_cycle),
                "clock_period_ns": clk_ns,
                "energy_nj": 0.0,
                "event_count": 0,
            },
        )
        op_bucket["start_cycle"] = min(op_bucket["start_cycle"], int(e.cycle))
        op_bucket["end_cycle"] = max(op_bucket["end_cycle"], int(e.end_cycle))
        op_bucket["energy_nj"] += float(getattr(e, "energy", 0.0) or 0.0)
        op_bucket["event_count"] += 1

    interface_breakdown: Dict[str, Dict[str, Any]] = {}
    for e in all_chain:
        meta = getattr(e, "metadata", None) or {}
        iface = str(meta.get("sim_interface", "unknown") or "unknown")
        clk_ns = float(getattr(e, "clock_period_ns", 1.0) or 1.0)
        bucket = interface_breakdown.setdefault(
            iface,
            {
                "energy_nj": 0.0,
                "event_count": 0,
                "duration_cycles_sum": 0,
                "duration_ns_sum": 0.0,
            },
        )
        dur = int(getattr(e, "duration", 0) or 0)
        bucket["energy_nj"] += float(getattr(e, "energy", 0.0) or 0.0)
        bucket["event_count"] += 1
        bucket["duration_cycles_sum"] += dur
        bucket["duration_ns_sum"] += dur * clk_ns

    per_layer: List[Dict[str, Any]] = []
    sum_layer_makespan_ns = 0.0
    for _layer_key, bucket in sorted(layers.items(), key=lambda kv: kv[1]["start_cycle"]):
        makespan_cycles = int(bucket["end_cycle"]) - int(bucket["start_cycle"]) + 1
        makespan_ns = makespan_cycles * float(bucket["clock_period_ns"])
        sum_layer_makespan_ns += makespan_ns
        op_rows: List[Dict[str, Any]] = []
        for op_name, op_bucket in sorted(bucket["operations"].items(), key=lambda kv: kv[1]["start_cycle"]):
            op_cycles = int(op_bucket["end_cycle"]) - int(op_bucket["start_cycle"]) + 1
            op_rows.append(
                {
                    "operation": op_name,
                    "makespan_cycles": op_cycles,
                    "makespan_ns": op_cycles * float(op_bucket["clock_period_ns"]),
                    "energy_nj": float(op_bucket["energy_nj"]),
                    "event_count": int(op_bucket["event_count"]),
                }
            )
        per_layer.append(
            {
                "layer_name": bucket["layer_name"],
                "start_cycle": int(bucket["start_cycle"]),
                "end_cycle": int(bucket["end_cycle"]),
                "makespan_cycles": makespan_cycles,
                "makespan_ns": makespan_ns,
                "energy_nj": float(bucket["energy_nj"]),
                "event_count": int(bucket["event_count"]),
                "query_count": len(bucket["query_indices"]),
                "array_count": len(bucket["array_indices"]),
                "operation_breakdown": op_rows,
            }
        )

    overall_makespan_cycles = overall_end - overall_start + 1
    return {
        "modeling_note": (
            "PointCIM distance summary: total_energy_nj includes all distance-chain ops "
            "(overlapped + non-overlapped). non_overlapped_makespan_* is the serial-style "
            "makespan on events marked overlapped=False. timeline_span_ns uses every op on "
            "the unified cycle axis (cycle × clock_period_ns) for wall-span under that clock model."
        ),
        "non_overlapped_makespan_cycles": overall_makespan_cycles,
        "non_overlapped_makespan_ns": overall_makespan_cycles * overall_clk_ns,
        "timeline_span_ns": float(timeline_span_ns),
        "sum_layer_makespan_ns": sum_layer_makespan_ns,
        "total_energy_nj": float(total_energy_nj),
        "event_count_non_overlapped": len(relevant),
        "event_count_all_ops": len(all_chain),
        "interface_breakdown": interface_breakdown,
        # Legacy: total count of distance-chain ops (was non-overlapped-only before).
        "event_count": len(all_chain),
        "layers": per_layer,
        # Back-compat keys (same as former overall_*)
        "overall_makespan_cycles": overall_makespan_cycles,
        "overall_makespan_ns": overall_makespan_cycles * overall_clk_ns,
    }


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        解析后的命名空间，包含 demo / network / accelerator / output 等字段。
    """
    parser = argparse.ArgumentParser(
        description="PointKAN CIM PPA 仿真器（点云存算一体性能/功耗评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m pointkan_sim.main --demo
  python -m pointkan_sim.main --network pointkan --accelerator baseline
  python -m pointkan_sim.main --network pointkan-elite --accelerator acc --output report.json
  python -m pointkan_sim.main --list_accelerators
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示（PointKAN + baseline 加速器）",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        choices=[
            "pointkan",
            "pointkan-elite",
            "pointkan-elite-scanobj",
            "pointkan-elite-partseg",
            "pointnet2-cls-ssg",
            "pointnet2-partseg-ssg",
            "pointnet2-sem-ssg",
        ],
        help=(
            "要仿真的网络; "
            "pointkan-elite-scanobj: ScanObjectNN 分类(1024pts,15类); "
            "pointkan-elite-partseg: ShapeNetPart 零件分割(2048pts,50类)"
        ),
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=None,
        metavar="C",
        help="每点输入特征维度；仅 pointnet2-sem-ssg 需要指定（默认 12）",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="baseline",
        help="加速器配置名称（默认: baseline）",
    )
    parser.add_argument(
        "--list_accelerators",
        action="store_true",
        help="列出可用加速器配置后退出",
    )
    parser.add_argument(
        "--input_points",
        type=int,
        default=1024,
        help="输入点云点数（默认: 1024）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=40,
        help="分类类别数（默认: 40）",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )
    parser.add_argument(
        "--show_events",
        action="store_true",
        help="打印详细事件统计（按算子/阶段等）",
    )
    parser.add_argument(
        "--show_layers",
        action="store_true",
        help="打印每层周期与事件数",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出路径，按扩展名自动选择 JSON 或 CSV",
    )
    parser.add_argument(
        "--cim_backend",
        type=str,
        default=None,
        choices=["neurosim", "fallback"],
        help="Conv/Linear 单层估算后端：neurosim（DNN_NeuroSim）、fallback；无全网络注入时生效",
    )
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default=None,
        help="NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    )
    parser.add_argument(
        "--neurosim_adc_clk_freq",
        type=float,
        default=None,
        metavar="HZ",
        help="NeuroSim SAR ADC 专用时钟频率(Hz)；仅 NeuroSim 模式生效",
    )
    parser.add_argument(
        "--neurosim_sar_column_power_w",
        type=float,
        default=None,
        metavar="W",
        help="NeuroSim SAR ADC Column_Power（W）；仅 NeuroSim 模式生效",
    )
    parser.add_argument(
        "--neurosim_sar_column_power_uw",
        type=float,
        default=None,
        metavar="UW",
        help="NeuroSim SAR ADC Column_Power（uW）；与 --neurosim_sar_column_power_w 二选一",
    )
    parser.add_argument(
        "--neurosim_max_in_vectors",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim 全网络每层输入向量上限；不设=完整 workload，不做缩放",
    )
    parser.add_argument(
        "--neurosim_num_col_muxed",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim numColMuxed（列共享 ADC）；经 main 可选 argv 传入；不设则沿用配置/NeuroSim 默认",
    )
    parser.add_argument(
        "--no_layer_buffer",
        action="store_true",
        help="禁用层间 buffer 读写（仅仿真计算单元，与旧行为一致）",
    )
    parser.add_argument(
        "--show_shapes",
        action="store_true",
        help="打印每层激活 shape（input_shape → output_shape）与数据量",
    )
    parser.add_argument(
        "--output_timeline_csv",
        type=str,
        default=None,
        help="输出「每 cycle 下工作的模块」时间线到 CSV（可用 --timeline_stride 采样）",
    )
    parser.add_argument(
        "--timeline_stride",
        type=int,
        default=500,
        help="时间线采样步长（cycle），默认 500；仅与 --output_timeline_csv 配合",
    )
    parser.add_argument(
        "--output_timeline_png",
        type=str,
        default=None,
        help="输出 block/resource 级 cycle timeline PNG（gantt 风格）",
    )
    parser.add_argument(
        "--timeline_max_cycle",
        type=int,
        default=2000,
        help="cycle timeline 可视化上限，默认前 2000 cycles",
    )
    parser.add_argument(
        "--enable_w8a8_quant_dequant",
        action="store_true",
        help="在 Conv/Linear 前后插入量化和反量化节点（FP32↔INT8），参与 cycle/功耗仿真；使用 logic_circuit.quant_pipeline_8bit / dequant_pipeline 配置",
    )
    parser.add_argument(
        "--sim_log_dir",
        type=str,
        default=SIM_LOG_DIR,
        help="仿真日志与报告等输出统一保存目录，默认 sim_log",
    )
    parser.add_argument("--logic_par_layernorm", type=int, default=None, metavar="N", help="逻辑电路 LayerNorm 并行度（≥1）")
    parser.add_argument("--logic_par_relu", type=int, default=None, metavar="N", help="逻辑电路 ReLU 并行度（≥1）")
    parser.add_argument("--logic_par_spool", type=int, default=None, metavar="N", help="逻辑电路 SPool 并行度（≥1）")
    parser.add_argument("--logic_par_quant_dequant", type=int, default=None, metavar="N", help="逻辑电路量化/反量化并行度（≥1）")
    parser.add_argument("--logic_clk_freq", type=float, default=None, metavar="Hz", help="数字逻辑时钟频率(Hz)，默认 1e9=1GHz")
    parser.add_argument("--use_scalesim_systolic", action="store_true", help="base_v1/base_v2 下用 SCALE-Sim 统计 Conv/Linear 周期与 MAC/访问；能耗仍用本工程内存建模")
    parser.add_argument("--scalesim_root", type=str, default=None, help="SCALE-Sim 根目录，默认用环境变量 SCALE_SIM_ROOT 或 /workspace/SCALE-Sim")
    parser.add_argument("--scalesim_config", type=str, default=None, help="SCALE-Sim 配置文件路径，默认 scalesim_root/configs/demo_64x64_32kb_ws.cfg")
    parser.add_argument("--knn_conv_pipeline_mode", type=str, choices=["disabled", "strict"], default="disabled", help="KNN→Conv producer-consumer 调度模式：disabled 保留原始层流水；strict 仅在 Conv1d 输入长度严格满足 query_points×k 时启用")
    parser.add_argument("--pointcim_sa_conv_pipeline", action="store_true", default=False, help="PointCIM 模式专用：每个 query 的 BallQuery 完成后立即驱动后续 Conv2d(kernel_size=1)，与下一 query 的 FPS+BQ 并行（SA sample_and_group ↔ MLP 流水线）")
    parser.add_argument("--fc_num_slices", type=int, default=1, help="Feature Computing parallel slices (1=single-block)")
    parser.add_argument("--fc_cores_per_slice", type=int, default=16, help="Cores per FC slice")
    parser.add_argument("--fc_muls_per_core", type=int, default=1024, help="MACs per FC core")
    parser.add_argument("--fc_coalescing_factor", type=float, default=0.0, help="DRAM read savings from LHT coalescing (0.0-1.0)")

    return parser.parse_args()


def _apply_neurosim_cli_overrides(hw_config: HardwareConfig, args: argparse.Namespace) -> None:
    if getattr(args, "neurosim_root", None):
        hw_config.neurosim_root = args.neurosim_root
    if getattr(args, "neurosim_pybind", False):
        hw_config.neurosim_pybind = True
    clk_freq = getattr(args, "neurosim_clk_freq", None)
    if clk_freq is not None and float(clk_freq) > 0:
        hw_config.neurosim_clk_freq_hz = float(clk_freq)
    adc_clk = getattr(args, "neurosim_adc_clk_freq", None)
    if adc_clk is not None and float(adc_clk) > 0:
        hw_config.neurosim_adc_clk_freq_hz = float(adc_clk)
    sar_power_w = resolve_neurosim_sar_column_power_w(
        getattr(args, "neurosim_sar_column_power_w", None),
        getattr(args, "neurosim_sar_column_power_uw", None),
    )
    if sar_power_w is not None:
        hw_config.neurosim_sar_column_power_w = float(sar_power_w)
    ncm = getattr(args, "neurosim_num_col_muxed", None)
    if ncm is not None:
        v = int(ncm)
        hw_config.neurosim_num_col_muxed = v if v > 0 else None


def _enforce_pointcim_neurosim_backend(hw_config: HardwareConfig) -> None:
    if bool(getattr(hw_config, "is_pointcim", False)):
        hw_config.cim_backend = "neurosim"


# =============================================================================
# 网络加载
# =============================================================================

def load_network(
    name: str,
    input_points: int = 1024,
    num_classes: int = 40,
    enable_distance_reuse: bool = False,
    **kwargs,
) -> List[LayerDef]:
    """
    根据名称加载网络层定义（LayerDef 有序列表）。

    Args:
        name: 网络名称，支持:
            "pointkan"                    — PointKAN 分类 (ModelNet40)
            "pointkan-elite"              — PointKAN-Elite 分类 (ModelNet40)
            "pointkan-elite-scanobj"      — PointKAN-Elite 分类 (ScanObjectNN)
            "pointkan-elite-partseg"      — PointKAN-Elite 零件分割 (ShapeNetPart)
            "pointnet2-cls-ssg"           — PointNet2ClsSSG 分类 (ModelNet40)
            "pointnet2-partseg-ssg"       — PointNet2PartSegSSG 分割 (ShapeNetPart)
            "pointnet2-sem-ssg"           — PointNet2SemSSG 语义分割 (S3DIS)
        input_points: 输入点云点数
        num_classes:  分类/分割类别数
        enable_distance_reuse: base_v3 距离矩阵复用，第一层 SA/GAM 存 DRAM，后续层读子矩阵
        **kwargs:     网络专属参数 (如 input_channels=12 for sem-ssg)

    Returns:
        按前向顺序排列的 LayerDef 列表。

    Raises:
        ValueError: 未知网络名时抛出。
    """
    if name == "pointkan":
        return create_pointkan_classification(
            input_points=input_points,
            num_classes=num_classes,
        )
    if name == "pointkan-elite":
        return create_pointkan_elite_classification(
            input_points=input_points,
            num_classes=num_classes,
        )
    if name == "pointkan-elite-scanobj":
        _nc = num_classes if num_classes != 40 else 15
        return create_pointkan_elite_scanobjectnn_classification(
            input_points=input_points,
            num_classes=_nc,
        )
    if name == "pointkan-elite-partseg":
        _pts = input_points if input_points != 1024 else 2048
        _nc = num_classes if num_classes != 40 else 50
        return create_pointkan_elite_partseg(
            input_points=_pts,
            num_classes=_nc,
            enable_distance_reuse=enable_distance_reuse,
        )
    if name == "pointnet2-cls-ssg":
        return create_pointnet2_cls_ssg(
            input_points=input_points,
            num_classes=num_classes,
        )
    if name == "pointnet2-partseg-ssg":
        return create_pointnet2_partseg_ssg(
            input_points=input_points,
            num_classes=num_classes,
            enable_distance_reuse=enable_distance_reuse,
        )
    if name == "pointnet2-sem-ssg":
        input_channels = kwargs.get("input_channels", 12)
        return create_pointnet2_sem_ssg(
            input_points=input_points,
            num_classes=num_classes,
            input_channels=input_channels,
            enable_distance_reuse=enable_distance_reuse,
        )
    raise ValueError(
        f"未知网络: {name}。支持: pointkan, pointkan-elite, "
        "pointkan-elite-scanobj, pointkan-elite-partseg, "
        "pointnet2-cls-ssg, pointnet2-partseg-ssg, pointnet2-sem-ssg"
    )


# =============================================================================
# 仿真核心
# =============================================================================

def _add_events_to_scheduler(
    scheduler: Scheduler,
    events: List[Any],
    base_cycle: int,
    layer_name: str,
    layer_type: str,
) -> None:
    """
    将本层产生的一批仿真事件加入调度器，并统一打上当前层名与层类型。
    用于后续按层聚合能耗、以及 operator_sequence 中的 layer_name 字段。
    """
    for event in events:
        meta = dict(event.metadata or {})
        meta["layer_name"] = layer_name
        meta["layer_type"] = layer_type
        new_ev = scheduler.event_queue.add_event(
            cycle=event.cycle + base_cycle,
            operator=event.operator,
            stage=event.stage,
            sub_stage=event.sub_stage,
            circuit_module=event.circuit_module,
            operation=event.operation,
            power=event.power,
            duration=event.duration,
            clock_period_ns=event.clock_period_ns,
            metadata=meta,
        )
        if getattr(event, "overlapped", False):
            new_ev.overlapped = True


def _is_pointcim_sa_conv_cross_layer_streaming_target(layer: Any, hw_config: Any) -> bool:
    """
    Whether this layer should skip monolithic feature input reads and allow
    xyz-phase cross-layer launch before the previous layer's feature spill/refill.

    This is intentionally narrow:
    - PointNet++-style SA layers only
    - PointCIM fused FPS/BQ path only
    - pointcim_sa_conv_pipeline must be enabled
    """
    if layer is None:
        return False
    if str(getattr(layer, "layer_type", "") or "") != "SA":
        return False
    if not bool(getattr(hw_config, "is_acc", False)):
        return False
    if not bool(getattr(hw_config, "use_rram_dist", False)):
        return False
    if not bool(getattr(hw_config, "pointcim_sa_conv_pipeline", False)):
        return False

    ops = list(getattr(layer, "operators", []) or [])
    op_types = [str(getattr(op, "op_type", "") or "") for op in ops]
    if "fps" not in op_types or "ballquery" not in op_types:
        return False

    for op in ops:
        if str(getattr(op, "op_type", "") or "") not in ("conv1d", "conv2d"):
            continue
        params = getattr(op, "params", {}) or {}
        if int(params.get("kernel_size", 1) or 1) == 1:
            return True
    return False


def simulate_network(
    layers: List[LayerDef],
    hw_config: HardwareConfig,
    show_layers: bool = False,
    cim_full_context: Optional[Dict[str, Any]] = None,
    activation_shapes: Optional[List[ActivationShapeInfo]] = None,
    enable_layer_buffer: bool = True,
    show_shapes: bool = False,
    repo_root: Optional[str] = None,
    real_conv_linear_list: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    对整网进行周期级仿真：按层顺序派发算子、生成事件并送入调度器执行。

    - 若提供 cim_full_context，conv1d/linear 使用全网络 CIM（NeuroSim）注入的延迟与能耗。
    - 若 enable_layer_buffer 且提供 activation_shapes，在每层首尾插入层间 buffer 读写事件
      （首层前：点云 N×3 从 DRAM 加载；层间：buffer 读/写，超出 buffer 部分按 DRAM spill 建模）。
    - 每个事件的 metadata 写入当前层名与层类型，供 breakdown 按层聚合及 operator_sequence 的 layer_name。

    Args:
        layers: 网络层定义列表（如 embedding → GAM-stage0 → …）
        hw_config: 硬件配置（时钟、CIM 并行度、层间 buffer、BN 融合等）
        show_layers: 是否在控制台打印每层周期与事件数
        cim_full_context: 可选全网络 CIM 注入结果（键名见 integration.cim_full_context）
        activation_shapes: 每层激活 shape 与字节数（用于 buffer 事件与报告）
        enable_layer_buffer: 是否启用层间 buffer 读写与 buffer 容量约束
        show_shapes: 是否打印每层激活 shape（input → output）
        repo_root: 项目根目录（PointKAN-pytorch，含 hw_sim/cycle_simulator/memory_model）。用于 CACTI/DRAMSim3 的 configs/outputs，与 demo_pointcloud_pipeline 的 memory_model 目录约定一致；None 时 adapter 尝试从 memory_model 包位置自动解析。

    Returns:
        字典，含 total_cycles、total_energy_nJ、layer_results、sim_result、
        operator_breakdown、operator_sequence（与 constants 键名一致）等。
    """
    scheduler = Scheduler(hw_config=hw_config)
    layer_results: List[Dict[str, Any]] = []
    current_cycle = 0
    clock_ns = hw_config.clock_period
    use_buffer = enable_layer_buffer and activation_shapes is not None and len(activation_shapes) == len(layers)
    memory_runtime_context = None
    feature_residency_plan = []
    if repo_root:
        setattr(hw_config, "repo_root", repo_root)

    def _infer_acc_input_points_local() -> int:
        max_points = 0
        for layer in layers:
            for op in getattr(layer, "operators", []) or []:
                params = getattr(op, "params", {}) or {}
                try:
                    pts = int(params.get("input_points", 0) or 0)
                except Exception:
                    pts = 0
                if pts > max_points:
                    max_points = pts
        if max_points > 0:
            return max_points
        if activation_shapes:
            first = activation_shapes[0]
            shape = getattr(first, "input_shape", None) or []
            ints = []
            for x in shape:
                try:
                    ints.append(int(x))
                except Exception:
                    pass
            if ints:
                return max(ints)
        return 1024

    def _acc_active_banks_local() -> int:
        if not getattr(hw_config, "is_acc", False):
            return 1
        macro = getattr(hw_config, "sram_cim_macro", None)
        array = getattr(macro, "array", None) if macro is not None else None
        cim_parallel = max(1, int(getattr(hw_config, "cim_parallelism", 128) or 128))
        array_rows = max(cim_parallel, int(getattr(array, "rows", cim_parallel) or cim_parallel))
        bank_capacity_points = max(cim_parallel, array_rows)
        input_points_local = max(1, _infer_acc_input_points_local())
        return max(1, int(math.ceil(input_points_local / float(bank_capacity_points))))

    # 统一 SRAM CACTI 回填：
    # - input_buffer / output_buffer 不应依赖 layer buffer 开关
    # - dist/self_dist/fps_min/topk/temp_max 等命名 SRAM 同样始终尝试回填
    if repo_root:
        from pointkan_sim.integration.memory_model_adapter import (
            fill_buffers_from_cacti,
            fill_dram_per_access_from_dramsim3,
            fill_named_sram_buffers_from_cacti,
        )
        ok_cim_buffers = fill_buffers_from_cacti(hw_config, repo_root)
        if not ok_cim_buffers:
            raise RuntimeError(
                "SRAM-CIM input/output buffers require CACTI simulation. "
                "CACTI was not run or failed for sram_cim_macro buffers."
            )
        ok_named_sram = fill_named_sram_buffers_from_cacti(hw_config, repo_root)
        if not ok_named_sram:
            raise RuntimeError(
                "Dedicated SRAM buffers require CACTI simulation. "
                "CACTI was not run or failed for one or more named SRAM buffers."
            )
        _t0 = time.perf_counter()
        log_info("DRAMSim3(per-access): start calibrating read/write energy ...")
        ok_dram_per_access = fill_dram_per_access_from_dramsim3(hw_config, repo_root)
        log_info(f"DRAMSim3(per-access): done in {(time.perf_counter() - _t0):.1f}s (ok={bool(ok_dram_per_access)})")
        if not ok_dram_per_access:
            raise RuntimeError(
                "DRAM per-access energy requires DRAMSim3 simulation. "
                "DRAMSim3 was not run or failed to calibrate read/write per-access energy."
            )

    # 层间 SRAM：通过 CACTI 仿真得到 cycle/energy（仅 CACTI 回填，无回退）；同时初始化持久 memory runtime context。
    if use_buffer:
        from pointkan_sim.integration.memory_model_adapter import (
            fill_inter_layer_buffer_from_cacti,
        )
        from pointkan_sim.integration.memory_runtime import create_memory_runtime_context
        from pointkan_sim.memory.feature_residency import build_feature_residency_plan, plan_by_layer_name, get_effective_feature_buffer_bytes, get_feature_buffer_slots
        feature_residency_plan = build_feature_residency_plan(layers, activation_shapes, hw_config)
        setattr(hw_config, "feature_residency_plan", feature_residency_plan)
        setattr(hw_config, "feature_residency_plan_by_layer", plan_by_layer_name(feature_residency_plan))
        ok = fill_inter_layer_buffer_from_cacti(hw_config, repo_root)
        if not ok:
            raise RuntimeError(
                "SRAM (inter-layer buffer) energy/cycle require CACTI simulation. "
                "CACTI was not run or failed. Check repo_root, memory_model path (hw_sim/cycle_simulator/memory_model), "
                "cacti binary and configs (configs/cacti, outputs/cacti)."
            )
        memory_runtime_context = create_memory_runtime_context(hw_config, repo_root)

    # 与 main 中 DRAM 事件顺序一致：首层 load，再每层 spill write / 下一层 spill read；用于 trace→DRAMSim3 预计算；未触发或失败则 raise
    dram_precomputed_iter = None
    dram_transfers: List[tuple] = []
    feature_residency_plan_by_layer = getattr(hw_config, "feature_residency_plan_by_layer", {}) if use_buffer else {}
    feature_buffer_slots = get_feature_buffer_slots(hw_config) if use_buffer else 1

    def _tag_feature_buffer_slot_events(events, slot_id: int, phase: str) -> None:
        if not use_buffer:
            return
        sid = max(0, int(slot_id))
        for ev in events:
            if (ev.operator or "").strip() != "LayerBuffer":
                continue
            meta = getattr(ev, "metadata", None) or {}
            meta["buffer_name"] = "feature_sram_buffer"
            meta["buffer_slot"] = sid
            meta["resource_id"] = f"feature_sram_slot_{sid}"
            meta["exclusive"] = True
            meta["reset_scope"] = "layer_boundary"
            meta["buffer_phase"] = phase
            ev.metadata = meta

    def _apply_feature_chunk_pipeline_overlap(events: List[Any], overlap_tag: str) -> None:
        chunked_layerbuffer = []
        for ev in events:
            if (ev.operator or "").strip() != "LayerBuffer":
                continue
            meta = getattr(ev, "metadata", None) or {}
            chunk_count = int(meta.get("chunk_count", 1) or 1)
            chunk_idx = int(meta.get("chunk_idx", 0) or 0)
            if chunk_count > 1:
                chunked_layerbuffer.append((chunk_idx, ev, meta))
        if not chunked_layerbuffer:
            return
        for chunk_idx, ev, meta in chunked_layerbuffer:
            if chunk_idx <= 0:
                continue
            ev.overlapped = True
            meta["pipeline_overlap"] = overlap_tag
            ev.metadata = meta

    if use_buffer and getattr(hw_config, "dram", None):
        buf = getattr(hw_config, "inter_layer_buffer", None)
        spill = getattr(buf, "spill_to_dram", False) if buf else False
        size = max(1, get_effective_feature_buffer_bytes(hw_config) or getattr(buf, "size_bytes", 65536)) if buf else 65536
        for i in range(len(layers)):
            shape_info = activation_shapes[i] if i < len(activation_shapes) else None
            if not shape_info:
                continue
            if i == 0 and shape_info.input_bytes > 0:
                dram_transfers.append((shape_info.input_bytes, True))
            if i < len(layers) - 1 and shape_info.output_bytes > size and spill:
                dram_transfers.append((shape_info.output_bytes - size, False))
            if (
                i > 0
                and shape_info.input_bytes > size
                and spill
                and not _is_pointcim_sa_conv_cross_layer_streaming_target(layers[i], hw_config)
            ):
                dram_transfers.append((shape_info.input_bytes - size, True))
    if use_buffer and dram_transfers and memory_runtime_context is not None:
        _t0 = time.perf_counter()
        log_info(f"DRAMSim3(layer-buffer): prime_batch_transfers start (transfers={len(dram_transfers)}) ...")
        dram_results = memory_runtime_context.prime_batch_transfers(dram_transfers)
        log_info(f"DRAMSim3(layer-buffer): prime_batch_transfers done in {(time.perf_counter() - _t0):.1f}s")
        dram_precomputed_iter = iter((r["cycles"], r["energy_nj"]) for r in dram_results)
    elif use_buffer and dram_transfers and memory_runtime_context is None:
        if not getattr(hw_config.dram, "dramsim3_bin", None) or not getattr(hw_config.dram, "dramsim3_config_ini", None):
            raise RuntimeError(
                "DRAM energy/cycle require DRAMSim3 simulation. "
                "hw_config.dram.dramsim3_bin and dramsim3_config_ini must be set when using layer buffer with DRAM."
            )
        from pointkan_sim.integration.memory_model_adapter import run_dram_trace_for_layer_buffer_dram
        _t0 = time.perf_counter()
        log_info(f"DRAMSim3(layer-buffer): start trace (transfers={len(dram_transfers)}) ...")
        dram_results = run_dram_trace_for_layer_buffer_dram(dram_transfers, hw_config, repo_root)
        log_info(f"DRAMSim3(layer-buffer): done in {(time.perf_counter() - _t0):.1f}s")
        if not dram_results or len(dram_results) != len(dram_transfers):
            raise RuntimeError(
                "DRAM energy/cycle require DRAMSim3 simulation. "
                "DRAMSim3 was not run or returned invalid results. Check repo_root, dramsim3_bin, dramsim3_config_ini, "
                "and memory_model path (hw_sim/cycle_simulator/memory_model)."
            )
        dram_precomputed_iter = iter(dram_results)

    # 硬件算子顺序来源：按「层」顺序遍历，每层内按「算子」顺序派发。
    # 顺序 = 层顺序（由 network 定义，如 pointkan_elite_net：embedding → GAM-stage0 → LFP-stage0 → stage-0 → …）
    #       + 每层内： [buffer读/首层DRAM加载] → dispatch_layer(layer.operators) → [buffer写]
    # 因此 GAM 的 FPS/KNN 会紧接在 embedding 之后出现，因为第 2 层是 GAM-stage0，且 GAM 的 operators 定义为 [FPS, KNN, AffineNorm, S-Pool]。
    # 仅纯 acc 模式下 FPS 在首层执行后可复用，后续层读 SRAM 索引；
    # PointCIM（is_acc + use_rram_dist）和 baseline 系列后续层 FPS 需要重新计算。
    first_layer_with_fps: Optional[int] = None
    _acc_fps_reuse = (getattr(hw_config, "is_acc", False)
                      and not getattr(hw_config, "use_rram_dist", False))
    if _acc_fps_reuse:
        for idx, layer in enumerate(layers):
            if any(op.op_type == "fps" for op in layer.operators):
                first_layer_with_fps = idx
                break

    # 跨 SA 层 xyz_phase 流水线：识别连续 SA 层序列，将 FPS/BQ 提前调度
    # xyz_phase 仅依赖上层 FPS idx，不需等待 MLP/MaxPool
    _sa_xyz_phase_end: Dict[int, int] = {}   # layer_index → xyz_phase 完成周期
    _sa_feature_phase_end: Dict[int, int] = {}  # layer_index → feature_phase 完成周期
    _sa_cross_layer_pipeline = getattr(hw_config, "is_acc", False)

    def _split_sa_operators(operators):
        """将 SA 层算子拆分为 xyz_phase (FPS+BQ) 和 feature_phase (其余)"""
        xyz_ops = []
        feature_ops = []
        for op in operators:
            if op.op_type in ("fps", "ballquery"):
                xyz_ops.append(op)
            else:
                feature_ops.append(op)
        return xyz_ops, feature_ops

    def _is_sa_with_fps_bq(layer):
        op_types = [op.op_type for op in layer.operators]
        return "fps" in op_types and "ballquery" in op_types
    real_conv_linear_index = 0
    for i, layer in enumerate(layers):
        shape_info = activation_shapes[i] if (activation_shapes and i < len(activation_shapes)) else None

        # 无算子层（如全局池化）只记录占位信息，不产生事件
        if not layer.operators:
            layer_results.append({
                "name": layer.name,
                "type": layer.layer_type,
                "cycles": 0,
                "events": 0,
                "description": layer.description,
                "input_shape": list(shape_info.input_shape) if shape_info and shape_info.input_shape else None,
                "output_shape": list(shape_info.output_shape) if shape_info and shape_info.output_shape else None,
            })
            if show_layers and show_shapes and shape_info:
                log_info(
                    f"  {layer.name:<25s}  —  (无算子)  "
                    f"in {format_activation_shape(shape_info.input_shape)}  "
                    f"→ out {format_activation_shape(shape_info.output_shape)}"
                )
            continue

        layer_start_cycle = current_cycle
        num_events_this_layer = 0

        # 1) 首层：点云输入 N×3 从 DRAM 加载
        if use_buffer and shape_info and i == 0 and shape_info.input_bytes > 0:
            load_events = gen_first_layer_input_load_events(
                shape_info.input_bytes, hw_config, layer.name, clock_ns,
                dram_precomputed=dram_precomputed_iter,
                memory_context=memory_runtime_context,
            )
            _add_events_to_scheduler(scheduler, load_events, current_cycle, layer.name, layer.layer_type)
            if load_events:
                current_cycle += sum(e.duration for e in load_events)
                num_events_this_layer += len(load_events)

        # 2) 非首层：从 buffer（及可选 DRAM spill）读入本层输入
        if (
            use_buffer
            and shape_info
            and i > 0
            and shape_info.input_bytes > 0
            and not _is_pointcim_sa_conv_cross_layer_streaming_target(layer, hw_config)
        ):
            read_events = gen_layer_buffer_read_events(
                shape_info.input_bytes, hw_config, layer.name, clock_ns,
                dram_precomputed=dram_precomputed_iter,
                memory_context=memory_runtime_context,
            )
            _tag_feature_buffer_slot_events(read_events, (i - 1) % max(1, feature_buffer_slots), "input_read")
            current_plan = feature_residency_plan_by_layer.get(layer.name)
            buf_cfg = getattr(hw_config, "inter_layer_buffer", None)
            pipeline_mode = str(getattr(buf_cfg, "pipeline_mode", "without_pipeline") or "without_pipeline").strip().lower() if buf_cfg else "without_pipeline"
            double_buffering = bool(getattr(buf_cfg, "double_buffering", False)) if buf_cfg else False
            allow_feature_read_stream = (
                pipeline_mode == "pipeline"
                and double_buffering
                and current_plan is not None
                and int(getattr(current_plan, "input_dram_bytes", 0) or 0) == 0
            )
            if allow_feature_read_stream:
                _apply_feature_chunk_pipeline_overlap(read_events, "feature_sram_double_buffer_read")
            _add_events_to_scheduler(scheduler, read_events, current_cycle, layer.name, layer.layer_type)
            if read_events:
                current_cycle += sum(e.duration for e in read_events if not getattr(e, "overlapped", False))
                num_events_this_layer += len(read_events)

        # 3) 本层计算事件（传入 layer_index / first_layer_with_fps，供 acc 下 FPS 仅首层执行、后续复用）
        dispatch_context = dict(cim_full_context) if cim_full_context else {}
        if real_conv_linear_list:
            dispatch_context["real_conv_linear_list"] = real_conv_linear_list
            dispatch_context["real_conv_linear_index"] = real_conv_linear_index
        dispatch_context["layer_index"] = i
        dispatch_context["layer_name"] = layer.name
        dispatch_context["memory_runtime_context"] = memory_runtime_context
        if _acc_fps_reuse and first_layer_with_fps is not None:
            dispatch_context["first_layer_with_fps"] = first_layer_with_fps

        # Cross-SA layer xyz_phase pipeline:
        # SA 层拆分为 xyz_phase (FPS+BQ) 和 feature_phase (Conv+BN+ReLU+MaxPool)。
        # xyz_phase 仅依赖上一 SA 层的 xyz_phase（FPS idx），不需等待上层 MLP/MaxPool。
        # feature_phase 依赖 xyz_phase + 上层 feature_phase（需要 points 特征）。
        _prev_sa_indices = [k for k in sorted(_sa_xyz_phase_end.keys()) if k < i]
        _prev_sa_idx = _prev_sa_indices[-1] if _prev_sa_indices else None

        if (_sa_cross_layer_pipeline
                and _is_sa_with_fps_bq(layer)
                and layer.layer_type == "SA"):
            xyz_ops, feature_ops = _split_sa_operators(layer.operators)

            # PointCIM SA-Conv pipeline: dispatch xyz+feature ops together so that
            # dispatch_layer_ops sees fps -> ballquery -> conv2d(k=1) as consecutive ops
            # and applies the per-query BQ→Conv producer-consumer pipeline.
            # _sa_xyz_phase_end is derived from FPS/BallQuery events in the combined result
            # so the cross-layer SA pipeline is preserved.
            _pointcim_sa_conv_pipe = (
                getattr(hw_config, "pointcim_sa_conv_pipeline", False)
                and getattr(hw_config, "use_rram_dist", False)
                and getattr(hw_config, "is_acc", False)
                and xyz_ops and feature_ops
            )

            if _pointcim_sa_conv_pipe:
                # Combined dispatch: xyz_ops + feature_ops in one call so that
                # dispatch_layer_ops can pipeline BQ→Conv per-query
                combined_start = current_cycle
                if _prev_sa_idx is not None and _prev_sa_idx in _sa_xyz_phase_end:
                    # Allow next-layer xyz/BQ to launch from the previous xyz-phase end.
                    # Feature availability for per-query gather remains modeled inside the
                    # combined pipeline via gather timing and shared-resource scheduling.
                    combined_start = _sa_xyz_phase_end[_prev_sa_idx]

                combined_events = dispatch_layer(xyz_ops + feature_ops, hw_config, dispatch_context)
                _add_events_to_scheduler(scheduler, combined_events, combined_start,
                                         layer.name + "_xyz", layer.layer_type)

                # Extract xyz_phase end from FPS/BallQuery events to maintain cross-layer pipeline
                xyz_evts_in_combined = [
                    e for e in combined_events
                    if (getattr(e, "operator", "") or "").strip() in ("FPS", "BallQuery")
                    and not getattr(e, "overlapped", False)
                ]
                if xyz_evts_in_combined:
                    _sa_xyz_phase_end[i] = combined_start + max(
                        e.cycle + e.duration for e in xyz_evts_in_combined
                    )
                else:
                    _sa_xyz_phase_end[i] = combined_start

                combined_non_ov = [e for e in combined_events if not getattr(e, "overlapped", False)]
                combined_span = (max((e.cycle + e.duration for e in combined_non_ov), default=0)
                                 if combined_non_ov
                                 else max((e.cycle + e.duration for e in combined_events), default=0))
                _sa_feature_phase_end[i] = combined_start + combined_span
                num_events_this_layer += len(combined_events)

            else:
                if xyz_ops:
                    # xyz_phase 起点：上一 SA 的 xyz_phase 完成 (而非整个上层完成)
                    xyz_start = current_cycle
                    if _prev_sa_idx is not None and _prev_sa_idx in _sa_xyz_phase_end:
                        xyz_start = max(xyz_start, _sa_xyz_phase_end[_prev_sa_idx])

                    xyz_events = dispatch_layer(xyz_ops, hw_config, dispatch_context)
                    _add_events_to_scheduler(scheduler, xyz_events, xyz_start,
                                             layer.name + "_xyz", layer.layer_type)
                    xyz_non_ov = [e for e in xyz_events if not getattr(e, "overlapped", False)]
                    xyz_span = (max((e.cycle + e.duration for e in xyz_non_ov), default=0)
                                if xyz_non_ov
                                else max((e.cycle + e.duration for e in xyz_events), default=0))
                    _sa_xyz_phase_end[i] = xyz_start + xyz_span
                    num_events_this_layer += len(xyz_events)

                if feature_ops:
                    # feature_phase 起点：max(本层 xyz_phase, 上层 feature_phase)
                    feature_start = _sa_xyz_phase_end.get(i, current_cycle)
                    if _prev_sa_idx is not None and _prev_sa_idx in _sa_feature_phase_end:
                        feature_start = max(feature_start, _sa_feature_phase_end[_prev_sa_idx])

                    feature_events = dispatch_layer(feature_ops, hw_config, dispatch_context)
                    _add_events_to_scheduler(scheduler, feature_events, feature_start,
                                             layer.name + "_feat", layer.layer_type)
                    feat_non_ov = [e for e in feature_events if not getattr(e, "overlapped", False)]
                    feat_span = (max((e.cycle + e.duration for e in feat_non_ov), default=0)
                                 if feat_non_ov
                                 else max((e.cycle + e.duration for e in feature_events), default=0))
                    _sa_feature_phase_end[i] = feature_start + feat_span
                    num_events_this_layer += len(feature_events)

            # current_cycle 推进到本层完全结束
            current_cycle = max(
                _sa_xyz_phase_end.get(i, current_cycle),
                _sa_feature_phase_end.get(i, current_cycle),
            )

            if cim_full_context is not None and CIM_LAYER_INDEX in dispatch_context:
                cim_full_context[CIM_LAYER_INDEX] = dispatch_context[CIM_LAYER_INDEX]
            if cim_full_context is not None and dispatch_context.get(ACC_CIM_KEY_PRELOADED):
                cim_full_context[ACC_CIM_KEY_PRELOADED] = True
            if real_conv_linear_list:
                real_conv_linear_index = int(dispatch_context.get("real_conv_linear_index", real_conv_linear_index) or real_conv_linear_index)

        else:
            # 非 SA 层或不启用跨层流水线：保持原有串行逻辑
            layer_events = dispatch_layer(layer.operators, hw_config, dispatch_context)
            if cim_full_context is not None and CIM_LAYER_INDEX in dispatch_context:
                cim_full_context[CIM_LAYER_INDEX] = dispatch_context[CIM_LAYER_INDEX]
            if cim_full_context is not None and dispatch_context.get(ACC_CIM_KEY_PRELOADED):
                cim_full_context[ACC_CIM_KEY_PRELOADED] = True
            if real_conv_linear_list:
                real_conv_linear_index = int(dispatch_context.get("real_conv_linear_index", real_conv_linear_index) or real_conv_linear_index)
            _add_events_to_scheduler(scheduler, layer_events, current_cycle, layer.name, layer.layer_type)
            non_ov = [e for e in layer_events if not getattr(e, "overlapped", False)]
            layer_span = max((e.cycle + e.duration for e in non_ov), default=0) if non_ov else max((e.cycle + e.duration for e in layer_events), default=0)
            current_cycle += layer_span
            num_events_this_layer += len(layer_events)

        # 4) 非末层：将本层输出写回 buffer（及可选 DRAM spill）
        if use_buffer and shape_info and i < len(layers) - 1 and shape_info.output_bytes > 0:
            write_events = gen_layer_buffer_write_events(
                shape_info.output_bytes, hw_config, layer.name, clock_ns,
                dram_precomputed=dram_precomputed_iter,
                memory_context=memory_runtime_context,
            )
            _tag_feature_buffer_slot_events(write_events, i % max(1, feature_buffer_slots), "output_write")
            _add_events_to_scheduler(scheduler, write_events, current_cycle, layer.name, layer.layer_type)
            if write_events:
                buf_cfg = getattr(hw_config, "inter_layer_buffer", None)
                pipeline_mode = str(getattr(buf_cfg, "pipeline_mode", "without_pipeline") or "without_pipeline").strip().lower() if buf_cfg else "without_pipeline"
                double_buffering = bool(getattr(buf_cfg, "double_buffering", False)) if buf_cfg else False
                next_layer = layers[i + 1] if i + 1 < len(layers) else None
                next_layer_name = str(getattr(next_layer, "name", "") or "") if next_layer is not None else ""
                next_plan = feature_residency_plan_by_layer.get(next_layer_name)
                allow_spill_overlap = (
                    pipeline_mode == "pipeline"
                    and double_buffering
                    and next_plan is not None
                    and int(getattr(next_plan, "input_dram_bytes", 0) or 0) == 0
                )
                if allow_spill_overlap:
                    _apply_feature_chunk_pipeline_overlap(write_events, "feature_sram_double_buffer_write")
                    for ev in write_events:
                        if (ev.operator or "").strip() == "DRAM":
                            ev.overlapped = True
                            meta = getattr(ev, "metadata", None) or {}
                            meta["pipeline_overlap"] = "feature_sram_double_buffer_dram"
                            ev.metadata = meta
                current_cycle += sum(e.duration for e in write_events if not getattr(e, "overlapped", False))
                num_events_this_layer += len(write_events)

        layer_results.append({
            "name": layer.name,
            "type": layer.layer_type,
            "cycles": current_cycle - layer_start_cycle,
            "events": num_events_this_layer,
            "description": layer.description,
            "input_shape": list(shape_info.input_shape) if shape_info and shape_info.input_shape else None,
            "output_shape": list(shape_info.output_shape) if shape_info and shape_info.output_shape else None,
            "input_bytes": shape_info.input_bytes if shape_info else None,
            "output_bytes": shape_info.output_bytes if shape_info else None,
        })

        if show_layers:
            shape_str = ""
            if show_shapes and shape_info:
                shape_str = (
                    f"  in {format_activation_shape(shape_info.input_shape)} "
                    f"→ out {format_activation_shape(shape_info.output_shape)}  "
                    f"({shape_info.output_bytes} B)"
                )
            log_info(
                f"  {layer.name:<25s}  {current_cycle - layer_start_cycle:>10,} cycles  "
                f"{num_events_this_layer:>4} events  [{layer.layer_type}]{shape_str}"
            )

    # 运行周期级仿真，汇总总周期、总能耗、峰值功率等
    log_info(
        f"\n运行仿真: {len(scheduler.event_queue)} 个事件, "
        f"约 {current_cycle:,} 周期..."
    )
    sim_result = scheduler.run_simulation()

    clock_period = hw_config.clock_period
    _scheduler_raw_cycles = sim_result.total_cycles  # 调度器原始周期（混合时钟域，不可直接用于延迟换算）
    total_energy = sim_result.total_energy
    # 端到端延迟：按每个事件的 clock_period_ns 换算后累加（NeuroSim 与 1 GHz 模块混合时正确）
    # 跳过 overlapped 事件（如 streaming DRAM writes），它们与其他操作并行执行
    total_latency_sum_ns = sum(
        e.duration * getattr(e, "clock_period_ns", 1.0)
        for e in sim_result.events
        if not getattr(e, "overlapped", False)
    )
    latency_sum_ns = total_latency_sum_ns if total_latency_sum_ns > 0 else (_scheduler_raw_cycles * clock_period)
    # 关键路径口径（真实事件时间轴 makespan）：
    # current_cycle 混合了 NeuroSim (10ns) 和数字逻辑 (1ns) 的 duration，
    # 因此 cycle * clock_period_ns 不能得到正确的 ns 位置。
    # 正确做法：按 scheduler cycle 分组（同 cycle = 并行），累加每组 max(duration*clk_ns)。
    all_events = list(sim_result.events or [])
    if all_events:
        from itertools import groupby as _groupby
        _sorted_evts = sorted(all_events, key=lambda _e: int(_e.cycle))
        _ns_makespan = 0.0
        for _cyc_val, _grp_iter in _groupby(_sorted_evts, key=lambda _e: int(_e.cycle)):
            _non_ov = [_e for _e in _grp_iter if not getattr(_e, "overlapped", False)]
            if _non_ov:
                _ns_makespan += max(
                    int(_e.duration) * float(getattr(_e, "clock_period_ns", 1.0))
                    for _e in _non_ov
                )
        latency_critical_path_ns = max(0.0, _ns_makespan)
    else:
        latency_critical_path_ns = latency_sum_ns
    latency_ns = latency_critical_path_ns
    latency_ms = latency_ns / 1e6
    avg_power = (total_energy / latency_ns * 1e3) if latency_ns > 0 else 0.0
    if clock_period > 0 and latency_critical_path_ns > 0:
        total_cycles = int(math.ceil(latency_critical_path_ns / clock_period))
    else:
        total_cycles = _scheduler_raw_cycles
    if clock_period > 0 and latency_sum_ns > 0:
        total_cycles_sum_equiv = int(math.ceil(latency_sum_ns / clock_period))
    else:
        total_cycles_sum_equiv = _scheduler_raw_cycles

    # 按算子类型聚合周期、能耗与延迟(ns)；Quant/Dequant/BN 单独成行（stage 区分）
    operator_breakdown, operator_cycle_detail = _summarize_operator_timeline_stats(sim_result.events)
    # 距离链关键路径分析：(op, layer) → sorted list of (start_cycle, end_cycle) 非 overlapped 事件区间
    _dist_chain_ops = {"FPS", "KNN"}
    _dist_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    for e in sim_result.events:
        op_name = _display_operator_name(e)
        if not getattr(e, "overlapped", False):
            # 收集 FPS/KNN 非 overlapped 事件区间，用于距离链关键路径分析
            if op_name in _dist_chain_ops:
                meta_dc = getattr(e, "metadata", None) or {}
                layer_key = str(meta_dc.get("layer_name", "") or "")
                dk = (op_name, layer_key)
                _dist_intervals.setdefault(dk, []).append((int(e.cycle), int(e.end_cycle)))

    # ── 距离链关键路径 / makespan 分析 ──────────────────────────────────────────
    # 对 FPS+KNN 事件按 (op, layer) 做区间合并，得到各 stage 的实际忙碌时间（busy cycles）
    # 以及整体距离链 makespan（从首个事件到末个事件的时间窗口）。
    # 注：cross-stage comparator_shared 资源约束会把各阶段事件序列化到全局时间轴上，
    #     makespan 因此反映距离链硬件占用的真实时间窗口（含各阶段间流水间隙）。
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        srt = sorted(intervals)
        merged: List[Tuple[int, int]] = [list(srt[0])]  # type: ignore[assignment]
        for s, e_c in srt[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e_c)  # type: ignore[index]
            else:
                merged.append([s, e_c])  # type: ignore[arg-type]
        return [tuple(x) for x in merged]  # type: ignore[return-value]

    dist_chain_analysis: Dict[str, Any] = {}
    _all_dist_intervals: List[Tuple[int, int]] = []
    per_op_layer: Dict[str, Any] = {}
    for (dc_op, dc_layer), ivs in _dist_intervals.items():
        merged = _merge_intervals(ivs)
        busy = sum(e_c - s + 1 for s, e_c in merged)
        makespan_cyc = (merged[-1][1] - merged[0][0] + 1) if merged else 0
        clk_ns_dc = getattr(hw_config, "clock_period", 1.0)
        per_op_layer.setdefault(dc_op, {})[dc_layer] = {
            "busy_cycles": busy,
            "makespan_cycles": makespan_cyc,
            "busy_ns": round(busy * clk_ns_dc, 2),
            "makespan_ns": round(makespan_cyc * clk_ns_dc, 2),
            "start_cycle": merged[0][0] if merged else 0,
            "end_cycle": merged[-1][1] if merged else 0,
        }
        _all_dist_intervals.extend(ivs)

    if _all_dist_intervals:
        all_merged = _merge_intervals(_all_dist_intervals)
        clk_ns_dc = getattr(hw_config, "clock_period", 1.0)
        total_busy = sum(e_c - s + 1 for s, e_c in all_merged)
        total_makespan = all_merged[-1][1] - all_merged[0][0] + 1 if all_merged else 0
        dist_chain_analysis = {
            "total_busy_cycles": total_busy,
            "total_busy_ns": round(total_busy * clk_ns_dc, 2),
            "total_busy_us": round(total_busy * clk_ns_dc / 1e3, 4),
            "total_makespan_cycles": total_makespan,
            "total_makespan_ns": round(total_makespan * clk_ns_dc, 2),
            "total_makespan_us": round(total_makespan * clk_ns_dc / 1e3, 4),
            "chain_start_cycle": all_merged[0][0],
            "chain_end_cycle": all_merged[-1][1],
            "utilization_pct": round(total_busy / total_makespan * 100, 2) if total_makespan > 0 else 0.0,
            "by_op": per_op_layer,
        }
    # ────────────────────────────────────────────────────────────────────────────

    # SCALE-Sim 逐层 Conv/Linear：周期、利用率、MACs、能耗（仅 baseline+scalesim 时有）
    scalesim_layer_details: List[Dict[str, Any]] = []
    for e in sim_result.events:
        meta = getattr(e, "metadata", None) or {}
        if meta.get("source") != "scalesim":
            continue
        op = (e.operator or "").strip()
        if op not in ("Conv1d", "Linear"):
            continue
        row = {
            "layer_name": meta.get("layer_name", ""),
            "operator": op,
            "total_cycles": meta.get("scalesim_total_cycles", e.duration),
            "stall_cycles": meta.get("scalesim_stall_cycles"),
            "overall_util_pct": meta.get("overall_util_pct"),
            "mapping_eff_pct": meta.get("mapping_eff_pct"),
            "compute_util_pct": meta.get("compute_util_pct"),
            "macs": meta.get("macs"),
            "energy_nj": e.energy,
            "mac_energy_nj": meta.get("scalesim_mac_energy_nj", e.energy if op in ("Conv1d", "Linear") else 0.0),
            "sram_energy_nj": meta.get("scalesim_sram_energy_nj", 0.0),
            "dram_energy_nj": meta.get("scalesim_dram_energy_nj", 0.0),
            "sram_cycles": meta.get("scalesim_sram_cycles", 0),
            "dram_cycles": meta.get("scalesim_dram_cycles", 0),
            "sram_clock_period_ns": meta.get("scalesim_sram_clock_period_ns"),
            "dram_clock_period_ns": meta.get("scalesim_dram_clock_period_ns"),
            "dram_trace_dir": meta.get("scalesim_dram_trace_dir", ""),
            "feature_input_dram_bytes": meta.get("feature_input_dram_bytes", 0),
            "feature_output_dram_bytes": meta.get("feature_output_dram_bytes", 0),
            "weight_dram_bytes": meta.get("weight_dram_bytes", 0),
            "weight_tensor_bytes": meta.get("weight_tensor_bytes", 0),
            "weight_dram_trace_bytes": meta.get("weight_dram_trace_bytes", 0),
            "feature_local_sram_trace_bytes": meta.get("feature_local_sram_trace_bytes", 0),
            "weight_sram_trace_bytes": meta.get("weight_sram_trace_bytes", 0),
            "feature_input_dram_trace_dir": meta.get("feature_input_dram_trace_dir", ""),
            "weight_dram_trace_dir": meta.get("weight_dram_trace_dir", ""),
            "feature_output_dram_trace_dir": meta.get("feature_output_dram_trace_dir", ""),
            "weight_refill_tiles": meta.get("weight_refill_tiles", 1),
            "weight_refill_tile_bytes": meta.get("weight_refill_tile_bytes", 0),
            "weight_sram_energy_nj": meta.get("weight_sram_energy_nj", 0.0),
            "weight_sram_cycles": meta.get("weight_sram_cycles", 0),
            "feature_local_sram_energy_nj": meta.get("feature_local_sram_energy_nj", 0.0),
            "feature_local_sram_cycles": meta.get("feature_local_sram_cycles", 0),
            "im2col_total_cycles": meta.get("im2col_total_cycles", 0),
            "im2col_input_read_bytes": meta.get("im2col_input_read_bytes", 0),
            "im2col_patch_write_bytes": meta.get("im2col_patch_write_bytes", 0),
            "im2col_matrix_read_bytes": meta.get("im2col_matrix_read_bytes", 0),
            "im2col_ifmap_source_tensor_bytes": meta.get("im2col_ifmap_source_tensor_bytes", 0),
            "im2col_ifmap_materialized_bytes": meta.get("im2col_ifmap_materialized_bytes", 0),
            "scalesim_ifmap_effective_bytes": meta.get("scalesim_ifmap_effective_bytes", 0),
        }
        # 每层激活 shape，便于核对 SCALE-Sim 周期
        row["activation_input_shape"] = meta.get("activation_input_shape", "")
        row["activation_output_shape"] = meta.get("activation_output_shape", "")
        row["systolic_array_rows"] = meta.get("systolic_array_rows")
        row["systolic_array_cols"] = meta.get("systolic_array_cols")
        if op in ("Conv1d", "Conv2d"):
            row["num_points"] = meta.get("num_points")
            row["in_channels"] = meta.get("in_channels")
            row["out_channels"] = meta.get("out_channels")
            row["kernel_size"] = meta.get("kernel_size")
        else:
            row["tokens"] = meta.get("tokens")
            row["in_features"] = meta.get("in_features")
            row["out_features"] = meta.get("out_features")
        scalesim_layer_details.append(row)

    # 按执行顺序生成完整硬件算子序列（与 constants 中键名一致，便于导出 operator_sequence.txt）
    events_sorted = sorted(sim_result.events, key=lambda e: (e.cycle, getattr(e, "event_id", 0)))
    operator_sequence: List[Dict[str, Any]] = []
    for idx, e in enumerate(events_sorted):
        meta = getattr(e, "metadata", None) or {}
        row = {
            KEY_IDX: idx,
            KEY_CYCLE: e.cycle,
            KEY_END_CYCLE: e.cycle + e.duration - 1,
            KEY_DURATION: e.duration,
            KEY_OPERATOR: (e.operator or "").strip() or "other",
            KEY_STAGE: (e.stage or "").strip(),
            KEY_OPERATION: (e.operation or "").strip(),
            KEY_CIRCUIT_MODULE: (e.circuit_module or "").strip(),
            KEY_LAYER_NAME: meta.get("layer_name", ""),
            KEY_ENERGY_NJ: round(e.energy, 4),
        }
        for k, v in meta.items():
            if k not in row:
                row[k] = v
        operator_sequence.append(row)

    def _buffer_area_um2(buf_cfg: Any) -> float:
        if buf_cfg is None:
            return 0.0
        area = float(getattr(buf_cfg, "area", 0.0) or 0.0)
        if area <= 0.0:
            return 0.0
        # InterLayerBuffer / named SRAM buffers are CACTI-backed and documented in mm^2.
        return area * 1e6

    def _ball_query_area(hw_cfg: Any) -> float:
        """读取 sorting_circuit.ball_query 中对应模式（baseline/acc）的面积占位值。"""
        bq = getattr(getattr(hw_cfg, "sorting_circuit", None), "ball_query", None)
        if bq is None:
            return 0.0
        is_baseline = getattr(hw_cfg, "is_baseline", False)
        return float(getattr(bq, "area_baseline" if is_baseline else "area_acc", 0.0) or 0.0)

    # 逻辑模块面积（共享模块只计一次）：未从 NeuroSim 得到面积的算子用 hw_config 对应模块面积
    logic = getattr(hw_config, "logic_circuit", None)
    # KAN/KAT(GRF)：acc 用 rram_lut_macro.rram_lut.area，baseline 用 logic_circuit.grf_base_engine.area
    kan_area = 0.0
    if getattr(hw_config, "rram_lut_macro", None) and getattr(hw_config.rram_lut_macro, "rram_lut", None):
        kan_area = float(getattr(hw_config.rram_lut_macro.rram_lut, "area", 0) or 0)
    if kan_area == 0.0 and logic and getattr(logic, "grf_base_engine", None):
        kan_area = float(getattr(logic.grf_base_engine, "area", 0) or 0)
    if logic:
        acc_active_banks = _acc_active_banks_local()
        sram_area_um2 = _buffer_area_um2(getattr(hw_config, "inter_layer_buffer", None))
        for _buf_name in (
            "xyz_sram_buffer",
            "query_xyz_sram_buffer",
            "dist_sram_buffer",
            "self_distance_sram_buffer",
            "fps_min_distance_table_sram",
            "residual_sram_buffer",
            "state_sram_buffer",
            "nonlinear_workspace_sram",
            "nonlinear_param_sram",
            "temp_max_distance_sram",
            "topk_sram_buffer",
            "fps_index_sram_buffer",
            "sort_cache_sram",
            "im2col_buffer",
        ):
            scale = acc_active_banks if (getattr(hw_config, "is_acc", False) and _buf_name == "self_distance_sram_buffer") else 1
            sram_area_um2 += _buffer_area_um2(getattr(hw_config, _buf_name, None)) * scale
        logic_area_map = {
            "LayerNorm": float(getattr(getattr(logic, "layernorm", None), "area", 0) or 0),
            "ReLU": float(getattr(getattr(logic, "relu", None), "area", 0) or 0),
            "SPool": float(getattr(getattr(logic, "spool", None), "area", 0) or 0),
            "AdaptiveMaxPool1d": float(getattr(getattr(logic, "adaptive_max_pool", None), "area", 0) or 0),
            "SAMaxPool": float(getattr(getattr(logic, "sa_maxpool", None), "area", 0) or 0),
            "BN": float(getattr(getattr(logic, "batchnorm1d", None), "area", 0) or 0),
            "BallQuery": _ball_query_area(hw_config),
            "AffineNorm": 0.0,
            "Quant": float(getattr(getattr(logic, "quant_pipeline_8bit", None), "area", 0) or 0),
            "Dequant": float(getattr(getattr(logic, "dequant_pipeline", None), "area", 0) or 0),
            "ADD": float(getattr(getattr(logic, "fp32_add", None), "area", 0) or 0),
            "MUL": float(getattr(getattr(logic, "fp32_mul", None), "area", 0) or 0),
            "DIV": float(getattr(getattr(logic, "fp32_div", None), "area", 0) or 0),
            "InterpN3": float(getattr(getattr(logic, "interp_n3_engine", None), "area", 0) or 0),
            "KAN": kan_area,
            "SRAM": sram_area_um2,
        }
        if getattr(hw_config, "is_acc", False):
            macro = getattr(hw_config, "sram_cim_macro", None)
            array = getattr(macro, "array", None) if macro is not None else None
            shift_add = getattr(macro, "shift_add", None) if macro is not None else None
            merge_adder = getattr(macro, "dist2_merge_adder", None) if macro is not None else None
            logic_area_map["sram_cim_array"] = float(getattr(array, "area", 0.0) or 0.0) * acc_active_banks
            logic_area_map["shift_add"] = float(getattr(shift_add, "area", 0.0) or 0.0) * acc_active_banks
            logic_area_map["dist2_merge_adder"] = float(getattr(merge_adder, "area", 0.0) or 0.0) * acc_active_banks
        for op_name, area in logic_area_map.items():
            if op_name in operator_breakdown and operator_breakdown[op_name].get("area_um2", 0) == 0:
                operator_breakdown[op_name]["area_um2"] = area
    total_area_um2 = sum(v.get("area_um2", 0) for v in operator_breakdown.values())

    # 确保所有条目有 area_um2 键（用于报告）
    for v in operator_breakdown.values():
        v.setdefault("area_um2", 0.0)

    # 统一内存事务接口：将不同层级内存交互映射为 src/dst/kind，便于复用分析
    memory_txns = extract_memory_transactions(operator_sequence)
    memory_txn_summary = summarize_memory_transactions(memory_txns)
    pointcim_distance_summary = _summarize_pointcim_distance_events(sim_result.events)

    out: Dict[str, Any] = {
        "total_cycles": total_cycles,           # 等效基础时钟周期 = ceil(latency_ns / base_clock)
        "scheduler_raw_cycles": _scheduler_raw_cycles,  # 调度器混合域原始周期（仅供调试）
        "total_energy_nJ": total_energy,
        "total_latency_ns": latency_critical_path_ns,
        "total_latency_sum_ns": latency_sum_ns,
        "total_latency_critical_path_ns": latency_critical_path_ns,
        "latency_ns": latency_ns,
        "latency_ms": latency_ms,
        "latency_sum_ns": latency_sum_ns,
        "latency_sum_ms": latency_sum_ns / 1e6,
        "latency_critical_path_ns": latency_critical_path_ns,
        "latency_critical_path_ms": latency_critical_path_ns / 1e6,
        "total_cycles_sum_equiv": total_cycles_sum_equiv,
        "avg_power_mW": avg_power,
        "peak_power_mW": sim_result.peak_power,
        "total_events": len(sim_result.events),
        "layer_results": layer_results,
        "sim_result": sim_result,
        "operator_breakdown": operator_breakdown,
        "operator_cycle_detail": operator_cycle_detail,
        "dist_chain_analysis": dist_chain_analysis,
        "scalesim_layer_details": scalesim_layer_details,
        "operator_sequence": operator_sequence,
        "memory_runtime_summary": (memory_runtime_context.summary() if memory_runtime_context is not None else None),
        "feature_residency_plan": [entry.to_dict() for entry in feature_residency_plan],
        "memory_transactions": [tx.to_dict() for tx in memory_txns],
        "memory_transactions_summary": [
            {"src": src, "dst": dst, "kind": kind, "count": cnt}
            for (src, dst, kind), cnt in sorted(memory_txn_summary.items(), key=lambda x: (-x[1], x[0]))
        ],
        "total_area_um2": total_area_um2,
        "pointcim_distance_summary": pointcim_distance_summary,
    }
    if activation_shapes is not None:
        out["activation_shapes"] = [s.to_dict() for s in activation_shapes]
    return out


# =============================================================================
# 结果打印
# =============================================================================

def print_simulation_summary(
    results: Dict[str, Any],
    hw_config: HardwareConfig,
    network_name: str,
) -> None:
    """在控制台打印仿真汇总：延迟、能耗、平均/峰值功率、事件数等。"""
    log_section("仿真结果")
    log_info(f"  网络:       {network_name}")
    log_info(
        f"  加速器:     {hw_config.accelerator_name} "
        f"({hw_config.accelerator_type.name})"
    )
    log_info(
        f"  时钟:       {hw_config.clock_period} ns "
        f"({1000 / hw_config.clock_period:.0f} MHz)"
    )
    log_separator("-")
    log_info(
        f"  延迟:       {results['total_cycles']:,} 周期(等效基础时钟) "
        f"({results['latency_ms']:.3f} ms)"
    )
    log_info(
        f"  能耗:       {results['total_energy_nJ']:.2f} nJ "
        f"({results['total_energy_nJ'] / 1000:.4f} uJ)"
    )
    log_info(f"  平均功率:   {results['avg_power_mW']:.2f} mW")
    log_info(f"  峰值功率:   {results['peak_power_mW']:.2f} mW")
    log_info(f"  事件数:     {results['total_events']:,}")
    log_separator()


def print_event_statistics(
    sim_result: Any,
    hw_config: HardwareConfig,
) -> None:
    """打印按算子、按阶段、按功率的详细事件统计。"""
    stats = getattr(sim_result, "statistics", {})

    if "events_by_operator" in stats:
        log_info("\n  按算子事件数:")
        for op_name, count in sorted(stats["events_by_operator"].items()):
            log_info(f"    {op_name}: {count:,}")

    if "events_by_stage" in stats:
        log_info("\n  按阶段事件数:")
        for stage_name, count in sorted(stats["events_by_stage"].items()):
            log_info(f"    {stage_name}: {count:,}")

    if "power" in stats:
        log_info("\n  功率统计:")
        log_info(f"    平均: {stats['power']['average_mW']:.2f} mW")
        log_info(f"    峰值: {stats['power']['peak_mW']:.2f} mW")
        log_info(f"    最小: {stats['power']['min_mW']:.2f} mW")


# =============================================================================
# Breakdown 与 PPA 结果
# =============================================================================

def _attach_breakdown_and_ppa(
    results: Dict[str, Any],
    hw_config: HardwareConfig,
) -> None:
    """
    根据仿真事件列表计算按层/按算子/按模块的 breakdown，并写入 results。

    会向 results 中写入:
    - breakdown: 原始 breakdown 字典（by_layer / by_operator / by_module，含 cycles、energy_nj、event_count）
    - ppa_result: PPAResult 实例，含总延迟/能耗及 latency_breakdown、energy_breakdown，供 Reporter 输出。
    """
    def _buffer_area_um2_local(buf_cfg: Any) -> float:
        if buf_cfg is None:
            return 0.0
        area = float(getattr(buf_cfg, "area", 0.0) or 0.0)
        return area * 1e6 if area < 1.0 else area

    def _infer_acc_input_points_from_results() -> int:
        max_points = 0
        for detail in operator_cycle_detail:
            try:
                pts = int(detail.get("num_points", 0) or 0)
            except Exception:
                pts = 0
            if pts > max_points:
                max_points = pts
        if max_points > 0:
            return max_points
        return 1024

    def _acc_active_banks_for_breakdown() -> int:
        if not getattr(hw_config, "is_acc", False):
            return 1
        macro = getattr(hw_config, "sram_cim_macro", None)
        array = getattr(macro, "array", None) if macro is not None else None
        cim_parallel = max(1, int(getattr(hw_config, "cim_parallelism", 128) or 128))
        array_rows = max(cim_parallel, int(getattr(array, "rows", cim_parallel) or cim_parallel))
        bank_capacity_points = max(cim_parallel, array_rows)
        input_points_local = max(1, _infer_acc_input_points_from_results())
        return max(1, int(math.ceil(input_points_local / float(bank_capacity_points))))

    events = getattr(results.get("sim_result"), "events", [])
    clock_ns = hw_config.clock_period
    breakdown = analyze_simulation_events_breakdown(events, clock_ns)

    operator_breakdown = results.get("operator_breakdown") or {}
    operator_cycle_detail = results.get("operator_cycle_detail") or []

    for op_name, stats in (breakdown.get("by_operator") or {}).items():
        stats.setdefault("area_um2", 0.0)
        op_area = float((operator_breakdown.get(op_name) or {}).get("area_um2", 0.0) or 0.0)
        if op_area > 0.0 and float(stats.get("area_um2", 0.0) or 0.0) <= 0.0:
            stats["area_um2"] = op_area

    layer_area_map: Dict[str, float] = {}
    for detail in operator_cycle_detail:
        layer = str(detail.get("layer_name") or "unknown")
        area_u = float(detail.get("area_um2", 0.0) or 0.0)
        if area_u > 0.0:
            layer_area_map[layer] = layer_area_map.get(layer, 0.0) + area_u
    for layer_name, stats in (breakdown.get("by_layer") or {}).items():
        stats.setdefault("area_um2", 0.0)
        if float(stats.get("area_um2", 0.0) or 0.0) <= 0.0:
            stats["area_um2"] = float(layer_area_map.get(layer_name, 0.0) or 0.0)

    acc_active_banks = _acc_active_banks_for_breakdown()
    module_area_map: Dict[str, float] = {
        "inter_layer_buffer": _buffer_area_um2_local(getattr(hw_config, "inter_layer_buffer", None)),
        "xyz_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "xyz_sram_buffer", None)),
        "query_xyz_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "query_xyz_sram_buffer", None)),
        "dist_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "dist_sram_buffer", None)),
        "self_distance_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "self_distance_sram_buffer", None)) * (acc_active_banks if getattr(hw_config, "is_acc", False) else 1),
        "fps_min_distance_table_sram": _buffer_area_um2_local(getattr(hw_config, "fps_min_distance_table_sram", None)),
        "residual_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "residual_sram_buffer", None)),
        "state_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "state_sram_buffer", None)),
        "nonlinear_workspace_sram": _buffer_area_um2_local(getattr(hw_config, "nonlinear_workspace_sram", None)),
        "nonlinear_param_sram": _buffer_area_um2_local(getattr(hw_config, "nonlinear_param_sram", None)),
        "temp_max_distance_sram": _buffer_area_um2_local(getattr(hw_config, "temp_max_distance_sram", None)),
        "topk_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "topk_sram_buffer", None)),
        "fps_index_sram_buffer": _buffer_area_um2_local(getattr(hw_config, "fps_index_sram_buffer", None)),
        "sort_cache_sram": _buffer_area_um2_local(getattr(hw_config, "sort_cache_sram", None)),
        "im2col_buffer": _buffer_area_um2_local(getattr(hw_config, "im2col_buffer", None)),
        "systolic_array": float(getattr(getattr(hw_config, "systolic_array", None), "area", 0.0) or 0.0),
        "grf_base_engine": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "grf_base_engine", None), "area", 0.0) or 0.0),
        "relu": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "relu", None), "area", 0.0) or 0.0),
        "spool": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "spool", None), "area", 0.0) or 0.0),
        "layernorm": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "layernorm", None), "area", 0.0) or 0.0),
        "adaptive_max_pool": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "adaptive_max_pool", None), "area", 0.0) or 0.0),
        "affine_norm": float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "batchnorm1d", None), "area", 0.0) or 0.0),
        "min_distance_table": float(getattr(getattr(getattr(hw_config, "sorting_circuit", None), "min_distance_table", None), "area", 0.0) or 0.0),
        "threshold_gen": float(getattr(getattr(getattr(hw_config, "sorting_circuit", None), "threshold_gen", None), "area", 0.0) or 0.0),
        "dist2_engine": float(getattr(getattr(hw_config, "baseline_dist2_engine", None), "area", 0.0) or 0.0),
        "sram_cim_array": float(getattr(getattr(getattr(hw_config, "sram_cim_macro", None), "array", None), "area", 0.0) or 0.0) * (acc_active_banks if getattr(hw_config, "is_acc", False) else 1),
        "shift_add": float(getattr(getattr(getattr(hw_config, "sram_cim_macro", None), "shift_add", None), "area", 0.0) or 0.0) * (acc_active_banks if getattr(hw_config, "is_acc", False) else 1),
        "dist2_merge_adder": float(getattr(getattr(getattr(hw_config, "sram_cim_macro", None), "dist2_merge_adder", None), "area", 0.0) or 0.0) * (acc_active_banks if getattr(hw_config, "is_acc", False) else 1),
        "rram_lut": float(getattr(getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut", None), "pwl_area", 0.0) or 0.0),
        "rram_lut_speculative": float(getattr(getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut_speculative", None), "pwl_area", 0.0) or 0.0),
        "coarse_filter": float(getattr(getattr(getattr(hw_config, "sorting_circuit", None), "coarse_filter", None), "area", 0.0) or 0.0),
        "fine_filter": float(getattr(getattr(getattr(hw_config, "sorting_circuit", None), "fine_filter", None), "area", 0.0) or 0.0),
    }
    for mod_name, stats in (breakdown.get("by_module") or {}).items():
        stats.setdefault("area_um2", 0.0)
        mod_area = float(module_area_map.get(mod_name, 0.0) or 0.0)
        if mod_area > 0.0 and float(stats.get("area_um2", 0.0) or 0.0) <= 0.0:
            stats["area_um2"] = mod_area

    results["breakdown"] = breakdown

    latency_breakdown = {
        "by_layer": {k: v["cycles"] for k, v in breakdown["by_layer"].items()},
        "by_operator": {k: v["cycles"] for k, v in breakdown["by_operator"].items()},
        "by_module": {k: v["cycles"] for k, v in breakdown["by_module"].items()},
    }
    energy_breakdown = {
        "by_layer": {k: v["energy_nj"] for k, v in breakdown["by_layer"].items()},
        "by_operator": {k: v["energy_nj"] for k, v in breakdown["by_operator"].items()},
        "by_module": {k: v["energy_nj"] for k, v in breakdown["by_module"].items()},
    }

    results["ppa_result"] = create_ppa_result(
        latency_cycles=results["total_cycles"],
        power_mW=results["avg_power_mW"],
        area_mm2=0.0,
        clock_period_ns=clock_ns,
        latency_breakdown=latency_breakdown,
        energy_breakdown=energy_breakdown,
    )
    # Use the real end-to-end totals from the simulator rather than reconstructing
    # them from a single clock or average-power approximation.
    ppa = results["ppa_result"]
    latency_ns = float(results.get("latency_ns", 0.0) or 0.0)
    if latency_ns > 0:
        ppa.latency_ns = latency_ns
        ppa.latency_ms = latency_ns / 1e6
    total_energy_nj = float(results.get("total_energy_nJ", 0.0) or 0.0)
    if total_energy_nj > 0:
        ppa.energy_nJ = total_energy_nj
        ppa.energy_uJ = total_energy_nj / 1000.0


def _write_report_to_file(
    results: Dict[str, Any],
    output_path: str,
    network_name: str,
    accelerator_name: str,
) -> None:
    """
    将 PPA 结果与 breakdown 写入文件（JSON 或 CSV，由扩展名决定）。

    Args:
        results: 必须包含 ppa_result 与可选的 breakdown
        output_path: 输出文件路径
        network_name: 网络名称，写入报告元数据
        accelerator_name: 加速器名称，写入报告元数据
    """
    ppa = results.get("ppa_result")
    if ppa is None:
        return

    reporter = Reporter(title="PointKAN PPA 报告")
    reporter.set_results(ppa)
    reporter.add_metadata("network", network_name)
    reporter.add_metadata("accelerator", accelerator_name)
    if results.get("breakdown"):
        reporter.add_section("breakdown_detail", results["breakdown"])
    if results.get("activation_shapes"):
        reporter.add_section("activation_shapes", results["activation_shapes"])

    if output_path.lower().endswith(".csv"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(reporter.generate_csv_header() + "\n")
            f.write(reporter.generate_csv_row() + "\n")
        log_info(f"\nCSV 报告已保存: {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(reporter.generate_json_report())
        log_info(f"\nJSON 报告已保存: {output_path}")


# =============================================================================
# 主流程：演示与指定网络仿真
# =============================================================================

def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    """
    运行演示流程：PointKAN + 指定加速器，默认打印每层信息并汇总 PPA。
    点云输入为 N×3（N=input_points），并启用层间 buffer 与激活 shape 显示（若 --show_shapes）。
    """
    log_section("演示: PointKAN on baseline")
    hw_config = get_accelerator_config(args.accelerator)
    set_digital_clock_freq_hz(hw_config, getattr(args, "logic_clk_freq", None))
    _apply_neurosim_cli_overrides(hw_config, args)
    hw_config.ns_simulation_mode = str(getattr(args, "ns_simulation_mode", "event") or "event")
    hw_config.enable_w8a8_quant_dequant = getattr(args, "enable_w8a8_quant_dequant", False)
    hw_config.bn_fuse_with_conv_linear = bool(getattr(args, "bn_en", 1))
    qdq_parallelism = getattr(args, "logic_par_quant_dequant", None)
    if getattr(hw_config, "is_acc", False) and getattr(hw_config, "enable_w8a8_quant_dequant", False) and qdq_parallelism is None:
        qdq_parallelism = 64
    set_logic_parallelism(
        hw_config,
        layernorm=getattr(args, "logic_par_layernorm", None),
        relu=getattr(args, "logic_par_relu", None),
        spool=getattr(args, "logic_par_spool", None),
        quant_dequant=qdq_parallelism,
    )
    layers = load_network("pointkan", args.input_points, args.num_classes)
    activation_shapes = compute_activation_shapes(layers, args.input_points, input_channels=3)
    real_conv_linear_list = getattr(args, "real_conv_linear_list", None)

    log_info(f"网络: PointKAN（{len(layers)} 层）")
    log_info(f"点云输入: N×3，N={args.input_points}（每样本 {args.input_points * 3 * 4} B）")
    log_info(f"加速器: {hw_config.accelerator_name}")
    log_info(f"CIM 并行度: {hw_config.cim_parallelism}")
    log_info("")

    results = simulate_network(
        layers,
        hw_config,
        show_layers=True,
        activation_shapes=activation_shapes,
        enable_layer_buffer=not getattr(args, "no_layer_buffer", False),
        show_shapes=getattr(args, "show_shapes", False),
    )
    _attach_breakdown_and_ppa(results, hw_config)
    print_simulation_summary(results, hw_config, "PointKAN")

    if args.show_events:
        print_event_statistics(results["sim_result"], hw_config)

    return results


def run_network_simulation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    运行指定网络的完整仿真：点云 N×3 输入、层间 buffer、可选 NeuroSim 全网络注入、
    每层打印与激活 shape（--show_shapes）、事件统计与报告输出。
    """
    network_name = args.network
    log_section(f"网络仿真: {network_name}")

    hw_config = get_accelerator_config(args.accelerator)
    set_digital_clock_freq_hz(hw_config, getattr(args, "logic_clk_freq", None))
    hw_config.ns_simulation_mode = str(getattr(args, "ns_simulation_mode", "event") or "event")
    hw_config.enable_w8a8_quant_dequant = getattr(args, "enable_w8a8_quant_dequant", False)
    hw_config.bn_fuse_with_conv_linear = bool(getattr(args, "bn_en", 1))
    if getattr(args, "cim_backend", None):
        hw_config.cim_backend = args.cim_backend
    _apply_neurosim_cli_overrides(hw_config, args)
    _enforce_pointcim_neurosim_backend(hw_config)
    if getattr(args, "use_scalesim_systolic", False) and getattr(hw_config, "is_baseline", False):
        hw_config.use_scalesim_for_systolic = True
        if getattr(args, "scalesim_root", None):
            hw_config.scalesim_root = args.scalesim_root
        if getattr(args, "scalesim_config", None):
            hw_config.scalesim_config_path = args.scalesim_config
    hw_config.knn_conv_pipeline_mode = str(getattr(args, "knn_conv_pipeline_mode", "disabled") or "disabled")
    hw_config.pointcim_sa_conv_pipeline = bool(getattr(args, "pointcim_sa_conv_pipeline", False))
    _fc_slices = getattr(args, "fc_num_slices", 1) or 1
    if _fc_slices > 1 or getattr(args, "fc_coalescing_factor", 0.0):
        hw_config.feature_computing = FeatureComputingConfig(
            num_slices=_fc_slices,
            cores_per_slice=getattr(args, "fc_cores_per_slice", 16) or 16,
            muls_per_core=getattr(args, "fc_muls_per_core", 1024) or 1024,
            coalescing_factor=getattr(args, "fc_coalescing_factor", 0.0) or 0.0,
        )
    qdq_parallelism = getattr(args, "logic_par_quant_dequant", None)
    if getattr(hw_config, "is_acc", False) and getattr(hw_config, "enable_w8a8_quant_dequant", False) and qdq_parallelism is None:
        qdq_parallelism = 64
    set_logic_parallelism(
        hw_config,
        layernorm=getattr(args, "logic_par_layernorm", None),
        relu=getattr(args, "logic_par_relu", None),
        spool=getattr(args, "logic_par_spool", None),
        quant_dequant=qdq_parallelism,
    )
    _net_input_channels = getattr(args, "input_channels", None) or (
        12 if network_name == "pointnet2-sem-ssg" else 3
    )
    _net_kwargs: Dict[str, Any] = {}
    if network_name == "pointnet2-sem-ssg":
        _net_kwargs["input_channels"] = int(_net_input_channels)
    _enable_dr = getattr(hw_config, "distance_matrix_reuse", False)
    layers = load_network(
        network_name, args.input_points, args.num_classes,
        enable_distance_reuse=_enable_dr, **_net_kwargs,
    )
    activation_shapes = compute_activation_shapes(
        layers, args.input_points, input_channels=int(_net_input_channels)
    )
    real_conv_linear_list = getattr(args, "real_conv_linear_list", None)

    log_info(f"网络: {network_name}（{len(layers)} 层）")
    log_info(
        f"点云输入: N×{_net_input_channels}，N={args.input_points}"
        f"（每样本 {args.input_points * int(_net_input_channels) * 4} B）"
    )
    log_info(
        f"加速器: {hw_config.accelerator_name} "
        f"({hw_config.accelerator_type.name})"
    )
    log_info(f"CIM 并行度: {hw_config.cim_parallelism}")
    log_info(f"Bit-serial 周期: {hw_config.bit_serial_cycles}")
    # 层间 buffer 与量化：来自 hw_config（非 load_network 的 layers）
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is not None:
        log_info(
            f"层间 buffer: size={buf.size_bytes} B, bandwidth={buf.bandwidth_bytes_per_cycle} B/cycle, "
            f"spill_to_dram={getattr(buf, 'spill_to_dram', False)}, "
            f"pipeline_mode={getattr(buf, 'pipeline_mode', 'without_pipeline')}, "
            f"double_buffering={getattr(buf, 'double_buffering', False)}"
        )
    else:
        log_info("层间 buffer: 未配置（inter_layer_buffer 为空）")
    log_info(
        f"层间量化/反量化: {'启用 (FP32↔INT8 计入 cycle/功耗)' if getattr(hw_config, 'enable_w8a8_quant_dequant', False) else '未启用 (默认，假定 W8A8 已就绪)'}"
    )
    log_info("")

    # 为 conv1d/linear 启用 NeuroSim 全网络（Tile/NoC 级）延迟与能耗
    cim_full_context = None
    use_neurosim_full = bool(getattr(args, "neurosim_full", False))
    if getattr(hw_config, "is_acc", False) and use_neurosim_full:
        from pointkan_sim.integration.neurosim_full import run_neurosim_full

        conv_linear_list = list(real_conv_linear_list or [])
        if conv_linear_list:
            neurosim_root = getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")
            _ns_cap = getattr(args, "neurosim_max_in_vectors", None)
            _ns_cap = int(_ns_cap) if _ns_cap is not None and int(_ns_cap) > 0 else None
            _timeout = getattr(args, "neurosim_timeout_s", None)
            _timeout = int(_timeout) if _timeout is not None and int(_timeout) > 0 else None
            _clk = getattr(args, "neurosim_clk_freq", None)
            try:
                if _clk is not None and float(_clk) <= 0:
                    _clk = None
            except (TypeError, ValueError):
                _clk = None
            if _clk is None:
                _clk = getattr(hw_config, "neurosim_clk_freq_hz", None)
            neurosim_result = run_neurosim_full(
                conv_linear_list,
                neurosim_root=neurosim_root,
                timeout_s=_timeout,
                clk_freq_hz=_clk,
                technode=getattr(args, "neurosim_technode", None),
                adc_clk_freq_hz=getattr(args, "neurosim_adc_clk_freq", None),
                sar_column_power_w=resolve_neurosim_sar_column_power_w(
                    getattr(args, "neurosim_sar_column_power_w", None),
                    getattr(args, "neurosim_sar_column_power_uw", None),
                ),
                num_col_muxed=getattr(hw_config, "neurosim_num_col_muxed", None),
                clk_mode=str(getattr(hw_config, "neurosim_clk_mode", "capped") or "capped"),
                max_in_vectors=_ns_cap,
            )
            if neurosim_result.success:
                neurosim_clock_ns = getattr(neurosim_result, "clock_period_ns", None) or 0.0
                cim_full_context = build_cim_full_context_from_neurosim_lists(
                    conv_linear_list,
                    layer_latency_ns=list(neurosim_result.layer_latency_ns),
                    layer_energy_nj=list(neurosim_result.layer_energy_nj),
                    clock_period_ns=float(neurosim_clock_ns) if neurosim_clock_ns > 0 else None,
                    layer_area_um2=list(getattr(neurosim_result, "layer_area_um2", []) or []),
                )
                log_info("已启用 NeuroSim 全网络（Tile/NoC）用于 conv1d/linear。")
            else:
                log_info(f"NeuroSim 全网络未启用: {neurosim_result.error}")
        else:
            log_info("NeuroSim 全网络未启用: real_conv_linear_list 为空，无法构造 full-network context。")
    log_info("")

    results = simulate_network(
        layers,
        hw_config,
        show_layers=args.show_layers,
        cim_full_context=cim_full_context,
        activation_shapes=activation_shapes,
        enable_layer_buffer=not getattr(args, "no_layer_buffer", False),
        show_shapes=getattr(args, "show_shapes", False),
        repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    print_simulation_summary(results, hw_config, network_name)

    if args.show_events:
        print_event_statistics(results["sim_result"], hw_config)

    _attach_breakdown_and_ppa(results, hw_config)

    # 模块活动时间线：不同 cycle 下有哪些模块在工作（前段算完可供给后段）
    events = getattr(results.get("sim_result"), "events", [])
    if events:
        stride = getattr(args, "timeline_stride", 500)
        results["module_activity_timeline"] = build_module_activity_timeline(
            events, cycle_stride=stride
        )
        results["module_activity_segments"] = build_module_activity_segments(events)
        sim_result = results.get("sim_result")
        block_intervals = getattr(sim_result, "block_intervals", None) or build_block_intervals(events)
        results["block_intervals"] = block_intervals
    out_tl = getattr(args, "output_timeline_csv", None)
    if out_tl and results.get("module_activity_timeline"):
        with open(out_tl, "w", encoding="utf-8") as f:
            f.write("\n".join(timeline_to_csv_rows(results["module_activity_timeline"])))
        log_info(f"模块活动时间线已写入: {out_tl}")
    out_png = getattr(args, "output_timeline_png", None)
    if out_png and results.get("block_intervals"):
        saved = plot_timeline(
            results["block_intervals"],
            max_cycle=int(getattr(args, "timeline_max_cycle", 2000) or 2000),
            output_path=out_png,
        )
        if saved:
            log_info(f"Cycle-level timeline PNG已写入: {saved}")
            json_path = str(Path(out_png).with_suffix(".json"))
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(block_intervals_to_json_rows(results["block_intervals"]))
            log_info(f"Block intervals JSON已写入: {json_path}")

    return results


# =============================================================================
# 程序入口（仅在被直接运行时执行，与文件顶部的路径修正是两处不同的 __main__ 用途）
# =============================================================================
# 说明：
# - 顶部 "if __name__ == '__main__' and __package__ is None"：在「直接运行本文件」时
#   先修正 sys.path，保证后续 import pointkan_sim 能找到包，只执行一次且必须在 import 之前。
# - 此处 "if __name__ == '__main__'"：真正的入口，解析参数、选择演示或指定网络、写报告并退出。
# =============================================================================

def main() -> int:
    """
    主入口：解析参数、初始化日志、执行演示或网络仿真，可选写报告并返回退出码。
    """
    args = parse_args()

    # 统一输出目录：创建 sim_log 并将日志、报告、时间线 CSV 等写入该目录
    Path(args.sim_log_dir).mkdir(parents=True, exist_ok=True)
    network_tag = _sanitize_log_tag(getattr(args, "network", "") or "sim")
    _log_path = Path(args.sim_log_dir) / f"simulation_{network_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(level=LogLevel[args.log_level], log_file=str(_log_path))

    # 相对路径的报告/时间线输出统一落到 sim_log_dir
    if getattr(args, "output", None) and not os.path.isabs(args.output):
        args.output = str(Path(args.sim_log_dir) / os.path.basename(args.output))
    if getattr(args, "output_timeline_csv", None) and not os.path.isabs(args.output_timeline_csv):
        args.output_timeline_csv = str(Path(args.sim_log_dir) / os.path.basename(args.output_timeline_csv))
    if getattr(args, "output_timeline_png", None) and not os.path.isabs(args.output_timeline_png):
        args.output_timeline_png = str(Path(args.sim_log_dir) / os.path.basename(args.output_timeline_png))

    if args.list_accelerators:
        print("可用加速器配置:")
        for name in list_accelerator_configs():
            cfg = get_accelerator_config(name)
            print(
                f"  {name:<15s}  [{cfg.accelerator_type.name}, "
                f"CIM parallelism={cfg.cim_parallelism}, "
                f"modules: {', '.join(cfg.get_all_modules().keys())}]"
            )
        return 0

    log_section("PointKAN CIM PPA 仿真器")
    log_info(f"支持的算子类型: {list_supported_op_types()}")
    log_info("")

    try:
        if args.demo:
            results = run_demo(args)
        elif args.network:
            results = run_network_simulation(args)
        else:
            log_info("未指定操作，运行演示...")
            results = run_demo(args)

        if args.output and results and results.get("ppa_result") is not None:
            _write_report_to_file(
                results,
                args.output,
                network_name=args.network or "pointkan",
                accelerator_name=get_accelerator_config(args.accelerator).accelerator_name,
            )

        return 0

    except Exception as e:
        log_info(f"错误: 仿真失败 - {e}")
        import traceback
        traceback.print_exc()
        return 1


# 仅当本文件被直接运行（而非 import）时执行 main，并以其返回值退出进程
if __name__ == "__main__":
    sys.exit(main())
