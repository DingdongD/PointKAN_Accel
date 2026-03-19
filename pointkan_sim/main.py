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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    HardwareConfig,
    set_digital_clock_freq_hz,
    set_logic_parallelism,
)
from pointkan_sim.network.pointkan_net import (
    LayerDef,
    OperatorRef,
    create_pointkan_classification,
)
from pointkan_sim.network.pointkan_elite_net import create_pointkan_elite_classification
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
from pointkan_sim.integration.mnsim_netstruct import (
    build_netstruct_from_layers,
    run_mnsim_full,
)
from pointkan_sim.simulator.scheduler import Scheduler
from pointkan_sim.report.reporter import Reporter, create_ppa_result
from pointkan_sim.analysis.breakdown import analyze_simulation_events_breakdown
from pointkan_sim.analysis.timeline import (
    build_module_activity_timeline,
    build_module_activity_segments,
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
            "pointnet2-cls-ssg",
            "pointnet2-partseg-ssg",
            "pointnet2-sem-ssg",
        ],
        help="要仿真的网络",
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
        "--mnsim_full",
        action="store_true",
        help="对 conv1d/linear 使用 MNSIM 全 NetStruct（Tile/NoC 级）的延迟与能耗",
    )
    parser.add_argument(
        "--no_mnsim_full",
        action="store_true",
        help="禁用 MNSIM 全 NetStruct，仅使用 PE 级估算",
    )
    parser.add_argument(
        "--cim_backend",
        type=str,
        default=None,
        choices=["mnsim", "neurosim", "fallback"],
        help="Conv/Linear 单层估算后端：mnsim、neurosim（DNN_NeuroSim）、fallback；无全 NetStruct 时生效",
    )
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default=None,
        help="NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
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

    return parser.parse_args()


# =============================================================================
# 网络加载
# =============================================================================

def load_network(
    name: str,
    input_points: int = 1024,
    num_classes: int = 40,
    **kwargs,
) -> List[LayerDef]:
    """
    根据名称加载网络层定义（LayerDef 有序列表）。

    Args:
        name: 网络名称，支持:
            "pointkan"               — PointKAN 分类 (ModelNet40)
            "pointkan-elite"         — PointKAN-Elite 分类 (ModelNet40)
            "pointnet2-cls-ssg"      — PointNet2ClsSSG 分类 (ModelNet40)
            "pointnet2-partseg-ssg"  — PointNet2PartSegSSG 分割 (ShapeNetPart)
            "pointnet2-sem-ssg"      — PointNet2SemSSG 语义分割 (S3DIS)
        input_points: 输入点云点数
        num_classes:  分类/分割类别数
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
    if name == "pointnet2-cls-ssg":
        return create_pointnet2_cls_ssg(
            input_points=input_points,
            num_classes=num_classes,
        )
    if name == "pointnet2-partseg-ssg":
        return create_pointnet2_partseg_ssg(
            input_points=input_points,
            num_classes=num_classes,
        )
    if name == "pointnet2-sem-ssg":
        input_channels = kwargs.get("input_channels", 12)
        return create_pointnet2_sem_ssg(
            input_points=input_points,
            num_classes=num_classes,
            input_channels=input_channels,
        )
    raise ValueError(
        f"未知网络: {name}。支持: pointkan, pointkan-elite, "
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
        scheduler.event_queue.add_event(
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


def simulate_network(
    layers: List[LayerDef],
    hw_config: HardwareConfig,
    show_layers: bool = False,
    mnsim_context: Optional[Dict[str, Any]] = None,
    activation_shapes: Optional[List[ActivationShapeInfo]] = None,
    enable_layer_buffer: bool = True,
    show_shapes: bool = False,
    repo_root: Optional[str] = None,
    real_conv_linear_list: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    对整网进行周期级仿真：按层顺序派发算子、生成事件并送入调度器执行。

    - 若提供 mnsim_context，conv1d/linear 使用 MNSIM 全 NetStruct 的延迟与能耗。
    - 若 enable_layer_buffer 且提供 activation_shapes，在每层首尾插入层间 buffer 读写事件
      （首层前：点云 N×3 从 DRAM 加载；层间：buffer 读/写，超出 buffer 部分按 DRAM spill 建模）。
    - 每个事件的 metadata 写入当前层名与层类型，供 breakdown 按层聚合及 operator_sequence 的 layer_name。

    Args:
        layers: 网络层定义列表（如 embedding → GAM-stage0 → …）
        hw_config: 硬件配置（时钟、CIM 并行度、层间 buffer、BN 融合等）
        show_layers: 是否在控制台打印每层周期与事件数
        mnsim_context: 可选 MNSIM 全 NetStruct 结果（mnsim_latency_ns / mnsim_energy_nj）
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

    # 层间 SRAM：通过 CACTI 仿真得到 cycle/energy（仅 CACTI 回填，无回退）；同时初始化持久 memory runtime context。
    if use_buffer:
        from pointkan_sim.integration.memory_model_adapter import fill_inter_layer_buffer_from_cacti
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
            if i > 0 and shape_info.input_bytes > size and spill:
                dram_transfers.append((shape_info.input_bytes - size, True))
    if use_buffer and dram_transfers and memory_runtime_context is not None:
        dram_results = memory_runtime_context.prime_batch_transfers(dram_transfers)
        dram_precomputed_iter = iter((r["cycles"], r["energy_nj"]) for r in dram_results)
    elif use_buffer and dram_transfers and memory_runtime_context is None:
        if not getattr(hw_config.dram, "dramsim3_bin", None) or not getattr(hw_config.dram, "dramsim3_config_ini", None):
            raise RuntimeError(
                "DRAM energy/cycle require DRAMSim3 simulation. "
                "hw_config.dram.dramsim3_bin and dramsim3_config_ini must be set when using layer buffer with DRAM."
            )
        from pointkan_sim.integration.memory_model_adapter import run_dram_trace_for_layer_buffer_dram
        dram_results = run_dram_trace_for_layer_buffer_dram(dram_transfers, hw_config, repo_root)
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
    # acc 模式下 FPS 仅在第一层（含 FPS 的层）执行，后续层复用第一层 FPS 结果；KNN 每层仍重新计算。
    first_layer_with_fps: Optional[int] = None
    if getattr(hw_config, "is_acc", False):
        for idx, layer in enumerate(layers):
            if any(op.op_type == "fps" for op in layer.operators):
                first_layer_with_fps = idx
                break
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
        if use_buffer and shape_info and i > 0 and shape_info.input_bytes > 0:
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
        dispatch_context = dict(mnsim_context) if mnsim_context else {}
        if real_conv_linear_list:
            dispatch_context["real_conv_linear_list"] = real_conv_linear_list
            dispatch_context["real_conv_linear_index"] = real_conv_linear_index
        dispatch_context["layer_index"] = i
        dispatch_context["memory_runtime_context"] = memory_runtime_context
        if getattr(hw_config, "is_acc", False) and first_layer_with_fps is not None:
            dispatch_context["first_layer_with_fps"] = first_layer_with_fps
        layer_events = dispatch_layer(layer.operators, hw_config, dispatch_context)
        if real_conv_linear_list:
            real_conv_linear_index = int(dispatch_context.get("real_conv_linear_index", real_conv_linear_index) or real_conv_linear_index)
        _add_events_to_scheduler(scheduler, layer_events, current_cycle, layer.name, layer.layer_type)
        layer_span = max((e.cycle + e.duration for e in layer_events), default=0)
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
    total_cycles = sim_result.total_cycles
    total_energy = sim_result.total_energy
    # 端到端延迟：按每个事件的 clock_period_ns 换算后累加（NeuroSim 与 1 GHz 模块混合时正确）
    total_latency_ns = sum(
        e.duration * getattr(e, "clock_period_ns", 1.0) for e in sim_result.events
    )
    latency_ns = total_latency_ns if total_latency_ns > 0 else (total_cycles * clock_period)
    latency_ms = latency_ns / 1e6
    avg_power = (total_energy / latency_ns * 1e3) if latency_ns > 0 else 0.0

    # 按算子类型聚合周期、能耗与延迟(ns)；Quant/Dequant/BN 单独成行（stage 区分）
    operator_breakdown: Dict[str, Dict[str, float]] = {}
    # 每个算子每次出现的 cycle 构成：(operator_display, layer_name) -> duration，用于打印明细
    operator_cycle_detail: List[Dict[str, Any]] = []
    for e in sim_result.events:
        stage = (e.stage or "").strip()
        if stage == "quant":
            op_name = "Quant"
        elif stage == "dequant":
            op_name = "Dequant"
        else:
            op_name = (e.operator or e.operation or "other").strip() or "other"
        clk_ns = getattr(e, "clock_period_ns", 1.0)
        if op_name not in operator_breakdown:
            operator_breakdown[op_name] = {"cycles": 0.0, "energy_nj": 0.0, "latency_ns": 0.0, "area_um2": 0.0}
        operator_breakdown[op_name]["cycles"] += e.duration
        operator_breakdown[op_name]["energy_nj"] += e.energy
        operator_breakdown[op_name]["latency_ns"] += e.duration * clk_ns
        meta = getattr(e, "metadata", None) or {}
        area_um2 = meta.get("area_um2")
        if area_um2 is not None and isinstance(area_um2, (int, float)):
            operator_breakdown[op_name]["area_um2"] = operator_breakdown[op_name].get("area_um2", 0) + float(area_um2)
        layer_name = meta.get("layer_name", "")
        area_um2 = meta.get("area_um2")
        area_um2_val = float(area_um2) if area_um2 is not None and isinstance(area_um2, (int, float)) else 0.0
        row = {
            "operator": op_name,
            "layer_name": layer_name,
            "duration": e.duration,
            "latency_ns": e.duration * clk_ns,
            "energy_nj": e.energy,
            "area_um2": area_um2_val,
        }
        # KAN/KAT 等带激活 shape 的算子：写入便于查看
        for k in ("num_points", "in_features", "out_features", "activation_input_shape", "activation_output_shape"):
            if k in meta and meta[k] is not None:
                row[k] = meta[k]
        operator_cycle_detail.append(row)

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
        }
        # 每层激活 shape，便于核对 SCALE-Sim 周期
        row["activation_input_shape"] = meta.get("activation_input_shape", "")
        row["activation_output_shape"] = meta.get("activation_output_shape", "")
        row["systolic_array_rows"] = meta.get("systolic_array_rows")
        row["systolic_array_cols"] = meta.get("systolic_array_cols")
        if op == "Conv1d":
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
        operator_sequence.append({
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
        })

    # 逻辑模块面积（共享模块只计一次）：未从 MNSIM/NeuroSim 得到面积的算子用 hw_config 对应模块面积
    logic = getattr(hw_config, "logic_circuit", None)
    # KAN/KAT(GRF)：acc 用 rram_lut_macro.rram_lut.area，baseline 用 logic_circuit.grf_base_engine.area
    kan_area = 0.0
    if getattr(hw_config, "rram_lut_macro", None) and getattr(hw_config.rram_lut_macro, "rram_lut", None):
        kan_area = float(getattr(hw_config.rram_lut_macro.rram_lut, "area", 0) or 0)
    if kan_area == 0.0 and logic and getattr(logic, "grf_base_engine", None):
        kan_area = float(getattr(logic.grf_base_engine, "area", 0) or 0)
    if logic:
        logic_area_map = {
            "LayerNorm": float(getattr(getattr(logic, "layernorm", None), "area", 0) or 0),
            "ReLU": float(getattr(getattr(logic, "relu", None), "area", 0) or 0),
            "SPool": float(getattr(getattr(logic, "spool", None), "area", 0) or 0),
            "AdaptiveMaxPool1d": float(getattr(getattr(logic, "adaptive_max_pool", None), "area", 0) or 0),
            "BN": float(getattr(getattr(logic, "batchnorm1d", None), "area", 0) or 0),
            "AffineNorm": 0.0,
            "Quant": float(getattr(getattr(logic, "quant_pipeline_8bit", None), "area", 0) or 0),
            "Dequant": float(getattr(getattr(logic, "dequant_pipeline", None), "area", 0) or 0),
            "ADD": float(getattr(getattr(logic, "fp32_add", None), "area", 0) or 0),
            "MUL": float(getattr(getattr(logic, "fp32_mul", None), "area", 0) or 0),
            "DIV": float(getattr(getattr(logic, "fp32_div", None), "area", 0) or 0),
            "KAN": kan_area,
        }
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

    out: Dict[str, Any] = {
        "total_cycles": total_cycles,
        "total_energy_nJ": total_energy,
        "total_latency_ns": total_latency_ns,
        "latency_ns": latency_ns,
        "latency_ms": latency_ms,
        "avg_power_mW": avg_power,
        "peak_power_mW": sim_result.peak_power,
        "total_events": len(sim_result.events),
        "layer_results": layer_results,
        "sim_result": sim_result,
        "operator_breakdown": operator_breakdown,
        "operator_cycle_detail": operator_cycle_detail,
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
        f"  延迟:       {results['total_cycles']:,} 周期 "
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
    events = getattr(results.get("sim_result"), "events", [])
    clock_ns = hw_config.clock_period
    breakdown = analyze_simulation_events_breakdown(events, clock_ns)
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
    hw_config.enable_w8a8_quant_dequant = getattr(args, "enable_w8a8_quant_dequant", False)
    hw_config.bn_fuse_with_conv_linear = bool(getattr(args, "bn_en", 1))
    set_logic_parallelism(
        hw_config,
        layernorm=getattr(args, "logic_par_layernorm", None),
        relu=getattr(args, "logic_par_relu", None),
        spool=getattr(args, "logic_par_spool", None),
        quant_dequant=getattr(args, "logic_par_quant_dequant", None),
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
    运行指定网络的完整仿真：点云 N×3 输入、层间 buffer、可选 MNSIM 全 NetStruct、
    每层打印与激活 shape（--show_shapes）、事件统计与报告输出。
    """
    network_name = args.network
    log_section(f"网络仿真: {network_name}")

    hw_config = get_accelerator_config(args.accelerator)
    set_digital_clock_freq_hz(hw_config, getattr(args, "logic_clk_freq", None))
    hw_config.enable_w8a8_quant_dequant = getattr(args, "enable_w8a8_quant_dequant", False)
    hw_config.bn_fuse_with_conv_linear = bool(getattr(args, "bn_en", 1))
    if getattr(args, "cim_backend", None):
        hw_config.cim_backend = args.cim_backend
    if getattr(args, "neurosim_root", None):
        hw_config.neurosim_root = args.neurosim_root
    if getattr(args, "use_scalesim_systolic", False) and getattr(hw_config, "is_baseline", False):
        hw_config.use_scalesim_for_systolic = True
        if getattr(args, "scalesim_root", None):
            hw_config.scalesim_root = args.scalesim_root
        if getattr(args, "scalesim_config", None):
            hw_config.scalesim_config_path = args.scalesim_config
    set_logic_parallelism(
        hw_config,
        layernorm=getattr(args, "logic_par_layernorm", None),
        relu=getattr(args, "logic_par_relu", None),
        spool=getattr(args, "logic_par_spool", None),
        quant_dequant=getattr(args, "logic_par_quant_dequant", None),
    )
    _net_input_channels = getattr(args, "input_channels", None) or (
        12 if network_name == "pointnet2-sem-ssg" else 3
    )
    _net_kwargs: Dict[str, Any] = {}
    if network_name == "pointnet2-sem-ssg":
        _net_kwargs["input_channels"] = int(_net_input_channels)
    layers = load_network(network_name, args.input_points, args.num_classes, **_net_kwargs)
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

    # 为 conv1d/linear 启用 MNSIM 全 NetStruct（Tile/NoC 级）延迟与能耗
    mnsim_context = None
    use_mnsim_full = getattr(args, "mnsim_full", False)
    no_mnsim_full = getattr(args, "no_mnsim_full", False)
    if use_mnsim_full and not no_mnsim_full:
        netstruct, conv_linear_order = build_netstruct_from_layers(layers)
        if netstruct:
            mnsim_result = run_mnsim_full(
                netstruct,
                conv_linear_order=conv_linear_order,
            )
            if mnsim_result.success:
                mnsim_context = {
                    "mnsim_latency_ns": mnsim_result.layer_latency_ns,
                    "mnsim_energy_nj": mnsim_result.layer_energy_nj,
                    "mnsim_index": 0,
                }
                log_info("已启用 MNSIM 全 NetStruct（Tile/NoC）用于 conv1d/linear。")
                # 每层 conv/linear 的 cycle、latency(ns)、energy(nJ) 详细日志
                clock_ns = getattr(hw_config, "clock_period", 1.0)
                order = mnsim_result.conv_linear_order or []
                for idx, (lat_ns, en_nj) in enumerate(
                    zip(mnsim_result.layer_latency_ns, mnsim_result.layer_energy_nj)
                ):
                    layer_name = order[idx][0] if idx < len(order) else f"layer_{idx}"
                    cycles = max(1, int(math.ceil(lat_ns / clock_ns)))
                    log_info(
                        f"  MNSIM [{layer_name}] latency={lat_ns:.1f} ns, "
                        f"cycles={cycles}, energy={en_nj:.2f} nJ"
                    )
            else:
                log_info(f"MNSIM 全 NetStruct 未启用: {mnsim_result.error}")
        else:
            log_info("MNSIM 全 NetStruct 未启用: 无 conv1d/linear 层。")
    log_info("")

    results = simulate_network(
        layers,
        hw_config,
        show_layers=args.show_layers,
        mnsim_context=mnsim_context,
        activation_shapes=activation_shapes,
        enable_layer_buffer=not getattr(args, "no_layer_buffer", False),
        show_shapes=getattr(args, "show_shapes", False),
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
    out_tl = getattr(args, "output_timeline_csv", None)
    if out_tl and results.get("module_activity_timeline"):
        with open(out_tl, "w", encoding="utf-8") as f:
            f.write("\n".join(timeline_to_csv_rows(results["module_activity_timeline"])))
        log_info(f"模块活动时间线已写入: {out_tl}")

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
    _log_path = Path(args.sim_log_dir) / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(level=LogLevel[args.log_level], log_file=str(_log_path))

    # 相对路径的报告/时间线输出统一落到 sim_log_dir
    if getattr(args, "output", None) and not os.path.isabs(args.output):
        args.output = str(Path(args.sim_log_dir) / os.path.basename(args.output))
    if getattr(args, "output_timeline_csv", None) and not os.path.isabs(args.output_timeline_csv):
        args.output_timeline_csv = str(Path(args.sim_log_dir) / os.path.basename(args.output_timeline_csv))

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
