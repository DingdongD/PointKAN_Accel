"""
算子派发器（Operator Dispatcher）
==================================

纯桥接：根据 op_type 字符串查找 handler，调用后得到 List[Event]。
不包含周期/功耗/事件的具体计算逻辑，计算细节在各算子类或 common_ops 中。

职责：
    1. op_type（如 conv1d, linear, fps, knn）→ 对应 handler
    2. 从 OperatorRef 取 params，构造 XxxParams，实例化 XxxOperator
    3. 调用 generate_events() 或直接由 handler 生成事件
    4. dispatch_layer：按层内算子顺序链式调用，累加 cycle 偏移

Conv1d/Linear 的 MNSIM 全 NetStruct 事件使用 constants 中的 OP_NAME_CONV1D/OP_NAME_LINEAR，
保证 operator_sequence.txt 中算子名统一为 PascalCase（Conv1d, Linear）。
"""

import math
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING

from ..constants import OP_NAME_CONV1D, OP_NAME_LINEAR
from ..simulator.event import Event
from .common_ops import (
    gen_conv1d_rram_cim_events,
    gen_linear_rram_cim_events,
    gen_quant_events,
    gen_dequant_events,
    gen_bn_events,
    gen_layernorm_events,
    gen_relu_events,
    gen_spool_events,
    gen_kan_rram_lut_events,
    gen_affine_norm_events,
    gen_adaptive_max_pool_events,
    gen_add_events,
    gen_mul_events,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig
    from ..network.pointkan_net import OperatorRef


# =============================================================================
# Per-op_type handlers
# =============================================================================
# Each handler signature:
#   (op_ref: OperatorRef, hw_config: HardwareConfig) -> List[Event]


def _gen_conv1d(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Conv1d：若有 mnsim_context 则用 MNSIM 全 NetStruct 的延迟/能耗生成事件（算子名 PascalCase）；否则走 PE 级 common_ops。"""
    if context and "mnsim_latency_ns" in context and "mnsim_energy_nj" in context:
        idx = context.get("mnsim_index", 0)
        lat_ns = context["mnsim_latency_ns"]
        en_nj = context["mnsim_energy_nj"]
        if idx < len(lat_ns) and idx < len(en_nj):
            # NeuroSim 使用自身时钟(如 100MHz=10ns)；未提供时用数字逻辑 1GHz
            clock_ns = context.get("mnsim_clock_period_ns") or getattr(hw_config, "clock_period", 1.0)
            clock_ns = float(clock_ns)
            cycles = max(1, int(math.ceil(lat_ns[idx] / clock_ns)))
            power_mw = (en_nj[idx] * 1e3 / (cycles * clock_ns)) if (cycles * clock_ns) > 0 else 0.0
            meta = {"source": "mnsim_full"}
            area_list = context.get("mnsim_area_um2")
            if isinstance(area_list, (list, tuple)) and idx < len(area_list):
                meta["area_um2"] = float(area_list[idx])
            layer_names = context.get("mnsim_layer_names")
            if isinstance(layer_names, (list, tuple)) and idx < len(layer_names):
                meta["layer_name"] = str(layer_names[idx])
            context["mnsim_index"] = idx + 1
            mnsim_ev = Event(
                event_id=0,
                cycle=0,
                operator=OP_NAME_CONV1D,
                circuit_module="mnsim_tile_noc",
                operation="conv1d_mnsim_full",
                power=power_mw,
                duration=cycles,
                clock_period_ns=clock_ns,
                metadata=meta,
            )
            if getattr(hw_config, "enable_w8a8_quant_dequant", False):
                p = op_ref.params
                in_ch = int(p.get("in_channels", 1))
                out_ch = int(p.get("out_channels", 1))
                num_points = int(p.get("num_points", 1))
                layer_name = str(p.get("layer_name", ""))
                in_numel = max(1, in_ch * num_points)
                out_numel = max(1, out_ch * num_points)
                q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
                d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
                q_ev, d_ev = q_evs[0], d_evs[0]
                mnsim_ev.cycle = q_ev.cycle + q_ev.duration
                d_ev.cycle = mnsim_ev.cycle + mnsim_ev.duration
                return [q_ev, mnsim_ev, d_ev]
            return [mnsim_ev]
    return gen_conv1d_rram_cim_events(op_ref.params, hw_config)


def _gen_linear(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Linear：若有 mnsim_context 则用 MNSIM 全 NetStruct 的延迟/能耗生成事件（算子名 PascalCase）；否则走 PE 级 common_ops。"""
    if context and "mnsim_latency_ns" in context and "mnsim_energy_nj" in context:
        idx = context.get("mnsim_index", 0)
        lat_ns = context["mnsim_latency_ns"]
        en_nj = context["mnsim_energy_nj"]
        if idx < len(lat_ns) and idx < len(en_nj):
            clock_ns = context.get("mnsim_clock_period_ns") or getattr(hw_config, "clock_period", 1.0)
            clock_ns = float(clock_ns)
            cycles = max(1, int(math.ceil(lat_ns[idx] / clock_ns)))
            power_mw = (en_nj[idx] * 1e3 / (cycles * clock_ns)) if (cycles * clock_ns) > 0 else 0.0
            meta = {"source": "mnsim_full"}
            area_list = context.get("mnsim_area_um2")
            if isinstance(area_list, (list, tuple)) and idx < len(area_list):
                meta["area_um2"] = float(area_list[idx])
            layer_names = context.get("mnsim_layer_names")
            if isinstance(layer_names, (list, tuple)) and idx < len(layer_names):
                meta["layer_name"] = str(layer_names[idx])
            context["mnsim_index"] = idx + 1
            mnsim_ev = Event(
                event_id=0,
                cycle=0,
                operator=OP_NAME_LINEAR,
                circuit_module="mnsim_tile_noc",
                operation="linear_mnsim_full",
                power=power_mw,
                duration=cycles,
                clock_period_ns=clock_ns,
                metadata=meta,
            )
            if getattr(hw_config, "enable_w8a8_quant_dequant", False):
                p = op_ref.params
                in_f = int(p.get("in_features", 1))
                out_f = int(p.get("out_features", 1))
                tokens = int(p.get("tokens", 1))
                layer_name = str(p.get("layer_name", ""))
                in_numel = max(1, in_f * max(1, tokens))
                out_numel = max(1, out_f * max(1, tokens))
                q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Linear")
                d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Linear")
                q_ev, d_ev = q_evs[0], d_evs[0]
                mnsim_ev.cycle = q_ev.cycle + q_ev.duration
                d_ev.cycle = mnsim_ev.cycle + mnsim_ev.duration
                return [q_ev, mnsim_ev, d_ev]
            return [mnsim_ev]
    return gen_linear_rram_cim_events(op_ref.params, hw_config)


def _gen_bn(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """BatchNorm1d generic logic-op model."""
    return gen_bn_events(op_ref.params, hw_config)


def _gen_layernorm(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """LayerNorm generic logic-op model."""
    return gen_layernorm_events(op_ref.params, hw_config)


def _gen_relu(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """ReLU generic logic-op model."""
    return gen_relu_events(op_ref.params, hw_config)


def _gen_spool(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """S-Pool generic logic-op model."""
    return gen_spool_events(op_ref.params, hw_config)


def _gen_kan(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """KAN LUT model (B-spline / GRF basis on RRAM-LUT)."""
    return gen_kan_rram_lut_events(op_ref.params, hw_config)


def _gen_fps(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    FPS：当 distance_in_phase2 时，先插入 Phase2 距离计算（每查询 12/2*ceil(N/128) 周期），
    再插入 FPS 引擎 bitonic；baseline 下与 KNN 串行（共享 bitonic 比较器）。
    """
    from .fps import FPSOperator, FPSParams
    from .layer_buffer_ops import gen_phase2_distance_events

    params = op_ref.params
    input_points = int(params.get('input_points', 1024))
    sampled_points_num = int(params.get('output_points', 512))
    fps_params = FPSParams(
        input_points=input_points,
        sampled_points_num=sampled_points_num,
    )
    fps_op = FPSOperator(params=fps_params, hw_config=hw_config)
    if not getattr(hw_config, "distance_in_phase2", False):
        return fps_op.generate_events()

    # Phase2 + FPS 引擎交错：每迭代 [距离计算] -> [bitonic]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    per_query_cycles = _phase2_distance_cycles_per_query(input_points, hw_config)
    all_events: List[Event] = []
    current_cycle = 0
    for i in range(sampled_points_num):
        evs_d = gen_phase2_distance_events(
            1,
            input_points,
            hw_config,
            "FPS",
            include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
            clock_period_ns=clock_ns,
        )
        for e in evs_d:
            e.cycle += current_cycle
        all_events.extend(evs_d)
        current_cycle += per_query_cycles
        one_bitonic = fps_op.generate_single_bitonic_event()
        one_bitonic.cycle = current_cycle
        all_events.append(one_bitonic)
        current_cycle += one_bitonic.duration
    return all_events


def _phase2_distance_cycles_per_query(num_keys: int, hw_config: 'HardwareConfig') -> int:
    """Phase2 单查询总周期（可含查询 DRAM 读取）。"""
    import math
    sram_ii = getattr(hw_config, "sram_ii", None)
    if sram_ii is not None:
        active = max(1, getattr(sram_ii, "active_rows", 128))
        bits = max(1, getattr(sram_ii, "bits_per_axis", 12))
        slice_bits = max(1, getattr(sram_ii, "bit_slice", 2))
    else:
        active, bits, slice_bits = 128, 12, 2

    dist_cycles = (bits // slice_bits) * max(1, int(math.ceil(num_keys / active)))
    if not getattr(hw_config, "distance_include_query_dram_read", True):
        return dist_cycles

    # 查询点默认 1x1x3，每特征 12bit
    num_features = 3
    bits_per_feature = 12
    query_bytes = int(math.ceil(num_features * bits_per_feature / 8.0))
    dram = getattr(hw_config, "dram", None)
    dram_bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32)) if dram is not None else 32
    dram_cycles = max(1, int(math.ceil(query_bytes / dram_bw)))
    return dist_cycles + dram_cycles


def _gen_knn(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    KNN：当 distance_in_phase2 时，先插入 Phase2 距离计算，再插入 KNN 引擎 bitonic top-k。
    Baseline 下与 FPS 串行（共享 bitonic 比较器）；fuse 模式可复用距离，待扩展。
    """
    from .knn import KNNOperator, KNNParams
    from .layer_buffer_ops import gen_phase2_distance_events

    params = op_ref.params
    input_points = int(params.get('input_points', 1024))
    query_points = int(params.get('query_points', 512))
    knn_params = KNNParams(
        input_points=input_points,
        query_points=query_points,
        k=params.get('k', 16),
    )
    knn_op = KNNOperator(params=knn_params, hw_config=hw_config)
    if not getattr(hw_config, "distance_in_phase2", False):
        return knn_op.generate_events()

    # Phase2 + KNN 引擎交错：每查询 [距离计算] -> [排序/筛选]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    per_query_cycles = _phase2_distance_cycles_per_query(input_points, hw_config)
    knn_mode = str(getattr(hw_config, "knn_engine_mode", "baseline")).lower()
    all_events = []
    current_cycle = 0
    for q in range(query_points):
        evs_d = gen_phase2_distance_events(
            1,
            input_points,
            hw_config,
            "KNN",
            include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
            clock_period_ns=clock_ns,
        )
        for e in evs_d:
            e.cycle += current_cycle
        all_events.extend(evs_d)
        current_cycle += per_query_cycles

        if knn_mode == "fuse":
            fuse_events = knn_op.generate_single_fuse_events()
            for fe in fuse_events:
                fe.cycle += current_cycle
            all_events.extend(fuse_events)
            current_cycle = max(fe.cycle + fe.duration for fe in fuse_events)
        else:
            one_bitonic = knn_op.generate_single_bitonic_event()
            one_bitonic.cycle = current_cycle
            all_events.append(one_bitonic)
            current_cycle += one_bitonic.duration
    return all_events


def _gen_stub(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Stub handler for ops not yet mapped to CIM (pipeline completeness only)."""
    return []


def _gen_affine_norm(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """AffineNorm generic logic-op model."""
    return gen_affine_norm_events(op_ref.params, hw_config)


def _gen_adaptive_max_pool(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """AdaptiveMaxPool1d generic logic-op model."""
    return gen_adaptive_max_pool_events(op_ref.params, hw_config)


def _gen_add(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Element-wise add, supports residual off-chip buffering."""
    return gen_add_events(op_ref.params, hw_config)


def _gen_mul(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Element-wise multiplication generic model."""
    return gen_mul_events(op_ref.params, hw_config)


# =============================================================================
# Dispatcher Registry & Public API
# =============================================================================

_OP_TYPE_HANDLERS: Dict[str, Callable] = {
    'conv1d':      _gen_conv1d,
    'linear':      _gen_linear,
    'bn':          _gen_bn,
    'batchnorm1d': _gen_bn,       # alias
    'layernorm':   _gen_layernorm,
    'relu':        _gen_relu,
    's-pool':      _gen_spool,
    'fps':         _gen_fps,
    'knn':         _gen_knn,
    'kan':         _gen_kan,
    'affine_norm':           _gen_affine_norm,
    'adaptive_max_pool1d':   _gen_adaptive_max_pool,
    'add':                   _gen_add,
    'mul':                   _gen_mul,
    'multiplication':        _gen_mul,
    'element_multiply':      _gen_mul,
}


def dispatch_operator(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Dispatch an OperatorRef to its Operator and return events.

    Args:
        op_ref: Operator reference from network layer definition.
        hw_config: Hardware configuration.
        context: Optional dict, e.g. mnsim_latency_ns, mnsim_energy_nj, mnsim_index for conv1d/linear.

    Returns:
        List of Event objects with relative cycles starting from 0.

    Raises:
        ValueError: If op_type is not registered.
    """
    handler = _OP_TYPE_HANDLERS.get(op_ref.op_type)
    if handler is None:
        raise ValueError(
            f"Unknown op_type '{op_ref.op_type}'. "
            f"Supported: {list_supported_op_types()}"
        )
    return handler(op_ref, hw_config, context)


def dispatch_layer(
    layer_operators: List['OperatorRef'],
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Dispatch all operators in a layer sequentially.

    - BN 融合：若 bn_fuse_with_conv_linear 且上一算子是 conv1d/linear，本层 BN 不计 cycle/能耗。
    - 其余同前：cycle 偏移、MNSIM context 等。
    """
    all_events: List[Event] = []
    current_cycle = 0
    prev_op_type: Optional[str] = None

    for op_ref in layer_operators:
        # BN 紧跟 Conv/Linear 时可选融合，不计 BN 开销
        if getattr(hw_config, "bn_fuse_with_conv_linear", False) and prev_op_type in ("conv1d", "linear"):
            if op_ref.op_type in ("bn", "batchnorm1d"):
                prev_op_type = op_ref.op_type
                continue
        events = dispatch_operator(op_ref, hw_config, context)
        for e in events:
            e.cycle += current_cycle
        all_events.extend(events)
        if events:
            current_cycle = max(e.cycle + e.duration for e in events)
        prev_op_type = op_ref.op_type

    return all_events


def list_supported_op_types() -> List[str]:
    """Return list of all supported op_type strings."""
    return sorted(_OP_TYPE_HANDLERS.keys())


def register_op_type(op_type: str, handler: Callable) -> None:
    """
    Register a custom op_type handler.

    Args:
        op_type: The operator type string.
        handler: Function(op_ref, hw_config, context?) -> List[Event].
    """
    _OP_TYPE_HANDLERS[op_type] = handler
