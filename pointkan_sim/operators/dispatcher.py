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

from __future__ import annotations

import math
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING, Tuple

from ..constants import OP_NAME_CONV1D, OP_NAME_LINEAR
from ..simulator.event import Event
from ..integration.memory_model_adapter import query_sram_cycles_energy, query_dram_cycles_energy
from ..memory.memory_contract import MemoryContractPolicy
from .common_ops import (
    gen_conv1d_rram_cim_events,
    gen_conv1d_systolic_events,
    gen_linear_rram_cim_events,
    gen_linear_systolic_events,
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
    gen_div_events,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig
    from ..network.pointkan_net import OperatorRef


# =============================================================================
# Per-op_type handlers
# =============================================================================


def _shape_to_str(shape):
    if not shape:
        return ""
    try:
        return "×".join(str(int(v)) for v in shape)
    except Exception:
        return str(shape)


def _merge_real_conv_linear_params(op_ref: 'OperatorRef', context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(op_ref.params)
    if not context:
        return params
    real_list = context.get('real_conv_linear_list')
    if not real_list:
        return params
    idx = int(context.get('real_conv_linear_index', 0) or 0)
    if idx >= len(real_list):
        return params
    info = real_list[idx]
    expected = 'conv' if op_ref.op_type == 'conv1d' else 'fc'
    if getattr(info, 'layer_type', None) != expected:
        return params
    context['real_conv_linear_index'] = idx + 1
    params['real_module_name'] = getattr(info, 'name', '')
    if getattr(info, 'input_shape', None):
        params['activation_input_shape'] = _shape_to_str(getattr(info, 'input_shape', ()))
    if getattr(info, 'output_shape', None):
        params['activation_output_shape'] = _shape_to_str(getattr(info, 'output_shape', ()))
    if op_ref.op_type == 'conv1d':
        if getattr(info, 'length', 0):
            params['num_points'] = int(info.length)
        if getattr(info, 'in_channels', 0):
            params['in_channels'] = int(info.in_channels)
        if getattr(info, 'out_channels', 0):
            params['out_channels'] = int(info.out_channels)
        if getattr(info, 'kernel_size', 0):
            params['kernel_size'] = int(info.kernel_size)
    else:
        if getattr(info, 'tokens', 0):
            params['tokens'] = int(info.tokens)
        if getattr(info, 'in_features', 0):
            params['in_features'] = int(info.in_features)
        if getattr(info, 'out_features', 0):
            params['out_features'] = int(info.out_features)
    return params



_ACC_MEM_POLICY = MemoryContractPolicy()


def _acc_buffer_capacity_bytes(buffer_name: str) -> int:
    p = _ACC_MEM_POLICY
    mapping = {
        "query_xyz_sram_buffer": max(1, p.cim_batch_points * p.point_xyz_bytes),
        "key_xyz_sram_buffer": max(1, p.cim_batch_points * p.point_xyz_bytes),
        "self_distance_sram_buffer": max(1, p.self_distance_sram_bytes),
        "dist_sram_buffer": max(1, p.dist_sram_bytes),
        "fps_min_distance_table_sram": max(1, p.fps_min_distance_table_sram_bytes),
        "temp_max_distance_sram": max(1, p.temp_max_distance_sram_bytes),
        "topk_sram_buffer": max(1, p.topk_sram_bytes),
        "sort_cache_sram": max(1, p.sort_cache_sram_bytes),
    }
    return int(mapping.get(buffer_name, 1 << 60))


def _acc_spill_bytes(buffer_name: str, total_bytes: int) -> Tuple[int, int]:
    total = max(0, int(total_bytes))
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name))
    if total <= cap:
        return total, 0
    return cap, total - cap


def _acc_mem_event(event_id: int, cycle: int, operator: str, stage: str, sub_stage: str, circuit_module: str, operation: str,
                   duration: int, energy_nj: float, clock_period_ns: float, metadata: Optional[Dict[str, Any]] = None) -> Event:
    dur = max(1, int(duration))
    clk = float(clock_period_ns)
    power = (float(energy_nj) * 1000.0 / (dur * clk)) if (dur > 0 and clk > 0) else 0.0
    return Event(event_id, cycle, operator, stage, sub_stage, circuit_module, operation, power, dur, clock_period_ns=clk, metadata=metadata or {})


def _acc_append_spill_events(out: List[Event], event_id: int, cycle: int, owner_operator: str, stage: str,
                             buffer_name: str, spill_bytes: int, clock_ns: float, phase: str,
                             direction: str, hw_config: 'HardwareConfig', memory_context: Optional[Any] = None) -> Tuple[int, int]:
    spill = max(0, int(spill_bytes))
    if spill <= 0:
        return event_id, 0
    is_read = direction == "read"
    if memory_context is not None:
        res = memory_context.dram_transfer(spill, is_read=is_read, cur_cycle=cycle)
        dur = int(res.get("cycles", 0) or 0)
        energy_nj = float(res.get("energy_nj", 0.0) or 0.0)
    else:
        dur, energy_nj = query_dram_cycles_energy(spill, is_read=is_read, hw_config=hw_config, clock_period_ns=clock_ns)
    ev = _acc_mem_event(
        event_id, cycle, "DRAM", stage, f"{buffer_name}_{'spill_read' if is_read else 'spill_write'}", "dram", f"{buffer_name}_{'spill_read' if is_read else 'spill_write'}",
        dur, energy_nj, clock_ns,
        {
            "owner_operator": owner_operator,
            "buffer_name": buffer_name,
            "buffer_phase": 'spill_read' if is_read else 'spill_write',
            "bytes": spill,
            "resource_id": "offchip_dram",
            "exclusive": True,
            "phase": phase,
        },
    )
    out.append(ev)
    return event_id + 1, dur


def _acc_append_sram_access_event(out: List[Event], event_id: int, cycle: int, owner_operator: str, stage: str,
                                  sub_stage: str, circuit_module: str, operation: str, buffer_name: str,
                                  phase: str, resource_id: str, num_bytes: int, hw_config: 'HardwareConfig',
                                  *, is_read: bool, overlapped: bool = False, extra_meta: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    onchip = max(0, int(num_bytes))
    if onchip <= 0:
        return event_id, 0
    clk = float(getattr(hw_config, "clock_period", 1.0))
    dur, energy_nj = query_sram_cycles_energy(onchip, is_read=is_read, hw_config=hw_config, clock_period_ns=clk)
    meta = {
        "owner_operator": owner_operator,
        "buffer_name": buffer_name,
        "buffer_phase": "read" if is_read else "write",
        "bytes": onchip,
        "resource_id": resource_id,
        "exclusive": True,
        "phase": phase,
    }
    if extra_meta:
        meta.update(extra_meta)
    ev = _acc_mem_event(event_id, cycle, owner_operator, stage, sub_stage, circuit_module, operation, dur, energy_nj, clk, meta)
    ev.overlapped = bool(overlapped)
    out.append(ev)
    return event_id + 1, (0 if overlapped else dur)


def _buffer_power_mw(buf_cfg: Any, clock_ns: float, default_power: float = 0.5, write: bool = True) -> float:
    if buf_cfg is None:
        return float(default_power)
    fn_name = "get_write_power_mw" if write else "get_read_power_mw"
    if hasattr(buf_cfg, fn_name):
        return float(getattr(buf_cfg, fn_name)(clock_ns))
    return float(getattr(buf_cfg, "power", default_power) or default_power)


def _acc_resampler_cycles(num_points: int, hw_config: 'HardwareConfig') -> int:
    logic = getattr(hw_config, "logic_circuit", None)
    cfg = getattr(logic, "resampler", None) if logic is not None else None
    if cfg is None:
        return max(1, int(num_points))
    explicit = int(getattr(cfg, "cycles", 0) or 0)
    if explicit > 0:
        return explicit
    parallelism = max(1, int(getattr(cfg, "parallelism", 1) or 1))
    return max(1, int(math.ceil(max(1, num_points) / float(parallelism))) + 2)


def _acc_resampler_lut(hw_config: 'HardwareConfig', mode: str) -> Tuple[float, int]:
    logic = getattr(hw_config, "logic_circuit", None)
    cfg = getattr(logic, "resampler", None) if logic is not None else None
    if cfg is None:
        return (0.094, 1)
    if mode == "downsample":
        return (
            float(getattr(cfg, "lut_power_downsample", 0.094) or 0.094),
            max(1, int(getattr(cfg, "lut_cycles_downsample", 1) or 1)),
        )
    return (
        float(getattr(cfg, "lut_power_upsample", 0.094) or 0.094),
        max(1, int(getattr(cfg, "lut_cycles_upsample", 1) or 1)),
    )


def _acc_predictor_cycles(hw_config: 'HardwareConfig') -> int:
    lut_cfg = getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut_speculative", None)
    if lut_cfg is None:
        return 1
    return max(1, int(getattr(lut_cfg, "cycles", 1) or 1))


def _acc_frontend_pipeline_enabled(hw_config: 'HardwareConfig') -> bool:
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return False
    pipeline_mode = str(getattr(buf, "pipeline_mode", "without_pipeline") or "without_pipeline").strip().lower()
    double_buffering = bool(getattr(buf, "double_buffering", False))
    return pipeline_mode == "pipeline" and double_buffering


def _acc_segment_end_points(iteration_idx: int, sampled_points_num: int, segment_len: int) -> int:
    point_index = max(1, int(iteration_idx) + 1)
    segment_len = max(1, int(segment_len) or 1)
    return min(int(sampled_points_num), int(math.ceil(point_index / float(segment_len))) * segment_len)


def _gen_acc_distance_query_events(
    start_event_id: int,
    base_cycle: int,
    operator_name: str,
    num_keys: int,
    hw_config: 'HardwareConfig',
    query_index: int = 0,
    include_query_dram_read: bool = True,
    memory_context: Optional[Any] = None,
) -> Tuple[List[Event], int, int]:
    """
    ACC 距离前端：key preload -> self-square -> self-distance write -> query read
    -> cross-term -> merge accumulate -> distance write。

    这是一个 first-order runtime skeleton，优先把关键物理阶段显式化。
    """
    events: List[Event] = []
    eid = start_event_id
    clk = float(getattr(hw_config, "clock_period", 1.0))
    num_keys = max(1, int(num_keys))
    macro = getattr(hw_config, "sram_cim_macro", None)
    array = getattr(macro, "array", None) if macro is not None else None
    shift = getattr(macro, "shift_add", None) if macro is not None else None
    out_buf = getattr(macro, "output_buffer", None) if macro is not None else None
    dist_merge = getattr(macro, "dist2_merge_adder", None) if macro is not None else None
    parallelism = max(1, int(getattr(array, "cim_parallelism", 128) or 128))
    quant_bits = max(1, int(getattr(array, "quantization_bits", 12) or 12))
    bit_slice = max(1, int(getattr(array, "bit_slice_bits", 2) or 2))
    bit_serial_cycles = max(1, int(math.ceil(quant_bits / float(bit_slice))))
    batches = max(1, int(math.ceil(num_keys / float(parallelism))))
    batch_compute_cycles = batches * bit_serial_cycles
    key_preload_cycles = num_keys * max(1, int(getattr(array, "write_cycles_per_row", 1) or 1))
    key_preload_power = float(getattr(array, "write_power_per_row_mw", 2.0) or 2.0)
    cim_power = float(getattr(array, "cim_power_per_batch_mw", 6.0) or 6.0)
    shift_power = float(getattr(shift, "power", 1.3) or 1.3)
    merge_power = float(getattr(dist_merge, "power", 1.3) or 1.3)
    merge_cycles = max(1, int(getattr(dist_merge, "cycles", 1) or 1))
    out_write_power = _buffer_power_mw(out_buf, clk, default_power=0.5, write=True)
    out_cycles = max(1, int(getattr(out_buf, "cycles", 1) or 1))
    xyz_bytes = num_keys * 3 * 4
    dist_bytes = num_keys * 4
    cursor = base_cycle
    frontend_pipeline = _acc_frontend_pipeline_enabled(hw_config)

    key_refill_bytes = xyz_bytes
    if memory_context is not None:
        res = memory_context.dram_transfer(key_refill_bytes, is_read=True, cur_cycle=cursor)
        key_dram_cycles = int(res.get("cycles", 0) or 0)
        key_dram_energy_nj = float(res.get("energy_nj", 0.0) or 0.0)
    else:
        key_dram_cycles, key_dram_energy_nj = query_dram_cycles_energy(key_refill_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk)
    key_dram_power = float(key_dram_energy_nj) * 1000.0 / max(1.0, key_dram_cycles * clk) if key_dram_cycles > 0 else 0.0
    key_refill_ev = Event(
        eid, cursor, operator_name, "distance", "key_dram_read", "dram", "key_xyz_refill",
        key_dram_power, max(1, key_dram_cycles), clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "buffer_name": "key_xyz_sram_buffer",
            "buffer_phase": "key_refill",
            "bytes": key_refill_bytes,
            "resource_id": "offchip_dram",
            "exclusive": True,
            "phase": "distance_frontend",
        },
    )
    if frontend_pipeline and query_index > 0:
        key_refill_ev.overlapped = True
    events.append(key_refill_ev)
    eid += 1
    if not getattr(key_refill_ev, "overlapped", False):
        cursor += max(1, key_dram_cycles)

    key_preload_ev = Event(
        eid, cursor, operator_name, "distance", "key_preload", "sram_cim_array", "cim_keyset_row_write",
        key_preload_power, key_preload_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "buffer_name": "key_xyz_sram_buffer",
            "buffer_phase": "key_refill",
            "bytes": xyz_bytes,
            "resource_id": "key_xyz_slot_0",
            "exclusive": True,
            "phase": "distance_frontend",
        },
    )
    if frontend_pipeline and query_index > 0:
        key_preload_ev.overlapped = True
    events.append(key_preload_ev)
    eid += 1
    if not getattr(key_preload_ev, "overlapped", False):
        cursor += key_preload_cycles

    events.append(Event(
        eid, cursor, operator_name, "distance", "self_square", "sram_cim_array", "self_square_distance",
        cim_power + shift_power, batch_compute_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "phase": "distance_frontend",
            "buffer_name": "self_distance_sram_buffer",
            "buffer_phase": "self_square_compute",
            "bytes": dist_bytes,
            "resource_id": "distance_compute_shared",
            "exclusive": True,
        },
    ))
    eid += 1
    cursor += batch_compute_cycles

    self_onchip_bytes, self_spill_bytes = _acc_spill_bytes("self_distance_sram_buffer", dist_bytes)
    events.append(Event(
        eid, cursor, operator_name, "distance", "self_distance_write", "output_buffer", "self_distance_write",
        out_write_power, out_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "buffer_name": "self_distance_sram_buffer",
            "buffer_phase": "write",
            "bytes": self_onchip_bytes,
            "resource_id": "self_distance_shared",
            "exclusive": True,
        },
    ))
    eid += 1
    cursor += out_cycles
    eid, spill_dur = _acc_append_spill_events(
        events, eid, cursor, operator_name, "distance", "self_distance_sram_buffer", self_spill_bytes,
        clk, "distance_frontend", "write", hw_config, memory_context,
    )
    cursor += spill_dur

    if include_query_dram_read:
        query_bytes = int(math.ceil(3 * quant_bits / 8.0))
        if memory_context is not None:
            res = memory_context.dram_transfer(query_bytes, is_read=True, cur_cycle=cursor)
            dram_cycles = int(res.get("cycles", 0) or 0)
            dram_energy_nj = float(res.get("energy_nj", 0.0) or 0.0)
        else:
            from .layer_buffer_ops import query_dram_cycles_energy
            dram_cycles, dram_energy_nj = query_dram_cycles_energy(
                query_bytes,
                is_read=True,
                hw_config=hw_config,
                clock_period_ns=clk,
            )
        dram_power = float(dram_energy_nj) * 1000.0 / max(1.0, dram_cycles * clk) if dram_cycles > 0 else 0.0
        events.append(Event(
            eid, cursor, operator_name, "distance", "query_dram_read", "dram", "phase2_query_point_read",
            dram_power, max(1, dram_cycles), clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "buffer_name": "query_xyz_sram_buffer",
                "buffer_phase": "query_refill",
                "bytes": query_bytes,
                "resource_id": "offchip_dram",
                "exclusive": True,
                "phase": "distance_frontend",
            },
        ))
        eid += 1
        cursor += max(1, dram_cycles)

    events.append(Event(
        eid, cursor, operator_name, "distance", "cross_term", "sram_cim_array", "cross_term_distance",
        cim_power + shift_power, batch_compute_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "phase": "distance_frontend",
            "buffer_name": "query_xyz_sram_buffer",
            "buffer_phase": "query_read",
            "bytes": 12,
            "resource_id": "distance_compute_shared",
            "exclusive": True,
        },
    ))
    eid += 1
    cursor += batch_compute_cycles

    eid, spill_dur = _acc_append_spill_events(
        events, eid, cursor, operator_name, "distance", "self_distance_sram_buffer", self_spill_bytes,
        clk, "distance_frontend", "read", hw_config, memory_context,
    )
    cursor += spill_dur
    events.append(Event(
        eid, cursor, operator_name, "distance", "merge_accumulate", "dist2_merge_adder", "distance_merge_accumulate",
        merge_power, merge_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "buffer_name": "self_distance_sram_buffer",
            "buffer_phase": "read",
            "bytes": self_onchip_bytes,
            "resource_id": "distance_merge_shared",
            "exclusive": True,
            "phase": "distance_frontend",
        },
    ))
    eid += 1
    cursor += merge_cycles

    dist_onchip_bytes, dist_spill_bytes = _acc_spill_bytes("dist_sram_buffer", dist_bytes)
    events.append(Event(
        eid, cursor, operator_name, "distance", "distance_write", "output_buffer", "distance_write",
        out_write_power, out_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "buffer_name": "dist_sram_buffer",
            "buffer_phase": "write",
            "bytes": dist_onchip_bytes,
            "resource_id": "dist_sram_shared",
            "exclusive": True,
            "phase": "distance_frontend",
        },
    ))
    eid += 1
    cursor += out_cycles
    eid, spill_dur = _acc_append_spill_events(
        events, eid, cursor, operator_name, "distance", "dist_sram_buffer", dist_spill_bytes,
        clk, "distance_frontend", "write", hw_config, memory_context,
    )
    cursor += spill_dur
    return events, eid, cursor - base_cycle


# Each handler signature:
#   (op_ref: OperatorRef, hw_config: HardwareConfig) -> List[Event]


def _gen_conv1d(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Conv1d：base_v1/base_v2 仅通过 SCALE-Sim 脉动阵列仿真（FXP16）；acc 用 MNSIM/NeuroSim 或 RRAM-CIM（可含量化/反量化）；其他配置 raise。
    """
    is_baseline = getattr(hw_config, "is_baseline", False)
    has_systolic = getattr(hw_config, "systolic_array", None) is not None
    if is_baseline:
        if not has_systolic:
            raise RuntimeError("Conv1d not supported: baseline (base_v1/base_v2) requires systolic_array and SCALE-Sim.")
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_conv1d_systolic_events(params, hw_config)
    # acc：MNSIM/NeuroSim（全 NetStruct 或 PE 级 RRAM-CIM）
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
    # acc：仅支持全网络 NeuroSim/MNSIM；禁止静默回退到单层 RRAM-CIM 估算
    if not is_baseline:
        raise RuntimeError("ACC Conv1d currently requires a full-network backend context (neurosim_full or mnsim_full). PE-level cim_backend fallback is disabled.")
    raise RuntimeError("Conv1d not supported: unknown accelerator mode (neither baseline with SCALE-Sim nor acc).")


def _gen_linear(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Linear：base_v1/base_v2 仅通过 SCALE-Sim 脉动阵列仿真（FXP16）；acc 用 MNSIM/NeuroSim 或 RRAM-CIM（可含量化/反量化）；其他配置 raise。
    """
    is_baseline = getattr(hw_config, "is_baseline", False)
    has_systolic = getattr(hw_config, "systolic_array", None) is not None
    if is_baseline:
        if not has_systolic:
            raise RuntimeError("Linear not supported: baseline (base_v1/base_v2) requires systolic_array and SCALE-Sim.")
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_linear_systolic_events(params, hw_config)
    # acc：MNSIM/NeuroSim
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
    # acc：仅支持全网络 NeuroSim/MNSIM；禁止静默回退到单层 RRAM-CIM 估算
    if not is_baseline:
        raise RuntimeError("ACC Linear currently requires a full-network backend context (neurosim_full or mnsim_full). PE-level cim_backend fallback is disabled.")
    raise RuntimeError("Linear not supported: unknown accelerator mode (neither baseline with SCALE-Sim nor acc).")


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
    acc 模式下仅第一层跑 FPS，后续层复用第一层 FPS 结果（0 周期占位）。
    """
    # acc：仅第一层有 FPS 的层跑 FPS，后续层复用第一层结果，不产生周期/能耗
    if getattr(hw_config, "is_acc", False) and context is not None:
        first_fps = context.get("first_layer_with_fps")
        layer_index = context.get("layer_index")
        if first_fps is not None and layer_index is not None and layer_index != first_fps:
            clock_ns = getattr(hw_config, "clock_period", 1.0)
            return [
                Event(
                    0, 0, "FPS", "reuse", "fps_reuse", "fps", "fps_reuse_first_layer",
                    0.0, 0, clock_ns, metadata={"fps_reuse": True},
                )
            ]

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

    # ACC：显式 phase-based runtime（warmup full-sort -> predictor gate -> post-predict coarse/fine）
    if getattr(hw_config, "is_acc", False):
        clock_ns = getattr(hw_config, "clock_period", 1.0)
        all_events: List[Event] = []
        current_cycle = 0
        eid = 0
        warmup_iters = max(1, int(math.ceil(sampled_points_num * 0.1)))
        mdt_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "min_distance_table", None)
        coarse_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "coarse_filter", None)
        fine_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "fine_filter", None)
        coarse_power = float(getattr(coarse_cfg, "power", 81.12) or 81.12)
        coarse_cycles = max(1, int(getattr(coarse_cfg, "cycles", 1) or 1))
        fine_power = float(getattr(fine_cfg, "power", 38.0) or 38.0)
        fine_cycles = max(1, int(getattr(fine_cfg, "cycles", 2) or 2))
        mdt_power = float(getattr(mdt_cfg, "power", 30.0) or 30.0)
        predictor_power = float(getattr(getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut_speculative", None), "power", 5.0) or 5.0)
        predictor_cycles = _acc_predictor_cycles(hw_config)
        segment_len = max(1, int(params.get("segment_len", 64) or 64))
        prediction_ready = False
        upsampled_until = 0

        for i in range(sampled_points_num):
            candidate_points = max(1, input_points - i)
            base_cycle = current_cycle
            evs_d, eid, dist_cycles = _gen_acc_distance_query_events(
                eid,
                base_cycle,
                "FPS",
                candidate_points,
                hw_config,
                query_index=i,
                include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
                memory_context=context.get("memory_runtime_context") if context else None,
            )
            all_events.extend(evs_d)
            current_cycle += dist_cycles

            mdt_total_bytes = int(candidate_points) * 8
            mdt_onchip_bytes, mdt_spill_bytes = _acc_spill_bytes("fps_min_distance_table_sram", mdt_total_bytes)
            warmup_temp_total_bytes = 32 * 8
            warmup_temp_onchip_bytes, warmup_temp_spill_bytes = _acc_spill_bytes("temp_max_distance_sram", warmup_temp_total_bytes)

            # min-table update and warmup/post-predict selection both overlap the distance stream in this first-order model
            min_ev = Event(
                eid, base_cycle, "FPS", "distance", "min_distance_update", "min_distance_table", "min_distance_compare",
                mdt_power, 1, clock_period_ns=clock_ns,
                metadata={
                    "iteration": i,
                    "buffer_name": "fps_min_distance_table_sram",
                    "buffer_phase": "update",
                    "bytes": mdt_onchip_bytes,
                    "resource_id": "fps_min_table_shared",
                    "exclusive": True,
                    "phase": "warmup_full_sort" if not prediction_ready else "post_predict_coarse_fine",
                },
            )
            min_ev.overlapped = True
            all_events.append(min_ev)
            eid += 1
            eid, spill_dur = _acc_append_spill_events(
                all_events, eid, current_cycle, "FPS", "distance", "fps_min_distance_table_sram", mdt_spill_bytes,
                clock_ns, "warmup_full_sort" if not prediction_ready else "post_predict_coarse_fine", "write", hw_config,
                context.get("memory_runtime_context") if context else None,
            )
            current_cycle += spill_dur

            if not prediction_ready:
                coarse_ev = Event(
                    eid, base_cycle, "FPS", "warmup_full_sort", "coarse_filter", "coarse_filter", "fps_warmup_coarse_top32",
                    coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "iteration": i,
                        "buffer_name": "temp_max_distance_sram",
                        "buffer_phase": "warmup_coarse_write",
                        "bytes": warmup_temp_onchip_bytes,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "warmup_full_sort",
                    },
                )
                coarse_ev.overlapped = True
                all_events.append(coarse_ev)
                eid += 1
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, current_cycle, "FPS", "warmup_full_sort", "temp_max_distance_sram", warmup_temp_spill_bytes,
                    clock_ns, "warmup_full_sort", "write", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                current_cycle += spill_dur

                fine_ev = Event(
                    eid, base_cycle + coarse_cycles, "FPS", "warmup_full_sort", "fine_filter", "fine_filter", "fps_warmup_fine_max",
                    fine_power, fine_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "iteration": i,
                        "buffer_name": "temp_max_distance_sram",
                        "buffer_phase": "warmup_fine_write",
                        "bytes": 8,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "warmup_full_sort",
                    },
                )
                fine_ev.overlapped = True
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, current_cycle, "FPS", "post_predict", "temp_max_distance_sram", max(0, 32 * 8 - _acc_buffer_capacity_bytes("temp_max_distance_sram")),
                    clock_ns, "post_predict_coarse_fine", "read", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                current_cycle += spill_dur
                all_events.append(fine_ev)
                eid += 1

                prefix_points = i + 1
                if prefix_points >= 2:
                    downsample_points = min(64, prefix_points)
                    down_cycles = _acc_resampler_cycles(prefix_points, hw_config)
                    down_power = float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "power", 3.95) or 3.95)
                    down_lut_power, down_lut_cycles = _acc_resampler_lut(hw_config, "downsample")
                    down_lut_ev = Event(
                        eid, base_cycle, "FPS", "speculative", "resampler_downsample_lut", "resampler", "resampler_downsample_lut",
                        down_lut_power, down_lut_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_prefix_downsample_lut",
                            "bytes": downsample_points * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "speculative_overlap",
                        },
                    )
                    down_lut_ev.overlapped = True
                    all_events.append(down_lut_ev)
                    eid += 1
                    down_ev = Event(
                        eid, base_cycle + down_lut_cycles, "FPS", "speculative", "resampler_downsample", "resampler", "resampler_downsample",
                        down_power, down_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_prefix_downsample",
                            "bytes": downsample_points * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "speculative_overlap",
                        },
                    )
                    down_ev.overlapped = True
                    all_events.append(down_ev)
                    eid += 1

                if prefix_points == warmup_iters:
                    # Predictor gate: must complete before switching to post-predict coarse/fine
                    pred_ev = Event(
                        eid, current_cycle, "FPS", "speculative", "predictor", "rram_lut_speculative", "spec_efficient_kan",
                        predictor_power, predictor_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_predictor",
                            "bytes": min(64, warmup_iters) * 4,
                            "resource_id": "predictor_shared",
                            "exclusive": True,
                            "phase": "predictor_gate",
                        },
                    )
                    all_events.append(pred_ev)
                    eid += 1
                    current_cycle += predictor_cycles

                    target_segment_end = _acc_segment_end_points(i, sampled_points_num, segment_len)
                    upsample_needed = max(0, target_segment_end - warmup_iters)
                    up_lut_power, up_lut_cycles = _acc_resampler_lut(hw_config, "upsample")
                    up_lut_ev = Event(
                        eid, current_cycle, "FPS", "speculative", "resampler_upsample_lut", "resampler", "resampler_upsample_lut",
                        up_lut_power, up_lut_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_resample_up_lut",
                            "bytes": max(64, upsample_needed) * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "predictor_gate",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    all_events.append(up_lut_ev)
                    eid += 1
                    current_cycle += up_lut_cycles

                    up_cycles = _acc_resampler_cycles(upsample_needed, hw_config)
                    up_power = float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "power", 3.95) or 3.95)
                    up_ev = Event(
                        eid, current_cycle, "FPS", "speculative", "resampler_upsample", "resampler", "resampler_upsample",
                        up_power, up_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_resample_up",
                            "bytes": max(upsample_needed, 0) * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "predictor_gate",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    all_events.append(up_ev)
                    eid += 1
                    current_cycle += up_cycles
                    eid, spill_dur = _acc_append_spill_events(
                        all_events, eid, current_cycle, "FPS", "speculative", "temp_max_distance_sram", max(0, max(upsample_needed, 0) * 4 - _acc_buffer_capacity_bytes("temp_max_distance_sram")),
                        clock_ns, "predictor_gate", "write", hw_config,
                    )
                    current_cycle += spill_dur
                    upsampled_until = max(upsampled_until, target_segment_end)
                    prediction_ready = True
            else:
                target_segment_end = _acc_segment_end_points(i, sampled_points_num, segment_len)
                if target_segment_end > upsampled_until:
                    upsample_needed = target_segment_end - upsampled_until
                    up_lut_power, up_lut_cycles = _acc_resampler_lut(hw_config, "upsample")
                    up_lut_ev = Event(
                        eid, current_cycle, "FPS", "speculative", "resampler_upsample_lut", "resampler", "resampler_upsample_lut",
                        up_lut_power, up_lut_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "segment_resample_up_lut",
                            "bytes": upsample_needed * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "post_predict_coarse_fine",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    all_events.append(up_lut_ev)
                    eid += 1
                    current_cycle += up_lut_cycles

                    up_cycles = _acc_resampler_cycles(upsample_needed, hw_config)
                    up_power = float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "power", 3.95) or 3.95)
                    up_ev = Event(
                        eid, current_cycle, "FPS", "speculative", "resampler_upsample", "resampler", "resampler_upsample",
                        up_power, up_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "segment_resample_up",
                            "bytes": upsample_needed * 4,
                            "resource_id": "resampler_shared",
                            "exclusive": True,
                            "phase": "post_predict_coarse_fine",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    all_events.append(up_ev)
                    eid += 1
                    current_cycle += up_cycles
                    eid, spill_dur = _acc_append_spill_events(
                        all_events, eid, current_cycle, "FPS", "speculative", "temp_max_distance_sram", max(0, upsample_needed * 4 - _acc_buffer_capacity_bytes("temp_max_distance_sram")),
                        clock_ns, "post_predict_coarse_fine", "write", hw_config,
                    )
                    current_cycle += spill_dur
                    upsampled_until = target_segment_end

                coarse_ev = Event(
                    eid, base_cycle, "FPS", "post_predict", "coarse_filter", "coarse_filter", "fps_coarse_filter_max",
                    coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "iteration": i,
                        "buffer_name": "temp_max_distance_sram",
                        "buffer_phase": "coarse_filter_write",
                        "bytes": min(32 * 8, _acc_buffer_capacity_bytes("temp_max_distance_sram")),
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "post_predict_coarse_fine",
                    },
                )
                coarse_ev.overlapped = True
                all_events.append(coarse_ev)
                eid += 1
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, current_cycle, "FPS", "post_predict", "temp_max_distance_sram", max(0, 32 * 8 - _acc_buffer_capacity_bytes("temp_max_distance_sram")),
                    clock_ns, "post_predict_coarse_fine", "write", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                current_cycle += spill_dur

                fine_ev = Event(
                    eid, base_cycle + coarse_cycles, "FPS", "post_predict", "fine_filter", "fine_filter", "fps_fine_filter_max",
                    fine_power, fine_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "iteration": i,
                        "buffer_name": "temp_max_distance_sram",
                        "buffer_phase": "fine_filter_read",
                        "bytes": 32 * 8,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "post_predict_coarse_fine",
                    },
                )
                fine_ev.overlapped = True
                all_events.append(fine_ev)
                eid += 1
        return all_events

    # Baseline Phase2 + FPS 引擎交错：每迭代 [距离计算] -> [bitonic]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    per_query_cycles = _phase2_distance_cycles_per_query(input_points, hw_config)
    all_events: List[Event] = []
    current_cycle = 0
    mdt_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "min_distance_table", None)
    mdt_power = float(getattr(mdt_cfg, "power", 0.0) or 0.0)
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
        min_table_ev = Event(
            len(all_events), current_cycle, "FPS", "min_distance_update", "table_update", "min_distance_table", "min_distance_compare",
            mdt_power, 1, clock_ns,
            metadata={"iteration": i, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "update", "bytes": int(input_points) * 8, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
        )
        all_events.append(min_table_ev)
        current_cycle += 1
        if str(getattr(hw_config, "fps_engine_mode", "baseline")).lower() == "fuse":
            # fuse：coarse+fine 专用路径（FPS MAX）
            fuse_evs = fps_op.generate_single_fuse_events()
            for fe in fuse_evs:
                fe.cycle += current_cycle
            all_events.extend(fuse_evs)
            current_cycle += sum(e.duration for e in fuse_evs)
        elif getattr(hw_config, "is_baseline_v2", False):
            sort_evs = fps_op._generate_single_merge_sort_iteration_events()
            critical_cy = 0
            for e in sort_evs:
                e.cycle += current_cycle
                if not getattr(e, "overlapped", False):
                    critical_cy = max(critical_cy, e.cycle + e.duration - current_cycle)
            all_events.extend(sort_evs)
            batches_fps = math.ceil(input_points / int(getattr(fps_params, "cim_parallelism", 64)))
            ms = getattr(hw_config.sorting_circuit, "merge_sort", None)
            reg = getattr(ms, "reg_file", None) if ms is not None else None
            if ms is not None and reg is not None:
                current_cycle += batches_fps * (ms.cycles_max + getattr(reg, "cycles_per_read", 1))
            else:
                current_cycle += sum(e.duration for e in sort_evs if not getattr(e, "overlapped", False))
        else:
            one_bitonic = fps_op.generate_single_bitonic_event()
            one_bitonic.cycle = current_cycle
            meta = dict(getattr(one_bitonic, "metadata", {}) or {})
            meta.update({"buffer_name": "fps_min_distance_table_sram", "buffer_phase": "sort_read", "bytes": int(input_points) * 4})
            one_bitonic.metadata = meta
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

    # ACC：动态阈值 KNN（距离前端 + coarse filter + topk SRAM + 可选 refine + fine filter）
    if getattr(hw_config, "is_acc", False):
        clock_ns = getattr(hw_config, "clock_period", 1.0)
        all_events: List[Event] = []
        current_cycle = 0
        eid = 0
        coarse_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "coarse_filter", None)
        fine_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "fine_filter", None)
        coarse_power = float(getattr(coarse_cfg, "power", 81.12) or 81.12)
        coarse_cycles = max(1, int(getattr(coarse_cfg, "cycles", 1) or 1))
        fine_power = float(getattr(fine_cfg, "power", 38.0) or 38.0)
        fine_cycles = max(1, int(getattr(fine_cfg, "cycles", 2) or 2))
        out_buf = getattr(getattr(hw_config, "sram_cim_macro", None), "output_buffer", None)
        topk_write_power = _buffer_power_mw(out_buf, clock_ns, default_power=0.5, write=True)
        refine_rounds = max(
            0,
            int(
                getattr(
                    hw_config,
                    "knn_dynamic_threshold_rounds_mean",
                    getattr(hw_config, "knn_coarse_rounds_mean", 1),
                )
                or 0
            ),
        )
        k_value = max(1, int(params.get('k', 16)))
        block_size = max(1, int(params.get('block_size', 128) or 128))
        num_blocks = max(1, int(math.ceil(input_points / float(block_size))))
        block_issue_cycles = max(1, int(getattr(hw_config, 'bit_serial_cycles', 1) or 1))
        topk_cap_bytes = _acc_buffer_capacity_bytes("topk_sram_buffer")
        thresh_cfg = getattr(getattr(hw_config, 'sorting_circuit', None), 'threshold_gen', None)
        thresh_gen_power = float(getattr(thresh_cfg, 'power', 3.66) or 3.66)
        thresh_gen_cycles = max(1, int(getattr(thresh_cfg, 'cycles', 1) or 1))

        for q in range(query_points):
            aggregate_candidate_bytes = 0
            base_cycle = current_cycle
            evs_d, eid, dist_cycles = _gen_acc_distance_query_events(
                eid,
                base_cycle,
                "KNN",
                input_points,
                hw_config,
                query_index=q,
                include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
                memory_context=context.get("memory_runtime_context") if context else None,
            )
            all_events.extend(evs_d)
            current_cycle += dist_cycles

            for blk in range(num_blocks):
                blk_start = base_cycle + blk * block_issue_cycles
                blk_points = min(block_size, input_points - blk * block_size)
                candidate_bytes = max(k_value * 8, blk_points * 4)
                coarse_ev = Event(
                    eid, blk_start, "KNN", "dynamic_threshold", "coarse_filter", "coarse_filter", "knn_coarse_filter",
                    coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "block_idx": blk,
                        "num_blocks": num_blocks,
                        "buffer_name": "dist_sram_buffer",
                        "buffer_phase": "coarse_filter_read",
                        "bytes": blk_points * 4,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                    },
                )
                coarse_ev.overlapped = True
                all_events.append(coarse_ev)
                eid += 1

                topk_write_ev = Event(
                    eid, blk_start + coarse_cycles, "KNN", "dynamic_threshold", "topk_write", "output_buffer", "topk_candidate_write",
                    topk_write_power, 1, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "block_idx": blk,
                        "num_blocks": num_blocks,
                        "buffer_name": "topk_sram_buffer",
                        "buffer_phase": "write",
                        "bytes": candidate_bytes,
                        "resource_id": "topk_sram_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                    },
                )
                topk_write_ev.overlapped = True
                all_events.append(topk_write_ev)
                eid += 1
                aggregate_candidate_bytes += candidate_bytes

            topk_onchip_bytes, topk_spill_bytes = _acc_spill_bytes("topk_sram_buffer", aggregate_candidate_bytes)
            eid, spill_dur = _acc_append_spill_events(
                all_events, eid, current_cycle, "KNN", "dynamic_threshold", "topk_sram_buffer", topk_spill_bytes,
                clock_ns, "knn_dynamic_threshold", "write", hw_config,
                context.get("memory_runtime_context") if context else None,
            )
            current_cycle += spill_dur
            refine_cursor = current_cycle
            for refine_idx in range(refine_rounds):
                if topk_onchip_bytes > 0:
                    sram_cycles, sram_energy = query_sram_cycles_energy(topk_onchip_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clock_ns)
                    all_events.append(_acc_mem_event(
                        eid, refine_cursor, "KNN", "dynamic_threshold", "topk_aggregate_read", "topk_sram_buffer", "topk_candidate_aggregate_read",
                        sram_cycles, sram_energy, clock_ns,
                        {"query": q, "buffer_name": "topk_sram_buffer", "buffer_phase": "aggregate_read", "bytes": topk_onchip_bytes, "resource_id": "topk_sram_shared", "exclusive": True, "phase": "knn_dynamic_threshold", "refine_round": refine_idx + 1, "refine_rounds": refine_rounds},
                    ))
                    eid += 1
                    refine_cursor += sram_cycles
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, refine_cursor, "KNN", "dynamic_threshold", "topk_sram_buffer", topk_spill_bytes,
                    clock_ns, "knn_dynamic_threshold", "read", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                refine_cursor += spill_dur
                thresh_ev = Event(
                    eid, refine_cursor, "KNN", "dynamic_threshold", "threshold_gen", "threshold_gen", "knn_threshold_update",
                    thresh_gen_power, thresh_gen_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "buffer_name": "topk_sram_buffer",
                        "buffer_phase": "threshold_gen",
                        "bytes": topk_onchip_bytes,
                        "resource_id": "threshold_gen_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                    },
                )
                all_events.append(thresh_ev)
                eid += 1

                refine_ev = Event(
                    eid, refine_cursor + thresh_gen_cycles, "KNN", "dynamic_threshold", "topk_refine", "coarse_filter", "knn_threshold_refine",
                    coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "buffer_name": "topk_sram_buffer",
                        "buffer_phase": "read",
                        "bytes": topk_onchip_bytes,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                    },
                )
                all_events.append(refine_ev)
                eid += 1
                refine_cursor += thresh_gen_cycles + coarse_cycles

            if topk_onchip_bytes > 0:
                sram_cycles, sram_energy = query_sram_cycles_energy(topk_onchip_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clock_ns)
                all_events.append(_acc_mem_event(
                    eid, refine_cursor, "KNN", "dynamic_threshold", "topk_fine_read", "topk_sram_buffer", "topk_candidate_fine_read",
                    sram_cycles, sram_energy, clock_ns,
                    {"query": q, "buffer_name": "topk_sram_buffer", "buffer_phase": "fine_filter_read", "bytes": topk_onchip_bytes, "resource_id": "topk_sram_shared", "exclusive": True, "phase": "knn_dynamic_threshold", "refine_rounds": refine_rounds, "num_blocks": num_blocks},
                ))
                eid += 1
                refine_cursor += sram_cycles
            eid, spill_dur = _acc_append_spill_events(
                all_events, eid, refine_cursor, "KNN", "dynamic_threshold", "topk_sram_buffer", topk_spill_bytes,
                clock_ns, "knn_dynamic_threshold", "read", hw_config,
                context.get("memory_runtime_context") if context else None,
            )
            refine_cursor += spill_dur
            fine_ev = Event(
                eid, refine_cursor, "KNN", "dynamic_threshold", "fine_filter", "fine_filter", "knn_fine_filter_topk",
                fine_power, fine_cycles, clock_period_ns=clock_ns,
                metadata={
                    "query": q,
                    "buffer_name": "topk_sram_buffer",
                    "buffer_phase": "fine_filter_read",
                    "bytes": topk_onchip_bytes,
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                    "phase": "knn_dynamic_threshold",
                    "refine_rounds": refine_rounds,
                    "num_blocks": num_blocks,
                },
            )
            all_events.append(fine_ev)
            eid += 1
        return all_events

    # Baseline Phase2 + KNN 引擎交错：每查询 [距离计算] -> [排序/筛选]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    per_query_cycles = _phase2_distance_cycles_per_query(input_points, hw_config)
    knn_mode = str(getattr(hw_config, "knn_engine_mode", "baseline")).lower()
    use_fuse = knn_mode == "fuse"
    all_events = []
    current_cycle = 0
    ob = getattr(getattr(hw_config, "sram_cim_macro", None), "output_buffer", None)
    write_power = float(ob.get_write_power_mw(clock_ns) if (ob is not None and hasattr(ob, "get_write_power_mw")) else getattr(ob, "power", 0.5) if ob is not None else 0.5)
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
        dist_write_ev = Event(
            len(all_events), current_cycle, "KNN", "sram_write", "distance_write", "output_buffer", "sram_write",
            write_power, 1, clock_ns,
            metadata={"query": q, "buffer_name": "dist_sram_buffer", "buffer_phase": "write", "bytes": int(input_points) * 4, "resource_id": "dist_sram_shared", "exclusive": True, "reset_scope": "query_boundary"},
        )
        all_events.append(dist_write_ev)
        current_cycle += 1

        if use_fuse:
            fuse_events = knn_op.generate_single_fuse_events()
            for fe in fuse_events:
                fe.cycle += current_cycle
            all_events.extend(fuse_events)
            current_cycle = max(fe.cycle + fe.duration for fe in fuse_events)
        elif getattr(hw_config, "is_baseline_v2", False):
            reg_read_ev = knn_op._generate_merge_sort_reg_file_read_event(
                current_cycle, len(all_events), num_reads=32,
            )
            if reg_read_ev is not None:
                all_events.append(reg_read_ev)
                current_cycle += reg_read_ev.duration
            one_sort = knn_op._generate_single_merge_sort_event()
            one_sort.cycle = current_cycle
            meta = dict(getattr(one_sort, "metadata", {}) or {})
            meta.update({"buffer_name": "sort_cache_sram", "buffer_phase": "sort_read", "bytes": 32 * 4, "resource_id": "merge_sort_shared", "exclusive": True, "reset_scope": "batch_boundary"})
            one_sort.metadata = meta
            all_events.append(one_sort)
            current_cycle += one_sort.duration
            reg_ev = knn_op._generate_merge_sort_reg_file_event(
                current_cycle, len(all_events), num_writes=32, overlapped=True,
            )
            if reg_ev is not None:
                all_events.append(reg_ev)
        else:
            one_bitonic = knn_op.generate_single_bitonic_event()
            one_bitonic.cycle = current_cycle
            meta = dict(getattr(one_bitonic, "metadata", {}) or {})
            meta.update({"buffer_name": "dist_sram_buffer", "buffer_phase": "sort_read", "bytes": int(input_points) * 4})
            one_bitonic.metadata = meta
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


def _gen_div(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """Element-wise division generic model."""
    return gen_div_events(op_ref.params, hw_config)


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
    'div':                   _gen_div,
    'division':              _gen_div,
    'element_divide':        _gen_div,
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
