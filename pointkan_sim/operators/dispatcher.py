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

Conv1d/Linear 的全网络 CIM（NeuroSim 注入）事件使用 constants 中的 OP_NAME_CONV1D/OP_NAME_LINEAR，
保证 operator_sequence.txt 中算子名统一为 PascalCase（Conv1d, Linear）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING, Tuple

from ..constants import OP_NAME_CONV1D, OP_NAME_LINEAR
from ..simulator.event import Event
from ..integration.cim_full_context import (
    ACC_CIM_KEY_PRELOADED,
    CIM_CLOCK_PERIOD_NS,
    CIM_FULL_BACKEND,
    CIM_LAT_ENERGY_BY_NAME,
    CIM_LAYER_AREA_UM2,
    CIM_LAYER_ENERGY_NJ,
    CIM_LAYER_INDEX,
    CIM_LAYER_LATENCY_NS,
    CIM_LAYER_NAMES,
)
from ..integration.memory_model_adapter import query_sram_cycles_energy, query_dram_batch_cycles_energy, query_dram_cycles_energy, query_dram_issue_cycles
from ..integration.rram_matrix_estimator import compute_rram_dist_primitives
from ..memory.memory_contract import MemoryContractPolicy
from ..utils.shared_helpers import index_bytes_for_points as _index_bytes_for_points
from ..utils.math_utils import safe_div_ceil
from ..utils.shared_helpers import retag_events as _retag_events
from ..utils.shared_helpers import split_bytes as _split_bytes
from .common_ops import (
    _clock as _base_clock_ns,
    _normalized_scheduler_timing,
    _use_fixed_neurosim_subarray,
    gen_conv1d_rram_cim_events,
    gen_conv1d_systolic_events,
    get_conv1d_per_group_cycles,
    gen_pointwise_conv_per_group_events,
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
    gen_sa_max_pool_events,
    gen_add_events,
    gen_mul_events,
    gen_div_events,
    gen_interp_n3_events,
    gen_interp_n3_per_query_events,
)
from .distance_ops import gen_distance_events_one_batch
from .neurosim_adapter import estimate_linear_neurosim

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
    expected = 'conv' if op_ref.op_type in ('conv1d', 'conv2d') else 'fc'
    if getattr(info, 'layer_type', None) != expected:
        return params
    context['real_conv_linear_index'] = idx + 1
    params['progress_index'] = idx + 1
    params['progress_total'] = len(real_list)
    params['real_module_name'] = getattr(info, 'name', '')
    if getattr(info, 'input_shape', None):
        params['activation_input_shape'] = _shape_to_str(getattr(info, 'input_shape', ()))
    if getattr(info, 'output_shape', None):
        params['activation_output_shape'] = _shape_to_str(getattr(info, 'output_shape', ()))
    if op_ref.op_type in ('conv1d', 'conv2d'):
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


def _resolve_cim_name_candidates(op_ref: 'OperatorRef', context: Optional[Dict[str, Any]]) -> List[str]:
    """
    Resolve best-effort name candidates for cim_full by-name lookup.

    OperatorRef currently has no ``name`` field. We therefore search in:
      1) op_ref.params["real_module_name"] / ["layer_name"] / ["name"]
      2) op_ref.description (last resort)
    """
    cands: List[str] = []
    params = getattr(op_ref, "params", {}) or {}
    for k in ("real_module_name", "layer_name", "name"):
        v = str(params.get(k, "") or "").strip()
        if v:
            cands.append(v)
    desc = str(getattr(op_ref, "description", "") or "").strip()
    if desc:
        cands.append(desc)
    out: List[str] = []
    seen = set()
    for v in cands:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


_ACC_MEM_POLICY = MemoryContractPolicy()


def _acc_buffer_config(hw_config: Optional["HardwareConfig"], buffer_name: str) -> Optional[Any]:
    if hw_config is None:
        return None
    return getattr(hw_config, buffer_name, None)


def _acc_buffer_capacity_bytes(buffer_name: str, hw_config: Optional["HardwareConfig"] = None) -> int:
    cfg = _acc_buffer_config(hw_config, buffer_name)
    if cfg is not None:
        size_bytes = int(getattr(cfg, "size_bytes", 0) or 0)
        if size_bytes > 0:
            return size_bytes
    p = _ACC_MEM_POLICY
    mapping = {
        "query_xyz_sram_buffer": max(1, p.cim_batch_points * p.point_xyz_onchip_bytes),
        "self_distance_sram_buffer": max(1, p.self_distance_sram_bytes) if p.self_distance_sram_bytes > 0 else (1 << 60),
        "dist_sram_buffer": max(1, p.dist_sram_bytes) if p.dist_sram_bytes > 0 else (1 << 60),
        "xyz_sram_buffer": max(1, p.xyz_sram_bytes),
        "fps_min_distance_table_sram": max(1, p.fps_min_distance_table_sram_bytes) if p.fps_min_distance_table_sram_bytes > 0 else (1 << 60),
        "temp_max_distance_sram": max(1, p.temp_max_distance_sram_bytes),
        "topk_sram_buffer": max(1, p.topk_sram_bytes),
        # Persisted FPS indices are provisioned as a small dedicated SRAM in hw_config.
        # MemoryContractPolicy tracks runtime traffic but does not carry sampled_points_num.
        "fps_index_sram_buffer": 2 * 1024,
        "sort_cache_sram": max(1, p.sort_cache_sram_bytes),
    }
    return int(mapping.get(buffer_name, 1 << 60))


def _acc_buffer_spill_to_dram(buffer_name: str, hw_config: Optional["HardwareConfig"] = None) -> bool:
    cfg = _acc_buffer_config(hw_config, buffer_name)
    if cfg is not None:
        return bool(getattr(cfg, "spill_to_dram", False))
    return False


def _fps_mdt_entry_bytes() -> int:
    # Runtime FPS min-distance table stores only the persistent min distance per point.
    return 4


def _fps_mdt_total_bytes(num_points: int) -> int:
    return max(1, int(num_points)) * _fps_mdt_entry_bytes()


def _fps_mdt_update_block_points(hw_config: "HardwareConfig") -> int:
    if getattr(hw_config, "is_acc", False):
        return max(1, int(getattr(hw_config, "cim_parallelism", 128) or 128))
    dist2_eng = getattr(hw_config, "baseline_dist2_engine", None)
    return max(1, int(getattr(dist2_eng, "pe_count", getattr(dist2_eng, "c_dim", 64)) or 64))


def _fps_mdt_superblock_points(num_points: int, hw_config: "HardwareConfig") -> int:
    total_points = max(1, int(num_points))
    granularity = _fps_mdt_update_block_points(hw_config)
    cap_entries = max(1, _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config) // _fps_mdt_entry_bytes())
    aligned = max(granularity, (cap_entries // granularity) * granularity)
    return max(granularity, min(total_points, aligned))


def _fps_mdt_superblock_bytes(num_points: int, superblock_idx: int, hw_config: "HardwareConfig") -> int:
    sb_points = _fps_mdt_superblock_points(num_points, hw_config)
    sb_start = max(0, int(superblock_idx)) * sb_points
    pts = min(sb_points, max(0, int(num_points) - sb_start))
    return max(1, pts * _fps_mdt_entry_bytes())


def _fps_mdt_superblock_index_for_block(block_idx: int, num_points: int, hw_config: "HardwareConfig") -> int:
    block_points = _fps_mdt_update_block_points(hw_config)
    sb_points = _fps_mdt_superblock_points(num_points, hw_config)
    blocks_per_superblock = max(1, int(math.ceil(sb_points / float(block_points))))
    return max(0, int(block_idx)) // blocks_per_superblock


def _fps_mdt_segment_bytes(num_points: int, block_idx: int, hw_config: "HardwareConfig") -> int:
    block_size = _fps_mdt_update_block_points(hw_config)
    blk_points = min(block_size, max(1, int(num_points) - int(block_idx) * block_size))
    return max(1, blk_points * _fps_mdt_entry_bytes())


def _acc_query_point_offchip_bytes() -> int:
    return max(1, int(_ACC_MEM_POLICY.point_xyz_offchip_bytes))


def _gen_baseline_phase2_distance_events(
    num_keys: int,
    hw_config: "HardwareConfig",
    operator_name: str,
    *,
    start_cycle: int = 0,
    start_event_id: int = 0,
    include_query_dram_read: bool = True,
    num_features: int = 3,
    bits_per_feature: int = 12,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Event], int, int]:
    """Generate one baseline Phase2 distance pass with explicit key preload."""
    if num_keys <= 0:
        return [], start_event_id, start_cycle

    clk = float(getattr(hw_config, "clock_period", 1.0))
    event_id = int(start_event_id)
    cycle = int(start_cycle)
    events: List[Event] = []
    base_meta = dict(metadata or {})

    point_bytes = int(math.ceil(max(1, num_features) * max(1, bits_per_feature) / 8.0))
    query_bytes = point_bytes
    query_chunk_bytes = query_bytes
    if include_query_dram_read and query_bytes > 0:
        query_dram_results = query_dram_batch_cycles_energy(
            [{"num_bytes": query_bytes, "is_read": True, "start_cycle": cycle}],
            hw_config,
            clk,
        )
        dram_cycles, dram_energy_nj = query_dram_results[0]
        dram_power = (
            dram_energy_nj * 1000.0 / (max(1, dram_cycles) * clk)
            if dram_cycles > 0 and clk > 0
            else 0.0
        )
        dram_meta = dict(base_meta)
        dram_meta.update(
            {
                "bytes": query_bytes,
                "phase": 2,
                "kind": "query_fetch",
                "resource_id": "dram_port",
                "dram_pipelined": True,
                "dram_issue_cycles": query_dram_issue_cycles(query_bytes, hw_config),
            }
        )
        events.append(
            Event(
                event_id,
                cycle,
                operator_name,
                "distance",
                "query_dram_read",
                "dram",
                "phase2_query_point_read",
                dram_power,
                dram_cycles,
                clk,
                metadata=dram_meta,
            )
        )
        event_id += 1
        cycle += dram_cycles

        # Producer-consumer: query DRAM → query_xyz_sram_buffer write → read
        # The query point must be written to on-chip SRAM before dist2_engine
        # can use it for distance computation against key batches.
        event_id, qsram_write_dur, _ = _acc_append_buffer_fill_events(
            events, event_id, cycle, operator_name, "distance",
            sub_stage="query_buffer_write",
            circuit_module="query_xyz_sram_buffer",
            operation="query_xyz_buffer_write",
            buffer_name="query_xyz_sram_buffer",
            phase="query_refill",
            resource_id="query_xyz_sram_shared",
            total_bytes=query_bytes,
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=query_chunk_bytes,
            extra_meta={**base_meta, "bytes": query_bytes, "phase": 2, "kind": "query_store"},
        )
        cycle += qsram_write_dur

        event_id, qsram_read_dur = _acc_append_buffer_consume_events(
            events, event_id, cycle, operator_name, "distance",
            sub_stage="query_buffer_read",
            circuit_module="query_xyz_sram_buffer",
            operation="query_xyz_buffer_read",
            buffer_name="query_xyz_sram_buffer",
            phase="query_read",
            resource_id="query_xyz_sram_shared",
            total_bytes=query_bytes,
            hw_config=hw_config,
            resident_bytes=min(query_bytes, _acc_buffer_capacity_bytes("query_xyz_sram_buffer", hw_config)),
            preferred_chunk_bytes=query_chunk_bytes,
            extra_meta={**base_meta, "bytes": query_bytes, "phase": 2, "kind": "query_load"},
        )
        cycle += qsram_read_dur

    eng = getattr(hw_config, "baseline_dist2_engine", None)
    batch_points = max(1, int(getattr(eng, "pe_count", getattr(eng, "c_dim", 64)) or 64))
    num_batches = int(math.ceil(num_keys / float(batch_points)))
    per_batch_cycles = int(getattr(eng, "cycles", 0) or 0)
    if per_batch_cycles <= 0:
        per_distance_cycles = max(1, int(getattr(eng, "per_distance_cycles", 7) or 7))
        per_batch_cycles = batch_points - 1 + per_distance_cycles
    xyz_chunk_bytes = batch_points * point_bytes
    dist_chunk_bytes = batch_points * 4
    dist_resident_bytes = 0
    key_dram_requests: List[Dict[str, Any]] = []
    batch_cycle_cursor = int(cycle)
    for batch_idx in range(num_batches):
        batch_m = min(batch_points, num_keys - batch_idx * batch_points)
        batch_bytes = max(1, int(batch_m) * point_bytes)
        key_dram_requests.append(
            {
                "num_bytes": batch_bytes,
                "is_read": True,
                "start_cycle": batch_cycle_cursor,
            }
        )
        predicted_dram_issue = int(query_dram_issue_cycles(batch_bytes, hw_config))
        batch_cycle_cursor += max(1, predicted_dram_issue)
        sram_wr_cycles, _ = query_sram_cycles_energy(
            batch_bytes,
            is_read=False,
            hw_config=hw_config,
            clock_period_ns=clk,
            buffer_name="xyz_sram_buffer",
        )
        batch_cycle_cursor += int(sram_wr_cycles)
        sram_rd_cycles, _ = query_sram_cycles_energy(
            batch_bytes,
            is_read=True,
            hw_config=hw_config,
            clock_period_ns=clk,
            buffer_name="xyz_sram_buffer",
        )
        batch_cycle_cursor += int(sram_rd_cycles)
        batch_cycle_cursor += per_batch_cycles
    key_dram_results = query_dram_batch_cycles_energy(key_dram_requests, hw_config, clk)
    for batch_idx in range(num_batches):
        batch_m = min(batch_points, num_keys - batch_idx * batch_points)
        batch_bytes = max(1, int(batch_m) * point_bytes)
        batch_meta = dict(base_meta)
        batch_meta.update(
            {
                "batch": batch_idx,
                "num_batches": num_batches,
                "batch_size_m": batch_m,
                "num_keys": num_keys,
                "phase": 2,
            }
        )

        key_dram_cycles, key_dram_energy_nj = key_dram_results[batch_idx]
        key_dram_power = (
            key_dram_energy_nj * 1000.0 / (max(1, key_dram_cycles) * clk)
            if key_dram_cycles > 0 and clk > 0
            else 0.0
        )
        key_dram_meta = dict(batch_meta)
        key_dram_meta.update(
            {
                "bytes": batch_bytes,
                "buffer_name": "xyz_sram_buffer",
                "buffer_phase": "refill_read",
                "resource_id": "dram_port",
                "dram_pipelined": True,
                "dram_issue_cycles": query_dram_issue_cycles(batch_bytes, hw_config),
            }
        )
        events.append(
            Event(
                event_id,
                cycle,
                operator_name,
                "distance",
                "key_dram_read",
                "dram",
                "phase2_key_batch_read",
                key_dram_power,
                max(1, key_dram_cycles),
                clk,
                metadata=key_dram_meta,
            )
        )
        event_id += 1
        cycle += max(1, key_dram_cycles)

        event_id, key_write_dur, _ = _acc_append_buffer_fill_events(
            events,
            event_id,
            cycle,
            operator_name,
            "distance",
            sub_stage="xyz_buffer_write",
            circuit_module="xyz_sram_buffer",
            operation="xyz_buffer_write",
            buffer_name="xyz_sram_buffer",
            phase="write",
            resource_id="xyz_sram_shared",
            total_bytes=batch_bytes,
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=xyz_chunk_bytes,
            extra_meta={**batch_meta, "bytes_src_offchip": batch_bytes, "bytes_dst_onchip": batch_bytes},
        )
        cycle += key_write_dur

        event_id, key_read_dur = _acc_append_buffer_consume_events(
            events,
            event_id,
            cycle,
            operator_name,
            "distance",
            sub_stage="xyz_buffer_key_read",
            circuit_module="xyz_sram_buffer",
            operation="xyz_buffer_key_read",
            buffer_name="xyz_sram_buffer",
            phase="read",
            resource_id="xyz_sram_shared",
            total_bytes=batch_bytes,
            hw_config=hw_config,
            resident_bytes=min(batch_bytes, _acc_buffer_capacity_bytes("xyz_sram_buffer", hw_config)),
            preferred_chunk_bytes=xyz_chunk_bytes,
            extra_meta=batch_meta,
        )
        cycle += key_read_dur

        dist_events, event_id, cycle = gen_distance_events_one_batch(
            operator_name,
            hw_config,
            event_id,
            cycle,
            batch_meta,
            batch_size_m=batch_m,
        )
        events.extend(dist_events)

        # Producer-consumer: dist2_engine output → dist_sram_buffer write
        # Each batch produces batch_m distance values (FP32) that must be
        # written to the on-chip distance buffer before being consumed by
        # the downstream sort / min-distance-table update stage.
        batch_output_bytes = max(1, int(batch_m) * 4)
        event_id, dist_out_dur, dist_resident_bytes = _acc_append_buffer_fill_events(
            events, event_id, cycle, operator_name, "distance",
            sub_stage="dist_output_write",
            circuit_module="dist_sram_buffer",
            operation="dist2_output_buffer_write",
            buffer_name="dist_sram_buffer",
            phase="batch_output",
            resource_id="dist_sram_shared",
            total_bytes=batch_output_bytes,
            hw_config=hw_config,
            is_read=False,
            resident_bytes_before=dist_resident_bytes,
            preferred_chunk_bytes=dist_chunk_bytes,
            extra_meta={
                **batch_meta,
                "bytes": batch_output_bytes,
                "kind": "dist_output",
            },
        )
        cycle += dist_out_dur

    return events, event_id, cycle


def _acc_query_point_onchip_bytes() -> int:
    return max(1, int(_ACC_MEM_POLICY.point_xyz_onchip_bytes))


def _distance_index_entry_bytes(num_points: int) -> int:
    return int(_ACC_MEM_POLICY.distance_bytes + _index_bytes_for_points(num_points))


def _sorter_depth_points(hw_config: Optional["HardwareConfig"], fallback: int = 1024) -> int:
    sort_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "bitonic_sort", None) if hw_config is not None else None
    depth = int(getattr(sort_cfg, "sorter_depth", 0) or 0) if sort_cfg is not None else 0
    return max(1, depth if depth > 0 else fallback)


def _group_ready_cycles(events: List[Event], operation: str, key_field: str = "batch") -> Dict[int, int]:
    ready: Dict[int, int] = {}
    for ev in events:
        if getattr(ev, "operation", "") != operation:
            continue
        meta = getattr(ev, "metadata", {}) or {}
        if key_field not in meta:
            continue
        try:
            key = int(meta[key_field])
        except Exception:
            continue
        ready[key] = max(ready.get(key, 0), int(getattr(ev, "cycle", 0)) + int(getattr(ev, "duration", 0)))
    return ready


def _acc_spill_bytes(buffer_name: str, total_bytes: int, hw_config: Optional["HardwareConfig"] = None) -> Tuple[int, int]:
    total = max(0, int(total_bytes))
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    if total <= cap:
        return total, 0
    if not _acc_buffer_spill_to_dram(buffer_name, hw_config):
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
                             direction: str, hw_config: 'HardwareConfig', memory_context: Optional[Any] = None,
                             precomputed_result: Optional[Dict[str, Any]] = None,
                             extra_meta: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    spill = max(0, int(spill_bytes))
    if spill <= 0:
        return event_id, 0
    is_read = direction == "read"
    if precomputed_result is not None:
        dur = int(precomputed_result.get("cycles", 0) or 0)
        energy_nj = float(precomputed_result.get("energy_nj", 0.0) or 0.0)
    elif memory_context is not None:
        res = memory_context.dram_transfer(spill, is_read=is_read, cur_cycle=cycle)
        dur = int(res.get("cycles", 0) or 0)
        energy_nj = float(res.get("energy_nj", 0.0) or 0.0)
    else:
        dur, energy_nj = query_dram_cycles_energy(spill, is_read=is_read, hw_config=hw_config, clock_period_ns=clock_ns)
    meta = {
            "owner_operator": owner_operator,
            "buffer_name": buffer_name,
            "buffer_phase": 'spill_read' if is_read else 'spill_write',
            "bytes": spill,
            "resource_id": "dram_port",
            "dram_pipelined": True,
            "dram_issue_cycles": query_dram_issue_cycles(spill, hw_config),
            "phase": phase,
        }
    if extra_meta:
        meta.update(extra_meta)
    ev = _acc_mem_event(
        event_id, cycle, "DRAM", stage, f"{buffer_name}_{'spill_read' if is_read else 'spill_write'}", "dram", f"{buffer_name}_{'spill_read' if is_read else 'spill_write'}",
        dur, energy_nj, clock_ns, meta,
    )
    out.append(ev)
    return event_id + 1, dur


def _acc_append_sram_access_event(out: List[Event], event_id: int, cycle: int, owner_operator: str, stage: str,
                                  sub_stage: str, circuit_module: str, operation: str, buffer_name: str,
                                  phase: str, resource_id: str, num_bytes: int, hw_config: 'HardwareConfig',
                                  *, is_read: bool, overlapped: bool = False, extra_meta: Optional[Dict[str, Any]] = None,
                                  return_actual_duration: bool = False) -> Tuple[int, int]:
    onchip = max(0, int(num_bytes))
    if onchip <= 0:
        return event_id, 0
    clk = float(getattr(hw_config, "clock_period", 1.0))
    dur, energy_nj = query_sram_cycles_energy(
        onchip,
        is_read=is_read,
        hw_config=hw_config,
        clock_period_ns=clk,
        buffer_name=buffer_name,
    )
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
    ev = _acc_mem_event(event_id, cycle, "SRAM", stage, sub_stage, circuit_module, operation, dur, energy_nj, clk, meta)
    ev.overlapped = bool(overlapped)
    out.append(ev)
    if return_actual_duration:
        return event_id + 1, dur
    return event_id + 1, (0 if overlapped else dur)


def _acc_buffer_stream_chunk_bytes(
    buffer_name: str,
    total_bytes: int,
    hw_config: 'HardwareConfig',
    *,
    preferred_chunk_bytes: Optional[int] = None,
) -> int:
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    total = max(1, int(total_bytes))
    if preferred_chunk_bytes is None:
        return min(cap, total)
    return max(1, min(cap, total, int(preferred_chunk_bytes)))


def _acc_plan_fill_spill_requests(
    *,
    buffer_name: str,
    total_bytes: int,
    cycle: int,
    hw_config: 'HardwareConfig',
    resident_bytes_before: int = 0,
    preferred_chunk_bytes: Optional[int] = None,
    is_read: bool,
) -> List[Dict[str, Any]]:
    total = max(0, int(total_bytes))
    if total <= 0 or not _acc_buffer_spill_to_dram(buffer_name, hw_config):
        return []
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    resident = max(0, min(cap, int(resident_bytes_before)))
    chunk_bytes = _acc_buffer_stream_chunk_bytes(
        buffer_name,
        total,
        hw_config,
        preferred_chunk_bytes=preferred_chunk_bytes,
    )
    clk = float(getattr(hw_config, "clock_period", 1.0))
    elapsed = 0
    requests: List[Dict[str, Any]] = []
    for chunk in _split_bytes(total, chunk_bytes):
        free_bytes = max(0, cap - resident)
        onchip = min(chunk, free_bytes)
        spill = max(0, chunk - onchip)
        if onchip > 0:
            sram_dur, _ = query_sram_cycles_energy(
                onchip,
                is_read=is_read,
                hw_config=hw_config,
                clock_period_ns=clk,
                buffer_name=buffer_name,
            )
            elapsed += int(sram_dur)
            resident += onchip
        if spill > 0:
            requests.append({
                "start_cycle": int(cycle + elapsed),
                "num_bytes": spill,
                "is_read": bool(is_read),
            })
            dram_dur, _ = query_dram_cycles_energy(
                spill,
                is_read=is_read,
                hw_config=hw_config,
                clock_period_ns=clk,
            )
            elapsed += int(dram_dur)
    return requests


def _acc_plan_consume_spill_requests(
    *,
    buffer_name: str,
    total_bytes: int,
    cycle: int,
    hw_config: 'HardwareConfig',
    resident_bytes: Optional[int] = None,
    preferred_chunk_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    total = max(0, int(total_bytes))
    if total <= 0 or not _acc_buffer_spill_to_dram(buffer_name, hw_config):
        return []
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    resident = min(total, cap if resident_bytes is None else max(0, int(resident_bytes)))
    chunk_bytes = _acc_buffer_stream_chunk_bytes(
        buffer_name,
        total,
        hw_config,
        preferred_chunk_bytes=preferred_chunk_bytes,
    )
    clk = float(getattr(hw_config, "clock_period", 1.0))
    elapsed = 0
    consumed = 0
    requests: List[Dict[str, Any]] = []
    for chunk in _split_bytes(total, chunk_bytes):
        chunk_start = consumed
        chunk_end = consumed + chunk
        onchip_start = min(chunk_start, resident)
        onchip_end = min(chunk_end, resident)
        onchip = max(0, onchip_end - onchip_start)
        spill = max(0, chunk - onchip)
        if spill > 0:
            requests.append({
                "start_cycle": int(cycle + elapsed),
                "num_bytes": spill,
                "is_read": True,
            })
            dram_dur, _ = query_dram_cycles_energy(
                spill,
                is_read=True,
                hw_config=hw_config,
                clock_period_ns=clk,
            )
            elapsed += int(dram_dur)
        sram_bytes = chunk if _acc_buffer_spill_to_dram(buffer_name, hw_config) else max(chunk, onchip)
        sram_dur, _ = query_sram_cycles_energy(
            sram_bytes,
            is_read=True,
            hw_config=hw_config,
            clock_period_ns=clk,
            buffer_name=buffer_name,
        )
        elapsed += int(sram_dur)
        consumed += chunk
    return requests


def _acc_precompute_fill_batch_results(
    memory_context: Optional[Any],
    *,
    buffer_name: str,
    total_bytes: int,
    cycle: int,
    hw_config: 'HardwareConfig',
    resident_bytes_before: int = 0,
    preferred_chunk_bytes: Optional[int] = None,
    dir_prefix: str,
) -> Optional[List[Dict[str, Any]]]:
    if memory_context is None:
        return None
    reqs = _acc_plan_fill_spill_requests(
        buffer_name=buffer_name,
        total_bytes=total_bytes,
        cycle=cycle,
        hw_config=hw_config,
        resident_bytes_before=resident_bytes_before,
        preferred_chunk_bytes=preferred_chunk_bytes,
        is_read=False,
    )
    if not reqs:
        return None
    return memory_context.prime_explicit_batch_transfers(reqs, dir_prefix=dir_prefix)


def _acc_precompute_consume_batch_results(
    memory_context: Optional[Any],
    *,
    buffer_name: str,
    total_bytes: int,
    cycle: int,
    hw_config: 'HardwareConfig',
    resident_bytes: Optional[int] = None,
    preferred_chunk_bytes: Optional[int] = None,
    dir_prefix: str,
) -> Optional[List[Dict[str, Any]]]:
    if memory_context is None:
        return None
    reqs = _acc_plan_consume_spill_requests(
        buffer_name=buffer_name,
        total_bytes=total_bytes,
        cycle=cycle,
        hw_config=hw_config,
        resident_bytes=resident_bytes,
        preferred_chunk_bytes=preferred_chunk_bytes,
    )
    if not reqs:
        return None
    return memory_context.prime_explicit_batch_transfers(reqs, dir_prefix=dir_prefix)


def _acc_precompute_banked_startup_dram_results(
    *,
    bank_specs: List[Dict[str, int]],
    preload_cursor: int,
    operator_name: str,
    query_index: int,
    num_banks: int,
    hw_config: 'HardwareConfig',
    clk: float,
    q12_cycles: int,
    memory_context: Optional[Any] = None,
) -> Tuple[List[int], List[Dict[str, Any]], List[int], List[int]]:
    if not bank_specs:
        return [], [], [], []

    setup_overheads: List[int] = []
    for spec in bank_specs:
        bank_idx = spec["bank_idx"]
        bank_keys = spec["bank_keys"]
        bank_xyz_sram_bytes = spec["xyz_sram_bytes"]
        dummy_events: List[Event] = []
        _eid = 0
        _eid, xyz_sram_write_dur, _ = _acc_append_buffer_fill_events(
            dummy_events,
            _eid,
            0,
            operator_name,
            "distance",
            sub_stage="xyz_buffer_write",
            circuit_module="xyz_sram_buffer",
            operation="xyz_buffer_write",
            buffer_name="xyz_sram_buffer",
            phase="write",
            resource_id=f"xyz_sram_bank{bank_idx}",
            total_bytes=bank_xyz_sram_bytes,
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=bank_xyz_sram_bytes,
            extra_meta={
                "query_index": query_index,
                "bank_idx": bank_idx,
                "num_banks": num_banks,
                "bank_keys": bank_keys,
                "phase": "distance_frontend_banked",
            },
            memory_context=None,
        )
        _eid, xyz_key_read_dur = _acc_append_buffer_consume_events(
            dummy_events,
            _eid,
            0,
            operator_name,
            "distance",
            sub_stage="xyz_buffer_key_read",
            circuit_module="xyz_sram_buffer",
            operation="xyz_buffer_key_read",
            buffer_name="xyz_sram_buffer",
            phase="read",
            resource_id=f"xyz_sram_bank{bank_idx}",
            total_bytes=bank_xyz_sram_bytes,
            hw_config=hw_config,
            resident_bytes=bank_xyz_sram_bytes,
            preferred_chunk_bytes=max(1, spec["bank_keys"] * _acc_query_point_onchip_bytes()),
            extra_meta={
                "query_index": query_index,
                "bank_idx": bank_idx,
                "num_banks": num_banks,
                "bank_keys": bank_keys,
                "phase": "distance_frontend_banked",
            },
            memory_context=None,
        )
        setup_overheads.append(int(q12_cycles + xyz_sram_write_dur + xyz_key_read_dur))

    def _build_requests(start_cycles: List[int]) -> List[Dict[str, Any]]:
        return [
            {
                "num_bytes": int(spec["xyz_dram_bytes"]),
                "is_read": True,
                "start_cycle": int(start),
            }
            for spec, start in zip(bank_specs, start_cycles)
        ]

    estimated_starts: List[int] = []
    cursor = int(preload_cursor)
    for spec, setup_overhead in zip(bank_specs, setup_overheads):
        estimated_starts.append(cursor)
        cursor += max(1, int(query_dram_issue_cycles(int(spec["xyz_dram_bytes"]), hw_config))) + int(setup_overhead)

    def _run_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if memory_context is not None:
            return memory_context.prime_explicit_batch_transfers(
                requests,
                dir_prefix=f"banked_startup_q{query_index}",
            )
        results = query_dram_batch_cycles_energy(requests, hw_config, clk)
        return [
            {
                "cycles": int(cyc),
                "energy_nj": float(en),
                "start_cycle": int(req["start_cycle"]),
                "end_cycle": int(req["start_cycle"]) + int(cyc),
            }
            for req, (cyc, en) in zip(requests, results)
        ]

    results = _run_requests(_build_requests(estimated_starts))
    exact_starts: List[int] = []
    cursor = int(preload_cursor)
    for res, setup_overhead in zip(results, setup_overheads):
        exact_starts.append(cursor)
        cursor += int(res.get("cycles", 0) or 0) + int(setup_overhead)

    if exact_starts != estimated_starts:
        results = _run_requests(_build_requests(exact_starts))

    dram_cycles = [int(res.get("cycles", 0) or 0) for res in results]
    return exact_starts, results, setup_overheads, dram_cycles


def _acc_append_buffer_fill_events(
    out: List[Event],
    event_id: int,
    cycle: int,
    owner_operator: str,
    stage: str,
    *,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    buffer_name: str,
    phase: str,
    resource_id: str,
    total_bytes: int,
    hw_config: 'HardwareConfig',
    is_read: bool,
    resident_bytes_before: int = 0,
    preferred_chunk_bytes: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    memory_context: Optional[Any] = None,
    precomputed_spill_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int, int]:
    total = max(0, int(total_bytes))
    if total <= 0:
        return event_id, 0, max(0, int(resident_bytes_before))
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    spill_to_dram = _acc_buffer_spill_to_dram(buffer_name, hw_config)
    resident = max(0, min(cap, int(resident_bytes_before)))
    chunk_bytes = _acc_buffer_stream_chunk_bytes(
        buffer_name,
        total,
        hw_config,
        preferred_chunk_bytes=preferred_chunk_bytes,
    )
    elapsed = 0
    chunks = _split_bytes(total, chunk_bytes)
    chunk_count = len(chunks)
    spill_result_idx = 0
    for chunk_idx, chunk in enumerate(chunks):
        free_bytes = max(0, cap - resident)
        onchip = min(chunk, free_bytes) if spill_to_dram else chunk
        spill = max(0, chunk - onchip)
        if onchip > 0:
            chunk_meta = dict(extra_meta or {})
            if "bytes" in chunk_meta:
                chunk_meta["logical_bytes"] = int(chunk_meta.get("bytes", 0) or 0)
                chunk_meta.pop("bytes", None)
            chunk_meta.update(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_count": chunk_count,
                    "chunk_bytes": chunk,
                    "resident_bytes_before": resident,
                    "resident_bytes_after": resident + onchip,
                    "producer_consumer_mode": "chunked_fill",
                }
            )
            req_start = cycle + elapsed
            event_id, dur = _acc_append_sram_access_event(
                out,
                event_id,
                req_start,
                owner_operator,
                stage,
                sub_stage,
                circuit_module,
                operation,
                buffer_name,
                phase,
                resource_id,
                onchip,
                hw_config,
                is_read=is_read,
                overlapped=False,
                extra_meta=chunk_meta,
                return_actual_duration=True,
            )
            ev = out[-1]
            actual_end = int(getattr(ev, 'cycle', req_start)) + int(getattr(ev, 'duration', dur))
            elapsed = max(elapsed, actual_end - cycle)
            resident += onchip
        if spill > 0:
            spill_phase = phase if phase.endswith("_spill") else f"{phase}_spill"
            req_start = cycle + elapsed
            event_id, spill_dur = _acc_append_spill_events(
                out,
                event_id,
                req_start,
                owner_operator,
                stage,
                buffer_name,
                spill,
                float(getattr(hw_config, "clock_period", 1.0)),
                spill_phase,
                "read" if is_read else "write",
                hw_config,
                memory_context,
                precomputed_result=(
                    precomputed_spill_results[spill_result_idx]
                    if precomputed_spill_results is not None and spill_result_idx < len(precomputed_spill_results)
                    else None
                ),
            )
            spill_result_idx += 1
            ev = out[-1]
            actual_end = int(getattr(ev, 'cycle', req_start)) + int(getattr(ev, 'duration', spill_dur))
            elapsed = max(elapsed, actual_end - cycle)
    return event_id, elapsed, resident


def _acc_append_buffer_consume_events(
    out: List[Event],
    event_id: int,
    cycle: int,
    owner_operator: str,
    stage: str,
    *,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    buffer_name: str,
    phase: str,
    resource_id: str,
    total_bytes: int,
    hw_config: 'HardwareConfig',
    resident_bytes: Optional[int] = None,
    preferred_chunk_bytes: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    memory_context: Optional[Any] = None,
    precomputed_spill_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int]:
    total = max(0, int(total_bytes))
    if total <= 0:
        return event_id, 0
    cap = max(1, _acc_buffer_capacity_bytes(buffer_name, hw_config))
    spill_to_dram = _acc_buffer_spill_to_dram(buffer_name, hw_config)
    resident = min(total, cap if resident_bytes is None else max(0, int(resident_bytes)))
    chunk_bytes = _acc_buffer_stream_chunk_bytes(
        buffer_name,
        total,
        hw_config,
        preferred_chunk_bytes=preferred_chunk_bytes,
    )
    elapsed = 0
    chunk_list = _split_bytes(total, chunk_bytes)
    consumed = 0
    spill_result_idx = 0
    for chunk_idx, chunk in enumerate(chunk_list):
        chunk_start = consumed
        chunk_end = consumed + chunk
        onchip_start = min(chunk_start, resident)
        onchip_end = min(chunk_end, resident)
        onchip = max(0, onchip_end - onchip_start)
        spill = max(0, chunk - onchip)
        if spill > 0 and spill_to_dram:
            spill_phase = phase if phase.endswith("_reload") else f"{phase}_reload"
            req_start = cycle + elapsed
            event_id, spill_dur = _acc_append_spill_events(
                out,
                event_id,
                req_start,
                owner_operator,
                stage,
                buffer_name,
                spill,
                float(getattr(hw_config, "clock_period", 1.0)),
                spill_phase,
                "read",
                hw_config,
                memory_context,
                precomputed_result=(
                    precomputed_spill_results[spill_result_idx]
                    if precomputed_spill_results is not None and spill_result_idx < len(precomputed_spill_results)
                    else None
                ),
            )
            spill_result_idx += 1
            ev = out[-1]
            actual_end = int(getattr(ev, 'cycle', req_start)) + int(getattr(ev, 'duration', spill_dur))
            elapsed = max(elapsed, actual_end - cycle)
        sram_bytes = chunk if spill_to_dram else max(chunk, onchip)
        chunk_meta = dict(extra_meta or {})
        if "bytes" in chunk_meta:
            chunk_meta["logical_bytes"] = int(chunk_meta.get("bytes", 0) or 0)
            chunk_meta.pop("bytes", None)
        chunk_meta.update(
            {
                "chunk_idx": chunk_idx,
                "chunk_count": len(chunk_list),
                "chunk_bytes": chunk,
                "resident_bytes": resident,
                "resident_chunk_bytes": onchip,
                "reload_chunk_bytes": spill if spill_to_dram else 0,
                "producer_consumer_mode": "chunked_consume",
            }
        )
        req_start = cycle + elapsed
        event_id, dur = _acc_append_sram_access_event(
            out,
            event_id,
            req_start,
            owner_operator,
            stage,
            sub_stage,
            circuit_module,
            operation,
            buffer_name,
            phase,
            resource_id,
            sram_bytes,
            hw_config,
            is_read=True,
            overlapped=False,
            extra_meta=chunk_meta,
            return_actual_duration=True,
        )
        ev = out[-1]
        actual_end = int(getattr(ev, 'cycle', req_start)) + int(getattr(ev, 'duration', dur))
        elapsed = max(elapsed, actual_end - cycle)
        consumed += chunk
    return event_id, elapsed


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


def _acc_streaming_downsample_cycles(num_points: int, hw_config: 'HardwareConfig') -> int:
    """
    Streaming downsample pipeline:
      per-point latency = 3 cycles, then 1 point / cycle.
      N points total => N - 1 + 3 = N + 2 cycles.
    """
    return max(1, int(num_points) + 2)


def _acc_query_frontend_cycles(hw_config: 'HardwareConfig') -> int:
    """query_xyz_sram_buffer write/read + xyz_sram_buffer read for one query point."""
    clk = float(getattr(hw_config, "clock_period", 1.0))
    qbytes = _acc_query_point_onchip_bytes()
    r1, _ = query_sram_cycles_energy(qbytes, is_read=True, hw_config=hw_config, clock_period_ns=clk, buffer_name="xyz_sram_buffer")
    w1, _ = query_sram_cycles_energy(qbytes, is_read=False, hw_config=hw_config, clock_period_ns=clk, buffer_name="query_xyz_sram_buffer")
    r2, _ = query_sram_cycles_energy(qbytes, is_read=True, hw_config=hw_config, clock_period_ns=clk, buffer_name="query_xyz_sram_buffer")
    return int(r1 + w1 + r2)


def _pointcim_quant12_power_cycles(hw_config: 'HardwareConfig', num_values: int) -> Tuple[float, int]:
    logic = getattr(hw_config, "logic_circuit", None)
    cfg = getattr(logic, "quant_pipeline_12bit", None) if logic is not None else None
    power_base = float(getattr(cfg, "power", 2.28) or 2.28) if cfg is not None else 2.28
    par = max(1, int(getattr(cfg, "parallelism", 1) or 1)) if cfg is not None else 1
    latency = max(1, int(getattr(cfg, "pipeline_latency", 6) or 6)) if cfg is not None else 6
    power = power_base * par
    cycles = max(1, safe_div_ceil(max(1, int(num_values)), par) + latency)
    return power, cycles


def _pointcim_scope_tag(tag_scope: str, name: str) -> str:
    scope = str(tag_scope or "pointcim_runtime").strip()
    return f"{scope}_{name}"


def _gen_pointcim_distance_query_events(
    start_event_id: int,
    base_cycle: int,
    operator_name: str,
    num_keys: int,
    hw_config: 'HardwareConfig',
    query_index: int = 0,
    startup_once: bool = False,
    write_dist_buffer: bool = True,
    tag_scope: str = "",
) -> Tuple[List[Event], int, int, List[int], List[int], List[str], List[str]]:
    events: List[Event] = []
    eid = start_event_id
    clk = float(getattr(hw_config, "clock_period", 1.0))
    neurosim_freq_hz = getattr(hw_config, "neurosim_clk_freq_hz", None)
    neurosim_clk = (1e9 / float(neurosim_freq_hz)) if neurosim_freq_hz is not None and float(neurosim_freq_hz) > 0 else clk
    num_keys = max(1, int(num_keys))
    rram_cfg = getattr(hw_config, "rram_dist", None)
    if rram_cfg is None:
        return events, eid, 0, [], [], [], []
    rram_cfg._clock_period_ns = float(neurosim_clk)

    prims_total = compute_rram_dist_primitives(rram_cfg, num_keys)
    array_point_counts = list(prims_total.get("array_point_counts") or [])
    if not array_point_counts:
        mapping = dict(prims_total.get("mapping") or {})
        full_points = int(mapping.get("full_points_per_array", max(1, prims_total.get("points_per_array", num_keys))))
        last_points = int(mapping.get("points_in_last_array", min(full_points, num_keys)))
        num_arrays = max(1, int(prims_total.get("num_arrays_total", math.ceil(num_keys / float(max(1, full_points))))))
        array_point_counts = [
            full_points if idx < num_arrays - 1 else last_points
            for idx in range(num_arrays)
            if (full_points if idx < num_arrays - 1 else last_points) > 0
        ]
    num_arrays = max(1, len(array_point_counts))
    resident_array_points = max(array_point_counts) if array_point_counts else num_keys

    dist_merge_cfg = getattr(getattr(hw_config, "sram_cim_macro", None), "dist2_merge_adder", None)
    merge_parallelism_cfg = int(getattr(dist_merge_cfg, "parallelism", 0) or 0) if dist_merge_cfg is not None else 0
    if merge_parallelism_cfg <= 1:
        bq_adder_par = int(getattr(hw_config, "bq_merge_adder_parallelism", 0) or 0)
        if bq_adder_par > 0:
            merge_parallelism_cfg = bq_adder_par
        else:
            merge_parallelism_cfg = int(getattr(hw_config, "cim_parallelism", 128) or 128)
    merge_parallelism = max(1, merge_parallelism_cfg)
    merge_batches = max(1, int(math.ceil(num_keys / float(merge_parallelism))))

    point_offchip_bytes = _acc_query_point_offchip_bytes()
    point_onchip_bytes = _acc_query_point_onchip_bytes()
    query_point_bytes = point_onchip_bytes
    key_dram_bytes = num_keys * point_offchip_bytes
    key_sram_bytes = num_keys * point_onchip_bytes
    self_dist_total_bytes = num_keys * 4
    query_values = 3
    key_values = num_keys * 3

    key_dram_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_key_dram_ready")
    key_quant_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_key_quant_ready")
    key_sram_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_key_sram_ready")
    key_sram_read_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_key_sram_read_ready")
    rram_program_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_rram_program_ready")
    self_k2_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_self_k2_ready")
    self_cache_ready_tag = _pointcim_scope_tag(tag_scope, "pointcim_self_cache_ready")

    do_startup = (query_index == 0) or (not startup_once)
    startup_ready_cycle = base_cycle
    if do_startup:
        dram_cycles, dram_energy_nj = query_dram_cycles_energy(
            key_dram_bytes,
            is_read=True,
            hw_config=hw_config,
            clock_period_ns=clk,
        )
        dram_cycles = max(1, int(dram_cycles))
        dram_power = float(dram_energy_nj) * 1000.0 / max(1.0, dram_cycles * clk) if dram_cycles > 0 else 0.0
        dram_start = base_cycle
        events.append(Event(
            eid, dram_start, "DRAM", "distance", "xyz_dram_read", "dram", "xyz_refill",
            dram_power, dram_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "buffer_name": "xyz_sram_buffer",
                "buffer_phase": "refill_read",
                "bytes": key_dram_bytes,
                "resource_id": "dram_port",
                "dram_pipelined": True,
                "dram_issue_cycles": query_dram_issue_cycles(key_dram_bytes, hw_config),
                "phase": "distance_frontend_pointcim",
                "dependency_tag": key_dram_ready_tag,
                "sim_interface": "dramsim3",
            },
        ))
        eid += 1

        q12_power, q12_cycles = _pointcim_quant12_power_cycles(hw_config, key_values)
        quant_start = dram_start + dram_cycles
        events.append(Event(
            eid, quant_start, "Quant", "distance", "xyz_quantize", "quant_pipeline_12bit", "xyz_quantize",
            q12_power, q12_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "buffer_name": "xyz_sram_buffer",
                "buffer_phase": "quantize",
                "bytes": key_dram_bytes,
                "resource_id": "quant_pipeline_12bit",
                "exclusive": True,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": key_dram_ready_tag,
                "dependency_tag": key_quant_ready_tag,
                "sim_interface": "logic_quant12",
            },
        ))
        eid += 1

        eid, key_write_dur, _ = _acc_append_buffer_fill_events(
            events, eid, quant_start + q12_cycles, operator_name, "distance",
            sub_stage="key_xyz_buffer_write",
            circuit_module="xyz_sram_buffer",
            operation="key_xyz_buffer_write",
            buffer_name="xyz_sram_buffer",
            phase="write",
            resource_id="pointcim_key_xyz_sram",
            total_bytes=key_sram_bytes,
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=key_sram_bytes,
            extra_meta={
                "query_index": query_index,
                "phase": "distance_frontend_pointcim",
                "bytes_src_offchip": key_dram_bytes,
                "bytes_dst_onchip": key_sram_bytes,
                "depends_on_tag": key_quant_ready_tag,
                "dependency_tag": key_sram_ready_tag,
                "sim_interface": "cacti",
            },
        )
        key_read_start = quant_start + q12_cycles + key_write_dur

        eid, key_read_dur = _acc_append_buffer_consume_events(
            events, eid, key_read_start, operator_name, "distance",
            sub_stage="key_xyz_buffer_read",
            circuit_module="xyz_sram_buffer",
            operation="key_xyz_buffer_read",
            buffer_name="xyz_sram_buffer",
            phase="read",
            resource_id="pointcim_key_xyz_sram",
            total_bytes=key_sram_bytes,
            hw_config=hw_config,
            resident_bytes=key_sram_bytes,
            preferred_chunk_bytes=key_sram_bytes,
            extra_meta={
                "query_index": query_index,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": key_sram_ready_tag,
                "dependency_tag": key_sram_read_ready_tag,
                "sim_interface": "cacti",
            },
        )
        program_start = key_read_start + key_read_dur

        write_energy_nj = float(prims_total["write_energy_nj"])
        write_latency_ns = float(prims_total.get("write_latency_ns", max(1, int(prims_total["write_cycles"])) * neurosim_clk))
        write_timing = _normalized_scheduler_timing(
            latency_ns=write_latency_ns,
            energy_nj=write_energy_nj,
            base_clock_ns=clk,
        )
        events.append(Event(
            eid, program_start, operator_name, "distance", "rram_program", "rram_dist_array", "rram_program_keyset",
            float(write_timing["power_mw"]), int(write_timing["duration_cycles"]), clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "num_keys": num_keys,
                "bytes": key_sram_bytes,
                "resource_id": "pointcim_rram_program",
                "exclusive": True,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": key_sram_read_ready_tag,
                "dependency_tag": rram_program_ready_tag,
                "latency_ns": write_latency_ns,
                "energy_nj": write_energy_nj,
                "native_clock_period_ns": float(neurosim_clk),
                "native_cycles": max(1, int(math.ceil(write_latency_ns / max(1e-9, neurosim_clk)))),
                "scheduler_normalized": True,
                "sim_interface": "destiny_rram",
            },
        ))
        eid += 1

        self_k2_energy_nj = float(prims_total["self_k2_energy_nj"])
        self_k2_latency_ns = float(prims_total["self_k2_latency_ns"])
        self_k2_timing = _normalized_scheduler_timing(
            latency_ns=self_k2_latency_ns,
            energy_nj=self_k2_energy_nj,
            base_clock_ns=clk,
        )
        self_k2_start = program_start + int(write_timing["duration_cycles"])
        events.append(Event(
            eid, self_k2_start, operator_name, "distance", "self_k2_init", "rram_dist_array", "rram_self_k2_init",
            float(self_k2_timing["power_mw"]), int(self_k2_timing["duration_cycles"]), clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "num_keys": num_keys,
                "resource_id": "pointcim_rram_self_k2",
                "exclusive": True,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": rram_program_ready_tag,
                "dependency_tag": self_k2_ready_tag,
                "latency_ns": self_k2_latency_ns,
                "energy_nj": self_k2_energy_nj,
                "native_clock_period_ns": float(neurosim_clk),
                "native_cycles": max(1, int(math.ceil(self_k2_latency_ns / max(1e-9, neurosim_clk)))),
                "scheduler_normalized": True,
                "sim_interface": "neurosim_dist",
            },
        ))
        eid += 1

        eid, self_write_dur, _ = _acc_append_buffer_fill_events(
            events, eid, self_k2_start + int(self_k2_timing["duration_cycles"]), operator_name, "distance",
            sub_stage="self_distance_write",
            circuit_module="self_distance_sram_buffer",
            operation="self_distance_write",
            buffer_name="self_distance_sram_buffer",
            phase="write",
            resource_id="pointcim_self_distance_sram",
            total_bytes=self_dist_total_bytes,
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=max(1, merge_parallelism * 4),
            extra_meta={
                "query_index": query_index,
                "num_keys": num_keys,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": self_k2_ready_tag,
                "dependency_tag": self_cache_ready_tag,
                "sim_interface": "cacti",
            },
        )
        startup_ready_cycle = self_k2_start + int(self_k2_timing["duration_cycles"]) + self_write_dur

    # query path
    query_src_ready_tag = _pointcim_scope_tag(tag_scope, f"pointcim_query_src_ready_q{query_index}")
    query_sram_ready_tag = _pointcim_scope_tag(tag_scope, f"pointcim_query_sram_ready_q{query_index}")
    query_read_ready_tag = _pointcim_scope_tag(tag_scope, f"pointcim_query_read_ready_q{query_index}")
    query_self_ready_tag = _pointcim_scope_tag(tag_scope, f"pointcim_query_self_ready_q{query_index}")

    cursor = base_cycle
    eid, xyz_query_read_dur = _acc_append_buffer_consume_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="xyz_buffer_query_read",
        circuit_module="xyz_sram_buffer",
        operation="xyz_buffer_query_read",
        buffer_name="xyz_sram_buffer",
        phase="read",
        resource_id="pointcim_query_xyz_src",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        resident_bytes=min(query_point_bytes, _acc_buffer_capacity_bytes("xyz_sram_buffer", hw_config)),
        preferred_chunk_bytes=query_point_bytes,
        extra_meta={
            "query_index": query_index,
            "phase": "distance_frontend_pointcim",
            "dependency_tag": query_src_ready_tag,
            "sim_interface": "cacti",
        },
    )
    cursor += xyz_query_read_dur
    eid, query_write_dur, _ = _acc_append_buffer_fill_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="query_buffer_write",
        circuit_module="query_xyz_sram_buffer",
        operation="query_xyz_buffer_write",
        buffer_name="query_xyz_sram_buffer",
        phase="query_refill",
        resource_id="pointcim_query_xyz_sram",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        is_read=False,
        preferred_chunk_bytes=query_point_bytes,
        extra_meta={
            "query_index": query_index,
            "phase": "distance_frontend_pointcim",
            "depends_on_tag": query_src_ready_tag,
            "dependency_tag": query_sram_ready_tag,
            "sim_interface": "cacti",
        },
    )
    cursor += query_write_dur
    eid, query_read_dur = _acc_append_buffer_consume_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="query_buffer_read",
        circuit_module="query_xyz_sram_buffer",
        operation="query_xyz_buffer_read",
        buffer_name="query_xyz_sram_buffer",
        phase="query_read",
        resource_id="pointcim_query_xyz_sram",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        resident_bytes=min(query_point_bytes, _acc_buffer_capacity_bytes("query_xyz_sram_buffer", hw_config)),
        preferred_chunk_bytes=query_point_bytes,
        extra_meta={
            "query_index": query_index,
            "phase": "distance_frontend_pointcim",
            "depends_on_tag": query_sram_ready_tag,
            "dependency_tag": query_read_ready_tag,
            "sim_interface": "cacti",
        },
    )
    query_ready_cycle = cursor + query_read_dur

    q_self_cycles = max(1, int(prims_total.get("self_term_cycles", getattr(rram_cfg, "self_term_cycles", 3)) or getattr(rram_cfg, "self_term_cycles", 3) or 3))
    q_self_power = float(getattr(rram_cfg, "self_term_power_mw", 0.0) or 0.0)
    if q_self_power <= 0.0 and dist_merge_cfg is not None:
        q_self_power = float(getattr(dist_merge_cfg, "power", 0.0) or 0.0)
    query_self_start = query_ready_cycle
    events.append(Event(
        eid, query_self_start, operator_name, "distance", "query_self_q", "rram_dist_digital", "query_self_q",
        q_self_power, q_self_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "phase": "distance_frontend_pointcim",
            "resource_id": "pointcim_query_self_shared",
            "exclusive": True,
            "depends_on_tag": query_read_ready_tag,
            "dependency_tag": query_self_ready_tag,
            "sim_interface": "logic_dist2_self_query",
        },
    ))
    eid += 1

    block_ready_cycles: List[int] = []
    block_prefetch_cycles: List[int] = []
    block_ready_tags: List[str] = []
    block_prefetch_tags: List[str] = []
    vmm_total_energy_nj = float(prims_total["vmm_energy_nj"])
    vmm_latency_ns = float(prims_total["vmm_latency_ns"])
    vmm_timing = _normalized_scheduler_timing(
        latency_ns=vmm_latency_ns,
        energy_nj=vmm_total_energy_nj,
        base_clock_ns=clk,
    )
    sum_cycles = max(1, int(prims_total.get("sum_cycles", getattr(rram_cfg, "sum_cycles", 1)) or getattr(rram_cfg, "sum_cycles", 1) or 1))
    sum_power = float(getattr(rram_cfg, "sum_power_mw", 0.0) or 0.0)
    if sum_power <= 0.0 and dist_merge_cfg is not None:
        sum_power = float(getattr(dist_merge_cfg, "power", 0.0) or 0.0)

    cross_start = max(query_ready_cycle, startup_ready_cycle)
    cross_tag = _pointcim_scope_tag(tag_scope, f"pointcim_cross_ready_q{query_index}")
    vmm_power = (
        vmm_total_energy_nj * 1000.0 / (float(vmm_timing["duration_cycles"]) * clk)
        if float(vmm_timing["duration_cycles"]) > 0 and clk > 0
        else 0.0
    )
    events.append(Event(
        eid, cross_start, operator_name, "distance", "cross_term", "rram_dist_array", "rram_cross_term_distance",
        vmm_power, int(vmm_timing["duration_cycles"]), clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "num_arrays": num_arrays,
            "array_point_counts": array_point_counts,
            "resident_array_points": resident_array_points,
            "phase": "distance_frontend_pointcim",
            "resource_id": "pointcim_vmm_array_all",
            "depends_on_tags": [query_read_ready_tag, rram_program_ready_tag],
            "dependency_tag": cross_tag,
            "latency_ns": vmm_latency_ns,
            "energy_nj": vmm_total_energy_nj,
            "native_clock_period_ns": float(neurosim_clk),
            "native_cycles": max(1, int(math.ceil(vmm_latency_ns / max(1e-9, neurosim_clk)))),
            "scheduler_normalized": True,
            "sim_interface": "neurosim_dist",
            "consolidated_arrays": True,
        },
    ))
    eid += 1
    cross_ready_tags: List[str] = [cross_tag]

    merge_cursor = cross_start + int(vmm_timing["duration_cycles"])
    for blk in range(merge_batches):
        blk_points = min(merge_parallelism, max(0, num_keys - blk * merge_parallelism))
        if blk_points <= 0:
            continue
        blk_dist_bytes = max(1, blk_points * 4)
        needed_cross_tags = cross_ready_tags

        self_tag = _pointcim_scope_tag(tag_scope, f"pointcim_self_read_ready_q{query_index}_b{blk}")
        merge_tag = _pointcim_scope_tag(tag_scope, f"pointcim_merge_ready_q{query_index}_b{blk}")
        dist_tag = _pointcim_scope_tag(tag_scope, f"pointcim_dist_write_ready_q{query_index}_b{blk}")
        eid, self_read_dur = _acc_append_buffer_consume_events(
            events, eid, merge_cursor, operator_name, "distance",
            sub_stage="self_distance_read",
            circuit_module="self_distance_sram_buffer",
            operation="self_distance_read",
            buffer_name="self_distance_sram_buffer",
            phase="read",
            resource_id="pointcim_self_distance_sram",
            total_bytes=blk_dist_bytes,
            hw_config=hw_config,
            resident_bytes=self_dist_total_bytes,
            preferred_chunk_bytes=blk_dist_bytes,
            extra_meta={
                "query_index": query_index,
                "consumer_batch_idx": blk,
                "num_batches": merge_batches,
                "merge_points": blk_points,
                "phase": "distance_frontend_pointcim",
                "depends_on_tag": self_cache_ready_tag,
                "dependency_tag": self_tag,
                "sim_interface": "cacti",
            },
        )
        self_ready_cycle = merge_cursor + self_read_dur

        merge_start = max(cross_start + int(vmm_timing["duration_cycles"]), self_ready_cycle, query_self_start + q_self_cycles, merge_cursor)
        events.append(Event(
            eid, merge_start, operator_name, "distance", "merge_accumulate", "dist2_merge_adder", "distance_merge_accumulate",
            sum_power, sum_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "consumer_batch_idx": blk,
                "num_batches": merge_batches,
                "merge_points": blk_points,
                "phase": "distance_frontend_pointcim",
                "resource_id": "pointcim_distance_merge",
                "exclusive": True,
                "depends_on_tags": needed_cross_tags + [self_tag, query_self_ready_tag],
                "dependency_tag": merge_tag,
                "sim_interface": "logic_dist2_merge_adder",
            },
        ))
        eid += 1
        block_ready_cycles.append(merge_start + sum_cycles)
        block_prefetch_cycles.append(merge_cursor)
        block_ready_tags.append(merge_tag)
        block_prefetch_tags.append(self_tag)
        merge_cursor = merge_start + sum_cycles

        if write_dist_buffer:
            eid, dist_write_dur, _ = _acc_append_buffer_fill_events(
                events, eid, merge_start + sum_cycles, operator_name, "distance",
                sub_stage="distance_write",
                circuit_module="dist_sram_buffer",
                operation="distance_write",
                buffer_name="dist_sram_buffer",
                phase="write",
                resource_id="dist_sram_shared",
                total_bytes=blk_dist_bytes,
                hw_config=hw_config,
                is_read=False,
                preferred_chunk_bytes=blk_dist_bytes,
                extra_meta={
                    "query_index": query_index,
                    "consumer_batch_idx": blk,
                    "num_batches": merge_batches,
                    "merge_points": blk_points,
                    "phase": "distance_frontend_pointcim",
                    "depends_on_tag": merge_tag,
                    "dependency_tag": dist_tag,
                    "sim_interface": "cacti",
                },
            )

    dist_cycles = max(0, (max(block_ready_cycles) if block_ready_cycles else query_ready_cycle) - base_cycle)
    return events, eid, dist_cycles, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags


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


def _acc_predictor_kat_estimate(
    hw_config: 'HardwareConfig',
    *,
    hidden_dim: int,
    output_dim: int,
    clock_period_ns: float,
) -> Dict[str, float]:
    """
    Predictor KAT is the effective nonlinearity between fc1 and fc2.
    Model it like a compact Efficient-KAN activation:
      SRAM read -> LUT eval -> optional post-LUT MAC -> SRAM write
    on a single hidden vector rather than the old 1-cycle placeholder.
    """
    lut_cfg = getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut_speculative", None)
    kat_mode = str(getattr(lut_cfg, "default_mode", "rram_pwl") if lut_cfg is not None else "rram_pwl").lower()
    lut_parallel = max(1, int(getattr(hw_config, "cim_parallelism", 128) or 128))
    features = max(1, int(hidden_dim))
    out_features = max(1, int(output_dim))
    if kat_mode in {"pwl", "rram_pwl"}:
        pwl_parallel = max(1, int(getattr(lut_cfg, "pwl_parallelism", 1) or 1)) if lut_cfg is not None else 1
        pwl_overhead = max(1, int(getattr(lut_cfg, "pwl_overhead_cycles", 1) or 1)) if lut_cfg is not None else 1
        total_cycles = max(1, safe_div_ceil(features, pwl_parallel) + pwl_overhead)
        pwl_power = float(getattr(lut_cfg, "pwl_power", 0.0759164) or 0.0759164) if lut_cfg is not None else 0.0759164
        latency_ns = total_cycles * float(clock_period_ns)
        energy_nj = pwl_power * latency_ns / 1e3
        return {
            "cycles": float(total_cycles),
            "power_mw": float(pwl_power),
            "latency_ns": float(latency_ns),
            "energy_nj": float(energy_nj),
            "lut_parallel": float(pwl_parallel),
            "kat_mode": "rram_pwl",
        }

    lut_factor = 2  # Efficient-KAN / elite style
    lut_cycles = max(1, safe_div_ceil(features * lut_factor, lut_parallel))
    buffer_cycles = max(1, safe_div_ceil(features + out_features, lut_parallel))
    post_mac_cycles_cfg = int(getattr(lut_cfg, "post_mac_cycles", 0) or 0) if lut_cfg is not None else 0
    post_mac_cycles = max(1, post_mac_cycles_cfg) if post_mac_cycles_cfg > 0 else max(1, safe_div_ceil(out_features, lut_parallel))
    total_cycles = max(1, buffer_cycles + lut_cycles + buffer_cycles + post_mac_cycles)
    lut_power = float(getattr(lut_cfg, "power", 5.0) or 5.0) if lut_cfg is not None else 5.0
    post_mac_power = float(getattr(lut_cfg, "post_mac_power", lut_power) or lut_power) if lut_cfg is not None else lut_power
    avg_power = max(0.1, ((2 * buffer_cycles + lut_cycles) * lut_power + post_mac_cycles * post_mac_power) / float(total_cycles))
    latency_ns = total_cycles * float(clock_period_ns)
    energy_nj = avg_power * latency_ns / 1e3
    return {
        "cycles": float(total_cycles),
        "power_mw": float(avg_power),
        "latency_ns": float(latency_ns),
        "energy_nj": float(energy_nj),
        "lut_parallel": float(lut_parallel),
        "kat_mode": "rram_lut",
    }


def _acc_predictor_events(
    start_event_id: int,
    base_cycle: int,
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]],
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> Tuple[List[Event], int, int]:
    """
    First-order EfficientKAN predictor approximation:
      Linear(fc1) -> KAT(rram_lut_speculative) -> Linear(fc2)

    Acc 路径下 Predictor 的 FC1/FC2 一律用单层 NeuroSim（或内置回退），时钟优先全网络注入上下文中的 cim_clock_period_ns。
    """
    events: List[Event] = []
    eid = start_event_id
    logic_clk = float(getattr(hw_config, "clock_period", 1.0))
    predictor_clk = float((context or {}).get(CIM_CLOCK_PERIOD_NS) or logic_clk)
    predictor_clk_freq_hz = None
    explicit_ns_freq = getattr(hw_config, "neurosim_clk_freq_hz", None)
    if explicit_ns_freq is not None and float(explicit_ns_freq) > 0:
        predictor_clk_freq_hz = float(explicit_ns_freq)
    elif predictor_clk > 0:
        predictor_clk_freq_hz = 1e9 / predictor_clk
    neurosim_mode = "pybind" if bool(getattr(hw_config, "neurosim_pybind", False)) else "subprocess"
    fc1_est = estimate_linear_neurosim(
        in_features=max(1, int(input_dim)),
        out_features=max(1, int(hidden_dim)),
        tokens=1,
        clock_period_ns=predictor_clk,
        neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
        fallback_power_mw=20.0,
        clk_freq_hz=predictor_clk_freq_hz,
        technode=getattr(hw_config, "neurosim_technode", 32),
        adc_clk_freq_hz=getattr(hw_config, "neurosim_adc_clk_freq_hz", None),
        sar_column_power_w=getattr(hw_config, "neurosim_sar_column_power_w", None),
        num_col_muxed=int(getattr(hw_config, "neurosim_num_col_muxed", 0) or 0) or 8,
        clk_mode=str(getattr(hw_config, "neurosim_clk_mode", "capped") or "capped"),
        mode=neurosim_mode,
        allow_subarray_retry=not _use_fixed_neurosim_subarray(hw_config),
    )
    fc2_est = estimate_linear_neurosim(
        in_features=max(1, int(hidden_dim)),
        out_features=max(1, int(output_dim)),
        tokens=1,
        clock_period_ns=predictor_clk,
        neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
        fallback_power_mw=20.0,
        clk_freq_hz=predictor_clk_freq_hz,
        technode=getattr(hw_config, "neurosim_technode", 32),
        adc_clk_freq_hz=getattr(hw_config, "neurosim_adc_clk_freq_hz", None),
        sar_column_power_w=getattr(hw_config, "neurosim_sar_column_power_w", None),
        num_col_muxed=int(getattr(hw_config, "neurosim_num_col_muxed", 0) or 0) or 8,
        clk_mode=str(getattr(hw_config, "neurosim_clk_mode", "capped") or "capped"),
        mode=neurosim_mode,
        allow_subarray_retry=not _use_fixed_neurosim_subarray(hw_config),
    )

    kat_est = _acc_predictor_kat_estimate(
        hw_config,
        hidden_dim=max(1, int(hidden_dim)),
        output_dim=max(1, int(output_dim)),
        clock_period_ns=logic_clk,
    )
    kat_cycles = max(1, int(kat_est["cycles"]))
    kat_power = float(kat_est["power_mw"])

    cursor_ns = 0.0

    def _cursor_cycle(offset_ns: float) -> int:
        return int(base_cycle + math.ceil(max(0.0, offset_ns) / max(1e-9, logic_clk)))

    fc1_meta = {
        "phase": "predictor_gate",
        "predictor_stage": "fc1",
        "source": getattr(fc1_est, "source", "unknown"),
        "latency_ns": float(getattr(fc1_est, "latency_ns", fc1_est.cycles * predictor_clk)),
        "energy_nj": float(getattr(fc1_est, "energy_nj", 0.0)),
        "bytes": max(1, int(input_dim)) * 4,
        "resource_id": "predictor_shared",
        "exclusive": True,
    }
    fc1_timing = _normalized_scheduler_timing(
        latency_ns=float(getattr(fc1_est, "latency_ns", predictor_clk)),
        energy_nj=float(getattr(fc1_est, "energy_nj", 0.0) or 0.0),
        base_clock_ns=logic_clk,
    )
    fc1_meta["native_clock_period_ns"] = float(predictor_clk)
    fc1_meta["native_cycles"] = int(getattr(fc1_est, "cycles", 1) or 1)
    fc1_meta["scheduler_normalized"] = True
    events.append(Event(
        eid, _cursor_cycle(cursor_ns), "SpecPredictor", "speculative", "predictor_fc1", "rram_cim", "predictor_fc1_linear",
        float(fc1_timing["power_mw"]), int(fc1_timing["duration_cycles"]), clock_period_ns=float(fc1_timing["clock_period_ns"]), metadata=fc1_meta,
    ))
    eid += 1
    cursor_ns += float(getattr(fc1_est, "latency_ns", fc1_est.cycles * predictor_clk))

    kat_meta = {
        "phase": "predictor_gate",
        "predictor_stage": "kat",
        "bytes": max(1, int(hidden_dim)) * 4,
        "kat_mode": str(kat_est.get("kat_mode", "rram_lut")),
        "source": "predictor_kat_model",
        "latency_ns": float(kat_est["latency_ns"]),
        "energy_nj": float(kat_est["energy_nj"]),
        "lut_parallel": int(kat_est["lut_parallel"]),
        "resource_id": "predictor_shared",
        "exclusive": True,
    }
    events.append(Event(
        eid, _cursor_cycle(cursor_ns), "SpecPredictor", "speculative", "predictor_kat", "rram_lut_speculative", "predictor_kat",
        kat_power, kat_cycles, clock_period_ns=logic_clk, metadata=kat_meta,
    ))
    eid += 1
    cursor_ns += float(kat_est["latency_ns"])

    fc2_meta = {
        "phase": "predictor_gate",
        "predictor_stage": "fc2",
        "source": getattr(fc2_est, "source", "unknown"),
        "latency_ns": float(getattr(fc2_est, "latency_ns", fc2_est.cycles * predictor_clk)),
        "energy_nj": float(getattr(fc2_est, "energy_nj", 0.0)),
        "bytes": max(1, int(output_dim)) * 4,
        "resource_id": "predictor_shared",
        "exclusive": True,
    }
    fc2_timing = _normalized_scheduler_timing(
        latency_ns=float(getattr(fc2_est, "latency_ns", predictor_clk)),
        energy_nj=float(getattr(fc2_est, "energy_nj", 0.0) or 0.0),
        base_clock_ns=logic_clk,
    )
    fc2_meta["native_clock_period_ns"] = float(predictor_clk)
    fc2_meta["native_cycles"] = int(getattr(fc2_est, "cycles", 1) or 1)
    fc2_meta["scheduler_normalized"] = True
    events.append(Event(
        eid, _cursor_cycle(cursor_ns), "SpecPredictor", "speculative", "predictor_fc2", "rram_cim", "predictor_fc2_linear",
        float(fc2_timing["power_mw"]), int(fc2_timing["duration_cycles"]), clock_period_ns=float(fc2_timing["clock_period_ns"]), metadata=fc2_meta,
    ))
    eid += 1
    cursor_ns += float(getattr(fc2_est, "latency_ns", fc2_est.cycles * predictor_clk))
    return events, eid, int(math.ceil(cursor_ns / max(1e-9, logic_clk)))


def _acc_query_point_bytes() -> int:
    return _acc_query_point_onchip_bytes()


def _acc_query_refill_bytes(hw_config: 'HardwareConfig') -> int:
    point_bytes = _acc_query_point_offchip_bytes()
    dram = getattr(hw_config, "dram", None)
    dram_bw = max(1, int(getattr(dram, "bandwidth_bytes_per_cycle", 32) or 32)) if dram is not None else 32
    return max(point_bytes, dram_bw)


def _acc_xyz_sram_load_bytes(params: Dict[str, Any]) -> int:
    num_keys = max(1, int(params.get("num_keys", params.get("num_points", 1)) or 1))
    return num_keys * _acc_query_point_offchip_bytes()


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


def _acc_segment_len(sampled_points_num: int, params: Dict[str, Any], hw_config: Optional['HardwareConfig'] = None) -> int:
    segment_count = int(params.get("segment_count", 0) or 0)
    if segment_count <= 0 and hw_config is not None:
        segment_count = int(getattr(hw_config, "acc_segment_count", 0) or 0)
    if segment_count > 0:
        return max(1, int(math.ceil(sampled_points_num / float(segment_count))))
    return max(1, int(params.get("segment_len", 64) or 64))


def _build_linear_interp_schedule(in_len: int, out_len: int) -> List[Dict[str, Any]]:
    in_len = max(1, int(in_len))
    out_len = max(1, int(out_len))
    if out_len == 1:
        src_positions = [0.0]
    else:
        step = float(max(0, in_len - 1)) / float(max(1, out_len - 1))
        src_positions = [j * step for j in range(out_len)]
    table: List[Dict[str, Any]] = []
    for out_idx, src_pos in enumerate(src_positions):
        left = max(0, min(in_len - 1, int(math.floor(src_pos))))
        right = max(0, min(in_len - 1, left + 1))
        w = float(src_pos - float(left))
        table.append({
            "out_idx": out_idx,
            "src_pos": src_pos,
            "left": left,
            "right": right,
            "w": w,
        })
    return table


def _gen_conv1d(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Conv1d：base_v1/base_v2 仅通过 SCALE-Sim 脉动阵列仿真（FXP16）；acc 用全网络 NeuroSim 注入或单层 RRAM-CIM（可含量化/反量化）；其他配置 raise。
    """
    is_baseline = getattr(hw_config, "is_baseline", False)
    has_systolic = getattr(hw_config, "systolic_array", None) is not None
    if is_baseline:
        if not has_systolic:
            raise RuntimeError("Conv1d not supported: baseline (base_v1/base_v2) requires systolic_array and SCALE-Sim.")
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_conv1d_systolic_events(params, hw_config)
    # acc：全网络 NeuroSim 注入或 PE 级 RRAM-CIM
    if context and CIM_LAYER_LATENCY_NS in context and CIM_LAYER_ENERGY_NJ in context:
        lat_ns_arr = context[CIM_LAYER_LATENCY_NS]
        en_nj_arr = context[CIM_LAYER_ENERGY_NJ]
        # Resolve latency/energy: name-based lookup first (immune to cim_layer_index misalignment
        # caused by SA Conv layers dispatched via _pipeline_conv_after_knn which bypass the index).
        lat_ns_val = en_nj_val = None
        area_idx = None
        resolved_name = ""
        lat_by_name = context.get(CIM_LAT_ENERGY_BY_NAME)
        name_candidates = _resolve_cim_name_candidates(op_ref, context)
        matched_name = next((n for n in name_candidates if lat_by_name and n in lat_by_name), "")
        if lat_by_name and matched_name:
            lat_ns_val, en_nj_val = lat_by_name[matched_name]
            resolved_name = matched_name
            _layer_names = context.get(CIM_LAYER_NAMES)
            if isinstance(_layer_names, (list, tuple)):
                try:
                    area_idx = list(_layer_names).index(matched_name)
                except ValueError:
                    pass
        else:
            idx = context.get(CIM_LAYER_INDEX, 0)
            if idx < len(lat_ns_arr) and idx < len(en_nj_arr):
                lat_ns_val, en_nj_val = float(lat_ns_arr[idx]), float(en_nj_arr[idx])
                area_idx = idx
                _layer_names = context.get(CIM_LAYER_NAMES)
                resolved_name = str(_layer_names[idx]) if isinstance(_layer_names, (list, tuple)) and idx < len(_layer_names) else ""
                context[CIM_LAYER_INDEX] = idx + 1
        if lat_ns_val is not None:
            full_backend = str(context.get(CIM_FULL_BACKEND, "neurosim") or "neurosim").lower()
            native_clock_ns = float(context.get(CIM_CLOCK_PERIOD_NS) or getattr(hw_config, "clock_period", 1.0))
            native_cycles = max(1, int(math.ceil(float(lat_ns_val) / max(1e-9, native_clock_ns))))
            power_mw = float(en_nj_val) * 1e3 / (native_cycles * native_clock_ns) if (native_cycles * native_clock_ns) > 0 else 0.0
            meta = {"source": f"{full_backend}_full"}
            area_list = context.get(CIM_LAYER_AREA_UM2)
            if isinstance(area_list, (list, tuple)) and area_idx is not None and area_idx < len(area_list):
                meta["area_um2"] = float(area_list[area_idx])
            if resolved_name:
                meta["layer_name"] = resolved_name
            meta["latency_ns"] = float(lat_ns_val)
            meta["energy_nj"] = float(en_nj_val)
            meta["native_clock_period_ns"] = native_clock_ns
            meta["native_cycles"] = native_cycles
            circuit_module = "neurosim_tile_noc"
            operation = "conv1d_neurosim_full"
            cim_full_ev = Event(
                event_id=0,
                cycle=0,
                operator=OP_NAME_CONV1D,
                circuit_module=circuit_module,
                operation=operation,
                power=float(power_mw),
                duration=int(native_cycles),
                clock_period_ns=float(native_clock_ns),
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
                cim_full_ev.cycle = q_ev.cycle + q_ev.duration
                d_ev.cycle = cim_full_ev.cycle + cim_full_ev.duration
                return [q_ev, cim_full_ev, d_ev]
            return [cim_full_ev]
    if not is_baseline:
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_conv1d_rram_cim_events(params, hw_config)
    raise RuntimeError("Conv1d not supported: unknown accelerator mode (neither baseline with SCALE-Sim nor acc).")


def _gen_linear(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Linear：base_v1/base_v2 仅通过 SCALE-Sim 脉动阵列仿真（FXP16）；acc 用全网络 NeuroSim 注入或单层 RRAM-CIM（可含量化/反量化）；其他配置 raise。
    """
    is_baseline = getattr(hw_config, "is_baseline", False)
    has_systolic = getattr(hw_config, "systolic_array", None) is not None
    if is_baseline:
        if not has_systolic:
            raise RuntimeError("Linear not supported: baseline (base_v1/base_v2) requires systolic_array and SCALE-Sim.")
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_linear_systolic_events(params, hw_config)
    if context and CIM_LAYER_LATENCY_NS in context and CIM_LAYER_ENERGY_NJ in context:
        lat_ns_arr = context[CIM_LAYER_LATENCY_NS]
        en_nj_arr = context[CIM_LAYER_ENERGY_NJ]
        # Resolve latency/energy: name-based lookup first (immune to cim_layer_index misalignment).
        lat_ns_val = en_nj_val = None
        area_idx = None
        resolved_name = ""
        lat_by_name = context.get(CIM_LAT_ENERGY_BY_NAME)
        name_candidates = _resolve_cim_name_candidates(op_ref, context)
        matched_name = next((n for n in name_candidates if lat_by_name and n in lat_by_name), "")
        if lat_by_name and matched_name:
            lat_ns_val, en_nj_val = lat_by_name[matched_name]
            resolved_name = matched_name
            _layer_names = context.get(CIM_LAYER_NAMES)
            if isinstance(_layer_names, (list, tuple)):
                try:
                    area_idx = list(_layer_names).index(matched_name)
                except ValueError:
                    pass
        else:
            idx = context.get(CIM_LAYER_INDEX, 0)
            if idx < len(lat_ns_arr) and idx < len(en_nj_arr):
                lat_ns_val, en_nj_val = float(lat_ns_arr[idx]), float(en_nj_arr[idx])
                area_idx = idx
                _layer_names = context.get(CIM_LAYER_NAMES)
                resolved_name = str(_layer_names[idx]) if isinstance(_layer_names, (list, tuple)) and idx < len(_layer_names) else ""
                context[CIM_LAYER_INDEX] = idx + 1
        if lat_ns_val is not None:
            full_backend = str(context.get(CIM_FULL_BACKEND, "neurosim") or "neurosim").lower()
            native_clock_ns = float(context.get(CIM_CLOCK_PERIOD_NS) or getattr(hw_config, "clock_period", 1.0))
            native_cycles = max(1, int(math.ceil(float(lat_ns_val) / max(1e-9, native_clock_ns))))
            power_mw = float(en_nj_val) * 1e3 / (native_cycles * native_clock_ns) if (native_cycles * native_clock_ns) > 0 else 0.0
            meta = {"source": f"{full_backend}_full"}
            area_list = context.get(CIM_LAYER_AREA_UM2)
            if isinstance(area_list, (list, tuple)) and area_idx is not None and area_idx < len(area_list):
                meta["area_um2"] = float(area_list[area_idx])
            if resolved_name:
                meta["layer_name"] = resolved_name
            meta["latency_ns"] = float(lat_ns_val)
            meta["energy_nj"] = float(en_nj_val)
            meta["native_clock_period_ns"] = native_clock_ns
            meta["native_cycles"] = native_cycles
            circuit_module = "neurosim_tile_noc"
            operation = "linear_neurosim_full"
            cim_full_ev = Event(
                event_id=0,
                cycle=0,
                operator=OP_NAME_LINEAR,
                circuit_module=circuit_module,
                operation=operation,
                power=float(power_mw),
                duration=int(native_cycles),
                clock_period_ns=float(native_clock_ns),
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
                cim_full_ev.cycle = q_ev.cycle + q_ev.duration
                d_ev.cycle = cim_full_ev.cycle + cim_full_ev.duration
                return [q_ev, cim_full_ev, d_ev]
            return [cim_full_ev]
    if not is_baseline:
        params = _merge_real_conv_linear_params(op_ref, context)
        return gen_linear_rram_cim_events(params, hw_config)
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


def _gen_conv2d(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    events = _gen_conv1d(op_ref, hw_config, context)
    for ev in events:
        if getattr(ev, "operator", "") == "Conv1d":
            ev.operator = "Conv2d"
        op_name = str(getattr(ev, "operation", "") or "")
        if "conv1d" in op_name:
            ev.operation = op_name.replace("conv1d", "conv2d")
        meta = dict(getattr(ev, "metadata", None) or {})
        meta.setdefault("native_op", "pointnet2_sa_conv2d_k1")
        ev.metadata = meta
    return events


def _gen_acc_distance_query_events(
    start_event_id: int,
    base_cycle: int,
    operator_name: str,
    num_keys: int,
    hw_config: 'HardwareConfig',
    query_index: int = 0,
    include_query_dram_read: bool = True,
    memory_context: Optional[Any] = None,
    startup_once: bool = False,
    write_dist_buffer: bool = True,
    tag_scope: str = "",
    key_already_resident: bool = False,
) -> Tuple[List[Event], int, int, List[int], List[int], List[str], List[str]]:
    """
    ACC 距离前端（banked）：
      - key xyz 以 bank 为单位 preload 到 SRAM-CIM array
      - 每个 bank 先做 self-square，结果写到 bank-local self-term SRAM
      - query 到来后，对每个 bank 做 128-point block 的 cross-term + merge

    Bank 容量按单 bank 的 SRAM-CIM array.rows 决定；
    对大点云，num_banks = ceil(num_keys / bank_capacity_points)。

    当前实现允许 bank[i+1] 的 preload 与 bank[i] 的 self-square/后续 compute 流水，
    但 bank[i] 的 query-side cross-term 只能在该 bank preload+self-square 完成后触发。
    返回值中的 block_ready_cycles/block_prefetch_cycles 仍保持“每个 block 一条 ready/prefetch cycle”，
    同时返回每个 block 的 merge/self-read dependency tag，供 FPS/KNN 下游建立显式 producer-consumer 关系。
    """
    if getattr(hw_config, "use_rram_dist", False) and getattr(hw_config, "rram_dist", None) is not None:
        return _gen_pointcim_distance_query_events(
            start_event_id=start_event_id,
            base_cycle=base_cycle,
            operator_name=operator_name,
            num_keys=num_keys,
            hw_config=hw_config,
            query_index=query_index,
            startup_once=startup_once,
            write_dist_buffer=write_dist_buffer,
            tag_scope=tag_scope,
        )

    events: List[Event] = []
    eid = start_event_id
    clk = float(getattr(hw_config, "clock_period", 1.0))
    num_keys = max(1, int(num_keys))
    macro = getattr(hw_config, "sram_cim_macro", None)
    array = getattr(macro, "array", None) if macro is not None else None
    shift = getattr(macro, "shift_add", None) if macro is not None else None
    dist_merge = getattr(macro, "dist2_merge_adder", None) if macro is not None else None
    parallelism = max(1, int(getattr(array, "cim_parallelism", 128) or 128))
    quant_bits = max(1, int(getattr(array, "quantization_bits", 12) or 12))
    bit_slice = max(1, int(getattr(array, "bit_slice_bits", 2) or 2))
    bit_serial_cycles = max(1, int(math.ceil(quant_bits / float(bit_slice))))
    key_preload_power = float(getattr(array, "write_power_per_row_mw", 2.0) or 2.0)
    key_row_write_cycles = max(1, int(getattr(array, "write_cycles_per_row", 1) or 1))
    cim_power = float(getattr(array, "cim_power_per_batch_mw", 6.0) or 6.0)
    shift_power = float(getattr(shift, "power", 1.3) or 1.3)
    merge_power = float(getattr(dist_merge, "power", 1.3) or 1.3)
    merge_block_cycles = max(1, int(getattr(dist_merge, "cycles", 1) or 1))

    point_offchip_bytes = _acc_query_point_offchip_bytes()
    point_onchip_bytes = _acc_query_point_onchip_bytes()
    query_point_bytes = point_onchip_bytes
    query_stream_chunk_bytes = query_point_bytes
    xyz_stream_chunk_bytes = parallelism * point_onchip_bytes
    dist_stream_chunk_bytes = parallelism * 4
    frontend_pipeline = _acc_frontend_pipeline_enabled(hw_config)

    # One bank should match the physical SRAM-CIM bank capacity. The xyz_sram_buffer is
    # only a DRAM->array staging buffer for the current bank preload and should not decide
    # how many keys a bank can logically hold.
    array_rows = max(parallelism, int(getattr(array, "rows", parallelism) or parallelism))
    bank_capacity_points = max(parallelism, array_rows)
    num_banks = max(1, int(math.ceil(num_keys / float(bank_capacity_points))))

    bank_specs: List[Dict[str, int]] = []
    for bank_idx in range(num_banks):
        bank_keys = min(bank_capacity_points, max(0, num_keys - bank_idx * bank_capacity_points))
        if bank_keys <= 0:
            continue
        bank_batches = max(1, int(math.ceil(bank_keys / float(parallelism))))
        bank_xyz_dram_bytes = bank_keys * point_offchip_bytes
        bank_xyz_sram_bytes = bank_keys * point_onchip_bytes
        bank_dist_bytes = bank_keys * 4
        # self-term is bank-local. With the current design each bank owns its own self-distance
        # SRAM slice, so the effective resident bytes scale with the bank size rather than being
        # globally clipped by a single shared 4KB buffer instance.
        bank_self_resident_bytes = bank_dist_bytes
        bank_self_block_bytes = max(1, int(math.ceil(bank_dist_bytes / float(bank_batches))))
        bank_specs.append({
            "bank_idx": bank_idx,
            "bank_keys": bank_keys,
            "bank_batches": bank_batches,
            "xyz_dram_bytes": bank_xyz_dram_bytes,
            "xyz_sram_bytes": bank_xyz_sram_bytes,
            "dist_bytes": bank_dist_bytes,
            "self_resident_bytes": bank_self_resident_bytes,
            "self_block_bytes": bank_self_block_bytes,
        })

    # Preload/self-square are startup work that should only happen once when keys are first materialized.
    # key_already_resident=True means a previous GAM/SA stage already wrote the full point cloud into
    # the SRAM-CIM array; subsequent stages use a subset (FPS-selected points) and only need to mask
    # inactive columns -- no DRAM read, no SRAM write, no row-write, no self-square recomputation.
    do_startup = ((query_index == 0) or (not startup_once)) and not key_already_resident
    bank_ready_cycles: Dict[int, int] = {spec["bank_idx"]: base_cycle for spec in bank_specs}
    preload_cursor = base_cycle
    if do_startup:
        q12 = getattr(getattr(hw_config, "logic_circuit", None), "quant_pipeline_12bit", None)
        q12_cycles = max(1, int(getattr(q12, "cycles", 1) or 1)) if q12 is not None else 1
        q12_power = float(getattr(q12, "power", 0.094) or 0.094) if q12 is not None else 0.094
        bank_start_cycles, bank_dram_results, _bank_setup_overheads, _bank_dram_cycles = _acc_precompute_banked_startup_dram_results(
            bank_specs=bank_specs,
            preload_cursor=preload_cursor,
            operator_name=operator_name,
            query_index=query_index,
            num_banks=num_banks,
            hw_config=hw_config,
            clk=clk,
            q12_cycles=q12_cycles,
            memory_context=memory_context,
        )
        for spec in bank_specs:
            bank_idx = spec["bank_idx"]
            bank_keys = spec["bank_keys"]
            bank_batches = spec["bank_batches"]
            bank_xyz_dram_bytes = spec["xyz_dram_bytes"]
            bank_xyz_sram_bytes = spec["xyz_sram_bytes"]
            bank_dist_bytes = spec["dist_bytes"]
            bank_self_resident_bytes = spec["self_resident_bytes"]

            bank_setup_cursor = bank_start_cycles[bank_idx]
            res = bank_dram_results[bank_idx]
            xyz_dram_cycles = int(res.get("cycles", 0) or 0)
            xyz_dram_energy_nj = float(res.get("energy_nj", 0.0) or 0.0)
            xyz_dram_power = float(xyz_dram_energy_nj) * 1000.0 / max(1.0, xyz_dram_cycles * clk) if xyz_dram_cycles > 0 else 0.0
            events.append(Event(
                eid, bank_setup_cursor, "DRAM", "distance", "xyz_dram_read", "dram", "xyz_refill",
                xyz_dram_power, max(1, xyz_dram_cycles), clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "buffer_name": "xyz_sram_buffer",
                    "buffer_phase": "refill_read",
                    "bytes": bank_xyz_dram_bytes,
                    "resource_id": "dram_port",
                    "dram_pipelined": True,
                    "dram_issue_cycles": query_dram_issue_cycles(bank_xyz_dram_bytes, hw_config),
                    "phase": "distance_frontend_banked",
                },
            ))
            eid += 1
            quant_start = bank_setup_cursor + max(1, xyz_dram_cycles)
            events.append(Event(
                eid, quant_start, "Quant", "distance", "xyz_quantize", "quant_pipeline_12bit", "xyz_quantize",
                q12_power, q12_cycles, clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "buffer_name": "xyz_sram_buffer",
                    "buffer_phase": "quantize",
                    "bytes": bank_xyz_dram_bytes,
                    "resource_id": "quant_pipeline_12bit",
                    "exclusive": True,
                    "phase": "distance_frontend_banked",
                },
            ))
            eid += 1
            bank_setup_cursor = quant_start + q12_cycles

            eid, xyz_sram_write_dur, _ = _acc_append_buffer_fill_events(
                events, eid, bank_setup_cursor, operator_name, "distance",
                sub_stage="xyz_buffer_write",
                circuit_module="xyz_sram_buffer",
                operation="xyz_buffer_write",
                buffer_name="xyz_sram_buffer",
                phase="write",
                resource_id=f"xyz_sram_bank{bank_idx}",
                total_bytes=bank_xyz_sram_bytes,
                hw_config=hw_config,
                is_read=False,
                preferred_chunk_bytes=bank_xyz_sram_bytes,
                extra_meta={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "phase": "distance_frontend_banked",
                    "bytes_src_offchip": bank_xyz_dram_bytes,
                    "bytes_dst_onchip": bank_xyz_sram_bytes,
                },
                memory_context=memory_context,
            )
            bank_setup_cursor += xyz_sram_write_dur

            eid, xyz_key_read_dur = _acc_append_buffer_consume_events(
                events, eid, bank_setup_cursor, operator_name, "distance",
                sub_stage="xyz_buffer_key_read",
                circuit_module="xyz_sram_buffer",
                operation="xyz_buffer_key_read",
                buffer_name="xyz_sram_buffer",
                phase="read",
                resource_id=f"xyz_sram_bank{bank_idx}",
                total_bytes=bank_xyz_sram_bytes,
                hw_config=hw_config,
                resident_bytes=bank_xyz_sram_bytes,
                preferred_chunk_bytes=xyz_stream_chunk_bytes,
                extra_meta={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "phase": "distance_frontend_banked",
                },
                memory_context=memory_context,
            )
            bank_setup_cursor += xyz_key_read_dur

            key_preload_start = bank_setup_cursor
            key_preload_cycles = bank_keys * key_row_write_cycles
            key_preload_ev = Event(
                eid, key_preload_start, operator_name, "distance", "key_preload", "sram_cim_array", "cim_keyset_row_write",
                key_preload_power, key_preload_cycles, clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "buffer_name": "sram_cim_array",
                    "buffer_phase": "key_preload",
                    "bytes": bank_xyz_sram_bytes,
                    "resource_id": f"key_xyz_bank_{bank_idx}",
                    "exclusive": True,
                    "phase": "distance_frontend_banked",
                },
            )
            if frontend_pipeline and bank_idx > 0:
                key_preload_ev.overlapped = True
            events.append(key_preload_ev)
            eid += 1

            # Once the bank's key staging buffer is drained into the array, the next bank can begin
            # its own DRAM->SRAM staging even while this bank performs self-square and later query-side work.
            preload_cursor = bank_setup_cursor

            bank_cursor = key_preload_start + (0 if getattr(key_preload_ev, "overlapped", False) else key_preload_cycles)
            bank_self_square_cycles = bank_batches * bit_serial_cycles
            events.append(Event(
                eid, bank_cursor, operator_name, "distance", "self_square", "sram_cim_array", "self_square_distance",
                cim_power + shift_power, bank_self_square_cycles, clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "phase": "distance_frontend_banked",
                    "buffer_name": "self_distance_sram_buffer",
                    "buffer_phase": "self_square_compute",
                    "bytes": bank_dist_bytes,
                    "resource_id": f"distance_compute_bank{bank_idx}",
                    "exclusive": True,
                },
            ))
            eid += 1
            bank_cursor += bank_self_square_cycles

            eid, self_write_dur, _ = _acc_append_buffer_fill_events(
                events, eid, bank_cursor, operator_name, "distance",
                sub_stage="self_distance_write",
                circuit_module="self_distance_sram_buffer",
                operation="self_distance_write",
                buffer_name="self_distance_sram_buffer",
                phase="write",
                resource_id=f"self_distance_bank{bank_idx}",
                total_bytes=bank_dist_bytes,
                hw_config=hw_config,
                is_read=False,
                preferred_chunk_bytes=dist_stream_chunk_bytes,
                extra_meta={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "bank_keys": bank_keys,
                    "phase": "distance_frontend_banked",
                },
                memory_context=memory_context,
            )
            bank_ready_cycles[bank_idx] = bank_cursor + self_write_dur

    # Query path remains once-per-query. The query point is tiny and can overlap with bank startup.
    cursor = base_cycle
    eid, xyz_query_read_dur = _acc_append_buffer_consume_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="xyz_buffer_query_read",
        circuit_module="xyz_sram_buffer",
        operation="xyz_buffer_query_read",
        buffer_name="xyz_sram_buffer",
        phase="read",
        resource_id="xyz_sram_query_feed",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        resident_bytes=min(query_point_bytes, _acc_buffer_capacity_bytes("xyz_sram_buffer", hw_config)),
        preferred_chunk_bytes=query_stream_chunk_bytes,
        extra_meta={"query_index": query_index, "phase": "distance_frontend_banked"},
        memory_context=memory_context,
    )
    cursor += xyz_query_read_dur
    eid, sram_write_dur, _ = _acc_append_buffer_fill_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="query_buffer_write",
        circuit_module="query_xyz_sram_buffer",
        operation="query_xyz_buffer_write",
        buffer_name="query_xyz_sram_buffer",
        phase="query_refill",
        resource_id="query_xyz_sram_shared",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        is_read=False,
        preferred_chunk_bytes=query_stream_chunk_bytes,
        extra_meta={"query_index": query_index, "phase": "distance_frontend_banked"},
        memory_context=memory_context,
    )
    cursor += sram_write_dur
    eid, sram_read_dur = _acc_append_buffer_consume_events(
        events, eid, cursor, operator_name, "distance",
        sub_stage="query_buffer_read",
        circuit_module="query_xyz_sram_buffer",
        operation="query_xyz_buffer_read",
        buffer_name="query_xyz_sram_buffer",
        phase="query_read",
        resource_id="query_xyz_sram_shared",
        total_bytes=query_point_bytes,
        hw_config=hw_config,
        resident_bytes=min(query_point_bytes, _acc_buffer_capacity_bytes("query_xyz_sram_buffer", hw_config)),
        preferred_chunk_bytes=query_stream_chunk_bytes,
        extra_meta={"query_index": query_index, "phase": "distance_frontend_banked"},
        memory_context=memory_context,
    )
    query_ready_cycle = cursor + sram_read_dur

    cursor = query_ready_cycle
    block_ready_cycles: List[int] = []
    block_prefetch_cycles: List[int] = []
    block_ready_tags: List[str] = []
    block_prefetch_tags: List[str] = []
    for spec in bank_specs:
        bank_idx = spec["bank_idx"]
        bank_keys = spec["bank_keys"]
        bank_batches = spec["bank_batches"]
        bank_dist_bytes = spec["dist_bytes"]
        bank_self_resident_bytes = spec["self_resident_bytes"]
        bank_self_block_bytes = spec["self_block_bytes"]
        bank_start = max(cursor, bank_ready_cycles.get(bank_idx, base_cycle))
        block_base_cycle = bank_start
        for blk in range(bank_batches):
            blk_points = min(parallelism, max(0, bank_keys - blk * parallelism))
            cross_start = block_base_cycle + blk * bit_serial_cycles
            cross_end = cross_start + bit_serial_cycles
            events.append(Event(
                eid, cross_start, operator_name, "distance", "cross_term", "sram_cim_array", "cross_term_distance",
                cim_power + shift_power, bit_serial_cycles, clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "block_idx": blk,
                    "num_blocks": bank_batches,
                    "block_points": blk_points,
                    "phase": "distance_frontend_banked",
                    "buffer_name": "query_xyz_sram_buffer",
                    "buffer_phase": "query_read",
                    "bytes": query_point_bytes,
                    "resource_id": f"distance_compute_bank{bank_idx}",
                    "exclusive": True,
                },
            ))
            eid += 1
            self_read_start = cross_start
            self_ready = self_read_start
            self_tag = f"{str(operator_name).lower()}_self_q{query_index}_bank{bank_idx}_b{blk}"
            merge_tag = f"{str(operator_name).lower()}_merge_q{query_index}_bank{bank_idx}_b{blk}"
            eid, self_read_dur = _acc_append_buffer_consume_events(
                events, eid, self_ready, operator_name, "distance",
                sub_stage="self_distance_read",
                circuit_module="self_distance_sram_buffer",
                operation="self_distance_read",
                buffer_name="self_distance_sram_buffer",
                phase="read",
                resource_id=f"self_distance_bank{bank_idx}_b{blk}",
                total_bytes=max(1, int(math.ceil(bank_dist_bytes / float(bank_batches)))),
                hw_config=hw_config,
                resident_bytes=bank_self_resident_bytes,
                preferred_chunk_bytes=bank_self_block_bytes,
                extra_meta={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "block_idx": blk,
                    "num_blocks": bank_batches,
                    "phase": "distance_frontend_banked",
                    "dependency_tag": self_tag,
                },
                memory_context=memory_context,
            )
            self_ready += self_read_dur
            merge_start = max(cross_end, self_ready)
            merge_ev = Event(
                eid, merge_start, operator_name, "distance", "merge_accumulate", "dist2_merge_adder", "distance_merge_accumulate",
                merge_power, merge_block_cycles, clock_period_ns=clk,
                metadata={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "block_idx": blk,
                    "num_blocks": bank_batches,
                    "buffer_name": "self_distance_sram_buffer",
                    "buffer_phase": "read",
                    "bytes": bank_self_block_bytes,
                    "resource_id": f"distance_merge_bank{bank_idx}",
                    "exclusive": True,
                    "phase": "distance_frontend_banked",
                    "depends_on_tag": self_tag,
                    "dependency_tag": merge_tag,
                },
            )
            if blk < bank_batches - 1:
                merge_ev.overlapped = True
            events.append(merge_ev)
            eid += 1
            block_ready_cycles.append(merge_start + merge_block_cycles)
            block_prefetch_cycles.append(self_read_start)
            block_ready_tags.append(merge_tag)
            block_prefetch_tags.append(self_tag)
        bank_end = block_base_cycle + bank_batches * bit_serial_cycles + merge_block_cycles
        if write_dist_buffer:
            eid, dist_write_dur, _ = _acc_append_buffer_fill_events(
                events, eid, bank_end, operator_name, "distance",
                sub_stage="distance_write",
                circuit_module="dist_sram_buffer",
                operation="distance_write",
                buffer_name="dist_sram_buffer",
                phase="write",
                resource_id=f"dist_sram_bank{bank_idx}",
                total_bytes=bank_dist_bytes,
                hw_config=hw_config,
                is_read=False,
                preferred_chunk_bytes=dist_stream_chunk_bytes,
                extra_meta={
                    "query_index": query_index,
                    "bank_idx": bank_idx,
                    "num_banks": num_banks,
                    "phase": "distance_frontend_banked",
                },
                memory_context=memory_context,
            )
            bank_end += dist_write_dur
        cursor = bank_end

    return events, eid, cursor - base_cycle, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags


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
    # 纯 acc（非 PointCIM）：仅首层跑 FPS，后续层复用首层索引（SRAM 读）
    # PointCIM / baseline 后续层需重新计算 FPS
    _acc_pure = (getattr(hw_config, "is_acc", False)
                 and not getattr(hw_config, "use_rram_dist", False))
    if _acc_pure and context is not None:
        first_fps = context.get("first_layer_with_fps")
        layer_index = context.get("layer_index")
        if first_fps is not None and layer_index is not None and layer_index != first_fps:
            clock_ns = getattr(hw_config, "clock_period", 1.0)
            sampled_points_num = int(op_ref.params.get('output_points', 512))
            idx_bytes = _index_bytes_for_points(int(op_ref.params.get('input_points', 1024)))
            total_idx_bytes = max(1, sampled_points_num * idx_bytes)
            events: List[Event] = []
            eid, _ = _acc_append_sram_access_event(
                events, 0, 0, "FPS", "reuse",
                "fps_index_read", "fps_index_sram_buffer", "fps_index_reuse_read",
                "fps_index_sram_buffer", "read", "fps_index_shared", total_idx_bytes, hw_config,
                is_read=True, overlapped=False, extra_meta={"fps_reuse": True},
            )
            events.append(
                Event(
                    eid, 0, "FPS", "reuse", "fps_reuse", "fps_index_sram_buffer", "fps_reuse_first_layer",
                    0.0, 1, clock_ns, metadata={"fps_reuse": True, "buffer_name": "fps_index_sram_buffer", "bytes": total_idx_bytes},
                )
            )
            return events

    from .fps import FPSOperator, FPSParams

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
        # Respect fps_engine_mode:
        # - fuse: ACC speculative FPS path (warmup + predictor + resampler)
        # - baseline/non-fuse: pure full-sort style path without speculative events
        use_speculative_fps = str(getattr(hw_config, "fps_engine_mode", "fuse")).lower() == "fuse"
        warmup_iters = max(2, int(sampled_points_num * 0.1)) if use_speculative_fps else int(sampled_points_num)
        mdt_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "min_distance_table", None)
        coarse_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "coarse_filter", None)
        fine_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "fine_filter", None)
        coarse_power = float(getattr(coarse_cfg, "power", 81.12) or 81.12)
        coarse_cycles = max(1, int(getattr(coarse_cfg, "cycles", 1) or 1))
        fine_power = float(getattr(fine_cfg, "power", 38.0) or 38.0)
        fine_cycles = max(1, int(getattr(fine_cfg, "cycles", 2) or 2))
        mdt_power = float(getattr(mdt_cfg, "power", 30.0) or 30.0)
        segment_len = _acc_segment_len(sampled_points_num, params, hw_config)
        prediction_started = False
        predictor_ready_cycle = -1
        prediction_ready = False
        upsampled_until = 0
        upsample_issue_cycle = -1
        upsample_slots_used = 0
        downsample_started = False
        downsample_issue_cycle = -1
        downsample_done_cycle = -1
        exact_prefix_ready_cycles: List[int] = []
        predictor_output_points = max(1, int(params.get("predictor_output_points", 64) or 64))
        downsample_output_points = max(1, int(params.get("downsample_output_points", 32) or 32))
        downsample_table = _build_linear_interp_schedule(warmup_iters, downsample_output_points)
        downsample_next_out_idx = 0
        upsample_total_outputs = max(0, sampled_points_num - warmup_iters)
        upsample_table = _build_linear_interp_schedule(predictor_output_points, max(1, upsample_total_outputs)) if upsample_total_outputs > 0 else []
        upsample_next_out_idx = 0
        upsample_output_ready_cycles: List[int] = [-1 for _ in upsample_table]
        predictor_hidden_dim = max(1, int(params.get("predictor_hidden_dim", 256) or 256))
        temp_entry_bytes = _distance_index_entry_bytes(input_points)
        temp_block_topk = max(
            1,
            int(
                params.get(
                    "fps_block_topk",
                    getattr(hw_config, "fps_block_topk", 32),
                )
                or 32
            ),
        )
        fps_coarse_reduce_capacity = max(1, int(getattr(hw_config, 'cim_parallelism', 128) or 128))
        fps_fine_capacity = 32
        dist_tag_scope = f"fps_dist_{input_points}_{sampled_points_num}_eid{eid}"

        def _next_fps_reduce_candidate_total(current_total: int) -> int:
            if current_total <= fps_fine_capacity:
                return current_total
            group_count = max(1, int(math.ceil(current_total / float(fps_coarse_reduce_capacity))))
            reduced_total = group_count * min(temp_block_topk, fps_fine_capacity)
            if reduced_total >= current_total:
                reduced_total = fps_fine_capacity
            return max(fps_fine_capacity, reduced_total)
        fps_index_total_bytes = max(1, sampled_points_num * _index_bytes_for_points(input_points))
        for i in range(sampled_points_num):
            target_segment_end = _acc_segment_end_points(i, sampled_points_num, segment_len)
            if use_speculative_fps and prediction_started and current_cycle >= predictor_ready_cycle:
                up_lut_power, up_lut_cycles = _acc_resampler_lut(hw_config, "upsample")
                if upsample_issue_cycle < 0:
                    upsample_issue_cycle = predictor_ready_cycle
                    upsample_slots_used = 0

                required_outputs = max(0, target_segment_end - warmup_iters)
                lanes = max(1, int(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "parallelism", 1) or 1))
                up_power = float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "power", 3.95) or 3.95)
                while upsample_next_out_idx < min(required_outputs, len(upsample_table)):
                    if upsample_slots_used >= lanes:
                        upsample_issue_cycle += 1
                        upsample_slots_used = 0
                    row = upsample_table[upsample_next_out_idx]
                    lane = upsample_slots_used
                    step_start = max(upsample_issue_cycle, predictor_ready_cycle)
                    up_lut_ev = Event(
                        eid, step_start, "Resampler", "speculative", "resampler_upsample_lut", "resampler", "resampler_upsample_lut",
                        up_lut_power, up_lut_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "query_index": int(warmup_iters + row["out_idx"]),
                            "emit_query_index": int(i),
                            "output_idx": int(row["out_idx"]),
                            "pair_left": int(row["left"]),
                            "pair_right": int(row["right"]),
                            "src_pos": float(row["src_pos"]),
                            "interp_weight": float(row["w"]),
                            "target_query": int(warmup_iters + row["out_idx"]),
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_resample_up_lut",
                            "bytes": 4,
                            "resource_id": f"resampler_shared_lane{lane}",
                            "exclusive": True,
                            "phase": "predictor_gate",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    up_lut_ev.overlapped = True
                    all_events.append(up_lut_ev)
                    eid += 1
                    up_ev = Event(
                        eid, step_start, "Resampler", "speculative", "resampler_upsample_step", "resampler", "resampler_upsample_step",
                        up_power, 3, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "query_index": int(warmup_iters + row["out_idx"]),
                            "emit_query_index": int(i),
                            "output_idx": int(row["out_idx"]),
                            "pair_left": int(row["left"]),
                            "pair_right": int(row["right"]),
                            "src_pos": float(row["src_pos"]),
                            "interp_weight": float(row["w"]),
                            "target_query": int(warmup_iters + row["out_idx"]),
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "spec_resample_up",
                            "bytes": 4,
                            "resource_id": f"resampler_shared_lane{lane}",
                            "exclusive": True,
                            "phase": "predictor_gate",
                            "segment_len": segment_len,
                            "segment_end_points": target_segment_end,
                        },
                    )
                    all_events.append(up_ev)
                    eid += 1
                    upsample_output_ready_cycles[upsample_next_out_idx] = step_start + 3
                    upsample_next_out_idx += 1
                    upsample_slots_used += 1

                if required_outputs > 0 and all(c >= 0 for c in upsample_output_ready_cycles[:required_outputs]):
                    ready_cycle = max(upsample_output_ready_cycles[:required_outputs])
                    prediction_ready = current_cycle >= ready_cycle
                    if prediction_ready:
                        upsampled_until = max(upsampled_until, target_segment_end)
                else:
                    prediction_ready = False
            else:
                prediction_ready = False

            candidate_points = max(1, input_points - i)
            base_cycle = current_cycle
            evs_d, eid, dist_cycles, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags = _gen_acc_distance_query_events(
                eid,
                base_cycle,
                "FPS",
                input_points,
                hw_config,
                query_index=i,
                include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
                memory_context=context.get("memory_runtime_context") if context else None,
                startup_once=True,
                write_dist_buffer=False,
                tag_scope=dist_tag_scope,
                key_already_resident=bool(context and context.get(ACC_CIM_KEY_PRELOADED, False)),
            )
            all_events.extend(evs_d)
            current_cycle += dist_cycles
            if context is not None:
                context.setdefault("acc_block_ready_cycles_by_query", {})[i] = block_ready_cycles
                context.setdefault("acc_block_ready_tags_by_query", {})[i] = block_ready_tags
                context.setdefault("acc_block_prefetch_tags_by_query", {})[i] = block_prefetch_tags

            # The min-distance table is persistent across the whole reference set; each block
            # updates its corresponding table segment instead of a single local winner entry.
            mdt_table_total_bytes = _fps_mdt_total_bytes(input_points)
            mdt_requires_stream = mdt_table_total_bytes > _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)
            mdt_superblock_points = _fps_mdt_superblock_points(input_points, hw_config)
            mdt_blocks_per_superblock = max(1, int(math.ceil(mdt_superblock_points / float(_fps_mdt_update_block_points(hw_config)))))
            warmup_temp_total_bytes = temp_entry_bytes
            warmup_temp_onchip_bytes, warmup_temp_spill_bytes = _acc_spill_bytes("temp_max_distance_sram", warmup_temp_total_bytes, hw_config)
            memory_context = context.get("memory_runtime_context") if context else None

            if not prediction_ready:
                warmup_super_read_batch: Dict[int, Dict[str, Any]] = {}
                if memory_context is not None and mdt_requires_stream and block_ready_cycles and i != 0:
                    read_requests: List[Dict[str, Any]] = []
                    num_superblocks = max(1, int(math.ceil(len(block_ready_cycles) / float(mdt_blocks_per_superblock))))
                    for super_idx in range(num_superblocks):
                        first_blk = super_idx * mdt_blocks_per_superblock
                        if first_blk >= len(block_ready_cycles):
                            continue
                        blk_ready = block_ready_cycles[first_blk]
                        mdt_read_start = block_prefetch_cycles[first_blk] if first_blk < len(block_prefetch_cycles) else blk_ready
                        read_requests.append({
                            "super_idx": super_idx,
                            "num_bytes": _fps_mdt_superblock_bytes(input_points, super_idx, hw_config),
                            "is_read": True,
                            "start_cycle": mdt_read_start,
                        })
                    if read_requests:
                        read_results = memory_context.prime_explicit_batch_transfers(
                            read_requests,
                            dir_prefix=f"fps_warm_mdt_super_read_q{i}",
                        )
                        warmup_super_read_batch = {int(req["super_idx"]): res for req, res in zip(read_requests, read_results)}
                warmup_barrier_cycle = 0
                for blk, blk_ready in enumerate(block_ready_cycles):
                    is_initial_sample = (i == 0)
                    super_idx = _fps_mdt_superblock_index_for_block(blk, input_points, hw_config)
                    super_tag = f"fps_warm_mdt_super_read_q{i}_s{super_idx}"
                    is_superblock_first = (blk % mdt_blocks_per_superblock) == 0
                    is_superblock_last = ((blk + 1) % mdt_blocks_per_superblock) == 0 or blk == len(block_ready_cycles) - 1
                    mdt_segment_bytes = _fps_mdt_segment_bytes(input_points, blk, hw_config)
                    mdt_resident_bytes = mdt_segment_bytes
                    if (not is_initial_sample) and mdt_requires_stream and is_superblock_first:
                        mdt_read_start = block_prefetch_cycles[blk] if blk < len(block_prefetch_cycles) else blk_ready
                        eid, super_read_dur = _acc_append_spill_events(
                            all_events, eid, mdt_read_start, "FPS", "distance",
                            "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns,
                            "warmup_full_sort", "read", hw_config, memory_context,
                            precomputed_result=warmup_super_read_batch.get(super_idx),
                            extra_meta={
                                "iteration": i,
                                "query_index": i,
                                "superblock_idx": super_idx,
                                "superblock_points": _fps_mdt_superblock_points(input_points, hw_config),
                                "dependency_tag": super_tag,
                            },
                        )
                        warmup_barrier_cycle = max(warmup_barrier_cycle, mdt_read_start + super_read_dur)
                    if not is_initial_sample:
                        mdt_read_start = block_prefetch_cycles[blk] if blk < len(block_prefetch_cycles) else blk_ready
                        eid, mdt_read_dur = _acc_append_buffer_consume_events(
                            all_events, eid, mdt_read_start, "FPS", "distance",
                            sub_stage="min_distance_table_read",
                            circuit_module="fps_min_distance_table_sram",
                            operation="fps_min_distance_table_read",
                            buffer_name="fps_min_distance_table_sram",
                            phase="read",
                            resource_id="fps_min_table_read_shared",
                            total_bytes=mdt_segment_bytes,
                            hw_config=hw_config,
                            resident_bytes=mdt_resident_bytes,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                            extra_meta={
                                "iteration": i,
                                "query_index": i,
                                "block_idx": blk,
                                "num_blocks": len(block_ready_cycles),
                                "phase": "warmup_full_sort",
                                "depends_on_tags": [t for t in [(block_prefetch_tags[blk] if blk < len(block_prefetch_tags) else None), (super_tag if mdt_requires_stream else None)] if t],
                                "dependency_tag": f"fps_warm_mdt_read_q{i}_b{blk}",
                            },
                            memory_context=memory_context,
                        )
                        min_start = max(blk_ready, mdt_read_start + mdt_read_dur)
                        min_ev = Event(
                            eid, min_start, "FPS", "distance", "min_distance_update", "min_distance_table", "min_distance_compare",
                            mdt_power, 1, clock_period_ns=clock_ns,
                            metadata={
                                "iteration": i,
                                "query_index": i,
                                "block_idx": blk,
                                "num_blocks": len(block_ready_cycles),
                                "buffer_name": "fps_min_distance_table_sram",
                                "buffer_phase": "update",
                                "bytes": mdt_segment_bytes,
                                "resource_id": "fps_min_update_shared",
                                "exclusive": True,
                                "phase": "warmup_full_sort",
                                "depends_on_tags": [t for t in [f"fps_warm_mdt_read_q{i}_b{blk}", (block_ready_tags[blk] if blk < len(block_ready_tags) else None)] if t],
                                "dependency_tag": f"fps_warm_min_q{i}_b{blk}",
                            },
                        )
                        min_ev_id = eid
                        all_events.append(min_ev)
                        eid += 1
                    else:
                        mdt_read_dur = 0
                    write_start = blk_ready if is_initial_sample else (min_start + 1)
                    eid, mdt_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, eid, write_start, "FPS", "distance",
                        sub_stage="min_distance_table_write",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_min_distance_table_write",
                        buffer_name="fps_min_distance_table_sram",
                        phase="update",
                        resource_id="fps_min_table_write_shared",
                        total_bytes=mdt_segment_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        resident_bytes_before=0,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                        extra_meta={
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": len(block_ready_cycles),
                            "phase": "warmup_full_sort",
                            "depends_on_tag": f"fps_warm_min_q{i}_b{blk}" if not is_initial_sample else (block_ready_tags[blk] if blk < len(block_ready_tags) else None),
                            "dependency_tag": f"fps_warm_mdt_write_q{i}_b{blk}",
                        },
                        memory_context=memory_context,
                    )
                    if mdt_requires_stream and is_superblock_last:
                        eid, mdt_wb_dur = _acc_append_spill_events(
                            all_events, eid, write_start + mdt_write_dur, "FPS", "distance",
                            "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns, "warmup_full_sort", "write",
                            hw_config, memory_context,
                            extra_meta={
                                "iteration": i,
                                "query_index": i,
                                "superblock_idx": super_idx,
                                "superblock_points": _fps_mdt_superblock_points(input_points, hw_config),
                                "depends_on_tag": f"fps_warm_mdt_write_q{i}_b{blk}",
                                "dependency_tag": f"fps_warm_mdt_super_write_q{i}_s{super_idx}",
                            },
                        )
                        mdt_write_dur += mdt_wb_dur
                    fps_coarse_start = blk_ready if is_initial_sample else (min_start + 1)
                    coarse_meta = {
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": len(block_ready_cycles),
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "warmup_coarse_write",
                            "bytes": warmup_temp_onchip_bytes,
                            "resource_id": "comparator_shared",
                            "resource_priority": 20,
                            "exclusive": True,
                            "phase": "warmup_full_sort",
                        }
                    if not is_initial_sample:
                        coarse_meta["depends_on_tag"] = f"fps_warm_min_q{i}_b{blk}"
                    elif blk < len(block_ready_tags):
                        coarse_meta["depends_on_tag"] = block_ready_tags[blk]
                    coarse_meta["dependency_tag"] = f"fps_warm_coarse_q{i}_b{blk}"
                    coarse_ev = Event(
                        eid, fps_coarse_start, "FPS", "warmup_full_sort", "coarse_filter", "coarse_filter", "fps_warmup_coarse_top32",
                        coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                        metadata=coarse_meta,
                    )
                    coarse_ev_id = eid
                    all_events.append(coarse_ev)
                    eid += 1
                    fine_start = fps_coarse_start + coarse_cycles
                    fine_meta = {
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": len(block_ready_cycles),
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "warmup_fine_write",
                            "bytes": temp_entry_bytes,
                            "resource_id": "comparator_shared",
                            "resource_priority": 30,
                            "exclusive": True,
                            "phase": "warmup_full_sort",
                        }
                    fine_meta["depends_on_tag"] = f"fps_warm_coarse_q{i}_b{blk}"
                    fine_ev = Event(
                        eid, fine_start, "FPS", "warmup_full_sort", "fine_filter", "fine_filter", "fps_warmup_fine_max",
                        fine_power, fine_cycles, clock_period_ns=clock_ns,
                        metadata=fine_meta,
                    )
                    all_events.append(fine_ev)
                    eid += 1
                    warmup_barrier_cycle = max(
                        warmup_barrier_cycle,
                        fine_start + fine_cycles,
                        write_start + mdt_write_dur,
                    )
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, warmup_barrier_cycle, "FPS", "warmup_full_sort", "temp_max_distance_sram", warmup_temp_spill_bytes,
                    clock_ns, "warmup_full_sort", "write", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                warmup_barrier_cycle += spill_dur
                current_cycle = max(current_cycle, warmup_barrier_cycle)
                eid, prefix_wr_dur, _ = _acc_append_buffer_fill_events(
                    all_events, eid, current_cycle, "FPS", "warmup_full_sort",
                    sub_stage="prefix_sample_write",
                    circuit_module="temp_max_distance_sram",
                    operation="fps_prefix_sample_write",
                    buffer_name="temp_max_distance_sram",
                    phase="speculative_overlap",
                    resource_id="temp_max_shared",
                    total_bytes=_ACC_MEM_POLICY.distance_bytes,
                    hw_config=hw_config,
                    is_read=False,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                    extra_meta={"iteration": i, "query_index": i},
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                prefix_ready_cycle = current_cycle + prefix_wr_dur
                exact_prefix_ready_cycles.append(prefix_ready_cycle)
                current_cycle = max(current_cycle, prefix_ready_cycle)

                prefix_points = i + 1
                if use_speculative_fps and prefix_points >= 2:
                    down_power = float(getattr(getattr(getattr(hw_config, "logic_circuit", None), "resampler", None), "power", 3.95) or 3.95)
                    down_lut_power, _down_lut_cycles = _acc_resampler_lut(hw_config, "downsample")
                    if not downsample_started:
                        second_ready = exact_prefix_ready_cycles[1]
                        downsample_started = True
                        downsample_issue_cycle = second_ready
                    if prefix_points <= warmup_iters:
                        available_input_idx = i
                        step_cycles = 3
                        while downsample_next_out_idx < len(downsample_table):
                            row = downsample_table[downsample_next_out_idx]
                            right_idx = int(row["right"])
                            if right_idx > available_input_idx:
                                break
                            right_ready = exact_prefix_ready_cycles[right_idx]
                            step_start = max(downsample_issue_cycle, right_ready)
                            down_lut_ev = Event(
                                eid, step_start, "Resampler", "speculative", "resampler_downsample_lut", "resampler", "resampler_downsample_lut",
                                down_lut_power, 1, clock_period_ns=clock_ns,
                                metadata={
                                    "iteration": i,
                                    "query_index": right_idx,
                                    "output_idx": int(row["out_idx"]),
                                    "pair_left": int(row["left"]),
                                    "pair_right": right_idx,
                                    "src_pos": float(row["src_pos"]),
                                    "interp_weight": float(row["w"]),
                                    "ready_query": right_idx,
                                    "buffer_name": "temp_max_distance_sram",
                                    "buffer_phase": "spec_prefix_downsample_lut",
                                    "bytes": 4,
                                    "resource_id": "resampler_shared",
                                    "exclusive": True,
                                    "phase": "speculative_overlap",
                                },
                            )
                            down_lut_ev.overlapped = True
                            all_events.append(down_lut_ev)
                            eid += 1
                            down_ev = Event(
                                eid, step_start, "Resampler", "speculative", "resampler_downsample_step", "resampler", "resampler_downsample_step",
                                down_power, step_cycles, clock_period_ns=clock_ns,
                                metadata={
                                    "iteration": i,
                                    "output_idx": int(row["out_idx"]),
                                    "pair_left": int(row["left"]),
                                    "pair_right": right_idx,
                                    "src_pos": float(row["src_pos"]),
                                    "interp_weight": float(row["w"]),
                                    "ready_query": right_idx,
                                    "buffer_name": "temp_max_distance_sram",
                                    "buffer_phase": "spec_prefix_downsample",
                                    "bytes": 4,
                                    "resource_id": "resampler_shared",
                                    "exclusive": True,
                                    "phase": "speculative_overlap",
                                },
                            )
                            down_ev.overlapped = True
                            all_events.append(down_ev)
                            eid += 1
                            downsample_issue_cycle = step_start + 1
                            downsample_done_cycle = max(downsample_done_cycle, step_start + step_cycles)
                            downsample_next_out_idx += 1

                if use_speculative_fps and prefix_points >= warmup_iters and not prediction_started:
                    gate_cycle = max(current_cycle, downsample_done_cycle)
                    pred_events, eid, pred_dur = _acc_predictor_events(
                        eid,
                        gate_cycle,
                        hw_config,
                        context,
                        input_dim=downsample_output_points,
                        hidden_dim=predictor_hidden_dim,
                        output_dim=predictor_output_points,
                    )
                    all_events.extend(pred_events)
                    prediction_started = True
                    predictor_ready_cycle = gate_cycle + pred_dur
            else:
                num_blocks = max(1, len(block_ready_cycles) if block_ready_cycles else int(math.ceil(input_points / float(max(1, getattr(hw_config, 'cim_parallelism', 128) or 128)))))
                block_issue_cycles = max(1, int(getattr(hw_config, 'bit_serial_cycles', 1) or 1))
                mdt_blocks_per_superblock = max(1, int(math.ceil(_fps_mdt_superblock_points(input_points, hw_config) / float(_fps_mdt_update_block_points(hw_config)))))
                temp_per_block_bytes = temp_block_topk * temp_entry_bytes
                temp_total_bytes = num_blocks * temp_per_block_bytes
                temp_onchip_bytes, temp_spill_bytes = _acc_spill_bytes("temp_max_distance_sram", temp_total_bytes, hw_config)
                post_super_read_batch: Dict[int, Dict[str, Any]] = {}
                if memory_context is not None and mdt_requires_stream and num_blocks > 0:
                    read_requests: List[Dict[str, Any]] = []
                    num_superblocks = max(1, int(math.ceil(num_blocks / float(mdt_blocks_per_superblock))))
                    for super_idx in range(num_superblocks):
                        first_blk = super_idx * mdt_blocks_per_superblock
                        if first_blk >= num_blocks:
                            continue
                        blk_start = block_ready_cycles[first_blk] if first_blk < len(block_ready_cycles) else (base_cycle + first_blk * block_issue_cycles)
                        mdt_read_start = block_prefetch_cycles[first_blk] if first_blk < len(block_prefetch_cycles) else blk_start
                        read_requests.append({
                            "super_idx": super_idx,
                            "num_bytes": _fps_mdt_superblock_bytes(input_points, super_idx, hw_config),
                            "is_read": True,
                            "start_cycle": mdt_read_start,
                        })
                    if read_requests:
                        read_results = memory_context.prime_explicit_batch_transfers(
                            read_requests,
                            dir_prefix=f"fps_post_mdt_super_read_q{i}",
                        )
                        post_super_read_batch = {int(req["super_idx"]): res for req, res in zip(read_requests, read_results)}

                barrier_cycle = current_cycle
                temp_sram_cursor = current_cycle
                post_append_tags: List[str] = []
                for blk in range(num_blocks):
                    blk_start = block_ready_cycles[blk] if blk < len(block_ready_cycles) else (base_cycle + blk * block_issue_cycles)
                    super_idx = _fps_mdt_superblock_index_for_block(blk, input_points, hw_config)
                    super_tag = f"fps_post_mdt_super_read_q{i}_s{super_idx}"
                    is_superblock_first = (blk % mdt_blocks_per_superblock) == 0
                    is_superblock_last = ((blk + 1) % mdt_blocks_per_superblock) == 0 or blk == num_blocks - 1
                    mdt_segment_bytes = _fps_mdt_segment_bytes(input_points, blk, hw_config)
                    mdt_resident_bytes = mdt_segment_bytes
                    mdt_read_start = block_prefetch_cycles[blk] if blk < len(block_prefetch_cycles) else blk_start
                    if mdt_requires_stream and is_superblock_first:
                        eid, super_read_dur = _acc_append_spill_events(
                            all_events, eid, mdt_read_start, "FPS", "distance",
                            "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns, "post_predict_coarse_fine", "read",
                            hw_config, memory_context,
                            precomputed_result=post_super_read_batch.get(super_idx),
                            extra_meta={
                                "iteration": i,
                                "query_index": i,
                                "superblock_idx": super_idx,
                                "superblock_points": _fps_mdt_superblock_points(input_points, hw_config),
                                "dependency_tag": super_tag,
                            },
                        )
                        barrier_cycle = max(barrier_cycle, mdt_read_start + super_read_dur)
                    eid, mdt_read_dur = _acc_append_buffer_consume_events(
                        all_events, eid, mdt_read_start, "FPS", "distance",
                        sub_stage="min_distance_table_read",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_min_distance_table_read",
                        buffer_name="fps_min_distance_table_sram",
                        phase="read",
                        resource_id="fps_min_table_read_shared",
                        total_bytes=mdt_segment_bytes,
                        hw_config=hw_config,
                        resident_bytes=mdt_resident_bytes,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                        extra_meta={
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": num_blocks,
                            "phase": "post_predict_coarse_fine",
                            "depends_on_tags": [t for t in [(block_prefetch_tags[blk] if blk < len(block_prefetch_tags) else None), (super_tag if mdt_requires_stream else None)] if t],
                            "dependency_tag": f"fps_post_mdt_read_q{i}_b{blk}",
                        },
                        memory_context=memory_context,
                    )
                    min_start = max(blk_start, mdt_read_start + mdt_read_dur)
                    min_ev = Event(
                        eid, min_start, "FPS", "distance", "min_distance_update", "min_distance_table", "min_distance_compare",
                        mdt_power, 1, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": num_blocks,
                            "buffer_name": "fps_min_distance_table_sram",
                            "buffer_phase": "update",
                            "bytes": mdt_segment_bytes,
                            "resource_id": "fps_min_update_shared",
                            "exclusive": True,
                            "phase": "post_predict_coarse_fine",
                            "depends_on_tags": [t for t in [f"fps_post_mdt_read_q{i}_b{blk}", (block_ready_tags[blk] if blk < len(block_ready_tags) else None)] if t],
                            "dependency_tag": f"fps_post_min_q{i}_b{blk}",
                        },
                    )
                    min_ev_id = eid
                    all_events.append(min_ev)
                    eid += 1
                    eid, mdt_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, eid, min_start + 1, "FPS", "distance",
                        sub_stage="min_distance_table_write",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_min_distance_table_write",
                        buffer_name="fps_min_distance_table_sram",
                        phase="update",
                        resource_id="fps_min_table_write_shared",
                        total_bytes=mdt_segment_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                        extra_meta={
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": num_blocks,
                            "phase": "post_predict_coarse_fine",
                            "depends_on_tag": f"fps_post_min_q{i}_b{blk}",
                            "dependency_tag": f"fps_post_mdt_write_q{i}_b{blk}",
                        },
                        memory_context=memory_context,
                    )
                    if mdt_requires_stream and is_superblock_last:
                        eid, mdt_wb_dur = _acc_append_spill_events(
                            all_events, eid, min_start + 1 + mdt_write_dur, "FPS", "distance",
                            "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns, "post_predict_coarse_fine", "write",
                            hw_config, memory_context,
                            extra_meta={
                                "iteration": i,
                                "query_index": i,
                                "superblock_idx": super_idx,
                                "superblock_points": _fps_mdt_superblock_points(input_points, hw_config),
                                "depends_on_tag": f"fps_post_mdt_write_q{i}_b{blk}",
                                "dependency_tag": f"fps_post_mdt_super_write_q{i}_s{super_idx}",
                            },
                        )
                        mdt_write_dur += mdt_wb_dur
                    # Post-predict coarse filtering depends on the just-computed min-distance
                    # values, while table write can proceed in parallel on the SRAM port.
                    fps_coarse_start = min_start + 1
                    coarse_meta = {
                            "iteration": i,
                            "query_index": i,
                            "block_idx": blk,
                            "num_blocks": num_blocks,
                            "buffer_name": "temp_max_distance_sram",
                            "buffer_phase": "append_write",
                            "bytes": temp_per_block_bytes,
                            "resource_id": "comparator_shared",
                            "exclusive": True,
                            "phase": "post_predict_coarse_fine",
                            "depends_on_tag": f"fps_post_min_q{i}_b{blk}",
                            "dependency_tag": f"fps_post_coarse_q{i}_b{blk}",
                        }
                    coarse_ev = Event(
                        eid, fps_coarse_start, "FPS", "post_predict", "coarse_filter", "coarse_filter", "fps_coarse_filter_max",
                        coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                        metadata=coarse_meta,
                    )
                    coarse_ev.overlapped = True
                    coarse_ev_id = eid
                    all_events.append(coarse_ev)
                    eid += 1
                    append_start = max(fps_coarse_start + coarse_cycles, temp_sram_cursor)
                    eid, append_dur, _ = _acc_append_buffer_fill_events(
                        all_events, eid, append_start, "FPS", "post_predict",
                        sub_stage="temp_max_append_write",
                        circuit_module="temp_max_distance_sram",
                        operation="temp_max_candidate_append",
                        buffer_name="temp_max_distance_sram",
                        phase="append_write",
                        resource_id="temp_max_shared",
                        total_bytes=temp_per_block_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
	                        extra_meta={"iteration": i, "block_idx": blk, "num_blocks": num_blocks, "phase": "post_predict_coarse_fine", "depends_on_tag": f"fps_post_coarse_q{i}_b{blk}", "dependency_tag": f"fps_post_append_q{i}_b{blk}"},
	                        memory_context=context.get("memory_runtime_context") if context else None,
	                    )
                    post_append_tags.append(f"fps_post_append_q{i}_b{blk}")
                    temp_sram_cursor = append_start + append_dur
                    barrier_cycle = max(
                        barrier_cycle,
                        temp_sram_cursor,
                        blk_start + 1 + mdt_write_dur,
                    )

                current_candidate_total = max(1, temp_total_bytes // max(1, temp_entry_bytes))
                current_total_bytes = temp_total_bytes
                current_buffer = "temp_max_distance_sram"
                current_resource_id = "temp_max_shared"
                current_onchip_bytes = temp_onchip_bytes
                current_spill_bytes = temp_spill_bytes

                reduce_round = 0
                prev_reduce_write_tag = ""
                while current_candidate_total > fps_fine_capacity:
                    read_start = max(barrier_cycle, temp_sram_cursor)
                    eid, read_dur = _acc_append_buffer_consume_events(
                        all_events, eid, read_start, "FPS", "post_predict",
                        sub_stage="temp_max_coarse_reduce_read",
                        circuit_module=current_buffer,
                        operation="fps_tempmax_reduce_read",
                        buffer_name=current_buffer,
                        phase="coarse_reduce_read",
                        resource_id=current_resource_id,
                        total_bytes=current_total_bytes,
                        hw_config=hw_config,
                        resident_bytes=current_onchip_bytes,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(current_buffer, hw_config),
                        extra_meta={
                            "iteration": i,
                            "query_index": i,
                            "num_blocks": num_blocks,
                            "phase": "post_predict_coarse_fine",
                            "candidate_count": current_candidate_total,
                            "depends_on_tags": post_append_tags if current_buffer == "temp_max_distance_sram" else [prev_reduce_write_tag],
                            "dependency_tag": f"fps_reduce_read_q{i}_r{reduce_round}",
                        },
                        memory_context=context.get("memory_runtime_context") if context else None,
                    )
                    temp_sram_cursor = read_start + read_dur
                    barrier_cycle = max(barrier_cycle, temp_sram_cursor)
                    eid, spill_dur = _acc_append_spill_events(
                        all_events, eid, barrier_cycle, "FPS", "post_predict", current_buffer, current_spill_bytes,
                        clock_ns, "post_predict_coarse_fine", "read", hw_config,
                        context.get("memory_runtime_context") if context else None,
                    )
                    barrier_cycle += spill_dur

                    next_candidate_total = _next_fps_reduce_candidate_total(current_candidate_total)
                    next_total_bytes = next_candidate_total * temp_entry_bytes
                    next_buffer = "sort_cache_sram"
                    next_resource_id = "sort_cache_shared"
                    next_onchip_bytes, next_spill_bytes = _acc_spill_bytes(next_buffer, next_total_bytes, hw_config)

                    coarse_ev = Event(
                        eid, barrier_cycle, "FPS", "post_predict", "coarse_filter", "coarse_filter", "fps_coarse_filter_max",
                        coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                        metadata={
                            "iteration": i,
                            "buffer_name": current_buffer,
                            "buffer_phase": "coarse_reduce_read",
                            "bytes": current_onchip_bytes,
                            "resource_id": "comparator_shared",
                            "exclusive": True,
                            "phase": "post_predict_coarse_fine",
                            "candidate_count": current_candidate_total,
                            "next_candidate_count": next_candidate_total,
                            "depends_on_tag": f"fps_reduce_read_q{i}_r{reduce_round}",
                            "dependency_tag": f"fps_reduce_coarse_q{i}_r{reduce_round}",
                        },
                    )
                    all_events.append(coarse_ev)
                    eid += 1
                    barrier_cycle += coarse_cycles

                    eid, write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, eid, barrier_cycle, "FPS", "post_predict",
                        sub_stage="temp_max_coarse_reduce_write",
                        circuit_module=next_buffer,
                        operation="fps_tempmax_reduce_write",
                        buffer_name=next_buffer,
                        phase="coarse_reduce_write",
                        resource_id=next_resource_id,
                        total_bytes=next_total_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(next_buffer, hw_config),
                        extra_meta={
                            "iteration": i,
                            "query_index": i,
                            "num_blocks": num_blocks,
                            "phase": "post_predict_coarse_fine",
                            "candidate_count": current_candidate_total,
                            "next_candidate_count": next_candidate_total,
                            "depends_on_tag": f"fps_reduce_coarse_q{i}_r{reduce_round}",
                            "dependency_tag": f"fps_reduce_write_q{i}_r{reduce_round}",
                        },
                        memory_context=context.get("memory_runtime_context") if context else None,
                    )
                    barrier_cycle += write_dur
                    eid, spill_dur = _acc_append_spill_events(
                        all_events, eid, barrier_cycle, "FPS", "post_predict", next_buffer, next_spill_bytes,
                        clock_ns, "post_predict_coarse_fine", "write", hw_config,
                        context.get("memory_runtime_context") if context else None,
                    )
                    barrier_cycle += spill_dur

                    current_candidate_total = next_candidate_total
                    current_total_bytes = next_total_bytes
                    current_buffer = next_buffer
                    current_resource_id = next_resource_id
                    current_onchip_bytes = next_onchip_bytes
                    current_spill_bytes = next_spill_bytes
                    prev_reduce_write_tag = f"fps_reduce_write_q{i}_r{reduce_round}"
                    reduce_round += 1

                read_start = max(barrier_cycle, temp_sram_cursor)
                eid, read_dur = _acc_append_buffer_consume_events(
                    all_events, eid, read_start, "FPS", "post_predict",
                    sub_stage="temp_max_fine_read",
                    circuit_module=current_buffer,
                    operation="fps_tempmax_global_read",
                    buffer_name=current_buffer,
                    phase="fine_filter_read",
                    resource_id=current_resource_id,
                    total_bytes=current_total_bytes,
                    hw_config=hw_config,
                    resident_bytes=current_onchip_bytes,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes(current_buffer, hw_config),
                    extra_meta={
                        "iteration": i,
                        "num_blocks": num_blocks,
                        "phase": "post_predict_coarse_fine",
                        "candidate_count": current_candidate_total,
                        "depends_on_tags": post_append_tags if current_buffer == "temp_max_distance_sram" else [prev_reduce_write_tag],
                        "dependency_tag": f"fps_fine_read_q{i}",
                    },
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                temp_sram_cursor = read_start + read_dur
                barrier_cycle = max(barrier_cycle, temp_sram_cursor)
                eid, spill_dur = _acc_append_spill_events(
                    all_events, eid, barrier_cycle, "FPS", "post_predict", current_buffer, current_spill_bytes,
                    clock_ns, "post_predict_coarse_fine", "read", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                barrier_cycle += spill_dur

                fine_ev = Event(
                    eid, barrier_cycle, "FPS", "post_predict", "fine_filter", "fine_filter", "fps_fine_filter_max",
                    fine_power, fine_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "iteration": i,
                        "buffer_name": current_buffer,
                        "buffer_phase": "fine_filter_read",
                        "bytes": current_onchip_bytes,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "post_predict_coarse_fine",
                        "num_blocks": num_blocks,
                        "candidate_count": current_candidate_total,
                        "depends_on_tag": f"fps_fine_read_q{i}",
                    },
                )
                all_events.append(fine_ev)
                eid += 1
                current_cycle = max(current_cycle, barrier_cycle + fine_cycles)
        eid, write_dur = _acc_append_sram_access_event(
            all_events, eid, current_cycle, "FPS", "reuse",
            "fps_index_write", "fps_index_sram_buffer", "fps_index_buffer_write",
            "fps_index_sram_buffer", "write", "fps_index_shared", fps_index_total_bytes, hw_config,
            is_read=False, overlapped=False, extra_meta={"phase": "fps_result_persist"},
        )
        current_cycle += write_dur
        if context is not None:
            context["acc_shared_distance_frontend_ready"] = True
            # Persist cross-stage flag: the SRAM-CIM array now holds the full key point cloud
            # for this stage.  Subsequent GAM/SA stages reuse a FPS-selected subset via masking
            # (no re-preload needed); only the per-query cross-term computation changes.
            context[ACC_CIM_KEY_PRELOADED] = True
        return all_events

    # Baseline Phase2 + FPS 引擎交错：每迭代 [距离计算] -> [bitonic]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    all_events: List[Event] = []
    current_cycle = 0
    memory_context = context.get("memory_runtime_context") if context else None
    mdt_cfg = getattr(getattr(hw_config, "sorting_circuit", None), "min_distance_table", None)
    mdt_power = float(getattr(mdt_cfg, "power", 0.0) or 0.0)
    for i in range(sampled_points_num):
        evs_d, _, current_cycle = _gen_baseline_phase2_distance_events(
            input_points,
            hw_config,
            "FPS",
            start_cycle=current_cycle,
            include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
            metadata={"iteration": i, "query_index": i},
        )
        all_events.extend(evs_d)

        if getattr(hw_config, "is_baseline_v2", False):
            candidate_points = max(1, int(input_points) - int(i))
            sorter_depth = _sorter_depth_points(hw_config)
            num_passes = max(1, int(math.ceil(candidate_points / float(sorter_depth))))
            temp_entry_bytes = _distance_index_entry_bytes(candidate_points)
            ms = getattr(hw_config.sorting_circuit, "merge_sort", None)
            reg = getattr(ms, "reg_file", None) if ms is not None else None
            merge_cycles = int(getattr(ms, "cycles_max", 22) or 22) if ms is not None else 22
            reg_read_cycles = int(getattr(reg, "cycles_per_read", 1) or 1) if reg is not None else 1
            dist2_eng = getattr(hw_config, "baseline_dist2_engine", None)
            local_batch_points = max(1, int(getattr(dist2_eng, "pe_count", getattr(dist2_eng, "c_dim", 64)) or 64))
            mdt_requires_stream = _fps_mdt_total_bytes(input_points) > _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)
            mdt_batches_per_superblock = max(1, int(math.ceil(_fps_mdt_superblock_points(input_points, hw_config) / float(local_batch_points))))
            batches_per_pass = max(1, int(math.ceil(sorter_depth / float(local_batch_points))))
            batch_ready = _group_ready_cycles(evs_d, "dist2_output_buffer_write", "batch")
            merge_cursor = 0
            pass_ready_cycles: List[int] = []

            for pass_idx in range(num_passes):
                pass_points = min(sorter_depth, max(0, candidate_points - pass_idx * sorter_depth))
                if pass_points <= 0:
                    continue
                pass_batches = max(1, int(math.ceil(pass_points / float(local_batch_points))))
                pass_batch_ready: List[int] = []
                for local_batch_idx in range(pass_batches):
                    global_batch_idx = pass_idx * batches_per_pass + local_batch_idx
                    batch_points = min(local_batch_points, max(0, pass_points - local_batch_idx * local_batch_points))
                    if batch_points <= 0:
                        continue
                    batch_start = max(int(batch_ready.get(global_batch_idx, 0)), merge_cursor)
                    dist_batch_bytes = int(batch_points) * 4
                    _, dist_read_dur = _acc_append_buffer_consume_events(
                        all_events, len(all_events), batch_start, "FPS", "min_distance_update",
                        sub_stage="dist_output_read",
                        circuit_module="dist_sram_buffer",
                        operation="dist_buffer_read_for_mdt",
                        buffer_name="dist_sram_buffer",
                        phase="read",
                        resource_id="dist_sram_shared",
                        total_bytes=dist_batch_bytes,
                        hw_config=hw_config,
                        resident_bytes=min(dist_batch_bytes, _acc_buffer_capacity_bytes("dist_sram_buffer", hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("dist_sram_buffer", hw_config),
                        extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "kind": "dist_to_mdt", "reset_scope": "fps_iteration"},
                        memory_context=context.get("memory_runtime_context") if context else None,
                    )
                    batch_cursor = batch_start + dist_read_dur
                    table_batch_bytes = int(batch_points) * _fps_mdt_entry_bytes()
                    super_idx = global_batch_idx // mdt_batches_per_superblock
                    is_superblock_first = (global_batch_idx % mdt_batches_per_superblock) == 0
                    is_superblock_last = ((global_batch_idx + 1) % mdt_batches_per_superblock) == 0 or global_batch_idx == (num_passes * batches_per_pass - 1) or local_batch_idx == pass_batches - 1
                    if i != 0:
                        if mdt_requires_stream and is_superblock_first:
                            _, sb_read_dur = _acc_append_spill_events(
                                all_events, len(all_events), batch_cursor, "FPS", "min_distance_update",
                                "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns, "fps_iteration", "read",
                                hw_config, memory_context,
                                extra_meta={"iteration": i, "pass_idx": pass_idx, "superblock_idx": super_idx},
                            )
                            batch_cursor += sb_read_dur
                        _, mdt_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), batch_cursor, "FPS", "min_distance_update",
                            sub_stage="min_distance_table_read",
                            circuit_module="fps_min_distance_table_sram",
                            operation="fps_min_distance_table_read",
                            buffer_name="fps_min_distance_table_sram",
                            phase="read",
                            resource_id="fps_min_table_shared",
                            total_bytes=table_batch_bytes,
                            hw_config=hw_config,
                            resident_bytes=table_batch_bytes,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                            extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "reset_scope": "fps_iteration"},
                            memory_context=memory_context,
                        )
                        batch_cursor += mdt_read_dur
                        all_events.append(Event(
                            len(all_events), batch_cursor, "FPS", "min_distance_update", "table_update", "min_distance_table", "min_distance_compare",
                            mdt_power, 1, clock_ns,
                            metadata={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "update", "bytes": table_batch_bytes, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
                        ))
                        batch_cursor += 1
                    _, mdt_write_dur = _acc_append_sram_access_event(
                        all_events, len(all_events), batch_cursor, "FPS", "min_distance_update",
                        sub_stage="min_distance_table_write",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_min_distance_table_write",
                        buffer_name="fps_min_distance_table_sram",
                        phase="update",
                        resource_id="fps_min_table_shared",
                        num_bytes=table_batch_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "reset_scope": "fps_iteration"},
                        return_actual_duration=True,
                    )
                    batch_cursor += mdt_write_dur
                    if mdt_requires_stream and is_superblock_last:
                        _, sb_write_dur = _acc_append_spill_events(
                            all_events, len(all_events), batch_cursor, "FPS", "min_distance_update",
                            "fps_min_distance_table_sram", _fps_mdt_superblock_bytes(input_points, super_idx, hw_config), clock_ns, "fps_iteration", "write",
                            hw_config, memory_context,
                            extra_meta={"iteration": i, "pass_idx": pass_idx, "superblock_idx": super_idx},
                        )
                        batch_cursor += sb_write_dur
                    _, sort_read_dur = _acc_append_buffer_consume_events(
                        all_events, len(all_events), batch_cursor, "FPS", "merge_sort",
                        sub_stage="local_reduce_read",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_merge_sort_reduce_read",
                        buffer_name="fps_min_distance_table_sram",
                        phase="local_reduce_read",
                        resource_id="fps_min_table_shared",
                        total_bytes=int(batch_points) * 4,
                        hw_config=hw_config,
                        resident_bytes=min(int(batch_points) * 4, _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                        extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx},
                        memory_context=context.get("memory_runtime_context") if context else None,
                    )
                    batch_cursor += sort_read_dur
                    all_events.append(Event(
                        len(all_events), batch_cursor, "FPS", "merge_sort", "merge_sort_local_max", "merge_sort", "merge_sort_max",
                        float(getattr(ms, "power_max_mw", 310.0) or 310.0), merge_cycles, clock_period_ns=clock_ns,
                        metadata={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "local_reduce_read", "bytes": int(batch_points) * 4, "resource_id": "merge_sort_shared", "exclusive": True},
                    ))
                    batch_cursor += merge_cycles
                    all_events.append(Event(
                        len(all_events), batch_cursor, "SRAM", "merge_sort_reg_file", "reg_file_read_1max", "sort_cache_sram", "reg_file_read",
                        0.0, reg_read_cycles, clock_period_ns=clock_ns,
                        metadata={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                    ))
                    batch_cursor += reg_read_cycles
                    temp_write_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="temp_max_distance_sram",
                        total_bytes=temp_entry_bytes,
                        cycle=batch_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        dir_prefix=f"fps_bv2_temp_write_i{i}_p{pass_idx}_b{global_batch_idx}",
                    )
                    _, temp_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), batch_cursor, "FPS", "merge_sort",
                        sub_stage="temp_max_local_write",
                        circuit_module="temp_max_distance_sram",
                        operation="fps_temp_max_local_write",
                        buffer_name="temp_max_distance_sram",
                        phase="local_candidate_write",
                        resource_id="temp_max_shared",
                        total_bytes=temp_entry_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx},
                        memory_context=memory_context,
                        precomputed_spill_results=temp_write_batch_results,
                    )
                    batch_cursor += temp_write_dur
                    pass_batch_ready.append(batch_cursor)
                    merge_cursor = batch_cursor

                pass_cursor = max(pass_batch_ready) if pass_batch_ready else merge_cursor
                source_buffer = "temp_max_distance_sram"
                source_item_bytes = temp_entry_bytes
                current_candidates = len(pass_batch_ready)
                while current_candidates > 1:
                    chunks = max(1, int(math.ceil(current_candidates / float(local_batch_points))))
                    dest_buffer = "sort_cache_sram" if source_buffer == "temp_max_distance_sram" else "temp_max_distance_sram"
                    produced = 0
                    for chunk_idx in range(chunks):
                        chunk_count = min(local_batch_points, max(0, current_candidates - chunk_idx * local_batch_points))
                        if chunk_count <= 0:
                            continue
                        read_bytes = chunk_count * source_item_bytes
                        reduce_read_batch_results = _acc_precompute_consume_batch_results(
                            memory_context,
                            buffer_name=source_buffer,
                            total_bytes=read_bytes,
                            cycle=pass_cursor,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            dir_prefix=f"fps_bv2_reduce_read_i{i}_p{pass_idx}_c{chunk_idx}",
                        )
                        _, reduce_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), pass_cursor, "FPS", "merge_sort",
                            sub_stage="temp_max_final_read",
                            circuit_module=source_buffer,
                            operation="fps_temp_max_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="temp_max_shared" if source_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                            total_bytes=read_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx},
                            memory_context=memory_context,
                            precomputed_spill_results=reduce_read_batch_results,
                        )
                        pass_cursor += reduce_read_dur
                        all_events.append(Event(
                            len(all_events), pass_cursor, "FPS", "merge_sort", "merge_sort_global_max", "merge_sort", "merge_sort_max",
                            float(getattr(ms, "power_max_mw", 310.0) or 310.0), merge_cycles, clock_period_ns=clock_ns,
                            metadata={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes, "resource_id": "merge_sort_shared", "exclusive": True},
                        ))
                        pass_cursor += merge_cycles
                        all_events.append(Event(
                            len(all_events), pass_cursor, "SRAM", "merge_sort_reg_file", "reg_file_read_1max", "sort_cache_sram", "reg_file_read",
                            0.0, reg_read_cycles, clock_period_ns=clock_ns,
                            metadata={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                        ))
                        pass_cursor += reg_read_cycles
                        reduce_write_batch_results = _acc_precompute_fill_batch_results(
                            memory_context,
                            buffer_name=dest_buffer,
                            total_bytes=temp_entry_bytes,
                            cycle=pass_cursor,
                            hw_config=hw_config,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            dir_prefix=f"fps_bv2_reduce_write_i{i}_p{pass_idx}_c{chunk_idx}",
                        )
                        _, reduce_write_dur, _ = _acc_append_buffer_fill_events(
                            all_events, len(all_events), pass_cursor, "FPS", "merge_sort",
                            sub_stage="temp_max_final_write",
                            circuit_module=dest_buffer,
                            operation="fps_temp_max_final_write",
                            buffer_name=dest_buffer,
                            phase="final_candidate_write",
                            resource_id="temp_max_shared" if dest_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                            total_bytes=temp_entry_bytes,
                            hw_config=hw_config,
                            is_read=False,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx},
                            memory_context=memory_context,
                            precomputed_spill_results=reduce_write_batch_results,
                        )
                        pass_cursor += reduce_write_dur
                        produced += 1
                    source_buffer = dest_buffer
                    current_candidates = produced
                if source_buffer != "temp_max_distance_sram":
                    pass_copy_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="temp_max_distance_sram",
                        total_bytes=temp_entry_bytes,
                        cycle=pass_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        dir_prefix=f"fps_bv2_pass_copy_i{i}_p{pass_idx}",
                    )
                    _, pass_copy_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), pass_cursor, "FPS", "merge_sort",
                        sub_stage="temp_max_pass_write",
                        circuit_module="temp_max_distance_sram",
                        operation="fps_temp_max_pass_write",
                        buffer_name="temp_max_distance_sram",
                        phase="pass_candidate_write",
                        resource_id="temp_max_shared",
                        total_bytes=temp_entry_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes},
                        memory_context=memory_context,
                        precomputed_spill_results=pass_copy_batch_results,
                    )
                    pass_cursor += pass_copy_dur
                pass_ready_cycles.append(pass_cursor)

            final_cursor = max(pass_ready_cycles) if pass_ready_cycles else current_cycle
            source_buffer = "temp_max_distance_sram"
            current_candidates = len(pass_ready_cycles)
            while current_candidates > 1:
                chunks = max(1, int(math.ceil(current_candidates / float(local_batch_points))))
                dest_buffer = "sort_cache_sram" if source_buffer == "temp_max_distance_sram" else "temp_max_distance_sram"
                produced = 0
                for chunk_idx in range(chunks):
                    chunk_count = min(local_batch_points, max(0, current_candidates - chunk_idx * local_batch_points))
                    if chunk_count <= 0:
                        continue
                    read_bytes = chunk_count * temp_entry_bytes
                    final_read_batch_results = _acc_precompute_consume_batch_results(
                        memory_context,
                        buffer_name=source_buffer,
                        total_bytes=read_bytes,
                        cycle=final_cursor,
                        hw_config=hw_config,
                        resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                        dir_prefix=f"fps_bv2_final_read_i{i}_c{chunk_idx}",
                    )
                    _, final_read_dur = _acc_append_buffer_consume_events(
                        all_events, len(all_events), final_cursor, "FPS", "merge_sort",
                        sub_stage="temp_max_final_read",
                        circuit_module=source_buffer,
                        operation="fps_temp_max_final_read",
                        buffer_name=source_buffer,
                        phase="final_sort_read",
                        resource_id="temp_max_shared" if source_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                        total_bytes=read_bytes,
                        hw_config=hw_config,
                        resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                        extra_meta={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx},
                        memory_context=memory_context,
                        precomputed_spill_results=final_read_batch_results,
                    )
                    final_cursor += final_read_dur
                    all_events.append(Event(
                        len(all_events), final_cursor, "FPS", "merge_sort", "merge_sort_global_max", "merge_sort", "merge_sort_max",
                        float(getattr(ms, "power_max_mw", 310.0) or 310.0), merge_cycles, clock_period_ns=clock_ns,
                        metadata={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes, "resource_id": "merge_sort_shared", "exclusive": True},
                    ))
                    final_cursor += merge_cycles
                    all_events.append(Event(
                        len(all_events), final_cursor, "SRAM", "merge_sort_reg_file", "reg_file_read_1max", "sort_cache_sram", "reg_file_read",
                        0.0, reg_read_cycles, clock_period_ns=clock_ns,
                        metadata={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                    ))
                    final_cursor += reg_read_cycles
                    final_write_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name=dest_buffer,
                        total_bytes=temp_entry_bytes,
                        cycle=final_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                        dir_prefix=f"fps_bv2_final_write_i{i}_c{chunk_idx}",
                    )
                    _, final_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), final_cursor, "FPS", "merge_sort",
                        sub_stage="temp_max_final_write",
                        circuit_module=dest_buffer,
                        operation="fps_temp_max_final_write",
                        buffer_name=dest_buffer,
                        phase="final_candidate_write",
                        resource_id="temp_max_shared" if dest_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                        total_bytes=temp_entry_bytes,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                        extra_meta={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx},
                        memory_context=memory_context,
                        precomputed_spill_results=final_write_batch_results,
                    )
                    final_cursor += final_write_dur
                    produced += 1
                source_buffer = dest_buffer
                current_candidates = produced
            current_cycle = max(current_cycle, final_cursor)
            continue

        # Producer-consumer: read accumulated batch distances from dist_sram_buffer
        # before feeding them into the min_distance_table update.
        dist_read_bytes = int(input_points) * 4
        _, dist_read_dur = _acc_append_buffer_consume_events(
            all_events, len(all_events), current_cycle, "FPS", "min_distance_update",
            sub_stage="dist_output_read",
            circuit_module="dist_sram_buffer",
            operation="dist_buffer_read_for_mdt",
            buffer_name="dist_sram_buffer",
            phase="read",
            resource_id="dist_sram_shared",
            total_bytes=dist_read_bytes,
            hw_config=hw_config,
            resident_bytes=min(dist_read_bytes, _acc_buffer_capacity_bytes("dist_sram_buffer", hw_config)),
            preferred_chunk_bytes=max(1, int(getattr(hw_config, "dist_sram_buffer", None).size_bytes if getattr(hw_config, "dist_sram_buffer", None) is not None else 4096)),
            extra_meta={"iteration": i, "bytes": dist_read_bytes, "kind": "dist_to_mdt", "reset_scope": "fps_iteration"},
            memory_context=context.get("memory_runtime_context") if context else None,
        )
        current_cycle += dist_read_dur

        mdt_read_batch_results = None
        if i != 0:
            if memory_context is not None:
                read_reqs = _acc_plan_consume_spill_requests(
                    buffer_name="fps_min_distance_table_sram",
                    total_bytes=int(input_points) * _fps_mdt_entry_bytes(),
                    cycle=current_cycle,
                    hw_config=hw_config,
                    resident_bytes=min(int(input_points) * _fps_mdt_entry_bytes(), _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)),
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                )
                if read_reqs:
                    mdt_read_batch_results = memory_context.prime_explicit_batch_transfers(
                        read_reqs,
                        dir_prefix=f"fps_base_mdt_read_i{i}",
                    )
            _, mdt_read_dur = _acc_append_buffer_consume_events(
                all_events, len(all_events), current_cycle, "FPS", "min_distance_update",
                sub_stage="min_distance_table_read",
                circuit_module="fps_min_distance_table_sram",
                operation="fps_min_distance_table_read",
                buffer_name="fps_min_distance_table_sram",
                phase="read",
                resource_id="fps_min_table_shared",
                total_bytes=int(input_points) * _fps_mdt_entry_bytes(),
                hw_config=hw_config,
                resident_bytes=min(int(input_points) * _fps_mdt_entry_bytes(), _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)),
                preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                extra_meta={"iteration": i, "reset_scope": "fps_iteration"},
                memory_context=memory_context,
                precomputed_spill_results=mdt_read_batch_results,
            )
            current_cycle += mdt_read_dur
        if i != 0:
            min_table_ev = Event(
                len(all_events), current_cycle, "FPS", "min_distance_update", "table_update", "min_distance_table", "min_distance_compare",
                mdt_power, 1, clock_ns,
                metadata={"iteration": i, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "update", "bytes": int(input_points) * _fps_mdt_entry_bytes(), "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
            )
            all_events.append(min_table_ev)
            current_cycle += 1
        mdt_write_batch_results = None
        if memory_context is not None:
            write_reqs = _acc_plan_fill_spill_requests(
                buffer_name="fps_min_distance_table_sram",
                total_bytes=int(input_points) * _fps_mdt_entry_bytes(),
                cycle=current_cycle,
                hw_config=hw_config,
                resident_bytes_before=0,
                preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                is_read=False,
            )
            if write_reqs:
                mdt_write_batch_results = memory_context.prime_explicit_batch_transfers(
                    write_reqs,
                    dir_prefix=f"fps_base_mdt_write_i{i}",
                )
        _, mdt_write_dur, _ = _acc_append_buffer_fill_events(
            all_events, len(all_events), current_cycle, "FPS", "min_distance_update",
            sub_stage="min_distance_table_write",
            circuit_module="fps_min_distance_table_sram",
            operation="fps_min_distance_table_write",
            buffer_name="fps_min_distance_table_sram",
            phase="update",
            resource_id="fps_min_table_shared",
            total_bytes=int(input_points) * _fps_mdt_entry_bytes(),
            hw_config=hw_config,
            is_read=False,
            preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
            extra_meta={"iteration": i, "reset_scope": "fps_iteration"},
            memory_context=memory_context,
            precomputed_spill_results=mdt_write_batch_results,
        )
        current_cycle += mdt_write_dur
        if str(getattr(hw_config, "fps_engine_mode", "baseline")).lower() == "fuse":
            # fuse：coarse+fine 专用路径（FPS MAX）
            fuse_evs = fps_op.generate_single_fuse_events()
            for fe in fuse_evs:
                fe.cycle += current_cycle
            all_events.extend(fuse_evs)
            current_cycle += sum(e.duration for e in fuse_evs)
        else:
            candidate_points = max(1, int(input_points) - int(i))
            sorter_depth = _sorter_depth_points(hw_config)
            num_passes = max(1, int(math.ceil(candidate_points / float(sorter_depth))))
            temp_entry_bytes = _distance_index_entry_bytes(candidate_points)
            for pass_idx in range(num_passes):
                pass_points = min(sorter_depth, max(0, candidate_points - pass_idx * sorter_depth))
                if pass_points <= 0:
                    continue
                pass_read_bytes = int(pass_points) * 4
                _, sort_read_dur = _acc_append_buffer_consume_events(
                    all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                    sub_stage="min_distance_table_local_sort_read",
                    circuit_module="fps_min_distance_table_sram",
                    operation="fps_min_distance_table_local_sort_read",
                    buffer_name="fps_min_distance_table_sram",
                    phase="local_sort_read",
                    resource_id="fps_min_table_shared",
                    total_bytes=pass_read_bytes,
                    hw_config=hw_config,
                    resident_bytes=min(pass_read_bytes, _acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config)),
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("fps_min_distance_table_sram", hw_config),
                    extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes, "reset_scope": "fps_iteration"},
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                current_cycle += sort_read_dur
                one_bitonic = fps_op.generate_single_bitonic_event()
                one_bitonic.cycle = current_cycle
                meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                meta.update({
                    "iteration": i,
                    "pass_idx": pass_idx,
                    "num_passes": num_passes,
                    "buffer_name": "fps_min_distance_table_sram",
                    "buffer_phase": "local_sort_read",
                    "bytes": pass_read_bytes,
                })
                one_bitonic.metadata = meta
                all_events.append(one_bitonic)
                current_cycle += one_bitonic.duration
                _, temp_write_dur, _ = _acc_append_buffer_fill_events(
                    all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                    sub_stage="temp_max_local_write",
                    circuit_module="temp_max_distance_sram",
                    operation="fps_temp_max_local_write",
                    buffer_name="temp_max_distance_sram",
                    phase="local_candidate_write",
                    resource_id="temp_max_shared",
                    total_bytes=temp_entry_bytes,
                    hw_config=hw_config,
                    is_read=False,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                    extra_meta={"iteration": i, "pass_idx": pass_idx, "num_passes": num_passes},
                    memory_context=memory_context,
                    precomputed_spill_results=_acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="temp_max_distance_sram",
                        total_bytes=temp_entry_bytes,
                        cycle=current_cycle,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        dir_prefix=f"fps_base_temp_write_i{i}_p{pass_idx}",
                    ),
                )
                current_cycle += temp_write_dur
            final_candidates = num_passes
            if final_candidates <= sorter_depth:
                final_candidate_bytes = final_candidates * temp_entry_bytes
                _, final_read_dur = _acc_append_buffer_consume_events(
                    all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                    sub_stage="temp_max_final_read",
                    circuit_module="temp_max_distance_sram",
                    operation="fps_temp_max_final_read",
                    buffer_name="temp_max_distance_sram",
                    phase="final_sort_read",
                    resource_id="temp_max_shared",
                    total_bytes=final_candidate_bytes,
                    hw_config=hw_config,
                    resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config)),
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                    extra_meta={"iteration": i, "num_passes": num_passes},
                    memory_context=memory_context,
                    precomputed_spill_results=_acc_precompute_consume_batch_results(
                        memory_context,
                        buffer_name="temp_max_distance_sram",
                        total_bytes=final_candidate_bytes,
                        cycle=current_cycle,
                        hw_config=hw_config,
                        resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("temp_max_distance_sram", hw_config),
                        dir_prefix=f"fps_base_final_read_i{i}",
                    ),
                )
                current_cycle += final_read_dur
                one_bitonic = fps_op.generate_single_bitonic_event()
                one_bitonic.cycle = current_cycle
                meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                meta.update({"iteration": i, "buffer_name": "temp_max_distance_sram", "buffer_phase": "final_sort_read", "bytes": final_candidate_bytes})
                one_bitonic.metadata = meta
                all_events.append(one_bitonic)
                current_cycle += one_bitonic.duration
            else:
                source_buffer = "temp_max_distance_sram"
                current_candidates = final_candidates
                while current_candidates > 1:
                    reduce_passes = max(1, int(math.ceil(current_candidates / float(sorter_depth))))
                    if current_candidates <= sorter_depth:
                        final_candidate_bytes = current_candidates * temp_entry_bytes
                        _, final_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                            sub_stage="temp_max_final_read",
                            circuit_module=source_buffer,
                            operation="fps_temp_max_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="temp_max_shared" if source_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                            total_bytes=final_candidate_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"iteration": i, "num_passes": num_passes, "reduce_level": current_candidates},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_consume_batch_results(
                                memory_context,
                                buffer_name=source_buffer,
                                total_bytes=final_candidate_bytes,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                                dir_prefix=f"fps_base_reduce_final_read_i{i}_lvl{current_candidates}",
                            ),
                        )
                        current_cycle += final_read_dur
                        one_bitonic = fps_op.generate_single_bitonic_event()
                        one_bitonic.cycle = current_cycle
                        meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                        meta.update({"iteration": i, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": final_candidate_bytes})
                        one_bitonic.metadata = meta
                        all_events.append(one_bitonic)
                        current_cycle += one_bitonic.duration
                        break
                    dest_buffer = "sort_cache_sram" if source_buffer == "temp_max_distance_sram" else "temp_max_distance_sram"
                    produced = 0
                    for chunk_idx in range(reduce_passes):
                        chunk_count = min(sorter_depth, max(0, current_candidates - chunk_idx * sorter_depth))
                        if chunk_count <= 0:
                            continue
                        read_bytes = chunk_count * temp_entry_bytes
                        _, reduce_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                            sub_stage="temp_max_final_read",
                            circuit_module=source_buffer,
                            operation="fps_temp_max_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="temp_max_shared" if source_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                            total_bytes=read_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx, "candidate_count": chunk_count},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_consume_batch_results(
                                memory_context,
                                buffer_name=source_buffer,
                                total_bytes=read_bytes,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                                dir_prefix=f"fps_base_reduce_read_i{i}_c{chunk_idx}",
                            ),
                        )
                        current_cycle += reduce_read_dur
                        one_bitonic = fps_op.generate_single_bitonic_event()
                        one_bitonic.cycle = current_cycle
                        meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                        meta.update({"iteration": i, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes})
                        one_bitonic.metadata = meta
                        all_events.append(one_bitonic)
                        current_cycle += one_bitonic.duration
                        _, reduce_write_dur, _ = _acc_append_buffer_fill_events(
                            all_events, len(all_events), current_cycle, "FPS", "bitonic_sort",
                            sub_stage="temp_max_final_write",
                            circuit_module=dest_buffer,
                            operation="fps_temp_max_final_write",
                            buffer_name=dest_buffer,
                            phase="final_candidate_write",
                            resource_id="temp_max_shared" if dest_buffer == "temp_max_distance_sram" else "sort_cache_shared",
                            total_bytes=temp_entry_bytes,
                            hw_config=hw_config,
                            is_read=False,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            extra_meta={"iteration": i, "num_passes": num_passes, "chunk_idx": chunk_idx},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_fill_batch_results(
                                memory_context,
                                buffer_name=dest_buffer,
                                total_bytes=temp_entry_bytes,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                                dir_prefix=f"fps_base_reduce_write_i{i}_c{chunk_idx}",
                            ),
                        )
                        current_cycle += reduce_write_dur
                        produced += 1
                    source_buffer = dest_buffer
                    current_candidates = produced
    return all_events


def _phase2_distance_cycles_per_query(num_keys: int, hw_config: 'HardwareConfig') -> int:
    """Phase2 单查询总周期（可含查询 DRAM 读取）。"""
    import math
    if getattr(hw_config, "is_baseline", False) and getattr(hw_config, "baseline_dist2_engine", None) is not None:
        eng = hw_config.baseline_dist2_engine
        batch_points = max(1, int(getattr(eng, "pe_count", getattr(eng, "c_dim", 64)) or 64))
        per_batch_cycles = int(getattr(eng, "cycles", 0) or 0)
        if per_batch_cycles <= 0:
            per_distance_cycles = max(1, int(getattr(eng, "per_distance_cycles", 7) or 7))
            per_batch_cycles = batch_points - 1 + per_distance_cycles
        num_batches = max(1, int(math.ceil(num_keys / float(batch_points))))
        point_bytes = int(math.ceil(3 * 12 / 8.0))
        dist_cycles = 0
        batch_requests: List[Dict[str, Any]] = []
        for batch_idx in range(num_batches):
            batch_m = min(batch_points, num_keys - batch_idx * batch_points)
            batch_bytes = max(1, int(batch_m) * point_bytes)
            batch_requests.append({"num_bytes": batch_bytes, "is_read": True, "start_cycle": dist_cycles})
        batch_results = query_dram_batch_cycles_energy(
            batch_requests,
            hw_config,
            float(getattr(hw_config, "clock_period", 1.0)),
        )
        for batch_idx in range(num_batches):
            batch_m = min(batch_points, num_keys - batch_idx * batch_points)
            batch_bytes = max(1, int(batch_m) * point_bytes)
            dram_cycles, _ = batch_results[batch_idx]
            sram_wr_cycles, _ = query_sram_cycles_energy(
                batch_bytes,
                is_read=False,
                hw_config=hw_config,
                clock_period_ns=float(getattr(hw_config, "clock_period", 1.0)),
                buffer_name="xyz_sram_buffer",
            )
            sram_rd_cycles, _ = query_sram_cycles_energy(
                batch_bytes,
                is_read=True,
                hw_config=hw_config,
                clock_period_ns=float(getattr(hw_config, "clock_period", 1.0)),
                buffer_name="xyz_sram_buffer",
            )
            dist_cycles += dram_cycles + sram_wr_cycles + sram_rd_cycles + per_batch_cycles
    else:
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
    query_result = query_dram_batch_cycles_energy(
        [{"num_bytes": query_bytes, "is_read": True, "start_cycle": dist_cycles}],
        hw_config,
        float(getattr(hw_config, "clock_period", 1.0)),
    )
    dram_cycles, _ = query_result[0]
    return dist_cycles + dram_cycles


def _extract_per_query_end_cycles(events: List[Event], num_queries: int) -> List[int]:
    """从 KNN 事件列表中提取每个 query 距离计算完成的周期。"""
    per_q: Dict[int, int] = {}
    for ev in events:
        q = ev.metadata.get("query", -1) if ev.metadata else -1
        if q >= 0:
            end = ev.cycle + ev.duration
            if q not in per_q or end > per_q[q]:
                per_q[q] = end
    return [per_q.get(q, 0) for q in range(num_queries)]


def _gen_knn(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    KNN：当 distance_in_phase2 时，先插入 Phase2 距离计算，再插入 KNN 引擎 bitonic top-k。
    Baseline 下与 FPS 串行（共享 bitonic 比较器）；fuse 模式可复用距离，待扩展。

    支持两个新标志:
    - distance_reuse: 跳过距离计算，从 DRAM 读已存储矩阵子集 + sort top-k
    - dist_matrix_dram_write: 计算完成后将距离矩阵写入 DRAM 供后续复用
    """
    from .knn import KNNOperator, KNNParams

    params = op_ref.params
    input_points = int(params.get('input_points', 1024))
    query_points = int(params.get('query_points', 512))
    distance_reuse = bool(params.get('distance_reuse', False))
    dist_matrix_dram_write = bool(params.get('dist_matrix_dram_write', False))
    is_transpose_read = bool(params.get('is_transpose_read', False))
    global_dist_cols = int(params.get('global_dist_cols', 0))
    transpose_store = bool(getattr(hw_config, "distance_matrix_transpose_store", False))
    if distance_reuse and not is_transpose_read:
        is_transpose_read = query_points > input_points
    knn_params = KNNParams(
        input_points=input_points,
        query_points=query_points,
        k=params.get('k', 16),
        distance_reuse=distance_reuse,
        dist_matrix_dram_write=dist_matrix_dram_write,
        is_transpose_read=is_transpose_read,
        global_dist_cols=global_dist_cols,
    )
    knn_op = KNNOperator(params=knn_params, hw_config=hw_config)

    ns_mode = str(getattr(hw_config, "ns_simulation_mode", "event")).strip().lower()
    # NSEngineCycleAccurate models DRAM/SRAM distance-matrix streaming + sort pipelines for
    # SRAM-CIM style KNN — not the PointCIM RRAM VMM distance micro-sequence.  For
    # use_rram_dist (PointCIM), keep the acc fuse-path event generator below.
    if ns_mode == "cycle_accurate" and not getattr(hw_config, "use_rram_dist", False):
        from .ns_engine_cycle import NSEngineCycleAccurate
        engine = NSEngineCycleAccurate(hw_config, knn_params)
        ca_events = engine.run()
        if dist_matrix_dram_write:
            use_streaming = getattr(hw_config, "distance_matrix_reuse", False)
            end_cycle = max((e.cycle + e.duration for e in ca_events), default=0)
            end_eid = max((e.event_id for e in ca_events), default=-1) + 1
            if use_streaming:
                per_q_cycles = _extract_per_query_end_cycles(ca_events, query_points)
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=0, start_event_id=end_eid,
                    streaming=True, per_query_ready_cycles=per_q_cycles,
                    transpose_store=transpose_store,
                )
            else:
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=end_cycle, start_event_id=end_eid,
                    transpose_store=transpose_store,
                )
            ca_events.extend(wb_events)
        return ca_events

    if distance_reuse:
        return knn_op.generate_distance_reuse_events(transpose_store=transpose_store)

    if not getattr(hw_config, "distance_in_phase2", False):
        events = knn_op.generate_events()
        if dist_matrix_dram_write:
            use_streaming = getattr(hw_config, "distance_matrix_reuse", False)
            end_cycle = max((e.cycle + e.duration for e in events), default=0)
            end_eid = max((e.event_id for e in events), default=-1) + 1
            if use_streaming:
                per_q_cycles = _extract_per_query_end_cycles(events, query_points)
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=0, start_event_id=end_eid,
                    streaming=True, per_query_ready_cycles=per_q_cycles,
                    transpose_store=transpose_store,
                )
            else:
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=end_cycle, start_event_id=end_eid,
                    transpose_store=transpose_store,
                )
            events.extend(wb_events)
        return events

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
        configured_refine_rounds = max(
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
        bytes_per_candidate = max(1, int(params.get('bytes_per_candidate', _distance_index_entry_bytes(input_points)) or _distance_index_entry_bytes(input_points)))
        candidates_per_block = max(
            1,
            int(
                params.get(
                    'knn_candidates_per_block',
                    getattr(hw_config, 'knn_candidates_per_block', 4),
                )
                or 4
            ),
        )
        thresh_cfg = getattr(getattr(hw_config, 'sorting_circuit', None), 'threshold_gen', None)
        thresh_gen_power = float(getattr(thresh_cfg, 'power', 3.66) or 3.66)
        thresh_gen_cycles = max(1, int(getattr(thresh_cfg, 'cycles', 1) or 1))
        fine_filter_floor_candidates = 32
        coarse_refine_capacity = max(1, block_size)
        fine_filter_capacity = max(k_value, fine_filter_floor_candidates)
        dist_tag_scope = f"knn_dist_{input_points}_{query_points}_{k_value}_eid{eid}"

        def _next_knn_refine_candidate_total(current_total: int) -> int:
            if current_total <= fine_filter_capacity:
                return current_total
            group_count = max(1, int(math.ceil(current_total / float(coarse_refine_capacity))))
            reduced_total = group_count * min(candidates_per_block, coarse_refine_capacity)
            if reduced_total >= current_total:
                reduced_total = fine_filter_capacity
            return max(fine_filter_capacity, reduced_total)

        def _required_knn_refine_rounds(initial_total: int) -> int:
            rounds = 0
            current_total = max(1, initial_total)
            while current_total > fine_filter_capacity:
                next_total = _next_knn_refine_candidate_total(current_total)
                rounds += 1
                if next_total >= current_total:
                    break
                current_total = next_total
            return rounds

        shared_frontend = bool(context and context.get("acc_shared_distance_frontend_ready"))
        for q in range(query_points):
            aggregate_candidate_bytes = 0
            base_cycle = current_cycle
            if not shared_frontend:
                evs_d, eid, dist_cycles, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags = _gen_acc_distance_query_events(
                    eid,
                    base_cycle,
                    "KNN",
                    input_points,
                    hw_config,
                    query_index=q,
                    include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
                    memory_context=context.get("memory_runtime_context") if context else None,
                    startup_once=True,
                    write_dist_buffer=False,
                    tag_scope=dist_tag_scope,
                    key_already_resident=bool(context and context.get(ACC_CIM_KEY_PRELOADED, False)),
                )
                all_events.extend(evs_d)
                current_cycle += dist_cycles
                if context is not None:
                    context.setdefault("acc_block_ready_cycles_by_query", {})[q] = block_ready_cycles
                    context.setdefault("acc_block_ready_tags_by_query", {})[q] = block_ready_tags
                    context.setdefault("acc_block_prefetch_tags_by_query", {})[q] = block_prefetch_tags
            block_ready_cycles = []
            block_ready_tags = []
            if context is not None:
                block_ready_cycles = list((context.get("acc_block_ready_cycles_by_query", {}) or {}).get(q, []) or [])
                block_ready_tags = list((context.get("acc_block_ready_tags_by_query", {}) or {}).get(q, []) or [])

            query_tail_start = current_cycle
            for blk in range(num_blocks):
                blk_ready = block_ready_cycles[blk] if blk < len(block_ready_cycles) else (base_cycle + blk * block_issue_cycles + 1)
                blk_points = min(block_size, input_points - blk * block_size)
                candidate_count = min(candidates_per_block, blk_points)
                candidate_bytes = candidate_count * bytes_per_candidate
                coarse_ev = Event(
                    eid, blk_ready, "KNN", "dynamic_threshold", "coarse_filter", "coarse_filter", "knn_coarse_filter",
                    coarse_power, coarse_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "query_index": q,
                        "block_idx": blk,
                        "num_blocks": num_blocks,
                        "buffer_name": "dist_sram_buffer",
                        "buffer_phase": "coarse_filter_read",
                        "bytes": blk_points * 4,
                        "resource_id": "comparator_shared",
                        "resource_priority": 0,
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                        "depends_on_tag": block_ready_tags[blk] if blk < len(block_ready_tags) else None,
                        "dependency_tag": f"knn_coarse_q{q}_b{blk}",
                    },
                )
                coarse_ev.overlapped = True
                all_events.append(coarse_ev)
                eid += 1

                topk_write_start = blk_ready + coarse_cycles
                eid, topk_write_dur, _ = _acc_append_buffer_fill_events(
                    all_events, eid, topk_write_start, "KNN", "dynamic_threshold",
                    sub_stage="topk_write",
                    circuit_module="topk_sram_buffer",
                    operation="topk_candidate_write",
                    buffer_name="topk_sram_buffer",
                    phase="write",
                    resource_id="topk_sram_shared",
                    total_bytes=candidate_bytes,
                    hw_config=hw_config,
                    is_read=False,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                    extra_meta={
                        "query": q,
                        "query_index": q,
                        "query_index": q,
                        "block_idx": blk,
                        "num_blocks": num_blocks,
                        "candidate_count": candidate_count,
                        "phase": "knn_dynamic_threshold",
                        "depends_on_tag": f"knn_coarse_q{q}_b{blk}",
                        "dependency_tag": f"knn_topk_write_q{q}_b{blk}",
                    },
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                aggregate_candidate_bytes += candidate_bytes
                query_tail_start = max(query_tail_start, topk_write_start + topk_write_dur)

            initial_candidate_total = max(1, aggregate_candidate_bytes // max(1, bytes_per_candidate))
            fine_candidate_total = min(initial_candidate_total, fine_filter_capacity)
            if initial_candidate_total <= fine_filter_capacity:
                refine_rounds = 0
            else:
                refine_rounds = max(
                    configured_refine_rounds,
                    _required_knn_refine_rounds(initial_candidate_total),
                )
            topk_onchip_bytes, topk_spill_bytes = _acc_spill_bytes("topk_sram_buffer", aggregate_candidate_bytes, hw_config)
            current_cycle = max(current_cycle, query_tail_start)
            block_write_tags = [f"knn_topk_write_q{q}_b{blk}" for blk in range(num_blocks)]
            eid, spill_dur = _acc_append_spill_events(
                all_events, eid, current_cycle, "KNN", "dynamic_threshold", "topk_sram_buffer", topk_spill_bytes,
                clock_ns, "knn_dynamic_threshold", "write", hw_config,
                context.get("memory_runtime_context") if context else None,
            )
            current_cycle += spill_dur
            refine_cursor = current_cycle
            current_candidate_total = initial_candidate_total
            current_onchip_bytes = topk_onchip_bytes
            current_spill_bytes = topk_spill_bytes
            prev_knn_write_tags = list(block_write_tags)
            for refine_idx in range(refine_rounds):
                round_read_phase = "aggregate_read" if refine_idx == 0 else "refine_read"
                round_read_op = "topk_candidate_aggregate_read" if refine_idx == 0 else "topk_candidate_refine_read"
                eid, topk_read_dur = _acc_append_buffer_consume_events(
                    all_events, eid, refine_cursor, "KNN", "dynamic_threshold",
                    sub_stage=f"topk_round{refine_idx + 1}_read",
                    circuit_module="topk_sram_buffer",
                    operation=round_read_op,
                    buffer_name="topk_sram_buffer",
                    phase=round_read_phase,
                    resource_id="topk_sram_shared",
                    total_bytes=current_candidate_total * bytes_per_candidate,
                    hw_config=hw_config,
                    resident_bytes=current_onchip_bytes,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                    extra_meta={
                        "query": q,
                        "query_index": q,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                        "candidate_count": current_candidate_total,
                        "depends_on_tags": prev_knn_write_tags,
                        "dependency_tag": f"knn_round_read_q{q}_r{refine_idx + 1}",
                    },
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                refine_cursor += topk_read_dur
                remaining_rounds = refine_rounds - refine_idx
                next_candidate_total = _next_knn_refine_candidate_total(current_candidate_total)
                if remaining_rounds == 1:
                    next_candidate_total = fine_candidate_total
                next_total_bytes = next_candidate_total * bytes_per_candidate
                next_onchip_bytes, next_spill_bytes = _acc_spill_bytes(
                    "topk_sram_buffer",
                    next_total_bytes,
                    hw_config,
                )
                thresh_ev = Event(
                    eid, refine_cursor, "KNN", "dynamic_threshold", "threshold_gen", "threshold_gen", "knn_threshold_update",
                    thresh_gen_power, thresh_gen_cycles, clock_period_ns=clock_ns,
                    metadata={
                        "query": q,
                        "buffer_name": "topk_sram_buffer",
                        "buffer_phase": "threshold_gen",
                        "bytes": current_onchip_bytes,
                        "resource_id": "threshold_gen_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                        "candidate_count": current_candidate_total,
                        "next_candidate_count": next_candidate_total,
                        "depends_on_tag": f"knn_round_read_q{q}_r{refine_idx + 1}",
                        "dependency_tag": f"knn_thresh_q{q}_r{refine_idx + 1}",
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
                        "bytes": current_onchip_bytes,
                        "resource_id": "comparator_shared",
                        "exclusive": True,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                        "candidate_count": current_candidate_total,
                        "next_candidate_count": next_candidate_total,
                        "depends_on_tag": f"knn_thresh_q{q}_r{refine_idx + 1}",
                        "dependency_tag": f"knn_refine_q{q}_r{refine_idx + 1}",
                    },
                )
                all_events.append(refine_ev)
                eid += 1
                refine_cursor += thresh_gen_cycles + coarse_cycles
                eid, topk_write_dur, _ = _acc_append_buffer_fill_events(
                    all_events, eid, refine_cursor, "KNN", "dynamic_threshold",
                    sub_stage=f"topk_round{refine_idx + 1}_write",
                    circuit_module="topk_sram_buffer",
                    operation="topk_candidate_refine_write",
                    buffer_name="topk_sram_buffer",
                    phase="refine_write",
                    resource_id="topk_sram_shared",
                    total_bytes=next_total_bytes,
                    hw_config=hw_config,
                    is_read=False,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                    extra_meta={
                        "query": q,
                        "query_index": q,
                        "phase": "knn_dynamic_threshold",
                        "refine_round": refine_idx + 1,
                        "refine_rounds": refine_rounds,
                        "candidate_count": current_candidate_total,
                        "next_candidate_count": next_candidate_total,
                        "depends_on_tag": f"knn_refine_q{q}_r{refine_idx + 1}",
                        "dependency_tag": f"knn_refine_write_q{q}_r{refine_idx + 1}",
                    },
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                refine_cursor += topk_write_dur
                eid, topk_round_spill_dur = _acc_append_spill_events(
                    all_events, eid, refine_cursor, "KNN", "dynamic_threshold", "topk_sram_buffer", next_spill_bytes,
                    clock_ns, "knn_dynamic_threshold", "write", hw_config,
                    context.get("memory_runtime_context") if context else None,
                )
                refine_cursor += topk_round_spill_dur
                current_candidate_total = next_candidate_total
                current_onchip_bytes = next_onchip_bytes
                current_spill_bytes = next_spill_bytes
                prev_knn_write_tags = [f"knn_refine_write_q{q}_r{refine_idx + 1}"]

            final_candidate_total = current_candidate_total
            final_total_bytes = final_candidate_total * bytes_per_candidate
            final_onchip_bytes, final_spill_bytes = _acc_spill_bytes(
                "topk_sram_buffer",
                final_total_bytes,
                hw_config,
            )
            eid, topk_fine_dur = _acc_append_buffer_consume_events(
                all_events, eid, refine_cursor, "KNN", "dynamic_threshold",
                sub_stage="topk_fine_read",
                circuit_module="topk_sram_buffer",
                operation="topk_candidate_fine_read",
                buffer_name="topk_sram_buffer",
                phase="fine_filter_read",
                resource_id="topk_sram_shared",
                total_bytes=final_total_bytes,
                hw_config=hw_config,
                resident_bytes=final_onchip_bytes,
                preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                extra_meta={
                    "query": q,
                    "phase": "knn_dynamic_threshold",
                    "refine_rounds": refine_rounds,
                    "num_blocks": num_blocks,
                    "candidate_count": final_candidate_total,
                    "depends_on_tags": prev_knn_write_tags,
                    "dependency_tag": f"knn_fine_read_q{q}",
                },
                memory_context=context.get("memory_runtime_context") if context else None,
            )
            refine_cursor += topk_fine_dur
            fine_ev = Event(
                eid, refine_cursor, "KNN", "dynamic_threshold", "fine_filter", "fine_filter", "knn_fine_filter_topk",
                fine_power, fine_cycles, clock_period_ns=clock_ns,
                metadata={
                    "query": q,
                    "buffer_name": "topk_sram_buffer",
                    "buffer_phase": "fine_filter_read",
                    "bytes": final_onchip_bytes,
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                    "phase": "knn_dynamic_threshold",
                    "refine_rounds": refine_rounds,
                    "num_blocks": num_blocks,
                    "candidate_count": final_candidate_total,
                    "depends_on_tag": f"knn_fine_read_q{q}",
                },
            )
            all_events.append(fine_ev)
            eid += 1
        if dist_matrix_dram_write:
            use_streaming = getattr(hw_config, "distance_matrix_reuse", False)
            end_cycle = max((e.cycle + e.duration for e in all_events), default=0)
            end_eid = max((e.event_id for e in all_events), default=-1) + 1
            if use_streaming:
                per_q_cycles = _extract_per_query_end_cycles(all_events, query_points)
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=0, start_event_id=end_eid,
                    streaming=True, per_query_ready_cycles=per_q_cycles,
                    transpose_store=transpose_store,
                )
            else:
                wb_events = knn_op.generate_dram_writeback_events(
                    start_cycle=end_cycle, start_event_id=end_eid,
                    transpose_store=transpose_store,
                )
            all_events.extend(wb_events)
        return all_events

    # Baseline Phase2 + KNN 引擎交错：每查询 [距离计算] -> [排序/筛选]
    clock_ns = getattr(hw_config, "clock_period", 1.0)
    knn_mode = str(getattr(hw_config, "knn_engine_mode", "baseline")).lower()
    use_fuse = knn_mode == "fuse"
    all_events = []
    current_cycle = 0
    memory_context = context.get("memory_runtime_context") if context else None
    for q in range(query_points):
        evs_d, _, current_cycle = _gen_baseline_phase2_distance_events(
            input_points,
            hw_config,
            "KNN",
            start_cycle=current_cycle,
            include_query_dram_read=getattr(hw_config, "distance_include_query_dram_read", True),
            metadata={"query": q, "query_index": q},
        )
        all_events.extend(evs_d)

        if getattr(hw_config, "is_baseline_v2", False):
            sorter_depth = _sorter_depth_points(hw_config)
            num_passes = max(1, int(math.ceil(input_points / float(sorter_depth))))
            bytes_per_candidate = _distance_index_entry_bytes(input_points)
            local_topk = max(32, int(getattr(knn_params, "k", 32) or 32))
            ms = getattr(hw_config.sorting_circuit, "merge_sort", None)
            reg = getattr(ms, "reg_file", None) if ms is not None else None
            merge_cycles = int(getattr(ms, "cycles_top32", 22) or 22) if ms is not None else 22
            reg_read_cycles = int(getattr(reg, "cycles_per_read", 1) or 1) if reg is not None else 1
            dist2_eng = getattr(hw_config, "baseline_dist2_engine", None)
            local_batch_points = max(1, int(getattr(dist2_eng, "pe_count", getattr(dist2_eng, "c_dim", 64)) or 64))
            batches_per_pass = max(1, int(math.ceil(sorter_depth / float(local_batch_points))))
            batch_ready = _group_ready_cycles(evs_d, "dist2_output_buffer_write", "batch")
            merge_cursor = 0
            pass_ready_cycles: List[int] = []
            pass_candidate_counts: List[int] = []

            for pass_idx in range(num_passes):
                pass_points = min(sorter_depth, max(0, input_points - pass_idx * sorter_depth))
                if pass_points <= 0:
                    continue
                pass_batches = max(1, int(math.ceil(pass_points / float(local_batch_points))))
                pass_batch_ready: List[int] = []
                pass_local_candidates = 0
                for local_batch_idx in range(pass_batches):
                    global_batch_idx = pass_idx * batches_per_pass + local_batch_idx
                    batch_points = min(local_batch_points, max(0, pass_points - local_batch_idx * local_batch_points))
                    if batch_points <= 0:
                        continue
                    batch_start = max(int(batch_ready.get(global_batch_idx, 0)), merge_cursor)
                    read_bytes = int(batch_points) * 4
                    _, sort_read_dur = _acc_append_buffer_consume_events(
                        all_events, len(all_events), batch_start, "KNN", "merge_sort",
                        sub_stage="local_reduce_read",
                        circuit_module="dist_sram_buffer",
                        operation="knn_merge_sort_reduce_read",
                        buffer_name="dist_sram_buffer",
                        phase="local_reduce_read",
                        resource_id="dist_sram_shared",
                        total_bytes=read_bytes,
                        hw_config=hw_config,
                        resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes("dist_sram_buffer", hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("dist_sram_buffer", hw_config),
                        extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx},
                        memory_context=context.get("memory_runtime_context") if context else None,
                    )
                    batch_cursor = batch_start + sort_read_dur
                    all_events.append(Event(
                        len(all_events), batch_cursor, "KNN", "merge_sort", "merge_sort_local_topk", "merge_sort", "merge_sort_top32",
                        float(getattr(ms, "power_top32_mw", 312.0) or 312.0), merge_cycles, clock_period_ns=clock_ns,
                        metadata={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "buffer_name": "dist_sram_buffer", "buffer_phase": "local_reduce_read", "bytes": read_bytes, "resource_id": "merge_sort_shared", "exclusive": True},
                    ))
                    batch_cursor += merge_cycles
                    out_count = min(local_topk, int(batch_points))
                    all_events.append(Event(
                        len(all_events), batch_cursor, "SRAM", "merge_sort_reg_file", "reg_file_read_32min", "sort_cache_sram", "reg_file_read",
                        0.0, reg_read_cycles, clock_period_ns=clock_ns,
                        metadata={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": out_count * 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                    ))
                    batch_cursor += reg_read_cycles
                    topk_write_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="topk_sram_buffer",
                        total_bytes=out_count * bytes_per_candidate,
                        cycle=batch_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        dir_prefix=f"knn_bv2_topk_write_q{q}_p{pass_idx}_b{global_batch_idx}",
                    )
                    _, topk_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), batch_cursor, "KNN", "merge_sort",
                        sub_stage="topk_local_write",
                        circuit_module="topk_sram_buffer",
                        operation="knn_topk_local_write",
                        buffer_name="topk_sram_buffer",
                        phase="local_candidate_write",
                        resource_id="topk_sram_shared",
                        total_bytes=out_count * bytes_per_candidate,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "batch": global_batch_idx, "local_batch_idx": local_batch_idx, "candidate_count": out_count},
                        memory_context=memory_context,
                        precomputed_spill_results=topk_write_batch_results,
                    )
                    batch_cursor += topk_write_dur
                    pass_batch_ready.append(batch_cursor)
                    pass_local_candidates += out_count
                    merge_cursor = batch_cursor

                pass_cursor = max(pass_batch_ready) if pass_batch_ready else merge_cursor
                source_buffer = "topk_sram_buffer"
                source_item_bytes = bytes_per_candidate
                current_candidates = pass_local_candidates
                while current_candidates > local_topk:
                    chunks = max(1, int(math.ceil(current_candidates / float(local_batch_points))))
                    dest_buffer = "sort_cache_sram" if source_buffer == "topk_sram_buffer" else "topk_sram_buffer"
                    produced = 0
                    for chunk_idx in range(chunks):
                        chunk_count = min(local_batch_points, max(0, current_candidates - chunk_idx * local_batch_points))
                        if chunk_count <= 0:
                            continue
                        read_bytes = chunk_count * source_item_bytes
                        reduce_read_batch_results = _acc_precompute_consume_batch_results(
                            memory_context,
                            buffer_name=source_buffer,
                            total_bytes=read_bytes,
                            cycle=pass_cursor,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            dir_prefix=f"knn_bv2_reduce_read_q{q}_p{pass_idx}_c{chunk_idx}",
                        )
                        _, reduce_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), pass_cursor, "KNN", "merge_sort",
                            sub_stage="topk_final_read",
                            circuit_module=source_buffer,
                            operation="knn_topk_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="topk_sram_shared" if source_buffer == "topk_sram_buffer" else "sort_cache_shared",
                            total_bytes=read_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx},
                            memory_context=memory_context,
                            precomputed_spill_results=reduce_read_batch_results,
                        )
                        pass_cursor += reduce_read_dur
                        all_events.append(Event(
                            len(all_events), pass_cursor, "KNN", "merge_sort", "merge_sort_global_topk", "merge_sort", "merge_sort_top32",
                            float(getattr(ms, "power_top32_mw", 312.0) or 312.0), merge_cycles, clock_period_ns=clock_ns,
                            metadata={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes, "resource_id": "merge_sort_shared", "exclusive": True},
                        ))
                        pass_cursor += merge_cycles
                        out_count = min(local_topk, chunk_count)
                        reduce_write_batch_results = _acc_precompute_fill_batch_results(
                            memory_context,
                            buffer_name=dest_buffer,
                            total_bytes=out_count * bytes_per_candidate,
                            cycle=pass_cursor,
                            hw_config=hw_config,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            dir_prefix=f"knn_bv2_reduce_write_q{q}_p{pass_idx}_c{chunk_idx}",
                        )
                        _, reduce_write_dur, _ = _acc_append_buffer_fill_events(
                            all_events, len(all_events), pass_cursor, "KNN", "merge_sort",
                            sub_stage="topk_final_write",
                            circuit_module=dest_buffer,
                            operation="knn_topk_final_write",
                            buffer_name=dest_buffer,
                            phase="final_candidate_write",
                            resource_id="topk_sram_shared" if dest_buffer == "topk_sram_buffer" else "sort_cache_shared",
                            total_bytes=out_count * bytes_per_candidate,
                            hw_config=hw_config,
                            is_read=False,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "chunk_idx": chunk_idx, "candidate_count": out_count},
                            memory_context=memory_context,
                            precomputed_spill_results=reduce_write_batch_results,
                        )
                        pass_cursor += reduce_write_dur
                        produced += out_count
                    source_buffer = dest_buffer
                    current_candidates = produced
                if source_buffer != "topk_sram_buffer":
                    pass_copy_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="topk_sram_buffer",
                        total_bytes=current_candidates * bytes_per_candidate,
                        cycle=pass_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        dir_prefix=f"knn_bv2_pass_copy_q{q}_p{pass_idx}",
                    )
                    _, pass_copy_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), pass_cursor, "KNN", "merge_sort",
                        sub_stage="topk_pass_write",
                        circuit_module="topk_sram_buffer",
                        operation="knn_topk_pass_write",
                        buffer_name="topk_sram_buffer",
                        phase="pass_candidate_write",
                        resource_id="topk_sram_shared",
                        total_bytes=current_candidates * bytes_per_candidate,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "candidate_count": current_candidates},
                        memory_context=memory_context,
                        precomputed_spill_results=pass_copy_batch_results,
                    )
                    pass_cursor += pass_copy_dur
                pass_ready_cycles.append(pass_cursor)
                pass_candidate_counts.append(current_candidates)

            final_cursor = max(pass_ready_cycles) if pass_ready_cycles else current_cycle
            source_buffer = "topk_sram_buffer"
            current_candidates = sum(pass_candidate_counts)
            while current_candidates > local_topk:
                chunks = max(1, int(math.ceil(current_candidates / float(local_batch_points))))
                dest_buffer = "sort_cache_sram" if source_buffer == "topk_sram_buffer" else "topk_sram_buffer"
                produced = 0
                for chunk_idx in range(chunks):
                    chunk_count = min(local_batch_points, max(0, current_candidates - chunk_idx * local_batch_points))
                    if chunk_count <= 0:
                        continue
                    read_bytes = chunk_count * bytes_per_candidate
                    final_read_batch_results = _acc_precompute_consume_batch_results(
                        memory_context,
                        buffer_name=source_buffer,
                        total_bytes=read_bytes,
                        cycle=final_cursor,
                        hw_config=hw_config,
                        resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                        dir_prefix=f"knn_bv2_final_read_q{q}_c{chunk_idx}",
                    )
                    _, final_read_dur = _acc_append_buffer_consume_events(
                        all_events, len(all_events), final_cursor, "KNN", "merge_sort",
                        sub_stage="topk_final_read",
                        circuit_module=source_buffer,
                        operation="knn_topk_final_read",
                        buffer_name=source_buffer,
                        phase="final_sort_read",
                        resource_id="topk_sram_shared" if source_buffer == "topk_sram_buffer" else "sort_cache_shared",
                        total_bytes=read_bytes,
                        hw_config=hw_config,
                        resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                        extra_meta={"query": q, "num_passes": num_passes, "chunk_idx": chunk_idx},
                        memory_context=memory_context,
                        precomputed_spill_results=final_read_batch_results,
                    )
                    final_cursor += final_read_dur
                    all_events.append(Event(
                        len(all_events), final_cursor, "KNN", "merge_sort", "merge_sort_global_topk", "merge_sort", "merge_sort_top32",
                        float(getattr(ms, "power_top32_mw", 312.0) or 312.0), merge_cycles, clock_period_ns=clock_ns,
                        metadata={"query": q, "num_passes": num_passes, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes, "resource_id": "merge_sort_shared", "exclusive": True},
                    ))
                    final_cursor += merge_cycles
                    out_count = min(local_topk, chunk_count)
                    final_write_batch_results = _acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name=dest_buffer,
                        total_bytes=out_count * bytes_per_candidate,
                        cycle=final_cursor,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                        dir_prefix=f"knn_bv2_final_write_q{q}_c{chunk_idx}",
                    )
                    _, final_write_dur, _ = _acc_append_buffer_fill_events(
                        all_events, len(all_events), final_cursor, "KNN", "merge_sort",
                        sub_stage="topk_final_write",
                        circuit_module=dest_buffer,
                        operation="knn_topk_final_write",
                        buffer_name=dest_buffer,
                        phase="final_candidate_write",
                        resource_id="topk_sram_shared" if dest_buffer == "topk_sram_buffer" else "sort_cache_shared",
                        total_bytes=out_count * bytes_per_candidate,
                        hw_config=hw_config,
                        is_read=False,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                        extra_meta={"query": q, "num_passes": num_passes, "chunk_idx": chunk_idx, "candidate_count": out_count},
                        memory_context=memory_context,
                        precomputed_spill_results=final_write_batch_results,
                    )
                    final_cursor += final_write_dur
                    produced += out_count
                source_buffer = dest_buffer
                current_candidates = produced
            current_cycle = max(current_cycle, final_cursor)
            continue

        if use_fuse:
            dist_total_bytes = int(input_points) * 4
            _, dist_read_dur = _acc_append_buffer_consume_events(
                all_events, len(all_events), current_cycle, "KNN", "sram_read",
                sub_stage="distance_read",
                circuit_module="dist_sram_buffer",
                operation="dist_buffer_read_for_sort",
                buffer_name="dist_sram_buffer",
                phase="read",
                resource_id="dist_sram_shared",
                total_bytes=dist_total_bytes,
                hw_config=hw_config,
                resident_bytes=min(dist_total_bytes, _acc_buffer_capacity_bytes("dist_sram_buffer", hw_config)),
                preferred_chunk_bytes=max(1, int(getattr(hw_config, "dist_sram_buffer", None).size_bytes if getattr(hw_config, "dist_sram_buffer", None) is not None else 4096)),
                extra_meta={"query": q, "bytes": dist_total_bytes, "kind": "dist_to_sort", "reset_scope": "query_boundary"},
                memory_context=context.get("memory_runtime_context") if context else None,
            )
            current_cycle += dist_read_dur
            fuse_events = knn_op.generate_single_fuse_events()
            for fe in fuse_events:
                fe.cycle += current_cycle
            all_events.extend(fuse_events)
            current_cycle = max(fe.cycle + fe.duration for fe in fuse_events)
        else:
            sorter_depth = _sorter_depth_points(hw_config)
            num_passes = max(1, int(math.ceil(input_points / float(sorter_depth))))
            bytes_per_candidate = _distance_index_entry_bytes(input_points)
            local_topk = max(32, int(getattr(knn_params, "k", 32) or 32))
            for pass_idx in range(num_passes):
                pass_points = min(sorter_depth, max(0, input_points - pass_idx * sorter_depth))
                if pass_points <= 0:
                    continue
                pass_read_bytes = int(pass_points) * 4
                _, sort_read_dur = _acc_append_buffer_consume_events(
                    all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                    sub_stage="distance_local_sort_read",
                    circuit_module="dist_sram_buffer",
                    operation="dist_sram_local_sort_read",
                    buffer_name="dist_sram_buffer",
                    phase="local_sort_read",
                    resource_id="dist_sram_shared",
                    total_bytes=pass_read_bytes,
                    hw_config=hw_config,
                    resident_bytes=min(pass_read_bytes, _acc_buffer_capacity_bytes("dist_sram_buffer", hw_config)),
                    preferred_chunk_bytes=max(1, int(getattr(hw_config, "dist_sram_buffer", None).size_bytes if getattr(hw_config, "dist_sram_buffer", None) is not None else 4096)),
                    extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "reset_scope": "query_boundary"},
                    memory_context=context.get("memory_runtime_context") if context else None,
                )
                current_cycle += sort_read_dur
                one_bitonic = knn_op.generate_single_bitonic_event()
                one_bitonic.cycle = current_cycle
                meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                meta.update({
                    "query": q,
                    "pass_idx": pass_idx,
                    "num_passes": num_passes,
                    "buffer_name": "dist_sram_buffer",
                    "buffer_phase": "local_sort_read",
                    "bytes": pass_read_bytes,
                })
                one_bitonic.metadata = meta
                all_events.append(one_bitonic)
                current_cycle += one_bitonic.duration
                local_topk_bytes = local_topk * bytes_per_candidate
                _, topk_write_dur, _ = _acc_append_buffer_fill_events(
                    all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                    sub_stage="topk_local_write",
                    circuit_module="topk_sram_buffer",
                    operation="knn_topk_local_write",
                    buffer_name="topk_sram_buffer",
                    phase="local_candidate_write",
                    resource_id="topk_sram_shared",
                    total_bytes=local_topk_bytes,
                    hw_config=hw_config,
                    is_read=False,
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                    extra_meta={"query": q, "pass_idx": pass_idx, "num_passes": num_passes, "candidate_count": local_topk},
                    memory_context=memory_context,
                    precomputed_spill_results=_acc_precompute_fill_batch_results(
                        memory_context,
                        buffer_name="topk_sram_buffer",
                        total_bytes=local_topk_bytes,
                        cycle=current_cycle,
                        hw_config=hw_config,
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        dir_prefix=f"knn_base_topk_write_q{q}_p{pass_idx}",
                    ),
                )
                current_cycle += topk_write_dur
            final_candidates = num_passes * local_topk
            if final_candidates <= sorter_depth:
                final_candidate_bytes = final_candidates * bytes_per_candidate
                _, final_read_dur = _acc_append_buffer_consume_events(
                    all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                    sub_stage="topk_final_read",
                    circuit_module="topk_sram_buffer",
                    operation="knn_topk_final_read",
                    buffer_name="topk_sram_buffer",
                    phase="final_sort_read",
                    resource_id="topk_sram_shared",
                    total_bytes=final_candidate_bytes,
                    hw_config=hw_config,
                    resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes("topk_sram_buffer", hw_config)),
                    preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                    extra_meta={"query": q, "num_passes": num_passes, "candidate_count": final_candidates},
                    memory_context=memory_context,
                    precomputed_spill_results=_acc_precompute_consume_batch_results(
                        memory_context,
                        buffer_name="topk_sram_buffer",
                        total_bytes=final_candidate_bytes,
                        cycle=current_cycle,
                        hw_config=hw_config,
                        resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes("topk_sram_buffer", hw_config)),
                        preferred_chunk_bytes=_acc_buffer_capacity_bytes("topk_sram_buffer", hw_config),
                        dir_prefix=f"knn_base_final_read_q{q}",
                    ),
                )
                current_cycle += final_read_dur
                one_bitonic = knn_op.generate_single_bitonic_event()
                one_bitonic.cycle = current_cycle
                meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                meta.update({"query": q, "buffer_name": "topk_sram_buffer", "buffer_phase": "final_sort_read", "bytes": final_candidate_bytes})
                one_bitonic.metadata = meta
                all_events.append(one_bitonic)
                current_cycle += one_bitonic.duration
            else:
                source_buffer = "topk_sram_buffer"
                current_candidates = final_candidates
                while current_candidates > local_topk:
                    reduce_passes = max(1, int(math.ceil(current_candidates / float(sorter_depth))))
                    if current_candidates <= sorter_depth:
                        final_candidate_bytes = current_candidates * bytes_per_candidate
                        _, final_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                            sub_stage="topk_final_read",
                            circuit_module=source_buffer,
                            operation="knn_topk_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="topk_sram_shared" if source_buffer == "topk_sram_buffer" else "sort_cache_shared",
                            total_bytes=final_candidate_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"query": q, "num_passes": num_passes, "candidate_count": current_candidates},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_consume_batch_results(
                                memory_context,
                                buffer_name=source_buffer,
                                total_bytes=final_candidate_bytes,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                resident_bytes=min(final_candidate_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                                dir_prefix=f"knn_base_reduce_final_read_q{q}_lvl{current_candidates}",
                            ),
                        )
                        current_cycle += final_read_dur
                        one_bitonic = knn_op.generate_single_bitonic_event()
                        one_bitonic.cycle = current_cycle
                        meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                        meta.update({"query": q, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": final_candidate_bytes})
                        one_bitonic.metadata = meta
                        all_events.append(one_bitonic)
                        current_cycle += one_bitonic.duration
                        break
                    dest_buffer = "sort_cache_sram" if source_buffer == "topk_sram_buffer" else "topk_sram_buffer"
                    produced = 0
                    for chunk_idx in range(reduce_passes):
                        chunk_count = min(sorter_depth, max(0, current_candidates - chunk_idx * sorter_depth))
                        if chunk_count <= 0:
                            continue
                        read_bytes = chunk_count * bytes_per_candidate
                        _, reduce_read_dur = _acc_append_buffer_consume_events(
                            all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                            sub_stage="topk_final_read",
                            circuit_module=source_buffer,
                            operation="knn_topk_final_read",
                            buffer_name=source_buffer,
                            phase="final_sort_read",
                            resource_id="topk_sram_shared" if source_buffer == "topk_sram_buffer" else "sort_cache_shared",
                            total_bytes=read_bytes,
                            hw_config=hw_config,
                            resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                            extra_meta={"query": q, "num_passes": num_passes, "chunk_idx": chunk_idx, "candidate_count": chunk_count},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_consume_batch_results(
                                memory_context,
                                buffer_name=source_buffer,
                                total_bytes=read_bytes,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                resident_bytes=min(read_bytes, _acc_buffer_capacity_bytes(source_buffer, hw_config)),
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(source_buffer, hw_config),
                                dir_prefix=f"knn_base_reduce_read_q{q}_c{chunk_idx}",
                            ),
                        )
                        current_cycle += reduce_read_dur
                        one_bitonic = knn_op.generate_single_bitonic_event()
                        one_bitonic.cycle = current_cycle
                        meta = dict(getattr(one_bitonic, "metadata", {}) or {})
                        meta.update({"query": q, "chunk_idx": chunk_idx, "buffer_name": source_buffer, "buffer_phase": "final_sort_read", "bytes": read_bytes})
                        one_bitonic.metadata = meta
                        all_events.append(one_bitonic)
                        current_cycle += one_bitonic.duration
                        out_count = min(local_topk, chunk_count)
                        _, reduce_write_dur, _ = _acc_append_buffer_fill_events(
                            all_events, len(all_events), current_cycle, "KNN", "bitonic_sort",
                            sub_stage="topk_final_write",
                            circuit_module=dest_buffer,
                            operation="knn_topk_final_write",
                            buffer_name=dest_buffer,
                            phase="final_candidate_write",
                            resource_id="topk_sram_shared" if dest_buffer == "topk_sram_buffer" else "sort_cache_shared",
                            total_bytes=out_count * bytes_per_candidate,
                            hw_config=hw_config,
                            is_read=False,
                            preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                            extra_meta={"query": q, "num_passes": num_passes, "chunk_idx": chunk_idx, "candidate_count": out_count},
                            memory_context=memory_context,
                            precomputed_spill_results=_acc_precompute_fill_batch_results(
                                memory_context,
                                buffer_name=dest_buffer,
                                total_bytes=out_count * bytes_per_candidate,
                                cycle=current_cycle,
                                hw_config=hw_config,
                                preferred_chunk_bytes=_acc_buffer_capacity_bytes(dest_buffer, hw_config),
                                dir_prefix=f"knn_base_reduce_write_q{q}_c{chunk_idx}",
                            ),
                        )
                        current_cycle += reduce_write_dur
                        produced += out_count
                    source_buffer = dest_buffer
                    current_candidates = produced
    if dist_matrix_dram_write:
        use_streaming = getattr(hw_config, "distance_matrix_reuse", False)
        end_cycle = max((e.cycle + e.duration for e in all_events), default=0)
        end_eid = max((e.event_id for e in all_events), default=-1) + 1
        if use_streaming:
            per_q_cycles = _extract_per_query_end_cycles(all_events, query_points)
            wb_events = knn_op.generate_dram_writeback_events(
                start_cycle=0, start_event_id=end_eid,
                streaming=True, per_query_ready_cycles=per_q_cycles,
                transpose_store=transpose_store,
            )
        else:
            wb_events = knn_op.generate_dram_writeback_events(
                start_cycle=end_cycle, start_event_id=end_eid,
                transpose_store=transpose_store,
            )
        all_events.extend(wb_events)
    return all_events


# =============================================================================
# PointCIM Fused FPS+BQ: 距离复用 + BQ Engine streaming + MDT parallel
# =============================================================================

def _gen_pointcim_fused_fps_bq(
    fps_ref: 'OperatorRef',
    bq_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    PointCIM 模式下 FPS+BQ/KNN 融合流水线。

    支持 PointNet++ SA (fps+ballquery) 和 PointKAN GAM (fps+knn)，
    硬件路径一致：BQ 叶+merge 引擎同时服务 BallQuery 和 KNN 操作。

    每个 FPS 采样迭代 i (i=0..npoint-1):
      1. RRAM VMM 距离计算 (一次算出 query_i 对所有 N key 的交叉项)
      2. merge adder (并行度=block_n=64) → 产出 N/64 个距离块
      3. 并行路径 A: 距离块流式喂入 BQ 叶+merge 引擎 → top-k 邻居索引
      4. 并行路径 B: 距离块更新 min_distance_table (FPS 用)
      5. BQ/KNN 完成 → query_i 的 group_idx 就绪
      6. MDT 更新 + argmax → 下一个 query_{i+1}
      7. 循环

    距离只计算一次，FPS 和 BQ/KNN 共享，消除重复计算。
    merge adder 并行度 = BQ 叶引擎 block_n，实现零等待 streaming。
    """
    from ..configs.hw_config import BQEngineConfig, select_bq_engine_pair

    fps_params = fps_ref.params
    bq_params = bq_ref.params
    input_points = int(fps_params.get('input_points', 1024))
    sampled_points = int(fps_params.get('output_points', 512))
    bq_k = int(bq_params.get('k', 32))
    neighbor_op_type = str(getattr(bq_ref, "op_type", "ballquery") or "ballquery")
    engine_mode = "knn" if neighbor_op_type == "knn" else "bq"

    clock_ns = float(getattr(hw_config, "clock_period", 1.0))

    sc = getattr(hw_config, "sorting_circuit", None)
    bq_eng: Optional[BQEngineConfig] = getattr(sc, "bq_engine", None) if sc else None
    if bq_eng is None:
        bq_eng = select_bq_engine_pair(k=bq_k, mode=engine_mode)

    mdt_cfg = getattr(sc, "min_distance_table", None) if sc else None
    mdt_power = float(getattr(mdt_cfg, "power", 30.0) or 30.0)
    mdt_cycles = max(1, int(getattr(mdt_cfg, "cycles", 1) or 1))

    all_events: List[Event] = []
    eid = 0
    current_cycle = 0

    dist_tag_scope = f"fused_fps_bq_{input_points}_{sampled_points}_"

    # 纯 acc（非 PointCIM）：仅首层跑 FPS+BQ，后续层复用首层索引
    # PointCIM 后续层需重新计算（距离矩阵来自 RRAM，不可跨层共享）
    _acc_pure = (getattr(hw_config, "is_acc", False)
                 and not getattr(hw_config, "use_rram_dist", False))
    if _acc_pure and context is not None:
        first_fps = context.get("first_layer_with_fps")
        layer_index = context.get("layer_index")
        if first_fps is not None and layer_index is not None and layer_index != first_fps:
            idx_bytes = _index_bytes_for_points(input_points)
            total_idx_bytes = max(1, sampled_points * idx_bytes)
            reuse_events: List[Event] = []
            _eid, _ = _acc_append_sram_access_event(
                reuse_events, 0, 0, "FPS", "reuse",
                "fps_index_read", "fps_index_sram_buffer", "fps_index_reuse_read",
                "fps_index_sram_buffer", "read", "fps_index_shared", total_idx_bytes, hw_config,
                is_read=True, overlapped=False, extra_meta={"fps_reuse": True},
            )
            reuse_events.append(Event(
                _eid, 0, "FPS", "reuse", "fps_reuse", "fps_index_sram_buffer", "fps_reuse_first_layer",
                0.0, 1, clock_ns, metadata={"fps_reuse": True, "buffer_name": "fps_index_sram_buffer", "bytes": total_idx_bytes},
            ))
            return reuse_events

    for i in range(sampled_points):
        iter_start = current_cycle
        candidate_points = max(1, input_points - i)

        # Step 1: RRAM 距离计算 (交叉项 + self-term merge) — 归属 "FPS" 算子
        evs_d, eid, dist_cycles, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags = _gen_pointcim_distance_query_events(
            start_event_id=eid,
            base_cycle=iter_start,
            operator_name="FPS",
            num_keys=input_points,
            hw_config=hw_config,
            query_index=i,
            startup_once=True,
            write_dist_buffer=False,
            tag_scope=dist_tag_scope,
        )
        for ev in evs_d:
            meta = dict(getattr(ev, "metadata", None) or {})
            meta["distance_shared_with_bq"] = True
            ev.metadata = meta
        all_events.extend(evs_d)

        # Step 2A: BQ/KNN engine streaming (距离块 → 叶引擎 → merge 链)
        neighbor_op_name = "KNN" if engine_mode == "knn" else "BallQuery"
        bq_evts, eid, bq_total = _gen_bq_engine_events(
            num_points=input_points,
            k=bq_k,
            hw_config=hw_config,
            start_cycle=iter_start,
            start_event_id=eid,
            operator_name=neighbor_op_name,
            query_index=i,
            block_ready_cycles=block_ready_cycles if block_ready_cycles else None,
            tag_scope=dist_tag_scope,
        )
        native_op_label = "pointkan_knn" if engine_mode == "knn" else "pointnet2_ballquery"
        for ev in bq_evts:
            meta = dict(getattr(ev, "metadata", None) or {})
            meta["native_op"] = native_op_label
            meta["fused_fps_bq"] = True
            meta["fps_iteration"] = i
            meta["query"] = i
            ev.metadata = meta
        all_events.extend(bq_evts)

        bq_end = iter_start + max(dist_cycles, bq_total) if bq_evts else iter_start + dist_cycles

        # Step 2B: MDT update (并行于 BQ, 使用相同的距离块 ready tags)
        mdt_barrier = iter_start
        for blk_idx, blk_ready in enumerate(block_ready_cycles or []):
            mdt_blk_start = max(blk_ready, mdt_barrier)
            mdt_tag = f"{dist_tag_scope}fused_mdt_q{i}_b{blk_idx}"
            dep_tags = []
            if blk_idx < len(block_ready_tags):
                dep_tags.append(block_ready_tags[blk_idx])

            mdt_ev = Event(
                eid, mdt_blk_start, "FPS", "distance", "min_distance_update",
                "min_distance_table", "min_distance_compare",
                mdt_power, mdt_cycles, clock_period_ns=clock_ns,
                metadata={
                    "iteration": i,
                    "query_index": i,
                    "block_idx": blk_idx,
                    "num_blocks": len(block_ready_cycles or []),
                    "buffer_name": "fps_min_distance_table_sram",
                    "buffer_phase": "update",
                    "resource_id": "fps_min_update_shared",
                    "exclusive": True,
                    "phase": "fused_fps_bq_mdt",
                    "dependency_tag": mdt_tag,
                    "depends_on_tags": dep_tags if dep_tags else None,
                    "fused_fps_bq": True,
                },
            )
            mdt_ev.overlapped = True
            all_events.append(mdt_ev)
            eid += 1
            mdt_barrier = mdt_blk_start + mdt_cycles

        # Step 3: FPS argmax (选下一个最远点, 依赖 MDT 更新完成)
        fps_argmax_start = max(mdt_barrier, bq_end)
        argmax_tag = f"{dist_tag_scope}fps_argmax_q{i}"
        fps_argmax_ev = Event(
            eid, fps_argmax_start, "FPS", "fused_fps_bq", "fps_argmax",
            "min_distance_table", "fps_select_farthest",
            mdt_power, 1, clock_period_ns=clock_ns,
            metadata={
                "iteration": i,
                "query_index": i,
                "phase": "fused_fps_bq_argmax",
                "resource_id": "fps_argmax_shared",
                "exclusive": True,
                "dependency_tag": argmax_tag,
                "fused_fps_bq": True,
            },
        )
        all_events.append(fps_argmax_ev)
        eid += 1

        current_cycle = fps_argmax_start + 1

    # Tag all events with fused metadata
    for ev in all_events:
        meta = dict(getattr(ev, "metadata", None) or {})
        meta.setdefault("fused_fps_bq", True)
        meta.setdefault("sampled_points", sampled_points)
        meta.setdefault("input_points", input_points)
        ev.metadata = meta

    return all_events


def _gen_bq_engine_events(
    num_points: int,
    k: int,
    hw_config: 'HardwareConfig',
    start_cycle: int = 0,
    start_event_id: int = 0,
    operator_name: str = "BallQuery",
    query_index: int = 0,
    block_ready_cycles: Optional[List[int]] = None,
    tag_scope: str = "",
) -> Tuple[List[Event], int, int]:
    """
    BQ 叶+merge 引擎事件生成：TDM 叶 + 单 merge 引擎 + SRAM buffer 链式串行。

    架构（对应 ball_query_engind.md）：
    - TDM 叶引擎（bq64_top32）: 每批处理 block_n 距离 → output_k 已排序槽
    - 单个 merge 引擎（merge32）: 链式串行，每次将新叶结果与 SRAM 中已有结果合并
    - topk_sram_buffer: merge 结果暂存，下次 merge 时读出

    时序（chain-serial, per query）：
      leaf_0 → sram_write(result_0)
      leaf_1 → sram_read(result_0) → merge(leaf_1, result_0) → sram_write(merged_01)
      leaf_2 → sram_read(merged_01) → merge(leaf_2, merged_01) → sram_write(merged_012)
      ...
    叶与前序 merge 可流水：leaf_{j+1} 在 merge_j 完成前即可启动（TDM 独立硬件）。

    streaming 模式：当 block_ready_cycles 给定时，叶引擎 batch j 的启动等待距离块就绪：
      leaf_start[j] = max(block_ready_cycles[j], leaf_end[j-1])

    Returns:
        (events, next_event_id, total_cycles)
    """
    from ..configs.hw_config import BQEngineConfig, select_bq_engine_pair

    sc = getattr(hw_config, "sorting_circuit", None)
    bq_eng: Optional[BQEngineConfig] = getattr(sc, "bq_engine", None) if sc else None
    if bq_eng is None:
        bq_eng = select_bq_engine_pair(k=k, mode="bq")

    leaf = bq_eng.leaf
    merge = bq_eng.merge
    clk = float(getattr(hw_config, "clock_period", 1.0))

    topk_buf = getattr(hw_config, "topk_sram_buffer", None)
    topk_bw = int(getattr(topk_buf, "bandwidth_bytes_per_cycle", 64) or 64) if topk_buf else 64
    topk_rd_power = float(getattr(topk_buf, "read_power_mw", 2.0) or 2.0) if topk_buf else 2.0
    topk_wr_power = float(getattr(topk_buf, "write_power_mw", 2.0) or 2.0) if topk_buf else 2.0
    # merge 结果大小：output_k 个 (距离, 索引) 对，每对 8 字节 (float32 + int32)
    merge_result_bytes = max(1, leaf.output_k * 8)
    sram_rw_cycles = max(1, int(math.ceil(merge_result_bytes / float(topk_bw))))

    n_batches = max(1, int(math.ceil(num_points / float(leaf.block_n))))
    events: List[Event] = []
    eid = start_event_id

    # --- Leaf phase (TDM, sequential) ---
    leaf_end_cycles: List[int] = []
    leaf_ready_tags: List[str] = []

    for j in range(n_batches):
        if block_ready_cycles is not None and j < len(block_ready_cycles):
            data_ready = block_ready_cycles[j]
        else:
            data_ready = start_cycle

        prev_leaf_end = leaf_end_cycles[-1] if leaf_end_cycles else start_cycle
        leaf_start = max(data_ready, prev_leaf_end)
        leaf_tag = f"{tag_scope}bq_leaf_q{query_index}_b{j}" if tag_scope else f"bq_leaf_q{query_index}_b{j}"
        depends: List[str] = []
        if block_ready_cycles is not None and tag_scope:
            dist_tag = f"{tag_scope}pointcim_merge_ready_q{query_index}_b{j}"
            depends.append(dist_tag)
        if j > 0:
            depends.append(leaf_ready_tags[-1])

        blk_points = min(leaf.block_n, max(0, num_points - j * leaf.block_n))
        ev = Event(
            eid, leaf_start, operator_name, "bq_engine", "bq_leaf_process",
            f"bq_leaf_{leaf.key}", f"bq_leaf_batch_{j}",
            leaf.power_mw, leaf.latency_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "batch_idx": j,
                "num_batches": n_batches,
                "batch_points": blk_points,
                "leaf_engine": leaf.key,
                "phase": "bq_engine_leaf",
                "resource_id": f"bq_leaf_{leaf.key}",
                "area_um2": float(getattr(leaf, "area_um2", 0.0) or 0.0),
                "exclusive": True,
                "dependency_tag": leaf_tag,
                "depends_on_tags": depends if depends else None,
            },
        )
        events.append(ev)
        eid += 1
        leaf_end_cycles.append(leaf_start + leaf.latency_cycles)
        leaf_ready_tags.append(leaf_tag)

    # --- Single merge engine, chain-serial with SRAM buffer ---
    if n_batches <= 1:
        # 单批无需 merge，但需写入 SRAM buffer
        wr_start = leaf_end_cycles[0] if leaf_end_cycles else start_cycle
        wr_tag = f"{tag_scope}bq_sram_wr_q{query_index}_final" if tag_scope else f"bq_sram_wr_q{query_index}_final"
        events.append(Event(
            eid, wr_start, operator_name, "bq_engine", "bq_topk_sram_write",
            "topk_sram_buffer", f"bq_topk_write_final",
            topk_wr_power, sram_rw_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "merge_step": 0,
                "bytes": merge_result_bytes,
                "phase": "bq_engine_sram",
                "resource_id": "topk_sram_buffer",
                "dependency_tag": wr_tag,
                "depends_on_tags": [leaf_ready_tags[0]],
            },
        ))
        eid += 1
        total = (wr_start + sram_rw_cycles) - start_cycle
        return events, eid, total

    # 链式 merge：
    #   step 0: leaf_0 完成 → write result to SRAM
    #   step j (j>=1): wait(leaf_j done AND prev_merge_sram_write done)
    #                   → read prev result from SRAM → merge → write merged to SRAM
    prev_sram_wr_end = leaf_end_cycles[0]
    prev_sram_wr_tag = leaf_ready_tags[0]

    # step 0: leaf_0 → sram write
    wr0_tag = f"{tag_scope}bq_sram_wr_q{query_index}_s0" if tag_scope else f"bq_sram_wr_q{query_index}_s0"
    events.append(Event(
        eid, leaf_end_cycles[0], operator_name, "bq_engine", "bq_topk_sram_write",
        "topk_sram_buffer", f"bq_topk_write_s0",
        topk_wr_power, sram_rw_cycles, clock_period_ns=clk,
        metadata={
            "query_index": query_index,
            "merge_step": 0,
            "bytes": merge_result_bytes,
            "phase": "bq_engine_sram",
            "resource_id": "topk_sram_buffer",
            "dependency_tag": wr0_tag,
            "depends_on_tags": [leaf_ready_tags[0]],
        },
    ))
    eid += 1
    prev_sram_wr_end = leaf_end_cycles[0] + sram_rw_cycles
    prev_sram_wr_tag = wr0_tag

    for j in range(1, n_batches):
        # wait for both: leaf_j done AND prev sram_write done
        merge_chain_start = max(leaf_end_cycles[j], prev_sram_wr_end)

        # 1) SRAM read: 读取上轮 merged 结果
        rd_tag = f"{tag_scope}bq_sram_rd_q{query_index}_s{j}" if tag_scope else f"bq_sram_rd_q{query_index}_s{j}"
        events.append(Event(
            eid, merge_chain_start, operator_name, "bq_engine", "bq_topk_sram_read",
            "topk_sram_buffer", f"bq_topk_read_s{j}",
            topk_rd_power, sram_rw_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "merge_step": j,
                "bytes": merge_result_bytes,
                "phase": "bq_engine_sram",
                "resource_id": "topk_sram_buffer",
                "dependency_tag": rd_tag,
                "depends_on_tags": [leaf_ready_tags[j], prev_sram_wr_tag],
            },
        ))
        eid += 1
        rd_end = merge_chain_start + sram_rw_cycles

        # 2) merge 引擎: 将 leaf_j 结果与 SRAM 读出的 prev 结果合并
        merge_tag = f"{tag_scope}bq_merge_q{query_index}_s{j}" if tag_scope else f"bq_merge_q{query_index}_s{j}"
        events.append(Event(
            eid, rd_end, operator_name, "bq_engine", "bq_merge",
            f"bq_merge_{merge.key}", f"bq_merge_step_{j}",
            merge.power_mw, merge.latency_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "merge_step": j,
                "num_merges": n_batches - 1,
                "merge_engine": merge.key,
                "phase": "bq_engine_merge",
                "resource_id": f"bq_merge_{merge.key}",
                "area_um2": float(getattr(merge, "area_um2", 0.0) or 0.0),
                "exclusive": True,
                "dependency_tag": merge_tag,
                "depends_on_tags": [rd_tag],
            },
        ))
        eid += 1
        merge_end = rd_end + merge.latency_cycles

        # 3) SRAM write: 将 merged 结果写回
        wr_tag = f"{tag_scope}bq_sram_wr_q{query_index}_s{j}" if tag_scope else f"bq_sram_wr_q{query_index}_s{j}"
        events.append(Event(
            eid, merge_end, operator_name, "bq_engine", "bq_topk_sram_write",
            "topk_sram_buffer", f"bq_topk_write_s{j}",
            topk_wr_power, sram_rw_cycles, clock_period_ns=clk,
            metadata={
                "query_index": query_index,
                "merge_step": j,
                "bytes": merge_result_bytes,
                "phase": "bq_engine_sram",
                "resource_id": "topk_sram_buffer",
                "dependency_tag": wr_tag,
                "depends_on_tags": [merge_tag],
            },
        ))
        eid += 1
        prev_sram_wr_end = merge_end + sram_rw_cycles
        prev_sram_wr_tag = wr_tag

    total = prev_sram_wr_end - start_cycle
    return events, eid, total


def _gen_ballquery(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    PointNet++ BallQuery native op.

    PointCIM/ACC 模式且有 bq_engine 配置时：使用叶+merge 引擎周期模型。
    否则回退到 KNN shadow 路径（兼容 baseline）。
    """
    sc = getattr(hw_config, "sorting_circuit", None)
    bq_engine = getattr(sc, "bq_engine", None) if sc else None

    if bq_engine is not None and getattr(hw_config, "is_acc", False):
        params = op_ref.params
        input_points = int(params.get('input_points', 1024))
        k = int(params.get('k', 32))
        events, _, total = _gen_bq_engine_events(
            num_points=input_points,
            k=k,
            hw_config=hw_config,
            start_cycle=0,
            start_event_id=0,
            operator_name="BallQuery",
            query_index=0,
            block_ready_cycles=None,
            tag_scope="",
        )
        for ev in events:
            meta = dict(getattr(ev, "metadata", None) or {})
            meta["native_op"] = "pointnet2_ballquery"
            meta["query_points"] = int(params.get("query_points", 512))
            ev.metadata = meta
        return events

    shadow = op_ref.__class__(
        op_type="knn",
        params=dict(op_ref.params),
        description=op_ref.description,
    )
    events = _gen_knn(shadow, hw_config, context)
    return _retag_events(
        events,
        operator="BallQuery",
        operation_replace={"knn": "ballquery"},
        metadata_updates={"native_op": "pointnet2_ballquery"},
    )


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


def _gen_sa_max_pool(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    return gen_sa_max_pool_events(op_ref.params, hw_config)


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


def _gen_interp_n3(
    op_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """InterpN3 插值引擎：gather + div/mul/add 统一建模。"""
    return gen_interp_n3_events(op_ref.params, hw_config)


# =============================================================================
# Dispatcher Registry & Public API
# =============================================================================

_OP_TYPE_HANDLERS: Dict[str, Callable] = {
    'conv1d':      _gen_conv1d,
    'conv2d':      _gen_conv2d,
    'linear':      _gen_linear,
    'bn':          _gen_bn,
    'batchnorm1d': _gen_bn,       # alias
    'layernorm':   _gen_layernorm,
    'relu':        _gen_relu,
    's-pool':      _gen_spool,
    'fps':         _gen_fps,
    'knn':         _gen_knn,
    'ballquery':   _gen_ballquery,
    'kan':         _gen_kan,
    'affine_norm':           _gen_affine_norm,
    'adaptive_max_pool1d':   _gen_adaptive_max_pool,
    'sa_maxpool':            _gen_sa_max_pool,
    'add':                   _gen_add,
    'mul':                   _gen_mul,
    'multiplication':        _gen_mul,
    'element_multiply':      _gen_mul,
    'div':                   _gen_div,
    'division':              _gen_div,
    'element_divide':        _gen_div,
    'interp_n3':             _gen_interp_n3,
}


def _dispatch_fps_knn_pipelined(
    fps_ref: 'OperatorRef',
    knn_ref: 'OperatorRef',
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Baseline 模式下 FPS→KNN 引擎间流水线重叠（producer-consumer）。

    FusionArch 风格：FPS 迭代 i 产出 centroid i 后，KNN query i 即可启动，
    无需等待全部 FPS 迭代完成。KNN query i 的开始时间取：
        max(FPS_iter_i 结束, KNN_query_{i-1} 结束)

    返回所有事件（cycle 相对 0），调用者负责加 current_cycle offset。
    """
    from .fps import FPSOperator, FPSParams
    from .knn import KNNOperator, KNNParams
    from itertools import accumulate

    # 1) 生成 FPS / KNN 事件（各自 cycle 从 0 开始，内部顺序已排好）
    fps_events = dispatch_operator(fps_ref, hw_config, context)
    knn_events = dispatch_operator(knn_ref, hw_config, context)

    if not fps_events or not knn_events:
        # 没有事件则退化为串行拼接
        fps_span = max((e.cycle + e.duration for e in fps_events), default=0) if fps_events else 0
        for e in knn_events:
            e.cycle += fps_span
        return fps_events + knn_events

    # 2) 重建 FPS 每迭代周期列表
    fps_params = fps_ref.params
    input_points = int(fps_params.get('input_points', 1024))
    sampled_points = int(fps_params.get('output_points', 512))
    fps_op = FPSOperator(
        params=FPSParams(input_points=input_points, sampled_points_num=sampled_points),
        hw_config=hw_config,
    )
    fps_iter_cycles = fps_op.get_per_iteration_cycles_list()

    # 3) 重建 KNN 每查询周期
    knn_params = knn_ref.params
    query_points = int(knn_params.get('query_points', sampled_points))
    knn_op = KNNOperator(
        params=KNNParams(
            input_points=input_points,
            query_points=query_points,
            k=int(knn_params.get('k', 16)),
        ),
        hw_config=hw_config,
    )
    knn_per_query = knn_op.get_per_query_cycles()

    # 4) 计算流水线调度：KNN query q 的起始周期
    fps_cum = list(accumulate(fps_iter_cycles))
    num_queries = min(query_points, len(fps_cum))
    knn_pipe_starts: List[int] = []
    for q in range(num_queries):
        fps_ready = fps_cum[q]
        prev_knn_end = (knn_pipe_starts[q - 1] + knn_per_query) if q > 0 else 0
        knn_pipe_starts.append(max(fps_ready, prev_knn_end))

    # 5) 找 KNN 事件按 query 分组的顺序起点
    knn_by_query: Dict[int, List[Event]] = {}
    for e in knn_events:
        q = (e.metadata or {}).get("query", 0)
        knn_by_query.setdefault(q, []).append(e)

    knn_query_seq_start: Dict[int, int] = {}
    for q, evts in knn_by_query.items():
        knn_query_seq_start[q] = min(e.cycle for e in evts)

    # 6) 将 KNN 各 query 事件从顺序位置搬到流水线位置
    for q, evts in knn_by_query.items():
        if q < len(knn_pipe_starts):
            shift = knn_pipe_starts[q] - knn_query_seq_start[q]
            for e in evts:
                e.cycle += shift

    return fps_events + knn_events


def _pipeline_conv_after_knn(
    preceding_events: List[Event],
    conv_ref: 'OperatorRef',
    num_queries: int,
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
    group_size_hint: Optional[int] = None,
    knn_in_channels: int = 3,
    force_strict: bool = False,
    use_sample_and_group: bool = False,
) -> List[Event]:
    """
    KNN→Conv1d 第二级 producer-consumer 流水线。

    KNN 每完成一个 query（产出 K 个邻居索引），先做 **feature gather**（从
    DRAM/SRAM 加载 K 个邻居的特征向量），再让 Conv1d 处理该 query 对应的
    group（kernel_size=1 时各 query 独立）。

    调度规则：
      gather_start[g] = max(knn_query_end[g], prev_group_end[g-1])
      conv_start[g]   = gather_end[g]
      group_end[g]     = conv_end[g]

    可选 multi-slice 模式（hw_config.feature_computing.num_slices > 1）：
    query 按 round-robin 分配到各 slice，slice 间并行，slice 内串行。

    前序事件的 cycle 应为相对 0。
    返回 preceding_events + feature_gather + Conv1d 流水线化事件，cycle 相对 0。
    """
    from .common_ops import (
        gen_feature_gather_events,
        gen_pointwise_conv_per_group_events,
        gen_sample_and_group_query_events,
    )

    # -- Fallback: dispatch full Conv1d and append serially ----------------
    def _serial_fallback() -> List[Event]:
        conv_events = dispatch_operator(conv_ref, hw_config, context)
        span = max((e.cycle + e.duration for e in preceding_events), default=0)
        for e in conv_events:
            e.cycle += span
        return preceding_events + conv_events

    if num_queries <= 0:
        return _serial_fallback()

    pipeline_mode = str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower()
    if pipeline_mode == "disabled":
        # force_strict=True: caller (PointCIM SA-Conv pipeline) guarantees
        # that preceding_events carry per-query BQ end times and strict
        # group_size_hint matches total_points — no need for user-level flag.
        if not force_strict:
            return _serial_fallback()
        pipeline_mode = "strict"  # auto-upgrade for PointCIM SA-Conv path

    # -- Build per-query KNN end-cycle map ---------------------------------
    knn_query_end: Dict[int, int] = {}
    knn_query_ready_tag: Dict[int, str] = {}
    for e in preceding_events:
        q = (e.metadata or {}).get("query")
        if q is not None:
            q_end = e.cycle + e.duration
            prev_end = knn_query_end.get(q, -1)
            knn_query_end[q] = max(knn_query_end.get(q, 0), q_end)
            if q_end >= prev_end:
                dep_tag = str((e.metadata or {}).get("dependency_tag", "") or "")
                if dep_tag:
                    knn_query_ready_tag[q] = dep_tag
    if not knn_query_end:
        return _serial_fallback()

    # Fix: use max of all values as fallback, not max-key lookup
    fallback_end = max(knn_query_end.values())

    # -- Strict mode: pointwise Conv num_points must equal query_points * k --
    conv_params = _merge_real_conv_linear_params(conv_ref, context)
    total_points = max(1, int(conv_params.get("num_points", 1) or 1))
    strict_match = (
        group_size_hint is not None
        and int(group_size_hint) > 0
        and int(group_size_hint) * num_queries == total_points
    )
    if pipeline_mode == "strict" and not strict_match:
        return _serial_fallback()

    # -- Group sizes -------------------------------------------------------
    if strict_match:
        group_sizes = [int(group_size_hint)] * num_queries
    else:
        group_sizes = [total_points // num_queries] * num_queries
        for idx in range(total_points % num_queries):
            group_sizes[idx] += 1
        group_sizes = [max(1, g) for g in group_sizes]

    # -- Feature computing config ------------------------------------------
    fc_cfg = getattr(hw_config, "feature_computing", None)
    num_slices = max(1, int(getattr(fc_cfg, "num_slices", 1) or 1)) if fc_cfg else 1
    coalescing = float(getattr(fc_cfg, "coalescing_factor", 0.0) or 0.0) if fc_cfg else 0.0

    # -- Determine bytes_per_element from quantisation setting -------------
    bpe = 1 if getattr(hw_config, "enable_w8a8_quant_dequant", False) else 4

    # -- Assign queries to slices (round-robin) ----------------------------
    slice_queries: List[List[int]] = [[] for _ in range(num_slices)]
    for g in range(num_queries):
        slice_queries[g % num_slices].append(g)

    # -- Resolve neurosim_full per-group latency/energy override (Gap B fix) --
    # When a neurosim_full context exists, SA Conv layers dispatched via this pipeline
    # bypass cim_layer_index.  Instead we look up the full-layer result by name and
    # distribute it proportionally across groups (weight = group_size / total_points).
    _neurosim_per_group_lat: Optional[List[float]] = None
    _neurosim_per_group_en: Optional[List[float]] = None
    if context:
        lat_by_name = context.get(CIM_LAT_ENERGY_BY_NAME)
        conv_name = ""
        # OperatorRef has no .name; resolve robustly from params/description.
        for n in _resolve_cim_name_candidates(conv_ref, context):
            if lat_by_name and n in lat_by_name:
                conv_name = n
                break
        if lat_by_name and conv_name:
            full_lat_ns, full_en_nj = lat_by_name[conv_name]
            _neurosim_per_group_lat = [
                full_lat_ns * group_sizes[g] / max(1, total_points)
                for g in range(num_queries)
            ]
            _neurosim_per_group_en = [
                full_en_nj * group_sizes[g] / max(1, total_points)
                for g in range(num_queries)
            ]
        # Name lookup may miss for some PointNet++ conv nodes (description/alias mismatch).
        # In that case consume one full-layer entry by index here (this pipeline path bypasses
        # _gen_conv1d/_gen_linear, so cim_layer_index must be advanced in this function).
        if _neurosim_per_group_lat is None:
            lat_arr = context.get(CIM_LAYER_LATENCY_NS)
            en_arr = context.get(CIM_LAYER_ENERGY_NJ)
            if isinstance(lat_arr, (list, tuple)) and isinstance(en_arr, (list, tuple)):
                idx = int(context.get(CIM_LAYER_INDEX, 0) or 0)
                if idx < len(lat_arr) and idx < len(en_arr):
                    full_lat_ns = float(lat_arr[idx])
                    full_en_nj = float(en_arr[idx])
                    context[CIM_LAYER_INDEX] = idx + 1
                    _neurosim_per_group_lat = [
                        full_lat_ns * group_sizes[g] / max(1, total_points)
                        for g in range(num_queries)
                    ]
                    _neurosim_per_group_en = [
                        full_en_nj * group_sizes[g] / max(1, total_points)
                        for g in range(num_queries)
                    ]

    # -- Per-group event generation (with SCALE-Sim template caching) ------
    eid = max((e.event_id for e in preceding_events), default=-1) + 1
    all_new_events: List[Event] = []
    _conv_template_cache: Dict[tuple, List[Event]] = {}
    current_layer_name = str((context or {}).get("layer_name", "") or "")

    for s in range(num_slices):
        prev_end_in_slice = 0
        for g in slice_queries[s]:
            knn_ready = knn_query_end.get(g, fallback_end)
            group_start = max(knn_ready, prev_end_in_slice)

            # 1) Feature gather events
            gather_dep_tag = None
            if use_sample_and_group:
                feature_channels = max(0, int(knn_in_channels) - 3)
                layer_tag_scope = f"{current_layer_name}_" if current_layer_name else ""
                gather_evts, eid, _gather_cycles, gather_dep_tag = gen_sample_and_group_query_events(
                    query_idx=g,
                    k=group_sizes[g],
                    xyz_channels=3,
                    feature_channels=feature_channels,
                    hw_config=hw_config,
                    start_cycle=group_start,
                    start_event_id=eid,
                    current_layer_name=current_layer_name,
                    tag_scope=f"{layer_tag_scope}q{g}_",
                    bq_ready_tag=knn_query_ready_tag.get(g, ""),
                )
                gather_end = max((ev.cycle + ev.duration for ev in gather_evts), default=group_start)
            else:
                gather_evts, eid, gather_end = gen_feature_gather_events(
                    query_idx=g,
                    k=group_sizes[g],
                    in_channels=knn_in_channels,
                    hw_config=hw_config,
                    start_cycle=group_start,
                    start_event_id=eid,
                    bytes_per_element=bpe,
                    coalescing_factor=coalescing,
                )
            for ev in gather_evts:
                ev.metadata["pipeline"] = "knn_conv"
                ev.metadata["pipeline_mode"] = pipeline_mode
                if num_slices > 1:
                    ev.metadata["slice_id"] = s
            all_new_events.extend(gather_evts)

            # 2) Pointwise Conv per-group events
            # Pass neurosim_full-derived per-group override when available (Gap B fix).
            _pg_lat = _neurosim_per_group_lat[g] if _neurosim_per_group_lat is not None else None
            _pg_en = _neurosim_per_group_en[g] if _neurosim_per_group_en is not None else None
            conv_evts, eid, conv_end = gen_pointwise_conv_per_group_events(
                conv_params,
                hw_config,
                group_points=group_sizes[g],
                query_idx=g,
                start_cycle=gather_end,
                start_event_id=eid,
                operator_kind=str(getattr(conv_ref, "op_type", "conv1d")),
                _template_cache=_conv_template_cache if _pg_lat is None else None,
                precomputed_lat_ns=_pg_lat,
                precomputed_en_nj=_pg_en,
            )
            for ev in conv_evts:
                ev.metadata["pipeline_mode"] = pipeline_mode
                ev.metadata["group_size"] = group_sizes[g]
                if gather_dep_tag:
                    deps = list(ev.metadata.get("depends_on_tags", []) or [])
                    deps.append(gather_dep_tag)
                    ev.metadata["depends_on_tags"] = deps
                if num_slices > 1:
                    ev.metadata["slice_id"] = s
            all_new_events.extend(conv_evts)

            prev_end_in_slice = conv_end

    return preceding_events + all_new_events


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
        context: Optional dict, e.g. cim_layer_latency_ns, cim_layer_energy_nj, cim_layer_index for conv1d/linear.

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


def _validate_channel_semantics(layer_operators: List['OperatorRef']) -> None:
    """
    校验算子链的通道语义一致性。

    检查规则：
    - SA MLP 首层 Conv2d 的 channel_semantics.relative_xyz + grouped_features == in_channels
    - GAM affine_norm 的 base_features + xyz_appended == num_features
    - GAM spool 的 anchor + neighbor + xyz == in_channels
    """
    import logging
    _log = logging.getLogger("pointkan_sim.dispatcher")
    for idx, op in enumerate(layer_operators):
        params = getattr(op, "params", {}) or {}
        sem = params.get("channel_semantics")
        if sem is None:
            continue
        op_type = str(getattr(op, "op_type", ""))

        if op_type in ("conv1d", "conv2d") and sem.get("source") == "sample_and_group":
            in_ch = int(params.get("in_channels", 0))
            rel_xyz = int(sem.get("relative_xyz", 0))
            feat = int(sem.get("grouped_features", 0))
            if in_ch != rel_xyz + feat:
                _log.warning(
                    "SA Conv2d[%d] channel mismatch: in_channels=%d != "
                    "relative_xyz(%d) + grouped_features(%d)",
                    idx, in_ch, rel_xyz, feat,
                )
            if not sem.get("has_relative_xyz"):
                _log.warning(
                    "SA Conv2d[%d] missing relative xyz subtraction "
                    "(sample_and_group should compute grouped_xyz - new_xyz)",
                    idx,
                )

        if op_type == "affine_norm" and sem.get("source") == "gam_affine_norm":
            total = int(sem.get("total", 0))
            base = int(sem.get("base_features", 0))
            xyz_app = int(sem.get("xyz_appended", 0))
            if total != base + xyz_app:
                _log.warning(
                    "GAM AffineNorm[%d] channel mismatch: total=%d != "
                    "base_features(%d) + xyz_appended(%d)",
                    idx, total, base, xyz_app,
                )

        if op_type == "s-pool" and sem.get("source") == "gam_spool":
            total = int(sem.get("total", 0))
            anchor = int(sem.get("anchor_features", 0))
            neighbor = int(sem.get("neighbor_features", 0))
            xyz_app = int(sem.get("xyz_appended", 0))
            if total != anchor + neighbor + xyz_app:
                _log.warning(
                    "GAM S-Pool[%d] channel mismatch: total=%d != "
                    "anchor(%d) + neighbor(%d) + xyz(%d)",
                    idx, total, anchor, neighbor, xyz_app,
                )


def dispatch_layer(
    layer_operators: List['OperatorRef'],
    hw_config: 'HardwareConfig',
    context: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """
    Dispatch all operators in a layer sequentially.

    - BN 融合：若 bn_fuse_with_conv_linear 且上一算子是 conv1d/linear，本层 BN 不计 cycle/能耗。
    - 其余同前：cycle 偏移、全网络 CIM 注入上下文（cim_* 键名，见 cim_full_context）等。
    """
    _validate_channel_semantics(layer_operators)
    all_events: List[Event] = []
    current_cycle = 0
    prev_op_type: Optional[str] = None

    i = 0
    while i < len(layer_operators):
        op_ref = layer_operators[i]
        # BN 紧跟 Conv/Linear 时可选融合，不计 BN 开销
        if getattr(hw_config, "bn_fuse_with_conv_linear", False) and prev_op_type in ("conv1d", "conv2d", "linear"):
            if op_ref.op_type in ("bn", "batchnorm1d"):
                prev_op_type = op_ref.op_type
                i += 1
                continue
        # PointCIM fused FPS+BQ/KNN: RRAM 距离复用 + BQ 叶引擎 streaming + MDT 并行
        # 适用于 PointNet++ SA (fps+ballquery) 和 PointKAN GAM (fps+knn)
        # 可选第二级流水线：每个 query 的 BQ 完成后立即驱动后续 Conv2d(kernel_size=1)
        if (getattr(hw_config, "use_rram_dist", False)
                and getattr(hw_config, "is_acc", False)
                and op_ref.op_type == "fps"
                and i + 1 < len(layer_operators)
                and layer_operators[i + 1].op_type in ("ballquery", "knn")):
            fused_events = _gen_pointcim_fused_fps_bq(
                op_ref, layer_operators[i + 1], hw_config, context,
            )
            consumed = 2  # FPS + BallQuery/KNN
            # sample_and_group gather: BQ/KNN 索引就绪 → xyz/feature gather → relative xyz → concat
            # 仅当下一算子为 Conv2d(k=1) 时插入（PointNet++ SA 模式；PointKAN GAM 由 affine_norm 内部处理）
            _next_is_conv = (i + 2 < len(layer_operators)
                             and layer_operators[i + 2].op_type in ("conv1d", "conv2d")
                             and int(layer_operators[i + 2].params.get("kernel_size", 1)) == 1)
            if _next_is_conv and not getattr(hw_config, "pointcim_sa_conv_pipeline", False):
                from .common_ops import gen_sample_and_group_events
                _fps_p = op_ref.params
                _bq_p = layer_operators[i + 1].params
                _conv_p = layer_operators[i + 2].params
                _sam_pts = int(_fps_p.get("output_points", 512))
                _bq_k_val = int(_bq_p.get("k", 32))
                _conv_in_ch = int(_conv_p.get("in_channels", 3))
                _xyz_ch = 3
                _feat_ch = max(0, _conv_in_ch - _xyz_ch)
                _fused_span = max((e.cycle + e.duration for e in fused_events), default=0)
                _gather_evs, _g_eid, _g_cyc, _gather_tag = gen_sample_and_group_events(
                    sampled_points=_sam_pts,
                    k=_bq_k_val,
                    xyz_channels=_xyz_ch,
                    feature_channels=_feat_ch,
                    hw_config=hw_config,
                    start_cycle=_fused_span,
                    start_event_id=max((e.event_id for e in fused_events), default=0) + 1,
                    tag_scope=f"sa_gather_l{context.get('layer_index', 0)}_",
                    current_layer_name=str((context or {}).get("layer_name", "") or ""),
                )
                fused_events.extend(_gather_evs)
            # PointCIM SA Conv pipeline: BQ/KNN query-i finish -> Conv2d group-i start
            if (getattr(hw_config, "pointcim_sa_conv_pipeline", False) and _next_is_conv):
                bq_params_i = layer_operators[i + 1].params
                fps_params_i = op_ref.params
                _sampled_pts = int(fps_params_i.get("output_points", 512))
                _bq_k = int(bq_params_i.get("k", 32))
                _in_ch = int(layer_operators[i + 2].params.get("in_channels", 3))
                fused_events = _pipeline_conv_after_knn(
                    fused_events, layer_operators[i + 2],
                    _sampled_pts, hw_config, context,
                    group_size_hint=_bq_k,
                    knn_in_channels=_in_ch,
                    force_strict=True,
                    use_sample_and_group=True,
                )
                consumed = 3  # FPS + BallQuery/KNN + Conv2d
            for e in fused_events:
                e.cycle += current_cycle
            all_events.extend(fused_events)
            span = 0
            if fused_events:
                span = max(e.cycle + e.duration - current_cycle for e in fused_events)
            current_cycle += span
            prev_op_type = layer_operators[i + consumed - 1].op_type
            i += consumed
            continue
        # PointNet++ SA: FPS -> BallQuery/KNN -> Conv(pointwise) strict producer-consumer pipeline
        if (str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() != "disabled"
                and op_ref.op_type == "fps"
                and i + 1 < len(layer_operators)
                and layer_operators[i + 1].op_type in ("knn", "ballquery")):
            pipelined_events = _dispatch_fps_knn_pipelined(
                op_ref, layer_operators[i + 1], hw_config, context,
            )
            consumed = 2  # FPS + KNN
            # sample_and_group gather (非 fused 路径)
            _pipe_next_conv = (i + 2 < len(layer_operators)
                               and layer_operators[i + 2].op_type in ("conv1d", "conv2d")
                               and int(layer_operators[i + 2].params.get("kernel_size", 1)) == 1)
            if _pipe_next_conv and str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() == "disabled":
                from .common_ops import gen_sample_and_group_events
                _ps = layer_operators[i + 1].params
                _pip_sam = int(op_ref.params.get("output_points", int(_ps.get("query_points", 512))))
                _pip_k = int(_ps.get("k", 16))
                _pip_conv_in = int(layer_operators[i + 2].params.get("in_channels", 3))
                _pip_feat_ch = max(0, _pip_conv_in - 3)
                _pip_span = max((e.cycle + e.duration for e in pipelined_events), default=0)
                _gev, _, _, _ = gen_sample_and_group_events(
                    sampled_points=_pip_sam, k=_pip_k,
                    xyz_channels=3, feature_channels=_pip_feat_ch,
                    hw_config=hw_config,
                    start_cycle=_pip_span,
                    start_event_id=max((e.event_id for e in pipelined_events), default=0) + 1,
                    tag_scope=f"sa_gather_pipe_l{context.get('layer_index', 0)}_",
                    current_layer_name=str((context or {}).get("layer_name", "") or ""),
                )
                pipelined_events.extend(_gev)
            # 第二级流水线：search -> Conv(kernel_size=1 时各 query 独立，如 PointNet++ SA)
            if (str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() != "disabled"
                    and _pipe_next_conv):
                search_params = layer_operators[i + 1].params
                _query_pts = int(search_params.get('query_points', 512))
                _k = int(search_params.get('k', 16))
                _in_ch = int(layer_operators[i + 2].params.get("in_channels", 3))
                pipelined_events = _pipeline_conv_after_knn(
                    pipelined_events, layer_operators[i + 2],
                    _query_pts,
                    hw_config, context,
                    group_size_hint=_k,
                    knn_in_channels=_in_ch,
                    use_sample_and_group=True,
                )
                consumed = 3  # FPS + KNN + Conv1d
            for e in pipelined_events:
                e.cycle += current_cycle
            all_events.extend(pipelined_events)
            span = 0
            if pipelined_events:
                span = max(e.cycle + e.duration - current_cycle for e in pipelined_events)
            current_cycle += span
            prev_op_type = layer_operators[i + consumed - 1].op_type
            i += consumed
            continue
        # Standalone search->Conv pipeline（如 PointNet++ FP/GroupAll path）
        if (str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() != "disabled"
                and op_ref.op_type in ("knn", "ballquery")
                and i + 1 < len(layer_operators)
                and layer_operators[i + 1].op_type in ("conv1d", "conv2d")
                and int(layer_operators[i + 1].params.get("kernel_size", 1)) == 1):
            knn_events = dispatch_operator(op_ref, hw_config, context)
            knn_params = op_ref.params
            _query_pts = int(knn_params.get('query_points', 512))
            _k = int(knn_params.get('k', 16))
            _in_ch = int(layer_operators[i + 1].params.get("in_channels", 3))
            pipelined_events = _pipeline_conv_after_knn(
                knn_events, layer_operators[i + 1],
                _query_pts,
                hw_config, context,
                group_size_hint=_k,
                knn_in_channels=_in_ch,
            )
            for e in pipelined_events:
                e.cycle += current_cycle
            all_events.extend(pipelined_events)
            span = 0
            if pipelined_events:
                span = max(e.cycle + e.duration - current_cycle for e in pipelined_events)
            current_cycle += span
            prev_op_type = layer_operators[i + 1].op_type
            i += 2
            continue
        # FP decoder pattern (new): KNN + InterpN3 + Conv1d
        if (str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() != "disabled"
                and op_ref.op_type == "knn"
                and i + 2 < len(layer_operators)
                and layer_operators[i + 1].op_type == "interp_n3"
                and layer_operators[i + 2].op_type == "conv1d"):
            knn_events = dispatch_operator(op_ref, hw_config, context)
            knn_query_end: Dict[int, int] = {}
            for e in knn_events:
                q = (e.metadata or {}).get("query")
                if q is not None:
                    knn_query_end[q] = max(knn_query_end.get(q, 0), e.cycle + e.duration)
            knn_params = op_ref.params
            _query_pts = int(knn_params.get('query_points', 512))
            _k = int(knn_params.get('k', 3))
            _in_ch = int(layer_operators[i + 2].params.get("in_channels", 3))
            interp_events, _eid, _query_interp_end = gen_interp_n3_per_query_events(
                layer_operators[i + 1].params,
                hw_config,
                _query_pts,
                knn_query_end,
                start_event_id=max((e.event_id for e in knn_events), default=-1) + 1,
            )
            combined = knn_events + interp_events
            pipelined_events = _pipeline_conv_after_knn(
                combined, layer_operators[i + 2],
                _query_pts,
                hw_config, context,
                group_size_hint=_k,
                knn_in_channels=_in_ch,
            )
            for e in pipelined_events:
                e.cycle += current_cycle
            all_events.extend(pipelined_events)
            span = 0
            if pipelined_events:
                span = max(e.cycle + e.duration - current_cycle for e in pipelined_events)
            current_cycle += span
            prev_op_type = layer_operators[i + 2].op_type
            i += 3  # KNN + InterpN3 + Conv1d
            continue
        # FP decoder pattern (legacy): KNN(k=3) + div + mul + add + Conv1d (backward compat)
        if (str(getattr(hw_config, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() != "disabled"
                and op_ref.op_type == "knn"
                and i + 4 < len(layer_operators)
                and layer_operators[i + 1].op_type == "div"
                and layer_operators[i + 2].op_type == "mul"
                and layer_operators[i + 3].op_type == "add"
                and layer_operators[i + 4].op_type == "conv1d"):
            knn_events = dispatch_operator(op_ref, hw_config, context)
            interp_events_legacy: List[Event] = []
            interp_cycle_l = max((e.cycle + e.duration for e in knn_events), default=0)
            for j in range(1, 4):  # div, mul, add
                ievts = dispatch_operator(layer_operators[i + j], hw_config, context)
                for e in ievts:
                    e.cycle += interp_cycle_l
                interp_events_legacy.extend(ievts)
                if ievts:
                    interp_cycle_l = max(e.cycle + e.duration for e in ievts)
            combined = knn_events + interp_events_legacy
            knn_params = op_ref.params
            _query_pts = int(knn_params.get('query_points', 512))
            _k = int(knn_params.get('k', 3))
            _in_ch = int(layer_operators[i + 4].params.get("in_channels", 3))
            pipelined_events = _pipeline_conv_after_knn(
                combined, layer_operators[i + 4],
                _query_pts,
                hw_config, context,
                group_size_hint=_k,
                knn_in_channels=_in_ch,
            )
            for e in pipelined_events:
                e.cycle += current_cycle
            all_events.extend(pipelined_events)
            span = 0
            if pipelined_events:
                span = max(e.cycle + e.duration - current_cycle for e in pipelined_events)
            current_cycle += span
            prev_op_type = layer_operators[i + 4].op_type
            i += 5  # KNN + div + mul + add + Conv1d
            continue
        # ACC stage0 expects FPS/KNN to stream from the same distance frontend.
        # Keep both operators anchored to the same layer offset instead of forcing
        # KNN to wait for the entire FPS query stream to finish.
        if (getattr(hw_config, "is_acc", False)
                and op_ref.op_type == "fps"
                and i + 1 < len(layer_operators)
                and layer_operators[i + 1].op_type == "knn"):
            fps_events = dispatch_operator(op_ref, hw_config, context)
            knn_events = dispatch_operator(layer_operators[i + 1], hw_config, context)
            paired_events = fps_events + knn_events
            for e in paired_events:
                e.cycle += current_cycle
            all_events.extend(paired_events)
            if paired_events:
                current_cycle = max(e.cycle + e.duration for e in paired_events)
            prev_op_type = layer_operators[i + 1].op_type
            i += 2
            continue

        events = dispatch_operator(op_ref, hw_config, context)
        for e in events:
            e.cycle += current_cycle
        all_events.extend(events)
        if events:
            current_cycle = max(e.cycle + e.duration for e in events)
        prev_op_type = op_ref.op_type
        i += 1

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
