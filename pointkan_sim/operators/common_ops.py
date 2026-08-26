"""
通用算子事件生成（Common Ops）
===============================

Conv1d/Linear 的量化/反量化/BN/LayerNorm/ReLU/SPool 等逻辑电路事件生成，
以及 PE 级 Conv1d/Linear 的 RRAM-CIM 事件（无全网络 NeuroSim 注入时的单层 NeuroSim/回退估算）。
算子名统一用 PascalCase（Conv1d, Linear, BN 等），与 constants.OP_NAME_* 一致。
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..simulator.event import Event
from ..utils.math_utils import safe_div_ceil
from ..utils.shared_helpers import retag_events as _retag_events
from ..integration.memory_model_adapter import query_dram_issue_cycles
from .neurosim_adapter import (
    NeuroSimEstimate,
    estimate_conv1d_neurosim,
    estimate_linear_neurosim,
)

from ..integration.memory_model_adapter import (
    query_dram_batch_cycles_energy,
    query_dram_cycles_energy,
    query_sram_cycles_energy,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def _clock(hw_config: "HardwareConfig") -> float:
    """从 hw_config 取时钟周期 (ns)。"""
    return float(getattr(hw_config, "clock_period", 1.0))


def _neurosim_clock_ns(hw_config: "HardwareConfig") -> float:
    """NeuroSim 相关事件使用的时钟周期 (ns)。"""
    freq_hz = getattr(hw_config, "neurosim_clk_freq_hz", None)
    if freq_hz is not None and float(freq_hz) > 0:
        return 1e9 / float(freq_hz)
    return _clock(hw_config)


def _use_fixed_neurosim_subarray(hw_config: "HardwareConfig") -> bool:
    return bool(getattr(hw_config, "is_acc", False) or getattr(hw_config, "is_pointcim", False))

def _normalized_scheduler_timing(*, latency_ns: float, energy_nj: float, base_clock_ns: float) -> Dict[str, float]:
    """
    将任意时钟域的 latency/energy 规约到调度器的 base_clock_ns 上，得到 duration_cycles 与等效功耗。

    说明：
      - 当事件本身用独立时钟（如 NeuroSim 的 Conv/Linear）时，不应使用该规约函数。
      - 仅当需要把某段时间/能耗折算到数字逻辑时钟进行统一调度时使用。
    """
    base_clk = max(1e-9, float(base_clock_ns))
    lat_ns = max(base_clk, float(latency_ns))
    dur = max(1, int(math.ceil(lat_ns / base_clk)))
    power = float(energy_nj) * 1000.0 / (dur * base_clk) if energy_nj > 0 else 0.0
    return {"duration_cycles": float(dur), "clock_period_ns": float(base_clk), "power_mw": float(power)}

def _logic_power(hw_config: "HardwareConfig", key: str, default_power: float) -> float:
    """从 hw_config.logic_circuit.<key> 取 power (mW)。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    p = getattr(sub, "power", default_power) if sub is not None else default_power
    return float(p if p is not None else default_power)


def _logic_cycles(hw_config: "HardwareConfig", key: str, default_cycles: int) -> int:
    """从 hw_config.logic_circuit.<key> 取 cycles。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    c = getattr(sub, "cycles", default_cycles) if sub is not None else default_cycles
    return int(c if c is not None else default_cycles)


def _logic_parallelism(hw_config: "HardwareConfig", key: str, default_parallelism: int = 1) -> int:
    """从 hw_config.logic_circuit.<key> 取并行度。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    par = getattr(sub, "parallelism", default_parallelism) if sub is not None else default_parallelism
    return max(1, int(par if par is not None else default_parallelism))


def _logic_cycles_per_element(hw_config: "HardwareConfig", key: str, default: int = 1) -> int:
    """从 hw_config.logic_circuit.<key> 取每元素周期数（如 fp32_add/fp32_mul=1，fp32_div=10）。"""
    logic = getattr(hw_config, "logic_circuit", None)
    sub = getattr(logic, key, None) if logic is not None else None
    c = getattr(sub, "cycles_per_element", default) if sub is not None else default
    return max(1, int(c if c is not None else default))


def _prod(values: List[int]) -> int:
    out = 1
    for v in values:
        out *= max(1, int(v))
    return out


def _event(
    event_id: int,
    cycle: int,
    operator: str,
    stage: str,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    power: float,
    duration: int,
    clock_period_ns: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """构造一条仿真事件（operator 建议用 PascalCase，如 Conv1d, BN）。"""
    return Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator,
        stage=stage,
        sub_stage=sub_stage,
        circuit_module=circuit_module,
        operation=operation,
        power=float(power),
        duration=max(1, int(duration)),
        clock_period_ns=clock_period_ns,
        metadata=metadata or {},
    )


def _energy_to_power_mw(energy_nj: float, duration_cycles: int, clock_period_ns: float) -> float:
    dur = max(1, int(duration_cycles))
    clk = float(clock_period_ns) if clock_period_ns else 1.0
    if energy_nj <= 0 or dur <= 0 or clk <= 0:
        return 0.0
    return float(energy_nj) * 1000.0 / (dur * clk)


@lru_cache(maxsize=64)
def _destiny_rram_read_params(
    repo_root: str,
    capacity_kb: int,
    word_width_bits: int,
    optimization_target: str,
) -> Dict[str, float]:
    destiny_dir = Path(repo_root) / "hw_sim" / "cycle_simulator" / "destiny_3d_cache"
    if str(destiny_dir) not in sys.path:
        sys.path.insert(0, str(destiny_dir))
    from destiny_runner import DestinyInput, DestinyRunner  # type: ignore

    result = DestinyRunner().run(DestinyInput(
        capacity=max(1, int(capacity_kb)),
        capacity_unit="KB",
        word_width_bits=max(1, int(word_width_bits)),
        optimization_target=str(optimization_target or "ReadEDP"),
    ))
    if result.read_latency_ns is None or result.read_energy_pj is None:
        raise RuntimeError("DESTINY did not return valid RRAM read latency/energy")
    return {
        "read_latency_ns": float(result.read_latency_ns),
        "read_energy_pj": float(result.read_energy_pj),
        "area_mm2": float(result.area_mm2 or 0.0),
        "leakage_power_uw": float(result.leakage_power_uw or 0.0),
    }


def _query_rram_lut_read_cycles_energy(
    num_bytes: int,
    hw_config: "HardwareConfig",
    lut_cfg: Any,
    params: Dict[str, Any],
    clk_ns: float,
) -> Dict[str, Any]:
    word_bits = max(1, int(params.get("rram_word_width_bits", getattr(lut_cfg, "rram_word_width_bits", 128) if lut_cfg is not None else 128)))
    word_bytes = max(1, int(math.ceil(word_bits / 8.0)))
    accesses = max(1, safe_div_ceil(max(1, int(num_bytes)), word_bytes))
    parallel = max(1, int(params.get("rram_read_parallelism", getattr(lut_cfg, "rram_read_parallelism", 1) if lut_cfg is not None else 1)))
    read_latency_ns = params.get("rram_read_latency_ns", getattr(lut_cfg, "rram_read_latency_ns", None) if lut_cfg is not None else None)
    read_energy_pj = params.get("rram_read_energy_pj", getattr(lut_cfg, "rram_read_energy_pj", None) if lut_cfg is not None else None)
    source = "config"
    extra: Dict[str, Any] = {}
    if read_latency_ns is None or read_energy_pj is None:
        repo_root = str(getattr(hw_config, "repo_root", "") or params.get("repo_root", "") or "/workspace/PointKAN-pytorch")
        capacity_kb = max(1, int(params.get("rram_capacity_kb", getattr(lut_cfg, "rram_capacity_kb", 1) if lut_cfg is not None else 1)))
        opt = str(params.get("rram_optimization_target", getattr(lut_cfg, "rram_optimization_target", "ReadEDP") if lut_cfg is not None else "ReadEDP"))
        destiny = _destiny_rram_read_params(repo_root, capacity_kb, word_bits, opt)
        read_latency_ns = destiny["read_latency_ns"]
        read_energy_pj = destiny["read_energy_pj"]
        source = "destiny"
        extra = dict(destiny)
        extra["capacity_kb"] = capacity_kb
        extra["optimization_target"] = opt

    cycles_per_access = max(1, int(math.ceil(float(read_latency_ns) / max(float(clk_ns), 1e-12))))
    read_batches = safe_div_ceil(accesses, parallel)
    cycles = max(1, read_batches * cycles_per_access)
    energy_nj = accesses * float(read_energy_pj) / 1000.0
    return {
        "cycles": cycles,
        "energy_nj": energy_nj,
        "power_mw": _energy_to_power_mw(energy_nj, cycles, clk_ns),
        "accesses": accesses,
        "word_width_bits": word_bits,
        "word_bytes": word_bytes,
        "parallelism": parallel,
        "read_latency_ns": float(read_latency_ns),
        "read_energy_pj": float(read_energy_pj),
        "source": source,
        **extra,
    }


def _named_sram_io_event(
    event_id: int,
    cycle: int,
    operator: str,
    sub_stage: str,
    operation: str,
    buffer_name: str,
    num_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clk: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    cycles, energy = query_sram_cycles_energy(
        max(1, int(num_bytes)),
        is_read=is_read,
        hw_config=hw_config,
        clock_period_ns=clk,
        buffer_name=buffer_name,
    )
    meta = dict(metadata or {})
    meta.update({
        "bytes": max(1, int(num_bytes)),
        "buffer_name": buffer_name,
        "buffer_phase": "read" if is_read else "write",
    })
    return _event(
        event_id,
        cycle,
        operator,
        sub_stage,
        "sram_read" if is_read else "sram_write",
        buffer_name,
        operation,
        _energy_to_power_mw(float(energy), int(cycles), clk),
        int(cycles),
        clk,
        meta,
    )


def _interp_n3_engine_cfg(hw_config: "HardwareConfig") -> Dict[str, Any]:
    logic = getattr(hw_config, "logic_circuit", None)
    cfg = getattr(logic, "interp_n3_engine", None) if logic is not None else None
    feature_parallelism = getattr(cfg, "feature_parallelism", getattr(hw_config, "interp_engine_parallelism", 16)) if cfg is not None else getattr(hw_config, "interp_engine_parallelism", 16)
    return {
        "power": float(getattr(cfg, "power", 68.0) if cfg is not None else 68.0),
        "area": float(getattr(cfg, "area", 240294.0) if cfg is not None else 240294.0),
        "weight_cycles": max(1, int(getattr(cfg, "cycles_per_query_weight", 22) if cfg is not None else 22)),
        "feature_parallelism": max(1, int(feature_parallelism or 16)),
        "weighted_cycles_numerator": max(1, int(getattr(cfg, "weighted_cycles_numerator", 4) if cfg is not None else 4)),
    }


def _merge_half_open_intervals(intervals: List[tuple]) -> List[tuple]:
    if not intervals:
        return []
    merged: List[tuple] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s)):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _build_interp_n3_engine_events(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    num_queries: int,
    query_ready_cycles: Dict[int, int],
    start_event_id: int = 0,
    *,
    include_query_markers: bool = False,
) -> tuple:
    nq = max(0, int(num_queries or 0))
    if nq <= 0:
        return ([], start_event_id, {})

    D2 = int(params.get("src_channels", params.get("num_features", 1)))
    k = int(params.get("k", 3))
    clk = _clock(hw_config)
    cfg = _interp_n3_engine_cfg(hw_config)
    gather_bytes = max(1, k * D2 * 4)
    gather_results = query_dram_batch_cycles_energy(
        [{"num_bytes": gather_bytes, "is_read": True, "start_cycle": 0} for _ in range(nq)],
        hw_config,
        clk,
    )
    gather_dur = max(1, int(gather_results[0][0])) if gather_results else 0
    gather_energy = float(gather_results[0][1]) if gather_results else 0.0
    gather_power = (gather_energy * 1e3 / (gather_dur * clk)) if (gather_dur > 0 and clk > 0) else 0.0
    weight_cycles = cfg["weight_cycles"]
    feature_cycles = max(1, safe_div_ceil(cfg["weighted_cycles_numerator"] * max(1, D2), cfg["feature_parallelism"]))
    initiation_interval = max(weight_cycles, feature_cycles)
    fallback_ready = max(query_ready_cycles.values()) if query_ready_cycles else 0

    gather_specs: List[tuple] = []
    stage1_specs: List[tuple] = []
    stage2_specs: List[tuple] = []
    query_end_cycles: Dict[int, int] = {}
    prev_gather_end = 0
    prev_stage1_end = 0
    prev_stage2_end = 0

    for q in range(nq):
        if gather_results and q < len(gather_results):
            gather_dur = max(1, int(gather_results[q][0]))
            gather_energy = float(gather_results[q][1])
            gather_power = (gather_energy * 1e3 / (gather_dur * clk)) if (gather_dur > 0 and clk > 0) else 0.0
        ready = int(query_ready_cycles.get(q, fallback_ready) or 0)
        gather_start = max(ready, prev_gather_end)
        gather_end = gather_start + gather_dur
        prev_gather_end = gather_end

        stage1_start = max(ready, prev_stage1_end)
        stage1_end = stage1_start + weight_cycles
        prev_stage1_end = stage1_end

        stage2_start = max(gather_end, stage1_end, prev_stage2_end)
        stage2_end = stage2_start + feature_cycles
        prev_stage2_end = stage2_end

        gather_specs.append((q, gather_start, gather_end))
        stage1_specs.append((q, stage1_start, stage1_end))
        stage2_specs.append((q, stage2_start, stage2_end))
        query_end_cycles[q] = stage2_end

    busy_intervals = _merge_half_open_intervals(
        [(s, e) for _q, s, e in stage1_specs] + [(s, e) for _q, s, e in stage2_specs]
    )

    events: List[Event] = []
    eid = start_event_id
    common_meta = {
        "N": nq,
        "k": k,
        "D2": D2,
        "weight_cycles_per_query": weight_cycles,
        "feature_cycles_per_query": feature_cycles,
        "feature_parallelism": cfg["feature_parallelism"],
        "initiation_interval": initiation_interval,
    }

    for q, start, end in gather_specs:
        meta = dict(common_meta)
        meta.update({
            "query": q,
            "bytes": gather_bytes,
            "buffer_name": "offchip_dram",
            "buffer_phase": "feature_gather",
            "resource_id": "dram_port",
            "pipeline": "knn_interp_conv",
        })
        ev = _event(
            eid, start, "InterpN3", "gather", "index_points_read", "dram",
            "interp_gather_read", gather_power, end - start, clk, meta,
        )
        ev.overlapped = True
        events.append(ev)
        eid += 1

    for idx, (start, end) in enumerate(busy_intervals):
        meta = dict(common_meta)
        meta.update({
            "busy_segment": idx,
            "engine_model": "interp_n3_engine",
        })
        events.append(_event(
            eid, start, "InterpN3", "interpolate", "engine_busy", "interp_n3_engine",
            "interp_engine_busy", cfg["power"], end - start, clk, meta,
        ))
        eid += 1

    if include_query_markers:
        for q, start, end in stage2_specs:
            meta = dict(common_meta)
            meta.update({
                "query": q,
                "pipeline": "knn_interp_conv",
                "marker": "query_complete",
            })
            ev = _event(
                eid, max(0, end - 1), "InterpN3", "interpolate", "query_done", "interp_n3_engine",
                "interp_query_complete", 0.0, 1, clk, meta,
            )
            ev.overlapped = True
            events.append(ev)
            eid += 1

    return (events, eid, query_end_cycles)


def _scalesim_memory_events(
    base_cycle: int,
    start_event_id: int,
    op_name: str,
    clk: float,
    hw_config: "HardwareConfig",
    meta: Dict[str, Any],
) -> List[Event]:
    events: List[Event] = []

    def _append_mem_event(
        event_id: int,
        operator: str,
        sub_stage: str,
        circuit_module: str,
        operation: str,
        energy_nj: float,
        cycles: int,
        clock_period_ns: float,
        extra_meta: Dict[str, Any],
    ) -> int:
        if energy_nj <= 0.0 or cycles <= 0:
            return event_id
        events.append(_event(
            event_id,
            base_cycle,
            operator,
            "buffer_io",
            sub_stage,
            circuit_module,
            operation,
            _energy_to_power_mw(energy_nj, cycles, clock_period_ns),
            cycles,
            clock_period_ns,
            {**meta, **extra_meta, "source": "scalesim_memory"},
        ))
        return event_id + 1

    event_id = start_event_id

    feature_local_sram_energy = float(meta.get("feature_local_sram_energy_nj") or 0.0)
    feature_local_sram_cycles = int(meta.get("feature_local_sram_cycles") or 0)
    feature_local_sram_clk = float(meta.get("scalesim_sram_clock_period_ns") or clk)
    feature_local_sram_bytes = int(meta.get("feature_local_sram_bytes_read", 0) or 0) + int(meta.get("feature_local_sram_bytes_write", 0) or 0)
    event_id = _append_mem_event(
        event_id,
        "LayerBuffer",
        "scalesim_feature_local_sram_access",
        "systolic_feature_sram",
        f"{op_name.lower()}_scalesim_feature_local_sram_access",
        feature_local_sram_energy,
        max(1, feature_local_sram_cycles) if feature_local_sram_energy > 0 else 0,
        feature_local_sram_clk,
        {
            "memory_kind": "sram",
            "buffer_name": "systolic_feature_sram_buffer",
            "buffer_phase": "compute_local",
            "bytes": feature_local_sram_bytes,
            "owner_operator": op_name,
            "owner_operator": op_name,
            "resource_id": "systolic_feature_sram_shared",
            "exclusive": True,
            "reset_scope": "operator_boundary",
        },
    )

    weight_sram_energy = float(meta.get("weight_sram_energy_nj") or 0.0)
    weight_sram_cycles = int(meta.get("weight_sram_cycles") or 0)
    weight_sram_clk = float(meta.get("scalesim_sram_clock_period_ns") or clk)
    weight_sram_bytes_read = int(meta.get("weight_sram_bytes_read", 0) or 0)
    weight_sram_bytes_write = int(meta.get("weight_sram_bytes_write", 0) or 0)
    weight_refill_tiles = max(1, int(meta.get("weight_refill_tiles", 1) or 1))
    weight_refill_tile_bytes = int(meta.get("weight_refill_tile_bytes", 0) or 0)
    if weight_sram_energy > 0.0 and weight_sram_cycles > 0:
        per_tile_write = weight_refill_tile_bytes if weight_refill_tile_bytes > 0 else int(math.ceil(weight_sram_bytes_write / float(weight_refill_tiles)))
        per_tile_energy = weight_sram_energy / float(weight_refill_tiles)
        per_tile_cycles = max(1, int(math.ceil(weight_sram_cycles / float(weight_refill_tiles))))
        remaining_read = weight_sram_bytes_read
        base_read = int(weight_sram_bytes_read // weight_refill_tiles)
        read_rem = int(weight_sram_bytes_read % weight_refill_tiles)
        for tile_idx in range(weight_refill_tiles):
            tile_read = base_read + (1 if tile_idx < read_rem else 0)
            tile_write = min(per_tile_write, max(0, weight_sram_bytes_write - tile_idx * per_tile_write))
            event_id = _append_mem_event(
                event_id,
                "LayerBuffer",
                "scalesim_weight_sram_access",
                "weight_sram_buffer",
                f"{op_name.lower()}_scalesim_weight_sram_access",
                per_tile_energy,
                per_tile_cycles,
                weight_sram_clk,
                {
                    "memory_kind": "sram",
                    "buffer_name": "weight_sram_buffer",
                    "buffer_phase": "weight_stage",
                    "bytes": tile_read + tile_write,
                    "owner_operator": op_name,
                    "owner_operator": op_name,
                    "bytes_read": tile_read,
                    "bytes_write": tile_write,
                    "tile_index": tile_idx,
                    "tile_count": weight_refill_tiles,
                    "resource_id": "weight_sram_shared",
                    "exclusive": True,
                    "reset_scope": "weight_tile_boundary",
                },
            )

    dram_clk = float(meta.get("scalesim_dram_clock_period_ns") or clk)
    dram_parts = [
        (
            "feature_input_read",
            float(meta.get("feature_input_dram_energy_nj") or 0.0),
            int(meta.get("feature_input_dram_cycles") or 0),
            int(meta.get("feature_input_dram_bytes", 0) or 0),
            meta.get("feature_input_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_feature_input_dram_access",
            1,
            int(meta.get("feature_input_dram_bytes", 0) or 0),
        ),
        (
            "weight_read",
            float(meta.get("weight_dram_energy_nj") or 0.0),
            int(meta.get("weight_dram_cycles") or 0),
            int(meta.get("weight_dram_bytes", 0) or 0),
            meta.get("weight_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_weight_dram_access",
            weight_refill_tiles,
            weight_refill_tile_bytes,
        ),
        (
            "feature_output_write",
            float(meta.get("feature_output_dram_energy_nj") or 0.0),
            int(meta.get("feature_output_dram_cycles") or 0),
            int(meta.get("feature_output_dram_bytes", 0) or 0),
            meta.get("feature_output_dram_trace_dir", ""),
            f"{op_name.lower()}_scalesim_feature_output_dram_access",
            1,
            int(meta.get("feature_output_dram_bytes", 0) or 0),
        ),
    ]
    for phase, energy_nj, cycles, num_bytes, trace_dir, operation, tile_count, tile_bytes in dram_parts:
        if energy_nj <= 0.0 or cycles <= 0 or num_bytes <= 0:
            continue
        per_tile_energy = energy_nj / float(max(1, tile_count))
        per_tile_cycles = max(1, int(math.ceil(cycles / float(max(1, tile_count)))))
        for tile_idx in range(max(1, tile_count)):
            this_tile_bytes = tile_bytes if tile_bytes > 0 else int(math.ceil(num_bytes / float(max(1, tile_count))))
            if tile_count > 1:
                remaining = max(0, num_bytes - tile_idx * this_tile_bytes)
                this_tile_bytes = min(this_tile_bytes, remaining)
            event_id = _append_mem_event(
                event_id,
                "DRAM",
                "scalesim_dram_access",
                "dram",
                operation,
                per_tile_energy,
                per_tile_cycles,
                dram_clk,
                {
                    "memory_kind": "dram",
                    "owner_operator": op_name,
                    "resource_id": "dram_port",
                    "buffer_name": "offchip_dram",
                    "buffer_phase": phase,
                    "bytes": this_tile_bytes,
                    "scalesim_dram_trace_dir": trace_dir,
                    "tile_index": tile_idx,
                    "tile_count": max(1, tile_count),
                },
            )

    return events


def _resolve_numel(params: Dict[str, Any], default_features: int = 1, default_points: int = 1) -> int:
    """从 params 解析元素总数（num_elements 或 num_features * num_points）。"""
    num_elements = int(params.get("num_elements", 0))
    if num_elements > 0:
        return num_elements
    num_features = int(params.get("num_features", params.get("normalized_shape", default_features)))
    num_points = int(params.get("num_points", default_points))
    return max(1, num_features * num_points)


def _quant_pipeline_power_cycles(
    hw_config: "HardwareConfig", numel: int, mode: str = "quant_8bit"
) -> tuple:
    """
    量化/反量化流水线：T = ceil(N/P) + pipeline_latency，功耗按并行度线性放大。
    mode='quant_8bit' 用 quant_pipeline_8bit（latency 6），'dequant' 用 dequant_pipeline（latency 4）。
    """
    lc = getattr(hw_config, "logic_circuit", None)
    if mode == "dequant":
        cfg = getattr(lc, "dequant_pipeline", None) if lc else None
        default_power, default_latency = 2.30, 4
    else:
        cfg = getattr(lc, "quant_pipeline_8bit", None) if lc else None
        default_power, default_latency = 2.234, 6
    power_base = float(getattr(cfg, "power", default_power) or default_power) if cfg else default_power
    par = max(1, int(getattr(cfg, "parallelism", 1) or 1)) if cfg else 1
    latency = int(getattr(cfg, "pipeline_latency", default_latency) or default_latency) if cfg else default_latency
    power = power_base * par
    cycles = max(1, safe_div_ceil(numel, par) + latency)
    return power, cycles


def gen_quant_events(
    numel: int,
    hw_config: "HardwareConfig",
    layer_name: str = "",
    op_name: str = "Conv1d",
) -> List[Event]:
    """W8A8：激活 FP32→INT8 量化事件（act_quant），在 MNSIM INT 计算前插入；op_name 用于 operator_sequence 显示（PascalCase）。"""
    clk = _clock(hw_config)
    power, cycles = _quant_pipeline_power_cycles(hw_config, numel, mode="quant_8bit")
    return [
        _event(
            0, 0,
            op_name, "quant", "quantize",
            "quant_pipeline_8bit", "activation_fp32_to_int8",
            power, cycles, clk,
            {"layer_name": layer_name, "numel": numel, "w8a8": True},
        )
    ]


def gen_dequant_events(
    numel: int,
    hw_config: "HardwareConfig",
    layer_name: str = "",
    op_name: str = "Conv1d",
) -> List[Event]:
    """W8A8：输出 INT8→FP32 反量化事件，供后续 BN/ReLU 等浮点运算；op_name 用于 operator_sequence 显示（PascalCase）。"""
    clk = _clock(hw_config)
    power, cycles = _quant_pipeline_power_cycles(hw_config, numel, mode="dequant")
    return [
        _event(
            0, 0,
            op_name, "dequant", "dequantize",
            "dequant_pipeline", "output_int8_to_fp32",
            power, cycles, clk,
            {"layer_name": layer_name, "numel": numel, "w8a8": True},
        )
    ]


def gen_bn_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """BatchNorm（1D/2D 共用）：2级流水线，并行度 P。

    BN1d 输入 [B, C, N]，BN2d 输入 [B, C, H, W]（调用方将 H*W 合并为 num_points）。
    流水线周期公式：
        steps  = ceil(B × C × spatial / P)
        cycles = steps + 1          ← 流水线深度 2：第 1 批出结果需 2 周期，此后每步 +1
    对应串行（P=1）公式：N-1+2 = N+1；并行（P>1）公式：ceil(N/P)-1+2 = ceil(N/P)+1。
    """
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    channels = int(params.get("num_channels", params.get("num_features", params.get("in_channels", 1))))
    # spatial 维度：BN1d 取 num_points（L），BN2d 调用方应将 H*W 合并后以 num_points 传入
    spatial = int(params.get("num_points", params.get("s", params.get("seq_len", 1))))
    if "input_shape" in params and isinstance(params["input_shape"], (list, tuple)):
        shp = params["input_shape"]
        if len(shp) >= 2:
            channels = int(shp[1])
        # shp[2] 为 spatial（BN1d: L；BN2d: H，若有 shp[3] 则 H*W）
        spatial = int(shp[2]) if len(shp) >= 3 else 1
        if len(shp) >= 4:
            spatial *= int(shp[3])

    par = _logic_parallelism(hw_config, "batchnorm1d", 1)
    numel = max(1, batch_size * channels * max(1, spatial))
    steps = safe_div_ceil(numel, par)
    cycles = max(1, steps + 1)          # 2-stage pipeline: (steps-1)+2 = steps+1
    power = _logic_power(hw_config, "batchnorm1d", 3.1718) * par
    clk = _clock(hw_config)
    param_bytes = max(1, channels * 2 * 4)
    ev0 = _named_sram_io_event(0, 0, "BN", "normalize", "batchnorm_param_read", "nonlinear_param_sram", param_bytes, True, hw_config, clk, {"batch": batch_size, "C": channels, "spatial": spatial, "numel": numel, "steps": steps, "parallelism": par})
    ev1 = _event(1, ev0.cycle + ev0.duration, "BN", "normalize", "batchnorm1d", "batchnorm1d", "batchnorm1d_affine", power, cycles, clk, {"batch": batch_size, "C": channels, "spatial": spatial, "numel": numel, "steps": steps, "parallelism": par})
    return [ev0, ev1]


def gen_layernorm_events(params: Dict[str, Any], hw_config: "HardwareConfig", op_name: str = "LayerNorm") -> List[Event]:
    """LayerNorm 周期公式：Nvec * (3*C/8 + 17)，C=features（归一化维度）。"""
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    num_points = int(params.get("num_points", params.get("tokens", 1)))
    features = int(params.get("num_features", params.get("normalized_shape", 1)))
    n_vec = max(1, batch_size * num_points)
    # 每向量周期 = 3*C/8 + 17（3*C/8 向上取整为整数）
    per_vec_cycles = (3 * features + 7) // 8 + 17
    base_cycles = n_vec * per_vec_cycles
    if params.get("num_elements", 0):
        numel = int(params["num_elements"])
    else:
        numel = max(1, batch_size * num_points * features)

    par = _logic_parallelism(hw_config, "layernorm", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "layernorm", 56.0) * par
    clk = _clock(hw_config)
    state_bytes = max(1, features * 2 * 4)
    param_bytes = max(1, features * 2 * 4)
    ev0 = _named_sram_io_event(0, 0, op_name, "normalize", "layernorm_param_read", "nonlinear_param_sram", param_bytes, True, hw_config, clk, {"numel": numel, "parallelism": par})
    ev1 = _named_sram_io_event(1, ev0.cycle + ev0.duration, op_name, "normalize", "layernorm_state_write", "state_sram_buffer", state_bytes, False, hw_config, clk, {"numel": numel, "parallelism": par})
    ev2 = _event(2, ev1.cycle + ev1.duration, op_name, "normalize", "layernorm", "layernorm", "layernorm_affine", power, cycles, clk, {"numel": numel, "parallelism": par})
    return [ev0, ev1, ev2]


def gen_relu_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    numel = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "relu", 1)
    cycles = max(1, safe_div_ceil(numel, par))
    power = _logic_power(hw_config, "relu", 0.45) * par
    clk = _clock(hw_config)
    lane_bytes = max(1, min(numel, par) * 4)
    ev0 = _named_sram_io_event(0, 0, "ReLU", "activation", "relu_state_write", "state_sram_buffer", lane_bytes, False, hw_config, clk, {"numel": numel, "parallelism": par})
    ev1 = _event(1, ev0.cycle + ev0.duration, "ReLU", "activation", "relu", "relu", "relu", power, cycles, clk, {"numel": numel, "parallelism": par})
    return [ev0, ev1]


def gen_spool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    channels = int(params.get("in_channels", params.get("num_features", 1)))
    num_points = int(params.get("num_points", 1))
    k_neighbors = int(params.get("k_neighbors", 1))

    # 更新公式：T = 56 + (N-1)*47, N=B*D*Ns
    n_total = max(1, batch_size * channels * num_points)
    cycles = max(1, 23 + (n_total - 1) * 12) # max(1, 56 + (n_total - 1) * 47)
    power = _logic_power(hw_config, "spool", 26.82)
    clk = _clock(hw_config)
    workspace_bytes = max(1, batch_size * channels * num_points * k_neighbors * 4)
    state_bytes = max(1, batch_size * channels * num_points * 4)
    ev0 = _named_sram_io_event(0, 0, "SPool", "pool", "spool_workspace_write", "nonlinear_workspace_sram", workspace_bytes, False, hw_config, clk, {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors, "N": n_total})
    ev1 = _named_sram_io_event(1, ev0.cycle + ev0.duration, "SPool", "pool", "spool_state_write", "state_sram_buffer", state_bytes, False, hw_config, clk, {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors, "N": n_total})
    ev2 = _event(2, ev1.cycle + ev1.duration, "SPool", "pool", "s_pool", "spool", "softmax_pool", power, cycles, clk, {"channels": channels, "num_points": num_points, "k_neighbors": k_neighbors, "N": n_total})
    return [ev0, ev1, ev2]


def gen_affine_norm_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """GAM 内仿射归一化（源码中为 mean/std + (x-mean)/(std+eps) + affine_alpha*x+affine_beta），
    并非简单 LayerNorm。暂按默认 1 cycle 占位，后续可再优化。"""
    s = int(params.get("s", params.get("num_points", 1)))
    k = int(params.get("k", params.get("k_neighbors", 1)))
    d = int(params.get("d", params.get("num_features", 1)))
    use_xyz = bool(params.get("use_xyz", False))
    coord_dims = int(params.get("coord_dims", 3))
    d_eff = d + (coord_dims if use_xyz else 0)
    l_work = max(1, s * k * d_eff)
    p_fixed = 32
    cycles = 2 * (safe_div_ceil(l_work, p_fixed) + 1) + 46

    clk = _clock(hw_config)
    power = _logic_power(hw_config, "layernorm", 56.0)
    param_bytes = max(1, d_eff * 2 * 4)
    workspace_bytes = max(1, l_work * 4)
    ev0 = _named_sram_io_event(0, 0, "AffineNorm", "normalize", "affinenorm_param_read", "nonlinear_param_sram", param_bytes, True, hw_config, clk, {"S": s, "K": k, "D": d_eff, "L": l_work})
    ev1 = _named_sram_io_event(1, ev0.cycle + ev0.duration, "AffineNorm", "normalize", "affinenorm_workspace_write", "nonlinear_workspace_sram", workspace_bytes, False, hw_config, clk, {"S": s, "K": k, "D": d_eff, "L": l_work})
    ev2 = _event(2, ev1.cycle + ev1.duration, "AffineNorm", "normalize", "affine_norm", "affine_norm", "groupnorm_affine", power, cycles, clk, {"S": s, "K": k, "D": d_eff, "L": l_work, "fixed_parallelism": p_fixed, "engine_parallelism": 1})
    return [ev0, ev1, ev2]


def gen_adaptive_max_pool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """AdaptiveMaxPool1d：对 (B,C,L) 沿 L 做 max 到 1，工作量 B*C*L。"""
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    input_shape = params.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
        base_cycles = _prod([int(x) for x in input_shape[:-1]])
    elif "num_elements" in params:
        base_cycles = int(params.get("num_elements", 1))
    else:
        num_features = int(params.get("num_features", params.get("num_points", 1)))
        num_points = int(params.get("num_points", params.get("input_points", 1)))
        # 沿最后一维做 max，工作量 = B × C × L
        base_cycles = max(1, batch_size * num_features * num_points)

    par = _logic_parallelism(hw_config, "adaptive_max_pool", 1)
    cycles = max(1, safe_div_ceil(base_cycles, par))
    power = _logic_power(hw_config, "adaptive_max_pool", 13.2) * par
    clk = _clock(hw_config)
    workspace_bytes = max(1, base_cycles * 4)
    state_bytes = max(1, batch_size * max(1, int(params.get("num_features", params.get("num_points", 1)))) * 4)
    ev0 = _named_sram_io_event(0, 0, "AdaptiveMaxPool1d", "pool", "adaptive_pool_workspace_write", "nonlinear_workspace_sram", workspace_bytes, False, hw_config, clk, {"base_cycles": base_cycles, "parallelism": par})
    ev1 = _named_sram_io_event(1, ev0.cycle + ev0.duration, "AdaptiveMaxPool1d", "pool", "adaptive_pool_state_write", "state_sram_buffer", state_bytes, False, hw_config, clk, {"base_cycles": base_cycles, "parallelism": par})
    ev2 = _event(2, ev1.cycle + ev1.duration, "AdaptiveMaxPool1d", "pool", "adaptive_max_pool1d", "adaptive_max_pool", "adaptive_max_pool1d", power, cycles, clk, {"base_cycles": base_cycles, "parallelism": par})
    return [ev0, ev1, ev2]


def gen_sa_max_pool_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    PointNet++ SA MaxPool：对分组后的邻居特征 [B, C, nsample, nq] 沿 nsample 维度取 max，
    输出 [B, C, nq]。

    相较旧的 adaptive_max_pool 借用模型，本实现做了以下修正：

    1. 显式区分 nq（query 点数/输出点数）和 nsample（每 query 的邻居数），
       避免将两者折叠为 num_points = nq * nsample 导致结构信息丢失。
       若 params 中不含 nq/nsample（旧路径兼容），则退化为视 num_points 为
       单一 group（nq=num_points, nsample=1）。

    2. 周期模型：总比较次数 = B × C × nq × nsample，按 parallelism 并行执行。
       （与旧模型总量相同，但结构语义正确。）

    3. 流式 workspace：片上只需缓冲 1 个 group 的 conv 输出
       （B × C × nsample × 4 bytes），而非全部 nq 个 group 同时在片上，
       实际 SRAM 压力低 nq 倍。

    4. 修正 state_bytes 为实际输出大小 B × C × nq × 4，
       旧模型此处漏乘 nq 倍。
    """
    batch_size = int(params.get("batch_size", params.get("batch", 1)))
    C = int(params.get("num_features", 1))

    # 优先从显式字段读取 nq / nsample（由 pointnet2_net._build_sa_operators 传入）
    nq = int(params.get("nq", 0))
    nsample = int(params.get("nsample", 0))
    if nq <= 0 or nsample <= 0:
        # 旧路径兼容：退化为视整体为单一 group
        num_points = int(params.get("num_points", params.get("input_points", 1)))
        nq = num_points
        nsample = 1

    par = _logic_parallelism(hw_config, "sa_maxpool", 32)
    # 周期公式：B × C × nq × ceil(nsample / parallelism)
    # 每个周期并行处理 par 个邻居，nq 个 query 串行（或流水），共 nq × ceil(nsample/par) 步
    steps_per_query = max(1, safe_div_ceil(nsample, par))
    cycles = max(1, batch_size * C * nq * steps_per_query)
    power = _logic_power(hw_config, "sa_maxpool", 8.2)
    clk = _clock(hw_config)

    # 流式 workspace：每次只缓存 1 个 group（nsample 邻居）的 conv 输出
    workspace_bytes = max(1, batch_size * C * nsample * 4)
    # 输出 state：nq 个 query，每个 C 通道（修正：旧模型缺少 nq 因子）
    state_bytes = max(1, batch_size * C * nq * 4)

    meta_base = {"nq": nq, "nsample": nsample, "C": C, "parallelism": par,
                 "steps_per_query": steps_per_query, "native_op": "sa_max_pool"}
    ev0 = _named_sram_io_event(
        0, 0, "SAMaxPool", "pool", "sa_maxpool_workspace_write",
        "nonlinear_workspace_sram", workspace_bytes, False, hw_config, clk,
        {**meta_base, "streaming_group": True},
    )
    ev1 = _named_sram_io_event(
        1, ev0.cycle + ev0.duration, "SAMaxPool", "pool", "sa_maxpool_state_write",
        "state_sram_buffer", state_bytes, False, hw_config, clk,
        meta_base,
    )
    ev2 = _event(
        2, ev1.cycle + ev1.duration,
        "SAMaxPool", "pool", "sa_max_pool", "sa_maxpool", "sa_max_pool",
        power, cycles, clk,
        meta_base,
    )
    return [ev0, ev1, ev2]


def gen_add_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_add", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_add", 1)
    add_cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    add_power = _logic_power(hw_config, "fp32_add", 1.16) * par
    clk = _clock(hw_config)

    events: List[Event] = []
    event_id = 0
    cycle = 0

    residual = bool(params.get("residual", False))
    # Residual paths default to on-chip retention; only force off-chip when explicitly requested.
    offchip = bool(params.get("offchip", False))
    residual_bytes = max(1, math.ceil(num_elements * 4))
    if residual and not offchip:
        sram_cycles, sram_energy = query_sram_cycles_energy(
            residual_bytes,
            is_read=True,
            hw_config=hw_config,
            clock_period_ns=clk,
            buffer_name="residual_sram_buffer",
        )
        sram_power = max(0.0, float(sram_energy) * 1000.0 / max(1.0, float(sram_cycles) * float(clk)))
        events.append(_event(event_id, cycle, "SRAM", "residual_buffer", "sram_read", "residual_sram_buffer", "residual_sram_read", sram_power, sram_cycles, clk, {"bytes": residual_bytes, "buffer_name": "residual_sram_buffer", "buffer_phase": "read"}))
        event_id += 1
        cycle += sram_cycles
    if offchip:
        bytes_rd = residual_bytes
        bw = int(params.get("dram_bw_bytes_per_cycle", 32))
        dram_cycles = safe_div_ceil(bytes_rd, bw)
        dram_power = float(params.get("dram_power_mw", 50.0))
        events.append(_event(event_id, cycle, "ADD", "residual_buffer", "dram_read", "dram", "offchip_residual_read", dram_power, dram_cycles, clk, {"bytes": bytes_rd, "resource_id": "dram_port", "dram_pipelined": True, "dram_issue_cycles": query_dram_issue_cycles(bytes_rd, hw_config)}))
        event_id += 1
        cycle += dram_cycles

    events.append(_event(event_id, cycle, "ADD", "elementwise", "add", "fp32_add", "elementwise_add", add_power, add_cycles, clk, {"num_elements": num_elements, "residual": residual}))
    event_id += 1
    cycle += add_cycles

    if offchip:
        bytes_wr = residual_bytes
        bw = int(params.get("dram_bw_bytes_per_cycle", 32))
        dram_cycles = safe_div_ceil(bytes_wr, bw)
        dram_power = float(params.get("dram_power_mw", 50.0))
        events.append(_event(event_id, cycle, "ADD", "residual_buffer", "dram_write", "dram", "offchip_residual_write", dram_power, dram_cycles, clk, {"bytes": bytes_wr}))

    return events


def gen_mul_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_mul", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_mul", 1)
    cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    power = _logic_power(hw_config, "fp32_mul", 2.5) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "MUL", "elementwise", "multiply", "fp32_mul", "elementwise_multiply", power, cycles, clk, {"num_elements": num_elements})]


def gen_div_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """FP32 逐元素除法，cycle/power/area 由 logic_circuit.fp32_div 统一管理。"""
    num_elements = _resolve_numel(params)
    par = _logic_parallelism(hw_config, "fp32_div", 1)
    per_elem = _logic_cycles_per_element(hw_config, "fp32_div", 10)
    cycles = max(1, safe_div_ceil(num_elements, par) * per_elem)
    power = _logic_power(hw_config, "fp32_div", 2.59) * par
    clk = _clock(hw_config)
    return [_event(0, 0, "DIV", "elementwise", "divide", "fp32_div", "elementwise_divide", power, cycles, clk, {"num_elements": num_elements})]


def gen_interp_n3_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    InterpN3 插值引擎事件：使用专用 InterpN3Engine 建模 FP 上采样中的 3-NN 插值。

    对每个 query：
      1. dist_recip / norm / weight 共固定 22 cycles
      2. 加权插值共 ceil(4 * D2 / 16) cycles（16 为默认 D2 并行度）

    两个阶段在 query 维度按流水线执行，整层 N 个 query 的计算段周期为：
      22 + ceil(4 * D2 / P) + (N - 1) * max(22, ceil(4 * D2 / P))
    其中 P = interp_n3_engine.feature_parallelism。

    邻居特征 gather 仍单独按 DRAM 事件建模，并与计算流水重叠。
    """
    N = int(params.get("num_output_points", params.get("num_points", 1)))
    events, _eid, _query_end = _build_interp_n3_engine_events(
        params,
        hw_config,
        N,
        {q: 0 for q in range(max(0, N))},
        start_event_id=0,
        include_query_markers=False,
    )
    return events


def gen_kan_rram_lut_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    KAT(GRF) 事件：grf_base_engine 存在时使用数字因果有理多项式（baseline / PointCIM）；
    否则走 rram_lut_macro.rram_lut（RRAM-LUT 查表，仅 acc 纯模式）。
    算子名统一为 KAN，breakdown 按 KAN 聚合。
    """
    in_features = int(params.get("in_features", 1))
    out_features = int(params.get("out_features", in_features))
    num_points = int(params.get("num_points", 1))
    variant = str(params.get("variant", "standard"))
    sequence_len = max(1, int(params.get("sequence_len", num_points)))
    kat_path = str(params.get("kat_path", ""))
    k_neighbors = int(params.get("k_neighbors", 0) or 0)
    if k_neighbors <= 0 and kat_path:
        if kat_path == "local" and num_points > 0 and sequence_len % num_points == 0:
            k_neighbors = max(1, sequence_len // num_points)
    num_groups = max(1, int(params.get("num_groups", 8 if variant == "elite" else 1) or 1))
    clk = _clock(hw_config)

    def _activation_shape(features: int) -> str:
        if k_neighbors > 1 and sequence_len == max(1, num_points * k_neighbors):
            return f"{num_points}×{k_neighbors}×{features}"
        return f"{sequence_len}×{features}"

    def _base_kat_meta(kat_impl: str, kat_mode: str = "") -> Dict[str, Any]:
        meta = {"variant": variant, "kat_impl": kat_impl}
        if kat_mode:
            meta["kat_mode"] = kat_mode
        if kat_path:
            meta["kat_path"] = kat_path
        meta["num_points"] = num_points
        meta["sequence_len"] = sequence_len
        if k_neighbors > 0:
            meta["k_neighbors"] = k_neighbors
        meta["in_features"] = in_features
        meta["out_features"] = out_features
        meta["num_groups"] = num_groups
        meta["activation_input_shape"] = _activation_shape(in_features)
        meta["activation_output_shape"] = _activation_shape(out_features)
        return meta

    # GRF 数字路径：baseline / PointCIM 配置了 grf_base_engine 时使用
    logic = getattr(hw_config, "logic_circuit", None)
    grf = getattr(logic, "grf_base_engine", None) if logic else None
    if grf is not None:
        cycles_per_data = int(getattr(grf, "cycles", 0) or 19)
        par = max(1, int(getattr(grf, "parallelism", 1) or 1))
        num_data = max(1, sequence_len * in_features)
        total_cycles = safe_div_ceil(num_data, par) * cycles_per_data
        power = float(getattr(grf, "power", 23.783) or 23.783)
        meta = _base_kat_meta("grf_base_engine")
        coeff_bytes = max(1, num_groups * 10 * 4)
        meta["coeff_bytes"] = coeff_bytes
        meta["coeff_source"] = "nonlinear_param_sram"
        meta["parallelism"] = par
        meta["cycles_per_data"] = cycles_per_data
        ev0 = _named_sram_io_event(0, 0, "KAN", "grf_kat", "kan_param_read", "nonlinear_param_sram", coeff_bytes, True, hw_config, clk, meta)
        ev1 = _event(1, ev0.cycle + ev0.duration, "KAN", "grf_kat", "grf_polynomial", "grf_base_engine", "kan_grf_causal", power, total_cycles, clk, meta)
        return [ev0, ev1]

    # acc：KAT 由 RRAM-LUT / RRAM-PWL 实现
    lut_cfg = getattr(getattr(hw_config, "rram_lut_macro", None), "rram_lut", None)
    kat_mode = str(params.get("kat_mode", getattr(lut_cfg, "default_mode", "rram_pwl"))).lower()

    if kat_mode in {"pwl", "rram_pwl"}:
        pwl_parallel = max(1, int(params.get("pwl_parallel", getattr(lut_cfg, "pwl_parallelism", 1) if lut_cfg is not None else 1)))
        pwl_overhead = max(1, int(getattr(lut_cfg, "pwl_overhead_cycles", 1) or 1)) if lut_cfg is not None else 1
        num_data = max(1, sequence_len * in_features)
        total_cycles = safe_div_ceil(num_data, pwl_parallel) + pwl_overhead
        pwl_power = float(getattr(lut_cfg, "pwl_power", 0.0759164) or 0.0759164) if lut_cfg is not None else 0.0759164
        meta = _base_kat_meta("rram_pwl", "rram_pwl")
        meta["parallelism"] = pwl_parallel
        meta["latency_ns"] = float(total_cycles * clk)
        meta["coeff_source"] = "rram_resident"
        coeff_bytes = max(1, num_groups * 10 * 4)
        meta["coeff_bytes"] = coeff_bytes
        rram_read = _query_rram_lut_read_cycles_energy(coeff_bytes, hw_config, lut_cfg, params, clk)
        read_meta = dict(meta)
        read_meta.update({
            "rram_read_accesses": rram_read["accesses"],
            "rram_word_width_bits": rram_read["word_width_bits"],
            "rram_word_bytes": rram_read["word_bytes"],
            "rram_read_parallelism": rram_read["parallelism"],
            "rram_read_latency_ns": rram_read["read_latency_ns"],
            "rram_read_energy_pj": rram_read["read_energy_pj"],
            "rram_read_source": rram_read["source"],
        })
        for key in ("area_mm2", "leakage_power_uw", "capacity_kb", "optimization_target"):
            if key in rram_read:
                read_meta[f"rram_{key}"] = rram_read[key]
        area_um2 = float(getattr(lut_cfg, "pwl_area", 235.199) or 235.199) if lut_cfg is not None else 235.199
        meta["area_um2"] = area_um2
        ev0 = _event(0, 0, "KAN", "rram_pwl", "rram_table_read", "rram_lut", "kan_rram_table_read", rram_read["power_mw"], rram_read["cycles"], clk, read_meta)
        ev1 = _event(1, ev0.cycle + ev0.duration, "KAN", "rram_pwl", "pwl_eval", "rram_pwl", "kan_rram_pwl", pwl_power, total_cycles, clk, meta)
        return [ev0, ev1]

    lut_factor = 4 if variant == "standard" else 2
    lut_parallel = int(params.get("lut_parallel", max(1, getattr(hw_config, "cim_parallelism", 128))))
    work = max(1, sequence_len * in_features * lut_factor)
    lut_cycles = safe_div_ceil(work, lut_parallel)

    lut_power = 5.0
    if lut_cfg is not None:
        lut_power = float(getattr(lut_cfg, "power", lut_power) or lut_power)

    input_bytes = max(1, sequence_len * in_features * 4)
    output_bytes = max(1, sequence_len * out_features * 4)
    sram_macro = getattr(hw_config, "sram_cim_macro", None)
    input_buf = getattr(sram_macro, "input_buffer", None) if sram_macro is not None else None
    output_buf = getattr(sram_macro, "output_buffer", None) if sram_macro is not None else None
    input_cycles, input_energy = query_sram_cycles_energy(
        input_bytes,
        is_read=True,
        hw_config=hw_config,
        clock_period_ns=clk,
        buffer_cfg=input_buf,
        buffer_name="input_buffer",
    )
    output_cycles, output_energy = query_sram_cycles_energy(
        output_bytes,
        is_read=False,
        hw_config=hw_config,
        clock_period_ns=clk,
        buffer_cfg=output_buf,
        buffer_name="output_buffer",
    )
    input_power = _energy_to_power_mw(float(input_energy), int(input_cycles), clk)
    output_power = _energy_to_power_mw(float(output_energy), int(output_cycles), clk)

    kat_mode = str(params.get("kat_mode", "lut")).lower()
    post_mac_power = float(getattr(lut_cfg, "post_mac_power", 0.0) or 0.0)
    post_mac_cycles = int(getattr(lut_cfg, "post_mac_cycles", 0) or 0)
    meta = _base_kat_meta("rram_lut", kat_mode)
    coeff_bytes = max(1, num_groups * 10 * 4)
    meta["coeff_bytes"] = coeff_bytes
    meta["coeff_source"] = "nonlinear_param_sram"
    events = [
        _named_sram_io_event(0, 0, "KAN", "buffer_io", "kan_param_read", "nonlinear_param_sram", coeff_bytes, True, hw_config, clk, meta),
        _event(1, 0, "KAN", "buffer_io", "sram_read", "buffer", "kan_input_read", input_power, input_cycles, clk, meta),
        _event(2, input_cycles, "KAN", "rram_lut", "lut_eval", "rram_lut", "kan_basis_eval", lut_power, lut_cycles, clk, meta),
    ]
    cursor = input_cycles + lut_cycles
    if kat_mode == "lut_mac":
        mac_cycles = max(1, post_mac_cycles or safe_div_ceil(max(1, sequence_len * out_features), lut_parallel))
        mac_power = post_mac_power if post_mac_power > 0 else lut_power
        events.append(_event(3, cursor, "KAN", "rram_lut", "lut_post_mac", "rram_lut", "kan_post_lut_mac", mac_power, mac_cycles, clk, meta))
        cursor += mac_cycles
    events.append(_named_sram_io_event(4, cursor, "KAN", "buffer_io", "kan_workspace_write", "nonlinear_workspace_sram", output_bytes, False, hw_config, clk, meta))
    events.append(_event(5, cursor, "KAN", "buffer_io", "sram_write", "buffer", "kan_output_write", output_power, output_cycles, clk, meta))
    return events


def _try_scalesim_conv1d(params: Dict[str, Any], hw_config: "HardwareConfig"):
    """若启用 use_scalesim_for_systolic 且 SCALE-Sim 可用，返回 (total_cycles, metadata)；否则返回 None。"""
    if not getattr(hw_config, "use_scalesim_for_systolic", False):
        return None
    try:
        from ..integration.scalesim_adapter import run_scalesim_conv1d_layer
    except ImportError:
        return None
    root = getattr(hw_config, "scalesim_root", "") or None
    cfg = getattr(hw_config, "scalesim_config_path", "") or None
    res = run_scalesim_conv1d_layer(
        in_channels=int(params.get("in_channels", 1)),
        out_channels=int(params.get("out_channels", 1)),
        kernel_size=int(params.get("kernel_size", 1)),
        num_points=int(params.get("num_points", 1)),
        layer_name=str(params.get("layer_name", "conv1d")),
        progress_index=params.get("progress_index"),
        progress_total=params.get("progress_total"),
        scalesim_root=root,
        config_path=cfg,
        hw_config=hw_config,
    )
    if not res.success:
        return None
    return (max(1, res.total_cycles), res.to_metadata())


def _try_scalesim_linear(params: Dict[str, Any], hw_config: "HardwareConfig"):
    """若启用 use_scalesim_for_systolic 且 SCALE-Sim 可用，返回 (total_cycles, metadata)；否则返回 None。"""
    if not getattr(hw_config, "use_scalesim_for_systolic", False):
        return None
    try:
        from ..integration.scalesim_adapter import run_scalesim_linear_layer
    except ImportError:
        return None
    root = getattr(hw_config, "scalesim_root", "") or None
    cfg = getattr(hw_config, "scalesim_config_path", "") or None
    res = run_scalesim_linear_layer(
        in_features=int(params.get("in_features", 1)),
        out_features=int(params.get("out_features", 1)),
        tokens=int(params.get("tokens", 1)),
        layer_name=str(params.get("layer_name", "linear")),
        progress_index=params.get("progress_index"),
        progress_total=params.get("progress_total"),
        scalesim_root=root,
        config_path=cfg,
        hw_config=hw_config,
    )
    if not res.success:
        return None
    return (max(1, res.total_cycles), res.to_metadata())




def _gen_im2col_engine_events(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    *,
    start_cycle: int = 0,
    start_event_id: int = 0,
) -> tuple[list[Event], int, int, Dict[str, Any]]:
    """Generate explicit im2col front-end events for baseline Conv1d."""
    in_ch = int(params.get("in_channels", 1))
    out_ch = int(params.get("out_channels", 1))
    kernel = int(params.get("kernel_size", 1))
    num_points = int(params.get("num_points", 1))
    layer_name = str(params.get("layer_name", ""))

    cfg = getattr(hw_config, "im2col", None)
    clk = _clock(hw_config)
    L = max(1, num_points)
    patch_elems = max(1, in_ch * kernel)
    total_patch_elems = patch_elems * L
    data_bytes = max(1, int(getattr(getattr(hw_config, "systolic_array", None), "data_bytes", 2) or 2))
    total_patch_bytes = total_patch_elems * data_bytes
    input_tensor_bytes = max(1, in_ch * L * data_bytes)

    addr_gen_per_cycle = max(1, int(getattr(cfg, "addr_gen_per_cycle", 16) or 16)) if cfg else 16
    pack_elems_per_cycle = max(1, int(getattr(cfg, "pack_elems_per_cycle", 16) or 16)) if cfg else 16
    addr_gen_power = float(getattr(cfg, "addr_gen_power_mw", 1.5) or 1.5) if cfg else 1.5
    input_ctrl_power = float(getattr(cfg, "input_read_ctrl_power_mw", 2.0) or 2.0) if cfg else 2.0
    pack_power = float(getattr(cfg, "pack_reorder_power_mw", getattr(cfg, "power", 15.0) if cfg else 15.0) or 15.0)
    col_write_ctrl_power = float(getattr(cfg, "col_write_ctrl_power_mw", 2.0) or 2.0) if cfg else 2.0
    col_read_ctrl_power = float(getattr(cfg, "col_read_ctrl_power_mw", 2.0) or 2.0) if cfg else 2.0
    overlap_addr_and_read = bool(getattr(cfg, "overlap_addr_and_read", True)) if cfg else True
    overlap_pack_with_write = bool(getattr(cfg, "overlap_pack_with_write", False)) if cfg else False
    input_from_dram = bool(getattr(cfg, "input_from_dram", False)) if cfg else False

    addr_gen_cycles = max(1, safe_div_ceil(total_patch_elems, addr_gen_per_cycle))
    pack_cycles = max(1, safe_div_ceil(total_patch_elems, pack_elems_per_cycle))

    dram_fill_cycles = dram_fill_energy = 0.0
    if input_from_dram:
        dram_fill_cycles, dram_fill_energy = query_dram_cycles_energy(input_tensor_bytes, True, hw_config, clk)

    input_read_cycles, input_read_energy = query_sram_cycles_energy(
        total_patch_bytes, True, hw_config, clk, buffer_name="inter_layer_buffer"
    )
    col_write_cycles, col_write_energy = query_sram_cycles_energy(
        total_patch_bytes, False, hw_config, clk, buffer_name="im2col_buffer"
    )
    col_read_cycles, col_read_energy = query_sram_cycles_energy(
        total_patch_bytes, True, hw_config, clk, buffer_name="im2col_buffer"
    )

    events: List[Event] = []
    event_id = start_event_id
    cursor = start_cycle
    if dram_fill_cycles > 0:
        events.append(_event(
            event_id, cursor, "Conv1d", "im2col", "dram_refill", "dram", "im2col_input_dram_refill",
            _energy_to_power_mw(float(dram_fill_energy), int(dram_fill_cycles), clk), int(dram_fill_cycles), clk,
            {"bytes": input_tensor_bytes, "layer_name": layer_name, "buffer_name": "inter_layer_buffer"},
        ))
        event_id += 1
        cursor += int(dram_fill_cycles)

    base = cursor
    events.append(_event(
        event_id, base, "Conv1d", "im2col", "addr_gen", "im2col", "conv_im2col_addr_gen",
        addr_gen_power, addr_gen_cycles, clk,
        {"layer_name": layer_name, "num_points": L, "patch_elems": patch_elems, "total_patch_elems": total_patch_elems},
    ))
    event_id += 1
    events.append(_event(
        event_id, base, "LayerBuffer", "im2col", "input_read", "inter_layer_buffer", "im2col_input_read",
        _energy_to_power_mw(float(input_read_energy), int(input_read_cycles), clk) + input_ctrl_power, int(input_read_cycles), clk,
        {"layer_name": layer_name, "bytes": total_patch_bytes, "buffer_name": "inter_layer_buffer", "buffer_phase": "im2col_source_read", "resource_id": "inter_layer_buffer_shared", "exclusive": True},
    ))
    event_id += 1
    stage1_end = base + (max(addr_gen_cycles, input_read_cycles) if overlap_addr_and_read else addr_gen_cycles + input_read_cycles)

    events.append(_event(
        event_id, stage1_end, "Conv1d", "im2col", "pack_reorder", "im2col", "conv_im2col_pack_reorder",
        pack_power, pack_cycles, clk,
        {"layer_name": layer_name, "bytes": total_patch_bytes, "patch_elems": patch_elems},
    ))
    event_id += 1
    events.append(_event(
        event_id, stage1_end if overlap_pack_with_write else stage1_end + pack_cycles,
        "LayerBuffer", "im2col", "col_write", "im2col_buffer", "im2col_patch_write",
        _energy_to_power_mw(float(col_write_energy), int(col_write_cycles), clk) + col_write_ctrl_power, int(col_write_cycles), clk,
        {"layer_name": layer_name, "bytes": total_patch_bytes, "buffer_name": "im2col_buffer", "buffer_phase": "im2col_patch_write", "resource_id": "im2col_buffer_shared", "exclusive": True},
    ))
    event_id += 1
    stage2_end = stage1_end + (max(pack_cycles, col_write_cycles) if overlap_pack_with_write else pack_cycles + col_write_cycles)

    events.append(_event(
        event_id, stage2_end, "LayerBuffer", "im2col", "col_read", "im2col_buffer", "im2col_matrix_read",
        _energy_to_power_mw(float(col_read_energy), int(col_read_cycles), clk) + col_read_ctrl_power, int(col_read_cycles), clk,
        {"layer_name": layer_name, "bytes": total_patch_bytes, "buffer_name": "im2col_buffer", "buffer_phase": "im2col_matrix_read", "resource_id": "im2col_buffer_shared", "exclusive": True},
    ))
    event_id += 1
    end_cycle = stage2_end + col_read_cycles

    meta = {
        "layer_name": layer_name,
        "num_points": L,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "kernel_size": kernel,
        "im2col_input_tensor_bytes": input_tensor_bytes,
        "im2col_patch_bytes": total_patch_bytes,
        "im2col_addr_gen_cycles": addr_gen_cycles,
        "im2col_input_read_cycles": input_read_cycles,
        "im2col_pack_cycles": pack_cycles,
        "im2col_col_write_cycles": col_write_cycles,
        "im2col_col_read_cycles": col_read_cycles,
        "im2col_dram_fill_cycles": int(dram_fill_cycles),
        "im2col_total_cycles": max(1, end_cycle - start_cycle),
        "im2col_input_read_bytes": total_patch_bytes,
        "im2col_patch_write_bytes": total_patch_bytes,
        "im2col_matrix_read_bytes": total_patch_bytes,
        "im2col_input_from_dram": input_from_dram,
        "im2col_ifmap_source_tensor_bytes": input_tensor_bytes,
        "im2col_ifmap_materialized_bytes": total_patch_bytes,
        "scalesim_ifmap_effective_bytes": total_patch_bytes,
    }
    return events, end_cycle, event_id, meta

def gen_interp_n3_per_query_events(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    num_queries: int,
    query_ready_cycles: Dict[int, int],
    start_event_id: int = 0,
    *,
    _template_cache: Optional[Dict[int, List[Event]]] = None,
) -> tuple:
    """
    为 FP 解码路径生成按 query 流水的 InterpN3 事件。

    新模型使用专用 InterpN3Engine：
      1. 权重生成阶段固定 22 cycles/query
      2. 加权插值阶段为 ceil(4 * D2 / P) cycles/query，P 为 D2 并行度（默认 16）

    两阶段在 query 维度形成 producer-consumer 流水；同时每个 query 的 feature gather
    仍按独立 DRAM 读建模，并与 InterpN3Engine 计算重叠。

    Returns:
        (events, next_event_id, query_end_cycles)
    """
    del _template_cache
    return _build_interp_n3_engine_events(
        params,
        hw_config,
        num_queries,
        query_ready_cycles,
        start_event_id=start_event_id,
        include_query_markers=True,
    )


def get_conv1d_per_group_cycles(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    num_groups: int,
    group_sizes: Optional[List[int]] = None,
) -> List[int]:
    """
    用 group 级 Conv1d 参数重建 baseline 事件，返回每个 group 的真实周期数。

    主要用于 KNN→Conv1d 流水线：KNN 每完成一个 query，就让 Conv1d 按该 query 对应的
    group 长度（通常是 K 个邻居）单独估算一次前端 im2col + systolic 的执行时间。
    """
    groups = max(0, int(num_groups))
    if groups <= 0:
        return []

    total_points = max(1, int(params.get("num_points", 1) or 1))
    if group_sizes is None:
        base = total_points // groups
        rem = total_points % groups
        group_sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    else:
        group_sizes = [max(1, int(v or 1)) for v in group_sizes[:groups]]
        if len(group_sizes) < groups:
            group_sizes.extend([1] * (groups - len(group_sizes)))

    per_group_cycles: List[int] = []
    cached_cycles: Dict[int, int] = {}
    for group_points in group_sizes:
        if group_points not in cached_cycles:
            group_params = dict(params)
            group_params["num_points"] = max(1, group_points)
            group_events = gen_conv1d_systolic_events(group_params, hw_config)
            cached_cycles[group_points] = max(
                1,
                max((e.cycle + e.duration for e in group_events), default=0),
            )
        per_group_cycles.append(cached_cycles[group_points])
    return per_group_cycles


# ---------------------------------------------------------------------------
# Feature-gather & per-group Conv1d event helpers (KNN→Conv pipeline)
# ---------------------------------------------------------------------------

def gen_feature_gather_events(
    query_idx: int,
    k: int,
    in_channels: int,
    hw_config: "HardwareConfig",
    start_cycle: int = 0,
    start_event_id: int = 0,
    bytes_per_element: int = 4,
    coalescing_factor: float = 0.0,
) -> tuple:
    """
    Generate DRAM read events for loading K neighbour features after KNN
    produces indices for one query.

    Total bytes = k * in_channels * bytes_per_element * (1 - coalescing_factor).

    Returns:
        (events, next_event_id, end_cycle)
    """
    raw_bytes = max(1, k) * max(1, in_channels) * max(1, bytes_per_element)
    effective_bytes = max(1, int(raw_bytes * max(0.0, 1.0 - min(1.0, float(coalescing_factor)))))
    clk = _clock(hw_config)

    dur, energy_nj = query_dram_cycles_energy(
        effective_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
    )
    dur = max(1, int(dur))
    power = (energy_nj * 1e3 / (dur * clk)) if (dur > 0 and clk > 0) else 0.0
    ev = _event(
        start_event_id, start_cycle,
        "DRAM", "feature_gather", "neighbor_feature_read",
        "dram", "feature_gather_read",
        power, dur, clk,
        {
            "query": query_idx,
            "phase": "feature_gather",
            "buffer_name": "feature_gather_dram",
            "k": k,
            "in_channels": in_channels,
            "bytes": effective_bytes,
            "raw_bytes": raw_bytes,
            "coalescing_factor": float(coalescing_factor),
            "resource_id": "dram_port",
            "dram_pipelined": True,
            "dram_issue_cycles": query_dram_issue_cycles(effective_bytes, hw_config),
        },
    )
    return ([ev], start_event_id + 1, start_cycle + dur)


# ---------------------------------------------------------------------------
# sample_and_group gather: index_points → relative xyz → feature gather → concat
# ---------------------------------------------------------------------------


def _resolve_sample_and_group_feature_source(
    hw_config: "HardwareConfig",
    feature_source: str,
    current_layer_name: str,
    sampled_points: int,
    feature_channels: int,
    bytes_per_element: int,
) -> str:
    if feature_source in ("sram", "dram"):
        return feature_source
    plan_map = getattr(hw_config, "feature_residency_plan_by_layer", {}) or {}
    plan = plan_map.get(str(current_layer_name or ""))
    if plan is not None:
        if int(getattr(plan, "input_dram_bytes", 0) or 0) > 0:
            return "dram"
        if int(getattr(plan, "input_sram_bytes", 0) or 0) > 0:
            return "sram"
    feat_buf = getattr(hw_config, "inter_layer_buffer", None)
    buf_cap = int(getattr(feat_buf, "capacity_bytes", 0) or 0) if feat_buf else 0
    total_feat_bytes = max(1, sampled_points * feature_channels * bytes_per_element)
    return "sram" if buf_cap > 0 and total_feat_bytes <= buf_cap else "dram"


def _resolve_sample_and_group_xyz_source(
    hw_config: "HardwareConfig",
    sampled_points: int,
    k: int,
    xyz_channels: int,
    bytes_per_element: int,
) -> str:
    xyz_buf = getattr(hw_config, "xyz_sram_buffer", None)
    if xyz_buf is None:
        return "dram"
    xyz_bytes = max(1, sampled_points * k * xyz_channels * bytes_per_element)
    buf_cap = int(getattr(xyz_buf, "size_bytes", 0) or 0)
    spill = bool(getattr(xyz_buf, "spill_to_dram", False))
    if spill or buf_cap <= 0 or xyz_bytes > buf_cap:
        return "dram"
    return "sram"


def _gen_sample_and_group_impl(
    sampled_points: int,
    k: int,
    xyz_channels: int,
    feature_channels: int,
    hw_config: "HardwareConfig",
    start_cycle: int,
    start_event_id: int,
    *,
    feature_source: str = "auto",
    tag_scope: str = "",
    bq_ready_tag: str = "",
    current_layer_name: str = "",
) -> tuple:
    clk = _clock(hw_config)
    eid = start_event_id
    events: List[Event] = []
    cursor = start_cycle
    bpe = 4

    dep_tags = [bq_ready_tag] if bq_ready_tag else []
    gather_op = "Gather"

    xyz_bytes = max(1, sampled_points * k * xyz_channels * bpe)
    xyz_gather_tag = f"{tag_scope}xyz_gather_ready"
    xyz_source = _resolve_sample_and_group_xyz_source(hw_config, sampled_points, k, xyz_channels, bpe)
    if xyz_source == "sram":
        ev_xyz = _named_sram_io_event(
            eid, cursor, gather_op, "xyz_gather", "xyz_index_read",
            "xyz_sram_buffer", xyz_bytes, True, hw_config, clk,
            metadata={
                "phase": "sample_and_group",
                "sub_op": "xyz_gather",
                "sampled_points": sampled_points,
                "k": k,
                "xyz_channels": xyz_channels,
                "bytes": xyz_bytes,
                "source_tier": "sram",
                "dependency_tag": xyz_gather_tag,
                "depends_on_tags": dep_tags,
            },
        )
    else:
        xyz_dur, xyz_energy_nj = query_dram_cycles_energy(
            xyz_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
        )
        xyz_dur = max(1, int(xyz_dur))
        ev_xyz = _event(
            eid, cursor, "DRAM", "xyz_gather", "xyz_index_dram_read",
            "dram", "xyz_gather_dram_read",
            _energy_to_power_mw(xyz_energy_nj, xyz_dur, clk), xyz_dur, clk,
            {
                "phase": "sample_and_group",
                "sub_op": "xyz_gather",
                "sampled_points": sampled_points,
                "k": k,
                "xyz_channels": xyz_channels,
                "bytes": xyz_bytes,
                "source_tier": "dram",
                "dependency_tag": xyz_gather_tag,
                "depends_on_tags": dep_tags,
            },
        )
    events.append(ev_xyz)
    eid += 1
    cursor += ev_xyz.duration

    rel_elements = max(1, sampled_points * k * xyz_channels)
    fp32_sub_cfg = getattr(getattr(hw_config, "logic_circuit", None), "fp32_add", None)
    sub_par = max(1, int(getattr(fp32_sub_cfg, "parallelism", 1) or 1))
    sub_power = float(getattr(fp32_sub_cfg, "power", 0.5) or 0.5)
    sub_cycles = max(1, safe_div_ceil(rel_elements, sub_par))
    rel_xyz_tag = f"{tag_scope}relative_xyz_ready"
    events.append(_event(
        eid, cursor, gather_op, "relative_xyz", "xyz_subtract_center",
        "fp32_add", "relative_xyz_sub",
        sub_power, sub_cycles, clk,
        {
            "phase": "sample_and_group",
            "sub_op": "relative_xyz",
            "elements": rel_elements,
            "parallelism": sub_par,
            "dependency_tag": rel_xyz_tag,
            "depends_on_tags": [xyz_gather_tag],
        },
    ))
    eid += 1
    cursor += sub_cycles

    feat_gather_tag = f"{tag_scope}feature_gather_ready"
    if feature_channels > 0:
        feat_bytes = max(1, sampled_points * k * feature_channels * bpe)
        feat_source = _resolve_sample_and_group_feature_source(
            hw_config,
            feature_source,
            current_layer_name,
            sampled_points,
            feature_channels,
            bpe,
        )
        if feat_source == "sram":
            ev_feat = _named_sram_io_event(
                eid, cursor, gather_op, "feature_gather", "feature_index_read",
                "inter_layer_buffer", feat_bytes, True, hw_config, clk,
                metadata={
                    "phase": "sample_and_group",
                    "sub_op": "feature_gather",
                    "sampled_points": sampled_points,
                    "k": k,
                    "feature_channels": feature_channels,
                    "bytes": feat_bytes,
                    "source_tier": "sram",
                    "dependency_tag": feat_gather_tag,
                    "depends_on_tags": dep_tags,
                },
            )
            events.append(ev_feat)
            eid += 1
            cursor += ev_feat.duration
        else:
            dur, energy_nj = query_dram_cycles_energy(
                feat_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
            )
            dur = max(1, int(dur))
            power = _energy_to_power_mw(energy_nj, dur, clk)
            events.append(_event(
                eid, cursor, "DRAM", "feature_gather", "feature_index_dram_read",
                "dram", "feature_gather_dram_read",
                power, dur, clk,
                {
                    "phase": "sample_and_group",
                    "sub_op": "feature_gather",
                    "sampled_points": sampled_points,
                    "k": k,
                    "feature_channels": feature_channels,
                    "bytes": feat_bytes,
                    "source_tier": "dram",
                    "dependency_tag": feat_gather_tag,
                    "depends_on_tags": dep_tags,
                },
            ))
            eid += 1
            cursor += dur
    else:
        feat_gather_tag = rel_xyz_tag

    concat_channels = xyz_channels + feature_channels
    concat_bytes = max(1, sampled_points * k * concat_channels * bpe)
    concat_tag = f"{tag_scope}concat_ready"
    concat_deps = [rel_xyz_tag]
    if feature_channels > 0:
        concat_deps.append(feat_gather_tag)
    concat_cycles = max(1, safe_div_ceil(sampled_points * k, sub_par))
    events.append(_event(
        eid, cursor, gather_op, "concat", "xyz_feature_concat",
        "fp32_add", "gather_concat",
        sub_power * 0.5, concat_cycles, clk,
        {
            "phase": "sample_and_group",
            "sub_op": "concat",
            "xyz_channels": xyz_channels,
            "feature_channels": feature_channels,
            "concat_channels": concat_channels,
            "bytes": concat_bytes,
            "dependency_tag": concat_tag,
            "depends_on_tags": concat_deps,
        },
    ))
    eid += 1
    cursor += concat_cycles
    return (events, eid, cursor - start_cycle, concat_tag)


def gen_sample_and_group_query_events(
    query_idx: int,
    k: int,
    xyz_channels: int,
    feature_channels: int,
    hw_config: "HardwareConfig",
    start_cycle: int = 0,
    start_event_id: int = 0,
    *,
    feature_source: str = "auto",
    tag_scope: str = "",
    bq_ready_tag: str = "",
    current_layer_name: str = "",
) -> tuple:
    events, eid, total_cycles, concat_tag = _gen_sample_and_group_impl(
        sampled_points=1,
        k=k,
        xyz_channels=xyz_channels,
        feature_channels=feature_channels,
        hw_config=hw_config,
        start_cycle=start_cycle,
        start_event_id=start_event_id,
        feature_source=feature_source,
        tag_scope=tag_scope,
        bq_ready_tag=bq_ready_tag,
        current_layer_name=current_layer_name,
    )
    for ev in events:
        meta = dict(ev.metadata or {})
        meta["query"] = query_idx
        ev.metadata = meta
    return (events, eid, total_cycles, concat_tag)

def gen_sample_and_group_events(
    sampled_points: int,
    k: int,
    xyz_channels: int,
    feature_channels: int,
    hw_config: "HardwareConfig",
    start_cycle: int = 0,
    start_event_id: int = 0,
    *,
    feature_source: str = "auto",
    tag_scope: str = "",
    bq_ready_tag: str = "",
    current_layer_name: str = "",
) -> tuple:
    """
    BQ/KNN 输出索引后的 sample_and_group 建模。

    对应 model_back.py::sample_and_group:
      1. xyz_gather: index_points(xyz, bq_idx) → [npoint, k, 3] 从 xyz SRAM buffer 按索引读
      2. relative_xyz: grouped_xyz - new_xyz → 逻辑减法
      3. feature_gather: index_points(points, bq_idx) → [npoint, k, D] 从 feature buffer 或 DRAM 读
      4. concat: cat(relative_xyz, grouped_features) → [npoint, k, 3+D]

    Args:
        sampled_points: npoint（FPS 采样点数）
        k: nsample（每个球/knn 邻居数）
        xyz_channels: 坐标维度（默认 3）
        feature_channels: 上一层输出特征维度，0 表示首层无 feature
        hw_config: 硬件配置
        feature_source: "sram" | "dram" | "auto"
            auto: 若 feature 总字节 <= feature SRAM buffer 容量则 sram，否则 dram
        tag_scope: 依赖标签前缀
        bq_ready_tag: BQ/KNN 完成的依赖标签

    Returns:
        (events, next_event_id, total_cycles, gather_ready_tag)
    """
    return _gen_sample_and_group_impl(
        sampled_points=sampled_points,
        k=k,
        xyz_channels=xyz_channels,
        feature_channels=feature_channels,
        hw_config=hw_config,
        start_cycle=start_cycle,
        start_event_id=start_event_id,
        feature_source=feature_source,
        tag_scope=tag_scope,
        bq_ready_tag=bq_ready_tag,
        current_layer_name=current_layer_name,
    )


def gen_conv1d_per_group_events(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    group_points: int,
    query_idx: int,
    start_cycle: int = 0,
    start_event_id: int = 0,
    *,
    _template_cache: Optional[Dict[int, List[Event]]] = None,
) -> tuple:
    """
    Generate full Conv1d events for a single KNN query group.

    Calls ``gen_conv1d_systolic_events`` with ``num_points = group_points``,
    then shifts to *start_cycle* and assigns sequential event IDs.  Results
    are cached by *group_points* in *_template_cache* (caller-managed dict)
    so that SCALE-Sim is invoked only once per unique group size.

    Returns:
        (events, next_event_id, end_cycle)
    """
    import copy as _copy

    gp = max(1, int(group_points))

    # Retrieve or build the template events for this group size
    if _template_cache is not None and gp in _template_cache:
        template = _template_cache[gp]
    else:
        group_params = dict(params)
        group_params["num_points"] = gp
        template = gen_conv1d_systolic_events(group_params, hw_config)
        if _template_cache is not None:
            _template_cache[gp] = template

    if not template:
        return ([], start_event_id, start_cycle)

    # Deep-copy, shift and re-tag
    base_cycle = min(e.cycle for e in template)
    events: List[Event] = []
    eid = start_event_id
    for orig in template:
        ev = _copy.copy(orig)
        ev.event_id = eid
        ev.cycle = start_cycle + (orig.cycle - base_cycle)
        ev.metadata = dict(orig.metadata or {})
        ev.metadata["query"] = query_idx
        ev.metadata["pipeline"] = "knn_conv"
        eid += 1
        events.append(ev)

    end_cycle = max(e.cycle + e.duration for e in events)
    return (events, eid, end_cycle)


def gen_pointwise_conv_per_group_events(
    params: Dict[str, Any],
    hw_config: "HardwareConfig",
    group_points: int,
    query_idx: int,
    start_cycle: int = 0,
    start_event_id: int = 0,
    *,
    operator_kind: str = "conv1d",
    _template_cache: Optional[Dict[tuple, List[Event]]] = None,
    precomputed_lat_ns: Optional[float] = None,
    precomputed_en_nj: Optional[float] = None,
) -> tuple:
    """
    Per-query/group pointwise convolution events for PointNet++ SA/FP pipelines.

    `conv2d` here specifically denotes PointNet++ SA shared MLP with kernel_size=1,
    which is mapped to the same spatial work as `conv1d(num_points=query_points*k)`.
    """
    kind = str(operator_kind or "conv1d").strip().lower()
    import copy as _copy

    gp = max(1, int(group_points))
    cache_key = (kind, gp)

    if getattr(hw_config, "is_baseline", False):
        sub_cache = None
        if _template_cache is not None:
            sub_cache = {gp: _template_cache[cache_key]} if cache_key in _template_cache else {}
        events, eid, end_cycle = gen_conv1d_per_group_events(
            params,
            hw_config,
            group_points=gp,
            query_idx=query_idx,
            start_cycle=start_cycle,
            start_event_id=start_event_id,
            _template_cache=sub_cache,
        )
        if _template_cache is not None and sub_cache is not None and gp in sub_cache and cache_key not in _template_cache:
            _template_cache[cache_key] = sub_cache[gp]
    else:
        if precomputed_lat_ns is not None and precomputed_en_nj is not None:
            # Gap B fix: use neurosim_full-derived per-group latency/energy directly.
            # Bypass gen_conv1d_rram_cim_events to keep results consistent with the
            # full-network NeuroSim estimation (which already accounts for all timing).
            # Keep Conv/Linear on the same NeuroSim clock domain (e.g. 100MHz => 10ns),
            # instead of collapsing to logic clock (1GHz => 1ns).
            clock_ns = float(_neurosim_clock_ns(hw_config))
            cycles = max(1, int(math.ceil(precomputed_lat_ns / clock_ns)))
            power_mw = (precomputed_en_nj * 1e3 / (cycles * clock_ns)) if (cycles * clock_ns) > 0 else 0.0
            op_name = "Conv2d" if kind == "conv2d" else "Conv1d"
            op_tag = "conv2d_neurosim_full_pg" if kind == "conv2d" else "conv1d_neurosim_full_pg"
            compute_ev = Event(
                event_id=start_event_id,
                cycle=start_cycle,
                operator=op_name,
                circuit_module="neurosim_tile_noc",
                operation=op_tag,
                power=power_mw,
                duration=cycles,
                clock_period_ns=clock_ns,
                metadata={
                    "source": "neurosim_full_pg",
                    "latency_ns": precomputed_lat_ns,
                    "energy_nj": precomputed_en_nj,
                    "query": query_idx,
                    "pipeline": "knn_conv",
                },
            )
            return ([compute_ev], start_event_id + 1, start_cycle + cycles)
        if _template_cache is not None and cache_key in _template_cache:
            template = _template_cache[cache_key]
        else:
            group_params = dict(params)
            group_params["num_points"] = gp
            template = gen_conv1d_rram_cim_events(group_params, hw_config)
            if kind == "conv2d":
                template = _retag_events(
                    template,
                    operator="Conv2d",
                    operation_replace={"conv1d": "conv2d"},
                    metadata_updates={"native_op": "pointnet2_sa_conv2d_k1"},
                )
            if _template_cache is not None:
                _template_cache[cache_key] = template

        if not template:
            return ([], start_event_id, start_cycle)

        base_cycle = min(e.cycle for e in template)
        events = []
        eid = start_event_id
        for orig in template:
            ev = _copy.copy(orig)
            ev.event_id = eid
            ev.cycle = start_cycle + (orig.cycle - base_cycle)
            ev.metadata = dict(orig.metadata or {})
            ev.metadata["query"] = query_idx
            ev.metadata["pipeline"] = "knn_conv"
            eid += 1
            events.append(ev)
        end_cycle = max(e.cycle + e.duration for e in events)

    if kind == "conv2d" and getattr(hw_config, "is_baseline", False):
        events = _retag_events(
            events,
            operator="Conv2d",
            operation_replace={"conv1d": "conv2d"},
            metadata_updates={"native_op": "pointnet2_sa_conv2d_k1"},
        )
    return (events, eid, end_cycle)


def gen_conv1d_systolic_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    Conv1d 的 baseline 路径（base_v1/base_v2）：显式 im2col 前处理引擎 + SCALE-Sim 脉动阵列。
    前处理阶段包含地址生成、源读取、pack/reorder、patch 写入 im2col buffer、
    以及 systolic 读取 im2col buffer 的矩阵。
    """
    scalesim_out = _try_scalesim_conv1d(params, hw_config)
    if scalesim_out is None:
        raise RuntimeError(
            "Conv1d in baseline (base_v1/base_v2) requires SCALE-Sim: set use_scalesim_for_systolic=True, "
            "ensure SCALE-Sim is installed and scalesim_root/config are valid, and that the layer run succeeds."
        )

    in_ch = int(params.get("in_channels", 1))
    out_ch = int(params.get("out_channels", 1))
    kernel = int(params.get("kernel_size", 1))
    num_points = int(params.get("num_points", 1))
    layer_name = str(params.get("layer_name", ""))
    clk = _clock(hw_config)
    L = max(1, num_points)
    M, K, N = out_ch, max(1, in_ch * kernel), L

    systolic_cycles, scalesim_meta = scalesim_out
    systolic_meta_base = {
        "M": M, "K": K, "N": N, "layer_name": layer_name, "source": "scalesim",
        "num_points": L, "in_channels": in_ch, "out_channels": out_ch, "kernel_size": kernel,
        "activation_input_shape": str(params.get("activation_input_shape") or f"1×{L} ch={in_ch}"),
        "activation_output_shape": str(params.get("activation_output_shape") or f"1×{L} ch={out_ch}"),
    }
    systolic_meta_base.update(scalesim_meta)
    mac_energy_nj = scalesim_meta.get("scalesim_mac_energy_nj")
    if mac_energy_nj is not None and systolic_cycles > 0 and clk > 0:
        sa_power = float(mac_energy_nj) * 1000.0 / (systolic_cycles * clk)
    else:
        macs = M * K * N
        sa_cfg = getattr(hw_config, "systolic_array", None)
        mac_power_mw = float(getattr(sa_cfg, "mac_power_mw", 7.144) or 7.144) if sa_cfg else 7.144
        sa_power = macs * mac_power_mw / max(1, systolic_cycles)

    events, cycle, ev_id, im2col_meta = _gen_im2col_engine_events(params, hw_config, start_cycle=0, start_event_id=0)
    systolic_meta_base.update(im2col_meta)
    events.append(_event(
        ev_id, cycle, "Conv1d", "systolic", "matrix_multiply", "systolic_array", "conv_systolic",
        sa_power, systolic_cycles, clk,
        systolic_meta_base,
    ))
    events.extend(_scalesim_memory_events(cycle, ev_id + 1, "Conv1d", clk, hw_config, systolic_meta_base))
    return events


def gen_linear_systolic_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """
    Linear 的 baseline 路径（base_v1/base_v2）：仅使用 SCALE-Sim 脉动模型（N×N 脉动阵列，无 im2col），FXP16 计算，不考虑量化/反量化。
    若未启用 use_scalesim_for_systolic 或 SCALE-Sim 不可用/失败，则 raise；不回退到其他路径。
    """
    scalesim_out = _try_scalesim_linear(params, hw_config)
    if scalesim_out is None:
        raise RuntimeError(
            "Linear in baseline (base_v1/base_v2) requires SCALE-Sim: set use_scalesim_for_systolic=True, "
            "ensure SCALE-Sim is installed and scalesim_root/config are valid, and that the layer run succeeds."
        )

    in_f = int(params.get("in_features", 1))
    out_f = int(params.get("out_features", 1))
    tokens = int(params.get("tokens", 1))
    layer_name = str(params.get("layer_name", ""))

    clk = _clock(hw_config)
    M, K, N = out_f, in_f, max(1, tokens)

    systolic_cycles, scalesim_meta = scalesim_out
    # 激活 shape 供输出/核对 SCALE-Sim 周期：Linear 映射为 ifmap (tokens,1) ch=in_f, filter (1,1) num_filter=out_f
    meta = {
        "M": M, "K": K, "N": N, "layer_name": layer_name, "source": "scalesim",
        "tokens": N, "in_features": in_f, "out_features": out_f,
        "activation_input_shape": str(params.get("activation_input_shape") or f"{N}×1 ch={in_f}"),
        "activation_output_shape": str(params.get("activation_output_shape") or f"{N}×1 ch={out_f}"),
    }
    meta.update(scalesim_meta)
    mac_energy_nj = meta.get("scalesim_mac_energy_nj")
    if mac_energy_nj is not None and systolic_cycles > 0 and clk > 0:
        sa_power = float(mac_energy_nj) * 1000.0 / (systolic_cycles * clk)
    else:
        macs = M * K * N
        sa_cfg = getattr(hw_config, "systolic_array", None)
        mac_power_mw = float(getattr(sa_cfg, "mac_power_mw", 7.144) or 7.144) if sa_cfg else 7.144
        sa_power = macs * mac_power_mw / max(1, systolic_cycles)

    ev = _event(
        0, 0, "Linear", "systolic", "matrix_multiply", "systolic_array", "linear_systolic",
        sa_power, systolic_cycles, clk,
        meta,
    )
    return [ev] + _scalesim_memory_events(0, 1, "Linear", clk, hw_config, meta)


def gen_conv1d_rram_cim_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """Conv1d 的 PE 级 RRAM-CIM 事件（无全网络 NeuroSim 注入时使用单层 NeuroSim/回退）；可选前量化+后反量化。算子名用 Conv1d（PascalCase）。"""
    in_ch = int(params.get("in_channels", 1))
    out_ch = int(params.get("out_channels", 1))
    kernel = int(params.get("kernel_size", 1))
    num_points = int(params.get("num_points", 1))
    layer_name = str(params.get("layer_name", ""))

    fallback_power = 25.0
    if hasattr(hw_config, "rram_cim_macro"):
        fallback_power = float(getattr(hw_config.rram_cim_macro, "power", fallback_power) or fallback_power)

    base_clk = _clock(hw_config)
    neurosim_clk = _neurosim_clock_ns(hw_config)
    neurosim_clk_freq_hz = getattr(hw_config, "neurosim_clk_freq_hz", None)
    if neurosim_clk_freq_hz is None and neurosim_clk > 0:
        neurosim_clk_freq_hz = 1e9 / neurosim_clk
    neurosim_mode = "pybind" if bool(getattr(hw_config, "neurosim_pybind", False)) else "subprocess"
    est = estimate_conv1d_neurosim(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        num_points=num_points,
        clock_period_ns=neurosim_clk,
        neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
        fallback_power_mw=fallback_power,
        clk_freq_hz=neurosim_clk_freq_hz,
        technode=getattr(hw_config, "neurosim_technode", None),
        adc_clk_freq_hz=getattr(hw_config, "neurosim_adc_clk_freq_hz", None),
        sar_column_power_w=getattr(hw_config, "neurosim_sar_column_power_w", None),
        num_col_muxed=int(getattr(hw_config, "neurosim_num_col_muxed", 0) or 0) or 8,
        clk_mode=str(getattr(hw_config, "neurosim_clk_mode", "capped") or "capped"),
        mode=neurosim_mode,
        allow_subarray_retry=not _use_fixed_neurosim_subarray(hw_config),
    )
    assert isinstance(est, NeuroSimEstimate)
    timing = _normalized_scheduler_timing(
        latency_ns=float(est.latency_ns),
        energy_nj=float(est.energy_nj),
        base_clock_ns=base_clk,
    )
    compute_ev = _event(
        0, 0, "Conv1d", "rram_cim", "matrix_vector", "rram_cim", "conv1d_rram_cim",
        float(timing["power_mw"]), int(timing["duration_cycles"]), float(timing["clock_period_ns"]),
        {
            "source": est.source,
            "latency_ns": float(est.latency_ns),
            "energy_nj": float(est.energy_nj),
            "native_clock_period_ns": float(neurosim_clk),
            "native_cycles": int(est.cycles),
            "scheduler_normalized": True,
        },
    )

    if not getattr(hw_config, "enable_w8a8_quant_dequant", False):
        return [compute_ev]

    # W8A8：前量化(激活 FP32→INT8)、后反量化(输出 INT8→FP32)，CIM 侧按 INT 路径建模
    in_numel = max(1, in_ch * num_points)
    out_numel = max(1, out_ch * num_points)
    q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
    d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Conv1d")
    q_ev, d_ev = q_evs[0], d_evs[0]
    c0 = q_ev.cycle + q_ev.duration
    compute_ev.cycle = c0
    d_ev.cycle = c0 + compute_ev.duration
    return [q_ev, compute_ev, d_ev]


def gen_linear_rram_cim_events(params: Dict[str, Any], hw_config: "HardwareConfig") -> List[Event]:
    """Linear 的 PE 级 RRAM-CIM 事件（无全网络 NeuroSim 注入时使用单层 NeuroSim/回退）；可选前量化+后反量化。算子名用 Linear（PascalCase）。"""
    in_f = int(params.get("in_features", 1))
    out_f = int(params.get("out_features", 1))
    tokens = int(params.get("tokens", 1))
    layer_name = str(params.get("layer_name", ""))

    fallback_power = 20.0
    if hasattr(hw_config, "rram_cim_macro"):
        fallback_power = float(getattr(hw_config.rram_cim_macro, "power", fallback_power) or fallback_power)

    base_clk = _clock(hw_config)
    neurosim_clk = _neurosim_clock_ns(hw_config)
    neurosim_clk_freq_hz = getattr(hw_config, "neurosim_clk_freq_hz", None)
    if neurosim_clk_freq_hz is None and neurosim_clk > 0:
        neurosim_clk_freq_hz = 1e9 / neurosim_clk
    neurosim_mode = "pybind" if bool(getattr(hw_config, "neurosim_pybind", False)) else "subprocess"
    est = estimate_linear_neurosim(
        in_features=in_f,
        out_features=out_f,
        tokens=max(1, tokens),
        clock_period_ns=neurosim_clk,
        neurosim_root=getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
        fallback_power_mw=fallback_power,
        clk_freq_hz=neurosim_clk_freq_hz,
        technode=getattr(hw_config, "neurosim_technode", None),
        adc_clk_freq_hz=getattr(hw_config, "neurosim_adc_clk_freq_hz", None),
        sar_column_power_w=getattr(hw_config, "neurosim_sar_column_power_w", None),
        num_col_muxed=int(getattr(hw_config, "neurosim_num_col_muxed", 0) or 0) or 8,
        clk_mode=str(getattr(hw_config, "neurosim_clk_mode", "capped") or "capped"),
        mode=neurosim_mode,
        allow_subarray_retry=not _use_fixed_neurosim_subarray(hw_config),
    )
    assert isinstance(est, NeuroSimEstimate)
    timing = _normalized_scheduler_timing(
        latency_ns=float(est.latency_ns),
        energy_nj=float(est.energy_nj),
        base_clock_ns=base_clk,
    )
    compute_ev = _event(
        0, 0, "Linear", "rram_cim", "matrix_vector", "rram_cim", "linear_rram_cim",
        float(timing["power_mw"]), int(timing["duration_cycles"]), float(timing["clock_period_ns"]),
        {
            "source": est.source,
            "latency_ns": float(est.latency_ns),
            "energy_nj": float(est.energy_nj),
            "native_clock_period_ns": float(neurosim_clk),
            "native_cycles": int(est.cycles),
            "scheduler_normalized": True,
        },
    )

    if not getattr(hw_config, "enable_w8a8_quant_dequant", False):
        return [compute_ev]

    # W8A8：前量化(激活 FP32→INT8)、后反量化(输出 INT8→FP32)，CIM 侧按 INT 路径建模
    in_numel = max(1, in_f * max(1, tokens))
    out_numel = max(1, out_f * max(1, tokens))
    q_evs = gen_quant_events(in_numel, hw_config, layer_name=layer_name, op_name="Linear")
    d_evs = gen_dequant_events(out_numel, hw_config, layer_name=layer_name, op_name="Linear")
    q_ev, d_ev = q_evs[0], d_evs[0]
    c0 = q_ev.cycle + q_ev.duration
    compute_ev.cycle = c0
    d_ev.cycle = c0 + compute_ev.duration
    return [q_ev, compute_ev, d_ev]
