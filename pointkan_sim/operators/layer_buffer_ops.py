"""
层间 Buffer 读写事件
====================

根据每层输入/输出字节数与 buffer 容量，生成「从 buffer 读入」「写回 buffer」事件；
当数据量超过 buffer_size 时，超出部分按 spill 到 DRAM 建模（可选）。
周期与能耗统一走 pointkan_sim.integration.memory_model_adapter（参考 memory_model demo_pointcloud_pipeline）：
仅使用 CACTI/DRAMSim3 回填的访问级模型；未回填时周期仍按带宽，能耗为 0。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from ..simulator.event import Event
from ..integration.memory_model_adapter import query_sram_cycles_energy, query_dram_cycles_energy
from ..integration.memory_runtime import MemoryRuntimeContext
from ..memory.feature_residency import get_effective_feature_buffer_bytes

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def _split_bytes(total_bytes: int, chunk_bytes: int) -> List[int]:
    total = max(0, int(total_bytes))
    chunk = max(1, int(chunk_bytes))
    out: List[int] = []
    while total > 0:
        take = min(total, chunk)
        out.append(take)
        total -= take
    return out


def _event(
    event_id: int,
    cycle: int,
    operator: str,
    stage: str,
    sub_stage: str,
    circuit_module: str,
    operation: str,
    duration: int,
    energy_nj: float,
    clock_period_ns: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """构造一条层间 buffer/DRAM 相关事件。power 由 energy_nj 反推以使 Event.energy == energy_nj。"""
    clk = float(clock_period_ns)
    dur = max(1, int(duration))
    power = (energy_nj * 1000.0 / (dur * clk)) if (dur > 0 and clk > 0) else 0.0
    return Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator,
        stage=stage,
        sub_stage=sub_stage,
        circuit_module=circuit_module,
        operation=operation,
        power=power,
        duration=dur,
        clock_period_ns=clock_period_ns,
        metadata=metadata or {},
    )


def gen_layer_buffer_read_events(
    input_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
    dram_precomputed: Optional[Iterator[Tuple[int, float]]] = None,
    memory_context: Optional[MemoryRuntimeContext] = None,
) -> List[Event]:
    """
    生成本层「从 buffer（及可选 DRAM spill）读入输入」的事件。
    """
    if input_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    size = max(1, get_effective_feature_buffer_bytes(hw_config) or buf.size_bytes)
    events: List[Event] = []
    cycle = 0
    event_id = 0
    meta_base = {"layer_name": layer_name, "buffer_phase": "input_read"}

    read_from_buf = min(input_bytes, size)
    if read_from_buf > 0:
        sram_chunks = _split_bytes(read_from_buf, size)
        total_chunks = len(sram_chunks)
        for chunk_idx, chunk_bytes in enumerate(sram_chunks):
            if memory_context is not None:
                stat = memory_context.sram_read(chunk_bytes, cycle)
                dur, energy_nj = stat.service_cycles, stat.dynamic_energy_nj
            else:
                dur, energy_nj = query_sram_cycles_energy(chunk_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk)
            events.append(_event(
                event_id, cycle,
                "LayerBuffer", "buffer_io", "buffer_read",
                "inter_layer_buffer", "buffer_read",
                dur, energy_nj, clk,
                {**meta_base, "bytes": chunk_bytes, "source": "buffer", "chunk_idx": chunk_idx, "chunk_count": total_chunks},
            ))
            event_id += 1
            cycle += dur

    spill_bytes = input_bytes - read_from_buf
    if spill_bytes > 0 and getattr(buf, "spill_to_dram", False):
        if dram_precomputed is not None:
            dur, energy_nj = next(dram_precomputed)
        elif memory_context is not None:
            res = memory_context.dram_transfer(spill_bytes, is_read=True, cur_cycle=cycle)
            dur, energy_nj = res["cycles"], res["energy_nj"]
        else:
            dur, energy_nj = query_dram_cycles_energy(
                spill_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
                bandwidth_bytes_per_cycle=getattr(buf, "dram_bandwidth_bytes_per_cycle", 32),
                read_power_mw=getattr(buf, "dram_read_power_mw", 50.0),
                write_power_mw=getattr(buf, "dram_write_power_mw", 50.0),
            )
        events.append(_event(
            event_id, cycle,
            "DRAM", "buffer_io", "dram_read",
            "dram", "inter_layer_spill_read",
            dur, energy_nj, clk,
            {**meta_base, "bytes": spill_bytes, "source": "dram_spill"},
        ))
        cycle += dur

    return events


def gen_layer_buffer_write_events(
    output_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
    dram_precomputed: Optional[Iterator[Tuple[int, float]]] = None,
    memory_context: Optional[MemoryRuntimeContext] = None,
) -> List[Event]:
    """
    生成本层「将输出写回 buffer（及可选 DRAM spill）」的事件。
    """
    if output_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    size = max(1, get_effective_feature_buffer_bytes(hw_config) or buf.size_bytes)
    events: List[Event] = []
    cycle = 0
    event_id = 0
    meta_base = {"layer_name": layer_name, "buffer_phase": "output_write"}

    write_to_buf = min(output_bytes, size)
    if write_to_buf > 0:
        sram_chunks = _split_bytes(write_to_buf, size)
        total_chunks = len(sram_chunks)
        for chunk_idx, chunk_bytes in enumerate(sram_chunks):
            if memory_context is not None:
                stat = memory_context.sram_write(chunk_bytes, cycle)
                dur, energy_nj = stat.service_cycles, stat.dynamic_energy_nj
            else:
                dur, energy_nj = query_sram_cycles_energy(chunk_bytes, is_read=False, hw_config=hw_config, clock_period_ns=clk)
            events.append(_event(
                event_id, cycle,
                "LayerBuffer", "buffer_io", "buffer_write",
                "inter_layer_buffer", "buffer_write",
                dur, energy_nj, clk,
                {**meta_base, "bytes": chunk_bytes, "target": "buffer", "chunk_idx": chunk_idx, "chunk_count": total_chunks},
            ))
            event_id += 1
            cycle += dur

    spill_bytes = output_bytes - write_to_buf
    if spill_bytes > 0 and getattr(buf, "spill_to_dram", False):
        if dram_precomputed is not None:
            dur, energy_nj = next(dram_precomputed)
        elif memory_context is not None:
            res = memory_context.dram_transfer(spill_bytes, is_read=False, cur_cycle=cycle)
            dur, energy_nj = res["cycles"], res["energy_nj"]
        else:
            dur, energy_nj = query_dram_cycles_energy(
                spill_bytes, is_read=False, hw_config=hw_config, clock_period_ns=clk,
                bandwidth_bytes_per_cycle=getattr(buf, "dram_bandwidth_bytes_per_cycle", 32),
                read_power_mw=getattr(buf, "dram_read_power_mw", 50.0),
                write_power_mw=getattr(buf, "dram_write_power_mw", 50.0),
            )
        events.append(_event(
            event_id, cycle,
            "DRAM", "buffer_io", "dram_write",
            "dram", "inter_layer_spill_write",
            dur, energy_nj, clk,
            {**meta_base, "bytes": spill_bytes, "target": "dram_spill"},
        ))

    return events


def gen_phase2_distance_events(
    num_queries: int,
    num_keys: int,
    hw_config: "HardwareConfig",
    operator_name: str = "DistanceCompute",
    num_features: int = 3,
    bits_per_feature: int = 12,
    include_query_dram_read: bool = True,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    Phase2：距离计算。每查询点 (12/2)*ceil(N/128) 周期，128 行 SRAM-II 激活、2bit 距离。
    用于 FPS/KNN 前：每迭代/每查询先跑本阶段再喂入 FPS bitonic 或 KNN 排序。

    增量改造：可选建模「查询点从 DRAM 读取」事件。
    - 查询点大小：1*1*num_features（默认 xyz=3），每特征 bits_per_feature（默认 12bit）
    - 若 include_query_dram_read=True，则每个 query 先插入 DRAM 读事件，再插入距离计算事件。
    """
    import math
    if num_queries <= 0 or num_keys <= 0:
        return []
    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    sram_ii = getattr(hw_config, "sram_ii", None)
    if sram_ii is not None:
        active = max(1, getattr(sram_ii, "active_rows", 128))
        bits = max(1, getattr(sram_ii, "bits_per_axis", 12))
        slice_bits = max(1, getattr(sram_ii, "bit_slice", 2))
        power = getattr(sram_ii, "power_mw", 0.0)
    else:
        active, bits, slice_bits, power = 128, 12, 2, 0.0
    per_query_cycles = (bits // slice_bits) * max(1, int(math.ceil(num_keys / active)))
    query_bytes = int(math.ceil(max(1, num_features) * max(1, bits_per_feature) / 8.0))
    dram_cycles, dram_energy_nj = 0, 0.0
    if include_query_dram_read and query_bytes > 0:
        dram_cycles, dram_energy_nj = query_dram_cycles_energy(
            query_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
        )

    events: List[Event] = []
    for i in range(num_queries):
        base_cycle = i * (per_query_cycles + (dram_cycles if include_query_dram_read else 0))
        event_id = i * (2 if include_query_dram_read else 1)
        if include_query_dram_read and dram_cycles > 0:
            events.append(
                _event(
                    event_id, base_cycle,
                    operator_name, "distance", "query_dram_read",
                    "dram", "phase2_query_point_read",
                    dram_cycles, dram_energy_nj, clk,
                    {
                        "query_index": i,
                        "bytes": query_bytes,
                        "phase": 2,
                        "kind": "query_fetch",
                    },
                )
            )
            base_cycle += dram_cycles
            event_id += 1
        dist_energy_nj = power * per_query_cycles * clk / 1000.0
        events.append(
            _event(
                event_id, base_cycle,
                operator_name, "distance", "sram_ii_128row_2bit",
                "sram_ii", "phase2_distance_compute",
                per_query_cycles, dist_energy_nj, clk,
                {"query_index": i, "num_keys": num_keys, "phase": 2},
            )
        )
    return events


def gen_first_layer_input_load_events(
    input_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
    dram_precomputed: Optional[Iterator[Tuple[int, float]]] = None,
    memory_context: Optional[MemoryRuntimeContext] = None,
) -> List[Event]:
    """
    首层/Phase1 点云输入加载的唯一入口：按 input_bytes 生成 DRAM→SRAM-II 加载事件。
    若提供 dram_precomputed（trace→DRAMSim3 预计算），则优先使用其 (cycles, energy_nj)。
    """
    if input_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    dram = getattr(hw_config, "dram", None)
    if dram_precomputed is not None:
        dur, energy_nj = next(dram_precomputed)
        op_name = "phase1_point_cloud_to_sram_ii" if dram is not None else "point_cloud_input_load"
    elif memory_context is not None:
        res = memory_context.dram_transfer(input_bytes, is_read=True, cur_cycle=0)
        dur, energy_nj = res["cycles"], res["energy_nj"]
        op_name = "phase1_point_cloud_to_sram_ii" if dram is not None else "point_cloud_input_load"
    elif dram is not None:
        op_name = "phase1_point_cloud_to_sram_ii"
        dur, energy_nj = query_dram_cycles_energy(input_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk)
    else:
        buf = getattr(hw_config, "inter_layer_buffer", None)
        if buf is None:
            return []
        op_name = "point_cloud_input_load"
        dur, energy_nj = query_dram_cycles_energy(
            input_bytes, is_read=True, hw_config=hw_config, clock_period_ns=clk,
            bandwidth_bytes_per_cycle=getattr(buf, "dram_bandwidth_bytes_per_cycle", 32),
            read_power_mw=getattr(buf, "dram_read_power_mw", 50.0),
            write_power_mw=getattr(buf, "dram_write_power_mw", 50.0),
        )
    return [
        _event(
            0, 0,
            "DRAM", "buffer_io", "dram_read",
            "dram", op_name,
            dur, energy_nj, clk,
            {"layer_name": layer_name, "bytes": input_bytes, "source": "point_cloud_Nx3"},
        )
    ]
