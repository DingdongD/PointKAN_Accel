"""
层间 Buffer 读写事件
====================

根据每层输入/输出字节数与 buffer 容量、带宽，生成「从 buffer 读入」「写回 buffer」事件；
当数据量超过 buffer_size 时，超出部分按 spill 到 DRAM 建模（可选），以模拟实际内存行为。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..simulator.event import Event
from ..utils.math_utils import safe_div_ceil

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


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
    """构造一条层间 buffer/DRAM 相关事件（operator 如 LayerBuffer）。"""
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


def gen_layer_buffer_read_events(
    input_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    生成本层「从 buffer（及可选 DRAM spill）读入输入」的事件。

    若 input_bytes <= buffer_size：仅从片上 buffer 读，周期 = ceil(input_bytes / bandwidth)。
    若 input_bytes > buffer_size 且 spill_to_dram：先读 buffer_size 从 buffer，再读剩余从 DRAM。

    Args:
        input_bytes: 本层输入数据量（字节）
        hw_config: 硬件配置（含 inter_layer_buffer）
        layer_name: 层名，写入 metadata
        clock_period_ns: 时钟周期，默认从 hw_config.clock_period 取

    Returns:
        按时间顺序的 Event 列表（相对周期从 0 开始），供调用方做 cycle 偏移。
    """
    if input_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    size = max(1, buf.size_bytes)
    bw = max(1, buf.bandwidth_bytes_per_cycle)
    events: List[Event] = []
    cycle = 0
    event_id = 0
    meta_base = {"layer_name": layer_name, "buffer_phase": "input_read"}

    # 从 buffer 读入的部分（最多 size_bytes）
    read_from_buf = min(input_bytes, size)
    if read_from_buf > 0:
        dur = safe_div_ceil(read_from_buf, bw)
        events.append(_event(
            event_id, cycle,
            "LayerBuffer", "buffer_io", "buffer_read",
            "inter_layer_buffer", "buffer_read",
            buf.read_power_mw, dur, clk,
            {**meta_base, "bytes": read_from_buf, "source": "buffer"},
        ))
        event_id += 1
        cycle += dur

    # 超出部分从 DRAM 读（spill）
    spill_bytes = input_bytes - read_from_buf
    if spill_bytes > 0 and getattr(buf, "spill_to_dram", False):
        dram_bw = max(1, getattr(buf, "dram_bandwidth_bytes_per_cycle", 32))
        dur = safe_div_ceil(spill_bytes, dram_bw)
        events.append(_event(
            event_id, cycle,
            "LayerBuffer", "buffer_io", "dram_read",
            "dram", "inter_layer_spill_read",
            getattr(buf, "dram_read_power_mw", 50.0), dur, clk,
            {**meta_base, "bytes": spill_bytes, "source": "dram_spill"},
        ))
        event_id += 1
        cycle += dur

    return events


def gen_layer_buffer_write_events(
    output_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    生成本层「将输出写回 buffer（及可选 DRAM spill）」的事件。

    若 output_bytes <= buffer_size：仅写入片上 buffer。
    若 output_bytes > buffer_size 且 spill_to_dram：先写满 buffer，剩余写 DRAM。

    Args:
        output_bytes: 本层输出数据量（字节）
        hw_config: 硬件配置
        layer_name: 层名
        clock_period_ns: 时钟周期

    Returns:
        按时间顺序的 Event 列表（相对周期从 0 开始）。
    """
    if output_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    size = max(1, buf.size_bytes)
    bw = max(1, buf.bandwidth_bytes_per_cycle)
    events: List[Event] = []
    cycle = 0
    event_id = 0
    meta_base = {"layer_name": layer_name, "buffer_phase": "output_write"}

    write_to_buf = min(output_bytes, size)
    if write_to_buf > 0:
        dur = safe_div_ceil(write_to_buf, bw)
        events.append(_event(
            event_id, cycle,
            "LayerBuffer", "buffer_io", "buffer_write",
            "inter_layer_buffer", "buffer_write",
            buf.write_power_mw, dur, clk,
            {**meta_base, "bytes": write_to_buf, "target": "buffer"},
        ))
        event_id += 1
        cycle += dur

    spill_bytes = output_bytes - write_to_buf
    if spill_bytes > 0 and getattr(buf, "spill_to_dram", False):
        dram_bw = max(1, getattr(buf, "dram_bandwidth_bytes_per_cycle", 32))
        dur = safe_div_ceil(spill_bytes, dram_bw)
        events.append(_event(
            event_id, cycle,
            "LayerBuffer", "buffer_io", "dram_write",
            "dram", "inter_layer_spill_write",
            getattr(buf, "dram_write_power_mw", 50.0), dur, clk,
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
    dram = getattr(hw_config, "dram", None)
    dram_bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32)) if dram is not None else 32
    dram_power = float(getattr(dram, "read_power_mw", 50.0)) if dram is not None else 50.0
    dram_cycles = safe_div_ceil(query_bytes, dram_bw)

    events: List[Event] = []
    for i in range(num_queries):
        base_cycle = i * (per_query_cycles + (dram_cycles if include_query_dram_read else 0))
        event_id = i * (2 if include_query_dram_read else 1)
        if include_query_dram_read:
            events.append(
                _event(
                    event_id, base_cycle,
                    operator_name, "distance", "query_dram_read",
                    "dram", "phase2_query_point_read",
                    dram_power, dram_cycles, clk,
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
        events.append(
            _event(
                event_id, base_cycle,
                operator_name, "distance", "sram_ii_128row_2bit",
                "sram_ii", "phase2_distance_compute",
                power, per_query_cycles, clk,
                {"query_index": i, "num_keys": num_keys, "phase": 2},
            )
        )
    return events


def gen_first_layer_input_load_events(
    input_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    首层/Phase1 点云输入加载的唯一入口：按 input_bytes 生成 DRAM→SRAM-II 加载事件。
    调用方传入字节数（如 activation_shapes[0].input_bytes 或 num_points*num_features*bits/8）。
    若 hw_config.dram 存在则用 DRAM 配置（operation=phase1_point_cloud_to_sram_ii）；否则用 inter_layer_buffer。
    """
    if input_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    dram = getattr(hw_config, "dram", None)
    if dram is not None:
        dram_bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        power = getattr(dram, "read_power_mw", 50.0)
        op_name = "phase1_point_cloud_to_sram_ii"
    else:
        buf = getattr(hw_config, "inter_layer_buffer", None)
        if buf is None:
            return []
        dram_bw = max(1, getattr(buf, "dram_bandwidth_bytes_per_cycle", 32))
        power = getattr(buf, "dram_read_power_mw", 50.0)
        op_name = "point_cloud_input_load"
    dur = safe_div_ceil(input_bytes, dram_bw)
    return [
        _event(
            0, 0,
            "LayerBuffer", "buffer_io", "dram_read",
            "dram", op_name,
            power, dur, clk,
            {"layer_name": layer_name, "bytes": input_bytes, "source": "point_cloud_Nx3"},
        )
    ]
