"""
层间 Buffer 读写事件
====================

根据每层输入/输出字节数与 buffer 容量、带宽，生成「从 buffer 读入」「写回 buffer」事件；
当数据量超过 buffer_size 时，超出部分按 spill 到 DRAM 建模（可选），以模拟实际内存行为。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..simulator.event import Event

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def _safe_div_ceil(a: int, b: int) -> int:
    return max(1, math.ceil(max(1, a) / max(1, b)))


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
        dur = _safe_div_ceil(read_from_buf, bw)
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
        dur = _safe_div_ceil(spill_bytes, dram_bw)
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
        dur = _safe_div_ceil(write_to_buf, bw)
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
        dur = _safe_div_ceil(spill_bytes, dram_bw)
        events.append(_event(
            event_id, cycle,
            "LayerBuffer", "buffer_io", "dram_write",
            "dram", "inter_layer_spill_write",
            getattr(buf, "dram_write_power_mw", 50.0), dur, clk,
            {**meta_base, "bytes": spill_bytes, "target": "dram_spill"},
        ))

    return events


def gen_phase1_point_cloud_load_events(
    num_points: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    num_features: int = 3,
    bits_per_feature: int = 12,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    Phase1：系统启动，从 DRAM 加载 1×N×3 点云（每特征 12bit）到 SRAM-II。
    Roofline：cycles = bytes / dram.bandwidth；能耗 = DRAM 读功耗 × 时间。
    """
    import math
    bytes_total = int(math.ceil(num_points * num_features * bits_per_feature / 8.0))
    if bytes_total <= 0:
        return []
    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    dram = getattr(hw_config, "dram", None)
    if dram is not None:
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        power = getattr(dram, "read_power_mw", 50.0)
    else:
        buf = getattr(hw_config, "inter_layer_buffer", None)
        bw = max(1, getattr(buf, "dram_bandwidth_bytes_per_cycle", 32)) if buf else 32
        power = getattr(buf, "dram_read_power_mw", 50.0) if buf else 50.0
    dur = _safe_div_ceil(bytes_total, bw)
    return [
        _event(
            0, 0,
            "LayerBuffer", "buffer_io", "dram_read",
            "dram", "phase1_point_cloud_to_sram_ii",
            power, dur, clk,
            {"layer_name": layer_name, "bytes": bytes_total, "source": "point_cloud_1_N_3", "phase": 1},
        )
    ]


def gen_phase2_distance_events(
    num_queries: int,
    num_keys: int,
    hw_config: "HardwareConfig",
    operator_name: str = "DistanceCompute",
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    Phase2：距离计算。每查询点 (12/2)*ceil(N/128) 周期，128 行 SRAM-II 激活、2bit 距离。
    用于 FPS/KNN 前：每迭代/每查询先跑本阶段再喂入 FPS bitonic 或 KNN 排序。
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
    events: List[Event] = []
    for i in range(num_queries):
        events.append(
            _event(
                i, i * per_query_cycles,
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
    生成第一层「点云输入从 DRAM/外部加载」事件（N×3 数据喂入）。

    若 hw_config.dram 存在则用 DRAM 配置（Phase1 语义）；否则回退到 inter_layer_buffer.dram。
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
    dur = _safe_div_ceil(input_bytes, dram_bw)
    return [
        _event(
            0, 0,
            "LayerBuffer", "buffer_io", "dram_read",
            "dram", op_name,
            power, dur, clk,
            {"layer_name": layer_name, "bytes": input_bytes, "source": "point_cloud_Nx3"},
        )
    ]
