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


def gen_first_layer_input_load_events(
    input_bytes: int,
    hw_config: "HardwareConfig",
    layer_name: str,
    clock_period_ns: Optional[float] = None,
) -> List[Event]:
    """
    生成第一层「点云输入从 DRAM/外部加载」事件（N×3 数据喂入）。

    用于模拟真实点云样本输入：每个样本 N×3（N 点数，3 为 xyz），
    按字节数折算为 DRAM 读周期与功耗。

    Args:
        input_bytes: 输入数据字节数（通常 N * 3 * bytes_per_element）
        hw_config: 硬件配置（使用 inter_layer_buffer 的 dram 参数）
        layer_name: 层名
        clock_period_ns: 时钟周期

    Returns:
        Event 列表（单段 DRAM 读或空）。
    """
    if input_bytes <= 0:
        return []

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return []

    dram_bw = max(1, getattr(buf, "dram_bandwidth_bytes_per_cycle", 32))
    dur = _safe_div_ceil(input_bytes, dram_bw)
    power = getattr(buf, "dram_read_power_mw", 50.0)
    return [
        _event(
            0, 0,
            "LayerBuffer", "buffer_io", "dram_read",
            "dram", "point_cloud_input_load",
            power, dur, clk,
            {"layer_name": layer_name, "bytes": input_bytes, "source": "point_cloud_Nx3"},
        )
    ]
