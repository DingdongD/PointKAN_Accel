"""
距离计算事件（Baseline dist2_engine / Acc SRAM-CIM / PointCIM RRAM VMM 统一入口）
===================================================================================

- **Baseline**：依赖 hw_config.baseline_dist2_engine（BaselineDist2EngineConfig），
  1×64 FP32 PE，单批周期 T = (PE_count - 1) + per_distance_cycles；当前默认 70 cycle/64 点批。
  每批 1 条事件；与排序电路无关。
- **Acc**：SRAM-CIM 流水线，每批 Stage 1–3（k²、-2qk、shift_add 累加），3 条事件。
- **PointCIM**：RRAM VMM 流水线，每批 Stage 2–4（VMM、q²、累加），3 条事件。
  写入事件（Stage 1 rram_write）由 FPS/KNN 层初始化时单独调用 gen_rram_dist_write_events。

统一入口 gen_distance_events_one_batch 根据 hw_config 自动选择路径：
  1. PointCIM（use_rram_dist=True）→ RRAM VMM
  2. Baseline（is_baseline + baseline_dist2_engine 非空）→ dist2_engine
  3. 其余（ACC）→ SRAM-CIM
不包含 FPS 的 min_distance_table、KNN 的 output_buffer 写回及 Sorter，由 FPS/KNN 自行追加 Stage 4/5。
"""

from typing import List, Tuple, TYPE_CHECKING

from ..simulator.event import Event
from .rram_dist_ops import (
    gen_rram_dist_events_one_batch,
    gen_rram_dist_write_events,
    gen_rram_dist_self_k2_events,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def gen_baseline_dist2_events_one_batch(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    batch_size_m: int,
    metadata: dict,
) -> Tuple[List[Event], int, int]:
    """
    生成**单批** Baseline 距离计算事件（dist2_engine，1×64 FP32 PE）。

    周期 T = (PE_count - 1) + per_distance_cycles，依赖 BaselineDist2EngineConfig。
    当前默认单批固定 70 cycle。每批仅 1 条事件。
    """
    eng = getattr(hw_config, "baseline_dist2_engine", None)
    if eng is None:
        return [], start_event_id, start_cycle
    pe_count = max(1, int(getattr(eng, "pe_count", getattr(eng, "c_dim", 64)) or 64))
    per_distance_cycles = max(1, int(getattr(eng, "per_distance_cycles", 7) or 7))
    explicit_cycles = int(getattr(eng, "cycles", 0) or 0)
    duration = explicit_cycles if explicit_cycles > 0 else (pe_count - 1 + per_distance_cycles)
    power = eng.power
    clock_period = hw_config.clock_period
    meta = dict(metadata)

    events: List[Event] = [
        Event(
            event_id=start_event_id,
            cycle=start_cycle,
            operator=operator_name,
            stage="dist2_systolic",
            sub_stage="batch_distance",
            circuit_module="dist2_engine",
            operation="systolic_dist2_batch",
            power=power,
            duration=duration,
            clock_period_ns=clock_period,
            metadata=meta,
        )
    ]
    return events, start_event_id + 1, start_cycle + duration


def gen_distance_events_one_batch(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    metadata: dict,
    *,
    dist_cycles: int = 0,
    batch_size_m: int = 0,
) -> Tuple[List[Event], int, int]:
    """
    单批距离计算统一入口，按模式选择路径：
      1. PointCIM（use_rram_dist=True）→ RRAM VMM（3 事件：VMM、q²、累加）
      2. Baseline（is_baseline + baseline_dist2_engine 非空）→ dist2_engine（1 事件）
      3. Acc（默认）→ SRAM-CIM（3 事件：k²、-2qk、shift_add）

    Args:
        dist_cycles:   Acc/SRAM-CIM 路径下每批 k²/-2qk 周期数。
        batch_size_m:  Baseline 路径下本批点数 M，用于 T=M+R+C-2。
    """
    # PointCIM：RRAM VMM 路径
    if getattr(hw_config, "use_rram_dist", False) and getattr(hw_config, "rram_dist", None) is not None:
        num_points = int(metadata.get("num_points", metadata.get("input_points", batch_size_m or 1024)))
        return gen_rram_dist_events_one_batch(
            operator_name, hw_config, start_event_id, start_cycle, metadata,
            num_points=num_points,
        )
    # Baseline：dist2_engine 脉动阵列路径
    if hw_config.is_baseline and getattr(hw_config, "baseline_dist2_engine", None) is not None:
        return gen_baseline_dist2_events_one_batch(
            operator_name, hw_config, start_event_id, start_cycle, batch_size_m or 128, metadata
        )
    # Acc（默认）：SRAM-CIM 路径
    return gen_sram_cim_distance_events_one_batch(
        operator_name, hw_config, start_event_id, start_cycle, dist_cycles, metadata
    )


def gen_sram_cim_distance_events_one_batch(
    operator_name: str,
    hw_config: "HardwareConfig",
    start_event_id: int,
    start_cycle: int,
    dist_cycles: int,
    metadata: dict,
) -> Tuple[List[Event], int, int]:
    """
    生成**单批** SRAM-CIM 距离计算事件（Stage 1–3：k²、-2qk、shift_add 累加），供 FPS/KNN 在每批循环内复用。

    Args:
        operator_name: 算子名，如 "FPS" 或 "KNN"
        hw_config: 硬件配置
        start_event_id: 起始 event_id
        start_cycle: 起始 cycle
        dist_cycles: 本批 k² / -2qk 的周期数
        metadata: 本批 metadata（如 {"iteration": i, "batch": b}）

    Returns:
        (events, next_event_id, next_cycle)：3 条事件、下一 event_id、下一 cycle（调用方随后追加该批的 Stage 4）
    """
    cim_power = hw_config.sram_cim_macro.array.cim_power
    shift_add_power = hw_config.sram_cim_macro.shift_add.power
    merge_cfg = getattr(hw_config.sram_cim_macro, "dist2_merge_adder", None)
    merge_power = float(getattr(merge_cfg, "power", shift_add_power) if merge_cfg is not None else shift_add_power)
    merge_cycles = int(getattr(merge_cfg, "cycles", 1) if merge_cfg is not None else 1)
    clock_period = hw_config.clock_period
    meta = dict(metadata)

    events: List[Event] = []
    event_id = start_event_id
    cycle = start_cycle

    # Stage 1: k^2
    events.append(Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator_name,
        stage="k_square",
        sub_stage="k_square_compute",
        circuit_module="sram_cim_array",
        operation="k2_bit_serial",
        power=cim_power + shift_add_power,
        duration=dist_cycles,
        clock_period_ns=clock_period,
        metadata=meta,
    ))
    event_id += 1
    cycle += dist_cycles

    # Stage 2: -2qk
    events.append(Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator_name,
        stage="cross_term",
        sub_stage="cross_term_compute",
        circuit_module="sram_cim_array",
        operation="two_qk_bit_serial",
        power=cim_power + shift_add_power,
        duration=dist_cycles,
        clock_period_ns=clock_period,
        metadata=meta,
    ))
    event_id += 1
    cycle += dist_cycles

    # Stage 3: accumulate
    events.append(Event(
        event_id=event_id,
        cycle=cycle,
        operator=operator_name,
        stage="accumulate",
        sub_stage="dist2_merge_accumulate",
        circuit_module="dist2_merge_adder",
        operation="distance_merge_accumulate",
        power=merge_power,
        duration=max(1, merge_cycles),
        clock_period_ns=clock_period,
        metadata=meta,
    ))
    event_id += 1
    cycle += max(1, merge_cycles)

    return events, event_id, cycle
