"""
距离计算事件（Baseline dist2_engine / Acc SRAM-CIM 统一入口）
=============================================================

- **Baseline**：依赖 hw_config.baseline_dist2_engine（BaselineDist2EngineConfig），
  3×64 脉动阵列，T = M + R + C - 2（M=批内点数，R=3，C=64），每批 1 条事件；与排序电路无关。
- **Acc**：SRAM-CIM 流水线，每批 Stage 1–3（k²、-2qk、shift_add 累加），3 条事件。

统一入口 gen_distance_events_one_batch 根据 hw_config 自动选择 baseline 或 acc 路径。
不包含 FPS 的 min_distance_table、KNN 的 output_buffer 写回及 Sorter，由 FPS/KNN 自行追加 Stage 4/5。
"""

from typing import List, Tuple, TYPE_CHECKING

from ..simulator.event import Event

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
    生成**单批** Baseline 距离计算事件（dist2_engine，3×64 脉动阵列）。

    周期 T = M + R + C - 2（M=批内点数，R=3，C=64），依赖 BaselineDist2EngineConfig。
    每批仅 1 条事件（整批由脉动阵列一次完成）。
    """
    eng = getattr(hw_config, "baseline_dist2_engine", None)
    if eng is None:
        return [], start_event_id, start_cycle
    m = max(1, batch_size_m)
    r = getattr(eng, "r_dim", 3)
    c = getattr(eng, "c_dim", 64)
    duration = m + r + c - 2
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
    单批距离计算统一入口：Baseline 用 dist2_engine，Acc 用 SRAM-CIM。

    - Baseline（hw_config.is_baseline 且 baseline_dist2_engine 非空）：使用 batch_size_m 计算 T=M+R+C-2。
    - Acc：使用 dist_cycles 生成 k²、-2qk、shift_add 三条事件。

    Args:
        dist_cycles: Acc 路径下每批 k²/-2qk 周期数（仅 acc 使用）。
        batch_size_m: Baseline 路径下本批点数 M（仅 baseline 使用），用于 T=M+R+C-2。
    """
    if hw_config.is_baseline and getattr(hw_config, "baseline_dist2_engine", None) is not None:
        return gen_baseline_dist2_events_one_batch(
            operator_name, hw_config, start_event_id, start_cycle, batch_size_m or 128, metadata
        )
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
        power=cim_power,
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
        power=cim_power,
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
        sub_stage="shift_add_accumulate",
        circuit_module="shift_add",
        operation="distance_accumulate",
        power=shift_add_power,
        duration=1,
        clock_period_ns=clock_period,
        metadata=meta,
    ))
    event_id += 1
    cycle += 1

    return events, event_id, cycle
