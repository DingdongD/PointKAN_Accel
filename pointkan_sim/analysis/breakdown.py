"""
Breakdown 分析模块
=================

对仿真事件进行多维度聚合，供 PPA 报告与饼图使用：
1. 按电路模块（circuit_module）聚合能耗
2. 按阶段/子阶段（stage/sub_stage）聚合周期与能耗
3. 按算子（operator）聚合，用于 FPS/KNN/Conv/Linear 等对比

与 main.py 配合：仿真结束后调用 analyze_simulation_events_breakdown，
得到 by_layer / by_operator / by_module（均含 cycles、energy_nj、event_count）。

概念说明：
- Activity Cycles：事件 duration 之和（流水线可导致大于 wall-clock 总周期）
- Total Cycles：实际墙钟周期
- Cycle% / Energy%：相对占比，用于饼图
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..operators.base_operator import BaseOperator
    from ..configs.hw_config import HardwareConfig
    from ..simulator.event import Event


def _get_cim_parallelism(hw_config: 'HardwareConfig') -> int:
    """Extract CIM parallelism factor from flat hw_config."""
    return hw_config.cim_parallelism


@dataclass
class ModuleBreakdown:
    """单个电路模块的周期/能耗汇总。"""
    name: str
    activity_cycles: int = 0   # 该模块上所有事件 duration 之和
    energy_nj: float = 0.0
    event_count: int = 0
    avg_power_mw: float = 0.0


@dataclass
class StageBreakdown:
    """单个阶段或子阶段的周期/能耗汇总。"""
    name: str
    activity_cycles: int = 0
    energy_nj: float = 0.0
    event_count: int = 0


@dataclass
class OperatorBreakdown:
    """单个算子的完整 breakdown（总周期、总能耗、按 stage/module 细分）。"""
    name: str
    total_cycles: int = 0             # 墙钟周期（calculate_cycles）
    total_activity_cycles: int = 0    # 所有事件 duration 之和
    total_energy_nj: float = 0.0
    latency_ms: float = 0.0

    stages: Dict[str, StageBreakdown] = field(default_factory=dict)
    sub_stages: Dict[str, StageBreakdown] = field(default_factory=dict)
    modules: Dict[str, ModuleBreakdown] = field(default_factory=dict)

    parallelism_factor: int = 1
    clock_period_ns: float = 100.0


def analyze_operator_breakdown(
    operator: 'BaseOperator',
    hw_config: 'HardwareConfig',
    events: Optional[List['Event']] = None
) -> OperatorBreakdown:
    """
    分析单个算子，返回按 stage/module 细分的周期与能耗。
    事件能耗为单 PE 份额，内部按 total_pes 缩放为芯片级能耗；周期不缩放。
    Args:
        operator: 待分析算子
        hw_config: 硬件配置
        events: 预生成事件（可选，为 None 时内部生成）
    Returns:
        OperatorBreakdown，含总周期、总能耗及 stages/modules 细分
    """
    # Get parallelism info from flat architecture
    cim_parallelism = _get_cim_parallelism(hw_config)
    total_pes = cim_parallelism  # flat arch: no tile/PE hierarchy
    clock_period = hw_config.clock_period

    # Get operator metrics
    total_cycles = operator.calculate_cycles()

    # Generate events if not provided
    if events is None:
        events = operator.generate_events()

    # REFACTORED: Task A2 - Compute chip-level energy from events.
    # Events represent single-PE workload; multiply by total_pes for chip-level.
    single_pe_energy = sum(e.energy for e in events)
    total_energy = single_pe_energy * total_pes

    # Calculate total activity cycles (sum of all event durations)
    total_activity_cycles = sum(e.duration for e in events)

    # Create breakdown object
    breakdown = OperatorBreakdown(
        name=operator.name,
        total_cycles=total_cycles,
        total_activity_cycles=total_activity_cycles,
        total_energy_nj=total_energy,
        latency_ms=total_cycles * clock_period / 1e6,
        parallelism_factor=total_pes,
        clock_period_ns=clock_period
    )

    # REFACTORED: Task A2 - Scheme A scaling: event.energy is single-PE share,
    # multiply by total_pes to get chip-level energy for each breakdown category.

    # Analyze by stage
    stage_data = defaultdict(lambda: {'cycles': 0, 'energy': 0.0, 'events': 0})
    for event in events:
        stage = event.stage or 'unknown'
        stage_data[stage]['cycles'] += event.duration
        stage_data[stage]['energy'] += event.energy * total_pes  # REFACTORED: Task A2
        stage_data[stage]['events'] += 1

    for stage_name, data in stage_data.items():
        breakdown.stages[stage_name] = StageBreakdown(
            name=stage_name,
            activity_cycles=data['cycles'],
            energy_nj=data['energy'],
            event_count=data['events']
        )

    # Analyze by sub-stage
    substage_data = defaultdict(lambda: {'cycles': 0, 'energy': 0.0, 'events': 0})
    for event in events:
        substage = event.sub_stage or event.stage or 'unknown'
        substage_data[substage]['cycles'] += event.duration
        substage_data[substage]['energy'] += event.energy * total_pes  # REFACTORED: Task A2
        substage_data[substage]['events'] += 1

    for substage_name, data in substage_data.items():
        breakdown.sub_stages[substage_name] = StageBreakdown(
            name=substage_name,
            activity_cycles=data['cycles'],
            energy_nj=data['energy'],
            event_count=data['events']
        )

    # Analyze by circuit module
    module_data = defaultdict(lambda: {'cycles': 0, 'energy': 0.0, 'events': 0, 'power_sum': 0.0})
    for event in events:
        module = event.circuit_module or 'UNKNOWN'
        module_data[module]['cycles'] += event.duration
        module_data[module]['energy'] += event.energy * total_pes  # REFACTORED: Task A2
        module_data[module]['events'] += 1
        module_data[module]['power_sum'] += event.power

    for module_name, data in module_data.items():
        avg_power = data['power_sum'] / data['events'] if data['events'] > 0 else 0
        breakdown.modules[module_name] = ModuleBreakdown(
            name=module_name,
            activity_cycles=data['cycles'],
            energy_nj=data['energy'],
            event_count=data['events'],
            avg_power_mw=avg_power
        )

    return breakdown


def format_breakdown_table(
    breakdown: OperatorBreakdown,
    show_stages: bool = True,
    show_substages: bool = True,
    show_modules: bool = True
) -> str:
    """
    Format breakdown as a readable string.

    Args:
        breakdown: The breakdown to format
        show_stages: Whether to show stage breakdown
        show_substages: Whether to show sub-stage breakdown
        show_modules: Whether to show module breakdown

    Returns:
        Formatted string
    """
    lines = []
    total_cycles = breakdown.total_cycles
    total_activity = breakdown.total_activity_cycles
    total_energy = sum(m.energy_nj for m in breakdown.modules.values())

    lines.append(f"\n{'='*80}")
    lines.append(f" {breakdown.name} Breakdown Analysis")
    lines.append(f"{'='*80}")

    lines.append(f"\nTotal Metrics:")
    lines.append(f"  Wall-Clock Cycles: {total_cycles:,}")
    lines.append(f"  Activity Cycles:   {total_activity:,} (sum of event durations)")
    lines.append(f"  Energy: {breakdown.total_energy_nj:.2f} nJ")
    lines.append(f"  Latency: {breakdown.latency_ms:.3f} ms")
    lines.append(f"  Parallelism: {breakdown.parallelism_factor}x")

    if show_stages and breakdown.stages:
        lines.append(f"\nStage Breakdown (Cycle% = stage_activity / total_activity):")
        lines.append("-" * 80)
        lines.append(f"{'Stage':<25} {'Activity':>12} {'Cycle%':>8} {'Energy(nJ)':>12} {'Energy%':>8}")
        lines.append("-" * 80)

        for name, stage in sorted(breakdown.stages.items(), key=lambda x: x[1].activity_cycles, reverse=True):
            cycle_pct = (stage.activity_cycles / total_activity * 100) if total_activity > 0 else 0
            energy_pct = (stage.energy_nj / total_energy * 100) if total_energy > 0 else 0
            lines.append(f"{name:<25} {stage.activity_cycles:>12,} {cycle_pct:>7.1f}% {stage.energy_nj:>12.2f} {energy_pct:>7.1f}%")

    if show_substages and breakdown.sub_stages:
        lines.append(f"\nSub-Stage Breakdown:")
        lines.append("-" * 80)
        lines.append(f"{'Sub-Stage':<25} {'Activity':>12} {'Cycle%':>8} {'Energy(nJ)':>12} {'Energy%':>8}")
        lines.append("-" * 80)

        for name, stage in sorted(breakdown.sub_stages.items(), key=lambda x: x[1].activity_cycles, reverse=True):
            cycle_pct = (stage.activity_cycles / total_activity * 100) if total_activity > 0 else 0
            energy_pct = (stage.energy_nj / total_energy * 100) if total_energy > 0 else 0
            lines.append(f"{name:<25} {stage.activity_cycles:>12,} {cycle_pct:>7.1f}% {stage.energy_nj:>12.2f} {energy_pct:>7.1f}%")

    if show_modules and breakdown.modules:
        lines.append(f"\nCircuit Module Breakdown:")
        lines.append("-" * 80)
        lines.append(f"{'Module':<15} {'Activity':>12} {'Cycle%':>8} {'Energy(nJ)':>12} {'Energy%':>8} {'AvgPower':>10}")
        lines.append("-" * 80)

        for name, module in sorted(breakdown.modules.items(), key=lambda x: x[1].energy_nj, reverse=True):
            cycle_pct = (module.activity_cycles / total_activity * 100) if total_activity > 0 else 0
            energy_pct = (module.energy_nj / total_energy * 100) if total_energy > 0 else 0
            lines.append(f"{name:<15} {module.activity_cycles:>12,} {cycle_pct:>7.1f}% {module.energy_nj:>12.2f} {energy_pct:>7.1f}% {module.avg_power_mw:>9.2f}mW")

    return "\n".join(lines)


def analyze_simulation_events_breakdown(
    events: List['Event'],
    clock_period_ns: float = 1.0,
) -> Dict[str, Any]:
    """
    从仿真事件列表按层/算子/电路模块三维度聚合周期与能耗。
    要求事件在入队时已设置 metadata['layer_name']（如 main.simulate_network 中 _add_events_to_scheduler）。

    cycles 字段采用「关键路径 makespan」：对每个 (分组维度, layer_name) 区间
    取 [min_cycle, max_end_cycle] 跨度，再按分组维度汇总。
    这比 sum(duration) 准确——后者在并行距离链中会把并行 CIM 读全部累加，高估墙钟延迟。

    Returns:
        by_layer: 层名 -> { cycles(makespan), energy_nj, event_count }
        by_operator: 算子名 -> { cycles(makespan), energy_nj, event_count }
        by_module: 电路模块名 -> { cycles(makespan), energy_nj, event_count }
        total_cycles: 墙钟总周期（max(end_cycle)-min(cycle)+1）
        total_energy_nj: 事件能耗之和
    """
    from collections import defaultdict

    by_layer: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})
    by_operator: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})
    by_module: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})

    # Makespan spans: (group_key, layer_name) → (min_cycle, max_end_cycle)
    _layer_spans: Dict[tuple, tuple] = {}     # (layer_name,) → (min, max)
    _op_spans: Dict[tuple, tuple] = {}        # (op_name, layer_name) → (min, max)
    _mod_spans: Dict[tuple, tuple] = {}       # (mod_name, layer_name) → (min, max)

    total_energy_nj = 0.0
    min_cycle = float("inf")
    max_end_cycle = -1

    for e in events:
        layer_name = str((e.metadata or {}).get("layer_name", "unknown") or "unknown")
        op_name = str(e.operator or "unknown")
        mod_name = str(e.circuit_module or "unknown")

        dur = getattr(e, "duration", 1)
        en = getattr(e, "energy", 0.0)
        overlapped = getattr(e, "overlapped", False)
        ec = int(getattr(e, "cycle", 0))
        ee = int(getattr(e, "end_cycle", ec + dur - 1))

        if not overlapped:
            # Track spans for makespan-based cycle counts
            for spans, key in (
                (_layer_spans, (layer_name,)),
                (_op_spans, (op_name, layer_name)),
                (_mod_spans, (mod_name, layer_name)),
            ):
                if key not in spans:
                    spans[key] = (ec, ee)
                else:
                    prev_min, prev_max = spans[key]
                    spans[key] = (min(prev_min, ec), max(prev_max, ee))

        by_layer[layer_name]["energy_nj"] += en
        by_layer[layer_name]["event_count"] += 1

        by_operator[op_name]["energy_nj"] += en
        by_operator[op_name]["event_count"] += 1

        by_module[mod_name]["energy_nj"] += en
        by_module[mod_name]["event_count"] += 1

        total_energy_nj += en
        min_cycle = min(min_cycle, ec)
        if not overlapped:
            max_end_cycle = max(max_end_cycle, ee)

    # Apply makespan-based cycles
    for (layer_name,), (mn, mx) in _layer_spans.items():
        by_layer[layer_name]["cycles"] += (mx - mn + 1)
    for (op_name, _layer), (mn, mx) in _op_spans.items():
        by_operator[op_name]["cycles"] += (mx - mn + 1)
    for (mod_name, _layer), (mn, mx) in _mod_spans.items():
        by_module[mod_name]["cycles"] += (mx - mn + 1)

    total_cycles = max(0, max_end_cycle - min_cycle + 1) if events else 0

    return {
        "by_layer": dict(by_layer),
        "by_operator": dict(by_operator),
        "by_module": dict(by_module),
        "total_cycles": total_cycles,
        "total_energy_nj": total_energy_nj,
        "latency_ms": total_cycles * clock_period_ns / 1e6,
    }
