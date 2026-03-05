"""
Breakdown Analysis Module
=========================

Provides detailed breakdown analysis for:
1. Energy consumption by circuit module
2. Cycle breakdown by stage/sub-stage
3. Operator-level analysis (FPS vs KNN)

This module is designed to be integrated with main.py for post-simulation analysis.

Key Concepts:
- Activity Cycles: Sum of event durations (can exceed total_cycles due to pipelining)
- Total Cycles: Actual wall-clock cycles from calculate_cycles()
- Cycle%: Relative contribution to total activity (sums to 100%)
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
    """Breakdown data for a single circuit module."""
    name: str
    activity_cycles: int = 0     # Sum of event durations for this module
    energy_nj: float = 0.0
    event_count: int = 0
    avg_power_mw: float = 0.0


@dataclass
class StageBreakdown:
    """Breakdown data for a stage or sub-stage."""
    name: str
    activity_cycles: int = 0     # Sum of event durations for this stage
    energy_nj: float = 0.0
    event_count: int = 0


@dataclass
class OperatorBreakdown:
    """Complete breakdown for an operator."""
    name: str
    total_cycles: int = 0           # Wall-clock cycles from calculate_cycles()
    total_activity_cycles: int = 0  # Sum of all event durations
    total_energy_nj: float = 0.0
    latency_ms: float = 0.0

    # Breakdowns
    stages: Dict[str, StageBreakdown] = field(default_factory=dict)
    sub_stages: Dict[str, StageBreakdown] = field(default_factory=dict)
    modules: Dict[str, ModuleBreakdown] = field(default_factory=dict)

    # Hardware info
    parallelism_factor: int = 1
    clock_period_ns: float = 100.0


def analyze_operator_breakdown(
    operator: 'BaseOperator',
    hw_config: 'HardwareConfig',
    events: Optional[List['Event']] = None
) -> OperatorBreakdown:
    """
    Analyze an operator and return detailed breakdown.

    # REFACTORED: Task A2 - Scheme A (single-PE share scaling):
    # generate_events() produces events for ONE PE's workload (batches_per_pe).
    # This function scales energy by total_pes to get chip-level energy.
    # Cycle counts are NOT scaled (they represent wall-clock time, already correct).

    Args:
        operator: The operator to analyze
        hw_config: Hardware configuration
        events: Pre-generated events (optional, will generate if None)

    Returns:
        OperatorBreakdown with all metrics
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


def analyze_network_breakdown(
    operators: List['BaseOperator'],
    hw_config: 'HardwareConfig',
    operator_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a full network of operators.

    # REFACTORED: Task A2 - Scheme A: events represent single-PE workload.
    # Energy is scaled by total_pes for chip-level values.

    Args:
        operators: List of operators
        hw_config: Hardware configuration
        operator_names: Optional names for operators

    Returns:
        Dictionary with network-level breakdown
    """
    if operator_names is None:
        operator_names = [f"Op_{i}" for i in range(len(operators))]

    cim_parallelism = _get_cim_parallelism(hw_config)
    total_pes = cim_parallelism  # flat arch: no tile/PE hierarchy
    clock_period = hw_config.clock_period

    total_cycles = 0
    total_energy = 0.0
    layer_data = []
    module_totals = defaultdict(float)

    for name, op in zip(operator_names, operators):
        cycles = op.calculate_cycles()
        # REFACTORED: Task A2 - Scheme A: scale single-PE energy to chip-level
        events = op.generate_events()
        single_pe_energy = sum(e.energy for e in events)
        energy = single_pe_energy * total_pes
        total_cycles += cycles
        total_energy += energy

        layer_data.append({
            'name': name,
            'cycles': cycles,
            'energy': energy,
            'breakdown': analyze_operator_breakdown(op, hw_config, events=events)
        })

        # Aggregate module energy (already using same events, avoid regeneration)
        for event in events:
            module = event.circuit_module or 'UNKNOWN'
            module_totals[module] += event.energy * total_pes  # REFACTORED: Task A2

    return {
        'total_cycles': total_cycles,
        'total_energy': total_energy,
        'latency_ms': total_cycles * clock_period / 1e6,
        'layers': layer_data,
        'module_totals': dict(module_totals),
        'parallelism': total_pes,
        'clock_period': clock_period
    }


def format_network_breakdown(network_data: Dict[str, Any]) -> str:
    """Format network breakdown as readable string."""
    lines = []
    total_cycles = network_data['total_cycles']
    total_energy = network_data['total_energy']
    module_totals = network_data['module_totals']
    total_module_energy = sum(module_totals.values())

    lines.append(f"\n{'='*80}")
    lines.append(f" Network Breakdown Analysis")
    lines.append(f"{'='*80}")

    lines.append(f"\nNetwork Summary:")
    lines.append(f"  Total Cycles: {total_cycles:,}")
    lines.append(f"  Total Energy: {total_energy:.2f} nJ")
    lines.append(f"  Total Latency: {network_data['latency_ms']:.3f} ms")
    lines.append(f"  Parallelism: {network_data['parallelism']}x")

    lines.append(f"\nLayer-by-Layer Breakdown:")
    lines.append("-" * 70)
    lines.append(f"{'Layer':<15} {'Cycles':>14} {'Cycle%':>8} {'Energy(nJ)':>12} {'Energy%':>8}")
    lines.append("-" * 70)

    for layer in network_data['layers']:
        c_pct = (layer['cycles'] / total_cycles * 100) if total_cycles > 0 else 0
        e_pct = (layer['energy'] / total_energy * 100) if total_energy > 0 else 0
        lines.append(f"{layer['name']:<15} {layer['cycles']:>14,} {c_pct:>7.1f}% {layer['energy']:>12.2f} {e_pct:>7.1f}%")

    lines.append(f"\nCircuit Module Energy Distribution:")
    lines.append("-" * 60)
    lines.append(f"{'Module':<20} {'Energy(nJ)':>15} {'Percentage':>12}")
    lines.append("-" * 60)

    for module, energy in sorted(module_totals.items(), key=lambda x: x[1], reverse=True):
        pct = (energy / total_module_energy * 100) if total_module_energy > 0 else 0
        lines.append(f"{module:<20} {energy:>15.2f} {pct:>11.1f}%")

    return "\n".join(lines)


def print_breakdown(breakdown: OperatorBreakdown, **kwargs):
    """Print breakdown to console."""
    print(format_breakdown_table(breakdown, **kwargs))


def print_network_breakdown(network_data: Dict[str, Any]):
    """Print network breakdown to console."""
    print(format_network_breakdown(network_data))


def analyze_simulation_events_breakdown(
    events: List['Event'],
    clock_period_ns: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute layer / operator / module breakdown from a flat list of Events.

    Events should have metadata['layer_name'] and metadata.get('layer_type') set
    when they were queued (e.g. in main.simulate_network).

    Returns:
        Dict with:
          - by_layer: { layer_name: { cycles, energy_nj, event_count } }
          - by_operator: { operator: { cycles, energy_nj, event_count } }
          - by_module: { circuit_module: { cycles, energy_nj, event_count } }
          - total_cycles: max(cycle + duration) - min(cycle) + 1
          - total_energy_nj: sum of event.energy
    """
    from collections import defaultdict

    by_layer: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})
    by_operator: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})
    by_module: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "energy_nj": 0.0, "event_count": 0})

    total_energy_nj = 0.0
    min_cycle = float("inf")
    max_end_cycle = -1

    for e in events:
        layer_name = (e.metadata or {}).get("layer_name", "unknown")
        op_name = e.operator or "unknown"
        mod_name = e.circuit_module or "unknown"

        dur = getattr(e, "duration", 1)
        en = getattr(e, "energy", 0.0)

        by_layer[layer_name]["cycles"] += dur
        by_layer[layer_name]["energy_nj"] += en
        by_layer[layer_name]["event_count"] += 1

        by_operator[op_name]["cycles"] += dur
        by_operator[op_name]["energy_nj"] += en
        by_operator[op_name]["event_count"] += 1

        by_module[mod_name]["cycles"] += dur
        by_module[mod_name]["energy_nj"] += en
        by_module[mod_name]["event_count"] += 1

        total_energy_nj += en
        min_cycle = min(min_cycle, e.cycle)
        max_end_cycle = max(max_end_cycle, e.cycle + dur - 1)

    total_cycles = max(0, max_end_cycle - min_cycle + 1) if events else 0

    return {
        "by_layer": dict(by_layer),
        "by_operator": dict(by_operator),
        "by_module": dict(by_module),
        "total_cycles": total_cycles,
        "total_energy_nj": total_energy_nj,
        "latency_ms": total_cycles * clock_period_ns / 1e6,
    }
