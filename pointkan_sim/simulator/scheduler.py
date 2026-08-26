"""
Scheduler Model
===============

Cycle-level scheduler for CIM simulation.
Optimized with event-driven simulation for large-scale workloads.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum, auto
import heapq

from .event import Event, EventQueue, EventStatus

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig
    from ..operators.base_operator import BaseOperator


class SchedulerStatus(Enum):
    """Scheduler status enumeration."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class SimulationResult:
    """
    Simulation result container.

    Attributes:
        total_cycles: Total simulation cycles
        total_energy: Total energy in nJ
        total_power: Average power in mW
        peak_power: Peak power in mW
        events: List of all events
        statistics: Detailed statistics
        status: Final simulation status
    """
    total_cycles: int = 0
    total_energy: float = 0.0
    total_power: float = 0.0
    peak_power: float = 0.0
    events: List[Event] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    block_intervals: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    status: SchedulerStatus = SchedulerStatus.COMPLETED
    error_message: str = ""



class Scheduler:
    """
    Cycle-level simulation scheduler.

    Uses event-driven simulation for efficiency with large workloads.
    Instead of iterating over every cycle, only processes cycles where
    events start or end, achieving O(events * log(events)) complexity.

    Attributes:
        hw_config: Hardware configuration
        sim_config: Simulation configuration
        event_queue: Event queue
        status: Current scheduler status
    """

    def __init__(
        self,
        hw_config: Optional['HardwareConfig'] = None,
        sim_config: Optional['SimulationConfig'] = None
    ):
        """
        Initialize scheduler.

        Args:
            hw_config: Hardware configuration
            sim_config: Simulation configuration
        """
        self.hw_config = hw_config
        self.sim_config = sim_config
        self.event_queue = EventQueue()
        self.status = SchedulerStatus.IDLE

        # Simulation state
        self._current_cycle: int = 0
        self._operators: List['BaseOperator'] = []
        self._cycle_power_trace: List[float] = []

        # Power trace sampling (for large simulations)
        self._power_sample_interval: int = 100  # Sample every N cycles
        self._sampled_power_trace: List[Tuple[int, float]] = []

    @property
    def current_cycle(self) -> int:
        """Get current simulation cycle."""
        return self._current_cycle

    def reset(self) -> None:
        """Reset scheduler state."""
        self.event_queue.clear()
        self._current_cycle = 0
        self._operators.clear()
        self._cycle_power_trace.clear()
        self._sampled_power_trace.clear()
        self.status = SchedulerStatus.IDLE

    def add_operator(self, operator: 'BaseOperator') -> None:
        """
        Add an operator to the simulation.

        Args:
            operator: Operator instance
        """
        self._operators.append(operator)

    def schedule_operator(
        self,
        operator: 'BaseOperator',
        start_cycle: int = 0
    ) -> int:
        """
        Schedule an operator for execution.

        Generates events for the operator and adds them to the queue.

        Args:
            operator: Operator to schedule
            start_cycle: Starting cycle

        Returns:
            Number of cycles for this operator.
        """
        # Generate events from operator
        events = operator.generate_events()

        # Offset events by start_cycle
        for event in events:
            self.event_queue.add_event(
                cycle=event.cycle + start_cycle,
                operator=operator.name,
                stage=event.stage,
                sub_stage=event.sub_stage,
                circuit_module=event.circuit_module,
                operation=event.operation,
                power=event.power,
                duration=event.duration,
                data_size=event.data_size,
                clock_period_ns=event.clock_period_ns,
                metadata=event.metadata
            )

        # Return total cycles
        return operator.calculate_cycles()

    def schedule_operators_sequential(
        self,
        operators: List['BaseOperator']
    ) -> int:
        """
        Schedule multiple operators sequentially.

        Args:
            operators: List of operators

        Returns:
            Total cycles.
        """
        current_cycle = 0

        for operator in operators:
            cycles = self.schedule_operator(operator, current_cycle)
            current_cycle += cycles

        return current_cycle

    def run_simulation(
        self,
        max_cycles: Optional[int] = None,
        progress_interval: int = 100000
    ) -> SimulationResult:
        """
        Run the simulation using event-driven approach.

        This optimized implementation only processes cycles where events
        start or end, instead of iterating over every single cycle.

        Complexity: O(events * log(events)) instead of O(cycles * events)

        Args:
            max_cycles: Maximum cycles to simulate (None = no limit)
            progress_interval: Print progress every N events processed

        Returns:
            SimulationResult with all metrics.
        """
        self.status = SchedulerStatus.RUNNING
        result = SimulationResult()

        try:
            events = self.event_queue.get_all_events()
            if not events:
                result.total_cycles = 0
                result.status = SchedulerStatus.COMPLETED
                return result

            # 增量改造：在进入事件驱动仿真前，先做一次"共享资源互斥"时序修正。
            # 约定：event.metadata 中可携带
            #   - resource_id: 资源名（如 bitonic_shared）
            #   - exclusive: 是否互斥占用（True 时同资源不可重叠）
            self._apply_exclusive_resource_constraints(events)

            # Determine end cycle
            last_event_cycle = self.event_queue.get_last_cycle()
            if max_cycles is not None:
                end_cycle = min(last_event_cycle, max_cycles)
            else:
                end_cycle = last_event_cycle

            # Use event-driven simulation
            result = self._run_event_driven_simulation(events, end_cycle, progress_interval)

        except Exception as e:
            result.status = SchedulerStatus.ERROR
            result.error_message = str(e)
            import traceback
            traceback.print_exc()

        self.status = result.status
        return result

    def _apply_exclusive_resource_constraints(self, events: List[Event]) -> None:
        """
        对带 resource_id 的事件做资源仲裁，支持两种模式：

        1) **exclusive**（互斥）：resource_id + exclusive=True。
           同资源上一事件「完成」后下一事件才能开始。
           next_free = cycle + duration。
           适用于片上共享单元（bitonic 比较器、SRAM port 等）。

        2) **dram_pipelined**（DRAM 流水线化）：resource_id + dram_pipelined=True。
           仅命令总线互斥：上一事件的 issue window 结束后，下一事件即可开始发射请求，
           无需等待前一事件的响应返回。
           next_free = cycle + dram_issue_cycles（metadata 中携带）。
           与 DRAMSim3 trace 模型一致：每 burst 事务占用命令总线 issue_gap 周期，
           多个传输可有 in-flight 请求重叠。
           事件 duration 仍为 issue + response 全程（数据可用时间），
           下游依赖应看 cycle + duration。

        调度规则：
        1) 按原始 cycle/event_id 排序；
        2) 对同一 resource_id，若新事件在资源 busy 期间到达，顺延到 next_free；
        3) 仅修改事件 cycle，不改变 duration/power。
        """
        if not events:
            return

        def _apply_resource_only(ev_list: List[Event]) -> None:
            next_free_cycle: Dict[str, int] = {}
            def _prio(ev: Event) -> int:
                try:
                    return int((ev.metadata or {}).get("resource_priority", 100))
                except Exception:
                    return 100
            for ev in sorted(ev_list, key=lambda x: (x.cycle, _prio(x), x.event_id)):
                meta = ev.metadata or {}
                rid = meta.get("resource_id")
                exclusive = bool(meta.get("exclusive", False))
                pipelined = bool(meta.get("dram_pipelined", False))
                if not rid or (not exclusive and not pipelined):
                    continue
                avail = int(next_free_cycle.get(str(rid), 0))
                if ev.cycle < avail:
                    ev.cycle = avail
                if pipelined:
                    issue_cycles = int(meta.get("dram_issue_cycles", ev.duration))
                    next_free_cycle[str(rid)] = ev.cycle + issue_cycles
                else:
                    next_free_cycle[str(rid)] = ev.cycle + ev.duration

        # 资源仲裁与依赖约束需要看彼此修正后的真实结束时刻。
        # 这里做固定点迭代，避免像 ACC 里 coarse/fine 依赖仍然拿到上游旧 cycle。
        event_by_id: Dict[int, Event] = {int(getattr(ev, "event_id", -1)): ev for ev in events}

        def _apply_dependencies_once(evts: List[Event]) -> bool:
            changed = False
            tag_end_cycle: Dict[str, int] = {}
            for ev in evts:
                meta = ev.metadata or {}
                tag = meta.get("dependency_tag")
                if tag not in (None, ""):
                    tag_end_cycle[str(tag)] = max(tag_end_cycle.get(str(tag), 0), int(ev.end_cycle) + 1)

            for e in sorted(evts, key=lambda x: getattr(x, "event_id", 0)):
                meta = e.metadata or {}
                dep_end = 0

                dep_one = meta.get("depends_on_event_id")
                dep_many = meta.get("depends_on_event_ids")
                dep_ids = []
                if dep_one not in (None, ""):
                    try:
                        dep_ids.append(int(dep_one))
                    except Exception:
                        pass
                if isinstance(dep_many, (list, tuple)):
                    for item in dep_many:
                        try:
                            dep_ids.append(int(item))
                        except Exception:
                            pass
                elif isinstance(dep_many, str) and dep_many.strip():
                    for item in dep_many.split(','):
                        try:
                            dep_ids.append(int(item.strip()))
                        except Exception:
                            pass
                for dep_id in dep_ids:
                    dep_ev = event_by_id.get(dep_id)
                    if dep_ev is not None:
                        dep_end = max(dep_end, int(dep_ev.end_cycle) + 1)

                dep_tag = meta.get("depends_on_tag")
                dep_tags = meta.get("depends_on_tags")
                tag_list = []
                if dep_tag not in (None, ""):
                    tag_list.append(str(dep_tag))
                if isinstance(dep_tags, (list, tuple)):
                    tag_list.extend(str(t) for t in dep_tags if t not in (None, ""))
                elif isinstance(dep_tags, str) and dep_tags.strip():
                    tag_list.extend(t.strip() for t in dep_tags.split(',') if t.strip())
                for tag in tag_list:
                    dep_end = max(dep_end, int(tag_end_cycle.get(tag, 0)))

                if dep_end and e.cycle < dep_end:
                    e.cycle = dep_end
                    changed = True
            return changed

        for _ in range(6):
            prev = [(int(ev.cycle), int(ev.end_cycle)) for ev in events]
            _apply_resource_only(events)
            dep_changed = _apply_dependencies_once(events)
            _apply_resource_only(events)
            curr = [(int(ev.cycle), int(ev.end_cycle)) for ev in events]
            if not dep_changed and curr == prev:
                break

    def _run_event_driven_simulation(
        self,
        events: List[Event],
        end_cycle: int,
        progress_interval: int
    ) -> SimulationResult:
        """
        Event-driven simulation core.

        Instead of iterating over every cycle, we:
        1. Collect all time points where events start or end
        2. Process only those time points
        3. Calculate power and energy for intervals between time points

        Args:
            events: List of all events
            end_cycle: Last cycle to simulate
            progress_interval: Progress reporting interval

        Returns:
            SimulationResult
        """
        result = SimulationResult()

        # Build timeline of event changes
        # Each entry: (cycle, event_id, is_start)
        # is_start: True = event starts, False = event ends
        timeline: List[Tuple[int, int, bool]] = []

        for i, event in enumerate(events):
            if event.cycle <= end_cycle:
                timeline.append((event.cycle, i, True))  # Start
                event_end = min(event.end_cycle, end_cycle)
                timeline.append((event_end + 1, i, False))  # End (exclusive)

        # Sort by cycle, then by is_start (ends before starts at same cycle)
        timeline.sort(key=lambda x: (x[0], x[2]))

        # Active events tracking
        active_event_ids: Set[int] = set()
        current_power: float = 0.0

        # Power statistics
        total_power_cycles: float = 0.0  # Sum of (power * cycles)
        peak_power: float = 0.0
        min_power: float = float('inf')

        # Sampled power trace for visualization
        self._sampled_power_trace = []
        last_sample_cycle = -self._power_sample_interval

        prev_cycle = 0
        processed_events = 0
        total_timeline_entries = len(timeline)

        print(f"    Processing {total_timeline_entries:,} timeline entries...")

        for cycle, event_id, is_start in timeline:
            if cycle > end_cycle + 1:
                break

            # Calculate power contribution for the interval [prev_cycle, cycle)
            if cycle > prev_cycle and active_event_ids:
                interval_length = cycle - prev_cycle
                total_power_cycles += current_power * interval_length

                # Track peak/min
                if current_power > peak_power:
                    peak_power = current_power
                if current_power < min_power and current_power > 0:
                    min_power = current_power

                # Sample power trace
                while last_sample_cycle + self._power_sample_interval < cycle:
                    sample_cycle = last_sample_cycle + self._power_sample_interval
                    if sample_cycle >= prev_cycle:
                        self._sampled_power_trace.append((sample_cycle, current_power))
                    last_sample_cycle = sample_cycle

            # Update active events
            if is_start:
                active_event_ids.add(event_id)
                current_power += events[event_id].power
                events[event_id].mark_executing()
            else:
                if event_id in active_event_ids:
                    active_event_ids.discard(event_id)
                    current_power -= events[event_id].power
                    current_power = max(0, current_power)  # Avoid floating point errors
                    events[event_id].mark_completed()

            prev_cycle = cycle
            processed_events += 1

            # Progress reporting
            if processed_events % progress_interval == 0:
                progress_pct = (processed_events / total_timeline_entries) * 100
                print(f"    Progress: {processed_events:,}/{total_timeline_entries:,} ({progress_pct:.1f}%)")

        # Calculate results
        result.total_cycles = end_cycle + 1
        result.events = events
        result.total_energy = self.event_queue.calculate_total_energy()

        if end_cycle > 0:
            result.total_power = total_power_cycles / (end_cycle + 1)
        result.peak_power = peak_power

        # Build full power trace from samples (for compatibility)
        self._cycle_power_trace = [p for _, p in self._sampled_power_trace]

        # Calculate statistics
        result.statistics = self._calculate_statistics_fast(events, peak_power, min_power)
        result.block_intervals = self._build_block_intervals(events)
        result.status = SchedulerStatus.COMPLETED

        print(f"    Simulation complete: {end_cycle + 1:,} cycles, {len(events):,} events")

        return result

    def _calculate_statistics_fast(
        self,
        events: List[Event],
        peak_power: float,
        min_power: float
    ) -> Dict[str, Any]:
        """
        Calculate statistics efficiently from events.

        Args:
            events: List of events
            peak_power: Peak power observed
            min_power: Minimum power observed

        Returns:
            Dictionary of statistics.
        """
        stats: Dict[str, Any] = {}

        # Count events by operator and stage
        events_by_operator: Dict[str, int] = {}
        events_by_stage: Dict[str, int] = {}

        for event in events:
            # By operator
            if event.operator:
                events_by_operator[event.operator] = events_by_operator.get(event.operator, 0) + 1

            # By stage
            if event.stage:
                events_by_stage[event.stage] = events_by_stage.get(event.stage, 0) + 1

        stats['total_events'] = len(events)
        stats['events_by_operator'] = events_by_operator
        stats['events_by_stage'] = events_by_stage

        # Power statistics
        if self._sampled_power_trace:
            powers = [p for _, p in self._sampled_power_trace]
            stats['power'] = {
                'average_mW': sum(powers) / len(powers) if powers else 0,
                'peak_mW': peak_power,
                'min_mW': min_power if min_power != float('inf') else 0,
            }

        # Operator statistics
        stats['operators'] = {}
        for operator in self._operators:
            stats['operators'][operator.name] = {
                'cycles': operator.calculate_cycles(),
                'stages': [s.name for s in operator.get_stages()],
            }

        return stats

    def _event_block_name(self, event: Event) -> str:
        meta = event.metadata or {}
        rid = str(meta.get("resource_id", "") or "").strip()
        if rid:
            return rid
        module = str(getattr(event, "circuit_module", "") or "").strip()
        if module:
            return module
        op = str(getattr(event, "operator", "") or "").strip()
        if op:
            return op
        return "unknown"

    def _build_block_intervals(self, events: List[Event]) -> Dict[str, List[Tuple[int, int]]]:
        raw: Dict[str, List[Tuple[int, int]]] = {}
        for event in events:
            start = int(getattr(event, "cycle", 0) or 0)
            end = int(getattr(event, "end_cycle", start) or start)
            if end < start:
                continue
            name = self._event_block_name(event)
            raw.setdefault(name, []).append((start, end))

        merged: Dict[str, List[Tuple[int, int]]] = {}
        for name, spans in raw.items():
            if not spans:
                continue
            spans = sorted(spans, key=lambda item: (item[0], item[1]))
            compact: List[Tuple[int, int]] = [spans[0]]
            for start, end in spans[1:]:
                last_start, last_end = compact[-1]
                if start <= last_end:
                    compact[-1] = (last_start, max(last_end, end))
                else:
                    compact.append((start, end))
            merged[name] = compact
        return merged

    def run_simulation_fast(
        self,
        max_cycles: Optional[int] = None
    ) -> SimulationResult:
        """
        Run fast simulation without cycle-by-cycle tracking.

        This is the fastest mode - it just calculates totals from events
        without tracking per-cycle power. Use this for very large simulations
        where you only need total cycles/energy.

        Args:
            max_cycles: Maximum cycles to simulate

        Returns:
            SimulationResult (without detailed power trace)
        """
        self.status = SchedulerStatus.RUNNING
        result = SimulationResult()

        try:
            events = self.event_queue.get_all_events()
            if not events:
                result.total_cycles = 0
                result.status = SchedulerStatus.COMPLETED
                return result

            # 与 run_simulation 保持一致：先做互斥资源时序修正
            self._apply_exclusive_resource_constraints(events)

            # Calculate totals directly from events（overlapped 不延伸关键路径）
            last_cycle = max(e.end_cycle for e in events if not getattr(e, "overlapped", False))
            if max_cycles is not None:
                last_cycle = min(last_cycle, max_cycles)

            result.total_cycles = last_cycle + 1
            result.total_energy = sum(e.energy for e in events if e.cycle <= last_cycle)

            # Estimate average power
            total_power_duration = sum(e.power * e.duration for e in events if e.cycle <= last_cycle)
            if result.total_cycles > 0:
                result.total_power = total_power_duration / result.total_cycles

            # Peak power estimation (from single events, may underestimate)
            result.peak_power = max(e.power for e in events) if events else 0

            result.events = events
            result.statistics = self._calculate_statistics_fast(events, result.peak_power, 0)
            result.block_intervals = self._build_block_intervals(events)
            result.status = SchedulerStatus.COMPLETED

            # Mark all events as completed
            for event in events:
                event.mark_completed()

        except Exception as e:
            result.status = SchedulerStatus.ERROR
            result.error_message = str(e)

        self.status = result.status
        return result

    def get_power_trace(self) -> List[float]:
        """
        Get power trace over simulation.

        Returns:
            List of power values (mW) - sampled, not every cycle.
        """
        return self._cycle_power_trace.copy()

    def get_sampled_power_trace(self) -> List[Tuple[int, float]]:
        """
        Get sampled power trace with cycle numbers.

        Returns:
            List of (cycle, power_mW) tuples.
        """
        return self._sampled_power_trace.copy()

    def get_event_statistics(self) -> Dict[str, Any]:
        """
        Get event queue statistics.

        Returns:
            Dictionary of statistics.
        """
        return self.event_queue.get_statistics()

    def __repr__(self) -> str:
        return f"Scheduler(status={self.status.name}, cycle={self._current_cycle})"
