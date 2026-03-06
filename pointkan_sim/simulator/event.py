"""
事件模型（Event Model）
======================

周期级仿真所用的事件定义。每条事件对应硬件上的一次操作（如一次 buffer 写、一次 MNSIM 矩阵乘），
含起始周期、持续周期、功耗、以及算子/阶段/操作名等，供调度器按周期推进并汇总能耗。
与 main 中 operator_sequence 的键名对应关系见 pointkan_sim.constants。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto


class EventStatus(Enum):
    """事件执行状态。"""
    PENDING = auto()      # 未开始
    EXECUTING = auto()    # 执行中
    COMPLETED = auto()    # 已完成
    CANCELLED = auto()    # 已取消


@dataclass
class Event:
    """
    单条仿真事件：表示周期级仿真中的一次硬件操作。

    Attributes:
        event_id: 唯一事件 ID
        cycle: 起始周期
        operator: 算子名称（如 FPS, KNN, Conv1d, LayerBuffer）
        stage: 阶段（如 buffer_io, quant, distance）
        sub_stage: 子阶段（可选）
        circuit_module: 涉及电路模块（如 DRAM, SRAM-II）
        operation: 具体操作描述（如 buffer_write, phase1_point_cloud_to_sram_ii）
        power: 功耗 (mW)
        duration: 持续周期数
        data_size: 数据量字节（可选）
        clock_period_ns: 时钟周期 (ns)，来自 hw_config
        status: 执行状态
        metadata: 额外元数据（如 layer_name、layer_type，用于 breakdown 与 operator_sequence）
    """
    event_id: int
    cycle: int
    operator: str = ""
    stage: str = ""
    sub_stage: str = ""
    circuit_module: str = ""
    operation: str = ""
    power: float = 0.0
    duration: int = 1
    data_size: int = 0
    clock_period_ns: float = 1.0  # Clock period in ns (default 1ns = 1GHz)
    status: EventStatus = EventStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_cycle(self) -> int:
        """事件结束周期（含）。"""
        return self.cycle + self.duration - 1

    @property
    def energy(self) -> float:
        """
        本事件的能耗 (nJ)。
        公式：power(mW) × duration(cycles) × clock_period_ns / 1000 = nJ。
        此为单 PE 能耗；芯片级需在 breakdown 中按 total_pes 缩放。
        """
        return self.power * self.duration * self.clock_period_ns / 1e3
    
    def is_active_at(self, cycle: int) -> bool:
        """判断给定周期是否处于本事件执行区间内（含起止）。"""
        return self.cycle <= cycle <= self.end_cycle

    def mark_executing(self) -> None:
        """标记为执行中。"""
        self.status = EventStatus.EXECUTING

    def mark_completed(self) -> None:
        """标记为已完成。"""
        self.status = EventStatus.COMPLETED

    def mark_cancelled(self) -> None:
        """标记为已取消。"""
        self.status = EventStatus.CANCELLED
    
    def __repr__(self) -> str:
        return (f"Event({self.event_id}, cycle={self.cycle}, "
                f"module={self.circuit_module}, op={self.operator})")


class EventQueue:
    """
    仿真事件队列：按周期管理事件，支持按周期查询、按算子/阶段筛选。
    Attributes:
        events: 全部事件列表
        current_cycle: 当前仿真周期
    """

    def __init__(self):
        """初始化空事件队列。"""
        self._events: List[Event] = []
        self._next_id: int = 0
        self._current_cycle: int = 0
        self._events_by_cycle: Dict[int, List[Event]] = {}
    
    @property
    def current_cycle(self) -> int:
        """Get current cycle."""
        return self._current_cycle
    
    @property
    def event_count(self) -> int:
        """Get total number of events."""
        return len(self._events)
    
    @property
    def pending_count(self) -> int:
        """Get number of pending events."""
        return sum(1 for e in self._events if e.status == EventStatus.PENDING)
    
    def add_event(
        self,
        cycle: int,
        operator: str = "",
        stage: str = "",
        sub_stage: str = "",
        circuit_module: str = "",
        operation: str = "",
        power: float = 0.0,
        duration: int = 1,
        data_size: int = 0,
        clock_period_ns: float = 100.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        向队列添加一条事件（起始周期、算子名、阶段、操作、功耗、持续周期等）。
        main 中会为事件 metadata 补全 layer_name/layer_type，用于 operator_sequence 与 breakdown。
        Returns:
            新创建的 Event 实例。
        """
        event = Event(
            event_id=self._next_id,
            cycle=cycle,
            operator=operator,
            stage=stage,
            sub_stage=sub_stage,
            circuit_module=circuit_module,
            operation=operation,
            power=power,
            duration=duration,
            data_size=data_size,
            clock_period_ns=clock_period_ns,
            metadata=metadata or {}
        )
        
        self._events.append(event)
        self._next_id += 1
        
        # Index by cycle
        if cycle not in self._events_by_cycle:
            self._events_by_cycle[cycle] = []
        self._events_by_cycle[cycle].append(event)
        
        return event
    
    def get_events_at_cycle(self, cycle: int) -> List[Event]:
        """
        Get all events starting at a specific cycle.
        
        Args:
            cycle: Cycle number
            
        Returns:
            List of events.
        """
        return self._events_by_cycle.get(cycle, [])
    
    def get_active_events_at_cycle(self, cycle: int) -> List[Event]:
        """
        Get all events active at a specific cycle.
        
        Args:
            cycle: Cycle number
            
        Returns:
            List of active events.
        """
        return [e for e in self._events if e.is_active_at(cycle)]
    
    def get_events_by_operator(self, operator: str) -> List[Event]:
        """
        Get all events for a specific operator.
        
        Args:
            operator: Operator name
            
        Returns:
            List of events.
        """
        return [e for e in self._events if e.operator == operator]
    
    def get_events_by_stage(self, stage: str) -> List[Event]:
        """
        Get all events for a specific stage.
        
        Args:
            stage: Stage name
            
        Returns:
            List of events.
        """
        return [e for e in self._events if e.stage == stage]
    
    def get_all_events(self) -> List[Event]:
        """Get all events."""
        return self._events.copy()
    
    def get_last_cycle(self) -> int:
        """Get the last cycle with any event."""
        if not self._events:
            return 0
        return max(e.end_cycle for e in self._events)
    
    def advance_cycle(self, cycles: int = 1) -> None:
        """
        Advance current cycle.
        
        Args:
            cycles: Number of cycles to advance
        """
        self._current_cycle += cycles
    
    def set_cycle(self, cycle: int) -> None:
        """
        Set current cycle.
        
        Args:
            cycle: Cycle number
        """
        self._current_cycle = cycle
    
    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._events_by_cycle.clear()
        self._next_id = 0
        self._current_cycle = 0
    
    def calculate_total_energy(self) -> float:
        """
        Calculate total energy of all events.
        
        Returns:
            Total energy in nJ.
        """
        return sum(e.energy for e in self._events)
    
    def calculate_power_at_cycle(self, cycle: int) -> float:
        """
        Calculate total power at a specific cycle.
        
        Args:
            cycle: Cycle number
            
        Returns:
            Total power in mW.
        """
        active_events = self.get_active_events_at_cycle(cycle)
        return sum(e.power for e in active_events)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event statistics.
        
        Returns:
            Dictionary of statistics.
        """
        stats = {
            'total_events': self.event_count,
            'last_cycle': self.get_last_cycle(),
            'total_energy_nJ': self.calculate_total_energy(),
            'events_by_operator': {},
            'events_by_stage': {},
        }

        # Count by operator
        operators = set(e.operator for e in self._events if e.operator)
        for op in operators:
            stats['events_by_operator'][op] = len(self.get_events_by_operator(op))
        
        # Count by stage
        stages = set(e.stage for e in self._events if e.stage)
        for stage in stages:
            stats['events_by_stage'][stage] = len(self.get_events_by_stage(stage))
        
        return stats
    
    def __len__(self) -> int:
        return len(self._events)
    
    def __repr__(self) -> str:
        return f"EventQueue({self.event_count} events, cycle={self._current_cycle})"
