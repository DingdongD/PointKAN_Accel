"""
Event Model
===========

Event definitions for cycle-level simulation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto


class EventStatus(Enum):
    """Event execution status."""
    PENDING = auto()            # Not yet executed
    EXECUTING = auto()          # Currently executing
    COMPLETED = auto()          # Execution completed
    CANCELLED = auto()          # Cancelled


@dataclass
class Event:
    """
    Simulation event.

    Represents a single operation in the cycle-level simulation.

    Attributes:
        event_id: Unique event identifier
        cycle: Cycle number when event occurs
        operator: Operator name (FPS, KNN, etc.)
        stage: Stage name
        sub_stage: Sub-stage name (optional)
        circuit_module: Circuit module involved
        operation: Specific operation description
        power: Power consumption (mW)
        duration: Duration in cycles
        data_size: Data size in bytes (optional)
        clock_period_ns: Clock period in nanoseconds (from hw_config)
        status: Event execution status
        metadata: Additional metadata
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
        """Get the cycle when this event completes."""
        return self.cycle + self.duration - 1

    @property
    def energy(self) -> float:
        """
        Calculate event energy.

        # REFACTORED: Task B - Hardware-semantic energy formula documentation.
        # Uses the clock_period_ns field from hardware configuration.
        # Formula: power(mW) × duration(cycles) × clock_period(ns/cycle) = pJ, then /1000 = nJ
        #
        # Example: ADC conversion at 16mW, 1 cycle, 100ns clock
        #   energy = 16 × 1 × 100 / 1000 = 1.6 nJ
        #
        # Note: This is SINGLE-PE energy. For chip-level energy,
        # multiply by total_pes (done in breakdown.py, Scheme A).

        Returns:
            Energy in nJ.
        """
        return self.power * self.duration * self.clock_period_ns / 1e3
    
    def is_active_at(self, cycle: int) -> bool:
        """
        Check if event is active at given cycle.
        
        Args:
            cycle: Cycle number
            
        Returns:
            True if event is active at this cycle.
        """
        return self.cycle <= cycle <= self.end_cycle
    
    def mark_executing(self) -> None:
        """Mark event as executing."""
        self.status = EventStatus.EXECUTING
    
    def mark_completed(self) -> None:
        """Mark event as completed."""
        self.status = EventStatus.COMPLETED
    
    def mark_cancelled(self) -> None:
        """Mark event as cancelled."""
        self.status = EventStatus.CANCELLED
    
    def __repr__(self) -> str:
        return (f"Event({self.event_id}, cycle={self.cycle}, "
                f"module={self.circuit_module}, op={self.operator})")


class EventQueue:
    """
    Event queue for managing simulation events.
    
    Events are stored and retrieved by cycle number.
    
    Attributes:
        events: List of all events
        current_cycle: Current simulation cycle
    """
    
    def __init__(self):
        """Initialize event queue."""
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
        Add a new event to the queue.

        Args:
            cycle: Cycle when event starts
            operator: Operator name
            stage: Stage name
            sub_stage: Sub-stage name
            circuit_module: Circuit module
            operation: Operation description
            power: Power consumption (mW)
            duration: Duration in cycles
            data_size: Data size in bytes
            clock_period_ns: Clock period in nanoseconds
            metadata: Additional metadata

        Returns:
            Created Event instance.
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
