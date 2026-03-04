"""
Base Operator Model
====================

Abstract base class for point cloud operators.
Defines the interface for stage-based modeling with flat macro-module architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..configs.hw_config import CircuitModule

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class SubStage:
    """
    Sub-stage within an operator stage.

    Attributes:
        name: Sub-stage name
        circuit_modules: List of circuit modules involved
        is_pipeline: Whether this sub-stage is pipelined
        description: Human-readable description
    """
    name: str
    circuit_modules: List[CircuitModule]
    is_pipeline: bool = False
    description: str = ""


@dataclass
class Stage:
    """
    Operator stage containing sub-stages.

    Attributes:
        name: Stage name
        sub_stages: List of computation sub-stages
        is_pipeline: Whether stages can be pipelined with other stages
        description: Human-readable description
    """
    name: str
    sub_stages: List[SubStage] = field(default_factory=list)
    is_pipeline: bool = False
    description: str = ""

    def add_sub_stage(self, sub_stage: SubStage) -> None:
        """Add a computation sub-stage."""
        self.sub_stages.append(sub_stage)


@dataclass
class OperatorParams:
    """
    Base parameters for operator execution.

    Attributes:
        input_points: Number of input points
    """
    input_points: int = 1024


class BaseOperator(ABC):
    """
    Abstract base class for point cloud operators.

    Defines the interface for dataflow modeling with stages and sub-stages.
    Each operator (FPS, KNN, etc.) must implement this interface.

    Attributes:
        name: Operator name
        stages: List of execution stages
        params: Operator parameters
        hw_config: Hardware configuration
    """

    def __init__(
        self,
        name: str,
        params: Optional[OperatorParams] = None,
        hw_config: Optional['HardwareConfig'] = None
    ):
        self.name = name
        self.params = params if params is not None else OperatorParams()
        self.hw_config = hw_config
        self.stages: List[Stage] = []

        # Initialize stages
        self._define_stages()

    @abstractmethod
    def _define_stages(self) -> None:
        """Define the stages and sub-stages of this operator."""
        pass

    @abstractmethod
    def calculate_cycles(self, params: Optional[OperatorParams] = None) -> int:
        """Calculate total execution cycles."""
        pass

    @abstractmethod
    def calculate_energy(self, params: Optional[OperatorParams] = None) -> float:
        """Calculate total energy consumption (nJ)."""
        pass

    @abstractmethod
    def generate_events(self, params: Optional[OperatorParams] = None) -> List[Any]:
        """Generate cycle-level events for simulation."""
        pass

    def get_stages(self) -> List[Stage]:
        """Get list of stages."""
        return self.stages

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
