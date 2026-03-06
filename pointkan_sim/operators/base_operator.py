"""
算子基类（Base Operator Model）
===============================

点云算子的抽象基类，定义「按阶段建模 + 扁平宏模块架构」的接口。
各具体算子（FPS、KNN、Conv1d、Linear 等）需实现 calculate_cycles、calculate_energy、generate_events。
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
    算子某一阶段内的子阶段。

    Attributes:
        name: 子阶段名称
        circuit_modules: 参与的电路模块列表
        is_pipeline: 是否可与其他子阶段流水
        description: 可读描述
    """
    name: str
    circuit_modules: List[CircuitModule]
    is_pipeline: bool = False
    description: str = ""


@dataclass
class Stage:
    """
    算子的一个执行阶段，可包含多个子阶段。

    Attributes:
        name: 阶段名称
        sub_stages: 子阶段列表
        is_pipeline: 是否可与其它阶段流水
        description: 可读描述
    """
    name: str
    sub_stages: List[SubStage] = field(default_factory=list)
    is_pipeline: bool = False
    description: str = ""

    def add_sub_stage(self, sub_stage: SubStage) -> None:
        """添加一个计算子阶段。"""
        self.sub_stages.append(sub_stage)


@dataclass
class OperatorParams:
    """
    算子执行所需的基础参数。

    Attributes:
        input_points: 输入点数量
    """
    input_points: int = 1024


class BaseOperator(ABC):
    """
    点云算子的抽象基类，按阶段/子阶段做数据流建模。

    各具体算子（FPS、KNN 等）需实现：_define_stages、calculate_cycles、calculate_energy、generate_events。

    Attributes:
        name: 算子名称
        stages: 执行阶段列表
        params: 算子参数
        hw_config: 硬件配置
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
        """定义本算子的阶段与子阶段。"""
        pass

    @abstractmethod
    def calculate_cycles(self, params: Optional[OperatorParams] = None) -> int:
        """计算总执行周期数。"""
        pass

    @abstractmethod
    def calculate_energy(self, params: Optional[OperatorParams] = None) -> float:
        """计算总能耗 (nJ)。"""
        pass

    @abstractmethod
    def generate_events(self, params: Optional[OperatorParams] = None) -> List[Any]:
        """生成周期级仿真事件列表。"""
        pass

    def get_stages(self) -> List[Stage]:
        """返回阶段列表。"""
        return self.stages

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
