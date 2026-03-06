"""
算子模块（Operators）
=====================

点云与神经网络算子在 CIM 上的部署：FPS、KNN、Conv1d、Linear、BN、ReLU 等，
通过 dispatcher 按 op_type 派发到对应 handler，生成周期级 Event 供仿真与 operator_sequence 输出。
"""

from .base_operator import (
    BaseOperator,
    Stage,
    SubStage,
    OperatorParams,
)
from ..configs.hw_config import CircuitModule
from .fps import FPSOperator, FPSParams
from .knn import KNNOperator, KNNParams
from .dispatcher import (
    dispatch_operator,
    dispatch_layer,
    list_supported_op_types,
    register_op_type,
)

__all__ = [
    'BaseOperator',
    'Stage',
    'SubStage',
    'CircuitModule',
    'OperatorParams',
    'FPSOperator',
    'FPSParams',
    'KNNOperator',
    'KNNParams',
    'dispatch_operator',
    'dispatch_layer',
    'list_supported_op_types',
    'register_op_type',
]
