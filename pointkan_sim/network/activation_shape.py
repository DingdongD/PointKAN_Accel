"""
激活 Shape 推导与层间数据量
============================

根据网络层定义与点云输入 N×3，逐层推导每层的输入/输出 shape、
元素个数与字节数，用于层间 buffer 读写与报告展示。

点云约定：每个样本输入为 (N, 3)，N 为点数，3 为 xyz 坐标。

与 PyTorch forward 的对应关系（忽略 batch 维）：
  - 推理脚本中 embedding out (B, 32, 1024) → 本处 output_shape = (1024, 32)，即 1024×32
  - GAM center (B, 64, 512) → (512, 64)；LFP/GFP out (B, C, G) → (G, C)
  - global_maxpool 后 (B, 512) → (512,)；classifier out (B, 40) → (40,)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pointkan_net import LayerDef, OperatorRef


@dataclass
class ActivationShapeInfo:
    """
    单层激活的输入/输出 shape 与数据量。

    Attributes:
        layer_name: 层名称
        layer_type: 层类型（embedding, GAM, LFP, stage, classifier 等）
        input_shape: 输入 shape，如 (N, 3) 或 (N, C) 或 (C,)；无上一层层时为 None
        output_shape: 本层输出 shape
        input_numel: 输入元素个数（用于 buffer 读）
        output_numel: 输出元素个数（用于 buffer 写）
        input_bytes: 输入字节数（input_numel * bytes_per_element）
        output_bytes: 输出字节数（output_numel * bytes_per_element）
    """
    layer_name: str
    layer_type: str
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_numel: int = 0
    output_numel: int = 0
    input_bytes: int = 0
    output_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "input_numel": self.input_numel,
            "output_numel": self.output_numel,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
        }


def _infer_layer_output_shape(layer: LayerDef) -> Optional[Tuple[int, ...]]:
    """
    从层内算子参数推断本层输出 shape，与 pointkanElite.py / pointkan.py 的 forward 一致。

    规则：
    - conv1d/conv2d：仅当 params 中有 num_points 时才更新 points，否则保留之前值。
    - linear：输出 (out_features,)。
    - s-pool（GAM）：(num_points, in_channels)。
    - LFP：输出为 (G, C)，即 fusion add 的 num_points × num_features；adaptive_max_pool1d 只做局部池化，不改变整层输出的 points 维。
    - adaptive_max_pool1d：只更新 channels（若有），不把 output_size 当作整层输出的 points。
    """
    if not layer.operators:
        return None

    points: Optional[int] = None
    channels: Optional[int] = None
    vector_dim: Optional[int] = None

    for op in layer.operators:
        p = op.params
        if op.op_type in ("conv1d", "conv2d"):
            if "num_points" in p:
                points = int(p["num_points"])
            channels = int(p.get("out_channels", 1))
        elif op.op_type == "linear":
            vector_dim = int(p.get("out_features", 1))
        elif op.op_type == "s-pool":
            points = int(p.get("num_points", 1))
            channels = int(p.get("in_channels", 1))
        elif op.op_type in ("adaptive_max_pool1d", "sa_maxpool"):
            # 仅局部池化，整层输出仍为 (num_points, channels)，不把 output_size 当作层输出
            if channels is None:
                channels = int(p.get("num_features", 1))
        elif op.op_type in ("add", "relu", "bn", "batchnorm1d", "layernorm", "kan", "mul", "affine_norm", "ballquery"):
            if points is None and "num_points" in p:
                points = int(p["num_points"])
            elif "num_points" in p:
                points = int(p["num_points"])
            if channels is None:
                channels = int(p.get("num_features", p.get("normalized_shape", p.get("in_features", 1))))
            elif "num_features" in p:
                channels = int(p["num_features"])
        elif op.op_type == "fps":
            points = int(p.get("output_points", 1))

    if vector_dim is not None:
        return (vector_dim,)
    if points is not None and channels is not None:
        return (points, channels)
    if channels is not None:
        return (channels,)
    return None


def compute_activation_shapes(
    layers: List[LayerDef],
    input_points: int,
    input_channels: int = 3,
    bytes_per_element: int = 4,
) -> List[ActivationShapeInfo]:
    """
    根据层序列与点云输入 (N, 3)，推导每层的 input_shape / output_shape 与字节数。

    Args:
        layers: 按前向顺序的 LayerDef 列表
        input_points: 点云点数 N（每个样本 N×3）
        input_channels: 输入通道数，默认 3（xyz）
        bytes_per_element: 每个元素的字节数（默认 4，float32）

    Returns:
        与 layers 同序的 ActivationShapeInfo 列表，便于确定层间激活交互次数与 buffer 数据量。
    """
    result: List[ActivationShapeInfo] = []
    # 上一层的输出 shape，第一层之前为 (input_points, input_channels)
    prev_output: Optional[Tuple[int, ...]] = (input_points, input_channels)
    prev_output_numel: int = input_points * input_channels

    for layer in layers:
        out_shape = _infer_layer_output_shape(layer)
        if out_shape is None:
            # 无算子层（如 global_maxpool）保持上一层的输出作为本层“输出”（下一层输入）
            if layer.layer_type == "global_maxpool" and prev_output and len(prev_output) >= 2:
                out_shape = (prev_output[-1],)
                out_numel = prev_output[-1]
            else:
                out_shape = prev_output
                out_numel = prev_output_numel
        else:
            if len(out_shape) == 1:
                out_numel = out_shape[0]
            else:
                out_numel = 1
                for d in out_shape:
                    out_numel *= d

        in_shape = prev_output
        in_numel = prev_output_numel

        info = ActivationShapeInfo(
            layer_name=layer.name,
            layer_type=layer.layer_type,
            input_shape=in_shape,
            output_shape=out_shape,
            input_numel=in_numel,
            output_numel=out_numel,
            input_bytes=in_numel * bytes_per_element,
            output_bytes=out_numel * bytes_per_element,
        )
        result.append(info)

        prev_output = out_shape
        prev_output_numel = out_numel

    return result


def format_activation_shape(shape: Optional[Tuple[int, ...]]) -> str:
    """将 shape 元组格式化为可读字符串，如 (1024, 3) -> '1024×3'。"""
    if not shape:
        return "—"
    return "×".join(str(s) for s in shape)
