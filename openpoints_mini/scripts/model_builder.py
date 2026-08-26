"""
根据任务、骨干（PointNeXt / PointVector）与数据集名称构建 ``BaseCls`` / ``BaseSeg`` / ``BasePartSeg``。
供 ``scripts/train.py``、``scripts/infer.py`` 与 ``train_s3dis_pointnext_iseda26_data.py`` 复用。
"""
from __future__ import annotations

import torch.nn as nn

from openpoints_mini.configs import presets
from openpoints_mini.models.build import build_model_from_cfg


def build_s3dis_semseg_model(
    backbone: str,
    variant: str,
    in_channels: int,
    device,
) -> nn.Module:
    """S3DIS 室内 HDF5 语义分割（13 类）。"""
    backbone = backbone.lower()
    if backbone == "pointnext":
        enc = presets.pointnext_encoder_s3dis_l(in_channels=in_channels)
        dec_name = "PointNextDecoder"
    elif backbone == "pointvector":
        if variant.lower() == "xl":
            enc = presets.pointvector_encoder_s3dis_xl(in_channels=in_channels)
        else:
            enc = presets.pointvector_encoder_s3dis_l(in_channels=in_channels)
        dec_name = "PointVectorDecoder"
    else:
        raise ValueError(f"未知 backbone: {backbone}，可选 pointnext / pointvector")
    return build_model_from_cfg(
        {
            "NAME": "BaseSeg",
            "encoder_args": enc,
            "decoder_args": {"NAME": dec_name},
            "cls_args": {
                "NAME": "SegHead",
                "num_classes": 13,
                "in_channels": None,
                "norm_args": {"norm": "bn"},
            },
        }
    ).to(device)


def build_modelnet40_cls_model(backbone: str, device) -> nn.Module:
    """ModelNet40 分类（40 类）。"""
    backbone = backbone.lower()
    if backbone == "pointnext":
        enc = presets.pointnext_encoder_modelnet40_s()
    elif backbone == "pointvector":
        enc = presets.pointvector_encoder_modelnet40_s()
    else:
        raise ValueError(f"未知 backbone: {backbone}")
    return build_model_from_cfg(
        {
            "NAME": "BaseCls",
            "encoder_args": enc,
            "cls_args": {
                "NAME": "ClsHead",
                "num_classes": 40,
                "mlps": [256],
                "norm_args": {"norm": "bn1d"},
                "act_args": {"act": "relu"},
                "dropout": 0.4,
            },
        }
    ).to(device)


def build_shapenet_part_model(
    backbone: str, device, in_channels: int | None = None
) -> nn.Module:
    """ShapeNet-Part 部件分割（50 类 part id；decoder 内类别 one-hot 维数为 16）。"""
    backbone = backbone.lower()
    if backbone == "pointnext":
        enc = presets.pointnext_encoder_shapenetpart_s(
            **({"in_channels": in_channels} if in_channels is not None else {})
        )
        dec_name = "PointNextPartDecoder"
    elif backbone == "pointvector":
        enc = presets.pointvector_encoder_shapenetpart_s(
            **({"in_channels": in_channels} if in_channels is not None else {})
        )
        dec_name = "PointVectorPartDecoder"
    else:
        raise ValueError(f"未知 backbone: {backbone}")
    return build_model_from_cfg(
        {
            "NAME": "BasePartSeg",
            "encoder_args": enc,
            "decoder_args": {
                "NAME": dec_name,
                "cls_map": "pointnet2",
                "num_classes": 16,
            },
            "cls_args": {
                "NAME": "SegHead",
                "num_classes": 50,
                "in_channels": None,
                "norm_args": {"norm": "bn"},
            },
        }
    ).to(device)
