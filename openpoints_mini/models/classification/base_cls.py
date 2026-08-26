import logging
from typing import List

import torch
import torch.nn as nn

from openpoints_mini.loss.build import build_criterion_from_cfg
from openpoints_mini.models.build import MODELS, build_model_from_cfg
from openpoints_mini.models.layers.conv import create_linearblock


@MODELS.register_module()
class BaseCls(nn.Module):
    """Classification model: encoder + optional head (same pattern as OpenPoints)."""

    def __init__(
        self,
        encoder_args=None,
        cls_args=None,
        criterion_args=None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning("unused kwargs in BaseCls: %s", kwargs)
        self.encoder = build_model_from_cfg(encoder_args)
        if cls_args is not None:
            in_ch = (
                self.encoder.out_channels
                if hasattr(self.encoder, "out_channels")
                else cls_args.get("in_channels", None)
            )
            cls_args = dict(cls_args)
            cls_args["in_channels"] = in_ch
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args else None

    def forward(self, data):
        global_feat = self.encoder.forward_cls_feat(data)
        return self.prediction(global_feat)

    def get_loss(self, pred, gt, inputs=None):
        if self.criterion is None:
            raise RuntimeError("criterion_args not set; pass criterion_args or use external loss")
        return self.criterion(pred, gt.long())

    def get_logits_loss(self, data, gt):
        logits = self.forward(data)
        return logits, self.get_loss(logits, gt)


@MODELS.register_module()
class ClsHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        mlps: List[int] = None,
        norm_args: dict = None,
        act_args: dict = None,
        dropout: float = 0.5,
        global_feat: str = None,
        point_dim: int = 2,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning("unused kwargs in ClsHead: %s", kwargs)
        self.global_feat = global_feat.split(",") if global_feat else None
        self.point_dim = point_dim
        in_channels = len(self.global_feat) * in_channels if self.global_feat else in_channels
        if mlps is not None:
            mlps = [in_channels] + mlps + [num_classes]
        else:
            mlps = [in_channels, num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(
                create_linearblock(mlps[i], mlps[i + 1], norm_args=norm_args, act_args=act_args)
            )
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        # Encoder forward_cls_feat 常为 (B, C, N)；分类头需要 (B, C) 时默认对点维做 max pool
        if self.global_feat is None and end_points.dim() == 3:
            end_points = torch.max(end_points, dim=-1)[0]
        if self.global_feat is not None:
            global_feats = []
            for preprocess in self.global_feat:
                if "max" in preprocess:
                    global_feats.append(torch.max(end_points, dim=self.point_dim, keepdim=False)[0])
                elif preprocess in ["avg", "mean"]:
                    global_feats.append(end_points.mean(dim=self.point_dim))
            end_points = torch.cat(global_feats, dim=1)
        return self.head(end_points)
