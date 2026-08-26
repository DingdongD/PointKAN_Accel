import copy
import logging
from typing import List

import torch
import torch.nn as nn

from openpoints_mini.models.build import MODELS, build_model_from_cfg
from openpoints_mini.models.layers.conv import create_convblock1d


@MODELS.register_module()
class BaseSeg(nn.Module):
    """语义分割：encoder + decoder + SegHead（与 OpenPoints BaseSeg 一致）。"""

    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning("unused kwargs in BaseSeg: %s", kwargs)
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            dec = copy.deepcopy(encoder_args)
            dec.update(decoder_args)
            dec["encoder_channel_list"] = (
                self.encoder.channel_list if hasattr(self.encoder, "channel_list") else None
            )
            self.decoder = build_model_from_cfg(dec)
        else:
            self.decoder = None

        if cls_args is not None:
            cls_args = dict(cls_args)
            if hasattr(self.decoder, "out_channels"):
                cls_args["in_channels"] = self.decoder.out_channels
            elif hasattr(self.encoder, "out_channels"):
                cls_args["in_channels"] = self.encoder.out_channels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)
        if self.head is not None:
            f = self.head(f)
        return f


@MODELS.register_module()
class BasePartSeg(BaseSeg):
    """部件分割（ShapeNet-Part）：需要 cls 条件时走 Point*PartDecoder。"""

    def forward(self, p0, f0=None, cls0=None):
        if hasattr(p0, "keys"):
            p0, f0, cls0 = p0["pos"], p0["x"], p0["cls"]
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(p0, f0)
        if self.decoder is not None:
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f


@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        mlps=None,
        norm_args=None,
        act_args=None,
        dropout=0.5,
        global_feat=None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning("unused kwargs in SegHead: %s", kwargs)
        if norm_args is None:
            norm_args = {"norm": "bn1d"}
        if act_args is None:
            act_args = {"act": "relu"}
        if global_feat is not None:
            self.global_feat = global_feat.split(",")
            multiplier = len(self.global_feat) + 1
        else:
            self.global_feat = None
            multiplier = 1
        in_channels *= multiplier

        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(
                create_convblock1d(mlps[i], mlps[i + 1], norm_args=norm_args, act_args=act_args)
            )
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        if self.global_feat is not None:
            global_feats = []
            for feat_type in self.global_feat:
                if "max" in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ["avg", "mean"]:
                    global_feats.append(end_points.mean(dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)
        return self.head(end_points)
