# -*- coding: utf-8 -*-
"""
FPS 优化版 PointKANElite（part segmentation）：
在 forward 开头预计算级联 FPS 索引，并在每层 GAM 中复用 fps_idx，
避免每层内部重复调用 furthest_point_sample 的调度开销。

注意：
- 这里的“复用”仅指复用 fps_idx（采样索引）。
- KNN 仍会在 knn_point 中计算 square_distance + topk；并未实现“FPS/KNN 复用同一份距离矩阵”。
"""

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .pointKANElite import (
    Geometric_Affine_Module,
    PointKANElite,
    index_points,
    knn_point,
    get_activation,
    ConvBNReLU1D,
    Local_Feature_Processing,
    Global_Feature_Processing,
    PointNetFeaturePropagation,
)


def get_stage_groups(points: int, reducers):
    anchor_points = int(points)
    groups_list = []
    for r in reducers:
        anchor_points = anchor_points // int(r)
        groups_list.append(anchor_points)
    return groups_list


def run_cascade_fps(xyz: torch.Tensor, groups_list):
    """
    xyz: [B, N, 3]
    returns: list[(fps_idx_current, fps_idx_in_N0)], each [B, G]
    """
    B, N, _ = xyz.shape
    device = xyz.device
    current_idx_in_N0 = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    results = []
    for g in groups_list:
        xyz_current = index_points(xyz, current_idx_in_N0)
        fps_idx_current = pointnet2_utils.furthest_point_sample(xyz_current, int(g)).long()
        fps_idx_in_N0 = torch.gather(current_idx_in_N0, 1, fps_idx_current)
        results.append((fps_idx_current, fps_idx_in_N0))
        current_idx_in_N0 = fps_idx_in_N0
    return results


class Geometric_Affine_Module_UseFPSIdx(Geometric_Affine_Module):
    def forward(self, xyz, points, fps_idx=None):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        if fps_idx is None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        e_new_points = torch.exp(new_points.permute(0, 3, 1, 2))
        up = (new_points.permute(0, 3, 1, 2) * e_new_points).mean(-1)
        down = e_new_points.mean(-1)
        center = torch.div(up, down)
        return new_xyz, new_points, center


class PointKANEliteFPSOptimized(nn.Module):
    """
    与原 PointKANElite 同结构同参数命名（GAM_list/LFP_blocks_list/GFP_blocks_list/decode_list...），
    仅将 GAM 替换为可接收 fps_idx 的版本，并在 forward 里预计算 cascade FPS。
    """

    def __init__(
        self,
        num_classes=50,
        points=2048,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 2),
        LFP_blocks=(2, 2, 2, 2),
        GFP_blocks=(2, 2, 2, 2),
        k_neighbors=(32, 32, 32, 32),
        reducers=(4, 4, 4, 4),
        de_dims=(512, 256, 128, 128),
        de_blocks=(2, 2, 2, 2),
        gmp_dim=64,
        cls_dim=64,
        **kwargs,
    ):
        super().__init__()

        self.stages = len(LFP_blocks)
        self.class_num = num_classes
        self.points = int(points)
        self._reducers = list(reducers)
        self._groups_list = get_stage_groups(self.points, self._reducers)

        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)

        assert len(LFP_blocks) == len(k_neighbors) == len(reducers) == len(GFP_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        self.GAM_list = nn.ModuleList()
        self.LFP_blocks_list = nn.ModuleList()
        self.GFP_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]

        # Encoder
        for i in range(len(LFP_blocks)):
            out_channel = last_channel * dim_expansion[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = Geometric_Affine_Module_UseFPSIdx(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.GAM_list.append(local_grouper)

            self.LFP_blocks_list.append(
                Local_Feature_Processing(
                    last_channel, out_channel, LFP_blocks[i], groups=groups,
                    res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz
                )
            )
            self.GFP_blocks_list.append(
                Global_Feature_Processing(
                    out_channel, GFP_blocks[i], groups=groups,
                    res_expansion=res_expansion, bias=bias, activation=activation
                )
            )

            last_channel = out_channel
            en_dims.append(last_channel)

        # Decoder
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims = list(de_dims)
        de_dims.insert(0, en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(
                    de_dims[i] + en_dims[i + 1],
                    de_dims[i + 1],
                    blocks=de_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )

        self.act = get_activation(activation)

        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation),
        )
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation)

        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + cls_dim + de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias),
        )
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):
        xyz = x.permute(0, 2, 1)  # [B,N,3]
        x = torch.cat([x, norm_plt], dim=1)
        x = self.embedding(x)

        xyz_list = [xyz]
        x_list = [x]

        fps_results = run_cascade_fps(xyz, self._groups_list)

        # Encoder
        for i in range(self.stages):
            fps_idx = fps_results[i][0]
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1), fps_idx=fps_idx)
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)

        # Decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)

        # global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))

        # cls token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


def pointKANEliteFPSOptimized(num_classes=50, **kwargs) -> PointKANEliteFPSOptimized:
    # mirror defaults from pointKANElite() in pointKANElite.py
    points = int(kwargs.pop("points", 2048))
    return PointKANEliteFPSOptimized(
        num_classes=num_classes,
        points=points,
        embed_dim=32,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 2],
        LFP_blocks=[1, 1, 1, 1],
        GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[32, 32, 32, 32],
        reducers=[4, 4, 4, 4],
        de_dims=[512, 256, 128, 128],
        de_blocks=[4, 4, 4, 4],
        gmp_dim=64,
        cls_dim=64,
        **kwargs,
    )

