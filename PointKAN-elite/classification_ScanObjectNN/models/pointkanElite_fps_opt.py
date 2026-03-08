# -*- coding: utf-8 -*-
"""
FPS 优化版 PointKANElite（ScanObjectNN）：推理时预计算级联 FPS 索引，与原始模型数值一致。
与 classification_ModelNet40 的 pointkanElite_fps_opt 同构，默认 num_classes=15 以匹配 ScanObjectNN。
"""

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .pointkanElite import (
    Geometric_Affine_Module,
    Model,
    index_points,
    knn_point,
    get_activation,
    ConvBNReLU1D,
    Local_Feature_Processing,
    Global_Feature_Processing,
)


def get_stage_groups(points=1024, reducers=None):
    if reducers is None:
        reducers = [2, 2, 2, 2]
    anchor_points = points
    groups_list = []
    for r in reducers:
        anchor_points = anchor_points // r
        groups_list.append(anchor_points)
    return groups_list


def run_cascade_fps(xyz, groups_list):
    B, N, _ = xyz.shape
    device = xyz.device
    current_idx_in_1024 = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    results = []
    for g in groups_list:
        xyz_current = index_points(xyz, current_idx_in_1024)
        fps_idx_current = pointnet2_utils.furthest_point_sample(xyz_current, g).long()
        fps_idx_in_1024 = torch.gather(current_idx_in_1024, 1, fps_idx_current)
        results.append((fps_idx_current, fps_idx_in_1024))
        current_idx_in_1024 = fps_idx_in_1024
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


class ModelFPSOptimized(Model):
    def __init__(self, points=1024, class_num=15, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], LFP_blocks=[2, 2, 2, 2], GFP_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        nn.Module.__init__(self)
        self.stages = len(LFP_blocks)
        self.class_num = class_num
        self.points = points
        self._reducers = reducers
        self._groups_list = get_stage_groups(points, reducers)
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(LFP_blocks) == len(k_neighbors) == len(reducers) == len(GFP_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.GAM_list = nn.ModuleList()
        self.LFP_blocks_list = nn.ModuleList()
        self.GFP_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(len(LFP_blocks)):
            out_channel = last_channel * dim_expansion[i]
            LFP_block_num = LFP_blocks[i]
            GFP_block_num = GFP_blocks[i]
            kneighbor = k_neighbors[i]
            anchor_points = self._groups_list[i]
            local_grouper = Geometric_Affine_Module_UseFPSIdx(
                last_channel, anchor_points, kneighbor, use_xyz, normalize
            )
            self.GAM_list.append(local_grouper)
            LFP_block_module = Local_Feature_Processing(
                last_channel, out_channel, LFP_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz
            )
            self.LFP_blocks_list.append(LFP_block_module)
            GFP_block_module = Global_Feature_Processing(
                out_channel, GFP_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation
            )
            self.GFP_blocks_list.append(GFP_block_module)
            last_channel = out_channel
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num),
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)
        fps_results = run_cascade_fps(xyz, self._groups_list)
        for i in range(self.stages):
            fps_idx = fps_results[i][0]
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1), fps_idx=fps_idx)
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pointKANEliteFPSOptimized(num_classes=15, **kwargs) -> ModelFPSOptimized:
    """与 pointKANElite() 相同配置（ScanObjectNN 默认 15 类），用于加载同一 checkpoint 的 FPS 优化推理。"""
    return ModelFPSOptimized(
        points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs
    )
