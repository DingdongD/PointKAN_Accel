# SPDX-License-Identifier: Apache-2.0
# 源自 ISEDA26/model_back.py PointNet++：分类 / 部件分割 / 语义分割（SSG / MSG）

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geom import (
    farthest_point_sample,
    index_points,
    query_ball_point,
    sample_and_group,
    sample_and_group_all,
    square_distance,
)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None, bits: int = 8):
        del bits
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=None):
        super().__init__()
        if dropout_rate is None:
            self.blks = nn.Linear(in_features, out_features)
        else:
            self.blks = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        return self.blks(x)


class PointNet2ClsSSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.normal = args.normal
        addition_channel = 3 if self.normal else 0
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3 + addition_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = FullyConnectedLayer(1024, 512, dropout_rate=0.4)
        self.fc2 = FullyConnectedLayer(512, 256, dropout_rate=0.4)
        self.fc3 = FullyConnectedLayer(256, args.num_classes)

    def forward(self, xyz: torch.Tensor):
        B, _, _ = xyz.shape
        if self.normal:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        bw = 4
        l1_xyz, l1_points = self.sa1(xyz, norm, bits=bw)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, bits=bw)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, bits=bw)
        x = l3_points.view(B, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, l3_points


class PointNet2ClsMSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.normal = args.normal
        addition_channel = 3 if self.normal else 0
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=addition_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=320,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=640 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

    def forward(self, xyz: torch.Tensor):
        B, _, _ = xyz.shape
        if self.normal:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(l3_points.shape[0], -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, l3_points


class PointNetFeaturePropagation(nn.Module):
    """语义分割上采样用（预留与 model_back 一致接口）。"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2PartSegSSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.normal = args.normal
        addition_channel = 3 if self.normal else 0
        # sample_and_group 输出通道 = 3(相对坐标) + 点特征维；6 维点(xyz+法向) => 3+6=9
        sa1_in = 3 + 6 + addition_channel
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=sa1_in,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fc2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        # cat([one_hot(16), l0_xyz(3), l0_points(6+addition)])
        self.fc3 = PointNetFeaturePropagation(
            in_channel=128 + 16 + 3 + 6 + addition_channel, mlp=[128, 128, 128]
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv2 = nn.Conv1d(128, args.num_classes, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, xyz: torch.Tensor, cls_label: torch.Tensor):
        B, _, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :].contiguous()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fc1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fc2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fc3(
            l0_xyz,
            l1_xyz,
            torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1),
            l1_points,
        )
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)
        return out, l3_points


class PointNet2PartSegMSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.normal = args.normal
        addition_channel = 3 if args.normal else 0
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=3 + addition_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=320,
            mlp_list=[[128, 128, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fc2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fc3 = PointNetFeaturePropagation(
            in_channel=150 + addition_channel, mlp=[128, 128]
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv2 = nn.Conv1d(128, args.num_classes, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, xyz: torch.Tensor, cls_label: torch.Tensor):
        B, _, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :].contiguous()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fc1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fc2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fc3(
            l0_xyz,
            l1_xyz,
            torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1),
            l1_points,
        )
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)
        return out, l3_points


class PointNet2SemSSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        nc = getattr(args, "num_classes", 13)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=9 + 3,
            mlp=[32, 32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16,
            radius=0.8,
            nsample=32,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
            group_all=False,
        )
        self.fc1 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fc2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fc3 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        self.fc4 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, nc, 1),
        )

    def forward(self, xyz: torch.Tensor):
        l0_points, l0_xyz = xyz, xyz[:, :3, :]
        bw = 8
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, bits=bw)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, bits=bw)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, bits=bw)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, bits=bw)
        l3_points = self.fc1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fc2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fc3(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fc4(l0_xyz, l1_xyz, None, l1_points)
        out = self.conv1(l0_points)
        return out, l4_points


class PointNet2SemMSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        nc = getattr(args, "num_classes", 13)
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=9,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=32 + 64,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.2, 0.4],
            nsample_list=[16, 32],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
        )
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=16,
            radius_list=[0.4, 0.8],
            nsample_list=[16, 32],
            in_channel=256 + 256,
            mlp_list=[[256, 256, 512], [256, 384, 512]],
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=512 + 512 + 256 + 256, mlp=[256, 256]
        )
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128 + 128, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 64 + 32, mlp=[256, 128])
        self.fp4 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, nc, 1),
        )

    def forward(self, xyz: torch.Tensor):
        l0_points, l0_xyz = xyz, xyz[:, 3:6, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp4(l0_xyz, l1_xyz, None, l1_points)
        out = self.conv1(l0_points)
        return out, l4_points
