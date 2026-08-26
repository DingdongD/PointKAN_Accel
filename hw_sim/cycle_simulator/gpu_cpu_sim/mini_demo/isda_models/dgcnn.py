# SPDX-License-Identifier: Apache-2.0
# 源自 ISEDA26/model_back.py；修正 get_graph_feature 中硬编码 cuda，保证 CPU/GPU 可跑。

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """x: (B, C, N) -> idx (B, N, k)"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
    dim9: bool = False,
    return_raw_features: bool = False,
    return_indices: bool = False,
):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = x.device

    if idx is None:
        if not dim9:
            idx_tmp = knn(x, k=k)
        else:
            idx_tmp = knn(x[:, 6:, :], k=k)
    else:
        idx_tmp = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = (idx_tmp + idx_base).view(-1)

    _, num_dims, _ = x.size()
    xt = x.transpose(2, 1).contiguous()
    feature = xt.view(batch_size * num_points, -1)[idx_flat, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    xc = xt.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if return_raw_features:
        if return_indices:
            return feature, idx_tmp
        return feature

    out = torch.cat((feature - xc, xc), dim=3).permute(0, 3, 1, 2).contiguous()
    if return_indices:
        return out, idx_tmp
    return out


class DGCNN_cls(nn.Module):
    """ModelNet40 风格 DGCNN 分类头。"""

    def __init__(self, args, output_channels: int = 40):
        super().__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_channels)

        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)

    def forward(self, x: torch.Tensor, bits: int = 6, collect_feats: bool = False):
        del bits  # 与原版 API 兼容
        batch_size = x.size(0)
        collected_feats = [] if collect_feats else None

        def collect_features(feat):
            if collect_feats:
                collected_feats.append(feat.detach().cpu())

        collect_features(x)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        collect_features(x1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        collect_features(x2)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        collect_features(x3)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1p = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x2p = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1p, x2p), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return (x, collected_feats) if collect_feats else x
