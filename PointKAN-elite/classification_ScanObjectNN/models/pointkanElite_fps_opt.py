# -*- coding: utf-8 -*-
"""
FPS 优化版 PointKANElite（ScanObjectNN）：推理时预计算级联 FPS 索引，与原始模型数值一致。
与 classification_ModelNet40 的 pointkanElite_fps_opt 同构，默认 num_classes=15 以匹配 ScanObjectNN。
"""

from __future__ import absolute_import

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .pointkanElite import (
    Geometric_Affine_Module,
    Model,
    index_points,
    knn_point,
    farthest_point_sample,
    square_distance,
    get_activation,
    ConvBNReLU1D,
    Local_Feature_Processing,
    Global_Feature_Processing,
)
from knn_fps_cuda import fps_knn_coarse_fine_cuda as _fps_knn_cf_cuda

@torch.no_grad()
def fps_with_knn_coarse_fine_fused(
    X: torch.Tensor,
    K: int,
    *,
    k_neighbors: int = 24,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    min_candidates_ratio: float = 1.2,
    seed: str = "first",
    exclude_self: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    来自 classification_ModelNet40/spec_kan_fps.py 的融合思路（做了简化，专供模型推理用）：
    在每一轮 FPS 选出 center 后，立刻用同一份距离 d 做 coarse-fine KNN。

    返回：
      - idx:     (B, K) FPS 索引（在当前点集空间）
      - knn_idx: (B, K, k) KNN 索引（在当前点集空间）
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(int(K), int(N))
    k_actual = min(int(k_neighbors), int(N))
    min_candidates = max(k_actual + 1, int(k_actual * float(min_candidates_ratio)))

    
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)

    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        x2 = (X * X).sum(dim=-1)  # [B, N]
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    # use dot-product distance (faster than torch.cdist for this use case)
    x2 = (X * X).sum(dim=-1)  # [B, N]

    for t in range(K):
        idx[:, t] = farthest

        C = X[batch_arange, farthest]  # [B, D]
        c2 = (C * C).sum(dim=-1, keepdim=True)  # [B, 1]
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)  # [B, N]
        d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()

        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d

        # Bidirectional threshold adjustment to enable real pruning:
        # stop when k < count <= min_candidates
        threshold = torch.full((B,), float(init_threshold), dtype=dtype, device=device)
        gamma = 1.1
        mask = None
        for _ in range(int(max_iter)):
            mask = d_knn < threshold.unsqueeze(-1)
            count = mask.sum(dim=1)
            too_few = count <= k_actual
            too_many = count > min_candidates
            done = ~(too_few | too_many)
            if done.all():
                break
            # increase threshold where too_few; decrease where too_many; keep where done
            threshold = torch.where(too_few, threshold * gamma, threshold)
            threshold = torch.where(too_many, threshold / gamma, threshold)
        if mask is None:
            mask = d_knn < threshold.unsqueeze(-1)

        masked_d = d_knn.masked_fill(~mask, float("inf"))
        _knn_vals, knn_idx_t = torch.topk(masked_d, k=k_actual, dim=1, largest=False)
        knn_idx[:, t, :] = knn_idx_t

        torch.minimum(min_dist, d, out=min_dist)
        min_dist[batch_arange, farthest] = 0.0

        if t < K - 1:
            farthest = torch.argmax(min_dist, dim=1)

    return idx, knn_idx


@torch.no_grad()
def fps_with_knn_coarse_fine_fused_blockwise(
    X: torch.Tensor,
    K: int,
    *,
    k_neighbors: int = 24,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    min_candidates_ratio: float = 1.2,
    seed: str = "first",
    exclude_self: bool = False,
    block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    原型实现移植（来自 classification_ModelNet40/spec_kan_fps.py）：
    融合 FPS + coarse-fine KNN，并用 blockwise 扫描构造候选 buffer mask。

    返回：
      - idx:     [B, K]
      - knn_idx: [B, K, k]
    """
    assert X.ndim == 3
    assert block_size > 0
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(int(K), int(N))
    k_actual = min(int(k_neighbors), int(N))
    min_candidates = max(k_actual + 1, int(k_actual * float(min_candidates_ratio)))

    x2 = (X * X).sum(dim=-1)  # [B, N]

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)

    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    for t in range(K):
        idx[:, t] = farthest

        C = X[batch_arange, farthest]  # [B, D]
        c2 = (C * C).sum(dim=-1, keepdim=True)  # [B, 1]
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)  # [B, N]
        d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()

        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d

        threshold = torch.full((B,), float(init_threshold), dtype=dtype, device=device)
        final_candidate_mask: Optional[torch.Tensor] = None

        for _ in range(int(max_iter)):
            candidate_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            for start in range(0, N, int(block_size)):
                end = min(start + int(block_size), N)
                d_blk = d_knn[:, start:end]
                blk_mask = d_blk < threshold.unsqueeze(-1)
                candidate_mask[:, start:end] |= blk_mask
            final_candidate_mask = candidate_mask
            count = candidate_mask.sum(dim=1)
            done = count >= min_candidates
            if bool(done.all()):
                break
            threshold = torch.where(done, threshold, threshold * 1.1)

        if final_candidate_mask is None:
            final_candidate_mask = d_knn < threshold.unsqueeze(-1)

        masked_d = d_knn.masked_fill(~final_candidate_mask, float("inf"))
        _vals, knn_idx_t = torch.topk(masked_d, k=k_actual, dim=1, largest=False, sorted=False)
        knn_idx[:, t, :] = knn_idx_t

        torch.minimum(min_dist, d, out=min_dist)
        min_dist[batch_arange, farthest] = 0.0
        if t < K - 1:
            farthest = torch.argmax(min_dist, dim=1)

    return idx, knn_idx


@torch.no_grad()
def fps_with_knn_vanilla_fused(
    X: torch.Tensor,
    K: int,
    *,
    k_neighbors: int = 24,
    seed: str = "first",
    exclude_self: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    spec_kan_fps 风格的 vanilla（纯 PyTorch 循环）：
    每一步 FPS 选出 center 后，直接在全体 N 点上对距离 d 做 topk 得 KNN（无 coarse-fine）。

    返回：
      - idx:     [B, K]
      - knn_idx: [B, K, k]
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(int(K), int(N))
    k_actual = min(int(k_neighbors), int(N))

    x2 = (X * X).sum(dim=-1)  # [B, N]
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)

    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    for t in range(K):
        idx[:, t] = farthest

        C = X[batch_arange, farthest]  # [B, D]
        c2 = (C * C).sum(dim=-1, keepdim=True)  # [B, 1]
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)  # [B, N]
        d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()

        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d

        _vals, knn_idx_t = torch.topk(d_knn, k=k_actual, dim=1, largest=False, sorted=False)
        knn_idx[:, t, :] = knn_idx_t

        torch.minimum(min_dist, d, out=min_dist)
        min_dist[batch_arange, farthest] = 0.0
        if t < K - 1:
            farthest = torch.argmax(min_dist, dim=1)

    return idx, knn_idx


@torch.no_grad()
def fps_with_knn_coarse_fine_cuda(
    X: torch.Tensor,
    K: int,
    *,
    k_neighbors: int = 24,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    min_candidates_ratio: float = 1.2,
    seed: str = "first",
    exclude_self: bool = False,
    block_size: int = 256,
    mmax_factor: int = 4,
    strict_semantic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA extension: fused FPS + coarse-fine KNN.
    note: CUDA kernel currently expects float32 input.
    """
    if X.device.type != "cuda":
        raise ValueError("fps_with_knn_coarse_fine_cuda requires CUDA tensor input")
    x32 = X.contiguous().float()
    return _fps_knn_cf_cuda(
        x32,
        int(K),
        k_neighbors=int(k_neighbors),
        init_threshold=float(init_threshold),
        max_iter=int(max_iter),
        min_candidates_ratio=float(min_candidates_ratio),
        seed=str(seed),
        exclude_self=bool(exclude_self),
        block_size=int(block_size),
        mmax_factor=int(mmax_factor),
        strict_semantic=bool(strict_semantic),
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
        fps_idx_current = pointnet2_utils.furthest_point_sample(xyz_current.contiguous().float(), g).long()
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
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous().float(), self.groups).long()
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


class Geometric_Affine_Module_PyTorchFPS(Geometric_Affine_Module):
    """
    仅把 pointnet2_utils.furthest_point_sample 替换为 PyTorch 版 farthest_point_sample（Python for-loop）。
    KNN 仍使用 knn_point(square_distance+topk)。
    """

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        fps_idx = farthest_point_sample(xyz, int(self.groups)).long()
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


class Geometric_Affine_Module_FusedFPSKNN(Geometric_Affine_Module):
    """
    用融合版 FPS + coarse-fine KNN（由距离 d 直接得 knn_idx），避免再调用 knn_point(square_distance+topk)。
    注意：这里的 FPS 是纯 PyTorch 循环实现（非 pointnet2_utils kernel），数值与原实现可能不同。
    """

    def __init__(
        self,
        channel,
        groups,
        kneighbors,
        use_xyz=True,
        normalize="center",
        *,
        init_threshold: float = 0.1,
        max_iter: int = 10,
        min_candidates_ratio: float = 1.2,
        seed: str = "first",
        exclude_self: bool = False,
        block_size: int = 256,
        **kwargs,
    ):
        super().__init__(channel, groups, kneighbors, use_xyz=use_xyz, normalize=normalize, **kwargs)
        self._init_threshold = float(init_threshold)
        self._max_iter = int(max_iter)
        self._min_candidates_ratio = float(min_candidates_ratio)
        self._seed = str(seed)
        self._exclude_self = bool(exclude_self)
        self._block_size = int(block_size)

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = int(self.groups)
        xyz = xyz.contiguous()

        fps_idx, knn_idx = fps_with_knn_coarse_fine_fused_blockwise(
            xyz,
            S,
            k_neighbors=int(self.kneighbors),
            init_threshold=self._init_threshold,
            max_iter=self._max_iter,
            min_candidates_ratio=self._min_candidates_ratio,
            seed=self._seed,
            exclude_self=self._exclude_self,
            block_size=self._block_size,
        )

        new_xyz = index_points(xyz, fps_idx)      # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, d]

        grouped_xyz = index_points(xyz, knn_idx)       # [B, S, k, 3]
        grouped_points = index_points(points, knn_idx) # [B, S, k, d]

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

        new_points = torch.cat(
            [grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, int(self.kneighbors), 1)],
            dim=-1,
        )
        e_new_points = torch.exp(new_points.permute(0, 3, 1, 2))
        up = (new_points.permute(0, 3, 1, 2) * e_new_points).mean(-1)
        down = e_new_points.mean(-1)
        center = torch.div(up, down)
        return new_xyz, new_points, center


class Geometric_Affine_Module_SpecFPSKNN(Geometric_Affine_Module):
    """
    公平对比用：完全走 spec_kan_fps 风格的 FPS+KNN（纯 PyTorch 循环），不使用 pointnet2 FPS CUDA kernel。

    variant:
      - "vanilla": 全量 topk
      - "coarse_fine": blockwise coarse-fine（候选 buffer + masked topk）
    """

    def __init__(
        self,
        channel,
        groups,
        kneighbors,
        use_xyz=True,
        normalize="center",
        *,
        variant: str = "vanilla",
        init_threshold: float = 0.1,
        max_iter: int = 10,
        min_candidates_ratio: float = 1.2,
        seed: str = "first",
        exclude_self: bool = False,
        block_size: int = 256,
        mmax_factor: int = 4,
        strict_semantic: bool = False,
        **kwargs,
    ):
        super().__init__(channel, groups, kneighbors, use_xyz=use_xyz, normalize=normalize, **kwargs)
        self._variant = str(variant)
        self._init_threshold = float(init_threshold)
        self._max_iter = int(max_iter)
        self._min_candidates_ratio = float(min_candidates_ratio)
        self._seed = str(seed)
        self._exclude_self = bool(exclude_self)
        self._block_size = int(block_size)
        self._mmax_factor = int(mmax_factor)
        self._strict_semantic = bool(strict_semantic)

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        S = int(self.groups)
        xyz = xyz.contiguous()

        if self._variant == "vanilla":
            fps_idx, knn_idx = fps_with_knn_vanilla_fused(
                xyz,
                S,
                k_neighbors=int(self.kneighbors),
                seed=self._seed,
                exclude_self=self._exclude_self,
            )
        elif self._variant == "coarse_fine":
            fps_idx, knn_idx = fps_with_knn_coarse_fine_fused_blockwise(
                xyz,
                S,
                k_neighbors=int(self.kneighbors),
                init_threshold=float(self._init_threshold),
                max_iter=int(self._max_iter),
                min_candidates_ratio=float(self._min_candidates_ratio),
                seed=str(self._seed),
                exclude_self=bool(self._exclude_self),
                block_size=int(self._block_size),
            )
        elif self._variant == "cuda_cf":
            fps_idx, knn_idx = fps_with_knn_coarse_fine_cuda(
                xyz,
                S,
                k_neighbors=int(self.kneighbors),
                init_threshold=float(self._init_threshold),
                max_iter=int(self._max_iter),
                min_candidates_ratio=float(self._min_candidates_ratio),
                seed=str(self._seed),
                exclude_self=bool(self._exclude_self),
                block_size=int(self._block_size),
                mmax_factor=int(self._mmax_factor),
                strict_semantic=bool(self._strict_semantic),
            )
        else:
            raise ValueError(f"Unknown variant={self._variant}")

        new_xyz = index_points(xyz, fps_idx)        # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, d]

        grouped_xyz = index_points(xyz, knn_idx)         # [B, S, k, 3]
        grouped_points = index_points(points, knn_idx)   # [B, S, k, d]

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = (
                torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points_cat = torch.cat(
            [grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, int(self.kneighbors), 1)],
            dim=-1,
        )
        e_new_points = torch.exp(new_points_cat.permute(0, 3, 1, 2))
        up = (new_points_cat.permute(0, 3, 1, 2) * e_new_points).mean(-1)
        down = e_new_points.mean(-1)
        center = torch.div(up, down)
        return new_xyz, new_points_cat, center


class Geometric_Affine_Module_PointNet2FPS_CoarseFineKNN(Geometric_Affine_Module):
    """
    保留 pointnet2 的 FPS CUDA kernel（furthest_point_sample），仅将 KNN 替换为 coarse-fine（阈值筛选 + topk）。

    注意：
    - 这里 coarse-fine 只改变“选邻居”的策略；距离矩阵仍需要计算（square_distance）。
    - 该实现目标是：先对齐 baseline（FPS 仍走 pointnet2 kernel），再观察 coarse-fine 对整体耗时的影响。
    """

    def __init__(
        self,
        channel,
        groups,
        kneighbors,
        use_xyz=True,
        normalize="center",
        *,
        init_threshold: float = 0.15,
        max_iter: int = 10,
        min_candidates_ratio: float = 1.1,
        exclude_self: bool = False,
        **kwargs,
    ):
        super().__init__(channel, groups, kneighbors, use_xyz=use_xyz, normalize=normalize, **kwargs)
        self._init_threshold = float(init_threshold)
        self._max_iter = int(max_iter)
        self._min_candidates_ratio = float(min_candidates_ratio)
        self._exclude_self = bool(exclude_self)

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        S = int(self.groups)
        k_actual = min(int(self.kneighbors), int(N))
        min_candidates = max(k_actual + 1, int(k_actual * float(self._min_candidates_ratio)))

        xyz = xyz.contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous().float(), S).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        # d2: [B, S, N]
        d2 = square_distance(new_xyz, xyz)
        if self._exclude_self:
            # 将自身点距离设为 +inf（只有当 new_xyz 是 xyz 的子集时才有意义）
            # 这里用 fps_idx 对应到 N 维度，做 per-(B,S) scatter。
            b = torch.arange(B, device=xyz.device).unsqueeze(1).expand(B, S)
            s = torch.arange(S, device=xyz.device).unsqueeze(0).expand(B, S)
            d2[b, s, fps_idx] = float("inf")

        # coarse-fine 阈值迭代（按 (B,S) 并行）
        thr = torch.full((B, S), float(self._init_threshold), dtype=d2.dtype, device=d2.device)
        gamma = 1.1
        mask = None
        for _ in range(int(self._max_iter)):
            mask = d2 < thr.unsqueeze(-1)  # [B,S,N]
            cnt = mask.sum(dim=-1)         # [B,S]
            too_few = cnt <= k_actual
            too_many = cnt > min_candidates
            done = ~(too_few | too_many)
            if bool(done.all()):
                break
            thr = torch.where(too_few, thr * gamma, thr)
            thr = torch.where(too_many, thr / gamma, thr)
        if mask is None:
            mask = d2 < thr.unsqueeze(-1)

        # 重要：这里仍对 N 做 topk，但大量非候选被置为 inf（便于先观察“只换策略”的效果）
        masked_d2 = d2.masked_fill(~mask, float("inf"))
        _, idx = torch.topk(masked_d2, k=k_actual, dim=-1, largest=False, sorted=False)  # [B,S,k]

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
            std = (
                torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points_cat = torch.cat(
            [grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, int(self.kneighbors), 1)],
            dim=-1,
        )
        e_new_points = torch.exp(new_points_cat.permute(0, 3, 1, 2))
        up = (new_points_cat.permute(0, 3, 1, 2) * e_new_points).mean(-1)
        down = e_new_points.mean(-1)
        center = torch.div(up, down)
        return new_xyz, new_points_cat, center


class ModelPointNet2FPSCoarseFineKNNOptimized(Model):
    """保留 pointnet2 FPS kernel，仅替换 KNN 为 coarse-fine 的 ScanObjectNN 分类模型。"""

    def __init__(
        self,
        points=1024,
        class_num=15,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="center",
        dim_expansion=[2, 2, 2, 2],
        LFP_blocks=[2, 2, 2, 2],
        GFP_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        *,
        init_threshold_coarse_fine: float = 0.15,
        max_iter_coarse_fine: int = 10,
        min_candidates_ratio: float = 1.1,
        **kwargs,
    ):
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

            local_grouper = Geometric_Affine_Module_PointNet2FPS_CoarseFineKNN(
                last_channel,
                anchor_points,
                kneighbor,
                use_xyz,
                normalize,
                init_threshold=float(init_threshold_coarse_fine),
                max_iter=int(max_iter_coarse_fine),
                min_candidates_ratio=float(min_candidates_ratio),
                exclude_self=False,
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
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1))
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


class ModelFusedFPSKNNOptimized(Model):
    """
    融合版 FPS+KNN（探索版）：
    - 每层 GAM 内部用 fps_with_knn_coarse_fine_fused 直接生成 fps_idx + knn_idx
    - 不再调用 pointnet2_utils.furthest_point_sample，也不再调用 knn_point

    注意：该实现与原模型数值不保证一致，仅用于性能探索/对比。
    """

    def __init__(
        self,
        points=1024,
        class_num=15,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="center",
        dim_expansion=[2, 2, 2, 2],
        LFP_blocks=[2, 2, 2, 2],
        GFP_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        # fused hyperparams
        init_threshold_coarse_fine: float = 0.15,
        max_iter_coarse_fine: int = 10,
        min_candidates_ratio: float = 1.1,
        seed: str = "first",
        **kwargs,
    ):
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

            local_grouper = Geometric_Affine_Module_FusedFPSKNN(
                last_channel,
                anchor_points,
                kneighbor,
                use_xyz,
                normalize,
                init_threshold=float(init_threshold_coarse_fine),
                max_iter=int(max_iter_coarse_fine),
                min_candidates_ratio=float(min_candidates_ratio),
                seed=str(seed),
                exclude_self=False,
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
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1))
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


class ModelSpecFPSKNNOptimized(Model):
    """完全走 spec_kan_fps 风格 FPS+KNN（vanilla 或 coarse_fine）的 ScanObjectNN 分类模型。"""

    def __init__(
        self,
        points=1024,
        class_num=15,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="center",
        dim_expansion=[2, 2, 2, 2],
        LFP_blocks=[2, 2, 2, 2],
        GFP_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        *,
        variant: str = "vanilla",
        init_threshold_coarse_fine: float = 0.1,
        max_iter_coarse_fine: int = 10,
        min_candidates_ratio: float = 1.2,
        seed: str = "first",
        block_size: int = 256,
        mmax_factor: int = 4,
        **kwargs,
    ):
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

            local_grouper = Geometric_Affine_Module_SpecFPSKNN(
                last_channel,
                anchor_points,
                kneighbor,
                use_xyz,
                normalize,
                variant=str(variant),
                init_threshold=float(init_threshold_coarse_fine),
                max_iter=int(max_iter_coarse_fine),
                min_candidates_ratio=float(min_candidates_ratio),
                seed=str(seed),
                exclude_self=False,
                block_size=int(block_size),
                mmax_factor=int(mmax_factor),
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
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1))
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


class ModelPyTorchFPSOptimized(Model):
    """仅替换 FPS 为 PyTorch 版 farthest_point_sample 的 ScanObjectNN 分类模型。"""

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
            local_grouper = Geometric_Affine_Module_PyTorchFPS(
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
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1))
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


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


class ModelFPSPrefixOptimized(ModelFPSOptimized):
    """
    Prefix-approx 版：
    - 仅在原始 N 点上做一次 FPS（得到 stage0 的 groups 点，保持与 vanilla stage0 一致）
    - 后续 stage 直接取 stage0 FPS 输出序列的前缀：等价于在“上一层输出的点集（且保持 FPS 序列顺序）”上取前 G 个点

    重要：这不是严格等价于 cascade FPS（在子集上重新跑 FPS），因此可能影响精度/表现；
    但可显著减少后续层 FPS kernel 的调用次数。
    """

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        x = self.embedding(x)

        # Stage0: exact FPS on original xyz
        g0 = int(self._groups_list[0])
        fps0 = pointnet2_utils.furthest_point_sample(xyz.contiguous().float(), g0).long()

        for i in range(self.stages):
            if i == 0:
                fps_idx = fps0  # indices in current xyz (original N)
            else:
                # current xyz has length == self._groups_list[i-1]
                gi = int(self._groups_list[i])
                fps_idx = torch.arange(gi, device=xyz.device, dtype=torch.long).unsqueeze(0).expand(xyz.shape[0], -1)

            xyz, x_stage, center = self.GAM_list[i](xyz, x.permute(0, 2, 1), fps_idx=fps_idx)
            x = self.LFP_blocks_list[i](x_stage, center)
            x = self.GFP_blocks_list[i](x)

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pointKANEliteFPSOptimized(num_classes=15, **kwargs) -> ModelFPSOptimized:
    """与 pointKANElite() 相同配置（ScanObjectNN 默认 15 类），用于加载同一 checkpoint 的 FPS 优化推理。"""
    points = int(kwargs.pop("points", 1024))
    return ModelFPSOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs
    )


def pointKANEliteFPSPrefixOptimized(num_classes=15, **kwargs) -> ModelFPSPrefixOptimized:
    """Prefix-approx FPS 优化版：仅 stage0 做一次 FPS，后续 stage 取前缀子集。"""
    points = int(kwargs.pop("points", 1024))
    return ModelFPSPrefixOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs
    )


def pointKANEliteFusedFPSKNNOptimized(num_classes=15, **kwargs) -> ModelFusedFPSKNNOptimized:
    """融合版 FPS+KNN（性能探索）：每层 GAM 内部用融合算法产生 fps_idx+knn_idx。"""
    points = int(kwargs.pop("points", 1024))
    init_threshold = float(kwargs.pop("init_threshold_coarse_fine", 0.15))
    max_iter = int(kwargs.pop("max_iter_coarse_fine", 10))
    min_candidates_ratio = float(kwargs.pop("min_candidates_ratio", 1.1))
    seed = str(kwargs.pop("seed", "first"))
    return ModelFusedFPSKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        init_threshold_coarse_fine=init_threshold,
        max_iter_coarse_fine=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        seed=seed,
        **kwargs
    )


def pointKANElitePointNet2FPSCoarseFineKNNOptimized(num_classes=15, **kwargs) -> ModelPointNet2FPSCoarseFineKNNOptimized:
    """保留 pointnet2 FPS kernel，仅替换 KNN 为 coarse-fine（用于观察 coarse-fine 在完整模型中的收益）。"""
    points = int(kwargs.pop("points", 1024))
    init_threshold = float(kwargs.pop("init_threshold_coarse_fine", 0.15))
    max_iter = int(kwargs.pop("max_iter_coarse_fine", 10))
    min_candidates_ratio = float(kwargs.pop("min_candidates_ratio", 1.1))
    return ModelPointNet2FPSCoarseFineKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        init_threshold_coarse_fine=init_threshold,
        max_iter_coarse_fine=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        **kwargs
    )


def pointKANEliteSpecVanillaOptimized(num_classes=15, **kwargs) -> ModelSpecFPSKNNOptimized:
    """公平对比 baseline：使用 spec_kan_fps 风格 FPS+全量 topk KNN（纯 PyTorch 循环）。"""
    points = int(kwargs.pop("points", 1024))
    seed = str(kwargs.pop("seed", "first"))
    block_size = int(kwargs.pop("block_size", 256))
    return ModelSpecFPSKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        variant="vanilla",
        seed=seed,
        block_size=block_size,
        **kwargs
    )


def pointKANEliteSpecCoarseFineOptimized(num_classes=15, **kwargs) -> ModelSpecFPSKNNOptimized:
    """公平对比 coarse-fine：使用 spec_kan_fps 风格 FPS + coarse-fine KNN（blockwise buffer）。"""
    points = int(kwargs.pop("points", 1024))
    init_threshold = float(kwargs.pop("init_threshold_coarse_fine", 0.1))
    max_iter = int(kwargs.pop("max_iter_coarse_fine", 10))
    min_candidates_ratio = float(kwargs.pop("min_candidates_ratio", 1.2))
    seed = str(kwargs.pop("seed", "first"))
    block_size = int(kwargs.pop("block_size", 256))
    return ModelSpecFPSKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        variant="coarse_fine",
        init_threshold_coarse_fine=init_threshold,
        max_iter_coarse_fine=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        seed=seed,
        block_size=block_size,
        **kwargs
    )


def pointKANEliteCudaCoarseFineOptimized(num_classes=15, **kwargs) -> ModelSpecFPSKNNOptimized:
    """CUDA 扩展版 coarse-fine：fused FPS+KNN，done=count>=min_candidates。"""
    points = int(kwargs.pop("points", 1024))
    init_threshold = float(kwargs.pop("init_threshold_coarse_fine", 0.1))
    max_iter = int(kwargs.pop("max_iter_coarse_fine", 10))
    min_candidates_ratio = float(kwargs.pop("min_candidates_ratio", 1.2))
    seed = str(kwargs.pop("seed", "first"))
    block_size = int(kwargs.pop("block_size", 256))
    mmax_factor = int(kwargs.pop("mmax_factor", 4))
    return ModelSpecFPSKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        variant="cuda_cf",
        init_threshold_coarse_fine=init_threshold,
        max_iter_coarse_fine=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        seed=seed,
        block_size=block_size,
        mmax_factor=mmax_factor,
        **kwargs
    )


def pointKANEliteCudaCoarseFineStrictOptimized(num_classes=15, **kwargs) -> ModelSpecFPSKNNOptimized:
    """严格语义版：与 python blockwise coarse-fine 逻辑对齐（先保证一致性）。"""
    points = int(kwargs.pop("points", 1024))
    init_threshold = float(kwargs.pop("init_threshold_coarse_fine", 0.1))
    max_iter = int(kwargs.pop("max_iter_coarse_fine", 10))
    min_candidates_ratio = float(kwargs.pop("min_candidates_ratio", 1.2))
    seed = str(kwargs.pop("seed", "first"))
    block_size = int(kwargs.pop("block_size", 256))
    mmax_factor = int(kwargs.pop("mmax_factor", 4))
    return ModelSpecFPSKNNOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        variant="cuda_cf",
        init_threshold_coarse_fine=init_threshold,
        max_iter_coarse_fine=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        seed=seed,
        block_size=block_size,
        mmax_factor=mmax_factor,
        strict_semantic=True,
        **kwargs
    )


def pointKANElitePyTorchFPSOptimized(num_classes=15, **kwargs) -> ModelPyTorchFPSOptimized:
    """将 pointnet2 FPS 替换为 PyTorch 版 farthest_point_sample（用于与 fused 版本做公平对比）。"""
    points = int(kwargs.pop("points", 1024))
    return ModelPyTorchFPSOptimized(
        points=points, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
        activation="relu", bias=False, use_xyz=False, normalize="anchor",
        dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
        k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs
    )
