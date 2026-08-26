# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M) squared Euclidean distance
    """
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[0] == dst.shape[0]
    assert src.shape[2] == dst.shape[2]
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:
        (B, K)      -> return (B, K, C)
        (B,)        -> return (B, C)
    """
    B, N, C = points.shape
    if idx.ndim == 1:
        assert idx.shape[0] == B
        return points[torch.arange(B, device=points.device), idx]  # (B, C)
    elif idx.ndim == 2:
        B2, K = idx.shape
        assert B2 == B
        batch = torch.arange(B, device=points.device)[:, None].expand(B, K)
        return points[batch, idx]  # (B, K, C)
    else:
        raise ValueError(f"Unsupported idx shape: {idx.shape}")


@torch.no_grad()
def fps_exact(
    X: torch.Tensor,
    K: int,
    seed: str = "first",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准 exact FPS（batch 版，PyTorch 原型）
    X: (B, N, C)
    return:
      idx:   (B, K)
      radii: (B, K)
             radii[:, t] 表示第 t 个被选中的点，在“被选中那一刻”的 min-dist
             对 t=0，定义为其到初始空集合的半径不自然，这里采用：
                 - 若 seed='centroid_farthest' / 'random' / 'first'
                 - 则 radii[:,0] 记为该点被选后，当前集合对全体点的 max min-dist
             主要用于和 vg_fps 的日志口径尽量一致
    """
    assert X.ndim == 3
    B, N, C = X.shape
    device = X.device
    K = min(K, N)

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=X.dtype, device=device)

    if seed == "first":
        farthest = torch.zeros((B,), dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "centroid_farthest":
        centroid = X.mean(dim=1, keepdim=True)                     # (B,1,C)
        d2 = torch.sum((X - centroid) ** 2, dim=-1)               # (B,N)
        farthest = torch.argmax(d2, dim=-1)
    else:
        raise ValueError(f"Unsupported seed mode: {seed}")

    min_dist = torch.full((B, N), float("inf"), dtype=X.dtype, device=device)

    for t in range(K):
        idx[:, t] = farthest

        # 记录“当前被选中点”在被选中瞬间的 min_dist
        if t == 0:
            # 第一个点没有前驱集合，可定义为 0；或定义成更新后的 max(min_dist)
            radii[:, t] = 0.0
        else:
            radii[:, t] = min_dist[torch.arange(B, device=device), farthest]

        chosen = gather_points(X, farthest)                       # (B,C)
        d2 = torch.sum((X - chosen[:, None, :]) ** 2, dim=-1)    # (B,N)
        min_dist = torch.minimum(min_dist, d2)
        farthest = torch.argmax(min_dist, dim=-1)

    return idx, radii

@torch.no_grad()
def build_voronoi_cells(
    X: torch.Tensor,
    M: int,
    coarse_seed: str = "centroid_farthest",
) -> Dict[str, torch.Tensor]:
    """
    用 M 个 coarse seeds 构建静态 Voronoi-like cells

    X: (B, N, C)
    return dict:
      seed_idx:     (B, M)
      seed_xyz:     (B, M, C)
      cell_id:      (B, N)     每个点属于哪个 cell
      cell_radius:  (B, M)     每个 cell 到其 seed 的最大平方半径
      cell_counts:  (B, M)
    """
    assert X.ndim == 3
    B, N, C = X.shape
    device = X.device
    M = min(M, N)

    seed_idx, _ = fps_exact(X, K=M, seed=coarse_seed)
    seed_xyz = gather_points(X, seed_idx)                        # (B,M,C)

    d2_ps = square_distance(X, seed_xyz)                         # (B,N,M)
    cell_id = torch.argmin(d2_ps, dim=-1)                        # (B,N)

    cell_radius = torch.zeros((B, M), dtype=X.dtype, device=device)
    cell_counts = torch.zeros((B, M), dtype=torch.long, device=device)

    # prototype implementation: 按 batch 处理 cell 统计
    for b in range(B):
        cid_b = cell_id[b]                                       # (N,)
        d2_b = d2_ps[b]                                          # (N,M)
        for m in range(M):
            mask = (cid_b == m)
            cnt = int(mask.sum().item())
            cell_counts[b, m] = cnt
            if cnt > 0:
                cell_radius[b, m] = d2_b[mask, m].max()
            else:
                cell_radius[b, m] = torch.tensor(0.0, dtype=X.dtype, device=device)

    return {
        "seed_idx": seed_idx,
        "seed_xyz": seed_xyz,
        "cell_id": cell_id,
        "cell_radius": cell_radius,
        "cell_counts": cell_counts,
    }


@torch.no_grad()
def _init_first_point(
    X: torch.Tensor,
    mode: str = "centroid_farthest",
) -> torch.Tensor:
    """
    X: (1, N, C) or (B, N, C)
    return: (B,)
    """
    B, N, C = X.shape
    device = X.device
    if mode == "first":
        return torch.zeros((B,), dtype=torch.long, device=device)
    elif mode == "random":
        return torch.randint(0, N, (B,), device=device)
    elif mode == "centroid_farthest":
        centroid = X.mean(dim=1, keepdim=True)
        d2 = torch.sum((X - centroid) ** 2, dim=-1)
        return torch.argmax(d2, dim=-1)
    else:
        raise ValueError(f"Unsupported init mode: {mode}")


@torch.no_grad()
def vg_fps(
    X: torch.Tensor,
    K: int,
    M: int = 32,
    init_seed: str = "centroid_farthest",
    coarse_seed: str = "centroid_farthest",
    eps: float = 1e-12,
    return_debug: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Voronoi-Guided FPS (VG-FPS)
    ------------------------------------------------------------
    这是一个“静态 cell + lazy 上界验证”的 FPS 实现。

    核心思想：
      1) 用 M 个 coarse seeds 把点集划为固定 cells
      2) 每个 cell 维护一个 upper bound:
             ub_j >= max_{x in cell j} min_{s in S} ||x-s||^2
      3) 每轮挑 ub 最大的 cell 做精算
      4) 若该 cell 精算值 >= 其它所有 cell 的 ub，则接受该点
         否则继续处理当前新的 ub 最大 cell

    参数:
      X: (B, N, C)
      K: 采样点数
      M: coarse Voronoi cells 数量
      init_seed: 第一个 FPS 点初始化方式
      coarse_seed: 构建 Voronoi cells 的 seed 方式
      eps: 数值容忍
    返回:
      fps_idx:   (B, K)
      fps_radii: (B, K)   每轮被接受点的 min-dist（平方距离）
      debug:     统计信息
    """
    assert X.ndim == 3
    B, N, C = X.shape
    device = X.device
    dtype = X.dtype
    K = min(K, N)
    M = min(M, N)

    # 1) 预构建 Voronoi-like cells
    cell_info = build_voronoi_cells(X, M=M, coarse_seed=coarse_seed)
    seed_xyz = cell_info["seed_xyz"]          # (B,M,C)
    cell_id = cell_info["cell_id"]            # (B,N)
    cell_radius = cell_info["cell_radius"]    # (B,M)

    fps_idx = torch.zeros((B, K), dtype=torch.long, device=device)
    fps_radii = torch.zeros((B, K), dtype=dtype, device=device)

    debug = {
        "num_refresh": torch.zeros((B,), dtype=torch.long, device=device),
        "num_accept": torch.zeros((B,), dtype=torch.long, device=device),
        "num_candidates_evaluated": torch.zeros((B,), dtype=torch.long, device=device),
        "num_bound_updates": torch.zeros((B,), dtype=torch.long, device=device),
    }

    # 按 batch 独立执行，逻辑更清晰，适合研究原型
    for b in range(B):
        xb = X[b]                              # (N,C)
        cid = cell_id[b]                       # (N,)
        sxyz = seed_xyz[b]                     # (M,C)
        crad = cell_radius[b]                  # (M,)

        # 预生成每个 cell 的点索引
        cell_points = []
        nonempty = torch.zeros((M,), dtype=torch.bool, device=device)
        for m in range(M):
            inds = torch.nonzero(cid == m, as_tuple=False).flatten()
            cell_points.append(inds)
            if inds.numel() > 0:
                nonempty[m] = True

        # 2) 初始化第一个点
        first_idx = _init_first_point(xb[None, ...], mode=init_seed)[0]
        fps_idx[b, 0] = first_idx

        chosen = xb[first_idx]
        min_dist = torch.sum((xb - chosen[None, :]) ** 2, dim=-1)
        fps_radii[b, 0] = 0.0

        # 3) 初始化每个 cell 的 upper bound
        # 对任意 x in cell_j:
        #   ||x - p|| <= ||seed_j - p|| + radius_j^(1/2)
        # 用平方距离形式时，写成更稳定一些：
        #   ub_j = max_{x in cell_j} min_dist(x)
        # 初始可以直接精确算，也可以用几何上界。这里采用：
        #   ub_j <= max_{x in cell_j} ||x-p||^2
        #       <= (||seed_j-p|| + r_j)^2
        # 其中 r_j = sqrt(cell_radius_sq)
        seed_d2 = torch.sum((sxyz - chosen[None, :]) ** 2, dim=-1)        # (M,)
        seed_d = torch.sqrt(torch.clamp(seed_d2, min=0.0))
        cell_r = torch.sqrt(torch.clamp(crad, min=0.0))
        ub = (seed_d + cell_r) ** 2                                        # (M,)
        ub = torch.where(nonempty, ub, torch.full_like(ub, -1.0))

        # 可选：每个 cell 当前是否“exact refreshed”
        exact_valid = torch.zeros((M,), dtype=torch.bool, device=device)

        # 4) 迭代采样剩余 K-1 个点
        for t in range(1, K):
            accepted = False

            while not accepted:
                # 取当前 ub 最大的 cell
                cell_j = torch.argmax(ub).item()
                ub_j = ub[cell_j].item()

                if ub_j < 0:
                    # 理论上不会发生，除非 cell 全空
                    fallback = torch.argmax(min_dist).item()
                    fps_idx[b, t] = fallback
                    fps_radii[b, t] = min_dist[fallback]
                    newp = xb[fallback]
                    d2_new = torch.sum((xb - newp[None, :]) ** 2, dim=-1)
                    min_dist = torch.minimum(min_dist, d2_new)
                    accepted = True
                    debug["num_accept"][b] += 1
                    break

                # 对该 cell 做精确刷新
                inds = cell_points[cell_j]
                assert inds.numel() > 0
                local_md = min_dist[inds]
                local_best_pos = torch.argmax(local_md)
                best_idx = inds[local_best_pos]
                best_val = local_md[local_best_pos]

                debug["num_refresh"][b] += 1
                debug["num_candidates_evaluated"][b] += inds.numel()

                # 刷新该 cell 的 ub 为 exact 值
                ub[cell_j] = best_val
                exact_valid[cell_j] = True

                # 判断能否接受：
                # 若该 cell exact best >= 其它所有 cell 的 upper bound，则它就是全局最优
                tmp = ub.clone()
                tmp[cell_j] = -1.0
                other_max = tmp.max()

                if best_val + eps >= other_max:
                    # 接受 best_idx
                    fps_idx[b, t] = best_idx
                    fps_radii[b, t] = best_val

                    newp = xb[best_idx]
                    d2_new = torch.sum((xb - newp[None, :]) ** 2, dim=-1)
                    min_dist = torch.minimum(min_dist, d2_new)

                    # 用新点更新所有 cell upper bound
                    # ub_m <= min(old ub_m, (||seed_m-newp|| + r_m)^2)
                    seed_d2_new = torch.sum((sxyz - newp[None, :]) ** 2, dim=-1)
                    seed_d_new = torch.sqrt(torch.clamp(seed_d2_new, min=0.0))
                    new_geom_ub = (seed_d_new + cell_r) ** 2
                    ub = torch.minimum(ub, new_geom_ub)
                    ub = torch.where(nonempty, ub, torch.full_like(ub, -1.0))

                    # 被几何上界更新之后，原先的 exact_valid 不再保证 exact
                    exact_valid[:] = False

                    accepted = True
                    debug["num_accept"][b] += 1
                    debug["num_bound_updates"][b] += M

        # end for t
    # end for b

    if return_debug:
        debug.update(cell_info)

    return fps_idx, fps_radii, debug


@torch.no_grad()
def compare_vg_fps_vs_exact(
    X: torch.Tensor,
    K: int,
    M: int = 32,
    init_seed: str = "centroid_farthest",
    coarse_seed: str = "centroid_farthest",
) -> Dict[str, torch.Tensor]:
    """
    简单对比 VG-FPS 和 exact FPS
    """
    idx_exact, radii_exact = fps_exact(X, K=K, seed=init_seed)
    idx_vg, radii_vg, dbg = vg_fps(
        X,
        K=K,
        M=M,
        init_seed=init_seed,
        coarse_seed=coarse_seed,
        return_debug=True,
    )

    same_idx_ratio = (idx_exact == idx_vg).float().mean(dim=-1)
    radius_gap = torch.abs(radii_exact - radii_vg).mean(dim=-1)

    return {
        "idx_exact": idx_exact,
        "radii_exact": radii_exact,
        "idx_vg": idx_vg,
        "radii_vg": radii_vg,
        "same_idx_ratio_per_batch": same_idx_ratio,
        "mean_radius_abs_gap_per_batch": radius_gap,
        "num_refresh": dbg["num_refresh"],
        "num_candidates_evaluated": dbg["num_candidates_evaluated"],
        "num_bound_updates": dbg["num_bound_updates"],
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, C = 2, 1024, 3
    K = 256
    M = 32

    X = torch.rand(B, N, C, device=device, dtype=torch.float32)

    idx_vg, radii_vg, dbg = vg_fps(
        X,
        K=K,
        M=M,
        init_seed="centroid_farthest",
        coarse_seed="centroid_farthest",
        return_debug=True,
    )

    print("VG-FPS idx shape:", idx_vg.shape)
    print("VG-FPS radii shape:", radii_vg.shape)
    print("num_refresh:", dbg["num_refresh"])
    print("num_candidates_evaluated:", dbg["num_candidates_evaluated"])

    cmp_res = compare_vg_fps_vs_exact(
        X, K=64, M=M,
        init_seed="centroid_farthest",
        coarse_seed="centroid_farthest",
    )
    print("same_idx_ratio_per_batch:", cmp_res["same_idx_ratio_per_batch"])
    print("mean_radius_abs_gap_per_batch:", cmp_res["mean_radius_abs_gap_per_batch"])