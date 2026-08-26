
"""
tree_fps_knn_accel.py

PyTorch implementation that reproduces the *algorithmic core ideas* of:
  1) Octree-based approximate KNN search
  2) FuseFPS-style bucket-based FPS with sampled-driven KD splitting
  3) FractalCloud-style Fractal partition + block-wise FPS/KNN

This file is intentionally written as a research prototype:
- Focuses on algorithmic behavior and batching interfaces.
- Stays close to the papers' high-level ideas.
- Does NOT reproduce RTL timing / HBM / double buffering / bank conflict hardware.

References embodied in this code:
- FractalCloud: threshold-controlled binary partition, min-max midpoint split,
  cyclic xyz split dimensions, depth-first block layout, block-wise FPS/KNN.
- FuseFPS: bucket-based FPS, arithmetic-mean split, sampled-driven KD-tree
  construction fused with bucket processing, per-bucket farthest-point metadata.
- Octree-based search: recursive 8-way spatial subdivision, leaf-local / nearby
  search candidates for approximate KNN.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Literal, Any

import math
import torch


# =============================================================================
# Utilities
# =============================================================================

def _check_xyz(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected xyz with shape (B, N, 3), got {tuple(x.shape)}")


@torch.no_grad()
def sqdist_points_to_point(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    x: (..., N, 3)
    c: (..., 3)
    return: (..., N)
    """
    return ((x - c.unsqueeze(-2)) ** 2).sum(dim=-1)


@torch.no_grad()
def topk_smallest(values: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    values: (..., N)
    return:
      topk_vals (..., k)
      topk_idx  (..., k)
    """
    k = min(k, values.shape[-1])
    return torch.topk(values, k=k, dim=-1, largest=False, sorted=True)


@torch.no_grad()
def bbox_from_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return points.amin(dim=0), points.amax(dim=0)


@torch.no_grad()
def bbox_center(bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    return 0.5 * (bmin + bmax)


@torch.no_grad()
def point_to_aabb_sqdist(point: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    """
    point: (3,)
    bmin/bmax: (..., 3)
    return: (...)
    """
    delta_low = torch.clamp(bmin - point, min=0.0)
    delta_high = torch.clamp(point - bmax, min=0.0)
    delta = delta_low + delta_high
    return (delta * delta).sum(dim=-1)


@torch.no_grad()
def pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: (M, 3)
    y: (N, 3)
    return: (M, N)
    """
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True).t()
    xy = x @ y.t()
    return xx + yy - 2.0 * xy


@torch.no_grad()
def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C)
    idx: (M,) or (..., M)
    """
    return x[idx]


# =============================================================================
# Exact baselines
# =============================================================================

@torch.no_grad()
def exact_fps(xyz: torch.Tensor,
              K: int,
              seed: Literal["first", "random", "centroid_farthest"] = "first",
              return_min_dist: bool = False):
    """
    Exact FPS baseline.
    xyz: (B, N, 3)
    return:
      idx: (B, K)
      optional min_dist: (B, N)
    """
    _check_xyz(xyz)
    B, N, _ = xyz.shape
    K = min(K, N)
    device = xyz.device

    if seed == "first":
        first = torch.zeros((B,), device=device, dtype=torch.long)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "centroid_farthest":
        c = xyz.mean(dim=1)
        first = sqdist_points_to_point(xyz, c).argmax(dim=1)
    else:
        raise ValueError(seed)

    out = torch.empty((B, K), device=device, dtype=torch.long)
    out[:, 0] = first

    selected = torch.zeros((B, N), device=device, dtype=torch.bool)
    selected.scatter_(1, first[:, None], True)

    center = xyz[torch.arange(B, device=device), first]
    min_dist = sqdist_points_to_point(xyz, center)
    min_dist[selected] = float("-inf")

    for t in range(1, K):
        nxt = min_dist.argmax(dim=1)
        out[:, t] = nxt
        selected.scatter_(1, nxt[:, None], True)

        center = xyz[torch.arange(B, device=device), nxt]
        d = sqdist_points_to_point(xyz, center)
        min_dist = torch.minimum(min_dist, d)
        min_dist[selected] = float("-inf")

    if return_min_dist:
        return out, min_dist
    return out


@torch.no_grad()
def exact_knn(query: torch.Tensor, ref: torch.Tensor, k: int):
    """
    query: (B, M, 3)
    ref:   (B, N, 3)
    return:
      idx:  (B, M, k)
      dist: (B, M, k)
    """
    if query.ndim != 3 or ref.ndim != 3 or query.shape[0] != ref.shape[0] or query.shape[-1] != 3 or ref.shape[-1] != 3:
        raise ValueError("query/ref must be (B,M,3)/(B,N,3)")
    B, M, _ = query.shape
    out_idx = []
    out_dist = []
    for b in range(B):
        d = pairwise_sqdist(query[b], ref[b])
        dist, idx = topk_smallest(d, k)
        out_idx.append(idx)
        out_dist.append(dist)
    return torch.stack(out_idx, dim=0), torch.stack(out_dist, dim=0)


# =============================================================================
# Octree core
# =============================================================================

@dataclass
class OctreeLeaf:
    point_ids: torch.Tensor
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    depth: int
    trace: int


@dataclass
class OctreeMeta:
    leaves: List[OctreeLeaf]
    depth_to_leaf_ids: Dict[int, List[int]]
    root_bbox_min: torch.Tensor
    root_bbox_max: torch.Tensor


@torch.no_grad()
def build_octree_single(points: torch.Tensor,
                        leaf_size: int = 128,
                        max_depth: int = 10) -> OctreeMeta:
    """
    Approximate octree representation aligned with octree-accelerator spirit:
    - recursive 8-way subdivision
    - trace-coded DFS organization
    """
    N = points.shape[0]
    ids = torch.arange(N, device=points.device, dtype=torch.long)
    root_bmin, root_bmax = bbox_from_points(points)

    stack: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = [
        (ids, root_bmin, root_bmax, 0, 0)  # ids, bmin, bmax, depth, trace
    ]

    leaves: List[OctreeLeaf] = []
    depth_to_leaf_ids: Dict[int, List[int]] = {}

    while stack:
        cur_ids, bmin, bmax, depth, trace = stack.pop()
        if cur_ids.numel() <= leaf_size or depth >= max_depth:
            leaf_id = len(leaves)
            leaves.append(OctreeLeaf(cur_ids, bmin.clone(), bmax.clone(), depth, trace))
            depth_to_leaf_ids.setdefault(depth, []).append(leaf_id)
            continue

        pts = points[cur_ids]
        center = bbox_center(bmin, bmax)
        code = (
            ((pts[:, 0] >= center[0]).long() << 2) |
            ((pts[:, 1] >= center[1]).long() << 1) |
            ((pts[:, 2] >= center[2]).long() << 0)
        )

        # DFS order, push high->low to visit low first
        for octant in range(7, -1, -1):
            mask = code == octant
            if mask.sum() == 0:
                continue
            child_ids = cur_ids[mask]

            child_min = bmin.clone()
            child_max = bmax.clone()
            if (octant >> 2) & 1:
                child_min[0] = center[0]
            else:
                child_max[0] = center[0]
            if (octant >> 1) & 1:
                child_min[1] = center[1]
            else:
                child_max[1] = center[1]
            if octant & 1:
                child_min[2] = center[2]
            else:
                child_max[2] = center[2]

            child_trace = (trace << 3) | octant
            stack.append((child_ids, child_min, child_max, depth + 1, child_trace))

    return OctreeMeta(
        leaves=leaves,
        depth_to_leaf_ids=depth_to_leaf_ids,
        root_bbox_min=root_bmin,
        root_bbox_max=root_bmax,
    )


@torch.no_grad()
def octree_leaf_assignment(query_pts: torch.Tensor, meta: OctreeMeta) -> torch.Tensor:
    """
    Assign each query to its containing leaf via bbox containment test.
    Falls back to nearest-center only for the (rare) out-of-root-bbox case.
    query_pts: (M, 3)
    return: leaf ids (M,)

    Fix: original nearest-center approach assigned 3/10 test points to the
    wrong (non-containing) leaf.  Proper assignment checks bbox containment.
    """
    M = query_pts.shape[0]
    result = torch.full((M,), -1, dtype=torch.long, device=query_pts.device)
    for li, leaf in enumerate(meta.leaves):
        inside = ((query_pts >= leaf.bbox_min.unsqueeze(0)) &
                  (query_pts <= leaf.bbox_max.unsqueeze(0))).all(dim=1)
        # Only assign if not yet assigned (DFS order: first match wins)
        unassigned = result == -1
        result[inside & unassigned] = li
    # Fallback for any point outside root bbox (should be empty in normal use)
    still_missing = result == -1
    if still_missing.any():
        centers = torch.stack([bbox_center(l.bbox_min, l.bbox_max) for l in meta.leaves], dim=0)
        d = pairwise_sqdist(query_pts[still_missing], centers)
        result[still_missing] = d.argmin(dim=1)
    return result


@torch.no_grad()
def octree_approx_knn_single(query: torch.Tensor,
                             ref: torch.Tensor,
                             k: int,
                             leaf_size: int = 128,
                             max_depth: int = 10,
                             expand_mode: Literal["same_leaf", "neighbor_bbox", "parent_like"] = "neighbor_bbox",
                             neighbor_leaves: int = 4,
                             return_debug: bool = False):
    """
    Approximate octree-based KNN:
      1) build octree on ref
      2) localize each query to a leaf
      3) search inside same leaf or nearby leaves

    This follows the core octree acceleration idea: shrink the candidate set
    via hierarchical 8-way subdivision instead of exhaustive search.
    """
    meta = build_octree_single(ref, leaf_size=leaf_size, max_depth=max_depth)
    leaf_ids = octree_leaf_assignment(query, meta)

    leaf_centers = torch.stack([bbox_center(l.bbox_min, l.bbox_max) for l in meta.leaves], dim=0)
    out_idx = []
    out_dist = []
    candidate_sizes = []

    for qi, q in enumerate(query):
        lid = int(leaf_ids[qi].item())

        if expand_mode == "same_leaf":
            cand_leaf_ids = torch.tensor([lid], device=query.device, dtype=torch.long)
        elif expand_mode == "neighbor_bbox":
            d_leaf = ((leaf_centers - q.unsqueeze(0)) ** 2).sum(dim=1)
            cand_leaf_ids = torch.topk(d_leaf, k=min(neighbor_leaves, len(meta.leaves)), largest=False).indices
        elif expand_mode == "parent_like":
            # software analogue of "expand local scope": include leaves with bbox
            # lower-bound among top few.
            bmins = torch.stack([l.bbox_min for l in meta.leaves], dim=0)
            bmaxs = torch.stack([l.bbox_max for l in meta.leaves], dim=0)
            lb = point_to_aabb_sqdist(q, bmins, bmaxs)
            cand_leaf_ids = torch.topk(lb, k=min(neighbor_leaves, len(meta.leaves)), largest=False).indices
        else:
            raise ValueError(expand_mode)

        cand_ids = torch.cat([meta.leaves[int(i.item())].point_ids for i in cand_leaf_ids], dim=0)
        cand_ids = torch.unique(cand_ids, sorted=False)
        candidate_sizes.append(int(cand_ids.numel()))

        cand_pts = ref[cand_ids]
        d = ((cand_pts - q.unsqueeze(0)) ** 2).sum(dim=1)
        dist, loc = topk_smallest(d.unsqueeze(0), k)
        dist = dist.squeeze(0)
        loc = loc.squeeze(0)
        out_idx.append(cand_ids[loc])
        out_dist.append(dist)

    out_idx_t = torch.stack(out_idx, dim=0)
    out_dist_t = torch.stack(out_dist, dim=0)

    if return_debug:
        dbg = {
            "num_leaves": len(meta.leaves),
            "avg_candidate_size": float(sum(candidate_sizes) / max(len(candidate_sizes), 1)),
            "candidate_sizes": candidate_sizes,
        }
        return out_idx_t, out_dist_t, dbg
    return out_idx_t, out_dist_t


@torch.no_grad()
def octree_approx_knn(query: torch.Tensor,
                      ref: torch.Tensor,
                      k: int,
                      leaf_size: int = 128,
                      max_depth: int = 10,
                      expand_mode: Literal["same_leaf", "neighbor_bbox", "parent_like"] = "neighbor_bbox",
                      neighbor_leaves: int = 4,
                      return_debug: bool = False):
    """
    Batched wrapper.
    """
    if query.ndim != 3 or ref.ndim != 3:
        raise ValueError
    B = query.shape[0]
    all_idx, all_dist, all_dbg = [], [], []
    for b in range(B):
        out = octree_approx_knn_single(
            query[b], ref[b], k,
            leaf_size=leaf_size,
            max_depth=max_depth,
            expand_mode=expand_mode,
            neighbor_leaves=neighbor_leaves,
            return_debug=return_debug,
        )
        if return_debug:
            idx, dist, dbg = out
            all_dbg.append(dbg)
        else:
            idx, dist = out
        all_idx.append(idx)
        all_dist.append(dist)
    if return_debug:
        return torch.stack(all_idx, 0), torch.stack(all_dist, 0), all_dbg
    return torch.stack(all_idx, 0), torch.stack(all_dist, 0)


# =============================================================================
# Fractal core (FractalCloud-style)
# =============================================================================

@dataclass
class FractalLeaf:
    point_ids: torch.Tensor
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    depth: int
    dft_index: int
    parent_dft_index: Optional[int]


@dataclass
class FractalMeta:
    leaves: List[FractalLeaf]
    root_bbox_min: torch.Tensor
    root_bbox_max: torch.Tensor


@torch.no_grad()
def build_fractal_single(points: torch.Tensor,
                         threshold: int = 256,
                         max_depth: int = 64,
                         start_dim: int = 0) -> FractalMeta:
    """
    Fractal partition:
      - threshold-controlled block division
      - binary tree
      - split = (min + max)/2 of current dimension
      - cyclic x/y/z dimensions
      - DFT memory order
    """
    N = points.shape[0]
    ids = torch.arange(N, device=points.device, dtype=torch.long)
    root_bmin, root_bmax = bbox_from_points(points)

    leaves: List[FractalLeaf] = []

    def _rec(cur_ids: torch.Tensor,
             d: int,
             depth: int,
             parent_dft_index: Optional[int]) -> None:
        cur_pts = points[cur_ids]
        bmin, bmax = bbox_from_points(cur_pts)

        if cur_ids.numel() <= threshold or depth >= max_depth:
            dft_idx = len(leaves)
            leaves.append(FractalLeaf(cur_ids, bmin, bmax, depth, dft_idx, parent_dft_index))
            return

        dim = d % 3
        cur_min = cur_pts[:, dim].amin()
        cur_max = cur_pts[:, dim].amax()
        mid = 0.5 * (cur_min + cur_max)

        left_mask = cur_pts[:, dim] < mid
        right_mask = ~left_mask

        # fallback, still preserve cyclic / midpoint spirit when possible
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            found = False
            for shift in [1, 2]:
                dim_try = (dim + shift) % 3
                cur_min_t = cur_pts[:, dim_try].amin()
                cur_max_t = cur_pts[:, dim_try].amax()
                mid_t = 0.5 * (cur_min_t + cur_max_t)
                lm = cur_pts[:, dim_try] < mid_t
                rm = ~lm
                if lm.sum() > 0 and rm.sum() > 0:
                    dim = dim_try
                    left_mask, right_mask = lm, rm
                    found = True
                    break
            if not found and (left_mask.sum() == 0 or right_mask.sum() == 0):
                order = torch.argsort(cur_pts[:, dim])
                half = max(order.numel() // 2, 1)
                left_ids = cur_ids[order[:half]]
                right_ids = cur_ids[order[half:]]
            else:
                left_ids = cur_ids[left_mask]
                right_ids = cur_ids[right_mask]
        else:
            left_ids = cur_ids[left_mask]
            right_ids = cur_ids[right_mask]

        this_parent = len(leaves) + 10_000_000 + depth  # virtual internal-node tag
        _rec(left_ids, d + 1, depth + 1, this_parent)
        _rec(right_ids, d + 1, depth + 1, this_parent)

    _rec(ids, start_dim, 0, None)
    return FractalMeta(leaves=leaves, root_bbox_min=root_bmin, root_bbox_max=root_bmax)


@torch.no_grad()
def fractal_blockwise_fps_single(points: torch.Tensor,
                                 K: int,
                                 threshold: int = 256,
                                 per_block_rate: Optional[float] = None,
                                 min_per_block: int = 1,
                                 seed: Literal["first", "centroid_farthest"] = "first",
                                 return_debug: bool = False):
    """
    FractalCloud-inspired block-wise FPS:
      - partition by Fractal
      - run local FPS inside each leaf block
      - aggregate results

    The paper's core idea is block-local FPS over Fractal leaves rather than
    global exhaustive search. It is approximate by design.
    """
    meta = build_fractal_single(points, threshold=threshold)
    num_leaves = len(meta.leaves)

    if per_block_rate is None:
        # fixed per-block rate in spirit of FractalCloud
        per_block_rate = K / max(points.shape[0], 1)

    sampled_all = []
    per_leaf_take = []

    for leaf in meta.leaves:
        n = leaf.point_ids.numel()
        take = max(min_per_block, int(round(n * per_block_rate)))
        take = min(take, n)
        per_leaf_take.append(take)

        sub_pts = points[leaf.point_ids].unsqueeze(0)
        local_idx = exact_fps(sub_pts, take, seed=seed)[0]
        sampled_all.append(leaf.point_ids[local_idx])

    sampled = torch.cat(sampled_all, dim=0)

    # If too many / too few, refine on the concatenated candidate set
    if sampled.numel() >= K:
        cand_pts = points[sampled].unsqueeze(0)
        final_local = exact_fps(cand_pts, K, seed=seed)[0]
        final_idx = sampled[final_local]
    else:
        # fallback: fill from exact FPS on full set for remaining quota
        remain = K - sampled.numel()
        global_idx = exact_fps(points.unsqueeze(0), min(remain, points.shape[0]), seed=seed)[0]
        merged = torch.unique(torch.cat([sampled, global_idx], dim=0), sorted=False)
        if merged.numel() > K:
            merged_pts = points[merged].unsqueeze(0)
            refine = exact_fps(merged_pts, K, seed=seed)[0]
            final_idx = merged[refine]
        else:
            final_idx = merged

    # Pad with global FPS fallback if still short (prevents stack size mismatch in batched wrapper)
    if final_idx.numel() < K:
        extra = exact_fps(points.unsqueeze(0), K, seed=seed)[0]
        merged2 = torch.unique(torch.cat([final_idx, extra], dim=0), sorted=False)
        final_idx = merged2[:K]

    final_idx = final_idx[:K]

    if return_debug:
        dbg = {
            "num_leaves": num_leaves,
            "avg_leaf_size": float(sum(l.point_ids.numel() for l in meta.leaves) / max(num_leaves, 1)),
            "per_leaf_take": per_leaf_take,
        }
        return final_idx, dbg
    return final_idx


@torch.no_grad()
def fractal_blockwise_knn_single(query: torch.Tensor,
                                 ref: torch.Tensor,
                                 k: int,
                                 threshold: int = 256,
                                 search_expand: Literal["same_leaf", "parent_bbox", "neighbor_bbox"] = "parent_bbox",
                                 neighbor_blocks: int = 2,
                                 return_debug: bool = False):
    """
    FractalCloud-inspired block-wise KNN:
      - queries are assigned to DFT-organized leaves
      - search space is local leaf / parent-expanded neighborhood
    """
    meta = build_fractal_single(ref, threshold=threshold)

    leaf_centers = torch.stack([bbox_center(l.bbox_min, l.bbox_max) for l in meta.leaves], dim=0)
    assign = pairwise_sqdist(query, leaf_centers).argmin(dim=1)

    out_idx, out_dist = [], []
    candidate_sizes = []

    all_bmins = torch.stack([l.bbox_min for l in meta.leaves], dim=0)
    all_bmaxs = torch.stack([l.bbox_max for l in meta.leaves], dim=0)

    for qi, q in enumerate(query):
        lid = int(assign[qi].item())
        leaf = meta.leaves[lid]

        if search_expand == "same_leaf":
            cand_leaf_ids = torch.tensor([lid], device=query.device)
        elif search_expand == "neighbor_bbox":
            d = pairwise_sqdist(q.unsqueeze(0), leaf_centers).squeeze(0)
            cand_leaf_ids = torch.topk(d, k=min(neighbor_blocks, len(meta.leaves)), largest=False).indices
        elif search_expand == "parent_bbox":
            lb = point_to_aabb_sqdist(q, all_bmins, all_bmaxs)
            cand_leaf_ids = torch.topk(lb, k=min(neighbor_blocks, len(meta.leaves)), largest=False).indices
        else:
            raise ValueError(search_expand)

        cand_ids = torch.cat([meta.leaves[int(i.item())].point_ids for i in cand_leaf_ids], dim=0)
        cand_ids = torch.unique(cand_ids, sorted=False)
        candidate_sizes.append(int(cand_ids.numel()))

        cand_pts = ref[cand_ids]
        d = ((cand_pts - q.unsqueeze(0)) ** 2).sum(dim=1)
        dist, loc = topk_smallest(d.unsqueeze(0), k)
        out_idx.append(cand_ids[loc.squeeze(0)])
        out_dist.append(dist.squeeze(0))

    out_idx_t = torch.stack(out_idx, dim=0)
    out_dist_t = torch.stack(out_dist, dim=0)

    if return_debug:
        dbg = {
            "num_leaves": len(meta.leaves),
            "avg_candidate_size": float(sum(candidate_sizes) / max(len(candidate_sizes), 1)),
            "candidate_sizes": candidate_sizes,
        }
        return out_idx_t, out_dist_t, dbg
    return out_idx_t, out_dist_t


@torch.no_grad()
def fractal_blockwise_fps(xyz: torch.Tensor,
                          K: int,
                          threshold: int = 256,
                          per_block_rate: Optional[float] = None,
                          min_per_block: int = 1,
                          seed: Literal["first", "centroid_farthest"] = "first",
                          return_debug: bool = False):
    _check_xyz(xyz)
    all_idx, all_dbg = [], []
    for b in range(xyz.shape[0]):
        out = fractal_blockwise_fps_single(
            xyz[b], K,
            threshold=threshold,
            per_block_rate=per_block_rate,
            min_per_block=min_per_block,
            seed=seed,
            return_debug=return_debug,
        )
        if return_debug:
            idx, dbg = out
            all_dbg.append(dbg)
        else:
            idx = out
        all_idx.append(idx)
    idx = torch.stack(all_idx, dim=0)
    if return_debug:
        return idx, all_dbg
    return idx


@torch.no_grad()
def fractal_blockwise_knn(query: torch.Tensor,
                          ref: torch.Tensor,
                          k: int,
                          threshold: int = 256,
                          search_expand: Literal["same_leaf", "parent_bbox", "neighbor_bbox"] = "parent_bbox",
                          neighbor_blocks: int = 2,
                          return_debug: bool = False):
    B = query.shape[0]
    all_idx, all_dist, all_dbg = [], [], []
    for b in range(B):
        out = fractal_blockwise_knn_single(
            query[b], ref[b], k,
            threshold=threshold,
            search_expand=search_expand,
            neighbor_blocks=neighbor_blocks,
            return_debug=return_debug,
        )
        if return_debug:
            idx, dist, dbg = out
            all_dbg.append(dbg)
        else:
            idx, dist = out
        all_idx.append(idx)
        all_dist.append(dist)
    if return_debug:
        return torch.stack(all_idx, 0), torch.stack(all_dist, 0), all_dbg
    return torch.stack(all_idx, 0), torch.stack(all_dist, 0)


# =============================================================================
# FuseFPS core (sampled-driven KD construction + bucket FPS)
# =============================================================================

@dataclass
class FuseBucket:
    point_ids: torch.Tensor
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    split_dim: Optional[int] = None
    split_value: Optional[torch.Tensor] = None
    far_point: Optional[int] = None
    far_point_dist: float = -float("inf")
    coord_sum: Optional[torch.Tensor] = None
    ref_buffer: List[int] = field(default_factory=list)
    depth: int = 0


@torch.no_grad()
def _choose_split_dim_from_bbox(bmin: torch.Tensor, bmax: torch.Tensor) -> int:
    return int(torch.argmax(bmax - bmin).item())


@torch.no_grad()
def _bucket_process_and_split(points: torch.Tensor,
                              min_dist: torch.Tensor,
                              bucket: FuseBucket,
                              ref_points: torch.Tensor,
                              max_ref_buffer: int = 4):
    """
    FuseFPS-style one-pass bucket processing:
      - choose split dimension by largest bbox range
      - use arithmetic mean split
      - while streaming bucket points, update farPoint metadata
      - simultaneously partition to two sub-buckets
    """
    ids = bucket.point_ids
    pts = points[ids]
    bmin, bmax = bbox_from_points(pts)
    split_dim = _choose_split_dim_from_bbox(bmin, bmax)
    split_value = pts[:, split_dim].mean()

    # "fused" point processing: update farthest metadata while partitioning
    local_md = min_dist[ids]
    local_best = torch.argmax(local_md)
    far_pid = int(ids[local_best].item())
    far_pdist = float(local_md[local_best].item())

    left_mask = pts[:, split_dim] < split_value
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        order = torch.argsort(pts[:, split_dim])
        half = max(order.numel() // 2, 1)
        left_ids = ids[order[:half]]
        right_ids = ids[order[half:]]
    else:
        left_ids = ids[left_mask]
        right_ids = ids[right_mask]

    left_pts = points[left_ids]
    right_pts = points[right_ids]

    left_bucket = FuseBucket(
        point_ids=left_ids,
        bbox_min=left_pts.amin(dim=0),
        bbox_max=left_pts.amax(dim=0),
        coord_sum=left_pts.sum(dim=0),
        depth=bucket.depth + 1,
    )
    right_bucket = FuseBucket(
        point_ids=right_ids,
        bbox_min=right_pts.amin(dim=0),
        bbox_max=right_pts.amax(dim=0),
        coord_sum=right_pts.sum(dim=0),
        depth=bucket.depth + 1,
    )

    bucket.split_dim = split_dim
    bucket.split_value = split_value
    bucket.far_point = far_pid
    bucket.far_point_dist = far_pdist
    bucket.coord_sum = pts.sum(dim=0)

    # reference buffer: keep recent sampled points (IDs)
    bucket.ref_buffer = bucket.ref_buffer[-max_ref_buffer:]

    return bucket, left_bucket, right_bucket


@torch.no_grad()
def _recompute_bucket_far(points: torch.Tensor, min_dist: torch.Tensor, bucket: FuseBucket) -> FuseBucket:
    ids = bucket.point_ids
    if ids.numel() == 0:
        bucket.far_point = None
        bucket.far_point_dist = -float("inf")
        return bucket
    local_md = min_dist[ids]
    loc = torch.argmax(local_md)
    bucket.far_point = int(ids[loc].item())
    bucket.far_point_dist = float(local_md[loc].item())
    return bucket


@torch.no_grad()
def _fusefps_should_process_bucket(points: torch.Tensor,
                                   sampled_point: torch.Tensor,
                                   bucket: FuseBucket) -> bool:
    """
    QuickFPS/FuseFPS spirit: process only necessary buckets.
    Use a geometric lower bound test against bucket bbox.
    """
    lb = point_to_aabb_sqdist(sampled_point, bucket.bbox_min.unsqueeze(0), bucket.bbox_max.unsqueeze(0)).item()
    return lb < bucket.far_point_dist


@torch.no_grad()
def fusefps_single(points: torch.Tensor,
                   K: int,
                   leaf_size: int = 256,
                   max_ref_buffer: int = 4,
                   seed: Literal["first", "centroid_farthest"] = "first",
                   return_debug: bool = False):
    """
    FuseFPS-inspired approximate FPS.

    Core ideas preserved:
      - bucket-based FPS
      - sampled-driven KD construction fused with point sampling
      - arithmetic-mean split instead of median sort
      - per-bucket farPoint/farPointDist metadata
      - process only necessary buckets based on pruning

    Approximate, not exact-FPS.
    """
    N = points.shape[0]
    K = min(K, N)

    if seed == "first":
        first = 0
    elif seed == "centroid_farthest":
        c = points.mean(dim=0)
        first = int(torch.argmax(((points - c.unsqueeze(0)) ** 2).sum(dim=1)).item())
    else:
        raise ValueError(seed)

    sampled = [first]
    selected = torch.zeros((N,), device=points.device, dtype=torch.bool)
    selected[first] = True

    min_dist = ((points - points[first].unsqueeze(0)) ** 2).sum(dim=1)
    min_dist[selected] = float("-inf")

    root_bmin, root_bmax = bbox_from_points(points)
    root = FuseBucket(
        point_ids=torch.arange(N, device=points.device, dtype=torch.long),
        bbox_min=root_bmin,
        bbox_max=root_bmax,
        coord_sum=points.sum(dim=0),
        depth=0,
    )
    root = _recompute_bucket_far(points, min_dist, root)
    buckets: List[FuseBucket] = [root]

    dbg = {"bucket_counts": [], "processed_buckets_per_iter": []}

    for _ in range(1, K):
        # choose bucket with largest far_point_dist
        far_dists = torch.tensor([b.far_point_dist for b in buckets], device=points.device)
        bid = int(torch.argmax(far_dists).item())
        cur_bucket = buckets[bid]

        if cur_bucket.far_point is None:
            nxt = int(torch.argmax(min_dist).item())
        else:
            nxt = cur_bucket.far_point

        sampled.append(nxt)
        selected[nxt] = True

        sampled_pt = points[nxt]
        d_new = ((points - sampled_pt.unsqueeze(0)) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, d_new)
        min_dist[selected] = float("-inf")

        # sampled-driven construction:
        # if current bucket still large, split it *now* while processing it.
        if cur_bucket.point_ids.numel() > leaf_size:
            cur_bucket.ref_buffer.append(nxt)
            _, left_b, right_b = _bucket_process_and_split(
                points, min_dist, cur_bucket, points[torch.tensor(sampled, device=points.device)],
                max_ref_buffer=max_ref_buffer,
            )
            left_b.ref_buffer = cur_bucket.ref_buffer.copy()
            right_b.ref_buffer = cur_bucket.ref_buffer.copy()

            left_b = _recompute_bucket_far(points, min_dist, left_b)
            right_b = _recompute_bucket_far(points, min_dist, right_b)

            buckets.pop(bid)
            buckets.append(left_b)
            buckets.append(right_b)
        else:
            buckets[bid] = _recompute_bucket_far(points, min_dist, cur_bucket)

        # prune/recompute only necessary buckets
        proc_cnt = 0
        for i in range(len(buckets)):
            if _fusefps_should_process_bucket(points, sampled_pt, buckets[i]):
                buckets[i] = _recompute_bucket_far(points, min_dist, buckets[i])
                proc_cnt += 1

        dbg["bucket_counts"].append(len(buckets))
        dbg["processed_buckets_per_iter"].append(proc_cnt)

    out = torch.tensor(sampled, device=points.device, dtype=torch.long)
    if return_debug:
        dbg["final_bucket_count"] = len(buckets)
        return out, dbg
    return out


@torch.no_grad()
def fusefps(xyz: torch.Tensor,
            K: int,
            leaf_size: int = 256,
            max_ref_buffer: int = 4,
            seed: Literal["first", "centroid_farthest"] = "first",
            return_debug: bool = False):
    _check_xyz(xyz)
    all_idx, all_dbg = [], []
    for b in range(xyz.shape[0]):
        out = fusefps_single(
            xyz[b], K,
            leaf_size=leaf_size,
            max_ref_buffer=max_ref_buffer,
            seed=seed,
            return_debug=return_debug,
        )
        if return_debug:
            idx, dbg = out
            all_dbg.append(dbg)
        else:
            idx = out
        all_idx.append(idx)
    idx = torch.stack(all_idx, dim=0)
    if return_debug:
        return idx, all_dbg
    return idx


# =============================================================================
# Unified façade
# =============================================================================

class PointCloudAccelOps:
    """
    Unified front-end for algorithm comparison.
    """

    @staticmethod
    @torch.no_grad()
    def fps(xyz: torch.Tensor,
            K: int,
            method: Literal["exact", "fusefps", "fractal"] = "exact",
            **kwargs):
        if method == "exact":
            return exact_fps(xyz, K, **kwargs)
        elif method == "fusefps":
            return fusefps(xyz, K, **kwargs)
        elif method == "fractal":
            return fractal_blockwise_fps(xyz, K, **kwargs)
        raise ValueError(method)

    @staticmethod
    @torch.no_grad()
    def knn(query: torch.Tensor,
            ref: torch.Tensor,
            k: int,
            method: Literal["exact", "octree", "fractal"] = "exact",
            **kwargs):
        if method == "exact":
            return exact_knn(query, ref, k)
        elif method == "octree":
            return octree_approx_knn(query, ref, k, **kwargs)
        elif method == "fractal":
            return fractal_blockwise_knn(query, ref, k, **kwargs)
        raise ValueError(method)


# =============================================================================
# Evaluation helpers
# =============================================================================

@torch.no_grad()
def fps_overlap_ratio(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """
    idx_a / idx_b: (K,)
    Returns Jaccard IoU = |A∩B| / |A∪B|, NOT recall.
    (Previous formula `|A∩B|/|A|` was recall and overestimates quality.)
    """
    sa = set(idx_a.tolist())
    sb = set(idx_b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


@torch.no_grad()
def knn_recall_at_k(ref_idx: torch.Tensor, approx_idx: torch.Tensor) -> float:
    """
    ref_idx, approx_idx: (M, k)
    """
    M = ref_idx.shape[0]
    total = 0.0
    for i in range(M):
        a = set(ref_idx[i].tolist())
        b = set(approx_idx[i].tolist())
        total += len(a & b) / max(len(a), 1)
    return total / max(M, 1)


@torch.no_grad()
def benchmark_demo(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    B, N, M, K, k = 2, 4096, 512, 256, 16
    xyz = torch.randn(B, N, 3, device=device)

    # FPS comparison
    idx_exact = exact_fps(xyz, K)
    idx_fuse, dbg_fuse = fusefps(xyz, K, return_debug=True)
    idx_frac, dbg_frac = fractal_blockwise_fps(xyz, K, threshold=256, return_debug=True)

    fps_overlap_fuse = sum(fps_overlap_ratio(idx_exact[b], idx_fuse[b]) for b in range(B)) / B
    fps_overlap_frac = sum(fps_overlap_ratio(idx_exact[b], idx_frac[b]) for b in range(B)) / B

    query = xyz[:, :M]
    knn_exact_idx, _ = exact_knn(query, xyz, k)
    knn_oct_idx, _, dbg_oct = octree_approx_knn(query, xyz, k, return_debug=True)
    knn_frac_idx, _, dbg_fk = fractal_blockwise_knn(query, xyz, k, return_debug=True)

    knn_recall_oct = sum(knn_recall_at_k(knn_exact_idx[b], knn_oct_idx[b]) for b in range(B)) / B
    knn_recall_frac = sum(knn_recall_at_k(knn_exact_idx[b], knn_frac_idx[b]) for b in range(B)) / B

    return {
        "fps_overlap_fusefps": fps_overlap_fuse,
        "fps_overlap_fractal": fps_overlap_frac,
        "knn_recall_octree": knn_recall_oct,
        "knn_recall_fractal": knn_recall_frac,
        "dbg_fuse": dbg_fuse,
        "dbg_frac": dbg_frac,
        "dbg_oct": dbg_oct,
        "dbg_fractal_knn": dbg_fk,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = benchmark_demo(device=device)
    print("device =", device)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {type(v).__name__}")
