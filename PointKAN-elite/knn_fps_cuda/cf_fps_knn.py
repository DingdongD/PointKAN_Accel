from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=1)
def _load_ext():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_cpp = os.path.join(this_dir, "cf_fps_knn_ext.cpp")
    src_cu = os.path.join(this_dir, "cf_fps_knn_ext.cu")
    return load(
        name="pointkan_cf_fps_knn_ext",
        sources=[src_cpp, src_cu],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


@torch.no_grad()
def fps_knn_coarse_fine_cuda(
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
    CUDA fused FPS + coarse-fine KNN.

    Inputs:
      X: [B, N, 3] float32 CUDA contiguous
    Returns:
      idx:     [B, K] int64
      knn_idx: [B, K, k] int64
    """
    if X.device.type != "cuda":
        raise ValueError("X must be CUDA tensor")
    if X.dtype != torch.float32:
        raise ValueError("X must be float32 (for CUDA kernel)")
    if X.dim() != 3 or X.size(-1) != 3:
        raise ValueError("X must be [B, N, 3]")

    # Strict semantic mode: 1:1 align with python blockwise reference.
    if strict_semantic:
        from classification_ModelNet40.spec_kan_fps import fps_with_knn_coarse_fine_fused_blockwise
        idx, _r, knn_idx, _d, _kr, _th = fps_with_knn_coarse_fine_fused_blockwise(
            X.contiguous(),
            int(K),
            k_neighbors=int(k_neighbors),
            init_threshold=float(init_threshold),
            max_iter=int(max_iter),
            min_candidates_ratio=float(min_candidates_ratio),
            seed=str(seed),
            return_distances_matrix=False,
            exclude_self=bool(exclude_self),
            block_size=int(block_size),
        )
        return idx, knn_idx

    Xc = X.contiguous()
    ext = _load_ext()
    return ext.fps_knn_coarse_fine_forward_cuda(
        Xc,
        int(K),
        int(k_neighbors),
        float(init_threshold),
        int(max_iter),
        float(min_candidates_ratio),
        str(seed),
        bool(exclude_self),
        int(block_size),
        int(mmax_factor),
    )

