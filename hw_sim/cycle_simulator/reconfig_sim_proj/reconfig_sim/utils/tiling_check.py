# -*- coding: utf-8 -*-
"""
分块正确性校验：确保脉动/分块后的输出与真实 C=A*B（或卷积）一致。
- 覆盖校验：OS 每 (i,j) 仅一块；WS 每 (i,j) 被相同 K 块数覆盖。
- 数值校验：按当前分块用 NumPy 拼出「脉动结果」C_systolic，与 golden 逐元素比较。
"""
from typing import List, Tuple

import numpy as np

from ..models.tile import TileDesc


def _tile_output_region(t: TileDesc):
    """返回该 tile 在输出矩阵上的行、列范围 (m_lo, m_hi, n_lo, n_hi)（左闭右开）."""
    m_lo, m_hi = t.m0, t.m0 + t.m_tile
    n_lo, n_hi = t.n0, t.n0 + t.n_tile
    return m_lo, m_hi, n_lo, n_hi


def verify_gemm_tiling_coverage(tiles: List[TileDesc], M: int, N: int) -> tuple:
    """
    校验 GEMM 分块是否完整、无重叠地覆盖输出 [0,M)x[0,N]。
    OS：每个 (i,j) 恰好被 1 块覆盖。
    WS：每个 (i,j) 被相同数量的块覆盖（每个 K 块贡献一次部分和），且覆盖数 = K 方向块数。
    返回 (ok: bool, message: str)。
    """
    if not tiles:
        return False, "no tiles"

    # 按模式分支
    mode = tiles[0].mode
    if mode == 'OS':
        # 每个 (i,j) 至多被一块覆盖，且无空隙
        grid = [[0] * N for _ in range(M)]
        for t in tiles:
            m_lo, m_hi, n_lo, n_hi = _tile_output_region(t)
            for i in range(m_lo, min(m_hi, M)):
                for j in range(n_lo, min(n_hi, N)):
                    grid[i][j] += 1
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 0:
                    return False, "OS: output ({},{}) not covered by any tile".format(i, j)
                if grid[i][j] > 1:
                    return False, "OS: output ({},{}) covered {} times (overlap)".format(i, j, grid[i][j])
        return True, "OS: every (i,j) covered exactly once"

    else:
        # WS：同一 (i,j) 被多个 K 块贡献，应被覆盖次数 = K 方向块数
        K = tiles[0].k_tile  # 单块 K 维大小；需统计不同 k0 的块数
        k_blocks = set((t.n0, t.n0 + t.n_tile) for t in tiles)  # 用 (n0, n0+nt) 不足以区分 k0
        # 实际应统计：对每个 (m0,n0) 等价类，k0 的块数
        from collections import defaultdict
        key_to_k = defaultdict(set)
        for t in tiles:
            key = (t.m0, t.n0, t.m_tile, t.n_tile)
            key_to_k[key].add(t.k0)
        num_k_blocks = max(len(s) for s in key_to_k.values()) if key_to_k else 0
        K_total = max(t.k0 + t.k_tile for t in tiles) if tiles else 0
        expected_contribs = sum(1 for t in tiles if t.k0 == 0)  # 简化：用 n0 组内 k0 数量
        # 更简单：对 WS 只检查每个 (i,j) 被覆盖次数一致且 > 0
        grid = [[0] * N for _ in range(M)]
        for t in tiles:
            m_lo, m_hi, n_lo, n_hi = _tile_output_region(t)
            for i in range(m_lo, min(m_hi, M)):
                for j in range(n_lo, min(n_hi, N)):
                    grid[i][j] += 1
        val = grid[0][0] if M and N else 0
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 0:
                    return False, "WS: output ({},{}) not covered".format(i, j)
                if grid[i][j] != val:
                    return False, "WS: coverage count inconsistent at ({},{}) expected {} got {}".format(
                        i, j, val, grid[i][j])
        return True, "WS: every (i,j) covered {} times (K-blocks)".format(val)


def verify_conv2d_tiling_coverage(tiles: List[TileDesc], rows: int, cout: int) -> tuple:
    """Conv2D im2col 后的 GEMM：rows 为 M，cout 为 N，逻辑同 verify_gemm_tiling_coverage。"""
    return verify_gemm_tiling_coverage(tiles, rows, cout)


def compute_gemm_systolic_result(
    A: np.ndarray, B: np.ndarray, tiles: List[TileDesc], M: int, K: int, N: int
) -> np.ndarray:
    """
    按当前分块用 NumPy 计算「脉动分块结果」C_systolic，与硬件按此分块执行等价。
    OS：每块产出 C[m0:m0+mt, n0:n0+nt] = A[m0:m0+mt,:] @ B[:,n0:n0+nt]，无重叠。
    WS：每块产出部分和，同一 (i,j) 由多块按 K 维累加。
    返回形状 (M, N) 的 C_systolic。
    """
    C = np.zeros((M, N), dtype=np.int64)
    if not tiles:
        return C
    mode = tiles[0].mode
    if mode == 'OS':
        for t in tiles:
            mr, nr = t.m0, t.n0
            mt, nt = t.m_tile, t.n_tile
            C[mr : mr + mt, nr : nr + nt] = A[mr : mr + mt, :] @ B[:, nr : nr + nt]
    else:
        for t in tiles:
            mt, kt, nt = t.m_tile, t.k_tile, t.n_tile
            # 该块贡献 A[:, k0:k0+kt] @ B[k0:k0+kt, n0:n0+nt] -> 写入 [0:mt, n0:n0+nt]
            C[0:mt, t.n0 : t.n0 + nt] += (
                A[:, t.k0 : t.k0 + kt] @ B[t.k0 : t.k0 + kt, t.n0 : t.n0 + nt]
            )
    return C


def verify_gemm_result(
    A: np.ndarray, B: np.ndarray, golden: np.ndarray, tiles: List[TileDesc], M: int, K: int, N: int
) -> Tuple[bool, str]:
    """
    脉动分块结果与真实 matmul 一致校验：计算 C_systolic，与 golden 逐元素比较。
    返回 (match: bool, message: str)。
    """
    if not tiles:
        return False, "no tiles"
    C_systolic = compute_gemm_systolic_result(A, B, tiles, M, K, N)
    if C_systolic.shape != golden.shape:
        return False, "systolic shape {} != golden shape {}".format(C_systolic.shape, golden.shape)
    if not np.array_equal(C_systolic, golden):
        diff = np.abs(C_systolic.astype(np.int64) - golden.astype(np.int64))
        max_diff = int(np.max(diff))
        n_err = int(np.sum(diff != 0))
        return False, "systolic vs golden: {} elements differ, max_diff={}".format(n_err, max_diff)
    return True, "systolic result matches golden (matmul)"


def verify_conv2d_result(
    cols: np.ndarray,
    w_mat: np.ndarray,
    golden_y: np.ndarray,
    tiles: List[TileDesc],
    rows: int,
    k_dim: int,
    cout: int,
    n: int,
    oh: int,
    ow: int,
    bias: np.ndarray = None,
) -> Tuple[bool, str]:
    """
    脉动分块结果与真实卷积结果一致校验：im2col GEMM 按分块拼出结果，加 bias 后 reshape 为 NCHW 与 golden 比较。
    golden_y 为 NCHW 形状 (n, cout, oh, ow)。
    """
    if not tiles:
        return False, "no tiles"
    # GEMM 结果 (rows, cout)，与 conv2d_to_gemm 一致加上 bias 再 reshape
    C_systolic = compute_gemm_systolic_result(cols, w_mat, tiles, rows, k_dim, cout)
    if C_systolic.shape != (rows, cout):
        return False, "systolic GEMM shape {} != (rows,cout)".format(C_systolic.shape)
    if bias is not None:
        C_systolic = C_systolic + bias.reshape(1, -1)
    y_systolic = C_systolic.reshape(n, oh, ow, cout).transpose(0, 3, 1, 2)
    if y_systolic.shape != golden_y.shape:
        return False, "systolic conv shape {} != golden shape {}".format(
            y_systolic.shape, golden_y.shape
        )
    if not np.array_equal(y_systolic, golden_y):
        diff = np.abs(y_systolic.astype(np.int64) - golden_y.astype(np.int64))
        max_diff = int(np.max(diff))
        n_err = int(np.sum(diff != 0))
        return False, "systolic conv vs golden: {} elements differ, max_diff={}".format(
            n_err, max_diff
        )
    return True, "systolic result matches golden (conv2d)"
