# -*- coding: utf-8 -*-
"""
分块器（Tiler）：将 GEMM 或 im2col 后的 GEMM 按阵列规模与数据流模式切分为多个 TileDesc。
"""
from typing import List
from ..models.tile import TileDesc


class GemmTiler:
    """GEMM 分块：按 arr_r x arr_c 与 OS/WS 规则将 MxKxN 切分为若干块"""

    def __init__(self, arr_r: int, arr_c: int):
        self.arr_r = arr_r  # 阵列行数
        self.arr_c = arr_c  # 阵列列数

    def tile(self, m: int, k: int, n: int, mode: str) -> List[TileDesc]:
        """
        对矩阵乘 A(MxK) * B(KxN) 分块。
        OS：M、N 维均按阵列脚印分块。每块必须 clear → compute → drain，把该块 C 取走后再做下一块，
            否则下一块会在同一 PE 上继续累加导致结果错误；故不能为“复用权重”而合并 M 维。
        WS：权重重用，M 维不切分（多行复用同一批权重）；仅按 N、K 分块，周期亚线性于 M。
        """
        tiles = []
        if mode == 'OS':
            # 每块 clear→compute→drain，取走 C 后再做下一块，避免 PE 上错误累加
            for m0 in range(0, m, self.arr_r):
                for n0 in range(0, n, self.arr_c):
                    mt = min(self.arr_r, m - m0)
                    nt = min(self.arr_c, n - n0)
                    tiles.append(TileDesc('gemm', 'OS', m0, 0, n0, mt, k, nt))
        elif mode == 'WS':
            # M 维不切：同一 (n0,k0) 块内多行复用权重，仅按 N、K 分块
            for n0 in range(0, n, self.arr_c):
                nt = min(self.arr_c, n - n0)
                for k0 in range(0, k, self.arr_r):
                    kt = min(self.arr_r, k - k0)
                    tiles.append(TileDesc('gemm', 'WS', 0, k0, n0, m, kt, nt))
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        return tiles


class Conv2DTiler:
    """Conv2D 经 im2col 化为 GEMM 后的分块：rows = N*OH*OW，k = Cin*Kh*Kw，cout = Cout"""

    def __init__(self, arr_r: int, arr_c: int):
        self.arr_r = arr_r
        self.arr_c = arr_c

    def tile_im2col_gemm(self, rows: int, k: int, cout: int, mode: str, batch_idx: int = 0):
        """
        对 im2col 后的 GEMM(rows, k, cout) 分块。
        rows：等价于 GEMM 的 M（通常为 N*OH*OW 或合并批后的行数）。
        """
        tiles = []
        if mode == 'OS':
            # 与 GEMM 一致：每块 clear→compute→drain，行维按 arr_r 分块
            for m0 in range(0, rows, self.arr_r):
                for n0 in range(0, cout, self.arr_c):
                    mt = min(self.arr_r, rows - m0)
                    nt = min(self.arr_c, cout - n0)
                    tiles.append(TileDesc('conv2d_gemm', 'OS', m0, 0, n0, mt, k, nt, batch_idx=batch_idx))
        elif mode == 'WS':
            # M(rows) 维不切，复用权重；仅按 cout、k 分块
            for n0 in range(0, cout, self.arr_c):
                nt = min(self.arr_c, cout - n0)
                for k0 in range(0, k, self.arr_r):
                    kt = min(self.arr_r, k - k0)
                    tiles.append(TileDesc('conv2d_gemm', 'WS', 0, k0, n0, rows, kt, nt, batch_idx=batch_idx))
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        return tiles
