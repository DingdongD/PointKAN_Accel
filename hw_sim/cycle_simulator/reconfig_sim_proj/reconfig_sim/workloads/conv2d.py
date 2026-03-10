# -*- coding: utf-8 -*-
"""
Conv2D 工作负载：卷积参数与输入/权重/偏置张量，经 im2col 后由仿真器按 GEMM 分块执行。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Conv2DWorkload:
    """单次 2D 卷积的维度、步长、padding 与数据（NCHW）"""

    n: int
    c_in: int
    h: int
    w: int
    c_out: int
    k_h: int
    k_w: int
    stride: int = 1
    padding: int = 0
    groups: int = 1
    x: np.ndarray = None    # 输入 [N, C_in, H, W]
    wgt: np.ndarray = None  # 权重 [C_out, C_in/groups, K_h, K_w]
    bias: np.ndarray = None # 偏置 [C_out]，可选
    name: str = "conv2d"

    @classmethod
    def random(cls, n: int, c_in: int, h: int, w: int, c_out: int, k_h: int, k_w: int,
               stride: int = 1, padding: int = 0, groups: int = 1, word_size: int = 16, seed: int = 0):
        """生成有符号 word_size 位范围内的随机输入、权重与偏置。"""
        rng = np.random.default_rng(seed)
        low = -(2 ** (word_size - 1))
        high = 2 ** (word_size - 1)
        x = rng.integers(low, high, size=(n, c_in, h, w), dtype=np.int64)
        wgt = rng.integers(low, high, size=(c_out, c_in // groups, k_h, k_w), dtype=np.int64)
        bias = rng.integers(low, high, size=(c_out,), dtype=np.int64)
        return cls(n=n, c_in=c_in, h=h, w=w, c_out=c_out, k_h=k_h, k_w=k_w,
                   stride=stride, padding=padding, groups=groups, x=x, wgt=wgt, bias=bias)
