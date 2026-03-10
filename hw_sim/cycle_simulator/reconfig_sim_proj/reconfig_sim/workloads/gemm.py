# -*- coding: utf-8 -*-
"""
GEMM 工作负载：矩阵乘 C = A*B，A(m,k)、B(k,n)，支持随机生成与 golden 校验和。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GemmWorkload:
    """单次 GEMM 的维度与数据，用于仿真与正确性校验"""

    m: int
    k: int
    n: int
    a: np.ndarray   # 矩阵 A，形状 (m, k)
    b: np.ndarray   # 矩阵 B，形状 (k, n)
    name: str = "gemm"

    @classmethod
    def random(cls, m: int, k: int, n: int, word_size: int = 16, seed: int = 0):
        """生成有符号 word_size 位范围内的随机 A、B，用于测试或报告校验和。"""
        rng = np.random.default_rng(seed)
        low = -(2 ** (word_size - 1))
        high = 2 ** (word_size - 1)
        a = rng.integers(low, high, size=(m, k), dtype=np.int64)
        b = rng.integers(low, high, size=(k, n), dtype=np.int64)
        return cls(m=m, k=k, n=n, a=a, b=b)

    def golden(self) -> np.ndarray:
        """返回 NumPy 计算的 C = A.dot(B)，形状 (m, n)，用于校验和比对。"""
        return self.a.dot(self.b)
