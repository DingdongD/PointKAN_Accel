# -*- coding: utf-8 -*-
"""
分块描述：单个 GEMM/Conv2D-GEMM 分块在逻辑矩阵中的位置与规模。
"""
from dataclasses import dataclass, asdict


@dataclass
class TileDesc:
    """单个计算块（tile）的描述，用于调度与周期/能耗估计"""
    op_type: str       # 算子类型：gemm / conv2d_gemm
    mode: str           # 数据流模式：OS / WS
    m0: int             # 在 M 维的起始行索引
    k0: int             # 在 K 维的起始列/行索引
    n0: int             # 在 N 维的起始列索引
    m_tile: int         # 本块 M 维大小
    k_tile: int         # 本块 K 维大小
    n_tile: int         # 本块 N 维大小
    batch_idx: int = 0  # 批索引（多批 Conv2D 时使用）
    extra: dict = None  # 扩展信息（可选）

    def to_dict(self):
        """转为字典，便于序列化与报告"""
        d = asdict(self)
        if self.extra is None:
            d['extra'] = {}
        return d
