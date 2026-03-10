# -*- coding: utf-8 -*-
"""
缓冲模型：按 A/B/C 读写字节数估算访存能耗与 stall（当前为解析模型，stall=0）。
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class BufferStats:
    """单块（tile）的缓冲访问统计与能耗"""
    a_read_bytes: int = 0   # A 矩阵读字节数
    b_read_bytes: int = 0   # B 矩阵读字节数
    c_read_bytes: int = 0   # C 矩阵读字节数（部分和读回）
    c_write_bytes: int = 0  # C 矩阵写字节数
    energy: float = 0.0     # 本块缓冲相关能耗（相对单位）
    stalls: int = 0         # 因缓冲不足导致的 stall 周期（当前模型恒为 0）


class BufferModel:
    """解析缓冲模型：根据每块 A/B/C 读写量计算能耗"""

    def __init__(self, cfg):
        self.cfg = cfg

    def estimate(self, *, a_bytes: int, b_bytes: int, c_read_bytes: int, c_write_bytes: int) -> BufferStats:
        """根据读写字节数与配置的每字节能耗，估算本块缓冲能耗"""
        e = (a_bytes + b_bytes + c_read_bytes) * self.cfg.read_energy_per_byte
        e += c_write_bytes * self.cfg.write_energy_per_byte
        return BufferStats(
            a_read_bytes=a_bytes,
            b_read_bytes=b_bytes,
            c_read_bytes=c_read_bytes,
            c_write_bytes=c_write_bytes,
            energy=e,
            stalls=0,
        )
