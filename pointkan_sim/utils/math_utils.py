# -*- coding: utf-8 -*-
"""
数学与整数运算工具（Math Utils）
================================

公共辅助函数，供 operators、layer_buffer_ops 等复用，避免重复实现。
"""

import math


def safe_div_ceil(a: int, b: int) -> int:
    """
    安全整除上取整：ceil(a/b)，且保证分母至少为 1，避免除零。
    用于周期数、字节/带宽等上取整。
    """
    return max(1, math.ceil(max(1, a) / max(1, b)))
