"""float32 位域拆分与有效尾数位宽估计。"""

from __future__ import annotations

import numpy as np


def float32_to_uint32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32).view(np.uint32)


def signed_bits_required(max_abs: int) -> int:
    """表示有符号整数 ±max_abs 所需的最小位数（含符号，保守上界）。"""
    v = int(max_abs)
    if v <= 0:
        return 1
    return int(np.ceil(np.log2(v + 1.0))) + 1


def quantize_xyz_int16(xyz: np.ndarray, scale: float = 2048.0) -> np.ndarray:
    return np.clip(np.round(xyz.astype(np.float64) * scale), -32768, 32767).astype(np.int32)


def extract_sign_exp_mant(arr: np.ndarray):
    """
    IEEE754 float32: sign(1), exp(8), mantissa(23).
    返回与 arr 同形状的 uint32 / uint32 / uint32（mant 为显式 23 位，不含隐含 1）。
    """
    u = float32_to_uint32(arr)
    sign = (u >> np.uint32(31)) & np.uint32(1)
    exp = (u >> np.uint32(23)) & np.uint32(0xFF)
    mant = u & np.uint32(0x7FFFFF)
    return sign, exp, mant


def mantissa_high_bits(arr: np.ndarray, k: int = 8) -> np.ndarray:
    """尾数高 k 位（从 23 位尾数的高位取）。"""
    _, _, mant = extract_sign_exp_mant(arr)
    shift = max(23 - k, 0)
    return (mant >> np.uint32(shift)) & np.uint32((1 << k) - 1)


def effective_mantissa_bits(arr: np.ndarray) -> np.ndarray:
    """
    leading-zero 风格：对规范化数，有效精度 ≈ 1 + mantissa 字段中非前导零宽度；
    次正规：仅为 mantissa 的最高置位宽度；0 为 0；Inf/NaN 为 nan。
    """
    u = float32_to_uint32(arr)
    exp = (u >> np.uint32(23)) & np.uint32(0xFF)
    mant = u & np.uint32(0x7FFFFF)
    out = np.zeros(arr.shape, dtype=np.float64)
    nan_mask = exp == np.uint32(255)
    out[nan_mask] = np.nan

    exp_i = exp.astype(np.int64)
    mant_i = mant.astype(np.int64)

    normal = (~nan_mask) & (exp_i != 0)
    subn = (~nan_mask) & (exp_i == 0) & (mant_i != 0)
    zero = (~nan_mask) & (exp_i == 0) & (mant_i == 0)

    def _bl(v: np.ndarray) -> np.ndarray:
        return np.frompyfunc(lambda x: int(x).bit_length(), 1, 1)(v).astype(np.float64)

    # 次正规：无隐含 1
    if np.any(subn):
        m = mant_i[subn]
        out[np.where(subn)] = _bl(m)

    # 正规：隐含前导 1；mant=0 表示 1.0 精确
    if np.any(normal):
        m = mant_i[normal]
        bl = _bl(m)
        eff = np.where(m == 0, 1.0, 1.0 + bl)
        out[np.where(normal)] = eff

    out[np.where(zero)] = 0.0
    return out


def bf16_from_fp32(arr: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16 (numpy 1.24+). 若不可用则手动截断。"""
    x = arr.astype(np.float32)
    try:
        return x.astype(np.dtype("bfloat16")).astype(np.float32)
    except (TypeError, ValueError):
        u = float32_to_uint32(x)
        # round to bf16: drop low 16 bits of float32
        u16 = (u >> np.uint32(16)).astype(np.uint32)
        u_bf = u16 << np.uint32(16)
        return u_bf.view(np.float32)


def fp16_from_fp32(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    f16 = x.astype(np.float16)
    return f16.astype(np.float32)


def float16_parts_f16(arr_f32_as_storage: np.ndarray):
    """将「由 fp16 转回 fp32 的数组」再按 IEEE fp16 语义拆域（用于统计）。"""
    f16 = arr_f32_as_storage.astype(np.float32).astype(np.float16)
    u = f16.view(np.uint16)
    sign = (u >> np.uint16(15)) & np.uint16(1)
    exp = (u >> np.uint16(10)) & np.uint16(0x1F)
    mant = u & np.uint16(0x3FF)
    return sign.astype(np.uint32), exp.astype(np.uint32), mant.astype(np.uint32)
