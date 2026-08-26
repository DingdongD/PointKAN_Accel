"""对 float32 标量流做 IEEE 字段分解熵（与阶段 1/2 共用）。"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .entropy_stats import joint_entropy_two, shannon_entropy_counts, topk_coverage_from_counts
from .float_bits import effective_mantissa_bits, extract_sign_exp_mant, mantissa_high_bits


def ieee32_stream_field_stats(vals: np.ndarray) -> Dict[str, float]:
    """
    vals: 任意形状 float32 数组，按元素视作标量流。
    返回 Hexp, Hsign, Hmant8, H(sign,exp), H(exp,mant8), top1..3 exp 覆盖, avg_effective_bits。
    """
    sign, exp, mant = extract_sign_exp_mant(vals)
    mant8 = mantissa_high_bits(vals, 8)

    sign_i = sign.astype(np.int64).ravel()
    exp_i = exp.astype(np.int64).ravel()
    mant8_i = mant8.astype(np.int64).ravel()

    c_exp = np.bincount(exp_i, minlength=256)
    c_sign = np.bincount(sign_i, minlength=2)
    c_m8 = np.bincount(mant8_i, minlength=256)

    top = topk_coverage_from_counts(c_exp, k=3)
    eff = effective_mantissa_bits(vals)
    avg_eff = float(np.nanmean(eff))

    return {
        "Hexp": shannon_entropy_counts(c_exp),
        "Hsign": shannon_entropy_counts(c_sign),
        "Hmant8": shannon_entropy_counts(c_m8),
        "Hsignexp": joint_entropy_two(sign_i, exp_i, 2, 256),
        "Hexpmant8": joint_entropy_two(exp_i, mant8_i, 256, 256),
        "top1_exp_cov": top["top1_exp_cov"],
        "top2_exp_cov": top["top2_exp_cov"],
        "top3_exp_cov": top["top3_exp_cov"],
        "avg_effective_bits": avg_eff,
    }
