from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OpDesc:
    name: str
    op_type: str
    m: int
    n: int
    k: int
    has_static_weight: bool = False
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.k


@dataclass(frozen=True)
class Conv2DIm2ColDesc(OpDesc):
    c_in: int = 0
    c_out: int = 0
    h_in: int = 0
    w_in: int = 0
    k_h: int = 0
    k_w: int = 0
    stride: int = 1
    padding: int = 0


def conv2d_to_im2col_op(
    name: str,
    batch: int,
    c_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    k_h: int,
    k_w: int,
    stride: int = 1,
    padding: int = 0,
) -> Conv2DIm2ColDesc:
    h_out = (h_in + 2 * padding - k_h) // stride + 1
    w_out = (w_in + 2 * padding - k_w) // stride + 1
    m = batch * h_out * w_out
    k = c_in * k_h * k_w
    n = c_out
    return Conv2DIm2ColDesc(
        name=name,
        op_type="conv2d_im2col",
        m=m,
        n=n,
        k=k,
        has_static_weight=True,
        attrs={"batch": batch, "h_out": h_out, "w_out": w_out},
        c_in=c_in,
        c_out=c_out,
        h_in=h_in,
        w_in=w_in,
        k_h=k_h,
        k_w=k_w,
        stride=stride,
        padding=padding,
    )
