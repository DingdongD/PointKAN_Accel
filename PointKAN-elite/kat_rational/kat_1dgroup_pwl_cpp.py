"""
KAT_Group_PWL_CPP: CPU fused C++ extension for PWL-v2 (mode 0/1/2/3).

Drop-in replacement for KAT_Group_PWL on CPU.
On CUDA it will fall back to the torch implementation (so keep using Triton there).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn

from .kat_1dgroup_pwl import KAT_Group_PWL


@lru_cache(maxsize=1)
def _load_ext():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "cpp_pwl", "pwl_ext.cpp")
    return load(
        name="kat_pwl_v2_ext",
        sources=[src],
        extra_cflags=["-O3"],
        with_cuda=False,
        verbose=False,
    )


class KAT_Group_PWL_CPP(nn.Module):
    def __init__(
        self,
        num_groups,
        x_min,
        x_max,
        modes,
        const_vals,
        lin_m,
        lin_b,
        split_stacked,
        slope_stacked,
        intercept_stacked,
        n_intervals,
        *,
        cuda_fallback: str = "torch",
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.cuda_fallback = str(cuda_fallback)

        self.register_buffer("_x_min", torch.tensor(float(x_min), dtype=torch.float32))
        self.register_buffer("_x_max", torch.tensor(float(x_max), dtype=torch.float32))
        self.register_buffer("_modes", torch.as_tensor(modes, dtype=torch.int64).contiguous())
        self.register_buffer("_const_vals", torch.as_tensor(const_vals, dtype=torch.float32).contiguous())
        self.register_buffer("_lin_m", torch.as_tensor(lin_m, dtype=torch.float32).contiguous())
        self.register_buffer("_lin_b", torch.as_tensor(lin_b, dtype=torch.float32).contiguous())
        self.register_buffer("_split_stacked", torch.as_tensor(split_stacked, dtype=torch.float32).contiguous())
        self.register_buffer("_slope_stacked", torch.as_tensor(slope_stacked, dtype=torch.float32).contiguous())
        self.register_buffer("_intercept_stacked", torch.as_tensor(intercept_stacked, dtype=torch.float32).contiguous())
        self.register_buffer("_n_intervals", torch.as_tensor(n_intervals, dtype=torch.int64).contiguous())

        # keep a torch fallback instance for CUDA or unsupported dtypes
        self._torch_impl: Optional[KAT_Group_PWL] = None

    @property
    def order(self):
        return (5, 4)

    def _get_torch_impl(self):
        if self._torch_impl is None:
            self._torch_impl = KAT_Group_PWL(
                num_groups=self.num_groups,
                x_min=float(self._x_min.item()),
                x_max=float(self._x_max.item()),
                modes=self._modes,
                const_vals=self._const_vals,
                lin_m=self._lin_m,
                lin_b=self._lin_b,
                split_stacked=self._split_stacked,
                slope_stacked=self._slope_stacked,
                intercept_stacked=self._intercept_stacked,
                n_intervals=self._n_intervals,
            )
        return self._torch_impl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() in (2, 3), "Input must be 2D (B, D) or 3D (B, L, D)."
        original_dim = input.dim()
        x = input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, D = x.shape
        G = self.num_groups
        C = D // G
        if G * C != D:
            raise ValueError(f"D={D} not divisible by num_groups={G}")

        # CPU fused path
        if x.device.type == "cpu":
            # extension expects float32 contiguous [N, D]
            x2 = x.contiguous().view(-1, D)
            x_f32 = x2.float().contiguous()
            ext = _load_ext()
            y_f32 = ext.pwl_v2_forward_cpu(
                x_f32,
                self._modes.cpu(),
                self._const_vals.cpu(),
                self._lin_m.cpu(),
                self._lin_b.cpu(),
                self._split_stacked.cpu(),
                self._slope_stacked.cpu(),
                self._intercept_stacked.cpu(),
                self._n_intervals.cpu(),
                float(self._x_min.item()),
                float(self._x_max.item()),
            )
            y = y_f32.to(dtype=x2.dtype).view(B, L, D)
        else:
            # CUDA: fall back to torch implementation
            y = self._get_torch_impl().to(x.device)(x)

        if original_dim == 2:
            y = y.squeeze(1)
        return y

    def extra_repr(self):
        return f"num_groups={self.num_groups}, x_range=[{self._x_min.item()}, {self._x_max.item()}], PWL-v2, impl=cpp(cpu)"

