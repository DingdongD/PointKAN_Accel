"""
KAT_Group_PWL_Triton: Triton fused kernel for PWL-v2 (mode 0/1/2/3).

Drop-in replacement for KAT_Group_PWL with the same input/output shapes:
  - input: (B, L, D) or (B, D)
  - output: same shape

Modes per group g:
  0: zero
  1: const
  2: linear (m*x + b)
  3: piecewise linear (segment lookup + m_i*x + b_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl


def _pwl_v2_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    C: tl.constexpr,
    x_min_ptr,
    x_max_ptr,
    modes_ptr,
    const_ptr,
    lin_m_ptr,
    lin_b_ptr,
    split_ptr,
    slope_ptr,
    intercept_ptr,
    n_intervals_ptr,
    MAX_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_min = tl.load(x_min_ptr).to(tl.float32)
    x_max = tl.load(x_max_ptr).to(tl.float32)
    x = tl.maximum(x, x_min)
    x = tl.minimum(x, x_max)

    # group id for each element based on channel index d
    d = offsets % D
    g = d // C  # [0, G)

    mode = tl.load(modes_ptr + g, mask=mask, other=0).to(tl.int32)

    # cheap branches
    y0 = tl.zeros([BLOCK], dtype=tl.float32)
    y1 = tl.load(const_ptr + g, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(lin_m_ptr + g, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(lin_b_ptr + g, mask=mask, other=0.0).to(tl.float32)
    y2 = m * x + b

    # mode==3: segment lookup (linear scan over MAX_K start boundaries)
    K = tl.load(n_intervals_ptr + g, mask=mask, other=0).to(tl.int32)  # intervals per group
    count = tl.zeros([BLOCK], dtype=tl.int32)

    # split_ptr layout: (G, MAX_K+1) float32, but we only need starts split[:, 0:MAX_K]
    # slope/intercept layout: (G, MAX_K) float32
    # Use static_range for unrolling.
    for i in tl.static_range(0, MAX_K):
        # only active when mode==3 and i < K
        active = (mode == 3) & (i < K) & mask
        start = tl.load(split_ptr + g * (MAX_K + 1) + i, mask=active, other=0.0).to(tl.float32)
        count += tl.where(active & (x >= start), 1, 0).to(tl.int32)

    seg = tl.maximum(count - 1, 0)
    seg = tl.minimum(seg, MAX_K - 1)

    sl = tl.load(slope_ptr + g * MAX_K + seg, mask=(mode == 3) & mask, other=0.0).to(tl.float32)
    ic = tl.load(intercept_ptr + g * MAX_K + seg, mask=(mode == 3) & mask, other=0.0).to(tl.float32)
    y3 = sl * x + ic

    y = tl.where(mode == 0, y0, tl.where(mode == 1, y1, tl.where(mode == 2, y2, y3)))
    # Store float32; cast back to input dtype on Python side.
    tl.store(y_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
# JIT / autotune wrapping
# -----------------------------------------------------------------------------
# Triton autotune uses triton.testing.do_bench which depends on torch._dynamo in
# some Triton versions. For older PyTorch builds (e.g., torch==2.0.0) without
# torch._dynamo, enable a safe fallback (no autotune).
_pwl_v2_fwd_kernel = triton.jit(_pwl_v2_fwd_kernel)

_CAN_AUTOTUNE = hasattr(torch, "_dynamo")
if _CAN_AUTOTUNE:
    _pwl_v2_fwd_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
        ],
        key=["D", "MAX_K"],
    )(_pwl_v2_fwd_kernel)


def _launch_pwl_v2(
    x: torch.Tensor,
    *,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    modes: torch.Tensor,
    const_vals: torch.Tensor,
    lin_m: torch.Tensor,
    lin_b: torch.Tensor,
    split_stacked: torch.Tensor,
    slope_stacked: torch.Tensor,
    intercept_stacked: torch.Tensor,
    n_intervals: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Triton kernel requires CUDA tensor."
    assert x.is_contiguous(), "x must be contiguous."
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    n_elements = x.numel()
    # Always compute into float32 buffer for stable accumulation and simple tl.store dtype.
    y = torch.empty((n_elements,), device=x.device, dtype=torch.float32)

    # Constants / metadata
    D = x.shape[-1]
    G = int(modes.numel())
    C = D // G
    max_k = int(slope_stacked.shape[1])
    assert split_stacked.shape[1] == max_k + 1, "split_stacked must be (G, max_k+1)"

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

    # Ensure dtypes on device
    x_min = x_min.to(device=x.device, dtype=torch.float32)
    x_max = x_max.to(device=x.device, dtype=torch.float32)
    modes = modes.to(device=x.device, dtype=torch.int32)
    const_vals = const_vals.to(device=x.device, dtype=torch.float32)
    lin_m = lin_m.to(device=x.device, dtype=torch.float32)
    lin_b = lin_b.to(device=x.device, dtype=torch.float32)
    split_stacked = split_stacked.to(device=x.device, dtype=torch.float32)
    slope_stacked = slope_stacked.to(device=x.device, dtype=torch.float32)
    intercept_stacked = intercept_stacked.to(device=x.device, dtype=torch.float32)
    n_intervals = n_intervals.to(device=x.device, dtype=torch.int32)

    # When autotune is unavailable, provide a fixed, reasonable config.
    launch_kwargs = {}
    if not _CAN_AUTOTUNE:
        launch_kwargs = {"BLOCK": 1024, "num_warps": 4, "num_stages": 2}

    _pwl_v2_fwd_kernel[grid](
        x,
        y,
        n_elements=n_elements,
        D=D,
        G=G,
        C=C,
        x_min_ptr=x_min,
        x_max_ptr=x_max,
        modes_ptr=modes,
        const_ptr=const_vals,
        lin_m_ptr=lin_m,
        lin_b_ptr=lin_b,
        split_ptr=split_stacked,
        slope_ptr=slope_stacked,
        intercept_ptr=intercept_stacked,
        n_intervals_ptr=n_intervals,
        MAX_K=max_k,
        **launch_kwargs,
    )
    return y.view_as(x).to(dtype=x.dtype)


class KAT_Group_PWL_Triton(nn.Module):
    """
    Triton fused forward for PWL-v2.
    Buffers are the same as KAT_Group_PWL (torch implementation).
    """

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
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.register_buffer("_x_min", torch.tensor(float(x_min), dtype=torch.float32))
        self.register_buffer("_x_max", torch.tensor(float(x_max), dtype=torch.float32))
        self.register_buffer("_modes", torch.as_tensor(modes, dtype=torch.int64))
        self.register_buffer("_const_vals", torch.as_tensor(const_vals, dtype=torch.float32))
        self.register_buffer("_lin_m", torch.as_tensor(lin_m, dtype=torch.float32))
        self.register_buffer("_lin_b", torch.as_tensor(lin_b, dtype=torch.float32))
        self.register_buffer("_split_stacked", torch.as_tensor(split_stacked, dtype=torch.float32))
        self.register_buffer("_slope_stacked", torch.as_tensor(slope_stacked, dtype=torch.float32))
        self.register_buffer("_intercept_stacked", torch.as_tensor(intercept_stacked, dtype=torch.float32))
        self.register_buffer("_n_intervals", torch.as_tensor(n_intervals, dtype=torch.int64))

    @property
    def order(self):
        return (5, 4)

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

        # Use contiguous flattened view for kernel
        x_flat = x.contiguous().view(-1, D)

        y = _launch_pwl_v2(
            x_flat,
            x_min=self._x_min,
            x_max=self._x_max,
            modes=self._modes,
            const_vals=self._const_vals,
            lin_m=self._lin_m,
            lin_b=self._lin_b,
            split_stacked=self._split_stacked,
            slope_stacked=self._slope_stacked,
            intercept_stacked=self._intercept_stacked,
            n_intervals=self._n_intervals,
        )
        y = y.view(B, L, D)
        if original_dim == 2:
            y = y.squeeze(1)
        return y

    def extra_repr(self):
        return f"num_groups={self.num_groups}, x_range=[{self._x_min.item()}, {self._x_max.item()}], PWL-v2, impl=triton"

