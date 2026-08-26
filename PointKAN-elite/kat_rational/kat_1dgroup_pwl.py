"""
KAT_Group_PWL: 用 v2 导出的分段线性（含 zero/const/linear/pwl）替代 GRF 有理函数，
与 KAT_Group / KAT_Group_LUT 相同的输入输出形状：(B, L, D) 或 (B, D)。
仅用于推理；参数由 grf_pwl_trainer_v2 导出的 JSON 经 inject 脚本填入。
"""

import torch
import torch.nn as nn


def _pwl_intervals(x, split_points, slopes, intercepts):
    """
    x: (*), split_points: (K+1), slopes/intercepts: (K,)
    与 GQA-LUT 中 piecewise_linear_approximation 区间语义一致：除最后一区间外左闭右开，最后一区间闭右端。
    """
    out = torch.zeros_like(x)
    K = int(slopes.shape[0])
    if K == 0:
        return out
    for i in range(K):
        lo = split_points[i]
        hi = split_points[i + 1]
        if i < K - 1:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        out = out + m.to(dtype=x.dtype) * (slopes[i] * x + intercepts[i])
    return out


class KAT_Group_PWL(nn.Module):
    """
    num_groups=G，输入通道 D=G*C，按 (B,L,G,C) 逐组标量映射。
    modes[g]: 0=zero, 1=const, 2=linear, 3=pwl
    pwl 组使用 split_stacked[g, :n_int+1], slope/intercept_stacked[g, :n_int]。
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

    def forward(self, input):
        assert input.dim() in (2, 3), "Input must be 2D (B, D) or 3D (B, L, D)."
        original_dim = input.dim()
        if input.dim() == 2:
            input = input.unsqueeze(1)
        B, L, D = input.shape
        G = self.num_groups
        C = D // G
        if G * C != D:
            raise ValueError(f"D={D} not divisible by num_groups={G}")
        x = input.view(B, L, G, C)
        x = x.clamp(self._x_min, self._x_max)

        # Vectorized PWL evaluation:
        # - modes==0: output 0
        # - modes==1: output const
        # - modes==2: output lin_m*x + lin_b
        # - modes==3: compute segment index for each element, then gather (slope, intercept)
        out = torch.zeros_like(x)

        modes = self._modes  # (G,)
        mask1 = (modes == 1).view(1, 1, G, 1)
        mask2 = (modes == 2).view(1, 1, G, 1)
        mask3 = (modes == 3).view(1, 1, G, 1)

        const_vals = self._const_vals.to(dtype=x.dtype).view(1, 1, G, 1)
        lin_m = self._lin_m.to(dtype=x.dtype).view(1, 1, G, 1)
        lin_b = self._lin_b.to(dtype=x.dtype).view(1, 1, G, 1)

        out = torch.where(mask1, const_vals, out)
        lin_val = lin_m * x + lin_b
        out = torch.where(mask2, lin_val, out)

        # PWL mode. For strict increasing split points, segment index can be derived from:
        #   idx = (# of active start boundaries <= x) - 1
        # where active start boundaries are split[i] for i in [0, K-1], i.e. K = n_intervals.
        max_k = int(self._slope_stacked.size(1))  # maximum number of intervals across groups
        if max_k > 0:
            # 只对 modes==3 的 group 计算 pwl_val，避免对 0/1/2 模式的 group 做无意义比较
            mask3_g = (modes == 3)  # (G,)
            if torch.any(mask3_g):
                g_idx = torch.nonzero(mask3_g, as_tuple=False).view(-1)  # (Gp,)
                Gp = int(g_idx.numel())

                x3 = x[:, :, g_idx, :]  # (B, L, Gp, C)
                n_intervals3 = self._n_intervals[g_idx]  # (Gp,)

                # starts: split_stacked[g, i] for i in [0, K-1]
                starts3 = self._split_stacked[g_idx, :max_k].to(dtype=x.dtype)  # (Gp, max_k)
                slopes3 = self._slope_stacked[g_idx, :].to(dtype=x.dtype)  # (Gp, max_k)
                intercepts3 = self._intercept_stacked[g_idx, :].to(dtype=x.dtype)  # (Gp, max_k)

                start_active3 = (torch.arange(max_k, device=x.device).view(1, max_k) < n_intervals3.view(Gp, 1))
                # start_active3: (Gp, max_k)

                cmp = (x3.unsqueeze(-1) >= starts3.view(1, 1, Gp, 1, max_k)) & start_active3.view(
                    1, 1, Gp, 1, max_k
                )
                count = cmp.sum(dim=-1)  # int64, (B, L, Gp, C)
                idx = (count - 1).clamp(min=0, max=max_k - 1).to(dtype=torch.int64)  # (B, L, Gp, C)

                slopes_exp = slopes3.view(1, 1, Gp, 1, max_k).expand(B, L, Gp, C, max_k)
                intercepts_exp = intercepts3.view(1, 1, Gp, 1, max_k).expand(B, L, Gp, C, max_k)
                sl_val = torch.gather(slopes_exp, -1, idx.unsqueeze(-1)).squeeze(-1)  # (B, L, Gp, C)
                ic_val = torch.gather(intercepts_exp, -1, idx.unsqueeze(-1)).squeeze(-1)  # (B, L, Gp, C)

                pwl_val3 = sl_val * x3 + ic_val  # (B, L, Gp, C)
                out[:, :, g_idx, :] = pwl_val3

        out = out.view(B, L, D)
        if original_dim == 2:
            out = out.squeeze(1)
        return out

    def extra_repr(self):
        return f"num_groups={self.num_groups}, x_range=[{self._x_min.item()}, {self._x_max.item()}], PWL-v2"
