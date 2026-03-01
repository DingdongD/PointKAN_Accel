"""
KAT_Group_LUT: 使用均匀分割 LUT + 线性插值替代 GRF 有理函数计算，与 KAT_Group 同输入输出。
用于推理时查表，不参与训练；LUT 由 run_grf_lut_verify.py 生成并加载。

支持三种模式：
- vanilla：单表 table (G, n_bins+1)，可为浮点或量化（table_q + scale + zp）。
- base_offset：base 表 + offset 表，查表后合并 base(x) + offset_g(x)；可为浮点或量化。
每层 x_min/x_max 来自校准时由 run_grf_lut_verify 写入 config 的 x_range_source=calibration，否则为 default。
"""

import torch
import torch.nn as nn


def _linear_interp(x, x_min, x_max, n_bins, table):
    """table: (B, L, G, n_bins+1)，x: (B, L, G, C)。按最后一维 table 插值，索引与 x 对应。"""
    span = (x_max - x_min).clamp(min=1e-12)
    idx_float = (x - x_min) / span * n_bins
    idx_float = torch.clamp(idx_float, 0, n_bins)
    i0 = idx_float.long().clamp(0, n_bins - 1)
    i1 = (i0 + 1).clamp(max=n_bins)
    t = (idx_float - i0.float()).clamp(0, 1)
    # table (B,L,G,T), i0 (B,L,G,C) -> gather on dim -1 need index (B,L,G,C)
    v0 = torch.gather(table, -1, i0)
    v1 = torch.gather(table, -1, i1)
    return (1 - t) * v0 + t * v1


def _dequant(q, scale, zero_point=0):
    """q 与 scale 同设备，zero_point 标量。"""
    return (q.float() - zero_point) * scale


class KAT_Group_LUT(nn.Module):
    """
    与 KAT_Group 同输入输出：(B, L, D) 或 (B, D)。
    支持 vanilla / base_offset，以及浮点表或量化表（table_q + scale + zp）。
    每层 x_min/x_max 由 run_grf_lut_verify 生成：若提供校准集或 analysis_json，则为校准后的输入范围（x_range_source=calibration），否则为默认范围（default）。
    """

    def __init__(
        self,
        num_groups,
        x_min,
        x_max,
        n_bins,
        device=None,
        *,
        table=None,
        table_q=None,
        table_scale=None,
        table_zp=None,
        base_table=None,
        offset_table=None,
        base_table_q=None,
        base_table_scale=None,
        base_table_zp=None,
        offset_table_q=None,
        offset_table_scale=None,
        offset_table_zp=None,
    ):
        super(KAT_Group_LUT, self).__init__()
        self.num_groups = int(num_groups)
        self.n_bins = int(n_bins)
        self.register_buffer("_x_min", torch.tensor(float(x_min), dtype=torch.float32))
        self.register_buffer("_x_max", torch.tensor(float(x_max), dtype=torch.float32))

        use_base_offset = base_table is not None or base_table_q is not None
        use_quant = table_q is not None or base_table_q is not None

        if use_base_offset:
            if use_quant:
                assert base_table_q is not None and offset_table_q is not None
                self.register_buffer("_base_table_q", torch.as_tensor(base_table_q))
                self.register_buffer("_base_scale", torch.tensor(float(base_table_scale), dtype=torch.float32))
                self.register_buffer("_base_zp", torch.tensor(int(base_table_zp or 0), dtype=torch.int32))
                self.register_buffer("_offset_table_q", torch.as_tensor(offset_table_q))
                self.register_buffer("_offset_scale", torch.tensor(float(offset_table_scale), dtype=torch.float32))
                self.register_buffer("_offset_zp", torch.tensor(int(offset_table_zp or 0), dtype=torch.int32))
                self._base_table = self._offset_table = None
            else:
                self.register_buffer("_base_table", torch.as_tensor(base_table, dtype=torch.float32))
                self.register_buffer("_offset_table", torch.as_tensor(offset_table, dtype=torch.float32))
                self._base_table_q = self._offset_table_q = None
            self._table = self._table_q = None
        else:
            if use_quant:
                self.register_buffer("_table_q", torch.as_tensor(table_q))
                self.register_buffer("_table_scale", torch.tensor(float(table_scale), dtype=torch.float32))
                self.register_buffer("_table_zp", torch.tensor(int(table_zp or 0), dtype=torch.int32))
                self._table = None
            else:
                t = torch.as_tensor(table, dtype=torch.float32)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                self.register_buffer("_table", t)
                self._table_q = None
            self._base_table = self._offset_table = None
        assert self._table is not None or self._table_q is not None or (self._base_table is not None or getattr(self, "_base_table_q", None) is not None)

    @property
    def x_min(self):
        return self._x_min.item()

    @property
    def x_max(self):
        return self._x_max.item()

    @property
    def order(self):
        return (5, 4)

    def _forward_vanilla_float(self, x, B, L, G, C):
        table = self._table
        table = table.unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        return _linear_interp(x, self._x_min, self._x_max, self.n_bins, table)

    def _forward_vanilla_quantized(self, x, B, L, G, C):
        table_q = self._table_q.unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        v = _linear_interp(x, self._x_min, self._x_max, self.n_bins, table_q.float())
        return _dequant(v, self._table_scale, self._table_zp.item())

    def _forward_base_offset_float(self, x, B, L, G, C):
        base = self._base_table
        offset = self._offset_table
        base = base.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        offset = offset.unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        base_v = _linear_interp(x, self._x_min, self._x_max, self.n_bins, base)
        offset_v = _linear_interp(x, self._x_min, self._x_max, self.n_bins, offset)
        return base_v + offset_v

    def _forward_base_offset_quantized(self, x, B, L, G, C):
        base_q = self._base_table_q.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        base_v = _linear_interp(x, self._x_min, self._x_max, self.n_bins, base_q.float())
        base_v = _dequant(base_v, self._base_scale, self._base_zp.item())
        offset_q = self._offset_table_q.unsqueeze(0).unsqueeze(0).expand(B, L, G, -1)
        offset_v = _linear_interp(x, self._x_min, self._x_max, self.n_bins, offset_q.float())
        offset_v = _dequant(offset_v, self._offset_scale, self._offset_zp.item())
        return base_v + offset_v

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

        if self._table is not None:
            out = self._forward_vanilla_float(x, B, L, G, C)
        elif self._table_q is not None:
            out = self._forward_vanilla_quantized(x, B, L, G, C)
        elif self._base_table is not None:
            out = self._forward_base_offset_float(x, B, L, G, C)
        else:
            out = self._forward_base_offset_quantized(x, B, L, G, C)

        out = out.view(B, L, D)
        if original_dim == 2:
            out = out.squeeze(1)
        return out

    def extra_repr(self):
        mode = "base_offset" if (self._base_table is not None or getattr(self, "_base_table_q", None) is not None) else "vanilla"
        q = "quantized" if (self._table_q is not None or getattr(self, "_base_table_q", None) is not None) else "float"
        return f"num_groups={self.num_groups}, n_bins={self.n_bins}, x_range=[{self.x_min}, {self.x_max}], mode={mode}, {q}"