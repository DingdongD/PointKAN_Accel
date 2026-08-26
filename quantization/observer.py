# Ported from ISEDA26/iseda_quant for PointKAN-Elite (RTN: MinMax only)
from __future__ import absolute_import
import torch
import torch.nn as nn
from .quant_utils import fake_quantize_per_tensor_affine, fake_quantize_per_channel_affine


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    new_axis_list = [i for i in range(len(x.size()))]
    new_axis_list[ch_axis] = 0
    new_axis_list[0] = ch_axis
    return torch.flatten(x.permute(new_axis_list), start_dim=1)


class ObserverBase(nn.Module):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("1e6")))
        self.register_buffer("max_val", torch.tensor(float("-1e6")))

    def set_bit(self, bit):
        self.bit = bit
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1)
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            scale = torch.nan_to_num(scale, nan=1.0)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            scale = torch.nan_to_num(scale, nan=1.0)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch.aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x_orig


class PercentileObserver(ObserverBase):
    """
    Robust observer: use percentile instead of raw min/max to reduce outlier impact.
    """

    def __init__(
        self,
        bit=8,
        symmetric=False,
        ch_axis=-1,
        percentile_low=0.001,
        percentile_high=0.999,
        max_samples=8192,
    ):
        super(PercentileObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.max_samples = int(max_samples)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        # Clamp percentiles to valid range
        p_low = max(0.0, min(1.0, self.percentile_low))
        p_high = max(0.0, min(1.0, self.percentile_high))
        if p_high < p_low:
            p_low, p_high = p_high, p_low

        if self.ch_axis == -1:
            # torch.quantile 在极大 tensor 上可能触发限制；这里对元素做随机子采样近似分位数
            flat = x.reshape(-1)
            if flat.numel() > self.max_samples:
                idx = torch.randint(0, flat.numel(), (self.max_samples,), device=flat.device)
                sample = flat[idx]
            else:
                sample = flat
            min_val_cur = torch.quantile(sample, p_low)
            max_val_cur = torch.quantile(sample, p_high)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            # y: (C, ...flattened) after _transform_to_ch_axis
            # 同样对最后一维做子采样，避免 y 过大导致 quantile 失败
            if y.numel() > self.max_samples * y.size(0):
                m = y.size(1)
                idx = torch.randint(0, m, (min(self.max_samples, m),), device=y.device)
                y = y[:, idx]
            min_val_cur = torch.quantile(y, p_low, dim=1)
            max_val_cur = torch.quantile(y, p_high, dim=1)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x_orig


class HistogramObserver(ObserverBase):
    """
    Activation histogram observer with threshold search.

    The current quantization path uses per-tensor symmetric activations. For
    per-channel use, this observer falls back to the inherited min/max qparams.
    """

    def __init__(
        self,
        bit=8,
        symmetric=False,
        ch_axis=-1,
        bins=2048,
        method="mse",
        start_bin=128,
        max_samples=262144,
    ):
        super(HistogramObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bins = int(bins)
        self.method = str(method).lower()
        self.start_bin = int(start_bin)
        self.max_samples = int(max_samples)
        self._samples = []

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)

        flat = x.reshape(-1)
        if self.symmetric:
            flat = flat.abs()
        if flat.numel() > self.max_samples:
            idx = torch.randint(0, flat.numel(), (self.max_samples,), device=flat.device)
            flat = flat[idx]
        self._samples.append(flat.detach().float().cpu())
        self._trim_samples()
        return x_orig

    def _trim_samples(self):
        total = sum(t.numel() for t in self._samples)
        if total <= self.max_samples:
            return
        samples = torch.cat(self._samples)
        if samples.numel() > self.max_samples:
            idx = torch.randperm(samples.numel())[: self.max_samples]
            samples = samples[idx]
        self._samples = [samples]

    def calculate_qparams(self, min_val=None, max_val=None):
        if self.ch_axis != -1 or not self._samples:
            return super(HistogramObserver, self).calculate_qparams(self.min_val, self.max_val)

        samples = torch.cat(self._samples)
        if samples.numel() == 0:
            return super(HistogramObserver, self).calculate_qparams(self.min_val, self.max_val)
        if self.symmetric:
            threshold = self._symmetric_threshold(samples)
            min_q = torch.tensor(-threshold, dtype=self.min_val.dtype, device=self.min_val.device)
            max_q = torch.tensor(threshold, dtype=self.max_val.dtype, device=self.max_val.device)
            self.min_val = min_q
            self.max_val = max_q
            return super(HistogramObserver, self).calculate_qparams(self.min_val, self.max_val)

        return super(HistogramObserver, self).calculate_qparams(self.min_val, self.max_val)

    def _symmetric_threshold(self, abs_samples):
        abs_samples = abs_samples[torch.isfinite(abs_samples)]
        if abs_samples.numel() == 0:
            return float(self.eps.item())
        max_abs = float(abs_samples.max().item())
        if max_abs <= 0:
            return float(self.eps.item())

        bins = max(32, int(self.bins))
        hist = torch.histc(abs_samples, bins=bins, min=0.0, max=max_abs).double()
        edges = torch.linspace(0.0, max_abs, bins + 1, dtype=torch.double)
        centers = (edges[:-1] + edges[1:]) * 0.5
        start = max(2, min(int(self.start_bin), bins - 1))

        if self.method == "kl":
            best_bin = self._best_kl_bin(hist, start)
        else:
            best_bin = self._best_mse_bin(hist, centers, start)
        return max(float(edges[best_bin].item()), float(self.eps.item()))

    def _best_mse_bin(self, hist, centers, start):
        levels = max(2, int(self.quant_max))
        best_bin = hist.numel()
        best_err = None
        for end in range(start, hist.numel() + 1):
            threshold = centers[end - 1]
            scale = threshold / float(levels)
            if scale <= 0:
                continue
            clipped = torch.clamp(centers, 0, threshold)
            q = torch.clamp(torch.round(clipped / scale), 0, levels)
            deq = q * scale
            err = torch.sum(hist * (centers - deq).pow(2)).item()
            if best_err is None or err < best_err:
                best_err = err
                best_bin = end
        return best_bin

    def _best_kl_bin(self, hist, start):
        quant_bins = max(2, int(self.quant_max))
        best_bin = hist.numel()
        best_kl = None
        for end in range(max(start, quant_bins), hist.numel() + 1):
            p = hist[:end].clone()
            if end < hist.numel():
                p[-1] += hist[end:].sum()
            if p.sum() <= 0:
                continue
            q = torch.zeros_like(p)
            boundaries = torch.linspace(0, end, quant_bins + 1)
            for i in range(quant_bins):
                lo = int(torch.floor(boundaries[i]).item())
                hi = int(torch.floor(boundaries[i + 1]).item())
                hi = max(hi, lo + 1)
                hi = min(hi, end)
                mass = p[lo:hi].sum()
                if mass > 0:
                    q[lo:hi] = mass / float(hi - lo)
            p = p / p.sum()
            q = q / torch.clamp(q.sum(), min=1e-12)
            mask = p > 0
            kl = torch.sum(p[mask] * torch.log(p[mask] / torch.clamp(q[mask], min=1e-12))).item()
            if best_kl is None or kl < best_kl:
                best_kl = kl
                best_bin = end
        return best_bin
