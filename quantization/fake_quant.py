# Ported from ISEDA26/iseda_quant for PointKAN-Elite (RTN: FixedFakeQuantize only)
from __future__ import absolute_import
import torch
import torch.nn as nn
from .observer import ObserverBase
from .quant_utils import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
)

class QuantizeBase(nn.Module):
    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1, observer_kwargs=None):
        super().__init__()
        self.observer = observer(
            bit=bit,
            symmetric=symmetric,
            ch_axis=ch_axis,
            **(observer_kwargs or {}),
        )
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.drop_prob = 1.0

    def set_bit(self, bit):
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1


class FixedFakeQuantize(QuantizeBase):
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, observer_kwargs=None):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis, observer_kwargs=observer_kwargs)
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))
        self.drop_prob = 1.0

    def forward(self, X, value=None):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.scale.device != X.device or self.zero_point.device != X.device:
                self.scale = self.scale.to(X.device)
                self.zero_point = self.zero_point.to(X.device)
            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(),
                    self.ch_axis, self.quant_min, self.quant_max
                )
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max
                )
        return X
