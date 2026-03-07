# W8A8 RTN config for PointKAN-Elite (Round-To-Nearest, 8-bit weight & activation)
from __future__ import absolute_import


class QConfig:
    def __init__(self, bit=8, symmetric=True, ch_axis=-1, quantizer="FixedFakeQuantize", observer="MinMaxObserver"):
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.quantizer = quantizer
        self.observer = observer

    def __repr__(self):
        return f"QConfig(bit={self.bit}, symmetric={self.symmetric}, ch_axis={self.ch_axis}, quantizer={self.quantizer}, observer={self.observer})"


def get_w8a8_rtn_config():
    """W8A8 RTN: 8-bit weight, 8-bit activation, MinMax observer, FixedFakeQuantize."""
    w_qconfig = QConfig(bit=8, symmetric=True, ch_axis=-1, quantizer="FixedFakeQuantize", observer="MinMaxObserver")
    a_qconfig = QConfig(bit=8, symmetric=True, ch_axis=-1, quantizer="FixedFakeQuantize", observer="MinMaxObserver")
    return w_qconfig, a_qconfig


class QuantConfig:
    def __init__(self, model_name="pointKANElite", w_qconfig=None, a_qconfig=None):
        self.model_name = model_name
        self.w_qconfig, self.a_qconfig = w_qconfig, a_qconfig
        if self.w_qconfig is None or self.a_qconfig is None:
            self.w_qconfig, self.a_qconfig = get_w8a8_rtn_config()
