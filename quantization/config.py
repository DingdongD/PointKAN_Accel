# W8A8 RTN config for PointKAN-Elite (Round-To-Nearest, 8-bit weight & activation)
from __future__ import absolute_import


class QConfig:
    def __init__(
        self,
        bit=8,
        symmetric=True,
        ch_axis=-1,
        quantizer="FixedFakeQuantize",
        observer="MinMaxObserver",
        observer_kwargs=None,
    ):
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.quantizer = quantizer
        self.observer = observer
        self.observer_kwargs = observer_kwargs or {}

    def __repr__(self):
        return (
            f"QConfig(bit={self.bit}, symmetric={self.symmetric}, ch_axis={self.ch_axis}, "
            f"quantizer={self.quantizer}, observer={self.observer}, observer_kwargs={self.observer_kwargs})"
        )


def get_w8a8_rtn_config(
    w_observer="MinMaxObserver",
    a_observer="MinMaxObserver",
    a_percentile_low=0.001,
    a_percentile_high=0.999,
    a_hist_method="mse",
    a_hist_bins=2048,
    a_hist_start_bin=128,
    a_hist_max_samples=262144,
    w_percentile_low=0.001,
    w_percentile_high=0.999,
):
    """W8A8 RTN: 8-bit weight & activation with configurable observer."""
    w_kwargs = {}
    if w_observer == "PercentileObserver":
        w_kwargs = {"percentile_low": w_percentile_low, "percentile_high": w_percentile_high}
    a_kwargs = {}
    if a_observer == "PercentileObserver":
        a_kwargs = {"percentile_low": a_percentile_low, "percentile_high": a_percentile_high}
    if a_observer == "HistogramObserver":
        a_kwargs = {
            "method": a_hist_method,
            "bins": a_hist_bins,
            "start_bin": a_hist_start_bin,
            "max_samples": a_hist_max_samples,
        }

    w_qconfig = QConfig(bit=8, symmetric=True, ch_axis=-1, quantizer="FixedFakeQuantize", observer=w_observer, observer_kwargs=w_kwargs)
    a_qconfig = QConfig(bit=8, symmetric=True, ch_axis=-1, quantizer="FixedFakeQuantize", observer=a_observer, observer_kwargs=a_kwargs)
    return w_qconfig, a_qconfig


class QuantConfig:
    def __init__(
        self,
        model_name="pointKANElite",
        w_qconfig=None,
        a_qconfig=None,
        w_observer="MinMaxObserver",
        a_observer="MinMaxObserver",
        a_percentile_low=0.001,
        a_percentile_high=0.999,
        a_hist_method="mse",
        a_hist_bins=2048,
        a_hist_start_bin=128,
        a_hist_max_samples=262144,
    ):
        self.model_name = model_name
        self.w_qconfig, self.a_qconfig = w_qconfig, a_qconfig
        if self.w_qconfig is None or self.a_qconfig is None:
            self.w_qconfig, self.a_qconfig = get_w8a8_rtn_config(
                w_observer=w_observer,
                a_observer=a_observer,
                a_percentile_low=a_percentile_low,
                a_percentile_high=a_percentile_high,
                a_hist_method=a_hist_method,
                a_hist_bins=a_hist_bins,
                a_hist_start_bin=a_hist_start_bin,
                a_hist_max_samples=a_hist_max_samples,
            )
