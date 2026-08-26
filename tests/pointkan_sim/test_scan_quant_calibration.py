import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_class_balanced_indices_are_reproducible_and_balanced():
    from quantization.calibration_sampling import class_balanced_indices

    labels = [0] * 10 + [1] * 10 + [2] * 10
    idx_a = class_balanced_indices(labels, calibration_size=12, seed=7)
    idx_b = class_balanced_indices(labels, calibration_size=12, seed=7)

    assert idx_a == idx_b
    counts = {0: 0, 1: 0, 2: 0}
    for idx in idx_a:
        counts[labels[idx]] += 1
    assert counts == {0: 4, 1: 4, 2: 4}


def test_histogram_observer_mse_produces_finite_symmetric_qparams():
    from quantization.observer import HistogramObserver

    obs = HistogramObserver(
        bit=8,
        symmetric=True,
        ch_axis=-1,
        bins=128,
        method="mse",
        start_bin=16,
        max_samples=2048,
    )
    x = torch.cat([torch.linspace(-1.0, 1.0, 1000), torch.tensor([12.0, -11.0])])
    obs(x)
    scale, zero_point = obs.calculate_qparams(obs.min_val, obs.max_val)

    assert torch.isfinite(scale).all()
    assert scale.item() > 0
    assert zero_point.item() == 0
    assert obs.max_val.item() < 12.0


def test_histogram_observer_kl_produces_finite_symmetric_qparams():
    from quantization.observer import HistogramObserver

    obs = HistogramObserver(
        bit=8,
        symmetric=True,
        ch_axis=-1,
        bins=128,
        method="kl",
        start_bin=16,
        max_samples=2048,
    )
    x = torch.randn(2048) * 0.3
    obs(x)
    scale, zero_point = obs.calculate_qparams(obs.min_val, obs.max_val)

    assert torch.isfinite(scale).all()
    assert scale.item() > 0
    assert zero_point.item() == 0
