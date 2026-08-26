import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_scripts.evaluate_rram_weight_noise import (
    NoiseSpec,
    apply_weight_noise_,
    capture_weight_state,
    infer_num_classes,
    parse_float_list,
    restore_weight_state_,
)


def test_parse_float_list_accepts_csv_and_rejects_negative():
    assert parse_float_list("0,0.01, 0.05") == [0.0, 0.01, 0.05]

    with pytest.raises(ValueError, match="non-negative"):
        parse_float_list("0,-0.01")


def test_infer_num_classes_defaults_by_network():
    assert infer_num_classes("pointkan-elite") == 40
    assert infer_num_classes("pointkan-elite-scanobj") == 15
    assert infer_num_classes("pointkan-elite-scanobj", override=7) == 7


def test_zero_sigma_leaves_supported_weights_unchanged():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Conv1d(2, 2, 1))
    before = capture_weight_state(model)

    stats = apply_weight_noise_(model, NoiseSpec(kind="relative", sigma=0.0), seed=7)
    after = capture_weight_state(model)

    assert stats["modules"] == 2
    for name in before:
        assert torch.equal(before[name], after[name])


def test_relative_noise_is_reproducible_for_same_seed():
    model_a = nn.Sequential(nn.Linear(4, 3), nn.Conv1d(2, 2, 1))
    model_b = copy.deepcopy(model_a)

    apply_weight_noise_(model_a, NoiseSpec(kind="relative", sigma=0.05), seed=123)
    apply_weight_noise_(model_b, NoiseSpec(kind="relative", sigma=0.05), seed=123)

    weights_a = capture_weight_state(model_a)
    weights_b = capture_weight_state(model_b)
    for name in weights_a:
        assert torch.equal(weights_a[name], weights_b[name])


def test_restore_weight_state_recovers_original_weights():
    model = nn.Sequential(nn.Linear(4, 3), nn.Conv1d(2, 2, 1))
    original = capture_weight_state(model)

    apply_weight_noise_(model, NoiseSpec(kind="additive_std", sigma=0.10), seed=5)
    restore_weight_state_(model, original)

    restored = capture_weight_state(model)
    for name in original:
        assert torch.equal(original[name], restored[name])


def test_column_noise_changes_each_output_channel_coherently():
    layer = nn.Linear(4, 3, bias=False)
    layer.weight.data.fill_(2.0)
    model = nn.Sequential(layer)

    apply_weight_noise_(model, NoiseSpec(kind="column", sigma=0.05), seed=99)

    row0 = layer.weight[0]
    row1 = layer.weight[1]
    assert torch.allclose(row0, row0[0].expand_as(row0))
    assert torch.allclose(row1, row1[0].expand_as(row1))
    assert not torch.equal(row0, row1)
