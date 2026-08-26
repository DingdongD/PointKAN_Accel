"""ISEDA 移植模型随机推理（供 run_infer_random 统一入口调用）。"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# mini_demo 根目录（含 rapl_inference.py）
_MINI_DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MINI_DEMO_ROOT not in sys.path:
    sys.path.insert(0, _MINI_DEMO_ROOT)

import torch

from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

from . import (
    DGCNN_cls,
    DGCNN_partseg,
    DGCNN_semseg_s3dis,
    PointNet2ClsMSG,
    PointNet2ClsSSG,
    PointNet2PartSegMSG,
    PointNet2PartSegSSG,
    PointNet2SemMSG,
    PointNet2SemSSG,
)


def default_num_classes(model: str) -> int:
    m = model.lower()
    if "sem" in m:
        return 13
    if "part" in m:
        return 50
    return 40


def default_num_points(model: str) -> int:
    m = model.lower()
    if "sem" in m:
        return 1024
    if "part" in m:
        return 2048
    return 1024


def build_model(name: str, num_classes: int, device: torch.device):
    name = name.lower().strip()
    if name == "dgcnn":
        a = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
        net = DGCNN_cls(a, output_channels=num_classes)
    elif name == "pn2_ssg":
        a = SimpleNamespace(num_classes=num_classes, normal=False)
        net = PointNet2ClsSSG(a)
    elif name == "pn2_msg":
        a = SimpleNamespace(num_classes=num_classes, normal=False)
        net = PointNet2ClsMSG(a)
    elif name == "dgcnn_partseg":
        a = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
        net = DGCNN_partseg(a, seg_num_all=num_classes)
    elif name == "dgcnn_semseg":
        a = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5, num_classes=num_classes)
        net = DGCNN_semseg_s3dis(a)
    elif name == "pn2_part_ssg":
        a = SimpleNamespace(num_classes=num_classes, normal=False)
        net = PointNet2PartSegSSG(a)
    elif name == "pn2_part_msg":
        a = SimpleNamespace(num_classes=num_classes, normal=False)
        net = PointNet2PartSegMSG(a)
    elif name == "pn2_sem_ssg":
        a = SimpleNamespace(num_classes=num_classes)
        net = PointNet2SemSSG(a)
    elif name == "pn2_sem_msg":
        a = SimpleNamespace(num_classes=num_classes)
        net = PointNet2SemMSG(a)
    else:
        raise ValueError(f"未知 ISEDA 模型: {name}")
    return net.to(device)


def _forward(net, model: str, x: torch.Tensor, cls16: torch.Tensor):
    m = model.lower()
    if m in ("dgcnn_partseg", "pn2_part_ssg", "pn2_part_msg"):
        out = net(x, cls16)
    else:
        out = net(x)
    return out[0] if isinstance(out, tuple) else out


def _make_cls16_onehot(batch_size: int, device: torch.device, dtype=None):
    if dtype is None:
        dtype = torch.float32
    oh = torch.zeros(batch_size, 16, device=device, dtype=dtype)
    idx = torch.randint(0, 16, (batch_size, 1), device=device)
    oh.scatter_(1, idx, 1.0)
    return oh


def input_tensor(model: str, batch_size: int, num_points: int, device, dtype):
    m = model.lower()
    if m == "dgcnn_semseg":
        return torch.randn(batch_size, 9, num_points, device=device, dtype=dtype)
    if m == "pn2_sem_ssg":
        return torch.randn(batch_size, 9, num_points, device=device, dtype=dtype)
    if m == "pn2_sem_msg":
        return torch.randn(batch_size, 9, num_points, device=device, dtype=dtype)
    if m == "pn2_part_ssg":
        return torch.randn(batch_size, 6, num_points, device=device, dtype=dtype)
    if m == "pn2_part_msg":
        return torch.randn(batch_size, 3, num_points, device=device, dtype=dtype)
    return torch.randn(batch_size, 3, num_points, device=device, dtype=dtype)


def run_benchmark(
    model: str,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    num_points: int | None,
    num_classes: int | None,
    seed: int | None,
):
    if seed is not None:
        torch.manual_seed(seed)
    m = model.lower()
    if num_points is None:
        num_points = default_num_points(m)
    if num_classes is None:
        num_classes = default_num_classes(m)

    if "sem" in m and num_points < 1024:
        import warnings

        warnings.warn(f"{model}: 建议 num_points >= 1024")
    if m in ("pn2_part_ssg", "pn2_part_msg") and num_points < 512:
        import warnings

        warnings.warn(f"{model}: 建议 num_points >= 512")

    net = build_model(m, num_classes, device)
    net.eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(max_batches):
                x = input_tensor(m, batch_size, num_points, device, torch.float32)
                cls16 = _make_cls16_onehot(batch_size, device)

                def _one():
                    _forward(net, m, x, cls16)

                infer_total_s += time_forward_seconds(device, _one)
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)
