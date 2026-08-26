"""
使用 ModelNet40 真实样本与 checkpoint 权重的硬件仿真评估
=========================================================

- 真实样本调度：按 ModelNet40 测试集逐 batch 运行「真实前向」（算准确率）+「一次 PPA 仿真」（累加 cycles/能耗），
  得到与真实推理对应的总周期、总能耗及 per-sample 均值。
- 真实权重：加载 checkpoint 中的模型权重，前向使用真实权重；仿真器侧与 checkpoint 一致的网络结构（LayerDef）
  用于与 checkpoint 一致的 conv/linear 层序列，驱动 NeuroSim 全网络或仿真器中的单层 CIM 估算。

Tokens 说明：
Linear 层：表示该层输入里「位置维」的长度。
LFP：当前 stage 的点数，即 512 / 256 / 128 / 64。
Classifier：全局池化后每样本一个向量，为 1。
Conv1d：不用 tokens，用 length 表示空间长度 L（当前阶段的点数）。

"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from pointkan_sim.integration.cim_full_context import CIM_LAYER_INDEX, ACC_CIM_KEY_PRELOADED

# 工程根与 classification 目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CLASSIFICATION_DIR = _PROJECT_ROOT / "PointKAN-elite" / "classification_ModelNet40"
_SCANOBJ_DIR = _PROJECT_ROOT / "PointKAN-elite" / "classification_ScanObjectNN"
_PARTSEG_DIR = _PROJECT_ROOT / "PointKAN-elite" / "part_segmentation"
_ISEDA26_ROOT = Path("/workspace/ISEDA26")
_PARTSEG_NUM_PART = 50
_PARTSEG_NUM_CLASSES = 16


_POINTNET2_CLS_CKPT = _ISEDA26_ROOT / "ckpts" / "modelnet_pointnet2ssg_p1024_k20" / "checkpoints" / "best_model.pth"
_POINTNET2_PARTSEG_CKPT = _ISEDA26_ROOT / "ckpts" / "shapenet_pointnet2ssg_p1024_k20" / "checkpoints" / "best_model.pth"
_POINTNET2_SEMSEG_CKPT = _ISEDA26_ROOT / "ckpts" / "s3dis_pointnet2ssg_p4096_k40" / "checkpoints" / "best_model.pth"


def _network_name(network: Optional[str]) -> str:
    return str(network or "").strip().lower()


def _ensure_common_project_path() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    _pointnet2_lib = _PROJECT_ROOT / "pointnet2_ops_lib"
    if _pointnet2_lib.exists() and str(_pointnet2_lib) not in sys.path:
        sys.path.insert(0, str(_pointnet2_lib))


def _ensure_classification_path() -> None:
    _ensure_common_project_path()
    _elite_root = _CLASSIFICATION_DIR.parent
    if str(_elite_root) not in sys.path:
        sys.path.insert(0, str(_elite_root))
    if str(_CLASSIFICATION_DIR) not in sys.path:
        sys.path.insert(0, str(_CLASSIFICATION_DIR))


def _ensure_scanobj_path() -> None:
    _ensure_common_project_path()
    _elite_root = _SCANOBJ_DIR.parent
    if str(_elite_root) not in sys.path:
        sys.path.insert(0, str(_elite_root))
    if str(_SCANOBJ_DIR) not in sys.path:
        sys.path.insert(0, str(_SCANOBJ_DIR))


def _ensure_partseg_path() -> None:
    _ensure_common_project_path()
    if str(_PARTSEG_DIR) not in sys.path:
        sys.path.insert(0, str(_PARTSEG_DIR))
    _elite_root = _PARTSEG_DIR.parent
    if str(_elite_root) not in sys.path:
        sys.path.insert(0, str(_elite_root))


def _ensure_iseda26_path() -> None:
    if str(_ISEDA26_ROOT) not in sys.path:
        sys.path.insert(0, str(_ISEDA26_ROOT))


def _is_scanobj_network(network: Optional[str]) -> bool:
    return _network_name(network) == "pointkan-elite-scanobj"


def _is_pointnet2_cls_network(network: Optional[str]) -> bool:
    return _network_name(network) == "pointnet2-cls-ssg"


def _is_pointnet2_partseg_network(network: Optional[str]) -> bool:
    return _network_name(network) == "pointnet2-partseg-ssg"


def _is_pointnet2_semseg_network(network: Optional[str]) -> bool:
    return _network_name(network) == "pointnet2-sem-ssg"


def _is_pointnet2_network(network: Optional[str]) -> bool:
    name = _network_name(network)
    return name in ("pointnet2-cls-ssg", "pointnet2-partseg-ssg", "pointnet2-sem-ssg")


def _is_partseg_network(network: Optional[str]) -> bool:
    return _network_name(network) in ("pointkan-elite-partseg", "pointnet2-partseg-ssg")


def _is_semseg_network(network: Optional[str]) -> bool:
    return _is_pointnet2_semseg_network(network)


def _import_fresh(module_name: str):
    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(module_name + "."):
            sys.modules.pop(key, None)
    return importlib.import_module(module_name)


def _resolve_checkpoint_file(checkpoint_path: str, candidates: Tuple[str, ...]) -> Path:
    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        for name in candidates:
            candidate = ckpt / name
            if candidate.exists():
                return candidate
    return ckpt


def _resolve_partseg_checkpoint_path(checkpoint_path: str) -> Path:
    return _resolve_checkpoint_file(checkpoint_path, ("best_insiou_model.pth", "best_checkpoint.pth", "best_model.pth", "best_acc_model.pth", "checkpoint.pth"))


def _extract_state_dict(ckpt_data: Any) -> Dict[str, Any]:
    if isinstance(ckpt_data, dict):
        for key in ("net", "model", "state_dict"):
            if key in ckpt_data and isinstance(ckpt_data[key], dict) and ckpt_data[key]:
                return ckpt_data[key]
        if ckpt_data:
            return ckpt_data
    return ckpt_data


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, dict) and state:
        first = next(iter(state.keys()))
        if isinstance(first, str) and first.startswith("module."):
            return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_state_dict_into_model(model: Any, checkpoint_path: str, device: str, *, strict: bool = True, candidates: Tuple[str, ...] = ("best_checkpoint.pth", "best_model.pth", "checkpoint.pth")) -> Path:
    import torch
    ckpt = _resolve_checkpoint_file(checkpoint_path, candidates)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt_data = torch.load(str(ckpt), map_location=device)
    state = _strip_module_prefix(_extract_state_dict(ckpt_data))
    model.load_state_dict(state, strict=strict)
    return ckpt


def _primary_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


@dataclass
class ConvLinearLayerInfo:
    """单层 conv 或 linear 的信息（来自真实前向一次收集）。"""
    name: str
    layer_type: str  # "conv" | "fc"
    # conv
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: int = 1
    length: int = 0  # 空间长度 (L)，即输入序列长度（点云中为当前 stage 的点数）
    # fc (Linear)
    in_features: int = 0
    out_features: int = 0
    tokens: int = 0  # Linear 输入的有效位置数；高维输入按除 batch/feature 外维度乘积展开
    # 真实激活 shape（便于 baseline/Scale-Sim 对齐）
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    # 真实数据量（字节），用于按层 indata/rdata 记录与仿真侧对齐
    indata_bytes: float = 0.0
    rdata_bytes: float = 0.0


@dataclass
class NonlinearActivationInfo:
    """非线性/归一化/池化等算子的前向激活信息（hook 收集）。"""
    name: str
    op_type: str  # "batchnorm1d" | "batchnorm2d" | "layernorm" | "relu" | "s-pool" | "adaptive_max_pool1d" | ...
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    numel: int
    bytes_fp32: float = 0.0  # 输出 numel * 4（便于与 indata 对照）
    input_numel: int = 0  # 输入元素数（用于 SPool 等看输入激活大小）
    input_bytes_fp32: float = 0.0  # 输入 FP32 字节数


@dataclass
class RealEvalResult:
    """真实数据评估结果：准确率 + 按 ModelNet40 调度后的 PPA。"""
    num_samples: int = 0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    test_loss: float = 0.0
    total_cycles: int = 0
    total_energy_nj: float = 0.0
    total_latency_ns: float = 0.0
    total_latency_sum_ns: float = 0.0
    total_latency_critical_path_ns: float = 0.0
    cycles_per_sample: float = 0.0
    energy_per_sample_nj: float = 0.0
    latency_per_sample_ns: float = 0.0
    latency_sum_per_sample_ns: float = 0.0
    critical_path_latency_per_sample_ns: float = 0.0
    task_type: str = "classification"
    instance_miou: float = 0.0
    conv_linear_from_model: List[ConvLinearLayerInfo] = field(default_factory=list)
    sim_once: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None


def load_model_and_dataset(
    checkpoint_path: str,
    network: str = "pointkan-elite",
    num_points: int = 1024,
    num_classes: int = 40,
    batch_size: int = 32,
    workers: int = 0,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """
    加载真实模型与测试集。

    Returns:
        (net, test_loader, criterion) 或 失败时抛出。
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    net_name = _network_name(network)

    if net_name == "pointkan-elite-partseg":
        _ensure_partseg_path()
        PartNormalDataset = _import_fresh("util.data_util").PartNormalDataset
        partseg_models = _import_fresh("model")
        net = partseg_models.__dict__["pointKANElite"](num_classes=_PARTSEG_NUM_PART)
        _load_state_dict_into_model(
            net,
            checkpoint_path,
            device,
            candidates=("best_insiou_model.pth", "best_checkpoint.pth", "best_model.pth", "best_acc_model.pth", "checkpoint.pth"),
        )
        net = net.to(device)
        net.eval()
        test_dataset = PartNormalDataset(npoints=num_points, split="test", normalize=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
        return net, test_loader, None

    if net_name == "pointnet2-partseg-ssg":
        _ensure_iseda26_path()
        from data_config import ShapeNetPart
        from pointnet2_utils import PointNet2PartSegSSG

        pt_args = SimpleNamespace(normal=False, num_classes=num_classes)
        net = PointNet2PartSegSSG(pt_args)
        _load_state_dict_into_model(
            net,
            checkpoint_path,
            device,
            candidates=("best_model.pth", "checkpoint.pth"),
        )
        net = net.to(device)
        net.eval()
        test_dataset = ShapeNetPart(num_points=num_points, partition="test", class_choice=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
        return net, test_loader, None

    if net_name == "pointkan-elite-scanobj":
        _ensure_scanobj_path()
        ScanObjectNN = _import_fresh("ScanObjectNN").ScanObjectNN
        scan_models = _import_fresh("models")
        net = scan_models.__dict__["pointKANElite"](num_classes=num_classes)
        _load_state_dict_into_model(
            net,
            checkpoint_path,
            device,
            candidates=("best_checkpoint.pth", "best_model.pth", "checkpoint.pth"),
        )
        net = net.to(device)
        net.eval()
        test_dataset = ScanObjectNN(partition="test", num_points=num_points)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
        return net, test_loader, F.cross_entropy

    if net_name == "pointnet2-cls-ssg":
        _ensure_iseda26_path()
        from data_config import ModelNet40
        from pointnet2_utils import PointNet2ClsSSG

        pt_args = SimpleNamespace(normal=False, num_classes=num_classes)
        net = PointNet2ClsSSG(pt_args)
        _load_state_dict_into_model(
            net,
            checkpoint_path,
            device,
            candidates=("best_model.pth", "checkpoint.pth"),
        )
        net = net.to(device)
        net.eval()
        test_dataset = ModelNet40(num_points=num_points, partition="test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
        return net, test_loader, F.cross_entropy

    if net_name == "pointnet2-sem-ssg":
        _ensure_iseda26_path()
        from data_config import S3DIS
        from pointnet2_utils import PointNet2SemSSG

        pt_args = SimpleNamespace(num_classes=num_classes)
        net = PointNet2SemSSG(pt_args)
        _load_state_dict_into_model(
            net,
            checkpoint_path,
            device,
            candidates=("best_model.pth", "checkpoint.pth"),
        )
        net = net.to(device)
        net.eval()
        test_dataset = S3DIS(num_points=num_points, partition="test", test_area="1")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
        return net, test_loader, F.cross_entropy

    _ensure_classification_path()
    from data import ModelNet40
    models = _import_fresh("models")
    from utils import cal_loss

    net = models.__dict__["pointKANElite"]()
    _load_state_dict_into_model(
        net,
        checkpoint_path,
        device,
        candidates=("best_checkpoint.pth", "best_model.pth", "checkpoint.pth"),
    )
    net = net.to(device)
    net.eval()

    test_dataset = ModelNet40(partition="test", num_points=num_points)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    criterion = cal_loss
    return net, test_loader, criterion


def _run_model_forward(
    net: Any,
    device: str,
    network: str,
    num_points: int,
    batch_size: int = 2,
    batch: Any = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    net_name = _network_name(network)

    if net_name == "pointkan-elite-partseg":
        _ensure_partseg_path()
        from util.util import to_categorical

        if batch is not None:
            points, label, _target, norm_plt = batch
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
        else:
            points = torch.rand(batch_size, 3, num_points, device=device)
            norm_plt = torch.rand(batch_size, 3, num_points, device=device)
            label = torch.zeros(batch_size, dtype=torch.long, device=device)
        onehot = to_categorical(label, _PARTSEG_NUM_CLASSES)
        return net(points, norm_plt, onehot)

    if net_name == "pointnet2-partseg-ssg":
        if batch is not None:
            points, label, _target = batch
            points = points.float().to(device).transpose(2, 1)
            label = label.view(-1).long().to(device)
        else:
            points = torch.rand(batch_size, 3, num_points, device=device)
            label = torch.zeros(batch_size, dtype=torch.long, device=device)
        onehot = F.one_hot(label, num_classes=_PARTSEG_NUM_CLASSES).float()
        return net(points, onehot)

    if net_name == "pointnet2-sem-ssg":
        if batch is not None:
            batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            batch_x = batch_x.float().to(device)
            if batch_x.dim() == 3:
                batch_x = batch_x.permute(0, 2, 1)
        else:
            batch_x = torch.rand(batch_size, 9, num_points, device=device)
        return net(batch_x)

    if batch is not None:
        batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_x = batch_x.float().to(device)
        if batch_x.dim() == 3 and batch_x.shape[-1] >= 3:
            batch_x = batch_x.permute(0, 2, 1)
    else:
        batch_x = torch.rand(batch_size, 3, num_points, device=device)
    return net(batch_x)


def collect_conv_linear_from_forward(
    net: Any,
    device: str,
    num_points: int,
    batch_size: int = 2,
    input_bit: int = 8,
    weight_bit: int = 8,
    network: str = "pointkan-elite",
    batch: Any = None,
) -> List[ConvLinearLayerInfo]:
    """
    运行一次前向，用 hook 收集所有 Conv1d / Conv2d / Linear 的调用顺序、输入/权重 shape 及真实数据量（字节），
    返回与真实权重、真实激活一致的层序列表，用于 NeuroSim 全网络与 real_data_overrides。
    """
    import torch
    import torch.nn as nn

    order: List[ConvLinearLayerInfo] = []
    inv_bit = 1.0 / 8.0 * input_bit
    wv_bit = 1.0 / 8.0 * weight_bit

    def _linear_tokens(x) -> int:
        if x is None:
            return 1
        if x.dim() <= 2:
            return 1
        tokens = 1
        for dim in x.shape[1:-1]:
            tokens *= int(dim)
        return max(1, int(tokens))

    def _shape_tuple(t) -> Tuple[int, ...]:
        return tuple(int(v) for v in t.shape) if t is not None else tuple()

    def _conv_hook(name: str):
        def hook(module, inp, out):
            x = inp[0]
            w = getattr(module, "weight", None)
            if x is None or w is None:
                return
            x = x.detach()
            w = w.detach()
            indata_bytes = float(x.numel()) * inv_bit
            rdata_bytes = float(w.numel()) * wv_bit
            # Conv1d: x [B, C_in, L], w [C_out, C_in, K]
            if x.dim() == 3 and w.dim() == 3:
                _, c_in, length = x.shape
                c_out = w.shape[0]
                k = w.shape[2]
                order.append(ConvLinearLayerInfo(
                    name=name, layer_type="conv",
                    in_channels=c_in, out_channels=c_out, kernel_size=k, length=length,
                    input_shape=_shape_tuple(x), output_shape=_shape_tuple(out.detach() if out is not None else None),
                    indata_bytes=indata_bytes, rdata_bytes=rdata_bytes,
                ))
            # Conv2d: x [B, C_in, H, W], w [C_out, C_in, K_h, K_w]
            # PointNet++ shared MLP uses kernel=1 and is modeled as an equivalent
            # conv sequence with length = H * W (see pointnet2_net.py comments).
            elif x.dim() == 4 and w.dim() == 4:
                _, c_in, h, width = x.shape
                c_out = w.shape[0]
                k_h = int(w.shape[2])
                k_w = int(w.shape[3])
                eff_kernel = max(1, k_h * k_w)
                eff_length = max(1, int(h) * int(width))
                order.append(ConvLinearLayerInfo(
                    name=name, layer_type="conv",
                    in_channels=c_in, out_channels=c_out, kernel_size=eff_kernel, length=eff_length,
                    input_shape=_shape_tuple(x), output_shape=_shape_tuple(out.detach() if out is not None else None),
                    indata_bytes=indata_bytes, rdata_bytes=rdata_bytes,
                ))
            # Linear: 允许 [B, in_f] / [B, T, in_f] / [B, ..., in_f]，tokens 取除 batch/feature 外维度乘积
            elif w.dim() == 2:
                in_f = w.shape[1]
                out_f = w.shape[0]
                tokens = _linear_tokens(x)
                order.append(ConvLinearLayerInfo(
                    name=name, layer_type="fc",
                    in_features=in_f, out_features=out_f, tokens=tokens,
                    input_shape=_shape_tuple(x), output_shape=_shape_tuple(out.detach() if out is not None else None),
                    indata_bytes=indata_bytes, rdata_bytes=rdata_bytes,
                ))
        return hook

    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_conv_hook(name)))

    try:
        with torch.no_grad():
            _ = _run_model_forward(
                net,
                device,
                network=network,
                num_points=num_points,
                batch_size=batch_size,
                batch=batch,
            )
    finally:
        for h in hooks:
            h.remove()

    return order


def collect_nonlinear_activations_from_forward(
    net: Any,
    device: str,
    num_points: int,
    batch_size: int = 2,
    network: str = "pointkan-elite",
    batch: Any = None,
) -> List[NonlinearActivationInfo]:
    """
    运行一次前向，用 hook 收集 BatchNorm、LayerNorm、ReLU、池化等非线性算子的
    输入/输出激活 shape 与 numel，便于与 LayerDef 对照或分析数据流。
    """
    import torch
    import torch.nn as nn

    order: List[NonlinearActivationInfo] = []

    def _make_hook(name: str, op_type: str):
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            x = inp[0] if inp and inp[0] is not None else None
            y = out
            if x is not None:
                x = x.detach()
            if y is not None:
                y = y.detach()
            in_shape = tuple(x.shape) if x is not None else ()
            out_shape = tuple(y.shape) if y is not None else ()
            numel = int(y.numel()) if y is not None else 0
            bytes_fp32 = numel * 4.0
            input_numel = int(x.numel()) if x is not None else 0
            input_bytes_fp32 = input_numel * 4.0
            order.append(NonlinearActivationInfo(
                name=name,
                op_type=op_type,
                input_shape=in_shape,
                output_shape=out_shape,
                numel=numel,
                bytes_fp32=bytes_fp32,
                input_numel=input_numel,
                input_bytes_fp32=input_bytes_fp32,
            ))
        return hook

    def _make_gam_spool_hook(name: str):
        """Hook for Geometric_Affine_Module: forward 返回 (new_xyz, new_points, center)，
        new_points 为 S-Pool 输入，center 为 S-Pool 输出。"""
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                return
            new_points = out[1].detach() if out[1] is not None else None
            center = out[2].detach() if out[2] is not None else None
            in_shape = tuple(new_points.shape) if new_points is not None else ()
            out_shape = tuple(center.shape) if center is not None else ()
            numel = int(center.numel()) if center is not None else 0
            bytes_fp32 = numel * 4.0
            input_numel = int(new_points.numel()) if new_points is not None else 0
            input_bytes_fp32 = input_numel * 4.0
            order.append(NonlinearActivationInfo(
                name=name,
                op_type="s-pool",
                input_shape=in_shape,
                output_shape=out_shape,
                numel=numel,
                bytes_fp32=bytes_fp32,
                input_numel=input_numel,
                input_bytes_fp32=input_bytes_fp32,
            ))
        return hook

    _OP_MAP = [
        (nn.BatchNorm1d, "batchnorm1d"),
        (nn.BatchNorm2d, "batchnorm2d"),
        (nn.LayerNorm, "layernorm"),
        (nn.ReLU, "relu"),
        (nn.LeakyReLU, "leaky_relu"),
        (nn.RReLU, "rrelu"),
        (nn.AdaptiveMaxPool1d, "adaptive_max_pool1d"),
        (nn.AdaptiveAvgPool1d, "adaptive_avg_pool1d"),
        (nn.MaxPool1d, "max_pool1d"),
        (nn.AvgPool1d, "avg_pool1d"),
    ]

    hooks = []
    for name, m in net.named_modules():
        # PointKAN-elite GAM: forward 返回 (new_xyz, new_points, center)，捕获 S-Pool 的输入 new_points 与输出 center
        if type(m).__name__ == "Geometric_Affine_Module":
            hooks.append(m.register_forward_hook(_make_gam_spool_hook(name)))
            continue
        for module_cls, op_type in _OP_MAP:
            if isinstance(m, module_cls):
                hooks.append(m.register_forward_hook(_make_hook(name, op_type)))
                break

    try:
        with torch.no_grad():
            _ = _run_model_forward(
                net,
                device,
                network=network,
                num_points=num_points,
                batch_size=batch_size,
                batch=batch,
            )
    finally:
        for h in hooks:
            h.remove()

    return order


def log_nonlinear_activation_shapes(
    nonlinear_list: List[NonlinearActivationInfo],
    log_info_fn: Any = None,
) -> None:
    """
    打印经 hook 收集的 BatchNorm / LayerNorm / ReLU / 池化 等算子的输入与输出激活 shape。
    """
    if log_info_fn is None:
        log_info_fn = print
    if not nonlinear_list:
        log_info_fn("========== 非线性算子激活（BatchNorm / LayerNorm / ReLU / 池化 等）==========")
        log_info_fn("  未收集到（未注册 hook 或前向未经过）。")
        return
    log_info_fn("========== 非线性算子激活（BatchNorm / LayerNorm / ReLU / 池化 等）==========")
    log_info_fn(f"  共 {len(nonlinear_list)} 个算子（按前向顺序）")
    for idx, info in enumerate(nonlinear_list):
        in_s = "×".join(str(d) for d in info.input_shape) if info.input_shape else "?"
        out_s = "×".join(str(d) for d in info.output_shape) if info.output_shape else "?"
        in_size = f" ({info.input_numel} elem, {info.input_bytes_fp32:.0f} B)" if info.input_numel else ""
        out_size = f" ({info.numel} elem, {info.bytes_fp32:.0f} B)"
        log_info_fn(
            f"  [{idx}] {info.name}  {info.op_type}  in={in_s}{in_size}  out={out_s}{out_size}"
        )
    log_info_fn("  in/out = 输入/输出 shape；括号内为输入/输出元素数与 FP32 字节数（SPool 等可见输入激活大小）。")


def collect_conv_linear_tensors_from_forward(
    net: Any,
    device: str,
    batch_x: Any,
    input_bit: int = 8,
    weight_bit: int = 8,
) -> List[Tuple[ConvLinearLayerInfo, Any, Any]]:
    """
    运行一次前向（使用给定输入 batch_x），用 hook 收集所有 Conv1d/Linear 的
    **层信息 + 该层输入激活张量 + 该层权重张量**，用于导出真实权重/激活到 NeuroSim CSV。

    Args:
        net: 已加载权重的模型（如 PointKAN-elite）
        device: 运行设备
        batch_x: 输入 [B, 3, num_points]，一般为从 DataLoader 取的一个 batch
        input_bit / weight_bit: 仅用于计算 indata_bytes / rdata_bytes

    Returns:
        List[(ConvLinearLayerInfo, input_tensor, weight_tensor)]，顺序与前向一致。
        input_tensor / weight_tensor 已 detach，可在 CPU 上 .numpy() 后写入 CSV。
    """
    import torch
    import torch.nn as nn

    order: List[Tuple[ConvLinearLayerInfo, Any, Any]] = []
    inv_bit = 1.0 / 8.0 * input_bit
    wv_bit = 1.0 / 8.0 * weight_bit

    def _linear_tokens(x) -> int:
        if x is None or x.dim() <= 2:
            return 1
        tokens = 1
        for dim in x.shape[1:-1]:
            tokens *= int(dim)
        return max(1, int(tokens))

    def _shape_tuple(t) -> Tuple[int, ...]:
        return tuple(int(v) for v in t.shape) if t is not None else tuple()

    def _hook(name: str):
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            x = inp[0]
            w = getattr(module, "weight", None)
            if x is None or w is None:
                return
            x = x.detach()
            w = w.detach()
            indata_bytes = float(x.numel()) * inv_bit
            rdata_bytes = float(w.numel()) * wv_bit
            if x.dim() == 3 and w.dim() == 3:
                _, c_in, length = x.shape
                c_out = w.shape[0]
                k = w.shape[2]
                info = ConvLinearLayerInfo(
                    name=name, layer_type="conv",
                    in_channels=c_in, out_channels=c_out, kernel_size=k, length=length,
                    input_shape=_shape_tuple(x), output_shape=_shape_tuple(out.detach() if out is not None else None),
                    indata_bytes=indata_bytes, rdata_bytes=rdata_bytes,
                )
                order.append((info, x, w))
            elif w.dim() == 2:
                in_f = w.shape[1]
                out_f = w.shape[0]
                tokens = _linear_tokens(x)
                info = ConvLinearLayerInfo(
                    name=name, layer_type="fc",
                    in_features=in_f, out_features=out_f, tokens=tokens,
                    input_shape=_shape_tuple(x), output_shape=_shape_tuple(out.detach() if out is not None else None),
                    indata_bytes=indata_bytes, rdata_bytes=rdata_bytes,
                )
                order.append((info, x, w))
        return hook

    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(_hook(name)))

    try:
        with torch.no_grad():
            batch_x = batch_x.permute(0, 2, 1)
            _ = net(batch_x.to(device))
    finally:
        for h in hooks:
            h.remove()

    return order


def format_activation_weight_shapes(info: "ConvLinearLayerInfo") -> Tuple[str, str]:
    """返回 (activation_shape_str, weight_shape_str) 便于日志/报告对照。"""
    if info.input_shape:
        act = "[" + ", ".join(str(v) for v in info.input_shape) + "]"
    elif info.layer_type == "conv":
        act = f"[B, {info.in_channels}, {info.length}]"
    else:
        act = f"[B, {info.tokens}, {info.in_features}]" if info.tokens > 1 else f"[B, {info.in_features}]"
    if info.layer_type == "conv":
        w = f"[{info.out_channels}, {info.in_channels}, {info.kernel_size}]"
    else:
        w = f"[{info.out_features}, {info.in_features}]"
    return act, w


def real_data_overrides_from_conv_linear_list(
    conv_linear_list: List[ConvLinearLayerInfo],
) -> List[Dict[str, float]]:
    """
    从真实前向收集的 conv/linear 列表生成 real_data_overrides（每层 indata/rdata 字节）。
    """
    return [
        {"real_indata_bytes": info.indata_bytes, "real_rdata_bytes": info.rdata_bytes}
        for info in conv_linear_list
    ]


def _layerdef_conv_linear_shapes(layers: List[Any]) -> List[Tuple[str, int, str, str]]:
    """返回 LayerDef 中所有 conv1d/linear 的 (layer_name, op_idx, op_type, shape_desc)。"""
    result: List[Tuple[str, int, str, str]] = []
    for layer in layers:
        if not layer.operators:
            continue
        for op_idx, op_ref in enumerate(layer.operators):
            if op_ref.op_type not in ("conv1d", "linear"):
                continue
            p = op_ref.params
            if op_ref.op_type == "conv1d":
                shape = f"in_ch={p.get('in_channels', 0)}, out_ch={p.get('out_channels', 0)}, L={p.get('num_points', 0)}, K={p.get('kernel_size', 1)}"
            else:
                shape = f"in_f={p.get('in_features', 0)}, out_f={p.get('out_features', 0)}"
            result.append((layer.name, op_idx, op_ref.op_type, shape))
    return result


def log_conv_linear_shapes(
    layers: List[Any],
    conv_linear_list: List[ConvLinearLayerInfo],
    real_data_overrides: Optional[List[Dict[str, float]]] = None,
    log_info_fn: Any = None,
) -> None:
    """
    打印 LayerDef 与真实前向的 conv/linear 对照：每层输入激活与权重的 shape 及 indata/rdata 字节。
    - indata：该层输入激活数据量（字节），Conv1d 对应 [B,C_in,L]、Linear 对应 [B,in_f] 或 [B,T,in_f]。
    - rdata：该层权重量（字节），Conv1d 对应 [C_out,C_in,K]、Linear 对应 [out_f,in_f]。
    若真实前向层数(36)与 LayerDef 层数(20)不一致，会打印说明：真实前向包含模型中所有 Conv1d/Linear（含 KAN 内部），
    LayerDef 仅包含 CIM 映射的 20 层；当前 overrides 按索引前 N 条对应到 LayerDef，执行顺序可能不一致。
    """
    if log_info_fn is None:
        log_info_fn = print
    layerdef_entries = _layerdef_conv_linear_shapes(layers)
    overrides = real_data_overrides or []

    log_info_fn("========== LayerDef 中的 conv/linear（CIM 映射层）==========")
    log_info_fn(f"  共 {len(layerdef_entries)} 层（仅 CIM 映射的 conv1d/linear，不含 KAN 内部等）")
    for idx, (lname, op_idx, op_type, shape) in enumerate(layerdef_entries):
        extra = ""
        if idx < len(overrides) and isinstance(overrides[idx], dict):
            o = overrides[idx]
            extra = f"  indata_bytes={o.get('real_indata_bytes', 0):.0f}  rdata_bytes={o.get('real_rdata_bytes', 0):.0f}"
        log_info_fn(f"  [{idx}] {lname} op_idx={op_idx} {op_type}  {shape}{extra}")

    log_info_fn("========== 真实前向收集的 conv/linear（模型中所有 nn.Conv1d/nn.Linear）==========")
    log_info_fn(f"  共 {len(conv_linear_list)} 层")
    for idx, info in enumerate(conv_linear_list):
        if info.layer_type == "conv":
            shape = f"in_ch={info.in_channels}, out_ch={info.out_channels}, L={info.length}, K={info.kernel_size}"
        else:
            shape = f"in_f={info.in_features}, out_f={info.out_features}, tokens={info.tokens}"
        log_info_fn(f"  [{idx}] {info.name} {info.layer_type}  {shape}  indata_bytes={info.indata_bytes:.0f}  rdata_bytes={info.rdata_bytes:.0f}")

    if len(conv_linear_list) != len(layerdef_entries):
        log_info_fn("---------- 说明 ----------")
        log_info_fn(f"  真实前向 {len(conv_linear_list)} 层 = 模型中所有 Conv1d/Linear（含 KAN、GAM 等子模块内）；")
        log_info_fn(f"  LayerDef {len(layerdef_entries)} 层 = 仅 CIM 仿真映射的 conv1d/linear（embedding、LFP transfer、GFP、classifier）。")
        log_info_fn("  当前 real_data_overrides 按索引将「前 N 个真实层」对应到 LayerDef 的 N 层，若前向顺序与 LayerDef 不一致则需按名称对齐。")
        log_info_fn("  indata = 输入激活字节数；rdata = 权重复用/读取字节数，符合 Conv1d/Linear 的矩阵乘数据量语义。")


def run_real_inference_and_sim_over_dataset(
    net: Any,
    test_loader: Any,
    criterion: Any,
    device: str,
    layers: List[Any],
    hw_config: Any,
    cim_full_context: Optional[Dict[str, Any]],
    enable_layer_buffer: bool = True,
    activation_shapes: Optional[List[Any]] = None,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    repo_root: Optional[str] = None,
    real_conv_linear_list: Optional[List[Any]] = None,
    network: str = "pointkan-elite",
) -> RealEvalResult:
    """
    在真实测试集上：真实前向按 batch 算任务指标；仿真器按样本串行，每次只处理一个样本。
    """
    import numpy as np
    import sklearn.metrics as metrics
    from ..main import simulate_network

    result = RealEvalResult()
    test_true = []
    test_pred = []
    total_loss = 0.0
    total_cycles = 0
    total_energy_nj = 0.0
    total_latency_ns = 0.0
    total_latency_sum_ns = 0.0
    total_latency_critical_path_ns = 0.0
    last_sim_once = None
    num_samples = 0
    sim_samples_done = 0
    batch_count = 0
    is_partseg = _is_partseg_network(network)
    is_semseg = _is_semseg_network(network)
    result.task_type = "partseg" if is_partseg else ("semseg" if is_semseg else "classification")
    shape_ious_sum = 0.0

    net.eval()
    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if max_samples is not None and int(max_samples) > 0 and sim_samples_done >= int(max_samples):
            break

        if is_partseg:
            import torch
            import torch.nn.functional as F

            _ensure_partseg_path()
            from util.util import to_categorical, compute_overall_iou

            if _is_pointnet2_partseg_network(network):
                points, label, target = batch
                points = points.float().to(device).transpose(2, 1)
                label = label.long().to(device).view(-1)
                target = target.long().to(device)
                onehot = to_categorical(label, _PARTSEG_NUM_CLASSES)
                B = points.size(0)

                with torch.no_grad():
                    seg_pred = _primary_output(net(points, onehot))
                    loss = F.cross_entropy(seg_pred, target)

                seg_pred_for_iou = seg_pred.transpose(1, 2).contiguous()
                seg_pred_flat = seg_pred_for_iou.view(-1, _PARTSEG_NUM_PART)
                target_flat = target.reshape(-1)
                batch_shapeious = compute_overall_iou(seg_pred_for_iou, target, _PARTSEG_NUM_PART)
            else:
                points, label, target, norm_plt = batch
                points = points.float().to(device).transpose(2, 1)
                norm_plt = norm_plt.float().to(device).transpose(2, 1)
                label = label.squeeze(1).long().to(device)
                target = target.long().to(device)
                onehot = to_categorical(label, _PARTSEG_NUM_CLASSES)
                B = points.size(0)

                with torch.no_grad():
                    seg_pred = _primary_output(net(points, norm_plt, onehot))
                    seg_pred_flat = seg_pred.contiguous().view(-1, _PARTSEG_NUM_PART)
                    target_flat = target.view(-1, 1)[:, 0]
                    loss = F.nll_loss(seg_pred_flat, target_flat)

                batch_shapeious = compute_overall_iou(seg_pred, target, _PARTSEG_NUM_PART)

            shape_ious_sum += float(sum(batch_shapeious))
            pred_choice = seg_pred_flat.max(1)[1]
            total_loss += loss.item() * B
            test_true.append(np.atleast_1d(target_flat.cpu().numpy()))
            test_pred.append(np.atleast_1d(pred_choice.cpu().numpy()))
            num_samples += B
        elif is_semseg:
            import torch
            import torch.nn.functional as F

            data, target = batch
            data = data.float().to(device).permute(0, 2, 1)
            target = target.long().to(device)
            B = data.size(0)

            with torch.no_grad():
                logits = _primary_output(net(data))
                loss = F.cross_entropy(logits, target)
            total_loss += loss.item() * B
            preds = logits.max(dim=1)[1]
            test_true.append(np.atleast_1d(target.reshape(-1).cpu().numpy()))
            test_pred.append(np.atleast_1d(preds.reshape(-1).cpu().numpy()))
            num_samples += B
        else:
            import torch

            data, label = batch
            data = data.to(device)
            label = label.to(device).view(-1)
            data = data.permute(0, 2, 1)
            B = data.size(0)

            with torch.no_grad():
                logits = _primary_output(net(data))
                loss = criterion(logits, label)
            total_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(np.atleast_1d(label.cpu().numpy()))
            test_pred.append(np.atleast_1d(preds.cpu().numpy()))
            num_samples += B

        # 仿真器按样本串行（但 PPA 与样本内容无关，因此这里本质是“重复累加”）
        sim_repeat = B
        if max_samples is not None and int(max_samples) > 0:
            sim_repeat = min(sim_repeat, max(0, int(max_samples) - sim_samples_done))
        for _ in range(sim_repeat):
            # Important: reset full-network CIM cursor per sample.
            # Otherwise CIM_LAYER_INDEX carries over across samples and eventually
            # falls out of range, causing unintended fallback path for Conv/Linear.
            sim_cim_ctx = dict(cim_full_context) if cim_full_context else None
            if sim_cim_ctx is not None:
                if CIM_LAYER_INDEX in sim_cim_ctx:
                    sim_cim_ctx[CIM_LAYER_INDEX] = 0
                sim_cim_ctx.pop(ACC_CIM_KEY_PRELOADED, None)
            sim_result = simulate_network(
                layers,
                hw_config,
                show_layers=False,
                cim_full_context=sim_cim_ctx,
                activation_shapes=activation_shapes,
                enable_layer_buffer=enable_layer_buffer,
                show_shapes=False,
                repo_root=repo_root,
                real_conv_linear_list=real_conv_linear_list,
            )
            last_sim_once = sim_result
            total_cycles += sim_result.get("total_cycles", 0) or 0
            total_energy_nj += sim_result.get("total_energy_nJ", 0.0) or 0.0
            total_latency_critical_path_ns += (
                sim_result.get("total_latency_critical_path_ns", 0.0)
                or sim_result.get("latency_critical_path_ns", 0.0)
                or sim_result.get("total_latency_ns", 0.0)
                or sim_result.get("latency_ns", 0.0)
                or 0.0
            )
            total_latency_sum_ns += (
                sim_result.get("total_latency_sum_ns", 0.0)
                or sim_result.get("latency_sum_ns", 0.0)
                or 0.0
            )
            sim_samples_done += 1
            if max_samples is not None and int(max_samples) > 0 and sim_samples_done >= int(max_samples):
                break
        batch_count += 1
        if max_samples is not None and int(max_samples) > 0 and sim_samples_done >= int(max_samples):
            break

    if not test_true:
        result.error = "No batches processed"
        return result

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    result.num_samples = num_samples
    result.test_loss = total_loss / max(batch_count, 1)
    if is_partseg:
        result.accuracy = float((test_true == test_pred).mean() * 100.0)
        result.instance_miou = float(shape_ious_sum / max(num_samples, 1) * 100.0)
        result.balanced_accuracy = result.instance_miou
    else:
        result.accuracy = float(metrics.accuracy_score(test_true, test_pred) * 100.0)
        result.balanced_accuracy = float(metrics.balanced_accuracy_score(test_true, test_pred) * 100.0)
    result.total_cycles = total_cycles
    result.total_energy_nj = total_energy_nj
    total_latency_ns = total_latency_critical_path_ns if total_latency_critical_path_ns > 0 else total_latency_sum_ns
    result.total_latency_ns = total_latency_ns
    result.total_latency_sum_ns = total_latency_sum_ns
    result.total_latency_critical_path_ns = total_latency_critical_path_ns
    result.cycles_per_sample = total_cycles / num_samples if num_samples else 0
    result.energy_per_sample_nj = total_energy_nj / num_samples if num_samples else 0
    result.latency_per_sample_ns = total_latency_ns / num_samples if num_samples else 0
    result.latency_sum_per_sample_ns = total_latency_sum_ns / num_samples if num_samples else 0
    result.critical_path_latency_per_sample_ns = total_latency_critical_path_ns / num_samples if num_samples else 0
    result.sim_once = last_sim_once
    result.success = True
    return result
