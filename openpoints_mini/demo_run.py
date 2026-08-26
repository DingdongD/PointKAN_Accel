#!/usr/bin/env python3
"""
验证：PointNet2 CUDA 扩展 + PointNext/PointVector 骨干 + BaseCls / BaseSeg 前向。

用法（在 PointKAN-pytorch 根目录）:
  conda activate pointkan
  bash openpoints_mini/install_pointnet2_cuda.sh   # 首次
  python -m openpoints_mini.demo_run
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

import openpoints_mini.models  # noqa: F401 — 注册 MODELS
from openpoints_mini.configs import presets
from openpoints_mini.models.build import build_model_from_cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("警告: PointNet2 扩展主要为 CUDA 设计，当前为 CPU，可能失败。")

    from openpoints_mini.cpp import pointnet2_cuda  # noqa: F401

    B, N = 2, 256
    pos = torch.randn(B, N, 3, device=device)

    enc_cfg = presets.pointnext_encoder_modelnet40_s()
    enc = build_model_from_cfg(enc_cfg).to(device)
    enc.eval()
    with torch.no_grad():
        feat = enc.forward_cls_feat(pos)
    print(f"PointNextEncoder(ModelNet-S 预设) forward_cls_feat -> {tuple(feat.shape)}")

    cls_model = build_model_from_cfg(
        {
            "NAME": "BaseCls",
            "encoder_args": presets.pointvector_encoder_modelnet40_s(),
            "cls_args": {
                "NAME": "ClsHead",
                "num_classes": 40,
                "mlps": [256],
                "norm_args": {"norm": "bn1d"},
                "act_args": {"act": "relu"},
                "dropout": 0.0,
            },
        }
    ).to(device)
    cls_model.eval()
    with torch.no_grad():
        logits = cls_model(pos)
    print(f"BaseCls(PointVector-S 预设) logits -> {tuple(logits.shape)}")

    seg_model = build_model_from_cfg(
        {
            "NAME": "BaseSeg",
            "encoder_args": presets.pointnext_encoder_s3dis_l(),
            "decoder_args": {"NAME": "PointNextDecoder"},
            "cls_args": {
                "NAME": "SegHead",
                "num_classes": 13,
                "in_channels": None,
                "norm_args": {"norm": "bn"},
            },
        }
    ).to(device)
    seg_xyz = torch.randn(B, N, 3, device=device)
    seg_feat = torch.randn(B, N, 4, device=device).transpose(1, 2).contiguous()
    seg_model.eval()
    with torch.no_grad():
        seg_logits = seg_model({"pos": seg_xyz, "x": seg_feat})
    print(f"BaseSeg(PointNext-L + Decoder) logits -> {tuple(seg_logits.shape)}")

    print("OK: CUDA 扩展与 cls/seg 骨架前向完成。")


if __name__ == "__main__":
    main()
