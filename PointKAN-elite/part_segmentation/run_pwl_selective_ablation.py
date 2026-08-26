#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PartSeg PWL-GRF selective ablation.

This script leaves the original model files untouched. It loads a fresh model for
each variant, selectively replaces KAT_Group modules with KAT_Group_PWL, and then
evaluates the normal PartSeg test split.
"""

import argparse
import json
import os
import sys
from typing import Iterable

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "pointKANElite_part_exp")
DEFAULT_PWL_V2_JSON = os.path.join(SCRIPT_DIR, "grf_export_out", "pwl_v2.json")
DEFAULT_LAYER_ERROR_JSON = os.path.join(SCRIPT_DIR, "grf_export_out", "partseg_pwl_layerwise_error_b16.json")

for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from util.data_util import PartNormalDataset
from inference_and_kan_analysis import (
    get_checkpoint_path,
    build_model_and_load_checkpoint,
    run_evaluation,
)
from inject_grf_pwl_v2 import (
    _build_pwl_module_from_layer,
    _build_pwl_module_from_layer_triton,
    _find_kat_group_modules,
    _resolve_bit_key,
)


def _load_ranked_layers(path: str, score: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("layers", [])
    if not rows:
        raise ValueError(f"no layers found in {path}")
    if score not in rows[0]:
        raise ValueError(f"score {score!r} not found; available keys: {sorted(rows[0].keys())}")
    rows = sorted(rows, key=lambda r: float(r[score]), reverse=True)
    return [str(r["name"]) for r in rows]


def _inject_selective(
    model,
    v2_json_path: str,
    *,
    bit_key: str | None,
    device,
    impl: str,
    replace_names: Iterable[str] | None = None,
    skip_names: Iterable[str] | None = None,
) -> tuple[int, str, list[str]]:
    with open(v2_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    bit = _resolve_bit_key(payload, bit_key)
    v2_by_name = {str(l["name"]): l for l in payload.get("layers", [])}
    replace_set = None if replace_names is None else set(replace_names)
    skip_set = set(skip_names or [])
    dev = device if device is not None else next(model.parameters()).device

    replaced_names: list[str] = []
    for name, old_module in _find_kat_group_modules(model):
        if name not in v2_by_name:
            raise KeyError(f"module {name!r} missing in {v2_json_path}")
        if replace_set is not None and name not in replace_set:
            continue
        if name in skip_set:
            continue

        layer_v2 = v2_by_name[name]
        if bit not in layer_v2.get("bits", {}):
            raise ValueError(f"layer {name!r} has no bits[{bit}]")
        num_groups = int(old_module.num_groups)
        if impl == "triton":
            pwl_mod = _build_pwl_module_from_layer_triton(layer_v2, bit, num_groups, dev)
        else:
            pwl_mod = _build_pwl_module_from_layer(layer_v2, bit, num_groups, dev)

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], pwl_mod)
        replaced_names.append(name)
    return len(replaced_names), bit, replaced_names


def _parse_topks(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective PWL-GRF ablation for PartSeg")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--pwl_v2_json", type=str, default=DEFAULT_PWL_V2_JSON)
    parser.add_argument("--layer_error_json", type=str, default=DEFAULT_LAYER_ERROR_JSON)
    parser.add_argument("--score", type=str, default="rmse", choices=("mae", "rmse", "max_abs_diff", "rel_rmse"))
    parser.add_argument("--topks", type=str, default="1,2,4,6")
    parser.add_argument("--bit", type=str, default=None)
    parser.add_argument("--pwl_impl", type=str, default="torch", choices=("torch", "triton"))
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    ranked = _load_ranked_layers(args.layer_error_json, args.score)
    topks = _parse_topks(args.topks)

    test_dataset = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"pwl_v2_json: {os.path.abspath(args.pwl_v2_json)}")
    print(f"ranking score: {args.score}")
    print("top ranked layers:")
    for i, name in enumerate(ranked[: max(topks or [0])], 1):
        print(f"  {i:02d}. {name}")

    variants = []

    net = build_model_and_load_checkpoint(checkpoint_path, device)
    vanilla = run_evaluation(net, test_loader, device, max_batches=args.max_batches)
    variants.append({"variant": "vanilla", "replaced": 0, "kept_vanilla_layers": "all", "result": vanilla})
    print(f"vanilla: acc={vanilla['acc']:.4f} ins_miou={vanilla['shape_avg_iou']:.6f}")

    net = build_model_and_load_checkpoint(checkpoint_path, device)
    n_replaced, bit_used, replaced_names = _inject_selective(
        net,
        args.pwl_v2_json,
        bit_key=args.bit,
        device=device,
        impl=args.pwl_impl,
    )
    result = run_evaluation(net, test_loader, device, max_batches=args.max_batches)
    variants.append(
        {
            "variant": "all_pwl",
            "replaced": n_replaced,
            "bit_used": bit_used,
            "replaced_layers": replaced_names,
            "kept_vanilla_layers": [],
            "result": result,
        }
    )
    print(f"all_pwl: replaced={n_replaced} acc={result['acc']:.4f} ins_miou={result['shape_avg_iou']:.6f}")

    for k in topks:
        keep = ranked[:k]
        net = build_model_and_load_checkpoint(checkpoint_path, device)
        n_replaced, bit_used, replaced_names = _inject_selective(
            net,
            args.pwl_v2_json,
            bit_key=args.bit,
            device=device,
            impl=args.pwl_impl,
            skip_names=keep,
        )
        result = run_evaluation(net, test_loader, device, max_batches=args.max_batches)
        variants.append(
            {
                "variant": f"keep_top{k}_vanilla_by_{args.score}",
                "replaced": n_replaced,
                "bit_used": bit_used,
                "replaced_layers": replaced_names,
                "kept_vanilla_layers": keep,
                "result": result,
            }
        )
        print(
            f"keep_top{k}_vanilla: replaced={n_replaced} "
            f"acc={result['acc']:.4f} ins_miou={result['shape_avg_iou']:.6f}"
        )

    base_acc = vanilla["acc"]
    base_iou = vanilla["shape_avg_iou"]
    for row in variants:
        res = row["result"]
        row["acc_diff_vs_vanilla"] = res["acc"] - base_acc
        row["shape_avg_iou_diff_vs_vanilla"] = res["shape_avg_iou"] - base_iou

    out = {
        "task": "part_segmentation",
        "checkpoint": checkpoint_path,
        "pwl_v2_json": os.path.abspath(args.pwl_v2_json),
        "layer_error_json": os.path.abspath(args.layer_error_json),
        "score": args.score,
        "topks": topks,
        "max_batches": args.max_batches,
        "variants": variants,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote: {args.out_json}")


if __name__ == "__main__":
    main()
