#!/usr/bin/env python3
"""Evaluate PointKAN-Elite accuracy under RRAM-like weight variation.

This script is intentionally external to PointKAN-elite model files. It loads a
checkpoint, snapshots Conv/Linear weights, applies temporary in-memory noise for
each sweep point, evaluates accuracy, then restores the original weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)


@dataclass(frozen=True)
class NoiseSpec:
    """Weight-noise configuration.

    kind:
      - relative: per-weight multiplicative N(0, sigma), w * (1 + eps)
      - additive_std: per-weight additive N(0, sigma * std(layer_weight))
      - additive_abs: per-weight additive N(0, sigma)
      - column: output-channel/row-wise multiplicative N(0, sigma)
      - combined: relative plus additive_std, each using the same sigma
    """

    kind: str
    sigma: float
    clamp_multiple: Optional[float] = None


def parse_float_list(value: str) -> List[float]:
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one variation value is required")
    if any(v < 0 for v in vals):
        raise ValueError("variation values must be non-negative")
    return vals


def parse_int_list(value: str) -> List[int]:
    vals = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one seed is required")
    return vals


def _supported_weight_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, SUPPORTED_MODULES) and getattr(module, "weight", None) is not None:
            yield name, module


def capture_weight_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: module.weight.detach().clone() for name, module in _supported_weight_modules(model)}


def restore_weight_state_(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, weight in state.items():
            modules[name].weight.copy_(weight.to(device=modules[name].weight.device, dtype=modules[name].weight.dtype))


def _randn_like(weight: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)


def _output_channel_shape(weight: torch.Tensor) -> List[int]:
    return [int(weight.shape[0])] + [1] * (weight.dim() - 1)


def _apply_clamp(noisy: torch.Tensor, original: torch.Tensor, clamp_multiple: Optional[float]) -> torch.Tensor:
    if clamp_multiple is None or clamp_multiple <= 0:
        return noisy
    max_abs = original.detach().abs().max()
    if not torch.isfinite(max_abs) or float(max_abs) == 0.0:
        return noisy
    limit = max_abs * float(clamp_multiple)
    return noisy.clamp(min=-limit, max=limit)


def _noisy_weight(weight: torch.Tensor, spec: NoiseSpec, generator: torch.Generator) -> torch.Tensor:
    sigma = float(spec.sigma)
    if sigma == 0.0:
        return weight.clone()

    if spec.kind == "relative":
        noisy = weight * (1.0 + _randn_like(weight, generator) * sigma)
    elif spec.kind == "additive_std":
        std = weight.detach().float().std(unbiased=False).to(dtype=weight.dtype, device=weight.device)
        noisy = weight + _randn_like(weight, generator) * (sigma * std)
    elif spec.kind == "additive_abs":
        noisy = weight + _randn_like(weight, generator) * sigma
    elif spec.kind == "column":
        eps = torch.randn(
            _output_channel_shape(weight),
            generator=generator,
            device=weight.device,
            dtype=weight.dtype,
        )
        noisy = weight * (1.0 + eps * sigma)
    elif spec.kind == "combined":
        rel = _randn_like(weight, generator) * sigma
        std = weight.detach().float().std(unbiased=False).to(dtype=weight.dtype, device=weight.device)
        add = _randn_like(weight, generator) * (sigma * std)
        noisy = weight * (1.0 + rel) + add
    else:
        raise ValueError(f"unsupported noise kind: {spec.kind}")

    return _apply_clamp(noisy, weight, spec.clamp_multiple)


def apply_weight_noise_(model: nn.Module, spec: NoiseSpec, seed: int) -> Dict[str, Any]:
    if spec.sigma < 0:
        raise ValueError("sigma must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    stats = {"modules": 0, "weights": 0, "kind": spec.kind, "sigma": spec.sigma, "seed": int(seed)}
    with torch.no_grad():
        for _, module in _supported_weight_modules(model):
            weight = module.weight
            # torch.Generator is CPU-only on many installed torch builds; generate
            # CPU noise deterministically, then move through _noisy_weight helpers.
            if weight.device.type != "cpu":
                cpu_weight = weight.detach().cpu()
                noisy = _noisy_weight(cpu_weight, spec, generator).to(device=weight.device, dtype=weight.dtype)
            else:
                noisy = _noisy_weight(weight.detach(), spec, generator).to(dtype=weight.dtype)
            weight.copy_(noisy)
            stats["modules"] += 1
            stats["weights"] += int(weight.numel())
    return stats


def _primary_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _network_name(network: str) -> str:
    return str(network or "pointkan-elite").strip().lower()


def infer_num_classes(network: str, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    if _network_name(network) == "pointkan-elite-scanobj":
        return 15
    return 40


def evaluate_accuracy(
    model: nn.Module,
    test_loader: Any,
    criterion: Any,
    *,
    network: str,
    device: str,
    max_batches: Optional[int],
) -> Dict[str, float]:
    from pointkan_sim.integration.real_data_eval import _is_partseg_network, _is_pointnet2_partseg_network

    test_true: List[np.ndarray] = []
    test_pred: List[np.ndarray] = []
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    shape_ious_sum = 0.0
    net_name = _network_name(network)
    is_partseg = _is_partseg_network(net_name)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if is_partseg:
                from pointkan_sim.integration.real_data_eval import _PARTSEG_NUM_CLASSES, _PARTSEG_NUM_PART, _ensure_partseg_path

                _ensure_partseg_path()
                from util.util import compute_overall_iou, to_categorical

                if _is_pointnet2_partseg_network(net_name):
                    points, label, target = batch
                    points = points.float().to(device).transpose(2, 1)
                    label = label.long().to(device).view(-1)
                    target = target.long().to(device)
                    onehot = to_categorical(label, _PARTSEG_NUM_CLASSES)
                    seg_pred = _primary_output(model(points, onehot))
                    loss = F.cross_entropy(seg_pred, target)
                    seg_for_iou = seg_pred.transpose(1, 2).contiguous()
                    seg_flat = seg_for_iou.view(-1, _PARTSEG_NUM_PART)
                    target_flat = target.reshape(-1)
                    batch_shapeious = compute_overall_iou(seg_for_iou, target, _PARTSEG_NUM_PART)
                else:
                    points, label, target, norm_plt = batch
                    points = points.float().to(device).transpose(2, 1)
                    norm_plt = norm_plt.float().to(device).transpose(2, 1)
                    label = label.squeeze(1).long().to(device)
                    target = target.long().to(device)
                    onehot = to_categorical(label, _PARTSEG_NUM_CLASSES)
                    seg_pred = _primary_output(model(points, norm_plt, onehot))
                    seg_flat = seg_pred.contiguous().view(-1, _PARTSEG_NUM_PART)
                    target_flat = target.view(-1, 1)[:, 0]
                    loss = F.nll_loss(seg_flat, target_flat)
                    batch_shapeious = compute_overall_iou(seg_pred, target, _PARTSEG_NUM_PART)
                pred_choice = seg_flat.max(1)[1]
                bsz = int(points.size(0))
                shape_ious_sum += float(sum(batch_shapeious))
                test_true.append(np.atleast_1d(target_flat.cpu().numpy()))
                test_pred.append(np.atleast_1d(pred_choice.cpu().numpy()))
            else:
                data, label = batch
                data = data.to(device).permute(0, 2, 1)
                label = label.to(device).view(-1)
                logits = _primary_output(model(data))
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                bsz = int(data.size(0))
                test_true.append(np.atleast_1d(label.cpu().numpy()))
                test_pred.append(np.atleast_1d(preds.cpu().numpy()))

            total_loss += float(loss.item())
            num_samples += bsz
            num_batches += 1

    if not test_true:
        raise RuntimeError("No batches were processed; check dataset path and max_batches.")

    y_true = np.concatenate(test_true)
    y_pred = np.concatenate(test_pred)
    if is_partseg:
        acc = float((y_true == y_pred).mean() * 100.0)
        balanced = float(shape_ious_sum / max(num_samples, 1) * 100.0)
        return {
            "accuracy": acc,
            "balanced_accuracy": balanced,
            "instance_miou": balanced,
            "loss": total_loss / max(num_batches, 1),
            "num_samples": float(num_samples),
            "num_batches": float(num_batches),
        }
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred) * 100.0),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred) * 100.0),
        "loss": total_loss / max(num_batches, 1),
        "num_samples": float(num_samples),
        "num_batches": float(num_batches),
    }


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_results(output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "rram_weight_noise_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] == "baseline":
            continue
        grouped.setdefault((str(row["kind"]), float(row["sigma"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    baseline_acc = next((float(r["accuracy"]) for r in rows if r["kind"] == "baseline"), math.nan)
    for (kind, sigma), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        acc_mean, acc_std = _mean_std([float(r["accuracy"]) for r in group])
        bal_mean, bal_std = _mean_std([float(r["balanced_accuracy"]) for r in group])
        summary_rows.append({
            "kind": kind,
            "sigma": sigma,
            "seeds": len(group),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "accuracy_drop_mean": baseline_acc - acc_mean,
            "balanced_accuracy_mean": bal_mean,
            "balanced_accuracy_std": bal_std,
        })

    with (output_dir / "rram_weight_noise_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["kind"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "rram_weight_noise_metadata.json").open("w") as f:
        json.dump({"args": vars(args), "rows": rows, "summary": summary_rows}, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PointKAN-Elite RRAM weight variation accuracy sweep")
    parser.add_argument("--network", default="pointkan-elite", choices=["pointkan-elite", "pointkan-elite-scanobj", "pointkan-elite-partseg"])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or task default if omitted")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=None, help="Override classification classes; defaults by network")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None, help="Use a small value for quick smoke tests")
    parser.add_argument("--variations", default="0,0.01,0.03,0.05,0.1", help="Comma-separated sigma values")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--noise_kinds", default="relative,column,additive_std", help="Comma-separated kinds")
    parser.add_argument("--clamp_multiple", type=float, default=None, help="Optional clamp to +/- max(abs(original))*value")
    parser.add_argument("--output_dir", default="analysis_rram_weight_noise")
    return parser


def main() -> int:
    from pointkan_sim.integration.real_data_eval import load_model_and_dataset
    from pointkan_sim.run_sim_with_real_eval import resolve_checkpoint_path

    parser = build_parser()
    args = parser.parse_args()
    variations = parse_float_list(args.variations)
    seeds = parse_int_list(args.seeds)
    kinds = [x.strip() for x in str(args.noise_kinds).split(",") if x.strip()]
    for kind in kinds:
        if kind not in {"relative", "additive_std", "additive_abs", "column", "combined"}:
            raise ValueError(f"unsupported noise kind: {kind}")

    checkpoint = resolve_checkpoint_path(args.checkpoint, args.network)
    num_classes = infer_num_classes(args.network, args.num_classes)
    print(f"[rram-noise] network={args.network} checkpoint={checkpoint}")
    model, test_loader, criterion = load_model_and_dataset(
        checkpoint,
        network=args.network,
        num_points=args.num_points,
        num_classes=num_classes,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    model.to(args.device)
    original_state = capture_weight_state(model)
    print(f"[rram-noise] captured {len(original_state)} Conv/Linear weight tensors")

    rows: List[Dict[str, Any]] = []
    baseline = evaluate_accuracy(
        model,
        test_loader,
        criterion,
        network=args.network,
        device=args.device,
        max_batches=args.max_batches,
    )
    rows.append({"kind": "baseline", "sigma": 0.0, "seed": -1, **baseline})
    print(f"[baseline] acc={baseline['accuracy']:.3f}% balanced={baseline['balanced_accuracy']:.3f}%")

    for kind in kinds:
        for sigma in variations:
            for seed in seeds:
                restore_weight_state_(model, original_state)
                spec = NoiseSpec(kind=kind, sigma=sigma, clamp_multiple=args.clamp_multiple)
                stats = apply_weight_noise_(model, spec, seed=seed)
                result = evaluate_accuracy(
                    model,
                    test_loader,
                    criterion,
                    network=args.network,
                    device=args.device,
                    max_batches=args.max_batches,
                )
                row = {
                    "kind": kind,
                    "sigma": sigma,
                    "seed": seed,
                    "modules_noised": stats["modules"],
                    "weights_noised": stats["weights"],
                    **result,
                }
                rows.append(row)
                print(
                    f"[{kind} sigma={sigma:g} seed={seed}] "
                    f"acc={result['accuracy']:.3f}% drop={baseline['accuracy'] - result['accuracy']:.3f}%"
                )

    restore_weight_state_(model, original_state)
    output_dir = Path(args.output_dir)
    write_results(output_dir, rows, args)
    print(f"[rram-noise] wrote {output_dir / 'rram_weight_noise_results.csv'}")
    print(f"[rram-noise] wrote {output_dir / 'rram_weight_noise_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
