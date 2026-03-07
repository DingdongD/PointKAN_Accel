# Ported from ISEDA26/iseda_quant for PointKAN-Elite W8A8 RTN (no plot dependency)
from __future__ import absolute_import
import os
import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def grad_scale(t: torch.Tensor, scale: float) -> torch.Tensor:
    return (t - (t * scale)).detach() + (t * scale)


def fake_quantize_per_tensor_affine(
    x, scale, zero_point, quant_min, quant_max, save_flag=0, visualize_path=""
) -> torch.Tensor:
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def fake_quantize_per_channel_affine(
    x, scale, zero_point, ch_axis, quant_min, quant_max
) -> torch.Tensor:
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def fake_logquantize_per_tensor_affine(
    x, scale, quant_min, quant_max, tau=2.0
) -> torch.Tensor:
    levels = quant_max - quant_min + 1
    x = torch.clamp(x, 1e-20, None)
    x_int = round_ste(-1 * (x / scale).log2() * tau)
    softmax_mask = x_int >= levels
    x_q = torch.clamp(x_int, 0, levels - 1)
    X = scale * 2 ** (-1 * x_q / tau)
    X[softmax_mask] = torch.tensor(0.0, device=X.device)
    return X


fake_quantize_per_axis_affine = fake_quantize_per_channel_affine


def fake_quantize_learnable_per_tensor_affine_training(
    x, scale, zero_point, quant_min, quant_max, grad_factor, save_flag=0, visualize_path=""
) -> torch.Tensor:
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def fake_quantize_learnable_per_channel_affine_training(
    x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor
) -> torch.Tensor:
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def fake_quantize_learnableplus_per_tensor_affine_training(
    x, scale, zero_point, quant_min, quant_max, grad_factor
) -> torch.Tensor:
    zero_point = round_ste(zero_point)
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def fake_quantize_learnableplus_per_channel_affine_training(
    x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor
) -> torch.Tensor:
    zero_point = round_ste(zero_point)
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    return (x_quant - zero_point) * scale


def set_quant_axis(model, axis, mode="act"):
    from .observer import ObserverBase
    from .fake_quant import QuantizeBase
    for name, module in model.named_modules():
        if not (isinstance(module, QuantizeBase) or isinstance(module, ObserverBase)):
            continue
        if mode not in name:
            continue
        module.ch_axis = axis
    return model
