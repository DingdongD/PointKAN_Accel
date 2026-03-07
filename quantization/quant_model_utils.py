# Ported from ISEDA26/iseda_quant for PointKAN-Elite (no load_dataset; calibrate with dataloader)
from __future__ import absolute_import
import copy
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from .quantized_module import QuantizedLayer, PreQuantizedLayer
from .state_control import enable_calibration_woquantization, enable_quantization
from .fake_quant import QuantizeBase

logger = logging.getLogger("pointkan_quant")


def _sync_observer_buffers(module, world_size):
    """Sync min_val/max_val of QuantizeBase.observer across GPUs (optional)."""
    for name, m in module.named_modules():
        if not isinstance(m, QuantizeBase):
            continue
        obs = getattr(m, "observer", None)
        if obs is None or not hasattr(obs, "min_val"):
            continue
        obs.min_val.data.div_(world_size)
        obs.max_val.data.div_(world_size)
        try:
            import torch.distributed as dist
            dist.all_reduce(obs.min_val.data)
            dist.all_reduce(obs.max_val.data)
        except Exception:
            pass


@torch.no_grad()
def calibrate(model, calib_loader, calib_num, device, verbose=True, get_logits_fn=None):
    """
    校准模型以收集量化参数（激活 + 权重）。
    model: 已 quantize_model 后的模型。
    calib_loader: DataLoader，每个 batch 为 (points, target) 或 (data, label, seg) 等，由 get_logits_fn 消费。
    calib_num: 用于校准的样本数量（激活）。
    device: torch.device。
    get_logits_fn: 可选，签名为 (model, batch, device) -> 无返回值，仅跑前向。默认：batch = (points, target)，points 送 device 并 permute(0,2,1)，model(points) 取 [0] 若为 tuple。
    """
    st = time.time()
    if get_logits_fn is None:
        def _default_forward(m, batch, dev):
            points, _ = batch[0], batch[1]
            points = points.to(dev)
            points = points.permute(0, 2, 1)
            out = m(points)
            if isinstance(out, (list, tuple)):
                _ = out[0]
            else:
                _ = out
        get_logits_fn = _default_forward

    # Stage 1: activation calibration
    logger.info("Stage 1: Calibrating Activations")
    enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
    sample_count = 0
    for batch in tqdm(calib_loader, total=min(len(calib_loader), (calib_num // (batch[0].size(0) if batch[0] is not None else 1)) + 1), desc="Calibrating Activations"):
        if batch[0] is None:
            continue
        get_logits_fn(model, batch, device)
        batch_size = batch[0].size(0)
        sample_count += batch_size
        if sample_count >= calib_num:
            break
    if verbose:
        logger.info("Activation calibration: %d samples", sample_count)

    # Stage 2: weight calibration (one batch)
    logger.info("Stage 2: Calibrating Weights")
    enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
    for batch in calib_loader:
        get_logits_fn(model, batch, device)
        break
    if verbose:
        logger.info("Weight calibration: 1 batch")

    logger.info("Calibration completed in %.2f s", time.time() - st)
    return


@torch.no_grad()
def calibrate_partseg(model, calib_loader, calib_num, device, num_classes=16, verbose=True):
    """PartSeg 校准：batch 为 (data, label, seg)，forward 为 model(data, norm_plt, to_categorical(label, num_classes))。"""
    import numpy as np
    st = time.time()

    def to_categorical(y, num_classes):
        o = np.zeros((y.shape[0], num_classes), dtype=np.float32)
        for i in range(y.shape[0]):
            o[i, y[i]] = 1.0
        return torch.from_numpy(o).to(y.device)

    logger.info("Stage 1: Calibrating Activations (PartSeg)")
    enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
    sample_count = 0
    for batch in tqdm(calib_loader, desc="Calibrating Activations"):
        data, label, seg = batch[0], batch[1], batch[2]
        if data is None:
            continue
        data = data.to(device)
        label_one_hot = to_categorical(label.cpu().numpy(), num_classes).to(device)
        data = data.permute(0, 2, 1)
        # PartSeg model often takes (points, norm_plt, label_one_hot); norm_plt may be same as points or a second tensor
        if len(batch) >= 4:
            norm_plt = batch[3].to(device).permute(0, 2, 1)
        else:
            norm_plt = data
        out = model(data, norm_plt, label_one_hot)
        if isinstance(out, (list, tuple)):
            _ = out[0]
        sample_count += data.size(0)
        if sample_count >= calib_num:
            break

    logger.info("Stage 2: Calibrating Weights (PartSeg)")
    enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
    for batch in calib_loader:
        data, label, seg = batch[0], batch[1], batch[2]
        data = data.to(device)
        label_one_hot = to_categorical(label.cpu().numpy(), num_classes).to(device)
        data = data.permute(0, 2, 1)
        norm_plt = batch[3].to(device).permute(0, 2, 1) if len(batch) >= 4 else data
        model(data, norm_plt, label_one_hot)
        break

    logger.info("PartSeg calibration completed in %.2f s", time.time() - st)


def quantize_model(model, quant_config, use_input_quant=True, layer_bit_alloc=None, verbose=True):
    """
    将 model 中的 Linear/Conv1d/Conv2d 替换为 PreQuantizedLayer（或 QuantizedLayer）。
    quant_config: 具有 w_qconfig, a_qconfig 的对象；可选 model_name 用于日志。
    """
    NotQuantModule = []

    def get_layer_bit_config(new_path, layer_bit_alloc, w_qconfig, a_qconfig):
        if layer_bit_alloc is None:
            return w_qconfig, a_qconfig
        for pattern, bit_config in layer_bit_alloc.items():
            if pattern in new_path:
                layer_w = copy.deepcopy(w_qconfig)
                layer_a = copy.deepcopy(a_qconfig)
                if "w_bit" in bit_config:
                    layer_w.bit = bit_config["w_bit"]
                if "a_bit" in bit_config:
                    layer_a.bit = bit_config["a_bit"]
                return layer_w, layer_a
        return w_qconfig, a_qconfig

    def replace_module(module, w_qconfig, a_qconfig, use_input_quant, current_path=""):
        for name, child in module.named_children():
            new_path = f"{current_path}.{name}" if current_path else name
            if any(s in new_path for s in NotQuantModule):
                continue
            layer_w_qconfig, layer_a_qconfig = get_layer_bit_config(
                new_path, layer_bit_alloc, w_qconfig, a_qconfig
            )
            if isinstance(child, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if use_input_quant:
                    setattr(
                        module,
                        name,
                        PreQuantizedLayer(child, None, layer_w_qconfig, layer_a_qconfig, qinput=True),
                    )
                else:
                    setattr(
                        module,
                        name,
                        QuantizedLayer(child, None, layer_w_qconfig, layer_a_qconfig, qoutput=True),
                    )
            else:
                replace_module(child, w_qconfig, a_qconfig, use_input_quant, new_path)

    model_name = getattr(getattr(quant_config, "model", None), "model", None) or getattr(quant_config, "model_name", "model")
    if verbose:
        logger.info("Quantizing model: %s", model_name)
        logger.info("w_qconfig: %s", quant_config.w_qconfig)
        logger.info("a_qconfig: %s", quant_config.a_qconfig)
    replace_module(model, quant_config.w_qconfig, quant_config.a_qconfig, use_input_quant)
    return model
