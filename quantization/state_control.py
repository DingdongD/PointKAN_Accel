# Ported from ISEDA26/iseda_quant for PointKAN-Elite
from __future__ import absolute_import
import logging
import torch
from .fake_quant import QuantizeBase

logger = logging.getLogger("pointkan_quant")


def enable_calibration_woquantization(model, quantizer_type="fake_quant"):
    module = model if isinstance(model, torch.nn.Module) else model.model
    logger.info("Enable observer and Disable quantize for %s", quantizer_type)
    enabled_count = 0
    disabled_count = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                disabled_count += 1
                continue
            submodule.enable_observer()
            submodule.disable_fake_quant()
            enabled_count += 1
    logger.info("Enabled %d observers, disabled %d quantizers for type: %s",
                enabled_count, disabled_count, quantizer_type)


def enable_dynamic_quantization(model, quantizer_type="fake_quant"):
    module = model if isinstance(model, torch.nn.Module) else model.model
    logger.info("Enable observer and Enable quantize for %s", quantizer_type)
    enabled_count = 0
    disabled_count = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                disabled_count += 1
                continue
            submodule.enable_observer()
            submodule.enable_fake_quant()
            enabled_count += 1
    logger.info("Enabled %d dynamic quantizers for type: %s", enabled_count, quantizer_type)


def enable_quantization(model, quantizer_type="fake_quant"):
    module = model if isinstance(model, torch.nn.Module) else model.model
    logger.info("Disable observer and Enable quantize.")
    enabled_count = 0
    disabled_count = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                disabled_count += 1
                continue
            submodule.disable_observer()
            submodule.enable_fake_quant()
            enabled_count += 1
    logger.info("Enabled %d quantizers, disabled %d observers for type: %s",
                enabled_count, disabled_count, quantizer_type)


def disable_all(model):
    module = model if isinstance(model, torch.nn.Module) else model.model
    logger.info("Disable ALL Quantizer")
    for name, submodule in module.named_modules():
        if isinstance(submodule, QuantizeBase):
            submodule.disable_observer()
            submodule.disable_fake_quant()
