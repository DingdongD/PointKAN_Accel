from __future__ import absolute_import

import torch
import torch.nn as nn


def _fuse_linear_bn_eval(linear, bn):
    if linear.bias is None:
        bias = torch.zeros(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device)
    else:
        bias = linear.bias
    bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
    scale = bn.weight * bn_var_rsqrt
    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight.data.copy_(linear.weight * scale.reshape(-1, 1))
    fused.bias.data.copy_((bias - bn.running_mean) * scale + bn.bias)
    return fused.to(linear.weight.device)


def fuse_conv_linear_bn_eval(module):
    """
    Recursively fold eval-mode Conv1d/Conv2d/Linear followed by BatchNorm1d/2d
    inside Sequential containers. BatchNorm slots are replaced with Identity.
    """
    module.eval()
    fused_count = 0
    for child in module.children():
        fused_count += fuse_conv_linear_bn_eval(child)

    if isinstance(module, nn.Sequential):
        keys = list(module._modules.keys())
        for i in range(len(keys) - 1):
            k1, k2 = keys[i], keys[i + 1]
            first, second = module._modules[k1], module._modules[k2]
            if isinstance(first, (nn.Conv1d, nn.Conv2d)) and isinstance(second, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module._modules[k1] = torch.nn.utils.fuse_conv_bn_eval(first, second)
                module._modules[k2] = nn.Identity()
                fused_count += 1
            elif isinstance(first, nn.Linear) and isinstance(second, nn.BatchNorm1d):
                module._modules[k1] = _fuse_linear_bn_eval(first, second)
                module._modules[k2] = nn.Identity()
                fused_count += 1
    return fused_count
