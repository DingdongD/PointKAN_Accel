from torch import nn

from openpoints_mini.utils.registry import Registry

LOSS = Registry("loss")
LOSS.register_module(name="CrossEntropyLoss", module=nn.CrossEntropyLoss)
LOSS.register_module(name="CrossEntropy", module=nn.CrossEntropyLoss)


def build_criterion_from_cfg(cfg, **kwargs):
    if cfg is None:
        return None
    return LOSS.build(cfg, **kwargs)
