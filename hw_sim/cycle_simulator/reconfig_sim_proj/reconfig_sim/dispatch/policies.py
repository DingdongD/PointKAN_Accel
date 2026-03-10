from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from reconfig_sim.workloads.op_desc import OpDesc


Mode = str  # 'OS' or 'WS'


class MappingPolicy:
    def select_mode(self, op: OpDesc) -> Mode:
        raise NotImplementedError


@dataclass
class FixedModePolicy(MappingPolicy):
    mode: Mode

    def __post_init__(self) -> None:
        if self.mode not in {"OS", "WS"}:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def select_mode(self, op: OpDesc) -> Mode:
        return self.mode


@dataclass
class RuleBasedPolicy(MappingPolicy):
    default_mode: Mode = "WS"
    explicit_map: Dict[str, Mode] = field(default_factory=dict)

    def select_mode(self, op: OpDesc) -> Mode:
        if op.name in self.explicit_map:
            return self.explicit_map[op.name]
        if op.op_type in self.explicit_map:
            return self.explicit_map[op.op_type]

        if op.op_type in {"linear", "conv2d_im2col", "ffn", "proj", "attention_proj"}:
            return "WS"
        if op.op_type in {"attention_qk", "attention_av", "gemm_dynamic"}:
            return "OS"
        if op.has_static_weight:
            return "WS"
        return self.default_mode


@dataclass
class OverridePolicy(MappingPolicy):
    base: MappingPolicy
    per_op_mode: Dict[str, Mode] = field(default_factory=dict)

    def select_mode(self, op: OpDesc) -> Mode:
        return self.per_op_mode.get(op.name, self.base.select_mode(op))
