#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge high-precision PWL layers into a baseline PWL JSON.

The normal injector expects one global bit key. This utility creates a mixed JSON
where sensitive layers use a high-precision bit entry, while all remaining
layers expose their baseline bit entry under the same deploy bit key.
"""

import argparse
import copy
import json
import os
from pathlib import Path


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sensitive high-precision PWL layers")
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--sensitive_dir", required=True)
    parser.add_argument("--sensitive_layers", nargs="+", required=True)
    parser.add_argument("--base_bit", default="8")
    parser.add_argument("--sensitive_bit", default="10")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(args.base_json, "r", encoding="utf-8") as f:
        base = json.load(f)

    sensitive = {}
    for layer_name in args.sensitive_layers:
        path = Path(args.sensitive_dir) / f"{_safe_layer_name(layer_name)}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        layers = payload.get("layers", [])
        if len(layers) != 1:
            raise ValueError(f"{path} should contain exactly one layer, got {len(layers)}")
        layer = layers[0]
        if str(layer["name"]) != layer_name:
            raise ValueError(f"{path} contains {layer['name']!r}, expected {layer_name!r}")
        if args.sensitive_bit not in layer.get("bits", {}):
            raise ValueError(f"{path} has no bits[{args.sensitive_bit}]")
        sensitive[layer_name] = layer

    out = copy.deepcopy(base)
    out["meta"] = copy.deepcopy(base.get("meta", {}))
    out["meta"]["mixed_sensitive_pwl"] = {
        "base_json": os.path.abspath(args.base_json),
        "sensitive_dir": os.path.abspath(args.sensitive_dir),
        "sensitive_layers": args.sensitive_layers,
        "base_bit_source": str(args.base_bit),
        "deploy_bit_key": str(args.sensitive_bit),
    }
    out["meta"]["deploy_breakpoint_bits"] = int(args.sensitive_bit)

    replaced = []
    for layer in out.get("layers", []):
        name = str(layer["name"])
        if name in sensitive:
            high = copy.deepcopy(sensitive[name])
            high["bits"] = {str(args.sensitive_bit): high["bits"][str(args.sensitive_bit)]}
            layer.clear()
            layer.update(high)
            replaced.append(name)
        else:
            bits = layer.get("bits", {})
            if str(args.base_bit) not in bits:
                raise ValueError(f"baseline layer {name!r} has no bits[{args.base_bit}]")
            bits[str(args.sensitive_bit)] = copy.deepcopy(bits[str(args.base_bit)])
            bits[str(args.sensitive_bit)]["breakpoint_bits"] = int(args.sensitive_bit)

    missing = sorted(set(args.sensitive_layers) - set(replaced))
    if missing:
        raise ValueError(f"sensitive layers missing from base json: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote: {args.output_json}")
    print(f"sensitive layers replaced: {len(replaced)}")


if __name__ == "__main__":
    main()
