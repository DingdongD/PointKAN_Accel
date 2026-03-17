#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 ModelNet40 的 PointKAN-Elite 每层网络 shape 分析脚本

对加载好的模型用 ModelNet40 数据做前向传播，通过 forward hook 记录每一层的
输入/输出 tensor shape，并写入日志文件。

使用示例：
    python run_shape_analysis.py --checkpoint checkpoints/pointKANElite-pointkan_elite_exp-42
    python run_shape_analysis.py --checkpoint ... --log shape_log.txt --num_batches 3
    ./run_shape_analysis.sh
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# 与 inference_and_kan_analysis 相同的路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from data import ModelNet40
import models as models

# 复用推理脚本中的加载逻辑
from inference_and_kan_analysis import (
    get_checkpoint_path,
    load_state_dict_for_model,
    build_model_and_load_checkpoint,
)


def _collect_layer_param_shapes(net):
    """
    以脚本的 layer 命名粒度收集参数 shape。
    返回 dict: layer_name -> list[(param_name, shape_tuple)]。
    """
    out = {}

    def add(layer_name, module):
        params = []
        for n, p in module.named_parameters(recurse=True):
            params.append((n, tuple(p.shape)))
        out[layer_name] = params

    add("embedding", net.embedding)
    n_stages = getattr(net, "stages", len(net.GAM_list))
    for i in range(n_stages):
        add(f"GAM-stage{i}", net.GAM_list[i])
        add(f"LFP-stage{i}", net.LFP_blocks_list[i])
        add(f"GFP-stage{i}", net.GFP_blocks_list[i])
    add("classifier", net.classifier)
    return out


def _format_param_list(params, max_items=None):
    """
    将参数列表格式化为紧凑单行字符串：name:shape | name:shape ...
    """
    if not params:
        return ""
    items = params if (max_items is None or max_items <= 0) else params[:max_items]
    s = " | ".join([f"{n}:{sh}" for n, sh in items])
    if max_items is not None and max_items > 0 and len(params) > max_items:
        s += f" | ...(+{len(params) - max_items} params)"
    return s


def _write_tsv(rows, log_file, include_weights, weight_by_layer):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("PointKAN-Elite 每层 Shape 分析 (ModelNet40)\n")
        f.write("=" * 80 + "\n")
        if include_weights:
            f.write("Layer\tInput Shape(s)\tOutput Shape(s)\tWeight Shapes (name:shape)\n")
        else:
            f.write("Layer\tInput Shape(s)\tOutput Shape(s)\n")
        f.write("-" * 80 + "\n")
        for name, inp, out, w in rows:
            if include_weights:
                f.write(f"{name}\t{inp}\t{out}\t{w}\n")
            else:
                f.write(f"{name}\t{inp}\t{out}\n")
        f.write("=" * 80 + "\n")


def _write_csv(rows, csv_file, include_weights):
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if include_weights:
            writer.writerow(["Layer", "Input Shape(s)", "Output Shape(s)", "Weight Shapes (name:shape)"])
            for name, inp, out, w in rows:
                writer.writerow([name, inp, out, w])
        else:
            writer.writerow(["Layer", "Input Shape(s)", "Output Shape(s)"])
            for name, inp, out, _w in rows:
                writer.writerow([name, inp, out])


def collect_layer_shapes(
    net,
    device,
    data_loader,
    num_batches=1,
    log_file=None,
    csv_file=None,
    include_weights=False,
    max_weight_items=0,
):
    """
    使用 ModelNet40 的 batch 前向传播，收集每一层的输入/输出 shape，并写入 log_file。
    返回列表 [(layer_name, input_shapes_str, output_shapes_str), ...]。
    """
    input_shapes = {}
    output_shapes = {}

    def _make_pre_hook(name, multi_input=False):
        def hook(module, inp):
            if multi_input:
                parts = []
                for x in (inp if isinstance(inp, (list, tuple)) else [inp]):
                    if isinstance(x, torch.Tensor):
                        parts.append(tuple(x.shape))
                    else:
                        parts.append(type(x).__name__)
                input_shapes[name] = str(tuple(parts))
            else:
                t = inp[0] if isinstance(inp, (list, tuple)) else inp
                if isinstance(t, torch.Tensor):
                    input_shapes[name] = str(tuple(t.shape))
                else:
                    input_shapes[name] = str(type(t).__name__)
        return hook

    def _make_hook(name, multi_output=False):
        def hook(module, inp, out):
            if multi_output:
                parts = []
                for x in (out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(x, torch.Tensor):
                        parts.append(tuple(x.shape))
                    else:
                        parts.append(type(x).__name__)
                output_shapes[name] = str(tuple(parts))
            else:
                if isinstance(out, torch.Tensor):
                    output_shapes[name] = str(tuple(out.shape))
                elif isinstance(out, (list, tuple)):
                    output_shapes[name] = str(tuple(
                        tuple(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__
                        for o in out
                    ))
                else:
                    output_shapes[name] = str(type(out).__name__)
        return hook

    # 注册各层 hook
    hooks = []
    # embedding: 输入 (B,3,N), 输出 (B,D,N)
    hooks.append(net.embedding.register_forward_pre_hook(_make_pre_hook("embedding", multi_input=False)))
    hooks.append(net.embedding.register_forward_hook(_make_hook("embedding", multi_output=False)))

    n_stages = getattr(net, "stages", len(net.GAM_list))
    for i in range(n_stages):
        # GAM: 输入 (xyz, points) 输出 (new_xyz, new_points, center)
        hooks.append(net.GAM_list[i].register_forward_pre_hook(_make_pre_hook(f"GAM-stage{i}", multi_input=True)))
        hooks.append(net.GAM_list[i].register_forward_hook(_make_hook(f"GAM-stage{i}", multi_output=True)))
        hooks.append(net.LFP_blocks_list[i].register_forward_pre_hook(_make_pre_hook(f"LFP-stage{i}", multi_input=True)))
        hooks.append(net.LFP_blocks_list[i].register_forward_hook(_make_hook(f"LFP-stage{i}", multi_output=False)))
        hooks.append(net.GFP_blocks_list[i].register_forward_pre_hook(_make_pre_hook(f"GFP-stage{i}", multi_input=False)))
        hooks.append(net.GFP_blocks_list[i].register_forward_hook(_make_hook(f"GFP-stage{i}", multi_output=False)))

    hooks.append(net.classifier.register_forward_pre_hook(_make_pre_hook("classifier", multi_input=False)))
    hooks.append(net.classifier.register_forward_hook(_make_hook("classifier", multi_output=False)))

    # 前向：使用 ModelNet40 的 batch
    net.eval()
    batch_count = 0
    with torch.no_grad():
        for data, _ in data_loader:
            if batch_count >= num_batches:
                break
            data = data.to(device)
            # 模型 forward 期望 x: (B, 3, N)
            if data.dim() == 3 and data.size(1) != 3:
                data = data.permute(0, 2, 1)
            _ = net(data)
            batch_count += 1

    for h in hooks:
        h.remove()

    # 按固定顺序组装结果
    order = ["embedding"]
    for i in range(n_stages):
        order.append(f"GAM-stage{i}")
        order.append(f"LFP-stage{i}")
        order.append(f"GFP-stage{i}")
    order.append("classifier")

    rows = []
    for name in order:
        inp = input_shapes.get(name, "—")
        out = output_shapes.get(name, "—")
        rows.append((name, inp, out))

    # 合并权重列（可选）
    param_shapes = _collect_layer_param_shapes(net) if include_weights else {}
    rows4 = []
    for name, inp, out in rows:
        if include_weights:
            params = param_shapes.get(name, [])
            w = _format_param_list(params, max_items=max_weight_items if max_weight_items > 0 else None)
        else:
            w = ""
        rows4.append((name, inp, out, w))

    if log_file:
        _write_tsv(rows4, log_file, include_weights=include_weights, weight_by_layer=param_shapes)
    if csv_file:
        _write_csv(rows4, csv_file, include_weights=include_weights)

    return rows


def main():
    parser = argparse.ArgumentParser(description="PointKAN-Elite 基于 ModelNet40 的每层 shape 分析")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite-pointkan_elite_exp-42",
                        help="检查点目录或 best_checkpoint.pth 路径")
    parser.add_argument("--num_points", type=int, default=1024, help="点云点数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", type=str, default="test", choices=("train", "test"),
                        help="使用 ModelNet40 的 train 或 test 分区")
    parser.add_argument("--num_batches", type=int, default=1, help="用于统计的 batch 数（取最后一轮的 shape）")
    parser.add_argument("--log", type=str, default=None,
                        help="shape 分析日志保存路径，默认 shape_analysis_<timestamp>.log")
    parser.add_argument("--csv", type=str, default=None,
                        help="若指定，则额外输出 CSV 文件到该路径（推荐 .csv）")
    parser.add_argument("--include_weights", action="store_true",
                        help="若指定，则在日志中额外输出各层块的参数(权重)shape")
    parser.add_argument("--max_weight_items", type=int, default=0,
                        help="限制每个层块输出的权重条目数（0 表示不限制；用于避免日志过大）")
    args = parser.parse_args()

    if args.log is None:
        args.log = os.path.join(SCRIPT_DIR, f"shape_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_path = get_checkpoint_path(args.checkpoint)
    net = build_model_and_load_checkpoint(checkpoint_path, device)

    dataset = ModelNet40(num_points=args.num_points, partition=args.partition)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    rows = collect_layer_shapes(
        net,
        device,
        loader,
        num_batches=args.num_batches,
        log_file=args.log,
        csv_file=args.csv,
        include_weights=args.include_weights,
        max_weight_items=args.max_weight_items,
    )

    # 同时打印到 stdout
    print("\nPointKAN-Elite 每层 Shape 分析 (ModelNet40, partition={}, batch_size={}, num_points={})\n".format(
        args.partition, args.batch_size, args.num_points))
    print("{:<16} {:<45} {}".format("Layer", "Input Shape(s)", "Output Shape(s)"))
    print("-" * 100)
    for name, inp, out in rows:
        print("{:<16} {:<45} {}".format(name, inp[:44] if len(inp) > 44 else inp, out[:44] if len(out) > 44 else out))
    if args.log:
        print("\nLog 已保存: {}".format(os.path.abspath(args.log)))
    if args.csv:
        print("CSV 已保存: {}".format(os.path.abspath(args.csv)))


if __name__ == "__main__":
    main()
