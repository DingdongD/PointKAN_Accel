#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考模型：从输入到线性插值重采样输出的完整过程，输出 FP32 十六进制，
便于与 Verilog 波形/日志逐项比对；同时可生成 t LUT 的 hex 文件。

与 Verilog fp32_resampler 及 lin_resample_1d 一致：
  src_pos[j] = j * (L-1) / (out_len-1)
  left = floor(src_pos), right = min(left+1, L-1)
  w = src_pos - left  → t_lut[j]
  y[j] = (1-w)*x[left] + w*x[right] = x[left] + (x[right]-x[left])*w

用法示例:
  # 5 点下采样到 3 点，打印输入/输出/t 的 hex，并写出 t_lut 与 golden 文件
  python ref_resampler.py 5 3 -o gold_5_3.hex --t-lut-out t_lut_5_3.hex

  # 使用自定义输入（浮点或 hex 文件）
  python ref_resampler.py 5 3 --in-float "1.0 2.0 3.0 4.0 5.0" -o gold.hex
  python ref_resampler.py 5 3 --in-hex in_5.hex -o gold.hex --t-lut-out t_lut.hex

  # 仅生成 t LUT，不跑插值
  python ref_resampler.py 5 3 --t-lut-only -o t_lut_5_3.hex

  # 与 Verilog 波形/日志比对：输出格式与 TB 一致，便于 diff
  #   [ref] out[j] = <hex>  对应  [TB]   out[j] = <hex>
  # 生成 gold_5_3.hex 后，TB 可用 $readmemh("gold_5_3.hex", gold_ref) 做自动比对。
"""
import struct
import argparse
import sys


def float_to_hex(f: float) -> str:
    """float -> IEEE754 单精度十六进制，8 位小写。"""
    bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return "{:08x}".format(bits)


def hex_to_float(h: str) -> float:
    """8 位十六进制字符串 -> float。"""
    u = int(h.strip(), 16)
    return struct.unpack(">f", struct.pack(">I", u))[0]


def lin_resample_1d_weights(L: int, out_len: int):
    """生成 out_len 个权重 w in [0, 1)，与 lin_resample_1d 一致。"""
    if out_len <= 0:
        return []
    if L == 1 or out_len == 1:
        return [0.0] * out_len
    src_pos = [j * (L - 1) / (out_len - 1) for j in range(out_len)]
    w = []
    for j in range(out_len):
        left = int(src_pos[j])
        w.append(src_pos[j] - left)
    return w


def lin_resample_1d(x: list, out_len: int):
    """
    一维线性插值重采样：x[0..L-1] -> y[0..out_len-1]。
    与 PyTorch/Verilog 一致：src_pos = j*(L-1)/(out_len-1), left=floor, w=frac,
    y = (1-w)*x[left] + w*x[right]。
    """
    L = len(x)
    if L == 0:
        return []
    if out_len == 0:
        return []
    if L == 1:
        return [float(x[0])] * out_len
    if out_len == 1:
        return [float(x[0])]

    src_pos = [j * (L - 1) / (out_len - 1) for j in range(out_len)]
    y = []
    for j in range(out_len):
        left = int(src_pos[j])
        right = min(left + 1, L - 1)
        w = src_pos[j] - left
        v = (1.0 - w) * float(x[left]) + w * float(x[right])
        y.append(v)
    return y


def main():
    ap = argparse.ArgumentParser(
        description="参考模型：输入→线性插值→输出，打印 hex 并可选输出 t LUT / golden 文件"
    )
    ap.add_argument("L", type=int, nargs="?", help="输入长度 N_IN（与 out_len 二选一与 -o 配合）")
    ap.add_argument("out_len", type=int, nargs="?", help="输出长度 N_OUT")
    ap.add_argument("-o", "--output", help="Golden 输出 hex 文件路径（每行一个 8 位 hex）")
    ap.add_argument("--t-lut-out", help="t 权重 LUT 的 hex 文件路径（可被 $readmemh 读入）")
    ap.add_argument("--t-lut-pad", type=int, default=0, help="t LUT 填到该行数（0=不填，行数=out_len；RTL 需 512 时可传 512）")
    ap.add_argument("--t-lut-only", action="store_true", help="仅生成 t LUT，不跑插值")
    ap.add_argument(
        "--in-float",
        type=str,
        metavar="STR",
        help="输入序列，空格分隔浮点，如 \"1.0 2.0 3.0 4.0 5.0\"",
    )
    ap.add_argument("--in-hex", metavar="FILE", help="输入 hex 文件，每行一个 8 位 hex（覆盖 L 时自动取长度）")
    ap.add_argument("--in-out", metavar="FILE", help="将当前输入向量写出为 hex 文件（L 行），供 TB 的 in.hex")
    ap.add_argument("--quiet", action="store_true", help="不打印逐项 hex，只写文件与摘要")
    ap.add_argument("--gold-pad", type=int, default=0, help="Golden 文件填充到该行数（0 不填充，TB 读少行会告警可忽略）")
    args = ap.parse_args()

    # 解析输入数据
    if args.in_hex:
        with open(args.in_hex) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        x = [hex_to_float(h) for h in lines]
        L = len(x)
        if args.L is not None and args.L != L:
            sys.stderr.write(f"Warning: --in-hex has {L} values, L={args.L} ignored\n")
    elif args.in_float:
        x = [float(s) for s in args.in_float.split()]
        L = len(x)
        if args.L is not None and args.L != L:
            sys.stderr.write(f"Warning: --in-float has {L} values, L={args.L} ignored\n")
    else:
        if args.L is None or args.out_len is None:
            ap.error("需要指定 L 与 out_len，或使用 --in-hex/--in-float 指定输入")
        L = args.L
        out_len = args.out_len
        # 默认输入：1.0, 2.0, ..., L
        x = [float(i + 1) for i in range(L)]

    if args.t_lut_only:
        if args.out_len is None:
            ap.error("--t-lut-only 需要指定 out_len（即第二个参数）")
        out_len = args.out_len
        weights = lin_resample_1d_weights(L, out_len)
        pad_to = max(len(weights), args.t_lut_pad) if args.t_lut_pad > 0 else len(weights)
        out_path = args.t_lut_out or args.output
        if not out_path:
            ap.error("--t-lut-only 需指定 -o 或 --t-lut-out")
        with open(out_path, "w") as f:
            for i in range(pad_to):
                w = weights[i] if i < len(weights) else 0.0
                f.write(float_to_hex(w) + "\n")
        print(f"Wrote t LUT {pad_to} lines to {out_path} ({len(weights)} used)")
        return

    out_len = args.out_len
    if out_len is None:
        ap.error("需要指定 out_len（第二个参数）")

    # t 权重
    weights = lin_resample_1d_weights(L, out_len)

    if args.in_out:
        with open(args.in_out, "w") as f:
            for v in x:
                f.write(float_to_hex(v) + "\n")
        print(f"[ref] Input: {len(x)} lines -> {args.in_out}")

    # 写 t LUT 文件
    if args.t_lut_out:
        pad_to = max(len(weights), args.t_lut_pad) if args.t_lut_pad > 0 else len(weights)
        with open(args.t_lut_out, "w") as f:
            for i in range(pad_to):
                w = weights[i] if i < len(weights) else 0.0
                f.write(float_to_hex(w) + "\n")
        print(f"[ref] t LUT: {pad_to} lines -> {args.t_lut_out}")

    if not args.quiet:
        print("[ref] === Input (FP32 hex) ===")
        for i, v in enumerate(x):
            print(f"  in[{i}] = {float_to_hex(v)}  ({v})")
        print("[ref] === t LUT (w = src_pos - floor(src_pos)) ===")
        for j, w in enumerate(weights):
            print(f"  t_lut[{j}] = {float_to_hex(w)}  ({w})")

    # 插值
    y = lin_resample_1d(x, out_len)

    if not args.quiet:
        print("[ref] === Output (FP32 hex, 可与 Verilog out_data 比对) ===")
        for j, v in enumerate(y):
            print(f"  out[{j}] = {float_to_hex(v)}  ({v})")

    # 写 golden 文件
    if args.output:
        pad = max(len(y), args.gold_pad) if args.gold_pad else len(y)
        with open(args.output, "w") as f:
            for i in range(pad):
                v = y[i] if i < len(y) else 0.0
                f.write(float_to_hex(v) + "\n")
        print(f"[ref] Golden: {pad} lines -> {args.output}" + (f" ({len(y)} used, {pad - len(y)} padded)" if pad > len(y) else ""))

    print(f"[ref] Done: L={L} -> out_len={out_len}")


if __name__ == "__main__":
    main()
