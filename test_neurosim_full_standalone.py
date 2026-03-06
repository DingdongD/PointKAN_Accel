#!/usr/bin/env python3
"""
NeuroSim 全网络调用 — 独立测试脚本
====================================

用于单独运行、排查 NeuroSim main 崩溃（如 returncode=-11/SIGSEGV）。
不依赖 pointkan_sim 包，可直接：python test_neurosim_full_standalone.py [选项]

用法:
  python test_neurosim_full_standalone.py
  python test_neurosim_full_standalone.py --num_layers 2
  python test_neurosim_full_standalone.py --num_layers 8 --keep_tmp
  python test_neurosim_full_standalone.py --neurosim_root /path/to/DNN_NeuroSim_V1.4/Inference_pytorch

若 stdout 出现 "SubArray Size is too large"，可尝试: --subarray 64 --parallel_read 64（若仍 returncode=-11，
则可能是 NeuroSim 在报错后未正常 exit 而崩溃，需在 DNN_NeuroSim 内修改 Param 或错误处理）。
建议先跑 --num_layers 1，再逐步增大以定位是哪一层或层数导致崩溃。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy")
    sys.exit(1)


# ---------- 仿 pointkan_sim 的层信息（仅字段用于生成 config/CSV）----------
@dataclass
class LayerSpec:
    name: str
    layer_type: str  # "conv" | "fc"
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: int = 1
    length: int = 0
    in_features: int = 0
    out_features: int = 0
    tokens: int = 0


# ---------- 与 neurosim_adapter / neurosim_full 一致的 CSV 生成 ----------
def _dec2bin_vec(x, n: int) -> Tuple[list, list]:
    out, scale = [], []
    delta = 1.0 / (2 ** (n - 1))
    x_int = np.floor(x / delta).astype(np.int64)
    base = 2 ** (n - 1)
    y = (x_int < 0).astype(np.float64)
    rest = x_int + base * y
    out.append(y.copy())
    scale.append(-base * delta)
    for _ in range(n - 1):
        base /= 2
        y = (rest >= base).astype(np.float64)
        rest -= base * y
        out.append(y.copy())
        scale.append(base * delta)
    return out, scale


def _write_matrix_weight(weight_matrix: np.ndarray, filepath: str) -> None:
    w = weight_matrix.reshape(weight_matrix.shape[0], -1).T
    np.savetxt(filepath, w, delimiter=",", fmt="%.6f")


def _write_matrix_activation_conv(input_matrix: np.ndarray, bit_length: int, filepath: str) -> None:
    """NeuroSim 期望 input CSV 为 (K 行, num_patches*bit_length 列)，与 VMM_CIM 一致。"""
    data = input_matrix[0, :, :]
    num_patches, K = data.shape
    filled_bin, _ = _dec2bin_vec(data.ravel(), bit_length)
    filled_b = np.zeros((num_patches, K * bit_length), dtype="U1")
    for i, b in enumerate(filled_bin):
        b_2d = b.reshape(num_patches, K)
        filled_b[:, i::bit_length] = b_2d
    out = np.empty((K, num_patches * bit_length), dtype="U1")
    for k in range(K):
        for p in range(num_patches):
            out[k, p * bit_length : (p + 1) * bit_length] = filled_b[p, k * bit_length : (k + 1) * bit_length]
    np.savetxt(filepath, out, delimiter=",", fmt="%s")


def _stretch_input_conv1d(in_c: int, in_w: int, k_w: int, stride: int = 1) -> np.ndarray:
    num_patches = max(1, (in_w - k_w) // stride + 1)
    K = in_c * k_w
    return np.random.randn(1, num_patches, K).astype(np.float32)


# ---------- 解析 main 的 stdout ----------
PATTERN_READ_LATENCY = re.compile(r"layer\d+'s readLatency is:\s*([\d.eE+-]+)\s*ns")
PATTERN_READ_ENERGY = re.compile(r"layer\d+'s readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")
PATTERN_BUFFER_LATENCY = re.compile(r"layer\d+'s buffer latency is:\s*([\d.eE+-]+)\s*ns")
PATTERN_BUFFER_ENERGY = re.compile(r"layer\d+'s buffer readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ")


def parse_layer_results(stdout: str) -> List[Tuple[float, float, float, float]]:
    lines = stdout.splitlines()
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "layer" in line and "readLatency is:" in line and "buffer" not in line:
            read_lat = PATTERN_READ_LATENCY.search(line)
            read_ns = float(read_lat.group(1)) if read_lat else 0.0
            read_pj = buf_ns = buf_pj = 0.0
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "readDynamicEnergy is:" in l and "buffer" not in l and "leakage" not in l:
                    m = PATTERN_READ_ENERGY.search(l)
                    if m:
                        read_pj = float(m.group(1))
                    break
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "buffer latency" in l:
                    m = PATTERN_BUFFER_LATENCY.search(l)
                    if m:
                        buf_ns = float(m.group(1))
                    break
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j]
                if "buffer readDynamicEnergy" in l:
                    m = PATTERN_BUFFER_ENERGY.search(l)
                    if m:
                        buf_pj = float(m.group(1))
                    break
            results.append((read_ns, read_pj, buf_ns, buf_pj))
        i += 1
    return results


# ---------- 构造与 PointKAN 真实前向相近的层列表（前 N 层）----------
def make_pointkan_like_layers(num_layers: int) -> List[LayerSpec]:
    """前几层与 PointKAN-elite 真实前向的 conv/linear 顺序与维度接近。"""
    # 简化版：交替小 conv + fc，维度偏小便于 NeuroSim 不崩
    layers = []
    for i in range(num_layers):
        if i % 2 == 0:
            layers.append(LayerSpec(
                name=f"conv_{i}",
                layer_type="conv",
                in_channels=min(256, 128),
                out_channels=min(128, 256),
                kernel_size=1,
                length=min(512, 64),
            ))
        else:
            layers.append(LayerSpec(
                name=f"fc_{i}",
                layer_type="fc",
                in_features=min(256, 128),
                out_features=min(128, 256),
                tokens=1,
            ))
    return layers


def run_neurosim_main(
    config_path: str,
    weight_input_pairs: List[Tuple[str, str]],
    main_dir: str,
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    parallel_read: int = 128,
    timeout_s: int = 120,
) -> Tuple[bool, str, str, int]:
    """调用 NeuroSIM/main，返回 (ok, stdout, stderr, returncode)。"""
    """返回 (ok, stdout, stderr, returncode)。"""
    main_exe = os.path.join(main_dir, "main")
    if not os.path.isfile(main_exe):
        return False, "", f"main 不存在: {main_exe}", -1
    args = [
        main_exe,
        config_path,
        str(wl_weight),
        str(wl_activation),
        str(subarray),
        str(parallel_read),
    ]
    for wf, inf in weight_input_pairs:
        args.append(wf)
        args.append(inf)
    try:
        proc = subprocess.run(
            args,
            cwd=main_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            err = f"returncode={proc.returncode}"
            if proc.stderr:
                err += "\nstderr:\n" + (proc.stderr.strip()[:2000] or "")
            return False, proc.stdout or "", err, proc.returncode
        return True, proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return False, "", "timeout", -1
    except Exception as e:
        return False, "", str(e), -1


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSim 全网络独立测试（便于排查 main 崩溃）")
    parser.add_argument("--neurosim_root", type=str, default="/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")
    parser.add_argument("--num_layers", type=int, default=4, help="参与测试的层数，可从 1 逐步增大排查")
    parser.add_argument("--keep_tmp", action="store_true", help="保留临时目录不删除，便于查看生成的 CSV")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--wl_weight", type=int, default=8)
    parser.add_argument("--wl_activation", type=int, default=8)
    parser.add_argument("--subarray", type=int, default=32, help="NeuroSim SubArray 行数；若报 SubArray Size is too large 可改为 64")
    parser.add_argument("--parallel_read", type=int, default=32, help="NeuroSim parallel read")
    args = parser.parse_args()

    main_dir = os.path.join(args.neurosim_root, "NeuroSIM")
    if not os.path.isdir(main_dir):
        print(f"错误: NeuroSIM 目录不存在: {main_dir}")
        return 1

    layers = make_pointkan_like_layers(args.num_layers)
    print(f"层数: {len(layers)}")
    for i, L in enumerate(layers):
        if L.layer_type == "conv":
            print(f"  [{i}] {L.name} conv in_ch={L.in_channels} out_ch={L.out_channels} L={L.length} K={L.kernel_size}")
        else:
            print(f"  [{i}] {L.name} fc in_f={L.in_features} out_f={L.out_features} tokens={L.tokens}")

    tmpdir = tempfile.mkdtemp(dir=main_dir, prefix="pointkan_test_")
    rel_prefix = os.path.basename(tmpdir)
    print(f"\n临时目录: {tmpdir}")
    print(f"相对前缀: {rel_prefix} (cwd={main_dir})")

    config_rows = []
    weight_input_pairs: List[Tuple[str, str]] = []

    try:
        for idx, info in enumerate(layers):
            if info.layer_type == "conv":
                in_h, in_w = 1, max(1, info.length)
                in_c, out_c = max(1, info.in_channels), max(1, info.out_channels)
                k_h, k_w = 1, max(1, info.kernel_size)
                stride, pool = 1, 0
                config_rows.append(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}")
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                _write_matrix_weight(weight_data, weight_path)
                input_stretched = _stretch_input_conv1d(in_c, in_w, k_w, stride)
                _write_matrix_activation_conv(input_stretched, args.wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))
            else:
                in_f, out_f = max(1, info.in_features), max(1, info.out_features)
                M = max(1, info.tokens)
                in_h, in_w = 1, M
                in_c, out_c = in_f, out_f
                k_h, k_w = 1, 1
                stride, pool = 1, 0
                config_rows.append(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}")
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                _write_matrix_weight(weight_data, weight_path)
                input_stretched = np.random.randn(1, num_patches, K).astype(np.float32)
                _write_matrix_activation_conv(input_stretched, args.wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))

        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write("\n".join(config_rows) + "\n")

        rel_config = os.path.join(rel_prefix, "config.csv")
        cmd = [
            os.path.join(main_dir, "main"),
            rel_config,
            str(args.wl_weight),
            str(args.wl_activation),
            str(args.subarray),
            str(args.parallel_read),
        ]
        for wf, inf in weight_input_pairs:
            cmd.extend([wf, inf])
        print("\n执行的命令 (cwd=%s):" % main_dir)
        print("  " + " \\\n    ".join(cmd[:8] + ["..."] if len(cmd) > 8 else cmd))

        ok, out, err, returncode = run_neurosim_main(
            rel_config,
            weight_input_pairs,
            main_dir,
            wl_weight=args.wl_weight,
            wl_activation=args.wl_activation,
            subarray=args.subarray,
            parallel_read=args.parallel_read,
            timeout_s=args.timeout,
        )

        print("\n---------- 结果 ----------")
        print(f"  returncode: {returncode}")
        print(f"  ok: {ok}")
        print(f"  stdout 长度: {len(out)} 字符")
        if err:
            print(f"  stderr:\n{err}")
        if out:
            print("\n---------- stdout 前 80 行 ----------")
            for i, line in enumerate(out.splitlines()[:80]):
                print(f"  {i+1}: {line}")
            if out.count("\n") > 80:
                print("  ...")

        if ok and out:
            parsed = parse_layer_results(out)
            print(f"\n解析到 {len(parsed)} 层:")
            for i, (r_ns, r_pj, b_ns, b_pj) in enumerate(parsed):
                lat = r_ns + b_ns
                en = (r_pj + b_pj) / 1000.0
                print(f"  layer{i+1}: latency_ns={lat:.2f}, energy_nj={en:.4f}")
    finally:
        if not args.keep_tmp and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"\n已删除临时目录: {tmpdir}")
        elif args.keep_tmp:
            print(f"\n已保留临时目录: {tmpdir}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
