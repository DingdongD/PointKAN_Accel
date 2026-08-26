#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 NeuroSim 底层 ADC 动态能耗与延迟（readDynamicEnergy / readLatency）

NeuroSim main.cpp 在「每层」与「整网 Summary 后」打印：
  ----------- ADC (or S/As and precharger for SRAM) readLatency is : X ns
  ----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : X pJ

另在 Summary 中打印芯片级 ADC 面积：
  Total ADC (or S/As and precharger for SRAM) Area on chip : X um^2

若 NeuroSim 源码已启用 Param.printAdcSubBreakdown 并在 main.cpp 中打印，stdout 中还会有：
  ----------- ADC sub-breakdown readDynamicEnergyArray / precharger / senseAmp / multilevelSenseAmp / multilevelSAEncoder / sarADC ...

本脚本：生成最小 config + weight/input CSV，调用与 pointkan_sim 相同的 NeuroSIM/main，
解析上述字段，便于与 pointkan_sim 中使用的 readDynamicEnergy（整层 read+buffer，不含 ADC 分项）对照。

用法（在 PointKAN-pytorch 仓库根目录）:
  python pointkan_sim/tools/verify_neurosim_adc_power.py
  python pointkan_sim/tools/verify_neurosim_adc_power.py --num_layers 2 --neurosim_root /path/to/Inference_pytorch
  NEUROSIM_ROOT=/path/to/Inference_pytorch python pointkan_sim/tools/verify_neurosim_adc_power.py
  python pointkan_sim/tools/verify_neurosim_adc_power.py --num_col_muxed 8   # 与 Param.numColMuxed 一致
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 仓库根目录（PointKAN-pytorch），便于 `python pointkan_sim/tools/...` 直接运行
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pointkan_sim.operators.neurosim_adapter import (  # noqa: E402
    _run_neurosim_main,
    _write_matrix_activation_conv,
    _write_matrix_weight,
    parse_clock_period_ns,
)
from pointkan_sim.utils.shared_helpers import stretch_input_conv1d as _stretch_input_conv1d  # noqa: E402

_P_ADC_LAT = re.compile(
    r"-----------\s*ADC \(or S/As and precharger for SRAM\) readLatency is\s*:\s*([\d.eE+-]+)\s*ns"
)
_P_ADC_EN = re.compile(
    r"-----------\s*ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ"
)
_P_ADC_AREA = re.compile(
    r"Total ADC \(or S/As and precharger for SRAM\) Area on chip\s*:\s*([\d.eE+-]+)\s*um\^2",
    re.IGNORECASE,
)
_P_CHIP_READ_EN = re.compile(
    r"Chip total readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ",
    re.IGNORECASE,
)
# Param.cpp 默认 pipeline=true 时 main 打印为 pipeline-system 行，而非 Chip total readDynamicEnergy
_P_CHIP_PIPELINE_READ_EN = re.compile(
    r"Chip pipeline-system readDynamicEnergy \(per image\) is:\s*([\d.eE+-]+)\s*pJ",
    re.IGNORECASE,
)
_P_SUBARRAY_SIZE = re.compile(
    r"User-defined\s+SubArray\s+Size:\s*([\d.eE+-]+)x([\d.eE+-]+)",
    re.IGNORECASE,
)
_P_TILE_LAYER = re.compile(r"^layer(\d+)\s*:\s*([\d.eE+-]+)", re.MULTILINE)
_P_SAR_SUB_PJ = re.compile(
    r"ADC sub-breakdown sarADC is\s*:\s*([\d.eE+-]+)\s*pJ",
    re.IGNORECASE,
)


def _parse_all_adc_breakdown(stdout: str) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    返回 (per_layer [(adc_lat_ns, adc_en_pj), ...], chip).

    main 在每一层 breakdown 后打印一对 ADC lat/en；在 Summary 之后再打印一对 **芯片级**
    汇总（非 pipeline 时 chipEnergyADC 为各层 ADC 能量之和）。最后一对视为 chip 级。
    """
    lat_matches = [float(m.group(1)) for m in _P_ADC_LAT.finditer(stdout)]
    en_matches = [float(m.group(1)) for m in _P_ADC_EN.finditer(stdout)]
    n = min(len(lat_matches), len(en_matches))
    if n == 0:
        return [], None
    pairs: List[Tuple[float, float]] = [(lat_matches[i], en_matches[i]) for i in range(n)]
    chip = pairs[-1]
    per_layer = pairs[:-1]
    return per_layer, chip


def _parse_adc_area_um2(stdout: str) -> Optional[float]:
    m = _P_ADC_AREA.search(stdout)
    if m:
        return float(m.group(1))
    return None


def _parse_chip_total_read_energy_pj(stdout: str) -> Optional[float]:
    m = _P_CHIP_READ_EN.search(stdout)
    if m:
        return float(m.group(1))
    m = _P_CHIP_PIPELINE_READ_EN.search(stdout)
    if m:
        return float(m.group(1))
    return None


def _parse_subarray_size_row_col(stdout: str) -> Optional[Tuple[float, float]]:
    """解析 FloorPlan 中 User-defined SubArray Size: numRow x numCol（与 NeuroSim 打印一致）。"""
    m = _P_SUBARRAY_SIZE.search(stdout)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _parse_tile_count_per_layer(stdout: str) -> List[float]:
    """解析 '# of tile used for each layer' 下 layerN: M，返回每层 tile 数（索引 0 对应 layer1）。"""
    lines = stdout.splitlines()
    in_section = False
    tiles: List[float] = []
    for line in lines:
        if "# of tile used for each layer" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("---") or "Speed-up" in line or "Utilization" in line:
                break
            m = _P_TILE_LAYER.match(line.strip())
            if m:
                idx = int(m.group(1)) - 1
                val = float(m.group(2))
                while len(tiles) < idx:
                    tiles.append(0.0)
                if len(tiles) == idx:
                    tiles.append(val)
                else:
                    tiles[idx] = val
    return tiles


def _parse_sar_adc_energy_pj_first(stdout: str) -> Optional[float]:
    """取第一条 ADC sub-breakdown sarADC 行的 pJ（单层时与 chip 段重复，数值应一致）。"""
    m = _P_SAR_SUB_PJ.search(stdout)
    if m:
        return float(m.group(1))
    return None


def _extract_adc_sub_breakdown_lines(stdout: str) -> List[str]:
    """从 main stdout 中抽出 sarADC / MLSA 等 sub-breakdown 行（NeuroSim 补丁 + printAdcSubBreakdown=true）。"""
    out: List[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if "ADC sub-breakdown" in s or "Chip-level ADC sub-breakdown" in s:
            out.append(s)
    return out


def _num_col_per_synapse(synapse_bit: int, cell_bit: int = 1) -> int:
    """与 main.cpp 中 ceil(synapseBit/cellBit) 一致（默认 cellBit=1）。"""
    return max(1, (synapse_bit + cell_bit - 1) // cell_bit)


def _min_layer_channels_for_floorplan(num_row_subarray: int, num_col_per_synapse: int) -> Tuple[int, int]:
    """
    Param.novelMapping 默认 true（Chip.cpp）时：
    - 1x1 卷积且 numRowPerSynapse=1：若 in_c < numRowSubArray，该层不进 novel mapping，maxPESizeNM 可保持为 0，
      随后 ChipFloorPlan 要求 maxPESizeNM >= 2*numRowSubArray，直接报错。
      故需 min_in_c >= numRowSubArray。
    - novel 层还使 minCube=2^ceil(log2(out_c*ncs)) 参与 maxPESizeNM，需 >= 2*numRowSubArray。
    - 落在 conventional 的层需 maxTileSizeCM >= 4*numRowSubArray。
    取满足 novel + conventional 的最小 out_c，并与 32 取 max。
    """
    min_in_c = max(1, num_row_subarray)

    need_nm = 2 * num_row_subarray
    min_out_nm = 1
    while min_out_nm < 10_000_000:
        x = min_out_nm * num_col_per_synapse
        if x > 0 and 2.0 ** math.ceil(math.log2(x)) >= need_nm:
            break
        min_out_nm += 1

    need_cm = 4 * num_row_subarray
    min_out_cm = 1
    while min_out_cm < 10_000_000:
        x = min_out_cm * num_col_per_synapse
        if x > 0 and 2.0 ** math.ceil(math.log2(x)) >= need_cm:
            break
        min_out_cm += 1

    min_out_c = max(32, min_out_nm, min_out_cm)
    return min_in_c, min_out_c


def _min_cube_out(out_c: int, num_col_per_synapse: int) -> int:
    """与 Chip.cpp 一致：minCube = 2^ceil(log2(out_c * numColPerSynapse))。"""
    x = max(1, out_c * num_col_per_synapse)
    return int(2.0 ** math.ceil(math.log2(x)))


def _floorplan_violations(in_c: int, out_c: int, subarray: int, num_col_per_synapse: int) -> List[str]:
    """
    固定 in_c/out_c 时，检查是否满足 ChipFloorPlan（novelMapping 默认 true，1x1 卷积）。
    不满足时 NeuroSim 会报 SubArray Size is too large。
    """
    errs: List[str] = []
    min_cube = _min_cube_out(out_c, num_col_per_synapse)
    if min_cube < 4 * subarray:
        errs.append(
            f"minCube(由 out_c×numColPerSynapse 得)={min_cube} < 4×subarray={4 * subarray}，"
            f"需增大 --out_c 或减小 --subarray（或恢复默认自动抬高通道）。"
        )
    if min_cube < 2 * subarray:
        errs.append(
            f"minCube={min_cube} < 2×subarray={2 * subarray}（novel 映射侧 maxPESizeNM）。"
        )
    if in_c < subarray:
        errs.append(
            f"in_c={in_c} < --subarray={subarray}：1×1 卷积下需 in_c≥subarray 才会进入 novel mapping，"
            f"否则 maxPESizeNM=0 会直接报错。"
        )
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 NeuroSim 底层 ADC 能耗/延迟（stdout breakdown）")
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default=os.environ.get("NEUROSIM_ROOT", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"),
        help="Inference_pytorch 根目录（含 NeuroSIM/main）",
    )
    parser.add_argument("--num_layers", type=int, default=1, help="测试网络层数（每层一行 config）；默认 1 最快")
    parser.add_argument("--wl_weight", type=int, default=8)
    parser.add_argument("--wl_activation", type=int, default=8)
    parser.add_argument(
        "--subarray",
        type=int,
        default=32,
        help="numRowSubArray（与 Param.numColSubArray 独立；列数见 Param.cpp 默认 128）。"
        "大 subarray 时需更大 out_c 以满足 Chip FloorPlan（脚本可自动抬高 in_c/out_c）",
    )
    parser.add_argument(
        "--in_c",
        type=int,
        default=None,
        help="测试层输入通道数；默认按 subarray+wl_weight 自动取满足 FloorPlan 的最小值（与 out_c 至少 32）",
    )
    parser.add_argument(
        "--out_c",
        type=int,
        default=None,
        help="测试层输出通道数；默认同上。若手动设得过小，128 子阵列会报 SubArray Size is too large",
    )
    parser.add_argument(
        "--no-auto-channels",
        action="store_true",
        help="关闭按 FloorPlan 自动抬高 in_c/out_c；未写的通道默认 32。"
        "扫 subarray 时请改用固定通道 + 校验，或只改 --len 改变输入长度（不改变 out_c）。",
    )
    parser.add_argument(
        "--len",
        type=int,
        default=32,
        dest="spatial_len",
        help="IFM 宽度 in_w（脚本里 in_h=1），即「输入位置数」numInVector；只加长输入不改 out_c。默认 32",
    )
    parser.add_argument("--parallel_read", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--clk_freq",
        type=float,
        default=None,
        help="Hz，芯片/数字外围时钟 param->clkFreq；与 SAR ADC 时钟 --adc_clk_freq 可独立",
    )
    parser.add_argument(
        "--adc_clk_freq",
        type=float,
        default=None,
        help="Hz，仅 SAR ADC 的转换时钟 param->adcClkFreq，默认 1e9（不传 main 时）",
    )
    parser.add_argument(
        "--sar_column_power_w",
        type=float,
        default=None,
        help="W，用户 SAR Column_Power（与 SarADC.cpp 经验式中 Column_Power 同量纲）；>0 时跳过经验式，仍用 P×(1+temp)×(log2(level)+1)/adcClk 得能量",
    )
    parser.add_argument(
        "--sar_column_power_uw",
        type=float,
        default=None,
        help="µW，与 --sar_column_power_w 二选一；内部换算为 W 传入 main",
    )
    parser.add_argument(
        "--technode",
        type=int,
        default=32,
        help="nm；默认 32（NeuroSim 默认 Param 可能为不支持的 28 nm，会导致 exit 255）",
    )
    parser.add_argument("--keep_tmp", action="store_true", help="保留临时目录")
    parser.add_argument(
        "--num_col_muxed",
        type=int,
        default=8,
        help="numColMuxed：经 NeuroSIM/main 可选 argv 第 9 槽传入（与 main.cpp 一致）；--verbose 时亦用于估算 SAR 路数",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印完整解析与 sub-breakdown 原文；默认仅打印每 tile SAR 能耗/latency/功耗",
    )
    args = parser.parse_args()

    if args.technode == 28:
        print("注意: NeuroSim 不支持 28 nm，已改为 32 nm。")
        args.technode = 32

    if args.sar_column_power_w is not None and args.sar_column_power_uw is not None:
        print("错误: --sar_column_power_w 与 --sar_column_power_uw 不能同时指定。")
        return 1
    sar_column_power_w: Optional[float] = args.sar_column_power_w
    if args.sar_column_power_uw is not None:
        sar_column_power_w = args.sar_column_power_uw * 1e-6

    ncs = _num_col_per_synapse(args.wl_weight, 1)
    if args.no_auto_channels:
        in_c = args.in_c if args.in_c is not None else 32
        out_c = args.out_c if args.out_c is not None else 32
    else:
        min_in_auto, min_out_auto = _min_layer_channels_for_floorplan(args.subarray, ncs)
        # 某一维显式给出则不再对该维自动放大（便于固定 out_c 只让 in_c 随 FloorPlan 抬高）
        if args.in_c is None:
            in_c = max(32, min_in_auto)
        else:
            in_c = args.in_c
        if args.out_c is None:
            out_c = max(32, min_out_auto)
        else:
            out_c = args.out_c

    viol = _floorplan_violations(in_c, out_c, args.subarray, ncs)
    if viol:
        print("错误: 当前 in_c/out_c 与 --subarray 不满足 NeuroSim FloorPlan：")
        for v in viol:
            print("  -", v)
        print(
            "提示: 只增加「输入长度」请用 --len（不改 out_c）；"
            "或增大 out_c / 减小 --subarray；或去掉 --no-auto-channels 使用自动通道。"
        )
        return 1

    main_dir = os.path.join(args.neurosim_root, "NeuroSIM")
    if not os.path.isfile(os.path.join(main_dir, "main")):
        print(f"错误: 未找到 {main_dir}/main ，请指定 --neurosim_root 或编译 NeuroSIM。")
        return 1

    tmpdir = tempfile.mkdtemp(dir=main_dir, prefix="verify_adc_")
    rel_prefix = os.path.basename(tmpdir)
    try:
        config_rows: List[str] = []
        pairs: List[Tuple[str, str]] = []
        for idx in range(args.num_layers):
            L = args.spatial_len
            k_w = 1
            in_h, in_w = 1, L
            k_h = 1
            pool, stride = 0, 1
            config_rows.append(f"{in_h},{in_w},{in_c},{k_h},{k_w},{out_c},{pool},{stride}")
            wp = os.path.join(tmpdir, f"weight{idx}.csv")
            ip = os.path.join(tmpdir, f"input{idx}.csv")
            w = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
            _write_matrix_weight(w, wp)
            inp = _stretch_input_conv1d(in_c, in_w, k_w, stride)
            _write_matrix_activation_conv(inp, args.wl_activation, ip)
            pairs.append(
                (os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv"))
            )

        cfg_path = os.path.join(tmpdir, "config.csv")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(config_rows) + "\n")
        rel_config = os.path.join(rel_prefix, "config.csv")

        ok, out, err = _run_neurosim_main(
            rel_config,
            pairs,
            args.neurosim_root,
            wl_weight=args.wl_weight,
            wl_activation=args.wl_activation,
            subarray=args.subarray,
            parallel_read=args.parallel_read,
            timeout_s=args.timeout,
            clk_freq_hz=args.clk_freq,
            technode=args.technode,
            adc_clk_freq_hz=args.adc_clk_freq,
            sar_column_power_w=sar_column_power_w,
            num_col_muxed=int(args.num_col_muxed) if int(args.num_col_muxed) > 0 else None,
        )
        if not ok:
            print("NeuroSim 运行失败:", err or "")
            if out:
                print("--- stdout (尾部) ---\n", "\n".join(out.splitlines()[-40:]))
            return 1

        per_layer, chip = _parse_all_adc_breakdown(out)
        area = _parse_adc_area_um2(out)
        chip_total_read_pj = _parse_chip_total_read_energy_pj(out)
        clk_ns = parse_clock_period_ns(out)
        sar_pj = _parse_sar_adc_energy_pj_first(out)
        sub_rc = _parse_subarray_size_row_col(out)
        tile_counts = _parse_tile_count_per_layer(out)

        if args.verbose:
            print("======== NeuroSim ADC 分项（stdout）========")
            print(f"层数: {args.num_layers}")
            if clk_ns is not None:
                print(f"时钟周期: {clk_ns:.6f} ns（约 {1000.0 / clk_ns:.2f} MHz）")
            if area is not None:
                print(f"芯片 ADC 总面积: {area:.4f} um^2")
            if per_layer:
                for i, (lat_ns, en_pj) in enumerate(per_layer, start=1):
                    p_mw = (en_pj / lat_ns) if lat_ns > 0 else float("nan")
                    print(
                        f"  Layer {i}: ADC latency = {lat_ns:.4f} ns, "
                        f"ADC readDynamicEnergy = {en_pj:.6f} pJ; E/t ≈ {p_mw:.4f} mW"
                    )
            if chip is not None:
                c_lat_v, c_en_v = chip
                print(
                    f"芯片级 Summary — ADC latency: {c_lat_v:.4f} ns, "
                    f"ADC readDynamicEnergy: {c_en_v:.6f} pJ"
                )
                if chip_total_read_pj is not None:
                    frac = (c_en_v / chip_total_read_pj * 100.0) if chip_total_read_pj > 0 else float("nan")
                    print(f"整网 readDynamicEnergy: {chip_total_read_pj:.6f} pJ；ADC/整网 ≈ {frac:.2f}%")
            sub_lines = _extract_adc_sub_breakdown_lines(out)
            print()
            print("======== sarADC sub-breakdown（stdout）========")
            if sub_lines:
                for ln in sub_lines:
                    print(f"  {ln}")
            else:
                print("  （无 sub-breakdown；需补丁 main + Param.printAdcSubBreakdown）")
                print(f"  main: {os.path.join(args.neurosim_root, 'NeuroSIM', 'main')}")
            if sub_rc is not None:
                nrow, ncol = sub_rc
                mux = max(1, int(args.num_col_muxed))
                chains = int(math.ceil(ncol / float(mux)))
                print(f"SubArray {nrow:g}x{ncol:g}；numColMuxed={mux} → 每子阵列约 {chains} 路 SAR")

        # 默认：仅 4 行
        print("每 tile SAR ADC")
        if chip is None or sar_pj is None or not tile_counts or args.num_layers != 1:
            print("  无法解析（需单层 + sar sub-breakdown + tile）。加 -v 查看详情。")
        else:
            c_lat, _c_en = chip
            nt = tile_counts[0]
            if nt <= 0:
                print("  tile 数=0。")
            else:
                e_tile = sar_pj / nt
                p_tile_mw = e_tile / c_lat if c_lat > 0 else float("nan")
                print(f"  E_sar_per_tile = {e_tile:.4f} pJ  （E_sar_layer {sar_pj:.4f} pJ / N_tile {nt:g}）")
                print(f"  t_ADC_layer = {c_lat:.4f} ns  （NeuroSim 仅整层，无 per-tile）")
                print(f"  P_sar_per_tile = {p_tile_mw:.4f} mW  （E_sar_per_tile / t_ADC_layer）")
        if args.keep_tmp:
            print(f"临时目录: {tmpdir}")
        return 0
    finally:
        if not args.keep_tmp and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
