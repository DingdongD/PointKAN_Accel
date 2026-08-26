#!/usr/bin/env python3
"""
NeuroSim 位宽扫描：对给定 Conv/FC  workload 枚举 (wl_weight, wl_activation)，
调用全网络 NeuroSim（优先 pybind，可选子进程），汇总每层与总 latency / energy。

用法（任选其一；``python -m pointkan_sim...`` 要求 **包在 sys.path 上**，见下）::

  # 推荐：先进入含 pointkan_sim 的仓库根再 -m
  cd /path/to/PointKAN-pytorch
  python -m pointkan_sim.tools.neurosim_bitwidth_latency_sweep --help

  # 在任意 cwd：显式 PYTHONPATH
  PYTHONPATH=/path/to/PointKAN-pytorch python -m pointkan_sim.tools.neurosim_bitwidth_latency_sweep --help

  # 任意 cwd：直接跑脚本文件（会自行把仓库根插入 sys.path）
  python /path/to/PointKAN-pytorch/pointkan_sim/tools/neurosim_bitwidth_latency_sweep.py --help

若在 ``/workspace`` 下报 ``No module named 'pointkan_sim'``，说明未把
``.../PointKAN-pytorch`` 加入模块搜索路径；请 ``cd PointKAN-pytorch`` 或设 ``PYTHONPATH``。

依赖：DNN_NeuroSim 已编译（NeuroSIM/main 或 neurosim_core）。
"""

from __future__ import annotations

import argparse
import csv
import math
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 允许「直接 python 本文件」运行
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pointkan_sim.integration.neurosim_full import (  # noqa: E402
    NeuroSimFullResult,
    run_neurosim_full,
    run_neurosim_full_pybind,
)
from pointkan_sim.integration.real_data_eval import ConvLinearLayerInfo  # noqa: E402


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def _build_probe_layers(args: argparse.Namespace) -> List[ConvLinearLayerInfo]:
    if args.layer == "conv":
        return [
            ConvLinearLayerInfo(
                name="probe_conv",
                layer_type="conv",
                in_channels=int(args.in_ch),
                out_channels=int(args.out_ch),
                kernel_size=int(args.kernel),
                length=int(args.length),
            )
        ]
    return [
        ConvLinearLayerInfo(
            name="probe_fc",
            layer_type="fc",
            in_features=int(args.in_f),
            out_features=int(args.out_f),
            tokens=int(args.tokens),
        )
    ]


def _run_once(
    *,
    conv_linear_list: List[ConvLinearLayerInfo],
    wl_weight: int,
    wl_activation: int,
    backend: str,
    neurosim_root: str,
    neurosim_so_dir: str,
    subarray: int,
    parallel_read: int,
    clk_freq_hz: Optional[float],
    technode: Optional[int],
    adc_clk_freq_hz: Optional[float],
    sar_column_power_w: Optional[float],
    max_in_vectors: Optional[int],
    timeout_s: Optional[int],
    clk_mode: str,
    cell_bit: int,
    align_col_mux_to_weight: bool,
    num_col_muxed: Optional[int],
) -> Tuple[NeuroSimFullResult, Optional[int]]:
    """
    Returns (result, num_col_muxed_used)。
    numColMuxed 可通过 pybind（configure_param）或 subprocess（main.cpp 可选 argv）传入。
    """
    ncm: Optional[int] = None
    if align_col_mux_to_weight:
        ncm = max(1, int(math.ceil(float(wl_weight) / float(max(1, int(cell_bit))))))
    elif num_col_muxed is not None and int(num_col_muxed) > 0:
        ncm = int(num_col_muxed)

    b = (backend or "auto").strip().lower()
    # subprocess 已支持 num_col_muxed；显式 --backend subprocess 时不应再强制走 pybind。
    try_pybind = (b in {"auto", "pybind"}) or (
        bool(align_col_mux_to_weight) and b != "subprocess"
    )

    if try_pybind:
        try:
            kwargs = dict(
                conv_linear_list=conv_linear_list,
                neurosim_so_dir=neurosim_so_dir,
                wl_weight=wl_weight,
                wl_activation=wl_activation,
                subarray=subarray,
                parallel_read=parallel_read,
                clk_freq_hz=clk_freq_hz,
                technode=technode,
                adc_clk_freq_hz=adc_clk_freq_hz,
                sar_column_power_w=sar_column_power_w,
                max_in_vectors=max_in_vectors,
                cell_bit=max(1, int(cell_bit)),
            )
            if ncm is not None:
                kwargs["num_col_muxed"] = int(ncm)
            r = run_neurosim_full_pybind(**kwargs)
            mux_used = int(ncm) if ncm is not None else 8
            if r.success:
                return r, mux_used
            if b == "pybind":
                return r, mux_used
        except Exception as e:
            if b == "pybind":
                return NeuroSimFullResult(
                    layer_latency_ns=[],
                    layer_energy_nj=[],
                    success=False,
                    error=str(e),
                ), ncm

    r = run_neurosim_full(
        conv_linear_list,
        neurosim_root=neurosim_root,
        wl_weight=wl_weight,
        wl_activation=wl_activation,
        subarray=subarray,
        parallel_read=parallel_read,
        timeout_s=timeout_s,
        clk_freq_hz=clk_freq_hz,
        technode=technode,
        adc_clk_freq_hz=adc_clk_freq_hz,
        sar_column_power_w=sar_column_power_w,
        num_col_muxed=ncm,
        max_in_vectors=max_in_vectors,
        clk_mode=clk_mode,
    )
    return r, (int(ncm) if ncm is not None else None)


def _row(
    wl_w: int,
    wl_a: int,
    res: NeuroSimFullResult,
    num_col_muxed_used: Optional[int],
) -> Dict[str, Any]:
    lats = list(res.layer_latency_ns or [])
    ens = list(res.layer_energy_nj or [])
    sum_lat = float(sum(lats)) if lats else 0.0
    sum_en = float(sum(ens)) if ens else 0.0
    return {
        "wl_weight": wl_w,
        "wl_activation": wl_a,
        "success": bool(res.success),
        "error": (res.error or "")[:500],
        "clock_period_ns": res.clock_period_ns,
        "sum_latency_ns": sum_lat,
        "sum_latency_us": sum_lat / 1e3,
        "sum_energy_nj": sum_en,
        "layer_latency_ns": ";".join(f"{x:.2f}" for x in lats),
        "ratio_vs_first": "",
        "num_col_muxed": num_col_muxed_used if num_col_muxed_used is not None else "",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="NeuroSim wl_weight × wl_activation latency sweep.",
        epilog=(
            "若 ModuleNotFoundError: pointkan_sim：请在 PointKAN-pytorch 根目录执行，或设置 "
            "PYTHONPATH=<该根目录>，或使用「python 本脚本绝对路径」方式运行。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["auto", "pybind", "subprocess"], default="auto")
    p.add_argument(
        "--neurosim-root",
        default="/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
        help="Inference_pytorch 根目录（subprocess 用）",
    )
    p.add_argument(
        "--neurosim-so-dir",
        default="",
        help="含 neurosim_core 的 NeuroSIM 目录；默认 <root>/NeuroSIM",
    )
    p.add_argument("--weights-bits", type=str, default="4,8", help="逗号分隔，如 4,6,8")
    p.add_argument("--activations-bits", type=str, default="4,8", help="逗号分隔")
    p.add_argument("--layer", choices=["conv", "fc"], default="conv")
    p.add_argument("--in-ch", type=int, default=64)
    p.add_argument("--out-ch", type=int, default=128)
    p.add_argument("--kernel", type=int, default=1)
    p.add_argument("--length", type=int, default=512, help="Conv1d 序列长度 W_in")
    p.add_argument("--in-f", type=int, default=256)
    p.add_argument("--out-f", type=int, default=128)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--subarray", type=int, default=128)
    p.add_argument("--parallel-read", type=int, default=128)
    p.add_argument("--max-in-vectors", type=int, default=0, help=">0 时截断 W_in/tokens 并线性缩放")
    p.add_argument("--neurosim-clk-freq", type=float, default=0.0, help="Hz；0=使用 NeuroSim/pybind 默认")
    p.add_argument("--neurosim-adc-clk-freq", type=float, default=0.0, help="Hz；0=默认")
    p.add_argument("--neurosim-technode", type=int, default=32)
    p.add_argument("--neurosim-timeout-s", type=int, default=0, help="仅 subprocess；0 表示无限")
    p.add_argument("--neurosim-clk-mode", choices=["capped", "critical"], default="capped")
    p.add_argument("--csv", type=str, default="", help="可选，写出结果 CSV")
    p.add_argument(
        "--align-col-mux-to-weight",
        action="store_true",
        help=(
            "仅 pybind：令 num_col_muxed = ceil(wl_weight/cell_bit)，与突触列数对齐 "
            "（并行模式下 subprocess/main 默认仍用 Param 的 numColMuxed=8，无法经 CLI 改）。"
        ),
    )
    p.add_argument(
        "--cell-bit",
        type=int,
        default=1,
        help="存储单元精度（与 NeuroSim cellBit 一致），用于对齐 ceil(wl_weight/cell_bit)。默认 1。",
    )
    p.add_argument(
        "--num-col-muxed",
        type=int,
        default=0,
        help=">0 时仅 pybind 强制使用该 num_col_muxed；与 --align-col-mux-to-weight 互斥。",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if int(getattr(args, "num_col_muxed", 0) or 0) > 0 and bool(getattr(args, "align_col_mux_to_weight", False)):
        p.error("--num-col-muxed 与 --align-col-mux-to-weight 不能同时使用")
    _mux_cli = int(getattr(args, "num_col_muxed", 0) or 0)
    if args.backend == "subprocess" and (
        bool(getattr(args, "align_col_mux_to_weight", False)) or _mux_cli > 0
    ):
        p.error(
            "修改 num_col_muxed（--align-col-mux-to-weight 或 --num-col-muxed）仅 "
            "neurosim_core pybind 支持；当前 Inference_pytorch/NeuroSIM/main 的 subprocess "
            "路径不会在 argv 里传 col mux。请使用 --backend pybind 或 --backend auto。"
        )

    wl_weights = _parse_int_list(args.weights_bits)
    wl_acts = _parse_int_list(args.activations_bits)
    for w in wl_weights + wl_acts:
        if w <= 0 or w > 32:
            p.error(f"invalid bit width: {w}")

    so_dir = args.neurosim_so_dir or str(Path(args.neurosim_root) / "NeuroSIM")
    conv_linear_list = _build_probe_layers(args)
    max_in_vectors: Optional[int] = (
        int(args.max_in_vectors) if int(args.max_in_vectors or 0) > 0 else None
    )
    clk = float(args.neurosim_clk_freq) if float(args.neurosim_clk_freq or 0) > 0 else None
    adc = float(args.neurosim_adc_clk_freq) if float(args.neurosim_adc_clk_freq or 0) > 0 else None
    timeout_s = int(args.neurosim_timeout_s) if int(args.neurosim_timeout_s or 0) > 0 else None

    combos = list(itertools.product(wl_weights, wl_acts))
    ncm_arg = int(args.num_col_muxed) if int(getattr(args, "num_col_muxed", 0) or 0) > 0 else None
    print(
        f"# workload={args.layer} layers={len(conv_linear_list)} "
        f"combos={len(combos)} backend={args.backend} root={args.neurosim_root} "
        f"align_col_mux={bool(args.align_col_mux_to_weight)} cell_bit={int(args.cell_bit)} "
        f"num_col_muxed_fixed={ncm_arg or ''}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for wl_w, wl_a in combos:
        print(f"# running wl_weight={wl_w} wl_activation={wl_a} ...", flush=True)
        res, mux_used = _run_once(
            conv_linear_list=conv_linear_list,
            wl_weight=wl_w,
            wl_activation=wl_a,
            backend=args.backend,
            neurosim_root=args.neurosim_root,
            neurosim_so_dir=so_dir,
            subarray=int(args.subarray),
            parallel_read=int(args.parallel_read),
            clk_freq_hz=clk,
            technode=int(args.neurosim_technode),
            adc_clk_freq_hz=adc,
            sar_column_power_w=None,
            max_in_vectors=max_in_vectors,
            timeout_s=timeout_s,
            clk_mode=str(args.neurosim_clk_mode or "capped"),
            cell_bit=int(args.cell_bit),
            align_col_mux_to_weight=bool(args.align_col_mux_to_weight),
            num_col_muxed=ncm_arg,
        )
        row = _row(wl_w, wl_a, res, mux_used)
        rows.append(row)
        clk_ns = row["clock_period_ns"]
        clk_s = f"{float(clk_ns):.4f}" if clk_ns is not None else ""
        err_tail = (row.get("error") or "")[:120]
        mux_s = row.get("num_col_muxed", "")
        mux_part = f" num_col_muxed={mux_s}" if mux_s != "" else ""
        print(
            f"  wl_w={wl_w} wl_a={wl_a} ok={row['success']}{mux_part} "
            f"sum_lat_ns={row['sum_latency_ns']:.2f} clock_period_ns={clk_s} "
            f"sum_en_nj={row['sum_energy_nj']:.4f}"
            + (f" err={err_tail!r}" if err_tail else ""),
            flush=True,
        )

    ref_sum = next(
        (
            float(r["sum_latency_ns"])
            for r in rows
            if r["success"] and float(r["sum_latency_ns"] or 0) > 0
        ),
        None,
    )
    if ref_sum and ref_sum > 0:
        for r in rows:
            if r["success"] and float(r["sum_latency_ns"] or 0) > 0:
                r["ratio_vs_first"] = float(r["sum_latency_ns"]) / ref_sum

    hdr = [
        "wl_weight",
        "wl_activation",
        "num_col_muxed",
        "success",
        "sum_latency_ns",
        "sum_latency_us",
        "sum_energy_nj",
        "clock_period_ns",
        "ratio_vs_first",
        "layer_latency_ns",
        "error",
    ]
    print("\n" + ",".join(hdr))
    for r in rows:
        print(
            ",".join(
                str(r.get(k, ""))
                for k in [
                    "wl_weight",
                    "wl_activation",
                    "num_col_muxed",
                    "success",
                    "sum_latency_ns",
                    "sum_latency_us",
                    "sum_energy_nj",
                    "clock_period_ns",
                    "ratio_vs_first",
                    "layer_latency_ns",
                    "error",
                ]
            )
        )

    if args.csv:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in hdr})
        print(f"# wrote {outp}", flush=True)

    return 0 if all(r.get("success") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
