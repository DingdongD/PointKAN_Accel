#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实 checkpoint 与 ModelNet40 数据评估仿真器
================================================

与 PointKAN-elite/classification_ModelNet40/inference_and_kan_analysis.py 及
run_inference_kan.sh 中模型权重、ModelNet40 数据一致，先（可选）跑真实推理得到准确率，
再跑 PPA 仿真，所有日志与报告写入 sim_log 目录。

使用示例（在 PointKAN-pytorch 根目录）:
  python -m pointkan_sim.run_sim_with_real_eval
  python -m pointkan_sim.run_sim_with_real_eval --checkpoint PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42
  python -m pointkan_sim.run_sim_with_real_eval --run_real_inference --mnsim_full --output report.json

完整 shape 分析（Conv/Linear + BN/LayerNorm/ReLU/池化/S-Pool 等）:
  必须使用真实数据模式并开启 --show_shapes，会做一次真实前向并用 hook 收集各层维度后打印。
  方式一（推荐）:
    USE_REAL_DATA=1 SHOW_SHAPES=1 ./run_sim_with_real_eval.sh
  方式二（仅 shape，不跑 MNSIM/NeuroSim）:
    python -m pointkan_sim.run_sim_with_real_eval --use_real_data --show_shapes --batch_size 2
  日志中会先后出现「Conv/Linear 层 shape」与「非线性算子激活」两段，含 in/out shape、numel、字节数。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pointkan_sim.constants import (
    KEY_CYCLE,
    KEY_DURATION,
    KEY_END_CYCLE,
    KEY_ENERGY_NJ,
    KEY_IDX,
    KEY_LAYER_NAME,
    KEY_OPERATION,
    KEY_OPERATOR,
    KEY_STAGE,
    OPERATOR_SEQUENCE_HEADER,
)

# 工程根目录（PointKAN-pytorch）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = PROJECT_ROOT / "PointKAN-elite" / "classification_ModelNet40"

# 与 run_inference_kan.sh 默认一致
DEFAULT_CKPT = "checkpoints/pointKANElite-pointkan_elite_exp-42"
DEFAULT_NUM_POINTS = 1024
DEFAULT_NUM_CLASSES = 40


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（网络、加速器、checkpoint、真实数据/仿真模式、BN/量化开关、导出选项等）。"""
    parser = argparse.ArgumentParser(
        description="使用真实 checkpoint 与 ModelNet40 配置运行 PPA 仿真，可选先跑真实推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"检查点目录或 .pth 路径；默认相对 classification_ModelNet40 的 {DEFAULT_CKPT}",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="ModelNet40 数据根目录（预留；当前 inference 脚本使用 data.py 内路径）",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=f"点云点数，需与训练/推理一致，默认 {DEFAULT_NUM_POINTS}",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=f"分类数，默认 {DEFAULT_NUM_CLASSES}",
    )
    parser.add_argument(
        "--run_real_inference",
        action="store_true",
        help="先运行 inference_and_kan_analysis 在 ModelNet40 上推理，将日志写入 sim_log，再跑仿真",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="baseline",
        help="加速器配置名",
    )
    parser.add_argument(
        "--mnsim_full",
        action="store_true",
        help="对 conv1d/linear 使用 MNSIM 全 NetStruct",
    )
    parser.add_argument(
        "--neurosim_full",
        action="store_true",
        help="对 conv1d/linear 使用 NeuroSim 全网络仿真（DNN_NeuroSim main）；与 --mnsim_full 二选一时优先 NeuroSim",
    )
    parser.add_argument(
        "--cim_backend",
        type=str,
        default=None,
        choices=["mnsim", "neurosim", "fallback"],
        help="Conv/Linear 单层估算后端：mnsim（默认）、neurosim（DNN_NeuroSim）、fallback（解析回退）；无全网络时生效",
    )
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default=None,
        help="NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    )
    parser.add_argument(
        "--neurosim_clk_freq",
        type=float,
        default=None,
        help="NeuroSim 全网络仿真时工作频率上限(Hz)，如 100000000=100MHz；不设则使用 NeuroSim 关键路径频率",
    )
    parser.add_argument(
        "--neurosim_technode",
        type=int,
        default=None,
        help="NeuroSim 工艺节点(nm)，如 32；仅支持 130/90/65/45/32/22/14/10/7，28 会自动改为 32",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出文件名（写入 sim_log 目录），如 report.json",
    )
    parser.add_argument(
        "--sim_log_dir",
        type=str,
        default="sim_log",
        help="仿真与推理日志统一输出目录",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="真实推理时禁用可视化（与 inference_and_kan_analysis 一致）",
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="用 ModelNet40 真实样本与 checkpoint 权重：加载模型与数据集，从真实前向收集 conv/linear 维度驱动 MNSIM，按批调度仿真并汇总准确率与 PPA",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="--use_real_data 时测试集 batch size",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="--use_real_data 时最多跑多少 batch（默认全量测试集）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="--use_real_data 时运行设备（真实前向含 FPS 等需 GPU，无 GPU 时可改用 --use_real_data 不加或先跑仿真）",
    )
    parser.add_argument(
        "--show_shapes",
        action="store_true",
        help="--use_real_data 时打印 LayerDef 与真实前向的每层 conv/linear 的输入激活与权重 shape（及 indata/rdata 字节）",
    )
    parser.add_argument(
        "--sim_results_dir",
        type=str,
        default="sim_results",
        help="--use_real_data 时 PPA 报告 JSON 与 breakdown 图表输出目录，默认 sim_results",
    )
    parser.add_argument(
        "--bn_en",
        type=int,
        default=1,
        choices=[0, 1],
        help="BN 融合开关：1=启用融合（BN 不单独计入），0=关闭融合（保留 BN 事件）",
    )
    parser.add_argument(
        "--qd_en",
        type=int,
        default=0,
        choices=[0, 1],
        help="量化/反量化开关：1=插入 FP32↔INT8 量化事件，0=不插入",
    )
    parser.add_argument(
        "--export_memory_transactions",
        action="store_true",
        help="导出内存交互事务序列（memory_transactions.txt）",
    )
    return parser.parse_args()


def _ensure_sim_log(sim_log_dir: str) -> Path:
    p = Path(sim_log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_real_inference(
    checkpoint_path: str,
    classification_dir: Path,
    sim_log_dir: Path,
    num_points: int,
    no_vis: bool,
) -> int:
    """
    在 classification_ModelNet40 目录下执行 inference_and_kan_analysis.py，
    将 stdout/stderr 写入 sim_log/real_inference_log.txt，返回进程退出码。
    """
    script = classification_dir / "inference_and_kan_analysis.py"
    if not script.is_file():
        print(f"[run_sim_with_real_eval] 未找到 {script}，跳过真实推理。")
        return -1

    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        checkpoint_path,
        "--num_points",
        str(num_points),
        "--no_vis",
    ]

    log_file = sim_log_dir / "real_inference_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"CWD: {classification_dir}\n\n")
    with open(log_file, "a", encoding="utf-8") as f:
        ret = subprocess.run(
            cmd,
            cwd=str(classification_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"[run_sim_with_real_eval] 真实推理日志已写入: {log_file}")
    return ret.returncode


def resolve_checkpoint_path(checkpoint_arg: Optional[str]) -> str:
    """解析 checkpoint 路径：若为相对路径则优先相对 classification_ModelNet40。"""
    if not checkpoint_arg:
        path = CLASSIFICATION_DIR / DEFAULT_CKPT
    else:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            # 先尝试相对当前目录，再尝试相对 classification 目录
            if not path.exists():
                alt = CLASSIFICATION_DIR / checkpoint_arg
                if alt.exists() or (alt / "best_checkpoint.pth").exists():
                    path = alt
            path = path.resolve()
    return str(path)


def main() -> int:
    args = _parse_args()

    sim_log_path = _ensure_sim_log(args.sim_log_dir)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from datetime import datetime
    from pointkan_sim.utils.logger import setup_logger, LogLevel
    log_path = sim_log_path / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(level=LogLevel.INFO, log_file=str(log_path))

    from pointkan_sim.main import (
        load_network,
        get_accelerator_config,
        compute_activation_shapes,
        _write_report_to_file,
        log_info,
        log_section,
        log_separator,
    )
    from pointkan_sim.memory.transactions import extract_memory_transactions, summarize_memory_transactions

    # -------------------------------------------------------------------------
    # 模式：用 ModelNet40 真实样本 + 真实权重 调度仿真，MNSIM 由真实前向得到的层维度驱动
    # -------------------------------------------------------------------------
    if args.use_real_data:
        from pointkan_sim.integration.real_data_eval import (
            load_model_and_dataset,
            collect_conv_linear_from_forward,
            collect_nonlinear_activations_from_forward,
            real_data_overrides_from_conv_linear_list,
            log_conv_linear_shapes,
            log_nonlinear_activation_shapes,
            run_real_inference_and_sim_over_dataset,
            RealEvalResult,
        )
        from pointkan_sim.integration.mnsim_netstruct import run_mnsim_full
        import math

        log_section("真实数据评估：ModelNet40 样本 + checkpoint 权重 + MNSIM 对应层 PPA")
        log_info(f"Checkpoint: {checkpoint_path}")
        log_info(f"num_points={args.num_points}, num_classes={args.num_classes}, batch_size={args.batch_size}")

        try:
            net, test_loader, criterion = load_model_and_dataset(
                checkpoint_path,
                num_points=args.num_points,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                workers=0,
                device=args.device,
            )
        except Exception as e:
            log_info(f"加载模型/数据集失败: {e}")
            return 1

        # 层定义（与后续 MNSIM、仿真共用）
        hw_config = get_accelerator_config(args.accelerator)
        # 用户可控开关：BN 融合 / 量化反量化 / CIM 后端
        hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
        hw_config.enable_w8a8_quant_dequant = bool(args.qd_en)
        if getattr(args, "cim_backend", None):
            hw_config.cim_backend = args.cim_backend
        if getattr(args, "neurosim_root", None):
            hw_config.neurosim_root = args.neurosim_root
        layers = load_network("pointkan-elite", args.num_points, args.num_classes)

        # 打印加速器与层间 buffer/量化配置（与 main.run_network_simulation 一致，便于对照）
        at = getattr(hw_config, "accelerator_type", None)
        at_name = getattr(at, "name", str(at)) if at is not None else "N/A"
        log_info(f"加速器: {hw_config.accelerator_name} ({at_name})")
        log_info(f"CIM 并行度: {getattr(hw_config, 'cim_parallelism', 'N/A')}")
        log_info(f"Bit-serial 周期: {getattr(hw_config, 'bit_serial_cycles', 'N/A')}")
        buf = getattr(hw_config, "inter_layer_buffer", None)
        if buf is not None:
            log_info(f"层间 buffer: size={buf.size_bytes} B, bandwidth={buf.bandwidth_bytes_per_cycle} B/cycle, spill_to_dram={getattr(buf, 'spill_to_dram', False)}")
        else:
            log_info("层间 buffer: 未配置（inter_layer_buffer 为空）")
        qd = getattr(hw_config, "enable_w8a8_quant_dequant", False)
        bn_fuse = getattr(hw_config, "bn_fuse_with_conv_linear", True)
        log_info(f"层间量化/反量化: {'启用 (FP32↔INT8 计入 cycle/功耗)' if qd else '未启用 (默认，假定 W8A8 已就绪)'}")
        log_info(f"BN 融合: {'启用 (BN 不单独计入)' if bn_fuse else '关闭 (保留 BN 事件)'}")
        log_info("")

        # 先做一次真实前向，收集 conv/linear 顺序与真实 indata/rdata 字节数，用于 MNSIM 按层矩阵乘适配
        conv_linear_list = collect_conv_linear_from_forward(
            net, args.device, args.num_points, batch_size=min(2, args.batch_size)
        )
        real_data_overrides = real_data_overrides_from_conv_linear_list(conv_linear_list)

        if args.show_shapes:
            log_conv_linear_shapes(
                layers,
                conv_linear_list,
                real_data_overrides=real_data_overrides,
                log_info_fn=log_info,
            )
            nonlinear_list = collect_nonlinear_activations_from_forward(
                net, args.device, args.num_points, batch_size=min(2, args.batch_size)
            )
            log_nonlinear_activation_shapes(nonlinear_list, log_info_fn=log_info)

        # MNSIM / NeuroSim 全网络：二选一注入 mnsim_context（dispatcher 用同一 context 键）
        activation_shapes = compute_activation_shapes(layers, args.num_points, input_channels=3)

        mnsim_context = None
        n_mnsim = 0
        netstruct = None
        conv_linear_order = None
        mnsim_result = None
        used_neurosim_for_ppa = False
        neurosim_result = None
        neurosim_root = getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")

        if args.neurosim_full:
            from pointkan_sim.integration.neurosim_full import run_neurosim_full
            neurosim_result = run_neurosim_full(
                conv_linear_list,
                neurosim_root=neurosim_root,
                clk_freq_hz=getattr(args, "neurosim_clk_freq", None),
                technode=getattr(args, "neurosim_technode", None),
            )
            if neurosim_result.success:
                used_neurosim_for_ppa = True
                if getattr(args, "neurosim_clk_freq", None) is not None or getattr(args, "neurosim_technode", None) is not None:
                    log_info(f"NeuroSim 频率/工艺: clk_freq_hz={getattr(args, 'neurosim_clk_freq', None)}, technode={getattr(args, 'neurosim_technode', None)}")
                mnsim_context = {
                    "mnsim_latency_ns": neurosim_result.layer_latency_ns,
                    "mnsim_energy_nj": neurosim_result.layer_energy_nj,
                    "mnsim_index": 0,
                }
                n_mnsim = len(neurosim_result.layer_latency_ns)
                log_info("已用 NeuroSim 全网络仿真（DNN_NeuroSim main）得到逐层 latency/energy")
                # 使用 NeuroSim 实际时钟周期换算 cycle，不再用 1 GHz 默认
                if getattr(neurosim_result, "clock_period_ns", None) is not None and neurosim_result.clock_period_ns > 0:
                    hw_config.clock_period = neurosim_result.clock_period_ns
                    freq_mhz = 1000.0 / neurosim_result.clock_period_ns
                    log_info(f"使用 NeuroSim 实际时钟: clock_period_ns={neurosim_result.clock_period_ns:.4f} ns (频率={freq_mhz:.2f} MHz)，用于 latency->cycle 与端到端延迟换算")
                else:
                    log_info("未解析到 NeuroSim 时钟周期，使用 hw_config 默认 clock_period (1 ns = 1 GHz) 换算 cycle")
                log_info("以下层使用 NeuroSim 接口计算 PPA：")
                clock_ns = hw_config.clock_period
                from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                for idx, (lat_ns, en_nj) in enumerate(zip(neurosim_result.layer_latency_ns, neurosim_result.layer_energy_nj)):
                    name = conv_linear_list[idx].name if idx < len(conv_linear_list) else f"layer_{idx}"
                    cyc = max(1, int(math.ceil(lat_ns / clock_ns)))
                    act_s, w_s = format_activation_weight_shapes(conv_linear_list[idx]) if idx < len(conv_linear_list) else ("?", "?")
                    log_info(f"  NeuroSim [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                log_info(f"  （共 {n_mnsim} 层）")
            else:
                log_info(f"NeuroSim 全网络未启用: {neurosim_result.error}")
                if neurosim_result.error and len(neurosim_result.error) > 200:
                    for line in neurosim_result.error.split("\n")[:15]:
                        if line.strip():
                            log_info(f"  NeuroSim stderr: {line.strip()}")
                log_info("  提示: 若为主进程崩溃(returncode≠0)，可检查 DNN_NeuroSim 是否支持当前层数/维度；层数较多时可尝试减少 batch 或先单独运行 NeuroSIM/main 排查。")

        if mnsim_context is None and args.mnsim_full:
            from pointkan_sim.integration.mnsim_netstruct import build_netstruct_from_layers, run_mnsim_full
            netstruct, conv_linear_order = build_netstruct_from_layers(
                layers, real_data_overrides=real_data_overrides
            )
            if netstruct:
                mnsim_result = run_mnsim_full(netstruct, conv_linear_order=conv_linear_order)
                if mnsim_result.success:
                    mnsim_context = {
                        "mnsim_latency_ns": mnsim_result.layer_latency_ns,
                        "mnsim_energy_nj": mnsim_result.layer_energy_nj,
                        "mnsim_index": 0,
                    }
                    n_mnsim = len(mnsim_result.layer_latency_ns)
                    log_info("已用与 checkpoint 一致的层结构 + 真实按层 indata/rdata 驱动 MNSIM 全 NetStruct（Tile/NoC）")
                    clock_ns = hw_config.clock_period
                    from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                    for idx, (lat_ns, en_nj) in enumerate(zip(mnsim_result.layer_latency_ns, mnsim_result.layer_energy_nj)):
                        name = conv_linear_order[idx][0] if idx < len(conv_linear_order) else f"layer_{idx}"
                        cyc = max(1, int(math.ceil(lat_ns / clock_ns)))
                        act_s, w_s = format_activation_weight_shapes(conv_linear_list[idx]) if idx < len(conv_linear_list) else ("?", "?")
                        log_info(f"  MNSIM [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                else:
                    log_info(f"MNSIM 未启用: {mnsim_result.error}")

        if mnsim_context is not None:
            n_mnsim = len(mnsim_context["mnsim_latency_ns"])
        log_info(f"真实前向共 {len(conv_linear_list)} 个 conv/linear；全网络 PPA 使用 {n_mnsim} 层")
        if len(conv_linear_list) != n_mnsim:
            log_info(f"  注意：真实前向层数({len(conv_linear_list)})与 LayerDef 层数({n_mnsim})不一致；overrides 按索引取前 {n_mnsim} 条对应到 MNSIM。建议使用 --show_shapes 核对。")

        # 按 ModelNet40 批次调度：每批一次真实前向（准确率）+ 一次 PPA 仿真（累加）
        eval_result = run_real_inference_and_sim_over_dataset(
            net,
            test_loader,
            criterion,
            args.device,
            layers,
            hw_config,
            mnsim_context,
            enable_layer_buffer=True,
            activation_shapes=activation_shapes,
            max_batches=args.max_batches,
        )

        if not eval_result.success:
            log_info(f"真实数据评估失败: {eval_result.error}")
            return 1

        log_separator()
        if used_neurosim_for_ppa:
            log_info("真实数据评估结果 (ModelNet40 测试集 + 对应层 NeuroSim PPA)")
        elif mnsim_context and args.mnsim_full:
            log_info("真实数据评估结果 (ModelNet40 测试集 + 对应层 MNSIM PPA)")
        else:
            log_info("真实数据评估结果 (ModelNet40 测试集)")
        log_separator()
        log_info(f"  样本数:        {eval_result.num_samples}")
        log_info(f"  准确率:        {eval_result.accuracy:.2f}%")
        log_info(f"  平衡准确率:    {eval_result.balanced_accuracy:.2f}%")
        log_info(f"  Test Loss:     {eval_result.test_loss:.4f}")
        clock_ns = hw_config.clock_period
        # 端到端延迟：各模块按自身 clock_period_ns 换算为 ns 后累加（NeuroSim 与 1 GHz 数字模块混合正确）
        latency_per_sample_ns = getattr(eval_result, "latency_per_sample_ns", None) or (eval_result.cycles_per_sample * clock_ns)
        log_info(f"  总周期:        {eval_result.total_cycles:,} cycles")
        log_info(f"  总能耗:        {eval_result.total_energy_nj:.2f} nJ")
        log_info(f"  端到端延迟:    {latency_per_sample_ns/1e6:.4f} ms (每样本, 各模块按自身时钟换算 ns 后累加)")
        log_info(f"  每样本周期:    {eval_result.cycles_per_sample:.1f} cycles")
        log_info(f"  每样本能耗:    {eval_result.energy_per_sample_nj:.2f} nJ")
        log_separator()

        # 单次仿真获取全算子 breakdown（FPS/KNN/BN/Conv/Linear 等），用于饼图与报告
        from pointkan_sim.main import simulate_network
        mnsim_ctx = dict(mnsim_context) if mnsim_context else None
        if mnsim_ctx is not None:
            mnsim_ctx["mnsim_index"] = 0
        sim_once = simulate_network(
            layers,
            hw_config,
            show_layers=False,
            mnsim_context=mnsim_ctx,
            activation_shapes=activation_shapes,
            enable_layer_buffer=True,
            show_shapes=False,
        )
        full_operator_breakdown = sim_once.get("operator_breakdown") or {}
        operator_sequence = sim_once.get("operator_sequence") or []

        # 控制台打印：端到端延迟与能耗的具体构成（按算子类型）；延迟为各算子 duration×自身 clock_period_ns 累加
        if full_operator_breakdown:
            total_cy = sum(v.get("cycles", 0) for v in full_operator_breakdown.values())
            total_en = sum(v.get("energy_nj", 0) for v in full_operator_breakdown.values())
            latency_total_ns = sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values())
            if latency_total_ns <= 0:
                latency_total_ns = total_cy * clock_ns
            log_separator()
            log_info("端到端延迟与能耗构成 (每样本，按算子类型；延迟=各模块 cycles×自身时钟周期 累加)")
            log_separator()
            log_info(f"  {'算子':<20} {'cycles':>12} {'cycles%':>8} {'latency(ns)':>14} {'latency%':>8} {'energy(nJ)':>12} {'energy%':>8}")
            log_info("  " + "-" * 86)
            for op_name, v in sorted(full_operator_breakdown.items(), key=lambda x: -x[1].get("latency_ns", x[1].get("cycles", 0))):
                cy = v.get("cycles", 0)
                en = v.get("energy_nj", 0)
                lat_ns = v.get("latency_ns", cy * clock_ns if clock_ns > 0 else 0)
                cy_pct = (cy / total_cy * 100) if total_cy > 0 else 0
                lat_pct = (lat_ns / latency_total_ns * 100) if latency_total_ns > 0 else 0
                en_pct = (en / total_en * 100) if total_en > 0 else 0
                log_info(f"  {op_name:<20} {cy:>12,.0f} {cy_pct:>7.1f}% {lat_ns:>14,.1f} {lat_pct:>7.1f}% {en:>12,.2f} {en_pct:>7.1f}%")
            log_info("  " + "-" * 86)
            log_info(f"  {'合计':<20} {total_cy:>12,.0f} {100.0:>7.1f}% {latency_total_ns:>14,.1f} {100.0:>7.1f}% {total_en:>12,.2f} {100.0:>7.1f}%")
            log_info(f"  端到端延迟合计: {latency_total_ns/1e6:.4f} ms  |  能耗合计: {total_en:.2f} nJ")
            log_separator()

        # 端到端 PPA 报告与算子级 breakdown 可视化：report.json + latency/energy 饼图（由 write_ppa_report_and_plots 统一写入 sim_results）
        from pointkan_sim.integration.ppa_report import build_ppa_report, write_ppa_report_and_plots
        mnsim_lat = mnsim_context["mnsim_latency_ns"] if mnsim_context else None
        mnsim_en = mnsim_context["mnsim_energy_nj"] if mnsim_context else None
        ppa_report = build_ppa_report(
            eval_result,
            clock_ns,
            netstruct=netstruct,
            conv_linear_order=conv_linear_order,
            mnsim_latency_ns=mnsim_lat,
            mnsim_energy_nj=mnsim_en,
            full_operator_breakdown=full_operator_breakdown,
            checkpoint=checkpoint_path,
            mnsim_full=args.mnsim_full,
            rram_tile_stats=getattr(mnsim_result, "rram_tile_stats", None) if mnsim_result else None,
            operator_sequence=operator_sequence,
            latency_per_sample_ns=getattr(eval_result, "latency_per_sample_ns", None),
        )
        sim_results_dir = Path(args.sim_results_dir)
        sim_results_dir.mkdir(parents=True, exist_ok=True)
        # NeuroSim 模式：将参与计算的层名及 latency/energy/激活与权重 shape 写入 CSV
        if used_neurosim_for_ppa and mnsim_context and conv_linear_list:
            import csv
            from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
            lat_ns_list = mnsim_context["mnsim_latency_ns"]
            en_nj_list = mnsim_context["mnsim_energy_nj"]
            neurosim_csv_path = sim_results_dir / "neurosim_layer_ppa.csv"
            with open(neurosim_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["layer_name", "layer_type", "latency_ns", "energy_nj", "input_activation_shape", "weight_shape"])
                for idx in range(min(len(lat_ns_list), len(en_nj_list), len(conv_linear_list))):
                    info = conv_linear_list[idx]
                    act_s, w_s = format_activation_weight_shapes(info)
                    w.writerow([
                        info.name,
                        info.layer_type,
                        f"{lat_ns_list[idx]:.4f}",
                        f"{en_nj_list[idx]:.4f}",
                        act_s,
                        w_s,
                    ])
            log_info(f"NeuroSim 逐层 PPA 与 shape 已写入: {neurosim_csv_path}（共 {min(len(lat_ns_list), len(conv_linear_list))} 层）")
        report_filename = os.path.basename(args.output) if args.output else "report.json"
        write_ppa_report_and_plots(
            ppa_report,
            sim_results_dir,
            report_filename=report_filename,
            log_info_fn=log_info,
        )
        # 按执行顺序输出完整硬件算子序列到 operator_sequence.txt（列与 constants 键名一致）
        if operator_sequence:
            seq_path = sim_results_dir / "operator_sequence.txt"
            with open(seq_path, "w", encoding="utf-8") as f:
                f.write("# 硬件算子执行顺序（按 cycle 排序）\n")
                f.write(OPERATOR_SEQUENCE_HEADER)
                for item in operator_sequence:
                    f.write(
                        "{idx}\t{cycle}\t{end_cycle}\t{duration}\t{operator}\t{stage}\t{operation}\t{layer_name}\t{energy_nj}\n".format(
                            idx=item.get(KEY_IDX, ""),
                            cycle=item.get(KEY_CYCLE, ""),
                            end_cycle=item.get(KEY_END_CYCLE, ""),
                            duration=item.get(KEY_DURATION, ""),
                            operator=item.get(KEY_OPERATOR, ""),
                            stage=item.get(KEY_STAGE, ""),
                            operation=item.get(KEY_OPERATION, ""),
                            layer_name=item.get(KEY_LAYER_NAME, ""),
                            energy_nj=item.get(KEY_ENERGY_NJ, ""),
                        )
                    )
            log_info(f"完整硬件算子序列已写入: {seq_path}（共 {len(operator_sequence)} 条）")

        # 可选：导出内存交互事务（统一接口 src/dst/kind），并写聚合统计
        if args.export_memory_transactions and operator_sequence:
            mem_txn = extract_memory_transactions(operator_sequence)
            mem_path = sim_results_dir / "memory_transactions.txt"
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write("# 内存交互事务序列（统一接口，按 cycle 排序）\n")
                f.write("# idx\tcycle\tend_cycle\tduration\tkind\tsrc\tdst\toperator\toperation\tcircuit_module\tlayer_name\tenergy_nj\n")
                for tx in mem_txn:
                    row = tx.to_dict()
                    f.write(
                        "{idx}\t{cycle}\t{end_cycle}\t{duration}\t{kind}\t{src}\t{dst}\t{operator}\t{operation}\t{circuit_module}\t{layer_name}\t{energy_nj}\n".format(
                            **row
                        )
                    )
            route_counter = summarize_memory_transactions(mem_txn)
            summary_path = sim_results_dir / "memory_transactions_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("# 内存事务聚合统计（src->dst, kind）\n")
                f.write("# src\tdst\tkind\tcount\n")
                for (src, dst, kind), cnt in sorted(route_counter.items(), key=lambda x: (-x[1], x[0])):
                    f.write(f"{src}\t{dst}\t{kind}\t{cnt}\n")
            log_info(f"内存交互事务已写入: {mem_path}（共 {len(mem_txn)} 条）")
            log_info(f"内存交互聚合已写入: {summary_path}")
        return 0

    # -------------------------------------------------------------------------
    # 模式：仅跑仿真（可选先跑 inference 脚本）
    # -------------------------------------------------------------------------
    if args.run_real_inference:
        print("[run_sim_with_real_eval] 运行真实推理 (ModelNet40)...")
        run_real_inference(
            checkpoint_path=checkpoint_path,
            classification_dir=CLASSIFICATION_DIR,
            sim_log_dir=sim_log_path,
            num_points=args.num_points,
            no_vis=True,
        )
        print()

    from pointkan_sim.main import run_network_simulation

    class SimArgs:
        network = "pointkan-elite"
        accelerator = args.accelerator
        input_points = args.num_points
        num_classes = args.num_classes
        show_layers = True
        mnsim_full = args.mnsim_full
        no_mnsim_full = False
        no_layer_buffer = False
        show_shapes = False
        show_events = False
        enable_w8a8_quant_dequant = bool(args.qd_en)
        bn_en = int(args.bn_en)
        cim_backend = getattr(args, "cim_backend", None)
        neurosim_root = getattr(args, "neurosim_root", None)
        output_timeline_csv = None
        timeline_stride = 500

    sim_args = SimArgs()
    log_section("PPA 仿真（与 checkpoint/ModelNet40 配置一致）")
    log_info(f"Checkpoint: {checkpoint_path}")
    log_info(f"num_points={args.num_points}, num_classes={args.num_classes}")

    results = run_network_simulation(sim_args)
    hw_config = get_accelerator_config(sim_args.accelerator)
    hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
    hw_config.enable_w8a8_quant_dequant = bool(args.qd_en)
    if getattr(sim_args, "cim_backend", None):
        hw_config.cim_backend = sim_args.cim_backend
    if getattr(sim_args, "neurosim_root", None):
        hw_config.neurosim_root = sim_args.neurosim_root

    if args.output:
        out_path = sim_log_path / os.path.basename(args.output)
        _write_report_to_file(
            results,
            str(out_path),
            network_name=sim_args.network,
            accelerator_name=hw_config.accelerator_name,
        )
        log_info(f"报告已写入: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
