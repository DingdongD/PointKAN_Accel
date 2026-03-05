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
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 工程根目录（PointKAN-pytorch）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = PROJECT_ROOT / "PointKAN-elite" / "classification_ModelNet40"

# 与 run_inference_kan.sh 默认一致
DEFAULT_CKPT = "checkpoints/pointKANElite-pointkan_elite_exp-42"
DEFAULT_NUM_POINTS = 1024
DEFAULT_NUM_CLASSES = 40


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用真实 checkpoint 与 ModelNet40 配置运行 PPA 仿真，可选先跑真实推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42",
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

    # 构建与 main 一致的仿真参数，并统一输出到 sim_log
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    # 导入 main 模块并构造 args，然后调用 run_network_simulation
    from pointkan_sim.main import (
        run_network_simulation,
        get_accelerator_config,
        _write_report_to_file,
        log_info,
        log_section,
    )
    from pointkan_sim.utils.logger import setup_logger, LogLevel

    # 初始化日志到 sim_log
    from datetime import datetime
    log_path = sim_log_path / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(level=LogLevel.INFO, log_file=str(log_path))

    # 构造与 run_network_simulation 兼容的 args 命名空间
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
        enable_w8a8_quant_dequant = False
        output_timeline_csv = None
        timeline_stride = 500

    sim_args = SimArgs()
    log_section("PPA 仿真（与 checkpoint/ModelNet40 配置一致）")
    log_info(f"Checkpoint: {checkpoint_path}")
    log_info(f"num_points={args.num_points}, num_classes={args.num_classes}")

    results = run_network_simulation(sim_args)
    hw_config = get_accelerator_config(sim_args.accelerator)

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
