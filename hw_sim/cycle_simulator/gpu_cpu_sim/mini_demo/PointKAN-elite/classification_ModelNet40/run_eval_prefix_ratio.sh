#!/bin/bash
# 对某一组 prefix_ratio 的模型（如 KAN 某次训练）做测试集评测，并输出 N 个样本的 GT vs 预测 PNG
# 用法:
#   ./run_eval_prefix_ratio.sh ./runs/fps_radii_kan_K512_pr0.10        # 默认 20 个样本
#   ./run_eval_prefix_ratio.sh ./runs/fps_radii_kan_K512_pr0.10 20     # 指定 20 个样本
#   ./run_eval_prefix_ratio.sh ./runs/fps_radii_mlp_K512_pr0.15 10     # MLP 某组，10 个样本

set -e
cd "$(dirname "$0")"

SAVE_DIR="${1:?用法: $0 <save_dir> [num_vis]}"
NUM_VIS="${2:-20}"

echo "========== 评测指定 run（测试集 MSE + 样本对比图） =========="
echo "  SAVE_DIR=$SAVE_DIR"
echo "  NUM_VIS=$NUM_VIS"
echo "  (radii 从 checkpoint 或 --radii_dir 读取，prefix_sweep 跑过的 run 已含 radii_dir)"
echo "============================================================"

python eval_fps_radii.py \
  --save_dir "$SAVE_DIR" \
  --num_vis "$NUM_VIS"

echo "Done. 可视化: ${SAVE_DIR}/vis/ (共 ${NUM_VIS} 张 sample_*_idx*.png)"
