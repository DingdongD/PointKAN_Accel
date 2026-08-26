#!/bin/bash
# 一键：在训练根目录下自动发现「全部」已完成的 FPS radii 模型（best.pt + radii_test），
# 生成多数据集 GT vs 预测对比总图（与 run_fps_radii_one_click_all 目录布局一致）。
#
# 用法（在 classification_ModelNet40 下）:
#   chmod +x run_visualize_fps_radii_curves.sh
#   ./run_visualize_fps_radii_curves.sh
#
# 环境变量:
#   BASE         训练输出根目录，默认 ./runs/fps_radii_oneclick_all
#   PR           与目录名 pr0.xx 一致，默认 0.10
#   K            FPS 步数，默认 512
#   MODEL_TYPE   kan | mlp，默认 kan
#   OUT          可视化输出目录
#   NUM_VIS      对比图每行样本数，默认 4
#   DEVICE       默认 cuda:0
#
set -e
cd "$(dirname "$0")"

BASE="${BASE:-./runs/fps_radii_oneclick_all}"
PR="${PR:-0.10}"
K="${K:-512}"
MODEL_TYPE="${MODEL_TYPE:-kan}"
OUT="${OUT:-${BASE}/curve_vis_pr${PR}}"
NUM_VIS="${NUM_VIS:-4}"
DEVICE="${DEVICE:-cuda:0}"

exec python visualize_fps_radii_curves.py --compare \
  --auto_from "$BASE" \
  --auto_k "$K" \
  --auto_model_type "$MODEL_TYPE" \
  --auto_pr "$PR" \
  --prefix_ratio "$PR" \
  --out_dir "$OUT" \
  --num_vis "$NUM_VIS" \
  --device "$DEVICE"
