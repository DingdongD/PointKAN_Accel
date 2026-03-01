#!/usr/bin/env bash
# FPS 校准与 radii 曲线评估（ModelNet40）
# 在 classification_ModelNet40 目录下执行: bash opt_fps/run_eval_fps_radii.sh

set -e
cd "$(dirname "$0")/.."
echo "==> FPS radii calibration (cwd: $(pwd))"

# 默认：train 集、每类 50 样本、1024 步 FPS、输出到 opt_fps/figures
python opt_fps/eval_fps_radii.py \
  --partition train \
  --num_points 1024 \
  --samples_per_class 50 \
  --fps_steps 1024 \
  --batch_size 32 \
  --out_dir opt_fps/figures

echo "==> Done. Check opt_fps/figures/fps_radii.png (and .pdf)"
