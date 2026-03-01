#!/usr/bin/env bash
# PointKAN 在 ModelNet40 上推理 latency 与模块 breakdown
# 在 classification_ModelNet40 目录下执行: bash run_benchmark_latency.sh

set -e
cd "$(dirname "$0")"
echo "==> PointKAN latency benchmark (cwd: $(pwd))"

# 可选：指定 checkpoint（不指定则随机权重）
# CHECKPOINT="checkpoints/pointKAN-xxx/42/last_checkpoint.pth"
CHECKPOINT=""

mkdir -p opt_fps/figures

python benchmark_latency.py \
  --model pointKAN \
  --num_points 1024 \
  --warmup 20 \
  --repeat 100 \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  --out_json opt_fps/figures/latency_breakdown.json

echo "==> Done. Check opt_fps/figures/latency_breakdown.json"
