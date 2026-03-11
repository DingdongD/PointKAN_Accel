#!/usr/bin/env bash
set -euo pipefail

# 输出目录
OUT_DIR="${OUT_DIR:-./outputs/dmon}"

# GPU 编号
GPU_INDEX="${GPU_INDEX:-0}"

# dmon 采样间隔（秒）
INTERVAL="${INTERVAL:-1}"

# 总采样次数
COUNT="${COUNT:-60}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# nvidia-smi dmon:
# -i 指定 GPU
# -s pucvmet:
#   p = power
#   u = utilization
#   c = clocks
#   v = violation
#   m = memory
#   e = ecc/error
#   t = temperature
# -d 采样间隔
# -c 采样次数
nvidia-smi dmon \
  -i "$GPU_INDEX" \
  -s pucvmet \
  -d "$INTERVAL" \
  -c "$COUNT" \
  > "$OUT_DIR/dmon_${TS}.txt"

echo "dmon log: $OUT_DIR/dmon_${TS}.txt"