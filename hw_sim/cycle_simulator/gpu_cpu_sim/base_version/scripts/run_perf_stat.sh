#!/usr/bin/env bash
set -euo pipefail

# 获取工程根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 必须由外部传入待执行命令
CMD="${CMD:?Please set CMD='python infer.py ...'}"

# perf 输出目录
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/perf}"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# 说明：
# -x,           : CSV 风格输出，以逗号分隔
# -ddd          : 更详细的 perf 统计
# cycles        : 时钟周期数
# instructions  : 指令数
# branches      : 分支指令数
# branch-misses : 分支预测失败
# cache-*       : cache 行为
# task-clock    : 任务运行时间
# power/energy-* : 若平台支持，可读 RAPL 能量
perf stat -x, -ddd \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,task-clock,power/energy-pkg/,power/energy-ram/ \
  -o "$OUT_DIR/perf_${TS}.csv" \
  bash -lc "$CMD"

echo "perf output: $OUT_DIR/perf_${TS}.csv"