#!/usr/bin/env bash
set -euo pipefail

# 获取工程根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 用户可通过环境变量覆盖
CMD="${CMD:-python infer_cpu.py}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
ENABLE_PERF="${ENABLE_PERF:-0}"

# 调用 Python 主分析器，模式为 cpu
python3 "$ROOT_DIR/profiler.py" \
  --cmd "$CMD" \
  --mode cpu \
  --out-dir "$OUT_DIR" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT"

# 如启用 perf，则进一步采样底层 CPU 计数器
if [[ "$ENABLE_PERF" == "1" ]]; then
  CMD="$CMD" bash "$ROOT_DIR/scripts/run_perf_stat.sh"
fi