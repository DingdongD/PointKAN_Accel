#!/usr/bin/env bash
set -euo pipefail

# 获取工程根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 用户可通过环境变量覆盖这些参数
CMD="${CMD:-python infer.py}"            # 推理命令
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs}"  # 输出目录
GPU_INDEX="${GPU_INDEX:-0}"              # GPU 编号
WARMUP="${WARMUP:-1}"                    # 预热次数
REPEAT="${REPEAT:-3}"                    # 正式统计次数

# 是否额外启用 perf / nsys
ENABLE_PERF="${ENABLE_PERF:-0}"
ENABLE_NSYS="${ENABLE_NSYS:-0}"

# 调用 Python 主分析器，模式为 gpu
python3 "$ROOT_DIR/profiler.py" \
  --cmd "$CMD" \
  --mode gpu \
  --out-dir "$OUT_DIR" \
  --gpu-index "$GPU_INDEX" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT"

# 如有需要，再额外跑 perf 统计
if [[ "$ENABLE_PERF" == "1" ]]; then
  CMD="$CMD" bash "$ROOT_DIR/scripts/run_perf_stat.sh"
fi

# 如有需要，再额外跑 Nsight Systems
if [[ "$ENABLE_NSYS" == "1" ]]; then
  CMD="$CMD" bash "$ROOT_DIR/scripts/run_nsys_profile.sh"
fi