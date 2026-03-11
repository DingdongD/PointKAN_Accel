#!/usr/bin/env bash
# 仅 model(data) 推理：三任务依次采 推理 latency + energy，并可选做 breakdown；MODE=gpu（默认）或 cpu
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-gpu}"
RUN_BREAKDOWN="${RUN_BREAKDOWN:-1}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-2}"

if [[ "$MODE" == "cpu" ]]; then
  PROFILE_SCRIPT="$ROOT/scripts/profile_cpu.sh"
else
  PROFILE_SCRIPT="$ROOT/scripts/profile_gpu.sh"
fi
export ENABLE_PERF="${ENABLE_PERF:-0}"

echo ">>> profile_all: MODE=${MODE}，仅 model 推理 latency+energy (使用 ${PROFILE_SCRIPT})"
for TASK in modelnet40 scanobjectnn part_seg; do
  export TASK
  export WARMUP
  export REPEAT
  echo "========== Profiling task: $TASK (${MODE}) =========="
  bash "$PROFILE_SCRIPT" || true
done

if [[ "$RUN_BREAKDOWN" == "1" ]]; then
  echo "========== Running breakdown for all tasks =========="
  bash "$ROOT/scripts/breakdown_all.sh"
fi

echo ">>> 全部完成 (MODE=${MODE})。推理 latency/energy 在 outputs/${MODE}_<task>/ 下 summary.json"
