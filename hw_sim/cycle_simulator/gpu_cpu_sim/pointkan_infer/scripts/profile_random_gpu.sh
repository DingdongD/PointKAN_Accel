#!/usr/bin/env bash
# 仅 model(data) 推理：随机数据+随机权重 GPU，采 推理 latency（cuda.Event）+ GPU 功耗/能耗（单卡，不依赖 checkpoint/数据集）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PROFILER="${ROOT}/../base_version/profiler.py"
WORKSPACE="$(cd "$ROOT/../../../../" 2>/dev/null && pwd || true)"
ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${WORKSPACE}/PointKAN-elite}"

TASK="${TASK:-modelnet40}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/random_gpu_${TASK}}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-2}"
GPU_INDEX="${GPU_INDEX:-0}"
BATCH="${BATCH:-1}"
MAX_BATCHES="${MAX_BATCHES:-20}"
ENABLE_PERF="${ENABLE_PERF:-0}"

CMD="python3 ${ROOT}/run_infer_random.py --task ${TASK} --device cuda --batch-size ${BATCH} --max-batches ${MAX_BATCHES}"
export POINTKAN_ELITE_ROOT="${ELITE_ROOT}"

EXTRA_PROFILER_ARGS=()
[[ "$ENABLE_PERF" == "1" ]] && EXTRA_PROFILER_ARGS+=(--enable-perf)

echo ">>> PointKAN GPU 推理 latency+功耗+能耗 (仅 model，单卡 gpu=${GPU_INDEX}): task=${TASK} out_dir=${OUT_DIR}"
python3 "$BASE_PROFILER" \
  --cmd "$CMD" \
  --mode gpu \
  --out-dir "$OUT_DIR" \
  --gpu-index "$GPU_INDEX" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --print-inference-only \
  "${EXTRA_PROFILER_ARGS[@]}"

echo ">>> 结果目录: ${OUT_DIR}/<timestamp>"
