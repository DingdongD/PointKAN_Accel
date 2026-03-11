#!/usr/bin/env bash
# 仅 model(data) 推理：随机数据+随机权重 CPU，采 推理 latency（perf_counter）+ CPU energy（pyRAPL 仅包推理循环，不测进程）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PROFILER="${ROOT}/../base_version/profiler.py"
WORKSPACE="$(cd "$ROOT/../../../../" 2>/dev/null && pwd || true)"
ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${WORKSPACE}/PointKAN-elite}"

TASK="${TASK:-modelnet40}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/random_cpu_${TASK}}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
BATCH="${BATCH:-1}"
MAX_BATCHES="${MAX_BATCHES:-20}"
ENABLE_PERF="${ENABLE_PERF:-0}"

CMD="python3 ${ROOT}/run_infer_random.py --task ${TASK} --device cpu --batch-size ${BATCH} --max-batches ${MAX_BATCHES}"
export POINTKAN_ELITE_ROOT="${ELITE_ROOT}"

# 不启用进程级 perf，CPU 能耗由 run_infer_random 内 pyRAPL 仅包推理循环提供
EXTRA_PROFILER_ARGS=()
[[ "${ENABLE_PERF:-0}" == "1" ]] && EXTRA_PROFILER_ARGS+=(--enable-perf)

echo ">>> PointKAN CPU 推理 latency+energy (仅 model): task=${TASK} out_dir=${OUT_DIR}"
python3 "$BASE_PROFILER" \
  --cmd "$CMD" \
  --mode cpu \
  --out-dir "$OUT_DIR" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --print-inference-only \
  "${EXTRA_PROFILER_ARGS[@]}"

echo ">>> 结果目录: ${OUT_DIR}/<timestamp>"
