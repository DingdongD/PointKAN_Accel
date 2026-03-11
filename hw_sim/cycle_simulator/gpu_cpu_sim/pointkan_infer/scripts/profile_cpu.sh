#!/usr/bin/env bash
# 仅 model(data) 推理：PointKAN CPU，采 推理 latency + CPU energy（pyRAPL 仅包推理循环）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PROFILER="${ROOT}/../base_version/profiler.py"
WORKSPACE="$(cd "$ROOT/../../../../" 2>/dev/null && pwd || true)"
ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${WORKSPACE}/PointKAN-elite}"

TASK="${TASK:-modelnet40}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  case "$TASK" in
    modelnet40)   CHECKPOINT="${ELITE_ROOT}/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42" ;;
    scanobjectnn) CHECKPOINT="${ELITE_ROOT}/classification_ScanObjectNN/checkpoints/pointKANElite-ScanObjectNN" ;;
    part_seg)     CHECKPOINT="${ELITE_ROOT}/part_segmentation/checkpoints/pointKANElite_part_exp" ;;
    *) echo "Unknown TASK=$TASK"; exit 1 ;;
  esac
fi

OUT_DIR="${OUT_DIR:-$ROOT/outputs/cpu_${TASK}}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
BATCH="${BATCH:-1}"
MAX_BATCHES="${MAX_BATCHES:-1}"
# 不测进程级 perf/RAPL，CPU 能耗由 run_infer 内 pyRAPL 仅包推理循环提供
ENABLE_PERF="${ENABLE_PERF:-0}"

CMD="python3 ${ROOT}/run_infer.py --task ${TASK} --checkpoint ${CHECKPOINT} --batch-size ${BATCH} --max-batches ${MAX_BATCHES} --device cpu"
export POINTKAN_ELITE_ROOT="${ELITE_ROOT}"

EXTRA_PROFILER_ARGS=()
[[ "$ENABLE_PERF" == "1" ]] && EXTRA_PROFILER_ARGS+=(--enable-perf)

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
