#!/usr/bin/env bash
# 对三个任务各跑一次 breakdown，输出 JSON 到 outputs/breakdown_<task>.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/../../../../" 2>/dev/null && pwd || true)"
ELITE_ROOT="${POINTKAN_ELITE_ROOT:-${WORKSPACE}/PointKAN-elite}"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
MAX_BATCHES="${MAX_BATCHES:-5}"
export POINTKAN_ELITE_ROOT="${ELITE_ROOT}"

mkdir -p "$OUT_BASE"

for TASK in modelnet40 scanobjectnn part_seg; do
  case "$TASK" in
    modelnet40)   CKPT="${ELITE_ROOT}/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42" ;;
    scanobjectnn) CKPT="${ELITE_ROOT}/classification_ScanObjectNN/checkpoints/pointKANElite-ScanObjectNN" ;;
    part_seg)     CKPT="${ELITE_ROOT}/part_segmentation/checkpoints/pointKANElite_part_exp" ;;
  esac
  if [[ ! -d "$CKPT" && ! -f "$CKPT" ]]; then
    echo "Skip $TASK: checkpoint not found $CKPT"
    continue
  fi
  echo ">>> Breakdown: $TASK"
  python3 "$ROOT/breakdown_profiler.py" \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --max-batches "$MAX_BATCHES" \
    --out "$OUT_BASE/breakdown_${TASK}.json"
done

echo ">>> Breakdown 结果: $OUT_BASE/breakdown_*.json"
