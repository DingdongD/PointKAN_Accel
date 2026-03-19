#!/bin/bash
# spec_kan_fps_opt_streaming_eval.py：Streaming block seg_filter 与 seg_filter 的 IOU / 耗时 / speedup 对比
# 用法:
#   ./run_spec_kan_fps_opt_streaming_eval.sh
#   ./run_spec_kan_fps_opt_streaming_eval.sh ./data 512 64 32   # root K block_len heap_cap

set -e
cd "$(dirname "$0")"

ROOT="${1:-./data}"
K="${2:-512}"
BLOCK_LEN="${3:-64}"
HEAP_CAP="${4:-32}"
SAVE_DIR="${5:-./runs/fps_seg_filter_streaming_eval}"

echo "========== Streaming vs seg_filter 评估 =========="
echo "  ROOT=$ROOT  K=$K  block_len=$BLOCK_LEN  heap_cap=$HEAP_CAP"
echo "  SAVE_DIR=$SAVE_DIR"
echo "=================================================="

python spec_kan_fps_opt_streaming_eval.py \
  --root "$ROOT" \
  --partition test \
  --num_points 1024 \
  --K "$K" \
  --segment_len 32 \
  --block_len "$BLOCK_LEN" \
  --heap_cap "$HEAP_CAP" \
  --batch_size 16 \
  --save_dir "$SAVE_DIR"

echo "Done. 结果: ${SAVE_DIR}/streaming_vs_seg_filter_results.json"
