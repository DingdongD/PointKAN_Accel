#!/bin/bash
# spec_kan_fps_opt_streaming_eval.py：Streaming block seg_filter 与 seg_filter 的 IOU / 耗时 / speedup 对比
# 用法:
#   ./run_spec_kan_fps_opt_streaming_eval.sh
#   ./run_spec_kan_fps_opt_streaming_eval.sh ./data 512 64   # root K block_len
#   ./run_spec_kan_fps_opt_streaming_eval.sh ./data 512 64 ./runs/fps_radii_kan_K512_pr0.10   # (可选)启用 KAN radii_ref 分支

set -e
cd "$(dirname "$0")"

ROOT="${1:-./data}"
K="${2:-512}"
BLOCK_LEN="${3:-64}"
SAVE_DIR="${4:-./runs/fps_seg_filter_streaming_eval}"
KAN_SAVE_DIR="${5:-}"

echo "========== Streaming vs seg_filter 评估 =========="
echo "  ROOT=$ROOT  K=$K  block_len=$BLOCK_LEN"
echo "  SAVE_DIR=$SAVE_DIR"
echo "=================================================="

python spec_kan_fps_opt_streaming_eval.py \
  --root "$ROOT" \
  --partition test \
  --num_points 1024 \
  --K "$K" \
  --segment_len 32 \
  --block_len "$BLOCK_LEN" \
  --batch_size 16 \
  --save_dir "$SAVE_DIR" \
  ${KAN_SAVE_DIR:+--kan_save_dir "$KAN_SAVE_DIR"}

echo "Done. 结果: ${SAVE_DIR}/streaming_vs_seg_filter_results.json"
