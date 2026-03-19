#!/bin/bash
# spec_kan_fps_opt_kan.py：seg_filter FPS IOU 评估（对比 exact radii_ref 与 KAN 预测 radii_ref）
# 依赖：先运行 run_fps_radii_prefix_sweep.sh 或 run_eval_prefix_ratio.sh 得到 KAN 权重（如 runs/fps_radii_kan_K512_pr0.10/best.pt）
#
# 用法:
#   ./run_spec_kan_fps_opt_kan.sh                                    # 仅 exact_ref 分支（不传 KAN 目录）
#   ./run_spec_kan_fps_opt_kan.sh ./runs/fps_radii_kan_K512_pr0.10  # exact_ref + KAN 预测 radii_ref 对比
#   ./run_spec_kan_fps_opt_kan.sh ./runs/fps_radii_kan_K512_pr0.10 ./data 512  # 指定数据根目录与 K

set -e
cd "$(dirname "$0")"

KAN_SAVE_DIR="${1:-}"
ROOT="${2:-./data}"
K="${3:-512}"
SAVE_DIR="${4:-./runs/fps_seg_filter_iou_kan}"
BATCH_SIZE="${5:-16}"
SEGMENT_LEN="${6:-32}"

echo "========== spec_kan_fps_opt_kan: seg_filter FPS IOU (exact_ref vs KAN radii_ref) =========="
echo "  ROOT=$ROOT"
echo "  K=$K"
echo "  SAVE_DIR=$SAVE_DIR"
echo "  KAN_SAVE_DIR=$KAN_SAVE_DIR ${KAN_SAVE_DIR:+[KAN 分支开启]}"
echo "  BATCH_SIZE=$BATCH_SIZE  SEGMENT_LEN=$SEGMENT_LEN"
echo "================================================================================"

CMD=(
  python spec_kan_fps_opt_kan.py
  --root "$ROOT"
  --partition test
  --num_points 1024
  --K "$K"
  --segment_len "$SEGMENT_LEN"
  --batch_size "$BATCH_SIZE"
  --save_dir "$SAVE_DIR"
)
if [ -n "$KAN_SAVE_DIR" ]; then
  CMD+=(--kan_save_dir "$KAN_SAVE_DIR")
fi

"${CMD[@]}"

echo "Done. 结果目录: $SAVE_DIR"
