#!/bin/bash
# PartSeg: 校准后 W8A8 RTN 精度测试
# 用法: ./run_quant_partseg.sh [checkpoint_path]
# 示例: ./run_quant_partseg.sh ../PointKAN-elite/part_segmentation/checkpoints/pointKANElite/best_insiou_model.pth

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <path_to_partseg_checkpoint.pth>"
  echo "Example: $0 PointKAN-elite/part_segmentation/checkpoints/pointKANElite/best_insiou_model.pth"
  exit 1
fi
python -m quantization.evaluate_quant_pointkan \
  --task partseg \
  --checkpoint "$CHECKPOINT" \
  --calibration_size 512 \
  --batch_size 32 \
  --device cuda
