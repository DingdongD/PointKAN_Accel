#!/bin/bash
# ScanObjectNN: 校准后 W8A8 RTN 精度测试
# 用法: ./run_quant_scan.sh [checkpoint_path]
# 示例: ./run_quant_scan.sh ../PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite/best_checkpoint.pth

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <path_to_scan_checkpoint.pth>"
  echo "Example: $0 PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite/best_checkpoint.pth"
  exit 1
fi
python -m quantization.evaluate_quant_pointkan \
  --task scan \
  --checkpoint "$CHECKPOINT" \
  --num_points 1024 \
  --calibration_size 512 \
  --batch_size 32 \
  --device cuda
