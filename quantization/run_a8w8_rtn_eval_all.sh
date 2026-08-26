#!/bin/bash
# 一键串行跑完：ModelNet40 / ScanObjectNN / PartSeg 的 A8W8 RTN 精度评估
#
# 可选环境变量（都可不设）：
#   MODELNET_CKPT
#   SCAN_CKPT
#   PARTSEG_CKPT
#
# 默认 checkpoint 路径使用工程里 run_quant_*.sh 的示例值。

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ModelNet40: 默认使用实际存在的 best_checkpoint.pth
MODELNET_CKPT="${MODELNET_CKPT:-PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42/best_checkpoint.pth}"
SCAN_CKPT="${SCAN_CKPT:-PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42/best_checkpoint.pth}"
PARTSEG_CKPT="${PARTSEG_CKPT:-PointKAN-elite/part_segmentation/checkpoints/pointKANElite_part_exp/best_insiou_model.pth}"

echo "ModelNet40 checkpoint: ${MODELNET_CKPT}"
echo "ScanObjectNN checkpoint: ${SCAN_CKPT}"
echo "PartSeg checkpoint: ${PARTSEG_CKPT}"

echo "---- ModelNet40 ----"
bash "${SCRIPT_DIR}/run_a8w8_rtn_modelnet40.sh" "${MODELNET_CKPT}"

echo "---- ScanObjectNN ----"
bash "${SCRIPT_DIR}/run_a8w8_rtn_scanobjectnn.sh" "${SCAN_CKPT}"

echo "---- PartSeg ----"
bash "${SCRIPT_DIR}/run_a8w8_rtn_partseg.sh" "${PARTSEG_CKPT}"

echo "All evaluations finished."

