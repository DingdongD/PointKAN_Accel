#!/bin/bash
# ScanObjectNN: A8W8 RTN（两阶段校准）精度评估
#
# 说明：
# 当前 quantization 工程仅对 nn.Conv1d/nn.Conv2d/nn.Linear 做 fake-quant；
# BatchNorm/LayerNorm 本身不会被替换成 fake-quant 模块（保持浮点推理）。
#
# 用法：
#   ./run_a8w8_rtn_scanobjectnn.sh <checkpoint_path>
#
# 可选环境变量（都可不设）：
#   NUM_POINTS       默认 1024
#   CALIBRATION_SIZE 默认 512
#   BATCH_SIZE       默认 32
#   WORKERS          默认 4
#   DEVICE           默认 cuda
#
# 注意：
# ScanObjectNN 数据读取路径在模型代码中写死到 /workspace/VoxelNet/...

set -euo pipefail
IFS=$'\n\t'

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pointkan/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  echo "Usage: $0 <checkpoint_path>"
  echo "Example: $0 PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite/best_checkpoint.pth"
  exit 1
fi

NUM_POINTS="${NUM_POINTS:-1024}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="${LOG_DIR:-"${PWD}/my_quant_logs"}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/scanobjectnn_a8w8_rtn_${TS}.log"

CMD=(
  "${PYTHON_BIN}" -m quantization.evaluate_quant_pointkan
  --task scan
  --checkpoint "${CHECKPOINT}"
  --num_points "${NUM_POINTS}"
  --calibration_size "${CALIBRATION_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --device "${DEVICE}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"

