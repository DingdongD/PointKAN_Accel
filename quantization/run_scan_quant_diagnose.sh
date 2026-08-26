#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pointkan/bin/python}"
CHECKPOINT="${1:-PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42/best_checkpoint.pth}"
CALIB_SIZES="${CALIB_SIZES:-64 128 256 512}"
DEVICE="${DEVICE:-cuda}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"

A_OBSERVER="${A_OBSERVER:-MinMaxObserver}"
A_PERCENTILE_LOW="${A_PERCENTILE_LOW:-0.001}"
A_PERCENTILE_HIGH="${A_PERCENTILE_HIGH:-0.999}"

LOG_DIR="${LOG_DIR:-"${PWD}/my_quant_logs"}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${LOG_DIR}/scan_quant_diagnose_${TS}.json"
LOG_FILE="${LOG_DIR}/scan_quant_diagnose_${TS}.log"

OLD_IFS="${IFS}"
IFS=' '
read -r -a CALIB_SIZE_ARR <<< "${CALIB_SIZES}"
IFS="${OLD_IFS}"

CMD=(
  "${PYTHON_BIN}" -m quantization.scan_quant_diagnose
  --checkpoint "${CHECKPOINT}"
  --num_points "${NUM_POINTS}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --device "${DEVICE}"
  --calib_sizes "${CALIB_SIZE_ARR[@]}"
  --a_observer "${A_OBSERVER}"
  --a_percentile_low "${A_PERCENTILE_LOW}"
  --a_percentile_high "${A_PERCENTILE_HIGH}"
  --out_json "${OUT_JSON}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
echo "JSON: ${OUT_JSON}"

