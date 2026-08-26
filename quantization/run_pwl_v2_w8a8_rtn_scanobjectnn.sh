#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pointkan/bin/python}"

CHECKPOINT="${1:-PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42/best_checkpoint.pth}"
PWL_V2_JSON="${PWL_V2_JSON:-}"

CALIBRATION_SIZE="${CALIBRATION_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

A_OBSERVER="${A_OBSERVER:-MinMaxObserver}"
A_PERCENTILE_LOW="${A_PERCENTILE_LOW:-0.01}"
A_PERCENTILE_HIGH="${A_PERCENTILE_HIGH:-0.99}"

PWL_BIT="${PWL_BIT:-}"
OUT_JSON="${OUT_JSON:-${PWD}/my_pwl_w8a8_out/scanobjectnn_pwl_w8a8_rtn.json}"
mkdir -p "$(dirname "${OUT_JSON}")"

ARGS=(
  "${PYTHON_BIN}" -m quantization.evaluate_pwl_v2_w8a8_rtn_pointkan
  --task scan
  --checkpoint "${CHECKPOINT}"
  --pwl_v2_json "${PWL_V2_JSON}"
  --calibration_size "${CALIBRATION_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --device "${DEVICE}"
  --a_observer "${A_OBSERVER}"
  --a_percentile_low "${A_PERCENTILE_LOW}"
  --a_percentile_high "${A_PERCENTILE_HIGH}"
  --out_json "${OUT_JSON}"
)

if [ -n "${PWL_BIT}" ]; then
  ARGS+=(--pwl_bit "${PWL_BIT}")
fi

echo "Running: ${ARGS[*]}"
"${ARGS[@]}"

