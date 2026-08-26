#!/usr/bin/env bash
# 从 ScanObjectNN checkpoint 导出 GRF 系数，再调用 GQA-LUT grf_pwl_trainer_v2 生成 pwl_v2.json。
# FAST=1：仅用 checkpoint 导出（秒级），PWL 使用全局 X_RANGE_*（无 input_range）。
# FAST=0：走 inference 校准导出（需数据，CPU 上整网测试较慢）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GQA_LUT="${GQA_LUT:-${ELITE_ROOT}/../../GQA-LUT}"
CKPT="${CKPT:-${SCRIPT_DIR}/checkpoints/pointKANElite-scan_exp-42}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/grf_export_out}"
DEVICE="${DEVICE:-cpu}"
FAST="${FAST:-1}"
# PWL 遗传迭代；提高精度可增大 TOTAL_ITERS（如 200）
TOTAL_ITERS="${TOTAL_ITERS:-80}"
NUM_SPLITS="${NUM_SPLITS:-8}"
X_RANGE_MIN="${X_RANGE_MIN:--4}"
X_RANGE_MAX="${X_RANGE_MAX:-4}"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${ELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${FAST}" == "1" ]]; then
  echo "== [1/2] FAST：仅 checkpoint 导出 grf_coefficients.json =="
  python "${SCRIPT_DIR}/export_grf_coefficients_only.py" \
    --checkpoint "${CKPT}" \
    --out_json "${OUT_DIR}/grf_coefficients.json" \
    --device "${DEVICE}"
  EXTRA_XRANGE=(--x_range "${X_RANGE_MIN}" "${X_RANGE_MAX}")
else
  echo "== [1/2] 导出 grf_coefficients.json（含校准 input_range） =="
  python "${SCRIPT_DIR}/inference_and_kan_analysis.py" \
    --checkpoint "${CKPT}" \
    --no_vis \
    --device "${DEVICE}" \
    --calibration_samples "${CALIB_SAMPLES:-300}" \
    --export_grf_json "${OUT_DIR}/grf_coefficients.json"
  EXTRA_XRANGE=()
fi

if [[ ! -f "${GQA_LUT}/grf_pwl_trainer_v2.py" ]]; then
  echo "未找到 GQA-LUT: ${GQA_LUT}（可 export GQA_LUT=你的路径）"
  exit 1
fi

echo "== [2/2] GRF-PWL v2 训练/导出（DEAP，可能数分钟） =="
python "${GQA_LUT}/grf_pwl_trainer_v2.py" \
  --grf_coeff_json "${OUT_DIR}/grf_coefficients.json" \
  --output_json "${OUT_DIR}/pwl_v2.json" \
  --num_splits "${NUM_SPLITS}" \
  --total_iters "${TOTAL_ITERS}" \
  --decimal_bit_range 0 6 \
  --deploy_breakpoint_bits 6 \
  --w_b_bit 8 \
  --seed 42 \
  "${EXTRA_XRANGE[@]:-}"

echo "完成: ${OUT_DIR}/pwl_v2.json"
