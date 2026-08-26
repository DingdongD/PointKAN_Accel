#!/usr/bin/env bash
#
# ScanObjectNN：与 ModelNet40 相同管线 —— 校准集上 collect_activation_ranges，
# 导出带 input_range 的 grf_coefficients.json，再跑 GQA-LUT grf_pwl_trainer_v2.py。
#
# 依赖：ScanObjectNN 数据路径可用（见 ScanObjectNN.py 中 h5 路径）。
#
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ELITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${ELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ===================== 用户参数 =====================
GQA_LUT="${GQA_LUT:-${ELITE_ROOT}/../../GQA-LUT}"

DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-scan_exp-42}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/grf_export_out}"
GRF_JSON="${GRF_JSON:-${OUT_DIR}/grf_coefficients.json}"
PWL_JSON="${PWL_JSON:-${OUT_DIR}/pwl_v2.json}"

CALIBRATION_PARTITION="${CALIBRATION_PARTITION:-train}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-500}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"

NUM_SPLITS="${NUM_SPLITS:-8}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
DECIMAL_BIT_LO="${DECIMAL_BIT_LO:-0}"
DECIMAL_BIT_HI="${DECIMAL_BIT_HI:-6}"
DEPLOY_BREAKPOINT_BITS="${DEPLOY_BREAKPOINT_BITS:-6}"
W_B_BIT="${W_B_BIT:-8}"
X_STEP="${X_STEP:-0.01}"
POP_SIZE="${POP_SIZE:-50}"
PWL_EXTRA_ARGS=${PWL_EXTRA_ARGS:-}

# ===================== 流程 =====================
mkdir -p "${OUT_DIR}"

if [[ ! -f "${GQA_LUT}/grf_pwl_trainer_v2.py" ]]; then
  echo "错误: 未找到 ${GQA_LUT}/grf_pwl_trainer_v2.py"
  exit 1
fi

echo ">>> [1/2] 校准 + 导出 GRF（device=${DEVICE}）"
python "${SCRIPT_DIR}/inference_and_kan_analysis.py" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --num_points "${NUM_POINTS}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --calibration_partition "${CALIBRATION_PARTITION}" \
  --calibration_samples "${CALIBRATION_SAMPLES}" \
  --export_grf_json "${GRF_JSON}"

[[ -f "${GRF_JSON}" ]] || { echo "缺失 ${GRF_JSON}"; exit 1; }

echo ">>> [2/2] PWL v2"
# shellcheck disable=SC2086
python "${GQA_LUT}/grf_pwl_trainer_v2.py" \
  --grf_coeff_json "${GRF_JSON}" \
  --output_json "${PWL_JSON}" \
  --num_splits "${NUM_SPLITS}" \
  --total_iters "${TOTAL_ITERS}" \
  --decimal_bit_range "${DECIMAL_BIT_LO}" "${DECIMAL_BIT_HI}" \
  --deploy_breakpoint_bits "${DEPLOY_BREAKPOINT_BITS}" \
  --w_b_bit "${W_B_BIT}" \
  --x_step "${X_STEP}" \
  --pop_size "${POP_SIZE}" \
  --seed "${SEED}" \
  ${PWL_EXTRA_ARGS}

echo "完成: ${PWL_JSON}"
