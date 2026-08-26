#!/usr/bin/env bash
#
# 与 inference_and_kan_analysis.py 中逻辑一致：
#   - collect_activation_ranges：校准集上前向，forward_pre_hook 统计每层 KAT_Group 输入的 min/max
#   - export_grf_coefficients_to_json：将上述范围写入 grf_coefficients.json 各层的 input_range
#   - grf_pwl_trainer_v2.py：按每层 input_range 做 PWL v2（无需全局 --x_range）
#
# 使用：在本目录执行，按需修改下方「用户参数」区后运行：
#   chmod +x build_pwl_v2_calibrated.sh
#   ./build_pwl_v2_calibrated.sh
#
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ELITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${ELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ===================== 用户参数（请按需填写）=====================
# GQA-LUT 仓库根目录（含 grf_pwl_trainer_v2.py）；默认：与 PointKAN-pytorch 同级的 GQA-LUT
GQA_LUT="${GQA_LUT:-${ELITE_ROOT}/../../GQA-LUT}"

DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/grf_export_out}"
GRF_JSON="${GRF_JSON:-${OUT_DIR}/grf_coefficients.json}"
PWL_JSON="${PWL_JSON:-${OUT_DIR}/pwl_v2.json}"

# 校准集（与 inference.sh 一致）：train | test
CALIBRATION_PARTITION="${CALIBRATION_PARTITION:-train}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-500}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"

# PWL v2（grf_pwl_trainer_v2.py）
NUM_SPLITS="${NUM_SPLITS:-8}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
DECIMAL_BIT_LO="${DECIMAL_BIT_LO:-0}"
DECIMAL_BIT_HI="${DECIMAL_BIT_HI:-6}"
DEPLOY_BREAKPOINT_BITS="${DEPLOY_BREAKPOINT_BITS:-6}"
W_B_BIT="${W_B_BIT:-8}"
X_STEP="${X_STEP:-0.01}"
POP_SIZE="${POP_SIZE:-50}"
# 额外传给 trainer 的参数（数组）
PWL_EXTRA_ARGS=${PWL_EXTRA_ARGS:-}

# ===================== 流程 =====================
mkdir -p "${OUT_DIR}"

if [[ ! -f "${GQA_LUT}/grf_pwl_trainer_v2.py" ]]; then
  echo "错误: 未找到 ${GQA_LUT}/grf_pwl_trainer_v2.py ，请设置 GQA_LUT=/你的/GQA-LUT"
  exit 1
fi

echo ">>> [1/2] 校准激活范围并导出 GRF（GPU/设备: ${DEVICE}）"
echo "    checkpoint=${CHECKPOINT}"
echo "    输出: ${GRF_JSON}"
python "${SCRIPT_DIR}/inference_and_kan_analysis.py" \
  --checkpoint "${CHECKPOINT}" \
  --no_vis \
  --device "${DEVICE}" \
  --num_points "${NUM_POINTS}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --calibration_partition "${CALIBRATION_PARTITION}" \
  --calibration_samples "${CALIBRATION_SAMPLES}" \
  --export_grf_json "${GRF_JSON}"

if [[ ! -f "${GRF_JSON}" ]]; then
  echo "错误: 未生成 ${GRF_JSON}"
  exit 1
fi

echo ">>> [2/2] PWL v2 训练并写出 JSON（可能较久）"
echo "    输出: ${PWL_JSON}"
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
