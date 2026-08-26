#!/usr/bin/env bash
# 无数据集时：仅从 checkpoint 导出 GRF，再用全局 x_range 跑 GQA-LUT PWL v2。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GQA_LUT="${GQA_LUT:-${ELITE_ROOT}/../../GQA-LUT}"
CKPT="${CKPT:-${SCRIPT_DIR}/checkpoints/pointKANElite_part_exp}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/grf_export_out}"
DEVICE="${DEVICE:-cpu}"
TOTAL_ITERS="${TOTAL_ITERS:-80}"
NUM_SPLITS="${NUM_SPLITS:-8}"
# 无校准时用保守全局激活区间（与常见 KAT 输入量级一致；有数据后请改用 inference 校准导出）
X_RANGE_MIN="${X_RANGE_MIN:--4}"
X_RANGE_MAX="${X_RANGE_MAX:-4}"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${ELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "== [1/2] 仅 checkpoint 导出 grf_coefficients.json（无 input_range） =="
python "${SCRIPT_DIR}/export_grf_coefficients_only.py" \
  --checkpoint "${CKPT}" \
  --out_json "${OUT_DIR}/grf_coefficients.json" \
  --device "${DEVICE}"

if [[ ! -f "${GQA_LUT}/grf_pwl_trainer_v2.py" ]]; then
  echo "未找到 GQA-LUT: ${GQA_LUT}"
  exit 1
fi

echo "== [2/2] PWL v2（全局 --x_range ${X_RANGE_MIN} ${X_RANGE_MAX}） =="
python "${GQA_LUT}/grf_pwl_trainer_v2.py" \
  --grf_coeff_json "${OUT_DIR}/grf_coefficients.json" \
  --output_json "${OUT_DIR}/pwl_v2.json" \
  --num_splits "${NUM_SPLITS}" \
  --total_iters "${TOTAL_ITERS}" \
  --decimal_bit_range 0 6 \
  --deploy_breakpoint_bits 6 \
  --w_b_bit 8 \
  --x_range "${X_RANGE_MIN}" "${X_RANGE_MAX}" \
  --seed 42

echo "完成: ${OUT_DIR}/pwl_v2.json"
echo "注意: 未做激活校准，x_range 为粗设；有数据后请用 inference 导出含 input_range 的 grf 并重跑。"
