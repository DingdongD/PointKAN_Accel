#!/usr/bin/env bash
# GRF-KAN 均匀 LUT 构建、验证与导出（供模型推理查表使用）
# 使用：chmod +x run_grf_lut.sh && ./run_grf_lut.sh
# 通过注释/环境变量切换：checkpoint、analysis JSON、n_bins、输出目录等。

set -e
cd "$(dirname "$0")"

# 检查点目录（含 best_checkpoint.pth）；若仅用 --grf_json 可留空
CKPT="${CKPT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"

# 输出目录（LUT 与验证结果）
OUT_DIR="${OUT_DIR:-./grf_lut}"

# 均匀分割格点数
N_BINS="${N_BINS:-256}"

# base+offset LUT 的 base 类型
BASE_KIND="${BASE_KIND:-mean}"

# 表值量化比特数（默认 0=不量化；4/6/8 会生成量化表并输出量化误差）
# 可分别指定：VANILLA_QUANTIZE_BITS=8 BASE_OFFSET_QUANTIZE_BITS=4 表示 vanilla 8bit、base+offset 4bit
QUANTIZE_BITS="${QUANTIZE_BITS:-0}"

# 可选：kan_analysis_results.json，提供每层 x 校准范围
# ANALYSIS_JSON="${ANALYSIS_JSON:-./my_kan_analysis/kan_analysis_results.json}"

# ========== 通过注释切换执行方式 ==========

# 方式一：从 checkpoint 构建 LUT（并可选指定校准范围、量化）
# 使用校准集得到每层 x_min/x_max：加 --calibrate（及可选 --calibration_partition --calibration_samples）
python run_grf_lut_verify.py --checkpoint "$CKPT" --out_dir "$OUT_DIR" --n_bins "$N_BINS" --base_kind "$BASE_KIND" --quantize_bits "$QUANTIZE_BITS" \
  ${VANILLA_QUANTIZE_BITS:+--vanilla_quantize_bits "$VANILLA_QUANTIZE_BITS"} \
  ${BASE_OFFSET_QUANTIZE_BITS:+--base_offset_quantize_bits "$BASE_OFFSET_QUANTIZE_BITS"} \
  ${BASE_TABLE_QUANTIZE_BITS:+--base_table_quantize_bits "$BASE_TABLE_QUANTIZE_BITS"} \
  ${OFFSET_TABLE_QUANTIZE_BITS:+--offset_table_quantize_bits "$OFFSET_TABLE_QUANTIZE_BITS"} \
  ${CALIBRATE:+--calibrate} ${CALIBRATION_PARTITION:+--calibration_partition "$CALIBRATION_PARTITION"} ${CALIBRATION_SAMPLES:+--calibration_samples "$CALIBRATION_SAMPLES"}

# 方式二：从 checkpoint + 指定 analysis JSON（带每层 x 范围）
# python run_grf_lut_verify.py --checkpoint "$CKPT" --analysis_json "${ANALYSIS_JSON}" --out_dir "$OUT_DIR" --n_bins "$N_BINS" --base_kind "$BASE_KIND" --quantize_bits "$QUANTIZE_BITS"

# 方式三：从已有 GRF JSON 构建 LUT（不读 checkpoint）
# python run_grf_lut_verify.py --grf_json "$OUT_DIR/grf_coefficients.json" --out_dir "$OUT_DIR" --n_bins "$N_BINS" --base_kind "$BASE_KIND" --quantize_bits "$QUANTIZE_BITS"

# 方式四：不画图
# python run_grf_lut_verify.py --checkpoint "$CKPT" --out_dir "$OUT_DIR" --n_bins "$N_BINS" --no_vis

# 示例：指定 n_bins
# N_BINS=512 ./run_grf_lut.sh
# 示例：生成 8bit 量化表并输出量化误差
# QUANTIZE_BITS=8 ./run_grf_lut.sh
# 示例：vanilla 8bit、base+offset 中 base 8bit / offset 4bit 分别控制
# VANILLA_QUANTIZE_BITS=8 BASE_TABLE_QUANTIZE_BITS=8 OFFSET_TABLE_QUANTIZE_BITS=4 QUANTIZE_BITS=0 ./run_grf_lut.sh
# 示例：用校准集得到每层 x_min/x_max（替代默认 [-2,2]）
# CALIBRATE=1 ./run_grf_lut.sh
# CALIBRATE=1 CALIBRATION_PARTITION=test CALIBRATION_SAMPLES=300 ./run_grf_lut.sh
# 示例：指定输出目录
# OUT_DIR=./my_lut ./run_grf_lut.sh
#
# 生成 LUT 后，可用 run_inference_lut_compare.sh 对比 Vanilla 与 LUT 推理性能；
# 可用 run_grf_lut_quantization_analysis.sh 分析多档量化精度对推理性能的影响并出图：
#   ./run_inference_lut_compare.sh
# 或：python run_inference_lut_compare.py --checkpoint "$CKPT" --lut_dir "$OUT_DIR" --out_json "$OUT_DIR/inference_lut_compare.json"
