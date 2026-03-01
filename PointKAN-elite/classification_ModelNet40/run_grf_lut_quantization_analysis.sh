#!/usr/bin/env bash
# 分析不同量化精度（4/6/8 bit）对 LUT 推理性能的影响，并生成柱状图与直方图
# 使用：chmod +x run_grf_lut_quantization_analysis.sh && ./run_grf_lut_quantization_analysis.sh
# 依赖：先有基准浮点 LUT 目录（如 ./run_grf_lut.sh 生成的 ./grf_lut），且含 grf_coefficients.json

set -e
cd "$(dirname "$0")"

CKPT="${CKPT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
LUT_DIR="${LUT_DIR:-./grf_lut}"
OUT_DIR="${OUT_DIR:-./grf_lut_quantization_analysis}"

python run_grf_lut_quantization_analysis.py --checkpoint "$CKPT" --lut_dir "$LUT_DIR" --out_dir "$OUT_DIR" --bits 2 4 6 8

# 仅写 JSON 不画图：python run_grf_lut_quantization_analysis.py ... --no_plot
# 指定比特：--bits 4 8
