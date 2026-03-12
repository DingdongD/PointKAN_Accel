#!/usr/bin/env bash
# 对比 Vanilla GRF-KAN 与 LUT 查表推理在 ModelNet40 测试集上的性能
# 使用前请先运行 run_grf_lut.sh 生成 LUT 目录。
# 使用：chmod +x run_inference_lut_compare.sh && ./run_inference_lut_compare.sh

set -e
cd "$(dirname "$0")"

CKPT="${CKPT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
LUT_DIR="${LUT_DIR:-./grf_lut}"
LUT_MODE="${LUT_MODE:-vanilla}"
USE_QUANTIZED="${USE_QUANTIZED:-0}"

EXTRA=""
if [ "$USE_QUANTIZED" = "1" ]; then
  EXTRA="--use_quantized"
fi
python run_inference_lut_compare.py --checkpoint "$CKPT" --lut_dir "$LUT_DIR" --lut_mode "$LUT_MODE" $EXTRA --out_json "${LUT_DIR}/inference_lut_compare.json"
# 使用 base+offset 表：LUT_MODE=base_offset ./run_inference_lut_compare.sh
# 使用量化表：先 QUANTIZE_BITS=8 ./run_grf_lut.sh，再 USE_QUANTIZED=1 ./run_inference_lut_compare.sh

# 示例：指定 LUT 目录
# LUT_DIR=./my_lut ./run_inference_lut_compare.sh
