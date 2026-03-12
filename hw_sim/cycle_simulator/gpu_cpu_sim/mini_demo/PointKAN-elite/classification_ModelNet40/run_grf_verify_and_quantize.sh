#!/usr/bin/env bash
# GRF value 空间残差验证 + f(x) 曲线量化分析（通过 sh 调用）
# 使用：chmod +x run_grf_verify_and_quantize.sh && ./run_grf_verify_and_quantize.sh
# 通过注释切换：从 checkpoint 跑完整流程，或仅从已有 GRF JSON 做验证与量化。

set -e
cd "$(dirname "$0")"

# 检查点目录（含 best_checkpoint.pth）；若仅用 --grf_json 可留空
CKPT="${CKPT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"

# 输出目录
OUT_DIR="${OUT_DIR:-./grf_analysis}"

# 可选：kan_analysis_results.json 路径，用于每层 x 的校准范围（若不设则用默认 [-2,2]）
# ANALYSIS_JSON="${ANALYSIS_JSON:-./kan_analysis/kan_analysis_results.json}"

# f(x) 整曲线量化比特数
BITS="${BITS:-8 6 4}"
# base + offset 分量分别量化：base 比特、offset 比特（探索 offset 低比特）
BASE_BITS="${BASE_BITS:-8}"
OFFSET_BITS="${OFFSET_BITS:-8 6 4}"
BASE_KIND="${BASE_KIND:-mean}"

# ========== 通过注释切换：只保留要执行的一段 ==========

# 方式一：从 checkpoint 导出 + 残差验证 + f(x) 量化 + base/offset 分别量化与可视化
python run_grf_verify_and_quantize.py --checkpoint "$CKPT" --out_dir "$OUT_DIR" --bits $BITS \
  --base_bits "$BASE_BITS" --offset_bits $OFFSET_BITS --base_kind "$BASE_KIND"

# 方式二：从已有 GRF JSON 做验证与量化（不读 checkpoint）
# python run_grf_verify_and_quantize.py --grf_json "$OUT_DIR/grf_coefficients.json" --out_dir "$OUT_DIR" \
#   --bits $BITS --base_bits "$BASE_BITS" --offset_bits $OFFSET_BITS --base_kind "$BASE_KIND"

# 方式三：从 checkpoint + 指定 analysis JSON（带校准 x 范围）
# python run_grf_verify_and_quantize.py --checkpoint "$CKPT" --out_dir "$OUT_DIR" --analysis_json "${ANALYSIS_JSON}" --bits $BITS

# 方式四：只跑验证与量化，不生成量化对比图
# python run_grf_verify_and_quantize.py --checkpoint "$CKPT" --out_dir "$OUT_DIR" --bits $BITS --no_vis

# 示例：探索 offset 低比特（base=8bit, offset=6/4bit）
# BASE_BITS=8 OFFSET_BITS="8 6 4" ./run_grf_verify_and_quantize.sh
# 示例：指定输出目录
# OUT_DIR=./my_grf ./run_grf_verify_and_quantize.sh
