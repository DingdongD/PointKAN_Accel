#!/usr/bin/env bash
# 推理验证 + KAN 分组因果曲线 f(x) + 组间相似性绘图 + JSON 输出
# 使用：chmod +x run_inference_kan.sh && ./run_inference_kan.sh
# 只保留要执行的那一段取消注释，其余注释掉即可。
# 注意：若 inference_and_kan_analysis.py 功能或参数有更新，请同步更新本脚本。

set -e
cd "$(dirname "$0")"

# 检查点：目录（含 best_checkpoint.pth）或直接 .pth 路径
CKPT="${CKPT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"

# 校准集（决定 f(x) 与相似性分析的 x 轴范围）
CALIBRATION_PARTITION="${CALIBRATION_PARTITION:-train}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-500}"
CALIBRATION_MARGIN="${CALIBRATION_MARGIN:-0.05}"

# 可选：指定 JSON 输出路径（不设则默认写到 out_dir/kan_analysis_results.json）
# JSON_OUTPUT="${JSON_OUTPUT:-}"

# ========== 通过注释切换：只保留要执行的那一段，其余注释掉 ==========

# 方式一：只做推理（不生成图与 JSON）；可加 --print_shapes 输出每层 shape
# python inference_and_kan_analysis.py --checkpoint "$CKPT" --no_vis
# python inference_and_kan_analysis.py --checkpoint "$CKPT" --no_vis --print_shapes

# 方式二：推理 + 校准 + f(x) 曲线 + 组间相似性图 + JSON（激活范围与相似性/压缩分析）
# python inference_and_kan_analysis.py --checkpoint "$CKPT" \
#   --calibration_partition "$CALIBRATION_PARTITION" \
#   --calibration_samples "$CALIBRATION_SAMPLES" \
#   --calibration_margin "$CALIBRATION_MARGIN"
# 若需每层 shape 分析，在以上命令末尾加 --print_shapes


# 方式三：同上，并指定输出目录与 JSON 路径（可选加 --print_shapes 做每层 shape 分析）
python inference_and_kan_analysis.py --checkpoint "$CKPT" --out_dir ./my_kan_analysis \
  --calibration_partition "$CALIBRATION_PARTITION" \
  --calibration_samples "$CALIBRATION_SAMPLES" \
  --calibration_margin "$CALIBRATION_MARGIN" \
  --print_shapes \
  ${JSON_OUTPUT:+--json_output "$JSON_OUTPUT"}


# 示例：仅写 JSON、不画图
# python inference_and_kan_analysis.py --checkpoint "$CKPT" --no_vis --json_output ./kan_results.json \
#   --calibration_partition "$CALIBRATION_PARTITION" --calibration_samples "$CALIBRATION_SAMPLES"

# 示例：用 test 集 300 样本校准、扩展 10%
# CALIBRATION_PARTITION=test CALIBRATION_SAMPLES=300 CALIBRATION_MARGIN=0.10 ./run_inference_kan.sh

# 示例：导出 GRF 系数到 JSON（便于从 JSON 还原 GRF-KAN）
# python inference_and_kan_analysis.py --checkpoint "$CKPT" --no_vis --export_grf_json ./grf_analysis/grf_coefficients.json \
#   --calibration_partition "$CALIBRATION_PARTITION" --calibration_samples "$CALIBRATION_SAMPLES"

# 示例：GRF value 空间 base+残差 验证 + f(x) 曲线量化分析与可视化（见 run_grf_verify_and_quantize.sh）
# ./run_grf_verify_and_quantize.sh
# 或: python run_grf_verify_and_quantize.py --checkpoint "$CKPT" --out_dir ./grf_analysis --bits 8 6 4
