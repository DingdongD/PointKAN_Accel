#!/bin/bash
# 训练多组 prefix_ratio，并汇总测试集 loss 到 JSON + PNG；支持 model_type=mlp 或 kan
# 用法:
#   ./run_fps_radii_prefix_sweep.sh                         # 默认 base=./runs, K=512, mlp
#   ./run_fps_radii_prefix_sweep.sh ./runs 512 kan          # 使用 KAN
#   ./run_fps_radii_prefix_sweep.sh ./runs 256 "" "0.05 0.10 0.15 0.20 0.25"  # 自定义 pr 列表

set -e
cd "$(dirname "$0")"

BASE_DIR="${1:-./runs}"
FPS_STEPS="${2:-512}"
MODEL_TYPE="${3:-mlp}"   # mlp | kan
# 不同 prefix_ratio，用空格分隔（第 4 个参数，若为空则用默认）
PREFIX_RATIOS="${4:-0.05 0.10 0.15 0.20}"
# 目录名前缀：fps_radii_mlp 或 fps_radii_kan
NAME_PREFIX="fps_radii_${MODEL_TYPE}"
# radii 只构建一次，所有 prefix_ratio 共用
RADII_DIR="${BASE_DIR}/radii_cache_K${FPS_STEPS}"

echo "========== FPS Radii prefix_ratio 扫描 (model_type=$MODEL_TYPE) =========="
echo "  BASE_DIR=$BASE_DIR"
echo "  FPS_STEPS=$FPS_STEPS"
echo "  RADII_DIR=$RADII_DIR (共用，只构建一次)"
echo "  MODEL_TYPE=$MODEL_TYPE"
echo "  PREFIX_RATIOS=$PREFIX_RATIOS"
echo "==========================================================================="

mkdir -p "$BASE_DIR"

for pr in $PREFIX_RATIOS; do
  # 保存目录名含 model 类型与 prefix_ratio；radii 从 RADII_DIR 加载/保存，不重复构建
  SAVE_DIR="${BASE_DIR}/${NAME_PREFIX}_K${FPS_STEPS}_pr${pr}"
  echo ""
  echo "[Train] prefix_ratio=$pr model_type=$MODEL_TYPE -> $SAVE_DIR (radii: $RADII_DIR)"
  python spec_kan_fps.py \
    --save_dir "$SAVE_DIR" \
    --fps_steps "$FPS_STEPS" \
    --prefix_ratio "$pr" \
    --model_type "$MODEL_TYPE" \
    --radii_dir "$RADII_DIR"
done

echo ""
echo "[Collect] 汇总 results.json -> prefix_ratio_summary.json + .png"
python collect_prefix_ratio_results.py \
  --base_dir "$BASE_DIR" \
  --pattern "${NAME_PREFIX}_K${FPS_STEPS}_pr*" \
  --out_json "${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}.json" \
  --out_png "${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}.png"

echo "Done. 汇总: ${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}.json, ${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}.png"
