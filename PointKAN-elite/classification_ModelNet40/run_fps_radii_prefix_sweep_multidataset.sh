#!/bin/bash
# 多数据集 FPS radii KAN/MLP：扫描 prefix_ratio，共用 radii 缓存目录。
#
# 用法:
#   chmod +x run_fps_radii_prefix_sweep_multidataset.sh
#
# 与仓库内数据路径一致（等同 run_knn_coarse_fine_eval_*.sh / ScanObjectNN inference.sh）:
#   PRESET=voxelnet ./run_fps_radii_prefix_sweep_multidataset.sh kitti
#   （可不设 DATA_ROOT，脚本不传 --data_root，由 Python 填入默认绝对路径）
#
# 手动路径:
#   PRESET=none DATA_ROOT=./data ./run_fps_radii_prefix_sweep_multidataset.sh modelnet40
#   DATA_ROOT=/path/to/velodyne KITTI_USE_ROI=1 ./run_fps_radii_prefix_sweep_multidataset.sh kitti
#
# ScanObjectNN:
#   PRESET=voxelnet ./run_fps_radii_prefix_sweep_multidataset.sh scanobjectnn
#
# ShapeNet Part:
#   PRESET=voxelnet ./run_fps_radii_prefix_sweep_multidataset.sh shapenet_part
#
# Waymo（KITTI 格式 velodyne）:
#   PRESET=voxelnet KITTI_USE_ROI=1 ./run_fps_radii_prefix_sweep_multidataset.sh waymo
#
# 参数: DATASET [BASE_DIR] [FPS_STEPS] [MODEL_TYPE] [PREFIX_RATIOS]
#
set -e
cd "$(dirname "$0")"

DATASET="${1:?用法: $0 DATASET [BASE_DIR] [FPS_STEPS] [MODEL_TYPE] [PREFIX_RATIOS]}"
BASE_DIR="${2:-./runs}"
FPS_STEPS="${3:-512}"
MODEL_TYPE="${4:-kan}"
PREFIX_RATIOS="${5:-0.05 0.10 0.15 0.20}"

PRESET="${PRESET:-none}"
# preset=none 且未设置 DATA_ROOT 时默认 ./data；voxelnet 且未设置则不传 --data_root（由 Python 填路径）
if [[ "$PRESET" == "none" ]]; then
  DATA_ROOT="${DATA_ROOT:-./data}"
else
  DATA_ROOT="${DATA_ROOT:-}"
fi

RADII_DIR="${BASE_DIR}/radii_cache_md_${DATASET}_K${FPS_STEPS}"
NAME_PREFIX="fps_radii_${MODEL_TYPE}_K${FPS_STEPS}_${DATASET}"

EXTRA=()
[[ "$PRESET" != "none" ]] && EXTRA+=(--preset "$PRESET")
[[ -n "$DATA_ROOT" ]] && EXTRA+=(--data_root "$DATA_ROOT")

if [ -n "${SHAPENET_ROOT:-}" ]; then
  EXTRA+=(--shapenet_root "$SHAPENET_ROOT")
fi
if [ -n "${SCAN_TRAIN_H5:-}" ]; then
  EXTRA+=(--scanobjectnn_train_h5 "$SCAN_TRAIN_H5")
fi
if [ -n "${SCAN_TEST_H5:-}" ]; then
  EXTRA+=(--scanobjectnn_test_h5 "$SCAN_TEST_H5")
fi
if [ "${SHAPENET_NORMALIZE:-0}" = "1" ]; then
  EXTRA+=(--shapenet_normalize)
fi
if [ "${KITTI_USE_ROI:-0}" = "1" ]; then
  EXTRA+=(--kitti_use_roi)
fi
if [ -n "${MAX_LIST_FILES:-}" ] && [ "${MAX_LIST_FILES}" != "0" ]; then
  EXTRA+=(--max_list_files "$MAX_LIST_FILES")
fi
if [ -n "${MAX_TRAIN_FILES:-}" ] && [ "${MAX_TRAIN_FILES}" != "0" ]; then
  EXTRA+=(--max_train_files "$MAX_TRAIN_FILES")
fi
if [ -n "${MAX_TEST_FILES:-}" ] && [ "${MAX_TEST_FILES}" != "0" ]; then
  EXTRA+=(--max_test_files "$MAX_TEST_FILES")
fi
if [ -n "${OUTDOOR_TRAIN_RATIO:-}" ]; then
  EXTRA+=(--outdoor_train_ratio "$OUTDOOR_TRAIN_RATIO")
fi
if [ -n "${NUM_POINTS:-}" ] && [ "${NUM_POINTS}" != "0" ]; then
  EXTRA+=(--num_points "$NUM_POINTS")
fi
if [ -n "${EPOCHS:-}" ] && [ "${EPOCHS}" != "0" ]; then
  EXTRA+=(--epochs "$EPOCHS")
fi
if [ -n "${DEVICE:-}" ]; then
  EXTRA+=(--device "$DEVICE")
fi
if [ -n "${BATCH_SIZE_TRAIN:-}" ] && [ "${BATCH_SIZE_TRAIN}" != "0" ]; then
  EXTRA+=(--batch_size_train "$BATCH_SIZE_TRAIN")
fi
if [ -n "${BATCH_SIZE_FPS:-}" ] && [ "${BATCH_SIZE_FPS}" != "0" ]; then
  EXTRA+=(--batch_size_fps "$BATCH_SIZE_FPS")
fi

echo "========== multidataset FPS radii sweep =========="
echo "  DATASET=$DATASET  PRESET=$PRESET"
echo "  BASE_DIR=$BASE_DIR  FPS_STEPS=$FPS_STEPS  MODEL_TYPE=$MODEL_TYPE"
echo "  PREFIX_RATIOS=$PREFIX_RATIOS"
echo "  DATA_ROOT=${DATA_ROOT:-<auto via preset>}"
echo "  RADII_DIR=$RADII_DIR (共用)"
echo "===================================================="

mkdir -p "$BASE_DIR" "$RADII_DIR"

for pr in $PREFIX_RATIOS; do
  SAVE_DIR="${BASE_DIR}/${NAME_PREFIX}_pr${pr}"
  echo ""
  echo "[Train] dataset=$DATASET prefix_ratio=$pr -> $SAVE_DIR"
  python spec_kan_fps_multidataset.py \
    --dataset "$DATASET" \
    "${EXTRA[@]}" \
    --save_dir "$SAVE_DIR" \
    --fps_steps "$FPS_STEPS" \
    --prefix_ratio "$pr" \
    --model_type "$MODEL_TYPE" \
    --radii_dir "$RADII_DIR"
done

SUMMARY_JSON="${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}_${DATASET}_K${FPS_STEPS}.json"
SUMMARY_PNG="${BASE_DIR}/prefix_ratio_summary_${MODEL_TYPE}_${DATASET}_K${FPS_STEPS}.png"
echo ""
echo "[Collect] -> $SUMMARY_JSON , $SUMMARY_PNG"
python collect_prefix_ratio_results.py \
  --base_dir "$BASE_DIR" \
  --pattern "${NAME_PREFIX}_pr*" \
  --out_json "$SUMMARY_JSON" \
  --out_png "$SUMMARY_PNG"

echo "Done."
