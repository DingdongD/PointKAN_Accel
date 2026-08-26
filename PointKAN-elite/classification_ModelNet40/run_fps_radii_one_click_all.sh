#!/bin/bash
# 一键：按仓库内默认数据路径（与 FPS_KNN_Ablation/run_knn_coarse_fine_eval_*.sh、
# classification_ScanObjectNN/inference.sh 中 DATA_DIR/DATA_ROOT 一致）依次训练各数据集的
# 多组 prefix_ratio FPS radii 预测（KAN 或 MLP）。
#
# 用法（在 classification_ModelNet40 目录）:
#   chmod +x run_fps_radii_one_click_all.sh
#   ./run_fps_radii_one_click_all.sh
#
# 常用环境变量（可选）:
#   BASE_DIR=./runs/my_radii_sweep   输出根目录
#   MODEL_TYPE=kan                   kan | mlp
#   FPS_STEPS=512
#   PREFIX_RATIOS="0.05 0.10 0.15 0.20 0.30 0.40"
#   NUM_POINTS=1024                  每样本送入 FPS 的点数（室外为子采样）
#   MAX_LIST_FILES=200               仅处理前 N 个文件（调试用）
#   SKIP_DATASETS=kitti,waymo        逗号分隔，跳过这些数据集名
#   CUDA_VISIBLE_DEVICES=0
#   DEVICE=cuda:0
#
# 说明:
#   - KITTI / Waymo 默认开启 ROI（与 run_knn_coarse_fine_eval_kitti.sh / waymo.sh 默认一致，未设 NO_ROI）
#   - ModelNet40 若使用 data.py 内硬编码 h5 路径，与 --preset voxelnet 下的 pc_data 目录应对齐
#
set -eo pipefail
cd "$(dirname "$0")"

export PRESET="${PRESET:-voxelnet}"
export BASE_DIR="${BASE_DIR:-./runs/fps_radii_oneclick_all}"
export MODEL_TYPE="${MODEL_TYPE:-kan}"
export FPS_STEPS="${FPS_STEPS:-512}"
export PREFIX_RATIOS="${PREFIX_RATIOS:-0.05 0.10 0.15 0.20 0.30 0.40}"
export DEVICE="${DEVICE:-cuda:0}"
export NUM_POINTS="${NUM_POINTS:-1024}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-256}"
export BATCH_SIZE_FPS="${BATCH_SIZE_FPS:-16}"

# 与 run_knn_coarse_fine_eval_kitti.sh / waymo.sh 一致：默认使用 ROI（通过 KITTI_USE_ROI=1）
export KITTI_USE_ROI="${KITTI_USE_ROI:-1}"

_run_sweep() {
  local ds="$1"
  echo ""
  echo "###################################################################"
  echo "#  dataset=$ds  PRESET=$PRESET"
  echo "###################################################################"
  export DATA_ROOT=""
  # NuScenes 默认与 spec_kan_fps_multidataset 一致不加 ROI；可用 NUSCENES_USE_ROI=1 开启
  case "$ds" in
    nuscenes)
      export KITTI_USE_ROI="${NUSCENES_USE_ROI:-0}"
      ;;
    kitti|waymo)
      export KITTI_USE_ROI="${KITTI_USE_ROI:-1}"
      ;;
    *)
      unset KITTI_USE_ROI || true
      ;;
  esac

  local skipl
  skipl=",${SKIP_DATASETS:-},"
  if [[ "$skipl" == *",$ds,"* ]]; then
    echo "[skip] SKIP_DATASETS 包含 $ds"
    return 0
  fi

  ./run_fps_radii_prefix_sweep_multidataset.sh "$ds" "$BASE_DIR" "$FPS_STEPS" "$MODEL_TYPE" "$PREFIX_RATIOS"
}

echo "======== run_fps_radii_one_click_all ========"
echo "  PRESET=$PRESET  BASE_DIR=$BASE_DIR"
echo "  MODEL_TYPE=$MODEL_TYPE  FPS_STEPS=$FPS_STEPS"
echo "  PREFIX_RATIOS=$PREFIX_RATIOS"
echo "  NUM_POINTS=$NUM_POINTS  DEVICE=$DEVICE"
echo "============================================"

mkdir -p "$BASE_DIR"

# 顺序：物体类 -> 室外类
_run_sweep modelnet40
_run_sweep scanobjectnn
_run_sweep shapenet_part
_run_sweep kitti
_run_sweep nuscenes
_run_sweep s3dis
_run_sweep waymo

echo ""
echo "全部完成。汇总图与 JSON 位于: $BASE_DIR/prefix_ratio_summary_${MODEL_TYPE}_*_K${FPS_STEPS}.json"
echo "可使用: ls -1 $BASE_DIR/prefix_ratio_summary_${MODEL_TYPE}_*_K${FPS_STEPS}.png"
