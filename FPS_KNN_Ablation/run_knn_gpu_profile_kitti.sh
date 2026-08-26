#!/usr/bin/env bash
# KITTI FPS+KNN GPU 显存 & 计算开销分析
# 扫描 FPS 采样数 K = [1024, 2048, 4096, 8192]，固定原始大点云（num_points=50000）
# =====================================================================
# 用法：
#   ./run_knn_gpu_profile_kitti.sh                          # 默认参数
#   DATA_DIR=/path/to/velodyne ./run_knn_gpu_profile_kitti.sh
#   FPS_STEPS="1024 2048 4096 8192" ./run_knn_gpu_profile_kitti.sh
#   NUM_POINTS=40000 MAX_SAMPLES=50 ./run_knn_gpu_profile_kitti.sh
#   NVTX=1 ./run_knn_gpu_profile_kitti.sh                  # 开启 NVTX
#
# 配合 Nsight Systems 全程采集（开 NVTX 可在 timeline 中看各算子区间）：
#   NVTX=1 nsys profile \
#     --trace=cuda,nvtx \
#     --output=kitti_knn_profile \
#     bash run_knn_gpu_profile_kitti.sh
# =====================================================================

set -e
cd "$(dirname "$0")"

# 只暴露一张卡给脚本；改卡号：CUDA_VISIBLE_DEVICES=3 ./...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEVICE="${DEVICE:-cuda:0}"

DATA_DIR="${DATA_DIR:-/workspace/VoxelNet/KITTI_Mini/training/velodyne}"

# 固定每帧点数（模拟 KITTI 原始大点云）
NUM_POINTS="${NUM_POINTS:-50000}"

# 扫描的 FPS 采样数 K 列表（空格分隔）
FPS_STEPS="${FPS_STEPS:-1024 2048 4096 8192}"

K_NEIGHBORS="${K_NEIGHBORS:-32}"
MIN_CANDIDATES_RATIO="${MIN_CANDIDATES_RATIO:-1.2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"   # 每档帧数：20 帧快速测，0 = 全部
SEED="${SEED:-42}"
NVTX="${NVTX:-0}"
OUT_JSON="${OUT_JSON:-./knn_gpu_profile_kitti.json}"

EXTRA_ARGS=(
  --data_dir          "$DATA_DIR"
  --num_points        "$NUM_POINTS"
  --fps_steps_list    $FPS_STEPS          # 不加引号：展开为多个整数参数
  --k_neighbors       "$K_NEIGHBORS"
  --min_candidates_ratio "$MIN_CANDIDATES_RATIO"
  --batch_size        "$BATCH_SIZE"
  --max_samples       "$MAX_SAMPLES"
  --device            "$DEVICE"
  --seed              "$SEED"
  --out_json          "$OUT_JSON"
)
[[ "$NVTX" = "1" ]] && EXTRA_ARGS+=(--nvtx)

echo ">>> python run_knn_gpu_profile_kitti.py ${EXTRA_ARGS[*]}"
exec python run_knn_gpu_profile_kitti.py "${EXTRA_ARGS[@]}"
