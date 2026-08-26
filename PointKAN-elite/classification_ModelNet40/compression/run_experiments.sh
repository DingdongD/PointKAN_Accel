#!/usr/bin/env bash
# ModelNet40 区域熵算法级实验。数据路径与上层 data.py 默认一致，可用环境变量覆盖。
set -euo pipefail
cd "$(dirname "$0")"

export MODELNET40_H5_DIR="${MODELNET40_H5_DIR:-/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048}"
export MAX_SAMPLES="${MAX_SAMPLES:-200}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointkan

exec python run_region_entropy_experiments.py \
  --data_dir "$MODELNET40_H5_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --out_dir "${OUT_DIR:-$(pwd)/results}" \
  "$@"
