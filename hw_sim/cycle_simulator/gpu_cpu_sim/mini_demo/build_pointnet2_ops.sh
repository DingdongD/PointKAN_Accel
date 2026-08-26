#!/usr/bin/env bash
# 编译安装 mini_demo 自带的 pointnet2_ops（CUDA 扩展），供 PointKAN-elite 等使用。
# ISEDA 移植的 PointNet++（isda_models）使用 PyTorch FPS，不依赖本包。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$ROOT/pointnet2_ops_lib" ]]; then
  echo "未找到 $ROOT/pointnet2_ops_lib"
  exit 1
fi
pip install -e "$ROOT/pointnet2_ops_lib"
echo ">>> pointnet2_ops 已安装（可编辑模式）"
