#!/usr/bin/env bash
# 在已安装 PyTorch+CUDA 的环境下编译 PointNet++ batch CUDA 扩展（与 OpenPoints 同源）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/cpp/pointnet2_batch"
pip install -v .
