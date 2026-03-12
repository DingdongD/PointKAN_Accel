#!/usr/bin/env bash
# 新服务器上一键安装 mini_demo 运行环境（需已安装 Python 3.8+ 与 CUDA/驱动，若跑 GPU）
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMO_ROOT"

echo ">>> 1. 安装 PyTorch（若未安装，请先安装与 CUDA 版本匹配的 torch）"
if ! python3 -c "import torch" 2>/dev/null; then
  echo "    未检测到 PyTorch，请先安装，例如："
  echo "    pip install torch   # 或访问 https://pytorch.org 选择 CUDA 版本"
  exit 1
fi
python3 -c "import torch; print('    PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo ">>> 2. 安装 requirements.txt"
pip install -r requirements.txt

echo ">>> 3. 安装 pointnet2_ops（本目录 pointnet2_ops_lib，含 CPU FPS）"
if [[ -d "$DEMO_ROOT/pointnet2_ops_lib" ]]; then
  pip install -e "$DEMO_ROOT/pointnet2_ops_lib"
else
  echo "    未找到 pointnet2_ops_lib 目录，请确保已拷贝到 mini_demo 下或另行安装 pointnet2_ops"
fi

echo ">>> 安装完成。运行: ./run.sh"
