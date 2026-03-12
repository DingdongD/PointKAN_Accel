#!/usr/bin/env bash
# 打包 mini_demo 为 zip，只保留运行推理所需文件，排除 .pt、runs/、outputs、build 等
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$DEMO_ROOT")"
ZIP_NAME="${1:-mini_demo.zip}"

cd "$PARENT"
rm -f "$ZIP_NAME"

# 排除：.pt/.pth 权重、.pkl 数据、runs/ 与 checkpoints/ 训练产物、outputs、__pycache__、build、.egg-info
zip -r "$ZIP_NAME" mini_demo \
  -x "*.pt" \
  -x "*.pth" \
  -x "*.pkl" \
  -x "*/checkpoints/*" \
  -x "*/checkpoints/*/*" \
  -x "*/checkpoints/*/*/*" \
  -x "*/checkpoints/*/*/*/*" \
  -x "*/checkpoints/*/*/*/*/*" \
  -x "*/runs/*" \
  -x "*/runs/*/*" \
  -x "*/runs/*/*/*" \
  -x "*/runs/*/*/*/*" \
  -x "mini_demo/outputs/*" \
  -x "mini_demo/outputs/*/*" \
  -x "mini_demo/outputs/*/*/*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x "mini_demo/pointnet2_ops_lib/build/*" \
  -x "mini_demo/pointnet2_ops_lib/build/*/*" \
  -x "mini_demo/pointnet2_ops_lib/build/*/*/*" \
  -x "mini_demo/pointnet2_ops_lib/*.egg-info/*"

echo ">>> 已生成: $PARENT/$ZIP_NAME"
ls -lh "$PARENT/$ZIP_NAME"
