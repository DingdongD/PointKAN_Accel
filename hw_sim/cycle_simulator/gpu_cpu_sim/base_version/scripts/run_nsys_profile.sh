#!/usr/bin/env bash
set -euo pipefail

# 获取工程根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 必须由外部传入待执行命令
CMD="${CMD:?Please set CMD='python infer.py ...'}"

# 输出目录
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/nsys}"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# 使用 Nsight Systems 采样
# --trace=cuda,nvtx,osrt : 追踪 CUDA / NVTX / OS runtime
# --sample=none          : 不做 CPU sampling，仅做 trace
# --force-overwrite=true : 覆盖旧文件
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$OUT_DIR/nsys_${TS}" \
  bash -lc "$CMD"

echo "nsys report prefix: $OUT_DIR/nsys_${TS}"