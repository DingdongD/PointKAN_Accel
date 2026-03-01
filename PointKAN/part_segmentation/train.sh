#!/usr/bin/env bash
# PointKAN Part Segmentation 训练脚本（支持断点重训 + 早停）
# 使用前请在本目录下执行：chmod +x train.sh
#
# 示例：
#   从头训练：              ./train.sh
#   断点重训（目录）：       RESUME_PATH=checkpoints/PointKAN_part_exp ./train.sh
#   断点重训（.pth 文件）：  RESUME_PATH=checkpoints/PointKAN_part_exp/last_checkpoint.pth ./train.sh
#   不早停（跑满 epochs）：  PATIENCE=0 ./train.sh

set -e
cd "$(dirname "$0")"

# ---------- 可调参数 ----------
EXP_NAME="${EXP_NAME:-part_exp}"
RESUME_PATH="${RESUME_PATH:-}"
PATIENCE="${PATIENCE:-20}"

# ---------- 固定参数（可按需修改） ----------
BATCH_SIZE=18
TEST_BATCH_SIZE=18
EPOCHS=350
LR=0.002
WORKERS=12
SCHEDULER=cos
USE_SGD=false

# ---------- 执行 ----------
if [[ -n "$RESUME_PATH" ]]; then
  echo ">>> 断点重训: --resume_path $RESUME_PATH (patience=${PATIENCE})"
  python main.py \
    --model pointKAN \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler "$SCHEDULER" \
    --workers "$WORKERS" \
    --resume_path "$RESUME_PATH" \
    --patience "$PATIENCE"
else
  echo ">>> 从头训练 (checkpoints 将保存在 checkpoints/pointKAN_${EXP_NAME}, 早停 patience=${PATIENCE})"
  python main.py \
    --model pointKAN \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler "$SCHEDULER" \
    --workers "$WORKERS" \
    --patience "$PATIENCE"
fi
