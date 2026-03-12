#!/usr/bin/env bash
# PointKAN-Elite ScanObjectNN 训练脚本（支持断点重训 + 早停）
# 使用前请在本目录下执行：chmod +x train.sh
#
# 示例：
#   从头训练：              ./train.sh
#   断点重训（目录）：       RESUME=checkpoints/PointKAN-scan_exp-42 ./train.sh
#   断点重训（.pth 文件）：  RESUME=checkpoints/PointKAN-scan_exp-42/last_checkpoint.pth ./train.sh
#   不早停：                PATIENCE=0 ./train.sh

set -e
cd "$(dirname "$0")"

MSG="${MSG:-scan_exp}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"
PATIENCE="${PATIENCE:-20}"

BATCH_SIZE=64
EPOCH=200
LR=0.01
WEIGHT_DECAY=1e-4
NUM_POINTS=1024
NUM_CLASSES=15
WORKERS=4

if [[ -n "$RESUME" ]]; then
  echo ">>> 断点重训: --resume $RESUME (patience=${PATIENCE})"
  python main.py \
    --model PointKAN \
    --msg "$MSG" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --epoch "$EPOCH" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_points "$NUM_POINTS" \
    --num_classes "$NUM_CLASSES" \
    --workers "$WORKERS" \
    --patience "$PATIENCE" \
    --resume "$RESUME"
else
  echo ">>> 从头训练 (checkpoints 将保存在 checkpoints/PointKANElite-${MSG}-${SEED}, 早停 patience=${PATIENCE})"
  python main.py \
    --model pointKANElite \
    --msg "$MSG" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --epoch "$EPOCH" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_points "$NUM_POINTS" \
    --num_classes "$NUM_CLASSES" \
    --workers "$WORKERS" \
    --patience "$PATIENCE"
fi
