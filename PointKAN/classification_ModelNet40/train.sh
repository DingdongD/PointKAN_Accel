#!/usr/bin/env bash
# PointKAN ModelNet40 训练脚本（支持断点重训）
# 使用前请在本目录下执行：chmod +x train.sh
#
# 示例：
#   从头训练：              ./train.sh
#   断点重训（目录）：       RESUME=checkpoints/pointKAN-pointkan_exp-42 ./train.sh
#   断点重训（.pth 文件）：  RESUME=checkpoints/pointKAN-pointkan_exp-42/last_checkpoint.pth ./train.sh

set -e
cd "$(dirname "$0")"

# ---------- 可调参数 ----------
# 从头训练时使用的实验标识，便于复现与断点恢复时找到目录
MSG="${MSG:-pointkan_exp}"
# 随机种子，与 MSG 一起决定 checkpoints 下的目录名
SEED="${SEED:-42}"
# 断点重训：设为检查点目录或 last_checkpoint.pth 的路径则从该处恢复，留空则从头训练
RESUME="${RESUME:-}"
# 早停耐心值：验证损失连续多少 epoch 不下降则停止，0 表示不早停（跑满 epoch）
PATIENCE="${PATIENCE:-20}"

# ---------- 固定参数（可按需修改） ----------
BATCH_SIZE=64
EPOCH=300
LR=0.1
MIN_LR=0.005
WEIGHT_DECAY=2e-4
NUM_POINTS=1024
WORKERS=8


# ---------- 执行 ----------
if [[ -n "$RESUME" ]]; then
  echo ">>> 断点重训: --resume $RESUME"
  python main.py \
    --msg "$MSG" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --epoch "$EPOCH" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_points "$NUM_POINTS" \
    --workers "$WORKERS" \
    --patience "$PATIENCE" \
    --resume "$RESUME"
else
  echo ">>> 从头训练 (checkpoints 将保存在 checkpoints/pointKAN-${MSG}-${SEED}, 早停 patience=${PATIENCE})"
  python main.py \
    --msg "$MSG" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --epoch "$EPOCH" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_points "$NUM_POINTS" \
    --workers "$WORKERS" \
    --patience "$PATIENCE"
fi
