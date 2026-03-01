#!/bin/bash
# FPS Radii：训练（可断点续训）-> 评估 -> 可视化；支持 MLP 或 KAN (Efficient_KAN)
# 用法:
#   ./run_fps_radii.sh                                    # 默认 mlp, save_dir=./runs/fps_radii_mlp, K=512
#   ./run_fps_radii.sh ./runs/fps_radii_kan 512 kan       # 使用 KAN 训练
#   ./run_fps_radii.sh ./runs/exp1 512 mlp resume          # 断点续训

set -e
cd "$(dirname "$0")"

SAVE_DIR="${1:-./runs/fps_radii_mlp}"
FPS_STEPS="${2:-512}"
MODEL_TYPE="${3:-mlp}"   # mlp | kan
RESUME="${4:-}"          # 传 resume 则断点续训

echo "========== FPS Radii (model_type=$MODEL_TYPE) =========="
echo "  SAVE_DIR=$SAVE_DIR"
echo "  FPS_STEPS (radii 长度 K)=$FPS_STEPS"
echo "  MODEL_TYPE=$MODEL_TYPE"
echo "======================================================="

# 1) 训练：传入 resume 时从 last.pt 断点续训，否则从头训练
if [ "$RESUME" = "resume" ]; then
  echo "[Train] 断点续训 (--resume ${SAVE_DIR})"
  python spec_kan_fps.py --save_dir "$SAVE_DIR" --fps_steps "$FPS_STEPS" --model_type "$MODEL_TYPE" --resume "$SAVE_DIR"
else
  echo "[Train] 从头训练（若已有 radii_*_K${FPS_STEPS}.npy 会直接加载）"
  python spec_kan_fps.py --save_dir "$SAVE_DIR" --fps_steps "$FPS_STEPS" --model_type "$MODEL_TYPE"
fi

# 2) 评估测试集 + 可视化若干样本
echo "[Eval] 测试集 MSE 与样本可视化"
python eval_fps_radii.py --save_dir "$SAVE_DIR" --num_vis 6

echo "Done. 最佳权重: ${SAVE_DIR}/best.pt  可视化: ${SAVE_DIR}/vis/"
