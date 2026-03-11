#!/usr/bin/env bash
# PointKAN-Elite ScanObjectNN 推理脚本
# 使用前请在本目录下执行：chmod +x inference.sh
#
# 模式说明：
#   eval     - 完整推理 + KAN 因果曲线分析 + 可视化（默认）
#   no_vis   - 仅推理验证，不生成可视化
#   fps_opt  - 在常规推理外再跑 FPS 优化版，汇报精度与耗时对比
#
# 示例：
#   默认（完整推理+可视化）：     ./inference.sh
#   仅推理不画图：                MODE=no_vis ./inference.sh
#   FPS 优化版对比：             MODE=fps_opt ./inference.sh
#   指定检查点：                  CHECKPOINT=checkpoints/pointKANElite-xxx ./inference.sh

set -e
cd "$(dirname "$0")"

MODE="${MODE:-eval}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-ScanObjectNN}"
NUM_CLASSES="${NUM_CLASSES:-15}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
CALIBRATION_PARTITION="${CALIBRATION_PARTITION:-train}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-500}"
OUT_DIR="${OUT_DIR:-}"
JSON_OUTPUT="${JSON_OUTPUT:-}"
EXTRA_ARGS=("${@}")

if [[ "$MODE" == "help" ]] || [[ " ${EXTRA_ARGS[*]} " == *" --help "* ]]; then
    echo "用法: ./inference.sh [MODE] 或通过环境变量覆盖默认值"
    echo ""
    echo "  MODE:       eval | no_vis | fps_opt"
    echo "  CHECKPOINT: 检查点目录或 best_checkpoint.pth 路径"
    echo "  NUM_CLASSES: 类别数 (默认 15)"
    echo "  NUM_POINTS: 点云点数 (默认 1024)"
    echo "  BATCH_SIZE: 批大小 (默认 32)"
    echo "  DEVICE:     cuda | cpu"
    exit 0
fi

_run_inference_kan() {
    local args=(
        --checkpoint "$CHECKPOINT"
        --num_classes "$NUM_CLASSES"
        --num_points "$NUM_POINTS"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --device "$DEVICE"
        --seed "$SEED"
        --calibration_partition "$CALIBRATION_PARTITION"
        --calibration_samples "$CALIBRATION_SAMPLES"
    )
    [[ "$1" == "no_vis" ]] && args+=(--no_vis)
    [[ "$2" == "fps_opt" ]] && args+=(--eval_fps_opt)
    [[ -n "$OUT_DIR" ]] && args+=(--out_dir "$OUT_DIR")
    [[ -n "$JSON_OUTPUT" ]] && args+=(--json_output "$JSON_OUTPUT")
    echo ">>> 运行: python inference_and_kan_analysis.py ${args[*]} ${EXTRA_ARGS[*]}"
    python inference_and_kan_analysis.py "${args[@]}" "${EXTRA_ARGS[@]}"
}

case "$MODE" in
    eval)   _run_inference_kan "" "" ;;
    no_vis) _run_inference_kan "no_vis" "" ;;
    fps_opt) _run_inference_kan "" "fps_opt" ;;
    *)
        echo "错误: 未知模式 MODE=$MODE (可选: eval | no_vis | fps_opt)"
        exit 1
        ;;
esac
