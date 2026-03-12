#!/usr/bin/env bash
# PointKAN-Elite Part Segmentation 推理脚本
# 使用前请在本目录下执行：chmod +x inference.sh
#
# 模式说明：
#   eval   - 完整推理 + KAN 因果曲线分析 + 可视化（默认）
#   no_vis - 仅推理验证，不生成可视化
#
# 示例：
#   默认：                    ./inference.sh
#   仅推理不画图：            MODE=no_vis ./inference.sh
#   指定检查点：              CHECKPOINT=checkpoints/pointKANElite_part_exp ./inference.sh

set -e
cd "$(dirname "$0")"

MODE="${MODE:-eval}"
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite_part_exp}"
NUM_POINTS="${NUM_POINTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
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
    echo "  MODE:       eval | no_vis"
    echo "  CHECKPOINT: 检查点目录或 best_insiou_model.pth 路径"
    echo "  NUM_POINTS: 点云点数 (默认 2048)"
    echo "  BATCH_SIZE: 批大小 (默认 16)"
    echo "  DEVICE:     cuda | cpu"
    exit 0
fi

_run_inference() {
    local args=(
        --checkpoint "$CHECKPOINT"
        --num_points "$NUM_POINTS"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --device "$DEVICE"
        --seed "$SEED"
        --calibration_partition "$CALIBRATION_PARTITION"
        --calibration_samples "$CALIBRATION_SAMPLES"
    )
    [[ "$1" == "no_vis" ]] && args+=(--no_vis)
    [[ -n "$OUT_DIR" ]] && args+=(--out_dir "$OUT_DIR")
    [[ -n "$JSON_OUTPUT" ]] && args+=(--json_output "$JSON_OUTPUT")
    echo ">>> 运行: python inference_and_kan_analysis.py ${args[*]} ${EXTRA_ARGS[*]}"
    python inference_and_kan_analysis.py "${args[@]}" "${EXTRA_ARGS[@]}"
}

case "$MODE" in
    eval)   _run_inference "" ;;
    no_vis) _run_inference "no_vis" ;;
    *)
        echo "错误: 未知模式 MODE=$MODE (可选: eval | no_vis)"
        exit 1
        ;;
esac
