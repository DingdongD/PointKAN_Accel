#!/usr/bin/env bash
# PointKAN-Elite ModelNet40 推理脚本（覆盖多种推理场景）
# 使用前请在本目录下执行：chmod +x inference.sh
#
# 模式说明：
#   eval     - 完整推理 + KAN 因果曲线分析 + 可视化（默认）
#   no_vis   - 仅推理验证，不生成可视化
#   fps_opt  - 在常规推理外再跑 FPS 优化版（预计算级联 FPS 索引），汇报精度与耗时对比
#   lut      - Vanilla GRF-KAN vs LUT 查表推理对比（需先生成 LUT）
#   quant    - 不同量化精度 (2/4/6/8 bit) 下 LUT 推理性能分析
#
# 示例：
#   默认（完整推理+可视化）：     ./inference.sh
#   仅推理不画图：                MODE=no_vis ./inference.sh
#   FPS 优化版对比：             MODE=fps_opt ./inference.sh
#   指定检查点：                  CHECKPOINT=checkpoints/pointKANElite-xxx-42 ./inference.sh
#   LUT 对比：                   MODE=lut LUT_DIR=./grf_lut ./inference.sh
#   量化分析：                   MODE=quant LUT_DIR=./grf_lut ./inference.sh
#   校准集用 test、500 样本：    MODE=eval CALIBRATION_PARTITION=test CALIBRATION_SAMPLES=500 ./inference.sh

set -e
cd "$(dirname "$0")"

# ---------- 推理模式 ----------
# eval | no_vis | fps_opt | lut | quant
MODE="${MODE:-eval}"

# ---------- 通用参数 ----------
CHECKPOINT="${CHECKPOINT:-checkpoints/pointKANElite-pointkan_elite_exp-42}"
NUM_POINTS="${NUM_POINTS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

# ---------- inference_and_kan_analysis 专用 ----------
# 校准集：train 或 test
CALIBRATION_PARTITION="${CALIBRATION_PARTITION:-train}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-500}"
# 可视化结果与 JSON 输出目录（默认在检查点同目录下 kan_analysis）
OUT_DIR="${OUT_DIR:-}"
JSON_OUTPUT="${JSON_OUTPUT:-}"
EXPORT_GRF_JSON="${EXPORT_GRF_JSON:-}"

# ---------- LUT 相关（mode=lut 或 quant 时使用） ----------
LUT_DIR="${LUT_DIR:-./grf_lut}"
LUT_MODE="${LUT_MODE:-vanilla}"   # vanilla | base_offset
USE_QUANTIZED="${USE_QUANTIZED:-}"  # 非空时对 lut 模式传 --use_quantized

# ---------- 量化分析专用 ----------
QUANT_OUT_DIR="${QUANT_OUT_DIR:-./grf_lut_quantization_analysis}"
QUANT_BITS="${QUANT_BITS:-2 4 6 8}"
N_BINS="${N_BINS:-256}"

# ---------- 额外参数（直接追加到 Python 命令末尾） ----------
EXTRA_ARGS=("${@}")

# 支持 --help 或 MODE=help 打印用法
if [[ "$MODE" == "help" ]] || [[ " ${EXTRA_ARGS[*]} " == *" --help "* ]]; then
    echo "用法: ./inference.sh [MODE] 或通过环境变量覆盖默认值"
    echo ""
    echo "  MODE (默认 eval):  eval | no_vis | fps_opt | lut | quant"
    echo "  CHECKPOINT:        检查点目录或 best_checkpoint.pth 路径"
    echo "  NUM_POINTS:        点云点数 (默认 1024)"
    echo "  BATCH_SIZE:        批大小 (默认 32)"
    echo "  DEVICE:            cuda | cpu"
    echo "  CALIBRATION_*:      仅 eval/no_vis: partition (train|test), samples 数"
    echo "  LUT_DIR:           lut/quant 模式下的 LUT 目录 (默认 ./grf_lut)"
    echo "  LUT_MODE:          vanilla | base_offset"
    echo "  USE_QUANTIZED:     非空时 lut 模式使用量化表"
    echo "  QUANT_OUT_DIR:     quant 模式输出目录"
    echo "  QUANT_BITS:        空格分隔的比特数 (默认 \"2 4 6 8\")"
    exit 0
fi

_run_inference_kan() {
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
    [[ "$2" == "fps_opt" ]] && args+=(--eval_fps_opt)
    [[ -n "$OUT_DIR" ]] && args+=(--out_dir "$OUT_DIR")
    [[ -n "$JSON_OUTPUT" ]] && args+=(--json_output "$JSON_OUTPUT")
    [[ -n "$EXPORT_GRF_JSON" ]] && args+=(--export_grf_json "$EXPORT_GRF_JSON")

    echo ">>> 运行: python inference_and_kan_analysis.py ${args[*]} ${EXTRA_ARGS[*]}"
    python inference_and_kan_analysis.py "${args[@]}" "${EXTRA_ARGS[@]}"
}

_run_lut_compare() {
    local args=(
        --checkpoint "$CHECKPOINT"
        --lut_dir "$LUT_DIR"
        --lut_mode "$LUT_MODE"
        --num_points "$NUM_POINTS"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --device "$DEVICE"
    )
    [[ -n "$USE_QUANTIZED" ]] && args+=(--use_quantized)

    echo ">>> 运行: python run_inference_lut_compare.py ${args[*]} ${EXTRA_ARGS[*]}"
    python run_inference_lut_compare.py "${args[@]}" "${EXTRA_ARGS[@]}"
}

_run_quant_analysis() {
    local args=(
        --checkpoint "$CHECKPOINT"
        --lut_dir "$LUT_DIR"
        --out_dir "$QUANT_OUT_DIR"
        --bits $QUANT_BITS
        --n_bins "$N_BINS"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --device "$DEVICE"
    )

    echo ">>> 运行: python run_grf_lut_quantization_analysis.py ${args[*]} ${EXTRA_ARGS[*]}"
    python run_grf_lut_quantization_analysis.py "${args[@]}" "${EXTRA_ARGS[@]}"
}

# ---------- 执行 ----------
case "$MODE" in
    eval)
        _run_inference_kan "" ""
        ;;
    no_vis)
        _run_inference_kan "no_vis" ""
        ;;
    fps_opt)
        _run_inference_kan "" "fps_opt"
        ;;
    lut)
        _run_lut_compare
        ;;
    quant)
        _run_quant_analysis
        ;;
    *)
        echo "错误: 未知模式 MODE=$MODE"
        echo "可选: eval | no_vis | fps_opt | lut | quant"
        exit 1
        ;;
esac
