#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NETWORKS_STR="${NETWORKS:-pointkan-elite}"
ACCELERATORS_STR="${ACCELERATORS:-acc}"
POINTS_STR="${POINTS:-32768}"
MAX_BATCHES_VAL="${MAX_BATCHES:-1}"
BATCH_SIZE_VAL="${BATCH_SIZE:-1}"
OUT_ROOT="${OUT_ROOT:-analysis_real_eval_point_sweep}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
SWEEP_MODE="${SWEEP_MODE:-fast}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
EXPORT_MEM_TXN_VAL="${EXPORT_MEM_TXN:-}"
EXPORT_MEM_CONTRACT_VAL="${EXPORT_MEM_CONTRACT:-}"

case "$SWEEP_MODE" in
  fast)
    # Keep final report/breakdown accurate while skipping the slowest event dumps.
    [ -z "$EXPORT_MEM_TXN_VAL" ] && EXPORT_MEM_TXN_VAL=0
    [ -z "$EXPORT_MEM_CONTRACT_VAL" ] && EXPORT_MEM_CONTRACT_VAL=0
    ;;
  full)
    # Preserve detailed transaction/contract artifacts for deep debugging.
    [ -z "$EXPORT_MEM_TXN_VAL" ] && EXPORT_MEM_TXN_VAL=1
    [ -z "$EXPORT_MEM_CONTRACT_VAL" ] && EXPORT_MEM_CONTRACT_VAL=0
    ;;
  *)
    echo "Unsupported SWEEP_MODE=$SWEEP_MODE (expected: fast or full)" >&2
    exit 1
    ;;
esac

if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be a positive integer, got: $PARALLEL_JOBS" >&2
  exit 1
fi

PYTHON_BIN="$(type -P python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(type -P python || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find a usable python/python3 executable in PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

read -r -a NETWORKS <<< "$NETWORKS_STR"
read -r -a ACCELERATORS <<< "$ACCELERATORS_STR"
read -r -a POINTS <<< "$POINTS_STR"
read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"

ALL_RUNS_JSON="$OUT_ROOT/all_runs.json"
ALL_RUNS_CSV="$OUT_ROOT/all_runs.csv"
FAILED_RUNS_TXT="$OUT_ROOT/failed_runs.txt"
STATUS_DIR="$OUT_ROOT/.parallel_status"
ROOT_BASHPID="${BASHPID:-}"
rm -rf "$STATUS_DIR"
mkdir -p "$STATUS_DIR"
: > "$FAILED_RUNS_TXT"

cleanup() {
  if [ "${BASHPID:-}" != "$ROOT_BASHPID" ]; then
    return 0
  fi
  rm -rf "$STATUS_DIR"
}
trap cleanup EXIT

echo "========== point sweep real eval matrix =========="
echo "  networks:     ${NETWORKS[*]}"
echo "  accelerators: ${ACCELERATORS[*]}"
echo "  points:       ${POINTS[*]}"
echo "  max_batches:  $MAX_BATCHES_VAL"
echo "  batch_size:   $BATCH_SIZE_VAL"
echo "  sweep_mode:   $SWEEP_MODE"
echo "  parallel_jobs: $PARALLEL_JOBS"
echo "  export_mem_txn: $EXPORT_MEM_TXN_VAL"
echo "  export_mem_contract: $EXPORT_MEM_CONTRACT_VAL"
echo "  out_root:     $OUT_ROOT"
echo "=================================================="

total=$(( ${#NETWORKS[@]} * ${#ACCELERATORS[@]} * ${#POINTS[@]} ))
idx=0
any_failed=0
running_pids=()

run_case() {
  local seq="$1"
  local total_cases="$2"
  local network="$3"
  local accelerator="$4"
  local points="$5"
  local run_dir="$OUT_ROOT/${network}/${accelerator}_p${points}"
  local run_log="$run_dir/run.log"
  local status_file="$STATUS_DIR/${seq}.status"

  mkdir -p "$run_dir"

  if [ "$PARALLEL_JOBS" -gt 1 ]; then
    {
      echo "[$seq/$total_cases] START network=$network accelerator=$accelerator points=$points"
      echo "  output_dir=$run_dir"
      echo "  log_file=$run_log"
    } >&2

    (
      set +e
      ACCELERATOR="$accelerator"       NETWORK="$network"       NUM_POINTS="$points"       MAX_BATCHES="$MAX_BATCHES_VAL"       BATCH_SIZE="$BATCH_SIZE_VAL"       EXPORT_MEM_TXN="$EXPORT_MEM_TXN_VAL"       EXPORT_MEM_CONTRACT="$EXPORT_MEM_CONTRACT_VAL"       SIM_RESULTS_DIR="$run_dir"       ./run_sim_with_real_eval.sh "${EXTRA_ARGS[@]}" > "$run_log" 2>&1
      rc=$?
      mkdir -p "$STATUS_DIR"
      if [ "$rc" -eq 0 ]; then
        printf 'OK	%s	%s	%s	%s	%s
' "$seq" "$network" "$accelerator" "$points" "$run_dir" > "$status_file"
      else
        printf 'FAIL	%s	%s	%s	%s	%s	%s
' "$seq" "$network" "$accelerator" "$points" "$run_dir" "$rc" > "$status_file"
        printf '[%s/%s] FAILED network=%s accelerator=%s points=%s rc=%s log=%s
' "$seq" "$total_cases" "$network" "$accelerator" "$points" "$rc" "$run_log" >> "$FAILED_RUNS_TXT"
      fi
      exit "$rc"
    ) &

    running_pids+=("$!")
  else
    echo
    echo "[$seq/$total_cases] network=$network accelerator=$accelerator points=$points"
    echo "  output_dir=$run_dir"

    set +e
    ACCELERATOR="$accelerator"     NETWORK="$network"     NUM_POINTS="$points"     MAX_BATCHES="$MAX_BATCHES_VAL"     BATCH_SIZE="$BATCH_SIZE_VAL"     EXPORT_MEM_TXN="$EXPORT_MEM_TXN_VAL"     EXPORT_MEM_CONTRACT="$EXPORT_MEM_CONTRACT_VAL"     SIM_RESULTS_DIR="$run_dir"     ./run_sim_with_real_eval.sh "${EXTRA_ARGS[@]}" | tee "$run_log"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      any_failed=1
      printf '[%s/%s] FAILED network=%s accelerator=%s points=%s rc=%s log=%s
' "$seq" "$total_cases" "$network" "$accelerator" "$points" "$rc" "$run_log" >> "$FAILED_RUNS_TXT"
    fi
  fi
}

reap_finished() {
  local pid
  local new_pids=()
  for pid in "${running_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      set +e
      wait "$pid"
      rc=$?
      set -e
      if [ "$rc" -ne 0 ]; then
        any_failed=1
      fi
    fi
  done
  running_pids=("${new_pids[@]}")
}

wait_for_slot() {
  while [ "${#running_pids[@]}" -ge "$PARALLEL_JOBS" ]; do
    sleep 1
    reap_finished
  done
}

wait_for_all() {
  while [ "${#running_pids[@]}" -gt 0 ]; do
    sleep 1
    reap_finished
  done
}

for network in "${NETWORKS[@]}"; do
  for accelerator in "${ACCELERATORS[@]}"; do
    for points in "${POINTS[@]}"; do
      idx=$((idx + 1))
      wait_for_slot
      run_case "$idx" "$total" "$network" "$accelerator" "$points"
    done
  done
done

if [ "$PARALLEL_JOBS" -gt 1 ]; then
  echo
  echo "Waiting for ${#running_pids[@]} running job(s) to finish..."
  wait_for_all
fi

"$PYTHON_BIN" "$SCRIPT_DIR/collect_point_sweep_results.py" "$OUT_ROOT"

echo
echo "Sweep complete."
echo "  JSON: $ALL_RUNS_JSON"
echo "  CSV:  $ALL_RUNS_CSV"
if [ -s "$FAILED_RUNS_TXT" ]; then
  echo "  Failed runs: $FAILED_RUNS_TXT"
  exit 1
fi
