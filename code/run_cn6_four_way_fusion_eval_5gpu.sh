#!/usr/bin/env bash

set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

# Evaluation is light enough to use two processes on the single RTX 5090.
GPU_IDS="${GPU_IDS:-0 0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_EPISODES="${EVAL_EPISODES:-500}"
EVAL_SEED="${EVAL_SEED:-20270001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cn6_four_way_fusion_eval_500ep}"
LOG_ROOT="${LOG_ROOT:-logs/cn6_four_way_fusion_eval_500ep}"

ALL_PATTERN="${ALL_PATTERN:-outputs/db_match_cn6_all_100k_seed%s/checkpoint_final.pt}"
NOFUSION_ALL_PATTERN="${NOFUSION_ALL_PATTERN:-outputs/db_match_cn6_nofusion_100k_seed%s/checkpoint_final.pt}"
INVISIBLE_PATTERN="${INVISIBLE_PATTERN:-outputs/db_match_cn6_invisible_100k_seed%s/checkpoint_final.pt}"
NOFUSION_INVISIBLE_PATTERN="${NOFUSION_INVISIBLE_PATTERN:-outputs/db_match_cn6_nofusion_invisible_100k_seed%s/checkpoint_final.pt}"

METHODS=(db_all nofusion_all db_invisible nofusion_invisible)
PATTERNS=(
    "$ALL_PATTERN"
    "$NOFUSION_ALL_PATTERN"
    "$INVISIBLE_PATTERN"
    "$NOFUSION_INVISIBLE_PATTERN"
)

if [ ! -f "evaluate_match_cn6.py" ]; then
    echo "当前目录不是 MATCH-Reproduction 项目根目录：缺少 evaluate_match_cn6.py"
    exit 1
fi

read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_IDS 至少要包含一张 GPU。"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

TASKS=()
for method_index in "${!METHODS[@]}"; do
    for seed in 0 1 2 3 4; do
        TASKS+=("${method_index}|${seed}")
    done
done

run_job() {
    local gpu="$1"
    local task="$2"
    local method_index seed method pattern checkpoint output log_path

    IFS='|' read -r method_index seed <<< "$task"
    method="${METHODS[$method_index]}"
    pattern="${PATTERNS[$method_index]}"
    printf -v checkpoint "$pattern" "$seed"
    output="$OUTPUT_ROOT/${method}_seed${seed}.json"
    log_path="$LOG_ROOT/${method}_seed${seed}.log"

    if [ -s "$output" ]; then
        echo "[$(date '+%F %T')] 跳过已有评估：${method}_seed${seed}"
        return 0
    fi

    if [ ! -f "$checkpoint" ]; then
        echo "[$(date '+%F %T')] 缺少 checkpoint：$checkpoint" | tee "$log_path"
        return 1
    fi

    echo "[$(date '+%F %T')] 评估：${method}_seed${seed}，GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u evaluate_match_cn6.py \
        --checkpoint "$checkpoint" \
        --episodes "$EVAL_EPISODES" \
        --seed "$EVAL_SEED" \
        --packet-loss 0.3 \
        --output "$output" \
        > "$log_path" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date '+%F %T')] 完成：${method}_seed${seed}"
    else
        echo "[$(date '+%F %T')] 失败：${method}_seed${seed}，日志=$log_path"
    fi
    return "$exit_code"
}

worker() {
    local slot="$1"
    local gpu="$2"
    local failed=0
    local index
    for ((index=slot; index<${#TASKS[@]}; index+=${#GPUS[@]})); do
        if ! run_job "$gpu" "${TASKS[$index]}"; then
            failed=$((failed + 1))
        fi
    done
    return "$failed"
}

echo "============================================================"
echo "CN6 four-way fusion evaluation"
echo "20 checkpoints × $EVAL_EPISODES episodes"
echo "所有方法共用测试 seed 起点：$EVAL_SEED"
echo "使用 GPU：${GPUS[*]}"
echo "============================================================"

PIDS=()
for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    worker "$slot" "${GPUS[$slot]}" &
    PIDS+=("$!")
done

FAILED_WORKERS=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED_WORKERS=$((FAILED_WORKERS + 1))
    fi
done

if [ "$FAILED_WORKERS" -eq 0 ]; then
    "$PYTHON_BIN" summarize_cn6_four_way_fusion.py --input-dir "$OUTPUT_ROOT"
fi

echo "异常 worker 数量：$FAILED_WORKERS"
exit "$FAILED_WORKERS"

