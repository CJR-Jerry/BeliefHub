#!/usr/bin/env bash

# Keep pipefail, but avoid nounset because older Bash versions may treat an
# explicitly empty resume-argument array as unbound.
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

EPISODES="${EPISODES:-100000}"
# One AutoDL RTX 5090. Repeating GPU 0 creates two workers on that GPU.
GPU_IDS="${GPU_IDS:-0 0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_FILE="scripts/cn6_nofusion_invisible_tasks.tsv"
LOG_ROOT="logs/cn6_nofusion_invisible_100k"
PID_FILE="$LOG_ROOT/worker_pids.txt"

read -r -a GPUS <<< "$GPU_IDS"

if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_IDS 至少要包含一张 GPU。"
    exit 1
fi

if [ ! -f "$TASK_FILE" ]; then
    echo "缺少任务表：$TASK_FILE"
    exit 1
fi

if [ ! -f "train_match_cn6.py" ]; then
    echo "当前目录不是 MATCH-Reproduction 项目根目录：缺少 train_match_cn6.py"
    exit 1
fi

if ! "$PYTHON_BIN" train_match_cn6.py --help 2>&1 | grep -q -- "--disable-belief-fusion"; then
    echo "train_match_cn6.py 不支持 --disable-belief-fusion，请先安装原 CN6 ablation 补丁。"
    exit 1
fi

mkdir -p "$LOG_ROOT" outputs

if [ -f "$PID_FILE" ]; then
    while read -r _gpu pid; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            echo "检测到仍在运行的 worker PID=$pid，未重复启动。"
            exit 1
        fi
    done < "$PID_FILE"
fi

mapfile -t TASKS < <(
    awk 'NF && $1 !~ /^#/ {print $1 "|" $2 "|" $3}' "$TASK_FILE"
)

if [ "${#TASKS[@]}" -ne 5 ]; then
    echo "任务数量应为5，实际为${#TASKS[@]}。"
    exit 1
fi

COMMON_ARGS=(
    --episodes "$EPISODES"
    --batch-size 32
    --buffer-capacity 1000
    --warmup-episodes 32
    --target-update 100
    --eval-interval 100
    --eval-episodes 20
    --epsilon-decay 3000
    --packet-loss 0.3
    --sight-range 0.8
    --reconstruction-weight 0.1
    --learning-rate 5e-4
    --gamma 0.99
    --grad-clip 10.0
    --use-local-belief
    --belief-hidden-dim 32
    --belief-loss-weight 0.01
    --belief-supervision invisible
    --disable-belief-fusion
)

last_episode() {
    local csv_path="$1"
    if [ ! -s "$csv_path" ]; then
        echo 0
        return
    fi
    tail -n 1 "$csv_path" | cut -d, -f1 | sed 's/\r$//'
}

run_job() {
    local gpu="$1"
    local task="$2"
    local name output_dir seed current_episode log_path
    local -a resume_args=()

    IFS='|' read -r name output_dir seed <<< "$task"
    log_path="$LOG_ROOT/${name}.log"

    if [ -f "$output_dir/checkpoint_final.pt" ]; then
        echo "[$(date '+%F %T')] 跳过已完成任务：$name"
        return 0
    fi

    current_episode="$(last_episode "$output_dir/training_log.csv")"
    if ! [[ "$current_episode" =~ ^[0-9]+$ ]]; then
        echo "[$(date '+%F %T')] $name 的 CSV 末行 episode 无效。" | tee -a "$log_path"
        return 1
    fi

    if [ "$current_episode" -ge "$EPISODES" ]; then
        echo "[$(date '+%F %T')] $name 已到 $current_episode，但缺少 final checkpoint，请检查。" | tee -a "$log_path"
        return 1
    fi

    if [ "$current_episode" -gt 0 ]; then
        if [ ! -f "$output_dir/checkpoint_latest.pt" ]; then
            echo "[$(date '+%F %T')] $name 有 CSV 但无 latest checkpoint，拒绝覆盖。" | tee -a "$log_path"
            return 1
        fi
        resume_args=(--resume "$output_dir/checkpoint_latest.pt")
        echo "[$(date '+%F %T')] 续训：$name，GPU=$gpu，episode=$current_episode"
    else
        if [ -f "$output_dir/checkpoint_latest.pt" ]; then
            echo "[$(date '+%F %T')] $name 有 checkpoint 但无 CSV，拒绝覆盖。" | tee -a "$log_path"
            return 1
        fi
        mkdir -p "$output_dir"
        : > "$log_path"
        echo "[$(date '+%F %T')] 从头训练：$name，GPU=$gpu"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_match_cn6.py \
        "${COMMON_ARGS[@]}" \
        --seed "$seed" \
        --output-dir "$output_dir" \
        "${resume_args[@]}" \
        >> "$log_path" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date '+%F %T')] 完成：$name"
    else
        echo "[$(date '+%F %T')] 异常退出：$name，code=$exit_code，日志=$log_path"
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
echo "CN6 NoFusion-Invisible：5 seeds × 100K episodes"
echo "唯一消融：belief supervision=invisible，decision fusion=off"
echo "使用 GPU：${GPUS[*]}"
echo "开始时间：$(date '+%F %T')"
echo "============================================================"

: > "$PID_FILE"
PIDS=()

for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    worker "$slot" "${GPUS[$slot]}" &
    pid=$!
    PIDS+=("$pid")
    echo "${GPUS[$slot]} $pid" >> "$PID_FILE"
done

FAILED_WORKERS=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED_WORKERS=$((FAILED_WORKERS + 1))
    fi
done

: > "$PID_FILE"

echo "============================================================"
echo "调度结束：$(date '+%F %T')"
echo "异常 worker 数量：$FAILED_WORKERS"
echo "============================================================"

exit "$FAILED_WORKERS"
