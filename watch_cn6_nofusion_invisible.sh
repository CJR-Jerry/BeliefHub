#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

MODE="${1:-train}"

if [ "$MODE" = "train" ]; then
    TOTAL="${EPISODES:-100000}"
    TASK_FILE="scripts/cn6_nofusion_invisible_tasks.tsv"

    printf "CN6 NoFusion-Invisible 训练进度  %s\n" "$(date '+%F %T')"
    printf "%-34s %-15s %-12s %-12s %s\n" \
        "TASK" "EPISODE" "EVAL_POL" "BELIEF_RMSE" "STATUS"

    while read -r name output_dir _seed; do
        [ -z "${name:-}" ] && continue
        [[ "$name" == \#* ]] && continue

        csv_path="$output_dir/training_log.csv"
        episode=0
        eval_pol="-"
        belief_rmse="-"
        status="waiting"

        if [ -f "$output_dir/checkpoint_final.pt" ]; then
            status="completed"
        elif [ -s "$csv_path" ]; then
            read -r episode eval_pol belief_rmse < <(
                awk -F, '
                    NR == 1 {
                        for (i = 1; i <= NF; i++) {
                            if ($i == "episode") episode_col = i
                            if ($i == "evaluation_pol") pol_col = i
                            if ($i == "belief_rmse") rmse_col = i
                        }
                        next
                    }
                    {
                        episode = $episode_col
                        if ($pol_col != "") pol = $pol_col
                        if ($rmse_col != "" && $rmse_col != "nan") rmse = $rmse_col
                    }
                    END {
                        if (pol == "") pol = "-"
                        if (rmse == "") rmse = "-"
                        printf "%s %s %s\n", episode, pol, rmse
                    }
                ' "$csv_path"
            )
            episode="${episode%.*}"
            if pgrep -f -- "--output-dir $output_dir" >/dev/null; then
                status="running"
            else
                status="paused/queued"
            fi
        elif pgrep -f -- "--output-dir $output_dir" >/dev/null; then
            status="starting"
        fi

        printf "%-34s %7s/%-7s %-12s %-12s %s\n" \
            "$name" "$episode" "$TOTAL" "$eval_pol" "$belief_rmse" "$status"
    done < "$TASK_FILE"

elif [ "$MODE" = "eval" ]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cn6_four_way_fusion_eval_500ep}"
    METHODS=(db_all nofusion_all db_invisible nofusion_invisible)
    completed=0
    total=20

    printf "CN6 四版本统一评估进度  %s\n" "$(date '+%F %T')"
    printf "%-24s %s\n" "TASK" "STATUS"

    for method in "${METHODS[@]}"; do
        for seed in 0 1 2 3 4; do
            task="${method}_seed${seed}"
            if [ -s "$OUTPUT_ROOT/${task}.json" ]; then
                status="completed"
                completed=$((completed + 1))
            elif pgrep -f -- "${task}.json" >/dev/null; then
                status="running"
            else
                status="waiting/failed"
            fi
            printf "%-24s %s\n" "$task" "$status"
        done
    done
    echo
    echo "Total: $completed/$total completed"
else
    echo "用法：bash watch_cn6_nofusion_invisible.sh [train|eval]"
    exit 2
fi

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true

