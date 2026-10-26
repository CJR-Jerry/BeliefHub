#!/usr/bin/env bash
set -o pipefail

EPISODES="${EPISODES:-500}"
EVAL_SEED="${EVAL_SEED:-20270001}"
GPU_IDS_STR="${GPU_IDS_STR:-0 1 2 3 4 5 6}"
RESULT_ROOT="${RESULT_ROOT:-outputs/cn6_missing_information_sweep_500ep}"
LOG_ROOT="${LOG_ROOT:-logs/cn6_dense_grid_500ep}"
STATUS_DIR="${LOG_ROOT}/status"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
SIGHTS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2)
SIGHT_CODES=(02 03 04 05 06 07 08 09 10 11 12)
LOSSES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
LOSS_CODES=(00 10 20 30 40 50 60 70 80 90 99)
METHODS=(match db_all db_invisible)
SEEDS=(0 1 2 3 4)
TOTAL_TASKS=1815

if [[ "${#GPU_IDS[@]}" -lt 1 || "${#GPU_IDS[@]}" -gt 7 ]]; then
    echo "ERROR: provide one to seven GPU IDs."
    exit 1
fi
[[ -f evaluate_cn6_dense_grid_checkpoint.py ]] || {
    echo "ERROR: run from the MATCH-Reproduction root."
    exit 1
}

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${STATUS_DIR}"
exec 9>"${STATUS_DIR}/launcher.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "ERROR: another dense-grid launcher is active."
    exit 1
fi

checkpoint_for() {
    case "$1" in
        match) printf 'outputs/match_cn6_100k_seed%s/checkpoint_final.pt' "$2" ;;
        db_all) printf 'outputs/db_match_cn6_all_100k_seed%s/checkpoint_final.pt' "$2" ;;
        db_invisible) printf 'outputs/db_match_cn6_invisible_100k_seed%s/checkpoint_final.pt' "$2" ;;
        *) return 1 ;;
    esac
}

missing=0
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        checkpoint="$(checkpoint_for "${method}" "${seed}")"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "MISSING: ${checkpoint}"
            missing=$((missing + 1))
        fi
    done
done
[[ "${missing}" -eq 0 ]] || exit 1

summary_complete() {
    python - "$1" "$2" "$3" "$4" "$5" "${EPISODES}" "${EVAL_SEED}" <<'PY'
import json, math, sys
path, method, seed, sight, loss, episodes, eval_seed = sys.argv[1:]
try:
    d = json.load(open(path, encoding="utf-8"))
    ok = (
        d.get("status") == "completed"
        and d.get("method") == method
        and int(d.get("training_seed", -1)) == int(seed)
        and int(d.get("checkpoint_episode", -1)) == 100000
        and int(d.get("evaluation_episodes", -1)) == int(episodes)
        and int(d.get("evaluation_seed", -1)) == int(eval_seed)
        and math.isclose(float(d.get("sight_range")), float(sight))
        and math.isclose(float(d.get("packet_loss")), float(loss))
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

run_task() {
    local sight="$1" loss="$2" condition="$3" method="$4" seed="$5" gpu="$6"
    local name="${condition}_${method}_seed${seed}"
    local out="${RESULT_ROOT}/${condition}/${method}_seed${seed}"
    local summary="${out}/evaluation_summary.json"
    local checkpoint
    checkpoint="$(checkpoint_for "${method}" "${seed}")"

    if [[ -f "${summary}" ]] && summary_complete \
        "${summary}" "${method}" "${seed}" "${sight}" "${loss}"; then
        : > "${STATUS_DIR}/${name}.done"
        rm -f "${STATUS_DIR}/${name}.running" "${STATUS_DIR}/${name}.failed"
        echo "SKIP gpu=${gpu} ${name}"
        return 0
    fi

    rm -f "${STATUS_DIR}/${name}.done" "${STATUS_DIR}/${name}.failed"
    : > "${STATUS_DIR}/${name}.running"
    mkdir -p "${out}"
    echo "START gpu=${gpu} ${name} sight=${sight} loss=${loss}"

    if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PWD}" \
        python -u evaluate_cn6_dense_grid_checkpoint.py \
            --method "${method}" \
            --checkpoint "${checkpoint}" \
            --episodes "${EPISODES}" \
            --seed "${EVAL_SEED}" \
            --packet-loss "${loss}" \
            --sight-range "${sight}" \
            --expected-checkpoint-episode 100000 \
            --output-dir "${out}" \
            > "${LOG_ROOT}/${name}.log" 2>&1
    then
        mv "${STATUS_DIR}/${name}.running" "${STATUS_DIR}/${name}.done"
        echo "DONE  gpu=${gpu} ${name}"
    else
        code=$?
        mv "${STATUS_DIR}/${name}.running" "${STATUS_DIR}/${name}.failed"
        echo "FAILED gpu=${gpu} ${name} exit=${code}"
        tail -n 20 "${LOG_ROOT}/${name}.log" 2>/dev/null || true
        return "${code}"
    fi
}

TASKS=()
for si in "${!SIGHTS[@]}"; do
    for li in "${!LOSSES[@]}"; do
        condition="sr${SIGHT_CODES[$si]}_pl${LOSS_CODES[$li]}"
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                TASKS+=("${SIGHTS[$si]}|${LOSSES[$li]}|${condition}|${method}|${seed}")
            done
        done
    done
done

echo "CN6 dense 2D sight/loss grid"
echo "Grid: 11 x 11; tasks: ${TOTAL_TASKS}; episodes: $((TOTAL_TASKS * EPISODES))"
echo "GPUs: ${GPU_IDS[*]}"

worker_pids=()
gpu_count="${#GPU_IDS[@]}"
for rank in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$rank]}"
    (
        for index in "${!TASKS[@]}"; do
            (( index % gpu_count == rank )) || continue
            IFS='|' read -r sight loss condition method seed <<< "${TASKS[$index]}"
            run_task "${sight}" "${loss}" "${condition}" "${method}" "${seed}" "${gpu}" || true
        done
    ) &
    worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "${pid}" || true; done

completed="$(find "${STATUS_DIR}" -maxdepth 1 -name '*.done' | wc -l)"
failed="$(find "${STATUS_DIR}" -maxdepth 1 -name '*.failed' | wc -l)"
echo "Completed: ${completed}/${TOTAL_TASKS}"
echo "Failed:    ${failed}/${TOTAL_TASKS}"
[[ "${completed}" -eq "${TOTAL_TASKS}" && "${failed}" -eq 0 ]]
