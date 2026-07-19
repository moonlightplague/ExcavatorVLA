#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
DATASET_BASE="${DATASET_BASE:-${PROJECT_ROOT}/excavator_auto_dataset}"
BATCH_ID="${BATCH_ID:-batch_$(date +%Y%m%d_%H%M%S)}"

# GPU_IDS contains physical Isaac/Vulkan device indices. Workers cycle through
# this list when WORKERS is larger than the number of listed GPUs.
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
WORKERS="${WORKERS:-${#GPU_LIST[@]}}"

SUCCESS_COUNT="${SUCCESS_COUNT:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-300}"
LOG_MODE="${LOG_MODE:-data}"
FAST_SAMPLED_REPLAY="${FAST_SAMPLED_REPLAY:-0}"
ATTEMPT_MOTION_SPEED="${ATTEMPT_MOTION_SPEED:-}"
DATASET_HZ="${DATASET_HZ:-10}"
SHUTDOWN_ON_COMPLETE="${SHUTDOWN_ON_COMPLETE:-0}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-1}"
WORKER_RESTART_LIMIT="${WORKER_RESTART_LIMIT:-3}"
WORKER_RESTART_DELAY_SECONDS="${WORKER_RESTART_DELAY_SECONDS:-10}"
WORKER_START_STAGGER_SECONDS="${WORKER_START_STAGGER_SECONDS:-2}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Isaac Python is not executable: ${ISAAC_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/run_vla_train_scene.py" ]]; then
    echo "Project launcher not found: ${PROJECT_ROOT}/run_vla_train_scene.py" >&2
    exit 2
fi
if (( WORKERS < 1 )); then
    echo "WORKERS must be at least 1" >&2
    exit 2
fi
if (( ${#GPU_LIST[@]} < 1 )); then
    echo "GPU_IDS must contain at least one GPU index" >&2
    exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in GPU_IDS=${GPU_IDS}: ${gpu}" >&2
        exit 2
    fi
done
if [[ "${SHUTDOWN_ON_COMPLETE}" != "0" && "${SHUTDOWN_ON_COMPLETE}" != "1" ]]; then
    echo "SHUTDOWN_ON_COMPLETE must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${SHUTDOWN_DELAY_MINUTES}" =~ ^[1-9][0-9]*$ ]] || (( SHUTDOWN_DELAY_MINUTES > 60 )); then
    echo "SHUTDOWN_DELAY_MINUTES must be between 1 and 60" >&2
    exit 2
fi
for value_name in WORKER_RESTART_LIMIT WORKER_RESTART_DELAY_SECONDS WORKER_START_STAGGER_SECONDS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer" >&2
        exit 2
    fi
done

LOG_DIR="${DATASET_BASE}/.parallel_logs/${BATCH_ID}"
mkdir -p "${LOG_DIR}"

declare -a PIDS=()
declare -a LABELS=()
interrupted=0

stop_workers() {
    local pid
    interrupted=1
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap stop_workers INT TERM

echo "Starting ${WORKERS} Isaac Sim worker(s)"
echo "Project: ${PROJECT_ROOT}"
echo "Dataset: ${DATASET_BASE}"
echo "Batch:   ${BATCH_ID}"
echo "GPUs:    ${GPU_IDS}"
echo "Dashboard --root must be: ${DATASET_BASE}"
echo "Shutdown after verified completion: ${SHUTDOWN_ON_COMPLETE}"
echo "Worker restart limit: ${WORKER_RESTART_LIMIT}"

result_successes() {
    local result_path="$1"
    local value=""
    if [[ -f "${result_path}" ]]; then
        value="$(
            sed -nE 's/^[[:space:]]*"successes"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' \
                "${result_path}" | head -n 1
        )"
    fi
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s' "${value}"
    else
        printf '0'
    fi
}

result_exit_reason() {
    local result_path="$1"
    if [[ ! -f "${result_path}" ]]; then
        printf 'result_missing'
        return
    fi
    local value=""
    value="$(
        sed -nE 's/^[[:space:]]*"loop_exit_reason"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/p' \
            "${result_path}" | head -n 1
    )"
    printf '%s' "${value:-unknown}"
}

run_worker_supervisor() {
    local worker="$1"
    local gpu="$2"
    local worker_name="$3"
    local log_path="$4"
    local result_path="$5"
    local aggregate_successes=0
    local restart_index=0
    local child_pid=0
    local child_owns_process_group=0

    stop_supervised_processes() {
        local attempt
        if (( child_pid <= 0 )); then
            return
        fi
        if (( child_owns_process_group != 0 )); then
            kill -TERM -- "-${child_pid}" 2>/dev/null || true
            for attempt in $(seq 1 10); do
                if ! kill -0 -- "-${child_pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            kill -KILL -- "-${child_pid}" 2>/dev/null || true
        else
            pkill -TERM -P "${child_pid}" 2>/dev/null || true
            kill -TERM "${child_pid}" 2>/dev/null || true
        fi
        wait "${child_pid}" 2>/dev/null || true
        child_pid=0
        child_owns_process_group=0
    }

    stop_supervised_child() {
        stop_supervised_processes
        exit 143
    }
    trap stop_supervised_child INT TERM

    while (( aggregate_successes < SUCCESS_COUNT )); do
        local remaining=$((SUCCESS_COUNT - aggregate_successes))
        local run_suffix="${BATCH_ID}_${worker_name}"
        if (( restart_index > 0 )); then
            run_suffix="${run_suffix}_retry_$(printf '%02d' "${restart_index}")"
        fi
        local -a cmd=(
            "${ISAAC_PYTHON}"
            "${PROJECT_ROOT}/run_vla_train_scene.py"
            --headless
            --auto-collect
            --no-bridge
            --success-count "${remaining}"
            --max-attempts "${MAX_ATTEMPTS}"
            --dataset-root "${DATASET_BASE}"
            --log-mode "${LOG_MODE}"
            --graphics-api vulkan
            --active-gpu "${gpu}"
            --physics-gpu "${gpu}"
            --disable-export
            --no-wait-export
        )
        if [[ "${FAST_SAMPLED_REPLAY}" == "1" ]]; then
            cmd+=(--fast-sampled-replay)
        fi
        if [[ -n "${ATTEMPT_MOTION_SPEED}" ]]; then
            cmd+=(--attempt-motion-speed "${ATTEMPT_MOTION_SPEED}")
        fi
        if [[ -n "${DATASET_HZ}" ]]; then
            cmd+=(--dataset-hz "${DATASET_HZ}")
        fi

        rm -f -- "${result_path}"
        echo "[${worker_name}] launch=$((restart_index + 1)) gpu=${gpu} remaining=${remaining} log=${log_path}"
        {
            echo
            echo "===== launch $((restart_index + 1)) remaining=${remaining} suffix=${run_suffix} ====="
        } >>"${log_path}"
        if command -v setsid >/dev/null 2>&1; then
            EXCAVATOR_AUTO_RUN_ID_SUFFIX="${run_suffix}" \
                EXCAVATOR_AUTO_COLLECT_RESULT_FILE="${result_path}" \
                PYTHONUNBUFFERED=1 \
                setsid "${cmd[@]}" >>"${log_path}" 2>&1 &
            child_owns_process_group=1
        else
            EXCAVATOR_AUTO_RUN_ID_SUFFIX="${run_suffix}" \
                EXCAVATOR_AUTO_COLLECT_RESULT_FILE="${result_path}" \
                PYTHONUNBUFFERED=1 \
                "${cmd[@]}" >>"${log_path}" 2>&1 &
            child_owns_process_group=0
        fi
        child_pid=$!
        wait "${child_pid}"
        local status=$?
        child_pid=0
        child_owns_process_group=0

        local gained
        gained="$(result_successes "${result_path}")"
        aggregate_successes=$((aggregate_successes + gained))
        if (( aggregate_successes > SUCCESS_COUNT )); then
            aggregate_successes="${SUCCESS_COUNT}"
        fi
        local reason
        reason="$(result_exit_reason "${result_path}")"
        echo "[${worker_name}] exit=${status} reason=${reason} gained=${gained} aggregate=${aggregate_successes}/${SUCCESS_COUNT}"

        if (( aggregate_successes >= SUCCESS_COUNT )); then
            trap - INT TERM
            return 0
        fi
        if (( restart_index >= WORKER_RESTART_LIMIT )); then
            echo "[${worker_name}] restart limit reached before target" >&2
            trap - INT TERM
            return 1
        fi

        restart_index=$((restart_index + 1))
        echo "[${worker_name}] restarting in ${WORKER_RESTART_DELAY_SECONDS}s"
        sleep "${WORKER_RESTART_DELAY_SECONDS}"
    done

    trap - INT TERM
    return 0
}

for (( worker = 0; worker < WORKERS; worker++ )); do
    gpu="${GPU_LIST[$((worker % ${#GPU_LIST[@]}))]}"
    worker_name="worker_$(printf '%02d' "${worker}")"
    log_path="${LOG_DIR}/${worker_name}.log"
    result_path="${LOG_DIR}/${worker_name}.result.json"

    run_worker_supervisor \
        "${worker}" \
        "${gpu}" \
        "${worker_name}" \
        "${log_path}" \
        "${result_path}" &
    PIDS+=("$!")
    LABELS+=("${worker_name}")
    if (( WORKER_START_STAGGER_SECONDS > 0 && worker + 1 < WORKERS )); then
        sleep "${WORKER_START_STAGGER_SECONDS}"
    fi
done

failed=0
completed_workers=0
for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    label="${LABELS[$index]}"
    if wait "${pid}"; then
        echo "[${label}] aggregate target verified"
        completed_workers=$((completed_workers + 1))
    else
        status=$?
        echo "[${label}] supervisor failed with exit code ${status}" >&2
        failed=1
    fi
done

trap - INT TERM
echo "Batch complete: ${DATASET_BASE} (${BATCH_ID}); verified=${completed_workers}/${WORKERS}"
if [[ "${SHUTDOWN_ON_COMPLETE}" == "1" ]]; then
    if (( interrupted != 0 || failed != 0 || completed_workers != WORKERS )); then
        echo "Shutdown skipped: batch did not complete every worker target safely."
    elif ! command -v shutdown >/dev/null 2>&1; then
        echo "Shutdown requested but the shutdown command is unavailable." >&2
        failed=1
    else
        echo "All worker targets verified. Scheduling power-off in ${SHUTDOWN_DELAY_MINUTES} minute(s)."
        echo "Cancel before then with: shutdown -c"
        sync
        if ! shutdown -h "+${SHUTDOWN_DELAY_MINUTES}" "Excavator auto collect ${BATCH_ID} completed"; then
            echo "Failed to schedule system shutdown." >&2
            failed=1
        fi
    fi
fi
exit "${failed}"
