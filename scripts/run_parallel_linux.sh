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
DATASET_HZ="${DATASET_HZ:-}"
SHUTDOWN_ON_COMPLETE="${SHUTDOWN_ON_COMPLETE:-0}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-1}"

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

LOG_DIR="${DATASET_BASE}/.parallel_logs/${BATCH_ID}"
mkdir -p "${LOG_DIR}"

declare -a PIDS=()
declare -a LABELS=()
declare -a RESULT_PATHS=()
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

for (( worker = 0; worker < WORKERS; worker++ )); do
    gpu="${GPU_LIST[$((worker % ${#GPU_LIST[@]}))]}"
    worker_name="worker_$(printf '%02d' "${worker}")"
    log_path="${LOG_DIR}/${worker_name}.log"
    result_path="${LOG_DIR}/${worker_name}.result.json"

    cmd=(
        "${ISAAC_PYTHON}"
        "${PROJECT_ROOT}/run_vla_train_scene.py"
        --headless
        --auto-collect
        --no-bridge
        --success-count "${SUCCESS_COUNT}"
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

    echo "[${worker_name}] gpu=${gpu} log=${log_path}"
    EXCAVATOR_AUTO_RUN_ID_SUFFIX="${BATCH_ID}_${worker_name}" \
        EXCAVATOR_AUTO_COLLECT_RESULT_FILE="${result_path}" \
        PYTHONUNBUFFERED=1 \
        "${cmd[@]}" >"${log_path}" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${worker_name}")
    RESULT_PATHS+=("${result_path}")
done

failed=0
completed_workers=0
for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    label="${LABELS[$index]}"
    result_path="${RESULT_PATHS[$index]}"
    if wait "${pid}"; then
        if [[ -f "${result_path}" ]] && grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "${result_path}"; then
            echo "[${label}] target verified"
            completed_workers=$((completed_workers + 1))
        else
            echo "[${label}] exited without reaching its success target; result=${result_path}" >&2
            failed=1
        fi
    else
        status=$?
        echo "[${label}] failed with exit code ${status}" >&2
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
