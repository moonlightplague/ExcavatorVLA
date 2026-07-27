#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
POLICY_PYTHON=/opt/conda/envs/smolvla/bin/python
EPISODE_RUNNER="$PROJECT/run_isaacsim_ckpt18450_seed2_success_rate_episode.sh"
SUMMARIZER="$PROJECT/scripts/evaluation/summarize_isaacsim_success_rate.py"
SIMULATOR="$PROJECT/run_simulation.py"
CLIENT="$PROJECT/scripts/bridge_test/smolvla_policy_client.py"

RESULT_ROOT=/root/gpufree-data/excavator_isaacsim_eval
LOG_ROOT=/root/gpufree-data/excavator_logs
NUM_EPISODES="${NUM_EPISODES:-200}"
POLICY_STEPS="${POLICY_STEPS:-350}"
MAX_ATTEMPTS_PER_EPISODE="${MAX_ATTEMPTS_PER_EPISODE:-3}"
GPU_COOLDOWN_SECONDS="${GPU_COOLDOWN_SECONDS:-5}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-60}"
TRUCK_SUCCESS_MIN_PARTICLES="${TRUCK_SUCCESS_MIN_PARTICLES:-1}"
TRUCK_SUCCESS_NO_GROWTH_STEPS="${TRUCK_SUCCESS_NO_GROWTH_STEPS:-${TRUCK_SUCCESS_DWELL_STEPS:-3}}"
BUCKET_OPEN_THRESHOLD="${BUCKET_OPEN_THRESHOLD:--0.80}"
RECORD_VIDEO="${RECORD_VIDEO:-0}"
DUMP_INPUTS="${DUMP_INPUTS:-0}"
RUN_TRACE_ANALYSIS="${RUN_TRACE_ANALYSIS:-0}"
RUNTIME_TEMP_ROOT="${RUNTIME_TEMP_ROOT:-/root/gpufree-data/tmp}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/root/gpufree-data/runtime_cache}"
MIN_ROOT_FREE_MIB="${MIN_ROOT_FREE_MIB:-1024}"

if (( NUM_EPISODES <= 0 )); then
  echo "[ERROR] NUM_EPISODES must be positive." >&2
  exit 1
fi
if (( POLICY_STEPS <= 0 )); then
  echo "[ERROR] POLICY_STEPS must be positive." >&2
  exit 1
fi
if (( MAX_ATTEMPTS_PER_EPISODE <= 0 )); then
  echo "[ERROR] MAX_ATTEMPTS_PER_EPISODE must be positive." >&2
  exit 1
fi
if (( GPU_COOLDOWN_SECONDS < 5 )); then
  echo "[ERROR] GPU_COOLDOWN_SECONDS must be at least 5." >&2
  exit 1
fi
if (( GPU_IDLE_TIMEOUT_SECONDS < GPU_COOLDOWN_SECONDS )); then
  echo "[ERROR] GPU_IDLE_TIMEOUT_SECONDS must be at least GPU_COOLDOWN_SECONDS." >&2
  exit 1
fi
if [[ "$RUNTIME_TEMP_ROOT" != /* ]]; then
  echo "[ERROR] RUNTIME_TEMP_ROOT must be an absolute path." >&2
  exit 1
fi
if [[ "$RUNTIME_CACHE_ROOT" != /* ]]; then
  echo "[ERROR] RUNTIME_CACHE_ROOT must be an absolute path." >&2
  exit 1
fi
if [[ ! "$MIN_ROOT_FREE_MIB" =~ ^[0-9]+$ ]] || \
  (( MIN_ROOT_FREE_MIB <= 0 )); then
  echo "[ERROR] MIN_ROOT_FREE_MIB must be a positive integer." >&2
  exit 1
fi
for required in "$EPISODE_RUNNER" "$SUMMARIZER" "$SIMULATOR" "$CLIENT"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Missing required file: $required" >&2
    exit 1
  fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="isaacsim_seed2_ckpt18450_success_rate_${NUM_EPISODES}ep_${POLICY_STEPS}steps_${TIMESTAMP}"
BATCH_RUN_DIR="${BATCH_RUN_DIR:-$RESULT_ROOT/$RUN_NAME}"
ARCHIVE="${BATCH_RUN_DIR}.tar.gz"
LATEST_OUTPUT_LINK="$RESULT_ROOT/isaacsim_seed2_ckpt18450_success_rate_${NUM_EPISODES}ep_latest"
LATEST_ARCHIVE_LINK="${LATEST_OUTPUT_LINK}.tar.gz"
PROGRESS_LOG="$BATCH_RUN_DIR/progress.log"

mkdir -p "$BATCH_RUN_DIR/episodes" "$LOG_ROOT"
mkdir -p "$RUNTIME_TEMP_ROOT"
mkdir -p "$RUNTIME_CACHE_ROOT"
RUNTIME_TEMP_ROOT="$(readlink -f "$RUNTIME_TEMP_ROOT")"
RUNTIME_CACHE_ROOT="$(readlink -f "$RUNTIME_CACHE_ROOT")"
if ! BATCH_TEMP_DIR="$(
  mktemp -d "$RUNTIME_TEMP_ROOT/sr.XXXXXX"
)"; then
  echo "[ERROR] Could not create a runtime temporary directory under $RUNTIME_TEMP_ROOT." >&2
  exit 1
fi

cleanup_runtime_temp() {
  if [[
    -n "${BATCH_TEMP_DIR:-}"
    && -d "$BATCH_TEMP_DIR"
    && "$BATCH_TEMP_DIR" == "$RUNTIME_TEMP_ROOT/"*
  ]]; then
    rm -rf -- "$BATCH_TEMP_DIR"
  fi
}
trap cleanup_runtime_temp EXIT

export TMPDIR="$BATCH_TEMP_DIR"
export TMP="$BATCH_TEMP_DIR"
export TEMP="$BATCH_TEMP_DIR"

mkdir -p \
  "$RUNTIME_CACHE_ROOT/xdg" \
  "$RUNTIME_CACHE_ROOT/xdg_config" \
  "$RUNTIME_CACHE_ROOT/xdg_data" \
  "$RUNTIME_CACHE_ROOT/cuda" \
  "$RUNTIME_CACHE_ROOT/nvidia_gl" \
  "$RUNTIME_CACHE_ROOT/matplotlib" \
  "$RUNTIME_CACHE_ROOT/torch" \
  "$RUNTIME_CACHE_ROOT/triton" \
  "$RUNTIME_CACHE_ROOT/numba" \
  "$RUNTIME_CACHE_ROOT/pip"

export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$RUNTIME_CACHE_ROOT/xdg_config"
export XDG_DATA_HOME="$RUNTIME_CACHE_ROOT/xdg_data"
export CUDA_CACHE_PATH="$RUNTIME_CACHE_ROOT/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$RUNTIME_CACHE_ROOT/nvidia_gl"
export MPLCONFIGDIR="$RUNTIME_CACHE_ROOT/matplotlib"
export TORCH_HOME="$RUNTIME_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$RUNTIME_CACHE_ROOT/triton"
export NUMBA_CACHE_DIR="$RUNTIME_CACHE_ROOT/numba"
export PIP_CACHE_DIR="$RUNTIME_CACHE_ROOT/pip"

verify_root_capacity() {
  local available_kib
  local required_kib

  if ! available_kib="$(df -Pk / | awk 'NR == 2 {print $4}')"; then
    echo "[ERROR] Could not query system root capacity." >&2
    return 1
  fi
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid available-space value for system root: $available_kib" >&2
    return 1
  fi

  required_kib=$((MIN_ROOT_FREE_MIB * 1024))
  if (( available_kib < required_kib )); then
    echo "[ERROR] System root has less than ${MIN_ROOT_FREE_MIB} MiB available." >&2
    df -hT / >&2 || true
    return 1
  fi
}

reset_runtime_temp() {
  if [[
    -z "${BATCH_TEMP_DIR:-}"
    || "$BATCH_TEMP_DIR" != "$RUNTIME_TEMP_ROOT/"*
    || "$BATCH_TEMP_DIR" == "$RUNTIME_TEMP_ROOT"
  ]]; then
    echo "[ERROR] Refusing to reset an invalid runtime temporary directory: ${BATCH_TEMP_DIR:-<empty>}" >&2
    return 1
  fi

  rm -rf -- "$BATCH_TEMP_DIR"
  mkdir -p "$BATCH_TEMP_DIR"
  chmod 0700 "$BATCH_TEMP_DIR"
}

verify_runtime_storage() {
  local temp_probe=""
  local output_probe=""

  if ! temp_probe="$(mktemp "$TMPDIR/temp_probe.XXXXXX")"; then
    echo "[ERROR] Runtime temporary directory is not writable: $TMPDIR" >&2
    return 1
  fi
  rm -f -- "$temp_probe"

  if ! output_probe="$(mktemp "$BATCH_RUN_DIR/output_probe.XXXXXX")"; then
    echo "[ERROR] Batch output directory is not writable: $BATCH_RUN_DIR" >&2
    return 1
  fi
  rm -f -- "$output_probe"
}

if ! verify_runtime_storage || ! verify_root_capacity; then
  df -h "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" >&2 || true
  df -i "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" >&2 || true
  exit 1
fi
cd "$PROJECT"

"$POLICY_PYTHON" - \
  "$BATCH_RUN_DIR/batch_manifest.json" \
  "$NUM_EPISODES" \
  "$POLICY_STEPS" \
  "$MAX_ATTEMPTS_PER_EPISODE" \
  "$GPU_COOLDOWN_SECONDS" \
  "$GPU_IDLE_TIMEOUT_SECONDS" \
  "$TRUCK_SUCCESS_MIN_PARTICLES" \
  "$TRUCK_SUCCESS_NO_GROWTH_STEPS" \
  "$BUCKET_OPEN_THRESHOLD" \
  "$RECORD_VIDEO" \
  "$DUMP_INPUTS" \
  "$RUN_TRACE_ANALYSIS" \
  "$(sha256sum "$EPISODE_RUNNER" | awk '{print $1}')" \
  "$(sha256sum "$SIMULATOR" | awk '{print $1}')" \
  "$(sha256sum "$CLIENT" | awk '{print $1}')" <<'PY'
from pathlib import Path
import json
import sys

(
    output_path,
    num_episodes,
    policy_steps,
    max_attempts,
    gpu_cooldown_seconds,
    gpu_idle_timeout_seconds,
    minimum_particles,
    no_growth_steps,
    bucket_open_threshold,
    record_video,
    dump_inputs,
    run_trace_analysis,
    runner_sha256,
    simulator_sha256,
    client_sha256,
) = sys.argv[1:]
payload = {
    "schema_version": "excavator_success_rate_batch_v1",
    "checkpoint_step": 18450,
    "scene_seed": 2,
    "num_episodes": int(num_episodes),
    "max_policy_steps_per_episode": int(policy_steps),
    "replan_interval": 3,
    "temporal_ensemble_width": 1,
    "temporal_ensemble_decay": 0.35,
    "excavation_sequence_supervisor_enabled": True,
    "supervisor_position_max_steps": 15,
    "supervisor_dump_min_steps": 3,
    "supervisor_dump_bucket_target_rad": -0.80,
    "supervisor_dump_velocity_rad_s": 1.20,
    "supervisor_empty_load_completion_enabled": False,
    "truck_success_min_particles": int(minimum_particles),
    "truck_success_no_growth_steps": int(
        no_growth_steps
    ),
    "bucket_open_threshold_rad": float(bucket_open_threshold),
    "max_attempts_per_episode": int(max_attempts),
    "gpu_cooldown_seconds": int(gpu_cooldown_seconds),
    "gpu_idle_timeout_seconds": int(gpu_idle_timeout_seconds),
    "record_video": bool(int(record_video)),
    "dump_initial_model_inputs": bool(int(dump_inputs)),
    "run_per_episode_trace_analysis": bool(int(run_trace_analysis)),
    "episode_runner_sha256": runner_sha256,
    "run_simulation_sha256": simulator_sha256,
    "policy_client_sha256": client_sha256,
    "truck_bed_success_metric_method": (
        "world_aabb_excluding_bucket_v2"
    ),
}
path = Path(output_path)
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise RuntimeError(
            "Refusing to resume a batch with different settings: "
            f"existing={existing}, requested={payload}"
        )
path.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY

wait_for_gpu_idle() {
  local elapsed
  local gpu_processes

  echo "Waiting ${GPU_COOLDOWN_SECONDS}s for GPU cooldown..." | \
    tee -a "$PROGRESS_LOG"
  sleep "$GPU_COOLDOWN_SECONDS"
  elapsed="$GPU_COOLDOWN_SECONDS"

  while true; do
    if ! gpu_processes="$(
      nvidia-smi \
        --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader 2>/dev/null
    )"; then
      echo "[ERROR] nvidia-smi failed while waiting for GPU idle." | \
        tee -a "$PROGRESS_LOG"
      return 1
    fi
    if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
      echo "GPU compute process list is empty after ${elapsed}s." | \
        tee -a "$PROGRESS_LOG"
      return 0
    fi

    if (( elapsed >= GPU_IDLE_TIMEOUT_SECONDS )); then
      echo "[ERROR] GPU still has compute processes after ${elapsed}s:" | \
        tee -a "$PROGRESS_LOG"
      echo "$gpu_processes" | tee -a "$PROGRESS_LOG"
      return 1
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done
}

echo "Batch output: $BATCH_RUN_DIR" | tee -a "$PROGRESS_LOG"
echo "Episodes: $NUM_EPISODES; max policy steps: $POLICY_STEPS" | \
  tee -a "$PROGRESS_LOG"
echo "Success: at least $TRUCK_SUCCESS_MIN_PARTICLES new particle(s) in the truck bed with no further increase for $TRUCK_SUCCESS_NO_GROWTH_STEPS steps." | \
  tee -a "$PROGRESS_LOG"
echo "Early stop: positive truck count with no further increase for $TRUCK_SUCCESS_NO_GROWTH_STEPS steps and bucket >= $BUCKET_OPEN_THRESHOLD rad." | \
  tee -a "$PROGRESS_LOG"
echo "GPU handoff: mandatory cooldown=${GPU_COOLDOWN_SECONDS}s; idle timeout=${GPU_IDLE_TIMEOUT_SECONDS}s." | \
  tee -a "$PROGRESS_LOG"
echo "Runtime temporary directory: $BATCH_TEMP_DIR" | \
  tee -a "$PROGRESS_LOG"
echo "Runtime cache directory: $RUNTIME_CACHE_ROOT" | \
  tee -a "$PROGRESS_LOG"
echo "Minimum system-root free space: ${MIN_ROOT_FREE_MIB} MiB" | \
  tee -a "$PROGRESS_LOG"
df -hT / | tee -a "$PROGRESS_LOG" || true
df -h "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" | \
  tee -a "$PROGRESS_LOG" || true
df -i "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" | \
  tee -a "$PROGRESS_LOG" || true

BATCH_ABORTED=0
for EPISODE_INDEX in $(seq 1 "$NUM_EPISODES"); do
  printf -v EPISODE_NAME 'episode_%04d' "$EPISODE_INDEX"
  EPISODE_DIR="$BATCH_RUN_DIR/episodes/$EPISODE_NAME"
  STATUS_JSON="$EPISODE_DIR/episode_status.json"
  VALIDATION_JSON="$EPISODE_DIR/rollout_validation.json"

  if [[ -f "$STATUS_JSON" && -f "$VALIDATION_JSON" ]] && \
    "$POLICY_PYTHON" - "$STATUS_JSON" "$VALIDATION_JSON" <<'PY'
from pathlib import Path
import json
import sys

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
valid = (
    status.get("status") == "completed"
    and validation.get("status") == "passed"
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[$EPISODE_INDEX/$NUM_EPISODES] Already complete; skipping." | \
      tee -a "$PROGRESS_LOG"
    continue
  fi

  if ! verify_runtime_storage || ! verify_root_capacity; then
    echo "[$EPISODE_INDEX/$NUM_EPISODES] Runtime storage or system-root reserve is unavailable; stopping the batch." | \
      tee -a "$PROGRESS_LOG"
    df -hT / | tee -a "$PROGRESS_LOG" || true
    df -h "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" | \
      tee -a "$PROGRESS_LOG" || true
    df -i "$RUNTIME_TEMP_ROOT" "$BATCH_RUN_DIR" | \
      tee -a "$PROGRESS_LOG" || true
    BATCH_ABORTED=1
    break
  fi

  EPISODE_COMPLETE=0
  for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS_PER_EPISODE"); do
    if ! verify_root_capacity; then
      echo "[$EPISODE_INDEX/$NUM_EPISODES] System-root reserve is too low; stopping retries." | \
        tee -a "$PROGRESS_LOG"
      break
    fi

    mkdir -p "$EPISODE_DIR"
    ATTEMPT_LOG="$EPISODE_DIR/launcher_attempt_${ATTEMPT}.log"
    echo "[$EPISODE_INDEX/$NUM_EPISODES] Starting attempt $ATTEMPT/$MAX_ATTEMPTS_PER_EPISODE..." | \
      tee -a "$PROGRESS_LOG"

    set +e
    EPISODE_INDEX="$EPISODE_INDEX" \
    BATCH_RUN_DIR="$BATCH_RUN_DIR" \
    POLICY_STEPS="$POLICY_STEPS" \
    TRUCK_SUCCESS_MIN_PARTICLES="$TRUCK_SUCCESS_MIN_PARTICLES" \
    TRUCK_SUCCESS_NO_GROWTH_STEPS="$TRUCK_SUCCESS_NO_GROWTH_STEPS" \
    BUCKET_OPEN_THRESHOLD="$BUCKET_OPEN_THRESHOLD" \
    RECORD_VIDEO="$RECORD_VIDEO" \
    DUMP_INPUTS="$DUMP_INPUTS" \
    RUN_TRACE_ANALYSIS="$RUN_TRACE_ANALYSIS" \
    PACKAGE_EPISODE=0 \
      bash "$EPISODE_RUNNER" > "$ATTEMPT_LOG" 2>&1
    ATTEMPT_STATUS=$?
    set -e

    set +e
    wait_for_gpu_idle
    GPU_IDLE_STATUS=$?
    set -e
    if [[ "$GPU_IDLE_STATUS" -ne 0 ]]; then
      ATTEMPT_STATUS=90
    elif ! reset_runtime_temp; then
      ATTEMPT_STATUS=91
    fi

    if [[
      "$ATTEMPT_STATUS" -eq 0
      && -f "$STATUS_JSON"
      && -f "$VALIDATION_JSON"
    ]] && \
      "$POLICY_PYTHON" - "$STATUS_JSON" "$VALIDATION_JSON" <<'PY'
from pathlib import Path
import json
import sys

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
valid = (
    status.get("status") == "completed"
    and validation.get("status") == "passed"
)
raise SystemExit(0 if valid else 1)
PY
    then
      EPISODE_COMPLETE=1
      RESULT_LINE="$(
        "$POLICY_PYTHON" - "$STATUS_JSON" <<'PY'
from pathlib import Path
import json
import sys

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    f"success={p['success']} "
    f"task_completed={p['task_completed_success_and_bucket_open']} "
    f"early={p['early_terminated']} "
    f"steps={p['policy_steps_executed']} "
    f"truck_particles={p['final_newly_deposited_particle_count']} "
    f"policy_success_s={p['success_time_from_policy_start_seconds']} "
    f"simulation_success_s={p['success_time_from_simulation_launch_seconds']}"
)
PY
      )"
      echo "[$EPISODE_INDEX/$NUM_EPISODES] Complete: $RESULT_LINE" | \
        tee -a "$PROGRESS_LOG"
      break
    fi

    echo "[$EPISODE_INDEX/$NUM_EPISODES] Attempt $ATTEMPT failed with exit code $ATTEMPT_STATUS. Log: $ATTEMPT_LOG" | \
      tee -a "$PROGRESS_LOG"
    tail -n 40 "$ATTEMPT_LOG" | tee -a "$PROGRESS_LOG" || true
    if grep -q 'No usable temporary directory found' "$ATTEMPT_LOG"; then
      echo "[$EPISODE_INDEX/$NUM_EPISODES] The runtime temporary directory became unavailable; skipping further retries." | \
        tee -a "$PROGRESS_LOG"
      break
    fi
  done

  if [[ "$EPISODE_COMPLETE" -ne 1 ]]; then
    echo "[$EPISODE_INDEX/$NUM_EPISODES] No valid result after $MAX_ATTEMPTS_PER_EPISODE attempts; stopping the batch so a deterministic configuration error is not repeated." | \
      tee -a "$PROGRESS_LOG"
    BATCH_ABORTED=1
    break
  fi

  "$POLICY_PYTHON" "$SUMMARIZER" \
    --batch-dir "$BATCH_RUN_DIR" \
    --expected-episodes "$NUM_EPISODES" \
    > "$BATCH_RUN_DIR/latest_summary.log" 2>&1
done

if [[ "$BATCH_ABORTED" -eq 1 ]]; then
  echo "Batch stopped early after an infrastructure or validation error." | \
    tee -a "$PROGRESS_LOG"
fi

set +e
"$POLICY_PYTHON" "$SUMMARIZER" \
  --batch-dir "$BATCH_RUN_DIR" \
  --expected-episodes "$NUM_EPISODES" \
  --strict \
  > "$BATCH_RUN_DIR/final_summary.log" 2>&1
SUMMARY_STATUS=$?
set -e

tar -C "$(dirname "$BATCH_RUN_DIR")" \
  -czf "$ARCHIVE" "$(basename "$BATCH_RUN_DIR")"
ln -sfn "$BATCH_RUN_DIR" "$LATEST_OUTPUT_LINK"
ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"

cat "$BATCH_RUN_DIR/final_summary.log"
echo "Batch output: $BATCH_RUN_DIR"
echo "CSV results: $BATCH_RUN_DIR/episode_results.csv"
echo "JSON summary: $BATCH_RUN_DIR/success_rate_summary.json"
echo "Archive: $ARCHIVE"
echo "Latest archive: $LATEST_ARCHIVE_LINK"
echo
echo "Download command:"
echo "scp -P 30105 root@120.209.70.195:$LATEST_ARCHIVE_LINK ."

exit "$SUMMARY_STATUS"
