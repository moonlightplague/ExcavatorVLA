#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
ISAAC_PYTHON=/root/isaacsim/python.sh
PYTHON=/opt/conda/envs/smolvla/bin/python

SIMULATOR="$PROJECT/run_simulation.py"
REPLAY_CLIENT="$PROJECT/scripts/bridge_test/replay_episode_npz_client.py"
DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
DATASET_META="$DATASET/meta/info.json"
EVAL_ROOT=/root/gpufree-data/excavator_offline_eval
RESULT_ROOT=/root/gpufree-data/excavator_isaacsim_replay
LOG_ROOT=/root/gpufree-data/excavator_logs

EPISODE_ID="${EPISODE_ID:-0}"
EPISODE_NPZ="${EPISODE_NPZ:-}"
MAX_STEPS="${MAX_STEPS:-0}"
SCENE_SEED=2
SAND_AMOUNT=1
EXECUTION_CHUNK_SIZE=1
TASK_TEXT="Excavate one scoop of sand from the sand pile in front of the excavator's initial base pose, then carry and dump the collected material into the truck bed to the right of the excavator's initial base pose."

SIM_PID=""

stop_simulator() {
  if [[ -z "$SIM_PID" ]]; then
    return
  fi
  if kill -0 "$SIM_PID" 2>/dev/null; then
    kill -TERM -- "-$SIM_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$SIM_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SIM_PID" 2>/dev/null; then
      echo "[WARN] Isaac Sim did not stop within 60 seconds; sending SIGKILL."
      kill -KILL -- "-$SIM_PID" 2>/dev/null || true
    fi
  fi
  wait "$SIM_PID" 2>/dev/null || true
  SIM_PID=""
}

cleanup() {
  stop_simulator
}

trap cleanup EXIT INT TERM

cd "$PROJECT"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

for required in \
  "$ISAAC_PYTHON" \
  "$SIMULATOR" \
  "$REPLAY_CLIENT" \
  "$DATASET_META"
do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 1
  fi
done

if [[ "$EXECUTION_CHUNK_SIZE" != "1" ]]; then
  echo "[ERROR] This launcher requires EXECUTION_CHUNK_SIZE=1." >&2
  exit 1
fi
if ! [[ "$EPISODE_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] EPISODE_ID must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MAX_STEPS must be a non-negative integer." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "[ERROR] setsid is required." >&2
  exit 1
fi
if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[r]eplay_episode_npz_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulator, policy client, replay client, or training process is already running:" >&2
  pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[r]eplay_episode_npz_client.py|[l]erobot-train|[o]t_train.py' >&2
  exit 1
fi

if [[ -z "$EPISODE_NPZ" ]]; then
  LATEST_EVAL_LINK="$(ls -dt \
    "$EVAL_ROOT"/excavator_smolvla_seed2_fixed5ep_canonical300_from*_b144_latest3_5episodes_latest \
    2>/dev/null | head -1)"
  if [[ -z "$LATEST_EVAL_LINK" ]]; then
    echo "[ERROR] Latest offline five-episode evaluation was not found." >&2
    exit 1
  fi
  LATEST_EVAL="$(readlink -f "$LATEST_EVAL_LINK")"
  COMPARISON="$LATEST_EVAL/checkpoint_comparison.csv"
  if [[ ! -f "$COMPARISON" ]]; then
    echo "[ERROR] Missing checkpoint comparison: $COMPARISON" >&2
    exit 1
  fi
  BEST_STEP="$(
    "$PYTHON" - "$COMPARISON" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("Checkpoint comparison is empty")
best = max(rows, key=lambda row: float(row["micro_representative_score"]))
print(int(best["checkpoint_step"]))
PY
  )"
  EPISODE_NPZ="$LATEST_EVAL/checkpoint_$(printf '%06d' "$BEST_STEP")/episodes/episode_$(printf '%06d' "$EPISODE_ID").npz"
else
  EPISODE_NPZ="$(readlink -f "$EPISODE_NPZ")"
  BEST_STEP="external"
fi

if [[ ! -f "$EPISODE_NPZ" ]]; then
  echo "[ERROR] Episode output was not found: $EPISODE_NPZ" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  for socket_path in /tmp/.X11-unix/X*; do
    if [[ -S "$socket_path" ]]; then
      export DISPLAY=":${socket_path##*X}"
      break
    fi
  done
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] DISPLAY is unset and no X11 socket was found." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="isaacsim_offline_episode$(printf '%06d' "$EPISODE_ID")_chunk1_${TIMESTAMP}"
OUTPUT_DIR="$RESULT_ROOT/$RUN_NAME"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
LATEST_OUTPUT_LINK="$RESULT_ROOT/isaacsim_offline_episode_chunk1_latest"
LATEST_ARCHIVE_LINK="${LATEST_OUTPUT_LINK}.tar.gz"
SIM_LOG="$OUTPUT_DIR/simulator.log"
REPLAY_LOG="$OUTPUT_DIR/replay_client.log"
TRACE_LOG="$OUTPUT_DIR/replay_trace.jsonl"
VIDEO_DIR="$OUTPUT_DIR/videos"

mkdir -p "$OUTPUT_DIR" "$VIDEO_DIR"

"$PYTHON" - "$EPISODE_NPZ" "$OUTPUT_DIR/source_manifest.json" "$BEST_STEP" "$EPISODE_ID" "$MAX_STEPS" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
with np.load(source, allow_pickle=False) as payload:
    actions = np.asarray(payload["pred_action"])
    if actions.ndim != 3 or actions.shape[2] != 4:
        raise RuntimeError(f"Expected pred_action [F,H,4], got {actions.shape}")
    manifest = {
        "source_episode_npz": str(source),
        "checkpoint_step": sys.argv[3],
        "episode_id": int(sys.argv[4]),
        "stored_pred_action_shape": list(actions.shape),
        "replay_selection": "pred_action[:, 0, :]",
        "execution_chunk_size": 1,
        "max_steps": int(sys.argv[5]),
    }
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "Source episode: $EPISODE_NPZ"
echo "Execution chunk size: $EXECUTION_CHUNK_SIZE"
echo "Output: $OUTPUT_DIR"

setsid "$ISAAC_PYTHON" "$SIMULATOR" \
  --scene-seed "$SCENE_SEED" \
  --sand-amount "$SAND_AMOUNT" \
  --robot-initial-joints-deg 40 46 -62 -20 \
  --expected-state27-initial-base \
    -6.950658321380615 2.2325551509857178 0 \
  --state27-dig-target-world \
    -0.9082751274108887 7.224985599517822 0.2896101474761963 \
  --state27-unload-target-world \
    -9.09683609008789 -0.9592496156692505 4.222683429718018 \
  --task-text "$TASK_TEXT" \
  > "$SIM_LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > "$OUTPUT_DIR/simulator.pid"

BRIDGE_READY=0
for _ in $(seq 1 600); do
  if grep -q '\[bridge\] TCP server listening' "$SIM_LOG" 2>/dev/null; then
    BRIDGE_READY=1
    break
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "[ERROR] Isaac Sim exited before the bridge became ready." >&2
    tail -n 160 "$SIM_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$BRIDGE_READY" -ne 1 ]]; then
  echo "[ERROR] Timed out waiting for the simulator bridge." >&2
  tail -n 160 "$SIM_LOG" >&2 || true
  exit 1
fi

set +e
"$PYTHON" "$REPLAY_CLIENT" \
  --episode-npz "$EPISODE_NPZ" \
  --dataset-meta "$DATASET_META" \
  --execution-chunk-size 1 \
  --max-steps "$MAX_STEPS" \
  --host 127.0.0.1 \
  --port 5555 \
  --trace-log "$TRACE_LOG" \
  --record-video-dir "$VIDEO_DIR" \
  --print-every 1 \
  > "$REPLAY_LOG" 2>&1
REPLAY_STATUS=$?
set -e

stop_simulator

tar -C "$(dirname "$OUTPUT_DIR")" \
  -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
ln -sfn "$OUTPUT_DIR" "$LATEST_OUTPUT_LINK"
ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"

if [[ "$REPLAY_STATUS" -ne 0 ]]; then
  echo "[ERROR] Replay client failed with exit code $REPLAY_STATUS." >&2
  tail -n 160 "$REPLAY_LOG" >&2 || true
  echo "Failure archive: $ARCHIVE" >&2
  exit "$REPLAY_STATUS"
fi

echo "Replay trace: $TRACE_LOG"
echo "Replay log: $REPLAY_LOG"
echo "Videos: $VIDEO_DIR"
echo "Archive: $ARCHIVE"
echo "Latest archive link: $LATEST_ARCHIVE_LINK"
echo "Download command:"
echo "scp -P 30105 root@120.209.70.195:$LATEST_ARCHIVE_LINK ."
