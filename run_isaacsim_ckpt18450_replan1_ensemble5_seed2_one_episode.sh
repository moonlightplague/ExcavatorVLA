#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
ISAAC_PYTHON=/root/isaacsim/python.sh
POLICY_PYTHON=/opt/conda/envs/smolvla/bin/python

SIMULATOR="$PROJECT/run_simulation.py"
CLIENT="$PROJECT/scripts/bridge_test/smolvla_policy_client.py"
ANALYZER="$PROJECT/analyze_smolvla_200step_trace.py"
DEPLOYMENT_CONTRACT="$PROJECT/excavator_common/deployment_contract.py"
OBSERVATION_CONTRACT="$PROJECT/excavator_common/vla_observation_contract.py"

CHECKPOINT=/root/gpufree-data/outputs/train/excavator_smolvla_seed2_fixed5ep_canonical300_from16650_b144/checkpoints/018450/pretrained_model
DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
DATASET_META="$DATASET/meta/info.json"
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

RESULT_ROOT=/root/gpufree-data/excavator_isaacsim_eval
LOG_ROOT=/root/gpufree-data/excavator_logs

# Simulation scene and sand settings.
SCENE_SEED=2
SAND_AMOUNT=1.0
SAND_PARAMETER_MODE=soft_dig
SAND_RADIUS_SCALE=0.2
SAND_WALL_ENABLED=1
SAND_WALL_RADIUS_SCALE=1.5
SAND_SETTLE_FRAMES=60

# Policy and execution settings.
POLICY_STEPS=300
REPLAN_INTERVAL=3
TEMPORAL_ENSEMBLE_WIDTH=1
TEMPORAL_ENSEMBLE_DECAY=0.35
ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR=1
ENABLE_LOAD_RETENTION_CONSTRAINT=0
ENABLE_UNLOAD_GEOMETRY_GATE=0
SUPERVISOR_LOAD_TRIGGER=500
SUPERVISOR_LOAD_RISE=400
SUPERVISOR_DIG_DESCENT_HEIGHT_TRIGGER=2.4
SUPERVISOR_DIG_DESCENT_STAGE_TRIGGER=2
SUPERVISOR_DIG_DESCENT_VELOCITY=0.20
SUPERVISOR_DIG_BUCKET_HALF_SCALE_DISTANCE=0.85
SUPERVISOR_DIG_DEPTH_TOLERANCE=0.75
SUPERVISOR_DIG_REBOUND_HEIGHT=0.01
SUPERVISOR_DIG_REBOUND_DWELL_STEPS=2
SUPERVISOR_DIG_REBOUND_MIN_LOAD=500
SUPERVISOR_CURL_TARGET=-2.04
SUPERVISOR_CURL_VELOCITY=0.85
SUPERVISOR_MAX_LIFT_STEPS=30
SUPERVISOR_TURN_ENTRY_BOOM_TARGET=0.18
SUPERVISOR_TURN_ENTRY_VELOCITY=0.22
SUPERVISOR_TURN_ENTRY_STEPS=5
SUPERVISOR_TURN_HANDOFF_ANGLE=0.20
SUPERVISOR_PHASE_SYNC_BUCKET_CLOSED=-2.04
SUPERVISOR_PHASE_SYNC_BOOM_LIFTED=0.28
SUPERVISOR_SWING_TOLERANCE=0.03
SUPERVISOR_UNLOAD_HEIGHT_VELOCITY=0.12
MAX_INPUT_ABS_SIGMA=8
MAX_EFFORT_ABS_SIGMA=8
TASK_TEXT="Excavate one scoop of sand from the sand pile in front of the excavator's initial base pose, then carry and dump the collected material into the truck bed to the right of the excavator's initial base pose."

case "$SAND_WALL_ENABLED" in
  1) SAND_WALL_OPTION=--sand-wall ;;
  0) SAND_WALL_OPTION=--no-sand-wall ;;
  *)
    echo "[ERROR] SAND_WALL_ENABLED must be 0 or 1." >&2
    exit 2
    ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="isaacsim_seed2_ckpt18450_replan${REPLAN_INTERVAL}_ensemble${TEMPORAL_ENSEMBLE_WIDTH}_supervisor${ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR}_${POLICY_STEPS}steps_${TIMESTAMP}"
OUTPUT_DIR="$RESULT_ROOT/$RUN_NAME"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
LATEST_OUTPUT_LINK="$RESULT_ROOT/isaacsim_seed2_ckpt18450_replan${REPLAN_INTERVAL}_ensemble${TEMPORAL_ENSEMBLE_WIDTH}_supervisor${ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR}_latest"
LATEST_ARCHIVE_LINK="${LATEST_OUTPUT_LINK}.tar.gz"

SIM_LOG="$OUTPUT_DIR/simulator.log"
CLIENT_LOG="$OUTPUT_DIR/policy_client.log"
TRACE_LOG="$OUTPUT_DIR/policy_steps_trace.jsonl"
CHUNK_LOG="$OUTPUT_DIR/action_stage_chunks.jsonl"
VIDEO_DIR="$OUTPUT_DIR/videos"
INPUT_DUMP_DIR="$OUTPUT_DIR/input_dump"
ANALYSIS_DIR="$OUTPUT_DIR/trace_analysis"
ANALYSIS_LOG="$OUTPUT_DIR/trace_analysis.log"

SIM_PID=""

stop_simulator() {
  if [[ -z "$SIM_PID" ]]; then
    return
  fi

  if kill -0 "$SIM_PID" 2>/dev/null; then
    echo "Stopping Isaac Sim process group $SIM_PID..."
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
mkdir -p "$OUTPUT_DIR" "$VIDEO_DIR" "$INPUT_DUMP_DIR" "$LOG_ROOT"

echo "======================================================================"
echo "1. Validate deployment inputs"
echo "======================================================================"

for required in \
  "$ISAAC_PYTHON" \
  "$SIMULATOR" \
  "$CLIENT" \
  "$ANALYZER" \
  "$DEPLOYMENT_CONTRACT" \
  "$OBSERVATION_CONTRACT" \
  "$CHECKPOINT/model.safetensors" \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/train_config.json" \
  "$CHECKPOINT/../training_state/training_step.json" \
  "$DATASET_META" \
  "$VLM" \
  "$PRIOR"
do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 1
  fi
done

if ! command -v setsid >/dev/null 2>&1; then
  echo "[ERROR] setsid is required to manage the Isaac Sim process group." >&2
  exit 1
fi

if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulator, policy client, or training process is already running:" >&2
  pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >&2
  exit 1
fi

GPU_PROCESSES="$(
  nvidia-smi --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "[ERROR] GPU compute processes are already running:" >&2
  echo "$GPU_PROCESSES" >&2
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
  echo "[ERROR] run_simulation.py requires a viewport-capable Isaac Sim session." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/gpufree-data/hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$POLICY_PYTHON" - \
  "$CHECKPOINT" \
  "$DATASET_META" \
  "$REPLAN_INTERVAL" \
  "$TEMPORAL_ENSEMBLE_WIDTH" \
  "$TEMPORAL_ENSEMBLE_DECAY" \
  "$TASK_TEXT" <<'PY'
from pathlib import Path
import json
import math
import sys

import pandas as pd
from excavator_common import deployment_contract
from excavator_common import vla_observation_contract

checkpoint = Path(sys.argv[1]).resolve()
dataset_meta = Path(sys.argv[2]).resolve()
replan_interval = int(sys.argv[3])
temporal_ensemble_width = int(sys.argv[4])
temporal_ensemble_decay = float(sys.argv[5])
task_text = sys.argv[6]

required_deployment_symbols = (
    "OBSERVATION_SCHEMA_27D_PLUS_EFFORT",
    "OBSERVATION_SCHEMA_28D_PLUS_EFFORT",
    "OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT",
    "STATE_NAMES_27D",
    "STATE_NAMES_28D",
    "validate_client_contract",
)
missing_deployment_symbols = [
    name
    for name in required_deployment_symbols
    if not hasattr(deployment_contract, name)
]
if missing_deployment_symbols:
    raise RuntimeError(
        "excavator_common/deployment_contract.py is out of date; "
        f"missing={missing_deployment_symbols!r}"
    )
required_observation_symbols = (
    "STATE_NAMES_28D",
    "LEGACY_STATE_NAMES_28D_V3",
    "EFFORT_NAMES_4D",
)
missing_observation_symbols = [
    name
    for name in required_observation_symbols
    if not hasattr(vla_observation_contract, name)
]
if missing_observation_symbols:
    raise RuntimeError(
        "excavator_common/vla_observation_contract.py is out of date; "
        f"missing={missing_observation_symbols!r}"
    )

config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
train_config = json.loads(
    (checkpoint / "train_config.json").read_text(encoding="utf-8")
)
step_state = json.loads(
    (checkpoint.parent / "training_state" / "training_step.json").read_text(
        encoding="utf-8"
    )
)
info = json.loads(dataset_meta.read_text(encoding="utf-8"))

step = int(step_state["step"])
if step != 18450:
    raise RuntimeError(f"Expected checkpoint step 18450, got {step}")
if int(config.get("chunk_size", -1)) != 50:
    raise RuntimeError(f"Expected model chunk_size 50, got {config.get('chunk_size')}")
if not 1 <= temporal_ensemble_width <= int(config["chunk_size"]):
    raise RuntimeError(
        "Temporal ensemble width must be in [1, chunk_size], got "
        f"{temporal_ensemble_width}"
    )
if (
    not math.isfinite(temporal_ensemble_decay)
    or temporal_ensemble_decay < 0.0
):
    raise RuntimeError(
        "Temporal ensemble decay must be non-negative, got "
        f"{temporal_ensemble_decay}"
    )
if config.get("stage_source_key") != "observation.stage_current_id":
    raise RuntimeError(
        f"Unexpected stage_source_key: {config.get('stage_source_key')!r}"
    )
input_features = config.get("input_features", {})
state_shape = input_features.get("observation.state", {}).get("shape")
if state_shape != [27]:
    raise RuntimeError(f"Expected 27D checkpoint state, got {state_shape}")
if info.get("features", {}).get("observation.state", {}).get("shape") != [27]:
    raise RuntimeError("Deployment dataset metadata is not 27D")
if float(info.get("fps", 0)) <= 0:
    raise RuntimeError(f"Invalid dataset FPS: {info.get('fps')}")

tasks_path = dataset_meta.parent / "tasks.parquet"
if not tasks_path.is_file():
    raise RuntimeError(f"Missing dataset task table: {tasks_path}")
tasks = pd.read_parquet(tasks_path)
if "task" in tasks.columns:
    known_tasks = {str(value) for value in tasks["task"].tolist()}
elif "tasks" in tasks.columns:
    known_tasks = {str(value) for value in tasks["tasks"].tolist()}
else:
    known_tasks = {
        str(value)
        for value in tasks.index.tolist()
        if isinstance(value, str)
    }
if not known_tasks:
    raise RuntimeError(
        "Could not resolve task text from dataset task table: "
        f"columns={tasks.columns.tolist()!r}"
    )
if task_text not in known_tasks:
    raise RuntimeError(
        "Deployment task text is not an exact dataset task: "
        f"task={task_text!r}, known={sorted(known_tasks)!r}"
    )

print("checkpoint:", checkpoint)
print("checkpoint step:", step)
print("model output chunk size:", config["chunk_size"])
print("execution replan interval:", replan_interval)
print("temporal ensemble width:", temporal_ensemble_width)
print("temporal ensemble decay:", temporal_ensemble_decay)
print("dataset metadata:", dataset_meta)
print("training FPS:", info["fps"])
print("exact dataset task:", task_text)
print("configured training target:", train_config.get("steps"))
PY

echo "DISPLAY: $DISPLAY"
echo "Output: $OUTPUT_DIR"

echo
echo "======================================================================"
echo "2. Start Isaac Sim with the fixed seed-2 scene"
echo "======================================================================"

setsid "$ISAAC_PYTHON" "$SIMULATOR" \
  --scene-seed "$SCENE_SEED" \
  --sand-amount "$SAND_AMOUNT" \
  --sand-parameter-mode "$SAND_PARAMETER_MODE" \
  --sand-radius-scale "$SAND_RADIUS_SCALE" \
  "$SAND_WALL_OPTION" \
  --sand-wall-radius-scale "$SAND_WALL_RADIUS_SCALE" \
  --sand-settle-frames "$SAND_SETTLE_FRAMES" \
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
echo "Isaac Sim PID: $SIM_PID"
echo "Isaac Sim log: $SIM_LOG"

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
  echo "[ERROR] Timed out waiting 600 seconds for the simulator bridge." >&2
  tail -n 160 "$SIM_LOG" >&2 || true
  exit 1
fi

echo "Simulator bridge is ready."

echo "Execution assistance: sequence_supervisor=$ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR load_retention=$ENABLE_LOAD_RETENTION_CONSTRAINT unload_geometry_gate=$ENABLE_UNLOAD_GEOMETRY_GATE"
echo "Supervisor forced descent: height_trigger=$SUPERVISOR_DIG_DESCENT_HEIGHT_TRIGGER stage_trigger=$SUPERVISOR_DIG_DESCENT_STAGE_TRIGGER velocity=$SUPERVISOR_DIG_DESCENT_VELOCITY bucket_half_scale_distance=$SUPERVISOR_DIG_BUCKET_HALF_SCALE_DISTANCE depth_tolerance=$SUPERVISOR_DIG_DEPTH_TOLERANCE one_shot=true"
echo "Supervisor forced curl: target=$SUPERVISOR_CURL_TARGET rad velocity=$SUPERVISOR_CURL_VELOCITY rad/s"
echo "Supervisor loaded-bucket gate: load=$SUPERVISOR_LOAD_TRIGGER rise=$SUPERVISOR_LOAD_RISE"
echo "Supervisor dig gate: rebound_height=$SUPERVISOR_DIG_REBOUND_HEIGHT rebound_dwell_steps=$SUPERVISOR_DIG_REBOUND_DWELL_STEPS min_load=$SUPERVISOR_DIG_REBOUND_MIN_LOAD"
echo "Supervisor turn guide: boom_target=$SUPERVISOR_TURN_ENTRY_BOOM_TARGET velocity=$SUPERVISOR_TURN_ENTRY_VELOCITY steps=$SUPERVISOR_TURN_ENTRY_STEPS swing_tolerance=$SUPERVISOR_SWING_TOLERANCE max_lift_steps=$SUPERVISOR_MAX_LIFT_STEPS"
echo "Supervisor phase sync: turn_handoff=$SUPERVISOR_TURN_HANDOFF_ANGLE bucket_closed=$SUPERVISOR_PHASE_SYNC_BUCKET_CLOSED boom_lifted=$SUPERVISOR_PHASE_SYNC_BOOM_LIFTED"
echo "Supervisor unload height velocity: $SUPERVISOR_UNLOAD_HEIGHT_VELOCITY"
if [[ "$ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR" != "1" ]]; then
  echo "[ERROR] The excavation sequence supervisor must be enabled." >&2
  exit 1
fi
if [[ "$ENABLE_LOAD_RETENTION_CONSTRAINT" != "0" || "$ENABLE_UNLOAD_GEOMETRY_GATE" != "0" ]]; then
  echo "[ERROR] The legacy standalone constraints must remain disabled when the sequence supervisor is active." >&2
  exit 1
fi

echo
echo "======================================================================"
echo "3. Run checkpoint 18450 with K=$REPLAN_INTERVAL for one ${POLICY_STEPS}-step episode"
echo "======================================================================"

set +e
"$POLICY_PYTHON" "$CLIENT" \
  --host 127.0.0.1 \
  --port 5555 \
  --ckpt "$CHECKPOINT" \
  --vlm "$VLM" \
  --dataset-meta "$DATASET_META" \
  --warmup-steps 0 \
  --replan-interval "$REPLAN_INTERVAL" \
  --temporal-ensemble-width "$TEMPORAL_ENSEMBLE_WIDTH" \
  --temporal-ensemble-decay "$TEMPORAL_ENSEMBLE_DECAY" \
  --enable-excavation-sequence-supervisor \
  --supervisor-load-trigger "$SUPERVISOR_LOAD_TRIGGER" \
  --supervisor-load-rise "$SUPERVISOR_LOAD_RISE" \
  --supervisor-dig-descent-height-trigger "$SUPERVISOR_DIG_DESCENT_HEIGHT_TRIGGER" \
  --supervisor-dig-descent-stage-trigger "$SUPERVISOR_DIG_DESCENT_STAGE_TRIGGER" \
  --supervisor-dig-descent-velocity "$SUPERVISOR_DIG_DESCENT_VELOCITY" \
  --supervisor-dig-bucket-half-scale-distance "$SUPERVISOR_DIG_BUCKET_HALF_SCALE_DISTANCE" \
  --supervisor-dig-depth-tolerance "$SUPERVISOR_DIG_DEPTH_TOLERANCE" \
  --supervisor-dig-rebound-height "$SUPERVISOR_DIG_REBOUND_HEIGHT" \
  --supervisor-dig-rebound-dwell-steps "$SUPERVISOR_DIG_REBOUND_DWELL_STEPS" \
  --supervisor-dig-rebound-min-load "$SUPERVISOR_DIG_REBOUND_MIN_LOAD" \
  --supervisor-curl-target "$SUPERVISOR_CURL_TARGET" \
  --supervisor-curl-velocity "$SUPERVISOR_CURL_VELOCITY" \
  --supervisor-max-lift-steps "$SUPERVISOR_MAX_LIFT_STEPS" \
  --supervisor-turn-entry-boom-target "$SUPERVISOR_TURN_ENTRY_BOOM_TARGET" \
  --supervisor-turn-entry-velocity "$SUPERVISOR_TURN_ENTRY_VELOCITY" \
  --supervisor-turn-entry-steps "$SUPERVISOR_TURN_ENTRY_STEPS" \
  --supervisor-turn-handoff-angle "$SUPERVISOR_TURN_HANDOFF_ANGLE" \
  --supervisor-phase-sync-bucket-closed "$SUPERVISOR_PHASE_SYNC_BUCKET_CLOSED" \
  --supervisor-phase-sync-boom-lifted "$SUPERVISOR_PHASE_SYNC_BOOM_LIFTED" \
  --supervisor-swing-tolerance "$SUPERVISOR_SWING_TOLERANCE" \
  --supervisor-unload-height-velocity "$SUPERVISOR_UNLOAD_HEIGHT_VELOCITY" \
  --max-input-abs-sigma "$MAX_INPUT_ABS_SIGMA" \
  --max-effort-abs-sigma "$MAX_EFFORT_ABS_SIGMA" \
  --steps "$POLICY_STEPS" \
  --sleep 0 \
  --print-every 1 \
  --no-print-full-chunk \
  --chunk-log "$CHUNK_LOG" \
  --trace-log "$TRACE_LOG" \
  --record-video-dir "$VIDEO_DIR" \
  --record-video-fps 0 \
  --dump-dir "$INPUT_DUMP_DIR" \
  --dump-input-steps 10 \
  > "$CLIENT_LOG" 2>&1
CLIENT_STATUS=$?
set -e

stop_simulator

if [[ "$CLIENT_STATUS" -ne 0 ]]; then
  echo "[ERROR] Policy client failed with exit code $CLIENT_STATUS." >&2
  tail -n 160 "$CLIENT_LOG" >&2 || true
  tar -C "$(dirname "$OUTPUT_DIR")" \
    -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
  ln -sfn "$OUTPUT_DIR" "$LATEST_OUTPUT_LINK"
  ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"
  exit "$CLIENT_STATUS"
fi

echo
echo "======================================================================"
echo "4. Validate K=$REPLAN_INTERVAL execution and analyze the rollout"
echo "======================================================================"

"$POLICY_PYTHON" - \
  "$TRACE_LOG" \
  "$CHUNK_LOG" \
  "$OUTPUT_DIR/rollout_validation.json" \
  "$POLICY_STEPS" \
  "$REPLAN_INTERVAL" \
  "$TEMPORAL_ENSEMBLE_WIDTH" \
  "$TEMPORAL_ENSEMBLE_DECAY" \
  "$ENABLE_EXCAVATION_SEQUENCE_SUPERVISOR" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np

trace_path = Path(sys.argv[1]).resolve()
chunk_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
expected_steps = int(sys.argv[4])
replan_interval = int(sys.argv[5])
temporal_ensemble_width = int(sys.argv[6])
temporal_ensemble_decay = float(sys.argv[7])
sequence_supervisor_enabled = bool(int(sys.argv[8]))


def load_jsonl(path: Path):
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


trace = load_jsonl(trace_path)
chunks = load_jsonl(chunk_path)

if len(trace) != expected_steps:
    raise RuntimeError(f"Expected {expected_steps} trace rows, got {len(trace)}")
expected_chunk_steps = list(range(0, expected_steps, replan_interval))
if len(chunks) != len(expected_chunk_steps):
    raise RuntimeError(
        f"K={replan_interval} requires {len(expected_chunk_steps)} chunks, "
        f"got {len(chunks)}"
    )

expected = list(range(expected_steps))
policy_steps = [int(row["policy_step"]) for row in trace]
chunk_steps = [int(row["replan_step"]) for row in chunks]
if policy_steps != expected:
    raise RuntimeError(
        f"Policy steps are not exactly 0..{expected_steps - 1}"
    )
if chunk_steps != expected_chunk_steps:
    raise RuntimeError(
        f"K={replan_interval} replan steps differ: "
        f"actual={chunk_steps}, expected={expected_chunk_steps}"
    )

bad_trace = [
    int(row["policy_step"])
    for row in trace
    if int(row["chunk_index"])
    != int(row["policy_step"]) % replan_interval
    or int(row["chunk_origin_step"])
    != (int(row["policy_step"]) // replan_interval) * replan_interval
]
if bad_trace:
    raise RuntimeError(
        f"K={replan_interval} execution contract failed at steps: "
        f"{bad_trace[:20]}"
    )

action_names = ("swing", "boom", "arm", "bucket")
chunks_by_origin = {
    int(row["replan_step"]): row
    for row in chunks
}
bad_ensemble = []
bad_supervisor_execution = []
supervisor_modified_steps = []
supervisor_phases = []
phase_order = {
    name: index
    for index, name in enumerate(
        (
            "dig",
            "curl_secure",
            "lift_loaded",
            "transfer_loaded",
            "position_over_truck",
            "dump",
            "complete",
        )
    )
}
model_chunk_size = int(chunks[0]["chunk_size"])
for row in trace:
    step = int(row["policy_step"])
    metadata = row.get("temporal_ensemble") or {}
    available_origins = [
        origin
        for origin in expected_chunk_steps
        if origin <= step and step - origin < model_chunk_size
    ]
    expected_origins = list(
        reversed(available_origins[-temporal_ensemble_width:])
    )
    expected_indices = [step - origin for origin in expected_origins]
    if (
        bool(metadata.get("enabled"))
        != (temporal_ensemble_width > 1)
        or int(metadata.get("configured_width", -1))
        != temporal_ensemble_width
        or not np.isclose(
            float(metadata.get("decay", float("nan"))),
            temporal_ensemble_decay,
        )
        or int(metadata.get("num_predictions", -1))
        != len(expected_origins)
        or metadata.get("source_chunk_origins") != expected_origins
        or metadata.get("source_chunk_indices") != expected_indices
    ):
        bad_ensemble.append((step, "metadata"))
        continue

    weights = np.asarray(metadata.get("weights", []), dtype=np.float64)
    if weights.shape != (len(expected_origins),) or not np.isclose(
        float(np.sum(weights)),
        1.0,
        rtol=0.0,
        atol=1.0e-9,
    ):
        bad_ensemble.append((step, "weights"))
        continue

    source_actions = []
    for origin, chunk_index in zip(
        expected_origins,
        expected_indices,
        strict=True,
    ):
        item = chunks_by_origin[origin]["items"][chunk_index]
        source_actions.append(
            [float(item["action"][name]) for name in action_names]
        )
    expected_action = np.sum(
        np.asarray(source_actions, dtype=np.float64)
        * weights[:, None],
        axis=0,
    )
    executed_action = np.asarray(
        [float(row["action"][name]) for name in action_names],
        dtype=np.float64,
    )
    if not np.allclose(
        executed_action,
        expected_action,
        rtol=0.0,
        atol=2.0e-6,
    ):
        bad_ensemble.append((step, "action"))

    constraint_metadata = row.get("execution_constraints") or {}
    constrained_action = np.asarray(
        [
            float(row.get("executed_action", {}).get(name, float("nan")))
            for name in action_names
        ],
        dtype=np.float64,
    )
    phase = str(constraint_metadata.get("phase", ""))
    if sequence_supervisor_enabled:
        if (
            not bool(constraint_metadata.get("enabled", False))
            or constraint_metadata.get("mode")
            != "excavation_sequence_supervisor_v1"
            or phase not in phase_order
            or not np.all(np.isfinite(constrained_action))
        ):
            bad_supervisor_execution.append(step)
        else:
            supervisor_phases.append(phase)
            if bool(constraint_metadata.get("modified", False)):
                supervisor_modified_steps.append(step)
    elif (
        bool(constraint_metadata.get("enabled", True))
        or bool(constraint_metadata.get("modified", True))
        or not np.allclose(
            constrained_action,
            executed_action,
            rtol=0.0,
            atol=0.0,
        )
    ):
        bad_supervisor_execution.append(step)

if bad_ensemble:
    raise RuntimeError(
        "Temporal-ensemble execution validation failed: "
        f"{bad_ensemble[:20]}"
    )
if bad_supervisor_execution:
    raise RuntimeError(
        "Sequence-supervisor execution validation failed at steps: "
        f"{bad_supervisor_execution[:20]}"
    )
if sequence_supervisor_enabled:
    phase_ranks = [phase_order[phase] for phase in supervisor_phases]
    if any(right < left for left, right in zip(phase_ranks, phase_ranks[1:])):
        raise RuntimeError(
            "Sequence-supervisor phases regressed: "
            f"{supervisor_phases}"
        )

checkpoint_values = sorted({str(row.get("checkpoint", "")) for row in chunks})
if len(checkpoint_values) != 1 or "018450" not in checkpoint_values[0]:
    raise RuntimeError(f"Unexpected chunk checkpoint values: {checkpoint_values}")

stage_ids = [int(row["stage_id"]) for row in trace]
payload = {
    "status": "passed",
    "checkpoint_step": 18450,
    "scene_seed": 2,
    "policy_steps": expected_steps,
    "model_output_chunk_size": model_chunk_size,
    "execution_replan_interval": replan_interval,
    "temporal_ensemble_width": temporal_ensemble_width,
    "temporal_ensemble_decay": temporal_ensemble_decay,
    "temporal_ensemble_actions_validated": True,
    "excavation_sequence_supervisor_enabled": (
        sequence_supervisor_enabled
    ),
    "load_retention_constraint_enabled": False,
    "unload_geometry_gate_enabled": False,
    "pure_model_execution_validated": not sequence_supervisor_enabled,
    "supervisor_modified_steps": len(supervisor_modified_steps),
    "supervisor_phase_sequence": list(dict.fromkeys(supervisor_phases)),
    "final_supervisor_phase": (
        supervisor_phases[-1] if supervisor_phases else None
    ),
    "num_inference_chunks": len(chunks),
    "executed_chunk_indices_match_interval": True,
    "chunk_origins_match_interval": True,
    "first_stage_id": stage_ids[0],
    "last_stage_id": stage_ids[-1],
    "unique_stage_ids": sorted(set(stage_ids)),
    "checkpoint": checkpoint_values[0],
}
output_path.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY

set +e
"$POLICY_PYTHON" "$ANALYZER" \
  --trace-log "$TRACE_LOG" \
  --chunk-log "$CHUNK_LOG" \
  --prior "$PRIOR" \
  --output-dir "$ANALYSIS_DIR" \
  --steps "$POLICY_STEPS" \
  > "$ANALYSIS_LOG" 2>&1
ANALYZER_STATUS=$?
set -e

printf 'analyzer_exit_status=%s\n' "$ANALYZER_STATUS" \
  > "$OUTPUT_DIR/trace_analysis_status.txt"

if [[ "$ANALYZER_STATUS" -ne 0 ]]; then
  echo "[WARN] Trace analyzer failed with exit code $ANALYZER_STATUS." >&2
  echo "[WARN] Packaging will continue; details: $ANALYSIS_LOG" >&2
  tail -n 80 "$ANALYSIS_LOG" >&2 || true
fi

echo
echo "======================================================================"
echo "5. Package the complete rollout"
echo "======================================================================"

tar -C "$(dirname "$OUTPUT_DIR")" \
  -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
ln -sfn "$OUTPUT_DIR" "$LATEST_OUTPUT_LINK"
ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"

echo "Validation: $OUTPUT_DIR/rollout_validation.json"
echo "Trace analysis: $ANALYSIS_DIR"
echo "Simulator log: $SIM_LOG"
echo "Policy client log: $CLIENT_LOG"
echo "Videos: $VIDEO_DIR"
echo "Archive: $ARCHIVE"
echo "Latest archive link: $LATEST_ARCHIVE_LINK"
echo
echo "Run one of these commands from your local computer:"
echo
echo "Download everything as one archive (videos, logs, traces, and analysis):"
echo "scp -P 30105 root@120.209.70.195:$ARCHIVE ."
echo
echo "Download the complete uncompressed output directory:"
echo "scp -P 30105 -r root@120.209.70.195:$OUTPUT_DIR ."
echo
echo "Download videos only:"
echo "scp -P 30105 -r root@120.209.70.195:$VIDEO_DIR ."

if [[ "$ANALYZER_STATUS" -ne 0 ]]; then
  echo
  echo "[ERROR] Rollout was packaged, but trace analysis failed." >&2
  exit "$ANALYZER_STATUS"
fi
