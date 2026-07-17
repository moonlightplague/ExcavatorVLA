#!/usr/bin/env bash
set -euo pipefail

# True LeRobot resume from the checkpoint whose saved training step is 3840.
# The source checkpoint is copied into a new run directory, so the original
# run and checkpoint are never modified.
#
# Phase 2 changes:
#   - preserve the existing direction/sign stage-action loss
#   - enable the validated swing low-motion loss for GT stages 1..7
#   - continue for 30 additional dataset epochs
#   - test batch sizes 152, 148, 144, 140, 136, and 128 in that order
#   - run a two-step GPU memory smoke test for each candidate
#   - use the first candidate that passes

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python
TRAIN=/opt/conda/envs/smolvla/bin/lerobot-train

DATASET=/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3_stage10_h30_obslabels
OUTPUT_ROOT=/root/gpufree-data/outputs/train
LOG_ROOT=/root/gpufree-data/excavator_logs

# The active run used this older job name even though its actual loss weights
# were stage_loss_weight=0.5 and stage_action_loss_weight=0.2.
DEFAULT_SOURCE_OUTPUT="$OUTPUT_ROOT/excavator_smolvla_dynamic_stage_seq50_exc10_prior005_norm_3cam_state27_effort4_lr5e5_b128_epoch20"

# Override at launch when needed:
# SOURCE_OUTPUT=/another/run/path bash resume_smolvla_lowmotion_from3840_add30epochs.sh
SOURCE_OUTPUT="${SOURCE_OUTPUT:-$DEFAULT_SOURCE_OUTPUT}"

RESUME_STEP=3840
ADDITIONAL_EPOCHS=30

BATCH_CANDIDATES=(144 140 136 128)
NUM_WORKERS=8

# Keep the original global checkpoint cadence in optimizer-step units.
# The exact number of dataset epochs between checkpoints depends on the
# automatically selected batch size. A final checkpoint is always saved at
# TOTAL_STEPS.
SAVE_FREQ=1920
LOG_FREQ=10

STAGE_LOSS_WEIGHT=0.5
STAGE_EXCEPTION_WEIGHT=10.0
STAGE_ACTION_LOSS_WEIGHT=0.20
STAGE_ACTION_MIN_PRIOR_WEIGHT=0.8
STAGE_ACTION_PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

LOW_MOTION_WEIGHT=0.5
LOW_MOTION_MARGIN=0.10
LOW_MOTION_SWING_SCALE=2.1780849

POLICY_INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[27]},"observation.effort":{"type":"STATE","shape":[4]},"observation.images.0":{"type":"VISUAL","shape":[3,256,256]},"observation.images.1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.2":{"type":"VISUAL","shape":[3,256,256]}}'
POLICY_OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[4]}}'

SMOLVLA_DIR=/opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/policies/smolvla
CONFIG_PY="$SMOLVLA_DIR/configuration_smolvla.py"
MODELING_PY="$SMOLVLA_DIR/modeling_smolvla.py"

cd "$PROJECT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

unset PYTHONPATH PYTHONHOME LD_PRELOAD
unset ISAAC_PATH EXP_PATH CARB_APP_PATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/gpufree-data/hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "PATCH AND SOURCE CHECK"
echo "============================================================"

if [[ ! -f "$CONFIG_PY" || ! -f "$MODELING_PY" ]]; then
    echo "Missing SmolVLA source files under:"
    echo "$SMOLVLA_DIR"
    exit 1
fi

if ! grep -q 'stage_action_low_motion_weight' "$CONFIG_PY"; then
    echo "The low-motion configuration patch is not installed."
    echo "Run:"
    echo "$PYTHON $PROJECT/patch_smolvla_stage_action_low_motion_loss.py"
    exit 1
fi

if ! grep -q 'stage_action_low_motion_raw_loss' "$MODELING_PY"; then
    echo "The low-motion modeling patch is not installed."
    echo "Run:"
    echo "$PYTHON $PROJECT/patch_smolvla_stage_action_low_motion_loss.py"
    exit 1
fi

"$PYTHON" -m py_compile "$CONFIG_PY" "$MODELING_PY"

if [[ ! -d "$SOURCE_OUTPUT/checkpoints" ]]; then
    echo "Source run checkpoints directory does not exist:"
    echo "$SOURCE_OUTPUT/checkpoints"
    exit 1
fi

if [[ ! -f "$STAGE_ACTION_PRIOR" ]]; then
    echo "Missing stage-action prior:"
    echo "$STAGE_ACTION_PRIOR"
    exit 1
fi

# Locate the exact checkpoint by reading training_state/training_step.json,
# instead of assuming a particular zero-padded directory name.
SOURCE_CHECKPOINT="$(
"$PYTHON" - "$SOURCE_OUTPUT" "$RESUME_STEP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = int(sys.argv[2])
matches = []

for checkpoint in sorted((root / "checkpoints").iterdir()):
    # Ignore "last" and any other symlink that resolves to a real checkpoint.
    if checkpoint.is_symlink():
        continue
    if not checkpoint.is_dir():
        continue
    state_file = checkpoint / "training_state" / "training_step.json"
    if not state_file.is_file():
        continue
    try:
        step = int(json.loads(state_file.read_text())["step"])
    except Exception:
        continue
    if step == target:
        matches.append(checkpoint.resolve())

if len(matches) != 1:
    print(
        f"Expected exactly one checkpoint at step {target}, found {len(matches)}",
        file=sys.stderr,
    )
    for match in matches:
        print(match, file=sys.stderr)
    raise SystemExit(1)

print(matches[0])
PY
)"

SOURCE_PRETRAINED="$SOURCE_CHECKPOINT/pretrained_model"
SOURCE_TRAIN_CONFIG="$SOURCE_PRETRAINED/train_config.json"

for required in \
    "$SOURCE_PRETRAINED/model.safetensors" \
    "$SOURCE_PRETRAINED/config.json" \
    "$SOURCE_TRAIN_CONFIG" \
    "$SOURCE_CHECKPOINT/training_state/training_step.json" \
    "$SOURCE_CHECKPOINT/training_state/optimizer_param_groups.json" \
    "$SOURCE_CHECKPOINT/training_state/optimizer_state.safetensors" \
    "$SOURCE_CHECKPOINT/training_state/scheduler_state.json"
do
    if [[ ! -f "$required" ]]; then
        echo "Incomplete checkpoint; missing:"
        echo "$required"
        exit 1
    fi
done

"$PYTHON" - "$SOURCE_TRAIN_CONFIG" "$RESUME_STEP" <<'PY'
import json
import math
import sys

path = sys.argv[1]
target_step = int(sys.argv[2])
cfg = json.load(open(path, "r", encoding="utf-8"))
policy = cfg["policy"]

def require_close(name, actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")

require_close("policy.stage_loss_weight", policy["stage_loss_weight"], 0.5)
require_close(
    "policy.stage_action_loss_weight",
    policy["stage_action_loss_weight"],
    0.2,
)
require_close(
    "policy.stage_exception_weight",
    policy["stage_exception_weight"],
    10.0,
)

print("source train config:", path)
print("source output_dir:", cfg["output_dir"])
print("source configured steps:", cfg["steps"])
print("source batch size:", cfg["batch_size"])
print("resume checkpoint step:", target_step)
print("stage loss weight:", policy["stage_loss_weight"])
print("stage-action loss weight:", policy["stage_action_loss_weight"])
print("stage exception weight:", policy["stage_exception_weight"])
PY

echo "Source checkpoint: $SOURCE_CHECKPOINT"

echo
echo "============================================================"
echo "GPU CHECK"
echo "============================================================"

GPU_PROCESSES="$(
    nvidia-smi \
      --query-compute-apps=pid,used_memory,process_name \
      --format=csv,noheader 2>/dev/null || true
)"

if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
    echo "GPU still has compute processes:"
    echo "$GPU_PROCESSES"
    echo
    echo "Stop the previous training process before starting this resume run."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader || true

run_smoke_test() {
    local batch_size="$1"
    local smoke_job="smolvla_lowmotion_resume3840_smoke_b${batch_size}"
    local smoke_output="$OUTPUT_ROOT/$smoke_job"
    local smoke_log="$LOG_ROOT/${smoke_job}_$(date +%Y%m%d_%H%M%S).log"

    rm -rf "$smoke_output"

    echo
    echo "Testing batch size $batch_size ..."
    if "$TRAIN" \
      --dataset.repo_id="$DATASET" \
      --policy.path="$SOURCE_PRETRAINED" \
      --policy.input_features="$POLICY_INPUT_FEATURES" \
      --policy.output_features="$POLICY_OUTPUT_FEATURES" \
      --policy.device=cuda \
      --policy.optimizer_lr=5e-5 \
      --policy.stage_loss_weight="$STAGE_LOSS_WEIGHT" \
      --policy.stage_exception_weight="$STAGE_EXCEPTION_WEIGHT" \
      --policy.stage_loss_ema_beta=0.99 \
      --policy.stage_scale_min=0.02 \
      --policy.stage_scale_max=20.0 \
      --policy.stage_action_prior_path="$STAGE_ACTION_PRIOR" \
      --policy.stage_action_loss_weight="$STAGE_ACTION_LOSS_WEIGHT" \
      --policy.stage_action_min_prior_weight="$STAGE_ACTION_MIN_PRIOR_WEIGHT" \
      --policy.stage_action_loss_ema_beta=0.99 \
      --policy.stage_action_scale_min=0.02 \
      --policy.stage_action_scale_max=20.0 \
      --policy.stage_action_low_motion_weight="$LOW_MOTION_WEIGHT" \
      --policy.stage_action_low_motion_margin="$LOW_MOTION_MARGIN" \
      --policy.stage_action_low_motion_swing_scale="$LOW_MOTION_SWING_SCALE" \
      --policy.push_to_hub=false \
      --output_dir="$smoke_output" \
      --job_name="$smoke_job" \
      --batch_size="$batch_size" \
      --steps=2 \
      --num_workers=0 \
      --log_freq=1 \
      --save_freq=2 \
      --wandb.enable=false \
      > "$smoke_log" 2>&1
    then
        echo "Batch size $batch_size smoke test passed."
        echo "Smoke log: $smoke_log"
        rm -rf "$smoke_output"
        return 0
    fi

    echo "Batch size $batch_size smoke test failed."
    echo "Last 100 log lines:"
    tail -100 "$smoke_log" || true
    rm -rf "$smoke_output"
    return 1
}

echo
echo "============================================================"
echo "GPU MEMORY SMOKE TEST"
echo "============================================================"
echo "Candidate order: ${BATCH_CANDIDATES[*]}"

BATCH_SIZE=""

for candidate in "${BATCH_CANDIDATES[@]}"; do
    if run_smoke_test "$candidate"; then
        BATCH_SIZE="$candidate"
        break
    fi
    echo
    echo "Batch size $candidate did not pass; trying the next candidate."
done

if [[ -z "$BATCH_SIZE" ]]; then
    echo "No batch-size candidate passed the two-step smoke test."
    echo "Resume run was not created."
    exit 1
fi

echo
echo "Selected batch size: $BATCH_SIZE"

read -r TOTAL_FRAMES STEPS_PER_EPOCH ADDITIONAL_STEPS TOTAL_STEPS < <(
"$PYTHON" - \
  "$DATASET/meta/info.json" \
  "$BATCH_SIZE" \
  "$ADDITIONAL_EPOCHS" \
  "$RESUME_STEP" <<'PY'
import json
import math
import sys

info_path, batch_size, epochs, resume_step = sys.argv[1:]
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

total_frames = int(info["total_frames"])
batch_size = int(batch_size)
epochs = int(epochs)
resume_step = int(resume_step)

steps_per_epoch = math.ceil(total_frames / batch_size)
additional_steps = steps_per_epoch * epochs
total_steps = resume_step + additional_steps

print(total_frames, steps_per_epoch, additional_steps, total_steps)
PY
)

JOB_NAME="excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b${BATCH_SIZE}_add30ep"
OUTPUT_DIR="$OUTPUT_ROOT/$JOB_NAME"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/${JOB_NAME}_${TIMESTAMP}.log"
LATEST_LOG="$LOG_ROOT/${JOB_NAME}_latest.log"
PID_FILE="$LOG_ROOT/${JOB_NAME}.pid"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Target output directory already exists:"
    echo "$OUTPUT_DIR"
    echo "Nothing was overwritten."
    exit 1
fi

echo
echo "============================================================"
echo "PREPARING ISOLATED RESUME RUN"
echo "============================================================"

mkdir -p "$OUTPUT_DIR/checkpoints"

CHECKPOINT_NAME="$(basename "$SOURCE_CHECKPOINT")"
TARGET_CHECKPOINT="$OUTPUT_DIR/checkpoints/$CHECKPOINT_NAME"

cp -a "$SOURCE_CHECKPOINT" "$TARGET_CHECKPOINT"
ln -sfn "$CHECKPOINT_NAME" "$OUTPUT_DIR/checkpoints/last"

TARGET_PRETRAINED="$TARGET_CHECKPOINT/pretrained_model"
TARGET_TRAIN_CONFIG="$TARGET_PRETRAINED/train_config.json"
TARGET_POLICY_CONFIG="$TARGET_PRETRAINED/config.json"

# Patch only the copied checkpoint configuration.
# The original checkpoint remains untouched.
"$PYTHON" - \
  "$TARGET_TRAIN_CONFIG" \
  "$TARGET_POLICY_CONFIG" \
  "$OUTPUT_DIR" \
  "$JOB_NAME" \
  "$BATCH_SIZE" \
  "$TOTAL_STEPS" \
  "$SAVE_FREQ" \
  "$LOG_FREQ" \
  "$NUM_WORKERS" \
  "$STAGE_LOSS_WEIGHT" \
  "$STAGE_EXCEPTION_WEIGHT" \
  "$STAGE_ACTION_LOSS_WEIGHT" \
  "$STAGE_ACTION_MIN_PRIOR_WEIGHT" \
  "$STAGE_ACTION_PRIOR" \
  "$LOW_MOTION_WEIGHT" \
  "$LOW_MOTION_MARGIN" \
  "$LOW_MOTION_SWING_SCALE" <<'PY'
import json
import sys
from pathlib import Path

(
    train_config_path,
    policy_config_path,
    output_dir,
    job_name,
    batch_size,
    total_steps,
    save_freq,
    log_freq,
    num_workers,
    stage_loss_weight,
    stage_exception_weight,
    stage_action_loss_weight,
    stage_action_min_prior_weight,
    stage_action_prior_path,
    low_motion_weight,
    low_motion_margin,
    low_motion_swing_scale,
) = sys.argv[1:]

train_path = Path(train_config_path)
policy_path = Path(policy_config_path)

train_cfg = json.loads(train_path.read_text(encoding="utf-8"))
policy_cfg = train_cfg["policy"]

train_cfg["output_dir"] = output_dir
train_cfg["job_name"] = job_name
train_cfg["resume"] = True
train_cfg["batch_size"] = int(batch_size)
train_cfg["steps"] = int(total_steps)
train_cfg["save_freq"] = int(save_freq)
train_cfg["log_freq"] = int(log_freq)
train_cfg["num_workers"] = int(num_workers)
train_cfg["save_checkpoint"] = True
train_cfg.setdefault("wandb", {})["enable"] = False

policy_cfg["stage_loss_weight"] = float(stage_loss_weight)
policy_cfg["stage_exception_weight"] = float(stage_exception_weight)
policy_cfg["stage_action_loss_weight"] = float(stage_action_loss_weight)
policy_cfg["stage_action_min_prior_weight"] = float(stage_action_min_prior_weight)
policy_cfg["stage_action_prior_path"] = stage_action_prior_path

policy_cfg["stage_action_low_motion_weight"] = float(low_motion_weight)
policy_cfg["stage_action_low_motion_margin"] = float(low_motion_margin)
policy_cfg["stage_action_low_motion_swing_scale"] = float(
    low_motion_swing_scale
)
policy_cfg["push_to_hub"] = False

train_path.write_text(
    json.dumps(train_cfg, indent=4) + "\n",
    encoding="utf-8",
)

# Keep the copied policy config self-describing as well.
standalone_policy_cfg = json.loads(policy_path.read_text(encoding="utf-8"))
standalone_policy_cfg["stage_action_low_motion_weight"] = float(
    low_motion_weight
)
standalone_policy_cfg["stage_action_low_motion_margin"] = float(
    low_motion_margin
)
standalone_policy_cfg["stage_action_low_motion_swing_scale"] = float(
    low_motion_swing_scale
)
standalone_policy_cfg["stage_action_loss_weight"] = float(
    stage_action_loss_weight
)
standalone_policy_cfg["stage_loss_weight"] = float(stage_loss_weight)
standalone_policy_cfg["stage_exception_weight"] = float(
    stage_exception_weight
)
standalone_policy_cfg["stage_action_prior_path"] = stage_action_prior_path
standalone_policy_cfg["stage_action_min_prior_weight"] = float(
    stage_action_min_prior_weight
)
standalone_policy_cfg["push_to_hub"] = False

policy_path.write_text(
    json.dumps(standalone_policy_cfg, indent=4) + "\n",
    encoding="utf-8",
)

print("patched copied train config:", train_path)
print("patched copied policy config:", policy_path)
print("output_dir:", train_cfg["output_dir"])
print("batch_size:", train_cfg["batch_size"])
print("total target steps:", train_cfg["steps"])
print(
    "low-motion:",
    policy_cfg["stage_action_low_motion_weight"],
    policy_cfg["stage_action_low_motion_margin"],
    policy_cfg["stage_action_low_motion_swing_scale"],
)
PY

echo "source checkpoint: $SOURCE_CHECKPOINT"
echo "copied checkpoint: $TARGET_CHECKPOINT"
echo "copied train config: $TARGET_TRAIN_CONFIG"
echo
echo "total frames: $TOTAL_FRAMES"
echo "selected batch size: $BATCH_SIZE"
echo "steps per additional epoch: $STEPS_PER_EPOCH"
echo "additional epochs: $ADDITIONAL_EPOCHS"
echo "additional optimizer steps: $ADDITIONAL_STEPS"
echo "resume step: $RESUME_STEP"
echo "final global step: $TOTAL_STEPS"
echo "save frequency: $SAVE_FREQ global steps"
echo
echo "loss configuration:"
echo "  stage loss weight: $STAGE_LOSS_WEIGHT"
echo "  direction + low-motion outer weight: $STAGE_ACTION_LOSS_WEIGHT"
echo "  low-motion internal weight: $LOW_MOTION_WEIGHT"
echo "  low-motion normalized margin: $LOW_MOTION_MARGIN"
echo "  low-motion swing scale: $LOW_MOTION_SWING_SCALE"

echo
echo "============================================================"
echo "STARTING TRUE RESUME"
echo "============================================================"

# LeRobot loads model, optimizer, scheduler, RNG state, and the saved global
# step from TARGET_CHECKPOINT because config_path points directly to that
# checkpoint's pretrained_model/train_config.json.
nohup env \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TRAIN" \
  --config_path="$TARGET_TRAIN_CONFIG" \
  --resume=true \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
ln -sfn "$LOG_FILE" "$LATEST_LOG"

echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Latest log symlink: $LATEST_LOG"
echo
echo "Follow training:"
echo "tail -f \"$LATEST_LOG\""
echo
echo "Confirm the resume step and new configuration:"
echo "grep -E 'cfg.steps=|Output dir:|stage_action_low_motion|Training' \"$LATEST_LOG\" | head -80"
echo
echo "Follow the useful losses:"
echo "tail -f \"$LATEST_LOG\" | grep --line-buffered -E 'policy_metrics|stage_action_direction_raw_loss|stage_action_low_motion_raw_loss|stage_action_low_motion_violation_rate|stage_action_low_motion_abs_swing_mean|stage_action_violation_rate|weighted_stage_action_loss|action_loss|stage_accuracy|loss'"
