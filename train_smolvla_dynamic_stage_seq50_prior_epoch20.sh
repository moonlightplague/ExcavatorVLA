#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python
TRAIN=/opt/conda/envs/smolvla/bin/lerobot-train

# Reuse the existing dataset. No dataset conversion is performed.
DATASET=/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3_stage10_h30_obslabels
BASE_MODEL=/root/gpufree-data/checkpoints/smolvla_base

OUTPUT_ROOT=/root/gpufree-data/outputs/train
LOG_ROOT=/root/gpufree-data/excavator_logs

JOB_NAME=excavator_smolvla_dynamic_stage_seq50_exc10_prior005_norm_3cam_state27_effort4_lr5e5_b128_epoch20
OUTPUT_DIR="$OUTPUT_ROOT/$JOB_NAME"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/${JOB_NAME}_${TIMESTAMP}.log"
LATEST_LOG="$LOG_ROOT/${JOB_NAME}_latest.log"
PID_FILE="$LOG_ROOT/${JOB_NAME}.pid"

SMOKE_JOB="${JOB_NAME}_smoke"
SMOKE_OUTPUT="$OUTPUT_ROOT/$SMOKE_JOB"
SMOKE_LOG="$LOG_ROOT/${SMOKE_JOB}_${TIMESTAMP}.log"

BATCH_SIZE=128
EPOCHS=20
SAVE_EVERY_EPOCHS=4
STAGE_EXCEPTION_WEIGHT=10.0
STAGE_ACTION_LOSS_WEIGHT=0.20
STAGE_ACTION_MIN_PRIOR_WEIGHT=0.8
STAGE_ACTION_PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

POLICY_INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[27]},"observation.effort":{"type":"STATE","shape":[4]},"observation.images.0":{"type":"VISUAL","shape":[3,256,256]},"observation.images.1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.2":{"type":"VISUAL","shape":[3,256,256]}}'
POLICY_OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[4]}}'

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

read -r TOTAL_FRAMES STEPS_PER_EPOCH TRAIN_STEPS SAVE_FREQ < <(
"$PYTHON" - "$DATASET/meta/info.json" "$BATCH_SIZE" "$EPOCHS" "$SAVE_EVERY_EPOCHS" <<'PY'
import json
import math
import sys

info_path, batch_size, epochs, save_every = sys.argv[1:]
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

total_frames = int(info["total_frames"])
batch_size = int(batch_size)
epochs = int(epochs)
save_every = int(save_every)

steps_per_epoch = math.ceil(total_frames / batch_size)
train_steps = steps_per_epoch * epochs
save_freq = steps_per_epoch * save_every
print(total_frames, steps_per_epoch, train_steps, save_freq)
PY
)

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
    echo "Stop the previous training before starting this run."
    exit 1
fi

echo "GPU compute process list is empty."

echo
echo "============================================================"
echo "STAGE-ACTION PRIOR CHECK"
echo "============================================================"

if [[ ! -f "$STAGE_ACTION_PRIOR" ]]; then
    echo "Missing prepared prior:"
    echo "$STAGE_ACTION_PRIOR"
    echo "Run prepare_stage_action_prior_training.py first."
    exit 1
fi

"$PYTHON" - "$STAGE_ACTION_PRIOR" "$STAGE_ACTION_MIN_PRIOR_WEIGHT" <<'PY'
import json
import sys

path = sys.argv[1]
threshold = float(sys.argv[2])

with open(path, "r", encoding="utf-8") as f:
    prior = json.load(f)

direction = prior["direction"]
weight = prior["weight"]
reliable = prior["reliable"]

strong = 0
for stage_id in range(len(direction)):
    for action_id in range(len(direction[stage_id])):
        if (
            bool(reliable[stage_id][action_id])
            and float(weight[stage_id][action_id]) >= threshold
            and int(direction[stage_id][action_id]) != 0
        ):
            strong += 1

assert len(prior["action_mean"]) == 4
assert len(prior["action_std"]) == 4
assert len(prior["action_scale"]) == 4
assert strong > 0

print("prior:", path)
print("strong rules:", strong)
print("action mean:", prior["action_mean"])
print("action std:", prior["action_std"])
print("action scale:", prior["action_scale"])
PY

echo
echo "============================================================"
echo "DYNAMIC STAGE-SEQUENCE CHECK"
echo "============================================================"

"$PYTHON" - "$DATASET" "$BASE_MODEL" "$STAGE_EXCEPTION_WEIGHT" <<'PY'
import sys
from pathlib import Path

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

root = Path(sys.argv[1])
base = Path(sys.argv[2])
training_exception_weight = float(sys.argv[3])

cfg = SmolVLAConfig(device="cuda")
assert cfg.stage_source_key == "observation.stage_current_id"
assert cfg.stage_delta_indices == list(range(50))
assert cfg.stage_loss_weight == 1.0
assert training_exception_weight == 10.0

meta = LeRobotDatasetMetadata(str(root))
delta_timestamps = resolve_delta_timestamps(cfg, meta)

stage_key = cfg.stage_source_key
assert stage_key in delta_timestamps
assert len(delta_timestamps[stage_key]) == 50
assert len(delta_timestamps["action"]) == 50

dataset = LeRobotDataset(
    str(root),
    delta_timestamps=delta_timestamps,
)
sample = dataset[0]

state_shape = tuple(sample["observation.state"].shape)
effort_shape = tuple(sample["observation.effort"].shape)
action_shape = tuple(sample["action"].shape)
stage_shape = tuple(sample[stage_key].shape)
stage_pad_shape = tuple(sample[f"{stage_key}_is_pad"].shape)

print("raw observation.state shape:", state_shape)
print("raw observation.effort shape:", effort_shape)
print("raw action shape:", action_shape)
print("raw stage shape:", stage_shape)
print("raw stage pad shape:", stage_pad_shape)

# Observation features queried at only the current timestamp may contain a
# singleton temporal dimension before the policy preprocessor squeezes it.
assert state_shape in ((27,), (1, 27)), state_shape
assert effort_shape in ((4,), (1, 4)), effort_shape
assert action_shape == (50, 4), action_shape
assert stage_shape in ((50,), (50, 1)), stage_shape
assert stage_pad_shape == (50,), stage_pad_shape

print("dataset frames:", dataset.num_frames)
print("dataset episodes:", dataset.num_episodes)
print("observation.state:", tuple(sample["observation.state"].shape))
print("observation.effort:", tuple(sample["observation.effort"].shape))
print("effective state before padding: 31D")
print("action chunk:", tuple(sample["action"].shape))
print("stage chunk:", tuple(sample[stage_key].shape))
print("stage pad mask:", tuple(sample[f"{stage_key}_is_pad"].shape))
print("stage loss weight:", cfg.stage_loss_weight)
print("exception weight:", training_exception_weight)
print("EMA beta:", cfg.stage_loss_ema_beta)
PY

echo
echo "============================================================"
echo "TRAINING PLAN"
echo "============================================================"
echo "total frames: $TOTAL_FRAMES"
echo "batch size: $BATCH_SIZE"
echo "steps per epoch: $STEPS_PER_EPOCH"
echo "epochs: $EPOCHS"
echo "training steps: $TRAIN_STEPS"
echo "save frequency: $SAVE_FREQ steps"
echo "stage exception weight: $STAGE_EXCEPTION_WEIGHT"
echo "stage-action prior: $STAGE_ACTION_PRIOR"
echo "stage-action loss weight: $STAGE_ACTION_LOSS_WEIGHT"
echo "stage-action minimum prior weight: $STAGE_ACTION_MIN_PRIOR_WEIGHT"

echo
echo "============================================================"
echo "2-STEP SMOKE TEST"
echo "============================================================"

rm -rf "$SMOKE_OUTPUT"

"$TRAIN" \
  --dataset.repo_id="$DATASET" \
  --policy.path="$BASE_MODEL" \
  --policy.input_features="$POLICY_INPUT_FEATURES" \
  --policy.output_features="$POLICY_OUTPUT_FEATURES" \
  --policy.device=cuda \
  --policy.optimizer_lr=5e-5 \
  --policy.stage_loss_weight=1.0 \
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
  --policy.push_to_hub=false \
  --output_dir="$SMOKE_OUTPUT" \
  --job_name="$SMOKE_JOB" \
  --batch_size=2 \
  --steps=2 \
  --num_workers=0 \
  --log_freq=1 \
  --save_freq=2 \
  --wandb.enable=false \
  2>&1 | tee "$SMOKE_LOG"

echo
echo "Smoke test completed successfully."
echo "Smoke log: $SMOKE_LOG"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Final output directory already exists:"
    echo "$OUTPUT_DIR"
    exit 1
fi

echo
echo "============================================================"
echo "STARTING 20-EPOCH TRAINING"
echo "============================================================"

nohup env \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TRAIN" \
  --dataset.repo_id="$DATASET" \
  --policy.path="$BASE_MODEL" \
  --policy.input_features="$POLICY_INPUT_FEATURES" \
  --policy.output_features="$POLICY_OUTPUT_FEATURES" \
  --policy.device=cuda \
  --policy.optimizer_lr=5e-5 \
  --policy.stage_loss_weight=0.5 \
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
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --steps="$TRAIN_STEPS" \
  --num_workers=8 \
  --log_freq=10 \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
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
echo "Follow all output:"
echo "tail -f \"$LATEST_LOG\""
echo
echo "Follow detailed losses:"
echo "tail -f \"$LATEST_LOG\" | grep --line-buffered -E 'policy_metrics|action_loss|stage_ce_loss|stage_exception_loss|stage_raw_loss|stage_scale|stage_action_raw_loss|stage_action_scale|normalized_stage_action_loss|weighted_stage_action_loss|stage_action_prior_coverage|stage_action_violation_rate|loss'"
