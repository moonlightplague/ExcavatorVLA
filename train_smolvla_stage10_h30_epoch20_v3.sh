#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python
TRAIN=/opt/conda/envs/smolvla/bin/lerobot-train

DATASET=/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3_stage10_h30_obslabels
BASE_MODEL=/root/gpufree-data/checkpoints/smolvla_base

OUTPUT_ROOT=/root/gpufree-data/outputs/train
LOG_ROOT=/root/gpufree-data/excavator_logs

JOB_NAME=excavator_smolvla_stage10_h30_obslabels_3cam_state27_effort4_lr5e5_b128_epoch20
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
echo "PRE-FLIGHT CHECK"
echo "============================================================"
echo "total frames: $TOTAL_FRAMES"
echo "batch size: $BATCH_SIZE"
echo "steps per epoch: $STEPS_PER_EPOCH"
echo "epochs: $EPOCHS"
echo "training steps: $TRAIN_STEPS"
echo "save frequency: $SAVE_FREQ steps (${SAVE_EVERY_EPOCHS} epochs)"

"$PYTHON" - "$DATASET" <<'PY'
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

root = Path(sys.argv[1])
assert root.exists(), root

cfg = SmolVLAConfig()
dataset = LeRobotDataset(str(root))
sample = dataset[0]

assert tuple(sample["observation.state"].shape) == (27,)
assert tuple(sample["observation.effort"].shape) == (4,)
assert tuple(sample["action"].shape) == (4,)
assert "observation.stage_target_30" in sample
assert "observation.stage_valid_30" in sample

print("dataset frames:", dataset.num_frames)
print("dataset episodes:", dataset.num_episodes)
print("observation.state:", tuple(sample["observation.state"].shape))
print("observation.effort:", tuple(sample["observation.effort"].shape))
print("effective state before 32D padding: 31D")
print("action:", tuple(sample["action"].shape))
print("stage target:", sample["observation.stage_target_30"])
print("stage valid:", sample["observation.stage_valid_30"])
print("num_stages:", cfg.num_stages)
print("stage_loss_weight:", cfg.stage_loss_weight)
PY

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
  --policy.stage_loss_weight=0.5 \
  --policy.stage_target_key=observation.stage_target_30 \
  --policy.stage_valid_key=observation.stage_valid_30 \
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
    echo
    echo "ERROR: Final output directory already exists:"
    echo "$OUTPUT_DIR"
    echo "Move or rename it before starting a new run."
    exit 1
fi

echo
echo "============================================================"
echo "STARTING 20-EPOCH TRAINING"
echo "============================================================"

nohup env PYTHONUNBUFFERED=1 \
"$TRAIN" \
  --dataset.repo_id="$DATASET" \
  --policy.path="$BASE_MODEL" \
  --policy.input_features="$POLICY_INPUT_FEATURES" \
  --policy.output_features="$POLICY_OUTPUT_FEATURES" \
  --policy.device=cuda \
  --policy.optimizer_lr=5e-5 \
  --policy.stage_loss_weight=0.5 \
  --policy.stage_target_key=observation.stage_target_30 \
  --policy.stage_valid_key=observation.stage_valid_30 \
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
echo "Follow policy loss metrics:"
echo "tail -f \"$LATEST_LOG\" | grep --line-buffered -E 'policy_metrics|loss|action_loss|stage_loss|stage_accuracy|stage_valid_fraction'"
