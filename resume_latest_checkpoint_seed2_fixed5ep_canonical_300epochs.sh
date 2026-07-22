#!/usr/bin/env bash
set -euo pipefail

# Continue the latest complete checkpoint on the five canonical-stage seed-2
# episodes for 300 dataset epochs. The source run is never modified.

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python
TRAIN=/opt/conda/envs/smolvla/bin/lerobot-train

DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
OUTPUT_ROOT=/root/gpufree-data/outputs/train
LOG_ROOT=/root/gpufree-data/excavator_logs

DEFAULT_SOURCE_OUTPUT="$OUTPUT_ROOT/excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b144_add30ep"
SOURCE_OUTPUT="${SOURCE_OUTPUT:-$DEFAULT_SOURCE_OUTPUT}"

ADDITIONAL_EPOCHS=300
BATCH_SIZE=144
NUM_WORKERS=8
SAVE_EVERY_EPOCHS=50
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

echo "======================================================================"
echo "1. Validate dataset and installed SmolVLA patches"
echo "======================================================================"

for required in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/stage_schema.json" \
  "$SOURCE_OUTPUT/checkpoints" \
  "$STAGE_ACTION_PRIOR" \
  "$CONFIG_PY" \
  "$MODELING_PY"
do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 1
  fi
done

if ! grep -q 'stage_action_low_motion_weight' "$CONFIG_PY"; then
  echo "[ERROR] Low-motion configuration patch is not installed." >&2
  echo "Run: $PYTHON $PROJECT/scripts/training/patch_smolvla_stage_action_low_motion_loss.py" >&2
  exit 1
fi
if ! grep -q 'stage_action_low_motion_raw_loss' "$MODELING_PY"; then
  echo "[ERROR] Low-motion modeling patch is not installed." >&2
  echo "Run: $PYTHON $PROJECT/scripts/training/patch_smolvla_stage_action_low_motion_loss.py" >&2
  exit 1
fi
"$PYTHON" -m py_compile "$CONFIG_PY" "$MODELING_PY"

"$PYTHON" - "$DATASET" <<'PY'
from pathlib import Path
import json
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1]).resolve()
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
schema = json.loads(
    (root / "meta" / "stage_schema.json").read_text(encoding="utf-8")
)
expected = {
    "0": "pre_dig",
    "1": "approach_contact",
    "2": "insert_cut",
    "3": "pull_mid_cut",
    "4": "curl_to_hold_material",
    "5": "pull_exit_cut",
    "6": "secure_load",
    "7": "lift_carry",
    "8": "loaded_transit",
    "9": "unload_to_bin",
}
if schema.get("classes") != expected:
    raise RuntimeError(
        f"Non-canonical stage schema: {schema.get('classes')!r}"
    )
if int(info.get("total_frames", -1)) != 810:
    raise RuntimeError(f"Expected 810 frames, got {info.get('total_frames')}")
if int(info.get("total_episodes", -1)) != 5:
    raise RuntimeError(
        f"Expected five episodes, got {info.get('total_episodes')}"
    )
if info["features"]["observation.state"]["shape"] != [27]:
    raise RuntimeError("Dataset observation.state is not 27D")
if "observation.stage_current_id" not in info["features"]:
    raise RuntimeError("Dataset lacks observation.stage_current_id")

dataset = LeRobotDataset(repo_id=root.name, root=root, download_videos=False)
sample = dataset[0]
for key, shape in {
    "observation.state": (27,),
    "observation.effort": (4,),
    "action": (4,),
    "observation.images.0": (3, 256, 256),
    "observation.images.1": (3, 256, 256),
    "observation.images.2": (3, 256, 256),
}.items():
    actual = tuple(sample[key].shape)
    if actual != shape:
        raise RuntimeError(f"{key}: expected {shape}, got {actual}")

print("dataset:", root)
print("frames:", info["total_frames"])
print("episodes:", info["total_episodes"])
print("fps:", info["fps"])
print("stages:", schema["classes"])
print("task:", sample.get("task"))
PY

echo
echo "======================================================================"
echo "2. Locate the latest complete source checkpoint"
echo "======================================================================"

read -r SOURCE_STEP SOURCE_CHECKPOINT < <(
"$PYTHON" - "$SOURCE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
candidates = []

for checkpoint in sorted((root / "checkpoints").iterdir()):
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        continue
    state = checkpoint / "training_state" / "training_step.json"
    required = (
        checkpoint / "pretrained_model" / "model.safetensors",
        checkpoint / "pretrained_model" / "config.json",
        checkpoint / "pretrained_model" / "train_config.json",
        checkpoint / "training_state" / "optimizer_param_groups.json",
        checkpoint / "training_state" / "optimizer_state.safetensors",
        checkpoint / "training_state" / "scheduler_state.json",
    )
    if not state.is_file() or not all(path.is_file() for path in required):
        continue
    try:
        step = int(json.loads(state.read_text(encoding="utf-8"))["step"])
    except Exception:
        continue
    candidates.append((step, checkpoint.resolve()))

if not candidates:
    raise SystemExit(f"No complete checkpoint found under {root / 'checkpoints'}")

step, checkpoint = max(candidates, key=lambda item: item[0])
print(step, checkpoint)
PY
)

SOURCE_PRETRAINED="$SOURCE_CHECKPOINT/pretrained_model"
SOURCE_TRAIN_CONFIG="$SOURCE_PRETRAINED/train_config.json"

"$PYTHON" - "$SOURCE_TRAIN_CONFIG" "$SOURCE_STEP" <<'PY'
import json
import math
import sys

path, expected_step = sys.argv[1:]
cfg = json.load(open(path, "r", encoding="utf-8"))
policy = cfg["policy"]

def require_close(name, expected):
    actual = float(policy[name])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{name}: expected {expected}, got {actual}")

require_close("stage_loss_weight", 0.5)
require_close("stage_exception_weight", 10.0)
require_close("stage_action_loss_weight", 0.2)
require_close("stage_action_low_motion_weight", 0.5)
require_close("stage_action_low_motion_margin", 0.1)

if policy.get("stage_source_key") != "observation.stage_current_id":
    raise RuntimeError(
        f"Unexpected stage_source_key: {policy.get('stage_source_key')!r}"
    )
if list(policy["input_features"]["observation.state"]["shape"]) != [27]:
    raise RuntimeError("Source checkpoint is not a 27D-state checkpoint")

print("source train config:", path)
print("source saved step:", expected_step)
print("source configured steps:", cfg["steps"])
print("source dataset:", cfg["dataset"]["repo_id"])
print("source batch size:", cfg["batch_size"])
print("source optimizer lr:", policy["optimizer_lr"])
PY

echo "Latest checkpoint: $SOURCE_CHECKPOINT"
echo "Latest saved step: $SOURCE_STEP"

echo
echo "======================================================================"
echo "3. Check GPU and run a two-step smoke test"
echo "======================================================================"

GPU_PROCESSES="$(
  nvidia-smi --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "[ERROR] GPU still has compute processes:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

SMOKE_JOB="smolvla_seed2_fixed5ep_canonical300_smoke_b${BATCH_SIZE}"
SMOKE_OUTPUT="$OUTPUT_ROOT/$SMOKE_JOB"
SMOKE_LOG="$LOG_ROOT/${SMOKE_JOB}_$(date +%Y%m%d_%H%M%S).log"

rm -rf -- "$SMOKE_OUTPUT"

if ! "$TRAIN" \
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
  --output_dir="$SMOKE_OUTPUT" \
  --job_name="$SMOKE_JOB" \
  --batch_size="$BATCH_SIZE" \
  --steps=2 \
  --num_workers=0 \
  --log_freq=1 \
  --save_freq=2 \
  --wandb.enable=false \
  > "$SMOKE_LOG" 2>&1
then
  echo "[ERROR] Two-step smoke test failed. Last 120 lines:" >&2
  tail -120 "$SMOKE_LOG" >&2 || true
  rm -rf -- "$SMOKE_OUTPUT"
  exit 1
fi

rm -rf -- "$SMOKE_OUTPUT"
echo "Smoke test passed: $SMOKE_LOG"

echo
echo "======================================================================"
echo "4. Calculate the 300-epoch continuation schedule"
echo "======================================================================"

read -r TOTAL_FRAMES STEPS_PER_EPOCH ADDITIONAL_STEPS TOTAL_STEPS SAVE_FREQ < <(
"$PYTHON" - \
  "$DATASET/meta/info.json" \
  "$BATCH_SIZE" \
  "$ADDITIONAL_EPOCHS" \
  "$SOURCE_STEP" \
  "$SAVE_EVERY_EPOCHS" <<'PY'
import json
import math
import sys

info_path, batch_size, epochs, source_step, save_epochs = sys.argv[1:]
info = json.load(open(info_path, "r", encoding="utf-8"))
total_frames = int(info["total_frames"])
batch_size = int(batch_size)
epochs = int(epochs)
source_step = int(source_step)
save_epochs = int(save_epochs)

steps_per_epoch = math.ceil(total_frames / batch_size)
additional_steps = steps_per_epoch * epochs
total_steps = source_step + additional_steps
save_freq = steps_per_epoch * save_epochs
print(total_frames, steps_per_epoch, additional_steps, total_steps, save_freq)
PY
)

JOB_NAME="excavator_smolvla_seed2_fixed5ep_canonical300_from${SOURCE_STEP}_b${BATCH_SIZE}"
OUTPUT_DIR="$OUTPUT_ROOT/$JOB_NAME"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/${JOB_NAME}_${TIMESTAMP}.log"
LATEST_LOG="$LOG_ROOT/${JOB_NAME}_latest.log"
PID_FILE="$LOG_ROOT/${JOB_NAME}.pid"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[ERROR] Target output already exists; nothing was overwritten:" >&2
  echo "$OUTPUT_DIR" >&2
  exit 1
fi

echo "frames: $TOTAL_FRAMES"
echo "batch size: $BATCH_SIZE"
echo "steps per epoch: $STEPS_PER_EPOCH"
echo "additional epochs: $ADDITIONAL_EPOCHS"
echo "additional steps: $ADDITIONAL_STEPS"
echo "source step: $SOURCE_STEP"
echo "final target step: $TOTAL_STEPS"
echo "save every: $SAVE_FREQ steps ($SAVE_EVERY_EPOCHS epochs)"

echo
echo "======================================================================"
echo "5. Create an isolated true-resume run"
echo "======================================================================"

mkdir -p "$OUTPUT_DIR/checkpoints"
CHECKPOINT_NAME="$(basename "$SOURCE_CHECKPOINT")"
TARGET_CHECKPOINT="$OUTPUT_DIR/checkpoints/$CHECKPOINT_NAME"
cp -a "$SOURCE_CHECKPOINT" "$TARGET_CHECKPOINT"
ln -sfn "$CHECKPOINT_NAME" "$OUTPUT_DIR/checkpoints/last"

TARGET_PRETRAINED="$TARGET_CHECKPOINT/pretrained_model"
TARGET_TRAIN_CONFIG="$TARGET_PRETRAINED/train_config.json"
TARGET_POLICY_CONFIG="$TARGET_PRETRAINED/config.json"

"$PYTHON" - \
  "$TARGET_TRAIN_CONFIG" \
  "$TARGET_POLICY_CONFIG" \
  "$DATASET" \
  "$OUTPUT_DIR" \
  "$JOB_NAME" \
  "$BATCH_SIZE" \
  "$TOTAL_STEPS" \
  "$SAVE_FREQ" \
  "$LOG_FREQ" \
  "$NUM_WORKERS" <<'PY'
import json
import sys
from pathlib import Path

(
    train_config_path,
    policy_config_path,
    dataset,
    output_dir,
    job_name,
    batch_size,
    total_steps,
    save_freq,
    log_freq,
    num_workers,
) = sys.argv[1:]

train_path = Path(train_config_path)
policy_path = Path(policy_config_path)
train_cfg = json.loads(train_path.read_text(encoding="utf-8"))

dataset_cfg = train_cfg.setdefault("dataset", {})
dataset_cfg["repo_id"] = dataset
if "root" in dataset_cfg:
    dataset_cfg["root"] = dataset

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
train_cfg["policy"]["push_to_hub"] = False

train_path.write_text(
    json.dumps(train_cfg, indent=4) + "\n",
    encoding="utf-8",
)

policy_cfg = json.loads(policy_path.read_text(encoding="utf-8"))
policy_cfg["push_to_hub"] = False
policy_path.write_text(
    json.dumps(policy_cfg, indent=4) + "\n",
    encoding="utf-8",
)

print("patched copied train config:", train_path)
print("dataset:", train_cfg["dataset"])
print("output_dir:", train_cfg["output_dir"])
print("steps:", train_cfg["steps"])
print("batch_size:", train_cfg["batch_size"])
print("save_freq:", train_cfg["save_freq"])
PY

echo "Source checkpoint (unchanged): $SOURCE_CHECKPOINT"
echo "Copied checkpoint: $TARGET_CHECKPOINT"
echo "New output: $OUTPUT_DIR"

echo
echo "======================================================================"
echo "6. Start true-resume training"
echo "======================================================================"

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
echo "Latest log: $LATEST_LOG"
echo
echo "Follow training:"
echo "tail -f \"$LATEST_LOG\""
echo
echo "Inspect losses:"
echo "tail -f \"$LATEST_LOG\" | grep --line-buffered -E 'policy_metrics|stage_action_direction_raw_loss|stage_action_low_motion_raw_loss|weighted_stage_action_loss|action_loss|stage_accuracy|loss'"
