#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

# Prepared LeRobot dataset.  This directory is treated as read-only.
DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_legacy_v11_32d

CHECKPOINT=/root/gpufree-data/outputs/train/excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b144_add30ep/checkpoints/016650/pretrained_model
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

OUTPUT=/root/gpufree-data/excavator_offline_eval/seed2_prepared_legacy_v11_32d_ckpt16650_one_episode
EVAL_LOG=/root/gpufree-data/excavator_logs/offline_eval_seed2_prepared_legacy_v11_32d_ckpt16650_one_episode.log
RESULT_ARCHIVE="${OUTPUT}.tar.gz"

EVALUATOR="$PROJECT/evaluate_smolvla_random50_episodes.py"
PLOTTER="$PROJECT/plot_smolvla_median_episode.py"

cd "$PROJECT"

for path in \
  "$DATASET/meta/info.json" \
  "$CHECKPOINT/model.safetensors" \
  "$CHECKPOINT/config.json" \
  "$VLM" \
  "$PRIOR" \
  "$EVALUATOR"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$EVAL_LOG")"

# This evaluation needs the GPU, but it does not need Isaac Sim.
if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulation, policy client, or training process is using resources:" >&2
  pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >&2
  echo "[ERROR] Stop those processes before starting this offline evaluation." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================================================"
echo "1. Validate the prepared dataset against checkpoint 016650"
echo "======================================================================"

"$PYTHON" - "$DATASET" "$CHECKPOINT" <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_root = Path(sys.argv[1]).resolve()
checkpoint_root = Path(sys.argv[2]).resolve()

info = json.loads(
    (dataset_root / "meta" / "info.json").read_text(encoding="utf-8")
)
config = json.loads(
    (checkpoint_root / "config.json").read_text(encoding="utf-8")
)

stage_key = str(config.get("stage_source_key", "")).strip()
if not stage_key:
    raise RuntimeError("Checkpoint config.json has no non-empty stage_source_key")

features = info.get("features", {})
required_metadata_shapes = {
    "observation.state": (27,),
    "observation.effort": (4,),
    "action": (4,),
    "observation.images.0": (3, 256, 256),
    "observation.images.1": (3, 256, 256),
    "observation.images.2": (3, 256, 256),
}

print("dataset:", dataset_root)
print("checkpoint:", checkpoint_root)
print("total_frames:", info.get("total_frames"))
print("total_episodes:", info.get("total_episodes"))
print("fps:", info.get("fps"))
print("checkpoint stage_source_key:", stage_key)

for key, expected in required_metadata_shapes.items():
    feature = features.get(key)
    if feature is None:
        raise RuntimeError(f"Dataset metadata is missing required feature {key!r}")
    actual = tuple(int(value) for value in feature.get("shape", []))
    print(f"metadata {key}: {actual}")
    if actual != expected:
        raise RuntimeError(
            f"Dataset is incompatible with checkpoint 016650: "
            f"{key} must have shape {expected}, got {actual}"
        )

if stage_key not in features:
    raise RuntimeError(
        f"Dataset metadata is missing checkpoint stage label {stage_key!r}"
    )

stage_shape = tuple(int(value) for value in features[stage_key].get("shape", []))
if stage_shape not in ((), (1,)):
    raise RuntimeError(
        f"Checkpoint stage label {stage_key!r} must be scalar, got {stage_shape}"
    )
print(f"metadata {stage_key}: {stage_shape or 'scalar'}")

tasks_path = dataset_root / "meta" / "tasks.parquet"
if not tasks_path.is_file():
    raise FileNotFoundError(tasks_path)
tasks = pd.read_parquet(tasks_path)
print("tasks:")
print(tasks.to_string())
if len(tasks) == 0:
    raise RuntimeError("Dataset contains no task prompt")

# Load one real frame so video decoding and tensor shapes fail before policy
# weights consume GPU memory.  This only reads the dataset.
dataset = LeRobotDataset(
    repo_id=dataset_root.name,
    root=dataset_root,
    download_videos=False,
)
if len(dataset) == 0:
    raise RuntimeError("Dataset contains no frames")
sample = dataset[0]

for key, expected in required_metadata_shapes.items():
    if key not in sample:
        raise RuntimeError(f"Dataset sample is missing {key!r}")
    actual = tuple(int(value) for value in sample[key].shape)
    print(f"sample {key}: {actual}")
    if actual != expected:
        raise RuntimeError(f"{key}: expected {expected}, got {actual}")

if stage_key not in sample:
    raise RuntimeError(f"Dataset sample is missing {stage_key!r}")
stage_value = int(sample[stage_key].item())
if not 0 <= stage_value < int(config.get("num_stages", 10)):
    raise RuntimeError(
        f"Invalid {stage_key}={stage_value}; expected a checkpoint stage ID"
    )
print(f"sample {stage_key}: {stage_value}")

sample_task = str(sample.get("task", "")).strip()
if not sample_task:
    raise RuntimeError("Dataset sample has an empty task prompt")
print("sample task:", sample_task)

print("[OK] Prepared dataset is a readable 27D checkpoint dataset.")
PY

echo
echo "======================================================================"
echo "2. Run checkpoint 016650 on one expert episode"
echo "======================================================================"

# Only this dedicated result directory is replaced. DATASET is never changed.
rm -rf -- "$OUTPUT"
mkdir -p "$OUTPUT"

"$PYTHON" "$EVALUATOR" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --vlm "$VLM" \
  --prior "$PRIOR" \
  --output-dir "$OUTPUT" \
  --num-episodes 1 \
  --seed 2 \
  --batch-size 4 \
  --num-workers 2 \
  2>&1 | tee "$EVAL_LOG"

echo
echo "======================================================================"
echo "3. Generate figures and package all results"
echo "======================================================================"

if [[ -f "$PLOTTER" ]]; then
  "$PYTHON" "$PLOTTER" \
    --eval-dir "$OUTPUT" \
    --dataset "$DATASET" \
    --contact-sheet-frames 12
else
  echo "[WARN] Plotter not found; skipped: $PLOTTER"
fi

rm -f -- "$RESULT_ARCHIVE"
tar -C "$(dirname "$OUTPUT")" -czf "$RESULT_ARCHIVE" "$(basename "$OUTPUT")"

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Dataset (unchanged): $DATASET"
echo "Summary: $OUTPUT/summary.json"
echo "Per-episode metrics: $OUTPUT/episode_metrics.csv"
echo "Stage confusion: $OUTPUT/micro_stage_confusion.npy"
echo "Per-stage metrics: $OUTPUT/micro_per_stage_metrics.csv"
echo "Result archive: $RESULT_ARCHIVE"
echo "Evaluation log: $EVAL_LOG"

"$PYTHON" - "$OUTPUT/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)

micro = summary["micro_all_frames"]
macro = summary["aggregate"]["macro_episode_mean"]

print()
print("KEY RESULTS")
print(
    "stage current accuracy (macro/micro):",
    macro.get("stage_current_accuracy"),
    micro.get("stage_current_accuracy"),
)
print(
    "stage 50-step accuracy (macro/micro):",
    macro.get("stage_chunk_accuracy"),
    micro.get("stage_chunk_accuracy"),
)
print(
    "action current tolerance accuracy (macro/micro):",
    macro.get("action_current_within_tolerance_accuracy"),
    micro.get("action_current_within_tolerance_accuracy"),
)
print(
    "action current sign accuracy (macro/micro):",
    macro.get("action_current_sign_accuracy"),
    micro.get("action_current_sign_accuracy"),
)

for name in ("swing", "boom", "arm", "bucket"):
    print(
        name,
        "MAE=", micro.get(f"{name}_current_mae"),
        "RMSE=", micro.get(f"{name}_current_rmse"),
        "corr=", micro.get(f"{name}_current_corr"),
    )
PY
