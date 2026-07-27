#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
CHECKPOINT=/root/gpufree-data/outputs/train/excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b144_add30ep/checkpoints/016650/pretrained_model
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

OUTPUT=/root/gpufree-data/excavator_offline_eval/seed2_fixed_scene_pose_exact_ckpt16650_all_episodes
EVAL_LOG=/root/gpufree-data/excavator_logs/offline_eval_seed2_fixed_scene_pose_exact_ckpt16650_all_episodes.log
RESULT_ARCHIVE="${OUTPUT}.tar.gz"
NUM_EPISODES=5

EVALUATOR="$PROJECT/evaluate_smolvla_random50_episodes.py"
PLOTTER="$PROJECT/plot_smolvla_median_episode.py"

cd "$PROJECT"

for path in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/stage_schema.json" \
  "$CHECKPOINT/model.safetensors" \
  "$VLM" \
  "$PRIOR" \
  "$EVALUATOR"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulation, policy client, or training process is using resources:" >&2
  pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >&2
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

rm -rf -- "$OUTPUT"
rm -f -- "$RESULT_ARCHIVE"
mkdir -p "$OUTPUT" "$(dirname "$EVAL_LOG")"

echo "======================================================================"
echo "1. Validate canonical stage schema and compare all dataset episodes"
echo "======================================================================"

"$PYTHON" - "$DATASET" "$OUTPUT/dataset_episode_comparison.json" "$NUM_EPISODES" <<'PY'
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

root = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
expected_episode_count = int(sys.argv[3])

expected_classes = {
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

info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
schema = json.loads(
    (root / "meta" / "stage_schema.json").read_text(encoding="utf-8")
)
if schema.get("classes") != expected_classes:
    raise RuntimeError(
        "Dataset does not use canonical v4 stages: "
        f"{schema.get('classes')!r}"
    )
if int(info.get("total_episodes", -1)) != expected_episode_count:
    raise RuntimeError(
        f"Expected {expected_episode_count} episodes, "
        f"metadata reports {info.get('total_episodes')}"
    )

data_files = sorted((root / "data").rglob("*.parquet"))
if not data_files:
    raise RuntimeError("Dataset contains no data parquet files")
data = pd.concat(
    [pd.read_parquet(path) for path in data_files],
    ignore_index=True,
)
data = data.sort_values(["episode_index", "frame_index"], kind="stable")
episode_ids = sorted(int(value) for value in data["episode_index"].unique())
if len(episode_ids) != expected_episode_count:
    raise RuntimeError(f"Found episode IDs {episode_ids}")

columns = [
    "timestamp",
    "observation.state",
    "observation.effort",
    "action",
    "observation.stage_current_id",
    "task_index",
]
for column in columns:
    if column not in data.columns:
        raise RuntimeError(f"Dataset is missing comparison column {column!r}")

def column_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].tolist()
    try:
        return np.asarray(values)
    except ValueError:
        return np.stack([np.asarray(value) for value in values])


groups = {
    episode_id: data[data["episode_index"] == episode_id]
    for episode_id in episode_ids
}
reference_id = episode_ids[0]
reference = groups[reference_id]
episode_rows = []

for episode_id in episode_ids:
    frame = groups[episode_id]
    row = {
        "episode_id": episode_id,
        "num_frames": int(len(frame)),
        "reference_episode_id": reference_id,
        "columns": {},
    }
    for column in columns:
        lhs = column_array(reference, column)
        rhs = column_array(frame, column)
        same_shape = lhs.shape == rhs.shape
        exact = bool(same_shape and np.array_equal(lhs, rhs))
        detail = {
            "shape": list(rhs.shape),
            "exactly_equal_to_reference": exact,
        }
        if same_shape and np.issubdtype(lhs.dtype, np.number):
            difference = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
            detail["max_abs_difference"] = (
                float(np.max(difference)) if difference.size else 0.0
            )
        row["columns"][column] = detail
    row["all_tabular_columns_exactly_equal_to_reference"] = all(
        detail["exactly_equal_to_reference"]
        for detail in row["columns"].values()
    )
    episode_rows.append(row)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


video_hashes: dict[int, dict[str, str]] = {episode_id: {} for episode_id in episode_ids}
pattern = re.compile(r"episode_(\d+)")
videos_root = root / "videos"
if videos_root.is_dir():
    for path in sorted(videos_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = pattern.search(relative)
        if match is None:
            continue
        episode_id = int(match.group(1))
        if episode_id not in video_hashes:
            continue
        template = pattern.sub("episode_*", relative)
        video_hashes[episode_id][template] = sha256(path)

reference_video_hashes = video_hashes[reference_id]
for row in episode_rows:
    episode_id = row["episode_id"]
    row["video_file_count"] = len(video_hashes[episode_id])
    row["encoded_video_files_byte_identical_to_reference"] = (
        video_hashes[episode_id] == reference_video_hashes
    )

payload = {
    "dataset": str(root),
    "canonical_stage_classes": expected_classes,
    "episode_ids": episode_ids,
    "reference_episode_id": reference_id,
    "all_tabular_episodes_exactly_identical": all(
        row["all_tabular_columns_exactly_equal_to_reference"]
        for row in episode_rows
    ),
    "all_encoded_video_files_byte_identical": all(
        row["encoded_video_files_byte_identical_to_reference"]
        for row in episode_rows
    ),
    "episodes": episode_rows,
}
output_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("episode IDs:", episode_ids)
print("all tabular episodes exactly identical:", payload["all_tabular_episodes_exactly_identical"])
print("all encoded video files byte-identical:", payload["all_encoded_video_files_byte_identical"])
for row in episode_rows:
    print(
        f"episode {row['episode_id']}: frames={row['num_frames']}, "
        f"tabular_exact={row['all_tabular_columns_exactly_equal_to_reference']}, "
        f"video_bytes_exact={row['encoded_video_files_byte_identical_to_reference']}"
    )
PY

echo
echo "======================================================================"
echo "2. Evaluate checkpoint 016650 on all five episodes"
echo "======================================================================"

"$PYTHON" "$EVALUATOR" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --vlm "$VLM" \
  --prior "$PRIOR" \
  --output-dir "$OUTPUT" \
  --num-episodes "$NUM_EPISODES" \
  --seed 2 \
  --batch-size 4 \
  --num-workers 2 \
  2>&1 | tee "$EVAL_LOG"

echo
echo "======================================================================"
echo "3. Compare per-episode checkpoint metrics"
echo "======================================================================"

"$PYTHON" - "$OUTPUT/episode_metrics.csv" "$OUTPUT/episode_metric_spread.json" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

metrics_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
frame = pd.read_csv(metrics_path).sort_values("episode_id")

columns = [
    "episode_id",
    "num_frames",
    "representative_score",
    "stage_current_accuracy",
    "stage_chunk_accuracy",
    "stage_transition_f1",
    "action_current_within_tolerance_accuracy",
    "action_chunk_within_tolerance_accuracy",
    "action_current_sign_accuracy",
    "action_chunk_sign_accuracy",
    "swing_current_mae",
    "boom_current_mae",
    "arm_current_mae",
    "bucket_current_mae",
]
print(frame[columns].to_string(index=False))

metric_columns = [column for column in columns if column != "episode_id"]
spread = {
    column: {
        "min": float(frame[column].min()),
        "max": float(frame[column].max()),
        "range": float(frame[column].max() - frame[column].min()),
    }
    for column in metric_columns
}
payload = {
    "num_episodes": int(len(frame)),
    "all_reported_metrics_exactly_identical": all(
        values["range"] == 0.0 for values in spread.values()
    ),
    "spread": spread,
}
output_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("all reported metrics exactly identical:", payload["all_reported_metrics_exactly_identical"])
PY

echo
echo "======================================================================"
echo "4. Plot the median episode and package results"
echo "======================================================================"

if [[ -f "$PLOTTER" ]]; then
  "$PYTHON" "$PLOTTER" \
    --eval-dir "$OUTPUT" \
    --dataset "$DATASET" \
    --contact-sheet-frames 12
else
  echo "[WARN] Plotter not found; skipped: $PLOTTER"
fi

tar -C "$(dirname "$OUTPUT")" -czf "$RESULT_ARCHIVE" "$(basename "$OUTPUT")"

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Summary: $OUTPUT/summary.json"
echo "Per-episode metrics: $OUTPUT/episode_metrics.csv"
echo "Dataset equality: $OUTPUT/dataset_episode_comparison.json"
echo "Metric spread: $OUTPUT/episode_metric_spread.json"
echo "Result archive: $RESULT_ARCHIVE"
echo "Evaluation log: $EVAL_LOG"
