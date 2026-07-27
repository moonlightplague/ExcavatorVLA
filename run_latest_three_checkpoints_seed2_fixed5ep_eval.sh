#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
TRAIN_ROOT=/root/gpufree-data/outputs/train
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json
EVAL_ROOT=/root/gpufree-data/excavator_offline_eval
LOG_ROOT=/root/gpufree-data/excavator_logs

EVALUATOR="$PROJECT/evaluate_smolvla_random50_episodes.py"
PLOTTER="$PROJECT/plot_smolvla_median_episode.py"
NUM_EPISODES=5
EVAL_BATCH_SIZE=4
EVAL_NUM_WORKERS=2

cd "$PROJECT"
mkdir -p "$EVAL_ROOT" "$LOG_ROOT"

for required in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/stage_schema.json" \
  "$VLM" \
  "$PRIOR" \
  "$EVALUATOR"
do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 1
  fi
done

if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulation, policy client, or training process is still running:" >&2
  pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >&2
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

echo "======================================================================"
echo "1. Validate the fixed five-episode dataset"
echo "======================================================================"

"$PYTHON" - "$DATASET" "$NUM_EPISODES" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]).resolve()
expected_episodes = int(sys.argv[2])
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

if int(info.get("total_episodes", -1)) != expected_episodes:
    raise RuntimeError(
        f"Expected {expected_episodes} episodes, got {info.get('total_episodes')}"
    )
if int(info.get("total_frames", -1)) != 810:
    raise RuntimeError(f"Expected 810 frames, got {info.get('total_frames')}")
if info.get("features", {}).get("observation.state", {}).get("shape") != [27]:
    raise RuntimeError("Dataset observation.state is not 27D")
if "observation.stage_current_id" not in info.get("features", {}):
    raise RuntimeError("Dataset lacks observation.stage_current_id")
if schema.get("classes") != expected_classes:
    raise RuntimeError(
        f"Dataset does not use the canonical stage order: {schema.get('classes')!r}"
    )

print("dataset:", root)
print("frames:", info["total_frames"])
print("episodes:", info["total_episodes"])
print("state shape:", info["features"]["observation.state"]["shape"])
print("stages:", schema["classes"])
PY

echo
echo "======================================================================"
echo "2. Select the last three complete checkpoints from the completed run"
echo "======================================================================"

SELECTION_TMP="$(mktemp)"
trap 'rm -f -- "$SELECTION_TMP"' EXIT

mapfile -t SELECTION < <(
  "$PYTHON" - "$TRAIN_ROOT" "${TRAIN_RUN:-}" "$SELECTION_TMP" <<'PY'
from pathlib import Path
import json
import re
import sys

train_root = Path(sys.argv[1]).resolve()
requested_run = sys.argv[2].strip()
manifest_path = Path(sys.argv[3])
run_pattern = re.compile(
    r"^excavator_smolvla_seed2_fixed5ep_canonical300_from\d+_b144$"
)


def complete_checkpoints(run: Path):
    result = []
    checkpoints = run / "checkpoints"
    if not checkpoints.is_dir():
        return result

    for checkpoint in checkpoints.iterdir():
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            continue
        pretrained = checkpoint / "pretrained_model"
        state_path = checkpoint / "training_state" / "training_step.json"
        required = (
            pretrained / "model.safetensors",
            pretrained / "config.json",
            pretrained / "train_config.json",
            checkpoint / "training_state" / "optimizer_param_groups.json",
            checkpoint / "training_state" / "optimizer_state.safetensors",
            checkpoint / "training_state" / "scheduler_state.json",
        )
        if not state_path.is_file() or not all(path.is_file() for path in required):
            continue
        try:
            step = int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
        except Exception:
            continue
        result.append((step, checkpoint.resolve(), pretrained.resolve()))

    return sorted(result, key=lambda item: (item[0], item[1].stat().st_mtime))


if requested_run:
    run_candidates = [Path(requested_run).expanduser().resolve()]
else:
    run_candidates = [
        path.resolve()
        for path in train_root.iterdir()
        if path.is_dir() and run_pattern.fullmatch(path.name)
    ]

if not run_candidates:
    raise SystemExit(
        "No canonical300 training run was found. Set TRAIN_RUN to the exact run path."
    )

completed_runs = []
progress_rows = []
for run in run_candidates:
    checkpoints = complete_checkpoints(run)
    if not checkpoints:
        progress_rows.append((str(run), None, None))
        continue

    latest_step, _, latest_pretrained = checkpoints[-1]
    train_config = json.loads(
        (latest_pretrained / "train_config.json").read_text(encoding="utf-8")
    )
    target_step = int(train_config["steps"])
    progress_rows.append((str(run), latest_step, target_step))
    if latest_step >= target_step:
        completed_runs.append(
            (latest_step, target_step, run.stat().st_mtime, run, checkpoints)
        )

if not completed_runs:
    details = "\n".join(
        f"  {run}: latest={latest}, target={target}"
        for run, latest, target in progress_rows
    )
    raise SystemExit(
        "No completed canonical300 run was found. Current progress:\n" + details
    )

latest_step, target_step, _, run, checkpoints = max(completed_runs)
if len(checkpoints) < 3:
    raise SystemExit(f"Only {len(checkpoints)} complete checkpoints exist in {run}")

selected = checkpoints[-3:]
manifest = {
    "training_run": str(run),
    "configured_target_step": target_step,
    "latest_complete_step": latest_step,
    "selection_rule": "three highest training_step values among complete checkpoints",
    "selected_checkpoints": [
        {
            "step": step,
            "checkpoint_dir": str(checkpoint),
            "pretrained_model": str(pretrained),
        }
        for step, checkpoint, pretrained in selected
    ],
}
manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print(run)
for step, _, pretrained in selected:
    print(f"{step}|{pretrained}")
PY
)

if [[ "${#SELECTION[@]}" -ne 4 ]]; then
  echo "[ERROR] Expected one run path and three checkpoint records." >&2
  printf '%s\n' "${SELECTION[@]}" >&2
  exit 1
fi

TRAIN_RUN_SELECTED="${SELECTION[0]}"
RUN_BASENAME="$(basename "$TRAIN_RUN_SELECTED")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="$EVAL_ROOT/${RUN_BASENAME}_latest3_5episodes_${TIMESTAMP}"
ARCHIVE="${OUTPUT_ROOT}.tar.gz"
LATEST_LINK="$EVAL_ROOT/${RUN_BASENAME}_latest3_5episodes_latest"
LATEST_ARCHIVE_LINK="${LATEST_LINK}.tar.gz"

mkdir -p "$OUTPUT_ROOT"
cp -- "$SELECTION_TMP" "$OUTPUT_ROOT/checkpoint_selection.json"

echo "training run: $TRAIN_RUN_SELECTED"
echo "selected checkpoints:"
printf '  %s\n' "${SELECTION[@]:1}"
echo "output root: $OUTPUT_ROOT"

echo
echo "======================================================================"
echo "3. Evaluate every selected checkpoint on the same five episodes"
echo "======================================================================"

STEPS=()

for record in "${SELECTION[@]:1}"; do
  IFS='|' read -r STEP CHECKPOINT <<< "$record"
  STEPS+=("$STEP")

  CHECKPOINT_OUTPUT="$OUTPUT_ROOT/checkpoint_$(printf '%06d' "$STEP")"
  CHECKPOINT_LOG="$OUTPUT_ROOT/checkpoint_$(printf '%06d' "$STEP").log"

  echo
  echo "----------------------------------------------------------------------"
  echo "Evaluating checkpoint step $STEP"
  echo "checkpoint: $CHECKPOINT"
  echo "output: $CHECKPOINT_OUTPUT"
  echo "----------------------------------------------------------------------"

  "$PYTHON" "$EVALUATOR" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --vlm "$VLM" \
    --prior "$PRIOR" \
    --output-dir "$CHECKPOINT_OUTPUT" \
    --num-episodes "$NUM_EPISODES" \
    --seed 2 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$EVAL_NUM_WORKERS" \
    2>&1 | tee "$CHECKPOINT_LOG"

  if [[ -f "$PLOTTER" ]]; then
    "$PYTHON" "$PLOTTER" \
      --eval-dir "$CHECKPOINT_OUTPUT" \
      --dataset "$DATASET" \
      --contact-sheet-frames 12
  else
    echo "[WARN] Plotter not found; figures were skipped: $PLOTTER"
  fi
done

echo
echo "======================================================================"
echo "4. Build the cross-checkpoint comparison"
echo "======================================================================"

"$PYTHON" - "$OUTPUT_ROOT" "${STEPS[@]}" <<'PY'
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1]).resolve()
steps = [int(value) for value in sys.argv[2:]]
rows = []
summaries = []

for step in steps:
    eval_dir = root / f"checkpoint_{step:06d}"
    summary_path = eval_dir / "summary.json"
    episode_metrics_path = eval_dir / "episode_metrics.csv"
    if not summary_path.is_file() or not episode_metrics_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint {step} is missing summary.json or episode_metrics.csv"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    macro = summary["aggregate"]["macro_episode_mean"]
    micro = summary["micro_all_frames"]
    row = {
        "checkpoint_step": step,
        "checkpoint": summary["checkpoint"],
        "num_episodes": summary["aggregate"]["num_episodes"],
        "num_frames": summary["num_dataset_frames_evaluated"],
        "inference_minutes": summary["inference_minutes"],
        "median_episode": summary["median_episode"],
        "median_representative_score": summary["median_representative_score"],
    }
    row.update({f"macro_{key}": value for key, value in macro.items()})
    row.update({f"micro_{key}": value for key, value in micro.items()})
    rows.append(row)
    summaries.append(summary)

fieldnames = list(rows[0])
for row in rows[1:]:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

with (root / "checkpoint_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "dataset": summaries[0]["dataset"],
    "selected_episode_ids": summaries[0]["selected_episodes"],
    "checkpoints": summaries,
}
(root / "checkpoint_comparison.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

display_metrics = [
    "checkpoint_step",
    "micro_representative_score",
    "micro_stage_current_accuracy",
    "micro_stage_chunk_accuracy",
    "micro_stage_transition_f1",
    "micro_action_current_within_tolerance_accuracy",
    "micro_action_current_sign_accuracy",
    "micro_stage_action_prior_current_accuracy",
    "micro_low_motion_current_satisfaction",
    "micro_swing_current_mae",
    "micro_boom_current_mae",
    "micro_arm_current_mae",
    "micro_bucket_current_mae",
]

print(",".join(display_metrics))
for row in rows:
    print(",".join(str(row.get(key, "")) for key in display_metrics))
PY

tar -C "$(dirname "$OUTPUT_ROOT")" \
  -czf "$ARCHIVE" "$(basename "$OUTPUT_ROOT")"
ln -sfn "$OUTPUT_ROOT" "$LATEST_LINK"
ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Training run: $TRAIN_RUN_SELECTED"
echo "Selected steps: ${STEPS[*]}"
echo "Output root: $OUTPUT_ROOT"
echo "Comparison CSV: $OUTPUT_ROOT/checkpoint_comparison.csv"
echo "Comparison JSON: $OUTPUT_ROOT/checkpoint_comparison.json"
echo "Result archive: $ARCHIVE"
echo "Latest result link: $LATEST_LINK"
echo "Latest archive link: $LATEST_ARCHIVE_LINK"
echo
echo "Download command:"
echo "scp -P 30105 root@120.209.70.195:$LATEST_ARCHIVE_LINK ."
