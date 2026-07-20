#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

RUN=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053
EXPORT28="$RUN/lerobot_v3_eval28"
DATA27="$RUN/lerobot_v3_eval27_stage10_h30_obslabels"

CHECKPOINT=/root/gpufree-data/outputs/train/excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b144_add30ep/checkpoints/016650/pretrained_model
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json

OUTPUT=/root/gpufree-data/excavator_offline_eval/seed2_fixed_scene_pose_exact_ckpt16650
SIM_LOG=/root/gpufree-data/excavator_logs/simulation_dynamic_prompt_seed2.log
EVAL_LOG=/root/gpufree-data/excavator_logs/offline_eval_seed2_fixed_scene_pose_exact_ckpt16650.log

EXPORTER="$PROJECT/excavator_dataset_tools.py"
CONVERTER="$PROJECT/convert_lerobot_stage_dataset.py"
EVALUATOR="$PROJECT/evaluate_smolvla_random50_episodes.py"
PLOTTER="$PROJECT/plot_smolvla_median_episode.py"

cd "$PROJECT"

for path in \
  "$RUN" \
  "$CHECKPOINT/model.safetensors" \
  "$VLM" \
  "$PRIOR" \
  "$EXPORTER" \
  "$CONVERTER" \
  "$EVALUATOR"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$EVAL_LOG")"

# Offline evaluation needs the GPU but not Isaac Sim.
if pgrep -af '[r]un_simulation.py|[s]molvla_policy_client.py|[l]erobot-train|[o]t_train.py' >/dev/null; then
  echo "[ERROR] A simulation, policy client, or training process is still using resources:" >&2
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
echo "1. Extract exact prompt from the seed-2 simulation log"
echo "======================================================================"

if [[ -n "${TASK_TEXT:-}" ]]; then
  EXACT_TASK="$TASK_TEXT"
else
  if [[ ! -f "$SIM_LOG" ]]; then
    echo "[ERROR] Simulation log not found: $SIM_LOG" >&2
    echo "[ERROR] Re-run with TASK_TEXT='exact prompt' or update SIM_LOG." >&2
    exit 1
  fi

  EXACT_TASK="$(
    "$PYTHON" - "$SIM_LOG" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

for i, line in enumerate(lines):
    if "[PROMPT] task_text" not in line:
        continue

    after = line.split("[PROMPT] task_text", 1)[1].lstrip(" :=")
    if after.strip():
        print(after.strip())
        raise SystemExit(0)

    for following in lines[i + 1:]:
        candidate = following.strip()
        if candidate:
            print(candidate)
            raise SystemExit(0)

raise SystemExit("Could not find a non-empty [PROMPT] task_text entry")
PY
  )"
fi

if [[ -z "$EXACT_TASK" ]]; then
  echo "[ERROR] Exact task prompt is empty." >&2
  exit 1
fi

echo "[PROMPT] $EXACT_TASK"

echo
echo "======================================================================"
echo "2. Export the five raw trainable episodes to LeRobot v3 (28D runtime state)"
echo "======================================================================"

rm -rf "$EXPORT28"

"$PYTHON" "$EXPORTER" "$RUN" \
  --export-lerobot \
  --export-split trainable \
  --export-dir "$EXPORT28" \
  --export-overwrite

echo
echo "======================================================================"
echo "3. Replace the raw generic task with the exact seed-2 prompt"
echo "======================================================================"

"$PYTHON" - "$EXPORT28" "$EXACT_TASK" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

root = Path(sys.argv[1])
task = sys.argv[2].strip()
if not task:
    raise RuntimeError("Exact task is empty")

tasks_path = root / "meta" / "tasks.parquet"
if not tasks_path.is_file():
    raise FileNotFoundError(tasks_path)

tasks = pd.read_parquet(tasks_path)
print("[TASKS BEFORE]")
print(tasks)

if len(tasks) != 1:
    raise RuntimeError(
        "Expected exactly one task for the fixed seed-2 scene; "
        f"found {len(tasks)} tasks. Refusing to collapse different tasks."
    )

task_index = int(tasks["task_index"].iloc[0])
index_name = tasks.index.name
patched_tasks = pd.DataFrame(
    {"task_index": [task_index]},
    index=pd.Index([task], name=index_name),
)
patched_tasks.to_parquet(tasks_path)

# Keep episode metadata human-readable and consistent.
episodes_root = root / "meta" / "episodes"
if episodes_root.is_dir():
    for path in sorted(episodes_root.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "tasks" in frame.columns:
            frame["tasks"] = [[task] for _ in range(len(frame))]
            frame.to_parquet(path, index=False)

# Some exporters retain debug task text in data parquet.
data_root = root / "data"
if data_root.is_dir():
    for path in sorted(data_root.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        changed = False
        if "task" in frame.columns:
            frame["task"] = task
            changed = True
        if changed:
            frame.to_parquet(path, index=False)

print("[TASKS AFTER]")
print(pd.read_parquet(tasks_path))
PY

echo
echo "======================================================================"
echo "4. Convert 28D runtime state to the checkpoint's 27D state"
echo "   and create current/future stage labels"
echo "======================================================================"

rm -rf "$DATA27"

"$PYTHON" "$CONVERTER" \
  --src "$EXPORT28" \
  --dst "$DATA27" \
  --horizon 30 \
  --minimum-purity 0.70 \
  --video-mode hardlink \
  --overwrite

echo
echo "======================================================================"
echo "5. Validate the converted dataset contract"
echo "======================================================================"

"$PYTHON" - "$DATA27" "$EXACT_TASK" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1])
expected_task = sys.argv[2]

info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
print("total_frames:", info.get("total_frames"))
print("total_episodes:", info.get("total_episodes"))
print("fps:", info.get("fps"))
print("features:", list(info.get("features", {})))

tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
print("tasks:")
print(tasks)

ds = LeRobotDataset(str(root))
sample = ds[0]

required_shapes = {
    "observation.state": (27,),
    "observation.effort": (4,),
    "action": (4,),
    "observation.images.0": (3, 256, 256),
    "observation.images.1": (3, 256, 256),
    "observation.images.2": (3, 256, 256),
}
for key, expected in required_shapes.items():
    if key not in sample:
        raise RuntimeError(f"Dataset sample is missing {key}")
    actual = tuple(sample[key].shape)
    print(f"{key}: {actual}")
    if actual != expected:
        raise RuntimeError(f"{key}: expected {expected}, got {actual}")

sample_task = str(sample.get("task", "")).strip()
print("sample task:", sample_task)
if sample_task != expected_task:
    raise RuntimeError(
        "Dataset task does not match the exact simulation prompt:\n"
        f"dataset={sample_task!r}\nexpected={expected_task!r}"
    )

for key in (
    "stage_current_id",
    "stage_target_30",
    "stage_purity_30",
    "stage_valid_30",
):
    if key not in sample:
        raise RuntimeError(f"Converted dataset is missing {key}")
    print(f"{key}: {sample[key]}")
PY

echo
echo "======================================================================"
echo "6. Run checkpoint 016650 on all five expert episodes"
echo "======================================================================"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

"$PYTHON" "$EVALUATOR" \
  --dataset "$DATA27" \
  --checkpoint "$CHECKPOINT" \
  --vlm "$VLM" \
  --prior "$PRIOR" \
  --output-dir "$OUTPUT" \
  --num-episodes 5 \
  --seed 20260720 \
  --batch-size 4 \
  --num-workers 2 \
  2>&1 | tee "$EVAL_LOG"

echo
echo "======================================================================"
echo "7. Generate detailed figures for the median episode"
echo "======================================================================"

if [[ -f "$PLOTTER" ]]; then
  "$PYTHON" "$PLOTTER" \
    --eval-dir "$OUTPUT" \
    --dataset "$DATA27" \
    --contact-sheet-frames 12
else
  echo "[WARN] Plotter not found; skipped: $PLOTTER"
fi

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Dataset: $DATA27"
echo "Summary: $OUTPUT/summary.json"
echo "Per-episode metrics: $OUTPUT/episode_metrics.csv"
echo "Stage confusion: $OUTPUT/micro_stage_confusion.npy"
echo "Per-stage metrics: $OUTPUT/micro_per_stage_metrics.csv"
echo "Log: $EVAL_LOG"

"$PYTHON" - "$OUTPUT/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    summary = json.load(f)

micro = summary["micro_all_frames"]
macro = summary["aggregate"]["macro_episode_mean"]

print()
print("KEY RESULTS")
print("stage current accuracy (macro/micro):",
      macro.get("stage_current_accuracy"),
      micro.get("stage_current_accuracy"))
print("stage 50-step accuracy (macro/micro):",
      macro.get("stage_chunk_accuracy"),
      micro.get("stage_chunk_accuracy"))
print("action current tolerance accuracy (macro/micro):",
      macro.get("action_current_within_tolerance_accuracy"),
      micro.get("action_current_within_tolerance_accuracy"))
print("action current sign accuracy (macro/micro):",
      macro.get("action_current_sign_accuracy"),
      micro.get("action_current_sign_accuracy"))

for name in ("swing", "boom", "arm", "bucket"):
    print(
        name,
        "MAE=", micro.get(f"{name}_current_mae"),
        "RMSE=", micro.get(f"{name}_current_rmse"),
        "corr=", micro.get(f"{name}_current_corr"),
    )
PY
