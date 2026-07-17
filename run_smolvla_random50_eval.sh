#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

DATASET=/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3_stage10_h30_obslabels
TRAIN_ROOT=/root/gpufree-data/outputs/train
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json
OUTPUT=/root/gpufree-data/excavator_final_offline_eval

NUM_EPISODES=50
SEED=20260717
BATCH_SIZE=4
NUM_WORKERS=4

cd "$PROJECT"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if pgrep -af '[l]erobot-train|[l]erobot_train|[o]t_train.py' >/dev/null; then
    echo "A training process is still running. Stop it before evaluation."
    pgrep -af '[l]erobot-train|[l]erobot_train|[o]t_train.py'
    exit 1
fi

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

"$PYTHON" evaluate_smolvla_random50_episodes.py \
  --dataset "$DATASET" \
  --train-root "$TRAIN_ROOT" \
  --vlm "$VLM" \
  --prior "$PRIOR" \
  --output-dir "$OUTPUT" \
  --num-episodes "$NUM_EPISODES" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

"$PYTHON" plot_smolvla_median_episode.py \
  --eval-dir "$OUTPUT" \
  --dataset "$DATASET" \
  --contact-sheet-frames 12

echo
echo "Evaluation complete."
echo "Summary:"
echo "  $OUTPUT/summary.json"
echo "Per-episode metrics:"
echo "  $OUTPUT/episode_metrics.csv"
echo "Median episode:"
echo "  $OUTPUT/median_episode.json"
echo "Detailed median-episode figures:"
find "$OUTPUT" -maxdepth 1 -type d -name 'median_episode_*_figures' -print
