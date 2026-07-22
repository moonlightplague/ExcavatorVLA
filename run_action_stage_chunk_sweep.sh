#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python
EVAL_ROOT=/root/gpufree-data/excavator_offline_eval
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json
ANALYZER="$PROJECT/scripts/evaluation/analyze_action_stage_chunk_stitch.py"

cd "$PROJECT"

for required in "$PRIOR" "$ANALYZER"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 1
  fi
done

LATEST_RESULT_LINK="$(ls -dt \
  "$EVAL_ROOT"/excavator_smolvla_seed2_fixed5ep_canonical300_from*_b144_latest3_5episodes_latest \
  2>/dev/null | head -1)"

if [[ -z "$LATEST_RESULT_LINK" ]]; then
  echo "[ERROR] Latest-three-checkpoint evaluation result was not found." >&2
  exit 1
fi

LATEST_RESULT="$(readlink -f "$LATEST_RESULT_LINK")"
COMPARISON="$LATEST_RESULT/checkpoint_comparison.csv"

if [[ ! -f "$COMPARISON" ]]; then
  echo "[ERROR] Missing checkpoint comparison: $COMPARISON" >&2
  exit 1
fi

read -r BEST_STEP BEST_SCORE < <(
  "$PYTHON" - "$COMPARISON" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("Checkpoint comparison is empty")
best = max(rows, key=lambda row: float(row["micro_representative_score"]))
print(int(best["checkpoint_step"]), float(best["micro_representative_score"]))
PY
)

BEST_EVAL_DIR="$LATEST_RESULT/checkpoint_$(printf '%06d' "$BEST_STEP")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$EVAL_ROOT/action_stage_chunk_sweep_ckpt${BEST_STEP}_${TIMESTAMP}"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
LATEST_OUTPUT_LINK="$EVAL_ROOT/action_stage_chunk_sweep_ckpt${BEST_STEP}_latest"
LATEST_ARCHIVE_LINK="${LATEST_OUTPUT_LINK}.tar.gz"

if [[ ! -d "$BEST_EVAL_DIR/episodes" ]]; then
  echo "[ERROR] Best checkpoint evaluation arrays were not found: $BEST_EVAL_DIR" >&2
  exit 1
fi

echo "Latest evaluation: $LATEST_RESULT"
echo "Best checkpoint step: $BEST_STEP"
echo "Best representative score: $BEST_SCORE"
echo "Source checkpoint evaluation: $BEST_EVAL_DIR"
echo "Output: $OUTPUT_DIR"

"$PYTHON" "$ANALYZER" \
  --eval-dir "$BEST_EVAL_DIR" \
  --prior "$PRIOR" \
  --output-dir "$OUTPUT_DIR" \
  --chunks 1 2 3 5 10 15 20 30 50

tar -C "$(dirname "$OUTPUT_DIR")" \
  -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
ln -sfn "$OUTPUT_DIR" "$LATEST_OUTPUT_LINK"
ln -sfn "$ARCHIVE" "$LATEST_ARCHIVE_LINK"

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Best checkpoint step: $BEST_STEP"
echo "Summary: $OUTPUT_DIR/chunk_sweep_summary.json"
echo "Micro comparison: $OUTPUT_DIR/chunk_sweep_micro.csv"
echo "Macro comparison: $OUTPUT_DIR/chunk_sweep_macro.csv"
echo "Per-episode metrics: $OUTPUT_DIR/chunk_sweep_episode_metrics.csv"
echo "Plot: $OUTPUT_DIR/chunk_stitch_accuracy.png"
echo "Archive: $ARCHIVE"
echo "Latest archive link: $LATEST_ARCHIVE_LINK"
echo
echo "Download command:"
echo "scp -P 30105 root@120.209.70.195:$LATEST_ARCHIVE_LINK ."
