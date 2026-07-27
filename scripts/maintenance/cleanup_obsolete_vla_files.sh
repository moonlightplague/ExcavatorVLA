#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/maintenance/cleanup_obsolete_vla_files.sh [--dry-run|--apply]

Safely removes only the explicitly listed obsolete simulation, training, and
client files. The default is --dry-run. Datasets, checkpoints, logs, outputs,
and files outside the Git worktree are never touched.
EOF
}

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run) APPLY=0 ;;
  --apply) APPLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" || ! -f "$ROOT/run_simulation.py" || ! -f "$ROOT/scripts/bridge_test/smolvla_policy_client.py" ]]; then
  echo "[ERROR] Run this script inside the ExcavatorVLA Git worktree." >&2
  exit 1
fi
cd "$ROOT"

OBSOLETE_FILES=(
  "archive/standalone/run_excavator_standalone_backup_camera_bug.py"
  "assets/usd/excavator_scene.usd.bak_before_remove_sand"
  "convert_stage_seq50_dataset.py"
  "discover_stage_action_relationships.py"
  "docs/analysis/CODE_REDUNDANCY_ERROR_ANALYSIS.md"
  "docs/analysis/SCRIPT_FUNCTION_MEMO_INDEX.md"
  "docs/vla_32d_calculation_audit.md"
  "docs/vla_32d_deployment_contract.md"
  "analyze_action_stage_chunk_stitch.py"
  "analyze_smolvla_200step_trace.py"
  "evaluate_smolvla_random50_episodes.py"
  "fix_run_simulation_bbox_overflow.py"
  "inspect_lerobot_dataset.py"
  "inspect_stage_layout.py"
  "migrate_stage_labels_to_observation.py"
  "patch_run_simulation_exact_training_prompt.py"
  "patch_run_simulation_state27_hide_sand_bbox_root.py"
  "patch_smolvla_dynamic_stage_seq50.py"
  "patch_smolvla_stage_action_direction_loss.py"
  "patch_smolvla_stage_action_low_motion_loss.py"
  "patch_smolvla_stage_seq50.py"
  "patch_smolvla_stage_training.py"
  "patch_stage_batch_and_logging.py"
  "plot_smolvla_median_episode.py"
  "prepare_stage_action_prior_training.py"
  "resume_smolvla_lowmotion_from3840_add30epochs.sh"
  "resume_smolvla_lowmotion_from3840_autobatch_add30epochs.sh"
  "run.sh"
  "run_best_checkpoint_action_stage_chunk_sweep.sh"
  "run_dataset_dashboard_v9.py"
  "run_dataset_dashboard_v10.py"
  "run_isaacsim_ckpt18450_replan1_ensemble5_seed2_one_episode.sh"
  "run_isaacsim_ckpt18450_replan3_ensemble1_seed2_one_episode.sh"
  "run_isaacsim_replay_offline_episode_chunk1.sh"
  "run_simulation.py.before_bbox_overflow_fix"
  "run_simulation.py.before_exact_training_prompt"
  "run_simulation.py.before_state27_bbox"
  "run_smolvla_random50_eval.sh"
  "score_stage_action_alignment.py"
  "scripts/bridge_test/archive/smolvla_client_backup_raw_action.py"
  "scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py"
  "scripts/bridge_test/compare_100_frames.py"
  "scripts/bridge_test/config.txt"
  "scripts/bridge_test/manual_lower_test.py"
  "scripts/bridge_test/observation_context.example.json"
  "scripts/bridge_test/persistent_bridge_server.py"
  "scripts/bridge_test/persistent_policy_client.py"
  "scripts/bridge_test/smolvla_client.py"
  "scripts/bridge_test/smolvla_policy_client.py.bak_before_effort18"
  "scripts/bridge_test/smolvla_policy_client_32d.py"
  "scripts/bridge_test/smolvla_policy_client_old.py"
  "scripts/bridge_test/smolvla_policy_client_stage_control_v10.py"
  "scripts/excavator_app/excavator_runtime.py.bak"
  "scripts/excavator_app/sand_site_runtime.py.bak_materials_fix"
  "summarize_smolvla_log_50step.py"
  "summarize_smolvla_log_50step_v2.py"
  "summarize_smolvla_resume_50step.py"
  "train_smolvla_dynamic_stage_seq50_epoch20.sh"
  "train_smolvla_dynamic_stage_seq50_epoch20_v2.sh"
  "train_smolvla_dynamic_stage_seq50_epoch20_v4.sh"
  "train_smolvla_dynamic_stage_seq50_prior_epoch20.sh"
  "train_smolvla_stage10_h30_epoch20.sh"
  "train_smolvla_stage10_h30_epoch20_v2.sh"
  "train_smolvla_stage10_h30_epoch20_v3.sh"
  "train_smolvla_stage_seq50_epoch20.sh"
)

LIVE_FILES=(
  "resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh"
  "run_isaacsim_ckpt18450_seed2_one_episode.sh"
  "run_isaacsim_offline_episode_replay_chunk1.sh"
  "run_latest_three_checkpoints_seed2_fixed5ep_eval.sh"
  "run_action_stage_chunk_sweep.sh"
  "run_dataset_dashboard.py"
  "run_excavator_standalone.py"
  "run_simulation.py"
  "scripts/bridge_test/gui_client.py"
  "scripts/bridge_test/gui_tcp_bridge_server.py"
  "scripts/evaluation/analyze_smolvla_rollout_trace.py"
  "scripts/evaluation/evaluate_smolvla_episodes.py"
  "scripts/bridge_test/smolvla_policy_client.py"
  "scripts/bridge_test/replay_episode_npz_client.py"
  "scripts/training/patch_smolvla_stage_action_low_motion_loss.py"
)

for path in "${LIVE_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required current workflow file is missing: $path" >&2
    exit 1
  fi
done

for path in "${OBSOLETE_FILES[@]}"; do
  case "$path" in
    /*|../*|*/../*)
      echo "[ERROR] Unsafe allowlist entry: $path" >&2
      exit 1
      ;;
  esac

  if [[ ! -e "$path" && ! -L "$path" ]]; then
    printf '[absent]  %s\n' "$path"
    continue
  fi

  if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    status="$(git status --porcelain=v1 -- "$path")"
    if [[ -n "$status" ]]; then
      echo "[ERROR] Refusing to remove modified tracked file: $path ($status)" >&2
      exit 1
    fi
    if (( APPLY )); then
      git rm -- "$path"
    else
      printf '[tracked] %s\n' "$path"
    fi
  else
    if (( APPLY )); then
      rm -f -- "$ROOT/$path"
      printf '[removed untracked] %s\n' "$path"
    else
      printf '[untracked] %s\n' "$path"
    fi
  fi
done

if (( ! APPLY )); then
  echo
  echo "Dry run only. Re-run with --apply to remove the listed files."
  exit 0
fi

for path in "${LIVE_FILES[@]}"; do
  case "$path" in
    *.sh) bash -n "$path" ;;
    *.py)
      python3 - "$path" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
      ;;
  esac
done

echo
echo "Cleanup complete. Review the repository state before committing:"
git status --short
