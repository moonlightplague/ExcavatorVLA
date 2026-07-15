# ExcavatorVLA Documentation Status

Updated: 2026-07-15

This file is the current entry point for project documentation. Older reports
under `docs/analysis/` remain useful as historical measurements, but they may
describe earlier dataset schemas or runtime behavior.

## Current Project State

The data-generation branch currently exports:

- three RGB observations: `observation.images.0/1/2`, each `256 x 256 x 3`;
- a 28D `observation.state`;
- a separate 4D `observation.effort`;
- a 4D joint-command velocity `action` in canonical order
  `[swing, boom, arm, bucket]`;
- an episode task prompt and canonical phase index;
- per-episode videos with a four-frame keyframe interval;
- a uniform export time policy whose speed scaling also updates action and
  derivative fields.

The canonical export identifiers are:

```text
task prompt:  excavator_relative_task_v3_heading_frame
state schema: excavator_state_v3_28d_plus_4effort_phase_index10
video layout: per_episode
```

The exact field names are maintained in `excavator_dataset_tools.py` as
`LEROBOT_STATE_NAMES_28D`, `LEROBOT_CANONICAL_PHASE_NAMES`, and the export
feature metadata. Source constants, exported `meta/info.json`, and checkpoint
configuration take precedence over older prose documents.

## Deployment Handoff

- [18D SmolVLA deployment handoff](vla_18d_deployment_handoff.md): immediate
  guidance for the existing 14D state + 4D effort checkpoints, including the
  current simulator slowdown diagnosis and a later 32D migration path.
- [Execution-chain performance report](excavator_execution_chain_performance_report.md):
  data-generator execution and Isaac update costs.
- [Path-planning analysis](excavator_path_planning_analysis.md): dig and unload
  planning architecture.
- [Profile-mode stall postmortem](profile_mode_scene_commit_stall_postmortem.md):
  resolved mode-specific scene-commit stall.

## Compatibility Rule

Existing 18D checkpoints remain 18D models. Do not feed the current 28D state
or the effective 32D state-plus-effort vector into them. The deployment adapter
must be selected from the checkpoint's own feature metadata, normalization
statistics, and training dataset FPS. Migration to the current schema is a
separate model change.
