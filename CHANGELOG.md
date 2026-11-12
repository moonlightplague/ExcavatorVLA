# Changelog

## 2026-11-12 — Training artifact maintenance

- Ignore local assistant/editor state, runtime caches, experiment-tracking outputs and generated coverage reports.
- Preserve PI05 milestone checkpoints explicitly retained by the single-cycle and repeated-task comparison configurations.
- Verify 13 PI05 training and checkpoint tests; preserve pending training/evaluation changes separately.

## 2026-11-11

- Add latest-first, dual-GPU checkpoint comparisons with five-cycle model diagnostics, combined per-checkpoint summaries and mandatory repeated-trace validation.
- Require identical base weights, precision, fine-tuning method, history and action contracts when comparing checkpoints; keep the old single-cycle defaults unchanged.
- Document recorded task-goal schedules, mandatory final returns, cycle-local particle accounting, collection/strict score separation, step budgets, outputs and limitations.

## 2026-11-10

- Commit complete PI05 optimizer/RNG checkpoints before validation and use a separate Gloo CPU group for checkpoint coordination, preventing idle ranks from leaving CUDA collectives pending during validation.
- Resume the repeated-history experiment with a deterministic eight-frame-per-validation-episode preview, record selected frame IDs and mark metrics as partial; preserve the model, training data, effective batch, scheduler and 2k checkpoint retention.

## 2026-11-09

- Add user-approved shared-GPU restart from complete optimizer/RNG state, archive interrupted attempts/logs, and keep one rolling 100-step recovery checkpoint alongside the requested 2k milestones.
- Release gradients after the first-update audit and before validation/checkpoint inference without changing optimizer updates or batch contracts.
- Correct repeated expert-replay acceptance to use immutable archived collection payload thresholds, fix exclusive observation boundaries, and support verified reassessment of blocked runs without discarding the original failed gate or replay evidence.
- Add an opt-in bucket command-target slew profile of 20 rad/s while preserving legacy limits and rejecting checkpoint/evaluation profile mismatches.
- Add immutable repeated-data action relabeling, per-parent PhysX evidence, causal action-history updates and a frozen 80/10/10 split preserving prior pilot parent assignments.
- Gate C's requested dual-GPU 20k run and ten 2k checkpoints on complete five-cycle expert replay, measured trajectory checks and deposited-load checks.

## 2026-11-08

- Add opt-in PI05 scheme C with causal three-camera/state windows, shared online/offline padding, state-history prompts, and versioned checkpoint/resume contracts; retain single-frame behavior.
- Add repeated-data and verified single-cycle history configurations and document the remaining repeated-task/data prerequisites.
- Add the PhysX single-cycle PI05 dataset builder with parent-disjoint 20/4/4 splits, collection-backend evidence, and velocity-integrator compatibility filtering.
- Add the dual-GPU LoRA/full-expert 10k pilot, exact continuation configuration, and comparisons of the explicitly retained 2k/4k/6k/8k/10k checkpoints.
- Require requested soil backends to match verified dataset provenance and carry that evidence into training checkpoints.
- Document expert replay gates, model-video outputs, and checkpoint evaluation; cover backend tampering, command-target mismatches, and evaluation scheduling.

## 2026-10-30

- Ignore local datasets, checkpoint directories, runtime logs, and development caches.
- Add PI05 VLM LoRA training, checkpoint reconstruction, continuation and A/B evaluation tools with regression coverage.
- Add ACT ablation configurations and split manifests.
- Add rollout selection and fine-tuning dataset conversion; guard overwrite ownership and selection accounting.
- Handle missing rollout metrics when computing rewards.
