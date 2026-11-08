# Changelog

## 2026-11-08

- Add collection-matched evaluation scene preparation using the existing repeat sampler, initial pose, digging planner, and low-fill unload selection.
- Record feasible scene setups without executing expert digging actions, with deterministic seed streams and isolated-GPU renderer configuration.

## 2026-10-31

- Export each successful repeat episode immediately with worker backpressure and a shared append lock; restore the policy Python environment outside Isaac.
- Preserve a validated export receipt before deleting large raw episode files. Keep compact metadata, scores, cycle reports, and episode metrics; retain unexported raw data on export failure.
- Write combined batch totals and averages across workers and retry rounds, with separate trainable and incomplete episode populations and explicit missing-value denominators.
- Support metrics-only episode validation and cover export failures, concurrent workers, cleanup, batch summaries, and real two-episode Parquet/video append after raw pruning.

## 2026-10-30

- Plan continuous cuts with mesh-based floor headroom, rejecting clearance detours that raised and lowered the arm between insertion and pulling.
- Prefer a checked direct physical return; retain collision-checked clearance routes around obstacles.
- Record sampled source-ID payload, unique scooped material, retention, spillage, mass, cycle times, runtime, throughput, bed fill, and existing quality scores for successful and failed episodes.
- Persist episode reports in `repeat_metrics.json`, episode metadata, and the run-level metric index; add metric and continuous-cut audits.
- Validate with 57 baseline tests and a live three-scoop collection and strict LeRobot export; see `VALIDATION.md`.
- Ignore generated datasets, model weights, checkpoints, runtime logs, and development caches.
