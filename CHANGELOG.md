# Changelog

## 2026-10-30

- Plan continuous cuts with mesh-based floor headroom, rejecting clearance detours that raised and lowered the arm between insertion and pulling.
- Prefer a checked direct physical return; retain collision-checked clearance routes around obstacles.
- Record sampled source-ID payload, unique scooped material, retention, spillage, mass, cycle times, runtime, throughput, bed fill, and existing quality scores for successful and failed episodes.
- Persist episode reports in `repeat_metrics.json`, episode metadata, and the run-level metric index; add metric and continuous-cut audits.
- Validate with 57 baseline tests and a live three-scoop collection and strict LeRobot export; see `VALIDATION.md`.
- Ignore generated datasets, model weights, checkpoints, runtime logs, and development caches.
