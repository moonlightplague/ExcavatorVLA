# Changelog

## 2026-11-13 — Coarse MPM repeat collection

- Select the shared coarse/fine soil profile explicitly and reject requested/active profile mismatches.
- Preserve fine-reference mass in coarse planning thresholds, search actual avalanched source material, and query its measured surface beyond the generation box with the existing physical path checks.
- Clear source identity, cut history, scoop index and fallback candidates at the next episode boundary, retain scene/amount provenance, and keep coarse-specific search hooks off for fine MPM and PhysX.
- Require at least 50 kg of measured new source delivery in every coarse dig by default; independently enforce the gate before streaming publication and during episode validation.
- Add opt-in `coarse_mass` acceptance based on productive physical cycles and retained truck mass; preserve legacy decisions and convert native-grain contact/progress/load controls to equivalent kilograms without changing geometry limits.
- Preserve the full feasible target pool through mass ranking, deduplicate clamped targets, and continue the checked cut while an early-contact cue reports an underfilled bucket.
- Measure fixed-attempt export rates with a strict greater-than target, optional confidence-bound gate, disjoint worker seeds, model/source hashes, per-dig mass and elapsed-time accounting.
- Validate the selected amount-9 recipe on 16 fixed scenes: 9/16 exports (56.25%), with all 90 delivered loads at least 74.56 kg; the 95% Wilson interval of 33.18–76.90% does not establish a general above-50% guarantee.
- Independently audit all nine exports (22,186 frames per camera, 153 real-reader boundary samples); pass 196 CPU regression checks and document failed development variants without promoting them.
- Record a provisional amount-9 recommendation, throughput and full reproduction commands; remove task-generated test datasets and diagnostics while preserving ignored shared experiments.

## 2026-11-12 — Collection artifact maintenance

- Ignore local assistant/editor state, runtime caches, experiment-tracking outputs and generated coverage reports.
- Verify five collection-launcher contract tests while preserving collection defaults and export behavior.

## 2026-11-09

- Add a measured-swing reachable workspace with randomized side loading, configurable release clearance, and explicit Newton solve frequency/point budget.
- Default collection to ten cycles, Newton MPM at 60 Hz/32,000 points, sand amount 3, the reachable side-loading profile and explicit episode efficiency acceptance; retain strict acceptance as an option.
- Preserve avalanched source material identity and distinguish new source-ID deposits from losses of an earlier truck load.
- Add an explicit ten-cycle episode efficiency policy while retaining original strict decisions, hard capture/delivery/execution checks, physical returns, and the source audit.
- Add a fixed-attempt Newton benchmark that counts failed plans and rejections, records seed ranges and source hashes, and reports acceptance uncertainty and strict-equivalent outcomes.
- Retain scoop quality measurements after raw pruning; audit episode policy and active soil configuration, and report rejected cleanup separately from export.
- Document the diagnosis and parameter choices in `TUNING.md`.
- Complete a fixed four-scene Newton trial: 2/4 accepted and exported under the episode policy versus 1/4 strict-equivalent, with ten productive rounds/returns each; validate 4,929 frames per camera and the real LeRobot reader. See `VALIDATION.md` for limits and hard failures.

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
