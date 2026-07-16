# Current Auto-Collect Runtime Architecture

Status: current working-tree reference

Audited: 2026-07-15

This document is the canonical overview of the current auto-collect runtime.
It describes function names and behavior in the working tree on the audit
date. Older performance and analysis reports remain useful as historical run
evidence, but their line numbers and some call chains are no longer current.

## Auto-Collect Attempt Chain

The current top-level chain is:

```text
auto_collect_loop()
  -> auto_collect_one_episode()
     -> auto_collect_apply_scene_randomization()
     -> auto_collect_scene_pre_sample_gate()
     -> auto_collect_initial_pose_pre_sample_gate()
     -> auto_collect_prepare_environment()
     -> action-channel and run preflight
     -> auto_collect_find_plan()
     -> auto_collect_begin_episode()
     -> dataset_camera_warmup_for_auto_run_once()
     -> dataset_camera_start_background()
     -> execute_dig_target_ball()
     -> auto_collect_finish_episode()
```

Failed scene, initial-pose, preparation, action-channel, and planning gates are
recorded as diagnostics before a trainable episode begins. Camera recording is
started only after the episode and plan have been accepted.

## Planning Chain

The main dig-plan entry remains:

```text
auto_collect_find_plan()
  -> build_dig_plan_from_current_target()
     -> adaptive_dig_plan_candidates()
     -> rank_dig_plan_candidates_cheap()
     -> full planning for the primary candidate window
     -> expert-cost comparison
     -> staged-prefix or progressive fallback when required
```

The current inner candidate policy is `cheap_rank_v1`. By default it performs
full planning for the best three cheaply ranked candidates
(`DIG_PLAN_PRIMARY_FULL_TOPK=3`) and selects the lowest expert cost among that
primary window. If no primary candidate is complete, it can use a qualified
staged prefix or continue progressively until a fallback succeeds or the time
budget expires.

`PLAN_BUILD_SUMMARY` records the cheap-rank time, primary window size,
evaluated candidates, selection scope, timeout state, and selected candidate.
This is distinct from the outer auto-collect target/ring search, which can
evaluate several target points.

Dig and truck-unload planning share the lower-level path validation stack but
not one identical top-level planner. The current detailed planning reference is
`excavator_path_planning_analysis.md`, including its 2026-07-15 status note.

## Execution Chain

The current execution path is:

```text
execute_dig_target_ball()
  -> execute_dig_plan_step()
     -> move_to_profile()
        -> send joint targets
        -> step_updates(control_step_frames())
        -> record due dataset samples and stage gates
```

Normal motion is not a single joint assignment. A planned stage is divided
into control points, each of which advances Isaac/Kit and therefore PhysX,
particles, rendering, callbacks, and any due dataset work.

`CONTROL_HZ` is 30 Hz. `control_step_frames()` defaults to two Isaac updates
per control step unless overridden. Consequently, wall-clock stage duration is
strongly affected by the cost of `SimulationApp.update()` and by scene particle
and rendering load; it is not determined only by four-joint interpolation.

### Optional Sampled Replay

`move_to_profile_sampled_replay()` exists, but
`EXCAVATOR_FAST_SAMPLED_REPLAY` defaults to off. When explicitly enabled, the
current guard permits it only for `pre_dig`. Sand-contact, lift, loaded carry,
unload, and dump motions remain on the normal physics path so live payload
particles are not teleported or detached from the bucket.

## Clock And Sampling Semantics

The runtime separates three concepts:

```text
wall time       profiling, scheduling, timeout observation
train time      exported sample timestamps and episode timeline
derivative time dq, ddq, and action derivative calculation
```

`DATASET_USE_SIM_TIME` defaults to true. `dataset_record_sample()` and
`dataset_record_sample_async()` therefore use simulation/train time for sample
timestamps, and `dataset_motion_derivatives()` receives that same sample time.
State tracks separate wall, train, and derivative clocks. Older reports that
describe exported timestamps as wall time are no longer current.

The default dataset sample interval is 0.20 seconds, or a 5 Hz target. Camera
frequency defaults to the same nominal rate. These are target simulation-time
rates, not guarantees of real-time wall-clock throughput.

During auto collect, `AUTO_COLLECT_OWNS_STEP_CLOCK` defaults to true. The main
runtime loop becomes passive while auto collect is active, preventing two
coroutines from competing to advance Kit. Camera capture runs through the
background camera scheduler after one run-level warmup rather than owning a
second simulation clock.

## Camera And Sample Contract

Trainable samples require a complete three-camera observation. The camera
module prepares the supported render backend, performs the run-level warmup,
and captures in the background while execution advances. Dataset sample rows
are aligned by their recorded train timestamp; capture implementation may
differ by platform, but Windows and Linux export the same dataset feature
contract.

The canonical image features are:

```text
observation.images.0
observation.images.1
observation.images.2
```

Camera initialization failure, persistent black output, or unavailable views
must stop collection rather than produce a false trainable episode. Per-stage
late frames use bounded retry/fallback behavior instead of redefining task
success.

## Sand Metrics

The hot sample path primarily needs bucket occupancy from the bucket's closed
volume. Full pile/bin/spill classification is reserved for explicit phase,
lift, dump, and final quality gates where possible. Some forced metric calls
remain for correctness gates, so historical counts from older runs should not
be assumed to describe the current call frequency.

The bucket volume is mesh-backed and moves with the bucket transform. Candidate
particle filtering is a broad phase only; the closed-volume test remains the
authoritative inside test.

## Runtime Modes

`data`, `debug`, and `profile` use the same task, planning, physics, sampling,
and quality semantics. Their intended differences are observability only:

```text
data     minimal logging and no calculation visualization
debug    diagnostic logging and calculation visualization
profile  debug visibility plus wall-time telemetry
```

A mode-specific debug draw must not author or refresh scene geometry from the
passive main loop while auto collect owns the clock. The resolved incident is
documented in `profile_mode_scene_commit_stall_postmortem.md`.

## Canonical Function Map

Use symbol search instead of copied line numbers when navigating current code:

```text
auto_collect_loop
auto_collect_one_episode
auto_collect_apply_scene_randomization
auto_collect_scene_pre_sample_gate
auto_collect_prepare_environment
auto_collect_find_plan
auto_collect_begin_episode
auto_collect_finish_episode
build_dig_plan_from_current_target
rank_dig_plan_candidates_cheap
execute_dig_target_ball
execute_dig_plan_step
move_to_profile
move_to_profile_sampled_replay
dataset_record_sample
dataset_record_sample_async
dataset_motion_derivatives
dataset_camera_start_background
sand_metrics_current
```

Source code, emitted run metadata, and exported `meta/info.json` take
precedence if this document and a future working tree diverge.
