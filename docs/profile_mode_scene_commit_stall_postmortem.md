# Profile Mode Attempt-2 Scene Commit Stall Postmortem

Date: 2026-07-11

Status: Resolved and verified in a subsequent profile-mode run.

## 1. Problem Summary

Excavator auto-collect could run multiple attempts normally in `data` mode,
but `profile` mode could stop after the first episode. The terminal appeared
idle and no second attempt completed.

The important invariant is:

```text
data / debug / profile must run the same auto-collect state machine,
the same physics frames, the same planner, the same gates, and the same
training-data collection.
```

The modes may differ only in observability:

- `data`: essential status/errors and maximum throughput.
- `debug`: diagnostic logs and Calc Viz.
- `profile`: debug behavior plus detailed timing telemetry.

Mode selection must never shorten waits, skip physics, change a gate, or
change episode acceptance.

## 2. Relevant Execution Chain

All modes use the same attempt chain:

```text
attempt start
-> scene randomization
-> pause timeline
-> clear old particles
-> apply truck transform
-> apply sand parameters
-> refresh unload mesh
-> applied-scene geometry gate
-> scene commit (Kit update)
-> resume timeline
-> prepare/reset sand
-> plan
-> begin episode
-> execute dig/lift/unload stages
-> record state/camera/data
-> score and finalize
-> fixed between-episode physics updates
-> next attempt
```

Profile mode additionally enabled:

- function timing summaries;
- `step_updates()` caller/lock/update telemetry;
- launcher update snapshots;
- diagnostic payload fields;
- visible and live Calc Viz geometry.

The auto-collect control flow itself was not intended to differ.

## 3. Decisive Log Evidence

The most useful run was:

```text
excavator_auto_dataset/run_20260711_195642
```

Episode 1 completed and was rejected for an unrelated unload path collision.
The run then entered attempt 2 normally.

The final event sequence was:

```text
20:00:00.160  scene_pause:attempt_2          update_start
20:00:00.175  scene_pause:attempt_2          update_ok

20:00:00.275  scene_particle_clear:attempt_2 update_start
20:00:00.292  scene_particle_clear:attempt_2 update_ok

truck_transform                              ok
sand_parameters                              ok
unload_mesh_refresh                          ok
geometry_gate                                ok

20:00:10.784  scene_commit:attempt_2         update_start
              no update_ok followed
```

This isolated the hard stall to:

```python
await app.next_update_async()
```

inside the scene commit after all USD scene edits had completed.

Before the stall, the run was not generally slow:

```text
step_updates frames:        1295
average frame:              61.9 ms
maximum completed frame:    247 ms
total lock wait:            0.074 s
total Kit update time:      79.96 s
```

Therefore this was not:

- an asyncio lock deadlock;
- sand settle waiting;
- planner computation;
- dataset image writing;
- particle metrics;
- the fixed between-episode frame count.

The lock had already been acquired. Kit stopped while committing scene/Hydra
changes.

The native Kit log did not report a corresponding PhysX, GPU, CUDA, Hydra, or
viewport exception. The absence of `update_ok` was the strongest signal.

## 4. Root Cause

When auto-collect owns the Kit update clock, the normal `main()` loop becomes
passive. In profile mode, however, Calc Viz is enabled and the passive loop was
still allowed to call:

```python
draw_trace(force=False)
maybe_draw_bucket_sand_count_debug(force=False)
```

The passive loop sleeps only a few milliseconds between checks.

During attempt-2 scene randomization, the auto task performs a scene
transaction while the timeline is paused:

- particle prims are removed;
- truck transforms change;
- sand-site parameters change;
- unload mesh guides and markers change;
- geometry is queried and validated.

At the same time, the passive profile loop could author debug BasisCurves,
marker transforms, visibility, colors, and bucket-volume points into the same
USD stage. The applied geometry gate took about 10 seconds in the decisive
run, leaving enough time for repeated Calc Viz refreshes while the scene was
only partially updated.

The next `scene_commit` forced Kit/Hydra to process overlapping scene changes.
That update never returned.

Data mode did not reproduce the problem because Calc Viz is disabled there,
so the scene transaction had only one USD authoring owner.

## 5. Final Fix

The project already had a policy function:

```python
debug_visual_authoring_allowed()
```

It returns false while auto-collect is active but no episode is currently
recording. That is exactly the unsafe scene-reset/prepare interval.

The passive main-loop condition was changed from:

```python
if debug_visuals_enabled():
```

to:

```python
if debug_visuals_enabled() and debug_visual_authoring_allowed():
```

This establishes a single-writer rule for scene transactions:

```text
scene randomization/prepare owns USD authoring while between episodes;
passive Calc Viz authoring resumes after episode recording begins.
```

Existing Calc Viz prims are not deleted or hidden. They simply stop being
rewritten while the scene transaction is in progress, then resume normal live
updates afterward.

## 6. Why the Fix Preserves Mode Semantics

The fix does not change:

- auto-collect stages;
- physics frame counts;
- the 90 between-episode frames;
- randomization values;
- sand reset behavior;
- planner inputs or outputs;
- motion commands;
- camera sampling;
- state/action sampling;
- quality gates;
- trainable/rejected decisions.

It only serializes nonessential debug USD authoring around a scene transaction.
Training data and physical execution remain identical across modes.

## 7. Rejected or Incorrect Fix Directions

### Reducing profile-only physics frames

Changing the profile inter-attempt wait from 90 frames to 2 frames would have
hidden the symptom by reducing the number of expensive updates. It would also
change physical behavior between modes and violate the mode invariant. This
approach was reverted.

### Treating the camera wall-clock scheduler as the root cause

Camera scheduling can affect throughput, but the decisive run completed camera
work and then stopped specifically at attempt-2 `scene_commit`. A camera clock
change did not explain the missing `update_ok` and was not retained as this
fix.

### Changing sand settle/reset logic

The stall occurred before prepare/reset sand began. Sand settling was therefore
downstream of the hard stall and could not be its cause.

### Treating `summary.json` as final process state

The run had already written the episode-1 summary before entering attempt 2.
Consequently, `summary.json` still reported one attempt even while the live
debug timeline was inside attempt 2. For an interrupted run, the tail of
`debug_timeline.jsonl` is more authoritative than the last episode summary.

## 8. Verification Criteria

A successful profile-mode verification must show this complete sequence for
attempt 2 and later attempts:

```text
scene_pause     update_start -> update_ok
particle_clear  update_start -> update_ok
scene_commit    update_start -> update_ok
scene_resume    update_start -> update_ok
AUTO SCENE RANDOMIZE attempt=N ok=True
```

It must also satisfy:

- profile Calc Viz remains visible during normal episode execution;
- profile telemetry remains enabled;
- multiple attempts complete;
- data mode behavior and throughput do not regress;
- no profile-only frame-count or gate override exists.

The user subsequently confirmed that profile mode no longer stalled after this
change.

## 9. Engineering Rule Going Forward

USD/Hydra scene mutation should follow a single-writer model.

In particular:

```text
Do not author debug geometry from a passive/background task while another task
is pausing the timeline, deleting particle prims, moving scene roots, or
rebuilding collision/render geometry.
```

Diagnostics may observe any stage, but visual authoring must be suspended or
queued until the scene transaction is complete. This preserves full profile
visibility without allowing diagnostics to alter execution stability.

## 10. Relevant Code and Artifacts

- `scripts/excavator_app/excavator_runtime.py`
  - `step_updates()`
  - `debug_visual_authoring_allowed()`
  - `auto_collect_apply_scene_randomization()`
  - passive section of `main()`
- `run_vla_train_scene.py`
  - launcher update profiling
- `excavator_auto_dataset/run_20260711_195642/debug_timeline.jsonl`
  - decisive attempt-2 scene-commit trace

