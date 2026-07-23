# Excavator VLA Dataset v2.0

## Scope

Dataset v2.0 adds long-horizon episodes through `--log-mode data_multi`.
One episode contains multiple consecutive scoop-and-dump cycles in one
unchanged scene:

```text
prepare scene once
-> plan and execute scoop 1
-> inspect a fresh sand snapshot from the current actual joint pose
-> execute another scoop only when an effective, spatially distinct,
   fully plannable target remains
-> repeat the inspect/plan/execute cycle
-> finalize on evidence-based target exhaustion
```

There is no sand reset, robot home reset, or truck reset between scoops.
Planning failure or quality failure in any scoop rejects the complete
long-horizon episode.

The scoop count is adaptive and is not a fixed endpoint. The default
`--scoops-per-episode 1` only requires one valid scoop before a natural stop can
complete the episode. Collection then continues until the fresh sand snapshot
has no target with sufficient local/swept density or real surface support, no
spatially distinct candidate remains, or no candidate passes complete planning
from the current real pose. A value above one is an optional dataset quality
floor, not a requested endpoint. `--max-scoops-per-episode` is only a runaway
guard; reaching it rejects the episode.

## Collection Contract

- Dataset release: `v2.0`
- Episode mode: `multi_scoop`
- Default minimum scoops per episode: `1`
- Technical maximum guard: `64`
- Natural stop: no effective spatial target remains, or no remaining target
  is fully plannable from the current actual pose
- Sampling: strict simulation-time `10 Hz`
- Camera streams: Cam 0, Cam 1, and Cam 2 for every row
- Camera policy: fresh frame only; no reused image paths
- State: existing 28D continuous observation
- Effort: separate 4D measured joint effort
- Stage: categorical `observation.stage_current_id`, values 0 through 9
- Scoop supervision:
  - `observation.scoop_index`
  - `observation.scoops_target` (compatibility alias for the minimum)
  - `observation.scoops_min`
  - `observation.scoops_max`
  - `observation.scoops_completed`

The 28D continuous state remains unchanged. Stage and scoop fields are
separate categorical supervision, so v2.0 does not silently change the
continuous state dimension.

The task text explicitly requests continued excavation without resetting the
scene until no effective target remains, with the configured minimum count.
The exporter preserves this text in `meta/tasks.parquet` and retains
`task_index` in the data parquet.

## Recovery Supervision

`--log-mode data_multi` remains the clean expert-only dataset mode.

`--log-mode data_multi_recovery` adds one controlled, path-validated,
unloaded pose deviation before the second scoop. The next scoop is replanned
from the actual perturbed joint pose, so the correction is produced by the
same expert planner used for normal collection. It does not inject a
collision, scene overlap, material loss, or joint teleport.

Recovery rows add these fields without changing the 28D state or 4D action:

```text
observation.recovery_active
observation.recovery_type_id
observation.recovery_attempt_index
action_is_expert
action_loss_weight
```

The deliberate deviation has `action_is_expert=0` and
`action_loss_weight=0`. Expert correction and natural carry-posture recovery
have weight `1`. Training code must multiply the action loss by
`action_loss_weight`; otherwise the model is explicitly trained to reproduce
the injected mistake.

Recovery type IDs are:

```text
0 none
1 controlled_pose_offset
2 carry_posture
3 path_replan
4 underfill_redig
5 unload_alignment
```

The original task prompt is retained. The model is asked to complete the
excavation task, not to intentionally make a mistake.

## Production Command

Run randomized Windows headless collection:

```powershell
cd D:\450\apps\isaacsim

.\python.bat D:\450\assets\usd\URDF_real3\run_vla_train_scene.py `
  --headless `
  --auto-collect `
  --success-count 100 `
  --max-attempts 1000 `
  --log-mode data_multi `
  --max-scoops-per-episode 64 `
  --graphics-api d3d12
```

`success-count` counts complete multi-scoop episodes, not individual scoops.

Collect the compatible recovery tier with:

```powershell
cd D:\450\apps\isaacsim

.\python.bat D:\450\assets\usd\URDF_real3\run_vla_train_scene.py `
  --headless `
  --auto-collect `
  --success-count 100 `
  --max-attempts 1000 `
  --log-mode data_multi_recovery `
  --max-scoops-per-episode 64 `
  --graphics-api d3d12
```

Keep expert-only and recovery episodes identifiable during training. They
share the same v2.0 state/action/camera contract and can be combined after the
trainer enables the action-loss mask.

## Local Regression Command

The checked-in fixed profile is only for deterministic local regression. It
does not replace randomized production collection.

```powershell
cd D:\450\apps\isaacsim

.\python.bat D:\450\assets\usd\URDF_real3\run_vla_train_scene.py `
  --headless `
  --auto-collect `
  --success-count 1 `
  --max-attempts 1 `
  --log-mode data_multi `
  --graphics-api d3d12 `
  --fixed-scene-profile D:\450\assets\usd\URDF_real3\configs\data_multi_v2_validation_scene.json `
  --disable-export `
  --no-wait-export
```

## Acceptance Criteria

A v2.0 episode is trainable only when:

1. At least the configured minimum number of scoop cycles completed.
2. Every scoop passed execution and per-scoop bucket/lift/bin/spill gates.
3. The episode ended with evidence that no effective spatial target remains
   or no remaining target is fully plannable from the current actual pose.
4. The scene was not reset between scoops.
5. Scoop indices are monotonic and each next scoop starts at `pre_dig`.
6. All trajectory timestamps are strictly monotonic on the 10 Hz grid.
7. Every row has three unique, present, non-black camera frames.
8. Shared LeRobot export passes both standard and VLA validation.

## Verified Adaptive Local Run

Windows Isaac Sim headless validation:

```text
run: excavator_auto_dataset/run_20260723_210347
attempts: 1
trainable: 1
scoops completed: 1
natural stop: no_plannable_new_dig_target
trajectory rows: 197
Cam 0/1/2: 197/197/197 unique present frames
```

Randomized Windows headless production validation was completed in two
consecutive runs:

```text
run_20260723_210918: 16 trainable episodes
run_20260723_231258: 4 trainable episodes
total: 20 trainable episodes, 26 successful scoops
scoop distribution: 14 episodes with 1 scoop, 6 episodes with 2 scoops
trajectory rows: 5,196
Cam 0/1/2: 5,196/5,196/5,196 unique present non-empty frames
missing/empty camera files: 0
adaptive natural-stop violations: 0
```

All 20 episodes ended naturally with `no_plannable_new_dig_target`. The
technical maximum guard did not terminate any episode.

## Prior Recovery Contract Run

This earlier fixed-profile run validates recovery annotations and loss masks.
It is not evidence of a fixed scoop endpoint in the current adaptive
collector.

```text
run: excavator_auto_dataset/run_20260723_184254
attempts: 1
trainable: 1
scoops: 3/3
trajectory rows: 734
simulation duration: 73.3 s
timestamp violations: 0
Cam 0/1/2: 734/734/734 unique non-black frames
controlled perturbation rows: 6, action loss weight 0
expert recovery rows: 49, action loss weight 1
loss-mask inconsistencies: 0
controlled recovery injected/completed: true/true
LeRobot standard ready: true
VLA training ready: true
```

The validated recovery export is:

```text
excavator_auto_dataset/run_20260723_184254/lerobot_v3
```
