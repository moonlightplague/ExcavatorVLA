# VLA 32D Calculation Audit

Audited: 2026-07-17

> Historical v3 audit. Its findings motivated the v4 causal/local observation
> contract implemented on 2026-07-18. Use
> [vla_32d_deployment_contract.md](vla_32d_deployment_contract.md) for the
> current schema and deployment requirements.

Scope:

```text
scripts/excavator_app/excavator_runtime.py
scripts/excavator_app/excavator_dataset_camera.py
excavator_dataset_tools.py
run_dataset_dashboard_v10.py
excavator_auto_dataset/.dashboard_success
```

## Verdict

The 28D state plus 4D effort feature order is internally consistent, but the
current dataset is not yet safe to describe as fully time-aligned 32D
observation data. Three issues materially affect model semantics:

1. A mixed 2 Hz / 5 Hz success pool is exported with one pool-level FPS.
2. A row can combine camera-time q/timestamp with record-time effort, bucket
   geometry, particle count, and phase.
3. Export speed scaling recomputes kinematic derivatives but cannot recompute
   measured physical effort.

These should be fixed before the current 32D dataset is treated as a canonical
training contract.

## High-Severity Findings

### H1. Mixed source frequencies use one pool-level FPS

Relevant code:

```text
excavator_dataset_tools.py
  infer_export_fps()
  lerobot_export_config_for_run()
  apply_export_time_policy_to_trajectory()
  collect_lerobot_rows()

run_dataset_dashboard_v10.py
  dashboard_transfer_success_records()
```

`infer_export_fps()` reads only:

```text
.dashboard_success/camera_config.json
```

The dashboard copies that file only when the pool does not already have one.
It therefore represents the first transferred run, not every episode.

Observed pool state:

```text
source run_20260708_112942: 2 Hz
source run_20260708_195935: 2 Hz
source run_20260711_214811: 5 Hz
pool camera_config.json:    2 Hz
export speed_scale:        10x
published info.json FPS:   20 Hz
```

Every accepted row is retained. The exporter does not resample source episodes
to one physical observation rate; it simply assigns:

```python
new_time[index] = index / effective_fps
```

Consequences:

```text
2 Hz source -> 20 Hz timeline: 10x time compression
5 Hz source -> 20 Hz timeline:  4x time compression
```

The two source groups therefore receive different effective speed scaling even
though the export policy says one `speed_scale`.

Required fix:

```text
1. Store source sample-rate metadata per transferred episode.
2. Validate the raw median timestamp delta for every episode.
3. Choose one explicit policy:
   a. export separate 2 Hz and 5 Hz datasets;
   b. downsample 5 Hz episodes to a shared 2 Hz timeline; or
   c. perform real observation/action resampling to a canonical FPS.
4. Never infer the whole success pool rate from one camera_config.json.
```

Simply relabeling all rows as a higher FPS is not resampling.

### H2. One row can contain values from different physical times

Relevant code:

```text
excavator_dataset_camera.py
  store_latest_capture()
  latest_capture_payload()

excavator_runtime.py
  dataset_record_sample_async()
  dataset_record_sample()
  dataset_observation_state()
  dataset_joint_effort_observation()
```

The camera record correctly stores:

```text
capture_pose_sim_time
capture_q
capture_q_cmd
```

`dataset_record_sample_async()` then uses those values as the row timestamp,
measured q, and commanded q. However, `dataset_record_sample()` subsequently
reads these values at record time:

```text
measured effort
bucket particle occupancy
bucket tip world transform
bucket load-point world transform
phase passed by the current execution loop
```

Those fields are not stored in the camera capture snapshot.

Measured evidence from
`run_20260713_144724/episode_000001`:

```text
rows:                     113
camera frame age mean:    about 285 ms
camera frame age maximum: about 742 ms
```

The q/action timeline is therefore camera-aligned, but the complete 32D row is
not guaranteed to describe the same physical instant.

Required fix:

```text
CaptureObservationSnapshot:
  simulation_time
  q_real
  q_cmd
  measured_effort
  bucket_tip_world
  bucket_load_world
  bucket_particle_count
  phase
  camera triplet sequence
```

The dataset row must be built from one immutable snapshot. Alternatively, the
camera must be captured synchronously at the dataset sample boundary.

### H3. Speed-scaled kinematics and measured effort are physically inconsistent

`apply_export_time_policy_to_trajectory()` recomputes:

```text
obs.dq
obs.ddq
action
action.ddq
bucket_load_rate
```

on the compressed uniform timeline. `observation.effort` remains the original
Isaac measurement because torque cannot be derived from q alone.

Current pool policy:

```text
speed_scale = 10
base pool FPS = 2
effective export FPS = 20
```

Observed exported action maxima include approximately:

```text
swing  6.21 rad/s
boom   4.26 rad/s
arm    3.45 rad/s
bucket 5.87 rad/s
```

The row can therefore claim a much faster motion while retaining effort from
the slower physical simulation.

Required policy:

```text
For an effort-conditioned model:
  use speed_scale=1; or
  regenerate the trajectory physically at the target speed.

For an artificially time-scaled model:
  remove measured effort from the model observation; and
  clamp action to deployment joint-velocity limits.
```

### H4. Measured effort contains large solver/contact outliers

The effort indexing path is correct when named DOFs are available:

```python
JOINT_INDICES = [
    swing_real_index,
    boom_real_index,
    arm_real_index,
    bucket_real_index,
]
ROBOT.get_measured_joint_efforts(joint_indices=JOINT_INDICES)
```

However, a checked trainable episode contained absolute effort maxima of
approximately:

```text
swing:  492,402
boom: 2,543,199
arm:  1,061,432
bucket: 269,243
```

The exporter uses ordinary mean/std/min/max statistics. Large constraint or
solver impulses can dominate normalization and make normal effort variation
nearly invisible.

Required fix:

```text
1. Record effort validity and contact/freeze state per row.
2. Report p50/p95/p99 and invalid impulse counts.
3. Reject physically failed/frozen rows before stats.
4. Use training-time clipping based on a documented physical or robust
   percentile limit; preserve unclipped values only as diagnostics.
```

## Medium-Severity Findings

### M1. `bucket_load_rate` is an unsmoothed one-step particle-count derivative

Current calculation:

```python
bucket_load_rate = (
    current_bucket_particle_count - previous_bucket_particle_count
) / dt
```

It is sensitive to particles crossing the closed-volume boundary and to time
scaling. In the current exported pool:

```text
absolute p50:   140 particles/s
absolute p95: 3,820 particles/s
absolute p99: 8,360 particles/s
maximum:      15,640 particles/s
```

Recommended fix:

```text
rate = robust derivative of a 3-5 sample median-smoothed load count
rate = clip(rate, documented capacity-based bound)
```

A normalized fill fraction and fill-rate fraction would be more portable than
raw particle count when sand particle settings change.

### M2. Canonical `loaded_transit` does not map to its own phase index

The canonical list defines:

```text
8 loaded_transit
```

but:

```python
canonical_lerobot_phase_index("loaded_transit") is None
```

The runtime phase mapper has the same omission. Current route labels usually
contain `clearance_route_post`, `staged_unload`, or `high_carry`, which do map
to 8, so current episodes often avoid the defect. A literal
`phase="loaded_transit"` would be skipped by the exporter.

Required fix: explicitly map `loaded_transit` to index 8 in both phase mappers
and add a test covering every canonical phase name.

### M3. Fields named `*_local_z` contain world Z

`_point_in_initial_heading_frame()` rotates XY but returns:

```python
local_z = point_world_z
```

This is harmless while the fixed base always has world Z = 0, but the field is
not truly local and will be wrong for a raised, tilted, or differently authored
base.

Required fix: either rename the fields to `*_world_z` or subtract a recorded
base/reference Z.

### M4. Bucket volume is trusted as closed without topology validation

The bucket count path is otherwise sound:

```text
world AABB broad phase
-> bucket-local OBB broad phase
-> exact mesh inside test
```

However, `authored_bucket_volume_faces()` labels the mesh as a closed cavity
without verifying that every undirected edge has exactly two incident faces.
The ray test can also double-count shared triangle edges for boundary cases.

Required fix:

```text
validate watertight edge incidence once when loading the mesh
reject invalid topology instead of silently counting zero
use a deterministic ray jitter or a watertight point-in-mesh implementation
```

### M5. Non-finite exporter validation is incomplete

The final 28D state explicitly checks `math.isfinite()`. Generic
`vector_or_none()` accepts NaN/Inf, so old or imported `action` and
`observation.effort` values can still reach Parquet and normalization stats.

Required fix: validate finite state, action, effort, timestamps, and camera
metadata before accepting each export row.

## Semantic And Maintenance Risks

### S1. `robot_body_yaw_deg` is actually initial swing yaw

The fixed base yaw field is intentionally `0.0`. Scene randomization stores
`robot_body_yaw_deg`, but `auto_collect_initial_pose_for_attempt()` applies it
to the `swing` joint. The exported local target frame therefore uses the
initial upper-structure working heading, not a rotated base transform.

The math is consistent, but names and prompt wording are misleading:

```text
legacy name: robot_body_yaw_deg
actual meaning: initial_swing_working_heading_deg
prompt phrase: initial base pose
```

Rename the metadata in a future schema version or explicitly document the
legacy alias.

### S2. Phase index is categorical but encoded as one scalar

`phase_index` values 0-9 are ordered integers. A neural network can incorrectly
interpret distance between phases as continuous magnitude. This was an
intentional 1D budget choice, not an arithmetic error.

If dimensions permit, use an embedding or one-hot encoding. Otherwise retain
the scalar but mark it as categorical in the model configuration.

### S3. DOF fallback assumes articulation order

Named DOF mapping is correct. If names are missing and the articulation has
exactly four DOFs, initialization falls back to `[0,1,2,3]`. That fallback can
silently swap state and effort channels in a differently authored URDF.

For training collection, missing named DOFs should be a hard preflight failure.

## Calculations Confirmed Correct

The following calculations are internally correct under their documented
assumptions:

```text
joint order:
  [swing, boom, arm, bucket]

runtime action:
  shortest commanded-angle delta / train dt

runtime measured velocity:
  shortest measured-angle delta / train dt

swing handling:
  shortest angular delta in runtime
  unwrapped series before export finite differences

target XY transform:
  +X forward, +Y left in the initial swing working frame

truck relative heading:
  sin/cos(truck_yaw - initial_working_heading)

camera policy:
  three complete views required
  frame reuse disabled by default

export derivatives:
  recomputed consistently from q/q_cmd on the exported uniform timeline

current observed phase order:
  pre_dig -> approach_contact -> insert_cut -> pull_mid_cut
  -> curl_to_hold_material -> pull_exit_cut -> secure_load
  -> lift_carry -> unload_to_bin
```

## Recommended Repair Order

```text
P0  Per-episode source FPS validation and an explicit mixed-rate policy
P0  One immutable camera/state/effort/bucket/phase observation snapshot
P0  Disable 10x time scaling for the effort-conditioned 32D dataset
P1  Effort outlier audit, filtering, and robust normalization
P1  Fix literal loaded_transit phase mapping
P1  Smooth and normalize bucket_load_rate
P2  Validate bucket volume watertight topology
P2  Rename local Z and initial working-heading semantics in schema v4
P2  Reject all non-finite export features
```
