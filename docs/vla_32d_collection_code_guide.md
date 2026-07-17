# VLA 32D Observation Collection Code Guide

Audited: 2026-07-17

## 1. Important Data Contract

The simulator does not directly collect one 32D vector. The exported model
observation consists of two features:

```text
observation.state   28D
observation.effort   4D
-----------------------
effective input     32D
```

The source trajectory stores a raw 14D state, a measured 4D effort vector, and
the additional values needed to derive the exported 28D state.

```text
Isaac runtime:
  raw observation.state 14D
  observation.effort     4D
  obs.dq                  4D
  phase
  dig target
  episode scene/truck metadata

LeRobot exporter:
  raw 14D + derived 14D -> observation.state 28D
  measured effort remains a separate 4D feature
```

## 2. Raw 14D State

Source:

```text
scripts/excavator_app/excavator_runtime.py
DATASET_STATE_NAMES
dataset_observation_state()
```

The essential collection code is:

```python
def dataset_observation_state(q_real=None, bucket_load_metrics=None):
    if q_real is None:
        q_real = get_real_joint_positions()
    q_real = np.array(q_real, dtype=np.float32)
    base = get_prim_translation(ROBOT_BASE)
    tip = bucket_tip_pos()
    load = bucket_load_pos()
    bucket_load = dataset_bucket_load_estimate(bucket_load_metrics)

    return [
        base[0],
        base[1],
        get_base_yaw_rad(),
        q_real[swing_index],
        q_real[boom_index],
        q_real[arm_index],
        q_real[bucket_index],
        bucket_load,
        tip[0], tip[1], tip[2],
        load[0], load[1], load[2],
    ]
```

The field order and units are:

| Index | Field | Meaning | Unit |
| ---: | --- | --- | --- |
| 0 | `base_x` | Excavator base world X | m |
| 1 | `base_y` | Excavator base world Y | m |
| 2 | `base_yaw` | Fixed-base yaw placeholder; currently always `0.0` | rad |
| 3 | `swing` | Measured swing joint position | rad |
| 4 | `boom` | Measured boom joint position | rad |
| 5 | `arm` | Measured arm joint position | rad |
| 6 | `bucket` | Measured bucket joint position | rad |
| 7 | `bucket_load_estimate` | Particles inside the closed bucket volume | particle count |
| 8-10 | `bucket_tip_x/y/z` | Bucket tip in world coordinates | m |
| 11-13 | `bucket_load_x/y/z` | Bucket load reference point in world coordinates | m |

`bucket_load_estimate` is not kilograms. It is the tracked sand-particle count
inside the mesh-backed closed bucket volume.

## 3. Measured 4D Effort

Source:

```text
scripts/excavator_app/excavator_runtime.py
DATASET_EFFORT_NAMES
dataset_joint_effort_observation()
```

Essential code:

```python
def dataset_joint_effort_observation():
    values = ROBOT.get_measured_joint_efforts(
        joint_indices=JOINT_INDICES
    )
    effort = np.asarray(values, dtype=np.float32)[:4]
    return {
        "available": True,
        "observation.effort": effort.tolist(),
    }
```

Order:

```text
swing_measured_effort
boom_measured_effort
arm_measured_effort
bucket_measured_effort
```

These are Isaac articulation generalized efforts. For the four revolute joints
they are nominally torque-like values in N*m. They are not a direct bucket
payload-weight sensor and can be dominated by gravity, linkage geometry,
controller reaction, contact, and constraint forces.

`obs.joint_force_torque` is an optional diagnostic reaction-wrench field. It is
not part of the 32D model input.

## 4. Joint Motion And Action

Source:

```text
scripts/excavator_app/excavator_runtime.py
dataset_motion_derivatives()
```

Essential code:

```python
dt = sample_time - previous_sample_time

action = shortest_joint_delta(q_cmd, previous_q_cmd) / dt
dq_real = shortest_joint_delta(q_real, previous_q_real) / dt
ddq_real = (dq_real - previous_dq_real) / dt
action_ddq = (action - previous_action) / dt
```

Semantics:

```text
obs.q       measured joint position, rad
obs.dq      measured joint velocity, rad/s
obs.ddq     measured joint acceleration, rad/s^2
obs.q_cmd   commanded joint position, rad
action      commanded joint-position velocity, rad/s
action.ddq  commanded acceleration, rad/s^2
```

The 4D `action` order is always:

```text
[swing, boom, arm, bucket]
```

The first sample has zero finite-difference velocity because no previous sample
exists in the raw runtime trajectory. During LeRobot export, derivatives are
recomputed on the uniform timeline, so the first exported velocity uses a
forward difference instead.

## 5. Raw Trajectory Row

Source:

```text
scripts/excavator_app/excavator_runtime.py
dataset_record_sample()
dataset_record_sample_async()
```

The key row construction is:

```python
sample = {
    "timestamp": train_abs,
    "t": train_t,
    "timestamp.wall": wall_now,
    "timestamp.simulation": sample_sim_time,
    "task": dataset_task_text,
    "phase": phase,
    "phase.index": phase_features["index"],
    "observation.state": obs_state,          # raw 14D
    "observation.effort": measured_effort,   # measured 4D
    "obs.q": q_real,
    "obs.dq": joint_velocity,
    "obs.ddq": joint_acceleration,
    "obs.q_cmd": q_cmd,
    "action": command_velocity,
    "target": dig_target_world_xyz,
    "bucket.tip": bucket_tip_world_xyz,
    "bucket.load": bucket_load_world_xyz,
    "sand": compact_bucket_metrics,
    "observation.images.0": camera_0_path,
    "observation.images.1": camera_1_path,
    "observation.images.2": camera_2_path,
}
```

Simulation/train time is the default timestamp source. Wall time is retained
for profiling and diagnostics.

## 6. Camera Collection

Source:

```text
scripts/excavator_app/excavator_dataset_camera.py
background_capture_loop()
capture_observations()
```

`capture_observations()` reads the latest validated three-camera background
capture, writes image files, and returns relative paths:

```python
payload["observation.images.0"] = "images/0/000123.ppm"
payload["observation.images.1"] = "images/1/000123.ppm"
payload["observation.images.2"] = "images/2/000123.ppm"
```

The exporter requires all three camera files for a row. A row with a missing
camera file is excluded from the exported dataset.

## 7. Canonical Phase

The exported phase vocabulary has ten classes:

```text
0 pre_dig
1 approach_contact
2 insert_cut
3 pull_mid_cut
4 curl_to_hold_material
5 pull_exit_cut
6 secure_load
7 lift_carry
8 loaded_transit
9 unload_to_bin
```

Source:

```text
excavator_dataset_tools.py
LEROBOT_CANONICAL_PHASE_NAMES
canonical_lerobot_phase_index()
```

Clearance/high-carry/staged-unload route labels map to `loaded_transit`.
Dump/unload labels map to `unload_to_bin`.

## 8. Exported 28D State

Source:

```text
excavator_dataset_tools.py
LEROBOT_STATE_NAMES_28D
build_lerobot_state_28d()
```

Essential construction:

```python
state = (
    raw_base_state_14d
    + joint_velocity_4d
    + [phase_index]
    + dig_target_in_initial_heading_frame_3d
    + unload_landing_in_initial_heading_frame_3d
    + [sin(truck_relative_yaw), cos(truck_relative_yaw)]
    + [bucket_load_rate]
)
```

The additional 14 fields are:

| Index | Field | Unit |
| ---: | --- | --- |
| 14-17 | `swing/boom/arm/bucket_velocity` | rad/s |
| 18 | `phase_index` | integer encoded as scalar |
| 19-21 | `dig_target_local_x/y/z` | m |
| 22-24 | `unload_landing_local_x/y/z` | m |
| 25 | `truck_heading_relative_sin` | unitless |
| 26 | `truck_heading_relative_cos` | unitless |
| 27 | `bucket_load_rate` | particles/s |

The local target frame currently uses the initial upper-structure working
heading. The legacy metadata name is `robot_body_yaw_deg`, but the runtime
applies this value to the initial `swing` joint rather than rotating the fixed
base:

```python
dx = point_x - base_x
dy = point_y - base_y

local_x = cos(heading) * dx + sin(heading) * dy
local_y = -sin(heading) * dx + cos(heading) * dy
local_z = point_z
```

`local_x` is forward and `local_y` is left in that heading frame.

Consequently, prompt text that says "initial base pose" uses the base position
as its origin but the initial swing/upper-structure heading as its direction.

## 9. Export Time Policy

Source:

```text
excavator_dataset_tools.py
apply_export_time_policy_to_trajectory()
```

The exporter keeps every accepted row, assigns a uniform timeline, unwraps
joint angles, and recomputes derivatives:

```python
new_time[i] = i / effective_fps
dq = finite_difference(q_real, new_time)
ddq = finite_difference(dq, new_time)
action = finite_difference(q_cmd, new_time)
```

Therefore speed scaling changes timestamps, `obs.dq`, `obs.ddq`, `action`,
`action.ddq`, and `bucket_load_rate` consistently. It does not merely multiply
video playback speed.

## 10. LeRobot Row And Export Entry

Source:

```text
excavator_dataset_tools.py
collect_lerobot_rows()
export_lerobot_dataset()
```

Final training row:

```python
row = {
    "index": global_frame,
    "episode_index": export_episode_index,
    "frame_index": episode_frame_index,
    "timestamp": uniform_episode_time,
    "task_index": task_index,
    "observation.state": state_28d,
    "observation.effort": effort_4d,
    "action": action_4d,
}
```

Camera videos are stored as separate media features. Task text is stored in
`meta/tasks.parquet` and referenced through `task_index`.

The dashboard does not maintain a different 32D implementation.
`dashboard_export_success_pool()` calls the shared
`excavator_dataset_tools.export_lerobot_dataset()` exporter.
