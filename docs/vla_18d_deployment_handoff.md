# 18D SmolVLA Deployment Handoff

Updated: 2026-07-15

## 1. Purpose and Scope

This note is for the engineer maintaining the simulator-to-SmolVLA deployment
pipeline based on commit:

~~~text
4bf1134915373cb9ee372702069cb2c8b8230a40
~~~

The immediate objective is deliberately narrow:

> Make the existing 18D checkpoint execute correctly and efficiently.

Do not migrate this checkpoint to the newer 32-value observation contract as
part of the performance fix. The current training branch is documented later
only so that a future migration is explicit and does not silently break the old
model.

Static review covered:

~~~text
run_simulation.py
scripts/bridge_test/smolvla_policy_client.py
scripts/excavator_app/excavator_runtime.py
excavator_dataset_tools.py
~~~

Isaac Sim was not launched during this audit. Runtime DOF order, checkpoint FPS,
and timing percentiles therefore must be captured by the deployment engineer.

## 2. Existing 18D Model Contract

The old checkpoint input is the 14D state followed by four measured efforts.
All joint-related vectors use the canonical order:

~~~text
[swing, boom, arm, bucket]
~~~

### 2.1 State indices

| Index | Name | Unit / meaning |
| ---: | --- | --- |
| 0 | base_x | world metres |
| 1 | base_y | world metres |
| 2 | base_yaw | radians |
| 3 | swing | radians |
| 4 | boom | radians |
| 5 | arm | radians |
| 6 | bucket | radians |
| 7 | bucket_load_estimate | particles inside the closed bucket volume |
| 8 | bucket_tip_x | world metres |
| 9 | bucket_tip_y | world metres |
| 10 | bucket_tip_z | world metres |
| 11 | bucket_load_x | world metres |
| 12 | bucket_load_y | world metres |
| 13 | bucket_load_z | world metres |
| 14 | swing_measured_effort | simulator measured joint effort |
| 15 | boom_measured_effort | simulator measured joint effort |
| 16 | arm_measured_effort | simulator measured joint effort |
| 17 | bucket_measured_effort | simulator measured joint effort |

Rotational joint effort is normally torque-like in Isaac units, but deployment
must preserve the exact values and normalization statistics used during
training rather than introducing a new unit conversion.

The bucket geometry points used by the reviewed server match training:

~~~text
BUCKET_TIP_LOCAL  = [0.75, 0.0, -0.18]
BUCKET_LOAD_LOCAL = [0.35, 0.0,  0.08]
~~~

### 2.2 Visual and language input

The checkpoint receives three RGB observations:

~~~text
observation.images.0
observation.images.1
observation.images.2
~~~

Each model tensor is [1, 3, 256, 256], float32 in [0, 1]. It also receives the
episode task string and its tokenizer output.

### 2.3 Model output

The exported action is four commanded joint velocities:

~~~text
[swing_cmd_velocity, boom_cmd_velocity,
 arm_cmd_velocity, bucket_cmd_velocity]
~~~

The intended unit is radians per second. Deployment may integrate it into a
position target:

~~~text
q_target = q_current + action_velocity * control_dt
~~~

This is valid only when action normalization, canonical joint order, and
control_dt match the training contract.

## 3. Confirmed Causes of Abnormally Slow Execution

### 3.1 One policy request advances only 1/30 second

The reviewed bridge hard-codes:

~~~text
PHYSICS_DT          = 1/60 s
BRIDGE_STEP_SECONDS = 1/30 s
BRIDGE_STEP_TICKS   = 2
~~~

The client still sends ticks=4, but the server ignores it. The earlier bridge
advanced four 60 Hz ticks, or 1/15 s, per request. The reviewed lockstep version
therefore produces half as much simulated movement per expensive request.

The hard-coded 30 Hz policy rate is also not derived from the checkpoint's
training dataset. An old model trained at 2, 5, or 10 Hz should not silently be
deployed at 30 observation/action decisions per simulated second.

### 3.2 Camera work is serialized inside every request

For every control request, the bridge:

~~~text
advances two physics/render frames
switches to camera 0
waits two settle frames and up to five capture frames
switches to camera 1 and repeats
switches to camera 2 and repeats
downsamples each 1024 image to 256 with LANCZOS
zlib-compresses and base64-encodes all three images
serializes the result as JSON
~~~

This is approximately 11 render/update calls in the best case and up to 23 in
the worst case for only 0.0333 seconds of simulation progress. The request/reply
protocol does not overlap simulation, image preparation, transport, and model
inference; their wall times are added together.

### 3.3 Isaac continues RayTracing while the VLA is inferring

When no command is queued, the reviewed simulator repeatedly calls:

~~~python
world.step(render=True)
simulation_app.update()
~~~

The timeline is paused, so simulation time does not progress, but
RayTracedLighting continues consuming GPU resources. When Isaac and SmolVLA
share one GPU, this can starve inference while doing no useful control work.

The accepted --headless argument does not change this behavior. The reviewed
server always creates SimulationApp(headless=False) because it depends on a GUI
viewport capture path.

### 3.4 Resulting real-time factor

One loop advances 0.0333 seconds of simulation. Its wall time is approximately:

~~~text
physics/render
+ three-camera render/copy/resize
+ compression/JSON/TCP
+ client decode/tensor preparation
+ SmolVLA inference
~~~

If that sum is one wall-clock second, the real-time factor is only 0.033:
20 seconds of simulated motion takes about 10 minutes. This is a bridge
architecture problem, not evidence that the four-DOF arm intrinsically needs
that long to control.

## 4. Input/Output Correctness Risks

### 4.1 Raw articulation order is not a stable contract

The reviewed server labels the first four raw articulation values as
[swing, boom, arm, bucket], sends efforts from raw indices [0, 1, 2, 3], and
applies model actions in that same raw order.

The data generator does not make this assumption. It resolves robot.dof_names,
builds a name-to-index map, and converts between raw Isaac order and canonical
model order. An earlier deployment client also explicitly documented the raw
order as [bucket, arm, boom, swing].

Therefore, the reviewed direct mapping is unsafe even if it happens to work on
one imported articulation. The adapter must use names:

~~~python
canonical_names = ["swing", "boom", "arm", "bucket"]
canonical_to_raw = [find_dof_index(name) for name in canonical_names]

q_model = q_raw[canonical_to_raw]
effort_model = effort_raw[canonical_to_raw]

action_raw = zeros_like(q_raw)
action_raw[canonical_to_raw] = action_model
~~~

Print the resolved mapping once and terminate startup if any name is missing or
duplicated.

### 4.2 bucket_load_estimate is permanently zero

Training used the real number of particles inside the bucket's closed volume.
The reviewed deployment bridge always sends state index 7 as 0.0. The model
therefore sees an empty bucket during dig, curl, lift, carry, and dump.

This does not explain GPU latency, but it is a material observation distribution
shift and can produce repeated, late, or inappropriate actions. Update the value
at policy-observation frequency. It need not be scanned at every 60 Hz physics
tick.

### 4.3 Checkpoint schema is hard-coded instead of negotiated

smolvla_policy_client.py always builds an 18D tensor. That is correct for the old
checkpoint, but the bridge has no startup handshake proving:

~~~text
state dimension and names
effort dimension and names
action dimension, names, and units
camera keys and shapes
training FPS
normalization statistics identity/version
~~~

Read these from the checkpoint and its source dataset metadata. Fail before
motion if the runtime adapter does not exactly match.

### 4.4 Frequency and action semantics must agree

Do not use the server's hard-coded 30 Hz as the checkpoint frequency. Read the
actual meta/info.json associated with the old training dataset. The client should
request observations and consume policy actions at that frequency, while a
low-level 60 Hz controller holds or interpolates the most recent valid
velocity/waypoint.

Do not add an arbitrary speed multiplier to compensate for a slow wall-clock
loop. It changes the physical policy and can hide a timing bug.

### 4.5 Normalization must have one owner

The reviewed client manually normalizes observation.state and manually
unnormalizes action using checkpoint safetensors. Keep this only if the loaded
policy does not already run the corresponding pre/post processors. At startup,
print the loaded tensor shapes and abort if they are not exactly state 18 and
action 4 for this checkpoint.

## 5. Recommended Immediate Architecture for the 18D Model

The fastest low-risk path does not require retraining.

1. Keep the old 18D model input and its original normalization statistics.
2. Add a named canonical/raw DOF adapter for state, effort, and action.
3. Read policy FPS from the checkpoint's source dataset metadata.
4. Run physics/low-level control at 60 Hz using the last valid VLA command.
5. Request a new VLA decision only at the model's training frequency.
6. Maintain three persistent camera render products; do not switch one viewport
   among three cameras or wait settle frames after every switch.
7. Render/copy cameras only when a policy observation is due.
8. Stop the paused-timeline render=True busy loop during inference. A network
   thread or asynchronous queue should receive the result without forcing RTX
   frames merely to keep TCP alive.
9. Preserve 1024-to-256 downsampling if its image quality is required, but move
   it out of camera-switch settling and measure it separately.
10. Add real bucket load at policy frequency and preserve the training prompt
    convention.

Two valid timing modes are possible.

### Real-time deployment

~~~text
physics/control: 60 Hz continuously
VLA observation/inference: training FPS
between VLA results: hold/interpolate the last safe command
~~~

### Deterministic lockstep evaluation

~~~text
one policy result advances exactly 1 / training_fps simulation seconds
physics uses the required number of 60 Hz substeps
capture occurs once after the advance
no rendering busy loop while waiting for inference
~~~

Do not mix these modes implicitly.

## 6. Required Startup Evidence

The deployment log should print one contract block before enabling motion:

~~~text
checkpoint state_dim=18 action_dim=4
training_fps=<value read from dataset/checkpoint>
raw_dof_names=[...]
canonical_to_raw=[...]
state_names=[18 expected names]
action_names=[swing_cmd_velocity, boom_cmd_velocity,
              arm_cmd_velocity, bucket_cmd_velocity]
state_norm_shape=[18] action_norm_shape=[4]
camera_keys=[observation.images.0, observation.images.1,
             observation.images.2]
camera_tensor_shapes=[[1,3,256,256], ...]
~~~

For the first safe diagnostic episode, also log at a low rate:

~~~text
q_raw and q_canonical
action_model and action_raw
bucket_load_estimate
state min/max/non-finite count
action min/max/non-finite count
~~~

## 7. Required Timing Telemetry

Measure p50, p95, and maximum wall time for:

~~~text
physics advance
three-camera render/copy
resize
compress/base64/JSON
TCP send/receive
client decode and tensor preparation
SmolVLA select_action
total policy loop
~~~

Also report:

~~~text
simulated_seconds / wall_seconds
policy decisions / simulated_second
policy decisions / wall_second
GPU utilization and memory for Isaac and VLA processes
~~~

Without this split, a slow total loop cannot be attributed reliably to the
model, camera, simulator, or transport.

## 8. Acceptance Criteria for the Old Checkpoint

The 18D deployment fix is complete only when:

- state is exactly 18 finite values in the documented order;
- effort and action are mapped by DOF name, not raw array position;
- bucket_load_estimate changes when material enters or leaves the bucket;
- action is exactly four finite rad/s values after the checkpoint's one and only
  unnormalization step;
- policy frequency equals the old training dataset FPS;
- no client ticks field is silently ignored;
- no camera-switch settle loop runs per policy request;
- waiting for inference does not continuously RayTrace paused frames;
- all three camera tensors are fresh, non-black, and correspond to one
  observation timestamp;
- timing telemetry identifies the remaining wall-clock bottleneck;
- simulated movement is evaluated using real-time factor, not terminal print
  frequency alone.

## 9. Later Migration to the Current Observation Contract

The current exporter uses:

~~~text
28D observation.state
+ separate 4D observation.effort
= 32 effective robot observation values
~~~

The added 14 state values are:

~~~text
swing_velocity
boom_velocity
arm_velocity
bucket_velocity
phase_index                         # canonical integer 0..9
dig_target_local_x/y/z
unload_landing_local_x/y/z
truck_heading_relative_sin/cos
bucket_load_rate
~~~

This schema is excavator_state_v3_28d_plus_4effort_phase_index10. It also uses
the relative-heading task prompt, per-episode media, a four-frame video keyframe
interval, and a uniform export time policy.

An 18D checkpoint cannot consume this schema by concatenating or truncating
values. A later migration must either train a new checkpoint or deliberately
expand and fine-tune the input projection with new normalization statistics.
The runtime adapter must choose 18D or 28D+effort from checkpoint metadata,
never from a global hard-coded default.

## 10. Short Handoff Summary

For the current issue, keep the old 18D model. First fix canonical DOF mapping,
restore checkpoint-derived control frequency, eliminate serial camera switching
and paused RTX busy rendering, and provide real bucket load. These changes
address both meanings of "slow": poor wall-clock throughput and physically
small or incorrect excavator motion. Migrate to the current 32-value observation
contract only after the old deployment path is measured and stable.

