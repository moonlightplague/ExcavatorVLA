# 32D VLA Deployment Observation Contract

Updated: 2026-07-18

## Contract

The recommended policy input is:

```text
observation.state   28 float32 values
observation.effort   4 float32 values
```

The current stage is exported separately as
`observation.stage_current_id` (`int64`, 0..9). It is supervision and
telemetry, not part of the 32 continuous policy inputs.

The authoritative implementation is:

```text
excavator_common/vla_observation_contract.py
schema: excavator_state_v4_28d_plus_4effort_categorical_phase10
```

## State 28D

| Range | Values | Unit/source |
| --- | --- | --- |
| 0..3 | swing, boom, arm, bucket position | rad, measured articulation |
| 4..7 | joint velocity | rad/s, causal finite difference on dataset/simulation time |
| 8..11 | command tracking error | rad, `q_cmd - q`; shortest swing delta |
| 12..15 | previous issued action | rad/s, causal command velocity |
| 16..17 | bucket load-to-pour axis forward/up | normalized in the current upper frame |
| 18..20 | dig target minus bucket tip | m, current upper frame |
| 21..23 | unload landing minus bucket load point | m, current upper frame |
| 24..25 | truck heading relative to current upper heading | sin/cos |
| 26 | source-tracked bucket fill fraction | count / 6400, clipped to 1.5 |
| 27 | smoothed causal bucket fill-rate fraction | fraction/s |

The current upper heading is `base_yaw + measured_swing`. Position features
are relative to the current bucket and current upper frame, so the policy does
not need to learn arbitrary world origins.

`observation.effort` is the measured swing/boom/arm/bucket effort in canonical
DOF order. It remains separate because it has different units and
normalization statistics.

## Action Timing

At observation row `t`:

```text
state[12:16] = command velocity that produced the current observation
action          = command velocity for the next transition
```

Joint velocity, previous action, and bucket fill rate use backward-only
differences. The exporter never uses a centered difference for observation
features, so no future action leaks into policy input.

## Stage Label

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

The simulator may report this label from its phase estimator, but
`smolvla_policy_client_32d.py` does not feed it to the base policy.

## Bucket Semantics

Training and deployment both use the authored closed `bucket_cut` mesh and
initial-pile particle source tracking. Deployment must not substitute the old
hand-written box/profile proxy. Bucket tip/load/pour local points are:

```text
tip  = [0.75, 0.00, -0.18]
load = [0.35, 0.00,  0.08]
pour = [0.85, 0.00,  0.30]
```

## Export And Old Pool Upgrade

Re-exporting `.dashboard_success` automatically reconstructs v4 28D from the
raw trajectory fields. Existing per-episode videos are reusable; only parquet,
stats, info, and manifest need rebuilding when the schema changes.

The exported `meta/info.json` must contain:

```text
state_schema_version = excavator_state_v4_28d_plus_4effort_categorical_phase10
features.observation.state.names = the exact v4 28D names
features.observation.stage_current_id = int64 scalar
```

## Deployment Pipeline

The deployment repository supports four explicit protocols:

```text
legacy_18d_state
state_27d_plus_effort_4d
state_28d_plus_effort_4d              # legacy v3, phase inside state
state_28d_v4_plus_effort_4d           # recommended
```

The strict v4 client is:

```text
scripts/bridge_test/smolvla_policy_client_32d.py
```

It requires `--dataset-meta <lerobot_v3/meta/info.json>` and verifies the v4
schema and all 28 state names before sending any motion command. This prevents
an old shape-compatible 28D checkpoint from being silently interpreted with
new semantics.

The simulator bridge reconstructs live v4 features from measured joints,
current command, the selected truck/dig/unload context, three camera views,
and the authored bucket volume. Its reply includes:

```text
observation_state_28d
observation_effort
observation_stage_current_id
observation_32d_ready
observation_contract
task_text
```

## Verified

On 2026-07-18:

```text
old .dashboard_success: 45 episodes, 10445 rows, 0 skipped
all state/effort/action values finite
all stage IDs 0..9 represented
one-episode strict LeRobot export: ready and validated
Isaac profile/headless: 1/1 trainable, 207 samples
camera 0/1/2: 207/207/207
raw runtime state: 28D, measured q matches state[0:4]
```
