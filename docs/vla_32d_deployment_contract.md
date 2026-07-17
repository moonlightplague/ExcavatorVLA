# 32D VLA Deployment Observation Contract

Updated: 2026-07-17

## Scope

The current dataset exports:

```text
observation.state   28D
observation.effort   4D
```

They are two model features, not one raw 32-value articulation sensor. A
deployment adapter must reproduce both keys with the same names, order, units,
coordinate frames, and episode context used during export.

The shared schema and numeric builder live in:

```text
excavator_common/vla_observation_contract.py
```

The full-runtime bridge obtains live values through:

```text
excavator_runtime.deployment_vla_observation_payload()
scripts/bridge_test/sand_site_tcp_bridge_server.py
```

## Source Matrix

| Values | Deployment source | Availability |
| --- | --- | --- |
| base_x/base_y | robot base prim world translation | direct simulator state |
| base_yaw | fixed-base schema value, currently 0 | direct schema value |
| swing/boom/arm/bucket | articulation DOFs resolved by name | direct simulator state |
| joint velocity 4D | finite difference of canonical joint positions on simulation time, with shortest swing delta | derived online, matches dataset sampling |
| measured effort 4D | `get_measured_joint_efforts()`, resolved by DOF name | direct simulator state |
| bucket tip/load XYZ | training runtime `bucket_tip_pos()` / `bucket_load_pos()` | direct FK/world transform |
| bucket load | initial-pile source particles inside the authored `bucket_cut` closed volume, including the dataset's 32-particle/0.60 source fallback | simulator oracle, same dataset semantics |
| bucket load rate | finite difference of the live bucket count on simulation time | derived online |
| dig target | episode task/target provider or active plan | required task context |
| unload landing | selected unload mesh target or episode task provider | required task context |
| truck relative heading | selected truck prim transform | simulator world model |
| phase index | external supervisor, or the simulator's monotonic state/geometry FSM | derived semantic context, not a physical sensor |
| task prompt | same episode task provider used for target context | required task context |
| three RGB images | dataset camera prims 0/1/2 | direct simulator render |

## Phase Handling

`phase_index` is an expert stage label. A pure end-to-end VLA cannot measure it
directly from an articulation or camera API. Deployment therefore supports two
explicit modes:

1. `phase_mode=external`: a hierarchical controller owns the phase state and
   sends `phase_name` or `phase_index`.
2. `phase_mode=auto`: the simulator runs a monotonic 10-stage FSM using bucket
   tip/load geometry, dig/unload target distance, bucket load/rate, and joint
   motion. The reply reports `phase_source=simulator_auto_fsm` plus transition
   diagnostics.

Auto mode prevents a missing supervisor from becoming a constant zero phase.
It is an estimator, not hidden ground truth. A checkpoint that learned a very
strong dependence on exact expert stage boundaries should use external mode,
or be retrained with the same inferred phase labels.

## Episode Context

At the beginning of an episode, send one bridge command with:

```json
{
  "observation_context": {
    "phase_mode": "auto",
    "dig_target_xyz": [0.0, 6.5, 0.4],
    "unload_landing_xyz": [-6.8, -7.8, 4.2],
    "initial_origin_xy": [0.0, 0.0],
    "initial_heading_rad": 1.57,
    "truck_yaw_rad": -2.1,
    "task_text": "Excavate one scoop from the pile in front of the excavator and dump it into the truck bed."
  }
}
```

An editable template is available at:

```text
scripts/bridge_test/observation_context.example.json
```

For external mode, set `"phase_mode": "external"` and include `phase_name` or
`phase_index`; the supervisor may update those phase fields on later commands.
Initial origin and heading remain fixed for the episode, matching export
behavior.

If target, landing, or truck heading is omitted, the full runtime may use its
active target, selected unload mesh, and truck prim. Their resolved source is
reported in `observation_sources`.

## Bridge Reply

A valid reply contains:

```text
observation_state_28d    28 finite float values
observation_effort        4 finite float values
observation_32d_ready      true
observation_contract       exact names/shapes/action unit
observation_sources        provenance for every nontrivial group
task_text                  episode language prompt
```

The bridge also keeps `observation_state` as the legacy live 14D state. The
policy client may form a legacy 18D checkpoint input only by explicitly
concatenating that 14D vector with the measured 4D effort. A new checkpoint
must read `observation_state_28d` and `observation_effort` as separate feature
keys; it must not pad, truncate, or concatenate them based on a global guess.
Read the checkpoint feature contract first.

## Supported Simulator Paths

The full excavator runtime remains the reference source because it owns the
authored bucket volume, particle source tracking, selected unload mesh, and
canonical DOF mapping:

```text
run_vla_train_scene.py --bridge
```

The current viewport bridge is GUI-only. `run_vla_train_scene.py` rejects
`--headless --bridge`; a headless deployment needs a separately validated
offscreen camera backend before it can satisfy the same three-image contract.

The `vla-deploy-contract` worktree synced to `origin/smolvla` commit `c099da0`
also supports the contract through its deterministic lockstep simulator:

```text
run_simulation.py
scripts/bridge_test/smolvla_policy_client_32d.py
```

That server keeps the legacy 18D handshake as the default and adds a separate
`state_28d_plus_effort_4d` handshake. The new handshake is accepted only when
the authored `/World/URDF_real3/bucket_link/bucket_cut/node_/mesh_` is a valid
closed mesh, an initial-pile source-particle mask can be captured, and all
three cameras are available. The client must supply target, unload landing,
and task text. `phase_mode=auto` does not require a separate stage input.

Example client command:

```bash
python scripts/bridge_test/smolvla_policy_client_32d.py \
  --ckpt /path/to/checkpoint \
  --vlm /path/to/local/SmolVLM \
  --dataset-meta /path/to/lerobot_v3 \
  --observation-context scripts/bridge_test/observation_context.example.json
```

## Acceptance Check

Before enabling policy motion, verify:

```text
schema_version == excavator_state_v3_28d_plus_4effort_phase_index10
state names exactly match all 28 exported names
effort names exactly match [swing, boom, arm, bucket]
all 32 values are finite
bucket_load uses initial-pile particles inside bucket_cut, not a hand-written proxy
phase source is external_supervisor or simulator_auto_fsm, never a constant fallback
target/landing remain fixed in the initial heading frame for the episode
truck sin/cos matches the current truck transform
camera keys 0/1/2 are present and fresh
action is four canonical rad/s values
```
