# ExcavatorVLA

This repository contains the Isaac Sim excavator runtime, the full sand-site scene runtime, auto dataset collection, and SmolVLA bridge clients.

## Entry Points

Use these entry points according to the workflow:

- [`main.py`](main.py): Isaac Sim Script Editor entry. Use this when Isaac Sim GUI is already open and you want the full interactive runtime, UI controls, sand controls, and manual auto-collect buttons.
- [`run_isaacsim_ckpt18450_seed2_one_episode.sh`](run_isaacsim_ckpt18450_seed2_one_episode.sh): canonical checkpoint-18450, seed-2, one-episode Isaac Sim evaluation launcher. Runtime replanning and ensemble settings are defined inside the script.
- [`run_simulation.py`](run_simulation.py): protocol-v2 simulator bridge used by the canonical launcher.
- [`resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh`](resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh): canonical continuation-training launcher for the five fixed seed-2 episodes.
- [`run_vla_train_scene.py`](run_vla_train_scene.py): Isaac Sim Python launcher for the full training scene and headless auto collect. Its optional TCP bridge is the older ticks-based protocol and is not compatible with the current protocol-v2 policy client.
- [`run_excavator_standalone.py`](run_excavator_standalone.py): legacy bridge launcher. Do not use it as the primary VLA training scene when you need the full sand-site runtime; it has older scene setup behavior and can show the old unload-bin style scene.
- `main_zsp.py`: legacy/removed entry in the current checkout. Do not use it unless you intentionally restore that file.

There is currently one active `main.py`. Keep `main.py` for Isaac Sim GUI / Script Editor debugging and interactive auto collect, use `run_vla_train_scene.py` for new data-collection runs, and use the canonical shell launcher above for checkpoint evaluation.

## Repository Layout

Top-level files are kept for active launch commands, configuration, dataset tooling, and this single project README.

- [`main.py`](main.py): Script Editor launcher for the full GUI runtime.
- [`run_simulation.py`](run_simulation.py): current SmolVLA deployment bridge.
- [`run_isaacsim_ckpt18450_seed2_one_episode.sh`](run_isaacsim_ckpt18450_seed2_one_episode.sh): current Isaac Sim checkpoint evaluation launcher.
- [`resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh`](resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh): current five-episode continuation-training launcher.
- [`run_action_stage_chunk_sweep.sh`](run_action_stage_chunk_sweep.sh): offline action/stage chunk-size sweep launcher.
- [`run_isaacsim_offline_episode_replay_chunk1.sh`](run_isaacsim_offline_episode_replay_chunk1.sh): offline episode-action replay launcher for Isaac Sim.
- [`run_latest_three_checkpoints_seed2_fixed5ep_eval.sh`](run_latest_three_checkpoints_seed2_fixed5ep_eval.sh): fixed-five-episode comparison for the latest three checkpoints.
- [`run_vla_train_scene.py`](run_vla_train_scene.py): primary VLA training and headless auto-collect launcher.
- [`run_excavator_standalone.py`](run_excavator_standalone.py): legacy standalone bridge launcher.
- [`excavator_dataset_tools.py`](excavator_dataset_tools.py): dataset inspection, plotting, dashboard, and LeRobot export CLI.
- [`run_dataset_dashboard.py`](run_dataset_dashboard.py): small dashboard launcher around `excavator_dataset_tools.py`.
- [`excavator_config.json`](excavator_config.json): project/runtime configuration.
- [`CHANGELOG.md`](CHANGELOG.md): chronological project change notes.

Project folders:

- [`assets/`](assets): checked-in Isaac Sim and model assets.
  - `assets/usd/`: original USD scene.
  - `assets/fbx/`: truck FBX/USD assets.
  - `assets/urdf/`: excavator URDF package and mesh assets.
- [`docs/`](docs): non-runtime documentation.
  - [`docs/analysis/`](docs/analysis): generated/static analysis documents and dataset schema notes.
- [`excavator_common/`](excavator_common): shared Python helpers for paths, bridge protocol, geometry, and the strict deployment contract.
- [`scripts/`](scripts): runtime modules and helper scripts.
  - [`scripts/excavator_app/`](scripts/excavator_app): main Isaac Sim excavator/sand runtime modules.
  - [`scripts/bridge_test/`](scripts/bridge_test): current bridge servers and external bridge clients.
  - `scripts/dataset/`: dataset inspection utilities.
  - `scripts/evaluation/`: checkpoint, rollout, action, and stage analysis utilities.
  - `scripts/training/`: SmolVLA patch installers and training-log summaries.
- [`tests/`](tests): lightweight tests for shared helpers.

Repository/server cleanup is intentionally explicit and non-recursive. Run `bash scripts/maintenance/cleanup_obsolete_vla_files.sh` to preview the obsolete-file allowlist, then rerun it with `--apply` to delete those exact files. The script never deletes datasets, checkpoints, logs, or evaluation results.

## Legacy Optimized 18D SmolVLA Deployment

The July 16 bridge optimization targets the existing checkpoint contract:

```text
14D observation.state
+ 4D observation.effort
= 18D model input

4D action:
[swing, boom, arm, bucket] commanded velocity in rad/s
```

Do not use this path with a checkpoint trained on the newer `28D state + 4D effort = 32D` schema. The client validates the feature contract and normalization dimensions and stops at startup if they do not match.

### 1. Point to the Original Dataset Metadata

Set `SMOLVLA_DATASET_META` on the machine that runs the policy client:

```bash
export SMOLVLA_DATASET_META=/path/to/old_18d_dataset/meta/info.json
```

`/path/to/...` is a placeholder; replace it with the real location. This command does not export, copy, or upload a dataset. In a Unix shell, `export` creates an environment variable for the current shell and the programs launched from it.

The variable identifies the old checkpoint's source dataset metadata. The client reads `fps` from `meta/info.json`, and the simulator uses that value to decide how much 60 Hz physics time belongs to each policy action. It is separate from `SMOLVLA_CKPT`, which points to the model checkpoint.

Either the exact file or the dataset root is accepted:

```bash
export SMOLVLA_DATASET_META=/data/excavator_old_18d/meta/info.json
# Also accepted:
export SMOLVLA_DATASET_META=/data/excavator_old_18d
```

On Windows PowerShell, the equivalent is:

```powershell
$env:SMOLVLA_DATASET_META = "E:\datasets\excavator_old_18d\meta\info.json"
```

A typical layout is:

```text
/data/excavator_old_18d/
`-- meta/
    |-- info.json   # contains the training FPS
    `-- stats.json  # dataset statistics, when present
```

Verify the exact-file form before launching:

```bash
test -f "$SMOLVLA_DATASET_META"
python -c 'import json, os; p=os.environ["SMOLVLA_DATASET_META"]; print(json.load(open(p))["fps"])'
```

Do not guess the FPS. If the original metadata is unavailable but the correct training rate is independently known, the client has a diagnostic `--training-fps FPS` fallback; normal deployment should use the metadata.

### 2. Start the Authoritative Simulator Bridge

Run with Isaac Sim's Python:

```bash
cd /isaac-sim/ExcavatorVLA
/isaac-sim/python.sh run_simulation.py
```

The bridge maps joints by `robot.dof_names`, keeps three persistent camera viewports, advances physics at 60 Hz without rendering every substep, and derives each action duration from the training FPS supplied in the protocol-v2 handshake. Viewport capture currently requires a non-headless Isaac Sim session.

### 3. Start the Policy Client

From the SmolVLA policy environment:

```bash
cd /isaac-sim/ExcavatorVLA
python scripts/bridge_test/smolvla_policy_client.py \
  --host 127.0.0.1 \
  --port 5555 \
  --ckpt /path/to/old_18d_checkpoint \
  --vlm /path/to/local/SmolVLM2-500M-Video-Instruct \
  --dataset-meta "$SMOLVLA_DATASET_META"
```

### Current Validation Status

Static parsing, diff checks, and all 11 deployment-contract unit tests pass. Isaac Sim was not launched during this change, so runtime validation is still required before treating the bridge as production-ready:

- Confirm the runtime DOF names, action direction and units, effort ordering, and joint limits.
- Confirm three fresh, non-black, correctly posed camera images across repeated observations.
- Confirm the installed checkpoint exposes the exact 18D input, 4D output, and expected three camera keys, with normalization applied once.
- Confirm bucket load changes during scoop/carry/dump and compare the proxy result with training-time geometry.
- Measure policy and bridge p50/p95/max latency, real-time factor, GPU utilization, and shared-GPU contention.
- Test reconnect, shutdown, safe zero-action behavior, and non-divisor training FPS values.

The current implementation is optimized deterministic lockstep. Holding or interpolating the last safe action at 60 Hz while the next inference runs is a separate real-time mode that remains to be implemented and validated.

## Markdown Index

This is the only `README.md` in the project. Other Markdown files are indexed here:

- [`CHANGELOG.md`](CHANGELOG.md): project change log.
- [`docs/analysis/CODE_REDUNDANCY_ERROR_ANALYSIS.md`](docs/analysis/CODE_REDUNDANCY_ERROR_ANALYSIS.md): static redundancy and risk review.
- [`docs/analysis/SCRIPT_FUNCTION_MEMO_INDEX.md`](docs/analysis/SCRIPT_FUNCTION_MEMO_INDEX.md): generated function inventory and navigation memo.
- [`docs/analysis/excavator_dataset_schema_analysis.md`](docs/analysis/excavator_dataset_schema_analysis.md): dataset schema and export notes.

## Project Configuration

Set the Isaac Sim project path in [`excavator_config.json`](excavator_config.json):

```json
{
    "model_source": "original",
    "project_root": "/isaac-sim/ExcavatorVLA"
}
```

`model_source` is intentionally fixed to `"original"` for `main.py`; ZSP scene selection is no longer part of the active entrypoint.

You can also override the project root without editing the config:

```bash
export EXCAVATOR_PROJECT_ROOT=/isaac-sim/ExcavatorVLA
```

## Full Training Scene Without `main.py`

Run the full sand + truck + excavator scene from a terminal with Isaac Sim's Python:

```bash
cd /isaac-sim/ExcavatorVLA
/isaac-sim/python.sh run_vla_train_scene.py \
  --sand-amount 1.0 \
  --no-bridge
```

This starts a GUI Isaac Sim process, loads the full scene, and disables the Excavator/Sand control windows by default. Use this launcher for scene work and data collection. Use the earlier `run_simulation.py` instructions for optimized old-18D inference.

Useful variants:

```bash
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 0.75 --no-bridge
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.25 --no-bridge
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.0 --no-bridge --with-ui
```

`run_vla_train_scene.py --bridge` and [`scripts/bridge_test/sand_site_tcp_bridge_server.py`](scripts/bridge_test/sand_site_tcp_bridge_server.py) retain the older `ticks` protocol for legacy testing. Do not connect the protocol-v2 `smolvla_policy_client.py` to that bridge. The legacy bridge depends on a GUI viewport, so it also cannot be combined with `--headless`.

## Headless Auto Collect

Run headless auto dataset collection with the same launcher:

```bash
cd /isaac-sim/ExcavatorVLA
/isaac-sim/python.sh run_vla_train_scene.py \
  --headless \
  --auto-collect \
  --success-count 20 \
  --max-attempts 120 \
  --sand-amount 1.0 \
  --dataset-root excavator_auto_dataset
```

Important options:

- `--success-count`: target number of successful/trainable episodes.
- `--max-attempts`: maximum total episode attempts before stopping. Use `0` to let the runtime choose its default multiplier.
- `--sand-amount`: initial sand amount multiplier. In auto collect this fixes the sand amount unless `--random-sand-amount` is also passed.
- `--random-sand-amount`: allow per-attempt sand amount randomization even when `--sand-amount` is provided.
- `--dataset-root`: output root for auto collect runs.
- `--disable-export`: collect raw run data but skip automatic LeRobot v3 export.
- `--export-python`: Python executable used for LeRobot export.
- `--no-wait-export`: close Isaac Sim immediately after auto collect finishes instead of waiting for LeRobot export.

### Dataset Camera Clock Rules

Dataset image collection uses a single Isaac camera clock:

1. Physics: `world.step(render=True)`
2. Camera graph: `camera_global_tick()`
3. Vision read: `cam.get_rgb()`
4. Dataset writer

Rules:

- Camera capture must be driven by `camera_global_tick()`.
- Dataset collection must not depend on viewport capture.
- `world.step()` alone is not sufficient for dataset camera frames.
- Isaac camera RGB reads require an explicit Replicator orchestrator step.

## Isaac Sim Script Editor Workflow

If Isaac Sim is already open and you want the interactive GUI workflow, run this in Script Editor:

```python
import runpy

runpy.run_path("/isaac-sim/ExcavatorVLA/main.py", run_name="__main__")
```

`main.py` resolves the repo root, adds [`scripts`](scripts) to the Python import path, opens [`assets/usd/excavator_scene.usd`](assets/usd/excavator_scene.usd), and runs `excavator_app.bootstrap.run_excavator_with_sand()`.

For legacy ticks-based bridge testing against that already-running scene, run:

```python
import runpy

runpy.run_path("/isaac-sim/ExcavatorVLA/scripts/bridge_test/sand_site_tcp_bridge_server.py", run_name="__main__")
```

This Script Editor bridge is not compatible with the protocol-v2 production client. For current old-18D SmolVLA deployment, launch `run_simulation.py` instead.

## Legacy Standalone Bridge

[`run_excavator_standalone.py`](run_excavator_standalone.py) remains available for older bridge and GUI-client testing:

```bash
cd /isaac-sim/ExcavatorVLA
/isaac-sim/python.sh run_excavator_standalone.py
```

Useful legacy options:

```bash
/isaac-sim/python.sh run_excavator_standalone.py \
  --truck-usd assets/fbx/truck/truck.usd \
  --truck-glb /optional/path/to/truck.glb
```

Then run the pygame teleoperation client in a second terminal:

```bash
cd /isaac-sim/ExcavatorVLA/scripts/bridge_test/
python gui_client.py --host 127.0.0.1 --port 5555
```

Use this legacy path only when you intentionally want the older standalone bridge behavior. For full sand + truck + excavator training, prefer `run_vla_train_scene.py`.

## Environment Variables

The launchers set common runtime variables automatically, but you can set them manually when needed:

```bash
export EXCAVATOR_PROJECT_ROOT=/isaac-sim/ExcavatorVLA
export EXCAVATOR_SAND_AMOUNT=1.0
export EXCAVATOR_NO_UI=1
export EXCAVATOR_HEADLESS=1
export EXCAVATOR_DATASET_ROOT=/path/to/excavator_auto_dataset
export SMOLVLA_CKPT=/path/to/smolvla/checkpoint
export SMOLVLA_VLM=/path/to/local/SmolVLM2-500M-Video-Instruct
export SMOLVLA_DATASET_META=/path/to/old_18d_dataset/meta/info.json
```

Additional optional asset overrides:

- `EXCAVATOR_TRUCK_USD`: dump-truck USD path for legacy standalone launchers. Default: [`assets/fbx/truck/truck.usd`](assets/fbx/truck/truck.usd).
- `EXCAVATOR_TRUCK_GLB`: optional dump-truck GLB source for conversion in legacy standalone launchers.

## Quick Decision Guide

- Need interactive Isaac Sim GUI and manual controls: use `main.py` from Script Editor.
- Need to deploy the old 18D SmolVLA checkpoint: use `run_simulation.py` and the protocol-v2 `smolvla_policy_client.py`.
- Need the full scene without inference: use `run_vla_train_scene.py --no-bridge`.
- Need automated dataset generation: use `run_vla_train_scene.py --headless --auto-collect`.
- Need the old ticks-based full-scene bridge: use `run_vla_train_scene.py --bridge` with a compatible legacy client only.
- Need old pygame teleop bridge behavior: use `run_excavator_standalone.py`.
