# ExcavatorVLA

This repository contains the Isaac Sim excavator runtime, the full sand-site scene runtime, auto dataset collection, and SmolVLA bridge clients.

## Entry Points

Use these entry points according to the workflow:

- [`main.py`](main.py): Isaac Sim Script Editor entry. Use this when Isaac Sim GUI is already open and you want the full interactive runtime, UI controls, sand controls, and manual auto-collect buttons.
- [`run_vla_train_scene.py`](run_vla_train_scene.py): Isaac Sim Python launcher for VLA training. Use this when you do not want to run `main.py`. It creates `SimulationApp`, opens the full original scene, loads sand + truck + excavator, and can start either the SmolVLA TCP bridge or headless auto collect.
- [`run_excavator_standalone.py`](run_excavator_standalone.py): legacy bridge launcher. Do not use it as the primary VLA training scene when you need the full sand-site runtime; it has older scene setup behavior and can show the old unload-bin style scene.
- `main_zsp.py`: legacy/removed entry in the current checkout. Do not use it unless you intentionally restore that file.

There is currently one active `main.py`. Keep `main.py` for Isaac Sim GUI / Script Editor debugging and interactive auto collect. Use `run_vla_train_scene.py` for new training runs.

## Repository Layout

Top-level files are kept for active launch commands, configuration, dataset tooling, and this single project README.

- [`main.py`](main.py): Script Editor launcher for the full GUI runtime.
- [`run_vla_train_scene.py`](run_vla_train_scene.py): primary VLA training and headless auto-collect launcher.
- [`run_excavator_standalone.py`](run_excavator_standalone.py): legacy standalone bridge launcher.
- [`run.sh`](run.sh): shell wrapper for the legacy standalone launcher.
- [`excavator_dataset_tools.py`](excavator_dataset_tools.py): dataset inspection, plotting, dashboard, and LeRobot export CLI.
- [`run_dataset_dashboard.py`](run_dataset_dashboard.py): small dashboard launcher around `excavator_dataset_tools.py`.
- [`excavator_config.json`](excavator_config.json): project/runtime configuration.
- [`CHANGELOG.md`](CHANGELOG.md): chronological project change notes.

Project folders:

- [`archive/`](archive): old runtime variants retained only for reference.
  - [`archive/standalone/`](archive/standalone): archived standalone camera/API backup launcher.
- [`assets/`](assets): checked-in Isaac Sim and model assets.
  - `assets/usd/`: original USD scene.
  - `assets/fbx/`: truck FBX/USD assets.
  - `assets/urdf/`: excavator URDF package and mesh assets.
- [`docs/`](docs): non-runtime documentation.
  - [`docs/analysis/`](docs/analysis): generated/static analysis documents and dataset schema notes.
- [`excavator_common/`](excavator_common): shared Python helpers for paths, bridge protocol, and geometry.
- [`scripts/`](scripts): runtime modules and helper scripts.
  - [`scripts/excavator_app/`](scripts/excavator_app): main Isaac Sim excavator/sand runtime modules.
  - [`scripts/bridge_test/`](scripts/bridge_test): current bridge servers and external bridge clients.
  - `scripts/bridge_test/archive/`: archived SmolVLA client variants retained for reference.
- [`tests/`](tests): lightweight tests for shared helpers.

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

## Full VLA Training Scene Without `main.py`

Run the full sand + truck + excavator training scene from a terminal with Isaac Sim's Python:

```bash
cd /isaac-sim/ExcavatorVLA
/isaac-sim/python.sh run_vla_train_scene.py \
  --sand-amount 1.0 \
  --bridge
```

This starts a GUI Isaac Sim process, loads the full scene, disables the Excavator/Sand control windows by default, and opens the TCP bridge on `0.0.0.0:5555`.

Then connect SmolVLA from your policy Python environment:

```bash
cd /isaac-sim/ExcavatorVLA
python scripts/bridge_test/smolvla_policy_client.py \
  --host 127.0.0.1 \
  --port 5555 \
  --ckpt /path/to/smolvla/checkpoint \
  --vlm /path/to/local/SmolVLM2-500M-Video-Instruct
```

Useful variants:

```bash
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 0.75 --bridge
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.25 --bridge --bridge-port 5556
/isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.0 --bridge --with-ui
```

The bridge depends on a GUI viewport, so do not combine `--headless` with `--bridge`.

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

If you want to attach SmolVLA to that already-running full scene, run:

```python
import runpy

runpy.run_path("/isaac-sim/ExcavatorVLA/scripts/bridge_test/sand_site_tcp_bridge_server.py", run_name="__main__")
```

Then start the SmolVLA client from a normal Python environment:

```bash
python scripts/bridge_test/smolvla_policy_client.py \
  --host 127.0.0.1 \
  --port 5555 \
  --ckpt /path/to/smolvla/checkpoint \
  --vlm /path/to/local/SmolVLM2-500M-Video-Instruct
```

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

Older standalone backup variants are archived under [`archive/standalone/`](archive/standalone).

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
```

Additional optional asset overrides:

- `EXCAVATOR_TRUCK_USD`: dump-truck USD path for legacy standalone launchers. Default: [`assets/fbx/truck/truck.usd`](assets/fbx/truck/truck.usd).
- `EXCAVATOR_TRUCK_GLB`: optional dump-truck GLB source for conversion in legacy standalone launchers.

## Quick Decision Guide

- Need interactive Isaac Sim GUI and manual controls: use `main.py` from Script Editor.
- Need VLA training scene with bridge: use `run_vla_train_scene.py --bridge`.
- Need automated dataset generation: use `run_vla_train_scene.py --headless --auto-collect`.
- Need old pygame teleop bridge behavior: use `run_excavator_standalone.py`.
