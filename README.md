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

## Reproducible Fixed-Five-Episode SmolVLA Workflow

This is the authoritative workflow for continuing SmolVLA training on the five
fixed seed-2 episodes, evaluating the final checkpoints on those same episodes,
selecting an execution chunk size, and deploying checkpoint 18450 in Isaac Sim.
Run the commands on the Linux GPU server unless a command is explicitly marked
as PowerShell.

### 1. Fixed Paths And Runtime Assumptions

The checked-in launchers use these server paths:

```bash
PROJECT=/root/isaacsim/ExcavatorVLA
POLICY_PYTHON=/opt/conda/envs/smolvla/bin/python
TRAIN=/opt/conda/envs/smolvla/bin/lerobot-train
ISAAC_PYTHON=/root/isaacsim/python.sh

DATASET=/root/gpufree-data/excavator_route_compare/seed_2_fixed_scene_pose_exact/run_20260720_194053/lerobot_v3_eval27_stage10_h30_obslabels
VLM=/root/gpufree-data/checkpoints/SmolVLM2-500M-Video-Instruct
PRIOR=/root/gpufree-data/excavator_stage_action_analysis/stage_action_prior_training.json
TRAIN_ROOT=/root/gpufree-data/outputs/train
LOG_ROOT=/root/gpufree-data/excavator_logs
```

The dataset contract is deliberately strict:

| Item | Required value |
| --- | --- |
| Episodes | 5 |
| Frames | 810 |
| `observation.state` | 27 values |
| `observation.effort` | 4 values |
| Camera inputs | Three `3 x 256 x 256` images |
| Action | `[swing, boom, arm, bucket]`, 4 values |
| Current stage key | `observation.stage_current_id` |
| Scene and sampling seed | 2 |

The canonical stage IDs are:

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

Do not substitute the old stage order in which stages 4/5 or 8/9 were renamed
or exchanged. Both the training and evaluation launchers stop before model work
if the metadata differs from this contract.

### 2. Synchronize And Record The Code Revision

Refuse to update over locally modified tracked files:

```bash
set -euo pipefail

cd /root/isaacsim/ExcavatorVLA

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[ERROR] Tracked local changes exist."
    git status --short
    exit 1
fi

git fetch origin smolvla
git switch smolvla
git pull --ff-only origin smolvla

git rev-parse HEAD | tee /root/gpufree-data/excavator_logs/reproduction_git_revision.txt
bash scripts/maintenance/cleanup_obsolete_vla_files.sh --dry-run
bash scripts/maintenance/cleanup_obsolete_vla_files.sh --apply
```

Always keep `reproduction_git_revision.txt` with the training and evaluation
artifacts. It is the code identity for the run.

### 3. Validate Or Install The SmolVLA Training Patches

The continuation checkpoint depends on custom stage-sequence, stage-action,
direction, low-motion, padding, and local metric-logging changes in the installed
LeRobot package. The training launcher validates the final low-motion markers
and runs a two-step CUDA smoke test before starting the real job.

Check an already prepared environment first:

```bash
SMOLVLA_DIR=/opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/policies/smolvla

grep -n 'stage_action_low_motion_weight' "$SMOLVLA_DIR/configuration_smolvla.py"
grep -n 'stage_action_low_motion_raw_loss' "$SMOLVLA_DIR/modeling_smolvla.py"
grep -n 'stage_action_prior_direction' "$SMOLVLA_DIR/modeling_smolvla.py"
```

If this is a fresh compatible LeRobot environment and those patches have not
been applied, install the patch chain exactly once and in this order:

```bash
cd /root/isaacsim/ExcavatorVLA
PYTHON=/opt/conda/envs/smolvla/bin/python

"$PYTHON" scripts/training/patch_smolvla_stage_training.py
"$PYTHON" scripts/training/patch_smolvla_stage_seq50.py
"$PYTHON" scripts/training/patch_smolvla_dynamic_stage_seq50.py
"$PYTHON" scripts/training/patch_stage_batch_and_logging.py
"$PYTHON" scripts/training/patch_smolvla_stage_action_direction_loss.py
"$PYTHON" scripts/training/patch_smolvla_stage_action_low_motion_loss.py

"$PYTHON" -m py_compile \
  /opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/policies/smolvla/configuration_smolvla.py \
  /opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/policies/smolvla/modeling_smolvla.py \
  /opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/datasets/factory.py \
  /opt/conda/envs/smolvla/lib/python3.10/site-packages/lerobot/scripts/lerobot_train.py
```

The patchers match exact source text and intentionally fail on an incompatible
or unexpectedly modified LeRobot installation. Do not keep applying them after
the markers already exist. Record the effective Python environment as well:

```bash
/opt/conda/envs/smolvla/bin/python -m pip freeze \
  > /root/gpufree-data/excavator_logs/smolvla_environment.txt
```

### 4. Continue Training For 300 Dataset Epochs

The canonical launcher is
`resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh`. By default it
finds the highest complete checkpoint under:

```text
/root/gpufree-data/outputs/train/excavator_smolvla_stage05_prior020_lowmotion05_resume3840_autobatch_b144_add30ep
```

A complete source checkpoint must include model weights, train/policy configs,
optimizer state, optimizer parameter groups, scheduler state, and
`training_step.json`. The source run is never modified: the launcher copies its
latest complete checkpoint into a new isolated output and performs a true
optimizer/scheduler resume.

The continuation settings are:

| Parameter | Value |
| --- | ---: |
| Additional epochs | 300 |
| Batch size | 144 |
| Workers | 8 |
| Save interval | 50 epochs |
| Optimizer learning rate | `5e-5` |
| Stage loss weight | `0.5` |
| Stage exception weight | `10.0` |
| Stage-action loss weight | `0.20` |
| Low-motion weight / margin | `0.5` / `0.10` |

Confirm the GPU is idle, then launch:

```bash
cd /root/isaacsim/ExcavatorVLA
nvidia-smi

bash resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh
```

To reproduce from a different source run without editing the script:

```bash
cd /root/isaacsim/ExcavatorVLA

SOURCE_OUTPUT=/root/gpufree-data/outputs/train/EXACT_SOURCE_RUN \
  bash resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh
```

For the known source step 16650, 810 frames and batch size 144 produce six
optimizer steps per dataset epoch. Three hundred additional epochs therefore
add 1800 steps and target checkpoint step 18450. The output name is:

```text
/root/gpufree-data/outputs/train/excavator_smolvla_seed2_fixed5ep_canonical300_from16650_b144
```

The launcher starts training with `nohup`, writes a PID file, and maintains a
latest-log symlink. Monitor it with:

```bash
LATEST_LOG="$(ls -t \
  /root/gpufree-data/excavator_logs/excavator_smolvla_seed2_fixed5ep_canonical300_from*_b144_latest.log \
  | head -1)"

tail -f "$LATEST_LOG"
```

Inspect the custom losses separately:

```bash
tail -f "$LATEST_LOG" | grep --line-buffered -E \
  'policy_metrics|stage_action_direction_raw_loss|stage_action_low_motion_raw_loss|weighted_stage_action_loss|action_loss|stage_accuracy|loss'
```

Do not start evaluation or Isaac Sim while this training process is using the
GPU. The evaluation launcher also rejects incomplete training runs.

### 5. Evaluate The Last Three Checkpoints On The Same Five Episodes

Run the offline evaluator only after training reaches its configured target:

```bash
cd /root/isaacsim/ExcavatorVLA

TRAIN_RUN=/root/gpufree-data/outputs/train/excavator_smolvla_seed2_fixed5ep_canonical300_from16650_b144 \
  bash run_latest_three_checkpoints_seed2_fixed5ep_eval.sh
```

The launcher:

1. revalidates the five-episode/810-frame/27D/canonical-stage contract;
2. selects the three highest complete checkpoint steps from the completed run;
3. evaluates every checkpoint with seed 2 on the same five episodes;
4. saves per-frame arrays, per-episode metrics, plots, and median episode output;
5. builds `checkpoint_comparison.csv` and `checkpoint_comparison.json`; and
6. packages the result and updates a stable `latest.tar.gz` symlink.

The main comparison fields include current-stage accuracy, action tolerance and
sign accuracy, stage-transition F1, stage-action-prior accuracy, low-motion
satisfaction, and per-joint MAE. `micro_representative_score` is the mean of:

```text
stage_current_accuracy
action_current_within_tolerance_accuracy
action_current_sign_accuracy
stage_action_prior_current_accuracy
low_motion_current_satisfaction
```

For the known run, the stable archive path is:

```text
/root/gpufree-data/excavator_offline_eval/excavator_smolvla_seed2_fixed5ep_canonical300_from16650_b144_latest3_5episodes_latest.tar.gz
```

### 6. Sweep Offline Action/Stage Execution Chunks

After the latest-three evaluation completes, run:

```bash
cd /root/isaacsim/ExcavatorVLA
bash run_action_stage_chunk_sweep.sh
```

This first selects the checkpoint with the highest
`micro_representative_score`, then evaluates stitched action and stage outputs
for execution chunk sizes:

```text
1 2 3 5 10 15 20 30 50
```

Chunk 1 must exactly reproduce the original evaluator metrics; the sweep fails
if it does not. Review these outputs rather than selecting from a single metric:

```text
chunk_sweep_summary.json
chunk_sweep_micro.csv
chunk_sweep_macro.csv
chunk_sweep_episode_metrics.csv
chunk_stitch_accuracy.png
```

For a best checkpoint at step 18450, the stable archive is:

```text
/root/gpufree-data/excavator_offline_eval/action_stage_chunk_sweep_ckpt18450_latest.tar.gz
```

Important: this offline execution-chunk sweep is not the same setting as the
Isaac Sim policy client's replanning interval or temporal ensemble width.

### 7. Optional Chunk-1 Replay Sanity Check In Isaac Sim

Before running the live policy, replay the saved offline output into the same
seed-2 scene:

```bash
cd /root/isaacsim/ExcavatorVLA

EPISODE_ID=0 MAX_STEPS=0 \
  bash run_isaacsim_offline_episode_replay_chunk1.sh
```

If `EPISODE_NPZ` is omitted, the launcher resolves the best checkpoint from the
latest evaluation and uses that checkpoint's selected episode array. Override
it explicitly when exact artifact identity matters:

```bash
EPISODE_ID=0 \
EPISODE_NPZ=/absolute/path/to/episode_000000.npz \
MAX_STEPS=0 \
  bash run_isaacsim_offline_episode_replay_chunk1.sh
```

Replay results are packaged under
`/root/gpufree-data/excavator_isaacsim_replay`, with the stable archive:

```text
/root/gpufree-data/excavator_isaacsim_replay/isaacsim_offline_episode_chunk1_latest.tar.gz
```

### 8. Deploy Checkpoint 18450 In Isaac Sim

The current deployment launcher is deliberately pinned to checkpoint step
18450 and seed 2:

```bash
cd /root/isaacsim/ExcavatorVLA
bash run_isaacsim_ckpt18450_seed2_one_episode.sh
```

It requires an idle GPU, no existing training/simulator/policy process, and a
viewport-capable X11 `DISPLAY`. It validates the checkpoint, 27D dataset
metadata, exact dataset task text, deployment contracts, and complete training
state before starting Isaac Sim.

The current deployment settings are:

| Setting | Value |
| --- | ---: |
| Policy steps | 500 |
| Model output chunk size | 50 |
| Replan interval | 3 |
| Temporal ensemble width | 1 |
| Temporal ensemble decay | `0.35` |
| Excavation sequence supervisor | Enabled |
| Legacy load-retention constraint | Disabled |
| Legacy unload-geometry gate | Disabled |
| Normalized state/effort OOD limits | 8 sigma |

With temporal ensemble width 1, no averaging across overlapping historical
chunks occurs. The client infers a new model chunk every three policy steps and
executes indices 0, 1, and 2 before replanning. The sequence supervisor may
modify the model action for forced descent, curl/load security, lift, closed-loop
turning, positioning over the truck, and dumping. Therefore this run is not a
pure-model rollout; every modified step is recorded in the trace metadata.

The launcher writes simulator and client logs, the action/stage chunk log, the
executed policy trace, input dumps, video, analysis plots, and
`rollout_validation.json`. Validation checks all 500 step IDs, chunk origins,
chunk indices, ensemble weights/actions, checkpoint identity, and monotonic
supervisor phase progression.

The stable archive is:

```text
/root/gpufree-data/excavator_isaacsim_eval/isaacsim_seed2_ckpt18450_replan3_ensemble1_supervisor1_latest.tar.gz
```

### 9. Download The Reproducibility Artifacts

Run these commands in Windows PowerShell:

```powershell
$Remote = "root@120.209.70.195"
$Port = 30105
$Destination = "E:\Downloads"

scp -P $Port "${Remote}:/root/gpufree-data/excavator_offline_eval/excavator_smolvla_seed2_fixed5ep_canonical300_from16650_b144_latest3_5episodes_latest.tar.gz" $Destination
scp -P $Port "${Remote}:/root/gpufree-data/excavator_offline_eval/action_stage_chunk_sweep_ckpt18450_latest.tar.gz" $Destination
scp -P $Port "${Remote}:/root/gpufree-data/excavator_isaacsim_replay/isaacsim_offline_episode_chunk1_latest.tar.gz" $Destination
scp -P $Port "${Remote}:/root/gpufree-data/excavator_isaacsim_eval/isaacsim_seed2_ckpt18450_replan3_ensemble1_supervisor1_latest.tar.gz" $Destination
scp -P $Port "${Remote}:/root/gpufree-data/excavator_logs/reproduction_git_revision.txt" $Destination
scp -P $Port "${Remote}:/root/gpufree-data/excavator_logs/smolvla_environment.txt" $Destination
```

Keep the Git revision, environment manifest, dataset metadata, checkpoint
selection manifest, comparison CSV/JSON, chunk sweep, replay archive, and live
simulation archive together. Those files are the minimum evidence needed to
reconstruct which code, data contract, checkpoint, execution strategy, and
supervisor produced a result.

### 10. Reproduction Completion Checklist

- The Git worktree is clean and its exact revision was recorded.
- The dataset reports 5 episodes, 810 frames, 27D state, and canonical stages.
- The same dataset task text is present in `meta/tasks.parquet`.
- The installed LeRobot files contain the direction and low-motion markers.
- Training passed its two-step smoke test and true-resumed optimizer/scheduler state.
- The continuation run reached its configured target step and contains three complete checkpoints.
- All three checkpoints were evaluated with seed 2 on the same five episodes.
- The best checkpoint and execution chunk were selected from saved comparison files.
- Isaac Sim used the fixed seed-2 pose, targets, checkpoint, and viewport-capable display.
- `rollout_validation.json` reports `"status": "passed"`.
- All stable archives and the two reproducibility manifests were downloaded.

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
- [`docs/analysis/excavator_dataset_schema_analysis.md`](docs/analysis/excavator_dataset_schema_analysis.md): dataset schema and export notes.
- [`docs/vla_32d_collection_code_guide.md`](docs/vla_32d_collection_code_guide.md): 32D data-collection code guide retained for the collection workflow.

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
