# Changelog

## 2026-07-16 - Simulator Bridge Optimization

Author: ylx

This release optimizes deployment of the existing 18-dimensional SmolVLA checkpoint. It does not change the model to the newer 32-dimensional dataset schema.

### Changed

- Reworked [`run_simulation.py`](run_simulation.py) into the authoritative protocol-v2 simulator bridge for the old 18D policy (`14D observation.state + 4D observation.effort -> 4D commanded joint velocity`).
- Replaced raw articulation-array assumptions with a strict runtime mapping from `robot.dof_names` to `[swing, boom, arm, bucket]` for state, effort, and action handling.
- Replaced the fixed `1/30 s` action step and ignored client `ticks` value with a drift-free 60 Hz physics scheduler. Each policy action now advances `1 / training_fps` seconds, where `training_fps` comes from the original dataset metadata.
- Integrated commanded joint velocity on every physics substep instead of only once per policy request.
- Replaced serial camera switching and settle frames with three persistent camera viewports. Camera requests are submitted together, and image resize work runs concurrently.
- Moved TCP request handling to a dedicated network thread so socket waiting no longer drives a tight simulator render loop.
- Stopped rendering during physics-only substeps and throttled idle UI updates with `--idle-ui-hz` (default: 5 Hz), reducing unnecessary RayTracedLighting work while waiting for policy inference.
- Updated [`scripts/bridge_test/smolvla_policy_client.py`](scripts/bridge_test/smolvla_policy_client.py) and the former persistent policy client for the protocol-v2 handshake and removed the obsolete `ticks` contract.
- Converted the former persistent bridge server into a status helper; that redundant helper was later removed after `run_simulation.py` became authoritative.

### Added

- Added [`excavator_common/deployment_contract.py`](excavator_common/deployment_contract.py) with the shared state/action/camera schema, protocol version, normalization hashes, dataset-FPS loading, DOF resolution, and physics tick scheduling.
- Added fail-closed client validation for exact 18D state features, 4D action features, three expected camera keys, and 18D/4D normalization statistics.
- Added a startup handshake that checks schema names and dimensions, units, training FPS, and normalization asset hashes before control begins.
- Added a real `bucket_load_estimate` computed from simulated particle positions using the runtime's closed-bucket proxy profile instead of always sending zero.
- Added bridge timing telemetry for physics, camera render/copy, resize, encoding, bucket-load calculation, total request time, physics ticks, and simulated time. The policy client reports p50, p95, maximum latency, and real-time factor.
- Added [`tests/test_deployment_contract.py`](tests/test_deployment_contract.py) for schema, DOF mapping, dataset metadata, hashing, and fractional tick-scheduler behavior.

### Compatibility Notes

- This deployment path intentionally supports the old 18D checkpoint only. A new `28D state + 4D effort = 32D` checkpoint requires retraining or a separately trained input projection; truncating or padding features is not supported.
- The policy client must receive the original training FPS through `--dataset-meta` or `SMOLVLA_DATASET_META`. `--training-fps` is available only as a diagnostic fallback when the correct value is independently known.
- Legacy clients that still send `ticks` are rejected until they are migrated to protocol v2.
- The optimized bridge is deterministic lockstep: one inference is followed by the correct amount of simulated time. Continuous real-time execution of the last safe action while inference is running remains future work.
- Three viewport captures still require a non-headless Isaac Sim session.

### Verification Completed

- All 11 deployment-contract unit tests pass.
- Python AST parsing passed for the modified bridge, client, and shared-contract files.
- `git diff --check` passed.

### Runtime Validation Still Required

- Confirm the actual excavator `robot.dof_names`, action directions, velocity units, effort ordering, and joint-limit behavior in Isaac Sim.
- Confirm all three persistent viewports return fresh, non-black, correctly posed images after repeated requests.
- Verify the installed LeRobot checkpoint exposes exactly the expected state, effort, action, and camera features, with normalization applied exactly once.
- Confirm `SMOLVLA_DATASET_META` points to the old 18D dataset and that its `fps` matches the checkpoint's training data.
- Compare the particle-based bucket-load proxy with the training-time bucket geometry and verify that the reported load changes during scoop, carry, and dump phases.
- Measure end-to-end p50/p95/max latency, real-time factor, GPU utilization, and memory contention when Isaac Sim and SmolVLA share a GPU.
- Test reconnect, disconnect, shutdown, and safe zero-action behavior.
- Test fractional scheduling when the dataset FPS is not an integer divisor of 60 Hz.
- Validate the persistent real-time control design separately before enabling action hold/interpolation during inference.

## Unreleased

### Added

- Added [run_vla_train_scene.py](run_vla_train_scene.py) as the primary Isaac Sim Python launcher for the full VLA training scene. It loads the original sand-site scene with sand, truck, and excavator without requiring `main.py`.
- Added launcher options for VLA training and data collection, including `--sand-amount`, `--headless`, `--with-ui`, `--bridge`, `--bridge-port`, `--auto-collect`, `--success-count`, `--max-attempts`, `--dataset-root`, and export controls.
- Added [scripts/bridge_test/sand_site_tcp_bridge_server.py](scripts/bridge_test/sand_site_tcp_bridge_server.py) so SmolVLA can attach to an already-running full sand-site scene from the Isaac Sim Script Editor workflow.
- Added [excavator_common/](excavator_common) for shared path, geometry, and bridge-protocol helpers.
- Added [tests/test_excavator_common.py](tests/test_excavator_common.py) for lightweight shared-helper coverage.

### Changed

- Promoted the excavator runtime code from the extracted package into [main.py](main.py), [scripts/excavator_app/](scripts/excavator_app), and [excavator_dataset_tools.py](excavator_dataset_tools.py).
- Moved the promoted URDF configuration layers into [assets/urdf/URDF_real3/urdf/URDF_real3/configuration/](assets/urdf/URDF_real3/urdf/URDF_real3/configuration) so they live with the Isaac Sim assets.
- Updated [scripts/excavator_app/sand_site_runtime.py](scripts/excavator_app/sand_site_runtime.py) and [scripts/excavator_app/excavator_runtime.py](scripts/excavator_app/excavator_runtime.py) to support no-UI/headless runtime modes through `EXCAVATOR_NO_UI` and `EXCAVATOR_HEADLESS`.
- Updated the sand runtime to honor `EXCAVATOR_SAND_AMOUNT` and keep sandbox fill height aligned with the requested sand amount multiplier.
- Updated [scripts/bridge_test/smolvla_policy_client.py](scripts/bridge_test/smolvla_policy_client.py) to consume multi-camera bridge payloads and feed SmolVLA with the available excavator camera observations.
- Added three excavator-mounted camera feeds to [run_excavator_standalone.py](run_excavator_standalone.py) and updated [scripts/bridge_test/gui_client.py](scripts/bridge_test/gui_client.py) to render `left`, `front`, and `right` views side by side.
- Removed empty placeholder scripts from [scripts/](scripts) so only usable launch and diagnostic helpers remain.
- Expanded [README.md](README.md) into the single project runbook, folder guide, launch guide, and Markdown index.

### Organized

- Consolidated project documentation so [README.md](README.md) is the only `README.md` in the project.
- Moved analysis documents into [docs/analysis/](docs/analysis).
- Moved the old standalone camera/API backup into [archive/standalone/](archive/standalone).
- Moved backup SmolVLA bridge clients into [scripts/bridge_test/archive/](scripts/bridge_test/archive).
- Updated archived backup scripts and documentation references after the folder reorganization.
