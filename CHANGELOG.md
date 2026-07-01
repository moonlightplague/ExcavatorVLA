# Changelog

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
