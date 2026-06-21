# Changelog

## Unreleased

- Promoted the excavator runtime code from the extracted package into [main.py](main.py), [scripts/excavator_app/](scripts/excavator_app), and [excavator_dataset_tools.py](excavator_dataset_tools.py).
- Moved the promoted URDF configuration layers into [assets/urdf/URDF_real3/urdf/URDF_real3/configuration/](assets/urdf/URDF_real3/urdf/URDF_real3/configuration) so they live with the Isaac Sim assets.
- Expanded [README.md](README.md) into a file-by-file Isaac Sim usage guide and kept the viewport-based RGB bridge contract intact.
- Added three excavator-mounted camera feeds to [run_excavator_standalone.py](run_excavator_standalone.py) and updated [scripts/bridge_test/gui_client.py](scripts/bridge_test/gui_client.py) to render `left`, `front`, and `right` views side by side.
- Removed empty placeholder scripts from [scripts/](scripts) so only usable launch and diagnostic helpers remain.
