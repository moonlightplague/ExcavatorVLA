# Changelog

## Unreleased

- Promoted the excavator runtime code from the extracted package into [main.py](main.py), [scripts/excavator_app/](scripts/excavator_app), and [excavator_dataset_tools.py](excavator_dataset_tools.py).
- Moved the promoted URDF configuration layers into [assets/urdf/URDF_real3/urdf/URDF_real3/configuration/](assets/urdf/URDF_real3/urdf/URDF_real3/configuration) so they live with the Isaac Sim assets.
- Expanded [README.md](README.md) into a file-by-file Isaac Sim usage guide and kept the viewport-based RGB bridge contract intact.
- Removed empty placeholder scripts from [scripts/](scripts) so only usable launch and diagnostic helpers remain.