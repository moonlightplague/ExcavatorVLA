# ExcavatorVLA handover record

Handover date: 2026-07-23

## Source revisions

- Isaac Sim data collection: `isaac` at
  `dada72d0eefd1dda9a63739862a571fc0f1a7b8a`
- SmolVLA training and simulation demo: `smolvla` at
  `19ba3240d463a671a1147a018005f1fdf3ff458d`
- Repository: <https://github.com/jxbb824/ExcavatorVLA>

Both branch workspaces were clean when the handover was prepared, and both
branches contained their own README and CHANGELOG.

## Dataset

Remote source:

```text
root@120.209.70.195:/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3
```

Portable handover artifacts:

```text
ExcavatorVLA-lerobot_v3-handover.zip
ExcavatorVLA-lerobot_v3-handover.zip.sha256
```

Pre-transfer inventory:

- Reported size: 15 GB
- File count: 4,279
- Exact remote byte count: 15,955,271,812

The ZIP is deliberately excluded from the Git release and delivered separately
because of its size. It contains the top-level `lerobot_v3/` directory so the
LeRobot directory structure is preserved.

Before extraction, place the ZIP and checksum file together and verify them:

```bash
sha256sum -c ExcavatorVLA-lerobot_v3-handover.zip.sha256
unzip ExcavatorVLA-lerobot_v3-handover.zip
```

## Handover validation

- Read the branch-specific README before running a workflow.
- Confirm dataset metadata and feature dimensions before training.
- Keep Isaac Sim and policy environments separate.
- Use the protocol and checkpoint paths documented on the `smolvla` branch.
- Record any future data-collection schema change on both affected branches.
