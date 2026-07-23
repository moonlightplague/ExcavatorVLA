# ExcavatorVLA project handover

This release is the navigation and reproducibility point for the two project
workflows. The runtime branches remain separate because they target different
environments.

## Source revisions

- Isaac Sim data collection and LeRobot export:
  [`isaac`](https://github.com/jxbb824/ExcavatorVLA/tree/isaac) at
  `dada72d0eefd1dda9a63739862a571fc0f1a7b8a`
- SmolVLA training and protocol-v2 simulation demo:
  [`smolvla`](https://github.com/jxbb824/ExcavatorVLA/tree/smolvla) at
  `19ba3240d463a671a1147a018005f1fdf3ff458d`

Each branch contains its own README and CHANGELOG. Start with the
branch-specific README for setup and execution commands.

## Dataset handover

The LeRobot v3 dataset is delivered separately because it is too large for the
Git release:

```text
ExcavatorVLA-lerobot_v3-handover.zip
ExcavatorVLA-lerobot_v3-handover.zip.sha256
```

The archive preserves the top-level `lerobot_v3/` directory. Verify the ZIP
with the accompanying SHA-256 checksum before extracting it.

Source dataset inventory:

- 4,279 files
- 15,955,271,812 uncompressed bytes

See the [main handover guide](https://github.com/jxbb824/ExcavatorVLA)
and
[handover record](https://github.com/jxbb824/ExcavatorVLA/blob/main/HANDOVER.md)
for the workflow boundary and validation checklist.
