# ExcavatorVLA handover

This `main` branch is the handover index. Active development is intentionally
split into two branches because Isaac Sim data collection and SmolVLA
training/deployment have different runtime environments and dependencies.

## Choose the correct branch

| Workflow | Branch and documentation | Handover commit |
| --- | --- | --- |
| Isaac Sim scene setup, sand/truck randomization, automated data collection, and LeRobot export | [`isaac` branch](https://github.com/jxbb824/ExcavatorVLA/tree/isaac) / [`isaac` README](https://github.com/jxbb824/ExcavatorVLA/blob/isaac/README.md) | `dada72d0eefd1dda9a63739862a571fc0f1a7b8a` |
| SmolVLA training, checkpoint evaluation, protocol-v2 bridge, and simulation demo | [`smolvla` branch](https://github.com/jxbb824/ExcavatorVLA/tree/smolvla) / [`smolvla` README](https://github.com/jxbb824/ExcavatorVLA/blob/smolvla/README.md) | `19ba3240d463a671a1147a018005f1fdf3ff458d` |

Do not use `main` as the Isaac Sim or SmolVLA runtime checkout. Start with the
README on the relevant branch and keep branch-specific changes on that branch.

## Checkout

Use separate folders so both environments can remain available:

```bash
git clone --branch isaac https://github.com/jxbb824/ExcavatorVLA.git ExcavatorVLA-isaac
git clone --branch smolvla https://github.com/jxbb824/ExcavatorVLA.git ExcavatorVLA-smolvla
```

Existing clones can use:

```bash
git fetch origin
git switch isaac
# or
git switch smolvla
```

## Workflow boundary

1. Use `isaac` to configure the scene and collect/export episodes.
2. Transfer the exported LeRobot dataset to the training machine.
3. Use `smolvla` to inspect the dataset contract, train or resume a checkpoint,
   evaluate it, and run the protocol-v2 Isaac Sim demo.
4. Keep dataset directories, checkpoints, logs, and generated videos outside
   Git. They are large runtime artifacts, not source files.

The two branch READMEs are authoritative for commands and environment paths.
When their instructions differ, follow the README from the branch being run.

## Handover dataset

The dataset copied for this handover comes from:

```text
/root/gpufree-data/ExcavatorVLA/excavator_auto_dataset/.dashboard_success/lerobot_v3
```

The dataset is delivered separately from the Git release as:

```text
ExcavatorVLA-lerobot_v3-handover.zip
ExcavatorVLA-lerobot_v3-handover.zip.sha256
```

Download the handover dataset from
[SharePoint](https://utexas-my.sharepoint.com/:f:/g/personal/ly6487_eid_utexas_edu/IgABgzdy4w8FRr4zC0IYaAhgAdI010D5qpShhyt25JPLJUg?e=b35Ebc).

The archive contains the top-level `lerobot_v3/` directory. The source dataset
is approximately 15 GB and contains 4,279 files. Verify the ZIP against the
accompanying SHA-256 checksum before extracting it. See [`HANDOVER.md`](HANDOVER.md)
for verification and release details.

## Release

The handover release records the exact `isaac` and `smolvla` commits above.
Branch history remains separate; the release does not merge either runtime
branch into `main`.
