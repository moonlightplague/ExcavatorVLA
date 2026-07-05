# Linux Camera Black-Frame Diagnostic

This folder contains a standalone Isaac Sim camera diagnostic for Linux headless
black-frame issues. It does not run auto collect, planner, dashboard, or VLA
export. It only probes Replicator RGB output and writes a self-contained log.

## Remote Run

From the repository root on Linux:

```bash
cd /root/gpufree-data/ExcavatorVLA

SCENE_PATH=/root/gpufree-data/ExcavatorVLA/assets/usd/excavator_scene.usd \
OUT_BASE=/root/gpufree-data/ExcavatorVLA/scripts/linux_camera_diag/output \
bash scripts/linux_camera_diag/run_linux_camera_diag.sh
```

The output folder will be:

```text
scripts/linux_camera_diag/output/linux_camera_diag_YYYYMMDD_HHMMSS/
```

Important files:

```text
linux_camera_diag.log
camera_diag_results.json
clean_scene.ppm
loaded_camera_0.ppm
loaded_camera_1.ppm
loaded_camera_2.ppm
```

The wrapper also creates:

```text
scripts/linux_camera_diag/output/linux_camera_diag_YYYYMMDD_HHMMSS.tar.gz
```

Copy the tarball back for analysis:

```bash
scp root@<remote>:/root/gpufree-data/ExcavatorVLA/scripts/linux_camera_diag/output/linux_camera_diag_*.tar.gz .
```

## Interpretation

- `clean_scene` fails or stays black: Linux Isaac/RTX/Hydra/Replicator
  offscreen rendering is not working.
- `clean_scene` succeeds but loaded cameras fail: scene/camera/light/USD
  composition/render delegate is the likely issue.
- loaded cameras succeed here but auto collect fails: runtime camera timing or
  wrong module version is the likely issue.

