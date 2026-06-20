# Excavator Sand Dataset Demo

This package contains the Isaac Sim script UI for running the excavator, real particle sand site, dig planner, trace preview, and auto dataset collection.

## Requirements

- Isaac Sim Full 5.1.0
- The excavator USD scene:

```text
file:/D:/450/assets/usd/URDF_fixed/URDF_real3_fixed/URDF_real3_fixed.usd
```

## Start

1. Open Isaac Sim Full 5.1.0.

2. Open the excavator USD scene:

```text
File -> Open
D:/450/assets/usd/URDF_fixed/URDF_real3_fixed/URDF_real3_fixed.usd
```

3. Open the Script Editor:

```text
Window -> Script Editor
```

4. Extract this code package. For the easiest setup, place the extracted folder near the USD assets, for example as a sibling of the `URDF_fixed` folder.

In Script Editor, open and run the `main.py` from that extracted folder.

Example:

```text
<extracted_package_folder>/main.py
```

`main.py` will try to auto-detect the package root from:

- `EXCAVATOR_PROJECT_ROOT`
- `URDF_REAL3_ROOT`
- the original `main.py` folder
- the current working directory
- the current opened USD scene folder and nearby parent/sibling folders

If Script Editor copies the script to a temporary path and the package is not near the USD scene, run this small launcher instead. Replace `ROOT` with the extracted package folder:

```python
import os
import runpy

ROOT = r"D:\path\to\URDF_real3_code"
os.environ["EXCAVATOR_PROJECT_ROOT"] = ROOT
runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")
```

5. Wait for two windows:

- `Sand Site Control`
- `Excavator Slider Control v3`

## Recommended First Run

1. In `Sand Site Control`, click:

```text
Reset Sand
```

Wait until the sand settles.

2. In `Excavator Slider Control v3`, use:

```text
Build Dig Plan
Trace path
Run All
```

For automatic dataset generation, set `Target trainable Count`, then click:

```text
Start
```

## Main Controls

### Sand Site Control

- `Sand amount`: changes sand height/amount.
- `Efficiency <-> Realistic`: controls particle fidelity and performance.
- `Apply`: apply parameter values.
- `Apply + Rebuild`: apply and rebuild the sand site.
- `Reset Sand`: rebuild real particle sand.
- `Status`: print current sand/particle diagnostics.

### Excavator Slider Control v3

- Joint sliders: manually command swing, boom, arm, bucket.
- `Build Dig Plan`: build the current dig plan.
- `Trace target`: green target/semantic trace.
- `Trace path`: blue planned bucket path.
- `Run Next Step` / `1 Pre` ... `8 Unload`: debug individual stages.
- `Run All`: execute the cached dig plan.
- `Auto Dataset`: automatically plan, dig, carry, unload, score, and save episodes.
- `Dir`: open the current dataset output directory.
- `Replay`: replay the latest recorded episode.

## Dataset Output

Auto dataset output is written under the extracted package folder by default:

```text
<extracted_package_folder>/excavator_auto_dataset/
```

To write datasets somewhere else, set `EXCAVATOR_DATASET_ROOT` before running `main.py`.

Each run creates a folder like:

```text
run_YYYYMMDD_HHMMSS/
```

Important files:

- `episodes.jsonl`: all executed episode records.
- `trainable_episodes.jsonl`: successful training-quality episodes.
- `failed_episodes.jsonl`: execution failures.
- `rejected_episodes.jsonl`: completed but low-quality episodes.
- `planning_diagnostics.jsonl`: planning failures/diagnostics.
- `summary.json`: run summary.
- Each episode folder contains `trajectory.jsonl`, `events.jsonl`, `score.json`, and `plan_debug.json`.

## Notes

- Planning failures are diagnostics, not trainable robot trajectory data.
- Only complete dig -> carry -> unload episodes that pass quality gates are written to `trainable_episodes.jsonl`.
- If the sand looks unstable after startup, click `Reset Sand` once in `Sand Site Control`.
- If FPS is low, move the `Efficiency <-> Realistic` slider toward `Efficiency`, then click `Apply + Rebuild` and `Reset Sand`.
