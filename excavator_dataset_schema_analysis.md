# Excavator Auto Dataset Schema Analysis

本文说明当前 excavator auto dataset 的数据结构、旧 run 中已经存在/缺失/不可用的数据、新代码会新增的数据，以及如何读取 `excavator_auto_dataset` 文件夹。

参考旧 run：

```text
excavator_auto_dataset/run_20260624_050735
```

该 run 是修改 camera schema 之前生成的历史数据，因此它能代表“旧数据实际包含什么”。新 camera 字段只会在重新 reload runtime 并重新采集的新 run 中出现。

## 1. Auto Dataset 文件夹结构

一个 run 目录大致如下：

```text
excavator_auto_dataset/
  run_YYYYMMDD_HHMMSS/
    run_meta.json
    summary.json
    sand_config.json
    auto_dataset_config.json
    planner_config.json
    quality_gate_config.json
    camera_config.json                 # 新 run 会有
    episodes.jsonl
    successful_episodes.jsonl
    trainable_episodes.jsonl
    rejected_episodes.jsonl
    failed_episodes.jsonl
    diagnostic_episodes.jsonl
    planning_diagnostics.jsonl
    segment_dig.jsonl
    segment_dig_secure.jsonl
    segment_lift_carry.jsonl
    segment_unload.jsonl
    debug_timeline.jsonl
    episode_000001/
      trajectory.jsonl
      events.jsonl
      sand_metrics.jsonl
      meta.json
      plan_debug.json
      score.json
      images/                          # 新 run 会有
        front/
        bucket/
        side/
```

### Index 文件含义

```text
episodes.jsonl              所有完成记录索引
trainable_episodes.jsonl    可训练 full-chain 成功 episode
successful_episodes.jsonl   full-chain success episode
rejected_episodes.jsonl     执行过但被质量/执行原因拒绝的 episode
failed_episodes.jsonl       hard failure episode
diagnostic_episodes.jsonl   诊断样本
planning_diagnostics.jsonl  规划失败诊断，不作为行为训练数据
segment_*.jsonl             分段技能数据索引
```

每个 index row 通常包含：

```text
episode_id
episode_index
status
score
reason
warning_reason
target_xyz
unload_point_xyz
trajectory
events
sand_metrics
meta
score_path
plan_debug
samples
max_bucket_from_pile_particles
lift_bucket_from_pile_particles
final_bin_from_pile_particles
final_spill_from_pile_particles
```

## 2. 旧 run 已存在的数据

旧 run `run_20260624_050735` 的 trajectory 格式：

```text
schema: excavator_auto_state_action_v3
trajectory_format: compact_jsonl_v4
```

旧 run 里每帧 `trajectory.jsonl` 已经存在：

```text
v
ep
id
i
t
phase
phase.index
phase.one_hot
phase.context
label
obs.state
obs.q
obs.dq
obs.ddq
obs.q_cmd
obs.q_err
action
action.ddq
goal.q
target
bucket.tip
bucket.load
bucket.pour
sand
contact
env
cost
label.flags
```

### `obs.state`

旧 `obs.state` 是 14 维：

```text
[
  base_x,
  base_y,
  base_yaw,
  swing,
  boom,
  arm,
  bucket,
  bucket_load_estimate,
  bucket_tip_x,
  bucket_tip_y,
  bucket_tip_z,
  bucket_load_x,
  bucket_load_y,
  bucket_load_z
]
```

注意：

- 当前挖掘机是 fixed-base 采集，`base_yaw` 目前是固定读数/近似值。
- `bucket_load_estimate` 来自沙粒统计估计，不是力传感器。

### `action`

旧 `action` 是 4 维：

```text
[
  swing_cmd_velocity,
  boom_cmd_velocity,
  arm_cmd_velocity,
  bucket_cmd_velocity
]
```

这不是 6 维履带动作。当前 runtime 没有记录：

```text
track_left_velocity
track_right_velocity
```

如果后续要兼容 6 维 LeRobot action，可以在 export 阶段补成：

```text
[0, 0, swing_cmd_velocity, boom_cmd_velocity, arm_cmd_velocity, bucket_cmd_velocity]
```

但原始 dataset 不应伪造履带运动。

### `sand`

旧 `sand` 已包含：

```text
available
n
pile
bucket
bucket_from_pile
bucket_from_initial
bin
bin_from_pile
bin_from_initial
spill_from_pile
spill_from_initial
source_tracking
bucket_from_pile_mass
bin_from_pile_mass
spill_from_pile_mass
```

可用于构造：

```text
bucket_load
bin_transfer
spill_ratio
material_progress
```

### `env`

旧 `env` 已包含：

```text
dig_target
sand_surface_z
sand_surface_source
local_height_patch
soil
unload_landing
unload_release
bin_center
bin_half_size
bin_z_range
bin_aabb
rigid_obstacle_count
```

### `contact`

旧 `contact` 已包含：

```text
bucket_soil_phase
bucket_has_pile_sand
bucket_has_any_sand
bin_has_pile_sand
spill_from_pile
bucket_rigid_overlap
```

### `cost`

旧 `cost` 已包含：

```text
joint_error_max_deg
joint_error_l2_rad
action_l2
speed_l2
accel_l2
jerk_l2
command_accel_l2
energy_proxy
clearance_to_rigid_m
spill_ratio_instant
bucket_from_pile
spill_from_pile
```

## 3. 旧 run 缺失或不可用的数据

旧 run 缺失：

```text
task
observation.state
observation.images.front
observation.images.bucket
observation.images.side
observation.camera
camera_config.json
episode_xxxxxx/images/
```

旧 run 不可用：

```text
observation.effort
joint torque
hydraulic pressure
track_left_velocity
track_right_velocity
depth image
semantic segmentation
camera intrinsics/extrinsics
```

说明：

- `obs.state` 存在，但不是 LeRobot 风格的 `observation.state` 字段名。
- 旧数据没有图像文件，也没有 camera pose，因此不能回填真实 camera observation。
- effort/force/hydraulic pressure 当前 runtime 没有可靠读取接口，不能伪造。

旧 run schema 检查结果示例：

```text
samples_scanned: 40
obs.state: present
action: present
sand: present
env: present
task: missing
observation.state: missing
observation.images.front: missing
observation.images.bucket: missing
observation.images.side: missing
observation.camera: missing
```

## 4. 新代码会新增的数据

新 run 的 schema：

```text
schema: excavator_auto_state_action_v4
trajectory_format: compact_jsonl_v5
camera_schema: excavator_camera_observation_v1
```

每帧新增：

```text
task
observation.state
observation.images.front
observation.images.bucket
observation.images.side
observation.camera
```

### `task`

每帧写入：

```text
"Dig soil from the marked area and dump it into the target container."
```

### `observation.state`

`observation.state` 是 `obs.state` 的 LeRobot 风格 alias，仍是 14 维：

```text
[
  base_x,
  base_y,
  base_yaw,
  swing,
  boom,
  arm,
  bucket,
  bucket_load_estimate,
  bucket_tip_x,
  bucket_tip_y,
  bucket_tip_z,
  bucket_load_x,
  bucket_load_y,
  bucket_load_z
]
```

### `observation.images.*`

新 run 每帧会尝试写相对路径：

```json
{
  "observation.images.front": "images/front/000000.png",
  "observation.images.bucket": "images/bucket/000000.png",
  "observation.images.side": "images/side/000000.png"
}
```

如果 PIL 不可用，会 fallback 为：

```text
.ppm
```

### Camera views

```text
front   front / cabin / scene camera
bucket  bucket-focused camera
side    side-view / third-person camera
```

### `observation.camera`

每帧包含：

```json
{
  "schema": "excavator_camera_observation_v1",
  "available": true,
  "frame_index": 0,
  "image_format": "png",
  "resolution": [320, 240],
  "views": {
    "front": {
      "available": true,
      "path": "images/front/000000.png",
      "shape": [240, 320, 3],
      "dtype": "uint8",
      "format": "png",
      "prim_path": "...",
      "pose": {
        "available": true,
        "position": [x, y, z],
        "world_transform": [[...], [...], [...], [...]]
      }
    }
  }
}
```

### New run-level files

新 run 会新增：

```text
camera_config.json
```

`run_meta.json` 和每个 `episode_xxxxxx/meta.json` 也会新增：

```text
camera_observations
lerobot_schema_notes
trajectory_fields_added_v4
```

## 5. 新数据仍然不可用的字段

新代码仍不写：

```text
observation.effort
joint torque
hydraulic pressure
depth image
semantic segmentation
true mobile-base track action
```

原因：

- 当前 runtime 没有可靠 effort/hydraulic pressure 读取接口。
- 当前 auto collect 使用 fixed-base 挖掘机，上车 `swing/boom/arm/bucket` 是主要动作。
- RGB camera 已加入，但 depth/segmentation 需要单独接 Isaac render product / annotator。

## 6. 如何读取 Auto Dataset

### 使用内置分析工具

检查最新 run：

```powershell
D:\450\apps\isaacsim\kit\python\python.exe excavator_dataset_tools.py --latest --analysis --compact --no-timeline
```

生成 HTML 图表：

```powershell
D:\450\apps\isaacsim\kit\python\python.exe excavator_dataset_tools.py --latest --plots
```

打开：

```text
excavator_auto_dataset/<latest_run>/analysis_plots/report.html
```

### Python 读取 index 和 trajectory

```python
from pathlib import Path
import json

run_dir = Path(r"D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_20260624_050735")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

episodes = list(read_jsonl(run_dir / "trainable_episodes.jsonl"))
row = episodes[0]

trajectory_path = Path(row["trajectory"])
episode_dir = trajectory_path.parent

for sample in read_jsonl(trajectory_path):
    state = sample.get("observation.state") or sample.get("obs.state")
    action = sample["action"]
    task = sample.get("task", "Dig soil from the marked area and dump it into the target container.")

    front_rel = sample.get("observation.images.front")
    front_path = episode_dir / front_rel if front_rel else None

    print(sample["i"], sample["phase"], len(state), len(action), front_path)
    break
```

### 读取图像为 Tensor

```python
from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np

def image_to_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x

sample = next(read_jsonl(trajectory_path))
front_rel = sample.get("observation.images.front")

if front_rel:
    image = image_to_tensor(episode_dir / front_rel)
    print(image.shape)  # [3, H, W]
```

### LeRobot-style 映射建议

第一版可映射为：

```python
lerobot_sample = {
    "observation.state": sample.get("observation.state") or sample.get("obs.state"),
    "observation.images.front": front_tensor,
    "observation.images.bucket": bucket_tensor,
    "observation.images.side": side_tensor,
    "action": sample["action"],
    "task": sample.get("task", episode_meta.get("task", "")),
}
```

如果下游必须使用 6 维 action：

```python
action4 = sample["action"]
action6 = [0.0, 0.0] + action4
```

但推荐在 dataset metadata 里明确：

```text
native_action_dim = 4
native_action_names = [
  swing_cmd_velocity,
  boom_cmd_velocity,
  arm_cmd_velocity,
  bucket_cmd_velocity
]
```

## 7. 快速结论

旧数据：

- 有 state/action/sand/env/contact/cost。
- 没有 camera image。
- 没有 effort/force/hydraulic pressure。
- 没有 LeRobot 风格 `observation.state` 字段名，但可以从 `obs.state` 直接映射。

新数据：

- 保留所有旧字段。
- 新增 `task`。
- 新增 `observation.state`。
- 新增 front/bucket/side RGB image path。
- 新增 camera metadata 和 pose。
- 新增 `camera_config.json`。
- effort 仍不可用，必须后续接入真实传感/仿真接口。

