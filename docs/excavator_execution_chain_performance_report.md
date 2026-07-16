# Excavator Auto-Collect Execution Chain and Performance Report

> Historical performance report. The measured Isaac/Kit update costs and run
> evidence remain useful, but this file is not the current call-chain or clock
> specification. Since this report was written,
> `begin_auto_dataset_episode()` became `auto_collect_begin_episode()`,
> `execute_dig_plan_sequence()` was replaced by the
> `execute_dig_target_ball()` / `execute_dig_plan_step()` chain, auto collect
> became the sole step-clock owner by default, camera capture gained a
> background scheduler, and exported sample/derivative time now defaults to
> simulation/train time rather than wall time. See
> `docs/current_runtime_architecture.md` for the audited 2026-07-15 chain.

本报告用于交给外部专家分析当前自动采集速度问题。核心问题是：

> 路径规划通常几十秒以内，但一个 attempt 的动作执行经常需要数分钟。按直觉 4 自由度机械臂只需要控制 4 个关节角度，为什么执行比规划还慢？

结论先行：当前执行阶段不是简单发送一次 4D joint target，而是每个动作阶段被拆成很多控制小步；每小步都等待 Isaac/Kit 推进物理、粒子、渲染、相机/数据采样和各种 gate 检查。实测显示 `app.next_update_async()` 对应的 Isaac update frame 在当前场景中经常是 `100-500 ms/frame`，而不是正常交互期望的 `16-33 ms/frame`。因此 planned 1 秒动作会被放大成十几秒甚至几十秒。

---

## 1. 当前任务目标

目标是高效生成可用于 VLA / LeRobot 训练的数据：

- 三路相机：`observation.images.0/1/2`
- 关节状态：`q / dq / ddq`
- action / q_cmd / q_err
- bucket 内 sand 数量
- 最终 truck bed / unload mesh 内 sand 数量
- stage / task prompt / episode metadata

当前不是纯交互 demo，而是 scripted expert data generator。它需要真实执行挖掘、装载、抬升、移动到 truck、倾倒，并记录训练数据。

---

## 2. 关键代码入口

主要文件：

```text
scripts/excavator_app/excavator_runtime.py
scripts/excavator_app/auto_dataset_collect.py
scripts/excavator_app/joint_space_planner.py
scripts/excavator_app/excavator_dataset_camera.py
```

关键函数：

```text
auto_collect_loop()
  -> auto_collect_prepare_attempt / scene randomize / sand reset
  -> find_plan()
  -> begin_auto_dataset_episode()
  -> execute_dig_plan_sequence()
  -> move_to_profile()
  -> step_updates()
  -> dataset_record_sample_async()
```

关键配置：

```text
CONTROL_HZ = 30
control_step_frames() = 2 by default
dataset_sample_interval = 0.20 s
auto_collect_owns_step_clock = true
planner_version = dig_plan_v4_joint_space_world_debug_expert_v1
```

---

## 3. 控制链路：一个 stage 如何执行

以 `move_to_profile(q_goal, seconds, label, mode)` 为核心，当前动作 stage 的执行链路如下：

```text
1. wait_for_articulation_action_ready()
   确认 robot / physics view / action channel 可用。

2. sync_motion_start_q()
   读取当前真实/命令关节状态作为 q0。

3. clip / loaded carry posture adjustment
   对 q_goal 做 joint limit、loaded carry、bucket hold 等处理。

4. estimate_stage_motion_seconds()
   估算该 stage 逻辑时间 seconds_eff。

5. steps = int(seconds_eff * CONTROL_HZ)
   例如 planned 1.0s -> 30 个控制小步。

6. 对每个控制小步：
   a. interpolate_q_motion(q0, q1, s)
   b. CTRL.apply_target_direct(q)
   c. await step_updates(control_step_frames())
      默认 control_step_frames=2，所以每个小步等 2 个 Isaac update frame。
   d. dataset_record_sample_async() if sample due
   e. update_sand_contact_progress() for contact/carry stages
   f. carry/recovery/freeze/check gates

7. stage_audit done/failed
   记录 planned duration、wall duration、q_cmd/q_real、sand 等。
```

因此一个 planned 1 秒的 stage 不是一次命令，而是：

```text
1.0s * CONTROL_HZ 30 * control_step_frames 2
= 60 个 Isaac update frame
```

如果 Isaac update 是 `16.7 ms/frame`，这个 stage 接近实时。

如果 Isaac update 是 `250 ms/frame`，这个 stage 就变成：

```text
60 * 250ms = 15s
```

这正是当前日志中看到的现象。

---

## 4. step_updates() 的作用

当前 `step_updates(n)` 的核心是：

```python
async with _STEP_UPDATES_LOCK:
    await app.next_update_async()
```

也就是说它等待 Kit/Isaac 完成下一帧 update。这个 update 可能包含：

- articulation / PhysX step
- particle sand simulation
- render / viewport / render products
- camera backend 相关更新
- USD / timeline / event processing
- Python callbacks / async tasks

所以它不是一个轻量 sleep，也不是单纯发控制命令。

当前最大的怀疑点是：`app.next_update_async()` 在这个场景中每帧非常慢。

---

## 5. 数据记录链路

每当 `dataset_record_sample_due()` 成立时，会调用：

```text
dataset_record_sample_async()
  -> dataset_capture_camera_observations()
  -> dataset_record_sample()
```

`dataset_record_sample()` 会写一条训练 row，内容包括：

```text
timestamp
phase / label
observation.state
observation.effort
obs.q / obs.dq / obs.ddq
obs.q_cmd / obs.q_err
action / action.ddq
goal.q
bucket.tip / bucket.load / bucket.pour
sand.bucket_from_pile
camera image paths
```

当前 sample 间隔目标是：

```text
dataset_sample_interval = 0.20s
```

也就是目标 5Hz。但因为执行 update 很慢，实际 sample rate 经常低于 5Hz。

注意：这部分本身会有开销，但最新分析中最大问题仍是 Isaac update frame 慢，而不是 JSON 写盘。

---

## 6. 实测案例：run_20260709_194400 / episode_000053

路径：

```text
D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_20260709_194400\episode_000053
```

这个 episode 是关掉另一个并行 Isaac run 之后的后段 trainable episode。

总耗时：

```text
上一个 episode end -> episode_000053 end: 353.4s
episode_000053 start -> end: 266.0s
task_start -> end: 263.9s
前置 prepare / randomize / reset / find_plan: 约 89.5s
正式执行: 约 263.9s
```

stage 级耗时：

| Stage | planned duration | wall time | wall / planned |
|---|---:|---:|---:|
| `pre_dig` | 1.10s | 16.23s | 14.8x |
| `approach_contact` | 1.00s | 13.92s | 13.9x |
| `insert_cut` | 1.15s | 9.05s | 7.9x |
| `pull_mid_cut` | 2.20s | 9.90s | 4.5x |
| `curl_to_hold_material` | 1.45s | 47.27s | 32.6x |
| `pull_exit_cut` | 1.45s | 5.90s | 4.1x |
| `secure_load` | 0.85s | 3.70s | 4.4x |
| `secure_load_recovery` | 0.85s | 3.88s | 4.6x |
| `lift_carry` | 1.00s | 15.05s | 15.0x |
| `unload_to_bin` | 8.71s | 131.49s | 15.1x |

估算每个 Isaac update frame 的 wall cost：

| Stage | estimated update frame cost |
|---|---:|
| `pre_dig` | ~246 ms/frame |
| `approach_contact` | ~232 ms/frame |
| `insert_cut` | ~133 ms/frame |
| `pull_mid_cut` | ~75 ms/frame |
| `curl_to_hold_material` | ~550 ms/frame |
| `lift_carry` | ~251 ms/frame |
| `unload_to_bin` | ~252 ms/frame |

如果目标是实时/准实时，update frame 应该接近 `16-33 ms/frame`。当前明显不正常。

---

## 7. 关掉并行 run 前后的速度差异

同一个 `run_20260709_194400` 中，23:52 左右关掉另一个并行 run 后：

episode 完成间隔：

```text
关掉之前：平均 9.5 min / episode
关掉之后：平均 4.9 min / episode
```

stage 平均耗时变化：

| Stage | before 23:52 avg | after 23:52 avg |
|---|---:|---:|
| `unload_to_bin` | 302.8s | 100.2s |
| `curl_to_hold_material` | 65.7s | 34.9s |
| `pre_dig` | 45.7s | 18.3s |
| `approach_contact` | 33.1s | 11.4s |
| `lift_carry` | 30.4s | 11.9s |

结论：

```text
并行 Isaac / auto collect 进程会显著拖慢单个 episode。
但关掉并行后，单实例仍然慢，说明还有单实例内部瓶颈。
```

---

## 8. 为什么别人的 4 自由度机械臂可以实时避障，而这里慢？

通常实时交互的 4DOF 机械臂只做：

```text
joint target / velocity command
rigid-body collision check
kinematic interpolation
低粒子或无粒子仿真
低频或无相机记录
无训练数据写盘
```

而当前 excavator auto collect 同时做：

```text
1. 4DOF articulation 控制
2. 大量 particle sand simulation
3. bucket 内 sand 计数
4. truck/unload bin 最终 sand 计数
5. 三路 camera capture
6. VLA trajectory row 写盘
7. contact/carry/recovery gate
8. full task quality scoring
9. 随机 sand/truck/robot 场景
10. best-of-K expert planner
```

所以它不是一个纯实时机械臂控制器。

但是，即便考虑这些，`app.next_update_async()` 达到 `100-500 ms/frame` 仍然值得重点排查。当前系统可能存在架构问题：free-space 阶段也在等待完整粒子/渲染/数据链路，而不是只在真正接触沙子和倾倒时启用高成本物理检查。

---

## 9. 当前可能存在的问题

### 9.1 控制时钟绑定到 Kit update

执行速度完全受 `app.next_update_async()` 影响。如果 Kit frame 慢，动作就慢。

问题：

```text
planned duration 是逻辑轨迹时间；
wall duration 被 Kit update frame cost 放大；
dataset timestamp 目前记录 wall-clock time。
```

需要专家判断是否应改为：

```text
simulation time / planned trajectory time 作为 dataset timestamp
wall-clock time 只用于 profiler
```

### 9.2 free-space 与 contact stage 使用同样重的执行链

`pre_dig`、`lift_carry`、`unload_to_bin` 这类 stage 也等待完整 Isaac update。

可能优化：

```text
free-space: kinematic/rigid-only fast execution
contact/dig/dump: full particle physics
```

### 9.3 `control_step_frames=2` 会直接放大 wall time

当前每个控制小步等待 2 个 Kit frames。

如果单帧 250ms：

```text
每个控制小步 = 500ms
30Hz 逻辑控制 = 实际 2Hz
```

需要专家判断：

```text
data collection 是否必须每个控制小步等 2 frames？
是否可根据 stage 自适应：
  free-space = 1 frame 或更少 sample
  contact = 2 frames
  settle = state-based
```

### 9.4 `unload_to_bin` 太长

EP53 中：

```text
unload_to_bin planned 8.71s
wall 131.49s
```

这个 stage 本来逻辑上只是 loaded carry + dump 到 truck。它不应该单独占半个 episode。

可能原因：

```text
large swing + loaded carry route 被拆成 261 控制 steps
每 step 等 2 Kit frames
每 frame 约 250ms
```

需要判断是否应该：

```text
1. free-space loaded carry 用更少控制小步
2. route retiming 用 waypoint/chunk 控制，而不是 30Hz dense command
3. dump 前后才启用密集 physics/camera/sample
```

### 9.5 `curl_to_hold_material` gate 太重

EP53 中：

```text
curl_to_hold_material planned 1.45s
wall 47.27s
```

这个阶段会做 sand contact progress、bucket load、carry geometry、recovery 等检查。

可能问题：

```text
1. 它等待真实 bucket/sand 条件太久
2. 每次检查触发粒子 snapshot / bucket metrics
3. recovery 或 gate 逻辑导致 stage 停留过久
```

需要专家判断：

```text
curl/secure 是否应改成明确动作 primitive：
  close bucket -> slight lift/retract -> verify once/twice
而不是在循环里长时间等 gate。
```

### 9.6 粒子数量可能过高

日志中单个 episode 粒子数可达几十万。例如 EP53 phase metrics 显示：

```text
n = 308515 particles
```

之前 sand amount 范围增大后，粒子数有时更高。需要专家确认：

```text
当前粒子数对 Isaac/PhysX 的 expected update time 是多少？
GPU dynamics 是否正确使用？
是否存在 CPU fallback？
```

### 9.7 data mode 仍然可能做过多工作

即使不是 debug/profile，data mode 仍然必须记录训练数据。但需要检查是否还有不必要的内容：

```text
per-sample bucket metrics
camera capture
effort observation
jsonl append
stage audit
sand metrics
quality trackers
```

这些应分清：

```text
训练必需
质量评估必需
debug/profile 才需要
完全冗余
```

---

## 10. 需要专家重点回答的问题

### Q1. `app.next_update_async()` 为什么会达到 100-500ms/frame？

请帮助判断如何拆分：

```text
PhysX articulation
particle sand
render/camera
USD/Hydra
Python callbacks
dataset sampling
```

希望得到一个方法，能把 Kit update 内部分解成耗时项。

### Q2. 是否应该使用固定 simulation step，而不是等待 Kit update？

当前是：

```python
await app.next_update_async()
```

是否应改为 Isaac/SimulationContext/World 的显式 step，例如：

```text
simulation_context.step(render=False)
world.step(render=False)
physics-only step
```

并且只在需要 camera 时 render。

### Q3. free-space motion 是否可以绕过粒子/渲染？

例如：

```text
pre_dig
lift_carry
unload_to_bin transit
clearance route
```

这些阶段大部分不是铲斗与沙子强交互。是否可以：

```text
1. 用 kinematic interpolation + collision check 生成 state
2. 只在 contact/dump 阶段跑 full PhysX/particles
3. 或每 N 个 waypoint 才 physical verify
```

### Q4. 对 VLA 数据，timestamp 应该用 wall time 还是 simulation/planned time？

当前数据中 timestamp 是 wall-clock time。由于仿真慢，动作本身被记录成很慢。

是否应改为：

```text
dataset timestamp = simulated/planned trajectory time
wall time = profiler only
```

这样即使采集耗时 5 分钟，训练数据仍可表示一个 20-60 秒任务。

### Q5. control_step_frames 是否应该 stage-adaptive？

当前默认：

```text
control_step_frames = 2
CONTROL_HZ = 30
```

是否可改为：

```text
free-space: 1 frame / lower control density
dig/contact: 2 frames
dump/settle: state-based
```

### Q6. `unload_to_bin` 应如何优化？

当前 `unload_to_bin` 是最大耗时。是否应该：

```text
1. route 用较少 waypoint/chunk
2. carry stage 只做 collision/carry envelope，不做密集 physical stepping
3. 到 truck 上方后再开启密集 dump physics
4. dump bucket 不必完整反转到极限角
```

### Q7. `curl_to_hold_material` 应如何设计？

目前这个阶段容易长时间等待 bucket retaining geometry / sand contact gate。

是否应换成：

```text
close bucket -> slight boom/arm lift/retract -> verify bucket load
```

而不是循环等待 progress gate。

### Q8. 粒子 sand 是否应该分层处理？

训练中真正高频需要的是：

```text
bucket 内 sand count
最终 truck bin sand count
```

是否可以避免每帧维护/扫描全局 sand，改为：

```text
bucket volume spatial query only
final unload bin query only
```

### Q9. 4DOF 实时避障可否作为控制层，sand 作为验证层？

是否应该把系统拆为：

```text
fast kinematic planner/controller
  -> real-time 4DOF obstacle avoidance
slow physical validator
  -> sand contact/dump verification
dataset recorder
  -> resample to training timebase
```

---

## 11. 当前建议的改进方向

按优先级：

### P0: 先剖析 `app.next_update_async()`

必须知道单帧慢在哪里：

```text
physics?
particles?
render?
camera?
Python callback?
```

否则只能看到 `step_updates` 总耗时。

### P0: 将 dataset timestamp 与 wall-clock 解耦

数据集应表达任务真实/计划时间，而不是采集机器跑了多久。

建议：

```text
timestamp_train = simulation_time 或 stage accumulated planned time
wall_timestamp = profiler/debug only
```

### P0: `unload_to_bin` 专项优化

它现在占 attempt 最大头。

短期可试：

```text
1. 降低 loaded carry transit 控制步数
2. free-space carry 不每步完整采样
3. dump 附近再恢复密集采样/物理
```

### P0: `curl_to_hold_material` gate 简化

把长时间等待改为有限动作 primitive + 少量检查。

### P1: stage-adaptive physics/render/sample policy

```text
free-space: low sample, low render, fast stepping
dig/contact: full physics, full sample
dump/settle: state-based physics
```

### P1: separate controller clock and camera clock

相机应按训练需要的频率记录，不应该拖住控制循环。

### P1: particle count / GPU dynamics sanity check

确认当前 sand particle 数量和 PhysX backend 是否合理，是否发生 CPU fallback。

---

## 12. 给专家的一句话总结

当前慢不是 4DOF 关节目标计算慢，而是执行循环把 planned trajectory 细分成大量控制小步，每步等待 Isaac/Kit 完整 update。日志显示单个 update frame 常在 `100-500ms`，导致 planned `1s` stage 变成 `10-50s`，planned `8.7s unload_to_bin` 变成 `131s`。需要专家判断如何把实时/快速 4DOF 控制、粒子物理验证、相机采集和 VLA 数据时间轴解耦。

