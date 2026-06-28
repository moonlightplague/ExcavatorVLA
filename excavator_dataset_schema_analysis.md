# Excavator Auto Dataset Schema, Cost, and Optimization Report

This report describes the current excavator auto dataset format, the LeRobot v3 export folder, the latest measured runtime cost, and the highest-value optimization plan.

Latest checked run:

```text
D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_20260628_052305
```

## 1. Latest Run Result

The latest run is complete and exported successfully.

```text
requested episodes : 99
max attempts       : 200
attempts used      : 200
trainable/success  : 99
rejected           : 97
failed             : 4
diagnostic         : 0
LeRobot v3 export  : ok
```

LeRobot v3 output:

```text
run_20260628_052305/
  lerobot_v3/
    manifest.json
    README.md
    data/chunk-000/file-000.parquet
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    videos/observation.images.0/chunk-000/file-000.mp4
    videos/observation.images.1/chunk-000/file-000.mp4
    videos/observation.images.2/chunk-000/file-000.mp4
```

Export manifest says:

```text
standard_lerobot_ready : true
state_action_ready     : true
effort_available       : true
vla_training_ready     : true
fps                    : 10
total_frames           : 20420
total_episodes         : 99
video frames/view      : 20420
video shape            : [256, 256, 3]
```

Two raw samples were skipped because the first two frames of episode 0 had missing camera files. The exported parquet and all three videos are aligned at 20420 frames.

## 2. Raw Auto Dataset Structure

Each run contains raw debug and training files:

```text
run_YYYYMMDD_HHMMSS/
  run_meta.json
  summary.json
  sand_config.json
  auto_dataset_config.json
  planner_config.json
  quality_gate_config.json
  camera_config.json
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
    images/
      0/
      1/
      2/
```

The raw folder intentionally keeps debug-heavy files. The clean VLA/LeRobot training data is only inside `lerobot_v3/`.

## 3. Current LeRobot v3 Export Schema

The exported parquet columns are:

```text
index
episode_index
frame_index
timestamp
task_index
observation.state
action
observation.effort
```

Image observations are stored as videos, not parquet arrays:

```text
observation.images.0 -> videos/observation.images.0/chunk-000/file-000.mp4
observation.images.1 -> videos/observation.images.1/chunk-000/file-000.mp4
observation.images.2 -> videos/observation.images.2/chunk-000/file-000.mp4
```

The current three cameras are:

```text
0 : /World/URDF_real3/arm_link/Camera_0
1 : /World/URDF_real3/swing_link/Camera_1
2 : /World/URDF_real3/swing_link/Camera_2
```

### 3.1 `observation.state`

Current state shape:

```text
observation.state: [14]
```

Current meaning:

```text
[
  base_x,
  base_y,
  base_yaw,
  swing_angle,
  boom_angle,
  arm_angle,
  bucket_angle,
  bucket_load,
  bucket_tip_x,
  bucket_tip_y,
  bucket_tip_z,
  bucket_load_x,
  bucket_load_y,
  bucket_load_z
]
```

For the latest export:

```text
bucket_load state index : 7
bucket_load range       : 0 to 4055
base_x/base_y/base_yaw  : fixed-base values; mostly constant
```

### 3.2 `action`

Current action shape:

```text
action: [4]
```

Current meaning:

```text
[
  swing_cmd_velocity,
  boom_cmd_velocity,
  arm_cmd_velocity,
  bucket_cmd_velocity
]
```

There is no real tracked-base action in the current fixed-base auto collection. Do not fabricate `track_left_velocity` or `track_right_velocity` in the raw dataset. If a downstream model requires a 6D action, add a documented adapter layer:

```text
[0, 0, swing, boom, arm, bucket]
```

### 3.3 `observation.effort`

Current effort shape:

```text
observation.effort: [4]
```

The latest run has non-empty effort values. These are exported as real runtime readings, not invented placeholders. Their scale is large and should be normalized by `meta/stats.json` during training.

### 3.4 Image Stats

`meta/stats.json` contains visual stats entries for all three video keys:

```text
observation.images.0
observation.images.1
observation.images.2
```

These use identity/ImageNet-style visual normalization values so LeRobot/SmolVLA loaders can find expected keys. The images themselves are stored in mp4 videos.

## 4. Bucket Load Volume and Sand Count

Current bucket load volume source:

```text
/World/URDF_real3/bucket_link/bucket_cut/node_/mesh_
```

Current mode:

```text
mesh_authored_closed_cavity
```

Latest logs show:

```text
volume_source=mesh_authored_closed_cavity:/World/URDF_real3/bucket_link/bucket_cut/node_/mesh_;
meshes=1;
vertices=120;
faces=40;
edges=120;
cap_faces=0
```

Important interpretation:

- The runtime no longer generates artificial cap faces.
- `cap_faces=0` means the previous boundary-fill logic is disabled.
- `edges=120` comes from the authored USD mesh triangles as read by the runtime.
- Blender may show the source object as 22 vertices / 33 edges / 13 faces / 40 triangles, while USD/runtime reads 120 vertices and 40 triangle faces because vertices may be split per triangle/attribute boundary.

Current bucket count path:

```text
particle snapshot
  -> bucket spatial AABB/grid prefilter
  -> transform candidates into bucket_link local
  -> point-in-authored-bucket-volume mesh test
  -> bucket_from_pile
```

The value used in `observation.state[7]` follows this bucket count estimate.

## 5. Latest Quality Analysis

From 200 attempts:

```text
trainable : 99
rejected  : 97
failed    : 4
```

Trainable quality:

```text
score avg                         : 61.82
score min/max                     : 48.87 / 73.62
max_bucket_from_pile avg          : 2845
lift_bucket_from_pile avg         : 2834
final_bin_from_pile avg           : 774
spill_ratio avg                   : 0.719
freeze_count avg                  : 0
```

All attempts:

```text
score avg                         : 46.96
max_bucket_from_pile avg          : 2563
lift_bucket_from_pile avg         : 1497
final_bin_from_pile avg           : 741
final_spill_from_pile avg         : 28188
spill_ratio avg                   : 0.805
```

Main warnings:

```text
quality_warning/high_spill_ratio : 153
quality_warning/score_low        : 120
```

Interpretation:

- The authored bucket mesh restored nonzero bucket load counts.
- The collection can produce trainable episodes.
- The major remaining quality issue is transfer efficiency into the bin and high spill.
- The trainable threshold currently allows some episodes with high spill as long as they complete the full chain.

## 6. Latest Failure Analysis

Top rejection/failure reasons:

```text
planning_failed/staged_unload_dump_pose: best IK error too high: planar=1.128 m : 47
planning_failed/staged_unload_dump_pose: best IK error too high: planar=0.499 m : 16
execution_failed/freeze_detected / approach_contact bucket stall                 : ~31
quality_rejected/low_final_bin_particles:0                                      : 24
```

Interpretation:

1. Unload pose generation is still too often choosing targets that are hard or impossible for IK.
2. Some approach-contact poses still push the bucket into a physical stall.
3. Low final bin count can be a real physics outcome, but it is also affected by unload target quality and bin reachability.

## 7. Runtime Cost Analysis

The latest profile shows the largest inclusive costs:

```text
dataset_record_sample               total 1836.9s  count 58680  avg 31.30ms
dataset_record_sample.features       total  862.2s  count 32609  avg 26.44ms
dataset_metrics_frame.bucket_load    total  773.2s  count 66330  avg 11.66ms
dataset_record_sample.particle_bucket total 741.2s  count 32609  avg 22.73ms
dataset_record_sample.features.env   total  736.2s  count 32609  avg 22.58ms
dataset_metrics_frame.snapshot       total  716.6s  count 66330  avg 10.80ms
path_obstacle_check                  total  660.8s  count 20985  avg 31.49ms
bucket_load_fast_current             total  652.1s  count 66330  avg  9.83ms
sand_metrics_current                 total  645.3s  count 15409  avg 41.88ms
build_dig_plan_from_current_target   total  643.0s  count   200  avg  3.22s
path_segment_check                   total  327.2s  count 12537  avg 26.10ms
find_clearance_route                 total  178.2s  count   436  avg 408.8ms
dataset_save_rgb_image               total   87.9s  count 97821  avg 0.90ms
dataset_capture_camera_observations  total   84.5s  count 32609  avg 2.59ms
```

### 7.1 Camera is not the bottleneck

Camera capture and image write are not the main bottleneck:

```text
camera capture avg : 2.59ms/sample
image save avg     : 0.90ms/image
```

The expensive parts are particle/sand metrics, environment feature generation, collision/path checks, and planning.

### 7.2 Dataset sampling is now the largest total cost

`dataset_record_sample` dominates because it runs tens of thousands of times. Its expensive subspans are:

```text
particle_bucket : ~22.73ms/sample
features.env    : ~22.58ms/sample
camera          : ~2.62ms/sample
```

This means optimizing sand metrics and environment feature caching will matter more than optimizing image encoding.

### 7.3 Bucket load count remains expensive

Bucket load count is called very often:

```text
bucket_load_fast_current count : 66330
avg                           : 9.83ms
dataset_metrics_frame.bucket_load avg: 11.66ms
```

Spatial grid is enabled and effective:

```text
bucket_load_spatial.enabled              : true
bucket_load_spatial.hits                 : 66330
bucket_load_spatial.misses               : 0
bucket_load_spatial.fallbacks            : 0
dataset_metrics_frame_hits               : 1941
dataset_metrics_frame_misses             : 66330
```

But the number of calls is high, so total cost is still large.

### 7.4 Planning early accept is working

The planner no longer blindly consumes 10s per plan.

Latest plan summary:

```text
budget_seconds       : 10.0
candidate_count      : 10
evaluated            : 1
accepted_plan_ms     : 3348.94
early_accept         : true
early_accept_reason  : staged_prefix_ready:pull_exit_cut
elapsed_ms           : 3348.95
timeout              : false
```

This is a major improvement compared with earlier fixed 10s planning. Remaining planning cost is still meaningful because 200 attempts call it 200 times.

### 7.5 Path cache is connected but hit rate is still limited

Path cache telemetry:

```text
path_obstacle_invocations : 20985
path_obstacle_hits        : 3607
path_obstacle_misses      : 17378
hit rate                  : ~17.2%

path_segment_invocations  : 12537
path_segment_hits         : 3765
path_segment_misses       : 8772
hit rate                  : ~30.0%

predicted_segment_hits    : 9000
predicted_segment_misses  : 92811
```

Interpretation:

- The cache is now actually being queried.
- The hit rate is not high enough.
- Quantized keying and coarser equivalence classes should still help.
- `path_obstacle_entries=0` likely means the entry counter is not representing the active backing store, because hits/misses/puts are nonzero.

## 8. Optimization Priorities

### Priority 1: Reduce unnecessary full sand metrics

Requirement alignment:

- During motion, the dataset mainly needs bucket load.
- Final scoring needs final unload bin count.
- Full pile/bin/spill metrics do not need to run every sample.

Recommended changes:

1. Keep `bucket_from_pile` per recorded sample.
2. Compute full `sand_metrics_current` only at phase gates:
   - after_cut
   - after_secure_load
   - after_lift_carry
   - after_unload_settle
   - final score
3. During normal per-frame recording, avoid global pile/bin/spill scans.
4. Reuse the same particle snapshot and transformed bucket-local candidate set for all same-frame consumers.

Expected benefit:

- Reduces `sand_metrics_current`, `dataset_metrics_frame.snapshot`, and `dataset_record_sample.particle_bucket` total cost.
- This should be the highest-value optimization because these costs dominate total runtime.

### Priority 2: Cache environment features per frame/stage

`dataset_record_sample.features.env` costs ~736s total. Many fields are static or slowly changing:

```text
dig target
unload bin center/half size/z range
rigid obstacle count
planner context
local sand height patch, if not needed every frame
```

Recommended changes:

1. Split env features into:
   - static per run
   - static per episode
   - dynamic per frame
2. Store static/per-episode env data once in episode meta.
3. For every frame, write only a compact reference plus truly dynamic values.

Expected benefit:

- Reduces per-sample JSON construction and repeated expensive environment queries.
- Reduces raw trajectory size.

### Priority 3: Improve bucket mesh inside test cost

Current bucket volume uses authored mesh faces, which is correct for geometry. Cost can still improve:

1. Precompute triangulated mesh once per bucket volume cache version.
2. Precompute triangle normals/edges for point-in-mesh.
3. Vectorize ray intersections over candidate points instead of looping Python point/triangle operations.
4. Keep the spatial AABB/grid prefilter.
5. Only rebuild mesh cache when:
   - mesh source changes
   - stage reloads
   - bucket link changes

Expected benefit:

- Reduces `bucket_load_fast_current` and `dataset_metrics_frame.bucket_load`.
- Keeps the same authored mesh semantics.

### Priority 4: Improve path collision cache hit rate

Current hit rate:

```text
path_obstacle_check ~17%
path_segment_check  ~30%
```

Recommended changes:

1. Quantize joint states in cache key:
   - start with 0.25deg or 0.5deg
   - include obstacle snapshot version
   - include mode and relevant clearance margins
2. Normalize equivalent labels so the same geometry check does not miss due to debug labels.
3. Cache negative results as well as positive results.
4. Batch all samples in a path segment into a single numpy collision check.
5. Keep exact recheck near collision margins to avoid unsafe false positives.

Expected benefit:

- Reduces `path_obstacle_check` and `path_segment_check` totals.
- Directly improves planning throughput without changing behavior.

### Priority 5: Unload pose reachability and target selection

Failures are dominated by unload IK errors:

```text
best IK error too high: planar=1.128m
best IK error too high: planar=0.499m
```

Recommended changes:

1. Add stronger fail-fast before route planning if unload target is not IK-reachable.
2. Cache unload reachability by bin target cell and carry/dump posture class.
3. Prefer central high release targets that are reachable with positive margin.
4. If several targets are reachable, choose the one with:
   - lower IK error
   - higher wall clearance
   - lower expected drift outside bin

Expected benefit:

- Reduces rejected attempts.
- Reduces wasted route planning on impossible unload goals.

### Priority 6: Approach-contact stall reduction

Many non-trainable episodes freeze during approach contact.

Recommended changes:

1. Detect near-ground/bucket-penetration risk earlier.
2. Slightly lift/uncurl before pushing toward contact if bucket is already constrained.
3. Add a softer approach trajectory around the last contact segment.
4. Do not force bucket motion through hard physical resistance.

Expected benefit:

- Reduces `freeze_detected` and `path_deviation` rejections.

## 9. Debug/Profile Mode Guidance

The latest run produced:

```text
debug_timeline.jsonl: ~160 MB
```

For performance runs:

- Use quiet or normal mode.
- Disable detailed profile unless diagnosing costs.
- Keep Calc Viz off unless inspecting geometry.
- Keep camera export enabled only if producing VLA data.

For optimization runs:

- Use profile mode for short runs only.
- Keep `PLAN_BUILD_SUMMARY`, path cache counters, and dataset exclusive spans.
- Avoid writing huge debug timelines for 99+ episode collection unless needed.

## 10. Commands to Inspect the Latest Dataset

Read summary:

```powershell
D:\450\apps\isaacsim\python.bat - <<'PY'
import json
from pathlib import Path
root = Path(r"D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_20260628_052305")
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
print(summary["requested"], summary["attempts"], summary["trainable"])
print(summary["debug_profile_summary"].keys())
PY
```

Read LeRobot parquet:

```powershell
D:\450\conda\envs\isaaclab_py311\python.exe - <<'PY'
from pathlib import Path
import pandas as pd
root = Path(r"D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_20260628_052305\lerobot_v3")
df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
print(len(df), df.columns.tolist())
print(df["episode_index"].nunique())
PY
```

Generate analysis plots:

```powershell
D:\450\conda\envs\isaaclab_py311\python.exe excavator_dataset_tools.py --latest --root excavator_auto_dataset --plots
```

Export/regenerate LeRobot v3:

```powershell
D:\450\conda\envs\isaaclab_py311\python.exe excavator_dataset_tools.py --latest --root excavator_auto_dataset --export-lerobot --export-overwrite
```

## 11. Current Technical Assessment

The dataset schema is now suitable for VLA training:

- state/action are aligned
- three camera videos exist
- effort exists
- task metadata exists
- stats include visual keys
- clean LeRobot v3 subfolder exists

The main remaining problems are not schema-related. They are:

1. High per-sample sand/bucket metrics cost.
2. High path collision checking cost.
3. Unload IK target failures.
4. Approach-contact physical stalls.
5. High spill ratio and low final transfer efficiency.

Recommended next engineering step:

```text
Optimize the data and planning computation path first:
  1. make per-sample recording bucket-only
  2. move full bin/spill metrics to phase/final gates
  3. precompute/vectorize authored bucket mesh inside test
  4. improve path collision cache quantization and batching
  5. add unload reachability target cache/fail-fast
```

This keeps behavior and dataset schema stable while reducing runtime cost.

## 12. Detailed Algorithm-Time Optimization Plan

This section focuses on the user's current constraints:

```text
Per recorded frame:
  compute bucket sand only

Phase/final gates:
  compute unload bin count and spill count

Bucket volume mesh:
  topology is constant
  only bucket_link transform changes
```

These constraints allow a much cleaner and faster algorithm than the current mixed full-metrics path.

### 12.1 Current hot path

Current per-frame dataset recording does this:

```text
dataset_record_sample()
  -> dataset_metrics_frame_bundle(need_full=False)
    -> dataset_particle_snapshot(build_bucket_index=True)
      -> get_sand_snapshot()
        -> sand_particle_positions()
        -> sand_settle_status()
        -> filter_settled_sand_particles()
        -> optional spatial index
      -> build bucket spatial index
    -> bucket_load_fast_current(points=snapshot)
      -> bucket_load_volume_world_aabb()
      -> bucket_spatial_aabb_indices()
      -> project candidate points to bucket_link local
      -> bucket_load_volume_mask_from_local()
      -> initial pile mask check
  -> optional diagnostic env/contact/cost fields
  -> camera capture
  -> jsonl write
```

The latest profile shows the cost of this design:

```text
dataset_record_sample                 1836.9s total
dataset_record_sample.features         862.2s total
dataset_metrics_frame.bucket_load      773.2s total
dataset_record_sample.particle_bucket  741.2s total
dataset_record_sample.features.env     736.2s total
dataset_metrics_frame.snapshot         716.6s total
bucket_load_fast_current               652.1s total
```

The key problem is not one single slow call. It is that per-frame recording still performs too much diagnostic and particle bookkeeping.

### 12.2 First optimization: make per-frame recording bucket-only

The current code already has a `need_full=False` route, but per-frame recording still:

- builds a particle snapshot
- may build/query a spatial index
- computes bucket count
- may record diagnostic env/contact/cost fields
- writes a separate `sand_metrics.jsonl` row

Recommended runtime policy:

```text
NORMAL / QUIET data collection:
  per frame:
    bucket_count only
    observation.state
    action
    effort
    camera

  phase gates:
    after_cut bucket count
    after_secure_load bucket count
    after_lift_carry bucket count
    after_unload_settle full bin/spill count
    final score full bin/spill count

PROFILE / DEBUG:
  optionally enable env/contact/cost/full sand diagnostics
```

This means `DATASET_RECORD_DIAGNOSTIC_FIELDS` should not be enabled by default during production auto collect. In the current code it is `True`, which explains the large `features.env` cost.

Expected win:

```text
dataset_record_sample.features.env can drop close to zero in production runs.
trajectory.jsonl becomes smaller.
debug_timeline pressure decreases.
```

### 12.3 Second optimization: bucket mesh topology cache

The selected bucket volume mesh is fixed:

```text
/World/URDF_real3/bucket_link/bucket_cut/node_/mesh_
```

Only the bucket pose changes. Therefore the following should be built once per stage reload or mesh path change:

```text
BucketVolumeCache:
  local_vertices: float32[N,3]
  faces: int32[M,*]
  triangles: int32[T,3]
  triangle_v0: float32[T,3]
  triangle_e1: float32[T,3]
  triangle_e2: float32[T,3]
  triangle_normal or raycast precomputed terms
  local_aabb_min/max
  edge_pairs for debug drawing
```

Current code calls:

```text
bucket_load_volume_mesh_local()
triangulate_mesh_faces()
point_in_closed_mesh()
```

The triangulation and raycast triangle data should not be recomputed per bucket count. The runtime should only recompute:

```text
world_aabb = transform 8 local AABB corners by current bucket_link matrix
world_to_bucket = inverse(bucket_link local-to-world matrix)
candidate_local = candidate_world @ world_to_bucket
inside = cached_closed_mesh_contains(candidate_local)
```

Expected win:

```text
bucket_load_fast_current avg should fall from ~9.8ms toward low single-digit ms,
depending on candidate count.
```

### 12.4 Third optimization: matrix transform instead of repeated USD point transforms

Current `project_points_to_link_local()` computes local axes by calling:

```text
transform_local_point_to_world(link, [0,0,0])
transform_local_point_to_world(link, [1,0,0])
transform_local_point_to_world(link, [0,1,0])
transform_local_point_to_world(link, [0,0,1])
```

Then it does dot products with the derived axes.

Because bucket pose changes frame by frame, the correct optimized form is:

```text
M_bucket_to_world = ComputeLocalToWorldTransform(bucket_link)
M_world_to_bucket = inverse(M_bucket_to_world)
candidate_local = transform_points(M_world_to_bucket, candidate_world)
```

Compute the matrix once per sample or once per physics frame. Do not call multiple USD transform helper functions per bucket count.

Expected win:

```text
less Python/USD overhead
less repeated transform calculation
cleaner correctness because the full transform matrix is used directly
```

### 12.5 Fourth optimization: particle snapshot lifetime

Current `get_sand_snapshot()` can do more than bucket counting needs:

```text
sand_particle_positions()
sand_settle_status()
filter_settled_sand_particles()
build_sand_spatial_index()
```

For bucket-only per-frame count, the minimum required data is:

```text
points: float32[N,3]
optional particle source mask
optional bucket spatial index
```

Recommended split:

```text
get_particle_points_snapshot()
  points only
  no settle check
  no settled_points
  no general sand spatial index

get_full_sand_snapshot()
  current full behavior
  only for phase/final diagnostics
```

Expected win:

```text
dataset_metrics_frame.snapshot avg can drop from ~10.8ms.
```

### 12.6 Fifth optimization: bucket spatial index reuse

Current bucket spatial index is built from all particles and hit every time:

```text
bucket_load_spatial.hits   : 66330
misses                     : 0
fallbacks                  : 0
```

This proves the spatial path is working, but total call count is high.

Potential improvement:

```text
Per physics frame:
  read particle points once
  build/reuse bucket spatial grid once
  compute bucket count once
  share result with:
    dataset_record_sample
    quality tracker
    phase metrics if same frame
    debug timeline if needed
```

The current frame cache hit count is small relative to misses:

```text
dataset_metrics_frame_hits   : 1941
dataset_metrics_frame_misses : 66330
```

This indicates most consumers are still causing a new metrics frame. The cache key may be too strict (`q_sig` plus short time window), or calls happen with slightly different q/labels.

Recommended cache key:

```text
bucket_metrics_cache_key = (
  particle_snapshot_id or simulation_frame_index,
  bucket_link_transform_quantized,
  bucket_volume_cache_version
)
```

Do not include label. Labels do not change geometry.

### 12.7 Sixth optimization: point-in-mesh acceleration

Since the mesh is fixed, there are three possible inside-test levels.

#### Option A: cached triangle raycast

Keep the current point-in-closed-mesh semantics but precompute triangles.

Pros:

- same behavior
- safest first step

Cons:

- still O(candidate_count * triangle_count)

#### Option B: convex/half-space approximation

If `bucket_cut` is a simple closed convex-ish volume, convert it into planes:

```text
inside = all(dot(local_point, normal_i) <= offset_i + eps)
```

Pros:

- very fast
- vectorizes perfectly

Cons:

- only safe if volume is convex enough
- may misclassify concave regions

#### Option C: local voxel/SDF occupancy

Precompute a local voxel occupancy grid inside the bucket volume:

```text
grid resolution: e.g. 32 x 24 x 24
precompute inside/outside once
per point:
  local -> grid index
  occupancy lookup
```

Pros:

- very fast per point
- works with arbitrary closed mesh

Cons:

- approximate
- needs careful resolution/margin tuning

Recommended sequence:

```text
1. cached triangle raycast
2. benchmark
3. if still expensive, add optional voxel occupancy backend
```

### 12.8 Full bin/spill metrics should be event-driven

Current full metrics are still called in several places:

```text
sand_metrics_current(force=True)
dataset_metrics_frame_full()
record_phase_metrics(... need_full=True)
final scoring
unload settle
```

This is correct for phase/final gates. It should not be used per sample.

Recommended gate list:

```text
after_cut:
  bucket-only is enough unless debugging

after_secure_load:
  bucket-only is enough

after_lift_carry:
  bucket-only is enough

before_unload:
  bucket-only is enough

after_unload_settle:
  full bin/spill metrics

episode_final_score:
  full bin/spill metrics, can reuse after_unload_settle if no sand moved afterward
```

If final score currently calls full metrics twice, cache by final phase timestamp and reuse.

### 12.9 Practical implementation order

Lowest risk first:

```text
Step 1:
  default DATASET_RECORD_DIAGNOSTIC_FIELDS to false for production auto collect
  keep profile/debug modes able to turn it on

Step 2:
  add particle-points-only snapshot for bucket count
  avoid settle/filter/general spatial index in per-frame bucket-only path

Step 3:
  add BucketVolumeCache with pre-triangulated authored mesh
  cache edge pairs for debug

Step 4:
  replace project_points_to_link_local() in bucket load path with one matrix inverse transform

Step 5:
  cache bucket count per sim frame / particle snapshot / bucket transform
  remove label from cache key

Step 6:
  move bin/spill to after_unload_settle and final score only
  reuse final full metrics

Step 7:
  if still slow, add voxel/SDF occupancy backend for bucket volume
```

Expected impact:

```text
features.env:
  can be almost eliminated in production data collection

bucket/sand metrics:
  should drop significantly because full sand scans leave the per-frame path

bucket mesh inside:
  should drop after precompute + matrix transform

overall:
  target dataset_record_sample avg can reasonably move from ~31ms toward ~8-15ms
  without changing the exported LeRobot schema
```

### 12.10 What not to optimize first

Do not start with camera/video encoding. Latest profile:

```text
camera capture avg : ~2.6ms/sample
image save avg     : ~0.9ms/image
```

This is not the primary bottleneck.

Do not remove bucket count from per-frame samples. It is part of `observation.state[7]` and is useful for VLA training.

Do not fabricate bin/spill per-frame values if they are not computed. Store only bucket count per frame, and use phase/final metadata for bin/spill scoring.
