# Repeated dig, unload, and return baseline

From the repository root:

```bash
./collect_repeat.sh 10                  # 10 successful episodes, 3 cycles each
./collect_repeat.sh 10 2 0,1 4          # two GPUs, 4 cycles per episode
```

Arguments are successful **episodes**, worker count, GPU IDs, and cycles per
episode (2–64; default 3). Further arguments go to the baseline/Isaac launcher:

```bash
DATA_ROOT=/path/to/repeat_data POLICY_PYTHON=/path/to/training/python \
  ./collect_repeat.sh 1 1 0 3 --max-attempts 5 --sand-amount 1.0

./collect_repeat.sh 1 1 0 3 \
  --fixed-scene-profile /absolute/path/to/validated_scene.json

python baseline/collect_repeat.py --help
```

`ISAAC_PYTHON` defaults to the root `run_isaac_python.sh`. `POLICY_PYTHON`
defaults to the active Python and needs numpy, pandas, pyarrow, and imageio-ffmpeg
for export; the Isaac Python runtime is separate. `DATA_ROOT` defaults to
`data/repeat`, resolved relative to the caller's directory. The publication is
`$DATA_ROOT/export_repeat`; override it with `EXPORT_DIR`. `SCOOPS_PER_EPISODE`
sets the default cycle count; `COLLECT_STATUS_SECONDS` controls the heartbeat.
Sand amount defaults to 1.0, matching the single-scoop launcher. Override with
`SAND_AMOUNT` or `--sand-amount` for a larger source pile.

The baseline defaults to the established PhysX particle soil, keeping the
existing soil implementation untouched. To use the locally integrated Newton
MPM backend, select `--soil-backend newton_mpm` or set `EXCAVATOR_SOIL_BACKEND`.
Material behavior, resolution, and source-particle quality counts depend on the
backend. The baseline never changes quality thresholds to accept a backend.

## Controller

`collect_repeat.py` boots the original `run_vla_train_scene.py`, then installs
`repeat_runtime.py` after bootstrap reloads its modules. The adapter reuses the
original digging controller, current-sand target ranking, collision checks,
camera scheduler, quality gates, run indices, and LeRobot exporter. Soil source
files are untouched. A small optional curl-candidate hook in the runtime lets
this task enforce the endpoint clearance during planning. A small integration in
`ExcavatorVLA-isaac/excavator_dataset_tools.py` exports the explicit return flag
and validates legitimate unload-to-return transitions.

Each episode follows this sequence:

1. Randomize/reset the scene once, before recording. Start with a raised arm
   facing the pile, or honor an exact profile's `initial_pose_deg`.
2. Measure actual soil in the truck bed and choose a safe, reachable low-fill
   cell. Plan and execute a dig and unload using the existing physical controller.
3. Wait for the deposit to land, then plan a collision-checked return from the
   measured joint pose to the cycle's `pre_dig` pose. Execute the return through
   `move_to_profile`, with recording and physics running throughout.
4. Replan the next dig from the depleted pile and accumulated truck load. Repeat
   the requested number of times, including the **final physical return**.

Dig candidates cover the source polygon beyond the original narrow central
search. New cuts prefer at least 0.28 m from previous targets; target height stays
at least 0.15 m above ground. This avoids repeatedly curling against the floor
of the first crater. The original source-boundary and material-density checks
still filter candidates. Tune with `--dig-target-spacing` and `--min-dig-height`.
The load-point curl waypoint stays at least 1.05 m above ground, with escape,
secure, and lift waypoints above it. These Cartesian targets enter the original
IK and collision planner before execution, keeping the bucket rotation clear
of the crater floor. The curl endpoint must also pass the runtime's existing
gravity-closed check; a bucket-only rotation cannot bypass the raised target.
Tune with `--min-curl-height`. Candidate and overall planning budgets are at
least 12 and 60 seconds respectively to allow the additional posture constraints.

The normalized target rings include fresh cuts near the pile center. Spacing
is measured in meters, so the dense inner rings remain available. Planning
respects the enabled bucket collision meshes. Display meshes and the load-volume marker
do not set the floor clearance. Floor violations feed the IK height correction.
If internal tip markers miss a thin sand layer, real collider vertices can
establish cutting contact using the same particle surface and depth threshold.
Empty-query floor estimates cannot establish contact; actual captured and
delivered sand still has to pass the original quality gates.
The escape after curl uses a retaining carry
orientation instead of reopening to the cutting angle.
Additional closed-bucket IK candidates coordinate boom/arm raising with curl
when a low cut needs more rotation clearance than the stock small adjustments.
The complete curl arc must clear the floor and rigid obstacles. Escape and lift
heights follow the selected curl pose. Targets without a safe low curl are
rejected. If separated targets cannot be planned, fixed-count collection retries
the freshly measured, denser areas of the same pile, including previous cut
locations. Spatial exclusion alone does not mark the material exhausted. This
fallback is recorded as `repeat_dig_revisit` and does not reset the sand or arm.

Cutting phases use a continuous, collision-checked path. A failed direct path
from contact through insertion, pull and curl rejects that IK candidate instead
of inserting generic free-space clearance waypoints that lift the arm and lower
it back into the pile. The planner searches other candidates with the existing
ground, mesh and obstacle checks. Returns try the checked direct path first;
clearance routes remain available for returns and travel around the truck.
Mesh-based target heights reserve 0.10 m above the floor for the bucket's
rotation between cut endpoints; endpoint clearance alone missed that arc.

Predictions preserve the actual 3D marker offsets. Unload cells are preflighted
with bounded opening-center IK, a closed carry posture, and collision checks;
loaded transit uses the same collision sampling resolution as execution. Loaded
clearance waypoints stay in the continuous lift/unload motion group. The dump
opening profile counts physics frames and fails if neither release nor the
requested opening completes before its timeout.

The original `return_home=True` branch teleports joints with recording disabled;
this baseline always calls the digging executor with `return_home=False` and
records its own physical return. The reset/initial placement happens outside
the recorded episode. There are no scene or sand resets between scoops.

Completion requires the requested fixed cycle count, successful physical
returns, the existing per-scoop transfer/spill/score gates, and the original
uniform physical 10 Hz source audit. A blocked return, failed later dig, or bad
camera stream rejects the entire episode. A successful prefix is not silently
published. The existing adaptive truck-particle target and technical-cap stop
are disabled for this fixed-length task.

`bed_fill.py` constructs a grid along the truck collision polygon's long axis,
including rotated trucks. Default grid: 5 × 3. Landing points use the inset bed
polygon (at least 0.60 m wall margin) and the original reach window. Each cell's
90th-percentile surface height, occupancy, and previous landing proximity guide
placement. Empty cells use the bed floor, so they are not mistaken for filled
cells. Off-bed, underground, non-finite, and high airborne particles are excluded.
Small center and motion costs prefer a central first deposit over a wall corner.
The planner only chooses landing targets; it never relocates soil. Actual bed
coverage and surface variation are recorded so uneven loads remain visible.

Options include `--bed-long-cells`, `--bed-short-cells`, `--bed-wall-margin`,
`--return-plan-seconds` (12), and `--dump-settle-seconds` (1). This is a placement
baseline: a small number of scoops cannot guarantee a perfectly level truck bed.

## Data and SmolVLA

The existing raw and LeRobot layouts are preserved:

```text
data/repeat/
  raw/
    .collection_logs/repeat_*/round_*/worker_*.{log,result.json,kit.log}
    run_*/
      summary.json, episodes.jsonl, trainable_episodes.jsonl, ...
      episode_*/
        meta.json, score.json, trajectory.jsonl, events.jsonl, sand_metrics.jsonl
        images/...
        repeat_cycles.jsonl, repeat_metrics.json
      repeat_metrics.jsonl
  export_repeat/
    meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/...
    data/chunk-*/file-*.parquet
    videos/observation.images.{0,1,2}/chunk-*/file-*.mp4
```

Raw metadata retains `episode_mode=multi_scoop` and adds `repeat_baseline`,
`repeat_cycles`, and `repeat_returns_completed`. Per-cycle reports contain the
selected landing, bed fill before/after depositing/returning, return success,
return pose, and episode-local sample boundaries (end is exclusive). Existing
`observation.scoop_index` and stage labels identify all cycles. Return movement
uses phase 0 (`pre_dig`), preserving the established ten-stage vocabulary. The
scheduled writer normally labels these frames `pre_dig`; return event pairs and
the cycle's sample boundaries distinguish them from the next dig's approach.
The task text explicitly requests repeated
digging, spreading, and returns and is retained by the exporter.

Every recorded episode, including failed and rejected episodes, writes
`repeat_metrics.json`. The same report is embedded as `meta.json.repeat_metrics`
and appended with episode identity to the run's `repeat_metrics.jsonl`. Reports
include final payload, unique scooped material, retained material, spillage and
spillage fraction; particle counts and masses in kg; simulation cycle time;
wall runtime; preparation-inclusive attempt runtime; per-scoop execution/cycle
times and counts; payload throughput; final bed coverage and surface variation;
and the original score and quality metrics. Per-scoop metrics also appear in
`repeat_cycles.jsonl` for cycles that reached the return step.

Particle accounting reuses the online evaluator's `ParticleTransferTracker`:
only initial-pile IDs count, repeat scoops do not double-count IDs, initial truck
contents are excluded, and bucket/truck overlap belongs to the bucket.
Spillage includes scooped material returned to the pile or still airborne.
Sampling runs at recording start, each recorded 10 Hz frame, cycle boundaries,
and finish; transfers entirely between observations can be missed. Region
geometry uses this collector's bucket volume and truck polygon/Z bounds, so
the counts need not match evaluation's truck-local bounding box exactly.
Unavailable identity measurements are explicitly null with a reason.

`cycle_time_seconds` measures the entire repeated episode in simulation time,
including settling, planning between scoops and the final physical return;
`cycle_time_wall_seconds` is its wall-clock counterpart. Both are null for
incomplete/rejected episodes, whose elapsed times are still recorded in
`simulated_runtime_seconds` and `runtime_seconds`. `attempt_runtime_seconds`
also includes scene preparation and initial planning. Per-scoop times begin at
executor entry and include settling and return. Every report embeds these
definitions and the `repeat_episode_metrics_v1` schema identifier. Policy and
bridge inference latencies do not apply to this scripted collector.

The parent process exports successful episodes sequentially, avoiding append
contention between simulators. The publication uses `current-v4` 28D state,
4D command velocity actions in rad/s, three RGB 256×256 videos at the source
10 Hz, and excludes unstable effort from policy inputs. Scoop/stage supervision
remains available as extra columns. Source trajectories are not retimed.
`observation.return_to_pile` is a separate integer flag derived from completed
cycle sample bounds and phase 0. It identifies physical returns without adding
state dimensions or inventing another stage. Export validation accepts these
flagged returns and still rejects unmarked regressions or stale lift/dig stages
after unloading. Legacy single-scoop exports do not acquire this optional field.
Worker failures stop the launcher after available successes have been exported;
raw failures and diagnostic records remain inspectable. Ctrl+C stops worker
process groups. The parent retries incomplete targets only while progress is made.

To export an existing raw run again:

```bash
python ExcavatorVLA-isaac/excavator_dataset_tools.py /absolute/path/to/raw/run_... \
  --export-lerobot-v3 --export-append --export-dir /path/to/export \
  --export-state-schema current-v4 --export-effort-policy exclude \
  --export-require-standard --export-require-vla
```

The supplied training configuration targets the default repeat publication:

```bash
bash train.sh --config baseline/training_config.yaml --dry-run
bash train.sh --config baseline/training_config.yaml
```

For a custom `DATA_ROOT`, update `dataset.root` in a copy of that YAML. Paths
are relative to the YAML file, as in the existing trainer. Collection produces
training-ready data; it does not start a training run.

## Validation

Run the simulator-independent tests with the policy Python:

```bash
python -m unittest discover -s baseline/tests -v
```

They check rotated bed geometry, measurements and fill ranking, visit memory,
per-attempt isolation, physical return orchestration and failure propagation,
fixed-count completion, continuous-cut routing, sampled particle accounting,
successful/failed episode timing and metric persistence, exact initial-pose overrides, and the real shell's
multiworker argument handling and sequential export using mock child processes.

Audit a successful **live** run and its export:

```bash
python baseline/validate_repeat.py --run-dir /path/to/raw/run_... \
  --export-dir /path/to/export --output /path/to/validation.json
```

This checks all cycles and recorded nonzero return actions, source 10 Hz cadence,
canonical state/actions, camera-frame uniqueness and files, source-soil transfer,
v2 metric persistence and continuous cuts, and actual MP4 frame counts against Parquet. Live validation results for this
checkout are recorded in `VALIDATION.md`.
