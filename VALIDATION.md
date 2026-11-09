# Validation — 2026-11-09: ten-cycle Newton MPM

The original two worker runs in `data/repeat/raw` had no trainable export:
seven planning failures and three rejected recorded episodes across ten
attempts. Their camera audits passed. The original data was retained.

The tuned controller uses the installed Isaac Sim `6.1.0-rc.26` / bundled
Newton `1.5.0` implicit MPM implementation. This work repairs its local
static-mesh collision adapter and collection setup; it does not upgrade the
Newton library. See [TUNING.md](TUNING.md) for parameters and policy definitions.

## Fixed-attempt protocol

Four consecutive scene seeds, `80000–80003`, were assigned before execution:
two attempts on each RTX 4090, with disjoint seed ranges. The benchmark bypasses
success-target shell retries. Planning failures, empty later cuts and rejected
episodes remain in the denominator. Workspace probes and aborted development
pilots do not enter this cohort.

The fixed settings are Newton MPM, ten cycles, sand amount 3, reachable side
loading at a 6.7–7.0 m measured-swing radius, 1 m bed inset, 0.45 m minimum release
clearance, 1.5 s dump settling, 60 Hz MPM, a 32,000-point budget and a 512 × 512
headless viewport. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`; one worker per GPU.
The material resolved to 31,904 real points with approximately 9.09 cm spacing
and a 12.12 cm grid. Source RGB and exported videos remain 256 × 256.

The benchmark runs the explicit `episode` policy and retains each original
strict decision. This allows the completed strict-equivalent episode count to
be reported separately. All capture/delivery, execution, physical-return and
source-camera checks remain hard requirements.
The strict comparison uses the original thresholds with the corrected Newton
source-material identity; it does not replay the entire old sampler/tracker.

## Live results

All four assigned attempts completed. **Two episodes were accepted and exported:
2/4 = 50%**, including both rejections in the denominator. The benchmark reported
`complete=true`, `target_met=true`, with both worker processes exiting normally.
The original strict per-cycle gate would accept 1/4 (25%) under the corrected
source identity. The selected default is therefore the explicit episode policy.

| Seed | Outcome | Productive scoops / physical returns | Final source payload | Final source spillage | Exported frames |
|---|---|---:|---:|---:|---:|
| 80000 | Rejected: empty tenth cut | 9 / 10 | 1,928 points / 1,754.89 kg | 1.08% | 0 |
| 80001 | Rejected: secure-carry failure on round nine | 8 / 8 | 1,460 points / 1,328.90 kg | 6.29% | 0 |
| 80002 | Accepted and exported; all strict checks pass | 10 / 10 | 1,730 points / 1,574.81 kg | 2.59% | 2,509 |
| 80003 | Accepted and exported; 9/10 original strict checks pass | 10 / 10 | 1,931 points / 1,757.41 kg | 3.93% | 2,420 |

Seed 80000 revisited the same center target on cuts one, five and ten. The last
cut captured zero material despite completing its motion and final return.
Its rejection is preserved; a nine-scoop prefix is not published. Seed 80002
delivered source material on every round, including a 19-point capture / 17-point
deposit on round nine and a 76-point capture / deposit on round ten. Its mean
source score was 92.05 and final unique-ID spillage 2.59%.

Seed 80001 could not find a retaining secure-carry pose after the ninth cut,
with 29 particles still loaded. This execution failure remains a rejection.
Its earlier seventh round illustrates the accounting correction: 25 new IDs
arrived, while net bed count increased by only 15 as older material left.
Those older losses remain in the final unique-ID ledger.

Seed 80003 completed ten productive cuts and returns. Its fifth round captured
14 source points and delivered nine, but its original score of 55.3 failed the
strict 70-point threshold. The full episode passed with mean source score
90.16, capture-weighted spillage 3.08%, and final unique-ID spillage 3.93%.
Thus the acceptance improvement includes an explicit relaxation of per-cycle
efficiency, not a relaxation of execution, positive delivery, or fixed length.

The 95% Wilson interval for 2/4 is approximately **15.0–85.0%**. This reaches the
requested observed rate on the tested setup; four scenes do not establish a
population rate of at least 50%. The settings are the best completely validated
setup from this investigation, rather than a demonstrated global optimum.
Successful episodes took about 40–41 minutes of wall time each on an RTX 4090,
representing approximately four minutes of simulated digging and returns.

The first successful export passed `baseline/validate_repeat.py`: ten scoops,
ten physical returns, original physical uniform 10 Hz source audit, finite
28D states / 4D actions, 2,509 frames in each of three RGB videos and 275 explicit
physical-return frames. Export validation ran after raw pruning and verified
retained quality/metric reports and the committed export receipt.
The final combined publication also passed: **two episodes, 4,929 frames in
each of three videos, and 546 physical-return frames**, with matching Parquet,
canonical dimensions, all ten cycles per accepted episode, and both committed
export/cleanup receipts.

The actual LeRobot 0.4.4 reader also decoded frames 0, 207 (a physical return),
1,254 and 2,508 from the first export. Every sample had `[28]` state, `[10,4]`
action chunks and three `[3,256,256]` RGB tensors, including terminal padding.
This check used local files and CPU video decoding.
After appending the second episode, the reader also loaded both starts and ends
(indices 0, 2,508, 2,509 and 4,928) with the same shapes. Each terminal sample had
exactly nine padded future actions, preserving the episode boundary. The three
camera views were also visually inspected.

## Regression checks

- 81 baseline tests passed after promoting the final Newton/workspace/episode
  acceptance defaults, including the multiworker shell and export/pruning checks.
- 120 Isaac tests passed. The suite's Warp-only test was skipped in policy
  Python and passed separately with the real bundled Warp on CPU and CUDA.
- The swept-collision test crosses an entire 4 cm floor shell within one step;
  it verifies first-impact separation while preserving misses and outward motion.
- All 72 simulator-independent SmolVLA tests passed after syncing the identical
  Newton soil source into the deployment adapter.

This is a small conditional test in the narrower reachable workspace. It does
not establish a rate over the old broad scene distribution or other material
settings. Remaining depletion failures must count against acceptance even
when earlier deposits and camera audits pass.

Temporary live datasets, exports, JSON evaluation reports, development logs and
the four reader-test cache entries were removed after recording these results,
as required by the repository's universal skill. The original `data/repeat`
collection was retained. Reproduction commands are in `TUNING.md`.

# Validation — 2026-10-30

The v2 controller completed a live three-scoop episode on its first attempt
with PhysX soil and sand amount 1.0. The run reused scene namespace
`run_20261025_210534_repeat_20261025_210508_595202_r1_w0` and worker identity
`repeat_20261025_210508_595202_r1_w0` with the normal randomized collector.

The old plans contained duplicated 6.5 m clearance waypoints between insertion
and pulling. Rejecting those routes first exposed a small intermediate mesh
floor penetration despite valid endpoints. Reserving 0.10 m at mesh-based target
heights allowed continuous cuts while keeping the existing floor/path checks.
All three cuts in the final run had no clearance detours; all three physical
returns used checked direct paths with zero clearance waypoints.

| Check | Result |
|---|---:|
| Completed scoops / physical returns | 3 / 3 |
| Source / exported frames | 690 |
| Frames in each of three exported videos | 690 |
| Exported physical-return frames | 75 |
| Episode score | 90.0 |
| Strict source-ID payload | 1,559 particles / 561.24 kg |
| Unique scooped source IDs | 1,571 |
| Final retained / spilled source IDs | 0 / 24 |
| Spillage fraction | 1.53% |
| Full episode simulation time, including returns | 68.97 s |
| Mean scoop execution/return time | 22.94 simulation seconds |
| Recorded episode wall runtime / preparation-inclusive attempt | 256.47 / 309.87 s |

The per-episode metrics file, embedded metadata, and run-level metric index
agreed. `baseline/validate_repeat.py` passed the v2 metric/continuous-cut audit,
source clock and images, all returns, canonical 28D state/4D actions, and actual
video frame counts. Strict standard/VLA export also passed. Payload can exceed
observed captured IDs because transient bucket occupancy between observations
is not observed; the shared online evaluation tracker uses the same accounting.
Spillage includes material returned to the pile, unlike the collector's existing
quality-gate spill estimate.

98 automated tests passed: 57 baseline tests, 15 online evaluation metric tests,
15 existing export contract tests, and 11 append tests. They cover failed and
rejected episode reports, unique source identity, repeated scoops, retained
material, overlap exclusion, timing, cut routing, and v1/v2 return compatibility.
Temporary live data, exports and metric files were removed after validation.
This smoke test does not establish success rates across all scenes or soil backends.

# Validation — 2026-10-25

Local runtime: Isaac Sim `6.1.0-rc.26`, RTX 4090 GPUs, PhysX particle soil.
Policy/export Python: `ExcavatorVLA-current` (LeRobot 0.4.4).
The optional Newton MPM backend was not included in these live validations.

## Follow-up: five-scoop collection failure repair

The two runs in `data/repeat/raw` requested five-scoop episodes and exhausted
10 and 15 attempts with zero successes. Their recorded prefixes completed at
most three scoops. This was a controller/planner failure, not an export counter
error. The original collection was retained unchanged.

The repair corrects physics-frame timing in the dump opening loop and rejects
an opening timeout. Bucket predictions retain lateral marker offsets, floor
clearance uses enabled physical collider meshes, and IK refines a target that
violates that clearance. Cut engagement can use actual collider vertices when
internal markers miss the remaining sand; empty-query floor estimates cannot
validate engagement. The existing execution, transfer, spill, and final quality
gates still decide whether the episode is trainable.

Closed escape/curl planning, loaded route continuity, collision sampling, and
fixed-count dense-target retries were also repaired. The retries use the
current pile and preserve physical returns without resets between scoops.

Both original worker identities and scene-seed namespaces were replayed with
sand amount 1.0 and five requested scoops:

| Original worker | Attempts | Trainable episodes | Scoops / returns | Frames | Score |
|---|---:|---:|---:|---:|---:|
| `r1_w1` | 1 | 1 | 5 / 5 | 1,362 | 89.81 |
| `r1_w0` | 2 | 1 | 5 / 5 | 1,401 | 89.28 |

Worker 0's first randomized layout was correctly rejected before recording for
physical robot/truck overlap. Its next layout completed the episode. Worker 1's
later cuts illustrate the engagement correction: the internal tip marker had
zero sand depth, while real collider contact depths were 0.21–0.34 m, and the
last three deposits delivered 208, 147, and 132 source particles to the truck.

Both episodes passed `baseline/validate_repeat.py`: five successful deposits,
five recorded returns, source audit, uniform 10 Hz, finite 28D states/4D actions,
and complete camera images. Their strict standard/VLA exports were appended
into `outputs/repeat_fix/export_repeat`: 2 episodes, 2,763 frames in each of
three RGB videos, and 289 explicit physical-return frames.

78 automated tests passed: 44 baseline tests, 2 dump-motion tests, 5 route
contract tests, 1 IK movement test, 15 export-contract tests, and 11 append tests.
The dump-motion tests exercise the actual coroutine at both 30/60 Hz control
rates and verify that a stalled loaded bucket cannot report completion.

Evidence: `outputs/repeat_fix/validation.json`, `checks.json`, `diagnosis.json`,
`export_validation.log`, and the `physical_contact*/` raw runs and worker logs.
These two replay successes verify the repaired failure cases; they do not
establish a collection success rate across all randomized scenes.

The initial live trials exposed three issues addressed by this implementation:

- The old home return uses direct joint placement with recording disabled.
  The repeat baseline records collision-checked physical returns.
- Narrow central dig sampling repeatedly revisits the first low crater.
  The baseline expands candidate coverage and cut spacing and adds curl posture
  constraints before selecting a motion. The original selector prioritizes a
  bucket-only curl even if its Cartesian waypoint is raised; the optional
  endpoint filter now enforces clearance and the execution-time gravity-closed
  requirement during planning.
- The exporter assumes a same-scoop transition out of unload is a regression.
  Complete, audited cycle bounds now produce an optional physical-return flag;
  unmarked regressions and stale lift/dig stages remain invalid.

## Final three-cycle data and export

The final controller completed a three-cycle episode on the first attempt with
sand amount 1.0 (the default), including the last physical return. The root
shell launcher collected and automatically exported the episode successfully.

| Check | Result |
|---|---:|
| Completed digs / deposits / physical returns | 3 / 3 / 3 |
| Recorded/exported frames | 809 |
| Frames in each of three RGB videos | 809 |
| Explicit physical return frames | 88 |
| Episode score | 90.0 |
| Bed particles after each cycle | 812 / 1,139 / 1,457 |
| Bed coverage after each cycle | 73.3% / 80.0% / 86.7% |
| Bed surface standard deviation after each cycle | 0.170 / 0.253 / 0.252 m |

`baseline/validate_repeat.py` passed raw image, source-clock, transfer, curl
endpoint, physical-return, Parquet, and MP4 checks. The real LeRobot loader
decoded frames from all three returns and the terminal frame; the SmolVLA
configuration validated with `[28]` states, `[10,4]` action chunks, and three
`[3,256,256]` images. Increasing coverage is measured; absolute surface variation
need not decrease with every added load.

Local evidence: `outputs/repeat_validation/filtered_three_amount1/validation.json`,
`lerobot_load_validation.json`, `export/`, and
`raw/run_20261025_120717_repeat_20261025_120653_4113114_r1_w0/episode_000001/`.
The run used `./collect_repeat.sh 1 1 0 3 --sand-amount 1.0 --max-attempts 3`,
with `DATA_ROOT` set to that validation directory and `POLICY_PYTHON` set to the
installed `ExcavatorVLA-current` Python.

## Additional two-cycle data

An end-to-end two-cycle run completed both transfers and both physical returns.
It passed the original quality and 10 Hz source gates, the strict standard/VLA
export checks, and `baseline/validate_repeat.py`:

| Check | Result |
|---|---:|
| Recorded/exported frames | 482 |
| Frames in each of three RGB videos | 482 |
| State / action dimensions | 28 / 4 |
| Explicit physical return frames | 61 |
| Episode score | 79.87 |
| Bed coverage after first / second deposit | 53.3% / 80.0% |
| Bed surface standard deviation after first / second deposit | 0.273 m / 0.244 m |

The real LeRobot loader decoded representative first, middle, return, and final
frames. Images were `[3,256,256]`, states `[28]`, and ten-step action chunks
`[10,4]`. The supplied SmolVLA training configuration passed validation against
the publication; the optional return flag is auxiliary supervision, outside the
policy input. All three cameras were visually inspected. The task text is 56
tokens with the local SmolVLM tokenizer, within the 64-token training limit.

Inspect the local evidence under `outputs/repeat_validation/fresh_two/`:
`validation.json`, `lerobot_load_validation.json`, `cameras_first_last.png`,
`export/`, and the raw run's successful `episode_000003/`. This successful
trial predates the final curl endpoint constraint; its exact controller options
are stored in raw `meta.json`.

## Real append and training-loader check

The two-cycle and final three-cycle runs were appended into a fresh standard
publication. It passed strict export and video checks with 2 episodes,
1,291 frames per camera, 149 explicit return frames, and 2 task texts. The
LeRobot loader correctly padded a ten-step action chunk at the first episode's
last frame and began the second episode at global frame 482. Evidence:
`outputs/repeat_validation/append/validation.json` and `export/`.

## Automated checks

58 tests passed: 27 baseline tests, 15 existing export-contract tests, 11
existing append tests, 1 IK movement test, and 4 unload-route contract tests.

```bash
python -m unittest discover -s baseline/tests -v
cd ExcavatorVLA-isaac
python -m unittest discover -s tests -p test_lerobot_export_contract.py -v
python -m unittest discover -s tests -p test_export_append.py -v
python -m unittest discover -s tests -p test_ik_movement.py -v
python -m unittest discover -s tests -p test_unload_route_contract.py -v
```

The baseline tests cover geometry, measured fill, cycle/return failures,
episode isolation, initial-pose overrides, curl waypoint planning, shell
multiworker orchestration, raw audits, and explicit-return export constraints.
The two existing export suites cover legacy contracts and transactional append.

Collection still rejects physically invalid randomized scenes and failed digs,
carries, or dumps. These are retained as diagnostics or rejected raw episodes;
successful prefixes of failed repeat episodes are not published. Bed leveling
is measured placement behavior; perfect flatness and a policy success rate are
not claimed from these collection smoke tests.
