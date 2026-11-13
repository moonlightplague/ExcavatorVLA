# Ten-dig collection with coarse MPM — 2026-11-12

The latest completed fixed cohort exported **9/16 episodes (56.25%)** with
ten physical digs and returns and at least 50 kg delivered in every dig. Its
95% Wilson interval is **33.18–76.90%**: the observed-rate target passed, but
the stronger confidence criterion failed. A general greater-than-50% rate
guarantee is therefore **not established**. Additional workspace and cut-geometry
tests also failed to establish that guarantee. All experiment timestamps below
are UTC.

## Practical sand-amount recommendation and measured recipe

Use **sand amount 9** as the provisional starting recommendation for ten-dig
coarse collection. It is the only amount tested with the final selected
controller in a complete fixed cohort. Keep the **50 kg minimum on every dig**
and the final retention gate enabled: seven of the 16 scenes were rejected for
real underfilled cuts. Amount alone does not ensure productive late digs in
every scene. The recommendation supplies qualified collected episodes; its
population export rate remains uncertain and it is not an established optimum.

The completed fixed cohort used the following explicit recipe. Generic launcher
defaults are not sufficient to reproduce it. `collect_repeat.sh` requests a
number of successful episodes and may retry; use `benchmark_repeat.py` below
to measure the fraction of all attempts that export.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
EXCAVATOR_MPM_EXECUTION=graph EXCAVATOR_MPM_DISPLAY=rocks \
EXCAVATOR_MPM_GRID_CAPACITY=16384 POLICY_PYTHON="$PWD/.venv-pi05/bin/python" \
./collect_repeat.sh 10 4 0,1 10 \
  --soil-backend newton_mpm --mpm-profile coarse \
  --scene-profile reachable --acceptance-profile coarse_mass \
  --sand-amount 9 --mpm-hz 120 --mpm-particle-budget 12000 \
  --min-dig-height .12 --min-curl-height 1.4 \
  --min-delivered-kg 50 --min-retained-delivery-ratio .8 \
  --bed-wall-margin 1.0 --dump-settle-seconds 1.5 --release-clearance .45 \
  --width 512 --height 512
```

The four-worker recipe uses two simulator processes per RTX 4090. With one
GPU, use two workers and its single GPU ID. The recorded images remain
256 × 256. Keep the sparse-grid allocation explicit: its capacity is distinct
from the 12,000 material-point budget and the 20 cm physics grid.
The tested environment left `EXCAVATOR_MPM_VOXEL_SIZE` and
`EXCAVATOR_MPM_PARTICLE_SPACING` unset, using the coarse profile defaults.

For a fresh fixed-attempt confirmation, use a new output directory and a seed
range that has not been used for development:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
EXCAVATOR_MPM_EXECUTION=graph EXCAVATOR_MPM_DISPLAY=rocks \
EXCAVATOR_MPM_GRID_CAPACITY=16384 \
.venv-pi05/bin/python baseline/benchmark_repeat.py \
  --output outputs/repeat_coarse_confirmation --attempts 16 \
  --gpu-ids 0,1 --workers-per-gpu 2 --seed-start 97000 \
  --policy-python "$PWD/.venv-pi05/bin/python" \
  --require-target --require-confidence \
  --acceptance-profile coarse_mass --mpm-profile coarse \
  --sand-amount 9 --mpm-hz 120 --mpm-particle-budget 12000 \
  --min-dig-height .12 --min-curl-height 1.4 \
  --min-delivered-kg 50 --min-retained-delivery-ratio .8
```

The two exit criteria distinguish an observed rate above 50% from evidence
whose 95% Wilson lower bound exceeds 50%. Neither is a promise that every
future collection batch will achieve that rate. Exported episodes must always
pass the per-dig mass gate.

To reproduce the reported cohort, use `--seed-start 96000` with a fresh output
directory. The example range 97000–97015 was reserved but never executed.

## Scope and compatibility

`collect_repeat.py` loads the shared Isaac `newton_mpm_soil` adapter. The latest
adapter defaults to the `coarse` profile, CUDA graph execution and irregular
rock rendering. Select this model as `--soil-backend newton_mpm
--mpm-profile coarse`; `coarse_mpm` is not a separate backend name.

The previous 2/4 export result in [VALIDATION.md](VALIDATION.md) used fine MPM
with 31,904 material points and a roughly 12 cm grid. That result does not
validate the latest coarse configuration. A simulation success alone is also
insufficient: a counted success requires a complete, validated standard/VLA
publication, all ten productive cycles and all ten physical returns.

## Compatibility failures found

- The original 1,000-point preflight cutoff discarded the entire planning
  snapshot after coarse material spread outside the authored footprint. One
  test still had 618 points there, about 3.3 tonnes, but was reported exhausted.
  Coarse search/preflight thresholds now preserve the mass threshold of the
  previous fine/32k configuration, rounding conservatively to whole points.
  The reference and resolved thresholds are recorded in metadata. This first
  compatibility correction retained the existing acceptance policy; the later
  opt-in mass policy is defined separately below.
- A fixed point budget can increase the actual grid spacing. With a 6,000-point
  budget, amounts 3 and 5 resolved to 21.8 and 23.1 cm grids. A 12,000-point
  budget retained the requested 20 cm grid and 15 cm point spacing in the tests.
- The original candidate order prioritized inner rings, leading to successive
  small cuts in depleted areas. Coarse planning now prioritizes current swept
  material mass among the upstream feasible candidates; spatial history,
  full IK, floor and collision checks remain active.
- The stock swept estimate includes material centers up to 20 cm below the
  blade target. This can reward a dense ground sheet that the blade cannot
  collect. Isolated above-blade and increased-curl-clearance tests still rejected
  amount 5 on dig five and amount 9 on dig six; this correction alone is
  insufficient.
- Amount 7 overflowed the default 4,096-cell reusable sparse grid at creation.
  An explicit 8,192-cell allocation resolved that failure. Allocation changes
  preserve the 20 cm physics grid; they do not change the material model.
- Source identity correctly includes material that avalanches out of the
  authored pile box, but the original planning snapshot and target-region gate
  still exclude that material. An isolated measured-source-footprint test now
  searches actual Newton birth material within 3 m of the generation center,
  excluding particles in the bucket and truck. It reapplies the existing static
  settle checks and requires real local material; no particles are moved or
  added. The first amount-5 snapshot contains 6,082 eligible source points.

## Completed calibration attempts

These are development tests of seed **91000**, not a population-rate estimate.
All rejections are retained; successful prefixes are not exported.

| Configuration | Measured outcome |
|---|---|
| Amount 3; 6k points; 60 Hz; original planner | Rejected after one cycle; false snapshot exhaustion; 5/46 captured points delivered |
| Amount 5; 6k points; 60 Hz; original planner | No first plan; footprint below the 1,000-point snapshot cutoff |
| Amount 3; 12k budget; 120 Hz; mass-aware planning | Rejected on dig 7; new deliveries `[68, 36, 34, 8, 12, 36, 1]` |
| Amount 5; 12k budget; 120 Hz; mass-aware planning | Rejected on dig 9; new deliveries `[76, 37, 45, 46, 26, 21, 14, 5, 2]` |
| Amount 7; 12k budget; 120 Hz; 4,096 grid cells | Sparse-grid overflow before collection; no export |
| Amount 7; 12k budget; 120 Hz; 8,192 cells; ring order | Rejected on dig 7; four-point capture |
| Amount 5; 12k budget; 120 Hz; mass-priority order | Rejected on dig 6; three points loaded and staged lift unreachable |
| Amount 5; above-blade estimate; 1.4 m minimum curl | Rejected on dig 5; four-point capture |
| Amount 9; above-blade estimate; 1.4 m minimum curl | Rejected on dig 6; new deliveries `[85, 65, 60, 79, 38, 1]` |
| Amount 9; above-blade estimate; 5 cm extra planning headroom | Six strict-quality digs, then no next reachable plan |
| Amount 9; explicit 16 cm coarse grid; 24k budget; 5 cm planning headroom | Four strict-quality digs, then staged-lift failure on dig 5 |
| Amount 5; expanded actual-source snapshot and target region only | One strict-quality dig, then contact query rejected outside the old source box |
| Amount 9; expanded actual-source snapshot and target region only | One strict-quality dig, then contact query rejected outside the old source box |
| Amount 5; actual-source contact surface, expanded search, loaded pickup window | Six strict-quality digs; dig 7 captured two points and delivered none; no export |
| Amount 7; integrated current-source planner and 50 kg gate | All ten cycles/returns and original episode efficiency passed; rejected solely for 32.44 kg on dig 9; no export |
| Amount 9; actual-source contact surface, expanded search, loaded pickup window | All ten cycles/returns exported and independently validated; every load ≥50 kg |

The mass-priority amount-5 test delivered 76, 70 and 64 points on its first
three rounds, approximately 411, 379 and 346 kg. Its later rejection shows why
early payloads alone cannot validate a ten-dig recipe.

The 5 cm headroom experiments retain the existing 5 mm hard geometry floor
limit and all path collision checks. The 16 cm experiment is explicitly a
resolution override; its results must not be presented as default 20 cm MPM.

## Sand-amount interpretation

### Completed ten-dig development export

With the actual-source surface/query correction, amount **9** completed seed
91000 and exported **2,610 frames per camera**, with ten recorded physical
returns. An independent audit validated the publication and decoded 19 samples
at cycle boundaries through the real `LeRobotDataset` reader: 28D state, 4D
actions, three RGB cameras of 256 × 256 at 10 Hz. The final Newton diagnostics
remained healthy, coarse, 20 cm grid / 15 cm spacing, with 10,572 material points.

| Dig | Amount 7 delivered kg | Amount 9 delivered kg |
|---|---:|---:|
| 1 | 427.1 | 399.5 |
| 2 | 346.0 | 314.3 |
| 3 | 356.8 | 287.7 |
| 4 | 221.7 | 442.1 |
| 5 | 140.6 | 218.4 |
| 6 | 243.3 | 239.7 |
| 7 | 205.5 | 394.2 |
| 8 | 81.1 | 309.0 |
| 9 | **32.4 — rejected** | 79.9 |
| 10 | 216.3 | 53.3 |

Every amount-9 dig exceeded the requested 50 kg minimum. That pilot preceded activation
of the new final mass gate; the independent audit recomputed that stricter gate
from its retained measurements. The confirmation uses the active gate.

The existing episode policy passed: mean source score 88.18, weighted spill
7.22%, final unique-source spill 12.12%. Digs 9 and 10 failed the legacy strict
per-cycle efficiency rule, so this is **not** a strict-equivalent episode.
No efficiency threshold was relaxed during this comparison.

The ten cycle-local deliveries totaled 2,738 kg; the final unique-source load
remaining in the truck was 2,588.9 kg. Recording through completion took 1,504.7 s
(25.1 min); preparation plus recording took 1,597.1 s, excluding application
startup. MPM physics accounted for 468.4 s. The first six cycles' execution
median was 130.1 s at amount 9 versus 123.7 s at amount 5, a 5.2% increase in
this single-seed, two-GPU comparison. These are execution timings, excluding
inter-cycle planning, and do not establish an optimal amount or throughput.

Amount 5 failed with zero delivery on dig 7. Amount 7 completed all ten physical
cycles/returns and passed its source-camera audit and the original episode
efficiency gate (mean 88.17, weighted spill 4.11%, final spill 7.99%). The active
mass gate rejected it **solely** for 32.44 kg on dig 9, preventing publication.
The next dig recovered to 216 kg, showing why a total or average is insufficient.
Its wall timing overlaps other simulator trials and is not a fair isolated
performance comparison. At this pilot stage, amount 9 remained a candidate:
its 53.3 kg final load left little margin. The subsequent fixed-cohort result
below supports a provisional recommendation with a wide confidence interval.

### Source size

The source generator maps the amount to reference height as
`min(6.5 m, 3 m × amount^0.35)`. Amount 9 requests about 6.47 m, close to the
height cap. This is the generation reference, not the measured settled pile
height. More sand alone did not cure the planning failures above.

At a fixed 20 cm grid and 15 cm point spacing in seed 91000, amounts 5 and 9
used 8,581 and 10,572 material points. Amount 9 therefore adds about 23% more
material points.
Amount 7 used 9,677 points; amount 9 adds only 9.2% over that intermediate size.

| CLI sand amount | Material points | Initial modeled source mass |
|---:|---:|---:|
| 3 | 7,161 | 29.06 t |
| 5 | 8,581 | 46.42 t |
| 7 | 9,677 | 52.32 t |
| 9 | 10,572 | 56.32 t |

These measurements use the same development seed and resolution. Amount is
a generator multiplier, not kilograms or a linear mass setting. Much of the
initial source can settle outside a reachable working face; total source mass
does not guarantee a useful late cut. At amount 9, each measured point is about
5.33 kg: nine delivered points are only about 47.9 kg and must be rejected;
ten are about 53.3 kg. The gate uses the actual mass in each scene rather than
rounding a small load up to the requested minimum. Amount 9 is the only amount
in the completed cohort with the final selected controller; comparisons against
smaller amounts used earlier controller revisions and do not establish a
global performance optimum.

The user specified **at least 50 kg per dig**. Coarse collection now defaults to
`--min-delivered-kg 50`. A separate hard gate checks the measured new-source
delivery of every cycle after its dump and physical return. The streaming
exporter and independent episode validator recompute that gate from real
material-point counts and mass, reject missing measurements, and retain the
report after raw pruning. Early loads cannot compensate for a small late load.

## Incomplete confirmation

An eight-scene batch, assigned seeds 92000–92007 before execution, was stopped
after the matching calibration failed on dig six. Five episodes had started,
including a rejected episode at seed 92002; the remaining episodes were partial
or unstarted. This is an abandoned, incomplete experiment. It supplies no
fixed-cohort success-rate estimate or confidence claim.

The next confirmation must use fresh seeds and a recipe that first completes
ten productive digs. Planning failures, quality rejections and export failures
must remain in the denominator. Benchmark v2 uses a strict `rate > 0.5` test;
exactly 50% is not a pass. Confidence intervals describe finite experimental
evidence, not a guarantee for all future randomized scenes.

A later amount-9 batch assigned seeds 93000–93007 was also stopped, with four
partial first episodes and zero exports. A simulator-free regression probe
demonstrated that the old episode's cut history, index and candidate cache
survived the next-attempt wrapper, allowing them to affect the next scene's
first plan. The reset is now performed before scene preparation, alongside
source-ID reset. This incomplete batch supplies no rate estimate. No runtime
source was changed until its workers had stopped.

## Completed fixed cohort: legacy episode policy plus 50 kg per dig

After the complete amount-9 export and independent reader audit, eight fresh
seeds **94000–94007** were assigned before execution after the complete boundary
reset passed the full 102-test baseline suite, with four workers across
two RTX 4090 GPUs. The frozen recipe is amount 9, coarse profile, graph
execution, rocks display, 120 Hz MPM, 12,000-point cap, 16,384 sparse-grid
capacity, 0.12 m minimum dig height, 1.4 m minimum curl height and a measured
50 kg delivery minimum in **every** cycle. Existing episode efficiency,
physical execution, return and source-camera audits remain enabled.

All eight scene attempts stay in the denominator, including plans that fail
before recording and failures of the stricter mass gate or exporter. No seed
replacement is allowed. The predeclared stronger criterion requires the 95%
Wilson lower bound to exceed 50% as well as the observed rate. The amount-7
development trial overlaps initially on GPU 0; its resource contention remains
included in elapsed-time measurements and is not grounds for dropping failures.

All eight original attempts completed: **2/8 exported (25%)**, with a 95% Wilson
interval of **7.15–59.07%**. Both the observed-rate and confidence criteria
failed. Five scenes completed all ten physical cycles; three of those were
rejected for underfilled digs. Three other scenes stopped on failed digs.

| Seed | Result | Underfilled delivered load / stopping point |
|---|---|---|
| 94000 | Rejected | Dig 9: 10.65 kg; legacy lift/delivery gate |
| 94001 | Rejected | Dig 7: 21.30 kg; legacy lift gate |
| 94002 | Rejected after ten returns | Dig 8: 26.63 kg |
| 94003 | Exported | Minimum over ten: 117.19 kg |
| 94004 | Exported | Minimum over ten: 90.49 kg |
| 94005 | Rejected after ten returns | Dig 8: 47.94 kg |
| 94006 | Rejected after ten returns | Digs 7/8: 42.62/26.64 kg |
| 94007 | Rejected | Dig 9: 15.97 kg; legacy capture/lift gate |

An interruption terminated the benchmark parent while its original simulator
workers continued. A detached recovery supervisor recorded all eight original
outcomes without rerunning or replacing seeds, but could not recover their
process exit codes. The recovered result explicitly marks infrastructure
certification false. Runtime source hashes remained unchanged for the cohort.
The 25% observed rate is a descriptive result, with no process-exit certification.

Independent validation of both actual exports passed: **5,367 frames per
camera**, 525 recorded return frames, and 36 cycle-boundary samples decoded
through the real LeRobot reader, with the canonical 10 Hz / 28D state / 4D
action / three 256 × 256 RGB camera contract. Elapsed cohort time was 4,230.6 s,
about 1.70 qualifying exports/hour across two GPUs. Background development
work overlapped, so this is not isolated throughput.

Height-priority development replays also failed: seed 94000 stopped on dig 8;
seed 94006 stopped on dig 10. Lowering the curl height to 1.05 m completed seed
94000 physically, but delivered only 42.61/26.63 kg on digs 8/9. Extending pickup
reach to 9 m stopped at a staged secure-pose failure on dig 7. These are failed
development probes, not a new cohort or an accepted recipe.

## Opt-in mass policy and controller-unit correction

The user's revised instruction permits replacing the difficult small-grain
episode score for coarse MPM. `--acceptance-profile coarse_mass` therefore
grades outcomes in kilograms:

- All requested physical dig/carry/dump/return cycles must complete, without
  freezes or execution failures; source and camera audits remain mandatory.
- Every dig must deliver at least **50 kg** of new source material, measured
  after its dump and physical return. Large early deposits cannot offset a
  small later deposit.
- At least **80%** of the cumulative cycle-local delivery must remain in the
  truck at episode end. For ten minimally qualifying 50 kg deposits this also
  requires at least **400 kg** in the final truck load.
- Raw-point capture/lift/mean-score cutoffs are retained as comparison verdicts,
  rather than acceptance vetoes. The streaming exporter and independent
  validator recompute the mass verdict and require the real, healthy coarse
  model and matching material-point mass.

The controller also contained native-grain count thresholds for contact
progress, advancing the cut, recognizing a loaded bucket, and securing/lifting
material. An 80-point loaded threshold represented about 426 kg in these
coarse scenes. In the mass profile, those thresholds preserve their authored
native-source mass using the actual runtime grain mass. Minimum counts round
up and maximum spill counts round down. Angular, gravity and collision limits
are preserved; original and resolved counts are recorded in episode metadata.

Production-code development replays with this policy/control correction did
not meet the requested floor:

| Amount | Seed | Result |
|---|---:|---|
| 9 | 94000 | Rejected dig 7: 10.65 kg |
| 9 | 94006 | Rejected dig 5: 26.64 kg |
| 7 | 94000 | Rejected dig 8: 27.03 kg |
| 5 | 94006 | Rejected dig 7: 48.63 kg |

The seed-94006 amount-9 failure captured seven points and delivered five, with
zero execution freezes and complete physical return. It is a real underfilled
cut; changing its point-count score cannot qualify it. These are development
replays, not a population-rate estimate.

Further planning experiments retain the full fast-feasible candidate pool
before ranking and deduplicate identical target XYZ values created by depth
clamping. One initial snapshot had 688 feasible rows, while the inherited
per-ring selector passed only 96 to the new ranking. A first full-list mass
experiment consumed its early failure budget on repeated clamped targets;
the deduplicated variant tests that issue separately. Ring grouping preserves
input order, but the native per-group final cost selection can choose among
several candidates; the strict-priority development variant tries one at a time.

A saved late snapshot from the failed height-order replay at seed 94000 had
about **53.3 kg** of source material above 0.3 m inside the 7.2–8.4 m pickup
window, versus **783 kg** inside 7.2–9.0 m. These are inventory estimates based
on measured material centers and the real swing center, not expected bucket
deliveries. Wider-window development tests retain all joint, gravity, IK and
collision checks. Neither search variant is yet a validated rate recommendation.

The full-candidate, strict-priority development replay at seed **94006**
completed all ten digs and exported under `coarse_mass`. Loads were
`[261.0, 261.0, 207.8, 58.6, 106.5, 186.5, 69.3, 69.3, 191.8, 79.9]` kg.
Final truck mass was 1,486.3 kg, retaining 99.64% of cumulative delivery.
Independent validation passed for all **2,421 frames per camera**, 260 recorded
return frames, and 16 decoded boundary samples through the real LeRobot reader.
This single replay is development evidence, not an export-rate estimate.

That snapshot also contained about **4,964 kg** above 0.3 m at real swing radii
4.5–7.2 m, which the calibration window had excluded. This narrow pickup
window was introduced during this task's development; it is not a physical
limit imposed by the shared coarse model. Additional tests admit targets at
4.5–9.0 m and let the existing IK, gravity, floor and collision checks determine
feasibility. The outer-window-only seed-94006 test stopped after four completed
cycles when no loaded lift pose satisfied the gravity check; that remains a
physical planning failure, with no export.

The closer-source seed-94006 test completed five returns but rejected dig 5 at
21.31 kg. An environment interruption left the seed-94000 deduplicated search,
outer-window and closer-source replays incomplete after 7, 6 and 4 productive
cycles respectively, and the isolated combined-search replay incomplete after
two productive cycles. No simulator processes survived the resumption. These
partial trajectories are not complete tests or exports; none is included in a
population-rate estimate.

## Fresh fixed validation: complete unique mass priority

The selected candidate keeps the successful full-candidate approach, removes
duplicate clamped target XYZ values, and retains the **7.2–8.4 m** pickup
window. The native planner still performs all full IK, pre-sample, floor,
gravity and collision checks. Marked coarse candidates are tried individually
in mass order; fine MPM and PhysX retain native grouping.

Eight fresh seeds **95000–95007** were assigned before execution on two RTX
4090 GPUs, with two independent workers per GPU and two attempts per worker.
The recipe is coarse 20 cm / 15 cm, graph execution, rocks rendering, amount 9,
120 Hz, 12,000-point cap, 16,384 sparse-grid capacity, 0.12 m minimum dig height,
1.4 m minimum curl height, and `coarse_mass` acceptance: every dig ≥50 kg and
final deposit retention ≥80%. All attempts remain in the denominator, with no
seed replacement. Runtime sources are frozen and hashed, including the native
`auto_dataset_collect.py` planner. The predeclared criteria are an observed
export rate strictly above 50% and a 95% Wilson lower bound above 50%.

This cohort completed all eight original attempts with **3/8 exports (37.5%)**.
The observed-rate and confidence criteria both failed. All four worker exit
codes were zero and infrastructure validation passed. The 95% Wilson interval
was **13.68–69.43%**.

| Seed | Result | Minimum / rejection |
|---|---|---|
| 95000 | Rejected | Dig 8: 15.98 kg |
| 95001 | Exported | Every dig ≥50 kg |
| 95002 | Rejected | Dig 9: 42.59 kg |
| 95003 | Rejected | Dig 10: 42.61 kg |
| 95004 | Exported | Minimum 58.58 kg; last three 79.88, 58.58, 127.81 kg |
| 95005 | Rejected | Dig 7: 47.92 kg |
| 95006 | Rejected | Dig 7: 26.62 kg |
| 95007 | Exported | Every dig ≥50 kg |

Independent validation passed for all three exports: **7,470 frames per
camera**, 823 physical-return frames, and 50 decoded boundary samples through
the real LeRobot reader. Elapsed time was 3,480.45 s, about 3.10 qualifying
exports/hour across two GPUs. After completed workers released their slots,
development replays started on GPU 0 at 14:29:47 UTC and GPU 1 at 14:33:06 UTC.
Their overlap remains in wall time; concurrency stayed at two heavy simulators
per GPU. This timing is not an isolated throughput comparison.

## Checked-cut continuation: selected control correction

The native progress detector can advance a cut after about 20 kg of bucket
progress in these coarse scenes. With a required 50 kg final delivery, this
can terminate the checked stroke before enough material enters the bucket.
The selected controller policy, `checked_cut_continue_until_load_goal_v1`,
requires **100 kg currently in the bucket** before accepting native early
advance during `insert_cut` or `pull_mid_cut`. A completed planned endpoint
can still finish below this internal goal. Final acceptance remains **50 kg
delivered per dig**. Curl, lift, return, joint, gravity, ground and collision
decisions continue through their existing checks.

On a development replay of failed seed **95000**, this correction completed
and exported ten digs with loads
`[223.7, 154.5, 213.1, 63.9, 143.8, 133.2, 229.0, 63.9, 186.4, 79.9]` kg.
Final truck mass was 1,464.8 kg, retaining 98.21% of cumulative deposits. All
ten legacy strict scoop decisions also passed. Independent audit passed for
**2,516 frames per camera**, 274 return frames and 17 real-reader boundary
samples. This replay is development evidence, not an independent rate estimate.

Two other development variants were rejected: admitting closer source at
4.5–8.4 m completed nine productive digs but found no reachable tenth plan;
also capping the native bucket-advance threshold at 100 kg failed dig 5 at
47.94 kg. Those variants are not selected. The selected pickup window remains
7.2–8.4 m, with full candidate retention and target deduplication.

## Completed independent 16-scene validation

Before execution, seeds **96000–96015** were assigned to four workers across
two RTX 4090 GPUs, four attempts per worker. The selected checked-cut
continuation controller uses the same amount-9 coarse recipe and acceptance
policy above. All 16 attempts remain in the denominator; no seed replacement
or runtime source changes are allowed. The observed export rate and its 95%
Wilson lower bound must both exceed 50%. All 16 assigned attempts completed
with all four worker exit codes zero and unchanged runtime source hashes.
Nine episodes exported: the observed-rate criterion passed, while the 95%
lower-bound criterion failed. The confidence-required benchmark exited with
the unmet-target status; no failures or seeds were removed.

The protocol records SHA-256 hashes for 22 runtime Python files, including
the native soil adapter and candidate planner. The canonical hash-set digest
is `5f64c3ed1d9986d4842dcb69691fad32cbaaa37d853b08b78bfa1369e23ef545`
(SHA-256 of the `source_sha256` object serialized with sorted keys and compact
JSON separators). The shared Isaac checkout is
`c27e350af9f9d0a6d1a5571f2ca45e7b62af5cbc`.

| Seed | Result | Minimum / stopping load | Last three deliveries (kg) |
|---|---|---:|---|
| 96000 | Rejected dig 9 | 37.27 kg | — |
| 96001 | Rejected dig 10 | 10.65 kg | — |
| 96002 | Rejected dig 9 | 21.29 kg | — |
| 96003 | Rejected dig 10 | 15.98 kg | — |
| 96004 | Rejected dig 10 | 15.97 kg | — |
| 96005 | Exported | 95.84 kg | 266.21, 95.84, 298.16 |
| 96006 | Exported | 79.88 kg | 266.26, 234.31, 149.10 |
| 96007 | Exported | 90.54 kg | 287.59, 218.36, 175.75 |
| 96008 | Exported | 106.55 kg | 255.72, 106.55, 117.21 |
| 96009 | Exported | 106.51 kg | 303.56, 244.98, 159.77 |
| 96010 | Exported | 127.86 kg | 223.76, 133.19, 149.18 |
| 96011 | Rejected dig 7 | 26.62 kg | — |
| 96012 | Exported | 74.56 kg | 117.17, 197.06, 74.56 |
| 96013 | Exported | 106.53 kg | 170.44, 122.51, 175.77 |
| 96014 | Rejected dig 10 | 10.65 kg | — |
| 96015 | Exported | 79.89 kg | 202.38, 250.32, 79.89 |

Independent validation passed for all **nine exports**, including **22,186
frames per camera**, 2,394 recorded return frames, and **153 decoded cycle-
boundary samples** through the real LeRobot reader. All 90 delivered loads
were at least **74.56 kg**, and the minimum final deposit retention was
**93.15%**. The 10 Hz / 28D state / 4D action / three 256 × 256 RGB contract
passed. A late dig's real frames were also visually inspected in all three
camera views: the coarse grains, excavator and truck were visible.

Elapsed fixed-cohort wall time was **7,171.01 s (119.52 min)**, yielding
**4.52 qualifying exports/hour** across two RTX 4090 GPUs. Median recording-
through-completion time among exports was **1,685.73 s (28.10 min)**; median
preparation-plus-recording time was **1,736.94 s (28.95 min)**. Development
replays began after original workers released GPU slots, at 16:54:48 UTC on
GPU 0 and 17:04:21 UTC on GPU 1. This overlap remains in elapsed time, with
at most two heavy simulators per GPU; the timing is not isolated throughput.

A separate `coarse_workface` development snapshot translated the initial
source bank closer to the arm and admitted pickup radii 4.5–8.4 m. Its seed-
96000 replay delivered 335.53 kg on the first dig but found no safe second
plan; full IK and mesh-floor checks rejected it. It is not selected for
production or an independent rate estimate. Its conditional fresh eight-scene
confirmation was never started.

Further development retained the same physical model, amount 9, 120 Hz,
20 cm grid / 15 cm spacing and 50 kg delivery gate. None of these variants is
included in the production recipe or pooled into the 16-scene rate:

| Development variant | Seed | Completed outcome |
|---|---:|---|
| Closer source bank; pickup 4.5–8.4 m | 96000 | First delivery 335.53 kg; no safe second plan |
| Same closer-bank variant | 96002 | First delivery 500.8 kg; no safe second plan |
| Open pull ≥0.65 m, closed escape ≥0.85 m; 5 cm planning headroom | 96000 | First delivery 202.33 kg; no safe second plan |
| Same longer stroke | 96002 | No initial plan; mesh-floor guard rejected the cut |
| Longer stroke with 12 cm planning headroom | 96002 | Ten-dig export; minimum 90.50 kg |
| Same extra-headroom variant | 96000 | Rejected dig 3: 37.27 kg |
| Closer bank with pickup narrowed to 6.4–8.4 m | 96000 | Rejected dig 8: 0 kg |
| Same narrowed-window variant | 96002 | Rejected dig 7: 20.03 kg |

The successful extra-headroom replay passed independent audit: **2,464 frames
per camera**, 263 return frames and 18 decoded boundary samples. It retained
99.66% of cumulative delivery and passed all ten legacy strict scoop checks.
The paired rejection prevents treating that development result as a robust
replacement recipe. Increasing planning headroom did not change the 5 mm hard
mesh-floor guard or any collision limit.

The longer stroke tested a concrete mismatch: the swept-density estimate spans
76 cm, while the native mid-pull defaults to 35 cm. Both loading travel and
closed escape were fully planned and checked. The tests show that increasing
that travel alone does not guarantee sufficient delivered mass.

A diagnostic replay of the first closer-bank variant recorded 24 failed
second-plan attempts, all at measured swing radii **5.352–6.048 m**. Its first
successful target was at 6.900 m. Excluding those close candidates allowed the
narrowed-window replays to pass the second-plan blocker, but both later failed
the actual delivered-mass gate. One earlier diagnostic wrapper failed before
producing a usable trial and was corrected; it supplies no simulation or rate
evidence. The reserved fresh confirmation seeds **97000–97015 were never
executed**, because the development prerequisites failed.

No tested configuration established the predeclared 95% lower-bound target.
Underfilled episodes remain rejected rather than being exported as successful
prefixes. The selected recipe, recommendation and uncertainty at the top of
this report reflect the completed 16-scene test. Temporary datasets, diagnostic
outputs and evaluation JSON files were removed after this report was recorded:
**62,811 task files, 15.06 GB apparent size**, plus local baseline bytecode
caches. The original shared coarse experiment artifacts remain ignored.

## Regression validation

196 CPU checks passed: 122 baseline, 26 Isaac export/append, 32 deployment
contracts, 10 training-config and six collection/evaluation scene tests.
They cover lazy soil initialization, mass-equivalent planning, requested/active
profile mismatches, preservation of feasible targets, the strict rate boundary,
worker allocation, physical-return supervision and canonical dataset shapes.
Dataset audits used Python 3.12, LeRobot 0.4.4, Torch 2.10 and PyAV decoding.
