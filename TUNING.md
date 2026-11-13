# Ten-cycle Newton MPM repeat collection

This page records the historical **fine-MPM** configuration. Select
`--mpm-profile fine` to reproduce it. The shared adapter now defaults to coarse;
see [COARSE_MPM_VALIDATION.md](COARSE_MPM_VALIDATION.md) for its separate measured
tests and the 50 kg minimum delivered load required in every dig.

This setup targets complete episodes with ten physical dig, carry, dump and
return cycles. Measure acceptance as validated exported episodes divided by
**all scene attempts**, including planning failures and rejected episodes.
`collect_repeat.sh` collects a requested number of successes; use
`benchmark_repeat.py` when measuring the rate.

The final four-attempt cohort exported **2/4 episodes (50%)**; the original strict
thresholds would accept 1/4. Both accepted episodes completed ten productive
rounds and ten physical returns. This is a small test in the reachable workspace;
see [VALIDATION.md](VALIDATION.md) for every outcome and the uncertainty interval.

## Diagnosis

The two original worker runs in `data/repeat/raw` exhausted ten attempts without
an export. Seven failed planning; three recorded episodes were rejected during
their second, second and fifth scoops. Their camera source audits passed. The
exporter had no trainable episode to publish. These original runs were retained.

Several issues became visible in live Newton trials:

- The original reach gate used distance from the world origin. The actual swing
  joint is offset by roughly `(-6.95, 2.23)` m. A world-space polar range did not
  describe the arm's loaded carry workspace. The reachable profile samples from
  the measured swing joint and retains actual dump IK and collision preflight.
- Rear loading puts the long bed axis along the arm's radial reach. Side loading
  makes more inset, low-fill cells feasible without extending the arm.
- Newton source material avalanches outside the authored pile polygon during
  settling. Initial polygon membership discarded most valid material IDs.
  Initial source identity now includes all finite Newton source material except
  material already in the truck or bucket.
- Net truck count is a poor measure of a later scoop's delivery. One pilot
  delivered 95 new source IDs while the total bed count fell by one because
  older material left. Cycle-local source-ID tracking measures new deposits;
  the unique-ID final ledger still counts loss of old deposits.
- The actual truck floor is a closed shell only about 4 cm thick. Newton's
  existing pointwise projection can miss a particle that crosses the whole
  shell within one solve. A swept segment check against the imported static
  triangles now stops an inward crossing at its first impact and applies the
  same 0.5 Coulomb friction used by the static colliders. It uses no substitute
  truck floor. The Newton implicit MPM material and solver remain in use.

A development pilot with sand amount 5, before the final swept-collision repair,
increased forces and lost early transfer quality. The chosen amount for complete
validation is 3. This is not an isolated post-repair comparison or an exhaustive
optimum search. Increasing source volume alone did not fix workspace, accounting
or thin-shell collision failures. Development probes and interrupted pilots are
not part of the final acceptance denominator.

## Parameters

| Parameter | Setting |
|---|---|
| Soil backend | `newton_mpm` |
| Cycles, including final physical return | `10` |
| Scene profile | `reachable` |
| Acceptance profile | `episode` |
| Truck landing radius from measured swing | `6.7–7.0 m`, area-uniform |
| Truck bearing in world XY | `-100° to -80°` |
| Truck alignment | Side approach, `70–85°` to either side of rear alignment |
| Robot yaw | `-15° to +15°` |
| Sand center radius / bearing | World-space `6.0–6.7 m` / `84–96°` |
| Sand amount | `3` |
| MPM solve frequency / point budget | `60 Hz` / `32,000` |
| Bed grid / inset wall margin | `5 × 3` / `1.0 m` |
| Minimum opening-center clearance above wall | `0.45 m` |
| Dump settling | `1.5 simulation seconds` |
| Preferred separation between dig targets | `0.28 m` |
| Minimum dig / curl load-point height | `0.15 m` / `1.05 m` |
| Return planning budget | `12 seconds` |
| Data | Physical uniform `10 Hz`, three RGB cameras, 28D state, 4D action |
| Headless application / recorded RGB size | `512 × 512` / `256 × 256` |

The launcher defaults to these physics/workspace parameters and the explicit
episode efficiency policy described below. `--acceptance-profile strict` restores
the original per-cycle efficiency gate. `RepeatConfig()` keeps its legacy
internal defaults for existing scene-preparation tools; launcher defaults are
defined by `collect_repeat.py` and the root shell script.

The point budget changes quadrature resolution while preserving source mass.
At sand amount 3 it resolved to 31,904 real material points, approximately
9.09 cm point spacing and a 12.12 cm grid. Passive rendering grains do not count
toward capture, delivery or acceptance. The shared soil adapter retains its
120 Hz / 40,000-point defaults outside this baseline; use
`EXCAVATOR_MPM_HZ=60 EXCAVATOR_MPM_MAX_PARTICLES=32000` and sand amount 3
when matching these collection dynamics in deployment or evaluation.

This profile is a narrower scene distribution than `broad`. Its rate applies to
this reachable workspace, with randomized truck, robot and pile placement.
Candidate truck positions also pass the coarse world-space legality bounds
(radius 8.0–10.8 m, bearing -170° to -120°). The geometric collision, floor,
carrying-posture and source-clock checks remain
active. There are no sand resets between cycles and no successful-prefix exports.

## Acceptance policy

The `episode` profile retains these hard checks on every cycle:

- Successful physical execution, no freeze and a recorded collision-checked
  return, including the tenth return.
- At least five captured source material points, five at lift, and three newly
  delivered source points.
- Finite quality measurements and the physical uniform 10 Hz camera audit.

Efficiency is judged across all ten cycles: mean source-ID score at least 65,
capture-weighted spill at most 50%, and final unique-source-ID spillage at most
50%. Final spillage includes material returned to the pile or lost from an older
deposit. Every cycle's original strict outcome is retained in `strict_success`
and `strict_reason`; the original score below 70 or spill above 30% becomes an
efficiency warning rather than immediate rejection.

The source-ID score uses the existing 35/30/15/10/10 weights for capture,
transfer, retention, neutral smoothness and completion, and a 20-times-spill
penalty. Capture saturates at 25 points. Neutral smoothness contributes five
points, matching the original multi-scoop report. `source_score` and the legacy
net-gain `score` are both saved; the source score drives the episode mean.

Use `--acceptance-profile strict` for the original per-cycle efficiency gate.
Both profiles keep execution and physical-return checks. A finite fixed-attempt
benchmark is required to distinguish an actual rate increase from collecting
until enough successes occur.

## Reproduction

Use a Python environment with NumPy, pandas, PyArrow and imageio-ffmpeg as
`POLICY_PYTHON`; Isaac Python remains separate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
POLICY_PYTHON=/path/to/training/python ./collect_repeat.sh 10 2 0,1 10 \
  --soil-backend newton_mpm --scene-profile reachable --acceptance-profile episode \
  --mpm-profile fine --sand-amount 3 --mpm-hz 60 --mpm-particle-budget 32000 \
  --truck-reach-min 6.7 --truck-reach-max 7.0 --truck-approach side \
  --release-clearance .45 --bed-wall-margin 1.0 --dump-settle-seconds 1.5

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /path/to/training/python baseline/benchmark_repeat.py \
  --output outputs/repeat_newton_benchmark --attempts 8 --gpu-ids 0,1 \
  --seed-start 81000 --mpm-profile fine --mpm-hz 60 --mpm-particle-budget 32000
```

The measured run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4` and one worker
per GPU. The three recorded cameras and exported RGB remain 256 × 256; the application
size only reduces the headless viewport allocation.

The benchmark uses disjoint consecutive seed ranges. Failed attempts consume a
seed; fixed-layout replay is rejected. `commands.json` records commands and
source hashes. `results.json` includes acceptance/export rates, a 95% Wilson
interval, failure reasons and the original strict-equivalent episode count.
Validate the resulting publication with `baseline/validate_repeat.py`.

See [VALIDATION.md](VALIDATION.md) for measured outcomes and limitations.
