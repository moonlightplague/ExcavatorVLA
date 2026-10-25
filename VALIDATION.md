# Validation — 2026-10-25

Local runtime: Isaac Sim `6.1.0-rc.26`, RTX 4090 GPUs, PhysX particle soil.
Policy/export Python: `ExcavatorVLA-current` (LeRobot 0.4.4).
The optional Newton MPM backend was not included in these live validations.

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
