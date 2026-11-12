# PI05 C repeated digging: experimental bucket-target slew limit

This experiment increases only the bucket **command-target slew rate** from
6 to 20 rad/s. Swing/boom/arm remain 6/2.5/2.5 rad/s. It does not set the
measured mechanical joint velocity or hydraulic speed to 20 rad/s. Legacy
datasets and checkpoints retain `excavator_velocity_v1`; the new explicit
`excavator_bucket_slew20_v1` profile is not a standard-limit benchmark result.
An old checkpoint cannot silently evaluate with the new limits. Expert replay
is explicitly exempt because it executes no model prediction.

The original 100 five-cycle PhysX episodes have 803 target transitions exceeding
the old bucket limit. The largest sampled target change requires 16.1867 rad/s,
while the measured bucket motion at that example is about +0.5299 rad/s. These
are target resets, not evidence of physical motion at 16 rad/s.

`scripts/build_pi05_repeat_relabel.py` publishes a separate candidate at
`../outputs/datasets/pi05_repeat_physx_bucket20_v2`. It never edits the source
publication. Labels are adjacent sampled command-target differences at 10 Hz,
where target = measured position + recorded tracking error. Actions are never
clipped to make the audit pass. Terminal actions are zero. The only observation
change is causal previous action in state indices 12:16; geometry, images,
measured joints, tracking errors, timestamps and cycle boundaries are retained.
Source video containers are hard-linked where possible, otherwise copied
byte-for-byte between filesystems.

The candidate includes all 100 source parents, 122270 frames, verified per-parent
PhysX evidence, and an 80/10/10 parent-disjoint split preserving the earlier
pilot's 20/4/4 parent assignments. Statistics are recomputed from the new
training subset only. Sampled target integration matches within 0.000028 rad
over complete five-cycle episodes; this is **not** physical replay validation.
The deployed six-substep interpolation differs from an instantaneous source
target reset, and physical trajectories/material transfer may still diverge.

The formal-training gate requires three distinct complete five-cycle expert
replays (source episodes 0, 77, 99), PhysX, no early stop at the first scoop,
no clipped actions, the minimum deposit threshold from the archived collection
`quality_gate_config.json` files (3 particles for both source workers), maximum
measured joint discrepancy <=0.15 rad, and command-target discrepancy <=0.001 rad.
The gate retains trace/status checksums. Failure stops the workflow without
starting formal training; raising a bound alone cannot bypass it. This is not
the benchmark's 100-particle performance target. The replay uses net bed-count
change and its own bed geometry, while collection uses cycle-local particle IDs;
that remaining measurement difference is recorded, not claimed to be identical.

From the parent checkout:

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/build_pi05_repeat_relabel.py
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/run_pi05_repeat_history.py \
  --output outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01 \
  --detach
```

For the existing blocked experiment, verified replay traces can be reassessed
without rerunning physics or changing the dataset/split:

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/run_pi05_repeat_history.py \
  --output outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01 \
  --resume-existing --detach
```

The original failed gate and blocked status are archived. Reference checksums,
simulator/adapter/config source hashes, full trace/status checksums, and the
collection configuration evidence are verified before approval. The gate is
recomputed, never manually flipped to passed. Source cycle `sample_end` is an
exclusive observation boundary; the final synthetic zero-action observation
does not count toward the source cycle's payload.

If training exits abnormally, explicitly user-approved shared-GPU restart is:

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/run_pi05_repeat_history.py \
  --output outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01 \
  --resume-training --allow-shared-gpus --detach
```

This resumes the latest complete optimizer/RNG state, not the last logged step.
It archives the previous status/loss log, keeps the same two-GPU effective batch,
and uses zero video-loader subprocesses per rank. Extra recovery saves occur
every 100 steps; only the newest rolling state is kept in addition to the ten
requested 2k milestones. The GPU-sharing launch guard requires 18000 MiB free
per selected GPU but cannot reserve VRAM or guarantee other jobs do not grow.
No other GPU process is terminated. Gradient storage is released after the
parameter-update audit and before validation to reduce transient memory usage.

The November 10 restart commits each complete checkpoint and its latest pointer
**before** running validation. A separate Gloo CPU process group gathers RNG
state and coordinates checkpoint/validation waits, so idle ranks do not leave
NCCL collectives pending while the main rank evaluates. Restart validation uses
eight uniformly spaced frames per held-out episode (80 frames across all ten
validation parents), rather than evaluating the entire long sequence inline.
`validation_selection.json` records the exact selected observation IDs; checkpoint
metrics explicitly declare `partial_evaluation`. This is an offline progress
preview, not full held-out validation or repeated-task closed-loop success.
Optimizer, data addressing, LR schedule and the 20k total budget are unchanged.
If validation fails after a save, the saved step remains resumable.

After the gate passes, the runner checks the training configuration and runs
two real optimizer updates using two-GPU DDP to check memory/update execution.
It resumes those same optimizer/RNG states to 20000 updates, using the C window
[-0.6,-0.3,0] seconds, VLM LoRA + full action expert, effective batch 8,
FP32 trainable storage with BF16 compute, and peak LR 2.5e-5 with warmup/cosine.
It starts from the immutable base, not the old single-frame 6k checkpoint.
The explicitly requested 2k/4k/.../20k checkpoints are all retained. The temporary
2-update checkpoint is pruned by retention once ten scheduled checkpoints exist.

Inspect the experiment's `status.json`, `logs/`, full-reference replay videos,
and the candidate's `repair_report.json` / `expert_replay_gate.json` for actual
progress. The full-reference replay flag remains an expert-only diagnostic.

## Five-cycle model checkpoint videos

The opt-in `excavator_repeat_reference_v1` diagnostic evaluates the complete
five-cycle task, including the final physical return, in one unreinitialized
scene. All motion is predicted by the model. It uses the **recorded task-goal
coordinates** from the archived `episode_index_record.json`, not a live
bed-fill planner, expert actions, expert return poses, or expert frame timings.
This controlled reference-scene comparison is not a novel-scene or live-planner
generalization benchmark.

Each cycle needs newly observed scooped material (collection minimum 5 IDs),
at least the archived minimum deposit (3 IDs) and an open bucket for three
observations. It then requires the emptied bucket (at most 5 particles) to
return within 1.25 m XY of the current dig target and 0.15–2.5 m above it for
three observations. These are explicit task-region criteria, not equality to
an expert joint configuration. Only then is the next recorded task issued.
The final cycle must also return. No cycle advances on a timer or a failure.

The separate `repeat_strict_success` additionally requires **every** completed
cycle to deposit at least `--min-particles` (100 below). A collection-threshold
success is never presented as a 100-particle-per-cycle result. The video labels
both the collection threshold and the strict outcome.

New cycles exclude previously deposited particle IDs and initial bucket residue.
Episode-wide particle accounting remains continuous. The bridge updates goals
without physics ticks or resetting joints, material, clocks, fill filters or
previous actions. C's observation history remains continuous across cycles;
only stale queued old-goal actions are discarded to force a fresh prediction.

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/eval_pi05_checkpoints.py \
  --train-dir outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01/train \
  --output outputs/eval/pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01 \
  --checkpoint-steps 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --gpus 0,1 --reference-episodes 0 --repeat-cycles 5 --repeat-cycle-steps 500 \
  --rollout-steps 2500 --replan-interval 3 --min-particles 100 --latest-first
```

GPU 0 runs 20k/16k/12k/8k/4k, GPU 1 runs 18k/14k/10k/6k/2k. Each gets the
same initial scene, seed, task goals, PhysX backend, action limits and budgets.
There are at most 500 model actions per cycle, including return, and 2500 in
total. An incomplete cycle ends the task as failed, with later cycles marked
`not_attempted`; it does not reset the scene or fabricate five separate trials.

The top output contains live `status.json`, `summary.json` and `summary.csv`.
Videos are `gpu_N/step_NNNNNN/reference_0000/episodes/episode_0001/evaluation.mp4`.
Each episode also saves `repeat_task_spec.json`, `repeat_metrics.json`,
`goal_updates.json`, `repeat_audit.json` and the full `steps.jsonl`.
The independent audit reconstructs cycles, returns and per-cycle ID accounting,
checks continuous history and goal geometry, and verifies every goal update
preserved cameras/commanded joints with zero physics ticks. Unit tests include
a clearly synthetic full shared-evaluator run; `tests/run_repeat_goal_update_live.py`
performs the real PhysX transport/goal-update regression without model/expert motion.
