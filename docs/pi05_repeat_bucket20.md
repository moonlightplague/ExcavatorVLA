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
progress. The shared model evaluator still needs separate repeated-task goal
and completion handling; the full-reference flag here is an expert diagnostic,
not a new repeated-model benchmark or proof of generalization.
