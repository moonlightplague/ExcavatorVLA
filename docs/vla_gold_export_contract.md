# VLA Gold Export Contract

The raw `.dashboard_success` pool remains the immutable source of record. The
training publication is a separate LeRobot v3 folder produced with:

```text
state_schema=current-v4
state=28D
effort_policy=exclude
quality_policy=gold-v1
stage_policy=repair_stale_parent_boundary_v1
action=4D rad/s
camera=3 x RGB 256x256 at the source 10 Hz timeline
```

## Corrections

- Task directions use the episode's initial excavator base pose. Current 28D
  joint values are never interpreted as `base_x/base_y/base_yaw`.
- A stale parent `lift_carry_done_boundary` written after unload is exported as
  terminal unload stage 9. New collection skips that stale boundary row.
- Raw Isaac effort is retained in source episodes but excluded from the gold
  policy input because solver spikes make mean/std normalization unstable.
- Gold episodes require score >= 70, post-lift spill ratio <= 0.30, and measured
  boom/arm/bucket positions inside the physical limits with 1 degree tolerance.
- Action values remain physical rad/s. Deployment uses the shared limits
  `[6.0, 2.5, 2.5, 6.0]` without an extra `tanh` or hidden `0.02 rad/s` scale.

## Export

```bash
python excavator_dataset_tools.py excavator_auto_dataset/.dashboard_success \
  --export-lerobot-v3 \
  --export-dir excavator_auto_dataset/.dashboard_success/lerobot_v3_gold_v1 \
  --export-reuse-from-dir excavator_auto_dataset/.dashboard_success/lerobot_v3 \
  --export-overwrite \
  --export-require-vla \
  --export-state-schema current-v4 \
  --export-effort-policy exclude \
  --export-quality-policy gold-v1
```

Matching per-episode videos are hard-linked or copied from the existing export;
Parquet tables, task prompts, statistics, episode indices, and manifests are
rebuilt for the selected gold subset.
