# Changelog

## 2026-11-08

- Add the PhysX single-cycle PI05 dataset builder with parent-disjoint 20/4/4 splits, collection-backend evidence, and velocity-integrator compatibility filtering.
- Add the dual-GPU LoRA/full-expert 10k pilot, exact continuation configuration, and comparisons of the explicitly retained 2k/4k/6k/8k/10k checkpoints.
- Require requested soil backends to match verified dataset provenance and carry that evidence into training checkpoints.
- Document expert replay gates, model-video outputs, and checkpoint evaluation; cover backend tampering, command-target mismatches, and evaluation scheduling.

## 2026-10-30

- Ignore local datasets, checkpoint directories, runtime logs, and development caches.
- Add PI05 VLM LoRA training, checkpoint reconstruction, continuation and A/B evaluation tools with regression coverage.
- Add ACT ablation configurations and split manifests.
- Add rollout selection and fine-tuning dataset conversion; guard overwrite ownership and selection accounting.
- Handle missing rollout metrics when computing rewards.
