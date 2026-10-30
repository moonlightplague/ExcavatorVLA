# pi0.5 configuration

These configurations select expert-only pi0.5 fine-tuning and reuse the
benchmark's shared data contracts, frozen episode split and evaluation metrics.

Pretrained weights remain in the parent project's `models/pi05_base/`.
`full.yaml` defines the shared defaults. `smoke.yaml`, `overfit.yaml`, and
`pilot.yaml` inherit it and override their update budgets and validation scope.
Batch size is per GPU; effective batch also includes gradient accumulation.
All step counts refer to optimizer updates, not micro-batches.

`training.trainable_dtype: float32` keeps expert/projection/time parameters,
gradients and AdamW moments in FP32. `policy.dtype: bfloat16` applies to frozen
backbone initialization (with upstream FP32 normalization/patch exceptions),
not trainable storage. CUDA forward compute uses BF16 autocast for both training
and new-checkpoint inference. Old BF16 checkpoints remain evaluable but cannot
be used for an exact FP32 resume; start a new run from the immutable base.

Use only paths under `configs/pi05/`. See [the pipeline guide](../../docs/pi05.md)
for environment, licensed tokenizer preparation, smoke/resume and evaluation.

## Matched LoRA experiment

`expert_ab_overfit.yaml` and `lora_overfit.yaml` use two GPUs, micro-batch 1,
accumulation 4 and 2,000 updates: effective batch 8 for both arms. Each starts
fresh from the immutable base on training Episode 1; neither resumes the old
single-GPU run. `lora_smoke.yaml` runs two dual-GPU updates (effective batch 2)
for integration checks. `lora_pilot.yaml` is a separate 160-episode, 5k-update
experiment with accumulation 8/effective batch 16, not automatically launched.

`training.mode: vlm_lora_expert_full` enables unmerged rank-16/alpha-32 LoRA on
only the VLM Gemma q/k/v/o projections, while the expert remains fully FP32
trainable and vision/base VLM weights remain frozen. Dedicated adapter-init RNG
does not perturb the training sampler/noise sequence. No PEFT dependency or
upstream source modification is needed. Checkpoint manifests persist mode,
adapter hyperparameters and exact target-module paths for strict reconstruction.
