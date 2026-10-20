# pi0.5 configuration

These configurations select expert-only pi0.5 fine-tuning and reuse the
benchmark's shared data contracts, frozen episode split and evaluation metrics.

Pretrained weights remain in the parent project's `models/pi05_base/`.
`full.yaml` defines the shared defaults. `smoke.yaml`, `overfit.yaml`, and
`pilot.yaml` inherit it and override their update budgets and validation scope.
Batch size is per GPU; effective batch also includes gradient accumulation.
All step counts refer to optimizer updates, not micro-batches.

Use only paths under `configs/pi05/`. See [the pipeline guide](../../docs/pi05.md)
for environment, licensed tokenizer preparation, smoke/resume and evaluation.
