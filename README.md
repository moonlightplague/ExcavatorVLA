# ExcavatorVLA VLM benchmark

This submodule owns the policy-side code used to train and compare robot
policies in the excavator loading benchmark. Model checkpoints are kept outside
this repository under the parent project's `models/` directory.

## Layout

```text
excavator_vlm/
  policies/
    act/
      adapter.py               # ACT training loss and batch adaptation
      config.py                # ACT config construction/validation
      dataset.py               # ACT action chunks and normalization statistics
      train.py                 # ACT training + validation + checkpoint reload check
      inference.py             # ACT checkpoint/processor loading
    pi05/                      # expert-only fine-tuning, strict base + delta loading
  data/contract.py             # shared state/action contract, audit and episode split
  evaluation/
    offline.py                 # shared offline evaluator with injected policy adapters
    offline_metrics.py         # shared physical-unit action metrics
    isaac.py                   # access to the parent benchmark's shared Isaac evaluator
  common/                      # shared YAML/path handling, I/O and data-loader helpers
configs/
  act/                         # smoke.yaml / overfit.yaml / pilot.yaml / full.yaml
  pi05/                        # smoke.yaml / overfit.yaml / pilot.yaml / full.yaml
  splits/                      # frozen episode splits shared by all policies
scripts/                       # thin CLI wrappers; existing script paths are preserved
tests/                         # masking, splits, real-data and checkpoint tests
third_party/lerobot-v0.4.4/     # unchanged upstream reference source and license
../models/                     # pretrained weights/cache, outside this submodule
../outputs/                    # run artifacts, outside this submodule
```

Each policy owns its config construction, preprocessing, loss, training and
checkpoint loading. Data contracts, episode splits, physical-unit metrics and
Isaac execution are shared. `scripts/train_act.py`, `scripts/eval_act_offline.py`
and `scripts/eval_act_success.py` provide the ACT entrypoints. All ACT training
configurations live under `configs/act/`; use these paths in commands.

Runtime imports use the installed `lerobot==0.4.4`. The vendored source is an
unchanged reference. ACT starts with ImageNet ResNet-18 weights and a fresh
transformer/action head; it needs no ACT base checkpoint. Pi0.5 assets remain in
`../models/pi05_base/`. Its isolated environment, tokenizer authorization,
training/resume, delta checkpoint and evaluation commands are documented in
[the PI05 pipeline guide](docs/pi05.md). PI05 uses its own pinned Transformers
compatibility implementation; do not install it over the working ACT environment.

## Environment and data

Run these commands from the parent `ExcavatorVLA_new` checkout:

```bash
export BASE=/root/gpufree-data/ExcavatorVLA_new
cd "$BASE"
export POLICY_PYTHON=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
"$POLICY_PYTHON" -c 'import torch, lerobot; print(torch.cuda.is_available())'
```

Use an unpatched LeRobot 0.4.4 environment. For a new environment, install this
submodule's `requirements.txt`. ACT needs no SmolVLA/transformers extras. The
script uses the YAML's `hf_home` and `torch_home`; the first ResNet initialization
may download torchvision's standard ImageNet weights to `models/torch-cache`.
Local dataset loading uses offline Hugging Face mode.

The checked export is `data/export`: 200 episodes, 39,962 frames, 10 Hz, three
256x256 RGB cameras, 28D current-v4 state, and 4D command velocities in rad/s.
The adapter checks exact feature names/order against that contract. Separate
effort and categorical phase/recovery labels are not policy inputs. The 28D
state includes simulator-derived targets and fill information; this is an
instrumented-state benchmark. ACT does not consume language prompts.

`configs/splits/excavator_v4_seed1000.json` freezes a task-stratified episode
split of 160 train / 20 validation / 20 test. It is bound to the metadata and
parquet content fingerprint. State/action mean/std are fit only on the selected
training episodes; images use fixed ImageNet mean/std. The original dataset's
global `stats.json` is not used. A new export requires a new split file/version.
This split measures held-out episodes, not unseen task or spatial generalization.

## Training

All paths in YAML are relative to the YAML file. Each run gets a new output
directory and refuses to overwrite an existing one. Batch size is per GPU.

```bash
# Audit all episode timestamps/history and all camera videos; validate the split.
# Exclusively creates the manifest if it does not exist.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/smoke.yaml --prepare

"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/smoke.yaml --dry-run

# 20 steps, single GPU, batch 2: validate save/reload and finite gradients.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/smoke.yaml

# 2,000 steps on TRAIN episode 1; inspect fitting before larger runs.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/overfit.yaml

# 5,000-step pilot, two GPUs, global batch 16.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/pilot.yaml

# 40,000 steps. Run after checking overfit/pilot diagnostics.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/train_act.py \
  --config ExcavatorVLA-vlm/configs/act/full.yaml
```

CLI overrides include `--gpus 0`, `--episodes 1`, `--steps 20`, `--batch-size 2`,
`--num-workers 0`, and `--output /path/to/new/run`. Training episode overrides
must be subsets of the frozen train split. `--device cpu` selects one CPU
process with full precision. `--no-pretrained-backbone` is available for offline
implementation tests and must be reported if used for experiments.

Every config predicts 10 steps (1 second) and executes 3 steps (0.3 seconds)
before replanning. The ACT default transformer dimensions are retained; KL
weight is 10, learning rate is 1e-5 for both backbone and the remaining model,
and weight decay is 1e-4. AdamW has no LR scheduler in this initial baseline.
Temporal ensembling is disabled for comparable chunk execution.

The training loss uses each future action's `action_loss_weight`, excludes
episode padding from the denominator, and adds VAE KL for supervised samples.
The adapter explicitly preserves weights across LeRobot 0.4.4 preprocessing,
which otherwise drops arbitrary action auxiliary fields. DDP loss normalization
accounts for different valid target counts on each rank. The architecture and
weight keys remain compatible with stock `ACTPolicy`.

Run artifacts include resolved configuration, dataset audit/fingerprint, split,
train-only normalization stats, package/source hashes, JSON-lines `train.log`,
validation results, `best_checkpoint.json`, and `reload_check.json`.

```text
outputs/train/excavator_act_.../checkpoints/last/pretrained_model/
  config.json, model.safetensors, train_config.json,
  policy_preprocessor.json, policy_postprocessor.json,
  processor safetensors, benchmark_provenance.json
```

`last` means latest, not best. `best_checkpoint.json` ranks validation
first-action MAE; use closed-loop validation scenes to make the final selection.
Smoke/overfit validation uses a limited number of batches, marked explicitly in
that file; pilot/full runs evaluate the entire validation split. Optimizer/RNG
state is saved in LeRobot format, but this launcher currently starts fresh runs
and has no resume CLI. Rerunning a command creates a new experiment.

## Offline evaluation

```bash
export CHECKPOINT="$BASE/outputs/train/<run>/checkpoints/last/pretrained_model"
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/eval_act_offline.py \
  --checkpoint "$CHECKPOINT" --split validation \
  --output "$BASE/outputs/eval/act_validation_<unique-name>"

# Overfit diagnostic: explicitly evaluate the same training episode.
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/eval_act_offline.py \
  --checkpoint "$CHECKPOINT" --split train --episodes 1 \
  --output "$BASE/outputs/eval/act_fit_<unique-name>"
```

Outputs are `metrics.json` and `actions.npz`: first-action and full-chunk
MAE/RMSE in rad/s, each joint and horizon, moving-action sign accuracy, near-zero
accuracy (threshold 0.02 rad/s), saturation, action statistics, normal/recovery
targets, and zero-action baselines. Padding is excluded; supervision-filtered
metrics are also reported. Recovery labels follow each future target timestamp.
Timing covers preprocessing through CPU output transfer, excludes video decode,
and reports batch size and inclusion of the first warmup batch. Offline FPS is
observations/second; closed-loop control FPS is recorded separately.

Use `--split test` only after selecting a checkpoint. The evaluator rejects
held-out/training overlap and changed data fingerprints. Training-fit errors
and validation errors do not establish simulator success.

## Isaac evaluation

```bash
"$POLICY_PYTHON" ExcavatorVLA-vlm/scripts/eval_act_success.py \
  --checkpoint "$CHECKPOINT" --dataset "$BASE/data/export" \
  --output "$BASE/outputs/eval/act_success_<unique-name>" \
  --episodes 5 --steps 350 --min-particles 100 \
  --seed 2 --replan-interval 3 --gpus 0 --no-video
```

The wrapper calls the parent `scripts/eval_smolvla_success.py --policy-type act`.
It shares persistent Isaac workers, scene resets, initial-pose/seed handling,
camera preparation, safety clipping, and metrics with SmolVLA. The checkpoint's
saved normalization is loaded; no pretrained ResNet download is needed to reload
a complete ACT checkpoint. Actions remain velocity commands integrated by the
existing simulator position-target controller.

Evaluate a fixed reference scene, then 5 debug seeds, 20 selection seeds, and
100 held-out benchmark seeds. Use a disjoint final seed range for reporting.
Keep the same dataset, initial pose, seeds, camera/render settings, physics,
episode horizon, action limits, replan interval and success threshold for each
policy. Existing SmolVLA checkpoints trained on all 200 episodes must be
retrained on this same train split/stats before reporting a matched comparison.
The current success criterion is stable deposited particle count plus
bucket opening; payload and spillage are particle counts, not calibrated mass.
This existing evaluator ends after the current success criterion; repeated
multi-scoop task completion needs a separate benchmark extension.

## Tests

```bash
"$POLICY_PYTHON" -m unittest discover -s ExcavatorVLA-vlm/tests -v
```

Tests cover masking gradients, train-only stats/split leakage, real dataset tail
chunks, physical-unit postprocessing, stock ACT checkpoint compatibility, and a
two-step real-data training/validation/save/reload integration test using a small
CPU model with `num_workers=0`, suitable for restricted environments. Multi-worker
data loading uses spawn and requires local IPC socket permissions. A worker that
cannot return a batch fails after 60 seconds instead of waiting indefinitely.
GPU smoke, overfit and Isaac outcomes must be reported separately.

Implementation validation on 2026-10-19 completed the standard 20-step,
single-RTX-4090 smoke run with ImageNet initialization and bf16, validation,
checkpoint saving and a successful reload comparison (`[1, 10, 4]`). Artifacts:
`../outputs/train/excavator_act_smoke_20261019_implementation/`.
Subsequent ACT runs completed the 2,000-step overfit and two-GPU 5,000-step pilot
(`excavator_act_pilot_20261019_151536_911263_2626695`). The pilot's full validation
first-action MAE was 0.072255 rad/s. A single seed-2 Isaac diagnostic completed
with 682 deposited source particles and an open bucket; artifacts are in
`../outputs/eval/act_pilot_5000_seed2_video_20261019/`. This is one diagnostic
success, not a held-out success-rate estimate, real-time deployment result, or
repeated multi-scoop benchmark. The 40,000-step ACT run remains separate work.
