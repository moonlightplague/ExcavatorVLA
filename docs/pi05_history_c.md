# PI05 scheme C: causal image and state history

Scheme C conditions PI05 on synchronized image and state windows. It adds no
subtask labels, stage input, phase classifier, or task-switching controller.
The global language instruction is unchanged. The default offsets are
`[-6, -3, 0]` control observations, meaning `[-0.6, -0.3, 0]` seconds at 10 Hz.

## Input and action contract

For a batch of size B and T=3 observations:

| Field | Shape / meaning |
| --- | --- |
| Three canonical camera keys | Each `[B,T,3,256,256]`, RGB in [0,1] |
| `observation.state` | `[B,T,28]`, unchanged current-v4 field order |
| `history_is_pad` | Boolean `[B,T]`; true only for missing initial observations |
| `task` | One global instruction per batch element |
| `action` | Existing future `[B,10,4]` command velocities, rad/s |

Image preprocessing reuses upstream resizing and SigLIP scaling. Images enter
the existing visual prefix in oldest-to-newest order, with cameras 0,1,2 for
each time point: nine images in total. Existing transformer positions distinguish
the ordered image blocks; the prompt describes the ordering and gives the
relative time of each state. No learned temporal embedding or new model weights
are introduced. This is an adaptation requiring fine-tuning, not a claim that
the single-frame checkpoint already understands the extra views.

Every state uses the same train-only quantile statistics. The local history
processor quantizes each state using the upstream binning/padding convention and
includes valid past states in the language prefix. Missing states are labeled
padding, and their image blocks are attention-masked. The current observation
must always be valid. Prompt length defaults to 512 tokens; overflowing prompts
raise an error rather than being truncated.

Only observation timestamps become negative offsets. Action targets, recovery
weights and future padding remain at offsets `0..0.9` seconds. History cannot
cross an episode boundary or access future observations. It is retained across
cycles inside a single episode.

## Training and offline evaluation

`dataset.py` requests identical negative offsets for state and every camera.
LeRobot clamps missing initial samples to the first sample and supplies masks.
The preprocessor checks that camera/state masks agree. Training, validation,
offline evaluation, numerical diagnosis and checkpoint reload use the same
history contract and preserve these masks.

Configurations:

- `configs/pi05/history_c.yaml`: repeated-data template, VLM LoRA + full action
  expert, two-GPU DDP, nominal global batch 8, predict 10 / execute 3.
- `configs/pi05/history_c_smoke.yaml`: 20-update, one-GPU integration template.
- `configs/pi05/history_c_singlecycle.yaml`: the existing verified 20/4/4
  single-cycle dataset, suitable for checking this input pipeline. It does not
  establish repeated-digging capability.

From the parent checkout, a CPU configuration/data check on existing data is:

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/train_pi05.py \
  --config ExcavatorVLA-vlm/configs/pi05/history_c_singlecycle.yaml \
  --device cpu --dry-run
```

After preparing controller-compatible repeated data, verified backend evidence
and a parent-disjoint split, point the repeated configuration at those files.
The ordinary training entrypoint then accepts the C configuration:

```bash
.venv-pi05/bin/python ExcavatorVLA-vlm/scripts/train_pi05.py \
  --config ExcavatorVLA-vlm/configs/pi05/history_c.yaml
```

The existing repeated publication has unencoded control-target discontinuities
and lacks the required backend-provenance manifest. This change does not repair
those data or declare them suitable for full continuous training. Keep the
provenance gate enabled; fix/recollect the controller-compatible data before a
repeated training run. Do not mix cycles of one source parent across splits.

## Online inference

`ObservationHistory` is an explicit, episode-local buffer. Append the latest
observation on **every control step**, including the steps between replans.
Query a window only when predicting a new chunk. The shared evaluator does this
automatically for PI05 history checkpoints and recreates the buffer each episode.
Its step trace records `input_history_steps` and `input_history_is_pad`, making
the window used by each prediction auditable. During chunk execution those
fields describe the current available window; `policy_latency_ms` identifies
the steps that actually run a new prediction.

Independent clients should pass a current batch with state `[1,28]`, three
camera tensors `[1,3,256,256]`, and a one-element task list to
`ObservationHistory.append(batch, step)`. Pass `buffer.batch()` to the loaded
preprocessor and policy. Step indices must be consecutive from zero. Call
`reset()` on a new episode, not at each replan or scoop. The bridge still carries
one current observation per reply; history is maintained on the policy side.

The existing evaluator still terminates on single-cycle success. Supporting a
changing dig/unload target, repeated completion criteria and cycle recovery is
separate task-protocol work; adding a history window does not implement it.

## Checkpoints and verification

The native PI05 config stores `n_obs_steps`. The checkpoint manifest additionally
stores the versioned offsets, FPS, camera ordering, state encoding and padding
contract. Loading and exact resume reject missing or different multi-frame
contracts, including a different stride with the same number of frames.
Legacy single-frame checkpoints remain supported. A single-frame run cannot be
exactly resumed as a C run. Base weight keys and LoRA/expert delta handling are
unchanged; upstream reference sources are not edited.

CPU tests exercise real PI05 preprocessing/prediction adapters with an explicitly
synthetic small core, padding invariance, historical-image influence, gradients,
processor serialization, checkpoint reconstruction, and continuous versus
resumed training. Online rollout fixtures check per-step updates and per-episode
reset. These checks do not measure excavator success or real-model GPU memory,
latency or distributed numerical behavior. Nine image blocks increase encoder
work and transformer context; measure the actual GPU budget before training.
