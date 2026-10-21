"""Fixed-input PI05 precision diagnostics; never writes a trained checkpoint."""
import argparse
from contextlib import contextmanager
import copy
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ...common.paths import PROJECT_ROOT, SUBMODULE
from ...common.runtime import dump
from ...data.contract import CAMERAS, JOINTS, audit_dataset, file_hash
from .adapter import flow_loss_terms, training_batch
from .checkpoints import resolve_checkpoint
from .dataset import make_pi05_dataset
from .inference import load_pi05


def difference(left, right):
    """Shape-strict FP64 comparisons, without mixing physical units."""
    left, right = left.detach().cpu(), right.detach().cpu()
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError('Compared tensors must have equal, nonempty shapes')
    a, b = left.double(), right.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite diagnostic tensor')
    delta = a - b
    result = {'shape': list(left.shape), 'left_dtype': str(left.dtype), 'right_dtype': str(right.dtype),
              'exact_equal': bool(torch.equal(a, b)), 'mae': delta.abs().mean().item(),
              'rmse': delta.square().mean().sqrt().item(), 'max_abs': delta.abs().max().item()}
    if left.ndim == 3 and left.shape[-1] >= 4:
        result['first_four_mae'] = delta[..., :4].abs().mean().item()
        result['per_joint_mae'] = dict(zip(JOINTS, delta[..., :4].abs().mean((0, 1)).tolist()))
    return result


def compare_observations(training, inference):
    # Processor converters also emit 'action': None for observation-only calls.
    # It is a training target, not a shared observation input.
    keys = ('observation.state', *CAMERAS, 'observation.language.tokens',
            'observation.language.attention_mask')
    return {key: difference(training[key], inference[key]) for key in keys}


@contextmanager
def capture_outputs(modules):
    """Observe actual module calls, removing hooks even when a forward fails."""
    outputs, handles = {}, []
    for name, module in modules.items():
        outputs[name] = []

        def hook(_module, _inputs, output, key=name):
            outputs[key].append(output.detach().cpu().clone())

        handles.append(module.register_forward_hook(hook))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def cached_prefix(core, images, image_masks, tokens, masks):
    """Exact prefix-cache construction from upstream PI05Pytorch.sample_actions."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    embeddings, padding, attention = core.embed_prefix(images, image_masks, tokens, masks)
    mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
    core.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
    _, cache = core.paligemma_with_expert.forward(
        attention_mask=mask, position_ids=torch.cumsum(padding, dim=1) - 1,
        past_key_values=None, inputs_embeds=[embeddings, None], use_cache=True)
    return padding, cache


def probe_modules(core):
    return {'vision_patch': core.paligemma_with_expert.paligemma.vision_tower
            .vision_model.embeddings.patch_embedding,
            **{name: getattr(core, name) for name in
               ('action_in_proj', 'time_mlp_in', 'time_mlp_out', 'action_out_proj')}}


def vector_field(policy, batch, noise, timestep, path, amp, train_mode=False):
    """Same normalized x_t, t and weights, through real full or cached forward."""
    policy.train(train_mode)
    core = policy.model
    core.gradient_checkpointing_disable()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp), \
            capture_outputs(probe_modules(core)) as traced:
        images, image_masks = policy._preprocess_images(batch)
        tokens, masks = batch['observation.language.tokens'], batch['observation.language.attention_mask']
        actions = policy.prepare_action(batch)
        times = torch.full((actions.shape[0],), timestep, device=actions.device, dtype=torch.float32)
        if path == 'full':
            losses = core(images, image_masks, tokens, masks, actions, noise=noise, time=times)
            if len(traced['action_out_proj']) != 1:
                raise RuntimeError('Expected exactly one full-forward action projection')
            field = traced['action_out_proj'][0]
            # Confirm the captured tensor is the exact field used for supervision.
            expected = (noise.cpu() - actions.cpu() - field.float()).square()
            torch.testing.assert_close(losses.float().cpu(), expected, rtol=1e-5, atol=1e-6)
        elif path == 'cached':
            padding, cache = cached_prefix(core, images, image_masks, tokens, masks)
            x_t = times[:, None, None] * noise + (1 - times[:, None, None]) * actions
            field = core.denoise_step(padding, cache, x_t, times).detach().cpu()
        else:
            raise ValueError(f'Unknown forward path: {path}')
    return field, traced


def compare_traces(left, right):
    if left.keys() != right.keys():
        raise ValueError('Trace module sets differ')
    result = {}
    for name in left:
        if len(left[name]) != len(right[name]):
            raise ValueError(f'Trace call counts differ: {name}')
        result[name] = [difference(a, b) for a, b in zip(left[name], right[name])]
    return result


def parameter_group(name, original_dtype):
    if '.gemma_expert.' in name:
        return 'expert_bf16' if original_dtype == torch.bfloat16 else 'expert_fp32'
    return 'projections_and_time'


def update_counts(before, after, gradient):
    before, after = before.float(), after.float()
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError('Nonfinite optimizer trial weights')
    changed = before != after
    nonzero = torch.zeros_like(changed) if gradient is None else gradient != 0
    if gradient is not None and not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite optimizer trial gradient')
    delta = after.double() - before.double()
    return {'parameters': before.numel(), 'with_gradient': before.numel() if gradient is not None else 0,
            'nonzero_gradient': int(nonzero.sum()), 'changed': int(changed.sum()),
            'changed_nonzero_gradient': int((changed & nonzero).sum()),
            'squared_update_l2': delta.square().sum().item(), 'update_abs_sum': delta.abs().sum().item()}


def summarize_counts(counts):
    return {**counts, 'changed_fraction': counts['changed'] / max(1, counts['parameters']),
            'changed_nonzero_gradient_fraction': counts['changed_nonzero_gradient']
                / max(1, counts['nonzero_gradient']),
            'update_l2': counts['squared_update_l2'] ** .5,
            'mean_abs_update': counts['update_abs_sum'] / max(1, counts['parameters'])}


def optimizer_trial(snapshots, saved_optimizer, *, device, lr, fp32=False):
    """One real AdamW step on disposable copies, using identical saved moments/gradients.

    FP32 is a same-gradient counterfactual, NOT a full FP32 retraining experiment.
    Original model parameters and the CPU snapshots are never modified.
    """
    trial, optimizer = [], None
    try:
        for name, weight, grad in snapshots:
            dtype = torch.float32 if fp32 else weight.dtype
            parameter = torch.nn.Parameter(weight.to(device=device, dtype=dtype, copy=True))
            if grad is not None:
                parameter.grad = grad.to(device=device, dtype=dtype, copy=True)
            trial.append(parameter)
        optimizer = torch.optim.AdamW(trial)
        # load_state_dict may retain tensor aliases on CPU; isolate every trial.
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        for group in optimizer.param_groups:
            group['lr'] = lr
        state_dtypes = sorted({str(value.dtype) for state in optimizer.state.values()
                               for key, value in state.items() if key != 'step' and torch.is_tensor(value)})
        optimizer.step()
        groups, parameters = {}, []
        for (name, before, grad), after in zip(snapshots, trial):
            after_cpu = after.detach().cpu()
            counts = update_counts(before, after_cpu, grad)
            group = parameter_group(name, before.dtype)
            totals = groups.setdefault(group, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                totals[key] += value
            parameters.append({'name': name, 'original_dtype': str(before.dtype),
                               'trial_dtype': str(after.dtype), **summarize_counts(counts)})
        return {'lr': lr, 'fp32_parameter_counterfactual': fp32, 'moment_dtypes': state_dtypes,
                'groups': {key: summarize_counts(value) for key, value in groups.items()},
                'parameters': parameters}
    finally:
        del optimizer, trial
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def fixed_gradient(policy, batch, noise):
    """Actual masked training loss, fixed t=0.5, clipped just as in production."""
    policy.train()
    policy.zero_grad(set_to_none=True)
    core = policy.model
    core.gradient_checkpointing_enable()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        images, masks = policy._preprocess_images(batch)
        losses = core(images, masks, batch['observation.language.tokens'],
                      batch['observation.language.attention_mask'], policy.prepare_action(batch),
                      noise=noise, time=torch.full((1,), .5, device=noise.device))
        numerator, denominator = flow_loss_terms(losses, batch)
        if denominator.item() <= 0:
            raise ValueError('Diagnostic batch has no supervised actions')
        loss = numerator / denominator
    loss.backward()
    named = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True)
    frozen_with_grad = [name for name, p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
    if frozen_with_grad:
        raise RuntimeError(f'Frozen parameters have gradients: {frozen_with_grad[:5]}')
    snapshots = [(name, p.detach().cpu().clone(), None if p.grad is None else p.grad.detach().cpu().clone())
                 for name, p in named]
    policy.zero_grad(set_to_none=True)
    core.gradient_checkpointing_disable()
    policy.eval()
    return snapshots, {'flow_loss': loss.item(), 'grad_norm_before_clip': norm.item(),
                       'clip_norm': 1., 'valid_action_dimensions': denominator.item(),
                       'frozen_parameters_with_gradient': frozen_with_grad,
                       'scope': 'one recorded observation, fixed noise and t=0.5; not the historical 8-microbatch update'}


def run(args):
    from torch.utils.data import default_collate
    if not torch.cuda.is_available():
        raise RuntimeError('Real PI05 diagnostic requires an idle CUDA GPU')
    started = time.perf_counter()
    checkpoint = resolve_checkpoint(args.checkpoint)
    hashes = {name: file_hash(checkpoint / name) for name in
              ('checkpoint.json', 'config.json', 'model.safetensors', 'normalization_stats.json')}
    manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
    if manifest.get('precision_contract') is not None:
        raise ValueError('This autocast on/off diagnostic is for legacy mixed-dtype checkpoints. '
                         'FP32-expert checkpoints require BF16 autocast for mixed-backbone execution; '
                         'use the trainer smoke and independent checkpoint reload instead.')
    provenance = json.loads((checkpoint / 'benchmark_provenance.json').read_text())
    if args.episode not in provenance['training_episodes']:
        raise ValueError('This overfit diagnostic must select an actual training episode')
    report = {'status': 'running', 'checkpoint': str(checkpoint), 'checkpoint_hashes': hashes,
              'base_sha256': manifest['base_sha256'], 'episode': args.episode, 'frames': args.frames,
              'times': args.times, 'seed': args.seed, 'gpu': torch.cuda.get_device_name(),
              'packages': {p: version(p) for p in ('torch', 'transformers', 'lerobot', 'accelerate')},
              'source_sha256': file_hash(Path(__file__)), 'argv': sys.argv,
              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
              'notes': ['autocast off retains the checkpoint mixed BF16/FP32 weight dtypes; not an all-FP32 model',
                        'vector fields are in normalized flow units, not rad/s',
                        'action comparisons use physical rad/s, before safety clipping',
                        'no simulator rollout or further training is performed'],
              'preprocessing': [], 'vector_fields': [], 'sampling': []}
    dump(args.output / 'report.json', report)
    audit, _ = audit_dataset(args.dataset)
    if audit['fingerprint'] != provenance['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint differs from training')
    report['dataset_fingerprint'] = audit['fingerprint']
    policy, pre, post = load_pi05(checkpoint, 'cuda')
    import inspect
    from lerobot.policies.pi05 import modeling_pi05
    report['modeling_source_sha256'] = file_hash(inspect.getfile(modeling_pi05))
    report['trainable_parameters'] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    report['active_dropout'] = [name for name, m in policy.named_modules()
                                if isinstance(m, torch.nn.Dropout) and m.p > 0]
    dataset = make_pi05_dataset('local/excavator', args.dataset.resolve(), [args.episode], policy.config.chunk_size)
    arrays = {}
    first_batch, first_noise = None, None
    for frame in args.frames:
        if not 0 <= frame < len(dataset):
            raise ValueError(f'Frame {frame} outside episode length {len(dataset)}')
        batch = default_collate([dataset[frame]])
        if int(batch['episode_index'][0]) != args.episode or int(batch['frame_index'][0]) != frame:
            raise ValueError('Selected dataset index does not match the requested episode/frame')
        training = training_batch(pre, copy.deepcopy(batch))
        observation = pre({key: copy.deepcopy(batch[key]) for key in ('observation.state', *CAMERAS, 'task')})
        compared = compare_observations(training, observation)
        if not all(value['exact_equal'] for value in compared.values()):
            raise AssertionError('Training and inference preprocessing differ')
        report['preprocessing'].append({'frame': frame, 'comparisons': compared})
        generator = torch.Generator(device='cpu').manual_seed(args.seed + frame)
        noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator).cuda()
        arrays[f'frame_{frame}_noise'] = noise.cpu().numpy()
        arrays[f'frame_{frame}_target_rad_s'] = batch['action'].numpy()
        if first_batch is None:
            first_batch, first_noise = training, noise
        for timestep in args.times:
            results, traces = {}, {}
            variants = [('train_full_amp', 'full', True, True), ('eval_full_amp', 'full', True, False),
                        ('eval_cached_amp', 'cached', True, False), ('train_full_noamp', 'full', False, True),
                        ('eval_full_noamp', 'full', False, False), ('eval_cached_noamp', 'cached', False, False)]
            for label, path, amp, train_mode in variants:
                results[label], traces[label] = vector_field(policy, training, noise, timestep, path, amp, train_mode)
                arrays[f'frame_{frame}_t{timestep}_{label}'] = results[label].float().numpy()
            pairs = [('train_full_amp', 'eval_full_amp'), ('eval_full_amp', 'eval_cached_amp'),
                     ('train_full_noamp', 'eval_full_noamp'), ('eval_full_noamp', 'eval_cached_noamp'),
                     ('train_full_amp', 'eval_cached_noamp'), ('eval_cached_amp', 'eval_cached_noamp'),
                     ('train_full_amp', 'train_full_noamp')]
            report['vector_fields'].append({'frame': frame, 'time': timestep,
                'comparisons': {f'{a}__vs__{b}': {'field': difference(results[a], results[b]),
                                 'module_outputs': compare_traces(traces[a], traces[b])} for a, b in pairs}})
        chunks = {}
        for amp in (False, True):
            label = 'amp' if amp else 'noamp'
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                normalized = policy.predict_action_chunk(observation, noise=noise)
            chunks[label] = post(normalized.detach().clone()).cpu()
            arrays[f'frame_{frame}_chunk_{label}_rad_s'] = chunks[label].numpy()
        valid = ~batch['action_is_pad']
        report['sampling'].append({'frame': frame, 'amp__vs__noamp_rad_s': difference(chunks['amp'], chunks['noamp']),
            'first_action_amp__vs__noamp_rad_s': difference(chunks['amp'][:, :1], chunks['noamp'][:, :1]),
            'target_mae_rad_s': {key: (value[valid] - batch['action'][valid]).abs().mean().item()
                                 for key, value in chunks.items()}})
        dump(args.output / 'report.json', report)
        print(f'Completed fixed-input forward comparisons: episode={args.episode}, frame={frame}', flush=True)
    np.savez_compressed(args.output / 'arrays.npz', **arrays)
    snapshots, report['gradient'] = fixed_gradient(policy, first_batch, first_noise)
    dump(args.output / 'report.json', report)
    gc.collect()
    torch.cuda.empty_cache()
    state_path = checkpoint.parent / 'training_state.pt'
    saved = torch.load(state_path, map_location='cpu', weights_only=True)
    if saved['step'] != manifest['step']:
        raise ValueError('Optimizer and model checkpoint steps differ')
    report['optimizer_step'] = saved['step']
    report['optimizer_trials'] = []
    saved_lr = saved['optimizer']['param_groups'][0]['lr']
    for lr in dict.fromkeys([saved_lr, policy.config.optimizer_lr]):
        for fp32 in (False, True):
            print(f'AdamW disposable-copy trial: lr={lr}, fp32={fp32}', flush=True)
            result = optimizer_trial(snapshots, saved['optimizer'], device='cuda', lr=lr, fp32=fp32)
            report['optimizer_trials'].append(result)
            dump(args.output / 'report.json', report)
    # Optimizer trials used independent copies. Verify the actual in-memory model was not updated.
    named = dict(policy.named_parameters())
    if not all(torch.equal(named[name].detach().cpu(), weight) for name, weight, _ in snapshots):
        raise AssertionError('Diagnostic unexpectedly changed live model weights')
    after_hashes = {name: file_hash(checkpoint / name) for name in hashes}
    if after_hashes != hashes:
        raise AssertionError('Checkpoint files changed during diagnostic')
    report.update(status='completed', checkpoint_files_unchanged=True, live_weights_unchanged=True,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    dump(args.output / 'report.json', report)
    print(f'Diagnostic complete: {args.output / "report.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=PROJECT_ROOT / 'data/export')
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--frames', default='0,3,8,20,30')
    parser.add_argument('--times', default='1.0,0.5,0.1')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.frames = [int(value) for value in args.frames.split(',')]
    args.times = [float(value) for value in args.times.split(',')]
    if not args.frames or min(args.frames) < 0 or len(set(args.frames)) != len(args.frames):
        parser.error('frames must be unique nonnegative indices')
    if not args.times or any(not 0 < t <= 1 for t in args.times) or args.gpu < 0:
        parser.error('times must be in (0,1]; GPU index must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / 'models/hf-cache'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(4)
    # Atomic ownership before error handling: never relabel an existing report
    # as failed if a caller accidentally reuses a previous output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    try:
        run(args)
    except Exception as exc:
        path = args.output / 'report.json'
        if path.is_file():
            report = json.loads(path.read_text())
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            dump(path, report)
        raise


if __name__ == '__main__':
    main()
