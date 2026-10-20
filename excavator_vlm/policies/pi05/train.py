"""Expert-only PI05 training with valid-weight accumulation and exact data addressing."""
import argparse
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from ...common.config import setup_cache
from ...common.paths import CONFIGS, SCRIPTS, SUBMODULE
from ...common.runtime import dump
from ...data.contract import audit_dataset, file_hash, make_split, validate_split
from .config import build_policy_config, config_dict, resolve_settings
from .dataset import make_pi05_dataset, training_stats


def prepare(settings, create=False):
    report, frames = audit_dataset(settings['dataset']['root'], settings['policy']['chunk_size'])
    path = Path(settings['benchmark']['split_file'])
    if not path.exists() and create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(make_split(frames, report['fingerprint'], settings['benchmark']['split_seed']), stream, indent=2)
    if not path.exists():
        raise FileNotFoundError('Frozen split missing; use --prepare explicitly')
    split = json.loads(path.read_text())
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    episodes = settings['dataset']['episodes'] or split['train']
    if not set(episodes).issubset(split['train']):
        raise ValueError('PI05 training override overlaps held-out episodes')
    return report, split, frames, episodes


@contextmanager
def preserved_rng(seed=None):
    from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state
    from accelerate.utils import set_seed
    state = serialize_rng_state()
    try:
        if seed is not None:
            set_seed(seed)
        yield
    finally:
        deserialize_rng_state(state)


def schedule_spec(config, steps):
    return {'total': min(steps, config.scheduler_decay_steps),
            'warmup': max(1, min(config.scheduler_warmup_steps, steps // 30)),
            'floor': config.scheduler_decay_lr / config.optimizer_lr}


def lr_multiplier(step, spec):
    if step < spec['warmup']:
        return (step + 1) / spec['warmup']
    progress = min(1., max(0., (step - spec['warmup']) / max(1, spec['total'] - spec['warmup'])))
    return spec['floor'] + (1 - spec['floor']) * .5 * (1 + math.cos(math.pi * progress))


def _gather_rng(accelerator):
    from lerobot.utils.random_utils import serialize_rng_state
    state = serialize_rng_state()
    if accelerator.num_processes == 1:
        return [state]
    import torch.distributed as dist
    result = [None] * accelerator.num_processes
    dist.all_gather_object(result, state)
    return result


def run(settings, config, config_path, stop_after=None):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from lerobot.utils.random_utils import deserialize_rng_state
    from ...evaluation.offline_metrics import evaluate_loader
    from .adapter import policy_class, supervision_weights, training_batch
    from .checkpoints import (apply_delta, load_base, resolve_checkpoint, retain_checkpoints,
                              save_checkpoint, trainable_keys)
    from .inference import check_environment
    from .processors import make_processors
    from .sampling import ResumableBatchSampler

    check_environment()
    # Fail before allocating the large model if tokenizer assets are absent.
    tokenizer = Path(settings['assets']['tokenizer_path'])
    if not (tokenizer / 'tokenizer_config.json').is_file():
        raise FileNotFoundError(f'Tokenizer assets missing: {tokenizer}; run prepare_pi05.py --fetch-tokenizer')
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: launch in a GPU-enabled shell')
    torch.set_num_threads(settings['launcher']['cpu_threads'])
    accelerator = Accelerator(cpu=config.device == 'cpu', mixed_precision=settings['launcher']['mixed_precision'],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    config.device = str(accelerator.device)
    set_seed(settings['seed'])
    report, split, frames, episodes = prepare(settings)
    stats = training_stats(frames, episodes)
    dataset = make_pi05_dataset(settings['dataset']['repo_id'], settings['dataset']['root'], episodes,
                               config.chunk_size, settings['dataset']['video_backend'])
    pre, post = make_processors(config, stats, tokenizer)
    resume = settings['training']['resume']
    state, resume_path = None, None
    if resume:
        resume_path = resolve_checkpoint(resume)
        # Only our own tensor/container checkpoint state is accepted.
        state = torch.load(resume_path.parent / 'training_state.pt', map_location='cpu', weights_only=True)
        old = json.loads((resume_path / 'benchmark_provenance.json').read_text())
        expected = (old['dataset_fingerprint'], old['training_episodes'], old['split_manifest'])
        if expected != (report['fingerprint'], episodes, split):
            raise ValueError('Resume data/split differs from checkpoint')
        previous = json.loads((resume_path / 'normalization_stats.json').read_text())
        if previous != stats:
            raise ValueError('Resume normalization differs from checkpoint')
        contract = {'world_size': accelerator.num_processes, 'batch_size': settings['batch_size'],
                    'accumulation': settings['training']['gradient_accumulation_steps']}
        if state['batch_contract'] != contract:
            raise ValueError('Exact resume requires unchanged world size, micro-batch and accumulation')
        if state['step'] >= settings['steps']:
            raise ValueError('Resume target steps must exceed the saved optimizer step')

    policy = policy_class()(config)
    loaded = load_base(policy, settings['assets']['base_path'], settings['assets']['base_sha256'])
    if resume_path:
        manifest = apply_delta(policy, resume_path)
        if manifest['base_sha256'] != settings['assets']['base_sha256']:
            raise ValueError('Resume base checkpoint differs')
        from lerobot.configs.policies import PreTrainedConfig
        old_config = config_dict(PreTrainedConfig.from_pretrained(str(resume_path)))
        current = config_dict(config)
        for key in current:
            if key not in ('device',) and current[key] != old_config.get(key):
                raise ValueError(f'Resume policy setting differs: {key}')
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.optimizer_lr, betas=tuple(config.optimizer_betas),
                                  eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay)
    spec = state['schedule'] if state else schedule_spec(config, settings['steps'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, spec))
    if state:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    policy, optimizer = accelerator.prepare(policy, optimizer)
    raw = accelerator.unwrap_model(policy)
    seen = state['consumed_microbatches'] if state else 0
    sampler = ResumableBatchSampler(len(dataset), settings['batch_size'], settings['seed'],
                                   accelerator.process_index, accelerator.num_processes, seen)
    workers = settings['num_workers']
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        multiprocessing_context='spawn' if workers else None,
                        persistent_workers=workers > 0, timeout=60 if workers else 0)
    iterator = iter(loader)
    # Iterator construction may consume a base seed; restore training RNG afterwards.
    if state:
        deserialize_rng_state(state['rng'][accelerator.process_index])
    run_dir = Path(settings['output_dir'])
    provenance = {'adapter': 'excavator_pi05_v1', 'dataset_fingerprint': report['fingerprint'],
                  'split_manifest': split, 'training_episodes': episodes, 'language_conditioning': True,
                  'normalization': 'quantiles from selected training episodes only',
                  'batch_contract': {'world_size': accelerator.num_processes, 'batch_size': settings['batch_size'],
                                     'accumulation': settings['training']['gradient_accumulation_steps']}}
    validation_loader = None
    if accelerator.is_main_process:
        if not state:
            run_dir.mkdir(parents=True, exist_ok=False)
            dump(run_dir / 'resolved_settings.json', settings)
            dump(run_dir / 'dataset_contract.json', report)
            dump(run_dir / 'normalization_stats.json', stats)
            dump(run_dir / 'split_manifest.json', split)
            dump(run_dir / 'provenance.json', {**provenance, 'assets': loaded,
                'trainable_parameters': sum(p.numel() for p in parameters), 'trainable_keys': trainable_keys(raw),
                'nominal_global_batch': settings['batch_size'] * accelerator.num_processes
                                        * settings['training']['gradient_accumulation_steps'],
                'packages': {p: version(p) for p in ('lerobot', 'torch', 'transformers', 'accelerate')},
                'source_hashes': {str(p.relative_to(SUBMODULE)): file_hash(p)
                                 for p in sorted((SUBMODULE / 'excavator_vlm').rglob('*.py'))}})
            (run_dir / 'training_config.yaml').write_text(Path(config_path).read_text())
        else:
            if run_dir != resume_path.parents[2]:
                raise ValueError('Resume must use the original run directory')
            last = json.loads((run_dir / 'last_checkpoint.json').read_text())
            if Path(last['checkpoint']).resolve() != resume_path:
                raise ValueError('Resume requires the latest checkpoint; branching into an existing run is forbidden')
            dump(run_dir / f'resume_from_{state["step"]:06d}_{os.getpid()}.json', settings)
        if settings['benchmark']['validation_freq']:
            validation = make_pi05_dataset(settings['dataset']['repo_id'], settings['dataset']['root'],
                split['validation'], config.chunk_size, settings['dataset']['video_backend'])
            validation_loader = DataLoader(validation, batch_size=settings['training']['validation_batch_size'])
    accelerator.wait_for_everyone()
    log = (run_dir / 'train.log').open('a' if state else 'w') if accelerator.is_main_process else None
    start_step = state['step'] if state else 0
    best = state['best_validation_mae'] if state else float('inf')
    started = time.perf_counter()
    if config.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, settings['steps'] + 1):
            batches = [next(iterator) for _ in range(settings['training']['gradient_accumulation_steps'])]
            seen += len(batches)
            denominator = sum(float(supervision_weights(batch).sum()) * 4 for batch in batches)
            denominator = accelerator.reduce(torch.tensor(denominator, device=accelerator.device), reduction='sum')
            if denominator.item() <= 0:
                raise ValueError('Effective batch has no supervised actions')
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            local_numerator = torch.zeros((), device=accelerator.device)
            for index, batch in enumerate(batches):
                context = accelerator.no_sync(policy) if index < len(batches) - 1 else nullcontext()
                with context, accelerator.autocast():
                    loss, parts = policy(training_batch(pre, batch), denominator=denominator,
                                         world_size=accelerator.num_processes)
                    finite = accelerator.reduce(torch.isfinite(loss.detach()).int(), reduction='sum')
                    if finite.item() != accelerator.num_processes:
                        raise FloatingPointError(f'Nonfinite flow loss at step {step}')
                    accelerator.backward(loss)
                    local_numerator += parts['flow_numerator']
            norm = accelerator.clip_grad_norm_(parameters, settings['grad_clip_norm'])
            if accelerator.reduce(torch.isfinite(norm).int(), reduction='sum').item() != accelerator.num_processes:
                raise FloatingPointError(f'Nonfinite gradients at step {step}')
            optimizer.step()
            scheduler.step()
            flow = accelerator.reduce(local_numerator, reduction='sum') / denominator
            if step == start_step + 1 or step % settings['log_freq'] == 0:
                row = {'step': step, 'flow_loss': float(flow), 'valid_action_dimensions': float(denominator),
                       'grad_norm_before_clip': float(norm), 'lr': optimizer.param_groups[0]['lr'],
                       'elapsed_seconds': time.perf_counter() - started, 'consumed_microbatches': seen,
                       'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated() if config.device.startswith('cuda') else 0}
                if log:
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(row), flush=True)
            stopping = stop_after is not None and step >= stop_after
            validate = bool(settings['benchmark']['validation_freq']) and (
                step % settings['benchmark']['validation_freq'] == 0 or step == settings['steps'])
            save = validate or step % settings['save_freq'] == 0 or step == settings['steps'] or stopping
            if save:
                accelerator.wait_for_everyone()
                metrics = None
                if accelerator.is_main_process and validate:
                    with preserved_rng(0):
                        metrics, _ = evaluate_loader(raw, pre, post, validation_loader,
                            [6., 2.5, 2.5, 6.], settings['benchmark']['validation_max_batches'])
                rng = _gather_rng(accelerator)
                if accelerator.is_main_process:
                    improved = metrics is not None and metrics['first_action']['mae'] < best
                    if improved:
                        best = metrics['first_action']['mae']
                    state = {'step': step, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                             'schedule': spec, 'rng': rng, 'consumed_microbatches': seen,
                             'batch_contract': provenance['batch_contract'], 'best_validation_mae': best}
                    pretrained = save_checkpoint(run_dir / 'checkpoints' / f'{step:06d}', raw, pre, post, stats,
                                                 settings['assets'], provenance, state)
                    dump(run_dir / 'last_checkpoint.json', {'step': step, 'checkpoint': str(pretrained)})
                    if metrics is not None:
                        metrics['partial_evaluation'] = settings['benchmark']['validation_max_batches'] > 0
                        dump(pretrained.parent / 'validation.json', metrics)
                        # Keep learning curves even when large older checkpoints are pruned.
                        with (run_dir / 'validation_history.jsonl').open('a') as history:
                            history.write(json.dumps({'step': step, 'metrics': metrics}, allow_nan=False) + '\n')
                        print(f'Validation step {step}: first-action MAE={metrics["first_action"]["mae"]:.6f} rad/s', flush=True)
                    if improved:
                        dump(run_dir / 'best_checkpoint.json', {'step': step, 'checkpoint': str(pretrained),
                            'validation_first_action_mae_rad_s': best,
                            'partial_validation': settings['benchmark']['validation_max_batches'] > 0})
                    retain_checkpoints(run_dir, settings['training']['keep_checkpoints'])
                    print(f'Checkpoint: {pretrained}', flush=True)
                accelerator.wait_for_everyone()
            if stopping:
                break
        if accelerator.is_main_process:
            with preserved_rng(0), torch.inference_mode():
                check_batch = {key: batches[0][key] for key in ('observation.state', 'task',
                               'observation.images.0', 'observation.images.1', 'observation.images.2')}
                expected = raw.predict_action_chunk(pre(check_batch)).cpu()
            pretrained = resolve_checkpoint(run_dir)
            apply_delta(raw, pretrained)
            with preserved_rng(0), torch.inference_mode():
                actual = raw.predict_action_chunk(pre(check_batch)).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            dump(run_dir / 'reload_check.json', {'status': 'passed', 'shape': list(actual.shape),
                'check': 'delta roundtrip with fixed sampling seed; independent CLI reload must also be evaluated'})
    finally:
        if log:
            log.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIGS / 'pi05/smoke.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--episodes')
    parser.add_argument('--gpus')
    parser.add_argument('--accumulation', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='Checkpoint and stop after this absolute update (resume smoke tests)')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    overrides = {key: value for key, value in {'steps': args.steps, 'batch_size': args.batch_size,
                 'num_workers': args.num_workers, 'output_dir': str(args.output.resolve()) if args.output else None}.items()
                 if value is not None}
    if args.gpus:
        overrides['launcher'] = {'gpu_ids': [int(g) for g in args.gpus.split(',')]}
    if args.device:
        overrides['policy'] = {'device': args.device}
        if args.device == 'cpu':
            overrides['policy']['dtype'] = 'float32'
            overrides['launcher'] = {'gpu_ids': [0], 'mixed_precision': 'no'}
    if args.episodes:
        overrides['dataset'] = {'episodes': [int(e) for e in args.episodes.split(',')]}
    if args.accumulation is not None:
        overrides['training'] = {'gradient_accumulation_steps': args.accumulation}
    if args.resume:
        from .checkpoints import resolve_checkpoint
        checkpoint = resolve_checkpoint(args.resume)
        overrides.setdefault('training', {})['resume'] = str(checkpoint)
        overrides['output_dir'] = str(args.output.resolve()) if args.output else str(checkpoint.parents[2])
    settings = resolve_settings(args.config, overrides)
    if args.stop_after is not None and not 1 <= args.stop_after <= settings['steps']:
        parser.error('--stop-after must be within the configured update budget')
    setup_cache(settings)
    if not args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, settings['launcher']['gpu_ids']))
    config = build_policy_config(settings)
    if args.prepare or args.dry_run:
        report, split, _, episodes = prepare(settings, args.prepare)
        print(json.dumps({'settings': settings, 'policy': config_dict(config), 'dataset_audit': report,
                          'training_episodes': episodes, 'split_counts': {k: len(split[k]) for k in ('train', 'validation', 'test')}}, indent=2))
        return
    if len(settings['launcher']['gpu_ids']) > 1 and not args.worker:
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
                   f'--num_processes={len(settings["launcher"]["gpu_ids"])}', '--num_machines=1',
                   f'--mixed_precision={settings["launcher"]["mixed_precision"]}', '--dynamo_backend=no',
                   str(SCRIPTS / 'train_pi05.py'), *sys.argv[1:], '--worker']
        if args.output is None and not args.resume:
            command += ['--output', settings['output_dir']]
        subprocess.run(command, check=True)
    else:
        run(settings, config, args.config.resolve(), args.stop_after)


if __name__ == '__main__':
    main()
