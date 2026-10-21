"""Train Diffusion Policy with a frozen episode split and train-only normalization statistics."""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ...common.config import setup_cache
from ...common.paths import CONFIGS, SCRIPTS, SUBMODULE
from ...common.runtime import dump, loader_for
from ...data.contract import audit_dataset, file_hash, make_split, validate_split
from .config import build_train_config, resolve_settings
from .dataset import make_diffusion_dataset, training_stats


def prepare(settings, create=False):
    report, frames = audit_dataset(settings['dataset']['root'], settings['policy']['horizon'])
    path = Path(settings['benchmark']['split_file'])
    if not path.exists():
        if not create:
            raise ValueError(f'Frozen split missing: run train_diffusion.py --prepare first ({path})')
        split = make_split(frames, report['fingerprint'], settings['benchmark']['split_seed'])
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently replacing a split from another run.
        with path.open('x') as handle:
            json.dump(split, handle, indent=2)
            handle.write('\n')
    split = json.loads(path.read_text())
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    episodes = settings['dataset'].get('episodes') or split['train']
    if not set(episodes).issubset(split['train']):
        raise ValueError('Training episode override contains held-out episodes')
    return report, split, frames, episodes


def run(settings, cfg, config_path):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
    from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
    from .adapter import ExcavatorDiffusionPolicy, training_batch
    from .inference import load_diffusion
    from ...evaluation.offline_metrics import evaluate_loader

    if cfg.policy.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run on a GPU node with working NVIDIA driver access')
    torch.set_num_threads(settings['launcher']['cpu_threads'])
    accelerator = Accelerator(cpu=cfg.policy.device == 'cpu',
                              mixed_precision=settings['launcher']['mixed_precision'],
                              step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(cfg.seed)
    # All ranks construct the same immutable split/stats; only main writes outputs.
    report, split, frames, episodes = prepare(settings)
    cfg.dataset.episodes = episodes
    cfg.policy.device = str(accelerator.device)
    stats = training_stats(frames, episodes)
    dataset = make_diffusion_dataset(cfg.dataset.repo_id, cfg.dataset.root, episodes,
                                     cfg.policy.horizon, cfg.dataset.video_backend)
    policy = ExcavatorDiffusionPolicy(cfg.policy).to(accelerator.device)
    pre, post = make_diffusion_pre_post_processors(cfg.policy, stats)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    train_loader = loader_for(dataset, cfg.batch_size, cfg.num_workers, shuffle=True)
    policy, optimizer, train_loader = accelerator.prepare(policy, optimizer, train_loader)
    validation_loader = None
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=False)
        (cfg.output_dir / 'training_config.yaml').write_text(config_path.read_text())
        dump(cfg.output_dir / 'resolved_training_config.json', {'launcher': settings['launcher'],
             'training': cfg.to_dict(), 'benchmark': settings['benchmark'], 'adapter': 'excavator_diffusion_v1'})
        dump(cfg.output_dir / 'resolved_settings.json', settings)
        dump(cfg.output_dir / 'dataset_contract.json', report)
        dump(cfg.output_dir / 'split_manifest.json', split)
        dump(cfg.output_dir / 'normalization_stats.json', stats)
        dump(cfg.output_dir / 'provenance.json', {
            'packages': {key: version(key) for key in ('lerobot', 'torch', 'torchvision', 'numpy', 'av')},
            'source_hashes': {str(p.relative_to(SUBMODULE)): file_hash(p)
                              for directory in ('excavator_vlm', 'scripts')
                              for p in sorted((SUBMODULE / directory).rglob('*.py'))},
            'training_episodes': episodes, 'world_size': accelerator.num_processes,
            'global_batch_size': cfg.batch_size * accelerator.num_processes,
            'normalization': 'selected training episodes only; fixed ImageNet image stats',
            'language_conditioning': False,
        })
        if settings['benchmark']['validation_freq']:
            validation = make_diffusion_dataset(cfg.dataset.repo_id, cfg.dataset.root, split['validation'],
                                                cfg.policy.horizon, cfg.dataset.video_backend)
            validation_loader = loader_for(validation, cfg.batch_size, 0)
    accelerator.wait_for_everyone()
    iterator = iter(train_loader)
    best_mae = float('inf')
    log = (cfg.output_dir / 'train.log').open('w') if accelerator.is_main_process else None
    started = time.perf_counter()
    try:
        for step in range(1, cfg.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, parts = policy(training_batch(pre, batch))
            finite = accelerator.reduce(torch.isfinite(loss.detach()).int(), reduction='sum')
            if finite.item() != accelerator.num_processes:
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            accelerator.backward(loss)
            norm = accelerator.clip_grad_norm_(policy.parameters(), settings['grad_clip_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            optimizer.step()
            if scheduler:
                scheduler.step()
            if step == 1 or step % cfg.log_freq == 0:
                values = torch.tensor([parts[k] for k in parts], device=accelerator.device)
                values = accelerator.reduce(values, reduction='mean').cpu().tolist()
                if log:
                    row = dict(step=step, **dict(zip(parts, values)), grad_norm=float(norm),
                               lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.perf_counter() - started)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    print(json.dumps(row), flush=True)
            save = step % cfg.save_freq == 0 or step == cfg.steps
            validate = settings['benchmark']['validation_freq'] and (
                step % settings['benchmark']['validation_freq'] == 0 or step == cfg.steps)
            if save or validate:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    raw = accelerator.unwrap_model(policy)
                    checkpoint = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(checkpoint, step, cfg, raw, optimizer, scheduler, pre, post)
                    update_last_checkpoint(checkpoint)
                    pretrained = checkpoint / 'pretrained_model'
                    dump(pretrained / 'benchmark_provenance.json', {
                        'adapter': 'excavator_diffusion_v1', 'dataset_fingerprint': report['fingerprint'],
                        'split_manifest': split, 'training_episodes': episodes,
                        'normalization_stats_sha256': file_hash(cfg.output_dir / 'normalization_stats.json'),
                    })
                    if validate:
                        metrics, _ = evaluate_loader(raw, pre, post, validation_loader,
                                                     [6., 2.5, 2.5, 6.],
                                                     settings['benchmark']['validation_max_batches'])
                        dump(checkpoint / 'validation.json', metrics)
                        mae = metrics['first_action']['mae']
                        if mae < best_mae:
                            best_mae = mae
                            dump(cfg.output_dir / 'best_checkpoint.json', {
                                'step': step, 'validation_first_action_mae_rad_s': mae,
                                'checkpoint': str(pretrained),
                                'partial_validation': settings['benchmark']['validation_max_batches'] > 0})
                        print(f'Validation step {step}: first-action MAE={mae:.6f} rad/s', flush=True)
                accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Diffusion sampling is stochastic (DDPM), so verify the reloaded
            # checkpoint by comparing weights directly rather than sampled output.
            raw = accelerator.unwrap_model(policy).eval()
            device = str(accelerator.device)
            restored, _, _ = load_diffusion(cfg.output_dir / 'checkpoints/last/pretrained_model', device)
            raw_sd = raw.state_dict()
            restored_sd = restored.state_dict()
            if raw_sd.keys() != restored_sd.keys():
                raise ValueError('Checkpoint keys differ after reload')
            for key in raw_sd:
                torch.testing.assert_close(restored_sd[key].float(), raw_sd[key].float(), rtol=1e-4, atol=1e-5)
            dump(cfg.output_dir / 'reload_check.json', {'status': 'passed', 'params': len(raw_sd)})
            print(f'Checkpoint: {cfg.output_dir / "checkpoints/last/pretrained_model"}', flush=True)
    finally:
        if log:
            log.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIGS / 'diffusion/full.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='Audit data and exclusively create the frozen split')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--episodes', help='Comma-separated training episode IDs (must be in train split)')
    parser.add_argument('--gpus', help='Comma-separated GPU indices')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-pretrained-backbone', action='store_true', help='For offline synthetic tests only')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    overrides = {key: value for key, value in {'steps': args.steps, 'batch_size': args.batch_size,
                 'num_workers': args.num_workers, 'output_dir': str(args.output.resolve()) if args.output else None}.items()
                 if value is not None}
    if args.device:
        overrides['policy'] = {'device': args.device}
        if args.device == 'cpu':
            overrides['launcher'] = {'gpu_ids': [0], 'mixed_precision': 'no'}
    if args.no_pretrained_backbone:
        overrides.setdefault('policy', {})['pretrained_backbone_weights'] = None
    if args.gpus:
        overrides.setdefault('launcher', {})['gpu_ids'] = [int(g) for g in args.gpus.split(',')]
    if args.episodes:
        overrides['dataset'] = {'episodes': [int(e) for e in args.episodes.split(',')]}
    settings = resolve_settings(args.config, overrides)
    setup_cache(settings)
    if not args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, settings['launcher']['gpu_ids']))
    cfg = build_train_config(settings)
    report, split, _, episodes = prepare(settings, create=args.prepare)
    cfg.dataset.episodes = episodes
    if args.prepare or args.dry_run:
        print(json.dumps({'training': cfg.to_dict(), 'launcher': settings['launcher'],
                          'dataset_audit': report, 'split_counts': {key: len(split[key]) for key in
                              ('train', 'validation', 'test')}}, indent=2))
        return
    if len(settings['launcher']['gpu_ids']) > 1 and not args.worker:
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
                   f'--num_processes={len(settings["launcher"]["gpu_ids"])}', '--num_machines=1',
                   f'--mixed_precision={settings["launcher"]["mixed_precision"]}', '--dynamo_backend=no',
                   str(SCRIPTS / 'train_diffusion.py'), *sys.argv[1:], '--worker']
        if args.output is None:
            command += ['--output', settings['output_dir']]
        subprocess.run(command, check=True)
    else:
        run(settings, cfg, args.config.resolve())


if __name__ == '__main__':
    main()
