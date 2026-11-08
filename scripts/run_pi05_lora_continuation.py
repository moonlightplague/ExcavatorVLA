"""Continue one LoRA/expert run exactly, then compare before/after checkpoints.

Uses the existing dual-GPU launcher, optimizer/RNG resume and benchmark entrypoints.
The immutable starting checkpoint is hard-linked before rolling retention runs.
No other job, policy/controller implementation or dataset is changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_pi05_lora_ab import Experiment, PROJECT, SUBMODULE, read


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_sources(config):
    paths = {Path(__file__).resolve(), SUBMODULE / 'scripts/train_pi05.py', config}
    for folder in ('policies/pi05', 'common', 'data', 'evaluation'):
        paths.update((SUBMODULE / 'excavator_vlm' / folder).glob('*.py'))
    paths.update(SUBMODULE / 'configs/pi05' / name for name in
                 ('full.yaml', 'overfit.yaml', 'expert_ab_overfit.yaml', 'lora_overfit.yaml'))
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}


def eval_sources():
    paths = set((PROJECT / 'ExcavatorVLA-smolvla/excavator_common').rglob('*.py'))
    paths.update([PROJECT / 'ExcavatorVLA-smolvla/run_simulation.py',
                  PROJECT / 'scripts/eval_smolvla_success.py', PROJECT / 'scripts/eval_video.py',
                  PROJECT / 'scripts/success_eval_runtime.py', PROJECT / 'run_isaac_python.sh',
                  SUBMODULE / 'scripts/eval_pi05_success.py', SUBMODULE / 'scripts/eval_pi05_offline.py'])
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}


class Continuation(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.train_hashes = train_sources(args.config)
        self.eval_hashes = None
        self.state.pop('optimizer_steps_each', None)
        self.state.pop('effective_global_batch_each', None)
        self.state.update(training_run=str(args.run), target_step=args.steps,
                          global_batch=8, learning_rate=args.resume_lr,
                          fine_tuning='vlm_lora_expert_full',
                          evaluation_soil_backend=args.soil_backend,
                          training_source_hashes=self.train_hashes)
        self.save()

    def check_sources(self):
        if train_sources(self.args.config) != self.train_hashes:
            raise RuntimeError('Training/processing source changed during this continuation')
        if self.eval_hashes is not None and eval_sources() != self.eval_hashes:
            raise RuntimeError('Simulator/evaluator changed between before/after evaluations')
        if shutil.disk_usage(self.root).free < 25 * 1024**3:
            raise RuntimeError('Less than 25 GiB free; no user files were deleted')

    def run(self, name, commands):
        try:
            super().run(name, commands)
        except BaseException as exc:
            if self.state['stages'] and self.state['stages'][-1]['name'] == name:
                self.state['stages'][-1].update(status='failed', error=f'{type(exc).__name__}: {exc}',
                                               finished_utc=now())
            self.save()
            raise

    def train_argv(self, checkpoint):
        return [sys.executable, '-u', str(SUBMODULE / 'scripts/train_pi05.py'),
                '--config', str(self.args.config), '--gpus', ','.join(map(str, self.args.gpus)),
                '--output', str(self.args.run), '--steps', str(self.args.steps),
                '--resume', str(checkpoint), '--resume-lr', str(self.args.resume_lr)]

    def execute(self):
        args = self.args
        # Check the evaluation dependency before spending GPU time on training.
        sys.path.insert(0, str(PROJECT / 'scripts'))
        sys.path.insert(0, str(PROJECT / 'ExcavatorVLA-smolvla'))
        from excavator_diagnostics import reference_episode
        reference_episode(args.reference, 10)
        latest = read(args.run / 'last_checkpoint.json')
        checkpoint = Path(latest['checkpoint']).resolve()
        if checkpoint.parents[2] != args.run or checkpoint.parent.parent != args.run / 'checkpoints':
            raise ValueError('Starting checkpoint is outside the requested training run')
        manifest = read(checkpoint / 'checkpoint.json')
        if latest['step'] != args.start_step or manifest['step'] != args.start_step:
            raise ValueError('Latest checkpoint is no longer the requested starting step')
        if manifest['finetuning']['mode'] != 'vlm_lora_expert_full':
            raise ValueError('Expected VLM LoRA plus full expert fine-tuning')
        if not (checkpoint.parent / 'training_state.pt').is_file():
            raise FileNotFoundError('Optimizer/RNG state is required for exact continuation')
        self.state.update(start_step=args.start_step, extra_steps=args.steps - args.start_step,
                          source_checkpoint=str(checkpoint),
                          source_checkpoint_sha256=manifest['delta_sha256'])
        self.save()
        self.run('preflight', [(self.train_argv(checkpoint) + ['--dry-run'], {})])
        text = (self.root / 'logs/preflight_0.log').read_text()
        report = json.loads(text[text.index('{\n'):])
        provenance = read(checkpoint / 'benchmark_provenance.json')
        if (report['dataset_audit']['fingerprint'] != provenance['dataset_fingerprint']
                or report['training_episodes'] != provenance['training_episodes']
                or provenance['batch_contract'] != {'world_size': 2, 'batch_size': 1, 'accumulation': 4}):
            raise ValueError('Continuation dataset, episode set or batch contract differs')
        if file_hash(checkpoint / 'model.safetensors') != manifest['delta_sha256']:
            raise ValueError('Starting checkpoint checksum differs from manifest')
        baseline = self.root / f'baseline_step{args.start_step}'
        shutil.copytree(checkpoint.parent, baseline, copy_function=os.link)
        before = baseline / 'pretrained_model'
        self.state.update(baseline_checkpoint=str(before),
                          baseline_storage='hard links to immutable checkpoint files; retained after original pruning',
                          data_preflight=report['dataset_audit'])
        self.save()
        self.run('lora_ddp_continue', [(self.train_argv(checkpoint), {})])
        last = read(args.run / 'last_checkpoint.json')
        after = Path(last['checkpoint']).resolve()
        if last['step'] != args.steps or read(after / 'checkpoint.json')['step'] != args.steps:
            raise RuntimeError('Training did not reach the requested optimizer step')
        if read(args.run / 'reload_check.json')['status'] != 'passed':
            raise RuntimeError('Continued checkpoint roundtrip failed')
        self.state.update(training_completed=True, trained_checkpoint=str(after),
                          training_completed_utc=now())
        # Unrelated simulator edits during training do not change training math.
        # Pin the current simulator once for both newly run evaluations.
        self.eval_hashes = eval_sources()
        self.state['evaluation_source_hashes'] = self.eval_hashes
        self.save()
        checkpoints = {'before': before, 'after': after}
        outputs = {label: self.root / f'{label}_train_ep1' for label in checkpoints}
        self.run('offline_train_episode1', [
            self.offline_command(checkpoints[label], outputs[label], gpu)
            for label, gpu in zip(checkpoints, args.gpus)])
        import numpy as np
        arrays = [np.load(outputs[label] / 'actions.npz') for label in checkpoints]
        for key in ('target', 'padding', 'supervision', 'recovery', 'episode_index', 'frame_index'):
            np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
        offline = {label: read(outputs[label] / 'metrics.json') for label in checkpoints}
        comparison = {'start_step': args.start_step, 'end_step': args.steps,
                      'lr': args.resume_lr, 'execution_horizon': args.execution_horizon,
                      'scope': 'Same training episode1, same raw episode3 reference scene; not generalization.',
                      'offline': offline, 'closed_loop': {}}
        failures = []
        # Evaluate independently so one infrastructure failure cannot terminate
        # the other policy's rollout. Policy task failure itself is a valid result.
        for label, gpu in zip(checkpoints, args.gpus):
            output = self.root / f'{label}_reference_video'
            command = [sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_success.py'),
                '--checkpoint', str(checkpoints[label]), '--dataset', str(PROJECT / 'data/export'),
                '--reference-episode', str(args.reference), '--episodes', '1', '--steps', '350',
                '--min-particles', '100', '--seed', '2', '--replan-interval', str(args.execution_horizon),
                '--gpus', str(gpu), '--workers', '1', '--cpu-threads', '8', '--headless',
                '--fresh-simulator', '--render-every-physics-step', '--video',
                '--soil-backend', args.soil_backend, '--sand-walls-during-settle',
                '--output', str(output)]
            try:
                self.run(f'{label}_closed_loop_video', [(command, {})])
                result = read(output / 'episodes/episode_0001/episode_status.json')
                if result['status'] != 'completed' or result.get('error') or result.get('video_error'):
                    raise RuntimeError(f'Incomplete closed-loop evaluation: {result.get("error")}')
                subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', result['video_path'],
                                '-threads', '1', '-f', 'null', '-'], check=True)
                comparison['closed_loop'][label] = result
            except Exception as exc:
                failures.append(f'{label}: {type(exc).__name__}: {exc}')
                comparison['closed_loop'][label] = {'status': 'incomplete', 'error': str(exc)}
            (self.root / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
        if failures:
            raise RuntimeError('Training and offline comparison completed; video evaluation incomplete: '
                               + '; '.join(failures))
        self.state.update(status='completed', phase='completed', completed_utc=now(),
                          comparison=str(self.root / 'comparison.json'))
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=SUBMODULE / 'configs/pi05/lora_continue5000.yaml')
    parser.add_argument('--start-step', type=int, default=3000)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--resume-lr', type=float, default=2.5e-5)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--execution-horizon', type=int, default=3)
    parser.add_argument('--soil-backend', choices=('newton_mpm', 'physx'), default='newton_mpm',
                        help='Explicit evaluation soil backend; legacy successful reference uses Newton MPM.')
    parser.add_argument('--reference', type=Path, default=PROJECT /
        'data/raw/run_20261013_172047_collection_20261013_172021_158092_r1_w0/episode_000003')
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(',')]
    for key in ('output', 'run', 'config', 'reference'):
        setattr(args, key, getattr(args, key).resolve())
    if (len(set(args.gpus)) != 2 or args.steps <= args.start_step
            or not 1 <= args.execution_horizon <= 10 or not 0 < args.resume_lr < float('inf')):
        parser.error('Require two distinct GPUs, increasing steps, a 1..10 horizon and finite positive LR')
    experiment = Continuation(args)
    try:
        experiment.execute()
    except BaseException as exc:
        experiment.state.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_utc=now())
        experiment.save()
        raise


if __name__ == '__main__':
    main()
