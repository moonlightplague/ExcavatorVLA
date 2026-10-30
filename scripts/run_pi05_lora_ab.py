"""Owned, fail-fast dual-GPU smoke -> matched 2k A/B -> offline/Isaac evaluation.

No pilot/full training is started automatically. Existing runs/assets are never
overwritten. status.json and per-stage logs make the long-running job inspectable.
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
import time

SUBMODULE = Path(__file__).resolve().parents[1]
PROJECT = SUBMODULE.parent
sys.path.insert(0, str(SUBMODULE))


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = []
    for folder in ('policies/pi05', 'common', 'data', 'evaluation'):
        paths.extend((SUBMODULE / 'excavator_vlm' / folder).glob('*.py'))
    paths.extend((SUBMODULE / 'configs/pi05').glob('*.yaml'))
    paths.extend([Path(__file__), PROJECT / 'scripts/eval_smolvla_success.py',
                  PROJECT / 'ExcavatorVLA-smolvla/run_simulation.py'])
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def continuation_plan(parent, extra_steps):
    parent = Path(parent).resolve()
    previous = read(parent / 'status.json')
    if previous['status'] != 'completed' or extra_steps < 1:
        raise ValueError('Continuation requires a completed parent A/B and positive extra steps')
    runs, starts, protected = {}, {}, {}
    for label, expected_mode in (('expert', 'expert_only'), ('lora', 'vlm_lora_expert_full')):
        prior_runs = previous.get('continuation', {}).get('runs', {})
        directory = Path(prior_runs.get(label, parent / f'{label}_overfit')).resolve()
        latest = read(directory / 'last_checkpoint.json')
        checkpoint = Path(latest['checkpoint']).resolve()
        if checkpoint.parent.parent != (directory / 'checkpoints').resolve():
            raise ValueError('Continuation checkpoint is outside its owned training run')
        manifest = read(checkpoint / 'checkpoint.json')
        if manifest['step'] != latest['step'] or manifest['finetuning']['mode'] != expected_mode:
            raise ValueError('Continuation checkpoint step or fine-tuning mode differs')
        if not (checkpoint.parent / 'training_state.pt').is_file():
            raise FileNotFoundError('Optimizer/RNG state missing; cannot continue exactly')
        provenance = read(directory / 'provenance.json')
        if provenance['batch_contract'] != {'world_size': 2, 'batch_size': 1, 'accumulation': 4}:
            raise ValueError('Expected the matched two-GPU effective-batch-8 runs')
        runs[label], starts[label] = str(directory), latest['step']
        protected[label] = [str(p) for p in sorted((directory / 'checkpoints').glob('*/pretrained_model'))
                            if (p / 'checkpoint.json').is_file()]
        settings = read(directory / 'resolved_settings.json')
        end = latest['step'] + extra_steps
        saves = {end}
        for frequency in (settings['save_freq'], settings['benchmark']['validation_freq']):
            if frequency:
                saves.update(range((latest['step'] // frequency + 1) * frequency, end + 1, frequency))
        if len(protected[label]) + len(saves) > 6:
            raise ValueError('More checkpoints than continuation retention can protect; stop without pruning')
    if len(set(starts.values())) != 1:
        raise ValueError('A/B starting steps differ; refusing an unmatched continuation')
    start = next(iter(starts.values()))
    if previous.get('optimizer_steps_each', start) != start:
        raise ValueError('Latest checkpoints no longer match the parent comparison; use the latest completed continuation')
    return {'parent': str(parent), 'runs': runs, 'start_step': start,
            'target_step': start + extra_steps, 'extra_steps': extra_steps,
            'protected_checkpoints': protected}


class Experiment:
    def __init__(self, args):
        self.args, self.root = args, args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'logs').mkdir()
        self.state = {'status': 'running', 'phase': 'preflight', 'pid': os.getpid(),
                      'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_ids': args.gpus,
                      'optimizer_steps_each': args.steps, 'effective_global_batch_each': 8,
                      'source_hashes': source_hashes(), 'stages': []}
        self.environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                            'TOKENIZERS_PARALLELISM': 'false', 'NCCL_P2P_DISABLE': '1',
                            'NCCL_IB_DISABLE': '1', 'PYTHONUNBUFFERED': '1'}
        self.children = []
        self.save()

    def save(self):
        path = self.root / 'status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def idle(self):
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,uuid',
                                            '--format=csv,noheader,nounits'], text=True)
            devices = {int(x[0]): (int(x[1]), int(x[2]), int(x[3]), x[4].strip()) for x in
                       (line.split(',') for line in rows.strip().splitlines())}
            if getattr(self.args, 'allow_shared_gpus', False):
                ready = all(gpu in devices and devices[gpu][1] - devices[gpu][0] >= 20 * 1024
                            for gpu in self.args.gpus)
            else:
                processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                      '--format=csv,noheader,nounits'], text=True)
                computing = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
                ready = all(gpu in devices and devices[gpu][0] < 1000 and devices[gpu][2] < 10
                            and devices[gpu][3] not in computing for gpu in self.args.gpus)
            if ready:
                return
            print(f'Waiting for GPU availability (sharing={getattr(self.args, "allow_shared_gpus", False)}): {devices}', flush=True)
            time.sleep(10)

    def check_sources(self):
        if source_hashes() != self.state['source_hashes']:
            raise RuntimeError('Runtime code/config changed during A/B; refusing an unmatched comparison')
        if shutil.disk_usage(self.root).free < 25 * 1024**3:
            raise RuntimeError('Less than 25 GiB free; refusing the next stage without deleting user files')

    def run(self, name, commands):
        """Run a list of (argv, env overrides) concurrently; fail on any error."""
        self.check_sources()
        self.idle()
        self.state['phase'] = name
        record = {'name': name, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'commands': [], 'pids': []}
        self.state['stages'].append(record)
        opened = []
        self.children = []
        started = time.monotonic()
        try:
            for index, (command, overrides) in enumerate(commands):
                log_path = self.root / 'logs' / f'{name}_{index}.log'
                handle = log_path.open('x')
                opened.append(handle)
                process = subprocess.Popen(command, cwd=PROJECT, env={**self.environment, **overrides},
                                           stdout=handle, stderr=subprocess.STDOUT)
                self.children.append(process)
                record['commands'].append({'argv': list(map(str, command)), 'log': str(log_path)})
                record['pids'].append(process.pid)
            self.save()
            print(f'Started {name}: {record["pids"]}', flush=True)
            while any(process.poll() is None for process in self.children):
                failures = [process.returncode for process in self.children if process.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f'{name} failed with exit codes {failures}; inspect stage logs')
                time.sleep(5)
            codes = [process.returncode for process in self.children]
            if any(codes):
                raise RuntimeError(f'{name} failed with exit codes {codes}; inspect stage logs')
            record.update(status='completed', exit_codes=codes, elapsed_seconds=time.monotonic() - started)
            self.save()
            print(f'Completed {name}', flush=True)
        finally:
            for process in self.children:
                if process.poll() is None:
                    process.terminate()
            for process in self.children:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for handle in opened:
                handle.close()
            self.children = []

    def train_command(self, config, output, steps=None, resume=False):
        command = [sys.executable, '-u', str(SUBMODULE / 'scripts/train_pi05.py'),
                   '--config', str(SUBMODULE / 'configs/pi05' / config),
                   '--gpus', ','.join(map(str, self.args.gpus)), '--output', str(output)]
        if steps is not None:
            command += ['--steps', str(steps)]
        if resume:
            command += ['--resume', str(output)]
            if self.args.resume_lr is not None:
                command += ['--resume-lr', str(self.args.resume_lr)]
        return command, {}

    def offline_command(self, checkpoint, output, gpu, smoke=False):
        command = [sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_offline.py'),
                   '--checkpoint', str(checkpoint), '--dataset', str(PROJECT / 'data/export'),
                   '--batch-size', '1', '--output', str(output)]
        command += ['--split', 'validation', '--max-batches', '1'] if smoke else ['--split', 'train', '--episodes', '1']
        return command, {'CUDA_VISIBLE_DEVICES': str(gpu)}

    def execute(self):
        if self.args.resume_parent:
            return self.continue_existing()
        gates = {}
        for label in ('expert', 'replan1', 'replan3'):
            folder = getattr(self.args, f'gate_{label}').resolve()
            result = read(folder / 'episodes/episode_0001/episode_status.json')
            if result['status'] != 'completed' or result.get('error'):
                raise RuntimeError(f'Incomplete diagnostic gate: {label}')
            if label == 'expert' and not result['success']:
                raise RuntimeError('Current-simulator expert replay failed; no training started')
            gates[label] = {'path': str(folder), 'result': result,
                            'simulator_sha256': read(folder / 'provenance.json')['source_files']
                            ['ExcavatorVLA-smolvla/run_simulation.py']}
        if len({value['simulator_sha256'] for value in gates.values()}) != 1:
            raise RuntimeError('Diagnostic gates used different simulator source versions')
        self.state['diagnostic_gates'] = gates
        self.save()
        smoke = self.args.smoke_run.resolve() if self.args.smoke_run else self.root / 'lora_smoke'
        if not self.args.smoke_run:
            self.run('lora_ddp_smoke', [self.train_command('lora_smoke.yaml', smoke)])
        self.state['smoke_run'] = str(smoke)
        if read(smoke / 'zero_lora_check.json')['status'] != 'passed':
            raise RuntimeError('Zero-LoRA identity check failed')
        if read(smoke / 'reload_check.json')['status'] != 'passed':
            raise RuntimeError('Smoke delta reload failed')
        audit = read(smoke / 'first_update_check_000001.json')
        if not audit['groups']['lora_B']['changed']:
            raise RuntimeError('LoRA received no effective update')
        if read(smoke / 'provenance.json')['batch_contract']['world_size'] != 2:
            raise RuntimeError('Smoke did not use both GPUs')
        smoke_eval = self.root / 'lora_smoke_reload'
        self.run('smoke_independent_reload', [self.offline_command(smoke, smoke_eval, self.args.gpus[0], True)])
        a = read(smoke / 'checkpoints/000002/validation.json')['first_action']['mae']
        b = read(smoke_eval / 'metrics.json')['first_action']['mae']
        if abs(a - b) > 1e-6:
            raise RuntimeError(f'Independent smoke reload MAE differs: {a} vs {b}')
        runs = {'expert': self.root / 'expert_overfit', 'lora': self.root / 'lora_overfit'}
        for label, config in (('expert', 'expert_ab_overfit.yaml'), ('lora', 'lora_overfit.yaml')):
            self.run(f'{label}_ddp_overfit', [self.train_command(config, runs[label], self.args.steps)])
            provenance = read(runs[label] / 'provenance.json')
            if provenance['nominal_global_batch'] != 8 or provenance['batch_contract']['world_size'] != 2:
                raise RuntimeError('A/B effective batch or GPU count differs')
            if read(runs[label] / 'last_checkpoint.json')['step'] != self.args.steps:
                raise RuntimeError('Training did not reach the matched update budget')
            if read(runs[label] / 'reload_check.json')['status'] != 'passed':
                raise RuntimeError('A/B checkpoint roundtrip failed')
        if read(runs['expert'] / 'normalization_stats.json') != read(runs['lora'] / 'normalization_stats.json'):
            raise RuntimeError('A/B normalization differs')
        self.evaluate_and_finish(runs)

    def continue_existing(self):
        plan = continuation_plan(self.args.resume_parent, self.args.extra_steps)
        if plan['target_step'] != self.args.steps:
            raise RuntimeError('Starting checkpoint changed after continuation was planned')
        previous = read(Path(plan['parent']) / 'status.json')
        # Scheduler/CLI changes are the explicitly requested continuation, not a
        # change to model math, observation processing, loss or simulator physics.
        allowed = {str(Path(__file__).resolve().relative_to(PROJECT)),
                   'ExcavatorVLA-vlm/excavator_vlm/policies/pi05/train.py',
                   'ExcavatorVLA-vlm/excavator_vlm/policies/pi05/config.py'}
        changed = [name for name, digest in previous['source_hashes'].items()
                   if self.state['source_hashes'].get(name) != digest and name not in allowed]
        if changed:
            raise RuntimeError(f'Non-scheduler runtime changed since the original A/B: {changed}')
        self.state.update(continuation=plan, resume_learning_rate=self.args.resume_lr,
                          optimizer_moments_and_rng_preserved=True,
                          learning_rate_policy='explicit constant override' if self.args.resume_lr is not None
                                               else 'saved schedule unchanged',
                          sharing_allowed=self.args.allow_shared_gpus)
        self.save()
        runs = {label: Path(directory) for label, directory in plan['runs'].items()}
        for label, config in (('expert', 'expert_ab_continue.yaml'), ('lora', 'lora_continue.yaml')):
            self.run(f'{label}_ddp_continue',
                     [self.train_command(config, runs[label], self.args.steps, resume=True)])
            if read(runs[label] / 'last_checkpoint.json')['step'] != self.args.steps:
                raise RuntimeError('Continuation did not reach the requested target step')
            if read(runs[label] / 'reload_check.json')['status'] != 'passed':
                raise RuntimeError('Continued checkpoint reload check failed')
            for path in plan['protected_checkpoints'][label]:
                if not (Path(path) / 'model.safetensors').is_file():
                    raise RuntimeError('A protected pre-continuation checkpoint was removed')
        self.evaluate_and_finish(runs)
        old = read(Path(plan['parent']) / 'comparison.json')
        new = read(self.root / 'comparison.json')
        (self.root / 'step_comparison.json').write_text(json.dumps(
            {'before_step': plan['start_step'], 'after_step': plan['target_step'],
             'resume_lr': self.args.resume_lr, 'before': old, 'after': new},
            indent=2, allow_nan=False) + '\n')

    def evaluate_and_finish(self, runs):
        checkpoints = {label: path / 'checkpoints' / f'{self.args.steps:06d}' / 'pretrained_model'
                       for label, path in runs.items()}
        offline = {label: self.root / f'{label}_train_ep1' for label in runs}
        self.run('ab_offline', [self.offline_command(checkpoints[label], offline[label], gpu)
                               for label, gpu in zip(runs, self.args.gpus)])
        commands = []
        closed = {label: self.root / f'{label}_reference_video' for label in runs}
        for label, gpu in zip(runs, self.args.gpus):
            commands.append(([sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_success.py'),
                '--checkpoint', str(checkpoints[label]), '--dataset', str(PROJECT / 'data/export'),
                '--reference-episode', str(self.args.reference.resolve()), '--episodes', '1', '--steps', '350',
                '--min-particles', '100', '--replan-interval', '3', '--seed', '2', '--gpus', str(gpu),
                '--workers', '1', '--cpu-threads', '8', '--video', '--output', str(closed[label])], {}))
        self.run('ab_reference_video', commands)
        comparison = {label: {'offline': read(offline[label] / 'metrics.json'),
                             'closed_loop': read(closed[label] / 'episodes/episode_0001/episode_status.json')}
                      for label in runs}
        (self.root / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
        self.state.update(status='completed', phase='completed',
                          note='Single-training-episode A/B only. No pilot/full run automatically started.')
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-expert', type=Path)
    parser.add_argument('--gate-replan1', type=Path)
    parser.add_argument('--gate-replan3', type=Path)
    parser.add_argument('--reference', type=Path, default=PROJECT /
        'data/raw/run_20261013_172047_collection_20261013_172021_158092_r1_w0/episode_000003')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume-parent', type=Path, help='Continue both completed arms in place; evaluation goes to new --output')
    parser.add_argument('--extra-steps', type=int, default=1000)
    parser.add_argument('--resume-lr', type=float,
                        help='Keep optimizer/RNG state but explicitly use this constant LR during continuation')
    parser.add_argument('--allow-shared-gpus', action='store_true',
                        help='User-approved sharing; still require 20 GiB free on each selected GPU')
    parser.add_argument('--smoke-run', type=Path, help='Reuse a completed real dual-GPU LoRA smoke run')
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus.split(',')]
    if args.resume_parent:
        if args.steps is not None:
            parser.error('Use --extra-steps with --resume-parent, not --steps')
        args.steps = continuation_plan(args.resume_parent, args.extra_steps)['target_step']
    else:
        if not all((args.gate_expert, args.gate_replan1, args.gate_replan3)):
            parser.error('Fresh experiments require expert/replan1/replan3 gate paths')
        if args.resume_lr is not None:
            parser.error('--resume-lr requires --resume-parent')
        args.steps = args.steps if args.steps is not None else 2000
    if args.resume_lr is not None and (not 0 < args.resume_lr < float('inf')):
        parser.error('--resume-lr must be finite and positive')
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2 or args.steps < 1:
        parser.error('Exactly two distinct GPUs and a positive step budget are required')
    experiment = Experiment(args)
    try:
        experiment.execute()
    except BaseException as exc:
        experiment.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        experiment.save()
        raise


if __name__ == '__main__':
    main()
