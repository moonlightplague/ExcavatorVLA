"""Compare saved PI05 checkpoints in the same recorded simulation scenes."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_pi05_singlecycle_pilot import PROJECT, SUBMODULE, digest, source_manifest


DEFAULT_STEPS = [2000, 4000, 6000, 8000, 10000]
METRICS = ['success', 'payload_particles', 'scooped_particles', 'spillage_particles',
           'spillage_fraction', 'cycle_time_seconds', 'failure_mode',
           'policy_latency_ms_mean', 'policy_latency_ms_p95', 'fps', 'policy_steps_executed']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_plan(args):
    settings = read(args.train_dir / 'resolved_settings.json')
    provenance = read(args.train_dir / 'provenance.json')
    dataset = Path(settings['dataset']['root']).resolve()
    backend = provenance.get('collection_backend', {}).get('soil_backend')
    if backend not in ('physx', 'newton_mpm'):
        raise ValueError('Training run needs explicit verified collection-backend provenance')
    if read(dataset / 'backend_provenance.json')['soil_backend'] != backend:
        raise ValueError('Dataset backend differs from the training provenance')
    if args.replan_interval > settings['policy']['chunk_size']:
        raise ValueError('Replan interval exceeds the trained action chunk')
    split = read(args.train_dir / 'split_manifest.json')
    jobs = []
    for step in args.checkpoint_steps:
        checkpoint = args.train_dir / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
        for episode in args.reference_episodes:
            reference = dataset / 'references' / f'episode_{episode:04d}.json'
            setup = read(reference)
            if setup.get('provenance', {}).get('soil_backend') != backend:
                raise ValueError(f'Reference {episode} has a different or unknown backend')
            subset = next((name for name in ('train', 'validation', 'test')
                           if episode in split[name]), None)
            if subset is None:
                raise ValueError(f'Reference {episode} is absent from the frozen split')
            output = args.output / f'step_{step:06d}' / f'reference_{episode:04d}'
            command = [sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_success.py'),
                       '--checkpoint', str(checkpoint), '--dataset', str(dataset),
                       '--reference-episode', str(reference), '--episodes', '1',
                       '--steps', str(args.rollout_steps), '--min-particles', str(args.min_particles),
                       '--seed', str(args.seed), '--replan-interval', str(args.replan_interval),
                       '--gpus', str(args.gpu), '--workers', '1',
                       '--cpu-threads', str(args.cpu_threads), '--headless', '--fresh-simulator',
                       '--render-every-physics-step', '--video', '--soil-backend', backend,
                       '--sand-walls-during-settle' if backend == 'physx'
                       else '--no-sand-walls-during-settle', '--output', str(output)]
            jobs.append(dict(checkpoint_step=step, checkpoint=str(checkpoint),
                             reference_episode=episode, reference_split=subset,
                             source_export_episode=setup['provenance'].get('source_export_episode'),
                             reference=str(reference), output=str(output), command=command,
                             status='pending', checkpoint_available=(checkpoint / 'checkpoint.json').is_file()))
    return dict(training_run=str(args.train_dir), dataset=str(dataset), soil_backend=backend,
                dataset_fingerprint=provenance['dataset_fingerprint'], gpu=args.gpu,
                checkpoint_steps=args.checkpoint_steps, reference_episodes=args.reference_episodes,
                seed=args.seed, replan_interval=args.replan_interval, rollout_steps=args.rollout_steps,
                min_particles=args.min_particles, jobs=jobs)


def gpu_idle(gpu):
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(',') for line in result.splitlines() if line.strip()]
    device = next((row for row in rows if int(row[0]) == gpu), None)
    if device is None:
        raise ValueError(f'Physical GPU {gpu} does not exist')
    result = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in result.splitlines() if line.strip()}
    return device[1].strip() not in busy and int(device[2]) >= 20 * 1024 and int(device[3]) < 10


def wait_for_parent(path, wait):
    if path is None:
        return
    announced = False
    while True:
        parent = read(path / 'status.json')
        if parent['status'] == 'completed':
            return
        if parent['status'] in ('failed', 'cancelled'):
            raise RuntimeError('Parent experiment failed; inspect it before launching checkpoint comparisons')
        if not wait:
            raise RuntimeError('Parent experiment is still running; use --wait or wait for it to finish')
        pid = parent.get('pid')
        if pid is not None and not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Parent controller exited without a final status')
        if not announced:
            print('Waiting for training and its scheduled evaluations to finish.', flush=True)
            announced = True
        time.sleep(10)


def wait_for_gpu(gpu, wait):
    announced = False
    while not gpu_idle(gpu):
        if not wait:
            raise RuntimeError(f'GPU {gpu} is busy or has less than 20 GiB free')
        if not announced:
            print(f'Waiting for GPU {gpu} to be idle with at least 20 GiB free.', flush=True)
            announced = True
        time.sleep(10)


def check_checkpoints(plan):
    """Require every requested save; never silently skip missing checkpoints."""
    run = Path(plan['training_run'])
    stats_hash = digest(run / 'normalization_stats.json')
    baseline = None
    for step in plan['checkpoint_steps']:
        checkpoint = run / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
        manifest = read(checkpoint / 'checkpoint.json')
        if manifest['step'] != step or manifest['format'] != 'excavator_pi05_trainable_delta_v1':
            raise ValueError(f'Checkpoint {step} manifest is inconsistent')
        if digest(checkpoint / 'normalization_stats.json') != stats_hash:
            raise ValueError(f'Checkpoint {step} uses different normalization statistics')
        if read(checkpoint / 'benchmark_provenance.json')['dataset_fingerprint'] != plan['dataset_fingerprint']:
            raise ValueError(f'Checkpoint {step} comes from different training data')
        signature = (manifest['base_sha256'], manifest.get('precision_contract'), manifest['finetuning'])
        if baseline is not None and signature != baseline:
            raise ValueError('Checkpoints differ in base model, precision or fine-tuning method')
        baseline = signature
        if digest(checkpoint / 'model.safetensors') != manifest['delta_sha256']:
            raise ValueError(f'Checkpoint {step} weight checksum differs')
        for job in plan['jobs']:
            if job['checkpoint_step'] == step:
                job.update(checkpoint_available=True, delta_sha256=manifest['delta_sha256'])


def save_summary(output, rows):
    write(output / 'summary.json', rows)
    fields = ['checkpoint_step', 'reference_episode', 'reference_split', 'source_export_episode',
              'soil_backend', *METRICS, 'checkpoint', 'video_path', 'video_duration_seconds']
    temporary = output / 'summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / 'summary.csv')


def run(args, plan):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    snapshot = source_manifest(args.train_dir / 'training_config.yaml')
    snapshot[str(Path(__file__).resolve().relative_to(PROJECT))] = digest(Path(__file__))
    inputs = [args.train_dir / name for name in ('resolved_settings.json', 'provenance.json',
                                                'split_manifest.json', 'normalization_stats.json')]
    inputs.extend(Path(job['reference']) for job in plan['jobs'])
    input_hashes = {str(path): digest(path) for path in set(inputs)}
    state = dict(plan, status='queued', phase='waiting_for_parent', pid=os.getpid(),
                 created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=snapshot,
                 input_hashes=input_hashes,
                 wait_for_experiment=str(args.wait_for_experiment) if args.wait_for_experiment else None)
    rows, process = [], None

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'status.json', state)

    def unchanged():
        for name, expected in snapshot.items():
            if digest(PROJECT / name) != expected:
                raise RuntimeError(f'Evaluation source changed: {name}')
        for name, expected in input_hashes.items():
            if digest(Path(name)) != expected:
                raise RuntimeError(f'Evaluation input changed: {name}')

    save()
    try:
        wait_for_parent(args.wait_for_experiment, args.wait)
        state['phase'] = 'checking_dataset_and_checkpoints'
        save()
        unchanged()
        from excavator_vlm.data.contract import audit_dataset
        report, _ = audit_dataset(Path(plan['dataset']), read(args.train_dir / 'resolved_settings.json')['policy']['chunk_size'])
        if (report['fingerprint'] != plan['dataset_fingerprint']
                or report['collection_backend']['soil_backend'] != plan['soil_backend']):
            raise ValueError('Current dataset/backend differs from the saved training provenance')
        check_checkpoints(plan)
        state['dataset_audit'] = report
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
                       'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        for job in plan['jobs']:
            state.update(status='queued', phase='waiting_for_gpu', current_step=job['checkpoint_step'],
                         current_reference=job['reference_episode'])
            save()
            wait_for_gpu(args.gpu, args.wait)
            unchanged()
            job['status'] = 'running'
            state.update(status='running', phase='evaluating')
            save()
            label = f'{job["checkpoint_step"]:06d}_reference_{job["reference_episode"]:04d}'
            print(f'Evaluating {label} on GPU {args.gpu}', flush=True)
            with (args.output / 'logs' / f'{label}.log').open('x') as log:
                process = subprocess.Popen(job['command'], cwd=PROJECT, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                save()
                job['exit_code'] = process.wait()
            process = None
            if job['exit_code']:
                job['status'] = 'failed'
                raise RuntimeError(f'{label} evaluation failed; see its log')
            result = read(Path(job['output']) / 'episodes/episode_0001/episode_status.json')
            if (result['status'] != 'completed' or result.get('error')
                    or result.get('execution_source') != 'model'):
                raise RuntimeError(f'{label} did not complete a model rollout')
            video = Path(result['video_path'])
            if not video.is_file() or not video.stat().st_size:
                raise RuntimeError(f'{label} video is missing or empty')
            duration = float(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video)], text=True).strip())
            row = {key: job[key] for key in ('checkpoint_step', 'reference_episode', 'reference_split',
                                           'source_export_episode', 'checkpoint')}
            row.update(soil_backend=plan['soil_backend'], video_path=str(video), video_duration_seconds=duration)
            row.update({name: result.get(name) for name in METRICS})
            rows.append(row)
            save_summary(args.output, rows)
            job.update(status='completed', success=result['success'], video_path=str(video))
            save()
        state.update(status='completed', phase='completed', completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for job in plan['jobs']:
            if job['status'] == 'running':
                job['status'] = 'failed'
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-steps', type=int, nargs='+', default=DEFAULT_STEPS)
    parser.add_argument('--reference-episodes', type=int, nargs='+', default=[0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rollout-steps', type=int, default=350)
    parser.add_argument('--replan-interval', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--min-particles', type=int, default=100)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--wait-for-experiment', type=Path,
                        help='Wait for this controller, including its scheduled evaluations; auto-detects the training parent.')
    parser.add_argument('--wait', action=argparse.BooleanOptionalAction, default=True,
                        help='Wait for parent completion and an idle GPU; --no-wait fails if either is busy.')
    parser.add_argument('--dry-run', action='store_true', help='Print all commands without writing files or using CUDA.')
    args = parser.parse_args(argv)
    if (min(args.checkpoint_steps) < 1 or min(args.reference_episodes) < 0 or args.gpu < 0 or args.seed < 0
            or min(args.rollout_steps, args.replan_interval, args.min_particles, args.cpu_threads) < 1):
        parser.error('Steps/counts must be positive; episode IDs/GPU/seed must be nonnegative')
    if (len(set(args.checkpoint_steps)) != len(args.checkpoint_steps)
            or len(set(args.reference_episodes)) != len(args.reference_episodes)):
        parser.error('Duplicate checkpoint steps or reference episodes')
    args.checkpoint_steps = sorted(args.checkpoint_steps)
    args.train_dir = args.train_dir.resolve()
    args.output = args.output.resolve()
    if args.wait_for_experiment is None and (args.train_dir.parent / 'status.json').is_file():
        args.wait_for_experiment = args.train_dir.parent
    if args.wait_for_experiment is not None:
        args.wait_for_experiment = args.wait_for_experiment.resolve()
    return args


def main():
    args = parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    run(args, plan)


if __name__ == '__main__':
    main()
