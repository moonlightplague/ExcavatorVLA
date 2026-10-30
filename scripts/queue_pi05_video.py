"""Queue one recorded evaluation behind an existing GPU experiment.

This standalone queue does not edit or interrupt the active training runner.
Its status/log are separate from the evaluator output, which refuses overwrite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SUBMODULE = Path(__file__).resolve().parents[1]
PROJECT = SUBMODULE.parent


def read(path):
    return json.loads(Path(path).read_text())


def gpu_available(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,uuid',
                                    '--format=csv,noheader,nounits'], text=True)
    values = {int(parts[0]): (int(parts[1]), int(parts[2]), parts[3].strip())
              for parts in (line.split(',') for line in rows.strip().splitlines())}
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True)
    occupied = {line.split(',')[0].strip() for line in active.splitlines() if line.strip()}
    return (gpu in values and values[gpu][1] - values[gpu][0] >= 20 * 1024
            and values[gpu][2] not in occupied)


def execute(args, state, save):
    previous_phase = None
    while True:
        dependency = read(args.wait_for)
        phase = dependency.get('phase')
        if phase != previous_phase:
            print(f'Waiting for existing experiment: {dependency["status"]}, {phase}', flush=True)
            previous_phase = phase
        state.update(status='waiting_for_experiment', dependency_status=dependency['status'],
                     dependency_phase=phase)
        save()
        if dependency['status'] in ('completed', 'failed'):
            # A failed upstream training job does not invalidate this explicitly
            # pinned, already-trained checkpoint. Never stop any of its workers.
            break
        time.sleep(20)
    state['status'] = 'waiting_for_gpu'
    save()
    while not gpu_available(args.gpu):
        time.sleep(15)
    if args.output.exists():
        raise FileExistsError(f'Evaluation output already exists: {args.output}')
    current = read(args.checkpoint / 'checkpoint.json')
    if current['delta_sha256'] != state['checkpoint_delta_sha256']:
        raise RuntimeError('Pinned checkpoint manifest changed while queued')
    command = [sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_success.py'),
               '--checkpoint', str(args.checkpoint), '--dataset', str(PROJECT / 'data/export'),
               '--reference-episode', str(args.reference), '--episodes', '1', '--steps', '350',
               '--min-particles', '100', '--replan-interval', str(args.replan_interval),
               '--seed', '2', '--gpus', str(args.gpu), '--workers', '1', '--cpu-threads', '8',
               '--video', '--output', str(args.output)]
    state.update(status='running', command=command, started_utc=datetime.now(timezone.utc).isoformat())
    save()
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                   'TOKENIZERS_PARALLELISM': 'false'}
    with (args.queue / 'evaluation.log').open('x') as log:
        subprocess.run(command, cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    config = read(args.output / 'config.json')
    if config['replan_interval'] != args.replan_interval:
        raise AssertionError('Evaluator did not use the requested execution horizon')
    result = read(args.output / 'episodes/episode_0001/episode_status.json')
    video = Path(result['video_path'])
    if result['status'] != 'completed' or result.get('video_error') or not video.is_file():
        raise RuntimeError(f'Evaluation/video incomplete: {result.get("error")}')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'stream=codec_name,width,height,nb_frames:format=duration,size', '-of', 'json', str(video)], text=True))
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    state.update(status='completed', success=result['success'], failure_mode=result['failure_mode'],
                 payload_particles=result['payload_particles'], video_path=str(video), video_metadata=probe,
                 completed_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(f'Video ready: {video}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path, required=True, help='Existing experiment status.json')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Pinned pretrained_model directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--replan-interval', type=int, default=10)
    parser.add_argument('--reference', type=Path, default=PROJECT /
        'data/raw/run_20261013_172047_collection_20261013_172021_158092_r1_w0/episode_000003')
    args = parser.parse_args()
    for name in ('queue', 'wait_for', 'checkpoint', 'output', 'reference'):
        setattr(args, name, getattr(args, name).resolve())
    config = read(args.checkpoint / 'config.json')
    manifest = read(args.checkpoint / 'checkpoint.json')
    if args.gpu < 0 or not 1 <= args.replan_interval <= config['chunk_size']:
        parser.error('Invalid GPU or execution horizon exceeds the predicted chunk')
    if args.output.exists():
        parser.error('Use a new evaluator output directory')
    read(args.wait_for)  # Fail before claiming output ownership if the dependency is invalid.
    args.queue.mkdir(parents=True, exist_ok=False)
    state = {'status': 'queued', 'pid': os.getpid(), 'checkpoint': str(args.checkpoint),
             'checkpoint_step': manifest['step'], 'checkpoint_delta_sha256': manifest['delta_sha256'],
             'wait_for': str(args.wait_for), 'gpu': args.gpu, 'output': str(args.output),
             'chunk_size': config['chunk_size'], 'replan_interval': args.replan_interval,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        temporary = args.queue / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.queue / 'status.json')

    save()
    try:
        execute(args, state, save)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
