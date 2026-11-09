"""Full expert replay -> two-GPU memory/update check -> gated 20k C training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_pi05_lora_ab import Experiment, PROJECT, SUBMODULE, read, source_hashes

sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.contract import file_hash
from excavator_vlm.data.replay_gate import assess_replay
from excavator_vlm.data.action_limits import collection_payload_threshold

CONFIG = SUBMODULE / 'configs/pi05/history_c_repeat20k.yaml'
DATASET = PROJECT / 'outputs/datasets/pi05_repeat_physx_bucket20_v2'
REFERENCE_CHECKPOINT = PROJECT / 'outputs/experiments/pi05_singlecycle20_physx_10k_20261102_02/train/checkpoints/006000/pretrained_model'


class RepeatHistoryExperiment(Experiment):
    def __init__(self, args):
        if args.resume_existing:
            self.args, self.root = args, args.output.resolve()
            self.state = read(self.root / 'status.json')
            if (self.state.get('status') != 'blocked'
                    or self.state.get('phase') != 'expert_replay_gate_failed'
                    or self.state.get('training_started') is not False or (self.root / 'train').exists()):
                raise ValueError('Reuse requires this blocked, never-trained experiment and its original replays')
            # Only the acceptance-check implementation may change when reusing
            # old physics traces. Simulator, data adapters and train configs stay frozen.
            permitted = {'ExcavatorVLA-vlm/excavator_vlm/data/action_limits.py',
                         'ExcavatorVLA-vlm/excavator_vlm/data/replay_gate.py',
                         'ExcavatorVLA-vlm/scripts/run_pi05_repeat_history.py'}
            current = source_hashes()
            for name, digest in self.state['source_hashes'].items():
                if name not in permitted and current.get(name) != digest:
                    raise ValueError(f'Cannot reuse physics evidence after source change: {name}')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            with (self.root / f'prior_blocked_attempt_{stamp}.json').open('x') as handle:
                json.dump(self.state, handle, indent=2, allow_nan=False)
            self.state.update(status='running', phase='reassess_existing_replays', pid=os.getpid(),
                              reassessment_started_utc=datetime.now(timezone.utc).isoformat(),
                              source_hashes=current, reused_physics_evidence=True)
            self.environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                                'TOKENIZERS_PARALLELISM': 'false', 'NCCL_P2P_DISABLE': '1',
                                'NCCL_IB_DISABLE': '1', 'PYTHONUNBUFFERED': '1'}
            self.children = []
        else:
            super().__init__(args)
        self.state['source_hashes'][str(Path(__file__).relative_to(PROJECT))] = file_hash(Path(__file__))
        self.state.update(training_started=False, requested_steps=20000,
                          checkpoint_steps=list(range(2000, 20001, 2000)),
                          dataset=str(DATASET), config=str(CONFIG),
                          observation_history=[-6, -3, 0], soil_backend='physx',
                          action_limit_profile='excavator_bucket_slew20_v1',
                          training_mode='vlm_lora_expert_full',
                          replay_episodes=[0, 77, 99],
                          replay_gate='archived collection payload threshold; physical-joint error <=0.15 rad; target error <=0.001 rad')
        self.save()

    def check_sources(self):
        expected = dict(self.state['source_hashes'])
        own = str(Path(__file__).relative_to(PROJECT))
        if expected.pop(own) != file_hash(Path(__file__)) or source_hashes() != expected:
            raise RuntimeError('Experiment sources changed; refusing the next stage')
        import shutil
        if shutil.disk_usage(self.root).free < 110 * 1024**3:
            raise RuntimeError('Less than 110 GiB free for the explicitly retained ten checkpoints')

    def replay_command(self, eid, gpu):
        reference = read(DATASET / f'references/episode_{eid:04d}.json')
        return ([sys.executable, '-u', str(SUBMODULE / 'scripts/eval_pi05_success.py'),
            '--checkpoint', str(REFERENCE_CHECKPOINT), '--dataset', str(DATASET),
            '--reference-episode', str(DATASET / f'references/episode_{eid:04d}.json'),
            '--replay-reference', '--replay-full-reference', '--episodes', '1',
            '--steps', str(len(reference['actions'])), '--min-particles', str(self.payload_threshold), '--seed', '2',
            '--gpus', str(gpu), '--workers', '1', '--cpu-threads', '8', '--headless',
            '--fresh-simulator', '--render-every-physics-step', '--video', '--soil-backend', 'physx',
            '--sand-walls-during-settle', '--output', str(self.root / f'expert_reference_{eid:04d}')], {})

    def collection_criterion(self):
        manifest = read(DATASET / 'backend_provenance.json')
        runs = sorted({entry['parent_id'].split('/')[0] for entry in manifest['episodes']})
        configs, evidence = [], []
        folder = DATASET / 'provenance/replay_criteria'
        folder.mkdir(parents=True, exist_ok=True)
        for run in runs:
            source = PROJECT / 'data/repeat/raw' / run / 'quality_gate_config.json'
            config = read(source)
            dest = folder / f'{run}_quality_gate_config.json'
            if dest.exists():
                if file_hash(dest) != file_hash(source):
                    raise ValueError('Archived collection criterion has changed')
            else:
                shutil.copy2(source, dest)
            configs.append(config)
            evidence.append({'path': str(dest.relative_to(DATASET)), 'sha256': file_hash(dest),
                             'field': 'quality_min_dump_particles'})
        thresholds = {config['quality_min_dump_particles'] for config in configs}
        versions = {config['quality_gate_version'] for config in configs}
        if len(thresholds) != 1 or len(versions) != 1:
            raise ValueError('Collection workers have inconsistent payload thresholds or versions')
        criterion = {'version': 'collection_min_dump_particles_v1',
                     'min_payload_particles_per_cycle': thresholds.pop(),
                     'quality_gate_version': versions.pop(), 'source_evidence': evidence,
                     'scope': 'collection payload threshold and source success; independent replay geometry and trajectory checks'}
        self.payload_threshold = collection_payload_threshold(DATASET, criterion)
        return criterion

    def execute(self):
        if not CONFIG.is_file() or not (DATASET / 'repair_report.json').is_file():
            raise FileNotFoundError('Build the separate relabelled dataset before this experiment')
        if read(DATASET / 'repair_report.json')['training_ready']:
            raise ValueError('This runner requires a fresh candidate, not a reused replay approval')
        criterion = self.collection_criterion()
        if not self.args.resume_existing:
            self.run('expert_full_replay_pair', [self.replay_command(0, 0), self.replay_command(77, 1)])
            self.run('expert_full_replay_third', [self.replay_command(99, 0)])
        audit = read(DATASET / 'audit.json')
        results = []
        for eid in self.state['replay_episodes']:
            folder = self.root / f'expert_reference_{eid:04d}/episodes/episode_0001'
            reference_path = DATASET / f'references/episode_{eid:04d}.json'
            config = read(folder.parent.parent / 'config.json')
            if (config.get('reference_setup_sha256') != file_hash(reference_path)
                    or config.get('simulation_settings', {}).get('soil_backend') != 'physx'
                    or config.get('replay_scope') != 'full_reference'
                    or config.get('action_limit_profile') != audit['action_contract']['profile']):
                raise ValueError('Stored full-replay protocol/reference does not match the candidate')
            trace = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
            reference = read(reference_path)
            if any(c.get('returned') is not True or c.get('metrics', {}).get('success') is not True
                   for c in reference['repeat_cycles']):
                raise ValueError('Source must contain successful execution and physical returns for all cycles')
            result = assess_replay(reference, read(folder / 'episode_status.json'), trace,
                                   min_payload=self.payload_threshold)
            result['episode_index'] = eid
            # Retain immutable, inspectable evidence alongside the candidate.
            dest = DATASET / f'provenance/replay_checks/episode_{eid:04d}'
            dest.mkdir(parents=True, exist_ok=self.args.resume_existing)
            result['evidence'] = []
            for name in ('episode_status.json', 'steps.jsonl'):
                if (dest / name).exists():
                    if file_hash(dest / name) != file_hash(folder / name):
                        raise ValueError('Saved replay evidence changed; refusing reassessment')
                else:
                    shutil.copy2(folder / name, dest / name)
                result['evidence'].append({'path': str((dest / name).relative_to(DATASET)),
                                           'sha256': file_hash(dest / name)})
            results.append(result)
        gate = {'schema_version': 'excavator_repeat_expert_gate_v2',
                'status': 'passed' if all(r['success'] for r in results) else 'failed',
                'dataset_fingerprint': audit['fingerprint'], 'soil_backend': 'physx',
                'action_contract': audit['action_contract'], 'episodes': results,
                'collection_criterion': criterion,
                'experiment': str(self.root), 'source_hashes': self.state['source_hashes'],
                'completed_utc': datetime.now(timezone.utc).isoformat()}
        self.state['expert_replay_gate'] = gate
        self.save()
        path = DATASET / 'expert_replay_gate.json'
        if path.exists():
            if not self.args.resume_existing:
                raise FileExistsError('Refusing to replace an existing replay gate without explicit reassessment')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            archive = path.with_name(f'expert_replay_gate_before_reassessment_{stamp}.json')
            path.rename(archive)
            self.state['previous_gate_archive'] = str(archive)
        with path.open('x') as handle:
            json.dump(gate, handle, indent=2, allow_nan=False)
        if gate['status'] != 'passed':
            self.state.update(status='blocked', phase='expert_replay_gate_failed', training_started=False)
            self.save()
            print('Expert replay failed. Formal training was NOT started.', flush=True)
            return
        report_path = DATASET / 'repair_report.json'
        report = read(report_path)
        report.update(training_ready=True, status='numerical_and_sampled_physics_replay_passed',
                      replay_gate_sha256=file_hash(path))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        self.run('training_preflight', [([sys.executable, '-u', str(SUBMODULE / 'scripts/train_pi05.py'),
                                         '--config', str(CONFIG), '--dry-run'], {})])
        command, overrides = self.train_command('history_c_repeat20k.yaml', self.root / 'train')
        self.state['training_started'] = True
        self.save()
        # These are the first two real updates of the 20k run, not a new schedule.
        self.run('ddp_memory_update_check', [(command + ['--stop-after', '2'], overrides)])
        provenance = read(self.root / 'train/provenance.json')
        if provenance['batch_contract'] != {'world_size': 2, 'batch_size': 1, 'accumulation': 4}:
            raise ValueError('Expected two-GPU DDP, per-GPU batch1 and accumulation4')
        self.run('ddp_train_20000', [self.train_command('history_c_repeat20k.yaml', self.root / 'train', resume=True)])
        self.state.update(status='completed', phase='finished', completed_utc=datetime.now(timezone.utc).isoformat())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--resume-existing', action='store_true',
                        help='Reassess this blocked, never-trained experiment using verified existing replay traces')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.detach:
        if args.output.exists() and not args.resume_existing:
            raise FileExistsError('Refusing to overwrite an existing experiment')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        log = (args.output.with_suffix(f'.reassess_{stamp}.launcher.log') if args.resume_existing
               else args.output.with_suffix('.launcher.log'))
        with log.open('x') as handle:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(args.output)]
            if args.resume_existing:
                command.append('--resume-existing')
            child = subprocess.Popen(command, cwd=PROJECT,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        print(json.dumps({'pid': child.pid, 'status': str(args.output / 'status.json'), 'launcher_log': str(log)}))
        return
    args.gpus = [0, 1]; args.steps = 20000; args.allow_shared_gpus = False; args.resume_lr = None
    experiment = RepeatHistoryExperiment(args)
    try:
        experiment.execute()
    except BaseException as exc:
        experiment.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        experiment.save()
        raise


if __name__ == '__main__':
    main()
