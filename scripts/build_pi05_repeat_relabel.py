"""Publish a separate bucket20 candidate; never overwrite or claim physics parity."""
import argparse
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

SUBMODULE = Path(__file__).resolve().parents[1]
ROOT = SUBMODULE.parent
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.action_limits import BUCKET20_PROFILE, PROFILES
from excavator_vlm.data.contract import CAMERAS, audit_dataset, file_hash, validate_split
from excavator_vlm.data.relabel import relabel_episode
from excavator_vlm.policies.pi05.dataset import training_stats


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parent_records(source, raw):
    result = {}
    for run in sorted(raw.glob('run_*')):
        rows = {r['episode_id']: r for r in map(json.loads, (run / 'episodes.jsonl').read_text().splitlines())}
        for path in sorted(run.glob('episode_*/export.json')):
            receipt = json.loads(path.read_text())
            if not receipt.get('validated') or Path(receipt['export_dir']).resolve() != source:
                continue
            meta_path = path.with_name('meta.json')
            meta = json.loads(meta_path.read_text())
            if (not meta.get('success') or meta.get('repeat_baseline', {}).get('soil_backend') != 'physx'
                    or not meta.get('source_sampling_audit', {}).get('ok')
                    or len(meta.get('repeat_cycles', [])) != 5 or meta.get('scoops_completed') != 5
                    or any(not c['metrics']['success'] for c in meta['repeat_cycles'])):
                raise ValueError(f'Unverified five-cycle PhysX parent: {meta_path}')
            eid = int(receipt['export_episode_index'])
            if eid in result:
                raise ValueError('Duplicate source export mapping')
            result[eid] = (meta_path, path, meta, rows[meta['episode_id']])
    return result


def frozen_split(ids, pilot, seed):
    parts = {key: [] for key in ('train', 'validation', 'test')}
    selection = json.loads((pilot / 'selection.json').read_text())
    for entry in selection['selected']:
        parts[entry['split']].append(int(entry['source_export_episode']))
    fixed = sum(parts.values(), [])
    if len(fixed) != len(set(fixed)) or not set(fixed).issubset(ids):
        raise ValueError('Prior pilot parent mapping has duplicates or unknown parents')
    remaining = list(map(int, np.random.default_rng(seed).permutation(sorted(set(ids) - set(fixed)))))
    target = {'validation': len(ids) // 10, 'test': len(ids) // 10}
    for key in ('validation', 'test'):
        n = target[key] - len(parts[key])
        if n < 0:
            raise ValueError('Prior held-out parents exceed requested split size')
        parts[key].extend(remaining[:n]); remaining = remaining[n:]
    parts['train'].extend(remaining)
    return {'schema_version': 1, 'seed': seed,
            'method': 'parent_disjoint_80_10_10_preserving_prior_pilot_assignments',
            **{key: sorted(value) for key, value in parts.items()}}


def build(args):
    if args.output.exists() or args.output.with_name(args.output.name + '.building').exists():
        raise FileExistsError('Refusing to overwrite an existing dataset or build')
    source_report, frames = audit_dataset(args.source)
    info = json.loads((args.source / 'meta/info.json').read_text())
    parents = parent_records(args.source, args.raw)
    ids = list(map(int, frames.episode_index.unique()))
    if set(parents) != set(ids):
        raise ValueError('Verified parent evidence must cover every exported episode')
    split = frozen_split(ids, args.pilot, args.seed)
    staging = args.output.with_name(args.output.name + '.building')
    staging.mkdir(parents=True)
    shutil.copytree(args.source / 'meta', staging / 'meta')
    for source in sorted((args.source / 'videos').rglob('*.mp4')):
        dest = staging / source.relative_to(args.source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, dest) # Containers are read-only; no video bytes are rewritten.
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, dest) # Separate mounts: byte-identical private copy.
    metadata = pd.concat([pd.read_parquet(p) for p in args.source.glob('meta/episodes/**/*.parquet')])
    metadata = metadata.set_index('episode_index')
    task_table = pd.read_parquet(args.source / 'meta/tasks.parquet')
    task = str(task_table.index[0])
    evidence, reports, revised_frames = [], [], []
    for eid, episode in frames.groupby('episode_index', sort=True):
        eid = int(eid); episode = episode.copy(); row = metadata.loc[eid]
        states = np.stack(episode['observation.state']); actions = np.stack(episode.action)
        revised, candidate, report = relabel_episode(states, actions, PROFILES[BUCKET20_PROFILE])
        episode['observation.state'] = list(revised)
        episode['action'] = list(candidate)
        revised_frames.append(episode)
        report['episode_index'] = eid; reports.append(report)
        mp, ep, meta, index = parents[eid]
        if len(episode) != meta['source_sampling_audit']['sample_count']:
            raise ValueError('Parent/export frame count mismatch')
        provenance = staging / f'provenance/episode_{eid:04d}'
        provenance.mkdir(parents=True)
        shutil.copy2(mp, provenance / 'meta.json'); shutil.copy2(ep, provenance / 'export.json')
        dump(provenance / 'episode_index_record.json', index)
        evidence.append({'episode_index': eid, 'soil_backend': 'physx',
            'parent_id': f'{mp.parent.parent.name}/{mp.parent.name}', 'source_export_episode': eid,
            'evidence': {'path': f'provenance/episode_{eid:04d}/meta.json',
                         'sha256': file_hash(mp), 'field': 'repeat_baseline.soil_backend'}})
        applied = index['scene_randomization']['applied']
        setup = {'schema_version': 'excavator_reference_setup_v1', 'fps': 10, 'action_units': 'rad/s',
            'state_names': info['features']['observation.state']['names'],
            'action_names': info['features']['action']['names'], 'task': task,
            'initial_joints_deg': np.degrees(revised[0, :4].astype(float)).tolist(),
            'dig_target_xyz': index['target_xyz'], 'sand_amount': applied['sand_amount_multiplier'],
            'expected_initial_state': revised[0].tolist(), 'actions': candidate.tolist(),
            'expected_states': revised.tolist(), 'repeat_cycles': meta['repeat_cycles'],
            'profile': {'sand_xy': applied['sand_center'][:2],
                'truck_translation_xyz': applied['truck_translation_xyz'],
                'truck_yaw_deg': applied['truck_yaw_deg'],
                'unload_landing_xyz': meta['repeat_cycles'][0]['landing']['landing']},
            'provenance': {'kind': 'relabelled_full_repeat_candidate_not_physics_validated',
                'source_export_episode': eid, 'source_dataset_fingerprint': source_report['fingerprint'],
                'source_parent': evidence[-1]['parent_id'], 'soil_backend': 'physx',
                'action_limit_profile': BUCKET20_PROFILE}}
        dump(staging / f'references/episode_{eid:04d}.json', setup)
    merged = pd.concat(revised_frames, ignore_index=True)
    # Preserve existing per-episode table/video offsets and data file partitions.
    for (chunk, file), rows in metadata.groupby(['data/chunk_index', 'data/file_index']):
        path = staging / info['data_path'].format(chunk_index=int(chunk), file_index=int(file))
        path.parent.mkdir(parents=True, exist_ok=True)
        merged[merged.episode_index.isin(rows.index)].to_parquet(path, index=False)
    info = copy.deepcopy(info)
    info.update(action_limit_profile=BUCKET20_PROFILE, action_limits_rad_s=PROFILES[BUCKET20_PROFILE],
        action_policy_version='sampled_command_target_difference_v1',
        relabel={'source': str(args.source), 'source_fingerprint': source_report['fingerprint'],
                 'state_change': 'previous_action only; all other observed fields unchanged',
                 'physics_replay': 'required_before_training'})
    dump(staging / 'meta/info.json', info)
    dump(staging / 'meta/stats.json', training_stats(merged, split['train']))
    dump(staging / 'backend_provenance.json', {'schema_version': 'excavator_backend_provenance_v1',
                                             'soil_backend': 'physx', 'episodes': evidence})
    audit, _ = audit_dataset(staging)
    split['dataset_fingerprint'] = audit['fingerprint']
    validate_split(split, ids, audit['fingerprint'])
    dump(staging / 'audit.json', audit)
    dump(staging / 'split_manifest.json', split)
    dump(staging / 'repair_report.json', {'status': 'numerical_parity_passed_physics_replay_pending',
         'training_ready': False, 'source_dataset_fingerprint': source_report['fingerprint'],
         'dataset_fingerprint': audit['fingerprint'], 'action_contract': audit['action_contract'],
         'episodes': reports, 'video_bytes_and_frame_times_unchanged': True})
    staging.rename(args.output)
    print(json.dumps({'dataset': str(args.output), 'episodes': len(ids),
        'split_counts': {key: len(split[key]) for key in ('train', 'validation', 'test')},
        'max_integrator_error_rad': max(r['relabel_integrator_max_error_rad'] for r in reports),
        'training_ready': False, 'next_gate': 'three complete five-cycle expert replays'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/repeat/export_repeat')
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/repeat/raw')
    parser.add_argument('--pilot', type=Path, default=ROOT / 'data/pi05_singlecycle_physx_pilot_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/datasets/pi05_repeat_physx_bucket20_v2')
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    for key in ('source', 'raw', 'pilot', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    build(args)


if __name__ == '__main__':
    main()
