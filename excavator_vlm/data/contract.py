"""Validate the current-v4 export and freeze episode splits shared by all policies."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAMERAS = tuple(f'observation.images.{i}' for i in range(3))
JOINTS = ('swing', 'boom', 'arm', 'bucket')
ACTION_NAMES = tuple(f'{joint}_cmd_velocity' for joint in JOINTS)
STATE_NAMES = (
    *(f'{joint}_position' for joint in JOINTS),
    *(f'{joint}_velocity' for joint in JOINTS),
    *(f'{joint}_tracking_error' for joint in JOINTS),
    *(f'previous_{joint}_action' for joint in JOINTS),
    'bucket_pour_axis_forward', 'bucket_pour_axis_up',
    'dig_target_from_tip_forward', 'dig_target_from_tip_left', 'dig_target_from_tip_up',
    'unload_from_load_forward', 'unload_from_load_left', 'unload_from_load_up',
    'truck_heading_relative_sin', 'truck_heading_relative_cos',
    'bucket_fill_fraction', 'bucket_fill_rate_fraction_per_s',
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_info(root):
    info = json.loads((Path(root) / 'meta/info.json').read_text())
    features = info['features']
    if info.get('state_schema') != 'current-v4' or info.get('fps') != 10:
        raise ValueError('Expected current-v4 dataset at 10 Hz')
    for name, size, names in [('observation.state', 28, STATE_NAMES), ('action', 4, ACTION_NAMES)]:
        if features[name]['shape'] != [size] or features[name]['names'] != list(names):
            raise ValueError(f'{name}: shape/names/order differ from deployment contract')
    for name in CAMERAS:
        if features[name]['shape'] != [256, 256, 3] or features[name]['dtype'] != 'video':
            raise ValueError(f'{name}: expected 256x256 RGB video')
    if features.get('action_loss_weight', {}).get('shape') != [1]:
        raise ValueError('Missing scalar action_loss_weight supervision')
    if not np.array_equal(info.get('action_limits_rad_s'), [6., 2.5, 2.5, 6.]):
        raise ValueError('Unexpected deployment action limits')
    return info


def policy_features(info):
    return {
        'observation.state': {'type': 'STATE', 'shape': [28]},
        **{key: {'type': 'VISUAL', 'shape': [3, 256, 256]} for key in CAMERAS},
    }, {'action': {'type': 'ACTION', 'shape': [4]}}


def read_frames(root):
    paths = sorted(Path(root).glob('data/**/*.parquet'))
    if not paths:
        raise ValueError('No data parquet files found')
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values(
        ['episode_index', 'frame_index']).reset_index(drop=True)


def audit_dataset(root, chunk_size=10):
    """Check timestamps and causal action history, rather than just equal lengths.

    Video existence and first/last timestamp decoding are checked for every camera
    in every episode. This does not prove physical camera exposure synchronization.
    """
    import av

    root = Path(root)
    info = load_info(root)
    frames = read_frames(root)
    ids = sorted(map(int, frames.episode_index.unique()))
    if ids != list(range(info['total_episodes'])) or len(frames) != info['total_frames']:
        raise ValueError('Episode/frame totals differ from metadata')
    if frames['index'].tolist() != list(range(len(frames))):
        raise ValueError('Absolute frame indices must be unique and contiguous')
    states = np.stack(frames['observation.state']).astype(np.float64)
    actions = np.stack(frames.action).astype(np.float64)
    weights = frames.action_loss_weight.to_numpy(dtype=np.float64)
    if states.shape != (len(frames), 28) or actions.shape != (len(frames), 4):
        raise ValueError('Invalid state/action row dimensions')
    if not all(np.isfinite(x).all() for x in (states, actions, weights)) or (weights < 0).any():
        raise ValueError('Nonfinite state/action or invalid supervision weights')
    if np.any(np.abs(actions) > np.array(info['action_limits_rad_s']) + 1e-5):
        raise ValueError('Actions exceed deployment velocity limits')
    meta_files = sorted(root.glob('meta/episodes/**/*.parquet'))
    metadata = pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)
    if sorted(metadata.episode_index.tolist()) != ids:
        raise ValueError('Missing or duplicate episode metadata')
    metadata = metadata.set_index('episode_index')
    lengths, max_history_error, checked = [], 0., 0
    for episode_id, episode in frames.groupby('episode_index', sort=True):
        length = len(episode)
        lengths.append(length)
        if episode.frame_index.tolist() != list(range(length)):
            raise ValueError(f'Episode {episode_id}: noncontiguous frame indices')
        if not np.allclose(episode.timestamp, np.arange(length) / info['fps'], atol=1e-5):
            raise ValueError(f'Episode {episode_id}: incorrect timestamps')
        s, a = np.stack(episode['observation.state']), np.stack(episode.action)
        if length > 1:
            max_history_error = max(max_history_error, float(np.abs(s[1:, 12:16] - a[:-1]).max()))
        row = metadata.loc[episode_id]
        if int(row['length']) != length:
            raise ValueError('Episode metadata length mismatch')
        for camera in CAMERAS:
            path = root / info['video_path'].format(
                video_key=camera, chunk_index=int(row[f'videos/{camera}/chunk_index']),
                file_index=int(row[f'videos/{camera}/file_index']))
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height) != (256, 256):
                    raise ValueError(f'Wrong video resolution: {path}')
                for ts in [float(row[f'videos/{camera}/from_timestamp']),
                           float(row[f'videos/{camera}/from_timestamp']) + (length - 1) / info['fps']]:
                    container.seek(max(0, int(ts / stream.time_base)), stream=stream)
                    decoded = next((frame for frame in container.decode(stream)
                                    if frame.time is not None and frame.time >= ts - 1e-4), None)
                    if decoded is None or abs(decoded.time - ts) > 1 / info['fps'] + 1e-4:
                        raise ValueError(f'Video timestamp unavailable: {path} at {ts}')
            checked += 1
    if max_history_error > 1e-5:
        raise ValueError('Previous actions are not causally aligned with targets')
    # Bind the split to metadata and tabular content. Video payloads are checked
    # above; their full binary checksums are intentionally not part of this hash.
    digest = hashlib.sha256()
    for path in [root / 'meta/info.json', *sorted(root.glob('data/**/*.parquet')),
                 *meta_files, root / 'meta/tasks.parquet']:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_hash(path).encode())
    valid = sum(sum(min(chunk_size, length - i) for i in range(length)) for length in lengths)
    return {
        'status': 'passed', 'schema': 'current-v4', 'fps': info['fps'],
        'episodes': len(ids), 'frames': len(frames), 'fingerprint': digest.hexdigest(),
        'fingerprint_scope': 'info + data parquet + episode metadata + tasks; excludes video bytes',
        'max_previous_action_error': max_history_error, 'camera_episode_checks': checked,
        'chunk_size': chunk_size, 'padded_timestep_fraction': 1 - valid / (len(frames) * chunk_size),
    }, frames


def make_split(frames, fingerprint, seed=1000):
    """Task-stratified 80/10/10 split with exact global counts, at episode grain."""
    rng = np.random.default_rng(seed)
    groups = {}
    for episode_id, rows in frames.groupby('episode_index'):
        task = tuple(sorted(map(int, rows.task_index.unique())))
        groups.setdefault(task, []).append(int(episode_id))
    groups = sorted(groups.items())
    count = frames.episode_index.nunique()
    if count < 10:
        raise ValueError('At least 10 episodes are required for a three-way split')
    targets = np.array([count - 2 * (count // 10), count // 10, count // 10])
    ideal = np.array([len(ids) * targets / count for _, ids in groups])
    allocations = np.floor(ideal).astype(int)
    while not np.array_equal(allocations.sum(axis=0), targets):
        choices = [(ideal[i, j] - allocations[i, j], i, j)
                   for i, (_, ids) in enumerate(groups) if allocations[i].sum() < len(ids)
                   for j in range(3) if allocations[:, j].sum() < targets[j]]
        _, i, j = max(choices)
        allocations[i, j] += 1
    parts = dict(train=[], validation=[], test=[])
    for i, (_, ids) in enumerate(groups):
        ids = list(rng.permutation(ids).astype(int))
        cursor = 0
        for j, key in enumerate(parts):
            parts[key].extend(int(x) for x in ids[cursor:cursor + allocations[i, j]])
            cursor += allocations[i, j]
    return {'schema_version': 1, 'seed': seed, 'method': 'episode_task_stratified_80_10_10',
            'dataset_fingerprint': fingerprint, **{key: sorted(ids) for key, ids in parts.items()}}


def validate_split(split, episode_ids, fingerprint=None):
    parts = [split[key] for key in ('train', 'validation', 'test')]
    if any(not part or any(type(x) is not int for x in part) for part in parts):
        raise ValueError('Split parts must be nonempty integer episode lists')
    flat = sum(parts, [])
    if len(flat) != len(set(flat)) or set(flat) != set(episode_ids):
        raise ValueError('Split has leakage, duplicates, missing, or unknown episodes')
    if fingerprint is not None and split['dataset_fingerprint'] != fingerprint:
        raise ValueError('Dataset changed since the split was frozen; create a new versioned split')
