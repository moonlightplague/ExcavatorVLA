"""Turn a Best-of-N selection into a LeRobot v3.0 dataset SmolVLA can fine-tune on.

The rollout store already keeps everything a policy needs per step -- the
pre-action observation the model consumed, the command actually sent to the
simulator, and the three camera frames that went with them. So conversion is a
re-packing job, not a re-simulation one: read the selection manifest written by
``select_rollouts``, and for each chosen episode emit one data parquet, one
episodes parquet and three mp4 files laid out exactly the way the reference
export at ``data/export`` is laid out.

Three decisions in here carry real weight, so they are flags rather than
behaviour you have to read the source to find.

Normalization. ``lerobot/scripts/lerobot_train.py`` rebuilds the normalizer from
``dataset.meta.stats`` whenever ``policy.pretrained_path`` is set, which
``scripts/train_smolvla.py`` always does. The checkpoint's own saved statistics
are overridden, not reused. A fine-tuning set whose ``observation.state`` and
``action`` statistics differ from the ones the base model was trained under
therefore shifts every input at step zero, before any gradient has been taken.
``--stats-mode reference`` (the default) copies those two entries plus the image
entries from the reference dataset verbatim and computes only the bookkeeping
keys honestly, which keeps normalization bit-identical to pre-training.
``--stats-mode compute`` is there for the case where you *want* the shift, and
either way a containment check reports dimensions that leave the reference
range instead of silently clipping them.

Weighting. ``scripts/train_smolvla.py`` folds ``action_loss_weight`` into the
loss as ``sum(w * L) / sum(w)`` over the whole batch. Weights cancel inside a
sample but survive across samples, so one constant per trajectory is already a
trajectory-level importance weight -- no training code has to change. The
default ``uniform`` is plain rejection sampling: every selected rollout counts
once. The other modes re-use the score that justified the selection, and all of
them are rescaled to mean one so the reported ``supervised_action_fraction``
stays comparable across runs.

Schema. Only the eleven features the store actually measured are written.
``observation.effort``, the phase and scoop counters, the recovery triple and
``action_is_expert`` are all part of the reference export, and all of them are
absent from a rollout NPZ. Inventing plausible values for them would be
inventing supervision, so they are dropped; ``build_train_config`` validates
state, action and images only, and ``action_loss_weight`` is the one extra key
it looks for.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.contract import ACTION_NAMES, CAMERAS, STATE_NAMES

SCHEMA_VERSION = 'excavator_rl_finetune_dataset_v1'
SELECTION_SCHEMA = 'excavator_rollout_selection_v1'
CODEBASE_VERSION = 'v3.0'
ROBOT_TYPE = 'excavator'
STATE_SCHEMA = 'current-v4'
STATE_SCHEMA_VERSION = 'excavator_state_v4_28d_rl_best_of_n_v1'
ACTION_LIMITS_RAD_S = [6.0, 2.5, 2.5, 6.0]
IMAGE_SHAPE = [256, 256, 3]
CHUNKS_SIZE = 1000
DEFAULT_FPS = 10.0
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200

TRAJECTORY = Path('trajectory')
ARRAYS_NPZ = 'arrays.npz'
FRAMES_NPZ = 'frames.npz'
META_JSON = 'meta.json'
SCENE_CONTEXT = 'scene_context.json'
INITIAL_OBSERVATION = 'initial_observation.json'
PROVENANCE_NAME = 'rl_provenance.json'

STATE_KEY = 'observation.state'
ACTION_KEY = 'action'
WEIGHT_KEY = 'action_loss_weight'
STATE_SOURCES = {'pre': 'observation.state', 'post': 'observation.state_next'}
ACTION_SOURCES = {'applied': 'action.applied', 'predicted': 'action.predicted'}
WEIGHT_MODES = ('uniform', 'score', 'rank', 'softmax')
STATS_MODES = ('reference', 'compute')
# Copied from the reference export rather than re-derived, because these are the
# statistics the base checkpoint was trained under.
REFERENCE_STATS_KEYS = (STATE_KEY, ACTION_KEY, *CAMERAS)
STD_FLOOR = 1.0e-6


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _chunk_dir(index):
    return f'chunk-{int(index):03d}'


def _file_stem(index):
    return f'file-{int(index):03d}'


def load_selection(path):
    """Read a selection manifest and hand back its best-first ``selected`` list."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SELECTION_SCHEMA:
        raise ValueError(f'Unsupported selection schema: {manifest.get("schema_version")!r}')
    selected = manifest.get('selected') or []
    if not selected:
        raise ValueError(f'Selection has no episodes: {path}')
    return manifest, selected


def task_text_for(episode_dir, fallback=None):
    """Recover the language prompt the policy was actually given for this scene.

    SmolVLA is language conditioned, so a wrong task string is a wrong input,
    not a cosmetic label. ``trajectory/meta.json`` does not record it, which
    means the episode directory is the only source: ``scene_context.json``
    carries the generated text, and ``initial_observation.json`` echoes back
    what the simulator reported at the first observation.
    """
    folder = Path(episode_dir)
    scene = folder / SCENE_CONTEXT
    if scene.is_file():
        context = json.loads(scene.read_text(encoding='utf-8'))
        text = (context.get('observation_context') or {}).get('task_text')
        if text:
            return str(text), 'scene_context.json'
    initial = folder / INITIAL_OBSERVATION
    if initial.is_file():
        text = json.loads(initial.read_text(encoding='utf-8')).get('task')
        if text:
            return str(text), 'initial_observation.json'
    if fallback:
        return str(fallback), 'fallback'
    raise ValueError(f'No task text found for {folder}')


def load_trajectory(episode_dir, state_source, action_source):
    """Validate one rollout's stored trajectory and return its arrays.

    Frames are deliberately *not* decoded here. A rollout's ``frames.npz`` is
    tens of megabytes compressed and hundreds decoded, and planning walks every
    selected episode before writing any of them; holding all of them would make
    peak memory scale with the selection instead of with one episode. Planning
    therefore only checks that the camera members exist, and ``decode_frames``
    is called once per episode at write time.
    """
    folder = Path(episode_dir)
    arrays_path = folder / TRAJECTORY / ARRAYS_NPZ
    frames_path = folder / TRAJECTORY / FRAMES_NPZ
    meta_path = folder / TRAJECTORY / META_JSON
    for required in (arrays_path, frames_path, meta_path):
        if not required.is_file():
            raise ValueError(f'Missing {required}')

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        names = set(arrays.files)
        state_name = STATE_SOURCES[state_source]
        action_name = ACTION_SOURCES[action_source]
        for required in (state_name, action_name):
            if required not in names:
                raise ValueError(f'{arrays_path.name} has no {required!r}; has {sorted(names)}')
        state = np.asarray(arrays[state_name], dtype=np.float32)
        action = np.asarray(arrays[action_name], dtype=np.float32)
    with np.load(frames_path, allow_pickle=False) as frames:
        missing = [key for key in range(len(CAMERAS)) if str(key) not in frames.files]
        if missing:
            raise ValueError(f'{frames_path.name} has no camera {missing}; has {sorted(frames.files)}')

    length = int(meta.get('frames') or state.shape[0])
    if state.ndim != 2 or state.shape[1] != len(STATE_NAMES):
        raise ValueError(f'{state_name} has shape {state.shape}, expected (T, {len(STATE_NAMES)})')
    if action.ndim != 2 or action.shape[1] != len(ACTION_NAMES):
        raise ValueError(f'{action_name} has shape {action.shape}, expected (T, {len(ACTION_NAMES)})')
    if not (state.shape[0] == action.shape[0] == length):
        raise ValueError(f'Length mismatch: meta={length} state={state.shape[0]} action={action.shape[0]}')
    if not np.isfinite(state).all():
        raise ValueError(f'{state_name} contains non-finite values')
    if not np.isfinite(action).all():
        raise ValueError(f'{action_name} contains non-finite values')

    fps = float(meta.get('fps') or DEFAULT_FPS)
    return {'length': length, 'fps': fps, 'state': state, 'action': action,
            'frames_path': frames_path, 'meta': meta,
            'arrays_sha256': _sha256(arrays_path), 'frames_sha256': _sha256(frames_path)}


def decode_frames(frames_path, length):
    """Decode one episode's three camera stacks, checking the shape as we go."""
    images = {}
    with np.load(frames_path, allow_pickle=False) as frames:
        for index, key in enumerate(CAMERAS):
            block = np.asarray(frames[str(index)], dtype=np.uint8)
            if block.shape != (length, *IMAGE_SHAPE):
                raise ValueError(f'{key} has shape {block.shape}, expected {(length, *IMAGE_SHAPE)}')
            images[key] = block
    return images


def scalar_stats(values):
    """Population statistics in plain Python floats, matching the reference exporter.

    The reference ``stats.json`` was produced by summing Python floats, and its
    variance is ``E[x^2] - E[x]^2`` rather than a two-pass or Welford estimate.
    Reproducing the arithmetic is what makes the copied and computed keys agree
    to the last digit; a numpy float64 reimplementation does not.
    """
    count = len(values)
    if not count:
        return {'count': [0]}
    mean = sum(values) / count
    mean_square = sum(value * value for value in values) / count
    std = math.sqrt(max(0.0, mean_square - mean * mean))
    return {'count': [count], 'mean': [mean], 'std': [1.0 if std < STD_FLOOR else std],
            'min': [min(values)], 'max': [max(values)]}


def vector_stats(vectors, dim):
    """Per-dimension population statistics, again in plain Python floats."""
    count = len(vectors)
    if not count:
        return {'count': [0]}
    mins = [math.inf] * dim
    maxs = [-math.inf] * dim
    sums = [0.0] * dim
    sums_square = [0.0] * dim
    for vector in vectors:
        for index in range(dim):
            value = vector[index]
            if value < mins[index]:
                mins[index] = value
            if value > maxs[index]:
                maxs[index] = value
            sums[index] += value
            sums_square[index] += value * value
    means = [total / count for total in sums]
    stds = []
    for index in range(dim):
        variance = max(0.0, sums_square[index] / count - means[index] * means[index])
        std = math.sqrt(variance)
        stds.append(1.0 if std < STD_FLOOR else std)
    return {'count': [count], 'mean': means, 'std': stds, 'min': mins, 'max': maxs}


def visual_identity_stats():
    """ImageNet constants, because pixels are normalized by the processor, not by data."""
    return {'count': [0], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
            'min': [0.0, 0.0, 0.0], 'max': [1.0, 1.0, 1.0]}


def compute_stats(rows):
    """Statistics for every key this dataset declares, computed from its own rows."""
    stats = {
        STATE_KEY: vector_stats([row[STATE_KEY] for row in rows], len(STATE_NAMES)),
        ACTION_KEY: vector_stats([row[ACTION_KEY] for row in rows], len(ACTION_NAMES)),
    }
    for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index', WEIGHT_KEY):
        stats[key] = scalar_stats([float(row[key]) for row in rows])
    for key in CAMERAS:
        stats[key] = visual_identity_stats()
    return stats


def containment_report(computed, reference):
    """Flag dimensions whose real range leaves the reference range.

    Reference statistics are reused for normalization, so a value outside the
    reference min/max normalizes past +/-1. That is not an error -- the base
    model saw clipped commands in the same range and coped -- but it is exactly
    the kind of drift that should be visible in the manifest rather than
    discovered as a regression.
    """
    problems = []
    for key in (STATE_KEY, ACTION_KEY):
        if key not in reference:
            continue
        low, high = reference[key]['min'], reference[key]['max']
        actual_low, actual_high = computed[key]['min'], computed[key]['max']
        for index in range(len(low)):
            if actual_low[index] < float(low[index]):
                problems.append({'feature': key, 'dim': index,
                                 'name': (STATE_NAMES if key == STATE_KEY else ACTION_NAMES)[index],
                                 'side': 'min', 'reference': float(low[index]),
                                 'actual': actual_low[index]})
            if actual_high[index] > float(high[index]):
                problems.append({'feature': key, 'dim': index,
                                 'name': (STATE_NAMES if key == STATE_KEY else ACTION_NAMES)[index],
                                 'side': 'max', 'reference': float(high[index]),
                                 'actual': actual_high[index]})
    return problems


def merged_stats(rows, stats_mode, reference_stats):
    """Honest bookkeeping keys, plus reference statistics where the mode asks for them."""
    computed = compute_stats(rows)
    warnings = []
    if stats_mode == 'compute' or not reference_stats:
        return computed, computed, warnings
    stats = dict(computed)
    copied = {}
    for key in REFERENCE_STATS_KEYS:
        if key in reference_stats:
            stats[key] = json.loads(json.dumps(reference_stats[key]))
            copied[key] = stats[key]
        else:
            warnings.append(f'reference stats.json has no {key!r}; computed instead')
    return stats, copied, warnings + [
        f'{item["feature"]}[{item["dim"]}] {item["name"]} {item["side"]} '
        f'{item["actual"]:.6g} outside reference {item["reference"]:.6g}'
        for item in containment_report(computed, reference_stats)
    ]


def episode_weights(selected, mode, temperature):
    """One constant per trajectory, rescaled to mean one.

    The loss is ``sum(w * L) / sum(w)`` across the batch, so a per-trajectory
    constant is a trajectory-level importance weight and its overall scale is
    unobservable. Rescaling to mean one keeps ``supervised_action_fraction``
    near 1.0 and makes loss curves comparable between weighting modes.
    """
    scores = [float(item.get('score') or 0.0) for item in selected]
    count = len(scores)
    if mode == 'uniform':
        raw = [1.0] * count
    elif mode == 'score':
        floor = min(scores) if scores else 0.0
        raw = [max(0.0, score - floor) + 1.0 for score in scores]
    elif mode == 'rank':
        # ``selected`` arrives best-first, so rank 0 is the strongest rollout.
        raw = [float(count - rank) for rank in range(count)]
    elif mode == 'softmax':
        peak = max(scores) if scores else 0.0
        raw = [math.exp((score - peak) / float(temperature)) for score in scores]
    else:
        raise ValueError(f'Unknown weight mode: {mode}')
    if any(value < 0.0 for value in raw):
        raise ValueError('Negative weights are not representable as loss weights')
    total = sum(raw)
    if total <= 0.0:
        raw = [1.0] * count
        total = float(count)
    scale = count / total
    return [value * scale for value in raw], raw


def build_features():
    """The eleven features this dataset really has, in reference declaration form."""
    features = {
        STATE_KEY: {'dtype': 'float32', 'shape': [len(STATE_NAMES)], 'names': list(STATE_NAMES)},
        ACTION_KEY: {'dtype': 'float32', 'shape': [len(ACTION_NAMES)], 'names': list(ACTION_NAMES)},
        'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
        'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        WEIGHT_KEY: {'dtype': 'float32', 'shape': [1], 'names': [WEIGHT_KEY]},
    }
    for key in CAMERAS:
        features[key] = {'dtype': 'video', 'shape': list(IMAGE_SHAPE),
                         'names': ['height', 'width', 'channels']}
    return features


def build_info(episodes, frames, tasks, fps):
    return {
        'codebase_version': CODEBASE_VERSION,
        'robot_type': ROBOT_TYPE,
        'state_schema': STATE_SCHEMA,
        'state_schema_version': STATE_SCHEMA_VERSION,
        'state_dim': len(STATE_NAMES),
        'action_loss_mask_feature': WEIGHT_KEY,
        'action_policy_version': 'rl_best_of_n_rollout_v1',
        'action_limits_rad_s': list(ACTION_LIMITS_RAD_S),
        'total_episodes': episodes,
        'total_frames': frames,
        'total_tasks': tasks,
        'chunks_size': CHUNKS_SIZE,
        'data_files_size_in_mb': DATA_FILES_SIZE_IN_MB,
        'video_files_size_in_mb': VIDEO_FILES_SIZE_IN_MB,
        'fps': fps,
        'splits': {'train': f'0:{episodes}'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': build_features(),
    }


def encode_video(frames, output_path, fps):
    """Write one camera stream with the same encoder settings as the reference export.

    ``quality=8`` and ``macro_block_size=1`` are what produced the reference
    files (h264 High, yuv420p, 256x256, 10 fps, ~1.33 Mbit/s). ``macro_block_size``
    matters here: imageio otherwise pads to a multiple of 16, and 256 already is
    one, but leaving it at the default invites a silent resize on any future
    resolution change.
    """
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=float(fps), codec='libx264',
                                quality=8, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()


def collect_plan(selected, args):
    """Resolve every selected rollout to a concrete episode plan before writing anything."""
    plan = []
    skipped = []
    for rank, item in enumerate(selected):
        if args.max_episodes is not None and len(plan) >= args.max_episodes:
            break
        folder = Path(item['path'])
        reason = None
        try:
            text, source = task_text_for(folder, args.default_task_text)
            trajectory = load_trajectory(folder, args.state_source, args.action_source)
            if args.min_frames and trajectory['length'] < args.min_frames:
                reason = f'too_short:{trajectory["length"]}<{args.min_frames}'
            elif abs(trajectory['fps'] - float(args.fps)) > 1.0e-6:
                # Timestamps and to_timestamp are derived from --fps, so a
                # rollout recorded at any other rate would silently desync the
                # video from the tabular frames.
                reason = f'fps_mismatch:stored={trajectory["fps"]}:requested={args.fps}'
        except ValueError as exc:
            reason = str(exc)
            text, source, trajectory = None, None, None
        if reason is not None:
            skipped.append({'rank': rank, 'path': str(folder),
                            'scene_seed': item.get('scene_seed'), 'reason': reason})
            continue
        plan.append({'rank': rank, 'selection': item, 'path': folder, 'task_text': text,
                     'task_source': source, 'trajectory': trajectory})
    if not plan:
        raise ValueError(f'No convertible episodes; skipped {len(skipped)}')
    return plan, skipped


def assign_tasks(plan):
    """Map task texts to stable indices, ordered by first appearance."""
    order = []
    for episode in plan:
        if episode['task_text'] not in order:
            order.append(episode['task_text'])
    index_of = {text: position for position, text in enumerate(order)}
    for episode in plan:
        episode['task_index'] = index_of[episode['task_text']]
    return order


def build_rows(plan, fps, weights):
    """Flatten episodes into global frame rows, assigning ``index`` as we go."""
    rows = []
    cursor = 0
    for position, episode in enumerate(plan):
        trajectory = episode['trajectory']
        length = trajectory['length']
        weight = float(weights[position])
        episode['episode_index'] = position
        episode['from_index'] = cursor
        episode['to_index'] = cursor + length
        for frame in range(length):
            rows.append({
                'index': cursor + frame,
                'episode_index': position,
                'frame_index': frame,
                'timestamp': float(frame) / float(fps),
                'task_index': episode['task_index'],
                STATE_KEY: [float(value) for value in trajectory['state'][frame]],
                ACTION_KEY: [float(value) for value in trajectory['action'][frame]],
                WEIGHT_KEY: weight,
            })
        cursor += length
    return rows


def write_dataset(plan, rows, tasks, stats, info, output, fps, verbose):
    """Emit the v3.0 tree: one chunk per episode, matching the reference layout."""
    root = Path(output)
    meta = root / 'meta'
    meta.mkdir(parents=True, exist_ok=True)

    by_episode = {}
    for row in rows:
        by_episode.setdefault(row['episode_index'], []).append(row)

    for position, episode in enumerate(plan):
        chunk = _chunk_dir(episode['episode_index'])
        name = _file_stem(0)
        episode_rows = by_episode[position]
        frame = pd.DataFrame(episode_rows, columns=[
            'index', 'episode_index', 'frame_index', 'timestamp', 'task_index',
            STATE_KEY, ACTION_KEY, WEIGHT_KEY])
        data_dir = root / 'data' / chunk
        data_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_dir / f'{name}.parquet', index=False)

        length = len(episode_rows)
        meta_row = {
            'episode_index': episode['episode_index'],
            'tasks': [episode['task_text']],
            'length': length,
            'dataset_from_index': episode['from_index'],
            'dataset_to_index': episode['to_index'],
            'meta/episodes/chunk_index': episode['episode_index'],
            'meta/episodes/file_index': 0,
            'data/chunk_index': episode['episode_index'],
            'data/file_index': 0,
        }
        trajectory = episode['trajectory']
        images = decode_frames(trajectory['frames_path'], length)
        for key in CAMERAS:
            video_dir = root / 'videos' / key / chunk
            video_dir.mkdir(parents=True, exist_ok=True)
            encode_video(images[key], video_dir / f'{name}.mp4', fps)
            meta_row[f'videos/{key}/chunk_index'] = episode['episode_index']
            meta_row[f'videos/{key}/file_index'] = 0
            meta_row[f'videos/{key}/from_timestamp'] = 0.0
            meta_row[f'videos/{key}/to_timestamp'] = float(length - 1) / float(fps)
        episodes_dir = meta / 'episodes' / chunk
        episodes_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([meta_row]).to_parquet(episodes_dir / f'{name}.parquet', index=False)
        if verbose:
            print(f'  [{position + 1}/{len(plan)}] episode {episode["episode_index"]} '
                  f'seed={episode["selection"].get("scene_seed")} frames={length} '
                  f'score={episode["selection"].get("score")} '
                  f'weight={episode["weight"]:.6f}', flush=True)

    pd.DataFrame({'task_index': list(range(len(tasks)))},
                 index=pd.Index(list(tasks))).to_parquet(meta / 'tasks.parquet', index=True)
    _atomic_json(meta / 'info.json', info)
    _atomic_json(meta / 'stats.json', stats)


def build_provenance(args, manifest, plan, skipped, rows, tasks, stats, stats_warnings, fps):
    return {
        'schema_version': SCHEMA_VERSION,
        'created_utc': _utcnow(),
        'codebase_version': CODEBASE_VERSION,
        'source_selection': str(Path(args.selection).resolve()),
        'source_selection_schema': manifest.get('schema_version'),
        'source_selection_created_utc': manifest.get('created_utc'),
        'selection_criteria': manifest.get('criteria'),
        'reference_dataset': str(Path(args.reference_dataset).resolve()) if args.reference_dataset else None,
        'fps': fps,
        'state_source': args.state_source,
        'action_source': args.action_source,
        'stats_mode': args.stats_mode,
        'stats_reference_keys': sorted(REFERENCE_STATS_KEYS) if args.stats_mode == 'reference' else [],
        'stats_warnings': stats_warnings,
        'weight_mode': args.weight_mode,
        'weight_temperature': args.temperature,
        'weight_normalized_to_mean_one': True,
        'weight_basis': (f'computed over the {len(plan)} episode(s) that survived '
                         f'filtering, not the {len(plan) + len(skipped)} that were '
                         f'selected; weight_raw is indexed by episode_index, rank by '
                         f'the selection'),
        'total_episodes': len(plan),
        'total_frames': len(rows),
        'total_tasks': len(tasks),
        'features': sorted(build_features()),
        'dropped_reference_features': [
            'observation.effort', 'observation.stage_current_id', 'observation.scoop_index',
            'observation.scoops_target', 'observation.scoops_min', 'observation.scoops_max',
            'observation.scoops_completed', 'observation.recovery_active',
            'observation.recovery_type_id', 'observation.recovery_attempt_index',
            'action_is_expert',
        ],
        'drop_reason': 'not measured by the rollout store; writing them would fabricate supervision',
        'episodes': [{
            'episode_index': episode['episode_index'],
            'rank': episode['rank'],
            'path': str(episode['path']),
            'run': episode['selection'].get('run'),
            'scene_seed': episode['selection'].get('scene_seed'),
            'score': episode['selection'].get('score'),
            'reward': episode['selection'].get('reward'),
            'checkpoint_sha256': episode['selection'].get('checkpoint_sha256'),
            'policy_type': episode['selection'].get('policy_type'),
            'execution_source': episode['selection'].get('execution_source'),
            'replan_interval': episode['selection'].get('replan_interval'),
            'frames': episode['trajectory']['length'],
            'from_index': episode['from_index'],
            'to_index': episode['to_index'],
            'task_text': episode['task_text'],
            'task_text_source': episode['task_source'],
            'task_index': episode['task_index'],
            'weight_raw': episode['weight_raw'],
            'weight': episode['weight'],
            'arrays_sha256': episode['trajectory']['arrays_sha256'],
            'frames_sha256': episode['trajectory']['frames_sha256'],
        } for episode in plan],
        'skipped': skipped,
    }


def print_plan(plan, tasks, weights, limit):
    print(f'Episodes: {len(plan)}   Frames: {sum(e["trajectory"]["length"] for e in plan)}   '
          f'Tasks: {len(tasks)}')
    print(f'{"idx":>4} {"seed":>6} {"frames":>7} {"score":>7} {"weight":>9}  task')
    for position, episode in enumerate(plan[:limit]):
        selection = episode['selection']
        print(f'{position:>4} {str(selection.get("scene_seed")):>6} '
              f'{episode["trajectory"]["length"]:>7} {str(selection.get("score")):>7} '
              f'{weights[position]:>9.5f}  {episode["task_text"][:58]}')
    if len(plan) > limit:
        print(f'  ... {len(plan) - limit} more')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Build a LeRobot v3.0 fine-tuning dataset from a Best-of-N selection.')
    parser.add_argument('--selection', required=True,
                        help='selection.json produced by select_rollouts')
    parser.add_argument('--output', required=True, help='Dataset root to write')
    parser.add_argument('--reference-dataset', default='/root/gpufree-data/data/export',
                        help='Dataset whose stats.json is copied under --stats-mode reference')
    parser.add_argument('--stats-mode', choices=STATS_MODES, default='reference',
                        help='reference: copy state/action/image stats from the reference dataset')
    parser.add_argument('--weight-mode', choices=WEIGHT_MODES, default='uniform',
                        help='Per-trajectory action_loss_weight; uniform is plain rejection sampling')
    parser.add_argument('--temperature', type=float, default=100.0,
                        help='Softmax temperature in score units')
    parser.add_argument('--state-source', choices=sorted(STATE_SOURCES), default='pre',
                        help='pre: the observation the policy consumed (causally correct)')
    parser.add_argument('--action-source', choices=sorted(ACTION_SOURCES), default='applied',
                        help='applied: the clipped command actually sent to the simulator')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS)
    parser.add_argument('--min-frames', type=int, default=1)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--default-task-text', default=None,
                        help='Fallback when an episode directory records no task text')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing dataset that this script produced')
    parser.add_argument('--dry-run', action='store_true', help='Plan and report, write nothing')
    parser.add_argument('--print-limit', type=int, default=20)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def prepare_output(path, overwrite):
    folder = Path(path)
    if not folder.exists():
        return
    produced = (folder / PROVENANCE_NAME).is_file()
    contents = [item.name for item in folder.iterdir()]
    if not produced:
        raise ValueError(f'{folder} exists, was not produced by this script, and cannot be replaced: {contents[:6]}')
    if not overwrite:
        raise ValueError(f'{folder} already exists; pass --overwrite to replace it')
    shutil.rmtree(folder)


def build_dataset(args):
    manifest, selected = load_selection(args.selection)
    plan, skipped = collect_plan(selected, args)
    tasks = assign_tasks(plan)
    # Weights are computed over the plan, not over ``selected``. A rollout that
    # was rejected for being too short or recorded at the wrong fps contributes
    # no rows, and letting its score set the floor of ``score`` mode or enter
    # the partition function of ``softmax`` mode would compress the weights of
    # the episodes that *are* in the dataset. Consequence worth remembering when
    # reading rl_provenance.json: ``rank`` indexes the selection, ``weight_raw``
    # indexes the plan, and the two disagree as soon as anything was skipped.
    weights, weights_raw = episode_weights(
        [episode['selection'] for episode in plan], args.weight_mode, args.temperature)
    for episode, weight, raw in zip(plan, weights, weights_raw):
        episode['weight'] = weight
        episode['weight_raw'] = raw
    fps = float(args.fps)
    rows = build_rows(plan, fps, weights)

    reference_stats = {}
    if args.stats_mode == 'reference':
        if not args.reference_dataset:
            raise ValueError('--stats-mode reference needs --reference-dataset')
        reference_path = Path(args.reference_dataset) / 'meta' / 'stats.json'
        if not reference_path.is_file():
            raise ValueError(f'No stats.json at {reference_path}')
        reference_stats = json.loads(reference_path.read_text(encoding='utf-8'))
    stats, _, stats_warnings = merged_stats(rows, args.stats_mode, reference_stats)
    info = build_info(len(plan), len(rows), len(tasks), fps)
    provenance = build_provenance(args, manifest, plan, skipped, rows, tasks, stats,
                                  stats_warnings, fps)

    if not args.quiet:
        print_plan(plan, tasks, weights, args.print_limit)
        if skipped:
            print(f'Skipped {len(skipped)} selected rollout(s):')
            for item in skipped[:args.print_limit]:
                print(f'  seed={item["scene_seed"]} {item["reason"][:110]}')
        for warning in stats_warnings:
            print(f'  warning: {warning}')

    if args.dry_run:
        return {'dry_run': True, 'provenance': provenance, 'stats_warnings': stats_warnings}

    prepare_output(args.output, args.overwrite)
    write_dataset(plan, rows, tasks, stats, info, args.output, fps, not args.quiet)
    _atomic_json(Path(args.output) / PROVENANCE_NAME, provenance)
    return {'dry_run': False, 'provenance': provenance, 'stats_warnings': stats_warnings}


def main(argv=None):
    args = parse_args(argv)
    try:
        result = build_dataset(args)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    provenance = result['provenance']
    if not args.quiet:
        target = '(dry run, nothing written)' if result['dry_run'] else args.output
        print(f'\nWrote {provenance["total_episodes"]} episodes / '
              f'{provenance["total_frames"]} frames / {provenance["total_tasks"]} tasks '
              f'-> {target}')
        print(f'stats_mode={args.stats_mode} weight_mode={args.weight_mode} '
              f'state={args.state_source} action={args.action_source} fps={provenance["fps"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
