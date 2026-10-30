"""Persist scored rollouts so successful trajectories can be reused for fine-tuning.

The shared success evaluator scores a rollout but records only camera sha256
digests, so the frames themselves are thrown away. Best-of-N rejection sampling
needs the exact observation the policy consumed, paired with the action it
emitted and the particle counts that scored the outcome, which is what this
store writes next to ``steps.jsonl``.

Frame provenance is exact rather than approximate. The simulator renders at
1024x1024, downsamples to 256x256 and ships the uint8 array over the bridge
losslessly (zlib + base64). The evaluator feeds that same array to
``F.interpolate(..., (256, 256))``, which is an identity resample, so a stored
frame is bit-identical to the policy input and no re-encoding loss is added.

Two files are written on purpose. ``arrays.npz`` and ``meta.json`` are small and
carry everything needed to rank rollouts, so selection never has to touch the
~30 MB of frames belonging to trajectories it is about to discard.
"""
import json
import os
from pathlib import Path

import numpy as np

from ..data.contract import ACTION_NAMES, CAMERAS, STATE_NAMES

SCHEMA_VERSION = 'excavator_rollout_trajectory_v1'
CAMERA_IDS = tuple(str(index) for index in range(len(CAMERAS)))
PARTICLE_FIELDS = ('payload_particles', 'scooped_particles', 'retained_particles',
                   'spillage_particles', 'spillage_fraction')
IMAGE_SHAPE = (256, 256, 3)

# Row t of every array is line t of steps.jsonl, which mixes time references:
# its input_state precedes the action while its state and particle counts follow
# it. Both are kept so the causal pairing a policy needs is unambiguous.
FRAME_SEMANTICS = {
    'step': 'steps.jsonl line index',
    'observation.images.*': 'pre-action frame: the observation the policy consumed to emit action[t]',
    'observation.state': 'pre-action 28D state, steps.jsonl input_state',
    'observation.state_next': 'post-action 28D state, steps.jsonl state',
    'action.predicted': 'raw policy output before velocity clipping',
    'action.applied': 'clipped joint-velocity command actually sent to the simulator',
    'particles': 'particle_transfer_metrics read from the post-action reply, PARTICLE_FIELDS order',
    'bucket_position': 'post-action bucket joint position in rad',
    'replanned': 'True when this step ran inference instead of consuming a cached chunk action',
}


def _stack(rows, dtype, trailing=()):
    if not rows:
        return np.zeros((0, *trailing), dtype=dtype)
    return np.stack(rows).astype(dtype, copy=False)


def _atomic_npz(path, arrays):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def trajectory_reward(status):
    """Rank one finished rollout with the simulator's own particle counts.

    Binary success cannot order the rollouts that all end with an empty truck
    bed, which is the common case (5 of the 6 ACT pilot seeds). Subtracting
    spillage from delivered payload keeps them orderable, and it penalises
    exactly the behaviour that is missing: the bucket letting sand out on the
    way to the truck.

    Counts default to zero because an episode that fails before its first
    observation is finalized without any particle metric at all. The metrics are
    also written as JSON null rather than omitted in that case, so a missing key
    and a null value have to be handled the same way.
    """
    def count(key):
        value = status.get(key)
        return 0 if value is None else int(value)

    payload = count('payload_particles')
    spillage = count('spillage_particles')
    return {
        'episode_status': status.get('status'),
        'success': bool(status.get('success')),
        'failure_mode': status.get('failure_mode'),
        'payload_particles': payload,
        'spillage_particles': spillage,
        'scooped_particles': count('scooped_particles'),
        'max_bucket_particles': count('max_bucket_particles'),
        'score': payload - spillage,
    }


class TrajectoryStore:
    """Accumulate one rollout in memory and flush it when the episode ends.

    Frames are held until ``close`` rather than streamed, which peaks at roughly
    0.6 MB per step (about 200 MB for a full 350-step episode) per worker. That
    is affordable next to a simulator process and it keeps the writer simple
    enough that a truncated file cannot look like a valid trajectory.
    """

    def __init__(self, folder, fps, scene_seed, provenance=None):
        self.path = Path(folder) / 'trajectory'
        self.fps = float(fps)
        self.scene_seed = int(scene_seed)
        self.provenance = dict(provenance or {})
        self._frames = {camera: [] for camera in CAMERA_IDS}
        self._frame_sha256 = {camera: [] for camera in CAMERA_IDS}
        self._columns = {key: [] for key in (
            'step', 'state_pre', 'state_post', 'predicted', 'applied',
            'particles', 'bucket_position', 'replanned')}
        self._closed = False

    def __len__(self):
        return len(self._columns['step'])

    def add(self, *, step, frames, frame_sha256, state_pre, state_post,
            predicted, applied, particles, bucket_position, replanned):
        """Record one executed step; ``frames`` are the pre-action images."""
        for camera in CAMERA_IDS:
            rgb = np.ascontiguousarray(frames[camera], dtype=np.uint8)
            if rgb.shape != IMAGE_SHAPE:
                raise ValueError(f'Camera {camera} delivered {rgb.shape}, expected {IMAGE_SHAPE}')
            self._frames[camera].append(rgb)
            self._frame_sha256[camera].append(frame_sha256[camera])
        # spillage_fraction is null until something has been scooped, and the
        # column also holds integer counts, so it becomes float with NaN gaps.
        values = [particles[field] for field in PARTICLE_FIELDS]
        self._columns['step'].append(int(step))
        self._columns['state_pre'].append(np.asarray(state_pre, dtype=np.float32))
        self._columns['state_post'].append(np.asarray(state_post, dtype=np.float32))
        self._columns['predicted'].append(np.asarray(predicted, dtype=np.float32))
        self._columns['applied'].append(np.asarray(applied, dtype=np.float32))
        self._columns['particles'].append(
            np.array([np.nan if value is None else value for value in values], dtype=np.float64))
        self._columns['bucket_position'].append(float(bucket_position))
        self._columns['replanned'].append(bool(replanned))

    def close(self, status):
        """Flush arrays, frames and metadata. Idempotent; safe from a finally block."""
        if self._closed:
            return
        self._closed = True
        columns = self._columns
        self.path.mkdir(parents=True, exist_ok=True)
        _atomic_npz(self.path / 'arrays.npz', {
            'step': _stack(columns['step'], np.int32, ()),
            'observation.state': _stack(columns['state_pre'], np.float32, (28,)),
            'observation.state_next': _stack(columns['state_post'], np.float32, (28,)),
            'action.predicted': _stack(columns['predicted'], np.float32, (4,)),
            'action.applied': _stack(columns['applied'], np.float32, (4,)),
            'particles': _stack(columns['particles'], np.float64, (len(PARTICLE_FIELDS),)),
            'bucket_position': _stack(columns['bucket_position'], np.float32, ()),
            'replanned': _stack(columns['replanned'], np.bool_, ()),
        })
        _atomic_npz(self.path / 'frames.npz', {
            camera: _stack(self._frames[camera], np.uint8, IMAGE_SHAPE)
            for camera in CAMERA_IDS
        })
        _atomic_json(self.path / 'meta.json', {
            'schema_version': SCHEMA_VERSION,
            'frames': len(columns['step']),
            'fps': self.fps,
            'scene_seed': self.scene_seed,
            'image_shape': list(IMAGE_SHAPE),
            'image_dtype': 'uint8',
            'camera_keys': list(CAMERAS),
            'camera_ids': list(CAMERA_IDS),
            'state_names': list(STATE_NAMES),
            'action_names': list(ACTION_NAMES),
            'particle_fields': list(PARTICLE_FIELDS),
            'frame_semantics': FRAME_SEMANTICS,
            'frame_sha256': {camera: list(digests)
                             for camera, digests in self._frame_sha256.items()},
            'reward': trajectory_reward(status),
            'provenance': self.provenance,
            'files': {'frames': 'frames.npz', 'arrays': 'arrays.npz'},
        })
