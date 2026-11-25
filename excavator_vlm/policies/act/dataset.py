"""ACT future-action chunks and mean/std + ImageNet preprocessing statistics."""
from pathlib import Path

import numpy as np

from ...data.contract import CAMERAS, load_info


def training_stats(frames, episodes):
    selected = frames[frames.episode_index.isin(episodes)]
    if selected.empty:
        raise ValueError('No training frames selected')
    stats = {}
    for key in ('observation.state', 'action'):
        x = np.stack(selected[key]).astype(np.float64)
        stats[key] = {name: value.tolist() for name, value in {
            'mean': x.mean(0), 'std': x.std(0), 'min': x.min(0), 'max': x.max(0),
            'count': np.array([len(x)]),
        }.items()}
    # ImageNet constants use no validation/test image statistics.
    for key in CAMERAS:
        stats[key] = {'mean': [[[v]] for v in (.485, .456, .406)],
                      'std': [[[v]] for v in (.229, .224, .225)]}
    return stats


def make_act_dataset(repo_id, root, episodes, chunk_size, video_backend='pyav',
                     temporal_horizon=1, temporal_stride=3):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = load_info(root)
    delta = [i / info['fps'] for i in range(chunk_size)]
    timestamps = {'action': delta, 'action_loss_weight': delta,
                  'observation.recovery_active': delta}
    if temporal_horizon > 1:
        if temporal_horizon != 3 or temporal_stride < 1:
            raise ValueError('Invalid causal ACT training window')
        offsets = [i * temporal_stride / info['fps'] for i in range(1 - temporal_horizon, 1)]
        timestamps.update({key: offsets for key in ('observation.state', *CAMERAS)})
    return LeRobotDataset(repo_id, root=Path(root), episodes=list(episodes),
                          delta_timestamps=timestamps,
                          video_backend=video_backend)
