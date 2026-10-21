"""VQ-BeT future-action chunks and mean/std + ImageNet preprocessing statistics."""
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


def make_vqbet_dataset(repo_id, root, episodes, n_action_pred_token, action_chunk_size,
                       n_obs_steps=1, video_backend='pyav'):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = load_info(root)
    fps = info['fps']
    # action_delta_indices = range(1 - n_obs_steps, n_action_pred_token + action_chunk_size - 1)
    n_action_delta = n_action_pred_token + action_chunk_size - 1
    action_delta = [i / fps for i in range(n_action_delta)]
    obs_delta = [(1 - n_obs_steps + i) / fps for i in range(n_obs_steps)]
    return LeRobotDataset(repo_id, root=Path(root), episodes=list(episodes),
                          delta_timestamps={'action': action_delta, 'action_loss_weight': action_delta,
                                            'observation.recovery_active': action_delta,
                                            'observation.state': obs_delta},
                          video_backend=video_backend)
