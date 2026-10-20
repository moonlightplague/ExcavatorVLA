"""Causal PI05 chunks, language, and quantiles fitted only on training frames."""
from pathlib import Path

import numpy as np

from ...data.contract import load_info


def training_stats(frames, episodes):
    selected = frames[frames.episode_index.isin(episodes)]
    if selected.empty:
        raise ValueError('No training frames selected')
    stats = {}
    for key in ('observation.state', 'action'):
        values = np.stack(selected[key]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite training values: {key}')
        stats[key] = {name: value.tolist() for name, value in {
            'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0), 'max': values.max(0),
            'q01': np.quantile(values, .01, axis=0), 'q99': np.quantile(values, .99, axis=0),
            'count': np.array([len(values)]),
        }.items()}
    return stats


def make_pi05_dataset(repo_id, root, episodes, chunk_size, video_backend='pyav'):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    delta = [i / load_info(root)['fps'] for i in range(chunk_size)]
    return LeRobotDataset(repo_id, root=Path(root), episodes=list(episodes),
                          delta_timestamps={'action': delta, 'action_loss_weight': delta,
                                            'observation.recovery_active': delta},
                          video_backend=video_backend)
