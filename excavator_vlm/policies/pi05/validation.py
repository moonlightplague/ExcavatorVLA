"""CPU checkpoint coordination and deterministic, episode-balanced previews."""
from datetime import timedelta

import numpy as np
from torch.utils.data import Subset


def checkpoint_group(accelerator):
    if accelerator.num_processes == 1:
        return None
    import torch.distributed as dist
    # Never leave a CUDA collective pending while only rank 0 validates or
    # writes checkpoints. Gloo also keeps serialized RNG objects off the GPU.
    return dist.new_group(backend='gloo', timeout=timedelta(hours=6))


def checkpoint_barrier(accelerator, group):
    if accelerator.num_processes > 1:
        if group is None:
            raise ValueError('Distributed checkpoint coordination requires its CPU group')
        import torch.distributed as dist
        dist.barrier(group=group)


def validation_preview(dataset, samples_per_episode):
    """Use actual HF row positions; never decode images just to choose rows."""
    if type(samples_per_episode) is not int or samples_per_episode < 0:
        raise ValueError('validation_samples_per_episode must be a nonnegative integer')
    if not samples_per_episode:
        return dataset, {'method': 'all_frames', 'available_frames': len(dataset),
                         'selected_frames': len(dataset), 'partial_evaluation': False}
    rows = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
    if len(rows) != len(dataset) or rows.duplicated(['episode_index', 'frame_index']).any():
        raise ValueError('Validation frame metadata is missing or duplicated')
    indices, selected = [], []
    for episode in sorted(rows.episode_index.unique()):
        positions = np.flatnonzero(rows.episode_index.to_numpy() == episode)
        positions = positions[np.argsort(rows.frame_index.to_numpy()[positions], kind='stable')]
        offsets = np.linspace(0, len(positions) - 1, min(samples_per_episode, len(positions)), dtype=int)
        for index in positions[offsets]:
            indices.append(int(index))
            selected.append({'episode_index': int(episode), 'frame_index': int(rows.iloc[index].frame_index)})
    if not indices:
        raise ValueError('Validation preview is empty')
    return Subset(dataset, indices), {
        'method': 'uniform_frames_per_episode_v1', 'samples_per_episode': samples_per_episode,
        'available_frames': len(dataset), 'selected_frames': len(indices),
        'partial_evaluation': len(indices) < len(dataset), 'selected_observations': selected}
