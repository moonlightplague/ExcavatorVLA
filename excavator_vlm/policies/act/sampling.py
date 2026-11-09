"""Episode-balanced transition windows mixed with uniform training observations."""
from collections import Counter
import json
from pathlib import Path

import numpy as np


def observation_probabilities(keys, training_episodes, manifest=None, probability=0.0):
    """Address dataset rows by identity, never by a presumed parquet/HF row order."""
    keys = [(int(ep), int(frame)) for ep, frame in keys]
    allowed = set(map(int, training_episodes))
    if not keys or len(set(keys)) != len(keys) or not {ep for ep, _ in keys} <= allowed:
        raise ValueError('Sampling requires unique training-only frame identities')
    weights = np.full(len(keys), (1 - probability) / len(keys), dtype=np.float64)
    selected = set()
    if manifest is not None:
        if manifest.get('labels_changed') is not False:
            raise ValueError('Transition manifest must preserve the original action labels')
        windows = manifest['windows']
        selected = {(int(w['episode_index']), int(w['observation_frame'])) for w in windows}
        if len(selected) != len(windows) or not selected <= set(keys):
            raise ValueError('Transition manifest has duplicate, held-out, or missing dataset frames')
        if 'training_episodes' in manifest and set(manifest['training_episodes']) != allowed:
            raise ValueError('Transition manifest training episodes differ from the frozen split')
    if probability:
        if not selected:
            raise ValueError('Transition sampling requires nonempty training windows')
        counts = Counter(ep for ep, _ in selected)
        for i, (episode, frame) in enumerate(keys):
            if (episode, frame) in selected:
                weights[i] += probability / len(counts) / counts[episode]
    report = {'kind': 'replacement_uniform_plus_episode_balanced_transition_windows',
              'observations': len(keys), 'transition_windows': len(selected),
              'transition_probability': probability,
              'probability_sum': float(weights.sum()),
              'min_observation_probability': float(weights.min()),
              'max_observation_probability': float(weights.max())}
    return weights, report


def make_sampler(dataset, episodes, specification, seed):
    import torch
    from torch.utils.data import WeightedRandomSampler

    identities = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
    keys = list(zip(identities.episode_index, identities.frame_index))
    path = specification['manifest']
    manifest = json.loads(Path(path).read_text()) if path else None
    weights, report = observation_probabilities(keys, episodes, manifest,
                                               specification['transition_probability'])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True,
                                 generator=generator), report
