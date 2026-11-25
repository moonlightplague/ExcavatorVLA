"""Model-window training/validation helpers; ordinary rollout code is unchanged."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import training_batch
from .continuation import validate_continuation
from ...data.contract import CAMERAS, file_hash
from ...evaluation.offline_metrics import summarize_actions


def initialize_temporal(policy, checkpoint, config, report, split, episodes, stats):
    from safetensors.torch import load_file
    previous = json.loads((Path(checkpoint) / 'config.json').read_text())
    if previous.get('temporal_horizon', 1) != 1:
        raise ValueError('Temporal adapter initialization expects a single-frame ACT source')
    provenance = json.loads((Path(checkpoint) / 'benchmark_provenance.json').read_text())
    if provenance.get('adapter') != 'lerobot_act_original_v1':
        raise ValueError('Temporal initialization requires the original ACT loss checkpoint')
    initialization = validate_continuation(checkpoint, config, report, split, episodes, stats)
    result = policy.load_state_dict(load_file(str(Path(checkpoint) / 'model.safetensors')), strict=False)
    expected_new = {name for name in policy.state_dict() if name.startswith('model.temporal_context.')}
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise ValueError(f'Unexpected weight-transfer keys: {result}')
    initialization.update(mode='exact_core_transfer_plus_zero_output_temporal_adapter_fresh_optimizer',
        new_parameter_keys=sorted(expected_new), source_step=initialization['source_step'])
    return initialization


def zero_adapter_roundtrip(policy, pre, batch, checkpoint, device):
    from .inference import load_act
    source, _, _ = load_act(checkpoint, device, original_act=True)
    history = training_batch(pre, batch)
    current = {key: batch[key][:, -1] for key in ('observation.state', *CAMERAS)}
    policy.eval()
    with torch.inference_mode():
        expected = source.predict_action_chunk(pre(current)).float()
        actual = policy.predict_action_chunk(history).float()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    policy.reset()
    return dict(status='passed', max_normalized_action_difference=float((actual - expected).abs().max()),
                scope='Real causal window; zero-output temporal branch preserves the source ACT')


def evaluate_temporal_loader(policy, pre, post, loader, limits, max_batches=0):
    # Explicit windows must stay independent: no online cache is used in validation.
    predicted, targets, pads, weights, recovery, episodes, frames = [], [], [], [], [], [], []
    policy.eval()
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(training_batch(pre, batch))
            actions = torch.stack([post(chunk[:, i]) for i in range(chunk.shape[1])], dim=1).cpu().numpy()
        predicted.append(actions)
        targets.append(batch['action'].cpu().numpy())
        pads.append(batch['action_is_pad'].cpu().numpy())
        weights.append(batch['action_loss_weight'].cpu().numpy().reshape(actions.shape[:2]))
        recovery.append(batch['observation.recovery_active'].cpu().numpy().reshape(actions.shape[:2]))
        episodes.append(batch['episode_index'].cpu().numpy())
        frames.append(batch['frame_index'].cpu().numpy())
    if not predicted:
        raise ValueError('No temporal validation observations')
    arrays = {key: np.concatenate(value) for key, value in dict(predicted=predicted, target=targets,
        padding=pads, supervision=weights, recovery=recovery, episode_index=episodes, frame_index=frames).items()}
    metrics = summarize_actions(**{key: arrays[key] for key in
        ('predicted', 'target', 'padding', 'supervision', 'recovery')}, limits=limits)
    metrics.update(episodes=len(np.unique(arrays['episode_index'])),
        observation_history='Three causal decision observations; independent explicit validation windows',
        temporal_stride=policy.config.temporal_stride)
    policy.train()
    return metrics, arrays
