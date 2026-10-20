"""Physical-unit action errors, masked at each future target timestamp."""
import time

import numpy as np
import torch

from ..data.contract import ACTION_NAMES, CAMERAS


def error_metrics(predicted, target, mask, limits, epsilon=.02):
    p, t = predicted[mask], target[mask]
    if not len(t):
        return {'timesteps': 0, 'mae': None, 'rmse': None}
    error = p - t
    moving = np.abs(t) > epsilon
    stopped = ~moving
    return {
        'timesteps': len(t), 'mae': float(np.abs(error).mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'zero_action_baseline': {'mae': float(np.abs(t).mean()), 'rmse': float(np.sqrt(np.mean(t ** 2)))},
        'direction_accuracy_moving': float((np.sign(p[moving]) == np.sign(t[moving])).mean()) if moving.any() else None,
        'near_zero_accuracy': float((np.abs(p[stopped]) <= epsilon).mean()) if stopped.any() else None,
        'predicted_saturation_fraction': float((np.abs(p) >= np.asarray(limits)).mean()),
        'per_joint': {name: {'mae': float(np.abs(error[:, i]).mean()),
                              'rmse': float(np.sqrt(np.mean(error[:, i] ** 2)))}
                      for i, name in enumerate(ACTION_NAMES)},
        'predicted': {name: getattr(p, name)(axis=0).tolist() for name in ('mean', 'std', 'min', 'max')},
        'target': {name: getattr(t, name)(axis=0).tolist() for name in ('mean', 'std', 'min', 'max')},
    }


def summarize_actions(predicted, target, padding, supervision, recovery, limits):
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 4:
        raise ValueError('Expected matching [frames, chunk, 4] arrays')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite predictions or targets')
    if any(x.shape != predicted.shape[:2] for x in (padding, supervision, recovery)):
        raise ValueError('Mask arrays must have [frames, chunk] shape')
    valid = ~padding.astype(bool)
    supervised = valid & (supervision > 0)
    first = np.zeros_like(valid)
    first[:, 0] = True
    return {
        'units': 'rad/s', 'moving_threshold_rad_s': .02,
        'evaluation': 'offline_chunks_at_recorded_observations_not_closed_loop',
        'frames': len(predicted), 'valid_target_timesteps': int(valid.sum()),
        'supervised_target_timesteps': int(supervised.sum()),
        'all_valid': error_metrics(predicted, target, valid, limits),
        'supervised': error_metrics(predicted, target, supervised, limits),
        'first_action': error_metrics(predicted, target, valid & first, limits),
        'normal': error_metrics(predicted, target, valid & (recovery == 0), limits),
        'recovery': error_metrics(predicted, target, valid & (recovery != 0), limits),
        'per_horizon': [error_metrics(predicted[:, i], target[:, i], valid[:, i], limits)
                        for i in range(predicted.shape[1])],
    }


def evaluate_loader(policy, pre, post, loader, limits, max_batches=0):
    predicted, targets, pads, weights, recovery, episode_ids, frame_ids, times = [], [], [], [], [], [], [], []
    policy.eval()
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        observation = {key: batch[key] for key in ('observation.state', *CAMERAS)}
        if policy.config.type == 'pi05':
            if 'task' not in batch:
                raise ValueError('PI05 offline evaluation requires the recorded language task')
            observation['task'] = batch['task']
        # CPU transfer waits for CUDA completion; timing covers preprocessing,
        # prediction, physical-unit unnormalization and output transfer.
        if next(policy.parameters()).is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(pre(observation))
            actions = torch.stack([post(chunk[:, i]) for i in range(chunk.shape[1])], dim=1).cpu().numpy()
        times.append((time.perf_counter() - started) * 1000)
        predicted.append(actions)
        targets.append(batch['action'].cpu().numpy())
        pads.append(batch['action_is_pad'].cpu().numpy())
        weights.append(batch['action_loss_weight'].cpu().numpy().reshape(actions.shape[:2]))
        recovery.append(batch['observation.recovery_active'].cpu().numpy().reshape(actions.shape[:2]))
        episode_ids.append(batch['episode_index'].cpu().numpy())
        frame_ids.append(batch['frame_index'].cpu().numpy())
    if not predicted:
        raise ValueError('Evaluation loader is empty')
    arrays = {key: np.concatenate(value) for key, value in {
        'predicted': predicted, 'target': targets, 'padding': pads, 'supervision': weights,
        'recovery': recovery, 'episode_index': episode_ids, 'frame_index': frame_ids,
    }.items()}
    metrics = summarize_actions(**{key: arrays[key] for key in (
        'predicted', 'target', 'padding', 'supervision', 'recovery')}, limits=limits)
    metrics['episodes'] = len(np.unique(arrays['episode_index']))
    metrics['latency'] = {
        'scope': 'per batch; preprocessing + inference + postprocessing + CPU transfer; excludes video decode',
        'batches': len(times), 'batch_size': loader.batch_size, 'warmup': 'first batch included',
        'mean_ms': float(np.mean(times)), 'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)), 'max_ms': float(max(times)),
        'observations_per_second': len(arrays['target']) / (sum(times) / 1000),
    }
    policy.train()
    return metrics, arrays
