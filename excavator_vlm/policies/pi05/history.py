"""Versioned causal windows shared by PI05 datasets, checkpoints and rollouts."""
from collections import deque
import copy

import torch

from ...data.contract import CAMERAS

VERSION = 'excavator_pi05_causal_history_v1'
MASK = 'history_is_pad'


def make_history(delta_indices=(0,), fps=10):
    if not isinstance(delta_indices, (list, tuple)):
        raise ValueError('History offsets must be an explicit list of integers')
    indices = list(delta_indices)
    if (not indices or len(indices) > 8 or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices)) or indices[-1] != 0 or indices[0] < -100):
        raise ValueError('History needs 1..8 increasing unique integer offsets in [-100,0], ending at 0')
    if type(fps) is not int or fps != 10:
        raise ValueError('Excavator history requires the recorded/deployed 10 Hz control cadence')
    return dict(version=VERSION, delta_indices=indices, fps=fps,
                image_order='time_then_camera', state_encoding='quantile_tokens_per_frame',
                padding='repeat_first_with_mask', cameras=list(CAMERAS))


def validate_history(value, n_obs_steps):
    if type(n_obs_steps) is not int or not 1 <= n_obs_steps <= 8:
        raise ValueError('Invalid history observation count')
    if value is None:
        if n_obs_steps != 1:
            raise ValueError('Multi-frame PI05 requires an explicit observation-history contract')
        value = make_history()
    if not isinstance(value, dict) or 'delta_indices' not in value or 'fps' not in value:
        raise ValueError('Invalid observation-history manifest')
    expected = make_history(value['delta_indices'], value['fps'])
    if value != expected or len(expected['delta_indices']) != n_obs_steps:
        raise ValueError('Observation-history contract differs from the configured PI05 inputs')
    return expected


def history_spec(config):
    return validate_history(getattr(config, 'excavator_observation_history', None), config.n_obs_steps)


def bind_history(config, value):
    config.excavator_observation_history = validate_history(value, config.n_obs_steps)
    return config


def observation_batch(batch, *, action=False):
    """Keep temporal padding through both training and independent offline inference."""
    keys = ['observation.state', *CAMERAS, 'task']
    if action:
        keys.append('action')
    result = {key: batch[key] for key in keys}
    result.update({key: value for key, value in batch.items()
                   if key == MASK or key in [f'{name}_is_pad' for name in ('observation.state', *CAMERAS)]})
    return result


def window_mask(batch, spec):
    """Validate batched windows and a common, left-padding-only camera/state mask."""
    state = batch['observation.state']
    count = len(spec['delta_indices'])
    if state.ndim != 3 or state.shape[1:] != (count, 28):
        raise ValueError(f'Expected batched history state [B,{count},28]')
    masks = [batch[k] for k in [MASK, *[f'{name}_is_pad' for name in ('observation.state', *CAMERAS)]]
             if k in batch]
    if not masks:
        raise ValueError('History padding mask is required, including for full windows')
    mask = masks[0]
    if mask.dtype != torch.bool or mask.shape != state.shape[:2]:
        raise ValueError('Expected boolean history padding [B,T]')
    if any(m.dtype != torch.bool or m.shape != mask.shape or not torch.equal(m.to(mask.device), mask)
           for m in masks[1:]):
        raise ValueError('Camera/state history masks must agree')
    if mask[:, -1].any() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError('Only an initial history prefix may be padded; current observation must be valid')
    for key in CAMERAS:
        if batch[key].shape != (*state.shape[:2], 3, 256, 256):
            raise ValueError(f'Expected synchronized [B,T,3,256,256] camera window: {key}')
    return mask


def temporal_images(batch, spec, preprocess_single_frame):
    """Reuse upstream image scaling; concatenate time-major, then canonical cameras."""
    mask = window_mask(batch, spec)
    images, masks = [], []
    for index in range(len(spec['delta_indices'])):
        frame = {key: batch[key][:, index] for key in CAMERAS}
        encoded, valid = preprocess_single_frame(frame)
        images.extend(encoded)
        masks.extend(m & ~mask[:, index].to(m.device) for m in valid)
    return images, masks


class ObservationHistory:
    """One append per executed control step, even between action-chunk replans."""
    def __init__(self, spec):
        self.spec = validate_history(spec, len(spec['delta_indices']))
        self.frames = deque(maxlen=1 - spec['delta_indices'][0])
        self.last_step = -1

    def reset(self):
        self.frames.clear()
        self.last_step = -1

    def append(self, batch, step):
        if type(step) is not int or step != self.last_step + 1:
            raise ValueError('History must receive consecutive 10 Hz observations starting at step 0')
        if batch['observation.state'].shape != (1, 28):
            raise ValueError('Online history expects one current 28D observation')
        for key in CAMERAS:
            if batch[key].shape != (1, 3, 256, 256):
                raise ValueError('Online history expects one current image per camera')
        self.frames.append({key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                            for key, value in batch.items()})
        self.last_step = step

    def batch(self):
        if not self.frames:
            raise ValueError('Append the current observation before querying history')
        offsets = self.spec['delta_indices']
        selected = [self.frames[max(0, len(self.frames) - 1 + offset)] for offset in offsets]
        result = {key: torch.stack([frame[key] for frame in selected], dim=1)
                  for key in ('observation.state', *CAMERAS)}
        result['task'] = copy.deepcopy(self.frames[-1]['task'])
        result[MASK] = torch.tensor([[self.last_step + offset < 0 for offset in offsets]], dtype=torch.bool)
        return result
