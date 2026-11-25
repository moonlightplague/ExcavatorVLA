"""ACT-internal causal visual/state context; no task phases or action overrides.

The public inference API still accepts one current observation. Its bounded
feature buffer advances only when the policy predicts a chunk, and is cleared
by ACTPolicy.reset(). Explicit training windows never mutate that buffer.
"""
from collections import deque

import torch
from torch import nn
from lerobot.policies.act.modeling_act import ACT, ACTEncoder

HISTORY_MASK = 'observation_history_is_pad'
VERSION = 'act_internal_prediction_history_v1'


class TemporalContext(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        dim = config.dim_model
        self.state_projection = nn.Linear(config.robot_state_feature.shape[0], dim)
        self.visual_projection = nn.Linear(channels, dim)
        self.time_embedding = nn.Embedding(config.temporal_horizon, dim)
        self.slot_embedding = nn.Embedding(len(config.image_features) + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, config.n_heads, dim * 2,
            dropout=config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.temporal_layers,
                                             norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.output = nn.Linear(dim, dim)
        # Adding temporal context initially preserves the complete native ACT.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, visual, padding):
        if padding.dtype != torch.bool or padding.shape != state.shape[:2]:
            raise ValueError('Temporal padding must be boolean [batch, observations]')
        if padding[:, -1].any() or (padding[:, 1:] & ~padding[:, :-1]).any():
            raise ValueError('Only a missing initial history prefix may be padded')
        tokens = torch.cat((self.state_projection(state).unsqueeze(2),
                            self.visual_projection(visual)), dim=2)
        batch, times, slots, dim = tokens.shape
        tokens = (tokens + self.time_embedding.weight[None, :, None]
                  + self.slot_embedding.weight[None, None, :])
        encoded = self.encoder(tokens.reshape(batch, times * slots, dim),
            src_key_padding_mask=padding.repeat_interleave(slots, dim=1))
        # Latest state token attends to all valid history/camera tokens.
        return self.output(encoded[:, -slots])


class TemporalACTEncoder(ACTEncoder):
    history_residual = None

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        if self.history_residual is not None:
            x = torch.cat((x[:1], x[1:2] + self.history_residual.unsqueeze(0), x[2:]), dim=0)
        return super().forward(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)


class TemporalACT(ACT):
    def __init__(self, config):
        super().__init__(config)
        if not config.robot_state_feature or len(config.image_features) != 3:
            raise ValueError('Temporal ACT requires the current state and three canonical cameras')
        encoder = TemporalACTEncoder(config)
        encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.encoder = encoder
        channels = self.encoder_img_feat_input_proj.in_channels
        self.temporal_context = TemporalContext(config, channels)
        self.reset_history()

    def reset_history(self):
        self._history = deque(maxlen=self.config.temporal_horizon)

    @property
    def history_size(self):
        return len(self._history)

    def _visual_features(self, images):
        return torch.stack([self.backbone(image)['feature_map'].mean(dim=(-2, -1))
                            for image in images], dim=1)

    def _current_forward(self, current, state, visual, padding):
        residual = self.temporal_context(state, visual, padding)
        self.encoder.history_residual = residual
        try:
            return super().forward(current)
        finally:
            self.encoder.history_residual = None

    def forward(self, batch):
        state = batch['observation.state']
        if state.ndim != 3 or state.shape[1] != self.config.temporal_horizon:
            raise ValueError('Temporal training/offline prediction requires an explicit causal window')
        padding = batch.get(HISTORY_MASK)
        if padding is None:
            raise ValueError('Explicit windows must include their initial-history padding mask')
        images = batch['observation.images']
        if any(image.ndim != 5 or image.shape[:2] != state.shape[:2] for image in images):
            raise ValueError('Historical RGB and state windows must be synchronized')
        batch_size, times = state.shape[:2]
        pooled = self._visual_features([image.flatten(0, 1) for image in images])
        pooled = pooled.reshape(batch_size, times, len(images), -1)
        current = {**batch, 'observation.state': state[:, -1],
                   'observation.images': [image[:, -1] for image in images]}
        return self._current_forward(current, state, pooled, padding)

    def forward_online(self, batch):
        state = batch['observation.state']
        if state.ndim == 3:
            return self.forward(batch)
        if state.ndim != 2 or state.shape[0] != 1:
            raise ValueError('Online temporal ACT expects a single current observation; batch windows explicitly')
        pooled = self._visual_features(batch['observation.images'])
        if self._history and self._history[0][0].device != state.device:
            raise ValueError('Reset temporal history before changing inference device')
        self._history.append((state.detach().clone(), pooled.detach().clone()))
        missing = self.config.temporal_horizon - len(self._history)
        entries = [self._history[0]] * missing + list(self._history)
        states = torch.stack([entry[0] for entry in entries], dim=1)
        visuals = torch.stack([entry[1] for entry in entries], dim=1)
        padding = torch.tensor([[True] * missing + [False] * len(self._history)],
                               device=state.device, dtype=torch.bool)
        return self._current_forward(batch, states, visuals, padding)
