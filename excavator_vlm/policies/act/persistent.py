"""Episode-long learned ACT memory. No task labels or external action rules."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from lerobot.policies.act.temporal_extension import TemporalACT


class ActionMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def projected(self, features):
        projected = self.project(features)
        if hasattr(self, 'angle_projection'):
            angle = features[..., :4] * self.angle_std + self.angle_mean
            projected = projected + self.angle_projection(torch.cat((angle.sin(), angle.cos()), dim=-1))
        return projected

    def encode(self, features, hidden=None):
        return self.gru(self.projected(features), hidden)

    def forward(self, features, hidden=None):
        encoded, hidden = self.encode(features, hidden)
        return self.output(encoded), hidden

    @torch.no_grad()
    def prefix_hidden(self, features, lengths):
        """Rebuild the actual episode prefix using current weights, without gradients."""
        hidden = features.new_zeros(1, features.shape[0], self.gru.hidden_size)
        valid = lengths > 0
        if valid.any():
            projected = self.projected(features[valid.to(features.device)])
            packed = pack_padded_sequence(projected, lengths[valid].cpu(), batch_first=True,
                                           enforce_sorted=False)
            _, actual = self.gru(packed)
            hidden[:, valid.to(features.device)] = actual
        return hidden


class PersistentACT(TemporalACT):
    def __init__(self, config):
        # Match cached batched convolutions and online singleton convolutions.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        super().__init__(config)
        channels = self.encoder_img_feat_input_proj.in_channels
        self.action_shape = (config.chunk_size, config.action_feature.shape[0])
        size = self.action_shape[0] * self.action_shape[1]
        features = config.robot_state_feature.shape[0] + len(config.image_features) * channels + size
        self.memory_adapter = ActionMemory(features, config.memory_hidden_dim, size)
        if getattr(config, 'action_parameterization', 'velocity') == 'command_pose':
            from .command_pose_extension import CommandPoseMemory
            self.memory_adapter = CommandPoseMemory(features, config.memory_hidden_dim, size, config)
        if getattr(config, 'memory_progress', False):
            if config.action_parameterization != 'velocity':
                raise ValueError('Progress training is supported for velocity actions only')
            self.memory_adapter.angle_projection = nn.Linear(8, config.memory_hidden_dim, bias=False)
            nn.init.zeros_(self.memory_adapter.angle_projection.weight)
            self.memory_adapter.register_buffer('angle_mean', torch.tensor(config.state_normalization_mean[:4], dtype=torch.float32))
            self.memory_adapter.register_buffer('angle_std', torch.tensor(config.state_normalization_std[:4], dtype=torch.float32))
            self.memory_adapter.future_state_head = nn.Sequential(nn.LayerNorm(config.memory_hidden_dim),
                nn.Linear(config.memory_hidden_dim, 128), nn.SiLU(), nn.Linear(128, 11))
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith('memory_adapter.'))
        self.feature_only = False
        self.train(False)

    def train(self, mode=True):
        # Frozen ACT predictions must use the same deterministic latent and dropout
        # behavior in cache construction, training and ordinary online inference.
        super().train(False)
        if hasattr(self, 'memory_adapter'):
            self.memory_adapter.train(mode)
        return self

    def reset_history(self):
        super().reset_history()
        self._memory_hidden = None

    def _current_forward(self, current, state, visual, padding):
        with torch.no_grad():
            base, _ = super()._current_forward(current, state, visual, padding)
        features = torch.cat((state[:, -1], visual[:, -1].flatten(1), base.flatten(1)), dim=-1)
        if self.feature_only:
            self.cached_features = features.detach()
            return base, (None, None)
        if features.shape[0] != 1:
            raise ValueError('Persistent online memory requires one episode at a time')
        delta, hidden = self.memory_adapter(features.unsqueeze(1), self._memory_hidden)
        self._memory_hidden = hidden.detach()
        return base + delta[:, 0].reshape_as(base), (None, None)

    def forward(self, batch):
        if not self.feature_only:
            raise ValueError('Train/validate persistent memory with ordered episode sequences')
        return super().forward(batch)

    @torch.no_grad()
    def features_from_spatial(self, state, visual, padding, spatial):
        """Frozen training cache path; numerically equivalent to raw image encoding."""
        current = {'observation.state': state[:, -1],
                   'observation.images': [value[:, :3, :1, :1] for value in spatial],
                   '_act_cached_backbone': spatial}
        self.feature_only = True
        try:
            base, _ = self._current_forward(current, state, visual, padding)
            return self.cached_features, base
        finally:
            self.feature_only = False
