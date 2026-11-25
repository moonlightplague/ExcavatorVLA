"""Learned absolute command-pose action representation, decoded inside ACT.

Bounds and normalization come only from training episodes. This module does
not receive task phases, cycle counts, success flags or external goal logic.
"""
import torch
from torch import nn
from lerobot.policies.act.persistent_extension import ActionMemory


class CommandPoseMemory(ActionMemory):
    def __init__(self, input_dim, hidden_dim, action_dim, config):
        super().__init__(input_dim, hidden_dim, action_dim)
        self.chunk_size = config.chunk_size
        self.pose_state = nn.Linear(config.robot_state_feature.shape[0], action_dim)
        for name, value, size in (
            ('pose_center', config.command_pose_center, 4),
            ('pose_radius', config.command_pose_radius, 4),
            ('state_mean', config.state_normalization_mean, 28),
            ('state_std', config.state_normalization_std, 28),
            ('action_mean', config.action_normalization_mean, 4),
            ('action_std', config.action_normalization_std, 4)):
            if value is None or len(value) != size:
                raise ValueError(f'Missing training-fitted command-pose statistic: {name}')
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        if (self.pose_radius <= 0).any() or (self.action_std <= 0).any() or (self.state_std <= 0).any():
            raise ValueError('Invalid command-pose scales')

    def current_command(self, features):
        state = features[..., :28] * self.state_std + self.state_mean
        return state[..., :4] + state[..., 8:12]

    def target_poses(self, features, normalized_actions):
        physical = normalized_actions * self.action_std + self.action_mean
        return self.current_command(features).unsqueeze(-2) + .1 * physical.cumsum(-2)

    def decode(self, features, poses):
        previous = torch.cat((self.current_command(features).unsqueeze(-2), poses[..., :-1, :]), dim=-2)
        physical_velocity = (poses - previous) * 10.
        return (physical_velocity - self.action_mean) / self.action_std

    def predict_poses(self, features, hidden=None):
        encoded, hidden = self.gru(self.project(features), hidden)
        logits = self.output(encoded) + self.pose_state(features[..., :28])
        poses = self.pose_center + self.pose_radius * logits.reshape(*features.shape[:-1], self.chunk_size, 4).tanh()
        return poses, hidden

    def forward(self, features, hidden=None):
        poses, hidden = self.predict_poses(features, hidden)
        action = self.decode(features, poses).flatten(-2)
        # PersistentACT's public interface adds its frozen base prediction.
        return action - features[..., -self.chunk_size*4:], hidden
