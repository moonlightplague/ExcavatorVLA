"""Diffusion Policy training loss with correct episode padding."""
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def training_batch(preprocessor, batch):
    # LeRobot 0.4.4's batch converter drops arbitrary action_* labels. Keep
    # action_is_pad explicitly so the diffusion loss can mask padded targets.
    processed = preprocessor({key: value for key, value in batch.items()
                              if key in ('observation.state', 'action', 'action_is_pad')
                              or key.startswith('observation.images.')})
    return processed


class ExcavatorDiffusionPolicy(DiffusionPolicy):
    """Same architecture/checkpoint keys as DiffusionPolicy; only the training metrics differ."""

    def forward(self, batch):
        loss, _ = super().forward(batch)
        return loss, {'diffusion_loss': loss.detach().item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch, noise=None):
        # The stock implementation reads from a rollout queue that is empty during
        # offline evaluation. Rebuild the observation layout from the preprocessed
        # batch and sample the full horizon (matching the benchmark's chunk contract)
        # instead of the truncated n_action_steps slice that select_action returns.
        batch = dict(batch)
        if ACTION in batch:
            batch.pop(ACTION)
        if self.config.image_features:
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # State may be (B, dim) in live evaluation (no dataset delta); add the
        # n_obs_steps axis to match generate_actions' (B, n_obs_steps, dim) contract.
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        batch_size = batch[OBS_STATE].shape[0]
        global_cond = self.diffusion._prepare_global_conditioning(batch)
        return self.diffusion.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
