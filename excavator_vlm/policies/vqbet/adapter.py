"""VQ-BeT integration: multi-camera support, training loss and chunk prediction."""
import torch

from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _validate_features_multi_camera(self):
    if self.crop_shape is not None:
        for key, image_ft in self.image_features.items():
            if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                raise ValueError(
                    f'`crop_shape` should fit within the images shapes. Got {self.crop_shape} '
                    f'for `crop_shape` and {image_ft.shape} for `{key}`.'
                )
    first_image_key, first_image_ft = next(iter(self.image_features.items()))
    for key, image_ft in self.image_features.items():
        if image_ft.shape != first_image_ft.shape:
            raise ValueError(
                f'`{key}` does not match `{first_image_key}`, but we expect all image shapes to match.'
            )


# Relax the single-camera restriction; the model itself supports N cameras.
VQBeTConfig.validate_features = _validate_features_multi_camera


def training_batch(preprocessor, batch):
    # LeRobot 0.4.4's batch converter drops arbitrary action_* labels. Keep
    # action_is_pad explicitly so padded targets stay explicit downstream.
    processed = preprocessor({key: value for key, value in batch.items()
                              if key in ('observation.state', 'action', 'action_is_pad')
                              or key.startswith('observation.images.')})
    return processed


class ExcavatorVQBeTPolicy(VQBeTPolicy):
    """Same architecture/checkpoint keys as VQBeTPolicy; supports N cameras and a
    live-evaluation predict_action_chunk that does not depend on the rollout queue."""

    def _prepare_observations(self, batch):
        batch = dict(batch)
        if self.config.image_features:
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        return batch

    def forward(self, batch):
        batch = self._prepare_observations(batch)
        if not self.vqbet.action_head.vqvae_model.discretized.item():
            loss, n_different_codes, n_different_combinations, recon_l1_error = (
                self.vqbet.action_head.discretize(self.config.n_vqvae_training_steps, batch[ACTION])
            )
            return loss, {
                "n_different_codes": n_different_codes,
                "n_different_combinations": n_different_combinations,
                "recon_l1_error": recon_l1_error,
            }
        _, loss_dict = self.vqbet(batch, rollout=False)
        loss = loss_dict.pop("loss")
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch, noise=None):
        # The stock implementation reads from a rollout queue that is empty during
        # offline evaluation; rebuild the observation layout and run the model
        # directly in rollout mode.
        batch = dict(batch)
        if ACTION in batch:
            batch.pop(ACTION)
        batch = self._prepare_observations(batch)
        return self.vqbet(batch, rollout=True)
