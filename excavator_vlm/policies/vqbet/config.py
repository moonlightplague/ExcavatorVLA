"""VQ-BeT-only configuration validation and LeRobot configuration construction."""
import copy
from importlib.metadata import version

from ...common.config import resolve_settings as resolve_common_settings
from ...data.contract import load_info, policy_features


def _patch_validate_features():
    """Relax VQ-BeT's single-camera restriction.

    The model itself supports N cameras via ``num_images = len(image_features)``;
    only the stock validator is conservative and requires exactly one image.
    """
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

    def validate_features(self):
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

    VQBeTConfig.validate_features = validate_features


_patch_validate_features()


def resolve_settings(path, overrides=None):
    settings = resolve_common_settings(path, overrides)
    if settings['policy'].get('type') != 'vqbet':
        raise ValueError('VQ-BeT entrypoint requires policy.type: vqbet')
    return settings


def build_train_config(settings):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig  # registers config

    if version('lerobot') != '0.4.4':
        raise ValueError('This adapter requires lerobot==0.4.4')
    policy = copy.deepcopy(settings['policy'])
    if policy.get('type') != 'vqbet' or policy.get('pretrained_path') is not None:
        raise ValueError('VQ-BeT starts from a fresh policy, without policy.path/pretrained_path')
    inputs, outputs = policy_features(load_info(settings['dataset']['root']))
    policy.update(input_features=inputs, output_features=outputs, push_to_hub=False)
    data = {key: settings[key] for key in ('dataset', 'policy', 'steps', 'batch_size', 'num_workers',
                                          'seed', 'save_freq', 'log_freq', 'output_dir', 'job_name')}
    data['policy'] = policy
    data.update(eval_freq=0, save_checkpoint=True, wandb={'enable': False},
                dataset={**data['dataset'], 'use_imagenet_stats': True})
    cfg = draccus.decode(TrainPipelineConfig, data)
    # Preserve the requested device instead of silently falling back to CPU.
    cfg.policy.device = policy['device']
    cfg.validate()
    cfg.optimizer.grad_clip_norm = settings['grad_clip_norm']
    if cfg.policy.n_obs_steps != 1 or cfg.policy.action_chunk_size < 1:
        raise ValueError('Invalid VQ-BeT observation/action horizon')
    cfg.policy.validate_features()
    return cfg
