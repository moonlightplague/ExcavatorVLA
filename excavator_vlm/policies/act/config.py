"""ACT-only configuration validation and LeRobot configuration construction."""
import copy
from importlib.metadata import version

from ...common.config import resolve_settings as resolve_common_settings
from ...data.contract import load_info, policy_features


def resolve_settings(path, overrides=None):
    settings = resolve_common_settings(path, overrides)
    if settings['policy'].get('type') != 'act':
        raise ValueError('ACT entrypoint requires policy.type: act')
    if settings['policy'].get('temporal_ensemble_coeff') is not None:
        raise ValueError('First benchmark uses chunk execution; temporal ensembling must be null')
    return settings


def build_train_config(settings):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig  # registers config

    if version('lerobot') != '0.4.4':
        raise ValueError('This adapter requires lerobot==0.4.4')
    policy = copy.deepcopy(settings['policy'])
    if policy.get('type') != 'act' or policy.get('pretrained_path') is not None:
        raise ValueError('ACT starts from a fresh policy, without policy.path/pretrained_path')
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
    if cfg.policy.n_obs_steps != 1 or not 1 <= cfg.policy.n_action_steps <= cfg.policy.chunk_size:
        raise ValueError('Invalid ACT observation/action horizon')
    cfg.policy.validate_features()
    return cfg
