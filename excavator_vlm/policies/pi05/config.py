"""Explicit excavator PI05 feature binding and resource-conscious settings."""
import copy
from importlib.metadata import version
from pathlib import Path

from ...common.config import resolve_settings as resolve_common_settings
from ...data.contract import CAMERAS, load_info
from .lora import finetuning_spec

BASE_SHA256 = '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'


def config_dict(config):
    import draccus
    return draccus.encode(config)


def resolve_settings(path, overrides=None):
    path = Path(path).resolve()
    settings = resolve_common_settings(path, overrides, extra_sections=('assets', 'training'))
    assets, training = settings.get('assets'), settings.get('training')
    if not isinstance(assets, dict) or set(assets) != {'base_path', 'base_sha256', 'tokenizer_path'}:
        raise ValueError('assets must specify base_path, base_sha256 and tokenizer_path')
    for key in ('base_path', 'tokenizer_path'):
        assets[key] = str((path.parent / Path(assets[key]).expanduser()).resolve())
    if not isinstance(assets['base_sha256'], str) or len(assets['base_sha256']) != 64:
        raise ValueError('A pinned base weight SHA-256 is required')
    required = {'gradient_accumulation_steps', 'validation_batch_size', 'resume', 'keep_checkpoints',
                'trainable_dtype'}
    if (not isinstance(training, dict) or not required.issubset(training)
            or set(training) - required - {'mode', 'lora', 'resume_lr'}):
        raise ValueError(f'training must specify {sorted(required)}; optional mode/lora/resume_lr')
    training.update(finetuning_spec(training.get('mode', 'expert_only'), training.get('lora')))
    for key in ('gradient_accumulation_steps', 'validation_batch_size', 'keep_checkpoints'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if training['resume']:
        training['resume'] = str((path.parent / Path(training['resume']).expanduser()).resolve())
    resume_lr = training.get('resume_lr')
    if resume_lr is not None:
        import math
        if (not isinstance(resume_lr, (int, float)) or isinstance(resume_lr, bool)
                or not math.isfinite(resume_lr) or resume_lr <= 0 or not training['resume']):
            raise ValueError('resume_lr requires a resume checkpoint and a finite positive learning rate')
    if training['trainable_dtype'] != 'float32':
        raise ValueError('PI05 expert parameters and optimizer moments must use float32 storage')
    policy = settings['policy']
    if policy.get('type') != 'pi05':
        raise ValueError('PI05 entrypoint requires policy.type: pi05')
    if not policy.get('train_expert_only') or not policy.get('freeze_vision_encoder'):
        raise ValueError('Freeze base VLM/vision weights; enable adapters with training.mode instead')
    if policy.get('rtc_config') or policy.get('compile_model'):
        raise ValueError('Initial benchmark disables RTC and model compilation')
    if policy.get('max_state_dim') != 32 or policy.get('max_action_dim') != 32:
        raise ValueError('Keep the pretrained internal 32D state/action interface')
    if policy.get('n_obs_steps') != 1 or not 1 <= policy['n_action_steps'] <= policy['chunk_size']:
        raise ValueError('Invalid observation/action horizon')
    if policy.get('empty_cameras', 0) != 0:
        raise ValueError('All three real excavator cameras are required')
    if policy.get('normalization_mapping') != {'VISUAL': 'IDENTITY', 'STATE': 'QUANTILES', 'ACTION': 'QUANTILES'}:
        raise ValueError('PI05 uses identity visual and train-only quantile state/action normalization')
    if policy['device'] == 'cuda' and (policy.get('dtype') != 'bfloat16'
                                     or settings['launcher']['mixed_precision'] != 'bf16'):
        raise ValueError('CUDA PI05 uses a BF16 frozen backbone and matching BF16 autocast; '
                         'trainable storage remains FP32')
    return settings


def build_policy_config(settings):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if version('lerobot') != '0.4.4':
        raise ValueError('This adapter requires lerobot==0.4.4')
    load_info(settings['dataset']['root'])
    policy = copy.deepcopy(settings['policy'])
    policy.pop('type')
    policy['normalization_mapping'] = {k: NormalizationMode(v) for k, v in policy['normalization_mapping'].items()}
    policy.update(input_features={
        'observation.state': PolicyFeature(FeatureType.STATE, (28,)),
        **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in CAMERAS},
    }, output_features={'action': PolicyFeature(FeatureType.ACTION, (4,))}, push_to_hub=False)
    config = PI05Config(**policy)
    # Do not let LeRobot silently turn a requested GPU run into CPU execution.
    config.device = settings['policy']['device']
    config.validate_features()
    return config
