"""Local-tokenizer PI05 processors; no ImageNet scaling or relative actions."""
from pathlib import Path

import torch

from ...data.contract import CAMERAS


class CheckedPreprocessor:
    def __init__(self, pipeline, max_length):
        self.pipeline, self.max_length = pipeline, max_length

    def __call__(self, batch):
        if not batch.get('task'):
            raise ValueError('PI05 requires a nonempty language task')
        state = batch['observation.state']
        if state.shape[-1] != 28 or not torch.isfinite(state).all():
            raise ValueError('Expected finite 28D excavator state')
        for key in CAMERAS:
            image = batch[key]
            if image.shape[-3:] != (3, 256, 256) or not torch.isfinite(image).all():
                raise ValueError(f'Expected finite 256x256 RGB image: {key}')
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f'PI05 expects unnormalized RGB in [0,1]: {key}')
        result = self.pipeline(batch)
        validate_tokens(result, self.max_length)
        return result

    def save_pretrained(self, *args, **kwargs):
        return self.pipeline.save_pretrained(*args, **kwargs)


def make_processors(config, stats, tokenizer_path):
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (
        AddBatchDimensionProcessorStep, DeviceProcessorStep, NormalizerProcessorStep,
        PolicyProcessorPipeline, RenameObservationsProcessorStep, TokenizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    tokenizer_path = Path(tokenizer_path).resolve()
    if not (tokenizer_path / 'tokenizer_config.json').is_file():
        raise FileNotFoundError(f'Local PaliGemma tokenizer missing: {tokenizer_path}; run prepare_pi05.py --fetch-tokenizer')
    tensors = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in values.items()}
               for key, values in stats.items()}
    pre = PolicyProcessorPipeline(steps=[
        RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(features={**config.input_features, **config.output_features},
                                norm_map=config.normalization_mapping, stats=tensors),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(tokenizer_name=str(tokenizer_path), max_length=config.tokenizer_max_length,
                               padding='max_length', padding_side='right', truncation=False),
        DeviceProcessorStep(device=config.device),
    ], name='policy_preprocessor')
    post = PolicyProcessorPipeline(steps=[
        UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=tensors),
        DeviceProcessorStep(device='cpu'),
    ], name='policy_postprocessor', to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return CheckedPreprocessor(pre, config.tokenizer_max_length), post


def validate_tokens(batch, max_length):
    tokens = batch['observation.language.tokens']
    if tokens.ndim != 2 or tokens.shape[-1] != max_length:
        raise ValueError('PI05 prompt exceeds tokenizer_max_length; refusing silent truncation')
