"""Load VQ-BeT weights and its saved preprocessing/postprocessing pipelines."""


def load_vqbet(checkpoint, device='cpu'):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from .adapter import ExcavatorVQBeTPolicy

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if config.type != 'vqbet':
        raise ValueError('Expected a VQ-BeT checkpoint')
    config.device = device
    # A full checkpoint restores the backbone; avoid a redundant network fetch.
    config.pretrained_backbone_weights = None
    policy = ExcavatorVQBeTPolicy.from_pretrained(str(checkpoint), config=config, strict=True).to(device)
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': device}})
    return policy.eval(), pre, post
