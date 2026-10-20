"""Load ACT weights and its saved preprocessing/postprocessing pipelines."""


def load_act(checkpoint, device='cpu'):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from .adapter import ExcavatorACTPolicy

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if config.type != 'act':
        raise ValueError('Expected an ACT checkpoint')
    config.device = device
    # A full checkpoint restores the backbone; avoid a redundant network fetch.
    config.pretrained_backbone_weights = None
    policy = ExcavatorACTPolicy.from_pretrained(str(checkpoint), config=config, strict=True)
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': device}})
    return policy.eval(), pre, post
