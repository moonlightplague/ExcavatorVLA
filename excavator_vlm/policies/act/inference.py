"""Load ACT weights and its saved preprocessing/postprocessing pipelines."""


def load_act(checkpoint, device='cpu', *, original_act=None):
    import json
    from pathlib import Path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from .adapter import ExcavatorACTPolicy
    from lerobot.policies.act.modeling_act import ACTPolicy

    if original_act is None:
        provenance = Path(checkpoint) / 'benchmark_provenance.json'
        original_act = (provenance.is_file() and
                        json.loads(provenance.read_text()).get('adapter') == 'lerobot_act_original_v1')
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if config.type != 'act':
        raise ValueError('Expected an ACT checkpoint')
    config.device = device
    # A full checkpoint restores the backbone; avoid a redundant network fetch.
    config.pretrained_backbone_weights = None
    policy_type = ACTPolicy if original_act else ExcavatorACTPolicy
    policy = policy_type.from_pretrained(str(checkpoint), config=config, strict=True)
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': device}})
    return policy.eval(), pre, post
