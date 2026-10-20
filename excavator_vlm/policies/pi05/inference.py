"""Load a checked immutable base plus expert delta, without upstream silent fallbacks."""
import json
from pathlib import Path

from ...data.contract import CAMERAS
from .checkpoints import apply_delta, asset_overrides, load_base, resolve_checkpoint


def check_environment():
    try:
        from transformers.models.siglip import check
        compatible = check.check_whether_transformers_replace_is_installed_correctly()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError('Use the isolated PI05 environment and requirements-pi05.txt; '
                           'the installed Transformers lacks the OpenPI compatibility implementation') from exc
    if not compatible:
        raise RuntimeError('PI05 Transformers compatibility check failed')


def load_pi05(checkpoint, device='cpu'):
    check_environment()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # register pi05
    from .adapter import policy_class
    from .processors import make_processors

    checkpoint = resolve_checkpoint(checkpoint)
    manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
    assets = asset_overrides(manifest)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if config.type != 'pi05':
        raise ValueError('Expected a PI05 checkpoint')
    config.device = device
    if str(device) == 'cpu':
        config.dtype = 'float32'
    config.compile_model = False
    config.gradient_checkpointing = False
    if (list(config.input_features['observation.state'].shape) != [28]
            or list(config.output_features['action'].shape) != [4]
            or tuple(config.image_features) != CAMERAS):
        raise ValueError('PI05 checkpoint differs from the excavator 28D/4D/three-camera contract')
    stats = json.loads((checkpoint / 'normalization_stats.json').read_text())
    pre, post = make_processors(config, stats, assets['tokenizer_path'])
    policy = policy_class()(config)
    load_base(policy, assets['base_path'], assets['base_sha256'])
    apply_delta(policy, checkpoint)
    return policy.eval(), pre, post
