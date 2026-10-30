"""Fail-fast base loading and versioned expert-only checkpoints (no base copies)."""
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from ...common.runtime import dump
from ...data.contract import file_hash
from .precision import PRECISION_CONTRACT, assert_fp32_training, precision_summary
from .lora import assert_finetuning_matches, finetuning_metadata

FORMAT = 'excavator_pi05_trainable_delta_v1'


@lru_cache(maxsize=8)
def _cached_hash(path, size, modified_ns):
    return file_hash(path)


def verified_base(path, expected_hash):
    path = Path(path).resolve() / 'model.safetensors'
    if not path.is_file():
        raise FileNotFoundError(f'PI05 pretrained weights missing: {path}')
    stat = path.stat()
    digest = _cached_hash(str(path), stat.st_size, stat.st_mtime_ns)
    if digest != expected_hash:
        raise ValueError('PI05 base weight fingerprint differs from the pinned checkpoint')
    return path


def strict_state_load(policy, state, expected_keys=None):
    target = policy.state_dict()
    expected = set(target if expected_keys is None else expected_keys)
    if set(state) != expected:
        raise ValueError(f'Weight keys differ: missing={sorted(expected - set(state))[:8]}, '
                         f'unexpected={sorted(set(state) - expected)[:8]}')
    for key, tensor in state.items():
        if key not in target or tensor.shape != target[key].shape:
            raise ValueError(f'Weight shape mismatch: {key}')
    incompatible = policy.load_state_dict(state, strict=expected_keys is None)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(target) - expected:
        raise ValueError('Unexpected incomplete weight loading')


def complete_tied_aliases(policy, state):
    """Recover omitted safetensors aliases only for parameters actually tied in memory."""
    groups = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    result, recovered = dict(state), []
    for names in groups.values():
        present = [name for name in names if name in result]
        if not present:
            continue
        for name in names:
            if name not in result:
                result[name] = result[present[0]]
                recovered.append(name)
    return result, recovered


def load_base(policy, base_path, expected_hash):
    path = verified_base(base_path, expected_hash)
    original = load_file(str(path), device='cpu')
    fixed = policy._fix_pytorch_state_dict_keys(original, policy.config)
    remapped = {key if key.startswith('model.') else f'model.{key}': tensor for key, tensor in fixed.items()}
    remapped, aliases = complete_tied_aliases(policy, remapped)
    strict_state_load(policy, remapped)
    return {'base_path': str(path.parent), 'base_sha256': expected_hash, 'loaded_tensors': len(remapped),
            'verified_tied_aliases': aliases}


def trainable_keys(policy):
    return sorted(name for name, value in policy.named_parameters() if value.requires_grad)


def apply_delta(policy, checkpoint):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported PI05 checkpoint format')
    assert_finetuning_matches(policy, manifest)
    keys = trainable_keys(policy)
    if manifest.get('trainable_keys') != keys:
        raise ValueError('Checkpoint fine-tuning parameter set differs from the current policy')
    path = checkpoint / 'model.safetensors'
    if file_hash(path) != manifest['delta_sha256']:
        raise ValueError('PI05 checkpoint delta checksum mismatch')
    strict_state_load(policy, load_file(str(path), device='cpu'), keys)
    return manifest


def save_checkpoint(path, policy, pre, post, stats, assets, provenance, training_state):
    assert_fp32_training(policy)
    if precision_summary(policy)['contract'] != PRECISION_CONTRACT:
        raise ValueError('New PI05 checkpoints must declare the FP32 trainable precision contract')
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    temporary = path.with_name(f'{path.name}.incomplete')
    temporary.mkdir(parents=True, exist_ok=False)
    pretrained = temporary / 'pretrained_model'
    pretrained.mkdir()
    keys = trainable_keys(policy)
    tensors = policy.state_dict()
    save_file({key: tensors[key].detach().cpu().contiguous() for key in keys},
              str(pretrained / 'model.safetensors'))
    policy.config.save_pretrained(pretrained)
    pre.save_pretrained(pretrained)
    post.save_pretrained(pretrained)
    dump(pretrained / 'normalization_stats.json', stats)
    dump(pretrained / 'benchmark_provenance.json', provenance)
    dump(pretrained / 'checkpoint.json', {'format': FORMAT, **assets, 'trainable_keys': keys,
         'delta_sha256': file_hash(pretrained / 'model.safetensors'), 'step': training_state['step'],
         'precision_contract': PRECISION_CONTRACT, 'parameter_precision': precision_summary(policy),
         'finetuning': finetuning_metadata(policy)})
    # Tensor/container-only state; restore with weights_only=True, never arbitrary pickle objects.
    torch.save(training_state, temporary / 'training_state.pt')
    temporary.rename(path)
    return path / 'pretrained_model'


def resolve_checkpoint(path):
    path = Path(path).resolve()
    if (path / 'last_checkpoint.json').is_file():
        return resolve_checkpoint(json.loads((path / 'last_checkpoint.json').read_text())['checkpoint'])
    if (path / 'pretrained_model/checkpoint.json').is_file():
        path = path / 'pretrained_model'
    if not (path / 'checkpoint.json').is_file():
        raise FileNotFoundError(f'PI05 checkpoint manifest missing: {path}')
    return path


def retain_checkpoints(run, keep):
    """Prune only complete, owned PI05 checkpoints from this experiment."""
    run = Path(run).resolve()
    root = run / 'checkpoints'
    paths = sorted(p for p in root.glob('[0-9]*') if p.is_dir() and not p.is_symlink()
                   and p.name.isdecimal() and (p / 'pretrained_model/checkpoint.json').is_file())
    protected = set(paths[-keep:])
    for name in ('best_checkpoint.json', 'last_checkpoint.json'):
        if (run / name).exists():
            protected.add(Path(json.loads((run / name).read_text())['checkpoint']).parent.resolve())
    for path in paths:
        manifest = json.loads((path / 'pretrained_model/checkpoint.json').read_text())
        if path not in protected and path.resolve().parent == root and manifest.get('format') == FORMAT:
            shutil.rmtree(path)


def asset_overrides(manifest):
    return {'base_path': os.environ.get('EXCAVATOR_PI05_BASE', manifest['base_path']),
            'base_sha256': manifest['base_sha256'],
            'tokenizer_path': os.environ.get('EXCAVATOR_PI05_TOKENIZER', manifest['tokenizer_path'])}
