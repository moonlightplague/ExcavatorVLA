"""Strict YAML configuration and paths relative to the configuration file."""
import copy
from datetime import datetime
import os
from pathlib import Path

import yaml

def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(
            result.get(key), dict) else copy.deepcopy(value)
    return result


def read_yaml(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Circular configuration inheritance')
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Training configuration must be a mapping')
    parent = data.pop('extends', None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        if parent_path.parent != path.parent:
            raise ValueError('Inherited configs must share the same directory for relative paths')
        data = merge(read_yaml(parent_path, (*seen, path)), data)
    return data


def resolve_settings(path, overrides=None, *, extra_sections=()):
    path = Path(path).resolve()
    data = merge(read_yaml(path), overrides or {})
    allowed = {'dataset', 'policy', 'steps', 'batch_size', 'num_workers', 'seed', 'save_freq',
               'log_freq', 'output_dir', 'job_name', 'benchmark', 'launcher', 'grad_clip_norm',
               *extra_sections}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f'Unknown training settings: {sorted(unknown)}')
    for section in ('dataset', 'policy', 'benchmark', 'launcher'):
        if not isinstance(data.get(section), dict):
            raise ValueError(f'{section} must be a mapping')
    permitted = {
        'dataset': {'repo_id', 'root', 'episodes', 'video_backend'},
        'benchmark': {'split_file', 'split_seed', 'validation_freq', 'validation_max_batches'},
        'launcher': {'output_root', 'hf_home', 'torch_home', 'gpu_ids', 'mixed_precision', 'cpu_threads'},
    }
    for section, keys in permitted.items():
        if set(data[section]) - keys:
            raise ValueError(f'Unknown {section} settings: {sorted(set(data[section]) - keys)}')
    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve())
    for section, key in [('dataset', 'root'), ('benchmark', 'split_file'),
                         ('launcher', 'output_root'), ('launcher', 'hf_home'), ('launcher', 'torch_home')]:
        data[section][key] = resolve(data[section][key])
    for key in ('steps', 'batch_size', 'save_freq', 'log_freq'):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('validation_freq', 'validation_max_batches'):
        if type(data['benchmark'][key]) is not int or data['benchmark'][key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if type(data['seed']) is not int or type(data['num_workers']) is not int or data['num_workers'] < 0:
        raise ValueError('seed must be integer and num_workers must be nonnegative integer')
    if data['grad_clip_norm'] <= 0:
        raise ValueError('grad_clip_norm must be positive')
    device = data['policy'].get('device')
    if device not in ('cpu', 'cuda'):
        raise ValueError('policy.device must be cpu or cuda')
    gpus = data['launcher']['gpu_ids']
    if not isinstance(gpus, list) or not gpus or any(type(g) is not int or g < 0 for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('gpu_ids must be unique nonnegative integers')
    if data['launcher']['mixed_precision'] not in ('no', 'bf16', 'fp16'):
        raise ValueError('mixed_precision must be no, bf16, or fp16')
    if device == 'cpu' and (len(gpus) != 1 or data['launcher']['mixed_precision'] != 'no'):
        raise ValueError('CPU runs need one process and mixed_precision: "no"')
    if data['launcher']['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be positive')
    episodes = data['dataset'].get('episodes')
    if episodes is not None and (not isinstance(episodes, list) or not episodes or
                                any(type(e) is not int for e in episodes) or len(set(episodes)) != len(episodes)):
        raise ValueError('dataset.episodes must be a nonempty unique integer list or null')
    name = data['job_name']
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError('job_name must be a directory name')
    data['output_dir'] = resolve(data['output_dir']) if data.get('output_dir') else str(
        Path(data['launcher']['output_root']) / f'{name}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}')
    return data


def setup_cache(settings):
    os.environ['HF_HOME'] = settings['launcher']['hf_home']
    os.environ['TORCH_HOME'] = settings['launcher']['torch_home']
    os.environ['HF_DATASETS_CACHE'] = str(Path(settings['launcher']['hf_home']) / 'datasets')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
