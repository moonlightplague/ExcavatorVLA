"""Shared offline evaluation CLI; policy loading and dataset adapters are injected."""
import argparse
import json
import os
from pathlib import Path
from ..common.paths import PROJECT_ROOT


def main(load_policy, make_dataset, policy_type, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=PROJECT_ROOT / 'data/export')
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--episodes', help='Explicit episode subset within --split')
    parser.add_argument('--batch-size', type=int, default=1 if policy_type == 'pi05' else 8)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_batches < 0:
        parser.error('batch-size must be positive; max-batches nonnegative')
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / 'models/hf-cache'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ..data.contract import audit_dataset, file_hash, load_info, validate_split
    from .offline_metrics import evaluate_loader

    torch.set_num_threads(4)
    torch.manual_seed(0)
    checkpoint = args.checkpoint.resolve()
    if policy_type == 'pi05':
        from ..policies.pi05.checkpoints import resolve_checkpoint
        checkpoint = resolve_checkpoint(checkpoint)
    provenance = json.loads((checkpoint / 'benchmark_provenance.json').read_text())
    split = (json.loads(args.split_file.read_text()) if args.split_file else provenance['split_manifest'])
    report, frames = audit_dataset(args.dataset)
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    if report['fingerprint'] != provenance['dataset_fingerprint']:
        raise ValueError('Evaluation dataset differs from checkpoint dataset fingerprint')
    episodes = [int(e) for e in args.episodes.split(',')] if args.episodes else split[args.split]
    if not episodes or not set(episodes).issubset(split[args.split]) or len(set(episodes)) != len(episodes):
        raise ValueError('Episode subset must be unique and within the selected split')
    if args.split != 'train' and set(episodes) & set(provenance['training_episodes']):
        raise ValueError('Selected held-out evaluation overlaps actual training episodes')
    policy, pre, post = load_policy(checkpoint, args.device)
    if policy_type == 'vqbet':
        dataset = make_dataset('local/excavator', args.dataset.resolve(), episodes,
                               policy.config.n_action_pred_token, policy.config.action_chunk_size)
    else:
        chunk_len = policy.config.chunk_size if hasattr(policy.config, 'chunk_size') else policy.config.horizon
        dataset = make_dataset('local/excavator', args.dataset.resolve(), episodes, chunk_len)
    if policy_type == 'pi05':
        # Base construction initializes tensors before strict loading; reset after
        # that work so independent CLI sampling matches fixed-seed validation.
        torch.manual_seed(0)
    metrics, arrays = evaluate_loader(policy, pre, post, DataLoader(dataset, args.batch_size),
                                     load_info(args.dataset)['action_limits_rad_s'], args.max_batches)
    metrics.update(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint / 'model.safetensors'),
                   dataset_fingerprint=report['fingerprint'], split=args.split, selected_episodes=episodes,
                   overlaps_training=bool(set(episodes) & set(provenance['training_episodes'])),
                   partial_evaluation=args.max_batches > 0, policy_type=policy_type, seed=0)
    if policy_type == 'pi05':
        from ..policies.pi05.precision import precision_summary
        from ..policies.pi05.lora import finetuning_metadata
        manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
        metrics.update(checkpoint_format=manifest['format'], base_sha256=manifest['base_sha256'],
                       policy_dtype=policy.config.dtype, num_inference_steps=policy.config.num_inference_steps,
                       chunk_size=policy.config.chunk_size,
                       policy_dtype_scope='backbone initialization; see parameter_precision for actual storage',
                       precision_contract=manifest.get('precision_contract'),
                       parameter_precision=precision_summary(policy),
                       finetuning=finetuning_metadata(policy))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'actions.npz', **arrays)
    print(json.dumps(metrics, indent=2, allow_nan=False))
