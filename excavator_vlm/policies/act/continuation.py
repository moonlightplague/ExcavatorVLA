"""Validate weights-only fine-tuning against frozen data and preprocessing."""
import json
from pathlib import Path

from ...data.contract import file_hash


def validate_continuation(checkpoint, config, report, split, episodes, stats):
    import draccus

    checkpoint = Path(checkpoint)
    provenance = json.loads((checkpoint / 'benchmark_provenance.json').read_text())
    if (provenance['dataset_fingerprint'] != report['fingerprint']
            or provenance['split_manifest'] != split
            or provenance['training_episodes'] != episodes):
        raise ValueError('Continuation data, episode selection, or split differs from checkpoint')
    source_stats = checkpoint.parents[2] / 'normalization_stats.json'
    if (file_hash(source_stats) != provenance['normalization_stats_sha256']
            or json.loads(source_stats.read_text()) != stats):
        raise ValueError('Continuation must preserve the original training normalization')
    previous = json.loads((checkpoint / 'config.json').read_text())
    current = draccus.encode(config)
    immutable = ('type', 'input_features', 'output_features', 'normalization_mapping',
                 'n_obs_steps', 'chunk_size', 'n_action_steps', 'vision_backbone',
                 'replace_final_stride_with_dilation', 'pre_norm', 'dim_model', 'n_heads',
                 'dim_feedforward', 'feedforward_activation', 'n_encoder_layers',
                 'n_decoder_layers', 'use_vae', 'latent_dim', 'n_vae_encoder_layers',
                 'temporal_ensemble_coeff', 'dropout', 'kl_weight')
    changed = [key for key in immutable if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError(f'Continuation policy configuration differs: {changed}')
    source_step = json.loads((checkpoint.parent / 'training_state/training_step.json').read_text())['step']
    return {'mode': 'weights_only_finetuning_with_fresh_optimizer',
            'source_checkpoint': str(checkpoint.resolve()), 'source_step': source_step,
            'source_model_sha256': file_hash(checkpoint / 'model.safetensors'),
            'source_normalization_sha256': file_hash(source_stats),
            'source_processor_files': {p.name: file_hash(p) for p in checkpoint.glob('*processor*')
                                       if p.is_file()}}
