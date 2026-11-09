import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.policies.act.sampling import observation_probabilities
from excavator_vlm.policies.act.continuation import validate_continuation
from excavator_vlm.data.contract import file_hash


class TransitionSamplingTests(unittest.TestCase):
    def test_mixture_respects_frame_identity_and_balances_episodes(self):
        keys = [(2, 1), (1, 0), (2, 0), (1, 1)]
        manifest = {'labels_changed': False, 'training_episodes': [1, 2], 'windows': [
            {'episode_index': ep, 'observation_frame': frame} for ep, frame in [(1, 0), (2, 0), (2, 1)]]}
        p, report = observation_probabilities(keys, [1, 2], manifest, .4)
        np.testing.assert_allclose(p, [.25, .35, .25, .15])
        self.assertAlmostEqual(report['probability_sum'], 1)
        np.testing.assert_allclose(observation_probabilities(keys, [1, 2])[0], [.25] * 4)

    def test_rejects_leakage_duplicates_and_changed_labels(self):
        for windows, unchanged in (([(9, 0)], True), ([(1, 0), (1, 0)], True), ([(1, 0)], False)):
            manifest = {'labels_changed': not unchanged, 'windows': [
                {'episode_index': ep, 'observation_frame': frame} for ep, frame in windows]}
            with self.assertRaises(ValueError):
                observation_probabilities([(1, 0)], [1], manifest, .3)


class ContinuationContractTests(unittest.TestCase):
    def test_real_finetuning_loads_exact_source_weights_and_keeps_processors(self):
        import torch
        from safetensors.torch import load_file
        from excavator_vlm.policies.act import train
        from excavator_vlm.policies.act.adapter import ExcavatorACTPolicy
        from excavator_vlm.policies.act.config import resolve_settings, build_train_config

        if not (SUBMODULE.parent / 'data/export_phw/meta/info.json').exists():
            self.skipTest('PHW dataset unavailable')
        with tempfile.TemporaryDirectory() as directory:
            config_path = SUBMODULE / 'configs/act/full_phw.yaml'
            overrides = {'output_dir': str(Path(directory) / 'source'), 'steps': 2,
                'dataset': {'episodes': [1]}, 'batch_size': 2, 'num_workers': 0,
                'policy': {'device': 'cpu', 'pretrained_backbone_weights': None,
                           'dim_model': 32, 'n_heads': 4, 'dim_feedforward': 64,
                           'n_encoder_layers': 1, 'n_decoder_layers': 1,
                           'n_vae_encoder_layers': 1, 'latent_dim': 4},
                'benchmark': {'validation_freq': 0},
                'launcher': {'gpu_ids': [0], 'mixed_precision': 'no', 'cpu_threads': 2}}
            settings = resolve_settings(config_path, overrides)
            train.run(settings, build_train_config(settings), config_path)
            checkpoint = Path(settings['output_dir']) / 'checkpoints/last/pretrained_model'
            expected = load_file(str(checkpoint / 'model.safetensors'))
            overrides['output_dir'] = str(Path(directory) / 'continued')
            overrides['continuation'] = {'checkpoint': str(checkpoint.resolve())}
            settings = resolve_settings(config_path, overrides)
            original_forward = ExcavatorACTPolicy.forward
            checked = []

            def first_forward(policy, batch):
                if not checked:
                    for key, value in policy.state_dict().items():
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                    checked.append(True)
                return original_forward(policy, batch)

            with patch.object(ExcavatorACTPolicy, 'forward', first_forward):
                train.run(settings, build_train_config(settings), config_path)
            self.assertEqual(checked, [True])
            continued = Path(settings['output_dir']) / 'checkpoints/last/pretrained_model'
            self.assertNotEqual(file_hash(continued / 'model.safetensors'),
                                file_hash(checkpoint / 'model.safetensors'))
            for path in checkpoint.glob('*processor*.safetensors'):
                self.assertEqual(file_hash(path), file_hash(continued / path.name))

    def test_rejects_changed_data_normalization_and_architecture(self):
        import draccus
        from lerobot.policies.act.configuration_act import ACTConfig

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'checkpoints/000002/pretrained_model'
            checkpoint.mkdir(parents=True)
            (checkpoint.parent / 'training_state').mkdir()
            (checkpoint.parent / 'training_state/training_step.json').write_text('{"step": 2}')
            (checkpoint / 'model.safetensors').write_bytes(b'unit-test-placeholder')
            stats = {'action': {'mean': [0.] * 4}}
            (run / 'normalization_stats.json').write_text(json.dumps(stats))
            split = {'train': [1], 'validation': [2], 'test': [3]}
            provenance = {'dataset_fingerprint': 'frozen', 'split_manifest': split,
                          'training_episodes': [1],
                          'normalization_stats_sha256': file_hash(run / 'normalization_stats.json')}
            (checkpoint / 'benchmark_provenance.json').write_text(json.dumps(provenance))
            config = ACTConfig(device='cpu', pretrained_backbone_weights=None)
            (checkpoint / 'config.json').write_text(json.dumps(draccus.encode(config)))
            result = validate_continuation(checkpoint, config, {'fingerprint': 'frozen'}, split, [1], stats)
            self.assertEqual(result['source_step'], 2)
            for report, episodes, normalization in (({'fingerprint': 'changed'}, [1], stats),
                ({'fingerprint': 'frozen'}, [2], stats), ({'fingerprint': 'frozen'}, [1], {})):
                with self.assertRaises(ValueError):
                    validate_continuation(checkpoint, config, report, split, episodes, normalization)
            changed = copy.deepcopy(config)
            changed.chunk_size += 1
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                validate_continuation(checkpoint, changed, {'fingerprint': 'frozen'}, split, [1], stats)


if __name__ == '__main__':
    unittest.main()
