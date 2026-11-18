"""Verify stock ACT is actually trained and reloaded by the original-objective mode."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from excavator_vlm.policies.act.config import resolve_settings, build_train_config
from excavator_vlm.policies.act.inference import load_act
from excavator_vlm.policies.act import train
from lerobot.policies.act.modeling_act import ACTPolicy


class OriginalACTTests(unittest.TestCase):
    def settings(self, output):
        return resolve_settings(SUBMODULE.parent / 'training_config_act_original.yaml', {
            'output_dir': str(output), 'steps': 1, 'batch_size': 2, 'num_workers': 0,
            'policy': {'device': 'cpu', 'pretrained_backbone_weights': None,
                       'chunk_size': 4, 'n_action_steps': 4, 'dim_model': 32, 'n_heads': 4,
                       'dim_feedforward': 64, 'n_encoder_layers': 1, 'n_decoder_layers': 1,
                       'n_vae_encoder_layers': 1, 'latent_dim': 4},
            'launcher': {'gpu_ids': [0], 'mixed_precision': 'no', 'cpu_threads': 2},
            'benchmark': {'validation_freq': 0}})

    def test_stock_training_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(1000)
        frames = pd.DataFrame({'episode_index': [0] * 4,
            'observation.state': list(rng.normal(size=(4, 28))),
            'action': list(rng.normal(size=(4, 4))), 'action_loss_weight': [1.] * 4})
        samples = [{'observation.state': torch.tensor(row, dtype=torch.float32),
            **{f'observation.images.{i}': torch.rand(3, 32, 32) for i in range(3)},
            'action': torch.randn(4, 4), 'action_loss_weight': torch.ones(4),
            'action_is_pad': torch.tensor([False, False, True, True])}
            for row in frames['observation.state']]
        split = {'train': [0], 'validation': [1], 'test': [2]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'train'
            settings = self.settings(output)
            cfg = build_train_config(settings)
            with patch.object(train, 'prepare', return_value=({'fingerprint': 'test'}, split, frames, [0])), \
                 patch.object(train, 'make_act_dataset', return_value=samples), \
                 patch('excavator_vlm.policies.act.adapter.ExcavatorACTPolicy',
                       side_effect=AssertionError('The weighted policy must not be constructed')):
                train.run(settings, cfg, SUBMODULE.parent / 'training_config_act_original.yaml')
            provenance = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(provenance['policy_class'], 'lerobot.policies.act.modeling_act.ACTPolicy')
            self.assertEqual(provenance['objective'], 'original')
            self.assertEqual(json.loads((output / 'reload_check.json').read_text()),
                             {'status': 'passed', 'shape': [1, 4, 4]})
            restored, _, _ = load_act(output / 'checkpoints/last/pretrained_model')
            self.assertIs(type(restored), ACTPolicy)

    def test_original_baseline_rejects_sampling_and_finetuning(self):
        path = SUBMODULE.parent / 'training_config_act_original.yaml'
        for override in ({'continuation': {'checkpoint': '/tmp/weights'}},
                         {'sampling': {'manifest': '/tmp/windows', 'transition_probability': .3}}):
            with self.assertRaisesRegex(ValueError, 'starts fresh'):
                resolve_settings(path, override)


if __name__ == '__main__':
    unittest.main()
