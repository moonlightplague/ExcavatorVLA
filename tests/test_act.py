"""Exercise masking gradients, real export chunk boundaries and checkpoint portability."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
os.environ.setdefault('HF_HOME', '/tmp/excavator-act-test-hf')
os.environ.setdefault('HF_DATASETS_CACHE', '/tmp/excavator-act-test-hf/datasets')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from excavator_vlm.policies.act.adapter import ExcavatorACTPolicy, training_batch, weighted_l1
from excavator_vlm.policies.act.inference import load_act
from excavator_vlm.policies.act.config import build_train_config, resolve_settings
from excavator_vlm.policies.act.dataset import make_act_dataset, training_stats
from excavator_vlm.data.contract import CAMERAS, STATE_NAMES, load_info, make_split, validate_split
from excavator_vlm.evaluation.offline_metrics import summarize_actions

torch.set_num_threads(2)


class LossTests(unittest.TestCase):
    def test_padding_and_unsupervised_timesteps_have_no_gradient(self):
        prediction = torch.ones(2, 3, 4, requires_grad=True)
        padding = torch.tensor([[False, False, True], [False, True, True]])
        supervision = torch.tensor([[1., 0., 1.], [2., 1., 1.]])
        loss, _ = weighted_l1(prediction, torch.zeros_like(prediction), padding, supervision)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(prediction.grad[padding].abs().sum().item(), 0.)
        self.assertEqual(prediction.grad[0, 1].abs().sum().item(), 0.)
        self.assertAlmostEqual(prediction.grad[1, 0].sum().item(), 2 / 3, places=6)

    def test_all_padding_remains_differentiable(self):
        prediction = torch.ones(1, 3, 4, requires_grad=True)
        loss, _ = weighted_l1(prediction, torch.zeros_like(prediction), torch.ones(1, 3, dtype=torch.bool),
                              torch.ones(1, 3))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0)

    def test_fractional_weights_are_normalized_by_the_actual_weight_sum(self):
        loss, _ = weighted_l1(torch.ones(1, 1, 4), torch.zeros(1, 1, 4),
                              torch.zeros(1, 1, dtype=torch.bool), torch.full((1, 1), .01))
        self.assertEqual(loss.item(), 1.)

    def test_non_chunked_weights_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same timestamps'):
            weighted_l1(torch.zeros(1, 3, 4), torch.zeros(1, 3, 4),
                        torch.zeros(1, 3, dtype=torch.bool), torch.ones(1))


class SplitTests(unittest.TestCase):
    def test_deterministic_stratified_split_has_exact_counts_and_no_overlap(self):
        frames = pd.DataFrame({'episode_index': np.arange(200), 'task_index': np.arange(200) % 22})
        split = make_split(frames, 'fingerprint')
        self.assertEqual(split, make_split(frames, 'fingerprint'))
        self.assertEqual([len(split[k]) for k in ('train', 'validation', 'test')], [160, 20, 20])
        validate_split(split, range(200), 'fingerprint')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_split(split, range(200), 'another')
        broken = copy.deepcopy(split)
        broken['test'][0] = broken['train'][0]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_split(broken, range(200))

    def test_stats_exclude_held_out_extreme_values(self):
        frames = pd.DataFrame({'episode_index': [0, 1],
                               'observation.state': [np.zeros(28), np.ones(28) * 1e9],
                               'action': [np.zeros(4), np.ones(4) * 1e9]})
        stats = training_stats(frames, [0])
        self.assertEqual(stats['action']['mean'], [0.] * 4)
        self.assertEqual(stats['observation.state']['max'], [0.] * 28)


class MetricsTests(unittest.TestCase):
    def test_future_masks_and_recovery_do_not_include_padding(self):
        target = np.zeros((1, 3, 4))
        predicted = np.ones_like(target)
        predicted[:, 2] = 1000
        metrics = summarize_actions(predicted, target, np.array([[False, False, True]]),
                                    np.array([[1, 0, 1]]), np.array([[0, 1, 1]]), [6, 2.5, 2.5, 6])
        self.assertEqual(metrics['all_valid']['mae'], 1)
        self.assertEqual(metrics['supervised_target_timesteps'], 1)
        self.assertEqual(metrics['recovery']['timesteps'], 1)
        self.assertIsNone(metrics['per_horizon'][2]['mae'])


class ConfigTests(unittest.TestCase):
    def test_distributed_launch_uses_the_public_wrapper_after_module_move(self):
        from excavator_vlm.policies.act import train
        with tempfile.TemporaryDirectory() as directory:
            arguments = ['train_act.py', '--config', str(SUBMODULE / 'configs/act/full.yaml'),
                         '--output', str(Path(directory) / 'run')]
            fake_cfg = SimpleNamespace(dataset=SimpleNamespace(episodes=None))
            with patch.object(sys, 'argv', arguments), patch.dict(os.environ), \
                    patch.object(train, 'setup_cache'), \
                    patch.object(train, 'build_train_config', return_value=fake_cfg), \
                    patch.object(train, 'prepare', return_value=({}, {}, None, [1])), \
                    patch.object(train.subprocess, 'run') as launch:
                train.main()
            command = launch.call_args.args[0]
            worker_entrypoint = command[command.index('--dynamo_backend=no') + 1]
            self.assertEqual(worker_entrypoint, str(SUBMODULE / 'scripts/train_act.py'))
            self.assertEqual(command[-1], '--worker')

    def test_policy_config_paths_and_inherited_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            overrides = {'output_dir': str(Path(directory) / 'run')}
            for name, steps in {'full': 40000, 'smoke': 20, 'overfit': 2000, 'pilot': 5000}.items():
                with self.subTest(config=name):
                    path = SUBMODULE / 'configs/act' / f'{name}.yaml'
                    self.assertTrue(path.is_file())
                    self.assertFalse(path.is_symlink())
                    settings = resolve_settings(path, overrides)
                    self.assertEqual(settings['steps'], steps)
                    self.assertEqual(settings['policy']['chunk_size'], 10)
                    self.assertEqual(settings['policy']['n_action_steps'], 3)
                    self.assertEqual(Path(settings['dataset']['root']), (SUBMODULE.parent / 'data/export').resolve())
                    self.assertEqual(Path(settings['benchmark']['split_file']),
                                     SUBMODULE / 'configs/splits/excavator_v4_seed1000.json')

    def test_full_and_smoke_resolve_schema_without_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, steps in [('full.yaml', 40000), ('smoke.yaml', 20)]:
                settings = resolve_settings(SUBMODULE / 'configs/act' / name,
                    {'policy': {'device': 'cpu'}, 'launcher': {'gpu_ids': [0], 'mixed_precision': 'no'},
                     'output_dir': str(Path(directory) / name)})
                cfg = build_train_config(settings)
                self.assertEqual(cfg.steps, steps)
                self.assertIsNone(cfg.policy.pretrained_path)
                self.assertEqual(tuple(cfg.policy.image_features), CAMERAS)
                self.assertEqual(set(cfg.policy.output_features), {'action'})
                self.assertEqual(list(cfg.policy.robot_state_feature.shape), [28])


class DatasetTests(unittest.TestCase):
    def test_real_episode_tail_and_weights_share_future_indices(self):
        root = SUBMODULE.parent / 'data/export'
        if not (root / 'meta/info.json').exists():
            self.skipTest('Real export unavailable')
        dataset = make_act_dataset('local/excavator', root.resolve(), [0], 10)
        tail = dataset[len(dataset) - 1]
        self.assertEqual(tuple(tail['action'].shape), (10, 4))
        self.assertEqual(tuple(tail['action_loss_weight'].shape), (10,))
        self.assertEqual(tail['action_is_pad'].tolist(), [False] + [True] * 9)
        self.assertEqual(dataset.delta_indices['action'], dataset.delta_indices['action_loss_weight'])
        self.assertEqual(tuple(tail[CAMERAS[0]].shape), (3, 256, 256))
        self.assertEqual(load_info(root)['features']['observation.state']['names'], list(STATE_NAMES))


class CheckpointTests(unittest.TestCase):
    def test_forward_update_save_and_stock_act_reload(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        config = ACTConfig(device='cpu', pretrained_backbone_weights=None,
                           chunk_size=4, n_action_steps=3, dim_model=32, n_heads=4,
                           dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                           n_vae_encoder_layers=1, latent_dim=4,
                           input_features={'observation.state': PolicyFeature(FeatureType.STATE, (28,)),
                                           **{key: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for key in CAMERAS}},
                           output_features={'action': PolicyFeature(FeatureType.ACTION, (4,))}, push_to_hub=False)
        stats = {'observation.state': {'mean': [0.] * 28, 'std': [1.] * 28},
                 'action': {'mean': [2.] * 4, 'std': [3.] * 4},
                 **{key: {'mean': [[[0.]]] * 3, 'std': [[[1.]]] * 3} for key in CAMERAS}}
        pre, post = make_act_pre_post_processors(config, stats)
        policy = ExcavatorACTPolicy(config)
        batch = {'observation.state': torch.randn(2, 28),
                 **{key: torch.rand(2, 3, 32, 32) for key in CAMERAS},
                 'action': torch.randn(2, 4, 4), 'action_is_pad': torch.zeros(2, 4, dtype=torch.bool),
                 'action_loss_weight': torch.ones(2, 4)}
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        processed = training_batch(pre, copy.deepcopy(batch))
        torch.testing.assert_close(processed['action_loss_weight'], batch['action_loss_weight'])
        loss, _ = policy(processed)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        policy.eval()
        observation = {key: value for key, value in batch.items() if key in config.input_features}
        expected = policy.predict_action_chunk(pre(copy.deepcopy(observation)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            policy.save_pretrained(path)
            pre.save_pretrained(path)
            post.save_pretrained(path)
            restored, rpre, rpost = load_act(path)
            actual = restored.predict_action_chunk(rpre(copy.deepcopy(observation)))
            torch.testing.assert_close(actual, expected)
            stock = ACTPolicy.from_pretrained(path, config=config, strict=True)
            torch.testing.assert_close(stock.predict_action_chunk(pre(copy.deepcopy(observation))), expected)
            torch.testing.assert_close(rpost(torch.zeros(2, 4)), torch.full((2, 4), 2.))
            self.assertEqual(tuple(actual.shape), (2, 4, 4))
            self.assertEqual(set(restored.config.output_features), {'action'})

    def test_real_dataset_training_loop_and_validation(self):
        from excavator_vlm.policies.act import train
        if not (SUBMODULE / 'configs/splits/excavator_v4_seed1000.json').exists():
            self.skipTest('Frozen real dataset split unavailable')
        with tempfile.TemporaryDirectory() as directory:
            settings = resolve_settings(SUBMODULE / 'configs/act/smoke.yaml', {
                'output_dir': str(Path(directory) / 'run'), 'steps': 2,
                'dataset': {'episodes': [1]}, 'batch_size': 2, 'num_workers': 0,
                'policy': {'device': 'cpu', 'pretrained_backbone_weights': None,
                           'dim_model': 32, 'n_heads': 4, 'dim_feedforward': 64,
                           'n_encoder_layers': 1, 'n_decoder_layers': 1,
                           'n_vae_encoder_layers': 1, 'latent_dim': 4},
                'benchmark': {'validation_freq': 2, 'validation_max_batches': 1},
                'launcher': {'gpu_ids': [0], 'mixed_precision': 'no', 'cpu_threads': 2}})
            cfg = build_train_config(settings)
            train.run(settings, cfg, SUBMODULE / 'configs/act/smoke.yaml')
            output = Path(settings['output_dir'])
            reload_check = json.loads((output / 'reload_check.json').read_text())
            self.assertEqual(reload_check, {'status': 'passed', 'shape': [1, 10, 4]})
            validation = json.loads((output / 'checkpoints/last/validation.json').read_text())
            self.assertEqual(validation['frames'], 2)
            self.assertTrue((output / 'best_checkpoint.json').exists())
            # Exercise the public offline CLI against the actual saved model,
            # including provenance validation and NPZ/JSON serialization.
            import subprocess
            result = subprocess.run([
                sys.executable, str(SUBMODULE / 'scripts/eval_act_offline.py'),
                '--checkpoint', str(output / 'checkpoints/last/pretrained_model'),
                '--dataset', settings['dataset']['root'], '--device', 'cpu',
                '--batch-size', '2', '--max-batches', '1', '--output', str(output / 'offline'),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            offline = json.loads((output / 'offline/metrics.json').read_text())
            self.assertFalse(offline['overlaps_training'])
            self.assertEqual(offline['frames'], 2)
            self.assertTrue((output / 'offline/actions.npz').is_file())


if __name__ == '__main__':
    unittest.main()
