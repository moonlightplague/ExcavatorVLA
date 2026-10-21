"""PI05 contracts and real trainer/checkpoint flow with a clearly synthetic tiny policy."""
import copy
import itertools
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
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HOME', '/tmp/excavator-pi05-test-hf')

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from excavator_vlm.data.contract import CAMERAS
from excavator_vlm.policies.pi05.adapter import flow_loss_terms, training_batch
from excavator_vlm.policies.pi05.checkpoints import apply_delta, complete_tied_aliases, strict_state_load
from excavator_vlm.policies.pi05.config import build_policy_config, resolve_settings
from excavator_vlm.policies.pi05.dataset import training_stats
from excavator_vlm.policies.pi05.processors import make_processors
from excavator_vlm.policies.pi05.sampling import ResumableBatchSampler

torch.set_num_threads(2)


def toy_tokenizer(path):
    """Unit-test tokenizer only; never substitutes for PaliGemma in production."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, 'Task': 2}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.save_pretrained(path)


def cpu_settings(directory):
    directory = Path(directory)
    return resolve_settings(SUBMODULE / 'configs/pi05/smoke.yaml', {
        'output_dir': str(directory / 'run'), 'steps': 2, 'dataset': {'episodes': [1]},
        'policy': {'device': 'cpu', 'dtype': 'float32'},
        'launcher': {'gpu_ids': [0], 'mixed_precision': 'no', 'cpu_threads': 2},
        'assets': {'tokenizer_path': str(directory / 'tokenizer')},
        'training': {'gradient_accumulation_steps': 2},
        'benchmark': {'validation_freq': 2, 'validation_max_batches': 1}})


class ContractTests(unittest.TestCase):
    def test_four_configs_bind_excavator_and_inherit_the_same_base(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, steps in {'smoke': 20, 'overfit': 2000, 'pilot': 5000, 'full': 40000}.items():
                with self.subTest(name=name):
                    settings = resolve_settings(SUBMODULE / f'configs/pi05/{name}.yaml',
                                                {'output_dir': str(Path(directory) / name)})
                    self.assertEqual(settings['steps'], steps)
                    self.assertEqual(settings['training']['trainable_dtype'], 'float32')
                    self.assertEqual(Path(settings['assets']['base_path']), (SUBMODULE.parent / 'models/pi05_base').resolve())
                    config = build_policy_config(settings)
                    self.assertEqual(tuple(config.image_features), CAMERAS)
                    self.assertEqual(list(config.input_features['observation.state'].shape), [28])
                    self.assertEqual(list(config.output_features['action'].shape), [4])
                    self.assertEqual((config.max_state_dim, config.max_action_dim, config.chunk_size), (32, 32, 10))

    def test_rejects_unsupported_training_modes(self):
        for overrides in ({'policy': {'train_expert_only': False}},
                          {'policy': {'empty_cameras': 1}},
                          {'training': {'trainable_dtype': 'bfloat16'}},
                          {'launcher': {'mixed_precision': 'no'}},
                          {'training': {'gradient_accumulation_steps': 0}}):
            with self.assertRaises(ValueError):
                resolve_settings(SUBMODULE / 'configs/pi05/smoke.yaml', overrides)

    def test_quantiles_do_not_include_held_out_extremes(self):
        frames = pd.DataFrame({'episode_index': [1, 1, 2],
            'observation.state': [np.zeros(28), np.ones(28), np.full(28, 1e9)],
            'action': [np.zeros(4), np.ones(4), np.full(4, 1e9)]})
        stats = training_stats(frames, [1])
        np.testing.assert_allclose(stats['action']['q01'], .01)
        np.testing.assert_allclose(stats['observation.state']['q99'], .99)


class LossTests(unittest.TestCase):
    def test_padding_weights_and_unused_action_dimensions_have_zero_gradient(self):
        losses = torch.ones(1, 3, 32, requires_grad=True)
        batch = {'action_is_pad': torch.tensor([[False, False, True]]),
                 'action_loss_weight': torch.tensor([[.25, 0., 1.]])}
        numerator, denominator = flow_loss_terms(losses, batch)
        (numerator / denominator).backward()
        self.assertEqual(denominator.item(), 1.)
        self.assertEqual(losses.grad[:, 1:].sum().item(), 0.)
        self.assertEqual(losses.grad[..., 4:].sum().item(), 0.)
        self.assertAlmostEqual(losses.grad.sum().item(), 1.)

    def test_accumulation_uses_total_valid_weight_not_mean_of_means(self):
        parameter = torch.tensor(2., requires_grad=True)
        denominators, numerators = [], []
        for weights in (torch.tensor([[1., 0.]]), torch.tensor([[.5, .5]])):
            numerator, denominator = flow_loss_terms(parameter.square().expand(1, 2, 32),
                {'action_is_pad': torch.zeros(1, 2, dtype=torch.bool), 'action_loss_weight': weights})
            numerators.append(numerator)
            denominators.append(denominator)
        total = sum(denominators)
        for numerator in numerators:
            (numerator / total).backward()
        self.assertAlmostEqual(parameter.grad.item(), 4.)

    def test_all_padding_has_differentiable_zero_numerator(self):
        losses = torch.ones(1, 1, 32, requires_grad=True)
        numerator, denominator = flow_loss_terms(losses,
            {'action_is_pad': torch.ones(1, 1, dtype=torch.bool), 'action_loss_weight': torch.ones(1, 1)})
        numerator.backward()
        self.assertEqual(denominator.item(), 0.)
        self.assertEqual(losses.grad.sum().item(), 0.)


class SamplerTests(unittest.TestCase):
    def test_resume_uses_consumed_position_across_epoch_boundaries(self):
        original = list(itertools.islice(ResumableBatchSampler(7, 2, 123), 13))
        resumed = list(itertools.islice(ResumableBatchSampler(7, 2, 123, start_batch=5), 8))
        self.assertEqual(original[5:], resumed)

    def test_two_ranks_shard_the_same_shuffle(self):
        rank0 = list(itertools.islice(ResumableBatchSampler(8, 2, 123, 0, 2), 2))
        rank1 = list(itertools.islice(ResumableBatchSampler(8, 2, 123, 1, 2), 2))
        self.assertEqual(set(sum(rank0, [])) | set(sum(rank1, [])), set(range(8)))
        self.assertFalse(set(sum(rank0, [])) & set(sum(rank1, [])))


class ProcessorTests(unittest.TestCase):
    def test_language_quantile_roundtrip_and_weight_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = cpu_settings(directory)
            toy_tokenizer(settings['assets']['tokenizer_path'])
            config = build_policy_config(settings)
            stats = {key: {'q01': [0.] * dim, 'q99': [2.] * dim}
                     for key, dim in (('observation.state', 28), ('action', 4))}
            pre, post = make_processors(config, stats, settings['assets']['tokenizer_path'])
            batch = {'observation.state': torch.ones(1, 28), 'action': torch.ones(1, 10, 4),
                **{key: torch.full((1, 3, 256, 256), .5) for key in CAMERAS}, 'task': ['Dig then load.'],
                'action_is_pad': torch.zeros(1, 10, dtype=torch.bool),
                'action_loss_weight': torch.full((1, 10), .25)}
            processed = training_batch(pre, copy.deepcopy(batch))
            self.assertEqual(tuple(processed['observation.language.tokens'].shape), (1, 200))
            torch.testing.assert_close(processed['action_loss_weight'], batch['action_loss_weight'])
            torch.testing.assert_close(processed['action'], torch.zeros(1, 10, 4))
            torch.testing.assert_close(post(processed['action'][:, 0]), torch.ones(1, 4))
            torch.testing.assert_close(processed[CAMERAS[0]], batch[CAMERAS[0]])
            with self.assertRaises(ValueError):
                pre({**batch, 'task': ['too long ' * 300]})


class StrictLoadingTests(unittest.TestCase):
    def test_only_actual_tied_parameters_can_restore_omitted_aliases(self):
        policy = torch.nn.Module()
        policy.first = torch.nn.Linear(2, 2, bias=False)
        policy.second = torch.nn.Linear(2, 2, bias=False)
        policy.second.weight = policy.first.weight
        recovered, names = complete_tied_aliases(policy, {'first.weight': torch.ones(2, 2)})
        self.assertEqual(names, ['second.weight'])
        strict_state_load(policy, recovered)
        self.assertIs(policy.first.weight, policy.second.weight)

    def test_missing_unexpected_and_bad_shapes_fail_before_mutation(self):
        policy = torch.nn.Linear(2, 2)
        original = copy.deepcopy(policy.state_dict())
        for broken in ({'weight': torch.zeros(2, 2)}, {**original, 'extra': torch.zeros(1)},
                       {**original, 'weight': torch.zeros(3, 2)}):
            with self.assertRaises(ValueError):
                strict_state_load(policy, broken)
            for key, value in policy.state_dict().items():
                torch.testing.assert_close(value, original[key])


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, episodes):
        self.episode = episodes[0]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {'observation.state': torch.full((28,), index / 3),
            **{key: torch.full((3, 256, 256), .5) for key in CAMERAS}, 'task': 'Dig then load.',
            'action': torch.full((10, 4), .1 * index), 'action_is_pad': torch.zeros(10, dtype=torch.bool),
            'action_loss_weight': torch.ones(10), 'observation.recovery_active': torch.zeros(10),
            'episode_index': torch.tensor(self.episode), 'frame_index': torch.tensor(index)}


class TinyPolicy(torch.nn.Module):
    """Synthetic fixture for trainer/resume plumbing, NOT a PI05 performance test."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frozen = torch.nn.Linear(1, 1)
        self.frozen.requires_grad_(False)
        self.expert = torch.nn.Linear(28, 32)

    def forward(self, batch, denominator=None, world_size=1):
        predicted = self.expert(batch['observation.state']).unsqueeze(1).expand(-1, 10, -1)
        target = torch.nn.functional.pad(batch['action'], (0, 28)) + torch.randn_like(predicted) * .01
        numerator, local = flow_loss_terms((predicted - target).square(), batch)
        loss = numerator * world_size / (local if denominator is None else denominator).clamp_min(1)
        return loss, {'flow_numerator': numerator.detach(), 'valid_dimensions': local.detach()}

    def predict_action_chunk(self, batch):
        assert 'observation.language.tokens' in batch
        self.eval()
        return self.expert(batch['observation.state'])[:, None, :4].expand(-1, 10, -1) + torch.randn(1) * .001


class TrainerTests(unittest.TestCase):
    def test_save_delta_resume_rng_sampler_and_validation_with_tiny_fixture(self):
        from excavator_vlm.policies.pi05 import adapter, checkpoints, inference, train
        frames = pd.DataFrame({'episode_index': [1, 1],
            'observation.state': [np.zeros(28), np.ones(28)], 'action': [np.zeros(4), np.ones(4)]})
        fixture = ({'fingerprint': 'synthetic-unit-fixture'},
                   {'train': [1], 'validation': [2], 'test': [3]}, frames, [1])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(inference, 'check_environment'), \
                patch.object(adapter, 'policy_class', return_value=TinyPolicy), \
                patch.object(checkpoints, 'load_base', return_value={'test_fixture': True}), \
                patch.object(inference, 'load_base', return_value={'test_fixture': True}), \
                patch.object(train, 'prepare', return_value=fixture), \
                patch.object(train, 'make_pi05_dataset', side_effect=lambda repo, root, episodes, *args: ToyDataset(episodes)):
            root = Path(directory)
            outputs = []
            for name, stop in (('continuous', None), ('interrupted', 1)):
                settings = cpu_settings(root / name)
                toy_tokenizer(settings['assets']['tokenizer_path'])
                config = build_policy_config(settings)
                train.run(settings, config, SUBMODULE / 'configs/pi05/smoke.yaml', stop_after=stop)
                outputs.append((settings, config))
            settings, config = outputs[1]
            settings['training']['resume'] = str(Path(settings['output_dir']) / 'checkpoints/000001/pretrained_model')
            train.run(settings, config, SUBMODULE / 'configs/pi05/smoke.yaml')
            expected = load_file(str(Path(outputs[0][0]['output_dir']) / 'checkpoints/000002/pretrained_model/model.safetensors'))
            actual_path = Path(settings['output_dir']) / 'checkpoints/000002/pretrained_model'
            actual = load_file(str(actual_path / 'model.safetensors'))
            self.assertEqual(set(actual), {'expert.weight', 'expert.bias'})
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            manifest = json.loads((actual_path / 'checkpoint.json').read_text())
            from excavator_vlm.policies.pi05.precision import PRECISION_CONTRACT
            self.assertEqual(manifest['precision_contract'], PRECISION_CONTRACT)
            self.assertTrue(all(value.dtype == torch.float32 for value in actual.values()))
            restored, _, _ = inference.load_pi05(actual_path, 'cpu')
            for key in expected:
                torch.testing.assert_close(restored.state_dict()[key], actual[key], rtol=0, atol=0)
            self.assertEqual(manifest['step'], 2)
            self.assertTrue((actual_path.parent / 'validation.json').is_file())
            history = Path(settings['output_dir']) / 'validation_history.jsonl'
            self.assertEqual(json.loads(history.read_text().splitlines()[-1])['step'], 2)
            saved = torch.load(actual_path.parent / 'training_state.pt', weights_only=True)
            self.assertTrue(all(value.dtype == torch.float32 for state in saved['optimizer']['state'].values()
                                for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq')))
            self.assertEqual(saved['consumed_microbatches'], 4)
            with self.assertRaises(ValueError):
                apply_delta(torch.nn.Linear(2, 2), actual_path)


if __name__ == '__main__':
    unittest.main()
