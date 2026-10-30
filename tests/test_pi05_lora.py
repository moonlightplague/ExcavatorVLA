"""Small CPU tests for scoped LoRA, immutable bases and strict delta contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file

from excavator_vlm.policies.pi05.checkpoints import FORMAT, apply_delta, trainable_keys
from excavator_vlm.policies.pi05.config import resolve_settings
from excavator_vlm.policies.pi05.lora import (
    EXPERT_ONLY, VLM_LORA, LoRALinear, assert_finetuning_matches, check_first_update,
    finetuning_metadata, finetuning_spec, optimizer_groups, set_lora_training,
    setup_finetuning, setup_from_manifest, update_snapshot,
)
from excavator_vlm.data.contract import file_hash
from excavator_vlm.policies.pi05.precision import configure_trainable_precision, assert_fp32_training

SUBMODULE = Path(__file__).resolve().parents[1]


def fixture():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    core = policy.model.paligemma_with_expert = torch.nn.Module()
    core.paligemma = torch.nn.Module()
    language = core.paligemma.language_model = torch.nn.Module()
    language.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(layer.self_attn, name, torch.nn.Linear(4, 4, bias=False))
        language.layers.append(layer)
    core.paligemma.vision_tower = torch.nn.Linear(4, 4)
    core.paligemma.requires_grad_(False)
    core.gemma_expert = torch.nn.Linear(4, 4)
    configure_trainable_precision(policy)
    return policy


def spec(**overrides):
    return finetuning_spec(VLM_LORA, {'r': 2, 'alpha': 4, **overrides})


class LoRATests(unittest.TestCase):
    def test_zero_identity_scoping_fp32_and_rng_preservation(self):
        model = fixture()
        before = copy.deepcopy(model.state_dict())
        rng = torch.random.get_rng_state().clone()
        setup_finetuning(model, spec())
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        wrapped = model.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj
        self.assertIsInstance(wrapped, LoRALinear)
        x = torch.randn(3, 4)
        torch.testing.assert_close(wrapped(x), torch.nn.functional.linear(x, wrapped.weight), rtol=0, atol=0)
        self.assertEqual(len(finetuning_metadata(model)['target_modules']), 8)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for name, p in model.named_parameters():
            if 'paligemma.' in name:
                self.assertEqual(p.requires_grad, name.endswith(('.lora_A', '.lora_B')))
        self.assertEqual(len(optimizer_groups(model)), 2)
        assert_fp32_training(model)

    def test_first_update_changes_adapters_not_base_and_dropout_mode(self):
        model = fixture()
        setup_finetuning(model, spec(dropout=.1))
        attention = model.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
        model.eval()
        set_lora_training(model, True)
        self.assertTrue(attention.q_proj.training)
        self.assertFalse(model.model.paligemma_with_expert.paligemma.training)
        before = update_snapshot(model)
        optimizer = torch.optim.AdamW(optimizer_groups(model), lr=1e-3)
        loss = sum(getattr(attention, key)(torch.ones(2, 4)).square().mean()
                   for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        loss.backward()
        optimizer.step()
        result = check_first_update(model, before)
        self.assertGreater(result['groups']['lora_B']['changed'], 0)
        self.assertEqual(result['groups']['frozen_sample']['changed'], 0)
        set_lora_training(model, False)
        self.assertFalse(attention.q_proj.training)

    def test_strict_delta_roundtrip_restores_nonzero_lora(self):
        model = fixture()
        base = copy.deepcopy(model.state_dict())
        setup_finetuning(model, spec())
        for name, p in model.named_parameters():
            if name.endswith('.lora_B'):
                p.data.fill_(.03)
        keys = trainable_keys(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file({key: model.state_dict()[key].contiguous() for key in keys}, str(path / 'model.safetensors'))
            manifest = {'format': FORMAT, 'trainable_keys': keys,
                        'delta_sha256': file_hash(path / 'model.safetensors'),
                        'finetuning': finetuning_metadata(model)}
            (path / 'checkpoint.json').write_text(json.dumps(manifest))
            restored = fixture()
            restored.load_state_dict(base, strict=True)
            setup_from_manifest(restored, manifest)
            apply_delta(restored, path)
            for key in keys:
                torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
            wrong = fixture()
            setup_finetuning(wrong, spec(r=3))
            with self.assertRaisesRegex(ValueError, 'LoRA configuration'):
                apply_delta(wrong, path)

    def test_legacy_expert_manifest_and_reject_bad_targets(self):
        model = fixture()
        setup_from_manifest(model, {})
        self.assertEqual(finetuning_metadata(model)['mode'], EXPERT_ONLY)
        assert_finetuning_matches(model, {})
        with self.assertRaises(ValueError):
            finetuning_spec(VLM_LORA, {'targets': ['lm_head']})
        with self.assertRaises(ValueError):
            finetuning_spec(EXPERT_ONLY, {'r': 2})
        with self.assertRaises(ValueError):
            setup_finetuning(model, spec())

    def test_dual_gpu_configs_preserve_effective_batch(self):
        for filename in ('expert_ab_overfit.yaml', 'lora_overfit.yaml'):
            settings = resolve_settings(SUBMODULE / 'configs/pi05' / filename)
            self.assertEqual(settings['launcher']['gpu_ids'], [0, 1])
            self.assertEqual(settings['steps'], 2000)
            self.assertEqual(settings['batch_size'] * settings['training']['gradient_accumulation_steps'] * 2, 8)

    def test_tiny_trainer_lora_save_resume_and_independent_loader(self):
        # Explicitly synthetic fixture: validates plumbing, not real PI05 learning.
        import numpy as np
        import pandas as pd
        from test_pi05 import TinyPolicy, ToyDataset, cpu_settings, toy_tokenizer
        from excavator_vlm.policies.pi05 import adapter, checkpoints, inference, train
        from excavator_vlm.policies.pi05.adapter import flow_loss_terms
        from excavator_vlm.policies.pi05.config import build_policy_config
        from safetensors.torch import load_file

        class TinyLoRAPolicy(TinyPolicy):
            def __init__(self, config):
                super().__init__(config)
                self.model = fixture().model
                self.model.paligemma_with_expert.gemma_expert.requires_grad_(False)
                for p in self.model.paligemma_with_expert.paligemma.parameters():
                    p.data.fill_(.1)

            def predict(self, batch):
                attention = self.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
                prefix = sum(getattr(attention, key)(batch['observation.state'][:, :4])
                             for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
                return self.expert(batch['observation.state']) + torch.nn.functional.pad(prefix, (0, 28))

            def forward(self, batch, denominator=None, world_size=1):
                predicted = self.predict(batch)[:, None].expand(-1, 10, -1)
                target = torch.nn.functional.pad(batch['action'], (0, 28)) + torch.randn_like(predicted) * .01
                numerator, local = flow_loss_terms((predicted - target).square(), batch)
                loss = numerator * world_size / (local if denominator is None else denominator).clamp_min(1)
                return loss, {'flow_numerator': numerator.detach(), 'valid_dimensions': local.detach()}

            def predict_action_chunk(self, batch):
                return self.predict(batch)[:, None, :4].expand(-1, 10, -1)

        frames = pd.DataFrame({'episode_index': [1, 1],
            'observation.state': [np.zeros(28), np.ones(28)], 'action': [np.zeros(4), np.ones(4)]})
        prepared = ({'fingerprint': 'synthetic-lora-fixture'},
                    {'train': [1], 'validation': [2], 'test': [3]}, frames, [1])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(inference, 'check_environment'), \
                patch.object(adapter, 'policy_class', return_value=TinyLoRAPolicy), \
                patch.object(checkpoints, 'load_base', return_value={'test_fixture': True}), \
                patch.object(inference, 'load_base', return_value={'test_fixture': True}), \
                patch.object(train, 'prepare', return_value=prepared), \
                patch.object(train, 'make_pi05_dataset', side_effect=lambda repo, root, episodes, *args: ToyDataset(episodes)):
            runs = []
            for name, stop in (('continuous', None), ('interrupted', 1)):
                settings = cpu_settings(Path(directory) / name)
                settings['training'].update(spec())
                toy_tokenizer(settings['assets']['tokenizer_path'])
                config = build_policy_config(settings)
                train.run(settings, config, SUBMODULE / 'configs/pi05/lora_smoke.yaml', stop_after=stop)
                runs.append((settings, config))
            settings, config = runs[1]
            settings['training']['resume'] = str(Path(settings['output_dir']) / 'checkpoints/000001/pretrained_model')
            train.run(settings, config, SUBMODULE / 'configs/pi05/lora_smoke.yaml')
            paths = [Path(s['output_dir']) / 'checkpoints/000002/pretrained_model' for s, _ in runs]
            expected, actual = [load_file(str(p / 'model.safetensors')) for p in paths]
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            restored, pre, _ = inference.load_pi05(paths[1], 'cpu')
            for key in expected:
                torch.testing.assert_close(restored.state_dict()[key], expected[key], rtol=0, atol=0)
            self.assertTrue(any(torch.count_nonzero(value) for key, value in actual.items() if key.endswith('.lora_B')))


if __name__ == '__main__':
    unittest.main()
