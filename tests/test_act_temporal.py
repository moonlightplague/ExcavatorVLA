"""Causal model history, compatibility and training checks in the private runtime."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from excavator_vlm.policies.act.inference import load_act
from excavator_vlm.policies.act.temporal import HISTORY_MASK


class TemporalACTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.config = ACTConfig(device='cpu', chunk_size=4, n_action_steps=3,
            pretrained_backbone_weights=None, dim_model=32, n_heads=4,
            dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
            n_vae_encoder_layers=1, latent_dim=4, dropout=0., temporal_horizon=3,
            temporal_layers=1, input_features={
                'observation.state': {'type': 'STATE', 'shape': (28,)},
                **{f'observation.images.{i}': {'type': 'VISUAL', 'shape': (3, 32, 32)} for i in range(3)}},
            output_features={'action': {'type': 'ACTION', 'shape': (4,)}})
        # Use the standard dataclass decoder to construct typed PolicyFeature values.
        from lerobot.configs.types import FeatureType, PolicyFeature
        self.config.input_features = {'observation.state': PolicyFeature(FeatureType.STATE, (28,)),
            **{f'observation.images.{i}': PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for i in range(3)}}
        self.config.output_features = {'action': PolicyFeature(FeatureType.ACTION, (4,))}

    def window(self, batch=1):
        return {'observation.state': torch.randn(batch, 3, 28),
                **{f'observation.images.{i}': torch.rand(batch, 3, 3, 32, 32) for i in range(3)},
                HISTORY_MASK: torch.zeros(batch, 3, dtype=torch.bool)}

    def current(self, window, index=-1):
        return {key: value[:, index] for key, value in window.items() if key != HISTORY_MASK}

    def activate_context(self, policy):
        with torch.no_grad():
            policy.model.temporal_context.output.weight.copy_(torch.eye(32) * .2)

    def test_zero_adapter_preserves_source_policy(self):
        baseline_config = copy.deepcopy(self.config)
        baseline_config.temporal_horizon = 1
        baseline = ACTPolicy(baseline_config).eval()
        temporal = ACTPolicy(self.config).eval()
        result = temporal.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(k.startswith('model.temporal_context.') for k in result.missing_keys))
        self.assertEqual(result.unexpected_keys, [])
        window = self.window()
        expected = baseline.predict_action_chunk(self.current(window))
        actual = temporal.predict_action_chunk(window)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_past_state_images_and_order_influence_actions(self):
        policy = ACTPolicy(self.config).eval()
        self.activate_context(policy)
        window = self.window()
        expected = policy.predict_action_chunk(window)
        changed = copy.deepcopy(window)
        changed['observation.state'][:, 0] += 4
        self.assertGreater(float((policy.predict_action_chunk(changed) - expected).abs().max()), 1e-6)
        changed = copy.deepcopy(window)
        for i in range(3):
            changed[f'observation.images.{i}'][:, 0] = 1 - changed[f'observation.images.{i}'][:, 0]
        self.assertGreater(float((policy.predict_action_chunk(changed) - expected).abs().max()), 1e-7)
        changed = copy.deepcopy(window)
        for key in self.current(window):
            changed[key][:, :2] = changed[key][:, :2].flip(1)
        self.assertGreater(float((policy.predict_action_chunk(changed) - expected).abs().max()), 1e-6)

    def test_initial_padding_cannot_influence_current_output(self):
        policy = ACTPolicy(self.config).eval()
        self.activate_context(policy)
        window = self.window()
        window[HISTORY_MASK][:, :2] = True
        expected = policy.predict_action_chunk(window)
        changed = copy.deepcopy(window)
        for key in self.current(window):
            changed[key][:, :2] += 100
        torch.testing.assert_close(policy.predict_action_chunk(changed), expected, rtol=1e-6, atol=1e-7)

    def test_online_model_buffer_matches_window_and_reset(self):
        policy = ACTPolicy(self.config).eval()
        self.activate_context(policy)
        window = self.window()
        for i in range(3):
            actual = policy.predict_action_chunk(self.current(window, i))
        self.assertEqual(policy.model.history_size, 3)
        torch.testing.assert_close(actual, policy.predict_action_chunk(window), rtol=1e-4, atol=1e-5)
        policy.reset()
        self.assertEqual(policy.model.history_size, 0)
        first = policy.predict_action_chunk(self.current(window))
        policy.reset()
        torch.testing.assert_close(policy.predict_action_chunk(self.current(window)), first)

    def test_native_loss_trains_temporal_parameters(self):
        policy = ACTPolicy(self.config).train()
        window = self.window(batch=2)
        window.update(action=torch.randn(2, 4, 4), action_is_pad=torch.zeros(2, 4, dtype=torch.bool))
        optimizer = torch.optim.SGD(policy.parameters(), lr=.01)
        loss, _ = policy(window)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(policy.model.temporal_context.output.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy(window)
        loss.backward()
        self.assertGreater(float(policy.model.temporal_context.state_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(policy.model.temporal_context.visual_projection.weight.grad.abs().sum()), 0)
        self.assertEqual(policy.model.history_size, 0)

    def test_strict_checkpoint_load_preserves_temporal_predictions(self):
        policy = ACTPolicy(self.config).eval()
        self.activate_context(policy)
        window = self.window()
        expected = policy.predict_action_chunk(window)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            policy.save_pretrained(path)
            # Processor loading is tested by the real-data training preflight;
            # native strict reconstruction verifies all model weights here.
            restored = ACTPolicy.from_pretrained(path, strict=True)
            self.assertEqual(restored.model.history_size, 0)
            torch.testing.assert_close(restored.predict_action_chunk(window), expected)


if __name__ == '__main__':
    unittest.main()
