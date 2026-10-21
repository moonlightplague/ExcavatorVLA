"""CPU storage/contract tests; real CUDA AMP validation is reported separately."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from excavator_vlm.policies.pi05.precision import (
    PRECISION_CONTRACT, assert_fp32_training, configure_trainable_precision,
    inference_precision, precision_summary, validate_resume_precision,
)


class PrecisionTests(unittest.TestCase):
    def test_cast_before_loading_preserves_base_precision_and_frozen_storage(self):
        policy = torch.nn.Module()
        policy.expert = torch.nn.Linear(2, 2).bfloat16()
        policy.backbone = torch.nn.Linear(2, 2).bfloat16().requires_grad_(False)
        frozen = policy.backbone.weight.detach().clone()
        configure_trainable_precision(policy)
        state = policy.state_dict()
        state['expert.weight'] = torch.full((2, 2), .123456789, dtype=torch.float32)
        policy.load_state_dict(state)
        self.assertEqual(policy.expert.weight.dtype, torch.float32)
        torch.testing.assert_close(policy.expert.weight, state['expert.weight'], rtol=0, atol=0)
        self.assertEqual(policy.backbone.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(policy.backbone.weight, frozen, rtol=0, atol=0)
        self.assertEqual(precision_summary(policy)['trainable'], {'torch.float32': 6})
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad])
        policy.expert(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        assert_fp32_training(policy, optimizer)
        optimizer.state[policy.expert.weight]['exp_avg'] = torch.zeros(2, 2, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'exp_avg'):
            assert_fp32_training(policy, optimizer)

    def test_inference_autocast_matches_new_contract_and_leaves_legacy_alone(self):
        policy = SimpleNamespace(config=SimpleNamespace(dtype='bfloat16'),
                                 parameters=lambda: iter([SimpleNamespace(device=torch.device('cuda:0'))]))
        with patch('excavator_vlm.policies.pi05.precision.torch.autocast',
                   return_value=nullcontext()) as autocast:
            with inference_precision(policy):
                pass
            autocast.assert_not_called()
            policy._excavator_precision_contract = PRECISION_CONTRACT
            with inference_precision(policy):
                pass
            autocast.assert_called_once_with('cuda', dtype=torch.bfloat16)

    def test_rejects_legacy_or_unknown_exact_resume(self):
        for manifest in ({}, {'precision_contract': 'unknown'}):
            with self.assertRaisesRegex(ValueError, 'start a new run'):
                validate_resume_precision(manifest)
        validate_resume_precision({'precision_contract': PRECISION_CONTRACT})

    def test_configuration_refuses_casting_after_backward(self):
        policy = torch.nn.Linear(2, 2)
        policy(torch.ones(1, 2)).sum().backward()
        with self.assertRaises(ValueError):
            configure_trainable_precision(policy)


if __name__ == '__main__':
    unittest.main()
