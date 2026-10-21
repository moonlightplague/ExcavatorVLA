"""Small CPU numerical helper tests, not real PI05 precision evidence."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from excavator_vlm.data.contract import CAMERAS
from excavator_vlm.policies.pi05.diagnostics import (
    capture_outputs, compare_observations, difference, main, optimizer_trial,
)


class DiagnosticTests(unittest.TestCase):
    def test_reused_output_does_not_overwrite_completed_report(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            path = Path(directory) / 'report.json'
            path.write_text('{"status": "completed"}')
            argv = ['diagnose_pi05.py', '--checkpoint', '/unused', '--output', directory]
            with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(path.read_text(), '{"status": "completed"}')

    def test_observation_comparison_excludes_optional_action_target(self):
        observation = {'observation.state': torch.zeros(1, 28),
                       **{key: torch.zeros(1, 3, 2, 2) for key in CAMERAS},
                       'observation.language.tokens': torch.zeros(1, 2, dtype=torch.long),
                       'observation.language.attention_mask': torch.ones(1, 2, dtype=torch.bool),
                       'action': None}
        training = {**observation, 'action': torch.ones(1, 10, 4)}
        self.assertTrue(all(value['exact_equal'] for value in
                            compare_observations(training, observation).values()))

    def test_comparison_rejects_broadcast_and_nonfinite(self):
        a = torch.zeros(1, 2, 4)
        result = difference(a, a + 1)
        self.assertEqual(result['mae'], 1.)
        self.assertEqual(result['per_joint_mae']['bucket'], 1.)
        with self.assertRaises(ValueError):
            difference(a, torch.zeros(4))
        with self.assertRaises(FloatingPointError):
            difference(a, torch.full_like(a, float('nan')))

    def test_hook_cleanup_on_failure(self):
        layer = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with capture_outputs({'layer': layer}) as outputs:
                layer(torch.ones(1, 2))
                self.assertEqual(len(outputs['layer']), 1)
                raise RuntimeError('intentional')
        self.assertFalse(layer._forward_hooks)

    def test_trials_are_independent_and_never_mutate_sources(self):
        # Synthetic illustration of rounding, not evidence about trained PI05.
        parameter = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        snapshots = [('model.gemma_expert.weight', parameter.detach().clone(), torch.ones_like(parameter))]
        original_weight, original_moment = snapshots[0][1].clone(), state['state'][0]['exp_avg'].clone()
        native = optimizer_trial(snapshots, state, device='cpu', lr=1e-6)
        fp32 = optimizer_trial(snapshots, state, device='cpu', lr=1e-6, fp32=True)
        repeat = optimizer_trial(snapshots, state, device='cpu', lr=1e-6)
        self.assertEqual(native['groups']['expert_bf16']['changed'], 0)
        self.assertEqual(fp32['groups']['expert_bf16']['changed'], 16)
        self.assertEqual(native, repeat)
        torch.testing.assert_close(snapshots[0][1], original_weight, rtol=0, atol=0)
        torch.testing.assert_close(state['state'][0]['exp_avg'], original_moment, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
