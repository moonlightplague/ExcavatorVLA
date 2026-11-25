"""Causality and reset tests for the model-only recurrent ACT adapter."""
import unittest
import torch
from lerobot.policies.act.persistent_extension import ActionMemory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.memory = ActionMemory(12, 16, 8).eval()
        self.features = torch.randn(3, 13, 12)

    def test_zero_head_preserves_source(self):
        result, _ = self.memory(self.features)
        self.assertEqual(result.abs().max().item(), 0)

    def test_streaming_equals_sequence(self):
        torch.nn.init.normal_(self.memory.output.weight)
        expected, _ = self.memory(self.features)
        hidden = None
        actual = []
        for frame in self.features.unbind(1):
            value, hidden = self.memory(frame[:, None], hidden)
            actual.append(value)
        torch.testing.assert_close(torch.cat(actual, 1), expected)

    def test_recomputed_prefix_including_empty(self):
        torch.nn.init.normal_(self.memory.output.weight)
        starts = torch.tensor([0, 3, 8])
        hidden = self.memory.prefix_hidden(self.features[:, :8], starts)
        for i, start in enumerate(starts.tolist()):
            full = self.memory(self.features[i:i+1])[0][:, start:]
            suffix = self.memory(self.features[i:i+1, start:], hidden[:, i:i+1])[0]
            torch.testing.assert_close(full, suffix)

    def test_history_older_than_three_predictions_matters(self):
        torch.nn.init.normal_(self.memory.output.weight)
        altered = self.features.clone()
        altered[:, :4] += 3
        expected = self.memory(self.features)[0]
        actual = self.memory(altered)[0]
        self.assertGreater((expected[:, -1] - actual[:, -1]).abs().max().item(), 1e-6)
        torch.testing.assert_close(self.memory(self.features)[0], expected)

    def test_finite_learning_and_gru_gradients(self):
        optimizer = torch.optim.Adam(self.memory.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            out = self.memory(self.features)[0]
            loss = (out-1).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            optimizer.step()
        self.assertGreater(self.memory.gru.weight_ih_l0.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
