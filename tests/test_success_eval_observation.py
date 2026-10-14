"""Validate camera observations passed to the success evaluator."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F


class SuccessEvalObservationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / 'scripts/eval_smolvla_success.py'
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in ('observation', 'validate_state')]
        self.images = {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in ('0', '1', '2')}
        self.names = [f'state_{index}' for index in range(28)]
        namespace = {
            'np': np, 'torch': torch, 'F': F, 'STATE_NAMES_28D': self.names,
            'decode_camera_images': lambda *args, **kwargs: self.images,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        self.observe = namespace['observation']
        self.reply = {
            'observation_32d_ready': True,
            'observation_contract': {'observation.state': {'names': self.names}},
            'observation_state_28d': [0.0] * 28, 'task_text': 'Dig and unload.',
        }

    def test_rendered_black_images_reach_the_policy(self):
        batch = self.observe(self.reply)
        for name in ('0', '1', '2'):
            image = batch[f'observation.images.{name}']
            self.assertEqual(tuple(image.shape), (1, 3, 256, 256))
            self.assertFalse(torch.any(image))

    def test_malformed_image_is_still_rejected(self):
        self.images['0'] = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'Invalid camera 0'):
            self.observe(self.reply)


if __name__ == '__main__':
    unittest.main()
