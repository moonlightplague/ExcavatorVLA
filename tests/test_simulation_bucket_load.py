"""Exercise standalone bucket metrics without starting Kit or a GPU."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np


class BucketLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'run_simulation.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'estimate_bucket_load_particles')
        cls.code = compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec')

    def estimate_empty_bucket(self, tracker):
        class IdentityTransform:
            def Transform(self, point):
                return point

            def GetInverse(self):
                return self

        # One sand particle outside the bucket's [0, 1]^3 world bounds.
        api = {'particle_positions_fn': lambda: np.array([[10, 10, 10]])}
        namespace = {
            'np': np, 'time': time, 'sand_api': api,
            'builtins': SimpleNamespace(_SAND_SITE=api),
            'stage': SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
            'UsdGeom': SimpleNamespace(Xformable=lambda prim: SimpleNamespace(
                ComputeLocalToWorldTransform=lambda time: IdentityTransform())),
            'Usd': SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None)),
            'Gf': SimpleNamespace(Vec3d=lambda *values: values),
            'bucket_source_tracker': tracker,
            'bucket_volume_topology': {
                'available': True, 'source': 'authored_closed_mesh', 'reason': 'ok',
                'local_min': [0, 0, 0], 'local_max': [1, 1, 1],
            },
        }
        exec(self.code, namespace)
        return namespace['estimate_bucket_load_particles'](source_tracking_required=True)

    def test_empty_bucket_has_valid_source_tracking(self):
        result = self.estimate_empty_bucket({'available': True, 'mask': np.array([True])})
        self.assertEqual(result['count'], 0)
        self.assertEqual(result['candidate_count'], 0)
        self.assertTrue(result['source_tracking_valid'])

    def test_empty_bucket_still_rejects_missing_tracker(self):
        result = self.estimate_empty_bucket({'available': False, 'reason': 'not_captured'})
        self.assertFalse(result['source_tracking_valid'])
        self.assertEqual(result['source_tracking_reason'], 'not_captured')

    def test_empty_bucket_rejects_changed_particle_count(self):
        result = self.estimate_empty_bucket({'available': True, 'mask': np.array([True, True]), 'reason': 'ok'})
        self.assertFalse(result['source_tracking_valid'])
        self.assertEqual(result['source_tracking_reason'], 'particle_count_changed')


if __name__ == '__main__':
    unittest.main()
