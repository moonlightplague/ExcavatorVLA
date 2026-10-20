"""Check that evaluation supplies a cut target, rather than a sand height."""
import unittest
import ast
from pathlib import Path
import numpy as np

from excavator_common.collection_target import (
    dig_target_from_particles, PATH_SURFACE_PROBES_M, CENTER_SURFACE_RADIUS_M,
    PATH_SURFACE_RADIUS_M, CUT_DEPTH_M, EXTRA_BITE_M,
)


class CollectionDigTargetTests(unittest.TestCase):
    def test_surface_probe_configuration_matches_the_collection_runtime(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / 'ExcavatorVLA-isaac/scripts/excavator_app/excavator_runtime.py').read_text(encoding='utf-8-sig'))
        names = {'SAND_SURFACE_QUERY_RADIUS', 'AUTO_DIG_TARGET_LOCAL_SURFACE_RADIUS',
                 'AUTO_DIG_TARGET_PATH_SURFACE_PROBES_M', 'AUTO_COLLECT_TARGET_DEPTHS',
                 'AUTO_DIG_TARGET_EXTRA_BITE_M'}
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in names}
        self.assertEqual(CENTER_SURFACE_RADIUS_M, values['SAND_SURFACE_QUERY_RADIUS'])
        self.assertEqual(PATH_SURFACE_RADIUS_M, values['AUTO_DIG_TARGET_LOCAL_SURFACE_RADIUS'])
        self.assertEqual(PATH_SURFACE_PROBES_M, values['AUTO_DIG_TARGET_PATH_SURFACE_PROBES_M'])
        self.assertEqual(CUT_DEPTH_M, max(values['AUTO_COLLECT_TARGET_DEPTHS']))
        self.assertEqual(EXTRA_BITE_M, values['AUTO_DIG_TARGET_EXTRA_BITE_M'])

    def test_flat_surface_subtracts_collection_cut_depth_and_extra_bite(self):
        xy = np.asarray([(x, y) for x in np.arange(-1, 1.01, .04)
                         for y in np.arange(-.3, .31, .04)])
        # Include the interior of the pile: its volume-height percentile is
        # deliberately different from the surface used by the collector.
        points = np.concatenate([np.column_stack([xy, np.full(len(xy), z)])
                                 for z in np.linspace(0., 1., 11)])
        target, report = dig_target_from_particles(points, [0, 0], [-5, 0])
        self.assertAlmostEqual(report['center_surface_z'], .90)
        self.assertAlmostEqual(target[2], .20)
        self.assertAlmostEqual(report['planned_depth_m'], .70)
        self.assertEqual(len(report['probes']), 6)
        self.assertGreater(report['plan_surface_z'], target[2])

    def test_cut_path_uses_robot_relative_inward_direction_and_low_surface(self):
        points = np.asarray([[-offset, 0., 1.2 if offset < .2 else .9]
                             for offset in PATH_SURFACE_PROBES_M])
        target, report = dig_target_from_particles(points, [0, 0], [-5, 0])
        self.assertAlmostEqual(report['plan_surface_z'], .9)
        self.assertGreater(report['surface_uncertainty_margin_m'], 0)
        self.assertLess(target[2], .2)

    def test_shallow_pile_stays_above_floor_and_missing_surface_is_rejected(self):
        target, _ = dig_target_from_particles([[0, 0, .2]], [0, 0], [-5, 0])
        self.assertAlmostEqual(target[2], .04)
        for points in ([], [[10, 10, 1]], [[0, 0, float('nan')]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                dig_target_from_particles(points, [0, 0], [-5, 0])


if __name__ == '__main__':
    unittest.main()
