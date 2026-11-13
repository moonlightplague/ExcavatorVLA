import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from baseline.coarse_source import current_source_points, install_current_source_planning


class CurrentSourceTests(unittest.TestCase):
    def test_source_ids_bucket_truck_and_extent_are_all_respected(self):
        points = np.array([[0, 7, .4], [1.8, 7, .3], [2, 7, .2],
                           [0, 7, 2], [0, 7, 3], [3.1, 7, .2]])
        before = points.copy()
        selected = current_source_points(points, [0, 7, 0],
            [False, False, False, True, False, False],
            [False, False, False, False, True, False],
            [True, True, False, True, True, True])
        np.testing.assert_allclose(selected, points[:2])
        np.testing.assert_array_equal(points, before)
        with self.assertRaisesRegex(ValueError, 'identity'):
            current_source_points(points, [0, 7, 0], [False]*6, [False]*6, [True])
        points[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            current_source_points(points, [0, 7, 0], [False]*6, [False]*6)

    def runtime(self, settled=True, coarse=True):
        points = np.array([[0, 7, .4], [1.8, 7, .3], [3.1, 7, .2]])
        ctx = dict(pile_center=[0, 7, 0], pile_radius=[1.2, 1.2])
        rt = SimpleNamespace(STATE={'repeat_planning_resolution': {'policy': 'coarse'}} if coarse else {},
            GROUND_TOP_Z=0., AUTO_DIG_DENSITY_RADIUS=.3, AUTO_DIG_MIN_LOCAL_PARTICLES=5,
            AUTO_DIG_SWEEP_RADIUS=.3, task_scene_context=lambda: ctx,
            get_sand_snapshot=Mock(return_value=dict(points=points, settled_points=None)),
            sand_region_masks=lambda p: (np.ones(len(p), bool), np.zeros(len(p), bool), np.zeros(len(p), bool)),
            sand_settle_status=Mock(return_value={'ok': settled}),
            build_sand_spatial_index=Mock(return_value='index'),
            auto_dig_target_sand_region_check=Mock(return_value=(False, 'outside_sand_polygon', {})),
            sand_surface_query_at_xy=Mock(return_value=(None, 'outside')),
            auto_dig_swept_density_count=Mock(return_value=99),
            sand_snapshot_surface_height=Mock(return_value=.3),
            sand_snapshot_density_count=Mock(return_value=6),
            target_radius_from_swing_center=lambda p: float(np.linalg.norm(p[:2])),
            dig_direction_unit=lambda p: [0, -1])
        return rt, ctx

    def test_avalanched_source_is_visible_without_changing_global_context_or_settle_gate(self):
        rt, ctx = self.runtime()
        soil = SimpleNamespace(current=lambda: SimpleNamespace(closed=False, error=''))
        with patch.dict('sys.modules', {'excavator_app': SimpleNamespace(newton_mpm_soil=soil)}):
            install_current_source_planning(rt)
            snapshot = rt.get_sand_snapshot()
        self.assertEqual(ctx['pile_radius'], [1.2, 1.2])
        self.assertEqual(snapshot['settled_count'], 2)
        self.assertEqual(snapshot['ctx']['pile_radius'], [3., 3.])
        self.assertEqual(len(rt.sand_settle_status.call_args.kwargs['points']), 2)
        ok, reason, _ = rt.auto_dig_target_sand_region_check(1.8, 7, snapshot=snapshot)
        self.assertTrue(ok)
        self.assertEqual(reason, 'actual_avalanched_source_material')
        rt.sand_snapshot_density_count.return_value = 0
        self.assertFalse(rt.auto_dig_target_sand_region_check(1.8, 7, snapshot=snapshot)[0])
        self.assertFalse(rt.auto_dig_target_sand_region_check(1.8, 4, snapshot=snapshot)[0])

    def test_bare_ground_never_becomes_evidence_of_sand_contact(self):
        rt, _ = self.runtime()
        snapshot = {'repeat_avalanche_source': True, 'settled_points': np.array([[1.8, 7, .3]])}
        rt.get_sand_snapshot.return_value = snapshot
        original_surface = rt.sand_surface_query_at_xy
        install_current_source_planning(rt)
        self.assertEqual(rt.sand_surface_query_at_xy(1.8, 7), (.3, 'actual_Newton_source_snapshot'))
        original_surface.assert_not_called()
        rt.sand_snapshot_surface_height.return_value = None
        z, source = rt.sand_surface_query_at_xy(1.8, 7)
        self.assertEqual(z, 0.)
        self.assertTrue(source.startswith('fallback'))
        self.assertEqual(rt.sand_surface_query_at_xy(5, 7), (None, 'outside'))

    def test_unsettled_source_cannot_produce_a_planning_inventory(self):
        rt, _ = self.runtime(settled=False)
        soil = SimpleNamespace(current=lambda: SimpleNamespace(closed=False, error=''))
        with patch.dict('sys.modules', {'excavator_app': SimpleNamespace(newton_mpm_soil=soil)}):
            install_current_source_planning(rt)
            snapshot = rt.get_sand_snapshot()
        self.assertIsNone(snapshot['settled_points'])
        self.assertIsNone(snapshot['spatial_index'])
        self.assertEqual(rt.auto_dig_swept_density_count([0, 7, .2], .4, snapshot=snapshot), 0)
        rt.build_sand_spatial_index.assert_not_called()

    def test_fine_and_physx_keep_the_original_functions_and_counts(self):
        rt, _ = self.runtime(coarse=False)
        original_snapshot = rt.get_sand_snapshot
        install_current_source_planning(rt)
        self.assertIs(rt.get_sand_snapshot(), original_snapshot.return_value)
        self.assertEqual(rt.sand_surface_query_at_xy(1.8, 7), (None, 'outside'))
        self.assertFalse(rt.auto_dig_target_sand_region_check(1.8, 7)[0])
        self.assertEqual(rt.auto_dig_swept_density_count([0, 7, .2], .4), 99)
        rt.sand_settle_status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
