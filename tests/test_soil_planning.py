import math
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from baseline.soil_planning import (scale_planning_counts, configure_coarse_planning,
                                   rank_material_targets, accessible_swept_count,
                                   complete_material_targets, install_material_search_order)


class PlanningMassTests(unittest.TestCase):
    def test_depth_clamping_cannot_consume_retries_on_the_same_target(self):
        rows = [dict(target_xyz=[0, 6, .12], ring_index=1, swept_density_count=40),
                dict(target_xyz=[0, 6, .120000001], ring_index=1, swept_density_count=40),
                dict(target_xyz=[1, 6, .3], ring_index=3, swept_density_count=60)]
        ranked = complete_material_targets(rows, 5.)
        self.assertEqual([r['ring_index'] for r in ranked], [3, 1])
        self.assertEqual(len({tuple(round(v, 4) for v in r['target_xyz']) for r in ranked}), 2)

    def test_native_planner_group_selection_preserves_strict_mass_order_only_for_marked_rows(self):
        path = Path(__file__).resolve().parents[2] / 'ExcavatorVLA-isaac/scripts/excavator_app/auto_dataset_collect.py'
        spec = importlib.util.spec_from_file_location('_material_planner_test', path)
        planner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(planner)
        rows = [dict(target_xyz=[0, 6, .12], ring_index=2, swept_density_count=40),
                dict(target_xyz=[1, 6, .3], ring_index=2, swept_density_count=60)]
        original = planner._group_targets_by_ring(rows)
        install_material_search_order(SimpleNamespace(auto_dataset_collect=planner))
        self.assertEqual(planner._group_targets_by_ring(rows), original)
        ranked = complete_material_targets(rows, 5.)
        groups = planner._group_targets_by_ring(ranked)
        self.assertEqual([len(r) for _, r in groups], [1, 1])
        self.assertEqual([r[0]['swept_density_count'] for _, r in groups], [60, 40])

    def test_runtime_ranks_full_feasible_pool_before_native_per_ring_truncation(self):
        from baseline.tests.test_repeat_runtime import fake_runtime
        from baseline.repeat_runtime import install, RepeatConfig
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rows = [dict(target_xyz=[0, 6, .12], ring_index=0, swept_density_count=20, planned=True),
                    dict(target_xyz=[1, 6, .12], ring_index=3, swept_density_count=80, planned=True),
                    dict(target_xyz=[2, 6, .12], ring_index=4, swept_density_count=100, planned=False)]
            rt.STATE.update(last_auto_dig_target_scores=rows,
                            repeat_planning_resolution={'metric_point_mass_kg': 5.})
            rt.auto_collect_rank_dig_targets = lambda _attempt: rows[:1]
            install(rt, RepeatConfig(scoops=10, scene_profile='broad', acceptance_profile='strict'))
            ranked = rt.auto_collect_rank_dig_targets(1)
            self.assertEqual([r['swept_density_count'] for r in ranked], [80, 20])

    def test_coarse_mass_controls_preserve_original_kg_without_rescaling_twice(self):
        diagnostics = dict(profile='coarse', active=True, error='', points=2000,
                           point_mass_kg=5., metric_point_mass_kg=5.)
        soil_controller = SimpleNamespace(runtime={'PARTICLE_MASS': .3,
            'sand_particle_points': lambda: ([None]*10000, None)})
        soil = SimpleNamespace(current=lambda: soil_controller,
            choose_resolution=Mock(return_value=(.075, .1, 10000)))
        rt = SimpleNamespace(STATE={}, info_print=Mock(),
            get_sand_site_api=lambda: {'mpm_diagnostics': lambda: diagnostics},
            AUTO_PREFLIGHT_MIN_PARTICLES=1000, AUTO_DIG_MIN_LOCAL_PARTICLES=24,
            AUTO_DIG_MIN_SWEPT_PARTICLES=24, CURL_HOLD_MIN_BUCKET_PARTICLES=80,
            SAND_CONTACT_ADVANCE_BUCKET_MIN=900, LIFT_CARRY_HIGH_LOAD_RECOVERY_PARTICLES=1200,
            SECURE_HOLD_MAX_SPILL_PARTICLES=240, QUALITY_MIN_BUCKET_PARTICLES=5,
            repeat_baseline=SimpleNamespace(config=SimpleNamespace(acceptance_profile='coarse_mass'),
                                            floor=SimpleNamespace(clearance=.1)))
        with patch.dict('sys.modules', {'excavator_app': SimpleNamespace(newton_mpm_soil=soil)}):
            configure_coarse_planning(rt)
            configure_coarse_planning(rt)
        self.assertEqual(rt.CURL_HOLD_MIN_BUCKET_PARTICLES, 5)
        self.assertEqual(rt.SAND_CONTACT_ADVANCE_BUCKET_MIN, 54)
        self.assertEqual(rt.LIFT_CARRY_HIGH_LOAD_RECOVERY_PARTICLES, 72)
        self.assertEqual(rt.SECURE_HOLD_MAX_SPILL_PARTICLES, 14)
        self.assertEqual(rt.QUALITY_MIN_BUCKET_PARTICLES, 5)  # legacy verdict remains comparable
        self.assertEqual(rt._repeat_original_control_counts['CURL_HOLD_MIN_BUCKET_PARTICLES'], 80)
        self.assertFalse(rt.STATE['repeat_planning_resolution']['material_controls']['angular_and_collision_limits_changed'])

    def test_ground_sheet_does_not_masquerade_as_available_cut_material(self):
        points = np.array([[.2, 0, .1], [.2, 0, .2], [.6, .2, .4],
                           [1.2, 0, .4], [.3, .5, .3], [.3, 0, .9]])
        before = points.copy()
        self.assertEqual(accessible_swept_count(points, [0, 0, .2], .5, [1, 0]), 2)
        np.testing.assert_array_equal(points, before)
        self.assertEqual(accessible_swept_count(points[:1], [0, 0, .2], .5, [1, 0]), 0)
        self.assertEqual(accessible_swept_count(None, [0, 0, .2], .5, [1, 0]), 0)

    def test_material_priority_prefers_a_full_outer_cut_to_a_depleted_inner_cut(self):
        rows = [dict(ring_index=0, swept_density_count=6, score=90, plan_surface_z=.8),
                dict(ring_index=3, swept_density_count=42, score=70, plan_surface_z=.6),
                dict(ring_index=2, swept_density_count=18, score=80, plan_surface_z=.7)]
        ranked = rank_material_targets(rows, 5.)
        self.assertEqual([r['ring_index'] for r in ranked], [3, 2, 0])
        self.assertEqual({id(r) for r in ranked}, {id(r) for r in rows})
        self.assertEqual(ranked[0]['repeat_estimated_swept_mass_kg'], 210.)
        self.assertEqual(ranked[-1]['swept_density_count'], 6)

    def test_coarse_search_requires_at_least_the_reference_mass(self):
        original = {"preflight": 1000, "local": 24, "swept": 24}
        reference, coarse = .91, 5.35
        scaled = scale_planning_counts(original, reference, coarse)
        self.assertEqual(scaled, {"preflight": 171, "local": 5, "swept": 5})
        for name, count in scaled.items():
            self.assertGreaterEqual(count * coarse, original[name] * reference)
            self.assertLess((count - 1) * coarse, original[name] * reference)
        self.assertEqual(original, {"preflight": 1000, "local": 24, "swept": 24})

    def test_equal_resolution_and_invalid_masses(self):
        self.assertEqual(scale_planning_counts({"local": 24}, .91, .91), {"local": 24})
        for value in (0., -1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                scale_planning_counts({"local": 24}, value, 1.)

    def test_live_configuration_is_not_scaled_again_on_repeated_queries(self):
        diagnostics = dict(profile="coarse", active=True, error="", points=2000,
                           point_mass_kg=5., metric_point_mass_kg=5.)
        controller = SimpleNamespace(runtime={"sand_particle_points": lambda: ([None]*10000, None)})
        soil = SimpleNamespace(current=lambda: controller,
                               choose_resolution=Mock(return_value=(.075, .1, 10000)))
        rt = SimpleNamespace(STATE={}, info_print=Mock(),
            get_sand_site_api=lambda: {"mpm_diagnostics": lambda: diagnostics},
            AUTO_PREFLIGHT_MIN_PARTICLES=1000, AUTO_DIG_MIN_LOCAL_PARTICLES=24,
            AUTO_DIG_MIN_SWEPT_PARTICLES=24, MULTI_SCOOP_MIN_SCORE=70.)
        with patch.dict("sys.modules", {"excavator_app": SimpleNamespace(newton_mpm_soil=soil)}):
            configure_coarse_planning(rt)
            configure_coarse_planning(rt)
        self.assertEqual(rt.AUTO_PREFLIGHT_MIN_PARTICLES, 200)
        self.assertEqual(rt.AUTO_DIG_MIN_SWEPT_PARTICLES, 5)
        self.assertEqual(rt.MULTI_SCOOP_MIN_SCORE, 70.)
        soil.choose_resolution.assert_called_once()
        self.assertFalse(rt.STATE["repeat_planning_resolution"]["episode_quality_thresholds_changed"])

    def test_fine_and_physx_do_not_change_planning_counts(self):
        for state in ({}, {"profile": "fine"}):
            rt = SimpleNamespace(get_sand_site_api=lambda: {"mpm_diagnostics": lambda: state},
                                 AUTO_PREFLIGHT_MIN_PARTICLES=1000)
            configure_coarse_planning(rt)
            self.assertEqual(rt.AUTO_PREFLIGHT_MIN_PARTICLES, 1000)


if __name__ == "__main__":
    unittest.main()
