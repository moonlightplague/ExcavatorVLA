from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from baseline.repeat_runtime import RepeatConfig
from baseline.tuning import (episode_quality, install_quality_policy, configure_seed_sequence,
                             configure_workspace, install_source_identity, delivery_mass_quality)


class QualityPolicyTests(unittest.TestCase):
    def test_each_dig_must_deliver_fifty_kg_without_averaging_or_rounding_up(self):
        good = [dict(payload_particles=10)] * 10
        self.assertTrue(delivery_mass_quality(good, 5., 50., 10)['success'])
        bad = good[:9] + [dict(payload_particles=9)]
        report = delivery_mass_quality(bad, 5., 50., 10)
        self.assertFalse(report['success'])
        self.assertIn('low_delivered_mass_kg:10:45.000<50.000', report['reason'])
        self.assertFalse(delivery_mass_quality(good, 4.9999, 50., 10)['success'])
        for mass in (None, float('nan'), 0.):
            self.assertFalse(delivery_mass_quality(good, mass, 50., 10)['success'])
        self.assertFalse(delivery_mass_quality(good[:9], 5., 50., 10)['success'])

    def runtime(self, reason, **values):
        report = dict(success=False, reason=reason, scoop_index=0, score=54., spill_ratio=.819,
                      lift_bucket_from_pile_particles=100, spill_from_lift_particles=82)
        report.update(values)
        def original(*_args):
            rt.STATE["dataset_scoop_results"].append(report)
            rt.dataset_record_event("scoop_end", "strict decision")
            return report
        rt = SimpleNamespace(STATE={"dataset_scoop_results": []}, multi_scoop_cycle_report=original,
                             info_print=Mock(), dataset_record_event=Mock())
        return rt

    def test_efficiency_warning_does_not_erase_original_rejection(self):
        rt = self.runtime("high_spill_ratio:0.819;score_below_minimum:54.0")
        install_quality_policy(rt, RepeatConfig(acceptance_profile="episode"))
        result = rt.multi_scoop_cycle_report(True)
        self.assertTrue(result["success"])
        self.assertFalse(result["strict_success"])
        self.assertEqual(len(result["quality_warnings"]), 2)
        self.assertEqual(rt.STATE["dataset_scoops_completed"], 1)
        events = rt.dataset_record_event.call_args_list
        self.assertEqual([call.args[0] for call in events], ["repeat_strict_scoop_end", "scoop_end"])
        self.assertTrue(events[-1].kwargs['data']['success'])
        self.assertFalse(events[-1].kwargs['data']['strict_success'])

    def test_empty_deliveries_freezes_and_execution_failures_still_fail(self):
        for reason in ("low_bin_gain:0", "low_bucket_particles:0", "freeze_detected:1", "execution_failed/collision"):
            with self.subTest(reason=reason):
                rt = self.runtime(reason + ";high_spill_ratio:0.819")
                install_quality_policy(rt, RepeatConfig(acceptance_profile="episode"))
                result = rt.multi_scoop_cycle_report(True)
                self.assertFalse(result["success"])
                self.assertIn(reason, result["reason"])
                self.assertEqual(rt.STATE["dataset_scoops_completed"], 0)
        rt = self.runtime("high_spill_ratio:0.819")
        install_quality_policy(rt, RepeatConfig(acceptance_profile="episode"))
        self.assertFalse(rt.multi_scoop_cycle_report(False)["success"])

    def test_episode_gate_requires_ten_cycles_and_bounds_total_losses(self):
        cfg = RepeatConfig(scoops=10, acceptance_profile="episode")
        good = dict(success=True, strict_success=True, score=90.,
                    lift_bucket_from_pile_particles=100, spill_from_lift_particles=10)
        poor = dict(success=True, strict_success=False, score=54.,
                    lift_bucket_from_pile_particles=100, spill_from_lift_particles=82)
        result = episode_quality([good] * 8 + [poor] * 2, cfg)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["weighted_spill_ratio"], .244)
        self.assertEqual(result["strict_scoops"], 8)
        self.assertFalse(episode_quality([good] * 9, cfg)["success"])
        self.assertFalse(episode_quality([poor] * 10, cfg)["success"])
        self.assertFalse(episode_quality([dict(good, score=float('nan'))] * 10, cfg)["success"])

    def test_new_delivery_is_separate_from_loss_of_an_older_truck_load(self):
        rt = self.runtime("low_bin_gain:0;high_spill_ratio:1.000;score_below_minimum:45.0",
                          score=45., spill_ratio=1., lift_bucket_from_pile_particles=125,
                          bin_gain_from_pile_particles=0)
        measured = dict(scooped_particles=127, payload_particles=95, spillage_particles=32)
        rt.repeat_baseline = SimpleNamespace(cycles=[dict(scoop_index=0, metrics=measured)])
        config = RepeatConfig(scoops=10, acceptance_profile='episode')
        install_quality_policy(rt, config)
        report = rt.multi_scoop_cycle_report(True)
        self.assertTrue(report['success'])
        self.assertFalse(report['strict_success'])
        self.assertEqual(report['source_delivered_particles'], 95)
        self.assertGreater(report['source_score'], 80)
        self.assertTrue(episode_quality([report]*10, config, {'spillage_fraction': .34})['success'])
        self.assertFalse(episode_quality([report]*10, config, {'spillage_fraction': .6})['success'])
        self.assertFalse(episode_quality([report]*10, config)['success'])

    def test_zero_new_delivery_cannot_pass_on_old_soil_settling_back_into_truck(self):
        rt = self.runtime('', success=True, score=90., spill_ratio=.1,
                          lift_bucket_from_pile_particles=50, bin_gain_from_pile_particles=30)
        rt.repeat_baseline = SimpleNamespace(cycles=[dict(scoop_index=0,
            metrics=dict(scooped_particles=50, payload_particles=0, spillage_particles=50))])
        install_quality_policy(rt, RepeatConfig(acceptance_profile='episode'))
        report = rt.multi_scoop_cycle_report(True)
        self.assertFalse(report['success'])
        self.assertIn('low_source_delivery:0', report['reason'])

    def test_failed_attempts_consume_reproducible_distinct_seeds(self):
        rt = SimpleNamespace(auto_scene_seed_context=lambda attempt: {"scene_seed": 99, "scene_seed_worker": "w0"})
        configure_seed_sequence(rt, 700)
        self.assertEqual([rt.auto_scene_seed_context(i)["scene_seed"] for i in (1, 2, 3)], [700, 701, 702])
        self.assertEqual(rt.auto_scene_seed_context(1)["scene_seed_worker"], "w0")

    def test_workspace_is_centered_on_measured_swing_in_offset_scene(self):
        center = np.array([-6.95066, 2.23256])
        rt = SimpleNamespace(get_swing_xy_center=lambda: center,
                             auto_scene_xy_radius=lambda xy: float(np.linalg.norm(xy)),
                             auto_scene_xy_angle_deg=lambda xy: float(np.degrees(np.arctan2(xy[1], xy[0]))))
        configure_workspace(rt, "reachable")
        rng = np.random.default_rng(42)
        for _ in range(50):
            xy, world_radius, world_angle = rt.auto_scene_sample_truck_unload_xy(rng, None, None)
            self.assertTrue(6.7 - 1e-5 <= np.linalg.norm(xy-center) <= 7.0 + 1e-5)
            self.assertAlmostEqual(world_radius, np.linalg.norm(xy), places=5)
            self.assertAlmostEqual(world_angle, np.degrees(np.arctan2(xy[1], xy[0])), places=4)
        # A cell close to the world origin can still be in the arm's workspace.
        self.assertTrue(rt.unload_landing_radius_status(center+[6.8,0])['ok'])
        self.assertFalse(rt.unload_landing_radius_status(center+[15.,0])['ok'])

    def test_avalanched_source_keeps_its_identity_but_initial_truck_soil_is_excluded(self):
        rt = SimpleNamespace(capture_initial_pile_particle_ids=Mock(), STATE={},
            os=SimpleNamespace(environ={"EXCAVATOR_SOIL_BACKEND": "newton_mpm"}),
            sand_particle_positions=lambda: np.array([[0.,0.,0.], [20.,20.,0.], [2.,2.,0.]]),
            sand_region_masks=lambda points: (np.array([True,False,False]),
                                              np.array([False,False,False]), np.array([False,False,True])))
        install_source_identity(rt)
        self.assertEqual(rt.capture_initial_pile_particle_ids(), {0,1})
        np.testing.assert_array_equal(rt.STATE['dataset_initial_pile_particle_mask'], [True, True, False])


if __name__ == "__main__":
    unittest.main()
