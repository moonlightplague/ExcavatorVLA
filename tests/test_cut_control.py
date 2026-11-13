import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from baseline.cut_control import install_cut_load_goal


class CutLoadGoalTests(unittest.TestCase):
    def runtime(self, mass=5., profile='coarse_mass', coarse=True):
        original = Mock(return_value=(True, 'native_progress'))
        rt = SimpleNamespace(STATE={'repeat_planning_resolution': {'profile':'coarse'}} if coarse else {},
                             sand_particle_mass=lambda: mass, sand_contact_stage_should_advance=original)
        install_cut_load_goal(rt, SimpleNamespace(acceptance_profile=profile, min_delivered_kg=50.))
        return rt, original

    def test_small_progress_cannot_shorten_a_checked_cut_before_loading_goal(self):
        rt, original = self.runtime()
        for stage in ('insert_cut', 'pull_mid_cut'):
            decision, reason = rt.sand_contact_stage_should_advance(stage, {'bucket':4}, q_cmd=[1])
            self.assertFalse(decision)
            self.assertIn('20.000<goal_kg=100.000', reason)
        self.assertEqual(original.call_args.kwargs, {'q_cmd':[1]})

    def test_loaded_cut_preserves_original_decision_and_reason(self):
        rt, original = self.runtime()
        self.assertEqual(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':20}), (True,'native_progress'))
        original.return_value = (False, 'native_not_ready')
        self.assertEqual(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':100}), (False,'native_not_ready'))

    def test_curl_lift_exit_and_return_keep_native_gravity_and_motion_decisions(self):
        rt, _ = self.runtime()
        for stage in ('curl_to_hold_material', 'secure_load', 'pull_exit_cut', 'lift_carry', 'pre_dig'):
            self.assertEqual(rt.sand_contact_stage_should_advance(stage, {'bucket':0}), (True,'native_progress'))

    def test_other_model_profiles_and_policies_keep_native_controls(self):
        for profile, coarse in [('episode', True), ('strict', True), ('coarse_mass', False)]:
            rt, original = self.runtime(profile=profile, coarse=coarse)
            self.assertEqual(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':0}), (True,'native_progress'))

    def test_early_exit_is_resolution_independent_in_kg(self):
        for mass, count in [(1.,100), (5.,20), (20.,5)]:
            rt, _ = self.runtime(mass=mass)
            self.assertTrue(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':count})[0])
            self.assertFalse(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':count-1})[0])

    def test_unavailable_measurements_do_not_authorize_an_early_exit(self):
        for mass in (None, 0., -1., math.nan, math.inf):
            rt, _ = self.runtime(mass=mass)
            self.assertFalse(rt.sand_contact_stage_should_advance('pull_mid_cut', {'bucket':20})[0])


if __name__ == '__main__':
    unittest.main()
