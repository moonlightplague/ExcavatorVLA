import math
import tempfile
import unittest
from pathlib import Path

import excavator_dataset_tools as tools
from excavator_common import vla_observation_contract


class TaskPromptTests(unittest.TestCase):
    def test_current_28d_state_is_not_treated_as_base_pose(self):
        sample = {
            "observation.state": [100.0, 200.0, 1.5] + [0.0] * 25,
            "obs.state_legacy_14d": [0.0, 0.0, 0.0] + [0.0] * 11,
        }
        xy, yaw = tools._state_xy_yaw(
            sample,
            state_names=vla_observation_contract.STATE_NAMES_28D,
        )
        self.assertIsNone(xy)
        self.assertIsNone(yaw)

        episode = {
            "target_xyz": [5.0, 5.0, 0.0],
            "unload_landing_xyz": [0.0, -5.0, 2.0],
            "scene_randomization": {
                "applied": {"robot_body_yaw_deg": 0.0},
            },
        }
        prompt = tools.build_episode_task_text(
            sample,
            {},
            episode_row=episode,
            state_names=vla_observation_contract.STATE_NAMES_28D,
        )
        self.assertIn("front-left", prompt)
        self.assertIn("to the right", prompt)


class StagePolicyTests(unittest.TestCase):
    def test_stale_parent_boundary_after_unload_stays_unload(self):
        phase, corrected = tools.lerobot_export_phase_index(
            {"phase": "lift_carry", "label": "lift_carry_done_boundary"},
            previous_phase_index=9,
        )
        self.assertEqual(phase, 9)
        self.assertTrue(corrected)

    def test_regular_stage_transition_is_unchanged(self):
        phase, corrected = tools.lerobot_export_phase_index(
            {"phase": "pull_exit_cut", "label": "pull_exit_cut_motion"},
            previous_phase_index=4,
        )
        self.assertEqual(phase, 5)
        self.assertFalse(corrected)

    def test_runtime_rejects_stale_parent_boundary_before_write(self):
        runtime_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "excavator_app"
            / "excavator_runtime.py"
        )
        source = runtime_path.read_text(encoding="utf-8-sig")
        self.assertIn("stale_parent_boundary", source)
        self.assertIn("dataset_last_recorded_phase_index", source)


class GoldQualityPolicyTests(unittest.TestCase):
    @staticmethod
    def episode(score=85.0, lift=1000, spill=100):
        return {
            "score": score,
            "lift_bucket_from_pile_particles": lift,
            "final_spill_from_pile_particles": spill,
        }

    @staticmethod
    def trajectory(bucket_deg=-110.0):
        return [
            {
                "obs.q": [
                    0.0,
                    math.radians(10.0),
                    math.radians(-40.0),
                    math.radians(bucket_deg),
                ]
            }
        ]

    def test_gold_accepts_high_quality_episode(self):
        result = tools.lerobot_episode_quality_decision(
            self.episode(),
            self.trajectory(),
            policy="gold-v1",
        )
        self.assertTrue(result["selected"])

    def test_gold_rejects_excessive_spill(self):
        result = tools.lerobot_episode_quality_decision(
            self.episode(spill=301),
            self.trajectory(),
            policy="gold-v1",
        )
        self.assertFalse(result["selected"])
        self.assertEqual(result["reason"], "spill_ratio_above_maximum")

    def test_gold_rejects_bucket_beyond_limit(self):
        result = tools.lerobot_episode_quality_decision(
            self.episode(),
            self.trajectory(bucket_deg=-122.0),
            policy="gold-v1",
        )
        self.assertFalse(result["selected"])
        self.assertEqual(result["reason"], "observed_bucket_outside_physical_limit")

    def test_effort_policy_does_not_invalidate_video_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = tools.lerobot_export_config_for_run(tmp, effort_policy="raw")
            excluded = tools.lerobot_export_config_for_run(tmp, effort_policy="exclude")
        self.assertEqual(raw["media_config_hash"], excluded["media_config_hash"])
        self.assertEqual(raw["export_config"]["effort_dim"], 4)
        self.assertEqual(excluded["export_config"]["effort_dim"], 0)


class DeploymentActionContractTests(unittest.TestCase):
    def test_policy_client_preserves_physical_action_units(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "bridge_test"
            / "smolvla_policy_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        self.assertNotIn("np.tanh(a)", source)
        self.assertIn("ACTION_LIMITS_RAD_S_4D", source)
        self.assertIn("default=0.0", source)


if __name__ == "__main__":
    unittest.main()
