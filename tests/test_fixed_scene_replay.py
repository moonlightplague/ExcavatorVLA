import json
import unittest
from pathlib import Path

from excavator_common import deployment_scene_contract


ROOT = Path(__file__).resolve().parents[1]


class FixedSceneReplayTests(unittest.TestCase):
    def test_deployment_profile_is_replayable(self):
        kwargs = {
            "seed": 2,
            "robot_xy": (0.0, 0.0),
            "truck_center_xy": (-5.0, -9.0),
            "truck_dump_center_xy": (-6.8, -7.5),
            "truck_yaw_deg": -93.0,
        }
        first = deployment_scene_contract.sample_random_scene_profile(**kwargs)
        second = deployment_scene_contract.sample_random_scene_profile(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 2)
        self.assertGreaterEqual(
            first["sand_unload_distance_m"],
            deployment_scene_contract.TRAINING_MIN_SAND_UNLOAD_DISTANCE_M,
        )
        self.assertGreaterEqual(
            first["robot_truck_distance_m"],
            deployment_scene_contract.TRAINING_MIN_ROBOT_TRUCK_DISTANCE_M,
        )

    def test_different_seed_changes_replayed_scene(self):
        kwargs = {
            "robot_xy": (0.0, 0.0),
            "truck_center_xy": (-5.0, -9.0),
            "truck_dump_center_xy": (-6.8, -7.5),
            "truck_yaw_deg": -93.0,
        }
        seed_two = deployment_scene_contract.sample_random_scene_profile(
            seed=2,
            **kwargs,
        )
        seed_three = deployment_scene_contract.sample_random_scene_profile(
            seed=3,
            **kwargs,
        )
        self.assertNotEqual(seed_two, seed_three)

    def test_launcher_maps_replay_cli_before_runtime_import(self):
        source = (ROOT / "run_vla_train_scene.py").read_text(encoding="utf-8")
        env_index = source.index('os.environ["EXCAVATOR_AUTO_SCENE_REPLAY_SEED"]')
        runtime_index = source.index(
            "from excavator_app.bootstrap import run_excavator_with_sand"
        )
        self.assertLess(env_index, runtime_index)
        self.assertIn('"--scene-seed"', source)
        self.assertIn('"--fixed-scene-profile"', source)
        self.assertIn('"--sand-settle-frames"', source)
        self.assertIn(
            'os.environ["EXCAVATOR_SAND_RESET_SETTLE_ENFORCE_MIN"] = "1"',
            source,
        )

    def test_runtime_uses_deployment_contract_for_fixed_replay(self):
        source = (
            ROOT / "scripts" / "excavator_app" / "excavator_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "deployment_scene_contract.sample_random_scene_profile(",
            source,
        )
        self.assertIn(
            '"scene_seed_source": "deployment_contract_fixed_replay"',
            source,
        )
        self.assertIn(
            "truck_base = auto_scene_deployment_baseline()",
            source,
        )
        self.assertIn(
            '"fixed_scene_direct_apply": True',
            source,
        )
        self.assertIn(
            'STATE["auto_scene_fixed_replay_candidate_template"]',
            source,
        )
        self.assertIn(
            'AUTO_SCENE_FIXED_PROFILE.get("initial_pose_deg")',
            source,
        )
        self.assertIn(
            '"ok_authored_collision_aabbs_separated"',
            source,
        )
        self.assertIn(
            "if elapsed >= min_frames and stable_windows >= required_windows:",
            source,
        )

    def test_launcher_does_not_enable_attempt_randomization_for_replay(self):
        source = (ROOT / "run_vla_train_scene.py").read_text(encoding="utf-8")
        for name in (
            "EXCAVATOR_RANDOM_TRUCK",
            "EXCAVATOR_RANDOM_TRUCK_YAW",
            "EXCAVATOR_RANDOM_ROBOT_YAW",
            "EXCAVATOR_RANDOM_SAND_XY",
            "EXCAVATOR_RANDOM_SAND_AMOUNT",
        ):
            self.assertIn(f'os.environ["{name}"] = "0"', source)

    def test_seed_two_exact_profile_contains_final_deployment_pose(self):
        path = ROOT / "configs" / "deployment_scene_seed2.json"
        profile = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(profile["scene_seed"], 2)
        self.assertEqual(len(profile["truck_translation_xyz"]), 3)
        self.assertAlmostEqual(
            profile["truck_yaw_deg"],
            -91.94326,
            places=5,
        )
        self.assertEqual(len(profile["unload_landing_xyz"]), 3)
        self.assertEqual(
            profile["initial_pose_deg"],
            {
                "id": "fixed_scene_target_side_high",
                "swing": 40.0,
                "boom": 46.0,
                "arm": -62.0,
                "bucket": -20.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
