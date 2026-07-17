import math
import unittest
from pathlib import Path

from excavator_common import deployment_scene_contract as scene_contract


class DeploymentSceneContractTests(unittest.TestCase):
    def test_fixed_profile_uses_training_prims_and_workspace(self):
        profile = scene_contract.validate_fixed_scene_profile()
        self.assertEqual(profile["truck_root_path"], "/World/truck")
        self.assertEqual(
            profile["truck_bed_collision_path"],
            "/World/truck/DumpBedCollision/dump_bed_collision",
        )
        self.assertEqual(profile["sand_center_xy"], (0.0, 6.7))
        self.assertTrue(
            scene_contract.TRAINING_SAND_RADIUS_RANGE_M[0]
            <= profile["sand_world_radius_m"]
            <= scene_contract.TRAINING_SAND_RADIUS_RANGE_M[1]
        )
        self.assertTrue(
            scene_contract.TRAINING_UNLOAD_RADIUS_RANGE_M[0]
            <= profile["unload_world_radius_m"]
            <= scene_contract.TRAINING_UNLOAD_RADIUS_RANGE_M[1]
        )

    def test_verified_training_pose_values_are_finite(self):
        profile = scene_contract.validate_fixed_scene_profile()
        values = (
            list(profile["truck_translation_xyz"])
            + [profile["truck_yaw_deg"]]
            + list(profile["unload_landing_xyz"])
        )
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_wrapped_yaw_error_uses_shortest_difference(self):
        self.assertAlmostEqual(
            scene_contract.wrapped_yaw_error_deg(-179.0, 179.0),
            2.0,
        )
        self.assertAlmostEqual(
            scene_contract.wrapped_yaw_error_deg(266.625, -93.375),
            0.0,
        )

    def test_non_finite_yaw_is_rejected(self):
        with self.assertRaises(ValueError):
            scene_contract.wrapped_yaw_error_deg(float("nan"), 0.0)

    def test_random_scene_sampler_matches_training_workspace(self):
        profile = scene_contract.sample_random_scene_profile(
            seed=1234,
            robot_xy=(-9.2, 6.7),
            truck_center_xy=(-5.0, -9.0),
            truck_dump_center_xy=(-6.8, -7.5),
            truck_yaw_deg=-93.0,
        )
        self.assertEqual(profile["seed"], 1234)
        self.assertTrue(
            scene_contract.TRAINING_SAND_RADIUS_RANGE_M[0]
            <= profile["sand_radius_m"]
            <= scene_contract.TRAINING_SAND_RADIUS_RANGE_M[1]
        )
        self.assertTrue(
            scene_contract.TRAINING_UNLOAD_RADIUS_RANGE_M[0]
            <= profile["unload_radius_m"]
            <= scene_contract.TRAINING_UNLOAD_RADIUS_RANGE_M[1]
        )
        self.assertTrue(
            scene_contract.TRAINING_SAND_AMOUNT_RANGE[0]
            <= profile["sand_amount_multiplier"]
            <= scene_contract.TRAINING_SAND_AMOUNT_RANGE[1]
        )
        self.assertGreaterEqual(
            profile["sand_unload_distance_m"],
            scene_contract.TRAINING_MIN_SAND_UNLOAD_DISTANCE_M,
        )
        self.assertGreaterEqual(
            profile["robot_truck_distance_m"],
            scene_contract.TRAINING_MIN_ROBOT_TRUCK_DISTANCE_M,
        )

    def test_random_scene_seed_is_replayable(self):
        kwargs = {
            "seed": 9876,
            "robot_xy": (-9.2, 6.7),
            "truck_center_xy": (-5.0, -9.0),
            "truck_dump_center_xy": (-6.8, -7.5),
            "truck_yaw_deg": -93.0,
        }
        self.assertEqual(
            scene_contract.sample_random_scene_profile(**kwargs),
            scene_contract.sample_random_scene_profile(**kwargs),
        )
        different = dict(kwargs, seed=9877)
        self.assertNotEqual(
            scene_contract.sample_random_scene_profile(**kwargs),
            scene_contract.sample_random_scene_profile(**different),
        )

    def test_run_simulation_has_no_obsolete_target_helper_or_truck_path(self):
        source = (
            Path(__file__).resolve().parents[1] / "run_simulation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("world_point_to_robot_local_feature(", source)
        self.assertNotIn('GetPrimAtPath("/World/DumpTruck")', source)
        self.assertNotIn("SAND_INITIAL_CENTER", source)
        self.assertIn("prim_local_yaw_z_deg(truck_prim)", source)
        self.assertIn("apply_randomized_dataset_scene()", source)
        self.assertIn("apply_auto_scene_parameters", source)
        self.assertNotIn("apply_fixed_training_truck_pose", source)

    def test_sand_init_elevates_then_restores_excavator(self):
        source = (
            Path(__file__).resolve().parents[1] / "run_simulation.py"
        ).read_text(encoding="utf-8")
        elevate_call = source.index(
            "elevate_excavator_for_sand_init(",
            source.index("actual_sand_center ="),
        )
        settle_loop = source.index(
            "range(max(1, int(args.sand_settle_frames)))",
            elevate_call,
        )
        restore_call = source.index(
            "restore_excavator_pose_after_sand_init(",
            settle_loop,
        )
        self.assertLess(elevate_call, settle_loop)
        self.assertLess(settle_loop, restore_call)
        self.assertIn(
            "pending_pose_restore = _ACTIVE_SAND_POSE_RESTORE",
            source,
        )
        self.assertNotIn("suspend_excavator_collisions_for_sand_init", source)
        self.assertIn("target_min_z", source)


if __name__ == "__main__":
    unittest.main()
