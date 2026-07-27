import ast
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
        self.assertIn("create_retaining_walls=False", source)
        self.assertIn("Retaining-wall generation disabled", source)
        self.assertIn(
            '"create_circular_sand_retaining_wall"',
            source,
        )
        self.assertIn(
            "if sand_enabled and args.sand_wall:",
            source,
        )
        self.assertIn(
            "expected_wall_inner_radius = (",
            source,
        )
        self.assertNotIn("Removed temporary retaining walls", source)
        self.assertIn(
            "pending_pose_restore = _ACTIVE_SAND_POSE_RESTORE",
            source,
        )
        self.assertNotIn("suspend_excavator_collisions_for_sand_init", source)
        self.assertIn("target_min_z", source)

    def test_one_episode_launcher_exposes_sand_and_wall_settings(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher = (
            repo_root
            / "run_isaacsim_ckpt18450_replan1_ensemble5_seed2_one_episode.sh"
        ).read_text(encoding="utf-8")
        expected_settings = (
            "SAND_AMOUNT=1.0",
            "SAND_PARAMETER_MODE=soft_dig",
            "SAND_RADIUS_SCALE=0.2",
            "SAND_WALL_ENABLED=1",
            "SAND_WALL_RADIUS_SCALE=1.5",
            "SAND_SETTLE_FRAMES=60",
        )
        for setting in expected_settings:
            self.assertIn(setting, launcher)
        expected_arguments = (
            '--sand-parameter-mode "$SAND_PARAMETER_MODE"',
            '--sand-radius-scale "$SAND_RADIUS_SCALE"',
            '"$SAND_WALL_OPTION"',
            '--sand-wall-radius-scale "$SAND_WALL_RADIUS_SCALE"',
            '--sand-settle-frames "$SAND_SETTLE_FRAMES"',
        )
        for argument in expected_arguments:
            self.assertIn(argument, launcher)

        simulator = (repo_root / "run_simulation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'profile["sand_amount_multiplier"] = manual_sand_amount',
            simulator,
        )
        self.assertIn(
            'profile["sand_amount_source"] = "manual_cli"',
            simulator,
        )
        self.assertIn(
            "sand_enabled = float(args.sand_amount) > 0.0",
            simulator,
        )
        self.assertNotIn(
            '"dataset-randomized",\n        f"enable_value=',
            simulator,
        )

    def test_sand_runtime_supports_configurable_circular_wall(self):
        runtime = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "excavator_app"
            / "sand_site_runtime.py"
        ).read_text(encoding="utf-8")
        ast.parse(runtime)
        self.assertIn(
            "def create_circular_sand_retaining_wall(",
            runtime,
        )
        self.assertIn(
            "inner_radius = sand_radius * radius_scale",
            runtime,
        )
        self.assertIn(
            'f"{root}/CircularSandRetainingWall"',
            runtime,
        )
        self.assertIn(
            '"create_circular_sand_retaining_wall": (',
            runtime,
        )
        self.assertNotIn("External startup requires soft_dig", runtime)
        self.assertIn("parameter_mode=parameter_mode", runtime)

    def test_run_simulation_syntax_and_video_camera_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "run_simulation.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn(
            'VIDEO_OVERVIEW_CAMERA_PATH = "/World/VideoOverviewCamera"',
            source,
        )
        self.assertIn("def create_video_overview_camera():", source)
        self.assertIn(
            "overview_camera_path = create_video_overview_camera()",
            source,
        )
        self.assertIn(
            "display_camera_path = overview_camera_path",
            source,
        )
        self.assertIn("lock_visible_viewport_to_overview()", source)
        self.assertNotIn("DISPLAY_CAMERA_NAME", source)
        self.assertIn("CAPTURE_STARTUP_WAIT_FRAMES = 60", source)
        self.assertIn(
            "wait_frames=CAPTURE_STARTUP_WAIT_FRAMES",
            source,
        )
        self.assertIn(
            "capture_window.visible = True",
            source,
        )


if __name__ == "__main__":
    unittest.main()
