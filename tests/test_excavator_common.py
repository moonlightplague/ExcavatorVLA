import math
import os
import socket
import tempfile
import unittest
from pathlib import Path

import excavator_dataset_tools
from excavator_common import bridge_protocol, geometry, paths, planning, scene_randomization, vla_observation_contract
from scripts.excavator_app import auto_dataset_collect, ik_calculation

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class BridgeProtocolTests(unittest.TestCase):
    def test_json_round_trip(self):
        a, b = socket.socketpair()
        try:
            bridge_protocol.write_json(a, {"ticks": 4, "joint_positions": [1, 2, 3]})
            self.assertEqual(
                bridge_protocol.read_json(b),
                {"ticks": 4, "joint_positions": [1, 2, 3]},
            )
        finally:
            a.close()
            b.close()

    def test_rgb_round_trip(self):
        if np is None:
            self.skipTest("numpy is required for bridge RGB protocol tests.")
        rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
        payload = bridge_protocol.encode_rgb_payload(rgb, np_module=np, camera_path="/Camera")
        decoded = bridge_protocol.decode_rgb_payload(payload, np_module=np)
        self.assertEqual(payload["camera_path"], "/Camera")
        np.testing.assert_array_equal(decoded, rgb)

    def test_validate_bridge_command_rejects_mixed_actions(self):
        with self.assertRaises(ValueError):
            bridge_protocol.validate_bridge_command(
                {"joint_positions": [0.0], "joint_velocities": [0.0]}
            )


class PathTests(unittest.TestCase):
    def test_project_root_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "excavator_app").mkdir(parents=True)
            (root / "excavator_config.json").write_text("{}", encoding="utf-8")
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "scripts" / "excavator_app" / "bootstrap.py").write_text("", encoding="utf-8")
            old = os.environ.get("EXCAVATOR_PROJECT_ROOT")
            os.environ["EXCAVATOR_PROJECT_ROOT"] = str(root)
            try:
                self.assertEqual(paths.find_project_root(), str(root.resolve()))
            finally:
                if old is None:
                    os.environ.pop("EXCAVATOR_PROJECT_ROOT", None)
                else:
                    os.environ["EXCAVATOR_PROJECT_ROOT"] = old

    def test_load_config_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "excavator_config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                paths.load_config(root)


class GeometryTests(unittest.TestCase):
    def test_convex_hull_and_centroid(self):
        hull = geometry.convex_hull_xy([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(abs(geometry.polygon_signed_area(hull)), 1.0)
        cx, cy = geometry.polygon_centroid_xy(hull)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertTrue(geometry.point_in_polygon_xy(0.5, 0.5, hull))
        self.assertFalse(geometry.point_in_polygon_xy(2.0, 2.0, hull))


class PlanningTests(unittest.TestCase):
    def test_prunes_redundant_clearance_waypoint(self):
        valid_routes = {("high",)}
        route, report = planning.greedy_prune_waypoints(
            ["tuck", "high"],
            lambda candidate: tuple(candidate) in valid_routes,
            minimum_count=1,
        )
        self.assertEqual(route, ["high"])
        self.assertEqual(report["removed_count"], 1)

    def test_keeps_both_required_clearance_waypoints(self):
        route, report = planning.greedy_prune_waypoints(
            ["tuck", "swing"],
            lambda _candidate: False,
            minimum_count=1,
        )
        self.assertEqual(route, ["tuck", "swing"])
        self.assertEqual(report["removed_count"], 0)

    def test_route_efficiency_prefers_faster_coordinated_motion(self):
        staged = planning.route_efficiency_score(4.2, 190.0, 2)
        coordinated = planning.route_efficiency_score(2.8, 205.0, 1)
        self.assertLess(coordinated, staged)

    def test_route_efficiency_uses_angle_and_stops_as_secondary_costs(self):
        shorter = planning.route_efficiency_score(3.0, 120.0, 1)
        longer = planning.route_efficiency_score(3.0, 180.0, 2)
        self.assertLess(shorter, longer)

    def test_height_floor_accepts_small_negative_margin_with_warning(self):
        report = planning.height_floor_report(0.038, -0.0013, tolerance_m=0.02)
        self.assertTrue(report["ok"])
        self.assertTrue(report["within_tolerance"])
        self.assertEqual(report["warning"], "loaded_carry_height_near_floor")


class DigDepthPlanningTests(unittest.TestCase):
    def test_load_target_z_accounts_for_lower_bucket_tip(self):
        if np is None:
            self.skipTest("numpy is required for dig depth planning tests.")
        model = {
            "effectors": {
                "load": {"lengths": [1.0, 1.0, 0.3], "offsets": [0.0, 0.0, 0.0]},
                "tip": {"lengths": [1.0, 1.0, 0.7], "offsets": [0.0, 0.0, 0.0]},
                "mid": {"lengths": [1.0, 1.0, 0.5], "offsets": [0.0, 0.0, 0.0]},
                "pour": {"lengths": [1.0, 1.0, 0.6], "offsets": [0.0, 0.0, 0.0]},
            }
        }
        min_target_z, detail = ik_calculation.floor_safe_effector_target_z(
            model,
            end_effector="load",
            world_angle_rad=-0.5 * math.pi,
            point_min_z={
                "tip": -0.28,
                "load": -0.10,
                "mid": -0.40,
                "pour": -0.40,
                "bucket_joint": -0.40,
            },
            margin=0.015,
        )
        self.assertAlmostEqual(min_target_z, 0.135, places=6)
        self.assertEqual(detail["limiting_point"], "tip")
        self.assertGreater(min_target_z, 0.035)

    def test_height_floor_rejects_drop_beyond_tolerance(self):
        report = planning.height_floor_report(0.038, -0.021, tolerance_m=0.02)
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "loaded_carry_height_drop")

    def test_loaded_lift_recovery_treats_score_drop_as_warning(self):
        report = planning.loaded_lift_recovery_report(
            loaded_count=5397,
            minimum_loaded_count=2000,
            transitional_hold=True,
            loaded_carry_joint_ok=True,
            dump_branch=False,
            carry_score_before=-33.22,
            carry_score_after=-36.76,
            score_warning_threshold=3.0,
        )
        self.assertTrue(report["allowed"])
        self.assertTrue(report["score_warning"])
        self.assertAlmostEqual(report["score_drop"], 3.54, places=6)

    def test_loaded_lift_recovery_still_rejects_dump_branch(self):
        report = planning.loaded_lift_recovery_report(
            loaded_count=5397,
            minimum_loaded_count=2000,
            transitional_hold=True,
            loaded_carry_joint_ok=True,
            dump_branch=True,
            carry_score_before=-33.22,
            carry_score_after=-36.76,
        )
        self.assertFalse(report["allowed"])


class SceneRandomizationTests(unittest.TestCase):
    def test_area_uniform_radius_is_uniform_in_squared_radius(self):
        radius_range = (5.0, 9.0)
        self.assertAlmostEqual(
            scene_randomization.area_uniform_radius(0.0, radius_range),
            5.0,
        )
        midpoint = scene_randomization.area_uniform_radius(0.5, radius_range)
        self.assertAlmostEqual(midpoint * midpoint, 0.5 * (5.0**2 + 9.0**2))
        self.assertAlmostEqual(
            scene_randomization.area_uniform_radius(1.0, radius_range),
            9.0,
        )

    def test_scene_seed_varies_by_run_worker_and_attempt(self):
        seed = scene_randomization.stable_scene_seed("run-a", 1, worker_identity="worker-0")
        self.assertEqual(
            seed,
            scene_randomization.stable_scene_seed("run-a", 1, worker_identity="worker-0"),
        )
        self.assertNotEqual(
            seed,
            scene_randomization.stable_scene_seed("run-b", 1, worker_identity="worker-0"),
        )
        self.assertNotEqual(
            seed,
            scene_randomization.stable_scene_seed("run-a", 1, worker_identity="worker-1"),
        )
        self.assertNotEqual(
            seed,
            scene_randomization.stable_scene_seed("run-a", 2, worker_identity="worker-0"),
        )


class AutoDatasetReliabilityTests(unittest.TestCase):
    class Runtime:
        def __init__(self, last_result):
            self.STATE = {"auto_collect_last_result": last_result}

    def test_transient_scene_overlap_does_not_stop_worker(self):
        rt = self.Runtime("prepare_failed/initial_robot_truck_overlap")
        self.assertEqual(auto_dataset_collect.update_prepare_failure_streak(rt, 2), 0)

    def test_transient_pose_failure_does_not_stop_worker(self):
        rt = self.Runtime("prepare_failed/home_direct_failed:bucket lag")
        self.assertEqual(auto_dataset_collect.update_prepare_failure_streak(rt, 2), 0)

    def test_persistent_runtime_failure_still_counts(self):
        rt = self.Runtime("prepare_failed/action_channel_not_ready:physics_view=False")
        self.assertEqual(auto_dataset_collect.update_prepare_failure_streak(rt, 2), 3)

    def test_success_resets_prepare_failure_streak(self):
        rt = self.Runtime("episode_trainable")
        self.assertEqual(auto_dataset_collect.update_prepare_failure_streak(rt, 2), 0)


class VLAObservationContractTests(unittest.TestCase):
    def test_phase_mapping_includes_loaded_transit(self):
        self.assertEqual(vla_observation_contract.canonical_phase_index("pre_dig"), 0)
        self.assertEqual(vla_observation_contract.canonical_phase_index("loaded_transit"), 8)
        self.assertEqual(vla_observation_contract.canonical_phase_index("clearance_route_post_2"), 8)
        self.assertEqual(vla_observation_contract.canonical_phase_index("unload_to_bin"), 9)

    def test_joint_velocity_uses_shortest_swing_delta(self):
        velocity = vla_observation_contract.joint_velocity_from_samples(
            [-math.radians(179.0), 0.3, -0.4, 0.5],
            [math.radians(179.0), 0.2, -0.2, 0.1],
            0.5,
        )
        self.assertAlmostEqual(velocity[0], math.radians(4.0), places=6)
        self.assertAlmostEqual(velocity[1], 0.2, places=6)
        self.assertAlmostEqual(velocity[2], -0.4, places=6)
        self.assertAlmostEqual(velocity[3], 0.8, places=6)

    def test_build_state_28d_uses_current_upper_heading_frame(self):
        state = vla_observation_contract.build_state_28d(
            joint_positions_4d=[0.1, 0.2, 0.3, 0.4],
            joint_velocity_4d=[1.0, 2.0, 3.0, 4.0],
            joint_tracking_error_4d=[0.01, 0.02, 0.03, 0.04],
            previous_action_4d=[0.5, 0.6, 0.7, 0.8],
            bucket_tip_world_xyz=[0.0, 0.0, 1.0],
            bucket_load_world_xyz=[0.0, 0.0, 1.0],
            bucket_pour_world_xyz=[0.0, 1.0, 2.0],
            dig_target_world_xyz=[1.0, 0.0, 2.0],
            unload_landing_world_xyz=[0.0, 2.0, 3.0],
            upper_heading_rad=0.5 * 3.141592653589793,
            truck_yaw_rad=3.141592653589793,
            bucket_fill_fraction_value=0.5,
            bucket_fill_rate_fraction_per_s=0.25,
        )
        self.assertEqual(len(state), 28)
        self.assertEqual(state[4:8], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(state[12:16], [0.5, 0.6, 0.7, 0.8])
        self.assertAlmostEqual(state[18], 0.0, places=6)
        self.assertAlmostEqual(state[19], -1.0, places=6)
        self.assertAlmostEqual(state[20], 1.0, places=6)
        self.assertAlmostEqual(state[21], 2.0, places=6)
        self.assertAlmostEqual(state[22], 0.0, places=6)
        self.assertAlmostEqual(state[23], 2.0, places=6)
        self.assertEqual(state[26:], [0.5, 0.25])

    def test_validate_payload_rejects_missing_effort(self):
        with self.assertRaises(vla_observation_contract.ObservationContractError):
            vla_observation_contract.validate_payload(
                {
                    "observation_state_28d": [0.0] * 28,
                    "observation_effort": None,
                }
            )


class DatasetExportPerformanceTests(unittest.TestCase):
    def test_video_parallel_config_honors_safe_overrides(self):
        names = [
            excavator_dataset_tools.LEROBOT_VIDEO_WORKERS_ENV,
            excavator_dataset_tools.LEROBOT_VIDEO_ENCODER_THREADS_ENV,
            excavator_dataset_tools.LEROBOT_VIDEO_PRESET_ENV,
        ]
        previous = {name: os.environ.get(name) for name in names}
        try:
            os.environ[excavator_dataset_tools.LEROBOT_VIDEO_WORKERS_ENV] = "3"
            os.environ[excavator_dataset_tools.LEROBOT_VIDEO_ENCODER_THREADS_ENV] = "2"
            os.environ[excavator_dataset_tools.LEROBOT_VIDEO_PRESET_ENV] = "veryfast"
            config = excavator_dataset_tools.lerobot_video_parallel_config(total_jobs=10)
            self.assertEqual(config["workers"], 3)
            self.assertEqual(config["encoder_threads"], 2)
            self.assertEqual(config["encoder_preset"], "veryfast")
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_safe_reuse_file_preserves_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            target = Path(tmp) / "staging" / "target.mp4"
            source.write_bytes(b"media")
            ok, method = excavator_dataset_tools.safe_reuse_file(str(source), str(target))
            self.assertTrue(ok)
            self.assertIn(method, {"hardlink", "copy"})
            self.assertEqual(target.read_bytes(), b"media")


class DatasetExportTimePolicyTests(unittest.TestCase):
    def test_hold_real_transition_does_not_become_motion_action(self):
        samples = [
            {
                "i": 0,
                "t": 0.0,
                "phase": "unload_to_bin",
                "label": "dump_release",
                "obs.q": [0.0, 0.0, 0.0, 0.0],
                "obs.q_cmd": [0.0, 0.0, 0.0, 1.0],
                "action": [0.0] * 4,
            },
            {
                "i": 1,
                "t": 0.1,
                "phase": "unload_to_bin",
                "label": "after_dump_direct",
                "obs.q": [0.0, 0.0, 0.0, 0.6],
                "obs.q_cmd": [0.0, 0.0, 0.0, 0.6],
                "action": [0.0] * 4,
            },
            {
                "i": 2,
                "t": 0.2,
                "phase": "unload_to_bin",
                "label": "after_dump_settle",
                "obs.q": [0.0, 0.0, 0.0, 0.6],
                "obs.q_cmd": [0.0, 0.0, 0.0, 0.6],
                "action": [0.0] * 4,
            },
        ]

        result = excavator_dataset_tools.apply_export_time_policy_to_trajectory(samples, base_fps=10.0)

        self.assertEqual(result["samples"][0]["action"], [0.0] * 4)
        self.assertEqual(result["samples"][1]["action"], [0.0] * 4)
        self.assertEqual(result["samples"][1]["action.semantic"], "hold")
        self.assertEqual(result["action_semantic_audit"]["corrected_hold_transitions"], 1)
        self.assertEqual(result["action_semantic_audit"]["hard_limit_violations"], 0)

    def test_trajectory_transition_preserves_command_velocity(self):
        samples = []
        for index, bucket in enumerate([0.0, 0.1, 0.2]):
            samples.append(
                {
                    "i": index,
                    "t": index * 0.1,
                    "phase": "pre_dig",
                    "label": "pre_dig",
                    "control.intent": "trajectory",
                    "obs.q": [0.0, 0.0, 0.0, bucket],
                    "obs.q_cmd": [0.0, 0.0, 0.0, bucket],
                    "action": [0.0] * 4,
                }
            )

        result = excavator_dataset_tools.apply_export_time_policy_to_trajectory(samples, base_fps=10.0)

        self.assertAlmostEqual(result["samples"][0]["action"][3], 1.0, places=6)
        self.assertAlmostEqual(result["samples"][1]["action"][3], 1.0, places=6)
        self.assertEqual(result["action_semantic_audit"]["corrected_hold_transitions"], 0)

    def test_discontinuous_position_setpoint_uses_executed_transition_velocity(self):
        samples = [
            {
                "i": 0,
                "t": 0.0,
                "phase": "unload_to_bin",
                "label": "unload_to_bin",
                "obs.q": [0.0, 0.0, 0.0, 0.0],
                "obs.q_cmd": [0.0, 0.0, 0.0, -2.0],
                "action": [0.0] * 4,
            },
            {
                "i": 1,
                "t": 0.1,
                "phase": "unload_to_bin",
                "label": "unload_to_bin",
                "obs.q": [0.0, 0.0, 0.0, 0.1],
                "obs.q_cmd": [0.0, 0.0, 0.0, 0.5],
                "action": [0.0] * 4,
            },
        ]

        result = excavator_dataset_tools.apply_export_time_policy_to_trajectory(samples, base_fps=10.0)

        self.assertAlmostEqual(result["samples"][0]["action"][3], 1.0, places=6)
        self.assertEqual(result["samples"][0]["action.semantic"], "setpoint_fallback")
        self.assertEqual(result["action_semantic_audit"]["setpoint_fallback_transitions"], 1)
        self.assertEqual(result["action_semantic_audit"]["hard_limit_violations"], 0)

    def test_legacy_migration_transform_retimes_episode_to_uniform_10hz(self):
        source_fps = 6.0
        samples = []
        for index in range(7):
            t = index / source_fps
            swing_unwrapped = math.radians(170.0 + 20.0 * t)
            swing = (swing_unwrapped + math.pi) % (2.0 * math.pi) - math.pi
            q = [swing, 0.1 * t, -0.2 * t, 0.3 * t]
            q_cmd = [swing, 0.1 * t + 0.01, -0.2 * t, 0.3 * t]
            samples.append(
                {
                    "i": index,
                    "t": t,
                    "timestamp": 1000.0 + t,
                    "observation.timestamp": 1000.0 + t,
                    "action.timestamp": 1000.0 + t,
                    "timestamp.simulation": 50.0 + t,
                    "phase": "pre_dig" if index < 3 else "approach_contact",
                    "obs.q": q,
                    "obs.q_cmd": q_cmd,
                    "obs.state_legacy_14d": q + [100.0 * t] + [0.0] * 9,
                    "observation.effort": [1.0 + t, 2.0 + t, 3.0 + t, 4.0 + t],
                    "observation.images.0": f"images/0/{index:06d}.ppm",
                    "observation.images.1": f"images/1/{index:06d}.ppm",
                    "observation.images.2": f"images/2/{index:06d}.ppm",
                    "action": [0.0] * 4,
                    "sand": {"bucket_from_pile": 100.0 * t},
                }
            )

        result = excavator_dataset_tools.apply_export_time_policy_to_trajectory(
            samples,
            base_fps=10.0,
        )
        transformed = result["samples"]

        self.assertEqual(result["time_mode"], "uniform_fps")
        self.assertEqual(len(transformed), 7)
        self.assertAlmostEqual(result["raw_duration_s"], 1.0, places=6)
        self.assertAlmostEqual(result["duration_s"], 0.6, places=6)
        self.assertAlmostEqual(result["media_duration_s"], 0.7, places=6)
        self.assertAlmostEqual(result["source_time_scale"], 1.0 / 0.6, places=6)
        self.assertEqual(
            [round(row["t"], 6) for row in transformed],
            [index / 10.0 for index in range(7)],
        )
        self.assertAlmostEqual(
            transformed[3]["obs.dq"][0],
            math.radians(20.0) * (10.0 / source_fps),
            places=5,
        )
        self.assertAlmostEqual(transformed[6]["timestamp"], 1000.6, places=6)
        self.assertAlmostEqual(transformed[6]["observation.timestamp"], 1000.6, places=6)
        self.assertAlmostEqual(transformed[6]["action.timestamp"], 1000.6, places=6)
        self.assertAlmostEqual(transformed[6]["timestamp.simulation"], 50.6, places=6)
        self.assertEqual(transformed[6]["timestamp.source"], "export_uniform_fps")
        self.assertEqual(transformed[6]["observation.effort"], [2.0, 3.0, 4.0, 5.0])
        self.assertTrue(all(row["export_time_policy"]["version"] == 1 for row in transformed))


if __name__ == "__main__":
    unittest.main()
