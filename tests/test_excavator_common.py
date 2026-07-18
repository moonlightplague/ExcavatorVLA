import math
import os
import socket
import tempfile
import unittest
from pathlib import Path

import excavator_dataset_tools
from excavator_common import bridge_protocol, geometry, paths, planning, scene_randomization, vla_observation_contract

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


if __name__ == "__main__":
    unittest.main()
