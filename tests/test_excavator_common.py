import math
import os
import socket
import tempfile
import unittest
from pathlib import Path

from excavator_common import bridge_protocol, geometry, paths, vla_observation_contract

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

    def test_build_state_28d_uses_initial_heading_frame(self):
        state = vla_observation_contract.build_state_28d(
            base_state_14d=[0.0] * 14,
            joint_velocity_4d=[1.0, 2.0, 3.0, 4.0],
            phase="loaded_transit",
            dig_target_world_xyz=[1.0, 0.0, 2.0],
            unload_landing_world_xyz=[0.0, 2.0, 3.0],
            initial_origin_xy=[0.0, 0.0],
            initial_heading_rad=0.5 * 3.141592653589793,
            truck_yaw_rad=3.141592653589793,
            bucket_load_rate=12.0,
        )
        self.assertEqual(len(state), 28)
        self.assertEqual(state[14:18], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(state[18], 8.0)
        self.assertAlmostEqual(state[19], 0.0, places=6)
        self.assertAlmostEqual(state[20], -1.0, places=6)
        self.assertAlmostEqual(state[22], 2.0, places=6)
        self.assertAlmostEqual(state[23], 0.0, places=6)
        self.assertAlmostEqual(state[25], 1.0, places=6)
        self.assertAlmostEqual(state[26], 0.0, places=6)
        self.assertEqual(state[27], 12.0)

    def test_validate_payload_rejects_missing_effort(self):
        with self.assertRaises(vla_observation_contract.ObservationContractError):
            vla_observation_contract.validate_payload(
                {
                    "observation_state_28d": [0.0] * 28,
                    "observation_effort": None,
                }
            )


if __name__ == "__main__":
    unittest.main()
