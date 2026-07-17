import math
import unittest

from excavator_common import vla_observation_contract as contract


class VLAObservationContractTests(unittest.TestCase):
    def test_phase_mapping(self):
        self.assertEqual(contract.canonical_phase_index("pre_dig"), 0)
        self.assertEqual(contract.canonical_phase_index("curl_to_hold_material"), 4)
        self.assertEqual(contract.canonical_phase_index("pull_exit_cut"), 5)
        self.assertEqual(contract.canonical_phase_index("loaded_transit"), 8)
        self.assertEqual(contract.canonical_phase_index("unload_to_bin"), 9)

    def test_velocity_uses_shortest_swing_delta(self):
        velocity = contract.joint_velocity_from_samples(
            [-math.radians(179.0), 0.3, -0.4, 0.5],
            [math.radians(179.0), 0.2, -0.2, 0.1],
            0.5,
        )
        self.assertAlmostEqual(velocity[0], math.radians(4.0), places=6)
        self.assertAlmostEqual(velocity[1], 0.2, places=6)

    def test_builds_exact_28d_shape(self):
        state = contract.build_state_28d(
            base_state_14d=[0.0] * 14,
            joint_velocity_4d=[0.1, 0.2, 0.3, 0.4],
            phase="loaded_transit",
            dig_target_world_xyz=[1.0, 0.0, 0.5],
            unload_landing_world_xyz=[0.0, 2.0, 4.0],
            initial_origin_xy=[0.0, 0.0],
            initial_heading_rad=math.pi / 2.0,
            truck_yaw_rad=math.pi,
            bucket_load_rate=12.0,
        )
        self.assertEqual(len(state), 28)
        self.assertEqual(state[18], 8.0)
        self.assertEqual(state[27], 12.0)


if __name__ == "__main__":
    unittest.main()
