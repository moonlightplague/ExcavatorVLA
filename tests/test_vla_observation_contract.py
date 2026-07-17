import math
import unittest

from excavator_common import vla_observation_contract as contract


class VLAObservationContractTests(unittest.TestCase):
    def test_point_transform_uses_initial_heading_frame(self):
        point = contract.point_in_initial_heading_frame(
            [2.0, 3.0, 0.5],
            [1.0, 1.0],
            math.pi / 2.0,
        )
        self.assertAlmostEqual(point[0], 2.0, places=6)
        self.assertAlmostEqual(point[1], -1.0, places=6)
        self.assertAlmostEqual(point[2], 0.5, places=6)

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

    def test_bucket_load_source_tracking_matches_dataset_fallback(self):
        tracked = contract.source_tracked_bucket_load(100, 80)
        self.assertEqual(tracked["count"], 80)
        self.assertEqual(tracked["source_tracking"], "initial_mask")

        fallback = contract.source_tracked_bucket_load(100, 20)
        self.assertEqual(fallback["count"], 100)
        self.assertEqual(
            fallback["source_tracking"],
            "bucket_region_fallback",
        )

        below_minimum = contract.source_tracked_bucket_load(20, 5)
        self.assertEqual(below_minimum["count"], 5)
        self.assertEqual(
            below_minimum["source_tracking"],
            "initial_mask",
        )

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

    def test_auto_phase_estimator_uses_monotonic_dataset_stage_order(self):
        estimator = contract.DeploymentPhaseEstimator()
        dig_target = [0.0, 0.0, 0.0]
        unload = [5.0, 0.0, 4.0]

        def update(
            tip,
            load_point,
            load_count=0.0,
            load_rate=0.0,
            bucket_joint=-0.2,
            dt=0.5,
        ):
            return estimator.update(
                dt=dt,
                joint_positions_4d=[0.0, 0.0, 0.0, bucket_joint],
                bucket_tip_world_xyz=tip,
                bucket_load_world_xyz=load_point,
                bucket_load_estimate=load_count,
                bucket_load_rate=load_rate,
                dig_target_world_xyz=dig_target,
                unload_landing_world_xyz=unload,
            )

        self.assertEqual(
            update([2.0, 0.0, 1.0], [1.8, 0.0, 1.1])["phase_index"],
            0,
        )
        self.assertEqual(
            update([0.4, 0.0, 0.4], [0.5, 0.0, 0.5])["phase_index"],
            1,
        )
        self.assertEqual(
            update([0.3, 0.0, 0.2], [0.4, 0.0, 0.3])["phase_index"],
            2,
        )
        self.assertEqual(
            update([0.2, 0.0, 0.1], [0.3, 0.0, 0.2])["phase_index"],
            3,
        )
        self.assertEqual(
            update(
                [0.2, 0.0, 0.1],
                [0.3, 0.0, 0.2],
                load_count=40.0,
                load_rate=20.0,
            )["phase_index"],
            4,
        )
        self.assertEqual(
            update(
                [0.2, 0.0, 0.2],
                [0.3, 0.0, 0.3],
                load_count=40.0,
                bucket_joint=-0.5,
            )["phase_index"],
            5,
        )
        self.assertEqual(
            update(
                [0.2, 0.0, 0.4],
                [0.3, 0.0, 0.4],
                load_count=40.0,
                bucket_joint=-0.5,
            )["phase_index"],
            6,
        )
        self.assertEqual(
            update(
                [0.2, 0.0, 0.7],
                [0.3, 0.0, 0.7],
                load_count=40.0,
                bucket_joint=-0.5,
            )["phase_index"],
            7,
        )
        self.assertEqual(
            update(
                [0.2, 0.0, 1.1],
                [0.3, 0.0, 1.1],
                load_count=40.0,
                bucket_joint=-0.5,
            )["phase_index"],
            8,
        )
        report = update(
            [5.0, 0.0, 4.0],
            [5.1, 0.0, 4.0],
            load_count=40.0,
            bucket_joint=-0.5,
        )
        self.assertEqual(report["phase_index"], 9)
        self.assertEqual(report["phase_name"], "unload_to_bin")
        self.assertEqual(report["phase_source"], "simulator_auto_fsm")


if __name__ == "__main__":
    unittest.main()
