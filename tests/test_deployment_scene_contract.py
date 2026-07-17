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

    def test_run_simulation_has_no_obsolete_target_helper_or_truck_path(self):
        source = (
            Path(__file__).resolve().parents[1] / "run_simulation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("world_point_to_robot_local_feature(", source)
        self.assertNotIn('GetPrimAtPath("/World/DumpTruck")', source)
        self.assertNotIn("SAND_INITIAL_CENTER", source)


if __name__ == "__main__":
    unittest.main()
