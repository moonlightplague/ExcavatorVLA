from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from baseline.motion_geometry import BucketFloor, RigidBucketKinematics, hull_xz, rotated_min_z
from baseline.unload_planner import damped_ik, UnloadPlanner
from baseline.repeat_runtime import RepeatConfig, RepeatController


class MeshFloorTests(unittest.TestCase):
    def test_rigid_fk_keeps_lateral_marker_offset(self):
        points = {"swing": np.array([0., 0., 0.]), "boom": np.array([.3, 0., 1.]),
                  "arm": np.array([5.3, 0., 1.]), "bucket": np.array([9.3, 0., 1.])}
        end = np.array([10.3, .3, 1.])
        rt = SimpleNamespace(get_ik_world_points=lambda name: points,
                             get_real_joint_positions=lambda: np.zeros(4),
                             IK_MODEL={"signs": [1., 1., 1.]},
                             CTRL=SimpleNamespace(name_to_idx={name: i for i, name in enumerate(("swing", "boom", "arm", "bucket"))}),
                             bucket_point_world=lambda name: end)
        fk = RigidBucketKinematics(rt)
        np.testing.assert_allclose(fk.points(np.zeros(4), "pour")[-1], end)
        q = np.array([-1.2, .2, -.3, .4])
        expected_x = .3 + 5 * np.cos(.2) + 4 * np.cos(-.1) + np.cos(.3)
        expected_z = 1 + 5 * np.sin(.2) + 4 * np.sin(-.1) + np.sin(.3)
        expected = [expected_x * np.cos(-1.2) - .3 * np.sin(-1.2),
                    expected_x * np.sin(-1.2) + .3 * np.cos(-1.2), expected_z]
        np.testing.assert_allclose(fk.points(q, "pour")[-1], expected, atol=1e-6)
    def test_hull_preserves_rotated_mesh_floor(self):
        points = np.array([[-1, -.5], [-1, .5], [1, -.5], [1, .5], [0, 0], [.2, .1]])
        hull = hull_xz(points)
        self.assertEqual(len(hull), 4)
        for angle in np.linspace(-np.pi, np.pi, 41):
            expected = np.min(points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle))
            self.assertAlmostEqual(rotated_min_z(hull, angle), expected)

    def test_mesh_detects_penetration_when_reference_markers_are_clear(self):
        vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-.5, .5) for z in (-.5, .5)])
        world = vertices + [0., 0., 1.]
        rt = SimpleNamespace(
            BUCKET_LINK="bucket", GROUND_TOP_Z=0.,
            bucket_collision_mesh_candidates=lambda: [SimpleNamespace(GetPath=lambda: "physical"),
                                                       SimpleNamespace(GetPath=lambda: "display")],
            prim_collision_enabled=lambda prim: prim.GetPath() == "physical",
            bucket_load_source_mesh_local=Mock(return_value=((vertices, []), [], [])),
            transform_local_points_to_world=lambda link, points: world,
            get_real_joint_positions=lambda: np.array([1., 0.]),
            get_joint_anchor_world=lambda name: np.array([0., 0., 1.]),
            current_planar_chain=lambda name: {"radial": [1., 0.]},
            chain_angles_from_q=lambda q, end_effector: [0., 0., q[1]],
            bucket_point_world=lambda name: np.array([1., 0., 1.]),
            predicted_chain_world_points=lambda q, end_effector: [[0, 0, q[0]], [0, 0, q[0]], [0, 0, q[0]], [1, 0, q[0]]],
            info_print=Mock(),
        )
        floor = BucketFloor(rt)
        self.assertAlmostEqual(floor.minimum_z([.3, 0.]), -.2)
        self.assertAlmostEqual(floor.target_floor("tip", 0.), .535)
        self.assertAlmostEqual(floor.target_floor("tip", 90.), 2.035)
        rt.bucket_load_source_mesh_local.assert_called_once_with("physical")
        rt.CTRL = SimpleNamespace(name_to_idx={"swing": 1})
        rt.sand_depth_for_point = lambda point: (max(0., .65 - point[2]), .65, "planning_particle_snapshot")
        self.assertAlmostEqual(floor.sand_contact(np.array([1., 0.]))["depth_m"], .15)
        # Rotate and translate the actual support vertices, not marker offsets.
        rotated = floor.contact_points(np.array([2., np.pi / 2]))
        self.assertAlmostEqual(float(rotated[:, 2].min()), 1.)
        self.assertAlmostEqual(float(rotated[:, 2].max()), 3.)
        rt.sand_depth_for_point = lambda point: (.2, .2, "fallback_floor_no_particle_surface")
        self.assertEqual(floor.sand_contact(np.array([1., 0.]))["depth_m"], 0.)
        floor.reset()
        self.assertIsNone(floor.reference)


def arm_position(q):
    swing, boom, arm = q
    radial = 6 * np.cos(boom) + 5 * np.cos(boom + arm)
    z = 1 + 6 * np.sin(boom) + 5 * np.sin(boom + arm)
    return np.array([radial * np.cos(swing), radial * np.sin(swing), z])


class ExactUnloadTests(unittest.TestCase):
    def test_curl_search_continues_past_unsafe_low_rotation_arcs(self):
        rt, _landing, seed = self.unload_runtime()
        rt.GROUND_TOP_Z = 0.
        def phase_check(q_start, q_goal, mode, samples, deadline=None):
            ok = arm_position(q_goal[:3])[2] >= 2.
            return ok, "ok" if ok else "rotation_hits_floor", samples, {}
        rt.path_phase_check.side_effect = phase_check
        rows = RepeatController(rt, RepeatConfig()).curl_candidates(seed, arm_position([0., .5, -1.]))
        self.assertTrue(rows)
        self.assertTrue(all(arm_position(row["q"][:3])[2] >= 2. for row in rows))

    def test_near_fold_limit_does_not_force_unreachable_low_curl(self):
        rt, _landing, seed = self.unload_runtime()
        rt.GROUND_TOP_Z = 0.
        rt.planner_effective_joint_bounds_rad = lambda name: (
            np.radians([-70., 74.]) if name == "boom" else np.radians([-89., 92.]))
        def load_point(name, q, reference_q):
            swing, boom, arm, _bucket = q
            radius = 5.7 * np.cos(boom) + 2.95 * np.cos(boom + arm)
            return np.array([radius * np.cos(swing), radius * np.sin(swing),
                             2.6 + 5.7 * np.sin(boom) + 2.95 * np.sin(boom + arm)])
        rt.bucket_point_world = load_point
        rt.gravity_carry_closed_for_q = lambda q: (
            abs((np.degrees(sum(q[1:])) + 90.) % 180. - 90.) <= 26., {})
        rows = RepeatController(rt, RepeatConfig()).curl_candidates(seed, [6.1, 0., 1.05])
        self.assertEqual(rows, [])

    def test_coordinated_curl_reaches_raised_load_target_with_closed_bucket(self):
        rt, _landing, seed = self.unload_runtime()
        rt.GROUND_TOP_Z = 0.
        target = arm_position([0., .5, -1.])
        rows = RepeatController(rt, RepeatConfig()).curl_candidates(seed, target)
        self.assertTrue(rows)
        for row in rows:
            q = row["q"]
            self.assertAlmostEqual(float(np.degrees(q[3])), -120., places=5)
            self.assertGreaterEqual(arm_position(q[:3])[2], 1.2)
            np.testing.assert_allclose(arm_position(q[:3]), row["info"]["load_target"], atol=.025)

    def test_bounded_ik_reaches_opening_center(self):
        target = arm_position([-1.8, .9, -1.4])
        low, high = [-np.pi, -.2, -2.2], [np.pi, 1.4, -.1]
        seeds = [[-1.8, .4, -.4], [-1.8, 1.2, -1.8]]
        solutions = damped_ik(arm_position, target, seeds, low, high)
        self.assertTrue(solutions)
        for q, error in solutions:
            self.assertLess(error, .025)
            self.assertTrue(np.all(q >= low) and np.all(q <= high))
            self.assertLess(np.linalg.norm(arm_position(q) - target), .025)

    def test_unreachable_or_joint_limited_target_is_not_accepted(self):
        self.assertEqual(damped_ik(arm_position, [50, 0, 10], [[0, .5, -1]],
                                   [-np.pi, -.2, -2.2], [np.pi, 1.4, -.1]), [])
        self.assertEqual(damped_ik(arm_position, arm_position([-1.8, .9, -1.4]), [[-1.8, 0, -1]],
                                   [-np.pi, -.2, -2.2], [np.pi, .1, -.1]), [])

    def unload_runtime(self):
        landing = np.array([-7., -3., 2.56])
        seed = np.array([0., .5, -1., -2.])
        rt = SimpleNamespace(
            STATE={},
            CTRL=SimpleNamespace(name_to_idx={name: index for index, name in enumerate(("swing", "boom", "arm", "bucket"))}),
            CURL_HOLD_TARGET_DEG=-120., deg_to_rad=np.radians,
            PATH_CHECK_SAMPLES=24,
            path_phase_check=Mock(return_value=(True, "ok", 24, {})),
            path_obstacle_check=Mock(return_value=(True, "ok", 24, {})),
            target_to_swing_angle=lambda target: np.arctan2(target[1], target[0]),
            task_scene_context=lambda: {"unload_bin_z_range": [.5, 2.5]},
            planner_effective_joint_bounds_rad=lambda name: (-.2, 1.4) if name == "boom" else (-2.2, -.1),
            unload_release_alignment_bucket_deg=lambda value: value, unload_dump_target_deg=lambda: 88.,
            bucket_unload_release_point_world=lambda q, reference_q: arm_position(q[:3]),
            force_loaded_carry_bucket_q=lambda q, **kwargs: q,
            gravity_carry_closed_for_q=Mock(return_value=(True, {})),
            unload_goal_pose_collision_report=Mock(return_value=None),
            unload_drop_report=lambda q, reference_q: {"release_xy_err": float(np.linalg.norm(arm_position(q[:3])[:2] - landing[:2]))},
            unload_drop_execution_ready=lambda drop: True,
            bucket_point_world=lambda name, q, reference_q: arm_position(q[:3]),
            info_print=Mock(),
        )
        return rt, landing, seed

    def test_preflight_rejects_bad_carry_and_collision(self):
        rt, landing, seed = self.unload_runtime()
        planner = UnloadPlanner(rt)
        q, detail = planner.solve(landing, seed)
        self.assertIsNotNone(q)
        self.assertLess(detail["drop"]["release_xy_err"], .025)
        planner.reset()
        rt.gravity_carry_closed_for_q.return_value = (False, {})
        self.assertIsNone(planner.solve(landing, seed)[0])
        planner.reset()
        rt.gravity_carry_closed_for_q.return_value = (True, {})
        rt.unload_goal_pose_collision_report.return_value = {"obstacle": "truck_wall"}
        self.assertIsNone(planner.solve(landing, seed)[0])

    def test_preflight_uses_candidate_cell_and_restores_active_target(self):
        rt, landing, seed = self.unload_runtime()
        previous = np.array([2., 3., 4.])
        rt.STATE["active_unload_landing_point"] = previous
        rt.unload_drop_report = lambda q, reference_q: {
            "release_xy_err": float(np.linalg.norm(
                arm_position(q[:3])[:2] - rt.STATE["active_unload_landing_point"][:2]))}
        planner = UnloadPlanner(rt)
        self.assertIsNotNone(planner.solve(landing, seed)[0])
        self.assertIs(rt.STATE["active_unload_landing_point"], previous)
        planner.reset()
        rt.STATE.clear()
        self.assertIsNotNone(planner.solve(landing, seed)[0])
        self.assertNotIn("active_unload_landing_point", rt.STATE)
        planner.reset()
        rt.unload_drop_report = Mock(side_effect=RuntimeError("report failed"))
        with self.assertRaisesRegex(RuntimeError, "report failed"):
            planner.solve(landing, seed)
        self.assertNotIn("active_unload_landing_point", rt.STATE)

    def test_preflight_respects_release_angle_and_higher_retry(self):
        rt, landing, seed = self.unload_runtime()
        rt.unload_release_alignment_bucket_deg = lambda value: min(-25., value)
        planner = UnloadPlanner(rt)
        q, _ = planner.solve(landing, seed)
        self.assertAlmostEqual(float(np.degrees(q[3])), -25., places=5)
        low = arm_position(q[:3])[2]
        q, _ = planner.solve(landing, seed, min_release_z=low + .5)
        self.assertGreater(arm_position(q[:3])[2], low + .45)


if __name__ == "__main__":
    unittest.main()
