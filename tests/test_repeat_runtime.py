import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

from baseline.repeat_runtime import RepeatConfig, RepeatController, install


def fake_runtime(directory):
    q = np.array([.5, .4, -.8, .2])
    rt = SimpleNamespace(
        STATE={"dig_plan_sequence": [("pre_dig", q.copy(), 1.)],
               "dataset_samples": 10, "dataset_episode_sample_start": 10,
               "dataset_episode_dir": directory, "dataset_scoop_index": 0,
               "dataset_recording": True, "dataset_meta_path": str(Path(directory) / "meta.json")},
        CONTROL_DT=.1, PATH_CHECK_SAMPLES=10, MOVE_DURATION_MARGIN_SECONDS=.1,
        os=os, time=time, UNLOAD_BIN_SAFE_XY_MARGIN=.3, AUTO_UNLOAD_WALL_MARGIN=.6,
        AUTO_SCENE_UNLOAD_LANDING_RADIUS_SOFT_MARGIN=.1,
        task_scene_context=lambda: {"unload_bin_z_range": [1., 3.], "pile_center": [0., 7., 0.]},
        unload_context_polygon_xy=lambda ctx, safe_margin: np.array(
            [[-3 + safe_margin, -1 + safe_margin], [3 - safe_margin, -1 + safe_margin],
             [3 - safe_margin, 1 - safe_margin], [-3 + safe_margin, 1 - safe_margin]]),
        sand_particle_positions=lambda: np.zeros((0, 3)), bucket_pour_pos=lambda: [0, 0, 4],
        unload_landing_radius_status=lambda *args, **kwargs: {"ok": True, "radius_m": 8},
        sync_motion_start_q=Mock(return_value=q.copy() + 1),
        clear_planning_runtime_caches=Mock(),
        find_clearance_route=Mock(return_value=([q.copy() + .2], "checked")),
        set_execution_failure_reason=Mock(), start_task=Mock(return_value=42),
        dataset_record_event=Mock(), info_print=Mock(), plan_joint_motion_metrics=lambda *a, **k: {"estimated_time": 1.},
        move_to_profile=AsyncMock(return_value=True), get_real_joint_positions=lambda: q.copy(),
        control_step_frames=lambda: 1, append_jsonl_dataset=Mock(), write_json_file=Mock(),
        multi_scoop_set_stop=Mock(),
        multi_scoop_filter_new_spatial_targets=lambda rows: (list(rows), {}),
        auto_collect_one_episode=AsyncMock(), auto_collect_begin_episode=Mock(return_value={}),
        auto_collect_initial_pose_for_attempt=Mock(), auto_collect_find_plan=AsyncMock(),
        dig_plan_specs_from_candidate=Mock(return_value=[]),
        dig_floor_safe_target_z=Mock(return_value=(.035, {})),
        predicted_phase_ground_report=Mock(return_value={}), phase_ground_ok=Mock(return_value=(True, "ok")),
        predicted_phase_ground_vertical_correction=Mock(return_value=(0., {})),
        cut_front_edge_quality=Mock(return_value=(True, "ok", 0.)), FRONT_EDGE_MIN_TIP_DEPTH=.02,
        predicted_chain_world_points=Mock(return_value=[]),
        path_phase_check=Mock(return_value=(True, "ok", 10, {})),
        path_obstacle_check=Mock(return_value=(True, "ok", 10, {})),
        plan_dump_pose_to_bin=Mock(return_value=(None, "unavailable")),
        AUTO_SCENE_FIXED_PROFILE=None, CTRL=SimpleNamespace(q_cmd=q.copy()),
        AUTO_COLLECT_TARGET_MIN_Z=.04, GROUND_TOP_Z=0.,
        AUTO_COLLECT_CANDIDATE_PLAN_SECONDS=4., AUTO_COLLECT_FIND_PLAN_MAX_SECONDS=28.,
        gravity_carry_closed_for_q=Mock(return_value=(True, {})),
        target_to_swing_angle=lambda target: .7, q_from_pose_deg=Mock(return_value=q.copy()),
        q_deg_values=Mock(return_value=[0, 0, 0, 0]),
        auto_collect_finish_episode=Mock(return_value=True), execute_dig_target_ball=AsyncMock(return_value=True),
    )

    async def step(frames):
        if rt.STATE["dataset_recording"]:
            rt.STATE["dataset_samples"] += frames

    rt.step_updates = step
    return rt


class RepeatRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_physical_contact_can_engage_sand_below_named_tip_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.cut_front_edge_quality.return_value = (False, "front tip not engaged enough: tip=0.000", 80.)
            controller = install(rt, RepeatConfig())
            controller.floor.sand_contact = Mock(return_value={"depth_m": .08, "point": [0., 7., .04]})
            self.assertTrue(rt.cut_front_edge_quality("insert_cut", {"repeat_mesh_q_rad": [0.] * 4})[0])
            controller.floor.sand_contact.return_value["depth_m"] = 0.
            self.assertFalse(rt.cut_front_edge_quality("insert_cut", {"repeat_mesh_q_rad": [0.] * 4})[0])
            controller.floor.sand_contact.return_value["depth_m"] = .08
            original = rt.cut_front_edge_quality.__wrapped__
            original.return_value = (False, "bucket body much deeper than front tip", 100.)
            self.assertFalse(rt.cut_front_edge_quality("insert_cut", {"repeat_mesh_q_rad": [0.] * 4})[0])

    async def test_mesh_floor_shortfall_is_refined_before_rejecting_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            install(rt, RepeatConfig())
            amount, detail = rt.predicted_phase_ground_vertical_correction("pull_mid_cut", {"repeat_mesh_min_z": -.007})
            self.assertAlmostEqual(amount, .042)
            self.assertEqual(detail["limiting_point"], "bucket_collision_mesh")

    async def test_fixed_count_retries_dense_targets_without_resetting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            history = [[0., 7., .5]]
            all_targets = [{"target_xyz": [0., 7., .3]}]
            rt.STATE.update(dataset_scoop_target_history=history, multi_scoop_ranked_targets_all=all_targets,
                            dataset_scoop_index=2)
            calls = []
            async def plan(attempt, ranked_targets=None):
                calls.append((ranked_targets, rt.MULTI_SCOOP_TARGET_MIN_SPACING_M))
                return (None, None, ["fresh_failed"]) if len(calls) == 1 else ([0., 7., .3], ["complete"], ["retry"])
            rt.auto_collect_find_plan = plan
            install(rt, RepeatConfig())
            _, sequence, attempts = await rt.auto_collect_find_plan(102, ranked_targets=[])
            self.assertEqual(sequence, ["complete"])
            self.assertEqual(calls[1], (all_targets, 0.))
            self.assertEqual(attempts, ["fresh_failed", "retry"])
            self.assertEqual(rt.MULTI_SCOOP_TARGET_MIN_SPACING_M, .28)
            self.assertIs(rt.STATE["dataset_scoop_target_history"], history)

    async def test_spatial_exclusion_does_not_end_fixed_count_with_material_left(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.multi_scoop_filter_new_spatial_targets = lambda rows: ([], {"spatially_exhausted": True})
            install(rt, RepeatConfig())
            targets = [{"target_xyz": [0., 7., .3]}]
            usable, report = rt.multi_scoop_filter_new_spatial_targets(targets)
            self.assertEqual(usable, targets)
            self.assertFalse(report["spatially_exhausted"])
            self.assertTrue(report["repeat_target_fallback"])

    async def test_every_cycle_returns_physically_and_keeps_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            controller = RepeatController(rt, RepeatConfig(scoops=2, settle_seconds=.1))
            original = rt.execute_dig_target_ball
            for scoop in range(2):
                rt.STATE["dataset_scoop_index"] = scoop
                controller.choose_landing()
                self.assertTrue(await controller.execute_cycle(original, return_home=False))
            self.assertEqual(len(controller.cycles), 2)
            self.assertEqual(rt.move_to_profile.await_count, 2)
            self.assertTrue(rt.STATE["dataset_recording"])
            self.assertEqual(len(controller.bed.visits), 2)
            self.assertEqual(original.await_args.kwargs["return_home"], False)
            call = rt.move_to_profile.await_args
            self.assertEqual(call.kwargs["mode"], "pre_dig")
            self.assertTrue(call.kwargs["path_waypoints"])
            self.assertEqual(controller.cycles[-1]["scoop_index"], 1)

    async def test_return_failure_propagates_and_is_never_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.find_clearance_route.return_value = (None, "truck wall")
            controller = RepeatController(rt, RepeatConfig(scoops=2, settle_seconds=.1))
            controller.choose_landing()
            self.assertFalse(await controller.execute_cycle(rt.execute_dig_target_ball))
            rt.move_to_profile.assert_not_awaited()
            self.assertEqual(controller.bed.visits, [])
            self.assertFalse(controller.cycles[0]["returned"])
            self.assertIn("repeat_return_no_route", rt.set_execution_failure_reason.call_args.args[0])

    async def test_dig_failure_does_not_attempt_return(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.execute_dig_target_ball.return_value = False
            controller = RepeatController(rt, RepeatConfig(scoops=2))
            self.assertFalse(await controller.execute_cycle(rt.execute_dig_target_ball))
            rt.move_to_profile.assert_not_awaited()
            self.assertEqual(controller.cycles, [])

    async def test_install_uses_fixed_count_and_rejects_incomplete_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            original_finish = rt.auto_collect_finish_episode
            controller = install(rt, RepeatConfig(scoops=3))
            self.assertFalse(rt.MULTI_SCOOP_ADAPTIVE_STOP_ENABLED)
            self.assertEqual(rt.STATE["dataset_scoops_min"], 3)
            self.assertGreater(rt.AUTO_DIG_RING_RADII[-1], 1.)
            self.assertAlmostEqual(rt.AUTO_COLLECT_TARGET_MIN_Z, .15)
            self.assertAlmostEqual(rt.MULTI_SCOOP_TARGET_MIN_SPACING_M, .28)
            self.assertTrue(any(.28 <= .672 * radius < .55 for radius in rt.AUTO_DIG_RING_RADII))
            meta = rt.auto_collect_begin_episode()
            self.assertEqual(meta["repeat_baseline"]["config"]["scoops"], 3)
            rt.auto_collect_finish_episode(meta, True, "ok")
            self.assertFalse(original_finish.call_args.args[1])
            self.assertIn("incomplete_physical_returns", original_finish.call_args.args[2])
            controller.cycles = [{"returned": True}] * 3
            rt.auto_collect_finish_episode(meta, True, "ok")
            self.assertTrue(original_finish.call_args.args[1])
            rt.multi_scoop_set_stop.assert_called_once()
            with self.assertRaises(RuntimeError):
                install(rt, RepeatConfig(scoops=3))

    async def test_attempt_reset_forgets_previous_bed_and_visits(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            controller = install(rt, RepeatConfig(scoops=2))
            controller.choose_landing()
            controller.bed.commit(controller.selected["landing"])
            controller.cycles.append({"returned": True})
            await rt.auto_collect_one_episode()
            self.assertIsNone(controller.bed)
            self.assertEqual(controller.cycles, [])

    async def test_unreachable_bed_fails_without_unsafe_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.unload_landing_radius_status = lambda *a, **k: {"ok": False}
            controller = RepeatController(rt, RepeatConfig())
            with self.assertRaisesRegex(RuntimeError, "No safe truck-bed cell"):
                controller.choose_landing()
            original_find = rt.auto_collect_find_plan
            install(rt, RepeatConfig())
            target, sequence, _attempts = await rt.auto_collect_find_plan(1)
            self.assertIsNone(sequence)
            original_find.assert_not_awaited()

    async def test_initial_pose_faces_pile_and_honors_exact_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            original = rt.auto_collect_initial_pose_for_attempt
            install(rt, RepeatConfig())
            pose = rt.auto_collect_initial_pose_for_attempt(1)
            self.assertEqual(pose["id"], "repeat_pile_facing")
            self.assertAlmostEqual(pose["pose_deg"]["swing"], np.degrees(.7))
            original.assert_not_called()
            rt.AUTO_SCENE_FIXED_PROFILE = {"initial_pose_deg": {"swing": -30}}
            rt.auto_collect_initial_pose_for_attempt(2)
            original.assert_called_once_with(2)

    async def test_loaded_planning_catches_wall_between_coarse_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            rt.PATH_CHECK_SAMPLES = 24
            def obstacle_check(q_start, q_goal, mode, samples, deadline=None):
                hit = any(.86 < sample / samples < .89 for sample in range(1, samples + 1))
                return not hit, "truck_wall" if hit else "ok", samples, {}
            rt.path_obstacle_check = obstacle_check
            self.assertTrue(obstacle_check(None, None, "unload_to_bin", 6)[0])
            install(rt, RepeatConfig())
            self.assertFalse(rt.path_obstacle_check(None, None, "unload_to_bin", samples=6)[0])
            self.assertEqual(rt.path_obstacle_check(None, None, "pull_mid_cut", samples=6)[2], 6)

    async def test_curl_rotation_clearance_is_planned_before_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            names = ["insert_cut", "pull_mid_cut", "curl_to_hold_material", "pull_exit_cut", "secure_load", "lift_carry"]
            specs = [(name, np.array([1., 2., .35]), -120., None, "load", 1., True) for name in names]
            rt.dig_plan_specs_from_candidate.return_value = specs
            install(rt, RepeatConfig())
            candidate = {}
            planned = rt.dig_plan_specs_from_candidate([0, 0, .15], candidate)
            self.assertAlmostEqual(planned[0][1][2], .35, places=5)
            self.assertAlmostEqual(planned[1][1][2], .35, places=5)
            self.assertAlmostEqual(planned[2][1][2], 1.05, places=5)
            self.assertTrue(all(planned[i][1][2] < planned[i + 1][1][2] for i in range(2, 5)))
            self.assertEqual(planned[2][2:], specs[2][2:])
            self.assertEqual(planned[3][2:5], (None, "carry", "load"))
            self.assertFalse(rt.DIG_CURL_CANDIDATE_FILTER({"ground_report": {"load_z": .5}, "q": [0.] * 4}))
            self.assertTrue(rt.DIG_CURL_CANDIDATE_FILTER({"ground_report": {"load_z": 1.1}, "q": [0.] * 4}))
            rt.gravity_carry_closed_for_q.return_value = (False, {})
            self.assertFalse(rt.DIG_CURL_CANDIDATE_FILTER({"ground_report": {"load_z": 1.1}, "q": [0.] * 4}))

    async def test_reachable_curl_height_propagates_through_escape_and_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = fake_runtime(directory)
            controller = install(rt, RepeatConfig())
            rt.bucket_point_world = Mock(return_value=np.array([1., 2., 4.35]))
            point = controller.adjust_plan_point("pull_exit_cut", [1., 2., 1.1], np.zeros(4))
            self.assertAlmostEqual(float(point[2]), 4.39, places=5)
            rt.bucket_point_world.return_value[2] = 4.57
            point = controller.adjust_plan_point("lift_carry", [1., 2., 1.55], np.zeros(4))
            self.assertAlmostEqual(float(point[2]), 4.77, places=5)


if __name__ == "__main__":
    unittest.main()
