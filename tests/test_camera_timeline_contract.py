import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "excavator_app" / "excavator_runtime.py"
CAMERA_PATH = ROOT / "scripts" / "excavator_app" / "excavator_dataset_camera.py"
AUTO_COLLECT_PATH = ROOT / "scripts" / "excavator_app" / "auto_dataset_collect.py"


def function_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def called_attribute_names(node):
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute):
            names.append(func.attr)
        elif isinstance(func, ast.Name):
            names.append(func.id)
    return names


class CameraTimelineContractTests(unittest.TestCase):
    def test_strict_sample_wait_pauses_only_for_render_completion(self):
        node = function_node(RUNTIME_PATH, "dataset_camera_wait_for_scheduled_capture")
        calls = called_attribute_names(node)
        self.assertIn("pause", calls)
        self.assertIn("play", calls)
        self.assertIn("commit", calls)
        self.assertIn("simulation_timeline_is_playing", calls)
        self.assertIn("submit_viewport_capture_triplet", calls)
        self.assertIn("step_updates", calls)
        source = ast.unparse(node)
        self.assertIn("_EXCAVATOR_INTERNAL_TIMELINE_PAUSE_DEPTH", source)
        self.assertIn("raw_pending", source)
        self.assertIn("capture_not_submitted_before_pause", source)
        self.assertLess(
            source.index("submit_viewport_capture_triplet"),
            source.index("timeline.pause()"),
        )
        pause_tail = source[source.index("timeline.pause()") :]
        self.assertNotIn("submit_viewport_capture_triplet", pause_tail)
        self.assertGreaterEqual(source.count("timeline.commit()"), 2)

    def test_sample_gate_uses_fixed_next_sim_deadline(self):
        due_node = function_node(RUNTIME_PATH, "dataset_record_sample_due")
        prepare_node = function_node(RUNTIME_PATH, "dataset_camera_prepare_for_scheduled_update")
        record_node = function_node(RUNTIME_PATH, "dataset_record_sample")
        due_source = ast.unparse(due_node)
        prepare_source = ast.unparse(prepare_node)
        record_source = ast.unparse(record_node)
        self.assertIn("dataset_next_sim_sample_time", due_source)
        self.assertIn("dataset_next_sim_sample_time", prepare_source)
        self.assertIn("float(sample_index) * interval", record_source)
        self.assertIn("camera_simulation_uniform_grid", record_source)
        self.assertIn("timestamp.capture_offset_s", record_source)

    def test_success_requires_original_uniform_10hz_source(self):
        audit_node = function_node(RUNTIME_PATH, "dataset_episode_source_sampling_audit")
        finish_node = function_node(RUNTIME_PATH, "auto_collect_finish_episode")
        loop_node = function_node(RUNTIME_PATH, "auto_collect_loop")
        audit_source = ast.unparse(audit_node)
        finish_source = ast.unparse(finish_node)
        loop_source = ast.unparse(loop_node)
        self.assertIn("reused_camera_rows", audit_source)
        self.assertIn("non_physical_timestamp_rows", audit_source)
        self.assertIn("seen_camera_paths", audit_source)
        self.assertIn("max_capture_grid_error_s", audit_source)
        self.assertIn("dataset_source_not_original_uniform_10hz", finish_source)
        self.assertIn("reject_episode_and_continue_auto_collect", finish_source)
        self.assertNotIn("auto_collect_stop_requested", finish_source)
        self.assertIn("dataset_source_rate_must_be_original_10hz", loop_source)

    def test_execution_has_start_recovery_and_stage_timeline_gate(self):
        node = function_node(RUNTIME_PATH, "execute_dig_target_ball")
        calls = called_attribute_names(node)
        self.assertIn("ensure_timeline_playing_async", calls)
        self.assertIn("simulation_timeline_is_playing", calls)
        self.assertIn("set_execution_failure_reason", calls)

    def test_active_motion_cancels_on_unexpected_timeline_stop(self):
        node = function_node(RUNTIME_PATH, "motion_cancel_requested")
        calls = called_attribute_names(node)
        self.assertIn("internal_timeline_pause_active", calls)
        self.assertIn("simulation_timeline_is_playing", calls)

    def test_background_capture_skips_internal_pause(self):
        node = function_node(CAMERA_PATH, "background_capture_loop")
        source = ast.unparse(node)
        self.assertIn("internal_timeline_pause_active", source)
        self.assertIn("not internal_pause_active", source)

    def test_viewport_warmup_primes_render_frames_before_readback(self):
        node = function_node(CAMERA_PATH, "prime_dataset_viewports_async")
        calls = called_attribute_names(node)
        self.assertIn("ensure_dataset_viewports", calls)
        self.assertIn("set_dataset_viewports_capture_active", calls)
        self.assertIn("next_viewport_frame_async", calls)
        self.assertIn("wait_for", calls)

    def test_viewports_remain_visible_for_complete_warmup(self):
        visibility_node = function_node(CAMERA_PATH, "set_dataset_viewports_capture_active")
        visibility_source = ast.unparse(visibility_node)
        self.assertIn("dataset_camera_warmup_viewports_active", visibility_source)

        warmup_node = function_node(CAMERA_PATH, "warmup_for_episode")
        warmup_source = ast.unparse(warmup_node)
        self.assertIn("prime_dataset_viewports_async", warmup_source)
        self.assertIn("dataset_camera_warmup_viewports_active", warmup_source)
        self.assertIn("finally", warmup_source)

    def test_strict_mode_uses_sample_owned_camera_scheduler(self):
        start_node = function_node(RUNTIME_PATH, "dataset_camera_start_background")
        start_source = ast.unparse(start_node)
        self.assertIn("DATASET_STRICT_SAMPLE_WAIT", start_source)
        self.assertIn("sample_scheduler_owns_capture", start_source)
        self.assertIn("sample_owned_strict", start_source)

        prepare_node = function_node(RUNTIME_PATH, "dataset_camera_prepare_for_scheduled_update")
        prepare_source = ast.unparse(prepare_node)
        self.assertNotIn("dataset_camera_background_running", prepare_source)
        self.assertIn("dataset_record_sample_due", RUNTIME_PATH.read_text(encoding="utf-8-sig"))
        self.assertIn("submit_viewport_capture_triplet", prepare_source)

    def test_strict_samples_are_owned_by_the_single_update_clock(self):
        step_node = function_node(RUNTIME_PATH, "step_updates")
        recorder_node = function_node(RUNTIME_PATH, "dataset_record_scheduled_update_sample_async")
        boundary_node = function_node(RUNTIME_PATH, "dataset_record_stage_boundary_sample")
        step_source = ast.unparse(step_node)
        recorder_source = ast.unparse(recorder_node)
        boundary_source = ast.unparse(boundary_node)

        self.assertIn("dataset_record_scheduled_update_sample_async", step_source)
        self.assertIn("dataset_record_sample_due", recorder_source)
        self.assertIn("dataset_record_sample_async", recorder_source)
        self.assertIn("dataset_camera_pose_sync_active", recorder_source)
        self.assertIn("dataset_global_sample_active", recorder_source)
        self.assertIn("DATASET_STRICT_SAMPLE_WAIT", boundary_source)
        self.assertIn("auto_collect_active", boundary_source)

    def test_next_scoop_index_changes_only_after_replanning_succeeds(self):
        replan_node = function_node(RUNTIME_PATH, "auto_collect_plan_next_scoop")
        reset_node = function_node(RUNTIME_PATH, "multi_scoop_reset_cycle_trackers")
        replan_source = ast.unparse(replan_node)
        self.assertNotIn("STATE['dataset_scoop_index']", replan_source)
        self.assertIn("STATE['multi_scoop_planning_index']", replan_source)
        self.assertIn("finally:", replan_source)
        reset_source = ast.unparse(reset_node)
        self.assertIn("STATE['dataset_scoop_index']", reset_source)
        self.assertIn("STATE['last_action_mode'] = 'pre_dig'", reset_source)

    def test_adaptive_stop_requires_material_exhaustion_evidence(self):
        viability_node = function_node(
            RUNTIME_PATH,
            "multi_scoop_dig_viability_report",
        )
        episode_node = function_node(RUNTIME_PATH, "auto_collect_one_episode")
        viability_source = ast.unparse(viability_node)
        episode_source = ast.unparse(episode_node)

        self.assertIn("low_swept_sand_density", viability_source)
        self.assertIn("low_local_sand_density", viability_source)
        self.assertIn("no_real_surface_near_target", viability_source)
        self.assertIn("blocking_reasons", viability_source)
        self.assertIn("natural_exhaustion", viability_source)
        spatial_node = function_node(
            RUNTIME_PATH,
            "multi_scoop_filter_new_spatial_targets",
        )
        spatial_source = ast.unparse(spatial_node)
        self.assertIn("previous_scoop_nearest_xy_m", spatial_source)
        self.assertIn("spatially_exhausted", spatial_source)
        self.assertIn("completed >= scoops_target", episode_source)
        self.assertIn("no_plannable_new_dig_target", episode_source)
        self.assertIn("multi_scoop_repeat_target_candidates", episode_source)
        self.assertIn("technical_safety_cap_reached", episode_source)
        self.assertIn("multi_scoop_exhausted_before_minimum", episode_source)

    def test_target_topk_prefers_distinct_xy_before_depth_variants(self):
        rank_node = function_node(RUNTIME_PATH, "auto_collect_rank_dig_targets")
        rank_source = ast.unparse(rank_node)
        self.assertIn("seen_angle_indices", rank_source)
        self.assertIn("unique_xy_rows + alternate_depth_rows", rank_source)

    def test_later_scoop_planning_does_not_prune_after_two_route_failures(self):
        find_plan_node = function_node(AUTO_COLLECT_PATH, "find_plan")
        find_plan_source = ast.unparse(find_plan_node)
        self.assertIn("later_multi_scoop", find_plan_source)
        self.assertIn("max(signature_global_limit, 4)", find_plan_source)

    def test_dataset_row_uses_capture_pose_and_simulation_time(self):
        node = function_node(RUNTIME_PATH, "dataset_record_sample_async")
        source = ast.unparse(node)
        self.assertIn("capture_q", source)
        self.assertIn("capture_q_cmd", source)
        self.assertIn("capture_pose_sim_time", source)
        self.assertIn("camera_simulation", source)

    def test_controlled_recovery_is_path_validated_and_loss_masked(self):
        inject_node = function_node(
            RUNTIME_PATH,
            "multi_scoop_inject_controlled_recovery",
        )
        inject_source = ast.unparse(inject_node)
        self.assertIn("path_segment_check", inject_source)
        self.assertIn("action_is_expert=False", inject_source)
        self.assertIn("action_loss_weight=0.0", inject_source)
        self.assertIn("planner_restarts_from_actual_pose", inject_source)

    def test_recovery_mode_keeps_clean_multi_scoop_mode_separate(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8-sig")
        self.assertIn('"data_multi_recovery"', source)
        self.assertIn("MULTI_SCOOP_RECOVERY_ENABLED", source)
        self.assertIn("quality_rejected/recovery_supervision_incomplete", source)


if __name__ == "__main__":
    unittest.main()
