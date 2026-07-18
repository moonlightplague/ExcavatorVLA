import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "excavator_app" / "excavator_runtime.py"
CAMERA_PATH = ROOT / "scripts" / "excavator_app" / "excavator_dataset_camera.py"


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
    def test_strict_sample_wait_never_pauses_timeline(self):
        node = function_node(RUNTIME_PATH, "dataset_camera_wait_for_scheduled_capture")
        calls = called_attribute_names(node)
        self.assertNotIn("pause", calls)
        self.assertIn("simulation_timeline_is_playing", calls)
        self.assertIn("submit_viewport_capture_triplet", calls)
        self.assertIn("step_updates", calls)

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

    def test_dataset_row_uses_capture_pose_and_simulation_time(self):
        node = function_node(RUNTIME_PATH, "dataset_record_sample_async")
        source = ast.unparse(node)
        self.assertIn("capture_q", source)
        self.assertIn("capture_q_cmd", source)
        self.assertIn("capture_pose_sim_time", source)
        self.assertIn("camera_simulation", source)


if __name__ == "__main__":
    unittest.main()
