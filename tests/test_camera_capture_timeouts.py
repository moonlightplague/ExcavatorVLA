"""Exercise camera waits without requiring an Isaac renderer in the test process."""
import ast
import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


CAMERA = Path(__file__).resolve().parents[1] / "scripts/excavator_app/excavator_dataset_camera.py"


def load_function(name, namespace):
    tree = ast.parse(CAMERA.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(CAMERA), "exec"), namespace)
    return namespace[name]


class CameraCaptureTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_frame_returns_after_capture_helper_finishes(self):
        utility = ModuleType("omni.kit.viewport.utility")
        async def never_frame(_api):
            await asyncio.Future()
        utility.next_viewport_frame_async = never_frame
        utility.capture_viewport_to_buffer = lambda *args, **kwargs: object()
        capture = load_function("capture_viewport_rgb_async", {
            "asyncio": asyncio, "time": time,
            "compact_capture_meta": dict,
            "set_viewport_camera_path": lambda *args: None,
        })
        with patch.dict(sys.modules, {"omni.kit.viewport.utility": utility}):
            rgb, reason, _ = await asyncio.wait_for(
                capture(SimpleNamespace(), object(), "/Camera", timeout_s=0.02), 0.5
            )
        self.assertIsNone(rgb)
        self.assertEqual(reason, "viewport_capture_no_rgb")

    async def test_initial_frame_wait_is_bounded(self):
        utility = ModuleType("omni.kit.viewport.utility")
        async def never_frame(_api):
            await asyncio.Future()
        utility.next_viewport_frame_async = never_frame
        utility.capture_viewport_to_buffer = lambda *args, **kwargs: self.fail("capture before first frame")
        capture = load_function("capture_viewport_rgb_async", {
            "asyncio": asyncio, "time": time,
            "compact_capture_meta": dict,
            "set_viewport_camera_path": lambda *args: None,
        })
        with patch.dict(sys.modules, {"omni.kit.viewport.utility": utility}):
            rgb, reason, _ = await asyncio.wait_for(
                capture(SimpleNamespace(), object(), "/Camera", wait_frames=1, timeout_s=0.02), 0.5
            )
        self.assertIsNone(rgb)
        self.assertEqual(reason, "viewport_frame_timeout")

    async def test_failed_viewport_prime_aborts_warmup_before_capture(self):
        async def prime(*args, **kwargs):
            return {"ok": False, "reason": "viewport_prime_timeout:6.00s", "frames": 0}
        warmup = load_function("warmup_for_episode", {
            "backend": lambda rt: "viewport_capture",
            "initialize": lambda *args, **kwargs: True,
            "prime_dataset_viewports_async": prime,
            "set_dataset_viewports_capture_active": lambda *args: None,
        })
        rt = SimpleNamespace(STATE={})
        self.assertFalse(await warmup(rt))
        self.assertEqual(rt.STATE["dataset_camera_warmup_status"]["reason"], "viewport_prime_timeout:6.00s")
        self.assertFalse(rt.STATE["dataset_camera_warmup_viewports_active"])


class CameraResourceLifetimeTests(unittest.TestCase):
    def test_headless_windows_stay_visible_unless_explicitly_overridden(self):
        keep_visible = load_function("dataset_viewport_keep_visible", {"os": os})
        rt = SimpleNamespace(STATE={})
        with patch.dict(os.environ, {"EXCAVATOR_HEADLESS": "1"}, clear=True):
            self.assertTrue(keep_visible(rt))
            os.environ["EXCAVATOR_DATASET_VIEWPORT_KEEP_VISIBLE"] = "0"
            self.assertFalse(keep_visible(rt))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(keep_visible(rt))

    def test_repeated_captures_do_not_toggle_window_visibility(self):
        class Window:
            def __init__(self):
                self._visible = False
                self.changes = []
            @property
            def visible(self):
                return self._visible
            @visible.setter
            def visible(self, value):
                self._visible = value
                self.changes.append(value)
        namespace = {"dataset_viewport_keep_visible": lambda rt: True}
        load_function("set_dataset_viewports_visible", namespace)
        active = load_function("set_dataset_viewports_capture_active", namespace)
        window = Window()
        rt = SimpleNamespace(STATE={"dataset_viewport_capture": {"viewports": {"0": {"window": window}}}})
        for _ in range(100):
            active(rt, True)
            active(rt, False)
        self.assertEqual(window.changes, [True])

    def test_completed_capture_breaks_helper_callback_ownership_cycle(self):
        finalize = load_function("_finalize_pending_triplet", {
            "time": time,
            "background_clock_seconds": lambda rt: (0.0, "simulation"),
            "set_dataset_viewports_capture_active": lambda *args: None,
        })
        rt = SimpleNamespace(STATE={"dataset_camera_capture_generation": 2}, DATASET_CAMERA_NAMES=["0"])
        triplet = {"pending": ["0"], "helpers": [object()], "generation": 1}
        self.assertFalse(finalize(rt, triplet))
        self.assertEqual(len(triplet["helpers"]), 1)
        triplet["pending"].clear()
        self.assertTrue(finalize(rt, triplet))
        self.assertEqual(triplet["helpers"], [])
        self.assertTrue(triplet["done"])


if __name__ == "__main__":
    unittest.main()
