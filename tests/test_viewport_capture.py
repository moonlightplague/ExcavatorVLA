"""Test readback timing and ownership without starting Isaac Sim."""
import ast
import contextlib
import io
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest

import numpy as np


class ViewportCaptureTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'run_simulation.py'
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('capture_rgb_from_persistent_viewports',
                                       'cleanup_viewport_capture_helpers')]
        self.frames = 0
        self.pending = []
        self.complete_after = 1

        class Window:
            def __init__(self):
                self._visible = True
                self.changes = []

            @property
            def visible(self):
                return self._visible

            @visible.setter
            def visible(self, value):
                self._visible = value
                self.changes.append(value)

        self.window = Window()
        self.namespace = {
            'CAPTURE_WAIT_FRAMES': 5, 'CAPTURE_MAX_RENDER_UPDATES': 120,
            'CAPTURE_TIMEOUT_SECONDS': 5.0, 'PENDING_VIEWPORT_CAPTURE_HELPERS': [],
            'LAST_CAPTURE_TIMING_MS': {},
            'time': SimpleNamespace(perf_counter=lambda: self.frames * 0.1),
            'capsule_to_numpy_rgba': lambda *args: np.ones((2, 2, 4), dtype=np.uint8),
            'resize_rgb_for_model': lambda rgb, np_module: rgb,
            'CAPTURE_RESIZE_EXECUTOR': SimpleNamespace(submit=lambda fn, *args:
                SimpleNamespace(result=lambda: fn(*args))),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), self.namespace)

    def submit(self, viewport, callback):
        self.pending.append((self.frames, callback))
        return object()

    def update(self):
        self.frames += 1
        remaining = []
        for submitted, callback in self.pending:
            if self.complete_after is not None and self.frames - submitted >= self.complete_after:
                callback(None, 16, 2, 2, None)
            else:
                remaining.append((submitted, callback))
        self.pending = remaining

    def capture(self):
        return self.namespace['capture_rgb_from_persistent_viewports'](
            {'0': {'window': self.window, 'viewport': object()}},
            SimpleNamespace(update=self.update), self.submit, np,
        )

    def test_readback_can_finish_after_five_updates(self):
        self.complete_after = 9
        self.assertEqual(self.capture()['0'].shape, (2, 2, 3))
        self.assertEqual(self.frames, 9)
        self.assertEqual(self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS'], [])

    def test_overview_keeps_video_resolution_and_model_cameras_keep_default(self):
        calls = []

        def resize(rgb, np_module, width=256, height=256):
            calls.append((width, height))
            return np.full((height, width, 3), 1, dtype=np.uint8)

        self.namespace['resize_rgb_for_model'] = resize
        views = {name: {'window': self.window, 'viewport': object()}
                 for name in ('0', '1', '2', 'overview')}
        frames = self.namespace['capture_rgb_from_persistent_viewports'](
            views, SimpleNamespace(update=self.update), self.submit, np,
        )
        self.assertEqual(frames['overview'].shape, (768, 768, 3))
        for name in ('0', '1', '2'):
            self.assertEqual(frames[name].shape, (256, 256, 3))
        self.assertEqual(calls.count((256, 256)), 3)

    def test_repeated_captures_keep_windows_visible_and_release_helpers(self):
        for _ in range(100):
            self.capture()
            self.assertEqual(self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS'], [])
        self.assertTrue(self.window.visible)
        self.assertEqual(self.window.changes, [])

    def test_timeout_keeps_helper_alive_until_late_callback(self):
        self.complete_after = None
        with self.assertRaisesRegex(RuntimeError, 'timeout after .* render updates'):
            self.capture()
        self.assertLessEqual(self.frames, 51)
        holders = self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS']
        self.assertEqual(len(holders), 1)
        self.assertFalse(holders[0]['done'])
        self.assertIn('helper', holders[0])
        self.pending[0][1](None, 16, 2, 2, None)
        self.namespace['cleanup_viewport_capture_helpers']()
        self.assertEqual(holders, [])

    def test_black_readback_is_recaptured_with_fresh_pixels(self):
        attempts = []

        def pixels(*args):
            attempts.append(None)
            return np.full((2, 2, 4), int(len(attempts) > 1), dtype=np.uint8)

        self.namespace['capsule_to_numpy_rgba'] = pixels
        with contextlib.redirect_stdout(io.StringIO()):
            rgb = self.capture()['0']
        self.assertTrue(np.any(rgb))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS'], [])

    def test_persistent_black_readback_retries_are_bounded(self):
        self.namespace['capsule_to_numpy_rgba'] = lambda *args: np.zeros((2, 2, 4), dtype=np.uint8)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'all-zero RGB'):
            self.capture()
        self.assertEqual(self.frames, 3)
        self.assertEqual(self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS'], [])

    def test_opaque_black_render_is_delivered_without_fabricated_pixels(self):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        self.namespace['capsule_to_numpy_rgba'] = lambda *args: rgba
        with contextlib.redirect_stdout(io.StringIO()):
            rgb = self.capture()['0']
        np.testing.assert_array_equal(rgb, rgba[:, :, :3])
        self.assertEqual(self.frames, 3)
        self.assertEqual(self.namespace['PENDING_VIEWPORT_CAPTURE_HELPERS'], [])

    def test_bridge_returns_capture_error_to_client(self):
        source = Path(__file__).resolve().parents[1] / 'run_simulation.py'
        tree = ast.parse(source.read_text())
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                       and any(isinstance(child, ast.Constant)
                               and child.value == 'CameraCaptureError: '
                               for child in ast.walk(node)))
        # The production handler continues the server's command loop after
        # returning an error. Supply a one-iteration loop to exercise it here.
        loop = ast.For(target=ast.Name(id='_', ctx=ast.Store()),
                       iter=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                       body=[handler], orelse=[])
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        replies = queue.Queue()

        def fail_capture(**kwargs):
            raise RuntimeError('readback timeout')

        namespace = {
            'capture_rgb_from_persistent_viewports': fail_capture,
            'capture_views': {}, 'simulation_app': object(),
            'capture_viewport_to_buffer': object(), 'np': np,
            'CAPTURE_WAIT_FRAMES': 5, 'response_queue': replies,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(module, str(source), 'exec'), namespace)
        reply = replies.get_nowait()
        self.assertEqual(reply['type'], 'error')
        self.assertIn('CameraCaptureError: RuntimeError: readback timeout', reply['error'])


if __name__ == '__main__':
    unittest.main()
