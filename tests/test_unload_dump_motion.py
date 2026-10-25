"""Exercise the real dump coroutine without starting Isaac Sim."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np


RUNTIME = Path(__file__).resolve().parents[1] / "scripts/excavator_app/excavator_runtime.py"


def dump_runtime(control_hz, stalled=False):
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "execute_unload_bucket_dump_motion")
    real = np.array([0., 0., 0., np.radians(-120.)])
    ctrl = SimpleNamespace(name_to_idx={"bucket": 3}, q_cmd=real.copy())
    physics_dt = 1 / 60
    elapsed = [0.]

    def send(q, mode):
        ctrl.q_cmd = np.array(q).copy()
        return True, "ok"

    async def step(frames):
        for _ in range(frames):
            elapsed[0] += physics_dt
            if not stalled:
                real[:] += np.clip(ctrl.q_cmd - real, -np.radians(110.) * physics_dt,
                                   np.radians(110.) * physics_dt)

    def metrics(label):
        released = real[3] >= np.radians(45.)
        return {"bucket_from_pile_count": 0 if released else 100,
                "bin_from_pile_count": 100 if released else 0}

    ctrl.send_action = send
    failure = Mock()
    namespace = {
        "np": np, "math": math, "CTRL": ctrl, "STATE": {},
        "CONTROL_HZ": control_hz, "CONTROL_DT": 1 / control_hz,
        "control_step_frames": lambda: round(60 / control_hz),
        "UNLOAD_DUMP_OPEN_RATE_DEG_S": 150., "UNLOAD_DUMP_MAX_OPEN_SECONDS": 2.6,
        "UNLOAD_DUMP_METRIC_SAMPLE_FRAMES": 6, "UNLOAD_DUMP_STOP_BUCKET_FRACTION": .05,
        "UNLOAD_DUMP_FLOW_CENTER_BUCKET_DEG": -25.,
        "robot_articulation_action_ready": lambda: True,
        "sync_motion_start_q": lambda label: real.copy(),
        "clip_unload_dump_command": lambda q, reference: np.array(q).copy(),
        "get_real_joint_positions": lambda: real.copy(),
        "q_real_near_command": lambda q, reference: np.array(q).copy(),
        "motion_cancel_requested": lambda task_id: False,
        "step_updates": step, "record_phase_metrics": metrics,
        "dataset_record_sample_due": lambda: False,
        "record_unload_trajectory_sample_async": AsyncMock(),
        "deg_to_rad": np.radians, "rad_to_deg": np.degrees,
        "info_print": Mock(), "update_status": Mock(),
        "debug_diagnostics_enabled": lambda: False,
        "set_execution_failure_reason": failure,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNTIME), "exec"), namespace)
    return namespace[function.name], real, elapsed, failure


class UnloadDumpMotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_opening_duration_uses_physics_frames_at_both_control_rates(self):
        for control_hz in (30, 60):
            with self.subTest(control_hz=control_hz):
                dump, real, elapsed, failure = dump_runtime(control_hz)
                self.assertTrue(await dump(np.radians([0., 0., 0., 88.]), "unload_to_bin"))
                self.assertGreaterEqual(real[3], np.radians(45.))
                self.assertGreater(elapsed[0], 1.5)
                self.assertLess(elapsed[0], 2.7)
                failure.assert_not_called()

    async def test_timeout_with_loaded_stalled_bucket_is_not_success(self):
        dump, real, elapsed, failure = dump_runtime(30, stalled=True)
        self.assertFalse(await dump(np.radians([0., 0., 0., 88.]), "unload_to_bin"))
        self.assertGreaterEqual(elapsed[0], 2.6)
        self.assertAlmostEqual(float(np.degrees(real[3])), -120.)
        self.assertIn("unload_dump_incomplete", failure.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
