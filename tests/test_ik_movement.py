import unittest

import numpy as np

from scripts.excavator_app import ik_movement


class _Controller:
    def __init__(self):
        self.q_cmd = np.zeros(4, dtype=np.float32)

    @staticmethod
    def clip_limits(q):
        return np.asarray(q, dtype=np.float32)


class _Runtime:
    ACTION_READY_MIN_STABLE_FRAMES = 1
    ACTION_READY_STAGE_MAX_WAIT_FRAMES = 1
    PATH_CHECK_SAMPLES = 24

    def __init__(self):
        self.np = np
        self.CTRL = _Controller()
        self.STATE = {}
        self.clearance_calls = 0
        self.direct_calls = 0
        self.failure_reason = ""

    @staticmethod
    def consume_motion_continuous_handoff(_stage_name):
        return False

    @staticmethod
    def articulation_action_ready_detail():
        return {"ready": True}

    @staticmethod
    def format_action_ready_detail(_detail):
        return "ready"

    async def wait_for_articulation_action_ready(self, *_args, **_kwargs):
        return True, "ok", {"ready": True}

    @staticmethod
    def update_status(*_args, **_kwargs):
        return None

    @staticmethod
    def sync_motion_start_q(_stage_name):
        return np.zeros(4, dtype=np.float32)

    @staticmethod
    def clip_command_near(q, reference=None):
        return np.asarray(q, dtype=np.float32)

    @staticmethod
    def active_loaded_route_fast_exec(_stage_name):
        return False

    @staticmethod
    def cache_active_stage_trace_points(*_args, **_kwargs):
        return []

    @staticmethod
    def current_trace_mode():
        return 0

    @staticmethod
    def info_print(*_args, **_kwargs):
        return None

    @staticmethod
    def q_deg_values(q, **_kwargs):
        return list(np.asarray(q, dtype=float))

    @staticmethod
    def path_phase_check(*_args, **_kwargs):
        return True, "ok", 24, {}

    @staticmethod
    def path_obstacle_check(*_args, **_kwargs):
        return False, "rigid_obstacle", 5, {"obstacle": "truck"}

    @staticmethod
    def format_obstacle_report(*_args, **_kwargs):
        return "predicted truck contact"

    @staticmethod
    def strict_path_precheck_phase(_stage_name):
        return True

    async def move_to_profile_with_clearance(self, *_args, **_kwargs):
        self.clearance_calls += 1
        return True

    async def move_to_profile(self, *_args, **_kwargs):
        self.direct_calls += 1
        return True

    def set_execution_failure_reason(self, reason):
        self.failure_reason = str(reason)


class PlannedStageRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_pre_dig_obstacle_uses_clearance_route_before_failure(self):
        rt = _Runtime()

        ok = await ik_movement.move_planned_stage(
            rt,
            "pre_dig",
            np.ones(4, dtype=np.float32),
            1.0,
        )

        self.assertTrue(ok)
        self.assertEqual(rt.clearance_calls, 1)
        self.assertEqual(rt.direct_calls, 0)
        self.assertEqual(rt.failure_reason, "")


if __name__ == "__main__":
    unittest.main()
