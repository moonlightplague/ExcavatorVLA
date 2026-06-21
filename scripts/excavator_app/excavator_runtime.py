import asyncio
import hashlib
import json
import math
import os
import sys
import time
import traceback
import numpy as np
import builtins

import omni.usd
import omni.kit.app
import omni.ui as ui
try:
    import omni.timeline
    HAS_OMNI_TIMELINE = True
except Exception:
    HAS_OMNI_TIMELINE = False
try:
    import carb
except Exception:
    carb = None

from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

from . import auto_dataset_collect
from . import ik_calculation
from . import ik_movement
from . import trace_showing


APP_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_PARENT) if os.path.basename(PACKAGE_PARENT) == "scripts" else PACKAGE_PARENT


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# ============================================================
# Stop previous loop
# ============================================================

if hasattr(builtins, "_EXCAVATOR_MOUSE_SLIDER_STATE"):
    try:
        old_state = builtins._EXCAVATOR_MOUSE_SLIDER_STATE
        old_state["running"] = False
        old_state["follow"] = False
        old_state["auto_collect_stop_requested"] = True
        old_state["robot_state_reads_enabled"] = False
        old_state["robot_state_shutdown"] = True
        for task_name, task in list((old_state.get("async_tasks") or {}).items()):
            try:
                if task is not None and hasattr(task, "done") and not task.done():
                    print("[INFO]", "[TASK CANCEL]", f"name={task_name}", "reason=runtime_reload")
                    task.cancel()
            except Exception as e:
                print("[INFO]", "[WARN] task cancel during reload failed:", task_name, type(e).__name__, e)
        old_state["async_tasks"] = {}
        old_window = getattr(builtins, "_EXCAVATOR_UI_WINDOW", None)
        if old_window is not None:
            try:
                old_window.visible = False
            except Exception:
                pass
    except Exception:
        pass

builtins._EXCAVATOR_MOUSE_SLIDER_STATE = {
    "running": True,
    "follow": False,
    "trace": False,
    "trace_mode": 0,
    "trace_bucket_points": [],
    "trace_bucket_last_point": None,
    "trace_planned_bucket_points": [],
    "trace_render_dirty": True,
    "trace_render_signature": None,
    "trace_active_motion": None,
    "trace_building_plan": False,
    "trace_no_plan_notice_time": 0.0,
    "trace_no_plan_notice_shown": False,
    "excavator_render_mode": True,
    "request_calibrate": False,
    "request_home": False,
    "request_print": False,
    "joint_cmd": {
        "swing": 0.0,
        "boom": 0.0,
        "arm": 0.0,
        "bucket": 0.0,
    },
    "target_cmd": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "manual_unload_override_enabled": False,
    "manual_unload_point": None,
    "manual_unload_inner_size": None,
    "manual_unload_z_range": None,
    "manual_unload_radius": 0.45,
    "manual_unload_mesh_shrink_d": 0.12,
    "manual_unload_range_shape": "circle",
    "manual_unload_selected_size_xy": None,
    "manual_unload_polygon_xy": None,
    "manual_unload_selected_hull_xy": None,
    "manual_unload_source": "",
    "manual_unload_selected_path": "",
    "last_unload_model_xyz": None,
    "last_unload_model_z_range": None,
    "last_unload_model_radius": 0.45,
    "last_unload_model_shrink_d": 0.12,
    "last_unload_sync_time": 0.0,
    "speed_multiplier": 1.0,
    "last_status_time": 0.0,
    "status_interval": 0.5,
    "last_status_text": None,
    "last_status_repeat_time": 0.0,
    "status_repeat_interval": 2.5,
    "log_mode": "normal",
    "log_last_print_time": {},
    "log_suppressed_count": {},
    "log_repeat_interval": 0.35,
    "last_trace_time": 0.0,
    "trace_interval": 0.15,
    # target sync throttle
    "last_target_sync_time": 0.0,
    "target_sync_interval": 0.05,
    "last_target_model_xyz": None,
    # follow throttle
    "last_follow_time": 0.0,
    # use a safe default follow interval (will be honored later)
    "follow_interval": 1.0 / 60.0,
    "active_task_id": 0,
    "active_task_name": "idle",
    "async_tasks": {},
    "ui_syncing": False,
    "last_slider_sync_time": 0.0,
    "slider_sync_interval": 0.10,
    "manual_override": False,
    "manual_editing_until": 0.0,
    "manual_joint_active": False,
    "manual_joint_target": None,
    "manual_last_status_time": 0.0,
    "manual_recovery_last_status_time": 0.0,
    "dig_plan_target": None,
    "dig_plan_sequence": None,
    "dig_plan_points": None,
    "dig_plan_step_index": 0,
    "active_plan_stage_index": -1,
    "dig_plan_start_q": None,
    "dig_plan_trace_points": [],
    "dig_plan_trace_stage_breaks": [],
    "dig_plan_planning_active": False,
    "dig_plan_planning_version": 0,
    "dig_plan_planning_source": "",
    "dig_plan_best_failure": None,
    "dig_plan_last_build_ms": 0.0,
    "current_dig_plan": None,
    "last_auto_preflight": None,
    "last_auto_dig_target_scores": [],
    "last_auto_unload_scores": [],
    "active_unload_landing_point": None,
    "active_unload_release_point": None,
    "planner_uses_command_state": True,
    "last_action_diag_time": 0.0,
    "last_action_q": None,
    "last_action_unclipped_q": None,
    "last_action_time": 0.0,
    "last_action_mode": "idle",
    "robot_state_reads_enabled": False,
    "robot_state_shutdown": False,
    "last_valid_real_q": None,
    "last_valid_real_q_time": 0.0,
    "last_joint_read_warn_time": 0.0,
    "freeze_candidate_since": 0.0,
    "freeze_last_real_q": None,
    "freeze_last_real_time": 0.0,
    "freeze_last_print_time": 0.0,
    "freeze_last_signature": None,
    "sand_contact_stage": "",
    "sand_contact_stage_started": 0.0,
    "sand_contact_last_progress_time": 0.0,
    "sand_contact_last_log_time": 0.0,
    "sand_contact_start_bucket_from_pile": 0,
    "sand_contact_start_pile": 0,
    "sand_contact_start_spill_from_pile": 0,
    "sand_contact_start_tip": None,
    "sand_contact_start_load": None,
    "sand_contact_progress_bucket_base": 0,
    "sand_contact_progress_pile_base": 0,
    "sand_contact_progress_spill_base": 0,
    "sand_contact_progress_tip_base": None,
    "sand_contact_progress_load_base": None,
    "sand_contact_last_progress_seen": False,
    "sand_contact_last_report": None,
    "dataset_recording": False,
    "dataset_path": "excavator_dataset.jsonl",
    "dataset_event_path": "excavator_events.jsonl",
    "dataset_meta_path": "",
    "debug_timeline_path": "",
    "debug_timeline_last_error_time": 0.0,
    "dataset_episode_id": int(time.time()),
    "dataset_episode_uid": "",
    "dataset_episode_dir": "",
    "dataset_episode_start_time": 0.0,
    "dataset_episode_freezes": 0,
    "dataset_last_sample_time": 0.0,
    "dataset_sample_interval": 0.20,
    "dataset_last_q_cmd": None,
    "dataset_last_q_real": None,
    "dataset_current_q_goal": None,
    "last_execution_failure_reason": "",
    "dataset_samples": 0,
    "dataset_last_error_time": 0.0,
    "auto_collect_active": False,
    "auto_collect_stop_requested": False,
    "auto_collect_requested": 0,
    "auto_collect_attempts": 0,
    "auto_collect_successes": 0,
    "auto_collect_failures": 0,
    "auto_collect_rejections": 0,
    "auto_collect_planning_diagnostics": 0,
    "auto_collect_run_dir": "",
    "auto_collect_run_id": "",
    "auto_collect_last_result": "",
    "auto_collect_task": None,
    "auto_collect_sand_reset_done": False,
    "auto_collect_initial_pose": None,
    "auto_collect_initial_pose_id": "",
    "last_dig_plan_candidates": [],
    "dig_plan_candidate": None,
    "dataset_sand_metrics_path": "",
    "dataset_initial_pile_particle_ids": None,
    "dataset_initial_pile_particle_count": 0,
    "dataset_max_bucket_particles": 0,
    "dataset_max_bucket_from_pile_particles": 0,
    "dataset_lift_bucket_from_pile_particles": 0,
    "dataset_final_bin_from_pile_particles": 0,
    "dataset_final_spill_from_pile_particles": 0,
    "dataset_max_joint_error_deg": 0.0,
    "dataset_max_action_speed": 0.0,
    "dataset_phase_metrics": {},
    "sand_metrics_last_time": 0.0,
    "sand_metrics_last": None,
    "sand_site_stable_reset_done": False,
    "sand_site_reset_active": False,
    "sand_site_last_reset_label": "",
    "replay_active": False,
    "replay_last_path": "",
}

STATE = builtins._EXCAVATOR_MOUSE_SLIDER_STATE


def runtime_module():
    return sys.modules[__name__]


# ============================================================
# Config
# ============================================================

CONTROL_HZ = 30
CONTROL_DT = 1.0 / CONTROL_HZ

GROUND_TOP_Z = 0.0
GROUND_SIZE = 30.0
GROUND_THICKNESS = 0.08
PARKING_GROUND_MARGIN_XY = 0.80
PARKING_GROUND_MIN_SIZE_XY = 3.20
PARKING_GROUND_BASE_MAX_SIZE_XY = 8.00
PARKING_GROUND_SIZE_MULTIPLIER_XY = 18.0
PARKING_GROUND_MAX_SIZE_XY = PARKING_GROUND_BASE_MAX_SIZE_XY * PARKING_GROUND_SIZE_MULTIPLIER_XY
ROBOT_PARK_SPAWN_LIFT_Z = 1.60
ROBOT_OPERATIONAL_SUPPORT_CLEARANCE_Z = 0.45
ROBOT_OPERATIONAL_CLEARANCE_TOL_Z = 0.035
ROBOT_GROUND_CONTACT_EPSILON_Z = 0.005
ROBOT_PARK_CONTACT_SKIN = ROBOT_OPERATIONAL_SUPPORT_CLEARANCE_Z
ROBOT_PARK_FULL_COLLISION_CLEARANCE_Z = ROBOT_GROUND_CONTACT_EPSILON_Z
ROBOT_PARK_NEVER_LOWER = False
ROBOT_DROP_SETTLE_FRAMES = 180
ROBOT_GROUND_TARGET_CLEARANCE_Z = ROBOT_OPERATIONAL_SUPPORT_CLEARANCE_Z
ROBOT_GROUND_FULL_TARGET_CLEARANCE_Z = ROBOT_GROUND_CONTACT_EPSILON_Z
ROBOT_GROUND_FLOAT_TOL_Z = ROBOT_OPERATIONAL_SUPPORT_CLEARANCE_Z + 0.08
ROBOT_GROUND_PENETRATION_TOL_Z = ROBOT_OPERATIONAL_CLEARANCE_TOL_Z
ROBOT_GROUND_FULL_PENETRATION_TOL_Z = 0.025
ROBOT_GROUND_MAX_SCRIPT_CORRECTION_Z = 1.50
VALID_BBOX_ABS_MAX = 1.0e5

TARGET_MIN_Z = GROUND_TOP_Z + 0.04
TARGET_RADIUS = 0.16
TARGET_COLOR_DEFAULT = (1.0, 0.05, 0.05)
TARGET_COLOR_REACHABLE = (0.1, 0.85, 0.25)
TARGET_COLOR_UNREACHABLE = (1.0, 0.75, 0.05)
UNLOAD_MARKER_RADIUS = 0.13
UNLOAD_MARKER_COLOR = (0.05, 0.70, 1.0)
UNLOAD_RANGE_COLUMN_COLOR = (0.0, 0.55, 1.0)
UNLOAD_RANGE_COLUMN_OPACITY = 0.07
UNLOAD_RANGE_GUIDE_WIDTH = 0.040
UNLOAD_RANGE_VISUAL_MAX_VERTICES = 32
UNLOAD_RANGE_CIRCLE_SEGMENTS = 20

BUCKET_SAND_COLLIDER_CONTACT_OFFSET = 0.055
BUCKET_SAND_COLLIDER_REST_OFFSET = 0.010
BUCKET_SAND_SDF_RESOLUTION = 256
BUCKET_SAND_SDF_SUBGRID_RESOLUTION = 12
BUCKET_SAND_USE_VISUAL_MESH_COLLISION = True

MANUAL_UI_SYNC_HOLD_SECONDS = 1.0
MANUAL_SPEED_SCALE = 0.18
MANUAL_SPEED_MULTIPLIER_CAP = 1.0
MANUAL_SPEED_MULTIPLIER_FLOOR = 0.04
MANUAL_TARGET_DEADBAND_DEG = 0.35
MANUAL_STATUS_INTERVAL = 0.75
MANUAL_RECOVERY_BUCKET_PENETRATION_Z = 0.08
MANUAL_RECOVERY_SWING_HOLD_DEG = 0.25
MANUAL_RECOVERY_STATUS_INTERVAL = 0.75
AUTO_PLANNING_IGNORES_SAND_REACTION = True
BASE_GROUND_BLOCKS_COMMANDS = False

FREEZE_CMD_ERR_DEG = 2.0
FREEZE_STALL_MOTION_DEG = 0.12
FREEZE_MIN_DURATION = 0.45
FREEZE_SWING_ONLY_MIN_DURATION = 2.25
FREEZE_BUCKET_CUT_MIN_DURATION = 1.35
FREEZE_PRINT_INTERVAL = 1.0
SAND_CONTACT_PHASES = {"insert_cut", "pull_mid_cut", "pull_exit_cut"}
SAND_CONTACT_BUCKET_PROGRESS_MIN = 80
SAND_CONTACT_PILE_PROGRESS_MIN = 60
SAND_CONTACT_TIP_PROGRESS_MIN_M = 0.015
SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M = 0.04
SAND_CONTACT_BOOM_HARD_ERR_DEG = 10.0
SAND_CONTACT_Q_LAG_ACCEPT_DEG = 16.0
SAND_CUT_NO_PROGRESS_TIMEOUT = 3.5
SAND_CONTACT_LOG_INTERVAL = 0.70
SAND_CONTACT_STAGE_MIN_SECONDS = 0.65
SAND_CONTACT_ADVANCE_BUCKET_MIN = 900
SAND_CONTACT_ADVANCE_PILE_MIN = 900
SAND_CONTACT_ADVANCE_TIP_MIN_M = 0.18
SAND_CONTACT_ADVANCE_FALLBACK_BUCKET_MIN = 300
SAND_CONTACT_ADVANCE_FALLBACK_PILE_MIN = 300
SAND_CONTACT_MAX_STAGE_WALL_SECONDS = 8.0
MANUAL_FREEZE_STOP_ENABLED = True
AUTO_FREEZE_STOP_ENABLED = True

LIMIT_POLICY = "override"
PHYSX_TIMESTEPS_PER_SECOND = 60

AUTO_COLLECT_DATASET_ROOT = os.environ.get(
    "EXCAVATOR_DATASET_ROOT",
    project_path("excavator_auto_dataset"),
)
AUTO_COLLECT_DEFAULT_COUNT = 10
AUTO_COLLECT_MAX_PLAN_RETRIES = 3
AUTO_COLLECT_BETWEEN_EPISODE_FRAMES = 90
AUTO_COLLECT_SAND_RESET_POLICY = "once_per_run_after_home"
AUTO_COLLECT_REUSE_READY_SAND_RESET = True
AUTO_COLLECT_PRE_RESET_SETTLE_FRAMES = 45
AUTO_COLLECT_RESET_SETTLE_FRAMES = 180
AUTO_COLLECT_HOME_SECONDS = 1.40
DIRECT_HOME_SETTLE_FRAMES = 10
DIRECT_INITIAL_POSE_SETTLE_FRAMES = 18
DIRECT_PLAN_END_HOME_SETTLE_FRAMES = 8
AUTO_COLLECT_TARGET_CENTER = np.array([0.0, -6.7, 0.0], dtype=np.float32)
AUTO_COLLECT_TARGET_RADIUS_X = 0.82
AUTO_COLLECT_TARGET_RADIUS_Y = 0.82
AUTO_COLLECT_TARGET_DEPTHS = [0.24, 0.32, 0.40, 0.46]
AUTO_COLLECT_TARGET_MIN_Z = GROUND_TOP_Z + 0.04
AUTO_COLLECT_TARGET_MAX_Z = 5.00
AUTO_COLLECT_SCHEMA = "excavator_auto_state_action_v1"
DATASET_TRAJECTORY_FORMAT = "compact_jsonl_v2"
DATASET_DEBUG_PLAN_FILE = "plan_debug.json"
DATASET_DEBUG_TIMELINE_FILE = "debug_timeline.jsonl"
AUTO_COLLECT_MAX_ATTEMPT_MULTIPLIER = 5
PLANNER_VERSION = "dig_plan_v3_shared_auto_flatfill"
QUALITY_GATE_VERSION = "quality_gate_v2_particles_no_freeze"
AUTO_PREFLIGHT_MIN_PARTICLES = 1000
AUTO_DIG_GRID_SIZE = 7
AUTO_DIG_TOPK_TARGETS = 8
AUTO_DIG_CORE_NORM_MAX = 0.62
AUTO_DIG_DENSITY_RADIUS = 0.34
AUTO_DIG_MIN_LOCAL_PARTICLES = 24
AUTO_UNLOAD_GRID_SIZE = 5
AUTO_UNLOAD_WALL_MARGIN = 0.18
AUTO_UNLOAD_EMPTY_CELL_HEIGHT = -1.0
AUTO_UNLOAD_FILL_HEIGHT_WEIGHT = 1.0
AUTO_UNLOAD_CENTER_WEIGHT = 0.12
AUTO_UNLOAD_MOTION_WEIGHT = 0.18
AUTO_COLLECT_INITIAL_POSES_DEG = [
    {"id": "safe_front", "swing": 0.0, "boom": 42.0, "arm": -58.0, "bucket": -18.0},
    {"id": "safe_left_45", "swing": -45.0, "boom": 46.0, "arm": -62.0, "bucket": -20.0},
    {"id": "safe_right_45", "swing": 45.0, "boom": 46.0, "arm": -62.0, "bucket": -20.0},
    {"id": "safe_left_90", "swing": -90.0, "boom": 52.0, "arm": -68.0, "bucket": -22.0},
    {"id": "safe_right_90", "swing": 90.0, "boom": 52.0, "arm": -68.0, "bucket": -22.0},
    {"id": "safe_high_front", "swing": 0.0, "boom": 56.0, "arm": -70.0, "bucket": -24.0},
]

SAND_PARTICLE_PATH_SUFFIX = "RealSandParticles"
SAND_PARTICLE_MASS_DEFAULT = 0.535
SAND_METRICS_INTERVAL = 0.45
AUTO_RESET_SAND_AFTER_WORLD_READY = False
AUTO_RESET_SAND_AFTER_UI_READY = True
AUTO_RESET_SAND_UI_READY_DELAY_FRAMES = 60
SAND_RESET_SETTLE_MIN_FRAMES = 240
SAND_RESET_SETTLE_MAX_FRAMES = 840
SAND_RESET_STABLE_WINDOW_FRAMES = 30
SAND_RESET_STABLE_MEAN_DISPLACEMENT = 0.004
SAND_RESET_STABLE_P95_DISPLACEMENT = 0.020
SAND_RESET_ESCAPE_Z = -1.0
SAND_RESET_ESCAPE_CHECK_FRAMES = 150
SAND_RESET_MAX_NATIVE_ATTEMPTS = 2
SAND_BUCKET_LOCAL_MIN = np.array([-0.12, -0.58, -0.30], dtype=np.float32)
SAND_BUCKET_LOCAL_MAX = np.array([0.90, 0.58, 0.55], dtype=np.float32)
SAND_PILE_CENTER = AUTO_COLLECT_TARGET_CENTER.copy()
SAND_PILE_RADIUS_X = 1.18
SAND_PILE_RADIUS_Y = 1.18
SAND_PILE_Z_MIN = -0.05
SAND_PILE_Z_MAX = 2.20
SAND_BIN_HALF_X = 1.25
SAND_BIN_HALF_Y = 1.25
SAND_BIN_Z_MIN = -0.05
SAND_BIN_Z_MAX = 2.20
QUALITY_TARGET_BUCKET_PARTICLES = 25
QUALITY_MIN_BUCKET_PARTICLES = 5
QUALITY_MIN_DUMP_PARTICLES = 3
QUALITY_MIN_SCORE = 55.0
QUALITY_MAX_SPILL_RATIO = 0.65

DESIRED_LIMITS_DEG = {
    "swing": (-180.0, 180.0),
    "boom": (-75.0, 75.0),
    "arm": (-95.0, 95.0),
    "bucket": (-120.0, 90.0),
}

DQ_MAX = {
    "swing": 1.0,
    "boom": 0.7,
    "arm": 0.8,
    "bucket": 1.0,
}

ACTION_READY_MIN_STABLE_FRAMES = 5
ACTION_READY_MAX_WAIT_FRAMES = 180
ACTION_READY_STAGE_MAX_WAIT_FRAMES = 150
MOVE_REACH_WAIT_MIN_FRAMES = 12
MOVE_REACH_WAIT_MAX_FRAMES = 180
MOVE_REACH_EXTRA_TIME_RATIO = 0.35
MOVE_DURATION_MARGIN_SECONDS = 0.25

JOINT_DRIVE_GAINS = {
    "swing": {"stiffness": 1.8e7, "damping": 1.8e6, "max_force": 8.0e7},
    "boom": {"stiffness": 1.2e7, "damping": 1.2e6, "max_force": 6.0e7},
    "arm": {"stiffness": 9.0e6, "damping": 9.0e5, "max_force": 4.0e7},
    "bucket": {"stiffness": 6.0e6, "damping": 6.0e5, "max_force": 3.0e7},
}

HOME_SAFE_DEG = {
    "boom": 28.0,
    "arm": -62.0,
    "bucket": -8.0,
}

SPEED_MULTIPLIER_MIN = 0.1
SPEED_MULTIPLIER_MAX = 10.0

# Horizontal bucket-load direction in the boom/arm/bucket planar IK frame.
# Lift/carry uses a small mouth-up bias so carried sand is retained.
BUCKET_LIFT_LEVEL_WORLD_DEG = 0.0
BUCKET_LIFT_LEVEL_TOL_DEG = 4.0
BUCKET_CARRY_HOLD_TILT_DEG = 18.0
BUCKET_CARRY_HOLD_TOL_DEG = 8.0
BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z = 0.08
BUCKET_DIG_APPROACH_WORLD_DEG = -78.0
BUCKET_DIG_INSERT_WORLD_DEG = -96.0
BUCKET_DIG_PULL_WORLD_DEG = -108.0
BUCKET_DIG_EXIT_WORLD_DEG = -118.0
BUCKET_UNLOAD_DUMP_DEG = 82.0
UNLOAD_DUMP_BUCKET_TOL_DEG = 18.0
UNLOAD_DUMP_ACCEPT_ERR = 0.45
UNLOAD_DUMP_SECONDS = 0.90
BUCKET_UNLOAD_SETTLE_FRAMES = 10
UNLOAD_DUMP_SETTLE_MIN_FRAMES = 60
UNLOAD_DUMP_SETTLE_MAX_FRAMES = 180
UNLOAD_DUMP_SETTLE_SAMPLE_FRAMES = 12
UNLOAD_DUMP_STABLE_SAMPLES = 4
UNLOAD_TARGET_CLEARANCE_Z = 1.20
UNLOAD_TARGET_MIN_Z = 1.35
UNLOAD_BIN_DUMP_WALL_CLEARANCE_Z = 0.35
UNLOAD_BIN_SAFE_XY_MARGIN = 0.08
UNLOAD_DROP_XY_TOL = 0.22
UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z = 0.18
UNLOAD_DROP_GRAVITY = 9.81
UNLOAD_DROP_ROLL_OFFSET_BASE = 0.24
UNLOAD_DROP_ROLL_OFFSET_PER_M = 0.10
UNLOAD_DROP_TANGENTIAL_GAIN = 0.36
UNLOAD_DROP_MAX_ROLL_OFFSET = 1.25
UNLOAD_DROP_MAX_XY_CORRECTION = 1.35
UNLOAD_DROP_IK_CORRECTION_ITERS = 5
UNLOAD_PRE_DUMP_ALIGN_MIN_SECONDS = 0.35
UNLOAD_PRE_DUMP_ALIGN_MAX_SECONDS = 1.80
UNLOAD_PRE_DUMP_NON_BUCKET_TOL_DEG = 0.75
UNLOAD_ACTUAL_BUCKET_XY_MARGIN = 0.08
UNLOAD_ACTUAL_LOAD_MARKER_XY_TOL = 0.42
UNLOAD_ACTUAL_MIN_ABOVE_WALL_Z = 0.04
UNLOAD_FINAL_SWING_TOL_DEG = 0.75
UNLOAD_FINAL_JOINT_TOL_DEG = 2.50
UNLOAD_FINAL_LOAD_XY_TOL = 0.42
UNLOAD_FINAL_LOAD_Z_CLEARANCE = 0.05
UNLOAD_POINT_DEFAULT_RADIUS = 0.45
UNLOAD_SELECTED_EDGE_MARGIN = 0.12
DEFAULT_UNLOAD_SOURCE_MESH_PATH = "/World/SandSite/UnloadBin"

DIG_PLAN_CANDIDATES = [
    {
        "id": "balanced",
        "approach_offset": 0.25,
        "pre_z": 0.60,
        "contact_z": 0.03,
        "insert_depth": 0.14,
        "mid_pull": 0.35,
        "mid_depth": 0.28,
        "exit_pull": 0.55,
        "exit_depth": 0.08,
        "curl_z": 0.08,
        "lift_height": 0.70,
        "unload_height_delta": 0.00,
        "unload_dump_deg": 82.0,
    },
    {
        "id": "high_lift",
        "approach_offset": 0.32,
        "pre_z": 0.82,
        "contact_z": 0.06,
        "insert_depth": 0.12,
        "mid_pull": 0.35,
        "mid_depth": 0.24,
        "exit_pull": 0.58,
        "exit_depth": 0.06,
        "curl_z": 0.16,
        "lift_height": 1.05,
        "unload_height_delta": 0.18,
        "unload_dump_deg": 86.0,
    },
    {
        "id": "deep_short",
        "approach_offset": 0.20,
        "pre_z": 0.70,
        "contact_z": 0.04,
        "insert_depth": 0.18,
        "mid_pull": 0.28,
        "mid_depth": 0.34,
        "exit_pull": 0.46,
        "exit_depth": 0.10,
        "curl_z": 0.10,
        "lift_height": 0.88,
        "unload_height_delta": 0.10,
        "unload_dump_deg": 88.0,
    },
    {
        "id": "shallow_long",
        "approach_offset": 0.42,
        "pre_z": 0.78,
        "contact_z": 0.08,
        "insert_depth": 0.12,
        "mid_pull": 0.48,
        "mid_depth": 0.24,
        "exit_pull": 0.72,
        "exit_depth": 0.06,
        "curl_z": 0.18,
        "lift_height": 1.15,
        "unload_height_delta": 0.22,
        "unload_dump_deg": 86.0,
    },
]


# ============================================================
# Globals
# ============================================================

stage = omni.usd.get_context().get_stage()

ROBOT_ROOT = None
ROBOT_BASE = None
BUCKET_LINK = None
CONTROL_ROOT = None
TARGET_PATH = None
UNLOAD_MARKER_PATH = None

JOINT_PATHS = {}
LINK_PATHS = {}
FINAL_LIMITS_DEG = {}
FINAL_LIMITS_RAD = {}

WINDOW = None
STATUS_LABEL = None
UI_STATUS_MAX_CHARS = 132

ROBOT = None
DOF_ORDER = ["swing", "boom", "arm", "bucket"]
DOF_NAME_TO_REAL_IDX = {}
JOINT_INDICES = None

IK_MODEL = None
TRACE_ROOT = None
TRACE_PRIMS = []
TRACE_COUNT = 16

SLIDER_MODELS = {}
TARGET_MODELS = {}
UNLOAD_MODELS = {}


# ============================================================
# Helpers
# ============================================================

def sdf_path(path):
    if isinstance(path, Sdf.Path):
        return path
    if hasattr(path, "GetPath"):
        return path.GetPath()
    return Sdf.Path(str(path))


def get_prim(path):
    return stage.GetPrimAtPath(sdf_path(path))


async def step_updates(n=1):
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


def deg_to_rad(x):
    return float(x) * math.pi / 180.0


def rad_to_deg(x):
    return float(x) * 180.0 / math.pi


def safe_float(x, default=0.0):
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


LOG_MODES = ["quiet", "normal", "debug", "trace"]
LOG_IMPORTANT_TOKENS = [
    "[ERROR]",
    "[WARN]",
    "[GROUND WARN]",
    "[GUARD]",
    "[FREEZE]",
    "[AUTO FREEZE STOP]",
    "[MANUAL FREEZE STOP]",
    "[MOVE VERIFY FAILED]",
    "[GUARD BLOCKED]",
    " BLOCKED",
    "FAILED",
    "UNREACHABLE",
    "not ready",
]
LOG_NORMAL_TOKENS = [
    "Ready.",
    "Slider UI exited.",
    "[DIG PLAN]",
    "[DIG STEP",
    "[DIG FINISHED]",
    "[AUTO DATASET]",
    "[REPLAY]",
    "[UNLOAD DUMP]",
    "[UNLOAD TARGET]",
    "[UNLOAD ARRIVAL]",
    "[SAND RESET DONE]",
]
LOG_DEBUG_TOKENS = [
    "IK Follow",
    "IK solve failed",
    "==========",
    "[SAND PHYSX]",
    "[SAND RESET SETTLE]",
    "[SAND RESET NATIVE]",
    "[SAND RESET RETRY]",
    "[GROUND CONTACT DIAG]",
    "[GROUND PARK]",
    "[GROUND SETTLE",
    "[PARKING GROUND]",
    "[COLLISION OFF]",
    "[ROBOT BUCKET",
    "[ROBOT BUCKET DEINSTANCE]",
    "[LIMIT]",
    "[JOINT LIMIT",
    "[MOTION DIAG]",
    "[MOVE]",
    "[MOVE CARRY",
    "[MANUAL SMOOTH]",
    "[PATH DIRECT]",
    "[PATH VIA",
    "[PATH ROUTE",
    "[PATH OBSTACLE",
    "[PATH PREDICTED",
    "[PRE DIG COORD ALIGN]",
    "[SWING PRE-REBASE]",
    "[SWING EDGE]",
    "[SWING REBASE]",
    "[DIG IK]",
    "[DIG Q]",
    "[DIG TARGET GUARD]",
    "[DIG WORLD ANGLE]",
    "[DIG CARRY TARGET]",
    "[DIG BUCKET HOLD TARGET]",
    "[DIG PLAN CANDIDATE",
    "[DIG PLAN SELECT]",
    "[UNLOAD TARGET]",
    "[UNLOAD MARKER]",
    "[UNLOAD ALIGN]",
    "[UNLOAD DROP]",
    "[UNLOAD ACTUAL GATE]",
    "[UNLOAD READY]",
    "[UNLOAD ROUTE]",
    "[UNLOAD DUMP PLAN]",
    "[UNLOAD PRE-DUMP ALIGN]",
    "ROBOT dof_names",
    "DOF_NAME_TO_REAL_IDX",
    "JOINT_INDICES",
]


def normalize_log_mode(mode):
    text = str(mode).strip().lower()
    return text if text in LOG_MODES else "normal"


def current_log_mode():
    mode = normalize_log_mode(STATE.get("log_mode", "normal"))
    STATE["log_mode"] = mode
    return mode


def log_text_from_args(args, kwargs=None):
    sep = " "
    if isinstance(kwargs, dict):
        sep = str(kwargs.get("sep", " "))
    return sep.join(str(x) for x in args)


def log_matches_any(text, tokens):
    return any(token in text for token in tokens)


def log_signature(text):
    if len(text) <= 220:
        return text
    return text[:220]


def should_emit_log(text, force=False):
    if force:
        return True

    mode = current_log_mode()
    important = log_matches_any(text, LOG_IMPORTANT_TOKENS)
    if mode == "quiet" and not important:
        return False

    debug_like = log_matches_any(text, LOG_DEBUG_TOKENS)
    if mode == "normal" and debug_like and not important and not log_matches_any(text, LOG_NORMAL_TOKENS):
        return False

    if mode in ("normal", "quiet") and not important:
        now = time.time()
        signature = log_signature(text)
        last = STATE.setdefault("log_last_print_time", {})
        suppressed = STATE.setdefault("log_suppressed_count", {})
        interval = float(STATE.get("log_repeat_interval", 0.35))
        last_time = float(last.get(signature, 0.0))
        if now - last_time < interval:
            suppressed[signature] = int(suppressed.get(signature, 0)) + 1
            return False
        last[signature] = now

    return True


def info_print(*args, **kwargs):
    force_log = bool(kwargs.pop("force_log", False))
    text = log_text_from_args(args, kwargs)
    if should_emit_log(text, force=force_log):
        builtins.print("[INFO]", *args, **kwargs)


def set_log_mode(mode, announce=True):
    mode = normalize_log_mode(mode)
    STATE["log_mode"] = mode
    STATE["log_last_print_time"] = {}
    STATE["log_suppressed_count"] = {}
    if announce:
        info_print("[LOG MODE]", f"mode={mode}", force_log=True)
        update_status(f"[LOG MODE] {mode}", force=True)
    return mode


def toggle_debug_log():
    mode = current_log_mode()
    return set_log_mode("normal" if mode == "debug" else "debug")


def toggle_quiet_log():
    mode = current_log_mode()
    return set_log_mode("normal" if mode == "quiet" else "quiet")


def print_log_state():
    suppressed = STATE.get("log_suppressed_count", {}) or {}
    total_suppressed = sum(int(v) for v in suppressed.values())
    top = sorted(suppressed.items(), key=lambda item: -int(item[1]))[:5]
    info_print(
        "[LOG STATE]",
        f"mode={current_log_mode()}",
        f"repeat_interval={float(STATE.get('log_repeat_interval', 0.35)):.2f}",
        f"suppressed_total={total_suppressed}",
        f"suppressed_top={[(k[:72], int(v)) for k, v in top]}",
        force_log=True,
    )


def disable_usd_audio_extension():
    if bool(STATE.get("usd_audio_extension_disabled", False)):
        return
    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.usd.audio", False)
        STATE["usd_audio_extension_disabled"] = True
        info_print("[USD AUDIO] disabled omni.usd.audio to suppress timeline reset log")
    except Exception as e:
        STATE["usd_audio_extension_disabled"] = False
        info_print("[WARN] [USD AUDIO] could not disable omni.usd.audio:", type(e).__name__, e)


def cached_real_joint_positions():
    q = STATE.get("last_valid_real_q")
    if q is not None:
        try:
            return np.array(q, dtype=np.float32).copy()
        except Exception:
            pass
    try:
        return CTRL.q_cmd.copy()
    except Exception:
        return np.zeros(4, dtype=np.float32)


def simulation_timeline_is_playing():
    if not HAS_OMNI_TIMELINE:
        return True
    try:
        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            return True
        return bool(timeline.is_playing())
    except Exception:
        return True


def ensure_timeline_playing(label=""):
    if not HAS_OMNI_TIMELINE:
        return True
    try:
        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            return True
        if not timeline.is_playing():
            try:
                timeline.play()
                info_print("[TIMELINE RECOVER]", f"label={label}", "action=play")
            except Exception as e:
                info_print("[WARN] [TIMELINE RECOVER] play failed:", type(e).__name__, e)
        return bool(timeline.is_playing())
    except Exception:
        return True


def timeline_allows_background_work():
    return bool(STATE.get("running", False)) and bool(simulation_timeline_is_playing())


def handle_timeline_stop_if_needed(label=""):
    if simulation_timeline_is_playing():
        STATE["timeline_stop_handled"] = False
        return False
    if bool(STATE.get("timeline_stop_handled", False)):
        return True

    STATE["timeline_stop_handled"] = True
    STATE["auto_collect_stop_requested"] = True
    STATE["follow"] = False
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["sand_site_reset_active"] = False

    for task_name in [
        "auto_collect",
        "motion",
        "planner",
        "replay",
        "sand_reset",
        "startup_sand_reset",
    ]:
        cancel_registered_task(task_name, reason=f"timeline_stopped:{label}")

    info_print("[TIMELINE STOP]", f"label={label}", "action=cancel_background_tasks")
    update_status("[TIMELINE STOP] background tasks cancelled; press Play before running again", force=True)
    return True


def object_physics_view_state(obj, depth=0):
    if obj is None or depth > 3:
        return None
    found = False
    for attr_name in ["_physics_view", "_physics_sim_view"]:
        if hasattr(obj, attr_name):
            found = True
            try:
                if getattr(obj, attr_name) is not None:
                    return True
            except Exception:
                pass
    for attr_name in ["_articulation_view", "_view", "_articulation"]:
        try:
            child = getattr(obj, attr_name, None)
        except Exception:
            child = None
        state = object_physics_view_state(child, depth + 1)
        if state is True:
            return True
        if state is False:
            found = True
    return False if found else None


def robot_joint_read_ready():
    if ROBOT is None:
        return False
    if not bool(STATE.get("robot_state_reads_enabled", False)):
        return False
    if bool(STATE.get("robot_state_shutdown", False)):
        return False
    if not simulation_timeline_is_playing():
        return False
    physics_state = object_physics_view_state(ROBOT)
    if physics_state is False:
        return False
    return True


def robot_articulation_action_ready():
    if ROBOT is None or JOINT_INDICES is None:
        return False
    if bool(STATE.get("robot_state_shutdown", False)):
        return False
    if not simulation_timeline_is_playing():
        return False
    physics_state = object_physics_view_state(ROBOT)
    if physics_state is False:
        return False
    return True


def articulation_action_ready_detail():
    physics_state = None
    try:
        physics_state = object_physics_view_state(ROBOT)
    except Exception:
        physics_state = None
    detail = {
        "ready": bool(robot_articulation_action_ready()),
        "robot": ROBOT is not None,
        "joint_indices": JOINT_INDICES is not None,
        "timeline_playing": bool(simulation_timeline_is_playing()),
        "robot_state_shutdown": bool(STATE.get("robot_state_shutdown", False)),
        "physics_view_state": physics_state,
    }
    return detail


def format_action_ready_detail(detail):
    if not isinstance(detail, dict):
        return "action_channel_detail=unknown"
    return (
        f"timeline={detail.get('timeline_playing')} "
        f"robot={detail.get('robot')} "
        f"joint_indices={detail.get('joint_indices')} "
        f"physics_view={detail.get('physics_view_state')} "
        f"shutdown={detail.get('robot_state_shutdown')}"
    )


async def wait_for_articulation_action_ready(
    label="action",
    min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
    max_frames=ACTION_READY_MAX_WAIT_FRAMES,
    record_failure=True,
):
    stable = 0
    last_detail = articulation_action_ready_detail()
    max_frames = max(1, int(max_frames))
    min_stable_frames = max(1, int(min_stable_frames))
    for _ in range(max_frames):
        last_detail = articulation_action_ready_detail()
        if bool(last_detail.get("ready", False)):
            stable += 1
            if stable >= min_stable_frames:
                return True, "ok", last_detail
        else:
            stable = 0
            if not bool(last_detail.get("timeline_playing", True)):
                handle_timeline_stop_if_needed(label)
                reason = f"action_channel_not_ready {format_action_ready_detail(last_detail)}"
                info_print("[DIG EXEC FAILED]", f"label={label}", reason)
                if record_failure:
                    set_execution_failure_reason(f"execution_failed/action_channel_not_ready:{label}:{reason}")
                return False, reason, last_detail
        await step_updates(1)
    reason = f"action_channel_not_ready {format_action_ready_detail(last_detail)}"
    info_print("[DIG EXEC FAILED]", f"label={label}", reason)
    if record_failure:
        set_execution_failure_reason(f"execution_failed/action_channel_not_ready:{label}:{reason}")
    return False, reason, last_detail


def get_real_joint_positions():
    global ROBOT, DOF_NAME_TO_REAL_IDX

    if not robot_joint_read_ready():
        return cached_real_joint_positions()

    try:
        q_raw = ROBOT.get_joint_positions()
        if q_raw is None:
            return cached_real_joint_positions()

        q_raw = np.array(q_raw, dtype=np.float32)
        q = np.zeros(4, dtype=np.float32)

        for i, name in enumerate(DOF_ORDER):
            if name in DOF_NAME_TO_REAL_IDX:
                q[i] = q_raw[DOF_NAME_TO_REAL_IDX[name]]
            else:
                q[i] = CTRL.q_cmd[i]

        STATE["last_valid_real_q"] = q.copy()
        STATE["last_valid_real_q_time"] = time.time()
        return q
    except Exception as e:
        now = time.time()
        if now - float(STATE.get("last_joint_read_warn_time", 0.0)) > 2.0:
            STATE["last_joint_read_warn_time"] = now
            info_print("[WARN] joint read skipped:", type(e).__name__, e)
        return cached_real_joint_positions()


def update_q_cmd_from_real():
    q_real = get_real_joint_positions()
    q_unwrapped = q_real.copy()
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        q_unwrapped[swing_idx] = swing_target_near(
            q_real[swing_idx],
            CTRL.q_cmd[swing_idx],
        )
    except Exception:
        pass
    CTRL.q_cmd = q_unwrapped.copy()
    return q_real


def planner_joint_positions():
    return CTRL.q_cmd.copy()


def planner_end_effector_pos(end_effector="mid", q=None):
    if q is None:
        q = planner_joint_positions()
    try:
        p = predicted_end_world_point(q, end_effector=end_effector, reference_q=planner_joint_positions())
        if p is not None:
            return p
    except Exception:
        pass
    if end_effector == "tip":
        return bucket_tip_pos()
    if end_effector == "load":
        return bucket_load_pos()
    if end_effector == "pour":
        return bucket_pour_pos()
    return bucket_mid_pos()


def hold_manual_ui_sync():
    STATE["manual_editing_until"] = time.time() + MANUAL_UI_SYNC_HOLD_SECONDS


def manual_ui_sync_is_held():
    return bool(STATE.get("manual_joint_active", False)) or time.time() < float(STATE.get("manual_editing_until", 0.0))


def get_speed_multiplier():
    sm = safe_float(STATE.get("speed_multiplier", 1.0), 1.0)
    sm = max(SPEED_MULTIPLIER_MIN, min(SPEED_MULTIPLIER_MAX, sm))
    STATE["speed_multiplier"] = sm
    return sm


def get_manual_speed_multiplier():
    sm = min(get_speed_multiplier(), MANUAL_SPEED_MULTIPLIER_CAP)
    return max(MANUAL_SPEED_MULTIPLIER_FLOOR, float(sm) * MANUAL_SPEED_SCALE)


def ui_short_text(text, max_chars=UI_STATUS_MAX_CHARS):
    s = str(text).replace("\n", " ")
    if len(s) <= int(max_chars):
        return s
    return s[: max(0, int(max_chars) - 3)] + "..."


def apply_speed_to_physx_joint_limits():
    # Keep joint physics exactly as imported. Speed is handled in software only.
    get_speed_multiplier()


def update_status(text, force=False):
    now = time.time()
    text = str(text)
    same_text = text == STATE.get("last_status_text", None)
    repeat_interval = float(STATE.get("status_repeat_interval", 2.5))
    interval = float(STATE.get("status_interval", 0.5))
    should_print = force or now - float(STATE.get("last_status_time", 0.0)) > interval
    if same_text and now - float(STATE.get("last_status_repeat_time", 0.0)) < repeat_interval:
        should_print = False

    if should_print:
        info_print(text)
        STATE["last_status_time"] = now
        STATE["last_status_text"] = text
        STATE["last_status_repeat_time"] = now
        try:
            if STATUS_LABEL is not None:
                STATUS_LABEL.text = ui_short_text(text)
        except Exception:
            pass
    elif same_text:
        try:
            if STATUS_LABEL is not None:
                STATUS_LABEL.text = ui_short_text(text)
        except Exception:
            pass


def cancel_active_task(reason="manual override"):
    already_manual = bool(STATE.get("manual_override", False)) and STATE.get("active_task_name") == "manual"
    STATE["follow"] = False
    STATE["manual_override"] = True
    STATE["active_task_id"] = int(STATE.get("active_task_id", 0)) + 1
    STATE["active_task_name"] = "manual"
    try:
        cancel_registered_task("motion", reason=reason)
    except Exception:
        pass
    update_status(f"[CANCEL TASK] {reason}", force=not already_manual)


def start_task(name):
    STATE["follow"] = False
    STATE["manual_override"] = False
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["active_task_id"] = int(STATE.get("active_task_id", 0)) + 1
    STATE["active_task_name"] = str(name)
    try:
        dataset_record_event("task_start", str(name))
    except Exception:
        pass
    return STATE["active_task_id"]


def task_alive(task_id):
    return (
        STATE.get("running", False)
        and int(STATE.get("active_task_id", -1)) == int(task_id)
    )


def cancel_registered_task(name, reason=""):
    tasks = STATE.setdefault("async_tasks", {})
    task = tasks.get(str(name))
    if task is None:
        return False
    try:
        if hasattr(task, "done") and not task.done():
            info_print("[TASK CANCEL]", f"name={name}", f"reason={reason}")
            task.cancel()
    except Exception as e:
        info_print("[WARN] task cancel failed:", f"name={name}", type(e).__name__, e)
    try:
        if tasks.get(str(name)) is task:
            tasks.pop(str(name), None)
    except Exception:
        pass
    return True


def cancel_registered_tasks(reason="", keep=None):
    keep = set([] if keep is None else keep)
    for name in list(STATE.setdefault("async_tasks", {}).keys()):
        if name in keep:
            continue
        cancel_registered_task(name, reason=reason)


def _registered_task_done(name, task):
    try:
        if task.cancelled():
            info_print("[TASK DONE]", f"name={name}", "status=cancelled")
            return
        exc = task.exception()
        if exc is not None:
            info_print("[ERROR] [TASK FAILED]", f"name={name}", type(exc).__name__, exc)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6)).strip()
            if tb:
                info_print("[TASK TRACE]", tb)
    except asyncio.CancelledError:
        info_print("[TASK DONE]", f"name={name}", "status=cancelled")
    except Exception as e:
        info_print("[WARN] task done callback failed:", f"name={name}", type(e).__name__, e)
    finally:
        tasks = STATE.setdefault("async_tasks", {})
        if tasks.get(str(name)) is task:
            tasks.pop(str(name), None)


def register_async_task(name, coro_or_task, replace=True):
    name = str(name)
    if replace:
        cancel_registered_task(name, reason="replace")
    task = coro_or_task if hasattr(coro_or_task, "add_done_callback") else asyncio.ensure_future(coro_or_task)
    STATE.setdefault("async_tasks", {})[name] = task
    try:
        task.add_done_callback(lambda t, task_name=name: _registered_task_done(task_name, t))
    except Exception as e:
        info_print("[WARN] task registration callback failed:", f"name={name}", type(e).__name__, e)
    return task


def find_robot_paths():
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root = prim.GetPath().pathString
            base = prim.GetPath().GetParentPath().pathString
            return root, base
    raise RuntimeError("No ArticulationRootAPI found. 请先导入 URDF。")


def clear_xform(prim):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    return xform


def set_xform(prim, translate=None, scale=None):
    xform = clear_xform(prim)
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def set_color(prim, color):
    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )
    except Exception:
        pass


def set_prim_visibility(prim_or_path, visible):
    prim = prim_or_path if hasattr(prim_or_path, "IsValid") else get_prim(prim_or_path)
    if not prim or not prim.IsValid():
        return False
    token = UsdGeom.Tokens.inherited if bool(visible) else UsdGeom.Tokens.invisible
    try:
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(token)
        return True
    except Exception as e:
        info_print("[WARN] visibility set failed:", prim.GetPath().pathString, type(e).__name__, e)
        return False


def restore_excavator_control_visibility():
    paths = []
    if ROBOT_ROOT:
        paths.append(ROBOT_ROOT)
    paths.extend(LINK_PATHS.values())

    restored = 0
    visited = set()
    for path in paths:
        prim = get_prim(path)
        while prim and prim.IsValid():
            prim_path = prim.GetPath().pathString
            if prim_path in visited:
                break
            visited.add(prim_path)
            if set_prim_visibility(prim, True):
                restored += 1
            if prim_path in ("/World", "/"):
                break
            prim = prim.GetParent()
    return restored


def subtree_has_mesh(prim):
    if not prim or not prim.IsValid():
        return False
    for p in Usd.PrimRange(prim):
        try:
            if p.IsA(UsdGeom.Mesh):
                return True
        except Exception:
            if str(p.GetTypeName()) == "Mesh":
                return True
    return False


def collect_excavator_display_roots():
    physical_roots = []
    render_roots = []
    ignored_child_names = {
        "visuals",
        "collisions",
        "joints",
        "looks",
        "root_joint",
        "cameras",
        "sensors",
    }

    for link_name, link_path in LINK_PATHS.items():
        link_prim = get_prim(link_path)
        if not link_prim or not link_prim.IsValid():
            continue

        visuals = get_prim(f"{link_path}/visuals")
        if visuals and visuals.IsValid():
            physical_roots.append(visuals.GetPath().pathString)

        for child in link_prim.GetChildren():
            child_name = child.GetName()
            if child_name.lower() in ignored_child_names:
                continue
            if subtree_has_mesh(child):
                render_roots.append(child.GetPath().pathString)

    return physical_roots, render_roots


def apply_excavator_render_mode(render_on=None, force_status=True):
    if render_on is None:
        render_on = bool(STATE.get("excavator_render_mode", False))
    render_on = bool(render_on)
    STATE["excavator_render_mode"] = render_on

    control_visible_count = restore_excavator_control_visibility()
    world_model_visible = set_prim_visibility("/World/model", render_on)
    physical_roots, render_roots = collect_excavator_display_roots()
    for path in physical_roots:
        set_prim_visibility(path, not render_on)
    for path in render_roots:
        set_prim_visibility(path, render_on)

    mode = "render" if render_on else "physics"
    msg = (
        f"Excavator display mode = {mode}; "
        f"render_roots={len(render_roots)} physical_visuals={len(physical_roots)} "
        f"world_model_visible={world_model_visible}"
    )
    info_print(
        "[DISPLAY MODE]",
        f"mode={mode}",
        f"world_model_visible={world_model_visible}",
        f"render_roots={len(render_roots)}",
        f"physical_visuals={len(physical_roots)}",
    )
    if force_status:
        update_status(msg, force=True)
    return render_on


def toggle_excavator_render_mode():
    return apply_excavator_render_mode(not bool(STATE.get("excavator_render_mode", False)))


def set_prim_attr(prim, name, value, type_name=None):
    try:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            if type_name is None:
                if isinstance(value, bool):
                    type_name = Sdf.ValueTypeNames.Bool
                elif isinstance(value, int):
                    type_name = Sdf.ValueTypeNames.Int
                elif isinstance(value, float):
                    type_name = Sdf.ValueTypeNames.Float
                else:
                    type_name = Sdf.ValueTypeNames.Token
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)
        return True
    except Exception as e:
        info_print("[WARN] set physics scene attr failed:", name, e)
        return False


def with_root_edit_target(fn):
    old_target = stage.GetEditTarget()
    try:
        stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    except Exception as e:
        info_print("[WARN] could not switch to root edit target:", e)
    try:
        return fn()
    finally:
        try:
            stage.SetEditTarget(old_target)
        except Exception:
            pass


def get_physx_schema():
    global PhysxSchema
    if PhysxSchema is not None:
        return PhysxSchema
    try:
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx", True)
        from pxr import PhysxSchema as ImportedPhysxSchema
        PhysxSchema = ImportedPhysxSchema
        info_print("[SAND PHYSX] enabled omni.physx")
    except Exception as e:
        info_print("[WARN] cannot enable/import PhysxSchema:", e)
    return PhysxSchema


def apply_physx_api_by_names(prim, names):
    schema = get_physx_schema()
    if schema is None:
        return None
    for name in names:
        api_cls = getattr(schema, name, None)
        if api_cls is None:
            continue
        try:
            return api_cls.Apply(prim)
        except Exception:
            try:
                return api_cls(prim)
            except Exception:
                pass
    return None


def apply_physx_scene_api(prim):
    schema = get_physx_schema()
    if schema is None:
        return None
    api_cls = getattr(schema, "PhysxSceneAPI", None)
    if api_cls is None:
        return None
    try:
        return api_cls.Apply(prim)
    except Exception:
        try:
            return api_cls(prim)
        except Exception:
            return None


def set_schema_attr(obj, names, value):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn().Set(value)
                return True
            except Exception:
                pass
    return False


def enable_physx_gpu_runtime_settings():
    if carb is None:
        return
    try:
        settings = carb.settings.get_settings()
        for key in [
            "/physics/physx/useGpu",
            "/physics/physx/useGPU",
            "/physics/physx/useGpuDynamics",
            "/physics/physx/enableGPUDynamics",
            "/physics/physx/enableGpuDynamics",
            "/physics/physx/enable_gpu_dynamics",
            "/physics/physx/gpuDynamicsEnabled",
            "/physics/physx/enableCCD",
            "/physics/physx/enableCcd",
            "/persistent/physics/physx/useGpu",
            "/persistent/physics/physx/useGPU",
            "/persistent/physics/physx/enableGPUDynamics",
            "/persistent/physics/physx/enableGpuDynamics",
            "/persistent/physics/physx/useGpuDynamics",
            "/persistent/physics/physx/enable_gpu_dynamics",
            "/persistent/physics/physx/gpuDynamicsEnabled",
            "/persistent/physics/physx/enableCCD",
            "/persistent/physics/physx/enableCcd",
        ]:
            if key.lower().endswith("enableccd"):
                settings.set_bool(key, False)
            else:
                settings.set_bool(key, True)
        settings.set_string("/physics/physx/broadphaseType", "GPU")
        settings.set_string("/persistent/physics/physx/broadphaseType", "GPU")
    except Exception as e:
        info_print("[WARN] cannot set PhysX GPU runtime settings:", e)


def configure_world_gpu_physics(world, label=""):
    enable_physx_gpu_runtime_settings()
    context = None
    for attr_name in ["get_physics_context", "physics_context", "_physics_context"]:
        try:
            candidate = getattr(world, attr_name, None)
            context = candidate() if callable(candidate) else candidate
            if context is not None:
                break
        except Exception:
            context = None
    if context is None:
        info_print(f"[SAND PHYSX] no runtime physics context available {label}")
        return False

    changed = []

    def call_bool(names, value=True):
        for name in names:
            fn = getattr(context, name, None)
            if callable(fn):
                try:
                    fn(bool(value))
                    changed.append(name)
                    return True
                except Exception:
                    pass
        return False

    def call_value(names, value):
        for name in names:
            fn = getattr(context, name, None)
            if callable(fn):
                try:
                    fn(value)
                    changed.append(name)
                    return True
                except Exception:
                    pass
        return False

    call_bool([
        "enable_gpu_dynamics",
        "set_gpu_dynamics_enabled",
        "set_enable_gpu_dynamics",
        "enableGPUDynamics",
    ], True)
    call_bool([
        "enable_ccd",
        "set_enable_ccd",
        "set_ccd_enabled",
        "enableCCD",
    ], False)
    call_value([
        "set_broadphase_type",
        "set_broadphase",
        "set_broadphaseType",
    ], "GPU")
    call_value([
        "set_solver_type",
        "set_solver",
    ], "TGS")
    info_print(f"[SAND PHYSX] runtime GPU physics configured {label}: methods={changed}")
    return bool(changed)


def ensure_particle_gpu_physics_scene(label=""):
    enable_physx_gpu_runtime_settings()
    try:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    except Exception as e:
        info_print("[WARN] cannot set stage metersPerUnit=1.0:", e)
    prim = get_prim("/physicsScene")
    if not prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, sdf_path("/physicsScene"))
        prim = scene.GetPrim()
        info_print("[SAND PHYSX] created /physicsScene")

    scenes = []
    scene_paths = set()
    for p in stage.Traverse():
        if p.GetTypeName() == "PhysicsScene" or str(p.GetPath()) == "/physicsScene":
            p_path = str(p.GetPath())
            if p.IsValid() and p_path not in scene_paths:
                scenes.append(p)
                scene_paths.add(p_path)
    if str(prim.GetPath()) not in scene_paths:
        scenes.append(prim)

    for scene_prim in scenes:
        scene = UsdPhysics.Scene(scene_prim)
        try:
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
        except Exception as e:
            info_print("[WARN] cannot set scene gravity:", scene_prim.GetPath(), e)
        api = apply_physx_scene_api(scene_prim)
        if api is not None:
            set_schema_attr(api, ["CreateEnableGPUDynamicsAttr", "CreateGpuDynamicsEnabledAttr"], True)
            set_schema_attr(api, ["CreateBroadphaseTypeAttr"], "GPU")
            set_schema_attr(api, ["CreateSolverTypeAttr"], "TGS")
            set_schema_attr(api, ["CreateTimeStepsPerSecondAttr"], int(PHYSX_TIMESTEPS_PER_SECOND))
            set_schema_attr(api, ["CreateEnableCCDAttr", "CreateEnableCcdAttr"], False)

        set_prim_attr(scene_prim, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(scene_prim, "physxScene:enableGpuDynamics", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(scene_prim, "physxScene:gpuDynamicsEnabled", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(scene_prim, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token)
        set_prim_attr(scene_prim, "physxScene:gpuBroadphase", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(scene_prim, "physxScene:solverType", "TGS", Sdf.ValueTypeNames.Token)
        set_prim_attr(scene_prim, "physxScene:timeStepsPerSecond", int(PHYSX_TIMESTEPS_PER_SECOND), Sdf.ValueTypeNames.Int)
        set_prim_attr(scene_prim, "physxScene:enableCCD", False, Sdf.ValueTypeNames.Bool)
        set_prim_attr(scene_prim, "physics:enableCCD", False, Sdf.ValueTypeNames.Bool)
        gpu_attr = scene_prim.GetAttribute("physxScene:enableGPUDynamics")
        broadphase_attr = scene_prim.GetAttribute("physxScene:broadphaseType")
        gravity_attr = scene_prim.GetAttribute("physics:gravityMagnitude")
        gpu_value = gpu_attr.Get() if gpu_attr.IsValid() else None
        broadphase_value = broadphase_attr.Get() if broadphase_attr.IsValid() else None
        gravity_value = gravity_attr.Get() if gravity_attr.IsValid() else None
        info_print(
            f"[SAND PHYSX] GPU scene configured {label}:",
            scene_prim.GetPath(),
            f"gpu={gpu_value}",
            f"broadphase={broadphase_value}",
            f"gravity={gravity_value}",
        )

    return prim


def rebind_sand_particles_to_physics_scene(label=""):
    scene_prim = get_prim("/physicsScene")
    if not scene_prim.IsValid():
        return
    rebound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lname = prim.GetName().lower()
        if "particlesystem" not in lname and "realsandparticles" not in lname:
            continue
        for rel_name in [
            "simulationOwner",
            "physics:simulationOwner",
            "physxParticle:simulationOwner",
            "physxParticle:particleSystem",
        ]:
            try:
                rel = prim.CreateRelationship(rel_name)
                if "particleSystem" in rel_name and "realsandparticles" in lname:
                    system_path = path.rsplit("/", 1)[0] + "/ParticleSystem"
                    rel.SetTargets([sdf_path(system_path)])
                else:
                    rel.SetTargets([scene_prim.GetPath()])
            except Exception:
                pass
        rebound += 1
    if rebound:
        info_print(f"[SAND PHYSX] rebound sand particle prims {label}: count={rebound}")


def get_bbox(path):
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        return None

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    try:
        return cache.ComputeWorldBound(prim).ComputeAlignedBox()
    except Exception:
        return None


def bbox_center(path):
    box = get_bbox(path)
    if box is None:
        return np.zeros(3, dtype=np.float32)
    if not bbox_values_are_valid(box.GetMin(), box.GetMax()):
        return np.zeros(3, dtype=np.float32)
    c = box.GetMidpoint()
    return np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float32)


def bbox_min_z(path):
    box = get_bbox(path)
    if box is None:
        return None
    mn = box.GetMin()
    mx = box.GetMax()
    if not bbox_values_are_valid(mn, mx):
        return None
    return float(mn[2])


def bbox_values_are_valid(mn, mx):
    try:
        vals = [
            float(mn[0]), float(mn[1]), float(mn[2]),
            float(mx[0]), float(mx[1]), float(mx[2]),
        ]
    except Exception:
        return False
    if not all(math.isfinite(v) for v in vals):
        return False
    if any(abs(v) > VALID_BBOX_ABS_MAX for v in vals):
        return False
    if vals[3] < vals[0] or vals[4] < vals[1] or vals[5] < vals[2]:
        return False
    return True


def bbox_min_max(path):
    box = get_bbox(path)
    if box is None:
        return None, None
    mn = box.GetMin()
    mx = box.GetMax()
    if not bbox_values_are_valid(mn, mx):
        return None, None
    return (
        np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=np.float32),
        np.array([float(mx[0]), float(mx[1]), float(mx[2])], dtype=np.float32),
    )


def bbox_center_size(path):
    mn, mx = bbox_min_max(path)
    if mn is None or mx is None:
        return None, None, None, None
    center = 0.5 * (mn + mx)
    size = np.maximum(mx - mn, np.zeros(3, dtype=np.float32))
    return center.astype(np.float32), size.astype(np.float32), mn, mx


def selected_prim_paths():
    try:
        selection = omni.usd.get_context().get_selection()
        if selection is None:
            return []
        paths = selection.get_selected_prim_paths()
        return [str(p) for p in (paths or []) if str(p)]
    except Exception as e:
        info_print("[WARN] selection read failed:", type(e).__name__, e)
        return []


def first_selected_prim_path():
    paths = selected_prim_paths()
    return paths[0] if paths else ""


def mesh_world_xy_points_under(prim):
    if prim is None or not prim.IsValid():
        return np.empty((0, 2), dtype=np.float32)
    pts = []
    for p in Usd.PrimRange(prim):
        try:
            if not p.IsA(UsdGeom.Mesh):
                continue
            local_points = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not local_points:
                continue
            mat = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for lp in local_points:
                wp = mat.Transform(Gf.Vec3d(float(lp[0]), float(lp[1]), float(lp[2])))
                pts.append((float(wp[0]), float(wp[1])))
        except Exception:
            continue
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def convex_hull_xy(points):
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    pts = np.unique(np.round(pts, 6), axis=0)
    if pts.shape[0] < 3:
        return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-9:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-9:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull if hull.shape[0] >= 3 else None


def polygon_signed_area(poly):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid_xy(poly):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    area = polygon_signed_area(p)
    if abs(area) < 1.0e-8:
        return np.mean(p, axis=0).astype(np.float32)
    x = p[:, 0]
    y = p[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    return np.array([
        np.sum((x + x2) * cross) / (6.0 * area),
        np.sum((y + y2) * cross) / (6.0 * area),
    ], dtype=np.float32)


def shrink_convex_polygon_xy(poly, shrink_d):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return None
    if polygon_signed_area(p) < 0.0:
        p = p[::-1].copy()
    d = max(0.0, float(shrink_d))
    if d <= 1.0e-6:
        return p.astype(np.float32)

    def inside(pt, a, b):
        e = b - a
        length = max(1.0e-9, float(np.linalg.norm(e)))
        return float(e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0])) >= d * length - 1.0e-8

    def intersect(s, ept, a, b):
        edge = b - a
        seg = ept - s
        rhs = d * max(1.0e-9, float(np.linalg.norm(edge))) - (edge[0] * (s[1] - a[1]) - edge[1] * (s[0] - a[0]))
        denom = edge[0] * seg[1] - edge[1] * seg[0]
        if abs(float(denom)) < 1.0e-12:
            return ept
        return s + np.clip(float(rhs / denom), 0.0, 1.0) * seg

    out = p.copy()
    for i in range(p.shape[0]):
        a = p[i]
        b = p[(i + 1) % p.shape[0]]
        if out.shape[0] == 0:
            break
        new_pts = []
        s = out[-1]
        s_inside = inside(s, a, b)
        for ept in out:
            e_inside = inside(ept, a, b)
            if e_inside:
                if not s_inside:
                    new_pts.append(intersect(s, ept, a, b))
                new_pts.append(ept)
            elif s_inside:
                new_pts.append(intersect(s, ept, a, b))
            s = ept
            s_inside = e_inside
        out = np.array(new_pts, dtype=np.float64) if new_pts else np.empty((0, 2), dtype=np.float64)
    if out.shape[0] < 3 or abs(polygon_signed_area(out)) < 1.0e-6:
        c = polygon_centroid_xy(p).astype(np.float64)
        return (c + 0.80 * (p - c)).astype(np.float32)
    return out.astype(np.float32)


def visual_polygon_xy(poly, max_vertices):
    p = np.array(poly, dtype=np.float32).reshape(-1, 2)
    n = int(p.shape[0])
    limit = max(3, int(max_vertices))
    if n <= limit:
        return p
    idx = np.linspace(0, n - 1, limit, dtype=np.int32)
    return p[idx]


def circle_polygon_xy(cx, cy, radius, segments=20):
    n = max(8, int(segments))
    r = max(0.01, float(radius))
    pts = []
    for i in range(n):
        a = 2.0 * math.pi * float(i) / float(n)
        pts.append((float(cx) + r * math.cos(a), float(cy) + r * math.sin(a)))
    return np.array(pts, dtype=np.float32)


def prim_collision_enabled(prim):
    try:
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr.IsValid() and attr.Get() is False:
            return False
    except Exception:
        pass
    try:
        return prim.HasAPI(UsdPhysics.CollisionAPI)
    except Exception:
        return False


def disable_collision(prim, label=""):
    if prim is None or not prim.IsValid():
        return
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    except Exception:
        pass
    set_prim_attr(prim, "physics:collisionEnabled", False, Sdf.ValueTypeNames.Bool)
    if label:
        info_print("[COLLISION OFF]", label, prim.GetPath())


def iter_prim_subtree(root_prim):
    if root_prim is None or not root_prim.IsValid():
        return
    try:
        for prim in Usd.PrimRange(root_prim):
            yield prim
    except Exception:
        yield root_prim


def nearest_rigid_ancestor_path(prim):
    p = prim
    while p is not None and p.IsValid():
        try:
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(p.GetPath())
        except Exception:
            pass
        parent_path = p.GetPath().GetParentPath()
        parent_text = str(parent_path)
        if parent_path == p.GetPath() or parent_text == "":
            break
        p = stage.GetPrimAtPath(parent_path)
    return None


def collision_bbox_for_subtree(root_path):
    root = get_prim(root_path)
    if not root or not root.IsValid():
        return None, None, "missing"

    root_prefix = str(root.GetPath())
    mins = []
    maxs = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p != root_prefix and not p.startswith(root_prefix + "/"):
            continue
        p_lower = p.lower()
        collision_named = "/collisions" in p_lower or p_lower.endswith("/collisions")
        if not collision_named and not prim_collision_enabled(prim):
            continue
        mn, mx = bbox_min_max(p)
        if mn is None:
            continue
        mins.append(mn)
        maxs.append(mx)

    if not mins:
        mn, mx = bbox_min_max(root_path)
        return mn, mx, "fallback_bbox"

    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0), "collision_bbox"


def compact_path(path, max_len=120):
    text = str(path)
    max_len = int(max_len)
    if len(text) <= max_len:
        return text
    keep = max(12, (max_len - 3) // 2)
    return text[:keep] + "..." + text[-keep:]


def collision_bbox_lowest_contributors(root_path, limit=3):
    root = get_prim(root_path)
    if not root or not root.IsValid():
        return []

    root_prefix = str(root.GetPath())
    rows = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p != root_prefix and not p.startswith(root_prefix + "/"):
            continue
        p_lower = p.lower()
        collision_named = "/collisions" in p_lower or p_lower.endswith("/collisions")
        collision_enabled = prim_collision_enabled(prim)
        if not collision_named and not collision_enabled:
            continue
        mn, mx = bbox_min_max(p)
        if mn is None:
            continue
        try:
            has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        except Exception:
            has_collision_api = False
        rows.append({
            "path": p,
            "min_z": float(mn[2]),
            "max_z": float(mx[2]),
            "type": str(prim.GetTypeName()),
            "collision_named": bool(collision_named),
            "collision_enabled": bool(collision_enabled),
            "has_collision_api": bool(has_collision_api),
            "rigid": nearest_rigid_ancestor_path(prim),
        })

    rows.sort(key=lambda item: item["min_z"])
    return rows[:max(1, int(limit))]


def collision_lowest_detail(link_name, limit=2):
    path = LINK_PATHS.get(link_name)
    if not path:
        return f"{link_name}_lowest=missing_link"
    rows = collision_bbox_lowest_contributors(path, limit=limit)
    if not rows:
        return f"{link_name}_lowest=no_collision_bbox"
    parts = []
    for i, row in enumerate(rows):
        parts.append(
            f"{link_name}_lowest{i + 1}:z={row['min_z']:.3f}..{row['max_z']:.3f} "
            f"type={row['type']} api={row['has_collision_api']} enabled={row['collision_enabled']} "
            f"named={row['collision_named']} rigid={compact_path(row['rigid'])} path={compact_path(row['path'])}"
        )
    return "; ".join(parts)


def robot_support_bbox():
    base_path = LINK_PATHS.get("base_link")
    if not base_path:
        return None, None, "missing_base_link"
    return collision_bbox_for_subtree(base_path)


def robot_full_collision_bbox():
    if not ROBOT_BASE:
        return None, None, "missing_robot_base"
    return collision_bbox_for_subtree(ROBOT_BASE)


def sync_articulation_pose_from_usd(label=""):
    if ROBOT is None:
        return False
    try:
        root_pos = get_prim_translation(ROBOT_ROOT)
    except Exception as e:
        info_print("[WARN] [GROUND PARK] cannot read articulation root pose:", label, e)
        return False

    pos = np.array(root_pos, dtype=np.float32)
    for method_name in ["set_world_pose", "set_local_pose"]:
        fn = getattr(ROBOT, method_name, None)
        if not callable(fn):
            continue
        try:
            fn(position=pos)
            info_print("[GROUND PARK]", f"synced articulation pose via {method_name}", f"label={label}", f"root_pos={np.round(pos, 3)}")
            return True
        except TypeError:
            try:
                fn(pos)
                info_print("[GROUND PARK]", f"synced articulation pose via {method_name}", f"label={label}", f"root_pos={np.round(pos, 3)}")
                return True
            except Exception:
                pass
        except Exception:
            pass
    return False


def get_prim_translation(path):
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        return np.zeros(3, dtype=np.float32)

    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)


def make_cube(path, translate, scale, color, collision=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    set_xform(prim, translate=translate, scale=scale)
    set_color(prim, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def make_box(path, translate, scale, color, collision=False, opacity=None):
    prim = make_cube(path, translate=translate, scale=scale, color=color, collision=collision)
    if opacity is not None:
        set_opacity(prim, opacity)
    if not collision:
        disable_collision(prim, "box")
    return prim


def make_sphere(path, translate, radius, color, collision=False):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    prim = sphere.GetPrim()
    set_xform(prim, translate=translate)
    set_color(prim, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    else:
        disable_collision(prim, "sphere")
    return prim


def set_opacity(prim, opacity):
    try:
        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr([float(opacity)])
    except Exception:
        pass


def make_cylinder(path, translate, radius, height, color, collision=False, opacity=None):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    prim = cyl.GetPrim()
    set_xform(prim, translate=translate)
    set_color(prim, color)
    if opacity is not None:
        set_opacity(prim, opacity)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    else:
        disable_collision(prim, "cylinder")
    return prim


def make_polygon_prism(path, polygon_xy, z_min, z_max, color, opacity=None):
    poly = np.array(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = []
    for xy in poly:
        pts.append(Gf.Vec3f(float(xy[0]), float(xy[1]), float(z_min)))
    for xy in poly:
        pts.append(Gf.Vec3f(float(xy[0]), float(xy[1]), float(z_max)))
    n = int(poly.shape[0])
    counts = [n, n]
    indices = list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        indices.extend([i, j, n + j, n + i])
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    try:
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(False)
    except Exception:
        pass
    prim = mesh.GetPrim()
    set_color(prim, color)
    if opacity is not None:
        set_opacity(prim, opacity)
    disable_collision(prim, "polygon_prism")
    return prim


def make_polygon_wire_column(path, polygon_xy, z_min, z_max, color, width=None):
    poly = np.array(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None

    z0 = float(z_min)
    z1 = float(z_max)
    n = int(poly.shape[0])
    points = []
    counts = []

    bottom = [Gf.Vec3f(float(x), float(y), z0) for x, y in poly] + [
        Gf.Vec3f(float(poly[0][0]), float(poly[0][1]), z0)
    ]
    top = [Gf.Vec3f(float(x), float(y), z1) for x, y in poly] + [
        Gf.Vec3f(float(poly[0][0]), float(poly[0][1]), z1)
    ]
    points.extend(bottom)
    counts.append(len(bottom))
    points.extend(top)
    counts.append(len(top))

    step = max(1, n // 8)
    for i in range(0, n, step):
        x, y = poly[i]
        points.extend([Gf.Vec3f(float(x), float(y), z0), Gf.Vec3f(float(x), float(y), z1)])
        counts.append(2)

    prim = get_prim(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.BasisCurves):
        try:
            stage.RemovePrim(Sdf.Path(path))
        except Exception:
            pass
        curve = UsdGeom.BasisCurves.Define(stage, path)
    else:
        curve = UsdGeom.BasisCurves(prim)

    try:
        curve.GetPointsAttr().Set(points) if curve.GetPointsAttr().IsValid() else curve.CreatePointsAttr(points)
        curve.GetCurveVertexCountsAttr().Set(counts) if curve.GetCurveVertexCountsAttr().IsValid() else curve.CreateCurveVertexCountsAttr(counts)
    except Exception:
        curve.CreatePointsAttr(points)
        curve.CreateCurveVertexCountsAttr(counts)

    try:
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        try:
            curve.GetBasisAttr().Clear()
        except Exception:
            pass
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateWidthsAttr([float(UNLOAD_RANGE_GUIDE_WIDTH if width is None else width)])
    except Exception:
        pass

    prim = curve.GetPrim()
    set_color(prim, color)
    disable_collision(prim, "polygon_wire_column")
    return prim


def bucket_collision_mesh_candidates():
    bucket = get_prim(BUCKET_LINK)
    if not bucket or not bucket.IsValid():
        return []

    collision_meshes = []
    visual_meshes = []
    other_meshes = []
    for prim in iter_prim_subtree(bucket):
        if prim.GetTypeName() != "Mesh":
            continue
        path_text = str(prim.GetPath()).lower()
        if "/collisions/" in path_text:
            collision_meshes.append(prim)
        elif "/visuals/" in path_text:
            visual_meshes.append(prim)
        else:
            other_meshes.append(prim)

    selected = []
    if BUCKET_SAND_USE_VISUAL_MESH_COLLISION:
        selected.extend(visual_meshes)
    selected.extend(collision_meshes)

    if selected:
        seen = set()
        unique = []
        for prim in selected:
            path_text = str(prim.GetPath())
            if path_text in seen:
                continue
            seen.add(path_text)
            unique.append(prim)
        info_print(
            "[ROBOT BUCKET COLLISION SELECT]",
            f"visual_meshes={len(visual_meshes)}",
            f"collision_meshes={len(collision_meshes)}",
            f"selected={len(unique)}",
            f"use_visual={BUCKET_SAND_USE_VISUAL_MESH_COLLISION}",
        )
        return unique

    info_print(
        "[WARN] [ROBOT BUCKET COLLISION] no mesh found under bucket_link visuals/collisions"
    )
    if other_meshes:
        for prim in other_meshes[:8]:
            info_print("[ROBOT BUCKET COLLISION] skipped non-collision mesh:", prim.GetPath())
    return []


def configure_robot_bucket_collision_mesh(prim):
    def do_configure():
        UsdPhysics.CollisionAPI.Apply(prim)
        try:
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("sdf")
        except Exception:
            set_prim_attr(prim, "physics:approximation", "sdf", Sdf.ValueTypeNames.Token)

        sdf_api = apply_physx_api_by_names(prim, ["PhysxSDFMeshCollisionAPI"])
        if sdf_api is not None:
            try:
                sdf_api.CreateSdfResolutionAttr().Set(int(BUCKET_SAND_SDF_RESOLUTION))
            except Exception:
                pass
            try:
                sdf_api.CreateSdfSubgridResolutionAttr().Set(int(BUCKET_SAND_SDF_SUBGRID_RESOLUTION))
            except Exception:
                pass

        collision_api = apply_physx_api_by_names(prim, ["PhysxCollisionAPI"])
        if collision_api is not None:
            try:
                collision_api.CreateContactOffsetAttr().Set(float(BUCKET_SAND_COLLIDER_CONTACT_OFFSET))
            except Exception:
                pass
            try:
                collision_api.CreateRestOffsetAttr().Set(float(BUCKET_SAND_COLLIDER_REST_OFFSET))
            except Exception:
                pass

        set_prim_attr(prim, "physics:collisionEnabled", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(prim, "physxCollision:contactOffset", float(BUCKET_SAND_COLLIDER_CONTACT_OFFSET), Sdf.ValueTypeNames.Float)
        set_prim_attr(prim, "physxCollision:restOffset", float(BUCKET_SAND_COLLIDER_REST_OFFSET), Sdf.ValueTypeNames.Float)
        set_prim_attr(prim, "physxSDFMeshCollision:sdfResolution", int(BUCKET_SAND_SDF_RESOLUTION), Sdf.ValueTypeNames.Int)
        set_prim_attr(prim, "physxSDFMeshCollision:sdfSubgridResolution", int(BUCKET_SAND_SDF_SUBGRID_RESOLUTION), Sdf.ValueTypeNames.Int)
        return True

    try:
        ok = bool(with_root_edit_target(do_configure))
    except Exception as e:
        info_print("[WARN] [ROBOT BUCKET COLLISION] configure failed:", prim.GetPath(), e)
        return False

    enabled_attr = prim.GetAttribute("physics:collisionEnabled")
    approximation_attr = prim.GetAttribute("physics:approximation")
    contact_attr = prim.GetAttribute("physxCollision:contactOffset")
    rest_attr = prim.GetAttribute("physxCollision:restOffset")
    sdf_attr = prim.GetAttribute("physxSDFMeshCollision:sdfResolution")
    rigid_ancestor = nearest_rigid_ancestor_path(prim)
    mn, mx = bbox_min_max(str(prim.GetPath()))
    if rigid_ancestor != BUCKET_LINK:
        info_print(
            "[WARN] [ROBOT BUCKET COLLISION] mesh rigid ancestor is not bucket_link",
            "mesh=", prim.GetPath(),
            "rigid_ancestor=", rigid_ancestor,
            "expected=", BUCKET_LINK,
        )
    info_print(
        "[ROBOT BUCKET COLLISION]",
        "mesh=", prim.GetPath(),
        "ok=", ok,
        "enabled=", enabled_attr.Get() if enabled_attr.IsValid() else None,
        "approx=", approximation_attr.Get() if approximation_attr.IsValid() else None,
        "contact=", contact_attr.Get() if contact_attr.IsValid() else None,
        "rest=", rest_attr.Get() if rest_attr.IsValid() else None,
        "sdf=", sdf_attr.Get() if sdf_attr.IsValid() else None,
        "rigid_ancestor=", rigid_ancestor,
        "bbox_min=", None if mn is None else np.round(mn, 3),
        "bbox_max=", None if mx is None else np.round(mx, 3),
    )
    return ok


def deinstance_bucket_mesh_scopes():
    bucket = get_prim(BUCKET_LINK)
    if not bucket or not bucket.IsValid():
        return 0

    changed = []
    bucket_path = str(bucket.GetPath())
    for suffix in ["visuals", "collisions"]:
        scope = get_prim(f"{bucket_path}/{suffix}")
        if not scope or not scope.IsValid():
            continue
        for prim in iter_prim_subtree(scope):
            try:
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
                    changed.append(str(prim.GetPath()))
            except Exception:
                pass

    if changed:
        info_print("[ROBOT BUCKET DEINSTANCE]", f"count={len(changed)}", f"paths={changed[:8]}")
    return len(changed)


def configure_robot_bucket_sand_collision(label=""):
    if not sand_site_active():
        return False

    deinstance_bucket_mesh_scopes()
    meshes = bucket_collision_mesh_candidates()
    ok_count = 0
    for prim in meshes:
        if configure_robot_bucket_collision_mesh(prim):
            ok_count += 1

    info_print(
        "[ROBOT BUCKET COLLISION SUMMARY]",
        f"label={label}",
        f"bucket={BUCKET_LINK}",
        f"selected_collision_meshes={len(meshes)}",
        f"configured={ok_count}",
    )
    return ok_count > 0


def set_target_color(color):
    if TARGET_PATH is None:
        return
    prim = get_prim(TARGET_PATH)
    if prim and prim.IsValid():
        set_color(prim, color)


def get_sand_site_api():
    api = getattr(builtins, "_SAND_SITE", None)
    return api if isinstance(api, dict) else None


def sand_site_active():
    api = get_sand_site_api()
    return api is not None and bool(api.get("suppress_control_ground", False))


def api_array(api, key, default, shape=None):
    value = None if api is None else api.get(key)
    if callable(value):
        try:
            value = value()
        except Exception as e:
            info_print("[WARN] sand site api callable failed:", key, type(e).__name__, e)
            value = None
    if value is None:
        value = default
    try:
        arr = np.array(value, dtype=np.float32).reshape(-1)
        if shape is not None and len(arr) < int(shape):
            return np.array(default, dtype=np.float32).reshape(-1)
        return arr
    except Exception:
        return np.array(default, dtype=np.float32).reshape(-1)


def task_scene_context():
    api = get_sand_site_api()
    raw = api.get("scene_context") if api is not None else None
    if callable(raw):
        try:
            raw = raw()
        except Exception as e:
            info_print("[WARN] sand site scene_context failed:", type(e).__name__, e)
            raw = None
    if not isinstance(raw, dict):
        raw = {}

    pile_center = api_array(raw, "pile_center", [AUTO_COLLECT_TARGET_CENTER[0], AUTO_COLLECT_TARGET_CENTER[1], GROUND_TOP_Z], 3)
    pile_radius = api_array(raw, "diggable_radius", [SAND_PILE_RADIUS_X, SAND_PILE_RADIUS_Y], 2)
    bin_center = api_array(raw, "unload_bin_center", [float(AUTO_COLLECT_TARGET_CENTER[0]), -float(AUTO_COLLECT_TARGET_CENTER[1]), GROUND_TOP_Z], 3)
    bin_inner = api_array(raw, "unload_bin_inner_size", [2.0 * SAND_BIN_HALF_X, 2.0 * SAND_BIN_HALF_Y], 2)
    bin_z_range = api_array(raw, "unload_bin_z_range", [SAND_BIN_Z_MIN, SAND_BIN_Z_MAX], 2)

    dump_default = [
        float(bin_center[0]),
        float(bin_center[1]),
        max(GROUND_TOP_Z + UNLOAD_TARGET_MIN_Z, float(bin_z_range[1]) + UNLOAD_BIN_DUMP_WALL_CLEARANCE_Z),
    ]
    dump_point = api_array(raw, "unload_bin_dump_point", dump_default, 3)
    if api is not None and "unload_bin_dump_point" not in raw:
        fn = api.get("unload_bin_dump_point")
        try:
            if callable(fn):
                dump_point = np.array(fn(), dtype=np.float32).reshape(-1)
        except Exception as e:
            info_print("[WARN] sand site unload dump point failed:", type(e).__name__, e)
        if dump_point is None or len(dump_point) < 3:
            try:
                p = api.get("dump_point")
                if p is not None:
                    dump_point = np.array(p, dtype=np.float32).reshape(-1)
            except Exception as e:
                info_print("[WARN] sand site dump_point failed:", type(e).__name__, e)

    source = "sand_site" if api is not None else "fallback"
    if bool(STATE.get("manual_unload_override_enabled", False)):
        manual_point = STATE.get("manual_unload_point")
        if manual_point is not None:
            try:
                p = np.array(manual_point, dtype=np.float32).reshape(-1)[:3]
                if len(p) >= 3:
                    bin_center = np.array([float(p[0]), float(p[1]), GROUND_TOP_Z], dtype=np.float32)
                    dump_point = p.copy()
                    manual_inner = STATE.get("manual_unload_inner_size")
                    if manual_inner is not None:
                        size = np.array(manual_inner, dtype=np.float32).reshape(-1)[:2]
                        if len(size) >= 2:
                            bin_inner = np.maximum(size, np.array([0.2, 0.2], dtype=np.float32))
                    manual_z_range = STATE.get("manual_unload_z_range")
                    if manual_z_range is not None:
                        zr = np.array(manual_z_range, dtype=np.float32).reshape(-1)[:2]
                        if len(zr) >= 2:
                            bin_z_range = np.array([min(float(zr[0]), float(zr[1])), max(float(zr[0]), float(zr[1]))], dtype=np.float32)
                    else:
                        bin_z_range = np.array([GROUND_TOP_Z - 0.05, max(GROUND_TOP_Z, float(p[2]))], dtype=np.float32)
                    source = "manual_unload_override"
            except Exception as e:
                info_print("[WARN] manual unload override ignored:", type(e).__name__, e)

    bin_center = np.array(bin_center[:3], dtype=np.float32)
    bin_inner = np.maximum(bin_inner[:2], np.array([0.2, 0.2], dtype=np.float32))
    bin_half = 0.5 * bin_inner
    bin_z_range = np.array([float(bin_z_range[0]), float(bin_z_range[1])], dtype=np.float32)
    dump_raw = np.array(dump_point[:3], dtype=np.float32)
    dump_z = max(
        float(dump_raw[2]),
        GROUND_TOP_Z + UNLOAD_TARGET_MIN_Z,
        float(bin_z_range[1]) + UNLOAD_BIN_DUMP_WALL_CLEARANCE_Z,
    )
    unload_point = np.array([float(bin_center[0]), float(bin_center[1]), dump_z], dtype=np.float32)

    return {
        "source": source,
        "pile_center": pile_center[:3],
        "pile_radius": np.maximum(pile_radius[:2], np.array([0.1, 0.1], dtype=np.float32)),
        "unload_bin_center": bin_center,
        "unload_bin_inner_size": bin_inner,
        "unload_bin_half_size": bin_half,
        "unload_bin_z_range": bin_z_range,
        "unload_point_raw": dump_raw,
        "unload_point": unload_point,
    }


def unload_bin_dump_point(height_delta=0.0, xy_offset=None, ctx=None):
    ctx = task_scene_context() if ctx is None else ctx
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)
    p = np.array(ctx["unload_point"], dtype=np.float32).copy()
    active = STATE.get("active_unload_landing_point")
    if xy_offset is None and active is not None:
        try:
            active_arr = np.array(active, dtype=np.float32).reshape(-1)
            if len(active_arr) >= 2:
                p[0] = float(active_arr[0])
                p[1] = float(active_arr[1])
            else:
                p[0] = float(bin_center[0])
                p[1] = float(bin_center[1])
        except Exception:
            p[0] = float(bin_center[0])
            p[1] = float(bin_center[1])
    else:
        p[0] = float(bin_center[0])
        p[1] = float(bin_center[1])
    if xy_offset is not None:
        off = np.array(xy_offset, dtype=np.float32).reshape(-1)
        if len(off) >= 2:
            safe_half = np.maximum(bin_half[:2] - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
            p[0] += float(np.clip(float(off[0]), -float(safe_half[0]), float(safe_half[0])))
            p[1] += float(np.clip(float(off[1]), -float(safe_half[1]), float(safe_half[1])))
    p[2] += float(height_delta)
    return p


def unload_bin_landing_point(height_delta=0.06, xy_offset=None, ctx=None):
    ctx = task_scene_context() if ctx is None else ctx
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
    z = float(bin_z_range[1]) + float(height_delta) if len(bin_z_range) >= 2 else GROUND_TOP_Z + float(height_delta)
    p = np.array([float(bin_center[0]), float(bin_center[1]), z], dtype=np.float32)
    active = STATE.get("active_unload_landing_point")
    if xy_offset is None and active is not None:
        try:
            active_arr = np.array(active, dtype=np.float32).reshape(-1)
            if len(active_arr) >= 2:
                p[0] = float(active_arr[0])
                p[1] = float(active_arr[1])
        except Exception:
            pass
    if xy_offset is not None:
        off = np.array(xy_offset, dtype=np.float32).reshape(-1)
        if len(off) >= 2:
            safe_half = np.maximum(bin_half[:2] - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
            p[0] += float(np.clip(float(off[0]), -float(safe_half[0]), float(safe_half[0])))
            p[1] += float(np.clip(float(off[1]), -float(safe_half[1]), float(safe_half[1])))
    return p


def unload_landing_point_from_release(point=None):
    p = unload_bin_landing_point()
    if point is not None:
        src = np.array(point, dtype=np.float32).reshape(-1)[:3]
        p[0] = float(src[0])
        p[1] = float(src[1])
        if len(src) >= 3:
            p[2] = float(src[2])
    return p


def choose_unload_landing_point_for_flat_fill():
    ctx = task_scene_context()
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)[:2]
    safe_half = np.maximum(
        bin_half - max(float(UNLOAD_BIN_SAFE_XY_MARGIN), float(AUTO_UNLOAD_WALL_MARGIN)),
        np.array([0.04, 0.04], dtype=np.float32),
    )
    wall_top = float(bin_z_range[1]) if len(bin_z_range) >= 2 else GROUND_TOP_Z
    landing_z = wall_top + 0.06
    points = sand_particle_positions()
    grid = max(3, int(AUTO_UNLOAD_GRID_SIZE))
    xs = np.linspace(float(bin_center[0] - safe_half[0]), float(bin_center[0] + safe_half[0]), grid)
    ys = np.linspace(float(bin_center[1] - safe_half[1]), float(bin_center[1] + safe_half[1]), grid)
    cell_half = np.array([
        max(0.04, float(safe_half[0]) / max(1, grid - 1)),
        max(0.04, float(safe_half[1]) / max(1, grid - 1)),
    ], dtype=np.float32)
    bucket_ref = bucket_pour_pos()
    bucket_xy = np.array([float(bin_center[0]), float(bin_center[1])], dtype=np.float32)
    if bucket_ref is not None:
        bucket_xy = np.array(bucket_ref[:2], dtype=np.float32)

    rows = []
    for x in xs:
        for y in ys:
            center = np.array([float(x), float(y), 0.0], dtype=np.float32)
            if points is None or len(points) == 0:
                cell_count = 0
                cell_height = float(AUTO_UNLOAD_EMPTY_CELL_HEIGHT)
            else:
                mask = mask_points_in_box(points, center, cell_half, float(bin_z_range[0]), float(bin_z_range[1] + 0.60))
                cell_points = points[mask]
                cell_count = int(len(cell_points))
                cell_height = float(np.percentile(cell_points[:, 2], 90.0)) if cell_count > 0 else float(AUTO_UNLOAD_EMPTY_CELL_HEIGHT)
            center_dist = float(np.linalg.norm(np.array([x - bin_center[0], y - bin_center[1]], dtype=np.float32) / np.maximum(safe_half, 1e-4)))
            motion_dist = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - bucket_xy))
            score = (
                float(AUTO_UNLOAD_FILL_HEIGHT_WEIGHT) * cell_height
                + float(AUTO_UNLOAD_CENTER_WEIGHT) * center_dist
                + float(AUTO_UNLOAD_MOTION_WEIGHT) * motion_dist
            )
            rows.append(
                {
                    "landing": [float(x), float(y), float(landing_z)],
                    "cell_height": float(cell_height),
                    "cell_count": cell_count,
                    "center_norm": float(center_dist),
                    "motion_dist": float(motion_dist),
                    "score": float(score),
                }
            )

    rows.sort(key=lambda row: float(row.get("score", 1.0e9)))
    chosen = rows[0] if rows else {"landing": vec_list(unload_bin_landing_point(ctx=ctx), 3), "score": 0.0}
    landing = np.array(chosen["landing"], dtype=np.float32)
    STATE["active_unload_landing_point"] = landing.copy()
    STATE["last_auto_unload_scores"] = rows
    info_print(
        "[AUTO UNLOAD SELECT]",
        f"landing={vec_list(landing, 3)}",
        f"cell_height={fmt_optional(chosen.get('cell_height'))}",
        f"cell_count={chosen.get('cell_count')}",
        f"score={fmt_optional(chosen.get('score'))}",
    )
    return landing, rows


def log_unload_context(label, target_xyz=None, unload_point=None):
    ctx = task_scene_context()
    unload = unload_bin_dump_point(ctx=ctx) if unload_point is None else np.array(unload_point, dtype=np.float32).reshape(-1)[:3]
    landing = unload_bin_landing_point(ctx=ctx)
    pile_center = np.array(ctx.get("pile_center", [0.0, 0.0, GROUND_TOP_Z]), dtype=np.float32).reshape(-1)[:3]
    pile_radius = np.array(ctx.get("pile_radius", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
    target = None if target_xyz is None else np.array(target_xyz, dtype=np.float32).reshape(-1)[:3]
    target_dist = None
    if target is not None:
        target_dist = float(np.linalg.norm(unload[:2] - target[:2]))
    pile_dist = float(np.linalg.norm(unload[:2] - pile_center[:2]))
    info_print(
        f"[UNLOAD TARGET] {label}: "
        f"source={ctx.get('source')} "
        f"target={vec_list(target, 3)} "
        f"unload={vec_list(unload, 3)} "
        f"landing={vec_list(landing, 3)} "
        f"raw={vec_list(ctx.get('unload_point_raw'), 3)} "
        f"pile_center={vec_list(pile_center, 3)} "
        f"pile_radius={vec_list(pile_radius, 2)} "
        f"bin_center={vec_list(ctx.get('unload_bin_center'), 3)} "
        f"bin_half={vec_list(ctx.get('unload_bin_half_size'), 2)} "
        f"bin_z_range={vec_list(ctx.get('unload_bin_z_range'), 2)} "
        f"target_to_unload_xy={fmt_optional(target_dist)}m "
        f"pile_to_unload_xy={fmt_optional(pile_dist)}m"
    )
    if target_dist is not None and target_dist < 0.75:
        info_print(
            f"[WARN] [UNLOAD TARGET] {label}: unload point is very close to dig target; "
            f"target_to_unload_xy={target_dist:.3f}m"
        )
    if pile_dist < max(0.75, float(max(pile_radius)) * 0.60):
        info_print(
            f"[WARN] [UNLOAD TARGET] {label}: unload point is close to pile center; "
            f"pile_to_unload_xy={pile_dist:.3f}m"
        )


def ensure_unload_marker(point=None, label=""):
    if CONTROL_ROOT is None:
        return None
    global UNLOAD_MARKER_PATH
    UNLOAD_MARKER_PATH = f"{CONTROL_ROOT}/UnloadPointBall"
    p = unload_landing_point_from_release(point)
    prim = get_prim(UNLOAD_MARKER_PATH)
    if not prim or not prim.IsValid():
        prim = make_sphere(
            UNLOAD_MARKER_PATH,
            translate=(float(p[0]), float(p[1]), float(p[2])),
            radius=UNLOAD_MARKER_RADIUS,
            color=UNLOAD_MARKER_COLOR,
            collision=False,
        )
        disable_collision(prim, "unload_point_ball")
    else:
        set_xform(prim, translate=(float(p[0]), float(p[1]), float(p[2])))
        set_color(prim, UNLOAD_MARKER_COLOR)
        disable_collision(prim, "unload_point_ball")
    if label:
        info_print(
            f"[UNLOAD MARKER] {label}: "
            f"role=sand_landing_target_not_bucket_release "
            f"path={UNLOAD_MARKER_PATH} pos={vec_list(p, 3)}"
        )
    ensure_unload_range_column(p, label=label)
    return prim


def manual_unload_radius():
    try:
        return max(0.05, float(STATE.get("manual_unload_radius", UNLOAD_POINT_DEFAULT_RADIUS)))
    except Exception:
        return float(UNLOAD_POINT_DEFAULT_RADIUS)


def manual_unload_mesh_shrink_d():
    try:
        return max(0.0, float(STATE.get("manual_unload_mesh_shrink_d", UNLOAD_SELECTED_EDGE_MARGIN)))
    except Exception:
        return float(UNLOAD_SELECTED_EDGE_MARGIN)


def ensure_unload_range_column(point=None, label=""):
    if CONTROL_ROOT is None:
        return None
    p = unload_landing_point_from_release(point)
    z_min = float(GROUND_TOP_Z)
    z_max = max(float(p[2]), z_min + 0.25)
    z_range = STATE.get("manual_unload_z_range")
    if z_range is not None:
        try:
            zr = np.array(z_range, dtype=np.float32).reshape(-1)[:2]
            if len(zr) >= 2:
                z_min = min(float(zr[0]), float(zr[1]))
                z_max = max(float(zr[0]), float(zr[1]))
        except Exception:
            pass
    if z_max - z_min < 0.05:
        z_max = z_min + 0.05
    height = float(z_max - z_min)
    center = (float(p[0]), float(p[1]), z_min + 0.5 * height)
    shape = str(STATE.get("manual_unload_range_shape", "circle") or "circle")
    box_path = f"{CONTROL_ROOT}/UnloadSelectedRangeBox"
    cyl_path = f"{CONTROL_ROOT}/UnloadSelectedRangeCylinder"

    if shape in ("box", "mesh"):
        try:
            stage.RemovePrim(Sdf.Path(cyl_path))
        except Exception:
            pass
        inner = STATE.get("manual_unload_inner_size")
        if inner is None:
            radius = manual_unload_radius()
            inner = np.array([2.0 * radius, 2.0 * radius], dtype=np.float32)
        inner = np.maximum(np.array(inner, dtype=np.float32).reshape(-1)[:2], np.array([0.05, 0.05], dtype=np.float32))
        poly = STATE.get("manual_unload_polygon_xy")
        if poly is not None:
            visual_poly = visual_polygon_xy(poly, UNLOAD_RANGE_VISUAL_MAX_VERTICES)
            prim = make_polygon_wire_column(
                box_path,
                polygon_xy=visual_poly,
                z_min=z_min,
                z_max=z_max,
                color=UNLOAD_RANGE_COLUMN_COLOR,
                width=UNLOAD_RANGE_GUIDE_WIDTH,
            )
        else:
            half_x = 0.5 * float(inner[0])
            half_y = 0.5 * float(inner[1])
            visual_poly = np.array(
                [
                    [float(p[0]) - half_x, float(p[1]) - half_y],
                    [float(p[0]) + half_x, float(p[1]) - half_y],
                    [float(p[0]) + half_x, float(p[1]) + half_y],
                    [float(p[0]) - half_x, float(p[1]) + half_y],
                ],
                dtype=np.float32,
            )
            prim = make_polygon_wire_column(
                box_path,
                polygon_xy=visual_poly,
                z_min=z_min,
                z_max=z_max,
                color=UNLOAD_RANGE_COLUMN_COLOR,
                width=UNLOAD_RANGE_GUIDE_WIDTH,
            )
        if label:
            info_print(
                f"[UNLOAD RANGE] {label}: "
                f"shape=box path={box_path} center={vec_list(center, 3)} "
                f"inner_size={vec_list(inner, 2)} vertices={0 if poly is None else len(poly)} "
                f"visual_vertices={0 if poly is None else len(visual_polygon_xy(poly, UNLOAD_RANGE_VISUAL_MAX_VERTICES))} z_range=({z_min:.3f},{z_max:.3f}) height={height:.3f} "
                f"mesh_shrink_d={manual_unload_mesh_shrink_d():.3f}"
            )
        return prim

    try:
        stage.RemovePrim(Sdf.Path(box_path))
    except Exception:
        pass
    try:
        stage.RemovePrim(Sdf.Path(cyl_path))
    except Exception:
        pass
    radius = manual_unload_radius()
    circle_poly = circle_polygon_xy(float(p[0]), float(p[1]), radius, UNLOAD_RANGE_CIRCLE_SEGMENTS)
    prim = make_polygon_wire_column(
        cyl_path,
        polygon_xy=circle_poly,
        z_min=z_min,
        z_max=z_max,
        color=UNLOAD_RANGE_COLUMN_COLOR,
        width=UNLOAD_RANGE_GUIDE_WIDTH,
    )
    if label:
        info_print(
            f"[UNLOAD RANGE] {label}: "
            f"shape=circle path={cyl_path} center={vec_list(center, 3)} radius={radius:.3f} z_range=({z_min:.3f},{z_max:.3f}) height={height:.3f}"
        )
    return prim


def bucket_point_world(name, q=None, reference_q=None):
    name = str(name)
    if q is not None:
        ref = CTRL.q_cmd if reference_q is None else reference_q
        return predicted_end_world_point(q, end_effector=name, reference_q=ref)
    if name == "tip":
        return bucket_tip_pos()
    if name == "load":
        return bucket_load_pos()
    if name == "pour":
        return bucket_pour_pos()
    return bucket_mid_pos()


def unload_alignment_report(q=None, effector="pour", reference_q=None):
    ctx = task_scene_context()
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)[:2]
    safe_half = np.maximum(bin_half - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
    effector = "pour" if str(effector) == "dump" else str(effector)
    load = bucket_point_world("load", q=q, reference_q=reference_q)
    pour = bucket_point_world("pour", q=q, reference_q=reference_q)
    point = pour if effector == "pour" else load
    if point is None:
        return {
            "ok": False,
            "reason": f"{effector}_point_unavailable",
            "effector": effector,
            "source": str(ctx.get("source", "unknown")),
            "bin_center": vec_list(bin_center, 3),
            "bin_half": vec_list(bin_half, 2),
            "bin_z_range": vec_list(bin_z_range, 2),
            "load": vec_list(load, 3),
            "pour": vec_list(pour, 3),
        }

    point = np.array(point, dtype=np.float32).reshape(-1)[:3]
    dx = float(point[0] - bin_center[0])
    dy = float(point[1] - bin_center[1])
    dz_wall = float(point[2] - bin_z_range[1])
    inside_xy = abs(dx) <= float(safe_half[0]) and abs(dy) <= float(safe_half[1])
    above_wall = dz_wall >= 0.05
    return {
        "ok": bool(inside_xy and above_wall),
        "effector": effector,
        "inside_xy": bool(inside_xy),
        "above_wall": bool(above_wall),
        "source": str(ctx.get("source", "unknown")),
        "point": vec_list(point, 3),
        "load": vec_list(load, 3),
        "pour": vec_list(pour, 3),
        "unload_point": vec_list(unload_bin_dump_point(ctx=ctx), 3),
        "bin_center": vec_list(bin_center, 3),
        "bin_half": vec_list(bin_half, 2),
        "safe_half": vec_list(safe_half, 2),
        "bin_z_range": vec_list(bin_z_range, 2),
        "dx": dx,
        "dy": dy,
        "dz_wall": dz_wall,
    }


def log_unload_alignment(label, q=None, effector="pour", reference_q=None):
    report = unload_alignment_report(q=q, effector=effector, reference_q=reference_q)
    info_print(
        f"[UNLOAD ALIGN] {label}: "
        f"effector={report.get('effector')} "
        f"ok={report.get('ok')} "
        f"inside_xy={report.get('inside_xy')} "
        f"above_wall={report.get('above_wall')} "
        f"point={report.get('point')} "
        f"load={report.get('load')} "
        f"pour={report.get('pour')} "
        f"unload={report.get('unload_point')} "
        f"bin_center={report.get('bin_center')} "
        f"safe_half={report.get('safe_half')} "
        f"bin_z_range={report.get('bin_z_range')} "
        f"dx={fmt_optional(report.get('dx'))} "
        f"dy={fmt_optional(report.get('dy'))} "
        f"dz_wall={fmt_optional(report.get('dz_wall'))} "
        f"source={report.get('source')}"
    )
    return report


def bucket_dump_forward_xy(q=None, reference_q=None):
    if q is None:
        q = CTRL.q_cmd.copy()
    q = np.array(q, dtype=np.float32).reshape(-1)
    load = bucket_point_world("load", q=q, reference_q=reference_q)
    pour = bucket_point_world("pour", q=q, reference_q=reference_q)
    if load is not None and pour is not None:
        load = np.array(load, dtype=np.float32).reshape(-1)[:3]
        pour = np.array(pour, dtype=np.float32).reshape(-1)[:3]
        mouth_xy = pour[:2] - load[:2]
        if float(np.linalg.norm(mouth_xy)) > 0.03:
            return safe_norm(mouth_xy, default=(1.0, 0.0))

    swing = float(q[CTRL.name_to_idx["swing"]])
    radial = np.array([math.cos(swing), math.sin(swing)], dtype=np.float32)
    sign = 1.0
    try:
        angles = chain_angles_from_q(q, end_effector="pour")
        if angles is not None and math.cos(float(angles[2])) < 0.0:
            sign = -1.0
    except Exception:
        pass
    return safe_norm(radial * sign, default=(1.0, 0.0))


def unload_drop_drift_model(q=None, reference_q=None, release=None, load=None, wall_z=None):
    q_eval = CTRL.q_cmd.copy() if q is None else np.array(q, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = CTRL.q_cmd.copy() if reference_q is None else np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    forward_xy = bucket_dump_forward_xy(q_eval, reference_q=q_ref)

    release_arr = None if release is None else np.array(release, dtype=np.float32).reshape(-1)[:3]
    load_arr = None if load is None else np.array(load, dtype=np.float32).reshape(-1)[:3]
    if release_arr is None:
        release_arr = bucket_point_world("pour", q=q_eval, reference_q=q_ref)
        release_arr = None if release_arr is None else np.array(release_arr, dtype=np.float32).reshape(-1)[:3]
    if load_arr is None:
        load_arr = bucket_point_world("load", q=q_eval, reference_q=q_ref)
        load_arr = None if load_arr is None else np.array(load_arr, dtype=np.float32).reshape(-1)[:3]

    if release_arr is None:
        height_above_wall = 0.0
    else:
        wall = GROUND_TOP_Z if wall_z is None else float(wall_z)
        height_above_wall = max(0.0, float(release_arr[2]) - wall)

    gravity = max(1e-3, float(UNLOAD_DROP_GRAVITY))
    fall_time = math.sqrt(max(0.0, 2.0 * height_above_wall / gravity))
    lip_span_xy = 0.55
    if release_arr is not None and load_arr is not None:
        lip_span_xy = max(0.10, float(np.linalg.norm((release_arr - load_arr)[:2])))

    bucket_delta = abs(wrap_angle(float(q_eval[CTRL.name_to_idx["bucket"]]) - float(q_ref[CTRL.name_to_idx["bucket"]])))
    peak_bucket_omega = 1.875 * bucket_delta / max(0.08, float(UNLOAD_DUMP_SECONDS))
    tangential_offset = float(UNLOAD_DROP_TANGENTIAL_GAIN) * peak_bucket_omega * lip_span_xy * fall_time
    roll_offset = float(UNLOAD_DROP_ROLL_OFFSET_BASE) + float(UNLOAD_DROP_ROLL_OFFSET_PER_M) * height_above_wall
    total_offset = clamp(roll_offset + tangential_offset, 0.0, float(UNLOAD_DROP_MAX_ROLL_OFFSET))
    drift_xy = np.array(forward_xy, dtype=np.float32) * float(total_offset)

    return {
        "forward_xy": np.array(forward_xy, dtype=np.float32),
        "drift_xy": drift_xy,
        "drift_distance": float(total_offset),
        "roll_offset": float(roll_offset),
        "tangential_offset": float(tangential_offset),
        "fall_time": float(fall_time),
        "height_above_wall": float(height_above_wall),
        "lip_span_xy": float(lip_span_xy),
        "bucket_delta_deg": float(rad_to_deg(bucket_delta)),
        "peak_bucket_omega": float(peak_bucket_omega),
        "gravity": float(gravity),
    }


def unload_release_target_for_landing(landing_target, q_estimate=None, q_start=None, release_z=None, wall_z=None):
    landing = np.array(landing_target, dtype=np.float32).reshape(-1)[:3].copy()
    q_eval = CTRL.q_cmd.copy() if q_estimate is None else np.array(q_estimate, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = q_eval.copy() if q_start is None else np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    release = landing.copy()
    if release_z is not None:
        release[2] = float(release_z)
    load = bucket_point_world("load", q=q_eval, reference_q=q_ref)
    drift = unload_drop_drift_model(q=q_eval, reference_q=q_ref, release=release, load=load, wall_z=wall_z)
    release[0] -= float(drift["drift_xy"][0])
    release[1] -= float(drift["drift_xy"][1])
    return release, drift


def predict_unload_drop(q=None, reference_q=None):
    ctx = task_scene_context()
    q_ref = CTRL.q_cmd if reference_q is None else reference_q
    release = bucket_point_world("pour", q=q, reference_q=q_ref)
    load = bucket_point_world("load", q=q, reference_q=q_ref)
    if release is None:
        return None

    release = np.array(release, dtype=np.float32).reshape(-1)[:3]
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
    wall_z = float(bin_z_range[1]) if len(bin_z_range) >= 2 else GROUND_TOP_Z
    drift = unload_drop_drift_model(q=q if q is not None else CTRL.q_cmd, reference_q=q_ref, release=release, load=load, wall_z=wall_z)
    forward_xy = np.array(drift["forward_xy"], dtype=np.float32)
    landing = release.copy()
    landing[0] += float(drift["drift_xy"][0])
    landing[1] += float(drift["drift_xy"][1])
    landing[2] = wall_z

    return {
        "release": release,
        "load": None if load is None else np.array(load, dtype=np.float32).reshape(-1)[:3],
        "landing": landing,
        "forward_xy": forward_xy,
        "drift_xy": np.array(drift["drift_xy"], dtype=np.float32),
        "drift_distance": float(drift["drift_distance"]),
        "roll_offset": float(drift["roll_offset"]),
        "tangential_offset": float(drift["tangential_offset"]),
        "fall_time": float(drift["fall_time"]),
        "height_above_wall": float(drift["height_above_wall"]),
        "lip_span_xy": float(drift["lip_span_xy"]),
        "bucket_delta_deg": float(drift["bucket_delta_deg"]),
        "peak_bucket_omega": float(drift["peak_bucket_omega"]),
        "gravity": float(drift["gravity"]),
        "wall_z": float(wall_z),
        "source_clearance": float(release[2]) - float(wall_z),
        "source": str(ctx.get("source", "unknown")),
    }


def unload_drop_report(q=None, reference_q=None):
    ctx = task_scene_context()
    target = unload_bin_landing_point(ctx=ctx)
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
    safe_half = np.maximum(bin_half - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
    drop = predict_unload_drop(q=q, reference_q=reference_q)
    if drop is None:
        return {
            "ok": False,
            "reason": "release_point_unavailable",
            "target": vec_list(target, 3),
            "bin_center": vec_list(bin_center, 3),
            "safe_half": vec_list(safe_half, 2),
            "source": str(ctx.get("source", "unknown")),
        }

    landing = np.array(drop["landing"], dtype=np.float32).reshape(-1)[:3]
    release = np.array(drop["release"], dtype=np.float32).reshape(-1)[:3]
    dx = float(landing[0] - target[0])
    dy = float(landing[1] - target[1])
    xy_err = float(math.sqrt(dx * dx + dy * dy))
    bin_dx = float(landing[0] - bin_center[0])
    bin_dy = float(landing[1] - bin_center[1])
    inside_xy = abs(bin_dx) <= float(safe_half[0]) and abs(bin_dy) <= float(safe_half[1])
    above_wall = float(drop["source_clearance"]) >= UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z
    close_xy = xy_err <= UNLOAD_DROP_XY_TOL

    ok = bool(inside_xy and above_wall)
    return {
        "ok": ok,
        "reason": "ok" if ok else "landing_outside_bin_or_too_low",
        "inside_xy": bool(inside_xy),
        "above_wall": bool(above_wall),
        "close_xy": bool(close_xy),
        "target": vec_list(target, 3),
        "landing": vec_list(landing, 3),
        "release": vec_list(release, 3),
        "load": vec_list(drop.get("load"), 3),
        "forward_xy": vec_list(drop.get("forward_xy"), 2),
        "drift_xy": vec_list(drop.get("drift_xy"), 2),
        "drift_distance": float(drop.get("drift_distance", drop.get("roll_offset", 0.0))),
        "roll_offset": float(drop.get("roll_offset", 0.0)),
        "tangential_offset": float(drop.get("tangential_offset", 0.0)),
        "fall_time": float(drop.get("fall_time", 0.0)),
        "height_above_wall": float(drop.get("height_above_wall", 0.0)),
        "lip_span_xy": float(drop.get("lip_span_xy", 0.0)),
        "bucket_delta_deg": float(drop.get("bucket_delta_deg", 0.0)),
        "source_clearance": float(drop.get("source_clearance", 0.0)),
        "bin_center": vec_list(bin_center, 3),
        "safe_half": vec_list(safe_half, 2),
        "dx": dx,
        "dy": dy,
        "xy_err": xy_err,
        "source": str(ctx.get("source", "unknown")),
    }


def log_unload_drop(label, q=None, reference_q=None):
    report = unload_drop_report(q=q, reference_q=reference_q)
    info_print(
        f"[UNLOAD DROP] {label}: "
        f"ok={report.get('ok')} "
        f"inside_xy={report.get('inside_xy')} "
        f"above_wall={report.get('above_wall')} "
        f"close_xy={report.get('close_xy')} "
        f"target={report.get('target')} "
        f"landing={report.get('landing')} "
        f"release={report.get('release')} "
        f"load={report.get('load')} "
        f"dx={fmt_optional(report.get('dx'))} "
        f"dy={fmt_optional(report.get('dy'))} "
        f"xy_err={fmt_optional(report.get('xy_err'))} "
        f"source_clearance={fmt_optional(report.get('source_clearance'))} "
        f"drift={fmt_optional(report.get('drift_distance'))} "
        f"drift_xy={report.get('drift_xy')} "
        f"roll={fmt_optional(report.get('roll_offset'))} "
        f"tangent={fmt_optional(report.get('tangential_offset'))} "
        f"fall_t={fmt_optional(report.get('fall_time'))} "
        f"bucket_delta={fmt_optional(report.get('bucket_delta_deg'))}deg "
        f"forward_xy={report.get('forward_xy')} "
        f"source={report.get('source')}"
    )
    return report


def actual_unload_position_report():
    ctx = task_scene_context()
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)[:2]
    marker = unload_bin_landing_point(ctx=ctx)
    safe_half = np.maximum(bin_half - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
    gate_half = safe_half + float(UNLOAD_ACTUAL_BUCKET_XY_MARGIN)

    def point_report(name, point):
        if point is None:
            return {
                "name": name,
                "point": None,
                "inside_gate": False,
                "above_wall": False,
                "marker_xy_ok": False,
                "xy_err": None,
                "dx": None,
                "dy": None,
                "dz_wall": None,
            }
        p = np.array(point, dtype=np.float32).reshape(-1)[:3]
        dx = float(p[0] - bin_center[0])
        dy = float(p[1] - bin_center[1])
        dz_wall = float(p[2] - bin_z_range[1])
        marker_dx = float(p[0] - marker[0])
        marker_dy = float(p[1] - marker[1])
        xy_err = float(math.sqrt(marker_dx * marker_dx + marker_dy * marker_dy))
        inside_gate = abs(dx) <= float(gate_half[0]) and abs(dy) <= float(gate_half[1])
        above_wall = dz_wall >= float(UNLOAD_ACTUAL_MIN_ABOVE_WALL_Z)
        marker_xy_ok = xy_err <= float(UNLOAD_ACTUAL_LOAD_MARKER_XY_TOL)
        return {
            "name": name,
            "point": vec_list(p, 3),
            "inside_gate": bool(inside_gate),
            "above_wall": bool(above_wall),
            "marker_xy_ok": bool(marker_xy_ok),
            "xy_err": xy_err,
            "dx": dx,
            "dy": dy,
            "dz_wall": dz_wall,
        }

    load_report = point_report("load", bucket_load_pos())
    pour_report = point_report("pour", bucket_pour_pos())
    try:
        q_real = current_real_q_near()
        q_dump = bucket_only_dump_pose(q_real, unload_dump_target_deg())
        drop_report = unload_drop_report(q=q_dump, reference_q=q_real)
    except Exception as e:
        drop_report = {
            "ok": False,
            "reason": f"bucket_only_drop_check_failed:{type(e).__name__}:{e}",
            "target": vec_list(marker, 3),
        }
    release_report = point_report("release", drop_report.get("release"))
    ok = bool(
        drop_report.get("ok", False)
        and drop_report.get("close_xy", False)
        and release_report.get("inside_gate", False)
        and release_report.get("above_wall", False)
    )
    return {
        "ok": ok,
        "load": load_report,
        "pour": pour_report,
        "release": release_report,
        "drop": drop_report,
        "bin_center": vec_list(bin_center, 3),
        "safe_half": vec_list(safe_half, 2),
        "gate_half": vec_list(gate_half, 2),
        "bin_z_range": vec_list(bin_z_range, 2),
        "marker": vec_list(marker, 3),
        "source": str(ctx.get("source", "unknown")),
    }


def log_actual_unload_position(label):
    report = actual_unload_position_report()
    load = report.get("load", {})
    pour = report.get("pour", {})
    release = report.get("release", {})
    drop = report.get("drop", {})
    info_print(
        f"[UNLOAD ACTUAL GATE] {label}: "
        f"ok={report.get('ok')} "
        f"marker={report.get('marker')} "
        f"bin_center={report.get('bin_center')} "
        f"gate_half={report.get('gate_half')} "
        f"load_point={load.get('point')} load_inside={load.get('inside_gate')} "
        f"load_above={load.get('above_wall')} load_marker_ok={load.get('marker_xy_ok')} "
        f"load_xy_err={fmt_optional(load.get('xy_err'))} "
        f"pour_point={pour.get('point')} pour_inside={pour.get('inside_gate')} "
        f"pour_above={pour.get('above_wall')} pour_marker_ok={pour.get('marker_xy_ok')} "
        f"pour_xy_err={fmt_optional(pour.get('xy_err'))} "
        f"release_point={release.get('point')} release_inside={release.get('inside_gate')} "
        f"release_above={release.get('above_wall')} release_marker_ok={release.get('marker_xy_ok')} "
        f"release_xy_err={fmt_optional(release.get('xy_err'))} "
        f"drop_ok={drop.get('ok')} drop_target={drop.get('target')} "
        f"drop_landing={drop.get('landing')} drop_xy_err={fmt_optional(drop.get('xy_err'))} "
        f"drop_close_xy={drop.get('close_xy')} "
        f"drop_drift={fmt_optional(drop.get('drift_distance'))} "
        f"drop_drift_xy={drop.get('drift_xy')} "
        f"drop_source_clearance={fmt_optional(drop.get('source_clearance'))} "
        f"source={report.get('source')}"
    )
    return report


def unload_arrival_report(q_goal=None):
    ctx = task_scene_context()
    target = unload_bin_landing_point(ctx=ctx)
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)[:2]
    wall_z = float(bin_z_range[1]) if len(bin_z_range) >= 2 else GROUND_TOP_Z
    q_goal_arr = CTRL.q_cmd.copy() if q_goal is None else np.array(q_goal, dtype=np.float32).copy()

    try:
        q_real_raw = get_real_joint_positions()
        q_real = q_real_near_command(q_real_raw, q_goal_arr)
        err_deg = q_delta_abs_deg(q_goal_arr, q_real)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"real_joint_read_failed:{type(e).__name__}:{e}",
            "q_goal": q_deg_values(q_goal_arr, wrap_swing_for_display=True),
            "q_real": None,
        }

    swing_idx = CTRL.name_to_idx["swing"]
    q_action_goal = CTRL.clip_action_limits(q_goal_arr.copy())
    swing_err = float(err_deg[swing_idx])
    swing_action_err = abs(rad_to_deg(swing_delta(q_action_goal[swing_idx], q_real_raw[swing_idx])))
    swing_goal_raw = float(q_goal_arr[swing_idx])
    swing_goal_action = float(q_action_goal[swing_idx])
    swing_real_raw = float(q_real_raw[swing_idx])
    swing_near_boundary = (
        abs(abs(wrap_angle(swing_goal_raw)) - math.pi) <= deg_to_rad(3.0)
        or abs(abs(wrap_angle(swing_goal_action)) - math.pi) <= deg_to_rad(3.0)
        or abs(abs(wrap_angle(swing_real_raw)) - math.pi) <= deg_to_rad(3.0)
    )
    non_swing_err = 0.0
    blocked = []
    for name, idx in CTRL.name_to_idx.items():
        e = float(err_deg[idx])
        if name == "swing":
            if e > UNLOAD_FINAL_SWING_TOL_DEG or swing_action_err > UNLOAD_FINAL_SWING_TOL_DEG:
                blocked.append(f"{name}:near={e:.2f}deg action={swing_action_err:.2f}deg")
        elif name == "bucket":
            continue
        else:
            non_swing_err = max(non_swing_err, e)
            if e > UNLOAD_FINAL_JOINT_TOL_DEG:
                blocked.append(f"{name}:{e:.2f}deg")

    load = bucket_load_pos()
    load_point = None
    load_xy_err = None
    load_z_clearance = None
    if load is not None:
        load_arr = np.array(load, dtype=np.float32).reshape(-1)[:3]
        dx = float(load_arr[0] - target[0])
        dy = float(load_arr[1] - target[1])
        load_xy_err = float(math.sqrt(dx * dx + dy * dy))
        load_z_clearance = float(load_arr[2] - wall_z)
        load_point = vec_list(load_arr, 3)

    q_dump = bucket_only_dump_pose(q_real, unload_dump_target_deg())
    drop = unload_drop_report(q=q_dump, reference_q=q_real)
    drop_ok = bool(drop.get("ok", False) and drop.get("close_xy", False))
    release_inside_gate = False
    release_z_clearance = None
    release_xy_err = None
    release_point = None
    try:
        release = drop.get("release")
        if release is not None:
            release_arr = np.array(release, dtype=np.float32).reshape(-1)[:3]
            bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
            bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
            safe_half = np.maximum(bin_half - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
            gate_half = safe_half + float(UNLOAD_ACTUAL_BUCKET_XY_MARGIN)
            release_inside_gate = (
                abs(float(release_arr[0] - bin_center[0])) <= float(gate_half[0])
                and abs(float(release_arr[1] - bin_center[1])) <= float(gate_half[1])
            )
            release_z_clearance = float(release_arr[2] - wall_z)
            release_xy_err = float(math.sqrt(float(release_arr[0] - target[0]) ** 2 + float(release_arr[1] - target[1]) ** 2))
            release_point = vec_list(release_arr, 3)
    except Exception:
        release_inside_gate = False
    release_gate_ok = bool(
        release_inside_gate
        and release_z_clearance is not None
        and float(release_z_clearance) >= float(UNLOAD_ACTUAL_MIN_ABOVE_WALL_Z)
    )

    # Motion arrival is a hard execution condition. Drop/release gates are
    # diagnostics only; the real quality decision is made from particle metrics
    # after the dump settles.
    ok = bool(not blocked)
    if ok:
        if drop_ok and release_gate_ok:
            reason = "ok"
        else:
            reason = "motion_ok_drop_diagnostic_only"
    elif blocked:
        reason = "joint_not_reached"
    else:
        reason = "unload_motion_not_reached"
    return {
        "ok": ok,
        "reason": reason,
        "blocked_joints": blocked,
        "swing_err_deg": swing_err,
        "swing_action_err_deg": swing_action_err,
        "swing_near_boundary": bool(swing_near_boundary),
        "swing_goal_raw_deg": rad_to_deg(swing_goal_raw),
        "swing_goal_wrapped_deg": rad_to_deg(wrap_angle(swing_goal_raw)),
        "swing_action_goal_deg": rad_to_deg(swing_goal_action),
        "swing_real_raw_deg": rad_to_deg(swing_real_raw),
        "swing_real_wrapped_deg": rad_to_deg(wrap_angle(swing_real_raw)),
        "swing_real_near_goal_deg": rad_to_deg(float(q_real[swing_idx])),
        "max_non_swing_err_deg": non_swing_err,
        "load_xy_err": load_xy_err,
        "load_z_clearance": load_z_clearance,
        "load_point": load_point,
        "drop_ok": drop_ok,
        "drop_target": drop.get("target"),
        "drop_landing": drop.get("landing"),
        "drop_release": drop.get("release"),
        "drop_xy_err": drop.get("xy_err"),
        "drop_drift_xy": drop.get("drift_xy"),
        "drop_drift_distance": drop.get("drift_distance"),
        "drop_fall_time": drop.get("fall_time"),
        "drop_bucket_delta_deg": drop.get("bucket_delta_deg"),
        "drop_source_clearance": drop.get("source_clearance"),
        "drop_inside_xy": drop.get("inside_xy"),
        "drop_above_wall": drop.get("above_wall"),
        "drop_close_xy": drop.get("close_xy"),
        "release_inside_gate": bool(release_inside_gate),
        "release_z_clearance": release_z_clearance,
        "release_xy_err": release_xy_err,
        "release_point": release_point,
        "target": vec_list(target, 3),
        "wall_z": wall_z,
        "q_goal": q_deg_values(q_goal_arr, wrap_swing_for_display=True),
        "q_real": q_deg_values(q_real, wrap_swing_for_display=True),
        "source": str(ctx.get("source", "unknown")),
    }


def verify_unload_arrival(label, q_goal):
    report = unload_arrival_report(q_goal)
    info_print(
        f"[UNLOAD ARRIVAL] {label}: "
        f"ok={report.get('ok')} "
        f"reason={report.get('reason')} "
        f"blocked_joints={report.get('blocked_joints')} "
        f"swing_err={fmt_optional(report.get('swing_err_deg'))}deg "
        f"swing_action_err={fmt_optional(report.get('swing_action_err_deg'))}deg "
        f"swing_boundary={report.get('swing_near_boundary')} "
        f"swing_goal_raw={fmt_optional(report.get('swing_goal_raw_deg'))}deg "
        f"swing_goal_wrapped={fmt_optional(report.get('swing_goal_wrapped_deg'))}deg "
        f"swing_action_goal={fmt_optional(report.get('swing_action_goal_deg'))}deg "
        f"swing_real_raw={fmt_optional(report.get('swing_real_raw_deg'))}deg "
        f"swing_real_near={fmt_optional(report.get('swing_real_near_goal_deg'))}deg "
        f"max_non_swing_err={fmt_optional(report.get('max_non_swing_err_deg'))}deg "
        f"load_xy_err={fmt_optional(report.get('load_xy_err'))} "
        f"load_z_clearance={fmt_optional(report.get('load_z_clearance'))} "
        f"drop_ok={report.get('drop_ok')} "
        f"drop_landing={report.get('drop_landing')} "
        f"drop_xy_err={fmt_optional(report.get('drop_xy_err'))} "
        f"drop_close_xy={report.get('drop_close_xy')} "
        f"release_inside={report.get('release_inside_gate')} "
        f"release_z_clearance={fmt_optional(report.get('release_z_clearance'))} "
        f"release_xy_err={fmt_optional(report.get('release_xy_err'))} "
        f"release_point={report.get('release_point')} "
        f"drop_drift={fmt_optional(report.get('drop_drift_distance'))} "
        f"drop_drift_xy={report.get('drop_drift_xy')} "
        f"drop_source_clearance={fmt_optional(report.get('drop_source_clearance'))} "
        f"target={report.get('target')} "
        f"load={report.get('load_point')} "
        f"q_goal={report.get('q_goal')} "
        f"q_real={report.get('q_real')} "
        f"source={report.get('source')}"
    )
    if bool(report.get("ok", False)):
        return True
    update_status(
        f"[UNLOAD BLOCKED] {label}: unload motion arrival failed; "
        f"blocked_joints={report.get('blocked_joints')} "
        f"swing_err={fmt_optional(report.get('swing_err_deg'))}deg "
        f"swing_action_err={fmt_optional(report.get('swing_action_err_deg'))}deg "
        f"drop_xy_err={fmt_optional(report.get('drop_xy_err'))}",
        force=True,
    )
    set_execution_failure_reason(f"execution_failed/unload_arrival:{label}:{report.get('reason')}")
    dataset_record_event("unload_arrival_failed", f"{label}:{report}")
    return False


def notify_sand_site_step_done(stage_name):
    if "unload" not in str(stage_name).lower():
        return
    info_print("[SAND SITE] unload done; real sand physics only, no scripted deposit")


def ensure_sand_site_bucket_colliders(force=False):
    return configure_robot_bucket_sand_collision("sand_site_active")


def print_ground_contact_diagnostics(label=""):
    hardbase = None
    for path in ["/SandSite/HardBaseCollision", "/World/SandSite/HardBaseCollision"]:
        prim = get_prim(path)
        if prim and prim.IsValid():
            hardbase = prim
            break
    hardbase_enabled = None
    hardbase_path = "missing"
    hardbase_xy = "missing"
    base_inside_hardbase_xy = None
    if hardbase is not None:
        hardbase_path = str(hardbase.GetPath())
        attr = hardbase.GetAttribute("physics:collisionEnabled")
        hardbase_enabled = attr.Get() if attr.IsValid() else None
        hbox = get_bbox(hardbase_path)
        if hbox is not None:
            hmin = hbox.GetMin()
            hmax = hbox.GetMax()
            hardbase_xy = f"x=({float(hmin[0]):.2f},{float(hmax[0]):.2f}) y=({float(hmin[1]):.2f},{float(hmax[1]):.2f})"
    base_min = None
    base_center = None
    full_min = None
    full_source = None
    boom_min = None
    try:
        base_min = bbox_min_z(LINK_PATHS["base_link"])
        base_center = bbox_center(LINK_PATHS["base_link"])
    except Exception:
        pass
    try:
        full_min, _, full_source = robot_full_collision_bbox()
    except Exception:
        full_min, full_source = None, "failed"
    try:
        boom_min = bbox_min_z(LINK_PATHS["boom_link"])
    except Exception:
        boom_min = None
    if hardbase is not None and base_center is not None:
        hbox = get_bbox(hardbase_path)
        if hbox is not None:
            hmin = hbox.GetMin()
            hmax = hbox.GetMax()
            base_inside_hardbase_xy = (
                float(hmin[0]) <= float(base_center[0]) <= float(hmax[0])
                and float(hmin[1]) <= float(base_center[1]) <= float(hmax[1])
            )
    support_ground = None
    support_ground_top = None
    if base_center is not None:
        support_ground_top, support_ground = support_ground_top_at_xy(float(base_center[0]), float(base_center[1]))
    info_print(
        f"[GROUND CONTACT DIAG] {label}",
        f"hardbase={hardbase_path}",
        f"hardbase_collision={hardbase_enabled}",
        f"hardbase_xy={hardbase_xy}",
        f"base_inside_hardbase_xy={base_inside_hardbase_xy}",
        f"support_ground={support_ground}",
        f"support_ground_top={fmt_optional(support_ground_top)}",
        f"base_min_z={fmt_optional(base_min)}",
        f"boom_min_z={fmt_optional(boom_min)}",
        f"full_min_z={fmt_optional(None if full_min is None else full_min[2])}",
        f"full_source={full_source}",
        f"ground_z={GROUND_TOP_Z:.3f}",
    )


def print_motion_constraint_diagnostics(label=""):
    try:
        q_real = get_real_joint_positions()
    except Exception:
        q_real = None
    report = phase_ground_report("diagnostic")
    ok, reason = phase_ground_ok("diagnostic", report)
    api = get_sand_site_api()
    sand_status = None
    if api is not None and callable(api.get("get_status")):
        try:
            sand_status = api.get("get_status")()
        except Exception as e:
            sand_status = {"error": str(e)}

    info_print("========== MOTION CONSTRAINT DIAG ==========")
    info_print("[MOTION DIAG]", f"label={label}", f"robot_ready={ROBOT is not None}", f"joint_indices={JOINT_INDICES}")
    info_print("[MOTION DIAG]", "limits_deg=", {k: tuple(round(float(v), 3) for v in vals) for k, vals in FINAL_LIMITS_DEG.items()})
    info_print("[MOTION DIAG]", "q_cmd_deg=", [round(rad_to_deg(float(x)), 3) for x in CTRL.q_cmd])
    if q_real is not None:
        info_print("[MOTION DIAG]", "q_real_deg=", [round(rad_to_deg(float(x)), 3) for x in q_real])
    info_print("[MOTION DIAG]", "ground_guard_ok=", ok, "reason=", reason)
    info_print("[MOTION DIAG] " + format_ground_report("diagnostic", report, "current"))
    info_print("[MOTION DIAG]", collision_lowest_detail("boom_link", limit=3))
    if sand_status is not None:
        info_print(
            "[MOTION DIAG]",
            "sand_enabled=", sand_status.get("real_sand_enabled"),
            "sand_particles=", sand_status.get("real_sand_particle_count"),
        )


def q_deg_values(q, wrap_swing_for_display=False):
    if q is None:
        return None
    vals = []
    for i, x in enumerate(np.array(q, dtype=np.float32)):
        v = float(x)
        if wrap_swing_for_display and i == CTRL.name_to_idx.get("swing", 0):
            v = wrap_angle(v)
        vals.append(round(rad_to_deg(v), 2))
    return vals


def vec_list(v, n=None):
    if v is None:
        return None
    arr = np.array(v, dtype=np.float32).reshape(-1)
    if n is not None:
        arr = arr[:int(n)]
    return [float(x) for x in arr]


def compact_sand_metrics(metrics):
    if not isinstance(metrics, dict) or not bool(metrics.get("available", False)):
        return {"available": False}
    return {
        "available": True,
        "n": int(metrics.get("particle_count", 0)),
        "pile": int(metrics.get("pile_count", 0)),
        "bucket": int(metrics.get("bucket_count", 0)),
        "bucket_from_pile": int(metrics.get("bucket_from_pile_count", 0)),
        "bin": int(metrics.get("bin_count", 0)),
        "bin_from_pile": int(metrics.get("bin_from_pile_count", 0)),
        "spill_from_pile": int(metrics.get("spill_from_pile_count", 0)),
        "bucket_from_pile_mass": float(metrics.get("bucket_from_pile_mass", 0.0)),
        "bin_from_pile_mass": float(metrics.get("bin_from_pile_mass", 0.0)),
        "spill_from_pile_mass": float(metrics.get("spill_from_pile_mass", 0.0)),
    }


def compact_scene_context(ctx=None):
    ctx = task_scene_context() if ctx is None else ctx
    if not isinstance(ctx, dict):
        ctx = {}
    return {
        "source": str(ctx.get("source", "unknown")),
        "pile_center": vec_list(ctx.get("pile_center"), 3),
        "pile_radius": vec_list(ctx.get("pile_radius"), 2),
        "unload_bin_center": vec_list(ctx.get("unload_bin_center"), 3),
        "unload_bin_half_size": vec_list(ctx.get("unload_bin_half_size"), 2),
        "unload_bin_z_range": vec_list(ctx.get("unload_bin_z_range"), 2),
        "unload_point_raw": vec_list(ctx.get("unload_point_raw"), 3),
        "unload_point": vec_list(ctx.get("unload_point"), 3),
    }


def compact_unload_drop(report=None):
    report = unload_drop_report() if report is None else report
    if not isinstance(report, dict):
        return {"ok": False}
    return {
        "ok": bool(report.get("ok", False)),
        "inside_xy": bool(report.get("inside_xy", False)),
        "above_wall": bool(report.get("above_wall", False)),
        "close_xy": bool(report.get("close_xy", False)),
        "xy_err": None if report.get("xy_err") is None else float(report.get("xy_err")),
        "source_clearance": None if report.get("source_clearance") is None else float(report.get("source_clearance")),
        "target": vec_list(report.get("target"), 3),
        "landing": vec_list(report.get("landing"), 3),
        "release": vec_list(report.get("release"), 3),
        "drift_xy": vec_list(report.get("drift_xy"), 2),
        "drift_distance": None if report.get("drift_distance") is None else float(report.get("drift_distance")),
        "fall_time": None if report.get("fall_time") is None else float(report.get("fall_time")),
        "bucket_delta_deg": None if report.get("bucket_delta_deg") is None else float(report.get("bucket_delta_deg")),
    }


def compact_plan_stage(row):
    if not isinstance(row, dict):
        return {}
    out = {
        "phase": str(row.get("phase", "")),
        "planned": bool(row.get("planned", False)),
        "required": bool(row.get("required", False)),
        "target": vec_list(row.get("target_point"), 3),
    }
    if row.get("q_goal_deg") is not None:
        out["q_goal_deg"] = row.get("q_goal_deg")
    if row.get("q_dump_deg") is not None:
        out["q_dump_deg"] = row.get("q_dump_deg")
    if row.get("duration") is not None:
        out["duration"] = float(row.get("duration"))
    if row.get("drop") is not None:
        out["drop"] = row.get("drop")
    if row.get("reason"):
        out["reason"] = str(row.get("reason"))
    return out


def compact_plan_candidate(row, include_stages=False):
    if not isinstance(row, dict):
        return {}
    out = {
        "id": str(row.get("id", "")),
        "planned": bool(row.get("planned", False)),
        "selected": bool(row.get("selected", False)),
        "score": float(row.get("score", -1.0e9)),
        "score_reason": str(row.get("score_reason", "")),
        "steps": int(row.get("steps", 0)),
        "failed_stage": str(row.get("failed_stage", "")),
        "failure_reason": str(row.get("failure_reason", "")),
        "planned_prefix": int(row.get("planned_prefix", 0) or 0),
        "rank_cost": None if row.get("rank_cost") is None else float(row.get("rank_cost")),
        "planner_cost": None if row.get("planner_cost") is None else float(row.get("planner_cost")),
        "weighted_angle": None if row.get("weighted_angle") is None else float(row.get("weighted_angle")),
        "estimated_time": None if row.get("estimated_time") is None else float(row.get("estimated_time")),
        "unload_point_xyz": vec_list(row.get("unload_point_xyz"), 3),
        "unload_landing_xyz": vec_list(row.get("unload_landing_xyz"), 3),
    }
    if include_stages:
        out["stages"] = [compact_plan_stage(x) for x in row.get("stages", [])]
    return out


DATASET_STATE_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "swing",
    "boom",
    "arm",
    "bucket",
    "bucket_load_estimate",
    "bucket_tip_x",
    "bucket_tip_y",
    "bucket_tip_z",
    "bucket_load_x",
    "bucket_load_y",
    "bucket_load_z",
]

DATASET_ACTION_NAMES = [
    "swing_cmd_velocity",
    "boom_cmd_velocity",
    "arm_cmd_velocity",
    "bucket_cmd_velocity",
]


def json_sanitize(value):
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def ensure_parent_dir(path):
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)


def write_json_file(path, data):
    ensure_parent_dir(path)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(json_sanitize(data), f, ensure_ascii=True, indent=2)


def append_jsonl(path, data):
    ensure_parent_dir(path)
    with open(str(path), "a", encoding="utf-8") as f:
        f.write(json.dumps(json_sanitize(data), ensure_ascii=True) + "\n")


def jsonl_line_count(path):
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def debug_short_string(value, max_len=220):
    s = str(value).replace("\n", " ")
    if len(s) <= int(max_len):
        return s
    return s[: max(0, int(max_len) - 3)] + "..."


def debug_round_vec(value, n=None, digits=3):
    vals = vec_list(value, n)
    if vals is None:
        return None
    return [round(float(x), int(digits)) for x in vals]


def debug_compact_sand_counts(metrics=None):
    try:
        if metrics is None:
            metrics = sand_metrics_current(force=False)
        sand = compact_sand_metrics(metrics)
        if not sand.get("available", False):
            return {"ok": False}
        return {
            "n": int(sand.get("n", 0)),
            "pile": int(sand.get("pile", 0)),
            "b": int(sand.get("bucket_from_pile", 0)),
            "bin": int(sand.get("bin_from_pile", 0)),
            "spill": int(sand.get("spill_from_pile", 0)),
        }
    except Exception:
        return {"ok": False}


def debug_compact_data(data, depth=0, max_items=8):
    if data is None:
        return None
    if depth >= 2:
        if isinstance(data, (str, int, float, bool)) or data is None:
            return debug_short_string(data, 120) if isinstance(data, str) else data
        return debug_short_string(type(data).__name__, 80)
    if isinstance(data, str):
        return debug_short_string(data)
    if isinstance(data, np.generic):
        return debug_compact_data(data.item(), depth=depth, max_items=max_items)
    if isinstance(data, (bool, int)):
        return data
    if isinstance(data, float):
        return round(float(data), 5) if math.isfinite(float(data)) else None
    if isinstance(data, np.ndarray):
        return debug_round_vec(data.reshape(-1), n=max_items)
    if isinstance(data, (list, tuple)):
        out = [debug_compact_data(x, depth + 1, max_items=max_items) for x in list(data)[:max_items]]
        if len(data) > max_items:
            out.append(f"...+{len(data) - max_items}")
        return out
    if isinstance(data, dict):
        preferred = [
            "attempt",
            "status",
            "result",
            "reason",
            "phase",
            "stage",
            "score",
            "target_xyz",
            "unload_point_xyz",
            "unload_landing_xyz",
            "unload_release_xyz",
            "initial_pose_id",
            "path",
            "plan_debug",
            "score_path",
            "trajectory",
            "events",
        ]
        keys = [k for k in preferred if k in data]
        keys += [k for k in data.keys() if k not in keys][: max(0, max_items - len(keys))]
        return {str(k): debug_compact_data(data.get(k), depth + 1, max_items=max_items) for k in keys[:max_items]}
    return debug_short_string(data, 120)


def debug_timeline_path(create=False):
    path = str(STATE.get("debug_timeline_path", "") or "")
    if path:
        return path
    run_dir = str(STATE.get("auto_collect_run_dir", "") or "")
    if not run_dir:
        return ""
    if create:
        os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, DATASET_DEBUG_TIMELINE_FILE)
    STATE["debug_timeline_path"] = path
    return path


def debug_timeline_record(tag, stage="", result="", reason="", data=None, q_cmd=None, q_real=None, include_sand=False):
    path = debug_timeline_path(create=False)
    if not path:
        return
    now = time.time()
    try:
        row = {
            "t": round(float(now), 3),
            "ep": str(STATE.get("dataset_episode_uid", "")),
            "idx": int(STATE.get("dataset_episode_id", 0)),
            "tag": str(tag).upper(),
        }
        if stage:
            row["stage"] = str(stage)
        if result:
            row["result"] = str(result)
        if reason:
            row["reason"] = debug_short_string(reason)
        if q_cmd is not None:
            row["q"] = q_deg_values(q_cmd, wrap_swing_for_display=True)
        if q_real is not None:
            row["qr"] = q_deg_values(q_real, wrap_swing_for_display=True)
        target = get_target_pos() if TARGET_PATH else None
        if target is not None:
            row["target"] = debug_round_vec(target, 3)
        landing = STATE.get("active_unload_landing_point")
        release = STATE.get("active_unload_release_point")
        if landing is not None:
            row["landing"] = debug_round_vec(landing, 3)
        if release is not None:
            row["release"] = debug_round_vec(release, 3)
        if include_sand:
            row["sand"] = debug_compact_sand_counts()
        compact = debug_compact_data(data)
        if compact not in (None, {}, []):
            row["data"] = compact
        append_jsonl(path, row)
    except Exception as e:
        if now - float(STATE.get("debug_timeline_last_error_time", 0.0)) > 2.0:
            STATE["debug_timeline_last_error_time"] = now
            info_print("[WARN] debug timeline write failed:", type(e).__name__, e)


def stable_json_hash(data):
    try:
        payload = json.dumps(json_sanitize(data), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = repr(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def planner_config_snapshot():
    return {
        "planner_version": PLANNER_VERSION,
        "dig_plan_max_build_seconds": DIG_PLAN_MAX_BUILD_SECONDS,
        "dig_plan_max_candidates": DIG_PLAN_MAX_CANDIDATES,
        "dig_plan_beam_size": DIG_PLAN_BEAM_SIZE,
        "dig_plan_topk_ik": DIG_PLAN_TOPK_IK,
        "dig_plan_path_check_samples": DIG_PLAN_PATH_CHECK_SAMPLES,
        "motion_weights": vec_list(DIG_PLAN_MOTION_WEIGHTS, 4),
        "angle_cost_weight": DIG_PLAN_ANGLE_COST_WEIGHT,
        "time_cost_weight": DIG_PLAN_TIME_COST_WEIGHT,
        "candidate_family": DIG_PLAN_CANDIDATES,
        "auto_dig_grid_size": AUTO_DIG_GRID_SIZE,
        "auto_dig_topk_targets": AUTO_DIG_TOPK_TARGETS,
        "auto_dig_core_norm_max": AUTO_DIG_CORE_NORM_MAX,
        "auto_dig_density_radius": AUTO_DIG_DENSITY_RADIUS,
        "auto_dig_min_local_particles": AUTO_DIG_MIN_LOCAL_PARTICLES,
        "auto_unload_grid_size": AUTO_UNLOAD_GRID_SIZE,
    }


def quality_gate_config_snapshot():
    return {
        "quality_gate_version": QUALITY_GATE_VERSION,
        "quality_target_bucket_particles": QUALITY_TARGET_BUCKET_PARTICLES,
        "quality_min_bucket_particles": QUALITY_MIN_BUCKET_PARTICLES,
        "quality_min_dump_particles": QUALITY_MIN_DUMP_PARTICLES,
        "quality_min_score": QUALITY_MIN_SCORE,
        "quality_max_spill_ratio": QUALITY_MAX_SPILL_RATIO,
    }


def auto_dataset_config_snapshot():
    return {
        "schema": AUTO_COLLECT_SCHEMA,
        "trajectory_format": DATASET_TRAJECTORY_FORMAT,
        "dataset_root": AUTO_COLLECT_DATASET_ROOT,
        "default_count": AUTO_COLLECT_DEFAULT_COUNT,
        "max_attempt_multiplier": AUTO_COLLECT_MAX_ATTEMPT_MULTIPLIER,
        "max_plan_retries": AUTO_COLLECT_MAX_PLAN_RETRIES,
        "target_center": vec_list(AUTO_COLLECT_TARGET_CENTER, 3),
        "target_radius_x": AUTO_COLLECT_TARGET_RADIUS_X,
        "target_radius_y": AUTO_COLLECT_TARGET_RADIUS_Y,
        "target_depths": AUTO_COLLECT_TARGET_DEPTHS,
        "initial_pose_family": AUTO_COLLECT_INITIAL_POSES_DEG,
        "sand_reset_policy": AUTO_COLLECT_SAND_RESET_POLICY,
        "preflight_min_particles": AUTO_PREFLIGHT_MIN_PARTICLES,
    }


def sand_config_snapshot():
    ctx = compact_scene_context()
    status = dataset_sand_status()
    return {
        "scene_context": ctx,
        "status": status,
        "particle_mass": sand_particle_mass(),
        "sand_particle_path_suffix": SAND_PARTICLE_PATH_SUFFIX,
        "sand_bucket_local_min": vec_list(SAND_BUCKET_LOCAL_MIN, 3),
        "sand_bucket_local_max": vec_list(SAND_BUCKET_LOCAL_MAX, 3),
        "sand_metrics_interval": SAND_METRICS_INTERVAL,
    }


def full_config_snapshot():
    return {
        "sand": sand_config_snapshot(),
        "auto_dataset": auto_dataset_config_snapshot(),
        "planner": planner_config_snapshot(),
        "quality_gate": quality_gate_config_snapshot(),
    }


def current_config_hash():
    return stable_json_hash(full_config_snapshot())


def get_base_yaw_rad():
    # Fixed-base dataset v1: record yaw as 0 unless a mobile base pose reader is added.
    return 0.0


def dataset_sand_status():
    api = get_sand_site_api()
    if api is not None and callable(api.get("get_status")):
        try:
            status = api.get("get_status")()
            if isinstance(status, dict):
                keep = [
                    "real_sand_enabled",
                    "real_sand_particle_count",
                    "real_sand_error",
                    "excavated_volume",
                    "status",
                ]
                return {k: status.get(k) for k in keep if k in status}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    return {"sand_site_active": bool(sand_site_active())}


def dataset_bucket_load_estimate():
    api = get_sand_site_api()
    if api is not None and callable(api.get("get_status")):
        try:
            status = api.get("get_status")()
            for key in ["bucket_load", "scripted_bucket_load", "excavated_volume"]:
                if key in status:
                    return safe_float(status.get(key), 0.0)
        except Exception:
            pass
    return 0.0


def sand_particle_mass():
    api = get_sand_site_api()
    if api is not None and callable(api.get("get_status")):
        try:
            status = api.get("get_status")()
            value = status.get("particle_mass") if isinstance(status, dict) else None
            if value is not None:
                return safe_float(value, SAND_PARTICLE_MASS_DEFAULT)
        except Exception:
            pass
    return float(SAND_PARTICLE_MASS_DEFAULT)


def sand_particle_prim():
    api = get_sand_site_api()
    candidates = []
    if api is not None:
        root = api.get("root_path")
        if root:
            candidates.append(f"{root}/{SAND_PARTICLE_PATH_SUFFIX}")
    candidates.extend([
        f"/SandSite/{SAND_PARTICLE_PATH_SUFFIX}",
        f"/World/SandSite/{SAND_PARTICLE_PATH_SUFFIX}",
    ])
    for path in candidates:
        prim = get_prim(path)
        if prim and prim.IsValid():
            return prim
    return None


def sand_particle_positions():
    prim = sand_particle_prim()
    if prim is None:
        return None
    try:
        attr = prim.GetAttribute("points")
        points = attr.Get() if attr.IsValid() else None
        if points is None or len(points) == 0:
            return None
        return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)
    except Exception as e:
        info_print("[WARN] sand particle read failed:", type(e).__name__, e)
        return None


def sand_particle_snapshot():
    points = sand_particle_positions()
    if points is None or len(points) == 0:
        return None
    return np.array(points, dtype=np.float32).copy()


def sand_reset_displacement_stats(a, b):
    if a is None or b is None:
        return None
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if len(a) == 0 or len(a) != len(b):
        return None
    disp = np.linalg.norm(b - a, axis=1)
    if len(disp) == 0:
        return None
    return {
        "n": int(len(disp)),
        "mean": float(np.mean(disp)),
        "p95": float(np.percentile(disp, 95.0)),
        "max": float(np.max(disp)),
        "z_min": float(np.min(b[:, 2])),
        "z_max": float(np.max(b[:, 2])),
    }


async def wait_for_sand_particles_stable(label="sand_reset"):
    label = str(label)
    min_frames = int(SAND_RESET_SETTLE_MIN_FRAMES)
    max_frames = max(min_frames, int(SAND_RESET_SETTLE_MAX_FRAMES))
    window = max(1, int(SAND_RESET_STABLE_WINDOW_FRAMES))
    prev = sand_particle_snapshot()
    elapsed = 0
    last_stats = None

    while elapsed < max_frames:
        if not timeline_allows_background_work():
            handle_timeline_stop_if_needed(label)
            return False, {"aborted": True, "reason": "timeline_stopped_or_runtime_stopped"}
        await step_updates(window)
        elapsed += window
        cur = sand_particle_snapshot()
        stats = sand_reset_displacement_stats(prev, cur)
        prev = cur
        if stats is None:
            info_print("[SAND RESET SETTLE]", f"label={label}", f"frames={elapsed}", "particles=missing_or_rebuilt")
            continue

        last_stats = stats
        stable = (
            elapsed >= min_frames
            and stats["mean"] <= float(SAND_RESET_STABLE_MEAN_DISPLACEMENT)
            and stats["p95"] <= float(SAND_RESET_STABLE_P95_DISPLACEMENT)
        )
        info_print(
            "[SAND RESET SETTLE]",
            f"label={label}",
            f"frames={elapsed}",
            f"n={stats['n']}",
            f"mean={stats['mean']:.4f}",
            f"p95={stats['p95']:.4f}",
            f"max={stats['max']:.4f}",
            f"z=({stats['z_min']:.3f},{stats['z_max']:.3f})",
            f"stable={stable}",
        )
        if elapsed <= int(SAND_RESET_ESCAPE_CHECK_FRAMES) and stats["z_min"] < float(SAND_RESET_ESCAPE_Z):
            stats["escaped"] = True
            info_print(
                "[WARN] [SAND RESET ESCAPE]",
                f"label={label}",
                f"frames={elapsed}",
                f"z_min={stats['z_min']:.3f}",
                "action=retry_native_reset",
            )
            return False, stats
        if stable:
            return True, stats

    return False, last_stats


async def reset_sand_site_stably(label=""):
    label = str(label or "sand_reset")
    if not timeline_allows_background_work():
        handle_timeline_stop_if_needed(label)
        info_print("[SAND RESET] skipped: timeline stopped or runtime stopped", f"label={label}")
        return False
    if not sand_site_active():
        info_print("[SAND RESET] skipped: sand site inactive", f"label={label}")
        return False

    if bool(STATE.get("sand_site_reset_active", False)):
        info_print("[SAND RESET] already active; waiting", f"label={label}", f"active_label={STATE.get('sand_site_last_reset_label')}")
        waited = 0
        while bool(STATE.get("sand_site_reset_active", False)) and waited < int(SAND_RESET_SETTLE_MAX_FRAMES):
            if not timeline_allows_background_work():
                handle_timeline_stop_if_needed(label)
                return False
            await step_updates(15)
            waited += 15
        return bool(STATE.get("sand_site_stable_reset_done", False))

    api = get_sand_site_api()
    reset_fn = None if api is None else api.get("reset")
    stable_reset_fn = None if api is None else api.get("reset_stably")
    if not callable(reset_fn):
        info_print("[SAND RESET] skipped: sand site reset API missing", f"label={label}")
        return False

    STATE["sand_site_reset_active"] = True
    STATE["sand_site_last_reset_label"] = label
    try:
        max_attempts = max(1, int(SAND_RESET_MAX_NATIVE_ATTEMPTS))
        ok = False
        stats = None
        for attempt in range(1, max_attempts + 1):
            if not timeline_allows_background_work():
                handle_timeline_stop_if_needed(label)
                stats = {"aborted": True, "reason": "timeline_stopped_or_runtime_stopped"}
                break
            update_status(f"[SAND RESET] native reset start: {label} attempt={attempt}", force=True)
            if callable(stable_reset_fn):
                info_print("[SAND RESET NATIVE]", f"label={label}", f"attempt={attempt}/{max_attempts}", "calling=sand_site.reset_stably")
                result = stable_reset_fn(f"{label}_native_attempt{attempt}")
                if hasattr(result, "__await__"):
                    native_ok = await result
                else:
                    native_ok = bool(result)
                if native_ok:
                    ok = True
                    stats = {"native_stable_reset": True}
                    break
                if not native_ok and attempt < max_attempts:
                    info_print("[SAND RESET RETRY]", f"label={label}", f"attempt={attempt}", "reason=native_stable_reset_failed")
                    await step_updates(30)
                    continue
            else:
                info_print("[SAND RESET NATIVE]", f"label={label}", f"attempt={attempt}/{max_attempts}", "calling=sand_site.reset")
                reset_fn()

            ok, stats = await wait_for_sand_particles_stable(f"{label}_attempt{attempt}")
            if ok:
                break
            escaped = isinstance(stats, dict) and bool(stats.get("escaped", False))
            if escaped and attempt < max_attempts:
                info_print("[SAND RESET RETRY]", f"label={label}", f"attempt={attempt}", "reason=escaped_particles")
                await step_updates(30)
                continue
            break

        STATE["sand_site_stable_reset_done"] = bool(ok)
        if ok:
            update_status(f"[SAND RESET] stable: {label}", force=True)
        else:
            update_status(f"[SAND RESET] not fully stable before timeout: {label}", force=True)
        info_print("[SAND RESET DONE]", f"label={label}", f"stable={ok}", f"stats={stats}")
        return bool(ok)
    except Exception as e:
        STATE["sand_site_stable_reset_done"] = False
        info_print("[WARN] [SAND RESET] stable reset failed:", f"label={label}", type(e).__name__, e)
        update_status(f"[SAND RESET] failed: {type(e).__name__}", force=True)
        return False
    finally:
        STATE["sand_site_reset_active"] = False


def request_sand_site_stable_reset(label="ui"):
    try:
        register_async_task("sand_reset", reset_sand_site_stably(label), replace=True)
        return True
    except Exception as e:
        info_print("[WARN] [SAND RESET] request failed:", type(e).__name__, e)
        return False


builtins._EXCAVATOR_STABLE_SAND_RESET = request_sand_site_stable_reset


async def delayed_startup_sand_reset():
    await step_updates(max(1, int(AUTO_RESET_SAND_UI_READY_DELAY_FRAMES)))
    if not STATE.get("running", False):
        return
    if not simulation_timeline_is_playing():
        handle_timeline_stop_if_needed("after_ui_ready")
        info_print("[SAND RESET] skipped after_ui_ready", "reason=timeline_stopped")
        return
    api = get_sand_site_api()
    if isinstance(api, dict) and bool(api.get("last_reset_healthy", False)):
        reset_age = time.time() - float(api.get("last_reset_time", 0.0) or 0.0)
        if reset_age < 300.0:
            STATE["sand_site_stable_reset_done"] = True
            info_print(
                "[SAND RESET] skipped after_ui_ready",
                f"reason=sand_site_already_healthy",
                f"age={reset_age:.1f}s",
                f"label={api.get('last_reset_label', '')}",
            )
            return
    await reset_sand_site_stably("after_ui_ready")


def project_points_to_link_local(points, link_path):
    if points is None or len(points) == 0 or not link_path:
        return None
    origin = transform_local_point_to_world(link_path, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    xw = transform_local_point_to_world(link_path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    yw = transform_local_point_to_world(link_path, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    zw = transform_local_point_to_world(link_path, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if origin is None or xw is None or yw is None or zw is None:
        return None
    axes = []
    for axis_point in [xw, yw, zw]:
        axis = np.array(axis_point - origin, dtype=np.float32)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None
        axes.append(axis / n)
    d = np.array(points, dtype=np.float32) - np.array(origin, dtype=np.float32).reshape(1, 3)
    return np.stack([d @ axes[0], d @ axes[1], d @ axes[2]], axis=1)


def mask_points_in_box(points, center, half_xy, z_min, z_max):
    if points is None:
        return np.zeros(0, dtype=bool)
    center = np.array(center, dtype=np.float32)
    return (
        (points[:, 0] >= center[0] - half_xy[0])
        & (points[:, 0] <= center[0] + half_xy[0])
        & (points[:, 1] >= center[1] - half_xy[1])
        & (points[:, 1] <= center[1] + half_xy[1])
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )


def sand_region_masks(points):
    n = 0 if points is None else int(len(points))
    empty = np.zeros(n, dtype=bool)
    if points is None or n == 0:
        return empty, empty, empty

    pile_center = np.array(SAND_PILE_CENTER, dtype=np.float32)
    dx = (points[:, 0] - pile_center[0]) / max(1e-5, float(SAND_PILE_RADIUS_X))
    dy = (points[:, 1] - pile_center[1]) / max(1e-5, float(SAND_PILE_RADIUS_Y))
    pile_mask = (
        (dx * dx + dy * dy <= 1.0)
        & (points[:, 2] >= SAND_PILE_Z_MIN)
        & (points[:, 2] <= SAND_PILE_Z_MAX)
    )

    bucket_local = project_points_to_link_local(points, BUCKET_LINK)
    if bucket_local is None:
        bucket_mask = empty.copy()
    else:
        bucket_mask = np.all(bucket_local >= SAND_BUCKET_LOCAL_MIN.reshape(1, 3), axis=1) & np.all(
            bucket_local <= SAND_BUCKET_LOCAL_MAX.reshape(1, 3), axis=1
        )

    ctx = task_scene_context()
    bin_center_3 = np.array(ctx["unload_bin_center"], dtype=np.float32)
    bin_center = np.array([float(bin_center_3[0]), float(bin_center_3[1]), 0.0], dtype=np.float32)
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32)
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32)
    bin_mask = mask_points_in_box(
        points,
        bin_center,
        bin_half,
        float(bin_z_range[0]),
        float(bin_z_range[1]),
    )
    return pile_mask, bucket_mask, bin_mask


def particle_ids_from_mask(mask):
    if mask is None:
        return set()
    return set(int(x) for x in np.nonzero(mask)[0].tolist())


def capture_initial_pile_particle_ids():
    points = sand_particle_positions()
    pile_mask, _, _ = sand_region_masks(points)
    ids = particle_ids_from_mask(pile_mask)
    STATE["dataset_initial_pile_particle_ids"] = ids
    STATE["dataset_initial_pile_particle_count"] = len(ids)
    return ids


def sand_metrics_current(force=False):
    now = time.time()
    if (
        not force
        and STATE.get("sand_metrics_last") is not None
        and now - float(STATE.get("sand_metrics_last_time", 0.0)) < SAND_METRICS_INTERVAL
    ):
        return dict(STATE["sand_metrics_last"])

    points = sand_particle_positions()
    if points is None:
        metrics = {
            "available": False,
            "particle_count": 0,
            "reason": "missing_particle_points",
        }
        STATE["sand_metrics_last"] = metrics
        STATE["sand_metrics_last_time"] = now
        return metrics

    pile_mask, bucket_mask, bin_mask = sand_region_masks(points)
    pile_ids = particle_ids_from_mask(pile_mask)
    bucket_ids = particle_ids_from_mask(bucket_mask)
    bin_ids = particle_ids_from_mask(bin_mask)

    initial_ids = STATE.get("dataset_initial_pile_particle_ids")
    if initial_ids is None:
        initial_ids = pile_ids
    initial_ids = set(initial_ids)
    from_pile_bucket = initial_ids & bucket_ids
    from_pile_bin = initial_ids & bin_ids
    from_pile_pile = initial_ids & pile_ids
    from_pile_spill = initial_ids - from_pile_bucket - from_pile_bin - from_pile_pile
    mass = sand_particle_mass()
    ctx = task_scene_context()
    metrics = {
        "available": True,
        "particle_count": int(len(points)),
        "particle_mass": float(mass),
        "scene_source": str(ctx.get("source", "unknown")),
        "unload_bin_center": vec_list(ctx.get("unload_bin_center"), 3),
        "unload_bin_half_size": vec_list(ctx.get("unload_bin_half_size"), 2),
        "unload_bin_z_range": vec_list(ctx.get("unload_bin_z_range"), 2),
        "unload_point": vec_list(ctx.get("unload_point"), 3),
        "initial_pile_count": int(len(initial_ids)),
        "pile_count": int(len(pile_ids)),
        "bucket_count": int(len(bucket_ids)),
        "bucket_from_pile_count": int(len(from_pile_bucket)),
        "bin_count": int(len(bin_ids)),
        "bin_from_pile_count": int(len(from_pile_bin)),
        "pile_from_initial_count": int(len(from_pile_pile)),
        "spill_from_pile_count": int(len(from_pile_spill)),
        "bucket_from_pile_mass": float(len(from_pile_bucket) * mass),
        "bin_from_pile_mass": float(len(from_pile_bin) * mass),
        "spill_from_pile_mass": float(len(from_pile_spill) * mass),
    }
    STATE["sand_metrics_last"] = metrics
    STATE["sand_metrics_last_time"] = now
    return dict(metrics)


def is_sand_contact_phase(mode):
    m = str(mode).lower()
    return any(phase in m for phase in SAND_CONTACT_PHASES)


def sand_contact_snapshot(force=False):
    metrics = sand_metrics_current(force=force)
    try:
        tip = bucket_tip_pos()
    except Exception:
        tip = None
    try:
        load = bucket_load_pos()
    except Exception:
        load = None
    return {
        "available": bool(metrics.get("available", False)) if isinstance(metrics, dict) else False,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "bucket": int(metrics.get("bucket_from_pile_count", 0)) if isinstance(metrics, dict) else 0,
        "pile": int(metrics.get("pile_count", 0)) if isinstance(metrics, dict) else 0,
        "spill": int(metrics.get("spill_from_pile_count", 0)) if isinstance(metrics, dict) else 0,
        "tip": None if tip is None else np.array(tip, dtype=np.float32).copy(),
        "load": None if load is None else np.array(load, dtype=np.float32).copy(),
    }


def point_distance(a, b):
    if a is None or b is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))
    except Exception:
        return 0.0


def start_sand_contact_stage(stage_name):
    if not is_sand_contact_phase(stage_name):
        return
    snap = sand_contact_snapshot(force=True)
    now = time.time()
    STATE["sand_contact_stage"] = str(stage_name)
    STATE["sand_contact_stage_started"] = now
    STATE["sand_contact_last_progress_time"] = now
    STATE["sand_contact_last_log_time"] = 0.0
    STATE["sand_contact_start_bucket_from_pile"] = int(snap["bucket"])
    STATE["sand_contact_start_pile"] = int(snap["pile"])
    STATE["sand_contact_start_spill_from_pile"] = int(snap["spill"])
    STATE["sand_contact_start_tip"] = snap["tip"]
    STATE["sand_contact_start_load"] = snap["load"]
    STATE["sand_contact_progress_bucket_base"] = int(snap["bucket"])
    STATE["sand_contact_progress_pile_base"] = int(snap["pile"])
    STATE["sand_contact_progress_spill_base"] = int(snap["spill"])
    STATE["sand_contact_progress_tip_base"] = snap["tip"]
    STATE["sand_contact_progress_load_base"] = snap["load"]
    STATE["sand_contact_last_progress_seen"] = False
    STATE["sand_contact_last_report"] = None
    info_print(
        "[SAND CONTACT START]",
        f"stage={stage_name}",
        f"bucket={int(snap['bucket'])}",
        f"pile={int(snap['pile'])}",
        f"spill={int(snap['spill'])}",
        f"available={bool(snap['available'])}",
    )


def update_sand_contact_progress(stage_name, q_cmd=None, q_real=None, force=False, log=True):
    if not is_sand_contact_phase(stage_name):
        return {"active": False, "allow_continue": False, "progress_now": False}

    current_stage = str(STATE.get("sand_contact_stage", "") or "")
    if current_stage != str(stage_name) or float(STATE.get("sand_contact_stage_started", 0.0)) <= 0.0:
        start_sand_contact_stage(stage_name)

    now = time.time()
    snap = sand_contact_snapshot(force=force)
    bucket = int(snap["bucket"])
    pile = int(snap["pile"])
    spill = int(snap["spill"])

    base_bucket = int(STATE.get("sand_contact_progress_bucket_base", bucket))
    base_pile = int(STATE.get("sand_contact_progress_pile_base", pile))
    base_spill = int(STATE.get("sand_contact_progress_spill_base", spill))
    start_bucket = int(STATE.get("sand_contact_start_bucket_from_pile", bucket))
    start_pile = int(STATE.get("sand_contact_start_pile", pile))
    start_spill = int(STATE.get("sand_contact_start_spill_from_pile", spill))

    bucket_delta = max(0, bucket - base_bucket)
    pile_delta = max(0, base_pile - pile)
    spill_delta = max(0, spill - base_spill)
    tip_delta = point_distance(snap["tip"], STATE.get("sand_contact_progress_tip_base"))
    load_delta = point_distance(snap["load"], STATE.get("sand_contact_progress_load_base"))
    effector_delta = max(tip_delta, load_delta)

    total_bucket_delta = max(0, bucket - start_bucket)
    total_pile_delta = max(0, start_pile - pile)
    total_spill_delta = max(0, spill - start_spill)
    total_tip_delta = max(
        point_distance(snap["tip"], STATE.get("sand_contact_start_tip")),
        point_distance(snap["load"], STATE.get("sand_contact_start_load")),
    )

    progress_now = bool(
        snap["available"]
        and (
            bucket_delta >= SAND_CONTACT_BUCKET_PROGRESS_MIN
            or pile_delta >= SAND_CONTACT_PILE_PROGRESS_MIN
            or effector_delta >= SAND_CONTACT_TIP_PROGRESS_MIN_M
        )
    )
    if progress_now:
        STATE["sand_contact_last_progress_time"] = now
        STATE["sand_contact_progress_bucket_base"] = bucket
        STATE["sand_contact_progress_pile_base"] = pile
        STATE["sand_contact_progress_spill_base"] = spill
        STATE["sand_contact_progress_tip_base"] = snap["tip"]
        STATE["sand_contact_progress_load_base"] = snap["load"]
        STATE["sand_contact_last_progress_seen"] = True

    last_progress_time = float(STATE.get("sand_contact_last_progress_time", 0.0) or 0.0)
    progress_age = now - last_progress_time if last_progress_time > 0.0 else 999.0
    allow_continue = bool(snap["available"] and progress_age <= SAND_CUT_NO_PROGRESS_TIMEOUT)

    q_lag = None
    if q_cmd is not None and q_real is not None:
        try:
            q_lag = max(q_delta_abs_deg(q_cmd, q_real))
        except Exception:
            q_lag = None

    report = {
        "active": True,
        "stage": str(stage_name),
        "available": bool(snap["available"]),
        "progress_now": progress_now,
        "allow_continue": allow_continue,
        "progress_age": float(progress_age),
        "bucket_delta": int(bucket_delta),
        "pile_delta": int(pile_delta),
        "spill_delta": int(spill_delta),
        "tip_delta": float(tip_delta),
        "load_delta": float(load_delta),
        "effector_delta": float(effector_delta),
        "total_bucket_delta": int(total_bucket_delta),
        "total_pile_delta": int(total_pile_delta),
        "total_spill_delta": int(total_spill_delta),
        "total_tip_delta": float(total_tip_delta),
        "q_lag_deg": q_lag,
        "bucket": int(bucket),
        "pile": int(pile),
        "spill": int(spill),
    }
    STATE["sand_contact_last_report"] = report

    if log and now - float(STATE.get("sand_contact_last_log_time", 0.0) or 0.0) >= SAND_CONTACT_LOG_INTERVAL:
        STATE["sand_contact_last_log_time"] = now
        q_lag_text = "None" if q_lag is None else f"{float(q_lag):.2f}deg"
        info_print(
            "[SAND CONTACT]",
            f"stage={stage_name}",
            f"progress={progress_now}",
            f"continue={allow_continue}",
            f"bucket_total=+{total_bucket_delta}",
            f"bucket_window=+{bucket_delta}",
            f"pile_total=-{total_pile_delta}",
            f"pile_window=-{pile_delta}",
            f"spill_total=+{total_spill_delta}",
            f"tip_window={effector_delta:.3f}m",
            f"age={progress_age:.2f}s",
            f"q_lag={q_lag_text}",
        )
        debug_timeline_record(
            "SAND_CONTACT",
            stage=str(stage_name),
            result="continue" if allow_continue else "no_progress",
            reason=(
                f"progress={progress_now}; bucket_total={total_bucket_delta}; "
                f"pile_total={total_pile_delta}; spill_total={total_spill_delta}; "
                f"age={progress_age:.2f}; q_lag={q_lag_text}"
            ),
            data={
                "bucket_total": total_bucket_delta,
                "pile_total": total_pile_delta,
                "spill_total": total_spill_delta,
                "progress_age": round(float(progress_age), 3),
                "allow_continue": allow_continue,
            },
            include_sand=True,
        )

    return report


def sand_contact_should_suppress_freeze(stage_name, blocked_names, cmd_err_deg, q_cmd, q_real):
    if not is_sand_contact_phase(stage_name):
        return False, None
    if "swing" in blocked_names:
        return False, None
    try:
        boom_idx = CTRL.name_to_idx.get("boom", 1)
        if float(cmd_err_deg[boom_idx]) >= SAND_CONTACT_BOOM_HARD_ERR_DEG:
            return False, None
    except Exception:
        pass
    report = update_sand_contact_progress(stage_name, q_cmd=q_cmd, q_real=q_real, force=False, log=True)
    if not report.get("allow_continue", False):
        return False, report
    if bool(report.get("progress_now", False)):
        STATE["freeze_candidate_since"] = 0.0
    return True, report


def sand_contact_stage_can_advance(q_goal, label="", mode="auto", seconds_eff=0.0):
    stage_name = str(label or mode)
    if not is_sand_contact_phase(stage_name):
        return False
    try:
        q_goal = np.array(q_goal, dtype=np.float32)
        q_real = q_real_near_command(get_real_joint_positions(), q_goal)
        err_deg = q_delta_abs_deg(q_goal, q_real)
    except Exception:
        return False

    swing_idx = CTRL.name_to_idx.get("swing", 0)
    boom_idx = CTRL.name_to_idx.get("boom", 1)
    if float(err_deg[swing_idx]) > MOVE_FINAL_SWING_TOL_DEG * 1.5:
        return False
    if float(err_deg[boom_idx]) >= SAND_CONTACT_BOOM_HARD_ERR_DEG:
        return False

    report = update_sand_contact_progress(stage_name, q_cmd=q_goal, q_real=q_real, force=True, log=True)
    elapsed = time.time() - float(STATE.get("sand_contact_stage_started", time.time()) or time.time())
    min_seconds = max(SAND_CONTACT_STAGE_MIN_SECONDS, min(1.20, float(seconds_eff) * 0.55))
    if elapsed < min_seconds:
        return False

    max_err = max(err_deg) if err_deg else 0.0
    material_progress = (
        int(report.get("total_bucket_delta", 0)) >= SAND_CONTACT_BUCKET_PROGRESS_MIN
        or int(report.get("total_pile_delta", 0)) >= SAND_CONTACT_PILE_PROGRESS_MIN
    )
    tip_progress = float(report.get("total_tip_delta", 0.0) or 0.0) >= SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M
    q_close_enough = float(max_err) <= SAND_CONTACT_Q_LAG_ACCEPT_DEG
    if material_progress or (tip_progress and q_close_enough) or q_close_enough:
        info_print(
            "[SAND CONTACT DONE]",
            f"stage={stage_name}",
            f"elapsed={elapsed:.2f}s",
            f"max_err={max_err:.2f}deg",
            f"bucket_total=+{int(report.get('total_bucket_delta', 0))}",
            f"pile_total=-{int(report.get('total_pile_delta', 0))}",
            f"tip_total={float(report.get('total_tip_delta', 0.0) or 0.0):.3f}m",
            "reason=material_progress_or_compliant_arrival",
        )
        return True
    return False


def sand_contact_stage_should_advance(stage_name, report, q_cmd=None, q_real=None, seconds_eff=0.0):
    if not is_sand_contact_phase(stage_name) or not isinstance(report, dict):
        return False, ""

    now = time.time()
    started = float(STATE.get("sand_contact_stage_started", now) or now)
    elapsed = max(0.0, now - started)
    min_seconds = max(SAND_CONTACT_STAGE_MIN_SECONDS, min(1.20, float(seconds_eff) * 0.45))
    if elapsed < min_seconds:
        return False, ""

    max_err = 0.0
    swing_err = 0.0
    boom_err = 0.0
    if q_cmd is not None and q_real is not None:
        try:
            err_deg = q_delta_abs_deg(q_cmd, q_real)
            max_err = float(max(err_deg)) if err_deg else 0.0
            swing_idx = CTRL.name_to_idx.get("swing", 0)
            boom_idx = CTRL.name_to_idx.get("boom", 1)
            swing_err = float(err_deg[swing_idx])
            boom_err = float(err_deg[boom_idx])
            if swing_err > MOVE_FINAL_SWING_TOL_DEG * 1.5:
                return False, ""
            if boom_err >= SAND_CONTACT_BOOM_HARD_ERR_DEG:
                return False, ""
        except Exception:
            pass

    bucket_total = int(report.get("total_bucket_delta", 0) or 0)
    pile_total = int(report.get("total_pile_delta", 0) or 0)
    spill_total = int(report.get("total_spill_delta", 0) or 0)
    tip_total = float(report.get("total_tip_delta", 0.0) or 0.0)

    if (
        bucket_total >= SAND_CONTACT_ADVANCE_BUCKET_MIN
        or pile_total >= SAND_CONTACT_ADVANCE_PILE_MIN
        or tip_total >= SAND_CONTACT_ADVANCE_TIP_MIN_M
    ):
        return True, (
            f"material_progress elapsed={elapsed:.2f}s bucket={bucket_total} "
            f"pile={pile_total} tip={tip_total:.3f}m spill={spill_total} "
            f"max_err={max_err:.2f}deg boom_err={boom_err:.2f}deg"
        )

    if elapsed >= SAND_CONTACT_MAX_STAGE_WALL_SECONDS and (
        bucket_total >= SAND_CONTACT_ADVANCE_FALLBACK_BUCKET_MIN
        or pile_total >= SAND_CONTACT_ADVANCE_FALLBACK_PILE_MIN
        or tip_total >= SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M
    ):
        return True, (
            f"stage_time_cap elapsed={elapsed:.2f}s bucket={bucket_total} "
            f"pile={pile_total} tip={tip_total:.3f}m spill={spill_total} "
            f"max_err={max_err:.2f}deg boom_err={boom_err:.2f}deg"
        )

    return False, ""


def update_episode_quality_trackers(metrics, phase, q_cmd=None, q_real=None, action=None):
    if not isinstance(metrics, dict) or not metrics.get("available", False):
        return
    bucket = int(metrics.get("bucket_count", 0))
    bucket_from_pile = int(metrics.get("bucket_from_pile_count", 0))
    STATE["dataset_max_bucket_particles"] = max(int(STATE.get("dataset_max_bucket_particles", 0)), bucket)
    STATE["dataset_max_bucket_from_pile_particles"] = max(
        int(STATE.get("dataset_max_bucket_from_pile_particles", 0)),
        bucket_from_pile,
    )
    phase_text = str(phase).lower()
    if "lift" in phase_text or "carry" in phase_text or "unload" in phase_text:
        STATE["dataset_lift_bucket_from_pile_particles"] = max(
            int(STATE.get("dataset_lift_bucket_from_pile_particles", 0)),
            bucket_from_pile,
        )
    STATE["dataset_final_bin_from_pile_particles"] = int(metrics.get("bin_from_pile_count", 0))
    STATE["dataset_final_spill_from_pile_particles"] = int(metrics.get("spill_from_pile_count", 0))
    if q_cmd is not None and q_real is not None:
        err_deg = [abs(rad_to_deg(x)) for x in dataset_joint_error(q_cmd, q_real)]
        STATE["dataset_max_joint_error_deg"] = max(float(STATE.get("dataset_max_joint_error_deg", 0.0)), max(err_deg))
    if action is not None:
        try:
            STATE["dataset_max_action_speed"] = max(
                float(STATE.get("dataset_max_action_speed", 0.0)),
                float(np.max(np.abs(np.array(action, dtype=np.float32)))),
            )
        except Exception:
            pass


def dataset_action_from_q(q_cmd, now):
    q_cmd = np.array(q_cmd, dtype=np.float32)
    q_prev = STATE.get("dataset_last_q_cmd")
    t_prev = float(STATE.get("dataset_last_sample_time", 0.0))
    if q_prev is None or t_prev <= 0.0:
        STATE["dataset_last_q_cmd"] = q_cmd.copy()
        return [0.0, 0.0, 0.0, 0.0]

    q_prev = np.array(q_prev, dtype=np.float32)
    dt = max(1e-4, float(now) - t_prev)
    dq = q_cmd - q_prev
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        dq[swing_idx] = swing_delta(q_cmd[swing_idx], q_prev[swing_idx])
    except Exception:
        pass
    STATE["dataset_last_q_cmd"] = q_cmd.copy()
    return [float(x) for x in (dq / dt)]


def dataset_joint_velocity_from_real(q_real, now):
    q_real = np.array(q_real, dtype=np.float32)
    q_prev = STATE.get("dataset_last_q_real")
    t_prev = float(STATE.get("dataset_last_sample_time", 0.0))
    if q_prev is None or t_prev <= 0.0:
        STATE["dataset_last_q_real"] = q_real.copy()
        return [0.0, 0.0, 0.0, 0.0]

    q_prev = np.array(q_prev, dtype=np.float32)
    dt = max(1e-4, float(now) - t_prev)
    dq = q_real - q_prev
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        dq[swing_idx] = swing_delta(q_real[swing_idx], q_prev[swing_idx])
    except Exception:
        pass
    STATE["dataset_last_q_real"] = q_real.copy()
    return [float(x) for x in (dq / dt)]


def dataset_joint_error(q_cmd, q_real):
    q_cmd = np.array(q_cmd, dtype=np.float32).copy()
    q_real = np.array(q_real, dtype=np.float32).copy()
    err = q_cmd - q_real
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        err[swing_idx] = swing_delta(q_cmd[swing_idx], q_real[swing_idx])
    except Exception:
        pass
    return [float(x) for x in err]


def dataset_observation_state(q_real=None):
    if q_real is None:
        q_real = get_real_joint_positions()
    q_real = np.array(q_real, dtype=np.float32)
    base = get_prim_translation(ROBOT_BASE) if ROBOT_BASE else np.zeros(3, dtype=np.float32)
    tip = bucket_tip_pos()
    load = bucket_load_pos()
    bucket_load = dataset_bucket_load_estimate()
    state = [
        float(base[0]),
        float(base[1]),
        float(get_base_yaw_rad()),
        float(q_real[CTRL.name_to_idx["swing"]]),
        float(q_real[CTRL.name_to_idx["boom"]]),
        float(q_real[CTRL.name_to_idx["arm"]]),
        float(q_real[CTRL.name_to_idx["bucket"]]),
        float(bucket_load),
    ]
    if tip is not None:
        state.extend([float(tip[0]), float(tip[1]), float(tip[2])])
    else:
        state.extend([0.0, 0.0, 0.0])
    if load is not None:
        state.extend([float(load[0]), float(load[1]), float(load[2])])
    else:
        state.extend([0.0, 0.0, 0.0])
    return state


def dataset_record_sample(phase, q_cmd=None, q_real=None, label="", force=False):
    if not bool(STATE.get("dataset_recording", False)):
        return
    if not str(STATE.get("dataset_episode_uid", "") or ""):
        return
    if not str(STATE.get("dataset_episode_dir", "") or ""):
        return

    now = time.time()
    interval = float(STATE.get("dataset_sample_interval", 0.10))
    if (not force) and now - float(STATE.get("dataset_last_sample_time", 0.0)) < interval:
        return

    try:
        if q_cmd is None:
            q_cmd = CTRL.q_cmd.copy()
        if q_real is None:
            q_real = get_real_joint_positions()

        action = dataset_action_from_q(q_cmd, now)
        joint_velocity = dataset_joint_velocity_from_real(q_real, now)
        sand_metrics = sand_metrics_current(force=force)
        update_episode_quality_trackers(sand_metrics, phase, q_cmd=q_cmd, q_real=q_real, action=action)
        q_goal = STATE.get("dataset_current_q_goal")
        target = get_target_pos() if TARGET_PATH else None
        sample_index = int(STATE.get("dataset_samples", 0)) - int(STATE.get("dataset_episode_sample_start", 0))
        sample = {
            "v": DATASET_TRAJECTORY_FORMAT,
            "ep": int(STATE.get("dataset_episode_id", 0)),
            "id": str(STATE.get("dataset_episode_uid", "")),
            "i": int(sample_index),
            "t": float(now) - float(STATE.get("dataset_episode_start_time", now)),
            "phase": str(phase),
            "label": str(label),
            "obs.state": dataset_observation_state(q_real=q_real),
            "obs.q": vec_list(q_real, 4),
            "obs.dq": joint_velocity,
            "obs.q_cmd": vec_list(q_cmd, 4),
            "obs.q_err": dataset_joint_error(q_cmd, q_real),
            "action": action,
            "goal.q": vec_list(q_goal, 4),
            "target": vec_list(target, 3),
            "bucket.tip": vec_list(bucket_tip_pos(), 3),
            "bucket.load": vec_list(bucket_load_pos(), 3),
            "bucket.pour": vec_list(bucket_pour_pos(), 3),
            "sand": compact_sand_metrics(sand_metrics),
        }

        path = str(STATE.get("dataset_path", "excavator_dataset.jsonl"))
        append_jsonl(path, sample)
        sand_path = str(STATE.get("dataset_sand_metrics_path", ""))
        if sand_path:
            append_jsonl(
                sand_path,
                {
                    "episode_id": str(STATE.get("dataset_episode_uid", "")),
                    "timestamp": float(now),
                    "t_episode": float(now) - float(STATE.get("dataset_episode_start_time", now)),
                    "phase": str(phase),
                    "label": str(label),
                    "sand": compact_sand_metrics(sand_metrics),
                },
            )

        STATE["dataset_last_sample_time"] = now
        STATE["dataset_samples"] = int(STATE.get("dataset_samples", 0)) + 1
    except Exception as e:
        if now - float(STATE.get("dataset_last_error_time", 0.0)) > 2.0:
            STATE["dataset_last_error_time"] = now
            info_print("[WARN] dataset record failed:", type(e).__name__, e)


def dataset_record_event(event, detail="", data=None):
    if not bool(STATE.get("dataset_recording", False)):
        return
    try:
        sample = {
            "episode_index": int(STATE.get("dataset_episode_id", 0)),
            "episode_id": str(STATE.get("dataset_episode_uid", "")),
            "timestamp": float(time.time()),
            "event": str(event),
            "detail": str(detail),
            "active_task": str(STATE.get("active_task_name", "idle")),
            "schema": AUTO_COLLECT_SCHEMA,
        }
        if data is not None:
            sample["data"] = data
        path = str(STATE.get("dataset_event_path") or STATE.get("dataset_path", "excavator_dataset.jsonl"))
        append_jsonl(path, sample)
        debug_timeline_record(
            str(event),
            result="event",
            reason=detail,
            data=data,
            include_sand=str(event) in {"phase_metrics", "episode_end", "freeze", "execution_failure"},
        )
    except Exception as e:
        info_print("[WARN] dataset event failed:", type(e).__name__, e)


def auto_collect_status_text():
    run_dir = str(STATE.get("auto_collect_run_dir", ""))
    trainable = jsonl_line_count(os.path.join(run_dir, "trainable_episodes.jsonl")) if run_dir else 0
    planning = jsonl_line_count(os.path.join(run_dir, "planning_diagnostics.jsonl")) if run_dir else 0
    return (
        f"[AUTO DATASET] active={STATE.get('auto_collect_active')} "
        f"requested={STATE.get('auto_collect_requested')} "
        f"attempts={STATE.get('auto_collect_attempts')} "
        f"success={STATE.get('auto_collect_successes')} "
        f"trainable={trainable} "
        f"rejected={STATE.get('auto_collect_rejections')} "
        f"fail={STATE.get('auto_collect_failures')} "
        f"planning={planning} "
        f"run_dir={STATE.get('auto_collect_run_dir')}"
    )


def ensure_auto_collect_run_dir():
    run_dir = str(STATE.get("auto_collect_run_dir", ""))
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        if not str(STATE.get("debug_timeline_path", "") or ""):
            STATE["debug_timeline_path"] = os.path.join(run_dir, DATASET_DEBUG_TIMELINE_FILE)
        return run_dir

    run_id = "run_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(AUTO_COLLECT_DATASET_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    STATE["auto_collect_run_id"] = run_id
    STATE["auto_collect_run_dir"] = run_dir
    STATE["debug_timeline_path"] = os.path.join(run_dir, DATASET_DEBUG_TIMELINE_FILE)
    config_snapshot = full_config_snapshot()
    config_hash = stable_json_hash(config_snapshot)
    write_json_file(os.path.join(run_dir, "sand_config.json"), config_snapshot["sand"])
    write_json_file(os.path.join(run_dir, "auto_dataset_config.json"), config_snapshot["auto_dataset"])
    write_json_file(os.path.join(run_dir, "planner_config.json"), config_snapshot["planner"])
    write_json_file(os.path.join(run_dir, "quality_gate_config.json"), config_snapshot["quality_gate"])
    write_json_file(
        os.path.join(run_dir, "run_meta.json"),
        {
            "run_id": run_id,
            "created_at": time.time(),
            "schema": AUTO_COLLECT_SCHEMA,
            "planner_version": PLANNER_VERSION,
            "quality_gate_version": QUALITY_GATE_VERSION,
            "config_hash": config_hash,
            "dataset_root": AUTO_COLLECT_DATASET_ROOT,
            "state_names": DATASET_STATE_NAMES,
            "action_names": DATASET_ACTION_NAMES,
            "trajectory_fields": {
                "obs.state": DATASET_STATE_NAMES,
                "obs.q": DOF_ORDER,
                "obs.dq": DOF_ORDER,
                "obs.q_cmd": DOF_ORDER,
                "obs.q_err": DOF_ORDER,
                "action": DATASET_ACTION_NAMES,
                "goal.q": DOF_ORDER,
                "target": ["x", "y", "z"],
                "bucket.tip": ["x", "y", "z"],
                "bucket.load": ["x", "y", "z"],
                "bucket.pour": ["x", "y", "z"],
                "sand": [
                    "n",
                    "pile",
                    "bucket",
                    "bucket_from_pile",
                    "bin",
                    "bin_from_pile",
                    "spill_from_pile",
                    "bucket_from_pile_mass",
                    "bin_from_pile_mass",
                    "spill_from_pile_mass",
                ],
            },
            "trainable_index": "trainable_episodes.jsonl",
            "successful_index": "successful_episodes.jsonl",
            "rejected_index": "rejected_episodes.jsonl",
            "failed_index": "failed_episodes.jsonl",
            "diagnostic_index": "diagnostic_episodes.jsonl",
            "planning_diagnostics_index": "planning_diagnostics.jsonl",
            "debug_timeline_index": DATASET_DEBUG_TIMELINE_FILE,
            "trajectory_format": DATASET_TRAJECTORY_FORMAT,
            "scene_context": compact_scene_context(),
            "target_center": vec_list(AUTO_COLLECT_TARGET_CENTER, 3),
            "target_radius_x": AUTO_COLLECT_TARGET_RADIUS_X,
            "target_radius_y": AUTO_COLLECT_TARGET_RADIUS_Y,
            "target_depths": AUTO_COLLECT_TARGET_DEPTHS,
            "initial_pose_family": AUTO_COLLECT_INITIAL_POSES_DEG,
            "dig_plan_candidate_family": [
                {
                    "id": str(x.get("id", "")),
                    "approach_offset": float(x.get("approach_offset", 0.0)),
                    "lift_height": float(x.get("lift_height", 0.0)),
                    "unload_height_delta": float(x.get("unload_height_delta", 0.0)),
                    "unload_dump_deg": float(x.get("unload_dump_deg", BUCKET_UNLOAD_DUMP_DEG)),
                }
                for x in DIG_PLAN_CANDIDATES
            ],
            "sand_reset_policy": AUTO_COLLECT_SAND_RESET_POLICY,
            "auto_reset_sand_after_world_ready": AUTO_RESET_SAND_AFTER_WORLD_READY,
            "auto_reset_sand_after_ui_ready": AUTO_RESET_SAND_AFTER_UI_READY,
            "auto_reset_sand_ui_ready_delay_frames": AUTO_RESET_SAND_UI_READY_DELAY_FRAMES,
            "auto_collect_reuse_ready_sand_reset": AUTO_COLLECT_REUSE_READY_SAND_RESET,
            "pre_reset_settle_frames": AUTO_COLLECT_PRE_RESET_SETTLE_FRAMES,
            "post_reset_settle_frames": AUTO_COLLECT_RESET_SETTLE_FRAMES,
            "stable_reset_min_frames": SAND_RESET_SETTLE_MIN_FRAMES,
            "stable_reset_max_frames": SAND_RESET_SETTLE_MAX_FRAMES,
        },
    )
    for index_name in [
        "episodes.jsonl",
        "successful_episodes.jsonl",
        "trainable_episodes.jsonl",
        "rejected_episodes.jsonl",
        "failed_episodes.jsonl",
        "diagnostic_episodes.jsonl",
        "planning_diagnostics.jsonl",
        DATASET_DEBUG_TIMELINE_FILE,
    ]:
        open(os.path.join(run_dir, index_name), "a", encoding="utf-8").close()
    info_print("[AUTO DATASET] created run", f"run_dir={run_dir}")
    debug_timeline_record(
        "RUN_START",
        result="created",
        data={"run_dir": run_dir, "config_hash": config_hash, "planner_version": PLANNER_VERSION},
    )
    return run_dir


def auto_collect_write_run_summary():
    run_dir = str(STATE.get("auto_collect_run_dir", ""))
    if not run_dir:
        return
    episodes_count = jsonl_line_count(os.path.join(run_dir, "episodes.jsonl"))
    success_count = jsonl_line_count(os.path.join(run_dir, "successful_episodes.jsonl"))
    trainable_count = jsonl_line_count(os.path.join(run_dir, "trainable_episodes.jsonl"))
    rejected_count = jsonl_line_count(os.path.join(run_dir, "rejected_episodes.jsonl"))
    failed_count = jsonl_line_count(os.path.join(run_dir, "failed_episodes.jsonl"))
    diagnostic_count = jsonl_line_count(os.path.join(run_dir, "diagnostic_episodes.jsonl"))
    planning_count = jsonl_line_count(os.path.join(run_dir, "planning_diagnostics.jsonl"))
    attempts = max(int(STATE.get("auto_collect_attempts", 0)), episodes_count)
    STATE["auto_collect_attempts"] = attempts
    STATE["auto_collect_successes"] = max(int(STATE.get("auto_collect_successes", 0)), success_count)
    STATE["auto_collect_rejections"] = max(int(STATE.get("auto_collect_rejections", 0)), rejected_count)
    STATE["auto_collect_failures"] = max(int(STATE.get("auto_collect_failures", 0)), failed_count)
    STATE["auto_collect_planning_diagnostics"] = max(
        int(STATE.get("auto_collect_planning_diagnostics", 0)),
        planning_count,
    )
    write_json_file(
        os.path.join(run_dir, "summary.json"),
        {
            "run_id": STATE.get("auto_collect_run_id", ""),
            "updated_at": time.time(),
            "planner_version": PLANNER_VERSION,
            "quality_gate_version": QUALITY_GATE_VERSION,
            "config_hash": current_config_hash(),
            "requested": int(STATE.get("auto_collect_requested", 0)),
            "attempts": attempts,
            "episodes_index_rows": episodes_count,
            "successes": int(STATE.get("auto_collect_successes", 0)),
            "trainable": trainable_count,
            "rejections": int(STATE.get("auto_collect_rejections", 0)),
            "failures": int(STATE.get("auto_collect_failures", 0)),
            "diagnostic": diagnostic_count,
            "planning_diagnostics": int(STATE.get("auto_collect_planning_diagnostics", 0)),
            "last_result": STATE.get("auto_collect_last_result", ""),
            "run_dir": run_dir,
            "trainable_index": os.path.join(run_dir, "trainable_episodes.jsonl"),
            "successful_index": os.path.join(run_dir, "successful_episodes.jsonl"),
            "rejected_index": os.path.join(run_dir, "rejected_episodes.jsonl"),
            "failed_index": os.path.join(run_dir, "failed_episodes.jsonl"),
            "diagnostic_index": os.path.join(run_dir, "diagnostic_episodes.jsonl"),
            "planning_diagnostics_index": os.path.join(run_dir, "planning_diagnostics.jsonl"),
            "debug_timeline_index": os.path.join(run_dir, DATASET_DEBUG_TIMELINE_FILE),
            "scene_context": compact_scene_context(),
            "sand_reset_policy": AUTO_COLLECT_SAND_RESET_POLICY,
            "sand_reset_done": bool(STATE.get("auto_collect_sand_reset_done", False)),
            "sand_site_stable_reset_done": bool(STATE.get("sand_site_stable_reset_done", False)),
            "sand_site_last_reset_label": STATE.get("sand_site_last_reset_label", ""),
            "auto_reset_sand_after_ui_ready": AUTO_RESET_SAND_AFTER_UI_READY,
        },
    )


def auto_collect_episode_dir(attempt_index):
    run_dir = ensure_auto_collect_run_dir()
    return os.path.join(run_dir, f"episode_{int(attempt_index):06d}")


def auto_collect_plan_summary(seq):
    points = STATE.get("dig_plan_points", None) or []
    candidate = STATE.get("dig_plan_candidate")
    candidate_stages = candidate.get("stages", []) if isinstance(candidate, dict) else []
    rows = []
    if not seq:
        return rows
    for i, item in enumerate(seq):
        stage_name, q_goal, duration = item
        point = points[i] if i < len(points) else None
        row = {
            "index": i,
            "phase": str(stage_name),
            "duration": float(duration),
            "q_goal_rad": vec_list(q_goal, 4),
            "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
            "target_point": vec_list(point, 3),
        }
        if i < len(candidate_stages) and isinstance(candidate_stages[i], dict):
            stage = candidate_stages[i]
            if stage.get("q_dump_rad") is not None:
                row["q_dump_rad"] = vec_list(stage.get("q_dump_rad"), 4)
            if stage.get("q_dump_deg") is not None:
                row["q_dump_deg"] = stage.get("q_dump_deg")
            if stage.get("drop") is not None:
                row["drop"] = stage.get("drop")
        rows.append(row)
    return rows


def auto_collect_episode_metrics():
    q_real = get_real_joint_positions()
    q_cmd = CTRL.q_cmd.copy()
    sand_metrics = sand_metrics_current(force=True)
    return {
        "q_real_rad": vec_list(q_real, 4),
        "q_cmd_rad": vec_list(q_cmd, 4),
        "q_real_deg": q_deg_values(q_real, wrap_swing_for_display=True),
        "q_cmd_deg": q_deg_values(q_cmd, wrap_swing_for_display=True),
        "target_xyz": vec_list(get_target_pos() if TARGET_PATH else None, 3),
        "bucket_tip_xyz": vec_list(bucket_tip_pos(), 3),
        "bucket_mid_xyz": vec_list(bucket_mid_pos(), 3),
        "bucket_load_xyz": vec_list(bucket_load_pos(), 3),
        "bucket_pour_xyz": vec_list(bucket_pour_pos(), 3),
        "unload_drop": compact_unload_drop(),
        "bucket_load_estimate": dataset_bucket_load_estimate(),
        "sand": compact_sand_metrics(sand_metrics),
        "support_clearance": support_clearance_detail(),
        "freeze_count": int(STATE.get("dataset_episode_freezes", 0)),
        "samples": int(STATE.get("dataset_samples", 0)) - int(STATE.get("dataset_episode_sample_start", 0)),
        "dig_plan_step_index": int(STATE.get("dig_plan_step_index", 0)),
        "max_bucket_from_pile_particles": int(STATE.get("dataset_max_bucket_from_pile_particles", 0)),
        "lift_bucket_from_pile_particles": int(STATE.get("dataset_lift_bucket_from_pile_particles", 0)),
        "final_bin_from_pile_particles": int(STATE.get("dataset_final_bin_from_pile_particles", 0)),
        "final_spill_from_pile_particles": int(STATE.get("dataset_final_spill_from_pile_particles", 0)),
        "max_joint_error_deg": float(STATE.get("dataset_max_joint_error_deg", 0.0)),
        "max_action_speed": float(STATE.get("dataset_max_action_speed", 0.0)),
    }


def record_phase_metrics(label, q_cmd=None, q_real=None, action=None):
    label = str(label)
    metrics = sand_metrics_current(force=True)
    q_cmd_now = CTRL.q_cmd.copy() if q_cmd is None else np.array(q_cmd, dtype=np.float32).copy()
    q_real_now = get_real_joint_positions() if q_real is None else np.array(q_real, dtype=np.float32).copy()
    update_episode_quality_trackers(metrics, label, q_cmd=q_cmd_now, q_real=q_real_now, action=action)

    phase_metrics = STATE.get("dataset_phase_metrics")
    if not isinstance(phase_metrics, dict):
        phase_metrics = {}
    row = {
        "time": time.time(),
        "sand": compact_sand_metrics(metrics),
        "q_cmd_rad": vec_list(q_cmd_now, 4),
        "q_real_rad": vec_list(q_real_now, 4),
        "q_cmd_deg": q_deg_values(q_cmd_now, wrap_swing_for_display=True),
        "q_real_deg": q_deg_values(q_real_now, wrap_swing_for_display=True),
        "bucket_tip_xyz": vec_list(bucket_tip_pos(), 3),
        "bucket_mid_xyz": vec_list(bucket_mid_pos(), 3),
        "bucket_load_xyz": vec_list(bucket_load_pos(), 3),
        "bucket_pour_xyz": vec_list(bucket_pour_pos(), 3),
        "unload_drop": compact_unload_drop(),
    }
    phase_metrics[label] = row
    STATE["dataset_phase_metrics"] = phase_metrics

    if bool(STATE.get("dataset_recording", False)):
        dataset_record_event(
            "phase_metrics",
            (
                f"{label}: "
                f"bucket_from_pile={int(metrics.get('bucket_from_pile_count', 0))} "
                f"bin_from_pile={int(metrics.get('bin_from_pile_count', 0))} "
                f"spill_from_pile={int(metrics.get('spill_from_pile_count', 0))}"
            ),
            data=row,
        )
    return metrics


def auto_collect_preflight_report(target_successes=None):
    run_dir = ensure_auto_collect_run_dir()
    ctx = task_scene_context()
    metrics = sand_metrics_current(force=True)
    particles = sand_particle_positions()
    particle_count = int(len(particles)) if particles is not None else int(metrics.get("particle_count", 0) or 0)
    api = get_sand_site_api()
    bucket_collider_ready = False
    try:
        bucket_collider_ready = bool(ensure_sand_site_bucket_colliders(force=True))
    except Exception as e:
        bucket_collider_ready = False
        info_print("[WARN] [AUTO PREFLIGHT] bucket collider check failed:", type(e).__name__, e)

    trace_cache_ready = True
    try:
        ensure_trace_prims()
    except Exception as e:
        trace_cache_ready = False
        info_print("[WARN] [AUTO PREFLIGHT] trace prim check failed:", type(e).__name__, e)

    dataset_writable = False
    write_reason = ""
    try:
        os.makedirs(run_dir, exist_ok=True)
        test_path = os.path.join(run_dir, "_preflight_write_test.tmp")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok\n")
        try:
            os.remove(test_path)
        except Exception:
            pass
        dataset_writable = True
    except Exception as e:
        write_reason = f"{type(e).__name__}: {e}"

    bucket_frame_ready = bool(BUCKET_LINK and get_prim(BUCKET_LINK).IsValid())
    bucket_tip_ready = bucket_tip_pos() is not None
    action_detail = articulation_action_ready_detail()
    action_ready = bool(action_detail.get("ready", False))
    unload_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    unload_inner = np.array(ctx["unload_bin_inner_size"], dtype=np.float32).reshape(-1)[:2]
    pile_center = np.array(ctx["pile_center"], dtype=np.float32).reshape(-1)[:3]
    pile_radius = np.array(ctx["pile_radius"], dtype=np.float32).reshape(-1)[:2]

    checks = {
        "sand_site": api is not None and bool(sand_site_active()),
        "particle_system": sand_particle_prim() is not None,
        "particle_count": particle_count >= int(AUTO_PREFLIGHT_MIN_PARTICLES),
        "unload_bin": np.all(unload_inner > 0.05),
        "bucket_collider": bucket_collider_ready,
        "robot": ROBOT is not None and len(DOF_NAME_TO_REAL_IDX) >= 4,
        "action_channel": action_ready,
        "ik": IK_MODEL is not None,
        "safe_initial_pose": len(AUTO_COLLECT_INITIAL_POSES_DEG) > 0,
        "bucket_frame": bucket_frame_ready and bucket_tip_ready,
        "trace_cache": trace_cache_ready,
        "run_dir": bool(run_dir),
        "dataset_writable": dataset_writable,
        "target_success": target_successes is None or int(target_successes) > 0,
        "quality_thresholds": QUALITY_MIN_SCORE > 0 and QUALITY_MIN_BUCKET_PARTICLES >= 0 and QUALITY_MIN_DUMP_PARTICLES >= 0,
    }

    reason = ""
    reason_map = {
        "sand_site": "preflight_failed/sand_missing",
        "particle_system": "preflight_failed/particle_system_missing",
        "particle_count": "preflight_failed/not_enough_particles",
        "unload_bin": "preflight_failed/unload_bin_missing",
        "bucket_collider": "preflight_failed/bucket_collider_not_ready",
        "robot": "preflight_failed/robot_not_ready",
        "action_channel": "preflight_failed/action_channel_not_ready",
        "ik": "preflight_failed/ik_not_ready",
        "safe_initial_pose": "preflight_failed/safe_initial_pose_missing",
        "bucket_frame": "preflight_failed/bucket_frame_missing",
        "trace_cache": "preflight_failed/trace_cache_not_ready",
        "run_dir": "preflight_failed/run_dir_missing",
        "dataset_writable": "preflight_failed/dataset_not_writable",
        "target_success": "preflight_failed/invalid_target_success_count",
        "quality_thresholds": "preflight_failed/quality_thresholds_invalid",
    }
    for key, ok in checks.items():
        if not bool(ok):
            reason = reason_map.get(key, f"preflight_failed/{key}")
            break

    report = {
        "ok": not bool(reason),
        "reason": reason,
        "checks": checks,
        "sand": {
            "ok": checks["sand_site"] and checks["particle_system"] and checks["particle_count"],
            "center": vec_list(pile_center, 3),
            "radius": vec_list(pile_radius, 2),
            "particles": particle_count,
            "min_particles": int(AUTO_PREFLIGHT_MIN_PARTICLES),
            "status": dataset_sand_status(),
        },
        "bin": {
            "ok": checks["unload_bin"],
            "center": vec_list(unload_center, 3),
            "inner_size": vec_list(unload_inner, 2),
            "z_range": vec_list(ctx["unload_bin_z_range"], 2),
        },
        "robot": {
            "ok": checks["robot"],
            "action_ready": action_ready,
            "action_detail": action_detail,
            "ik_calibrated": checks["ik"],
            "bucket_collider_ready": bucket_collider_ready,
            "bucket_frame_ready": bucket_frame_ready,
            "bucket_tip_ready": bucket_tip_ready,
            "dof_names": list(DOF_NAME_TO_REAL_IDX.keys()),
        },
        "planner": {
            "ok": checks["safe_initial_pose"] and checks["bucket_frame"] and checks["trace_cache"],
            "trace_cache_ready": trace_cache_ready,
            "version": PLANNER_VERSION,
        },
        "dataset": {
            "ok": checks["run_dir"] and checks["dataset_writable"] and checks["target_success"],
            "run_dir": run_dir,
            "target_success": target_successes,
            "write_reason": write_reason,
            "config_hash": current_config_hash(),
        },
    }
    STATE["last_auto_preflight"] = report
    info_print(
        "[AUTO PREFLIGHT]",
        f"sand={'OK' if report['sand']['ok'] else 'BAD'} center={report['sand']['center']} radius={report['sand']['radius']} particles={particle_count}",
        f"bin={'OK' if report['bin']['ok'] else 'BAD'} center={report['bin']['center']} size={report['bin']['inner_size']}",
        f"robot={'OK' if report['robot']['ok'] else 'BAD'} IK={checks['ik']} action_ready={action_ready} bucket_collider={bucket_collider_ready}",
        f"planner={'OK' if report['planner']['ok'] else 'BAD'} trace_cache={trace_cache_ready}",
        f"dataset={'OK' if report['dataset']['ok'] else 'BAD'} run_dir={run_dir} target_success={target_successes}",
        f"reason={reason or 'ok'}",
    )
    debug_timeline_record(
        "PREFLIGHT",
        result="ok" if not reason else "failed",
        reason=reason or "ok",
        data={
            "sand_ok": report["sand"]["ok"],
            "particles": particle_count,
            "bin_ok": report["bin"]["ok"],
            "robot_ok": report["robot"]["ok"],
            "planner_ok": report["planner"]["ok"],
            "dataset_ok": report["dataset"]["ok"],
        },
        include_sand=False,
    )
    return not bool(reason), report, reason


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def auto_collect_record_planning_diagnostic(attempt_index, target, plan_attempts, reason, initial_info=None):
    run_dir = ensure_auto_collect_run_dir()
    best_failure = STATE.get("dig_plan_best_failure")
    chosen_plan = STATE.get("dig_plan_candidate")
    shared_plan = STATE.get("current_dig_plan")
    initial_info = initial_info if isinstance(initial_info, dict) else {}
    row = {
        "attempt": int(attempt_index),
        "status": "planning_diagnostic",
        "reason": str(reason),
        "created_at": time.time(),
        "planner_version": PLANNER_VERSION,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "config_hash": current_config_hash(),
        "target_xyz": vec_list(target, 3),
        "unload_landing_xyz": vec_list(unload_bin_landing_point(), 3),
        "unload_release_xyz": vec_list(unload_bin_dump_point(), 3),
        "initial_pose_id": str(initial_info.get("id", "")),
        "q_initial_deg": q_deg_values(initial_info.get("q"), wrap_swing_for_display=True) if initial_info.get("q") is not None else None,
        "preflight": STATE.get("last_auto_preflight", {}),
        "best_failure": compact_plan_candidate(best_failure, include_stages=True),
        "chosen_plan": compact_plan_candidate(chosen_plan, include_stages=True),
        "shared_plan": {
            "plan_id": shared_plan.get("plan_id", "") if isinstance(shared_plan, dict) else "",
            "cost": shared_plan.get("total_plan_cost") if isinstance(shared_plan, dict) else None,
            "stage_count": shared_plan.get("stage_count") if isinstance(shared_plan, dict) else None,
        },
        "plan_attempts": plan_attempts if isinstance(plan_attempts, list) else [],
        "dig_target_candidates": STATE.get("last_auto_dig_target_scores", []),
        "unload_flat_fill_candidates": STATE.get("last_auto_unload_scores", []),
    }
    append_jsonl(os.path.join(run_dir, "planning_diagnostics.jsonl"), row)
    debug_timeline_record(
        "PLAN_FAIL",
        result="planning_diagnostic",
        reason=reason,
        data={
            "attempt": int(attempt_index),
            "target_xyz": vec_list(target, 3),
            "best_failure": compact_plan_candidate(best_failure, include_stages=False),
            "chosen_plan": compact_plan_candidate(chosen_plan, include_stages=False),
        },
        include_sand=False,
    )
    STATE["auto_collect_planning_diagnostics"] = int(STATE.get("auto_collect_planning_diagnostics", 0)) + 1
    STATE["auto_collect_last_result"] = f"planning_diagnostic attempt={int(attempt_index)} reason={reason}"
    info_print(
        "[AUTO DATASET PLANNING DIAG]",
        f"attempt={int(attempt_index)}",
        f"target={vec_list(target, 3)}",
        f"reason={reason}",
    )
    auto_collect_write_run_summary()
    return row


def compute_episode_quality_score(execution_success, reason):
    final_metrics = auto_collect_episode_metrics()
    max_bucket = int(final_metrics.get("max_bucket_from_pile_particles", 0))
    lift_bucket = int(final_metrics.get("lift_bucket_from_pile_particles", 0))
    final_bin = int(final_metrics.get("final_bin_from_pile_particles", 0))
    final_spill = int(final_metrics.get("final_spill_from_pile_particles", 0))
    freezes = int(final_metrics.get("freeze_count", 0))
    samples = int(final_metrics.get("samples", 0))
    max_joint_err = float(final_metrics.get("max_joint_error_deg", 0.0))

    dig_load_score = clamp01(max_bucket / max(1.0, float(QUALITY_TARGET_BUCKET_PARTICLES)))
    dump_transfer_score = clamp01(final_bin / max(1.0, float(max(lift_bucket, max_bucket))))
    retention_score = clamp01(lift_bucket / max(1.0, float(max_bucket)))
    smoothness_score = clamp01(1.0 - max(0.0, max_joint_err - 8.0) / 35.0)
    completion_score = 1.0 if execution_success and freezes == 0 else 0.0
    spill_ratio = final_spill / max(1.0, float(max_bucket + final_bin + final_spill))

    score = (
        35.0 * dig_load_score
        + 30.0 * dump_transfer_score
        + 15.0 * retention_score
        + 10.0 * smoothness_score
        + 10.0 * completion_score
    )
    score -= min(30.0, freezes * 15.0)
    score -= min(20.0, spill_ratio * 20.0)
    score = max(0.0, min(100.0, score))

    quality_reasons = []
    if not execution_success:
        raw_reason = str(reason)
        if raw_reason.startswith(("preflight_failed/", "planning_failed/", "execution_failed/")):
            quality_reasons.append(raw_reason)
        elif "freeze" in raw_reason:
            quality_reasons.append(f"execution_failed/freeze_detected:{raw_reason}")
        elif "plan" in raw_reason:
            quality_reasons.append(f"planning_failed/{raw_reason}")
        else:
            quality_reasons.append(f"execution_failed/{raw_reason}")
    if freezes > 0:
        quality_reasons.append(f"execution_failed/freeze_detected:{freezes}")
    if max_bucket < QUALITY_MIN_BUCKET_PARTICLES:
        quality_reasons.append(f"quality_rejected/low_bucket_particles:{max_bucket}")
    if final_bin < QUALITY_MIN_DUMP_PARTICLES:
        quality_reasons.append(f"quality_rejected/low_final_bin_particles:{final_bin}")
    if spill_ratio > QUALITY_MAX_SPILL_RATIO:
        quality_reasons.append(f"quality_rejected/high_spill_ratio:{spill_ratio:.2f}")
    if score < QUALITY_MIN_SCORE:
        quality_reasons.append(f"quality_rejected/score_low:{score:.1f}")
    if samples <= 0:
        quality_reasons.append("diagnostic/no_samples")

    quality_success = execution_success and len(quality_reasons) == 0
    return {
        "success": bool(quality_success),
        "execution_success": bool(execution_success),
        "score": float(score),
        "failure_reason": "; ".join(quality_reasons),
        "raw_reason": str(reason),
        "components": {
            "dig_load_score": dig_load_score,
            "dump_transfer_score": dump_transfer_score,
            "retention_score": retention_score,
            "smoothness_score": smoothness_score,
            "completion_score": completion_score,
            "spill_ratio": spill_ratio,
        },
        "thresholds": {
            "quality_min_score": QUALITY_MIN_SCORE,
            "quality_target_bucket_particles": QUALITY_TARGET_BUCKET_PARTICLES,
            "quality_min_bucket_particles": QUALITY_MIN_BUCKET_PARTICLES,
            "quality_min_dump_particles": QUALITY_MIN_DUMP_PARTICLES,
            "quality_max_spill_ratio": QUALITY_MAX_SPILL_RATIO,
        },
        "counts": {
            "max_bucket_from_pile_particles": max_bucket,
            "lift_bucket_from_pile_particles": lift_bucket,
            "final_bin_from_pile_particles": final_bin,
            "final_spill_from_pile_particles": final_spill,
            "freeze_count": freezes,
            "samples": samples,
        },
        "phase_metrics": STATE.get("dataset_phase_metrics", {}),
        "final_metrics": final_metrics,
    }


def auto_collect_begin_episode(attempt_index, target, plan_attempts, seq, initial_info=None):
    episode_dir = auto_collect_episode_dir(attempt_index)
    os.makedirs(episode_dir, exist_ok=True)
    uid = f"episode_{int(attempt_index):06d}_{int(time.time())}"
    STATE["dataset_episode_id"] = int(attempt_index)
    STATE["dataset_episode_uid"] = uid
    STATE["dataset_episode_dir"] = episode_dir
    STATE["dataset_path"] = os.path.join(episode_dir, "trajectory.jsonl")
    STATE["dataset_event_path"] = os.path.join(episode_dir, "events.jsonl")
    STATE["dataset_meta_path"] = os.path.join(episode_dir, "meta.json")
    STATE["dataset_sand_metrics_path"] = os.path.join(episode_dir, "sand_metrics.jsonl")
    STATE["dataset_episode_start_time"] = time.time()
    STATE["dataset_episode_freezes"] = 0
    STATE["dataset_last_sample_time"] = 0.0
    STATE["dataset_last_q_cmd"] = None
    STATE["dataset_last_q_real"] = None
    STATE["dataset_current_q_goal"] = None
    STATE["last_execution_failure_reason"] = ""
    STATE["sand_metrics_last_time"] = 0.0
    STATE["sand_metrics_last"] = None
    STATE["dataset_initial_pile_particle_ids"] = None
    STATE["dataset_initial_pile_particle_count"] = 0
    STATE["dataset_max_bucket_particles"] = 0
    STATE["dataset_max_bucket_from_pile_particles"] = 0
    STATE["dataset_lift_bucket_from_pile_particles"] = 0
    STATE["dataset_final_bin_from_pile_particles"] = 0
    STATE["dataset_final_spill_from_pile_particles"] = 0
    STATE["dataset_max_joint_error_deg"] = 0.0
    STATE["dataset_max_action_speed"] = 0.0
    STATE["dataset_phase_metrics"] = {}
    initial_ids = capture_initial_pile_particle_ids()
    STATE["dataset_episode_sample_start"] = int(STATE.get("dataset_samples", 0))
    STATE["dataset_recording"] = True

    shared_plan = STATE.get("current_dig_plan")
    if not isinstance(shared_plan, dict):
        shared_plan = {}
    else:
        shared_plan = dict(shared_plan)
        shared_plan["episode_id"] = uid
        STATE["current_dig_plan"] = shared_plan

    chosen_plan = STATE.get("dig_plan_candidate")
    if not isinstance(chosen_plan, dict):
        chosen_plan = {}
    q_initial = None
    initial_pose_id = ""
    if isinstance(initial_info, dict):
        q_initial = initial_info.get("q")
        initial_pose_id = str(initial_info.get("id", ""))
    q_initial_arr = None if q_initial is None else np.array(q_initial, dtype=np.float32).reshape(-1)
    q_initial_deg = None if q_initial_arr is None or len(q_initial_arr) < 4 else q_deg_values(q_initial_arr[:4], wrap_swing_for_display=True)
    plan_unload_point = shared_plan.get("chosen_unload_release_point") or chosen_plan.get("unload_release_xyz") or chosen_plan.get("unload_point_xyz")
    if plan_unload_point is not None:
        unload_point = np.array(plan_unload_point, dtype=np.float32).reshape(-1)[:3]
    else:
        unload_point = unload_bin_dump_point()
    plan_unload_landing = shared_plan.get("chosen_unload_landing_point") or chosen_plan.get("unload_landing_xyz")
    if plan_unload_landing is not None:
        unload_landing = np.array(plan_unload_landing, dtype=np.float32).reshape(-1)[:3]
    else:
        unload_landing = unload_landing_point_from_release(unload_point)
    scene_ctx = compact_scene_context()
    chosen_plan_compact = compact_plan_candidate(chosen_plan, include_stages=True)
    candidates_compact = [
        compact_plan_candidate(x, include_stages=False)
        for x in STATE.get("last_dig_plan_candidates", [])
    ]
    plan_debug_path = os.path.join(episode_dir, DATASET_DEBUG_PLAN_FILE)
    episode_seed = stable_json_hash([STATE.get("auto_collect_run_id", ""), int(attempt_index), "episode"])
    sand_seed = stable_json_hash([STATE.get("auto_collect_run_id", ""), int(attempt_index), "sand"])
    candidate_sampling_seed = stable_json_hash([STATE.get("auto_collect_run_id", ""), int(attempt_index), "candidate_sampling"])
    config_hash = current_config_hash()
    write_json_file(
        plan_debug_path,
        {
            "episode_id": uid,
            "episode_seed": episode_seed,
            "sand_seed": sand_seed,
            "candidate_sampling_seed": candidate_sampling_seed,
            "planner_version": PLANNER_VERSION,
            "quality_gate_version": QUALITY_GATE_VERSION,
            "config_hash": config_hash,
            "target_xyz": vec_list(target, 3),
            "unload_point_xyz": vec_list(unload_point, 3),
            "unload_landing_xyz": vec_list(unload_landing, 3),
            "shared_dig_plan": shared_plan,
            "preflight": STATE.get("last_auto_preflight", {}),
            "dig_target_candidates": STATE.get("last_auto_dig_target_scores", []),
            "unload_flat_fill_candidates": STATE.get("last_auto_unload_scores", []),
            "chosen_plan": chosen_plan_compact,
            "best_failure": compact_plan_candidate(STATE.get("dig_plan_best_failure"), include_stages=True),
            "build_ms": float(STATE.get("dig_plan_last_build_ms", 0.0)),
            "trace_points_count": len(STATE.get("dig_plan_trace_points", []) or []),
            "trace_stage_breaks": STATE.get("dig_plan_trace_stage_breaks", []),
            "candidates": STATE.get("last_dig_plan_candidates", []),
        },
    )

    meta = {
        "episode_id": uid,
        "episode_index": int(attempt_index),
        "status": "running",
        "created_at": STATE["dataset_episode_start_time"],
        "schema": AUTO_COLLECT_SCHEMA,
        "planner_version": PLANNER_VERSION,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "config_hash": config_hash,
        "episode_seed": episode_seed,
        "sand_seed": sand_seed,
        "candidate_sampling_seed": candidate_sampling_seed,
        "task": "Dig soil from the marked area and dump it into the target container.",
        "trajectory_format": DATASET_TRAJECTORY_FORMAT,
        "target_xyz": vec_list(target, 3),
        "unload_point_xyz": vec_list(unload_point, 3),
        "unload_landing_xyz": vec_list(unload_landing, 3),
        "unload_release_xyz": vec_list(unload_point, 3),
        "scene_context": scene_ctx,
        "preflight": STATE.get("last_auto_preflight", {}),
        "initial_pose_id": initial_pose_id,
        "q_initial_rad": None if q_initial_arr is None or len(q_initial_arr) < 4 else vec_list(q_initial_arr[:4], 4),
        "q_initial_deg": q_initial_deg,
        "chosen_plan": chosen_plan_compact,
        "shared_dig_plan": shared_plan,
        "dig_target_candidates": STATE.get("last_auto_dig_target_scores", []),
        "unload_flat_fill_candidates": STATE.get("last_auto_unload_scores", []),
        "plan_candidates": candidates_compact,
        "plan_attempts": plan_attempts,
        "plan": auto_collect_plan_summary(seq),
        "trainable_rule": "full_dig_lift_dump_only",
        "state_names": DATASET_STATE_NAMES,
        "action_names": DATASET_ACTION_NAMES,
        "paths": {
            "trajectory": STATE["dataset_path"],
            "events": STATE["dataset_event_path"],
            "sand_metrics": STATE["dataset_sand_metrics_path"],
            "meta": STATE["dataset_meta_path"],
            "plan_debug": plan_debug_path,
            "debug_timeline": debug_timeline_path(create=False),
        },
        "initial_pile_particle_count": len(initial_ids),
        "initial_metrics": auto_collect_episode_metrics(),
    }
    write_json_file(STATE["dataset_meta_path"], meta)
    dataset_record_event(
        "episode_start",
        (
            f"episode_id={uid}; target={vec_list(target, 3)}; "
            f"unload={vec_list(unload_point, 3)}; landing={vec_list(unload_landing, 3)}; initial_pose={initial_pose_id}; "
            f"scene_source={scene_ctx.get('source')}"
        ),
    )
    dataset_record_event("plan_ready", f"steps={0 if seq is None else len(seq)}")
    debug_timeline_record(
        "EP_START",
        result="running",
        data={
            "attempt": int(attempt_index),
            "initial_pose_id": initial_pose_id,
            "target_xyz": vec_list(target, 3),
            "unload_landing_xyz": vec_list(unload_landing, 3),
            "unload_release_xyz": vec_list(unload_point, 3),
            "stage_count": 0 if seq is None else len(seq),
            "plan_debug": plan_debug_path,
        },
        include_sand=True,
    )
    return meta


def auto_collect_finish_episode(meta, success, reason):
    now = time.time()
    execution_success = bool(success)
    reason = str(reason)
    score_report = compute_episode_quality_score(execution_success, reason)
    success = bool(score_report.get("success", False))
    metrics = score_report.get("final_metrics", auto_collect_episode_metrics())
    moved_soil = max(
        int(metrics.get("max_bucket_from_pile_particles", 0)),
        int(metrics.get("lift_bucket_from_pile_particles", 0)),
    )
    samples = int(metrics.get("samples", 0))
    rejected = (not success) and samples > 0 and moved_soil >= QUALITY_MIN_BUCKET_PARTICLES
    diagnostic = (not success) and ("diagnostic/" in str(score_report.get("failure_reason", "")))
    status = "trainable" if success else ("rejected" if rejected else ("diagnostic" if diagnostic else "failed"))
    meta["status"] = status
    meta["success"] = success
    meta["execution_success"] = execution_success
    meta["failure_reason"] = "" if success else score_report.get("failure_reason", reason)
    meta["score_summary"] = {
        "score": score_report.get("score"),
        "success": score_report.get("success"),
        "failure_reason": score_report.get("failure_reason", ""),
        "components": score_report.get("components", {}),
    }
    meta["finished_at"] = now
    meta["duration"] = now - float(meta.get("created_at", now))
    meta["final_counts"] = score_report.get("counts", {})
    meta["final_metrics"] = metrics
    write_json_file(STATE.get("dataset_meta_path", ""), meta)
    score_path = os.path.join(str(STATE.get("dataset_episode_dir", "")), "score.json")
    write_json_file(score_path, score_report)

    dataset_record_event(
        "episode_end",
        f"status={status}; success={success}; execution_success={execution_success}; score={score_report.get('score'):.1f}; reason={meta['failure_reason']}",
    )
    run_dir = ensure_auto_collect_run_dir()
    index_row = {
        "episode_id": meta.get("episode_id"),
        "episode_index": meta.get("episode_index"),
        "success": success,
        "status": status,
        "score": score_report.get("score"),
        "reason": meta.get("failure_reason", reason),
        "planner_version": meta.get("planner_version", PLANNER_VERSION),
        "quality_gate_version": meta.get("quality_gate_version", QUALITY_GATE_VERSION),
        "config_hash": meta.get("config_hash", ""),
        "episode_seed": meta.get("episode_seed", ""),
        "target_xyz": meta.get("target_xyz"),
        "unload_point_xyz": meta.get("unload_point_xyz"),
        "unload_landing_xyz": meta.get("unload_landing_xyz"),
        "unload_release_xyz": meta.get("unload_release_xyz"),
        "initial_pose_id": meta.get("initial_pose_id", ""),
        "q_initial_deg": meta.get("q_initial_deg"),
        "shared_plan_id": (meta.get("shared_dig_plan") or {}).get("plan_id", ""),
        "shared_plan_cost": (meta.get("shared_dig_plan") or {}).get("total_plan_cost"),
        "shared_plan_duration": (meta.get("shared_dig_plan") or {}).get("estimated_duration"),
        "shared_plan_stage_count": (meta.get("shared_dig_plan") or {}).get("stage_count"),
        "chosen_plan_id": (meta.get("chosen_plan") or {}).get("id", ""),
        "chosen_plan_score": (meta.get("chosen_plan") or {}).get("score"),
        "chosen_plan_unload_xyz": (meta.get("chosen_plan") or {}).get("unload_point_xyz"),
        "chosen_plan_unload_landing_xyz": (meta.get("chosen_plan") or {}).get("unload_landing_xyz"),
        "plan_debug": (meta.get("paths") or {}).get("plan_debug", ""),
        "trajectory": STATE.get("dataset_path", ""),
        "events": STATE.get("dataset_event_path", ""),
        "sand_metrics": STATE.get("dataset_sand_metrics_path", ""),
        "meta": STATE.get("dataset_meta_path", ""),
        "score_path": score_path,
        "debug_timeline": debug_timeline_path(create=False),
        "samples": metrics.get("samples"),
        "freeze_count": metrics.get("freeze_count"),
        "max_bucket_from_pile_particles": metrics.get("max_bucket_from_pile_particles"),
        "lift_bucket_from_pile_particles": metrics.get("lift_bucket_from_pile_particles"),
        "final_bin_from_pile_particles": metrics.get("final_bin_from_pile_particles"),
        "final_spill_from_pile_particles": metrics.get("final_spill_from_pile_particles"),
    }
    append_jsonl(os.path.join(run_dir, "episodes.jsonl"), index_row)
    if success:
        bucket_name = "successful_episodes.jsonl"
        append_jsonl(os.path.join(run_dir, "trainable_episodes.jsonl"), index_row)
    elif rejected:
        bucket_name = "rejected_episodes.jsonl"
    elif diagnostic:
        bucket_name = "diagnostic_episodes.jsonl"
    else:
        bucket_name = "failed_episodes.jsonl"
    append_jsonl(os.path.join(run_dir, bucket_name), index_row)
    debug_timeline_record(
        "EP_END",
        result=status,
        reason=meta.get("failure_reason", reason),
        data={
            "score": score_report.get("score"),
            "success": success,
            "execution_success": execution_success,
            "bucket": metrics.get("max_bucket_from_pile_particles"),
            "lift": metrics.get("lift_bucket_from_pile_particles"),
            "bin": metrics.get("final_bin_from_pile_particles"),
            "spill": metrics.get("final_spill_from_pile_particles"),
            "samples": metrics.get("samples"),
            "freeze": metrics.get("freeze_count"),
            "score_path": score_path,
            "plan_debug": (meta.get("paths") or {}).get("plan_debug", ""),
        },
        include_sand=True,
    )
    info_print(
        "[AUTO DATASET SCORE]",
        f"episode={meta.get('episode_id')}",
        f"status={status}",
        f"score={score_report.get('score'):.1f}",
        f"bucket={metrics.get('max_bucket_from_pile_particles')}",
        f"lift={metrics.get('lift_bucket_from_pile_particles')}",
        f"bin={metrics.get('final_bin_from_pile_particles')}",
        f"spill={metrics.get('final_spill_from_pile_particles')}",
        f"reason={meta.get('failure_reason', '')}",
    )

    if success:
        STATE["auto_collect_successes"] = int(STATE.get("auto_collect_successes", 0)) + 1
    elif rejected:
        STATE["auto_collect_rejections"] = int(STATE.get("auto_collect_rejections", 0)) + 1
    elif diagnostic:
        pass
    else:
        STATE["auto_collect_failures"] = int(STATE.get("auto_collect_failures", 0)) + 1
    STATE["auto_collect_last_result"] = f"{meta.get('episode_id')} status={status} score={score_report.get('score'):.1f} reason={meta.get('failure_reason', '')}"
    auto_collect_write_run_summary()
    STATE["dataset_recording"] = False
    return success


def auto_collect_sample_target(attempt_index, retry_index=0):
    k = max(1, int(attempt_index) * AUTO_COLLECT_MAX_PLAN_RETRIES + int(retry_index))
    theta = k * 2.399963229728653
    frac = (k * 0.6180339887498949) % 1.0
    radius = math.sqrt(0.05 + 0.45 * frac)
    ctx = task_scene_context()
    center = np.array(ctx["pile_center"], dtype=np.float32)
    pile_radius = np.array(ctx["pile_radius"], dtype=np.float32)
    rx = min(float(AUTO_COLLECT_TARGET_RADIUS_X), max(0.12, float(pile_radius[0]) * 0.52))
    ry = min(float(AUTO_COLLECT_TARGET_RADIUS_Y), max(0.12, float(pile_radius[1]) * 0.52))
    x = float(center[0]) + rx * radius * math.cos(theta)
    y = float(center[1]) + ry * radius * math.sin(theta)

    surface_z = 0.75
    api = get_sand_site_api()
    fn = None if api is None else api.get("height_fn")
    if callable(fn):
        try:
            surface_z = float(fn(x, y))
        except Exception as e:
            info_print("[WARN] [AUTO DATASET] sand height query failed:", type(e).__name__, e)
    depth = float(AUTO_COLLECT_TARGET_DEPTHS[k % len(AUTO_COLLECT_TARGET_DEPTHS)])
    z = auto_dig_target_z_from_surface(surface_z, depth)
    return np.array([x, y, z], dtype=np.float32)


def auto_dig_target_z_from_surface(surface_z, depth):
    surface_z = float(surface_z)
    depth = max(0.04, float(depth))
    dynamic_max_z = max(float(AUTO_COLLECT_TARGET_MAX_Z), surface_z + 0.05)
    return float(max(AUTO_COLLECT_TARGET_MIN_Z, min(dynamic_max_z, surface_z - depth)))


def sand_surface_height_for_auto_target(x, y, particles=None):
    surface_z = 0.75
    api = get_sand_site_api()
    fn = None if api is None else api.get("height_fn")
    if callable(fn):
        try:
            surface_z = float(fn(float(x), float(y)))
        except Exception:
            pass
    if particles is not None and len(particles) > 0:
        p = np.array(particles, dtype=np.float32)
        d = np.linalg.norm(p[:, :2] - np.array([[float(x), float(y)]], dtype=np.float32), axis=1)
        near = p[d <= 0.22]
        if len(near) > 0:
            surface_z = max(float(surface_z), float(np.percentile(near[:, 2], 85.0)))
    return float(surface_z)


def auto_collect_rank_dig_targets(attempt_index):
    ctx = task_scene_context()
    center = np.array(ctx["pile_center"], dtype=np.float32).reshape(-1)[:3]
    pile_radius = np.array(ctx["pile_radius"], dtype=np.float32).reshape(-1)[:2]
    rx = min(float(AUTO_COLLECT_TARGET_RADIUS_X), max(0.12, float(pile_radius[0]) * 0.56))
    ry = min(float(AUTO_COLLECT_TARGET_RADIUS_Y), max(0.12, float(pile_radius[1]) * 0.56))
    grid = max(3, int(AUTO_DIG_GRID_SIZE))
    values = np.linspace(-float(AUTO_DIG_CORE_NORM_MAX), float(AUTO_DIG_CORE_NORM_MAX), grid)
    particles = sand_particle_positions()
    q_ref = CTRL.q_cmd.copy()
    depth = float(AUTO_COLLECT_TARGET_DEPTHS[(max(1, int(attempt_index)) - 1) % len(AUTO_COLLECT_TARGET_DEPTHS)])
    rows = []

    for ux in values:
        for uy in values:
            norm = float(ux * ux + uy * uy)
            if norm > float(AUTO_DIG_CORE_NORM_MAX) * float(AUTO_DIG_CORE_NORM_MAX):
                continue
            x = float(center[0]) + rx * float(ux)
            y = float(center[1]) + ry * float(uy)
            surface_z = sand_surface_height_for_auto_target(x, y, particles=particles)
            z = auto_dig_target_z_from_surface(surface_z, depth)
            target = np.array([x, y, z], dtype=np.float32)
            ok, reason = validate_dig_target(target, hard_block=False)
            if not ok:
                rows.append({"target_xyz": vec_list(target, 3), "planned": False, "score": -1.0e9, "reason": reason})
                continue

            density_count = 0
            if particles is not None and len(particles) > 0:
                dxy = np.linalg.norm(particles[:, :2] - np.array([[x, y]], dtype=np.float32), axis=1)
                density_count = int(np.count_nonzero(dxy <= float(AUTO_DIG_DENSITY_RADIUS)))
                if density_count < int(AUTO_DIG_MIN_LOCAL_PARTICLES):
                    rows.append({
                        "target_xyz": vec_list(target, 3),
                        "surface_z": float(surface_z),
                        "target_depth": float(surface_z - z),
                        "density_count": density_count,
                        "planned": False,
                        "score": -1.0e9,
                        "reason": f"low_local_sand_density:{density_count}",
                    })
                    continue
            density_score = clamp01(density_count / 220.0)
            depth_score = clamp01((surface_z - z) / max(0.01, max(AUTO_COLLECT_TARGET_DEPTHS)))
            boundary_score = clamp01(1.0 - math.sqrt(norm) / max(0.01, float(AUTO_DIG_CORE_NORM_MAX)))
            swing_goal = target_to_swing_angle(target)
            swing_delta_deg_abs = abs(rad_to_deg(swing_delta(swing_goal, q_ref[CTRL.name_to_idx["swing"]])))
            motion_score = clamp01(1.0 - swing_delta_deg_abs / 180.0)

            ik_ok = True
            ik_reason = "ok"
            try:
                q_probe, ik_info = solve_priority_ik_to_target(
                    target,
                    q_seed=q_ref,
                    preferred_bucket_rad=deg_to_rad(-50.0),
                    bucket_motion_weight=0.35,
                    bucket_preference_weight=2.0,
                    min_world_z=GROUND_TOP_Z - 0.02,
                    accept_err=0.75,
                    end_effector="tip",
                    allow_end_below=True,
                    min_end_z=GROUND_TOP_Z - DIG_MAX_TIP_DEPTH,
                    phase_mode="approach_contact",
                    use_refinement=False,
                    bucket_candidate_span_deg=90.0,
                    bucket_candidate_count=9,
                )
                if q_probe is None:
                    ik_ok = False
                    ik_reason = str(ik_info)
            except Exception as e:
                ik_ok = False
                ik_reason = f"ik_check_failed:{type(e).__name__}:{e}"

            reach_score = 1.0 if ik_ok else 0.0
            score = (
                35.0 * reach_score
                + 24.0 * density_score
                + 24.0 * depth_score
                + 12.0 * motion_score
                + 12.0 * boundary_score
            )
            rows.append(
                {
                    "target_xyz": vec_list(target, 3),
                    "surface_z": float(surface_z),
                    "target_depth": float(surface_z - z),
                    "density_count": density_count,
                    "density_score": float(density_score),
                    "depth_score": float(depth_score),
                    "motion_score": float(motion_score),
                    "boundary_score": float(boundary_score),
                    "reach_score": float(reach_score),
                    "score": float(score),
                    "planned": bool(ik_ok),
                    "reason": ik_reason,
                }
            )

    rows.sort(key=lambda row: float(row.get("score", -1.0e9)), reverse=True)
    usable = [row for row in rows if bool(row.get("planned", False))]
    if not usable:
        surface_z = sand_surface_height_for_auto_target(float(center[0]), float(center[1]), particles=particles)
        fallback_depth = max(float(AUTO_COLLECT_TARGET_DEPTHS))
        fallback = np.array([
            float(center[0]),
            float(center[1]),
            auto_dig_target_z_from_surface(surface_z, fallback_depth),
        ], dtype=np.float32)
        fallback_density = 0
        if particles is not None and len(particles) > 0:
            dxy = particles[:, :2] - fallback[:2]
            fallback_density = int(np.count_nonzero(np.sum(dxy * dxy, axis=1) <= float(AUTO_DIG_DENSITY_RADIUS) ** 2))
        usable = [{
            "target_xyz": vec_list(fallback, 3),
            "surface_z": float(surface_z),
            "target_depth": float(surface_z - float(fallback[2])),
            "density_count": fallback_density,
            "score": 0.0,
            "planned": True,
            "reason": "fallback_center_deep_after_no_usable_grid",
        }]
        rows.insert(0, usable[0])
    STATE["last_auto_dig_target_scores"] = rows
    info_print(
        "[AUTO DIG TARGET SELECT]",
        f"candidates={len(rows)}",
        f"usable={len(usable)}",
        f"best={usable[0].get('target_xyz')}",
        f"score={fmt_optional(usable[0].get('score'))}",
        f"depth={fmt_optional(usable[0].get('target_depth'))}",
        f"density={usable[0].get('density_count')}",
        f"reason={usable[0].get('reason')}",
    )
    return usable[: max(1, int(AUTO_DIG_TOPK_TARGETS))]


def set_target_models_from_xyz(p):
    p = np.array(p, dtype=np.float32)
    set_target_xyz(float(p[0]), float(p[1]), float(p[2]))
    for axis, value in [("x", p[0]), ("y", p[1]), ("z", p[2])]:
        model = TARGET_MODELS.get(axis)
        if model is not None:
            try:
                model.set_value(float(value))
            except Exception:
                pass
    STATE["last_target_model_xyz"] = p.copy()
    STATE["last_target_sync_time"] = time.time()


def set_unload_models_from_xyz(p):
    p = np.array(p, dtype=np.float32).reshape(-1)[:3]
    if len(p) < 3:
        return
    set_manual_unload_point_xyz(float(p[0]), float(p[1]), float(p[2]), source="set_models", range_shape="circle")
    update_unload_models_only(p)


def update_unload_models_only(p):
    p = np.array(p, dtype=np.float32).reshape(-1)[:3]
    if len(p) < 3:
        return
    for axis, value in [("x", p[0]), ("y", p[1]), ("z", p[2])]:
        model = UNLOAD_MODELS.get(axis)
        if model is not None:
            try:
                model.set_value(float(value))
            except Exception:
                pass
    z_range = STATE.get("manual_unload_z_range")
    if z_range is not None:
        try:
            zr = np.array(z_range, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            zr = np.array([float(GROUND_TOP_Z), float(p[2])], dtype=np.float32)
    else:
        zr = np.array([float(GROUND_TOP_Z), float(p[2])], dtype=np.float32)
    if len(zr) < 2:
        zr = np.array([float(GROUND_TOP_Z), float(p[2])], dtype=np.float32)
    zr = np.array([min(float(zr[0]), float(zr[1])), max(float(zr[0]), float(zr[1]))], dtype=np.float32)
    for key, value in [("z_min", zr[0]), ("z_max", zr[1])]:
        model = UNLOAD_MODELS.get(key)
        if model is not None:
            try:
                model.set_value(float(value))
            except Exception:
                pass
    radius_model = UNLOAD_MODELS.get("r")
    if radius_model is not None:
        try:
            radius_model.set_value(float(manual_unload_radius()))
        except Exception:
            pass
    shrink_model = UNLOAD_MODELS.get("d")
    if shrink_model is not None:
        try:
            shrink_model.set_value(float(manual_unload_mesh_shrink_d()))
        except Exception:
            pass
    STATE["last_unload_model_xyz"] = p.copy()
    STATE["last_unload_model_z_range"] = zr.copy()
    STATE["last_unload_model_radius"] = float(manual_unload_radius())
    STATE["last_unload_model_shrink_d"] = float(manual_unload_mesh_shrink_d())
    STATE["last_unload_sync_time"] = time.time()


async def auto_collect_prepare_environment():
    was_recording = bool(STATE.get("dataset_recording", False))
    STATE["dataset_recording"] = False
    reset_dig_plan()
    if handle_timeline_stop_if_needed("auto_collect_prepare_start"):
        STATE["dataset_recording"] = was_recording
        return False

    if IK_MODEL is None:
        update_status("[AUTO DATASET] calibrating IK", force=True)
        await calibrate_ik()

    task_id = start_task("auto_collect_prepare")
    q_home = safe_home_q()
    ok = await set_joint_pose_direct_and_settle(
        q_home,
        label="auto_collect_home",
        mode="auto_collect_home_direct",
        settle_frames=DIRECT_HOME_SETTLE_FRAMES,
        task_id=task_id,
    )
    await step_updates(AUTO_COLLECT_PRE_RESET_SETTLE_FRAMES)

    policy = str(AUTO_COLLECT_SAND_RESET_POLICY).lower()
    ready_reset_done = bool(STATE.get("sand_site_stable_reset_done", False))
    should_reset_sand = (
        ok
        and callable((get_sand_site_api() or {}).get("reset"))
        and policy in ["once_per_run_after_home", "per_episode_after_home"]
        and (policy == "per_episode_after_home" or not bool(STATE.get("auto_collect_sand_reset_done", False)))
        and not (
            AUTO_COLLECT_REUSE_READY_SAND_RESET
            and ready_reset_done
            and policy == "once_per_run_after_home"
        )
    )
    if should_reset_sand:
        try:
            if handle_timeline_stop_if_needed("auto_collect_prepare_reset"):
                STATE["dataset_recording"] = was_recording
                return False
            update_status("[AUTO DATASET] resetting sand after home pose", force=True)
            info_print(
                "[AUTO DATASET RESET]",
                f"policy={AUTO_COLLECT_SAND_RESET_POLICY}",
                "order=calibrate_then_home_then_stable_reset",
                f"min_settle_frames={SAND_RESET_SETTLE_MIN_FRAMES}",
                f"max_settle_frames={SAND_RESET_SETTLE_MAX_FRAMES}",
            )
            reset_ok = await reset_sand_site_stably("auto_collect_prepare")
            STATE["auto_collect_sand_reset_done"] = bool(reset_ok)
        except Exception as e:
            info_print("[WARN] [AUTO DATASET] sand reset failed:", type(e).__name__, e)
    else:
        if ready_reset_done and AUTO_COLLECT_REUSE_READY_SAND_RESET and policy == "once_per_run_after_home":
            STATE["auto_collect_sand_reset_done"] = True
        info_print(
            "[AUTO DATASET RESET]",
            f"policy={AUTO_COLLECT_SAND_RESET_POLICY}",
            f"reset_done={STATE.get('auto_collect_sand_reset_done')}",
            f"ready_stable_reset_done={ready_reset_done}",
            f"home_ok={ok}",
            "action=skip",
        )
        await step_updates(20)

    STATE["dataset_recording"] = was_recording
    return bool(ok and task_alive(task_id))


async def auto_collect_find_plan(attempt_index):
    return await auto_dataset_collect.find_plan(runtime_module(), attempt_index)


async def auto_collect_one_episode():
    attempt = int(STATE.get("auto_collect_attempts", 0)) + 1
    STATE["auto_collect_attempts"] = attempt
    update_status(f"[AUTO DATASET] episode {attempt} prepare", force=True)

    prepared = await auto_collect_prepare_environment()
    if not prepared:
        target = auto_collect_sample_target(attempt, 0)
        meta = auto_collect_begin_episode(attempt, target, [{"prepared": False}], None, initial_info=None)
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=prepare_failed",
            "executed=False",
        )
        return auto_collect_finish_episode(meta, False, "preflight_failed/prepare_environment_failed")

    initial_ok, initial_info = await auto_collect_move_to_initial_pose(attempt)
    if not initial_ok:
        target = auto_collect_sample_target(attempt, 0)
        meta = auto_collect_begin_episode(
            attempt,
            target,
            [{"prepared": True, "initial_pose_ok": False, "initial_pose_id": initial_info.get("id", "")}],
            None,
            initial_info=initial_info,
        )
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=initial_pose_failed",
            "executed=False",
            f"initial_pose_id={initial_info.get('id', '')}",
        )
        return auto_collect_finish_episode(meta, False, "execution_failed/initial_pose_failed")

    action_ready, action_reason, _action_detail = await wait_for_articulation_action_ready(
        "auto_preflight",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_MAX_WAIT_FRAMES,
        record_failure=False,
    )
    if not action_ready:
        target = auto_collect_sample_target(attempt, 0)
        meta = auto_collect_begin_episode(
            attempt,
            target,
            [{"prepared": True, "initial_pose_ok": True, "action_channel_ready": False}],
            None,
            initial_info=initial_info,
        )
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=preflight_failed/action_channel_not_ready",
            "executed=False",
            f"reason={action_reason}",
        )
        return auto_collect_finish_episode(meta, False, f"preflight_failed/action_channel_not_ready:{action_reason}")

    preflight_ok, preflight, preflight_reason = auto_collect_preflight_report(
        target_successes=STATE.get("auto_collect_requested")
    )
    if not preflight_ok:
        target = auto_collect_sample_target(attempt, 0)
        meta = auto_collect_begin_episode(
            attempt,
            target,
            [{"prepared": True, "initial_pose_ok": True, "preflight": preflight}],
            None,
            initial_info=initial_info,
        )
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            f"result={preflight_reason}",
            "executed=False",
        )
        return auto_collect_finish_episode(meta, False, preflight_reason)

    target, seq, plan_attempts = await auto_collect_find_plan(attempt)
    if not seq:
        best_failure = STATE.get("dig_plan_best_failure")
        if isinstance(best_failure, dict):
            failed_stage = str(best_failure.get("failed_stage", "unknown"))
            failed_detail = str(best_failure.get("failure_reason", "no_valid_candidate"))
            plan_reason = f"planning_failed/{failed_stage}:{failed_detail}"
        else:
            plan_reason = "planning_failed/no_reachable_dig_candidate"
        update_status("[AUTO DATASET] plan failed for sampled target", force=True)
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=plan_failed",
            "executed=False",
            f"target={vec_list(target, 3)}",
            f"plan_retries={len(plan_attempts)}",
            f"reason={plan_reason}",
        )
        auto_collect_record_planning_diagnostic(
            attempt,
            target,
            plan_attempts,
            plan_reason,
            initial_info=initial_info,
        )
        return False

    meta = auto_collect_begin_episode(attempt, target, plan_attempts, seq, initial_info=initial_info)

    shared_plan = STATE.get("current_dig_plan")
    if isinstance(shared_plan, dict):
        shared_plan["episode_id"] = meta.get("episode_id", "")
        STATE["current_dig_plan"] = shared_plan
        info_print(
            "[AUTO EPISODE PLAN]",
            f"episode_id={meta.get('episode_id')}",
            f"dig_target={shared_plan.get('chosen_dig_target')}",
            f"unload_landing={shared_plan.get('chosen_unload_landing_point')}",
            f"unload_release={shared_plan.get('chosen_unload_release_point')}",
            f"plan_cost={shared_plan.get('total_plan_cost')}",
            f"estimated_duration={shared_plan.get('estimated_duration')}",
            f"stage_count={shared_plan.get('stage_count')}",
            f"planner_version={shared_plan.get('planner_version')}",
        )
    try:
        if current_trace_mode() == 0:
            set_trace_mode(2, reset_real=False)
        else:
            draw_trace(force=True)
    except Exception as e:
        info_print("[WARN] [AUTO DATASET] trace refresh failed:", type(e).__name__, e)

    update_status(f"[AUTO DATASET] executing episode {attempt} steps={len(seq)}", force=True)
    info_print(
        "[AUTO DATASET ATTEMPT]",
        f"attempt={attempt}",
        "result=executing",
        "executed=True",
        f"steps={len(seq)}",
        f"target={vec_list(target, 3)}",
        f"unload_point={vec_list(unload_bin_dump_point(), 3)}",
        f"unload_landing={vec_list(unload_bin_landing_point(), 3)}",
    )
    result = await execute_dig_target_ball(rebuild_plan=False, task_name=f"auto_collect_episode_{attempt:06d}")
    freezes = int(STATE.get("dataset_episode_freezes", 0))
    if not result:
        failure_reason = str(STATE.get("last_execution_failure_reason", "") or "execution_failed/stage_failed")
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=execution_failed",
            "executed=True",
            f"freezes={freezes}",
            f"reason={failure_reason}",
        )
        return auto_collect_finish_episode(meta, False, failure_reason)
    if freezes > 0:
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            "result=freeze_failed",
            "executed=True",
            f"freezes={freezes}",
        )
        return auto_collect_finish_episode(meta, False, f"execution_failed/freeze_detected:{freezes}")
    info_print(
        "[AUTO DATASET ATTEMPT]",
        f"attempt={attempt}",
        "result=execution_ok",
        "executed=True",
    )
    return auto_collect_finish_episode(meta, True, "ok")


async def auto_collect_loop(count):
    if bool(STATE.get("auto_collect_active", False)):
        update_status("[AUTO DATASET] already running", force=True)
        return

    previous_dataset = {
        "dataset_recording": bool(STATE.get("dataset_recording", False)),
        "dataset_path": STATE.get("dataset_path", "excavator_dataset.jsonl"),
        "dataset_event_path": STATE.get("dataset_event_path", "excavator_events.jsonl"),
        "dataset_meta_path": STATE.get("dataset_meta_path", ""),
        "dataset_sand_metrics_path": STATE.get("dataset_sand_metrics_path", ""),
    }
    STATE["auto_collect_active"] = True
    STATE["auto_collect_stop_requested"] = False
    STATE["auto_collect_requested"] = int(count)
    STATE["auto_collect_attempts"] = 0
    STATE["auto_collect_successes"] = 0
    STATE["auto_collect_failures"] = 0
    STATE["auto_collect_rejections"] = 0
    STATE["auto_collect_planning_diagnostics"] = 0
    STATE["auto_collect_run_dir"] = ""
    STATE["auto_collect_run_id"] = ""
    STATE["debug_timeline_path"] = ""
    STATE["auto_collect_sand_reset_done"] = False
    ensure_auto_collect_run_dir()
    update_status(auto_collect_status_text(), force=True)

    try:
        target_successes = max(1, int(count))
        max_attempts = max(target_successes, target_successes * AUTO_COLLECT_MAX_ATTEMPT_MULTIPLIER)
        info_print(
            "[AUTO DATASET LOOP]",
            "count_mode=successful_episodes",
            f"target_successes={target_successes}",
            f"max_attempts={max_attempts}",
            f"attempt_multiplier={AUTO_COLLECT_MAX_ATTEMPT_MULTIPLIER}",
        )
        consecutive_prepare_failed = 0
        while (
            int(STATE.get("auto_collect_successes", 0)) < target_successes
            and int(STATE.get("auto_collect_attempts", 0)) < max_attempts
        ):
            if STATE.get("auto_collect_stop_requested", False) or not STATE.get("running", False):
                break
            await auto_collect_one_episode()
            consecutive_prepare_failed = auto_dataset_collect.update_prepare_failure_streak(
                runtime_module(),
                consecutive_prepare_failed,
            )
            if consecutive_prepare_failed >= 3:
                info_print(
                    "[WARN] [AUTO DATASET]",
                    "stopping_after_repeated_prepare_failed",
                    f"count={consecutive_prepare_failed}",
                    f"last={STATE.get('auto_collect_last_result', '')}",
                )
                STATE["auto_collect_last_result"] = "blocked=repeated_prepare_failed"
                break
            await step_updates(AUTO_COLLECT_BETWEEN_EPISODE_FRAMES)
        if int(STATE.get("auto_collect_successes", 0)) < target_successes:
            info_print(
                "[WARN] [AUTO DATASET]",
                f"target_successes={target_successes}",
                f"actual_successes={STATE.get('auto_collect_successes')}",
                f"attempts={STATE.get('auto_collect_attempts')}",
                "reason=max_attempts_or_stop",
            )
    except Exception as e:
        info_print("[ERROR] [AUTO DATASET] loop failed:", type(e).__name__, e)
        STATE["auto_collect_last_result"] = f"loop_failed={type(e).__name__}: {e}"
    finally:
        debug_timeline_record(
            "RUN_END",
            result=str(STATE.get("auto_collect_last_result", "")),
            data={
                "requested": int(STATE.get("auto_collect_requested", 0)),
                "attempts": int(STATE.get("auto_collect_attempts", 0)),
                "success": int(STATE.get("auto_collect_successes", 0)),
                "rejected": int(STATE.get("auto_collect_rejections", 0)),
                "failed": int(STATE.get("auto_collect_failures", 0)),
                "planning": int(STATE.get("auto_collect_planning_diagnostics", 0)),
            },
            include_sand=True,
        )
        STATE["auto_collect_active"] = False
        STATE["auto_collect_stop_requested"] = False
        STATE["dataset_recording"] = previous_dataset["dataset_recording"]
        STATE["dataset_path"] = previous_dataset["dataset_path"]
        STATE["dataset_event_path"] = previous_dataset["dataset_event_path"]
        STATE["dataset_meta_path"] = previous_dataset["dataset_meta_path"]
        STATE["dataset_sand_metrics_path"] = previous_dataset["dataset_sand_metrics_path"]
        auto_collect_write_run_summary()
        update_status(auto_collect_status_text(), force=True)


def request_auto_collect(count):
    if bool(STATE.get("auto_collect_active", False)):
        update_status("[AUTO DATASET] already running", force=True)
        return
    task = register_async_task("auto_collect", auto_collect_loop(max(1, int(count))), replace=True)
    STATE["auto_collect_task"] = task


def stop_auto_collect():
    STATE["auto_collect_stop_requested"] = True
    STATE["trace_active_motion"] = None
    STATE["trace_no_plan_notice_shown"] = True
    STATE["trace_no_plan_notice_time"] = time.time()
    cancel_active_task("auto dataset stop requested")
    cancel_registered_task("auto_collect", reason="stop_auto_collect")
    auto_collect_write_run_summary()
    update_status("[AUTO DATASET] stop requested", force=True)


def read_jsonl_rows(path):
    rows = []
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def read_json_file_or_none(path):
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def latest_successful_record_path():
    candidates = []
    run_dir = str(STATE.get("auto_collect_run_dir", ""))
    if run_dir:
        candidates.append(os.path.join(run_dir, "trainable_episodes.jsonl"))
        candidates.append(os.path.join(run_dir, "successful_episodes.jsonl"))
    root = AUTO_COLLECT_DATASET_ROOT
    try:
        if os.path.isdir(root):
            run_dirs = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
            run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for rd in run_dirs:
                candidates.append(os.path.join(rd, "trainable_episodes.jsonl"))
                candidates.append(os.path.join(rd, "successful_episodes.jsonl"))
    except Exception:
        pass

    for index_path in candidates:
        rows = read_jsonl_rows(index_path)
        if not rows:
            continue
        row = rows[-1]
        traj = row.get("trajectory")
        if traj and os.path.exists(str(traj)):
            return str(traj)
    return ""


def q_from_replay_sample(sample):
    for key in ["obs.q_cmd", "obs.q", "goal.q", "observation.joint_command", "observation.joint_position", "planned.q_goal"]:
        value = sample.get(key)
        if value is not None:
            arr = np.array(value, dtype=np.float32).reshape(-1)
            if len(arr) >= 4:
                return arr[:4].copy()
    return None


async def replay_record(path=None):
    if STATE.get("replay_active", False):
        update_status("[REPLAY] already running", force=True)
        return False
    if path is None or not str(path):
        path = latest_successful_record_path()
    if not path or not os.path.exists(str(path)):
        update_status("[REPLAY] no successful record found", force=True)
        return False

    rows = read_jsonl_rows(path)
    rows = [row for row in rows if q_from_replay_sample(row) is not None]
    if not rows:
        update_status("[REPLAY] record has no joint samples", force=True)
        return False

    was_recording = bool(STATE.get("dataset_recording", False))
    STATE["dataset_recording"] = False
    STATE["replay_active"] = True
    STATE["replay_last_path"] = str(path)
    task_id = start_task("replay_record")
    update_status(f"[REPLAY] start samples={len(rows)} path={path}", force=True)
    info_print("[REPLAY]", f"path={path}", f"samples={len(rows)}")

    try:
        ready, reason, _detail = await wait_for_articulation_action_ready(
            "replay",
            min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
            max_frames=ACTION_READY_MAX_WAIT_FRAMES,
            record_failure=True,
        )
        if not ready:
            update_status(f"[REPLAY BLOCKED] action_channel_not_ready; {reason}", force=True)
            return False

        meta = read_json_file_or_none(os.path.join(os.path.dirname(str(path)), "meta.json")) or {}
        first_target = meta.get("target_xyz") or rows[0].get("target_xyz") or rows[0].get("target")
        if first_target is not None:
            set_target_models_from_xyz(first_target)
        replay_unload = (
            meta.get("unload_landing_xyz")
            or meta.get("unload_point_xyz")
            or (meta.get("chosen_plan") or {}).get("unload_landing_xyz")
            or (meta.get("chosen_plan") or {}).get("unload_point_xyz")
        )
        if replay_unload is not None:
            ensure_unload_marker(replay_unload, label="replay_record")

        prev_t = None
        for row in rows:
            if not task_alive(task_id) or STATE.get("auto_collect_stop_requested", False):
                break
            q = q_from_replay_sample(row)
            if q is None:
                continue
            timestamp = row.get("t_episode", row.get("timestamp"))
            if prev_t is None:
                dt = 1.0 / CONTROL_HZ
            else:
                dt = max(1.0 / CONTROL_HZ, min(0.25, safe_float(timestamp, prev_t) - safe_float(prev_t, timestamp)))
            prev_t = timestamp
            ok, reason = CTRL.apply_target_direct(q, mode="replay")
            if not ok:
                update_status(f"[REPLAY BLOCKED] {reason}", force=True)
                return False
            await step_updates(max(1, int(dt * 60.0)))
        update_status("[REPLAY DONE]", force=True)
        return True
    finally:
        STATE["dataset_recording"] = was_recording
        STATE["replay_active"] = False


def request_replay_latest_record():
    register_async_task("replay", replay_record(), replace=True)


def q_real_near_command(q_real, q_cmd):
    q_real = np.array(q_real, dtype=np.float32).copy()
    q_cmd = np.array(q_cmd, dtype=np.float32)
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        q_real[swing_idx] = swing_target_near(q_real[swing_idx], q_cmd[swing_idx])
    except Exception:
        pass
    return q_real


def q_delta_abs_deg(q_a, q_b):
    q_a = np.array(q_a, dtype=np.float32)
    q_b = np.array(q_b, dtype=np.float32)
    out = []
    for i, name in enumerate(DOF_ORDER):
        if name == "swing":
            d = swing_delta(q_a[i], q_b[i])
        else:
            d = float(q_a[i] - q_b[i])
        out.append(abs(rad_to_deg(d)))
    return out


def plan_joint_motion_metrics(q_to, q_from, duration=0.0):
    return ik_calculation.joint_motion_metrics(runtime_module(), q_to, q_from, duration=duration)


def plan_path_penalty(q_start, q_goal, mode):
    return ik_calculation.path_penalty(runtime_module(), q_start, q_goal, mode)


def stop_manual_motion_after_freeze(q_real, detail=""):
    if not MANUAL_FREEZE_STOP_ENABLED:
        return
    if not bool(STATE.get("manual_joint_active", False)):
        return

    q_hold = np.array(q_real, dtype=np.float32).copy()
    q_hold = CTRL.clip_limits(q_hold)
    CTRL.q_cmd = q_hold.copy()
    CTRL.q_safe = q_hold.copy()
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["freeze_candidate_since"] = 0.0
    STATE["last_action_q"] = None
    STATE["last_action_unclipped_q"] = None
    STATE["last_action_time"] = 0.0

    try:
        CTRL.send_action(q_hold, mode="manual_freeze_hold")
    except Exception as e:
        info_print("[WARN] manual freeze hold action failed:", e)
    STATE["last_action_q"] = None
    STATE["last_action_unclipped_q"] = None
    STATE["last_action_time"] = 0.0
    STATE["last_action_mode"] = "manual_freeze_hold"

    try:
        hold_manual_ui_sync()
        sync_sliders_from_real_q(force=True)
    except Exception:
        pass

    update_status("[MANUAL FREEZE STOP] physical motion stalled; holding current real joint state", force=True)
    info_print("[MANUAL FREEZE STOP]", f"hold_deg={q_deg_values(q_hold, wrap_swing_for_display=True)}", f"detail={detail}")


def stop_auto_motion_after_freeze(q_real, detail="", action_mode=""):
    if not AUTO_FREEZE_STOP_ENABLED:
        return

    action_mode = str(action_mode)
    if action_mode.startswith("manual") or action_mode in ["idle", "auto_freeze_hold"]:
        return

    q_hold = np.array(q_real, dtype=np.float32).copy()
    q_hold = CTRL.clip_limits(q_hold)
    CTRL.q_cmd = q_hold.copy()
    CTRL.q_safe = q_hold.copy()
    STATE["follow"] = False
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["freeze_candidate_since"] = 0.0
    STATE["active_task_id"] = int(STATE.get("active_task_id", 0)) + 1

    try:
        CTRL.send_action(q_hold, mode="auto_freeze_hold")
    except Exception as e:
        info_print("[WARN] auto freeze hold action failed:", e)
    STATE["last_action_q"] = None
    STATE["last_action_unclipped_q"] = None
    STATE["last_action_time"] = 0.0
    STATE["last_action_mode"] = "auto_freeze_hold"

    update_status(f"[AUTO FREEZE STOP] {action_mode} stalled; holding current real joint state", force=True)
    info_print(
        "[AUTO FREEZE STOP]",
        f"mode={action_mode}",
        f"hold_deg={q_deg_values(q_hold, wrap_swing_for_display=True)}",
        f"detail={detail}",
    )


def freeze_contact_detail(mode="freeze"):
    details = []
    try:
        report = phase_ground_report(mode)
        for label, key, margin in [
            ("base", "base_min", -0.02),
            ("bucket", "bucket_min", -0.02),
        ]:
            z = report.get(key)
            if z is not None and float(z) < GROUND_TOP_Z + float(margin):
                details.append(f"{label}_below_ground={float(z):.3f}")
                if label == "bucket" and float(z) < GROUND_TOP_Z - MANUAL_RECOVERY_BUCKET_PENETRATION_Z:
                    details.append("bucket_penetration_locks_swing=True")
    except Exception as e:
        details.append(f"ground_report_failed={type(e).__name__}")

    try:
        full_min, _, full_source = robot_full_collision_bbox()
        if full_min is not None and float(full_min[2]) < GROUND_TOP_Z - 0.02:
            details.append(f"full_collision_min_z={float(full_min[2]):.3f} source={full_source}")
    except Exception:
        pass

    try:
        obstacle_detail = actual_rigid_obstacle_contact_detail()
        if obstacle_detail:
            details.append(obstacle_detail)
    except Exception:
        pass

    if sand_site_active():
        details.append("sand_site_active=True")

    if not details:
        details.append("no_below_ground_body_detected")
    return "; ".join(details)


def fmt_vec3(v):
    if v is None:
        return "None"
    try:
        return "(" + ",".join(f"{float(x):.3f}" for x in v[:3]) + ")"
    except Exception:
        return "None"


def swing_drive_detail():
    try:
        prim = get_prim(JOINT_PATHS.get("swing"))
        drive = get_joint_drive_api(prim)
        if drive is None:
            return "swing_drive=None"
        return (
            "swing_drive="
            f"stiffness:{read_drive_attr(drive, 'GetStiffnessAttr')},"
            f"damping:{read_drive_attr(drive, 'GetDampingAttr')},"
            f"max_force:{read_drive_attr(drive, 'GetMaxForceAttr')}"
        )
    except Exception as e:
        return f"swing_drive_diag_failed={type(e).__name__}"


def link_min_z_detail():
    parts = []
    for name in ["base_link", "swing_link", "boom_link", "arm_link", "bucket_link"]:
        try:
            path = LINK_PATHS.get(name)
            z = bbox_min_z(path) if path else None
            parts.append(f"{name}_min_z={fmt_optional(z)}")
        except Exception:
            parts.append(f"{name}_min_z=err")
    return "; ".join(parts)


def link_lowest_collision_detail():
    parts = []
    for name in ["base_link", "swing_link", "boom_link", "arm_link", "bucket_link"]:
        try:
            parts.append(collision_lowest_detail(name, limit=1))
        except Exception as e:
            parts.append(f"{name}_lowest_failed={type(e).__name__}")
    return "; ".join(parts)


def support_clearance_detail():
    try:
        support_min, support_max, support_source = robot_support_bbox()
        if support_min is None:
            return f"support_clearance=None source={support_source}"
        cx = 0.5 * (float(support_min[0]) + float(support_max[0]))
        cy = 0.5 * (float(support_min[1]) + float(support_max[1]))
        ground_top, ground_path = support_ground_top_at_xy(cx, cy)
        clearance = float(support_min[2]) - float(ground_top)
        target = float(ROBOT_GROUND_TARGET_CLEARANCE_Z)
        below_target = max(0.0, target - clearance)
        return (
            f"support_clearance={clearance:.3f} "
            f"target_support_clearance={target:.3f} "
            f"support_below_operational_target={below_target:.3f} "
            f"support_min={fmt_vec3(support_min)} support_max={fmt_vec3(support_max)} "
            f"ground={ground_path} ground_top={ground_top:.3f} source={support_source}"
        )
    except Exception as e:
        return f"support_clearance_failed={type(e).__name__}"


def swing_obstacle_prediction_detail(q_cmd, q_real):
    try:
        ok, reason, sample, report = path_obstacle_check(q_real, q_cmd, "swing_diag", samples=12)
        if ok:
            return "swing_obstacle_prediction=clear"
        return "swing_obstacle_prediction=" + format_obstacle_report("swing_diag", report, reason)
    except Exception as e:
        return f"swing_obstacle_prediction_failed={type(e).__name__}"


def swing_freeze_detail(q_cmd, q_real):
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        swing_err = rad_to_deg(swing_delta(float(q_cmd[swing_idx]), float(q_real[swing_idx])))
    except Exception:
        swing_err = 0.0
    return "; ".join([
        f"swing_err_signed={swing_err:.2f}deg",
        swing_drive_detail(),
        link_min_z_detail(),
        link_lowest_collision_detail(),
        support_clearance_detail(),
        swing_obstacle_prediction_detail(q_cmd, q_real),
    ])


def log_freeze(reason, mode="", q_cmd=None, q_real=None, extra="", force=False):
    now = time.time()
    reason = str(reason)
    mode = str(mode)
    extra = str(extra)
    signature_extra = extra[:160]
    if reason == "suspected_collision_or_physics_constraint" and "blocked_joints=" in extra:
        try:
            blocked_text = extra.split("blocked_joints=", 1)[1].split(";", 1)[0]
            names = []
            for item in blocked_text.strip("[]").split(","):
                name = item.strip().strip("'").strip('"').split(":", 1)[0].strip()
                if name:
                    names.append(name)
            signature_extra = "blocked_joints=" + ",".join(names)
        except Exception:
            signature_extra = extra[:160]
    signature = f"{reason}|{mode}|{signature_extra}"
    if (
        not force
        and signature == STATE.get("freeze_last_signature")
        and now - float(STATE.get("freeze_last_print_time", 0.0)) < FREEZE_PRINT_INTERVAL
    ):
        return

    args = [
        "[FREEZE]",
        f"reason={reason}",
        f"mode={mode}",
        f"active_task={STATE.get('active_task_name')}",
    ]
    if q_cmd is not None:
        args.append(f"cmd_deg={q_deg_values(q_cmd)}")
    if q_real is not None:
        args.append(f"real_deg={q_deg_values(q_real)}")
    if extra:
        args.append(f"detail={extra}")
    info_print(*args)
    STATE["dataset_episode_freezes"] = int(STATE.get("dataset_episode_freezes", 0)) + 1
    debug_timeline_record(
        "FREEZE",
        stage=mode,
        result="stalled",
        reason=f"{reason}; {extra}",
        q_cmd=q_cmd,
        q_real=q_real,
        include_sand=True,
    )
    dataset_record_event("freeze", f"reason={reason}; mode={mode}; {extra[:500]}")
    STATE["freeze_last_signature"] = signature
    STATE["freeze_last_print_time"] = now


def check_freeze_state(label="loop"):
    q_cmd = STATE.get("last_action_q")
    action_time = float(STATE.get("last_action_time", 0.0))
    if q_cmd is None or ROBOT is None or JOINT_INDICES is None or action_time <= 0.0:
        return
    if not robot_joint_read_ready():
        return

    now = time.time()
    action_mode = str(STATE.get("last_action_mode", label))
    action_mode_l = action_mode.lower()
    if "direct" in action_mode_l or "home_direct" in action_mode_l or "initial_direct" in action_mode_l:
        STATE["freeze_candidate_since"] = 0.0
        return

    try:
        q_cmd = np.array(q_cmd, dtype=np.float32)
        q_real_raw = get_real_joint_positions()
        q_real = q_real_near_command(q_real_raw, q_cmd)
    except Exception as e:
        log_freeze("real_joint_read_failed", mode=label, extra=repr(e), force=False)
        return

    cmd_err_deg = q_delta_abs_deg(q_cmd, q_real)
    max_err = max(cmd_err_deg) if cmd_err_deg else 0.0
    if max_err < FREEZE_CMD_ERR_DEG:
        STATE["freeze_candidate_since"] = 0.0
        STATE["freeze_last_real_q"] = q_real.copy()
        STATE["freeze_last_real_time"] = now
        return

    last_real = STATE.get("freeze_last_real_q")
    if last_real is None:
        STATE["freeze_last_real_q"] = q_real.copy()
        STATE["freeze_last_real_time"] = now
        return

    real_step_deg = q_delta_abs_deg(q_real, np.array(last_real, dtype=np.float32))
    max_step = max(real_step_deg) if real_step_deg else 0.0
    STATE["freeze_last_real_q"] = q_real.copy()
    STATE["freeze_last_real_time"] = now

    if max_step > FREEZE_STALL_MOTION_DEG:
        STATE["freeze_candidate_since"] = 0.0
        return

    since = float(STATE.get("freeze_candidate_since", 0.0))
    if since <= 0.0:
        STATE["freeze_candidate_since"] = now
        return
    candidate_age = now - since

    blocked_names = [name for i, name in enumerate(DOF_ORDER) if cmd_err_deg[i] >= FREEZE_CMD_ERR_DEG]
    suppress_contact_freeze, contact_report = sand_contact_should_suppress_freeze(
        action_mode,
        blocked_names,
        cmd_err_deg,
        q_cmd,
        q_real,
    )
    if suppress_contact_freeze:
        return
    action_mode_l = action_mode.lower()
    if "swing" in blocked_names:
        min_duration = FREEZE_SWING_ONLY_MIN_DURATION
    elif "bucket" in blocked_names and ("cut" in action_mode_l or "dig" in action_mode_l or "pull" in action_mode_l):
        min_duration = FREEZE_BUCKET_CUT_MIN_DURATION
    else:
        min_duration = FREEZE_MIN_DURATION
    if candidate_age < min_duration:
        return

    blocked = [f"{name}:{cmd_err_deg[DOF_ORDER.index(name)]:.2f}deg" for name in blocked_names]
    detail = (
        f"command accepted but real joints stalled; "
        f"blocked_joints={blocked}; "
        f"max_cmd_err={max_err:.2f}deg; "
        f"max_real_step={max_step:.3f}deg; "
        f"last_action_age={now - action_time:.2f}s; "
        f"candidate_age={candidate_age:.2f}s; "
        f"{freeze_contact_detail(label)}"
    )
    if contact_report is not None:
        detail += (
            f"; sand_contact_progress_age={float(contact_report.get('progress_age', 0.0)):.2f}s"
            f"; sand_bucket_total=+{int(contact_report.get('total_bucket_delta', 0))}"
            f"; sand_pile_total=-{int(contact_report.get('total_pile_delta', 0))}"
            f"; sand_tip_total={float(contact_report.get('total_tip_delta', 0.0) or 0.0):.3f}m"
        )
    if "swing" in blocked_names:
        detail += "; " + swing_freeze_detail(q_cmd, q_real)
    log_freeze(
        "suspected_collision_or_physics_constraint",
        mode=action_mode,
        q_cmd=q_cmd,
        q_real=q_real,
        extra=detail,
        force=False,
    )
    set_execution_failure_reason(f"execution_failed/freeze_detected:{action_mode}:{detail}")
    if action_mode.startswith("manual"):
        stop_manual_motion_after_freeze(q_real, detail=detail)
    else:
        stop_auto_motion_after_freeze(q_real, detail=detail, action_mode=action_mode)


def notify_sand_site_tool_sample(stage_name):
    return


# ============================================================
# Setup
# ============================================================

def setup_paths():
    global ROBOT_ROOT, ROBOT_BASE, BUCKET_LINK, JOINT_PATHS, LINK_PATHS

    ROBOT_ROOT, ROBOT_BASE = find_robot_paths()

    LINK_PATHS = {
        "base_link": f"{ROBOT_BASE}/base_link",
        "swing_link": f"{ROBOT_BASE}/swing_link",
        "boom_link": f"{ROBOT_BASE}/boom_link",
        "arm_link": f"{ROBOT_BASE}/arm_link",
        "bucket_link": f"{ROBOT_BASE}/bucket_link",
    }

    JOINT_PATHS = {
        "swing": f"{ROBOT_BASE}/joints/swing",
        "boom": f"{ROBOT_BASE}/joints/boom",
        "arm": f"{ROBOT_BASE}/joints/arm",
        "bucket": f"{ROBOT_BASE}/joints/bucket",
    }

    BUCKET_LINK = LINK_PATHS["bucket_link"]

    info_print("ROBOT_ROOT =", ROBOT_ROOT)
    info_print("ROBOT_BASE =", ROBOT_BASE)


def read_imported_limit_deg(joint_name):
    prim = get_prim(JOINT_PATHS[joint_name])
    rj = UsdPhysics.RevoluteJoint(prim)

    raw_lo = rj.GetLowerLimitAttr().Get()
    raw_hi = rj.GetUpperLimitAttr().Get()
    if joint_name == "swing" and (raw_lo is None or raw_hi is None):
        return -180.0, 180.0

    lo = safe_float(raw_lo)
    hi = safe_float(raw_hi)

    return min(lo, hi), max(lo, hi)


def compute_limits():
    global FINAL_LIMITS_DEG, FINAL_LIMITS_RAD

    FINAL_LIMITS_DEG = {}
    FINAL_LIMITS_RAD = {}

    for name in ["swing", "boom", "arm", "bucket"]:
        imp_lo, imp_hi = read_imported_limit_deg(name)
        des_lo, des_hi = DESIRED_LIMITS_DEG[name]

        if LIMIT_POLICY == "respect_imported":
            lo = max(imp_lo, des_lo)
            hi = min(imp_hi, des_hi)
            if lo >= hi:
                lo, hi = imp_lo, imp_hi
        else:
            lo, hi = des_lo, des_hi

        FINAL_LIMITS_DEG[name] = (lo, hi)
        FINAL_LIMITS_RAD[name] = (deg_to_rad(lo), deg_to_rad(hi))

        info_print(f"[LIMIT] {name}: imported=({imp_lo:.2f},{imp_hi:.2f}) final=({lo:.2f},{hi:.2f})")


def ensure_joint_limits_are_valid():
    for name in ["swing", "boom", "arm", "bucket"]:
        prim = get_prim(JOINT_PATHS[name])
        if not prim or not prim.IsValid():
            info_print("[WARN] joint prim unavailable for limit validation:", name, JOINT_PATHS.get(name))
            continue

        rj = UsdPhysics.RevoluteJoint(prim)
        lo_attr = rj.GetLowerLimitAttr()
        hi_attr = rj.GetUpperLimitAttr()
        raw_lo = lo_attr.Get()
        raw_hi = hi_attr.Get()

        valid = True
        try:
            lo = float(raw_lo)
            hi = float(raw_hi)
            valid = math.isfinite(lo) and math.isfinite(hi) and lo < hi
        except Exception:
            valid = False

        if valid:
            info_print(f"[JOINT LIMIT OK] {name}: lower={lo:.3f} upper={hi:.3f}")
            continue

        lo, hi = DESIRED_LIMITS_DEG[name]
        lo_attr.Set(float(lo))
        hi_attr.Set(float(hi))
        info_print(
            f"[JOINT LIMIT FIX] {name}: invalid imported lower/upper=({raw_lo},{raw_hi}) "
            f"set=({lo:.3f},{hi:.3f})"
        )


def get_joint_drive_api(joint_prim):
    if joint_prim is None or not joint_prim.IsValid():
        return None
    try:
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive and drive.GetPrim() and drive.GetPrim().IsValid():
            return drive
    except Exception:
        pass
    try:
        return UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    except Exception:
        return None


def set_drive_float_attr(drive, get_name, create_name, value):
    try:
        attr = getattr(drive, get_name)()
        if not attr.IsValid():
            attr = getattr(drive, create_name)()
        attr.Set(float(value))
        return True
    except Exception:
        return False


def read_drive_attr(drive, get_name):
    try:
        attr = getattr(drive, get_name)()
        if attr.IsValid():
            return attr.Get()
    except Exception:
        pass
    return None


def configure_joint_drive_gains():
    for name, gains in JOINT_DRIVE_GAINS.items():
        prim = get_prim(JOINT_PATHS.get(name))
        if not prim or not prim.IsValid():
            info_print("[WARN] joint prim unavailable for drive gains:", name, JOINT_PATHS.get(name))
            continue

        drive = get_joint_drive_api(prim)
        if drive is None:
            info_print("[WARN] drive API unavailable:", name, prim.GetPath())
            continue

        set_drive_float_attr(drive, "GetStiffnessAttr", "CreateStiffnessAttr", gains["stiffness"])
        set_drive_float_attr(drive, "GetDampingAttr", "CreateDampingAttr", gains["damping"])
        set_drive_float_attr(drive, "GetMaxForceAttr", "CreateMaxForceAttr", gains["max_force"])

        info_print(
            "[DRIVE GAIN]",
            f"{name}:",
            f"stiffness={read_drive_attr(drive, 'GetStiffnessAttr')}",
            f"damping={read_drive_attr(drive, 'GetDampingAttr')}",
            f"max_force={read_drive_attr(drive, 'GetMaxForceAttr')}",
            "control=ArticulationAction_only",
        )


def configure_joints():
    ensure_joint_limits_are_valid()
    compute_limits()
    configure_joint_drive_gains()
    apply_speed_to_physx_joint_limits()


def make_parking_ground():
    support_min, support_max, source = robot_support_bbox()
    if support_min is not None and support_max is not None:
        min_x = float(support_min[0])
        min_y = float(support_min[1])
        max_x = float(support_max[0])
        max_y = float(support_max[1])
        width_x = max_x - min_x
        width_y = max_y - min_y
        if (
            not all(math.isfinite(v) for v in [min_x, min_y, max_x, max_y, width_x, width_y])
            or width_x < 0.0
            or width_y < 0.0
            or width_x > PARKING_GROUND_BASE_MAX_SIZE_XY
            or width_y > PARKING_GROUND_BASE_MAX_SIZE_XY
        ):
            root_pos = get_prim_translation(ROBOT_BASE)
            cx, cy = float(root_pos[0]), float(root_pos[1])
            base_sx = base_sy = PARKING_GROUND_MIN_SIZE_XY
            source = f"{source}_invalid_bbox_fallback"
        else:
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)
            base_sx = min(
                PARKING_GROUND_BASE_MAX_SIZE_XY,
                max(PARKING_GROUND_MIN_SIZE_XY, width_x + 2.0 * PARKING_GROUND_MARGIN_XY),
            )
            base_sy = min(
                PARKING_GROUND_BASE_MAX_SIZE_XY,
                max(PARKING_GROUND_MIN_SIZE_XY, width_y + 2.0 * PARKING_GROUND_MARGIN_XY),
            )
    else:
        root_pos = get_prim_translation(ROBOT_BASE)
        cx, cy = float(root_pos[0]), float(root_pos[1])
        base_sx = base_sy = PARKING_GROUND_MIN_SIZE_XY
        source = "root_fallback"

    sx = min(PARKING_GROUND_MAX_SIZE_XY, base_sx * PARKING_GROUND_SIZE_MULTIPLIER_XY)
    sy = min(PARKING_GROUND_MAX_SIZE_XY, base_sy * PARKING_GROUND_SIZE_MULTIPLIER_XY)

    path = f"{CONTROL_ROOT}/ParkingGround"
    prim = make_cube(
        path,
        translate=(cx, cy, GROUND_TOP_Z - 0.5 * GROUND_THICKNESS),
        scale=(sx, sy, GROUND_THICKNESS),
        color=(0.34, 0.34, 0.34),
        collision=True,
    )
    info_print(
        "[PARKING GROUND]",
        f"path={path}",
        f"source={source}",
        f"center=({cx:.3f},{cy:.3f})",
        f"base_size=({base_sx:.3f},{base_sy:.3f})",
        f"multiplier={PARKING_GROUND_SIZE_MULTIPLIER_XY:.1f}",
        f"size=({sx:.3f},{sy:.3f},{GROUND_THICKNESS:.3f})",
    )
    return prim


def support_ground_top_at_xy(x, y):
    candidates = [
        f"{CONTROL_ROOT}/ParkingGround",
        f"{CONTROL_ROOT}/Ground",
        "/SandSite/HardBaseCollision",
        "/World/SandSite/HardBaseCollision",
    ]
    best = None
    for path in candidates:
        mn, mx = bbox_min_max(path)
        if mn is None:
            continue
        if float(mn[0]) <= float(x) <= float(mx[0]) and float(mn[1]) <= float(y) <= float(mx[1]):
            top = float(mx[2])
            if best is None or top > best[0]:
                best = (top, path)
    if best is not None:
        return best
    return GROUND_TOP_Z, "fallback_ground_z"


def park_robot_on_support_ground(label=""):
    prim = get_prim(ROBOT_BASE)
    if not prim or not prim.IsValid():
        info_print("[GROUND PARK] robot root unavailable", f"label={label}")
        return False

    current = get_prim_translation(ROBOT_BASE)
    lifted_z = float(current[2]) + ROBOT_PARK_SPAWN_LIFT_Z
    set_xform(prim, translate=(float(current[0]), float(current[1]), lifted_z))

    support_min, support_max, source = robot_support_bbox()
    if support_min is None or support_max is None:
        info_print("[GROUND PARK] support bbox unavailable", f"label={label}", f"source={source}")
        set_xform(prim, translate=(float(current[0]), float(current[1]), float(current[2])))
        return False

    full_min, full_max, full_source = robot_full_collision_bbox()
    if full_min is None or full_max is None:
        full_min = support_min
        full_max = support_max
        full_source = f"{source}_fallback_for_full"

    support_cx = 0.5 * (float(support_min[0]) + float(support_max[0]))
    support_cy = 0.5 * (float(support_min[1]) + float(support_max[1]))
    ground_top, ground_path = support_ground_top_at_xy(support_cx, support_cy)
    target_support_min_z = float(ground_top) + ROBOT_PARK_CONTACT_SKIN
    target_full_min_z = float(ground_top) + ROBOT_PARK_FULL_COLLISION_CLEARANCE_Z
    dz_for_support = target_support_min_z - float(support_min[2])
    full_safety_enabled = full_source == "collision_bbox"
    dz_for_full_clearance = target_full_min_z - float(full_min[2]) if full_safety_enabled else dz_for_support
    dz_down = max(dz_for_support, dz_for_full_clearance)
    if not all(math.isfinite(v) for v in [support_cx, support_cy, ground_top, target_support_min_z, target_full_min_z, dz_for_support, dz_for_full_clearance, dz_down]):
        info_print(
            "[WARN] [GROUND PARK] invalid parking computation",
            f"label={label}",
            f"support_source={source}",
            f"full_source={full_source}",
            "height_basis=fixed_base_operational_clearance",
            f"ground={ground_path}",
            f"support_min={support_min}",
            f"support_max={support_max}",
            f"full_min={full_min}",
            f"full_max={full_max}",
        )
        set_xform(prim, translate=(float(current[0]), float(current[1]), float(current[2])))
        return False
    if abs(dz_down) > ROBOT_PARK_SPAWN_LIFT_Z + 2.0:
        info_print(
            "[WARN] [GROUND PARK] parking dz too large; using original root pose",
            f"label={label}",
            f"dz_to_contact={dz_down:.3f}",
            f"support_source={source}",
            f"ground={ground_path}",
        )
        set_xform(prim, translate=(float(current[0]), float(current[1]), float(current[2])))
        return False

    before_min_z = float(support_min[2])
    before_full_min_z = float(full_min[2])
    target_z = lifted_z + dz_down
    if ROBOT_PARK_NEVER_LOWER:
        target_z = max(float(current[2]), float(target_z))
    set_xform(prim, translate=(float(current[0]), float(current[1]), target_z))
    sync_articulation_pose_from_usd(label)

    final_min, final_max, final_source = robot_support_bbox()
    final_full_min, final_full_max, final_full_source = robot_full_collision_bbox()
    final_min_z = None if final_min is None else float(final_min[2])
    final_full_min_z = None if final_full_min is None else float(final_full_min[2])
    final_clearance = None if final_min_z is None else final_min_z - float(ground_top)
    final_full_clearance = None if final_full_min_z is None else final_full_min_z - float(ground_top)
    info_print(
        "[GROUND PARK]",
        f"label={label}",
        f"support_source={source}",
        f"full_source={full_source}",
        "height_basis=fixed_base_operational_clearance",
        f"ground={ground_path}",
        f"ground_top={ground_top:.3f}",
        f"spawn_lift={ROBOT_PARK_SPAWN_LIFT_Z:.3f}",
        f"before_support_min_z={before_min_z:.3f}",
        f"before_full_min_z={before_full_min_z:.3f}",
        f"dz_support={dz_for_support:.3f}",
        f"dz_full={dz_for_full_clearance:.3f}",
        f"dz_applied={dz_down:.3f}",
        f"target_z={target_z:.3f}",
        f"never_lower={ROBOT_PARK_NEVER_LOWER}",
        f"full_safety_enabled={full_safety_enabled}",
        f"final_support_min_z={fmt_optional(final_min_z)}",
        f"final_full_min_z={fmt_optional(final_full_min_z)}",
        f"final_clearance={fmt_optional(final_clearance)}",
        f"final_full_clearance={fmt_optional(final_full_clearance)}",
        f"skin={ROBOT_PARK_CONTACT_SKIN:.3f}",
        f"full_skin={ROBOT_PARK_FULL_COLLISION_CLEARANCE_Z:.3f}",
        f"final_source={final_source}",
        f"final_full_source={final_full_source}",
    )
    return True


def settle_robot_on_ground_if_needed(label=""):
    prim = get_prim(ROBOT_BASE)
    if not prim or not prim.IsValid():
        info_print("[GROUND SETTLE] robot root unavailable", f"label={label}")
        return False

    support_min, support_max, source = robot_support_bbox()
    if support_min is None or support_max is None:
        info_print("[GROUND SETTLE] support bbox unavailable", f"label={label}", f"source={source}")
        return False

    support_cx = 0.5 * (float(support_min[0]) + float(support_max[0]))
    support_cy = 0.5 * (float(support_min[1]) + float(support_max[1]))
    ground_top, ground_path = support_ground_top_at_xy(support_cx, support_cy)
    clearance = float(support_min[2]) - float(ground_top)
    full_min, full_max, full_source = robot_full_collision_bbox()
    full_clearance = None if full_min is None else float(full_min[2]) - float(ground_top)
    target_support_min_z = float(ground_top) + ROBOT_GROUND_TARGET_CLEARANCE_Z
    target_full_min_z = float(ground_top) + ROBOT_GROUND_FULL_TARGET_CLEARANCE_Z
    full_safety_enabled = full_min is not None and full_source == "collision_bbox"
    too_high = clearance > ROBOT_GROUND_FLOAT_TOL_Z
    too_low = clearance < (ROBOT_GROUND_TARGET_CLEARANCE_Z - ROBOT_GROUND_PENETRATION_TOL_Z)
    full_too_low = (
        full_safety_enabled
        and full_clearance < -ROBOT_GROUND_FULL_PENETRATION_TOL_Z
    )

    if too_high and not (too_low or full_too_low):
        info_print(
            "[GROUND SETTLE WAIT]",
            f"label={label}",
            f"support_source={source}",
            f"full_source={full_source}",
            "height_basis=physics_drop_no_script_lower",
            f"ground={ground_path}",
            f"ground_top={ground_top:.3f}",
            f"support_min_z={support_min[2]:.3f}",
            f"clearance={clearance:.3f}",
            f"full_min_z={fmt_optional(None if full_min is None else full_min[2])}",
            f"full_clearance={fmt_optional(full_clearance)}",
            f"target_clearance={ROBOT_GROUND_TARGET_CLEARANCE_Z:.3f}",
            f"full_safety_enabled={full_safety_enabled}",
        )
        return True

    if not (too_high or too_low or full_too_low):
        info_print(
            "[GROUND SETTLE OK]",
            f"label={label}",
            f"support_source={source}",
            f"full_source={full_source}",
            "height_basis=fixed_base_operational_clearance",
            f"ground={ground_path}",
            f"ground_top={ground_top:.3f}",
            f"support_min_z={support_min[2]:.3f}",
            f"clearance={clearance:.3f}",
            f"full_min_z={fmt_optional(None if full_min is None else full_min[2])}",
            f"full_clearance={fmt_optional(full_clearance)}",
            f"target_clearance={ROBOT_GROUND_TARGET_CLEARANCE_Z:.3f}",
            f"full_safety_enabled={full_safety_enabled}",
        )
        return True

    if full_too_low:
        log_freeze(
            "initial_ground_penetration_collision_risk",
            mode="startup",
            extra=(
                f"support_min_z={support_min[2]:.3f}; "
                f"ground_top={ground_top:.3f}; "
                f"clearance={clearance:.3f}; "
                f"full_min_z={fmt_optional(None if full_min is None else full_min[2])}; "
                f"full_clearance={fmt_optional(full_clearance)}; "
                f"ground={ground_path}; "
                f"support_source={source}; "
                f"full_source={full_source}"
            ),
            force=True,
        )
    elif too_low:
        info_print(
            "[GROUND SETTLE LOW]",
            f"label={label}",
            f"support_source={source}",
            f"full_source={full_source}",
            "reason=support_clearance_below_operational_target",
            f"ground={ground_path}",
            f"ground_top={ground_top:.3f}",
            f"support_min_z={support_min[2]:.3f}",
            f"clearance={clearance:.3f}",
            f"target_clearance={ROBOT_GROUND_TARGET_CLEARANCE_Z:.3f}",
            f"full_min_z={fmt_optional(None if full_min is None else full_min[2])}",
            f"full_clearance={fmt_optional(full_clearance)}",
            f"full_safety_enabled={full_safety_enabled}",
        )

    current = get_prim_translation(ROBOT_BASE)
    dz_support = target_support_min_z - float(support_min[2])
    dz = max(0.0, dz_support)
    dz_full = None
    if full_safety_enabled:
        dz_full = target_full_min_z - float(full_min[2])
        if dz_full > dz:
            dz = dz_full
    if abs(float(dz)) > ROBOT_GROUND_MAX_SCRIPT_CORRECTION_Z:
        clipped = math.copysign(ROBOT_GROUND_MAX_SCRIPT_CORRECTION_Z, float(dz))
        info_print(
            "[WARN] [GROUND SETTLE] script correction clipped",
            f"label={label}",
            f"raw_dz={dz:.3f}",
            f"clipped_dz={clipped:.3f}",
            f"support_source={source}",
            f"full_source={full_source}",
        )
        dz = clipped
    target_z = float(current[2]) + dz
    set_xform(prim, translate=(float(current[0]), float(current[1]), target_z))
    sync_articulation_pose_from_usd(label)
    info_print(
        "[GROUND SETTLE CORRECT]",
        f"label={label}",
        f"support_source={source}",
        f"full_source={full_source}",
        "height_basis=fixed_base_operational_clearance",
        f"ground={ground_path}",
        f"ground_top={ground_top:.3f}",
        f"before_support_min_z={support_min[2]:.3f}",
        f"before_clearance={clearance:.3f}",
        f"before_full_min_z={fmt_optional(None if full_min is None else full_min[2])}",
        f"before_full_clearance={fmt_optional(full_clearance)}",
        f"dz_support={dz_support:.3f}",
        f"dz_full={fmt_optional(dz_full)}",
        f"dz={dz:.3f}",
        f"target_root_z={target_z:.3f}",
        f"target_clearance={ROBOT_GROUND_TARGET_CLEARANCE_Z:.3f}",
        f"target_full_clearance={ROBOT_GROUND_FULL_TARGET_CLEARANCE_Z:.3f}",
        f"contact_epsilon={ROBOT_GROUND_CONTACT_EPSILON_Z:.3f}",
        f"full_safety_enabled={full_safety_enabled}",
    )
    return True


def build_scene():
    global CONTROL_ROOT, TARGET_PATH, UNLOAD_MARKER_PATH

    for p in ["/ControlRig", "/World/ControlRig"]:
        if get_prim(p).IsValid():
            stage.RemovePrim(Sdf.Path(p))

    CONTROL_ROOT = "/World/ControlRig" if get_prim("/World").IsValid() else "/ControlRig"
    UsdGeom.Xform.Define(stage, CONTROL_ROOT)

    if sand_site_active():
        ensure_particle_gpu_physics_scene("build_scene")

    if sand_site_active():
        info_print("[SAND SITE] active; using local hard parking ground instead of flat full ground")
    make_parking_ground()
    park_robot_on_support_ground("build_scene")

    dome = UsdLux.DomeLight.Define(stage, f"{CONTROL_ROOT}/DomeLight")
    dome.CreateIntensityAttr(700.0)

    light = UsdLux.SphereLight.Define(stage, f"{CONTROL_ROOT}/WorkLight")
    light.CreateIntensityAttr(30000.0)
    light.CreateRadiusAttr(1.2)
    set_xform(light.GetPrim(), translate=(0.0, -5.0, 7.0))

    bc = bbox_center(BUCKET_LINK)
    target_pos = (bc[0], bc[1], max(bc[2], 1.0))

    TARGET_PATH = f"{CONTROL_ROOT}/TargetBall"
    make_sphere(
        TARGET_PATH,
        translate=target_pos,
        radius=TARGET_RADIUS,
        color=TARGET_COLOR_DEFAULT,
        collision=False,
    )
    disable_collision(get_prim(TARGET_PATH), "target_ball")
    UNLOAD_MARKER_PATH = f"{CONTROL_ROOT}/UnloadPointBall"
    if not apply_default_unload_source_mesh():
        ensure_unload_marker(label="build_scene")

    api = get_sand_site_api()
    move_target = None if api is None else api.get("move_target_to_entry")
    if callable(move_target):
        try:
            move_target()
            info_print("[SAND SITE] moved target ball to recommended sand entry")
        except Exception as e:
            info_print("[WARN] sand site target placement failed:", e)


# ============================================================
# Controller using Isaac articulation actions
# ============================================================

class ArticulationActionController:
    def __init__(self):
        self.dof_names = ["swing", "boom", "arm", "bucket"]
        self.name_to_idx = {n: i for i, n in enumerate(self.dof_names)}
        self.q_cmd = np.zeros(4, dtype=np.float32)
        self.q_safe = self.q_cmd.copy()

    def clip_limits(self, q):
        q = np.array(q, dtype=np.float32)
        for name, idx in self.name_to_idx.items():
            if name == "swing":
                continue
            if name in FINAL_LIMITS_RAD:
                lo, hi = FINAL_LIMITS_RAD[name]
                q[idx] = min(max(float(q[idx]), lo), hi)
        return q

    def clip_action_limits(self, q):
        q = self.clip_limits(q)
        idx = self.name_to_idx["swing"]
        lo, hi = FINAL_LIMITS_RAD["swing"]
        q[idx] = min(max(float(q[idx]), lo), hi)
        return q

    def ground_ok(self, mode="auto"):
        report = phase_ground_report(mode)
        base_min = report["base_min"]
        base_reason = None

        if base_min is not None and base_min < GROUND_TOP_Z - 0.02:
            base_reason = f"base below ground {base_min:.3f}"
            if BASE_GROUND_BLOCKS_COMMANDS:
                return False, format_ground_report(mode, report, base_reason)

        if mode == "manual":
            bucket_min = report["bucket_min"]
            if bucket_min is not None and bucket_min < GROUND_TOP_Z - 0.20:
                return True, format_ground_report(mode, report, "bucket command allowed for recovery")
            if base_reason is not None:
                return True, format_ground_report(mode, report, base_reason)
            return True, "ok"

        ok, reason = phase_ground_ok(mode, report)
        if not ok:
            return True, format_ground_report(
                mode,
                report,
                "predicted contact allowed: " + reason,
            )

        if base_reason is not None:
            return True, format_ground_report(mode, report, base_reason)

        return True, "ok"

    def command_ground_ok(self, q, mode="auto"):
        if mode == "manual":
            return self.ground_ok(mode=mode)

        report = predicted_phase_ground_report(q, mode)
        if report.get("tip_z") is None:
            return self.ground_ok(mode=mode)

        base_min = report["base_min"]
        if base_min is not None and base_min < GROUND_TOP_Z - 0.02:
            base_reason = f"base below ground {base_min:.3f}"
            if BASE_GROUND_BLOCKS_COMMANDS:
                return False, format_ground_report(mode, report, base_reason)
        else:
            base_reason = None

        ok, reason = phase_ground_ok(mode, report)
        if not ok:
            return True, format_ground_report(
                mode,
                report,
                "predicted contact allowed: " + reason,
            )

        if base_reason is not None:
            return True, format_ground_report(mode, report, base_reason)

        return True, "ok"

    def send_action(self, q, mode="action"):
        global ROBOT, JOINT_INDICES

        q = clip_command_near(q, reference=self.q_cmd)
        q = maybe_rebase_swing_for_bounded_joint(q)
        q_action = self.clip_action_limits(q)
        swing_idx = self.name_to_idx["swing"]
        lo, hi = FINAL_LIMITS_RAD["swing"]
        if float(q[swing_idx]) < lo or float(q[swing_idx]) > hi:
            q[swing_idx] = q_action[swing_idx]
        self.q_cmd = q.copy()
        self.q_safe = q.copy()

        if ROBOT is None or JOINT_INDICES is None:
            info_print("[WARN] ROBOT or JOINT_INDICES not ready")
            return False, "robot not ready"
        if not robot_articulation_action_ready():
            return False, "physics view not ready"

        if len(JOINT_INDICES) != len(q):
            msg = f"joint index count mismatch: q={len(q)}, indices={len(JOINT_INDICES)}"
            info_print("[ERROR]", msg)
            return False, msg

        try:
            ROBOT.apply_action(
                ArticulationAction(
                    joint_positions=q_action,
                    joint_indices=JOINT_INDICES,
                )
            )
            now = time.time()
            STATE["last_action_q"] = q_action.copy()
            STATE["last_action_unclipped_q"] = q.copy()
            STATE["last_action_time"] = now
            STATE["last_action_mode"] = str(mode)
            if now - float(STATE.get("last_action_diag_time", 0.0)) > 1.0:
                try:
                    q_real = get_real_joint_positions()
                    q_real_deg = [round(rad_to_deg(float(x)), 2) for x in q_real]
                except Exception:
                    q_real_deg = None
                info_print(
                    "[ACTION SENT]",
                    "cmd_deg=", [round(rad_to_deg(float(x)), 2) for x in q],
                    "action_deg=", [round(rad_to_deg(float(x)), 2) for x in q_action],
                    "real_deg=", q_real_deg,
                )
                STATE["last_action_diag_time"] = now
            return True, "ok"
        except Exception as e:
            info_print("[ERROR] apply_action failed:", e)
            return False, str(e)

    def hold_current_action(self):
        if ROBOT is None or JOINT_INDICES is None:
            return
        if not robot_articulation_action_ready():
            return
        if len(JOINT_INDICES) != len(self.q_cmd):
            return
        q_action = self.clip_action_limits(self.q_cmd)
        try:
            ROBOT.apply_action(
                ArticulationAction(
                    joint_positions=q_action,
                    joint_indices=JOINT_INDICES,
                )
            )
        except Exception as e:
            now = time.time()
            if now - float(STATE.get("last_action_diag_time", 0.0)) > 1.0:
                info_print("[WARN] hold_current_action failed:", e)
                STATE["last_action_diag_time"] = now

    def apply_target(self, q_target, mode="auto", speed_multiplier_override=None):
        q_target = clip_command_near(q_target, reference=self.q_cmd)

        if speed_multiplier_override is None:
            speed_multiplier = float(STATE.get("speed_multiplier", 1.0))
        else:
            speed_multiplier = float(speed_multiplier_override)
        # velocity limit per tick
        q_next = self.q_cmd.copy()
        for name, idx in self.name_to_idx.items():
            max_step = DQ_MAX[name] * CONTROL_DT * speed_multiplier
            if name == "swing":
                delta = swing_delta(q_target[idx], self.q_cmd[idx])
            else:
                delta = float(q_target[idx] - self.q_cmd[idx])
            if delta > max_step:
                q_next[idx] = self.q_cmd[idx] + max_step
            elif delta < -max_step:
                q_next[idx] = self.q_cmd[idx] - max_step
            else:
                q_next[idx] = q_target[idx]
            if name == "swing":
                q_next[idx] = normalize_swing_cmd(q_next[idx])

        ok, reason = self.command_ground_ok(q_next, mode=mode)
        if not ok:
            update_status("[GUARD] " + reason, force=True)
            info_print("[GUARD BLOCKED]", f"mode={mode}", reason)
            # restore safe
            if self.q_safe is not None:
                self.send_action(self.q_safe, mode=f"{mode}_guard_restore")
            return False, reason
        if reason != "ok":
            update_status("[GROUND WARN] " + reason, force=False)

        return self.send_action(q_next, mode=mode)

    def apply_target_direct(self, q_target, mode="manual"):
        # direct articulation command without velocity projection (for sliders)
        q = clip_command_near(q_target, reference=self.q_cmd)
        ok, reason = self.command_ground_ok(q, mode=mode)
        if not ok:
            update_status("[GUARD] " + reason, force=True)
            info_print("[GUARD BLOCKED]", f"mode={mode}", reason)
            return False, reason
        if reason != "ok":
            update_status("[GROUND WARN] " + reason, force=False)
        return self.send_action(q, mode=mode)

    def print_state(self):
        bc = bbox_center(BUCKET_LINK)
        target = get_prim_translation(TARGET_PATH)
        info_print("========== STATE ==========")
        info_print("q_cmd rad:", self.q_cmd)
        info_print("q_cmd deg:", [rad_to_deg(x) for x in self.q_cmd])
        info_print("bucket center:", bc)
        info_print("target:", target)
        info_print("error:", np.linalg.norm(target - bc))

        # Additional diagnostics
        try:
            center = get_swing_center_world()
            r = target_radius_from_swing_center(target)
            rmax = estimate_dynamic_reach_radius()
            info_print("swing_center:", center)
            info_print("target_radius_from_swing_center:", r)
            info_print("estimated_dynamic_reach:", rmax)
        except Exception as e:
            info_print("[WARN] diagnostics failed:", e)
        print_motion_constraint_diagnostics("print_state")


CTRL = ArticulationActionController()


def set_manual_joint_target(q_target, reason="slider"):
    q = clip_command_near(q_target, reference=CTRL.q_cmd)
    STATE["manual_joint_target"] = np.array(q, dtype=np.float32)
    STATE["manual_joint_active"] = True
    STATE["manual_override"] = True
    hold_manual_ui_sync()

    now = time.time()
    if now - float(STATE.get("manual_last_status_time", 0.0)) > MANUAL_STATUS_INTERVAL:
        STATE["manual_last_status_time"] = now
        update_status(
            f"[MANUAL TARGET] {reason}: smoothing to "
            f"swing={rad_to_deg(q[0]):.1f} boom={rad_to_deg(q[1]):.1f} "
            f"arm={rad_to_deg(q[2]):.1f} bucket={rad_to_deg(q[3]):.1f} "
            f"manual_speed={get_manual_speed_multiplier():.2f}",
            force=False,
        )


def manual_recovery_project_target(q_target):
    q = np.array(q_target, dtype=np.float32).copy()
    try:
        report = phase_ground_report("manual_recovery")
        bucket_min = report.get("bucket_min")
    except Exception:
        return q, False

    if bucket_min is None or float(bucket_min) >= GROUND_TOP_Z - MANUAL_RECOVERY_BUCKET_PENETRATION_Z:
        return q, False

    try:
        q_real = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
    except Exception:
        q_real = CTRL.q_cmd.copy()

    swing_idx = CTRL.name_to_idx["swing"]
    swing_err_deg = abs(rad_to_deg(swing_delta(float(q[swing_idx]), float(q_real[swing_idx]))))
    adjusted = False
    if swing_err_deg > MANUAL_RECOVERY_SWING_HOLD_DEG:
        q[swing_idx] = q_real[swing_idx]
        adjusted = True

    now = time.time()
    if now - float(STATE.get("manual_recovery_last_status_time", 0.0)) > MANUAL_RECOVERY_STATUS_INTERVAL:
        STATE["manual_recovery_last_status_time"] = now
        info_print(
            "[MANUAL RECOVERY]",
            f"bucket_min={float(bucket_min):.3f}",
            f"penetration={GROUND_TOP_Z - float(bucket_min):.3f}",
            f"swing_delta_requested={swing_err_deg:.2f}deg",
            f"swing_held={adjusted}",
            "reason=bucket is below hard ground; lift/curl before swinging",
        )
        update_status(
            f"[MANUAL RECOVERY] bucket below hard ground {float(bucket_min):.3f}; swing held until bucket is clear",
            force=False,
        )

    return q, adjusted


def apply_manual_joint_target_step():
    if not bool(STATE.get("manual_joint_active", False)):
        return

    q_target = STATE.get("manual_joint_target", None)
    if q_target is None:
        STATE["manual_joint_active"] = False
        return

    q_target = np.array(q_target, dtype=np.float32)
    q_cmd_target, recovery_projected = manual_recovery_project_target(q_target)
    ok, reason = CTRL.apply_target(
        q_cmd_target,
        mode="manual",
        speed_multiplier_override=get_manual_speed_multiplier(),
    )
    if not ok:
        update_status(f"[MANUAL BLOCKED] {reason}", force=True)
        STATE["manual_joint_active"] = False
        return

    err_deg = q_delta_abs_deg(q_cmd_target, CTRL.q_cmd)
    max_err = max(err_deg) if err_deg else 0.0
    if max_err <= MANUAL_TARGET_DEADBAND_DEG:
        STATE["manual_joint_active"] = False
        if recovery_projected:
            update_status("[MANUAL RECOVERY] swing command held; move boom/arm/bucket to clear bucket first", force=False)
        else:
            update_status(f"[MANUAL DONE] max_err={max_err:.2f}deg", force=False)
        return

    now = time.time()
    if now - float(STATE.get("manual_last_status_time", 0.0)) > MANUAL_STATUS_INTERVAL:
        STATE["manual_last_status_time"] = now
        update_status(f"[MANUAL SMOOTH] max_err={max_err:.2f}deg speed={get_manual_speed_multiplier():.2f}", force=False)


def safe_home_q():
    q = get_real_joint_positions().copy()
    q[CTRL.name_to_idx["boom"]] = deg_to_rad(HOME_SAFE_DEG["boom"])
    q[CTRL.name_to_idx["arm"]] = deg_to_rad(HOME_SAFE_DEG["arm"])
    q[CTRL.name_to_idx["bucket"]] = deg_to_rad(HOME_SAFE_DEG["bucket"])
    return clip_command_near(q, reference=CTRL.q_cmd)


def q_from_pose_deg(pose, reference=None):
    q = CTRL.q_cmd.copy() if reference is None else np.array(reference, dtype=np.float32).copy()
    for name in DOF_ORDER:
        if name not in pose or name not in CTRL.name_to_idx:
            continue
        q[CTRL.name_to_idx[name]] = deg_to_rad(float(pose[name]))
    return clip_command_near(q, reference=CTRL.q_cmd)


def auto_collect_initial_pose_for_attempt(attempt_index):
    poses = AUTO_COLLECT_INITIAL_POSES_DEG
    pose = poses[(max(1, int(attempt_index)) - 1) % max(1, len(poses))]
    q = q_from_pose_deg(pose, reference=CTRL.q_cmd)
    return {
        "id": str(pose.get("id", f"pose_{attempt_index}")),
        "pose_deg": dict(pose),
        "q": q.copy(),
        "q_deg": q_deg_values(q, wrap_swing_for_display=True),
    }


async def auto_collect_move_to_initial_pose(attempt_index):
    info = auto_collect_initial_pose_for_attempt(attempt_index)
    STATE["auto_collect_initial_pose"] = info
    STATE["auto_collect_initial_pose_id"] = info["id"]
    task_id = start_task(f"auto_collect_initial_{info['id']}")
    update_status(
        f"[AUTO DATASET] initial pose {info['id']} q={info['q_deg']}",
        force=True,
    )
    ok = await set_joint_pose_direct_and_settle(
        info["q"],
        label=f"auto_collect_initial_{info['id']}",
        mode="auto_collect_initial_direct",
        settle_frames=DIRECT_INITIAL_POSE_SETTLE_FRAMES,
        task_id=task_id,
    )
    return bool(ok and task_alive(task_id)), info


# ============================================================
# Target and trace
# ============================================================

def get_target_pos():
    return get_prim_translation(TARGET_PATH)


def set_target_xyz(x, y, z):
    prim = get_prim(TARGET_PATH)
    if prim and prim.IsValid():
        z = max(float(z), TARGET_MIN_Z)
        set_xform(prim, translate=(x, y, z))
        set_target_color(TARGET_COLOR_DEFAULT)


def set_manual_unload_point_xyz(x, y, z, source="ui", inner_size=None, z_range=None, selected_path="", range_shape=None):
    p = np.array([float(x), float(y), float(z)], dtype=np.float32)
    STATE["manual_unload_override_enabled"] = True
    STATE["manual_unload_point"] = p.copy()
    STATE["manual_unload_source"] = str(source)
    radius = manual_unload_radius()
    if selected_path:
        STATE["manual_unload_selected_path"] = str(selected_path)
    if inner_size is not None:
        arr = np.array(inner_size, dtype=np.float32).reshape(-1)[:2]
        if len(arr) >= 2:
            STATE["manual_unload_inner_size"] = np.maximum(arr, np.array([0.2, 0.2], dtype=np.float32))
            radius = 0.5 * float(np.min(STATE["manual_unload_inner_size"]))
            STATE["manual_unload_radius"] = max(0.05, radius)
            STATE["manual_unload_range_shape"] = str(range_shape or "box")
    elif STATE.get("manual_unload_inner_size") is None:
        d = 2.0 * radius
        STATE["manual_unload_inner_size"] = np.array([d, d], dtype=np.float32)
        STATE["manual_unload_range_shape"] = str(range_shape or "circle")
    else:
        d = 2.0 * radius
        STATE["manual_unload_inner_size"] = np.array([d, d], dtype=np.float32)
        STATE["manual_unload_range_shape"] = str(range_shape or "circle")
    if range_shape is not None:
        STATE["manual_unload_range_shape"] = str(range_shape)
        if str(range_shape) == "circle":
            STATE["manual_unload_polygon_xy"] = None
            STATE["manual_unload_selected_hull_xy"] = None
            STATE["manual_unload_selected_size_xy"] = None
    if z_range is not None:
        arr = np.array(z_range, dtype=np.float32).reshape(-1)[:2]
        if len(arr) >= 2:
            STATE["manual_unload_z_range"] = np.array([min(float(arr[0]), float(arr[1])), max(float(arr[0]), float(arr[1]))], dtype=np.float32)
    else:
        STATE["manual_unload_z_range"] = np.array([min(float(GROUND_TOP_Z), float(z)), max(float(GROUND_TOP_Z), float(z))], dtype=np.float32)
    ensure_unload_marker(p, label=f"manual_{source}")
    reset_dig_plan()
    return p


def set_manual_unload_from_mesh_path(path, source="selected_mesh", status=True):
    if not path:
        update_status("[UNLOAD SETUP] select a mesh/group first", force=True)
        return False
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        update_status(f"[UNLOAD SETUP] mesh path invalid: {path}", force=True)
        info_print("[WARN] [UNLOAD SETUP] invalid mesh path:", path)
        return False
    center, size, mn, mx = bbox_center_size(path)
    if center is None:
        update_status(f"[UNLOAD SETUP] selected prim has no valid bbox: {path}", force=True)
        return False
    shrink = get_unload_mesh_shrink_from_model()
    STATE["manual_unload_mesh_shrink_d"] = float(shrink)
    xy_points = mesh_world_xy_points_under(prim)
    hull = convex_hull_xy(xy_points)
    if hull is None:
        hull = np.array(
            [
                [float(mn[0]), float(mn[1])],
                [float(mx[0]), float(mn[1])],
                [float(mx[0]), float(mx[1])],
                [float(mn[0]), float(mx[1])],
            ],
            dtype=np.float32,
        )
    poly = shrink_convex_polygon_xy(hull, shrink)
    if poly is None:
        update_status(f"[UNLOAD SETUP] selected mesh footprint failed: {path}", force=True)
        return False
    footprint_center = polygon_centroid_xy(poly)
    poly_min = np.min(poly, axis=0)
    poly_max = np.max(poly, axis=0)
    inner = np.maximum(poly_max - poly_min, np.array([0.2, 0.2], dtype=np.float32))
    point = np.array([float(footprint_center[0]), float(footprint_center[1]), float(mx[2]) + 0.06], dtype=np.float32)
    STATE["manual_unload_selected_size_xy"] = np.maximum(np.max(hull, axis=0) - np.min(hull, axis=0), np.array([0.2, 0.2], dtype=np.float32))
    STATE["manual_unload_selected_hull_xy"] = np.array(hull, dtype=np.float32)
    STATE["manual_unload_polygon_xy"] = np.array(poly, dtype=np.float32)
    set_manual_unload_point_xyz(
        float(point[0]),
        float(point[1]),
        float(point[2]),
        source=source,
        inner_size=inner,
        z_range=np.array([float(mn[2]), float(point[2])], dtype=np.float32),
        selected_path=path,
        range_shape="box",
    )
    update_unload_models_only(point)
    info_print(
        "[UNLOAD SETUP]",
        f"source={source}",
        f"path={path}",
        f"center={vec_list([footprint_center[0], footprint_center[1], point[2]], 3)}",
        f"size={vec_list(size, 3)}",
        f"inner_size={vec_list(inner, 2)}",
        f"mesh_shrink_d={shrink:.3f}",
        f"hull_vertices={len(hull)}",
        f"active_vertices={len(poly)}",
        f"point={vec_list(point, 3)}",
    )
    if status:
        update_status(f"[UNLOAD SETUP] mesh/group selected: {path}", force=True)
    return True


def set_manual_unload_from_selected_mesh():
    return set_manual_unload_from_mesh_path(first_selected_prim_path(), source="selected_mesh", status=True)


def apply_default_unload_source_mesh():
    return set_manual_unload_from_mesh_path(DEFAULT_UNLOAD_SOURCE_MESH_PATH, source="preset_mesh", status=False)


def set_manual_unload_from_selected_point():
    path = first_selected_prim_path()
    if not path:
        update_status("[UNLOAD SETUP] select an unload point prim first", force=True)
        return False
    center, size, mn, mx = bbox_center_size(path)
    if center is None:
        center = get_prim_translation(path)
        size = np.zeros(3, dtype=np.float32)
    point = np.array(center[:3], dtype=np.float32)
    set_manual_unload_point_xyz(
        float(point[0]),
        float(point[1]),
        float(point[2]),
        source="selected_point",
        inner_size=np.array([2.0 * UNLOAD_POINT_DEFAULT_RADIUS, 2.0 * UNLOAD_POINT_DEFAULT_RADIUS], dtype=np.float32),
        selected_path=path,
        range_shape="circle",
    )
    update_unload_models_only(point)
    info_print(
        "[UNLOAD SETUP]",
        "source=selected_point",
        f"path={path}",
        f"point={vec_list(point, 3)}",
    )
    update_status(f"[UNLOAD SETUP] point selected: {path}", force=True)
    return True


def clear_manual_unload_override():
    STATE["manual_unload_override_enabled"] = False
    STATE["manual_unload_point"] = None
    STATE["manual_unload_inner_size"] = None
    STATE["manual_unload_z_range"] = None
    STATE["manual_unload_radius"] = float(UNLOAD_POINT_DEFAULT_RADIUS)
    STATE["manual_unload_mesh_shrink_d"] = float(UNLOAD_SELECTED_EDGE_MARGIN)
    STATE["manual_unload_range_shape"] = "circle"
    STATE["manual_unload_selected_size_xy"] = None
    STATE["manual_unload_source"] = ""
    STATE["manual_unload_selected_path"] = ""
    reset_dig_plan()
    p = unload_bin_landing_point()
    ensure_unload_marker(p, label="clear_manual_unload_override")
    update_unload_models_only(p)
    update_status("[UNLOAD SETUP] cleared manual override; using sand site/default unload bin", force=True)


def update_target_from_models():
    x = TARGET_MODELS["x"].as_float
    y = TARGET_MODELS["y"].as_float
    z = TARGET_MODELS["z"].as_float
    set_target_xyz(x, y, z)


def get_target_xyz_from_models():
    return np.array([
        TARGET_MODELS["x"].as_float,
        TARGET_MODELS["y"].as_float,
        max(TARGET_MODELS["z"].as_float, TARGET_MIN_Z),
    ], dtype=np.float32)


def get_unload_xyz_from_models():
    z_model = UNLOAD_MODELS.get("z_max") or UNLOAD_MODELS.get("z")
    try:
        z_value = float(z_model.as_float) if z_model is not None else float(unload_bin_landing_point()[2])
    except Exception:
        try:
            z_value = float(z_model.get_value_as_float()) if z_model is not None else float(unload_bin_landing_point()[2])
        except Exception:
            z_value = float(unload_bin_landing_point()[2])
    return np.array([
        UNLOAD_MODELS["x"].as_float,
        UNLOAD_MODELS["y"].as_float,
        z_value,
    ], dtype=np.float32)


def get_unload_z_range_from_models():
    p = get_unload_xyz_from_models()
    z_min_model = UNLOAD_MODELS.get("z_min")
    z_max_model = UNLOAD_MODELS.get("z_max") or UNLOAD_MODELS.get("z")

    def model_float(model, fallback):
        if model is None:
            return float(fallback)
        try:
            return float(model.as_float)
        except Exception:
            try:
                return float(model.get_value_as_float())
            except Exception:
                return float(fallback)

    z_min = model_float(z_min_model, GROUND_TOP_Z)
    z_max = model_float(z_max_model, p[2])
    if abs(z_max - z_min) < 0.03:
        z_max = z_min + 0.03
    return np.array([min(z_min, z_max), max(z_min, z_max)], dtype=np.float32)


def get_unload_radius_from_model():
    model = UNLOAD_MODELS.get("r")
    if model is None:
        return manual_unload_radius()
    try:
        return max(0.05, float(model.as_float))
    except Exception:
        try:
            return max(0.05, float(model.get_value_as_float()))
        except Exception:
            return manual_unload_radius()


def get_unload_mesh_shrink_from_model():
    model = UNLOAD_MODELS.get("d")
    if model is None:
        return manual_unload_mesh_shrink_d()
    try:
        return max(0.0, float(model.as_float))
    except Exception:
        try:
            return max(0.0, float(model.get_value_as_float()))
        except Exception:
            return manual_unload_mesh_shrink_d()


def sync_target_from_sliders_live(force=False):
    """
    target ball 随 slider 自动更新。为了避免每帧 author，做轻微 throttle。
    """
    if STATE.get("auto_collect_active", False):
        return

    now = time.time()

    if (not force) and now - STATE["last_target_sync_time"] < STATE["target_sync_interval"]:
        return

    if not TARGET_MODELS:
        return

    p = get_target_xyz_from_models()
    last = STATE.get("last_target_model_xyz")
    if not force and last is not None:
        try:
            if float(np.linalg.norm(p - np.array(last, dtype=np.float32).reshape(-1)[:3])) < 1.0e-4:
                STATE["last_target_sync_time"] = now
                return
        except Exception:
            pass
    set_target_xyz(float(p[0]), float(p[1]), float(p[2]))
    STATE["last_target_model_xyz"] = p.copy()
    STATE["last_target_sync_time"] = now


def sync_unload_from_sliders_live(force=False):
    if STATE.get("auto_collect_active", False):
        return

    now = time.time()
    if (not force) and now - float(STATE.get("last_unload_sync_time", 0.0) or 0.0) < float(STATE.get("target_sync_interval", 0.05)):
        return
    if not UNLOAD_MODELS:
        return

    p = get_unload_xyz_from_models()
    z_range = get_unload_z_range_from_models()
    radius = get_unload_radius_from_model()
    shrink_d = get_unload_mesh_shrink_from_model()
    last = STATE.get("last_unload_model_xyz")
    last_z_range = STATE.get("last_unload_model_z_range")
    last_radius = float(STATE.get("last_unload_model_radius", radius) or radius)
    last_shrink_d = float(STATE.get("last_unload_model_shrink_d", shrink_d) or shrink_d)
    if not force and last is not None:
        try:
            xyz_same = float(np.linalg.norm(p - np.array(last, dtype=np.float32).reshape(-1)[:3])) < 1.0e-4
            if last_z_range is None:
                z_range_same = False
            else:
                z_range_same = float(np.linalg.norm(z_range - np.array(last_z_range, dtype=np.float32).reshape(-1)[:2])) < 1.0e-4
            radius_same = abs(float(radius) - last_radius) < 1.0e-4
            shrink_same = abs(float(shrink_d) - last_shrink_d) < 1.0e-4
            if xyz_same and z_range_same and radius_same and shrink_same:
                STATE["last_unload_sync_time"] = now
                return
        except Exception:
            pass
    STATE["manual_unload_mesh_shrink_d"] = shrink_d
    if str(STATE.get("manual_unload_source", "")).startswith("selected_mesh"):
        hull = STATE.get("manual_unload_selected_hull_xy")
        if hull is not None:
            poly = shrink_convex_polygon_xy(hull, shrink_d)
            if poly is not None:
                STATE["manual_unload_polygon_xy"] = np.array(poly, dtype=np.float32)
                poly_min = np.min(poly, axis=0)
                poly_max = np.max(poly, axis=0)
                inner = np.maximum(poly_max - poly_min, np.array([0.2, 0.2], dtype=np.float32))
            else:
                inner = np.array(STATE.get("manual_unload_inner_size"), dtype=np.float32).reshape(-1)[:2]
            STATE["manual_unload_selected_size_xy"] = np.maximum(np.max(np.array(hull, dtype=np.float32), axis=0) - np.min(np.array(hull, dtype=np.float32), axis=0), np.array([0.2, 0.2], dtype=np.float32))
            set_manual_unload_point_xyz(
                float(p[0]),
                float(p[1]),
                float(p[2]),
                source="selected_mesh_ui",
                inner_size=inner,
                z_range=z_range,
                selected_path=str(STATE.get("manual_unload_selected_path", "")),
                range_shape="box",
            )
            STATE["last_unload_model_xyz"] = p.copy()
            STATE["last_unload_model_z_range"] = z_range.copy()
            STATE["last_unload_model_radius"] = float(manual_unload_radius())
            STATE["last_unload_model_shrink_d"] = float(shrink_d)
            STATE["last_unload_sync_time"] = now
            return
    STATE["manual_unload_radius"] = radius
    set_manual_unload_point_xyz(
        float(p[0]),
        float(p[1]),
        float(p[2]),
        source="ui",
        inner_size=np.array([2.0 * radius, 2.0 * radius], dtype=np.float32),
        z_range=z_range,
        range_shape="circle",
    )
    STATE["last_unload_model_xyz"] = p.copy()
    STATE["last_unload_model_z_range"] = z_range.copy()
    STATE["last_unload_model_radius"] = float(radius)
    STATE["last_unload_model_shrink_d"] = float(shrink_d)
    STATE["last_unload_sync_time"] = now


TRACE_CURVE_PATH = None
TRACE_REAL_CURVE_PATH = None
TRACE_BUCKET_PATH = None
TRACE_TARGET_PATH = None
TRACE_PLAN_END_PATH = None
TRACE_LEGACY_CURVE_PATH = None
TRACE_OLD_REAL_CURVE_PATH = None
TRACE_PLAN_MAX_POINTS = 360
TRACE_PLAN_SEGMENT_SAMPLES = 18
TRACE_MIN_CURVE_POINTS = 2


def configure_linear_trace_curve(path, color, width, default_points=None):
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        curve = UsdGeom.BasisCurves.Define(stage, path)
    else:
        curve = UsdGeom.BasisCurves(prim)
    points = default_points
    if points is None:
        points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.001, 0.0, 0.0)]
    if len(points) < TRACE_MIN_CURVE_POINTS:
        p = points[0] if points else Gf.Vec3f(0.0, 0.0, 0.0)
        points = [p, Gf.Vec3f(float(p[0]) + 0.001, float(p[1]), float(p[2]))]
    # Some Kit/Hydra builds keep a stale authored "bezier" basis on BasisCurves
    # even after the basis attr is cleared. Padding to a valid nonperiodic
    # bezier count prevents the noisy "Incorrect number of vertices" warning
    # while type=linear still renders the intended polyline.
    while len(points) < 4 or ((len(points) - 4) % 3) != 0:
        last = points[-1]
        points.append(Gf.Vec3f(float(last[0]), float(last[1]), float(last[2])))
    try:
        curve.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    except Exception:
        try:
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        except Exception:
            pass
    try:
        curve.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    except Exception:
        try:
            curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        except Exception:
            pass
    try:
        curve.GetBasisAttr().Block()
    except Exception:
        try:
            curve.GetBasisAttr().Clear()
        except Exception:
            pass
    try:
        curve.GetCurveVertexCountsAttr().Set([len(points)])
    except Exception:
        curve.CreateCurveVertexCountsAttr([len(points)])
    try:
        curve.GetWidthsAttr().Set([float(width)] * len(points))
    except Exception:
        curve.CreateWidthsAttr([float(width)] * len(points))
    try:
        curve.GetPointsAttr().Set(points)
    except Exception:
        curve.CreatePointsAttr(points)
    set_color(curve.GetPrim(), color)
    return curve

def ensure_trace_prims():
    global TRACE_CURVE_PATH, TRACE_REAL_CURVE_PATH, TRACE_BUCKET_PATH, TRACE_TARGET_PATH, TRACE_PLAN_END_PATH, TRACE_LEGACY_CURVE_PATH, TRACE_OLD_REAL_CURVE_PATH

    if CONTROL_ROOT is None:
        return

    trace_root = f"{CONTROL_ROOT}/TracePath"

    if not get_prim(trace_root).IsValid():
        UsdGeom.Xform.Define(stage, trace_root)

    TRACE_LEGACY_CURVE_PATH = f"{trace_root}/path_curve"
    TRACE_CURVE_PATH = f"{trace_root}/planned_path_curve"
    TRACE_REAL_CURVE_PATH = f"{trace_root}/planned_bucket_curve"
    TRACE_OLD_REAL_CURVE_PATH = f"{trace_root}/real_bucket_curve"
    TRACE_BUCKET_PATH = f"{trace_root}/bucket_marker"
    TRACE_TARGET_PATH = f"{trace_root}/target_marker"
    TRACE_PLAN_END_PATH = f"{trace_root}/planned_bucket_end_marker"

    default_green = [Gf.Vec3f(float(i) * 0.001, 0.0, 0.0) for i in range(max(TRACE_MIN_CURVE_POINTS, TRACE_COUNT))]
    configure_linear_trace_curve(TRACE_CURVE_PATH, (0.05, 1.0, 0.10), 0.035, default_green)
    configure_linear_trace_curve(TRACE_REAL_CURVE_PATH, (0.05, 0.35, 1.0), 0.080)

    if not get_prim(TRACE_BUCKET_PATH).IsValid():
        s = UsdGeom.Sphere.Define(stage, TRACE_BUCKET_PATH)
        s.CreateRadiusAttr(0.075)
        set_color(s.GetPrim(), (0.05, 0.25, 1.0))

    if not get_prim(TRACE_TARGET_PATH).IsValid():
        s = UsdGeom.Sphere.Define(stage, TRACE_TARGET_PATH)
        s.CreateRadiusAttr(0.075)
        set_color(s.GetPrim(), (1.0, 0.9, 0.05))

    if not get_prim(TRACE_PLAN_END_PATH).IsValid():
        s = UsdGeom.Sphere.Define(stage, TRACE_PLAN_END_PATH)
        s.CreateRadiusAttr(0.095)
        set_color(s.GetPrim(), (0.0, 0.75, 1.0))


def set_trace_visibility(paths, visible):
    token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    for p in paths:
        if p is None:
            continue
        prim = get_prim(p)
        if prim and prim.IsValid():
            try:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(token)
            except Exception:
                pass


def hide_trace_prims():
    set_trace_visibility(
        [
            TRACE_CURVE_PATH,
            TRACE_REAL_CURVE_PATH,
            TRACE_BUCKET_PATH,
            TRACE_TARGET_PATH,
            TRACE_PLAN_END_PATH,
            TRACE_LEGACY_CURVE_PATH,
            TRACE_OLD_REAL_CURVE_PATH,
        ],
        False,
    )


def show_trace_prims(mode=None):
    mode = current_trace_mode() if mode is None else int(mode)
    if mode == 1:
        set_trace_visibility([TRACE_CURVE_PATH, TRACE_BUCKET_PATH, TRACE_TARGET_PATH], True)
        set_trace_visibility([TRACE_REAL_CURVE_PATH, TRACE_PLAN_END_PATH, TRACE_LEGACY_CURVE_PATH, TRACE_OLD_REAL_CURVE_PATH], False)
    elif mode == 2:
        set_trace_visibility([TRACE_REAL_CURVE_PATH, TRACE_PLAN_END_PATH], True)
        set_trace_visibility(
            [TRACE_CURVE_PATH, TRACE_BUCKET_PATH, TRACE_TARGET_PATH, TRACE_LEGACY_CURVE_PATH, TRACE_OLD_REAL_CURVE_PATH],
            False,
        )
    else:
        hide_trace_prims()


def current_trace_mode():
    try:
        mode = int(STATE.get("trace_mode", 0))
    except Exception:
        mode = 1 if bool(STATE.get("trace", False)) else 0
    if mode not in (0, 1, 2):
        mode = 1 if bool(STATE.get("trace", False)) else 0
    STATE["trace_mode"] = mode
    STATE["trace"] = bool(mode)
    return mode


def set_trace_mode(mode, reset_real=False):
    mode = int(mode)
    if mode not in (0, 1, 2):
        mode = 0
    old_mode = int(STATE.get("trace_mode", 0) or 0)
    STATE["trace_mode"] = mode
    STATE["trace"] = bool(mode)
    if mode != old_mode or mode == 2:
        STATE["trace_no_plan_notice_shown"] = False
        STATE["trace_no_plan_notice_time"] = 0.0
    if reset_real or mode != old_mode:
        STATE["trace_bucket_points"] = []
        STATE["trace_bucket_last_point"] = None
    if mode == 0:
        hide_trace_prims()
    else:
        draw_trace(force=True)
    mode_name = {
        0: "off",
        1: "green target/stage path",
        2: "blue planned bucket path",
    }.get(mode, "off")
    update_status(f"Trace mode = {mode} ({mode_name})", force=True)


def trace_auto_carry_bucket_world(q0, q1, mode):
    level_mode = str(mode).lower()
    auto_carry_bucket = (
        ("lift_carry" in level_mode)
        or ("carry" in level_mode and "unload" not in level_mode)
        or ("unload_to_bin" in level_mode)
    )
    if not auto_carry_bucket:
        return None
    start_angles = chain_angles_from_q(q0, end_effector="load")
    goal_angles = chain_angles_from_q(q1, end_effector="load")
    reference_angles = goal_angles if goal_angles is not None else start_angles
    if reference_angles is None:
        return None
    return float(nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q1))


def planned_bucket_segment_points(q_start, q_goal, mode="auto", samples=None):
    if q_start is None or q_goal is None:
        return [], None
    q0 = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    q1 = clip_command_near(np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy(), reference=q0)
    n = max(2, int(TRACE_PLAN_SEGMENT_SAMPLES if samples is None else samples))
    carry_bucket_world_rad = trace_auto_carry_bucket_world(q0, q1, mode)
    points = []
    q_last = q0.copy()

    for i in range(n):
        u = float(i) / max(1, n - 1)
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        q = interpolate_q_shortest(q0, q1, s)
        if carry_bucket_world_rad is not None:
            carry_calc = bucket_joint_for_world_angle(q, carry_bucket_world_rad, end_effector="load")
            if carry_calc is not None:
                q[CTRL.name_to_idx["bucket"]] = carry_calc["bucket"]
                q = CTRL.clip_limits(q)
        p = predicted_end_world_point(q, end_effector="tip", reference_q=q0)
        if p is not None:
            points.append(np.array(p, dtype=np.float32).reshape(-1)[:3].copy())
        q_last = q.copy()

    return points, q_last


def cache_dig_plan_trace_points(seq=None, start_q=None):
    return trace_showing.cache_dig_plan_trace_points(runtime_module(), seq=seq, start_q=start_q)


def cache_active_stage_trace_points(stage_name, q_start, q_goal, stage_index=None, include_remaining=True):
    return trace_showing.cache_active_stage_trace_points(
        runtime_module(),
        stage_name,
        q_start,
        q_goal,
        stage_index=stage_index,
        include_remaining=include_remaining,
    )


def planned_bucket_points_from_dig_plan():
    return list(STATE.get("dig_plan_trace_points", []) or [])[:TRACE_PLAN_MAX_POINTS]


def dig_plan_target_matches_current(tol=0.05):
    plan_target = STATE.get("dig_plan_target", None)
    if plan_target is None:
        return False
    try:
        current = get_target_pos()
        current[2] = max(float(current[2]), GROUND_TOP_Z)
        planned = np.array(plan_target, dtype=np.float32).reshape(-1)[:3]
        return float(np.linalg.norm(current - planned)) <= float(tol)
    except Exception:
        return False


def ensure_trace_dig_plan_current():
    return


def planned_bucket_points_from_active_motion():
    motion = STATE.get("trace_active_motion")
    if not isinstance(motion, dict):
        return []
    if time.time() > float(motion.get("expires_at", 0.0)):
        return []
    q_goal = motion.get("q_goal")
    mode = motion.get("mode", "auto")
    if q_goal is None:
        return []
    points, _q_end = planned_bucket_segment_points(CTRL.q_cmd.copy(), q_goal, mode=mode)
    return points[:TRACE_PLAN_MAX_POINTS]


def planned_bucket_trace_points():
    points = planned_bucket_points_from_dig_plan()
    if not points:
        plan = STATE.get("current_dig_plan")
        if isinstance(plan, dict):
            points = [np.array(p, dtype=np.float32).reshape(-1)[:3].copy() for p in plan.get("planned_path_points", [])]
    source = str(STATE.get("trace_plan_source", "dig_plan") or "dig_plan")
    if not points:
        source = "none"
    STATE["trace_planned_bucket_points"] = points
    STATE["trace_plan_source"] = source
    return points


def trace_points_signature(trace_points):
    if not trace_points:
        return ("empty", 0)
    source = str(STATE.get("trace_plan_source", "") or "")
    samples = []
    count = len(trace_points)
    for idx in [0, count // 2, count - 1]:
        try:
            p = np.array(trace_points[idx], dtype=np.float32).reshape(-1)[:3]
            samples.append(tuple(round(float(x), 3) for x in p))
        except Exception:
            samples.append(None)
    return (source, int(count), tuple(samples))


def draw_trace(force=False):
    mode = current_trace_mode()
    if mode == 0:
        hide_trace_prims()
        return

    now = time.time()
    dirty = bool(STATE.get("trace_render_dirty", False))
    if (not force) and (not dirty) and now - STATE["last_trace_time"] < STATE["trace_interval"]:
        return

    if mode == 2:
        trace_points = planned_bucket_trace_points()
        if len(trace_points) < TRACE_MIN_CURVE_POINTS:
            ensure_trace_prims()
            show_trace_prims(mode)
            set_trace_visibility([TRACE_REAL_CURVE_PATH, TRACE_PLAN_END_PATH], False)
            if not bool(STATE.get("trace_no_plan_notice_shown", False)):
                STATE["trace_no_plan_notice_shown"] = True
                STATE["trace_no_plan_notice_time"] = now
                update_status("[TRACE PLAN] no cached dig plan; press Build Dig Plan first", force=False)
            STATE["last_trace_time"] = now
            return
        signature = trace_points_signature(trace_points)
        if (not force) and (not dirty) and signature == STATE.get("trace_render_signature"):
            STATE["last_trace_time"] = now
            return
        ensure_trace_prims()
        show_trace_prims(mode)
        points = [Gf.Vec3f(float(x[0]), float(x[1]), float(x[2])) for x in trace_points]
        try:
            if dirty or signature != STATE.get("trace_render_signature"):
                configure_linear_trace_curve(TRACE_REAL_CURVE_PATH, (0.05, 0.35, 1.0), 0.080, points)
                STATE["trace_render_signature"] = signature
                STATE["trace_render_dirty"] = False
            set_trace_visibility([TRACE_REAL_CURVE_PATH, TRACE_PLAN_END_PATH], True)
            set_trace_visibility([TRACE_CURVE_PATH, TRACE_BUCKET_PATH, TRACE_TARGET_PATH], False)
        except Exception:
            pass
        try:
            if get_prim(TRACE_PLAN_END_PATH).IsValid():
                p = trace_points[-1]
                set_xform(get_prim(TRACE_PLAN_END_PATH), translate=(p[0], p[1], p[2]))
        except Exception:
            pass
        STATE["last_trace_time"] = now
        return

    if mode != 1:
        STATE["last_trace_time"] = now
        return

    ensure_trace_prims()
    show_trace_prims(mode)

    if not get_prim(TRACE_CURVE_PATH).IsValid():
        return

    p0 = bbox_center(BUCKET_LINK)
    p1 = get_target_pos()
    p1[2] = max(float(p1[2]), TARGET_MIN_Z)

    dig_points = STATE.get("dig_plan_points", None)
    if dig_points:
        trace_points = [p0] + [np.array(p, dtype=np.float32) for p in dig_points]
    else:
        trace_points = []
        for i in range(TRACE_COUNT):
            s = i / max(1, TRACE_COUNT - 1)
            trace_points.append((1.0 - s) * p0 + s * p1)

    points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in trace_points]

    try:
        configure_linear_trace_curve(TRACE_CURVE_PATH, (0.05, 1.0, 0.10), 0.035, points)
    except Exception:
        pass

    try:
        if get_prim(TRACE_BUCKET_PATH).IsValid():
            set_xform(get_prim(TRACE_BUCKET_PATH), translate=(p0[0], p0[1], p0[2]))
        if get_prim(TRACE_TARGET_PATH).IsValid():
            set_xform(get_prim(TRACE_TARGET_PATH), translate=(p1[0], p1[1], p1[2]))
    except Exception:
        pass

    STATE["last_trace_time"] = now


# ============================================================
# IK follow
# ============================================================

async def move_to(q_goal, seconds=1.0, mode="auto"):
    q0 = CTRL.q_cmd.copy()
    q1 = clip_command_near(q_goal, reference=q0)

    steps = max(1, int(seconds * CONTROL_HZ))

    for i in range(steps):
        s = (i + 1) / steps
        q = interpolate_q_shortest(q0, q1, s)
        CTRL.apply_target(q, mode=mode)
        await step_updates(1)
        check_freeze_state(f"move_to:{mode}")


def hold_real_state_after_verify_failure(q_real, mode):
    q_hold = CTRL.clip_limits(np.array(q_real, dtype=np.float32).copy())
    CTRL.q_cmd = q_hold.copy()
    CTRL.q_safe = q_hold.copy()
    try:
        CTRL.send_action(q_hold, mode=f"{mode}_verify_hold")
    except Exception as e:
        info_print("[WARN] verify hold action failed:", e)
    STATE["last_action_q"] = None
    STATE["last_action_unclipped_q"] = None
    STATE["last_action_time"] = 0.0
    STATE["last_action_mode"] = f"{mode}_verify_hold"


def set_execution_failure_reason(reason):
    reason = str(reason)
    STATE["last_execution_failure_reason"] = reason
    debug_timeline_record("EXEC_FAIL", result="failed", reason=reason, include_sand=True)
    dataset_record_event("execution_failure", reason)


def execution_failure_status_text(stage_name):
    reason = str(STATE.get("last_execution_failure_reason", "") or "execution_failed/stage_failed")
    return f"[DIG EXEC FAILED] {stage_name} could not complete: {reason}"


def sync_motion_start_q(label=""):
    try:
        q_real = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
        q_start = CTRL.clip_limits(np.array(q_real, dtype=np.float32).copy())
        CTRL.q_cmd = q_start.copy()
        CTRL.q_safe = q_start.copy()
        STATE["dataset_current_q_goal"] = q_start.copy()
        return q_start
    except Exception as e:
        info_print("[WARN] [MOVE START SYNC] failed:", label, type(e).__name__, e)
        return CTRL.q_cmd.copy()


def estimate_stage_motion_seconds(q0, q1, requested_seconds=0.0):
    sm = max(0.05, get_speed_multiplier())
    q0 = np.array(q0, dtype=np.float32)
    q1 = np.array(q1, dtype=np.float32)
    max_joint_seconds = 0.0
    for name, idx in CTRL.name_to_idx.items():
        if name == "swing":
            delta = abs(float(swing_delta(q1[idx], q0[idx])))
        else:
            delta = abs(float(q1[idx] - q0[idx]))
        max_joint_seconds = max(max_joint_seconds, delta / max(1.0e-6, float(DQ_MAX[name]) * sm))
    requested = max(0.0, float(requested_seconds)) / sm
    return max(0.08, requested, max_joint_seconds + MOVE_DURATION_MARGIN_SECONDS)


def motion_reach_report(q_goal):
    try:
        q_goal = np.array(q_goal, dtype=np.float32)
        q_real = q_real_near_command(get_real_joint_positions(), q_goal)
    except Exception as e:
        info_print("[WARN] [MOVE VERIFY] cannot read real joints:", type(e).__name__, e)
        return True, "joint_read_unavailable", [], 0.0, 0.0, None

    err_deg = q_delta_abs_deg(q_goal, q_real)
    swing_idx = CTRL.name_to_idx["swing"]
    swing_err = float(err_deg[swing_idx])
    max_err = max(err_deg) if err_deg else 0.0
    blocked = [
        f"{name}:{float(err_deg[idx]):.2f}deg"
        for name, idx in CTRL.name_to_idx.items()
        if (
            (name == "swing" and float(err_deg[idx]) > MOVE_FINAL_SWING_TOL_DEG)
            or (name != "swing" and float(err_deg[idx]) > MOVE_FINAL_JOINT_TOL_DEG)
        )
    ]

    if not blocked:
        return True, "ok", blocked, swing_err, max_err, q_real

    detail = (
        f"blocked_joints={blocked}; swing_err={swing_err:.2f}deg; max_err={max_err:.2f}deg; "
        f"cmd_deg={q_deg_values(q_goal)} real_deg={q_deg_values(q_real, wrap_swing_for_display=True)}"
    )
    return False, detail, blocked, swing_err, max_err, q_real


def verify_motion_reached(q_goal, label="", mode="auto", record_failure=True):
    ok, detail, _blocked, _swing_err, _max_err, q_real = motion_reach_report(q_goal)
    if ok:
        return True
    if record_failure:
        set_execution_failure_reason(f"execution_failed/path_deviation:{label}:{mode}:{detail}")
        info_print("[DIG EXEC FAILED]", f"label={label}", "reason=path_deviation", detail)
        info_print("[MOVE VERIFY FAILED]", f"label={label}", f"mode={mode}", detail)
        dataset_record_event("move_verify_failed", f"{label}:{mode}:{detail}")
        if q_real is not None:
            hold_real_state_after_verify_failure(q_real, mode)
    return False


async def wait_for_motion_reached(q_goal, label="", mode="auto", seconds_eff=0.0):
    max_frames = max(
        int(MOVE_REACH_WAIT_MAX_FRAMES),
        int(max(0.0, float(seconds_eff)) * CONTROL_HZ * MOVE_REACH_EXTRA_TIME_RATIO),
    )
    min_frames = max(0, int(MOVE_REACH_WAIT_MIN_FRAMES))
    last_detail = "not_checked"
    for frame in range(max_frames):
        if frame < min_frames:
            await step_updates(1)
            continue
        ok, detail, _blocked, _swing_err, _max_err, _q_real = motion_reach_report(q_goal)
        last_detail = detail
        if ok:
            return True
        if sand_contact_stage_can_advance(q_goal, label=label, mode=mode, seconds_eff=seconds_eff):
            return True
        if not robot_articulation_action_ready():
            await wait_for_articulation_action_ready(
                f"{label}_reach_wait",
                min_stable_frames=1,
                max_frames=15,
                record_failure=False,
            )
        await step_updates(1)
    info_print("[MOVE VERIFY WAIT TIMEOUT]", f"label={label}", f"mode={mode}", f"detail={last_detail}")
    return verify_motion_reached(q_goal, label=label, mode=mode, record_failure=True)


async def move_to_profile(q_goal, seconds=1.0, label="", task_id=None, mode="auto", q_start_override=None):
    ready, reason, _detail = await wait_for_articulation_action_ready(
        f"{label}_stage_start",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_STAGE_MAX_WAIT_FRAMES,
        record_failure=True,
    )
    if not ready:
        update_status(f"[DIG EXEC FAILED] {label}: action_channel_not_ready; {reason}", force=True)
        dataset_record_event("move_action_channel_not_ready", f"{label}:{mode}:{reason}")
        return False

    if q_start_override is None:
        q0 = sync_motion_start_q(label)
    else:
        q0 = CTRL.clip_limits(np.array(q_start_override, dtype=np.float32).copy())
        CTRL.q_cmd = q0.copy()
        CTRL.q_safe = q0.copy()
        STATE["dataset_current_q_goal"] = q0.copy()
    q1 = clip_command_near(q_goal, reference=q0)
    q_final_cmd = q1.copy()
    contact_stage_name = str(label or mode)
    if is_sand_contact_phase(contact_stage_name):
        start_sand_contact_stage(contact_stage_name)

    sm = get_speed_multiplier()
    apply_speed_to_physx_joint_limits()

    seconds_eff = estimate_stage_motion_seconds(q0, q1, requested_seconds=seconds)
    steps = max(4, int(seconds_eff * CONTROL_HZ))
    STATE["dataset_current_q_goal"] = q1.copy()
    STATE["trace_active_motion"] = {
        "label": str(label),
        "mode": str(mode),
        "q_goal": q1.copy(),
        "expires_at": time.time() + seconds_eff + 1.0,
    }
    if current_trace_mode() == 2:
        draw_trace(force=False)

    update_status(f"[MOVE] {label} phase={mode} duration={seconds_eff:.2f}s speed={sm:.2f}", force=True)

    carry_bucket_world_rad = None
    carry_max_err_rad = 0.0
    carry_limited = False
    level_mode = str(mode).lower()
    auto_carry_bucket = (
        ("lift_carry" in level_mode)
        or ("carry" in level_mode and "unload" not in level_mode)
        or ("unload_to_bin" in level_mode)
    )
    if auto_carry_bucket:
        start_angles = chain_angles_from_q(q0, end_effector="load")
        goal_angles = chain_angles_from_q(q1, end_effector="load")
        reference_angles = goal_angles if goal_angles is not None else start_angles
        if reference_angles is not None:
            carry_bucket_world_rad = float(nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q1))
            start_world = None if start_angles is None else rad_to_deg(start_angles[2])
            goal_world = None if goal_angles is None else rad_to_deg(goal_angles[2])
            info_print(
                f"[MOVE CARRY HOLD] {label}: target_load_world={rad_to_deg(carry_bucket_world_rad):.2f}deg "
                f"start_load_world={fmt_optional(start_world)}deg goal_load_world={fmt_optional(goal_world)}deg"
            )

    contact_advanced = False
    contact_advance_reason = ""

    for i in range(steps):
        if task_id is not None and not task_alive(task_id):
            update_status(f"[MOVE STOPPED] {label}", force=True)
            return False

        u = float(i + 1) / steps
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        q = interpolate_q_shortest(q0, q1, s)
        if carry_bucket_world_rad is not None:
            carry_calc = bucket_joint_for_world_angle(q, carry_bucket_world_rad, end_effector="load")
            if carry_calc is not None:
                raw_bucket = float(carry_calc["bucket"])
                q[CTRL.name_to_idx["bucket"]] = carry_calc["bucket"]
                q = CTRL.clip_limits(q)
                if abs(wrap_angle(float(q[CTRL.name_to_idx["bucket"]]) - raw_bucket)) > deg_to_rad(0.25):
                    carry_limited = True
                actual_angles = chain_angles_from_q(q, end_effector="load")
                if actual_angles is not None:
                    carry_err_rad = abs(wrap_angle(float(actual_angles[2]) - carry_bucket_world_rad))
                    carry_max_err_rad = max(carry_max_err_rad, carry_err_rad)
                    if carry_err_rad > deg_to_rad(BUCKET_CARRY_HOLD_TOL_DEG):
                        update_status(
                            f"[MOVE CARRY DIAG] {label}: load_world_err={rad_to_deg(carry_err_rad):.2f}deg "
                            f"bucket_limit={carry_limited}; executing and scoring actual carried sand",
                            force=True,
                        )

        q_final_cmd = q.copy()
        ok, reason = CTRL.apply_target_direct(q, mode=mode)
        if not ok:
            if "physics view not ready" in str(reason).lower() or "robot not ready" in str(reason).lower():
                ready, ready_reason, _ready_detail = await wait_for_articulation_action_ready(
                    f"{label}_during_move",
                    min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
                    max_frames=ACTION_READY_STAGE_MAX_WAIT_FRAMES,
                    record_failure=True,
                )
                if ready:
                    ok, reason = CTRL.apply_target_direct(q, mode=mode)
                if not ok:
                    update_status(f"[DIG EXEC FAILED] {label}: action_channel_not_ready; {ready_reason}", force=True)
                    dataset_record_event("move_action_channel_not_ready", f"{label}:{mode}:{ready_reason}")
                    return False
            if not ok:
                update_status(f"[MOVE BLOCKED] {reason}", force=True)
                set_execution_failure_reason(f"execution_failed/move_blocked:{label}:{mode}:{reason}")
                dataset_record_event("move_blocked", f"{label}:{mode}:{reason}")
                return False

        await step_updates(max(1, int(60 / CONTROL_HZ)))
        dataset_record_sample(mode, q_cmd=CTRL.q_cmd.copy(), label=label)
        notify_sand_site_tool_sample(mode)
        if is_sand_contact_phase(contact_stage_name):
            try:
                q_contact_real = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
            except Exception:
                q_contact_real = None
            contact_report = update_sand_contact_progress(
                contact_stage_name,
                q_cmd=CTRL.q_cmd.copy(),
                q_real=q_contact_real,
                force=False,
                log=True,
            )
            should_advance, advance_reason = sand_contact_stage_should_advance(
                contact_stage_name,
                contact_report,
                q_cmd=CTRL.q_cmd.copy(),
                q_real=q_contact_real,
                seconds_eff=seconds_eff,
            )
            if should_advance:
                contact_advanced = True
                contact_advance_reason = advance_reason
                if q_contact_real is not None:
                    q_final_cmd = CTRL.clip_limits(np.array(q_contact_real, dtype=np.float32).copy())
                    CTRL.send_action(q_final_cmd, mode=f"{mode}_sand_contact_advance")
                    STATE["dataset_current_q_goal"] = q_final_cmd.copy()
                info_print(
                    "[SAND CONTACT ADVANCE]",
                    f"stage={contact_stage_name}",
                    f"reason={advance_reason}",
                )
                debug_timeline_record(
                    "SAND_CONTACT_ADVANCE",
                    stage=contact_stage_name,
                    result="advance",
                    reason=advance_reason,
                    q_cmd=CTRL.q_cmd.copy(),
                    q_real=q_contact_real,
                    data={
                        "bucket_total": int(contact_report.get("total_bucket_delta", 0) or 0),
                        "pile_total": int(contact_report.get("total_pile_delta", 0) or 0),
                        "spill_total": int(contact_report.get("total_spill_delta", 0) or 0),
                        "tip_total": round(float(contact_report.get("total_tip_delta", 0.0) or 0.0), 4),
                    },
                    include_sand=True,
                )
                break
        check_freeze_state(f"move_profile:{mode}")

    if carry_bucket_world_rad is not None:
        final_angles = chain_angles_from_q(CTRL.q_cmd, end_effector="load")
        final_world = None if final_angles is None else rad_to_deg(final_angles[2])
        info_print(
            f"[MOVE CARRY DONE] {label}: target_load_world={rad_to_deg(carry_bucket_world_rad):.2f}deg "
            f"final_load_world={fmt_optional(final_world)}deg "
            f"max_carry_err={rad_to_deg(carry_max_err_rad):.2f}deg limited={carry_limited}"
        )

    if contact_advanced:
        update_status(
            f"[MOVE CONTACT ADVANCE] {label}: {contact_advance_reason}",
            force=False,
        )
        return True

    if not await wait_for_motion_reached(q_final_cmd, label=label, mode=mode, seconds_eff=seconds_eff):
        update_status(f"[DIG EXEC FAILED] {label}: path_deviation; real joints did not reach planned command", force=True)
        return False

    return True


# ============================================================
# Correct swing-center / reach calculation
# ============================================================

DIG_MIN_RADIUS = 0.35
DIG_MAX_RADIUS_FALLBACK = 12.0
DIG_REACH_MARGIN = 1.35

SWING_REBASE_MARGIN_DEG = 0.65
SWING_REBASE_EPS_DEG = 0.05

BUCKET_TIP_LOCAL = np.array([0.75, 0.0, -0.18], dtype=np.float32)
BUCKET_LOAD_LOCAL = np.array([0.35, 0.0, 0.08], dtype=np.float32)
BUCKET_POUR_LOCAL = np.array([
    float(SAND_BUCKET_LOCAL_MAX[0]) - 0.05,
    0.0,
    float(SAND_BUCKET_LOCAL_MAX[2]) - 0.25,
], dtype=np.float32)

IK_MAX_DQ = np.array([0.075, 0.070, 0.080, 0.090], dtype=np.float32)
IK_CALIBRATION_STEP = 0.08
IK_MIN_SEGMENT_LENGTH = 1e-4
IK_BUCKET_CANDIDATE_SPAN_DEG = 70.0
IK_BUCKET_CANDIDATE_COUNT = 17
DIG_IK_BUCKET_CANDIDATE_SPAN_DEG = 170.0
DIG_IK_BUCKET_CANDIDATE_COUNT = 17
IK_COST_WEIGHTS = np.array([0.35, 0.45, 0.55, 4.50], dtype=np.float32)
DIG_PLAN_BEAM_SIZE = 3
DIG_PLAN_TOPK_IK = 2
DIG_PLAN_MAX_CANDIDATES = 10
DIG_PLAN_PATH_CHECK_SAMPLES = 8
DIG_PLAN_MAX_BUILD_SECONDS = 3.5
DIG_PLAN_MOTION_WEIGHTS = np.array([1.15, 1.0, 0.9, 0.7], dtype=np.float32)
DIG_PLAN_ANGLE_COST_WEIGHT = 0.65
DIG_PLAN_TIME_COST_WEIGHT = 0.35
DIG_PLAN_IK_ERR_COST = 18.0
DIG_PLAN_PATH_SOFT_PENALTY = 8.0
DIG_PLAN_OBSTACLE_SOFT_PENALTY = 16.0
DIG_PLAN_UNLOAD_XY_COST = 22.0
DIG_PLAN_LIFT_SOFT_ACCEPT_ERR = 0.85
DIG_PLAN_UNLOAD_SOFT_ACCEPT_ERR = 0.70
DIG_PLAN_CUT_SOFT_ACCEPT_ERR = 0.44
IK_FOLLOW_MIN_CLEARANCE = 0.12
IK_DIG_MIN_CLEARANCE = 0.02
IK_ACCEPT_ERR = 0.35
IK_REFINE_ITERS = 10
IK_REFINE_LAMBDA = 0.035
IK_REFINE_MAX_STEP = 0.12
DIG_MAX_TIP_DEPTH = 0.28
DIG_MAX_CURL_DEPTH = 0.14
DIG_MAX_BUCKET_BODY_DEPTH = 0.10
DIG_ARM_MIN_CLEARANCE = 0.03
PATH_CHECK_SAMPLES = 24
PATH_CLEARANCE_BODY_Z = 0.06
PATH_CLEARANCE_END_Z = 0.18
PATH_CLEARANCE_HEIGHTS = [0.22, 0.38, 0.60, 0.85, 1.15, 1.55, 2.05, 2.75, 3.45]
PATH_CLEARANCE_FRACTIONS = [0.25, 0.40, 0.60, 0.75]
PATH_CLEARANCE_DURATION = 0.85
PATH_OBSTACLE_MARGIN_XY = 0.18
PATH_OBSTACLE_MARGIN_Z = 0.10
PATH_OBSTACLE_OVER_CLEARANCE_Z = 0.45
PATH_ROUTE_SIDE_OFFSETS = [0.65, 1.10, 1.65, 2.25]
PRE_DIG_SWING_ALIGN_THRESHOLD_DEG = 8.0
MOVE_FINAL_SWING_TOL_DEG = 3.0
MOVE_FINAL_JOINT_TOL_DEG = 8.0
PRE_DIG_TRAVEL_DEG = {
    "boom": 56.0,
    "arm": -62.0,
    "bucket": -20.0,
}
PRE_DIG_TRAVEL_SECONDS = 0.85
PRE_DIG_SWING_ALIGN_MIN_SECONDS = 0.75
PRE_DIG_SWING_ALIGN_MAX_SECONDS = 2.40
PRE_DIG_TRAVEL_SWING_FRACTION = 0.45
PRE_DIG_ALIGN_JOINT_BLEND = 0.35
PATH_RIGID_OBSTACLE_PATHS = [
    "/SandSite/SandRetainingWalls/WallLeft",
    "/SandSite/SandRetainingWalls/WallRight",
    "/SandSite/SandRetainingWalls/WallFront",
    "/SandSite/SandRetainingWalls/WallBack",
    "/SandSite/UnloadBin/WallLeft",
    "/SandSite/UnloadBin/WallRight",
    "/SandSite/UnloadBin/WallFront",
    "/SandSite/UnloadBin/WallBack",
]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def q_deg(swing_rad, boom_deg, arm_deg, bucket_deg):
    return np.array([
        float(swing_rad),
        deg_to_rad(boom_deg),
        deg_to_rad(arm_deg),
        deg_to_rad(bucket_deg),
    ], dtype=np.float32)


def get_joint_anchor_world(joint_name):
    """
    返回 USD Physics joint anchor 的 world position。
    """
    joint_path = JOINT_PATHS.get(joint_name)
    if joint_path is None:
        return None

    prim = get_prim(joint_path)
    if not prim or not prim.IsValid():
        return None

    try:
        body0_rel = prim.GetRelationship("physics:body0")
        targets = body0_rel.GetTargets()
        if not targets:
            return None

        body0_path = str(targets[0])
        body0_prim = get_prim(body0_path)
        if not body0_prim or not body0_prim.IsValid():
            return None

        local_pos_attr = prim.GetAttribute("physics:localPos0")
        local_pos = local_pos_attr.Get() if local_pos_attr else None

        if local_pos is None:
            local_pos = Gf.Vec3f(0.0, 0.0, 0.0)

        mat = UsdGeom.Xformable(body0_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        wp = mat.Transform(
            Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2]))
        )

        return np.array([float(wp[0]), float(wp[1]), float(wp[2])], dtype=np.float32)

    except Exception as e:
        info_print("[WARN] get_joint_anchor_world failed:", joint_name, e)
        return None


def get_swing_center_world():
    """
    挖掘机工作装置的水平回转中心。
    """
    p = get_joint_anchor_world("swing")
    if p is not None:
        return p

    if "swing_link" in LINK_PATHS:
        p = bbox_center(LINK_PATHS["swing_link"])
        if np.linalg.norm(p) > 1e-6:
            return p

    return bbox_center(LINK_PATHS["base_link"])


def get_swing_xy_center():
    p = get_swing_center_world()
    return np.array([float(p[0]), float(p[1])], dtype=np.float32)


def estimate_dynamic_reach_radius():
    """
    根据当前 bucket tip 距离估计模型工作半径。
    """
    try:
        center = get_swing_center_world()
        tip = transform_local_point_to_world(BUCKET_LINK, BUCKET_TIP_LOCAL)

        r_now = float(np.linalg.norm(tip[:2] - center[:2]))
        r_max = clamp(r_now * DIG_REACH_MARGIN, 4.0, DIG_MAX_RADIUS_FALLBACK)

        return r_max

    except Exception:
        return DIG_MAX_RADIUS_FALLBACK


def transform_local_point_to_world(link_path, local_xyz):
    prim = get_prim(link_path)
    if not prim or not prim.IsValid():
        return bbox_center(link_path)

    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p = Gf.Vec3d(float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2]))
    wp = mat.Transform(p)
    return np.array([float(wp[0]), float(wp[1]), float(wp[2])], dtype=np.float32)


def bucket_tip_pos():
    return transform_local_point_to_world(BUCKET_LINK, BUCKET_TIP_LOCAL)


def bucket_load_pos():
    return transform_local_point_to_world(BUCKET_LINK, BUCKET_LOAD_LOCAL)


def bucket_pour_pos():
    return transform_local_point_to_world(BUCKET_LINK, BUCKET_POUR_LOCAL)


def bucket_mid_pos():
    return bbox_center(BUCKET_LINK)


def is_cutting_phase(mode):
    m = str(mode).lower()
    return "insert" in m or "pull" in m or "cut" in m


def is_curl_phase(mode):
    return "curl" in str(mode).lower()


def is_calibrate_phase(mode):
    return "calibrate" in str(mode).lower()


def sand_surface_z_at_xy(x, y):
    api = get_sand_site_api()
    ctx = task_scene_context()
    center = np.array(ctx.get("pile_center", [0.0, 0.0, GROUND_TOP_Z]), dtype=np.float32).reshape(-1)[:3]
    radius = np.array(ctx.get("pile_radius", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
    if len(radius) >= 2 and float(radius[0]) > 0.0 and float(radius[1]) > 0.0:
        if abs(float(x) - float(center[0])) > float(radius[0]) or abs(float(y) - float(center[1])) > float(radius[1]):
            return None

    for fn_key in ["height_fn", "initial_height_fn"]:
        fn = api.get(fn_key) if isinstance(api, dict) else None
        if callable(fn):
            try:
                z = float(fn(float(x), float(y)))
                if math.isfinite(z):
                    return z
            except Exception:
                pass
    if len(center) >= 3:
        return float(center[2])
    return None


def sand_depth_for_point(point):
    if point is None:
        return None, None
    try:
        p = np.array(point, dtype=np.float32).reshape(-1)
        if len(p) < 3:
            return None, None
        surface_z = sand_surface_z_at_xy(float(p[0]), float(p[1]))
        if surface_z is None:
            return None, None
        return max(0.0, float(surface_z) - float(p[2])), float(surface_z)
    except Exception:
        return None, None


def phase_ground_report(mode):
    tip = bucket_tip_pos()
    load = bucket_load_pos()
    pour = bucket_pour_pos()
    bucket_mid = bucket_mid_pos()
    tip_sand_depth, tip_sand_surface = sand_depth_for_point(tip)
    load_sand_depth, load_sand_surface = sand_depth_for_point(load)
    pour_sand_depth, pour_sand_surface = sand_depth_for_point(pour)
    mid_sand_depth, mid_sand_surface = sand_depth_for_point(bucket_mid)
    return {
        "mode": str(mode),
        "base_min": bbox_min_z(LINK_PATHS["base_link"]),
        "boom_min": bbox_min_z(LINK_PATHS["boom_link"]) if "boom_link" in LINK_PATHS else None,
        "arm_min": bbox_min_z(LINK_PATHS["arm_link"]) if "arm_link" in LINK_PATHS else None,
        "bucket_min": bbox_min_z(BUCKET_LINK),
        "tip_z": None if tip is None else float(tip[2]),
        "load_z": None if load is None else float(load[2]),
        "pour_z": None if pour is None else float(pour[2]),
        "bucket_mid_z": None if bucket_mid is None else float(bucket_mid[2]),
        "tip_sand_depth": tip_sand_depth,
        "load_sand_depth": load_sand_depth,
        "pour_sand_depth": pour_sand_depth,
        "bucket_mid_sand_depth": mid_sand_depth,
        "tip_sand_surface_z": tip_sand_surface,
        "load_sand_surface_z": load_sand_surface,
        "pour_sand_surface_z": pour_sand_surface,
        "bucket_mid_sand_surface_z": mid_sand_surface,
    }


def min_existing(values):
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return min(valid)


def predicted_chain_world_z(q, end_effector="tip"):
    if IK_MODEL is None:
        return None

    boom = get_joint_anchor_world("boom")
    if boom is None:
        return None

    part = ik_model_part(end_effector=end_effector)
    if part is None:
        return None

    lengths = np.array(part.get("lengths", []), dtype=np.float32)
    if len(lengths) != 3 or float(np.min(lengths)) < IK_MIN_SEGMENT_LENGTH:
        chain = current_planar_chain(end_effector=end_effector)
        if chain is None:
            return None
        lengths = np.array(chain["lengths"], dtype=np.float32)

    pts = planar_points_from_q(
        CTRL.clip_limits(q),
        lengths,
        end_effector=end_effector,
    )
    if pts is None:
        return None

    root_z = float(boom[2])
    return [root_z + float(p[1]) for p in pts]


def predicted_chain_world_points(q, end_effector="mid", reference_q=None):
    if IK_MODEL is None:
        return None

    swing_anchor = get_joint_anchor_world("swing")
    boom = get_joint_anchor_world("boom")
    if swing_anchor is None or boom is None:
        return None

    part = ik_model_part(end_effector=end_effector)
    if part is None:
        return None

    lengths = np.array(part.get("lengths", []), dtype=np.float32)
    if len(lengths) != 3 or float(np.min(lengths)) < IK_MIN_SEGMENT_LENGTH:
        chain = current_planar_chain(end_effector=end_effector)
        if chain is None:
            return None
        lengths = np.array(chain["lengths"], dtype=np.float32)

    q = CTRL.clip_limits(q)
    if reference_q is None:
        reference_q = CTRL.q_cmd

    swing_idx = CTRL.name_to_idx["swing"]
    swing = float(q[swing_idx])
    ref_swing = float(reference_q[swing_idx])

    boom_offset_xy = boom[:2] - swing_anchor[:2]
    boom_xy = swing_anchor[:2] + rotate_xy(boom_offset_xy, swing - ref_swing)
    boom_root = np.array([boom_xy[0], boom_xy[1], float(boom[2])], dtype=np.float32)
    radial = safe_norm(np.array([math.cos(swing), math.sin(swing)], dtype=np.float32), default=(1.0, 0.0))

    planar = planar_points_from_q(q, lengths, end_effector=end_effector)
    if planar is None:
        return None

    points = []
    for p in planar:
        points.append(
            np.array(
                [
                    float(boom_root[0]) + float(radial[0]) * float(p[0]),
                    float(boom_root[1]) + float(radial[1]) * float(p[0]),
                    float(boom_root[2]) + float(p[1]),
                ],
                dtype=np.float32,
            )
        )
    return points


def predicted_end_world_point(q, end_effector="mid", reference_q=None):
    pts = predicted_chain_world_points(q, end_effector=end_effector, reference_q=reference_q)
    if pts is None or len(pts) == 0:
        return None
    return pts[-1]


def predicted_phase_ground_report(q, mode):
    tip_pts = predicted_chain_world_points(q, "tip")
    mid_pts = predicted_chain_world_points(q, "mid")
    load_pts = predicted_chain_world_points(q, "load")
    pour_pts = predicted_chain_world_points(q, "pour")
    tip_zs = predicted_chain_world_z(q, "tip")
    mid_zs = predicted_chain_world_z(q, "mid")
    load_zs = predicted_chain_world_z(q, "load")
    pour_zs = predicted_chain_world_z(q, "pour")

    def z_at(zs, idx):
        if zs is None or len(zs) <= idx:
            return None
        return float(zs[idx])

    boom_root_z = z_at(tip_zs, 0)
    arm_joint_z = z_at(tip_zs, 1)
    bucket_joint_z = z_at(tip_zs, 2)
    tip_z = z_at(tip_zs, 3)
    mid_z = z_at(mid_zs, 3)
    load_z = z_at(load_zs, 3)
    pour_z = z_at(pour_zs, 3)
    tip_point = None if tip_pts is None or len(tip_pts) <= 3 else tip_pts[3]
    mid_point = None if mid_pts is None or len(mid_pts) <= 3 else mid_pts[3]
    load_point = None if load_pts is None or len(load_pts) <= 3 else load_pts[3]
    pour_point = None if pour_pts is None or len(pour_pts) <= 3 else pour_pts[3]
    tip_sand_depth, tip_sand_surface = sand_depth_for_point(tip_point)
    load_sand_depth, load_sand_surface = sand_depth_for_point(load_point)
    pour_sand_depth, pour_sand_surface = sand_depth_for_point(pour_point)
    mid_sand_depth, mid_sand_surface = sand_depth_for_point(mid_point)

    base_path = LINK_PATHS.get("base_link")

    return {
        "mode": str(mode),
        "base_min": bbox_min_z(base_path) if base_path else None,
        "boom_min": min_existing([boom_root_z, arm_joint_z]),
        "arm_min": min_existing([arm_joint_z, bucket_joint_z]),
        "bucket_min": min_existing([bucket_joint_z, tip_z, mid_z, load_z, pour_z]),
        "tip_z": tip_z,
        "load_z": load_z,
        "pour_z": pour_z,
        "bucket_mid_z": mid_z,
        "tip_sand_depth": tip_sand_depth,
        "load_sand_depth": load_sand_depth,
        "pour_sand_depth": pour_sand_depth,
        "bucket_mid_sand_depth": mid_sand_depth,
        "tip_sand_surface_z": tip_sand_surface,
        "load_sand_surface_z": load_sand_surface,
        "pour_sand_surface_z": pour_sand_surface,
        "bucket_mid_sand_surface_z": mid_sand_surface,
    }


def depth_below_ground(z):
    if z is None:
        return None
    return max(0.0, GROUND_TOP_Z - float(z))


def fmt_optional(x):
    return "None" if x is None else f"{float(x):.3f}"


def format_ground_report(mode, report, reason):
    tip_hard_depth = depth_below_ground(report.get("tip_z"))
    load_hard_depth = depth_below_ground(report.get("load_z"))
    pour_hard_depth = depth_below_ground(report.get("pour_z"))
    mid_hard_depth = depth_below_ground(report.get("bucket_mid_z"))
    if mode == "manual":
        phase_kind = "manual"
    elif is_cutting_phase(mode):
        phase_kind = "cut"
    elif is_curl_phase(mode):
        phase_kind = "curl"
    else:
        phase_kind = "free"
    return (
        f"phase={mode} kind={phase_kind} {reason}; "
        f"tip_z={fmt_optional(report.get('tip_z'))} tip_hard_depth={fmt_optional(tip_hard_depth)} tip_sand_depth={fmt_optional(report.get('tip_sand_depth'))} "
        f"load_z={fmt_optional(report.get('load_z'))} load_hard_depth={fmt_optional(load_hard_depth)} load_sand_depth={fmt_optional(report.get('load_sand_depth'))} "
        f"pour_z={fmt_optional(report.get('pour_z'))} pour_hard_depth={fmt_optional(pour_hard_depth)} pour_sand_depth={fmt_optional(report.get('pour_sand_depth'))} "
        f"bucket_mid_z={fmt_optional(report.get('bucket_mid_z'))} mid_hard_depth={fmt_optional(mid_hard_depth)} mid_sand_depth={fmt_optional(report.get('bucket_mid_sand_depth'))} "
        f"sand_surface_z={fmt_optional(report.get('tip_sand_surface_z'))} "
        f"bucket_min={fmt_optional(report.get('bucket_min'))} "
        f"arm_min={fmt_optional(report.get('arm_min'))} boom_min={fmt_optional(report.get('boom_min'))}"
    )


def log_phase_ground(prefix, mode):
    report = phase_ground_report(mode)
    info_print(f"{prefix} " + format_ground_report(mode, report, "check"))


def log_predicted_phase_ground(prefix, mode, q):
    report = predicted_phase_ground_report(q, mode)
    if report.get("tip_z") is None:
        ok, reason = False, "missing predicted phase report"
    else:
        ok, reason = phase_ground_ok(mode, report)
    info_print(f"{prefix} " + format_ground_report(mode, report, reason))
    return ok, reason, report


def phase_ground_ok(mode, report):
    arm_min = report.get("arm_min")
    bucket_min = report.get("bucket_min")
    tip_depth = depth_below_ground(report.get("tip_z"))
    load_depth = depth_below_ground(report.get("load_z"))

    arm_clearance = -0.02 if (is_cutting_phase(mode) or is_curl_phase(mode)) else DIG_ARM_MIN_CLEARANCE

    if (not is_calibrate_phase(mode)) and arm_min is not None and arm_min < GROUND_TOP_Z + arm_clearance:
        return False, "arm too low"
    # boom_min is kept in diagnostics, but not used as a hard blocker. It is
    # often the boom root / bbox low point, so it can be below the parking
    # plane while the actionable bucket/arm path is still clear.

    if is_cutting_phase(mode):
        if tip_depth is not None and tip_depth > DIG_MAX_TIP_DEPTH:
            return False, f"tip below hard floor too deep {tip_depth:.3f}"
        if load_depth is not None and load_depth > DIG_MAX_BUCKET_BODY_DEPTH:
            return False, f"bucket body below hard floor too deep {load_depth:.3f}"
        if bucket_min is not None and bucket_min < GROUND_TOP_Z - (DIG_MAX_TIP_DEPTH + 0.12):
            return False, "bucket bbox below hard floor too deep"
        return True, "ok"

    if is_curl_phase(mode):
        if tip_depth is not None and tip_depth > DIG_MAX_CURL_DEPTH:
            return False, f"curl below hard floor too deep {tip_depth:.3f}"
        if load_depth is not None and load_depth > DIG_MAX_BUCKET_BODY_DEPTH:
            return False, f"bucket body below hard floor too deep {load_depth:.3f}"
        return True, "ok"

    if tip_depth is not None and tip_depth > 0.03:
        return False, f"tip penetrates hard floor {tip_depth:.3f}"
    if bucket_min is not None and bucket_min < GROUND_TOP_Z - 0.08:
        return False, "bucket below hard floor in free-space phase"

    return True, "ok"


def path_phase_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES):
    samples = max(2, int(samples))
    last_report = None

    for i in range(1, samples + 1):
        s = float(i) / float(samples)
        q = interpolate_q_shortest(q_start, q_goal, s)
        report = predicted_phase_ground_report(q, mode)
        last_report = report

        if report.get("tip_z") is None:
            return False, "missing predicted phase report", i, report

        ok, reason = phase_ground_ok(mode, report)
        if not ok:
            return False, reason, i, report

    return True, "ok", samples, last_report


def rigid_obstacle_bboxes():
    roots = ["/SandSite", "/World/SandSite"]
    bboxes = []
    for base_path in PATH_RIGID_OBSTACLE_PATHS:
        suffix = base_path[len("/SandSite"):] if base_path.startswith("/SandSite") else base_path
        for root in roots:
            path = f"{root}{suffix}" if suffix.startswith("/") else f"{root}/{suffix}"
            prim = get_prim(path)
            if not prim or not prim.IsValid():
                continue
            if not prim_collision_enabled(prim):
                continue
            mn, mx = bbox_min_max(path)
            if mn is None or mx is None:
                continue
            if not bbox_values_are_valid(mn, mx):
                continue
            bboxes.append({
                "path": path,
                "min": np.array(mn, dtype=np.float32),
                "max": np.array(mx, dtype=np.float32),
            })
    return bboxes


def point_inside_expanded_bbox(point, mn, mx, margin_xy=0.0, margin_z=0.0):
    p = np.array(point, dtype=np.float32)
    lo = np.array(mn, dtype=np.float32).copy()
    hi = np.array(mx, dtype=np.float32).copy()
    lo[0] -= float(margin_xy)
    lo[1] -= float(margin_xy)
    hi[0] += float(margin_xy)
    hi[1] += float(margin_xy)
    lo[2] -= float(margin_z)
    hi[2] += float(margin_z)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def predicted_obstacle_check_points(q, reference_q=None):
    points = []
    for end_effector in ["tip", "mid", "load", "pour"]:
        chain_points = predicted_chain_world_points(q, end_effector=end_effector, reference_q=reference_q)
        if chain_points is None:
            continue
        for p in chain_points:
            if p is None:
                continue
            points.append(np.array(p, dtype=np.float32))

    unique = []
    seen = set()
    for p in points:
        key = tuple(round(float(v), 3) for v in p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def format_obstacle_report(mode, report, reason):
    point = report.get("point") if isinstance(report, dict) else None
    bbox_min = report.get("bbox_min") if isinstance(report, dict) else None
    bbox_max = report.get("bbox_max") if isinstance(report, dict) else None
    point_text = "None" if point is None else f"({float(point[0]):.3f},{float(point[1]):.3f},{float(point[2]):.3f})"
    min_text = "None" if bbox_min is None else f"({float(bbox_min[0]):.3f},{float(bbox_min[1]):.3f},{float(bbox_min[2]):.3f})"
    max_text = "None" if bbox_max is None else f"({float(bbox_max[0]):.3f},{float(bbox_max[1]):.3f},{float(bbox_max[2]):.3f})"
    obstacle = report.get("obstacle", "unknown") if isinstance(report, dict) else "unknown"
    return (
        f"phase={mode} kind=rigid_obstacle {reason}; "
        f"obstacle={obstacle} point={point_text} bbox_min={min_text} bbox_max={max_text}"
    )


def actual_rigid_obstacle_contact_detail():
    obstacles = rigid_obstacle_bboxes()
    if not obstacles:
        return ""

    points = [
        ("tip", bucket_tip_pos()),
        ("mid", bucket_mid_pos()),
        ("load", bucket_load_pos()),
        ("pour", bucket_pour_pos()),
    ]
    for point_name, point in points:
        if point is None:
            continue
        for obstacle in obstacles:
            if point_inside_expanded_bbox(
                point,
                obstacle["min"],
                obstacle["max"],
                margin_xy=PATH_OBSTACLE_MARGIN_XY,
                margin_z=PATH_OBSTACLE_MARGIN_Z,
            ):
                p = np.array(point, dtype=np.float32)
                return (
                    f"rigid_obstacle_overlap={obstacle['path']} "
                    f"bucket_point={point_name} "
                    f"point=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f})"
                )
    return ""


def path_obstacle_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES):
    obstacles = rigid_obstacle_bboxes()
    if not obstacles:
        return True, "ok", samples, None

    samples = max(2, int(samples))
    for i in range(1, samples + 1):
        s = float(i) / float(samples)
        q = interpolate_q_shortest(q_start, q_goal, s)
        points = predicted_obstacle_check_points(q, reference_q=q_start)
        for p in points:
            for obstacle in obstacles:
                if point_inside_expanded_bbox(
                    p,
                    obstacle["min"],
                    obstacle["max"],
                    margin_xy=PATH_OBSTACLE_MARGIN_XY,
                    margin_z=PATH_OBSTACLE_MARGIN_Z,
                ):
                    report = {
                        "mode": str(mode),
                        "obstacle": obstacle["path"],
                        "point": p,
                        "bbox_min": obstacle["min"],
                        "bbox_max": obstacle["max"],
                    }
                    return False, "predicted rigid obstacle contact", i, report

    return True, "ok", samples, None


def path_segment_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES):
    ok, reason, sample, report = path_phase_check(q_start, q_goal, mode, samples=samples)
    if not ok:
        return False, "phase", reason, sample, report

    ok, reason, sample, report = path_obstacle_check(q_start, q_goal, mode, samples=samples)
    if not ok:
        return False, "obstacle", reason, sample, report

    return True, "ok", "ok", samples, report


def path_block_report_text(mode, kind, report, reason):
    if kind == "obstacle":
        return format_obstacle_report(mode, report, reason)
    return format_ground_report(mode, report, reason)


def obstacle_top_z_for_segment(p_start, p_goal):
    obstacles = rigid_obstacle_bboxes()
    if not obstacles or p_start is None or p_goal is None:
        return None

    a = np.array(p_start, dtype=np.float32)
    b = np.array(p_goal, dtype=np.float32)
    seg_min = np.minimum(a[:2], b[:2]) - PATH_OBSTACLE_MARGIN_XY
    seg_max = np.maximum(a[:2], b[:2]) + PATH_OBSTACLE_MARGIN_XY
    top = None
    for obstacle in obstacles:
        mn = obstacle["min"]
        mx = obstacle["max"]
        overlap = (
            float(seg_max[0]) >= float(mn[0] - PATH_OBSTACLE_MARGIN_XY)
            and float(seg_min[0]) <= float(mx[0] + PATH_OBSTACLE_MARGIN_XY)
            and float(seg_max[1]) >= float(mn[1] - PATH_OBSTACLE_MARGIN_XY)
            and float(seg_min[1]) <= float(mx[1] + PATH_OBSTACLE_MARGIN_XY)
        )
        if overlap:
            top = float(mx[2]) if top is None else max(top, float(mx[2]))
    return top


def wrap_angle(x):
    y = (float(x) + math.pi) % (2.0 * math.pi) - math.pi
    return y


def normalize_swing_cmd(angle):
    return float(angle)


def swing_delta(target, current):
    return wrap_angle(float(target) - float(current))


def swing_target_near(target, current):
    current = float(current)
    return current + swing_delta(target, current)


def set_robot_joint_state(q, label="state"):
    if ROBOT is None:
        info_print(f"[WARN] {label}: ROBOT not ready for joint state set")
        return False
    if JOINT_INDICES is None:
        info_print(f"[WARN] {label}: JOINT_INDICES not ready for joint state set")
        return False
    if not robot_articulation_action_ready():
        info_print(f"[WARN] {label}: physics view not ready for joint state set")
        return False

    q_state = CTRL.clip_action_limits(q)
    try:
        ROBOT.set_joint_positions(q_state, joint_indices=JOINT_INDICES)
    except TypeError:
        try:
            ROBOT.set_joint_positions(positions=q_state, joint_indices=JOINT_INDICES)
        except Exception as e:
            info_print(f"[ERROR] {label}: set_joint_positions failed:", e)
            return False
    except Exception as e:
        info_print(f"[ERROR] {label}: set_joint_positions failed:", e)
        return False

    try:
        ROBOT.set_joint_velocities(np.zeros(len(q_state), dtype=np.float32), joint_indices=JOINT_INDICES)
    except Exception:
        pass

    try:
        ROBOT.apply_action(
            ArticulationAction(
                joint_positions=q_state,
                joint_indices=JOINT_INDICES,
            )
        )
    except Exception:
        pass

    return True


def set_joint_pose_direct(q_goal, label="direct_pose", mode="direct_pose", update_ui=True):
    q = clip_command_near(q_goal, reference=CTRL.q_cmd)
    q = CTRL.clip_limits(q)
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["trace_active_motion"] = None
    STATE["dataset_current_q_goal"] = q.copy()

    state_ok = set_robot_joint_state(q, label=label)
    CTRL.q_cmd = q.copy()
    CTRL.q_safe = q.copy()
    action_ok, action_reason = CTRL.send_action(q, mode=mode)
    ok = bool(state_ok or action_ok)

    if update_ui:
        try:
            sync_sliders_from_real_q(force=True)
        except Exception:
            pass

    info_print(
        "[DIRECT POSE]",
        f"label={label}",
        f"mode={mode}",
        f"ok={ok}",
        f"state_set={state_ok}",
        f"action_set={action_ok}",
        f"reason={action_reason}",
        f"q_deg={q_deg_values(q, wrap_swing_for_display=True)}",
    )
    return ok


async def set_joint_pose_direct_and_settle(q_goal, label="direct_pose", mode="direct_pose", settle_frames=0, task_id=None):
    ready, reason, _detail = await wait_for_articulation_action_ready(
        f"{label}_direct_pose",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_STAGE_MAX_WAIT_FRAMES,
        record_failure=False,
    )
    if not ready:
        update_status(f"[DIRECT POSE BLOCKED] {label}: action_channel_not_ready; {reason}", force=True)
        return False
    ok = set_joint_pose_direct(q_goal, label=label, mode=mode, update_ui=True)
    frames = max(0, int(settle_frames))
    for _ in range(frames):
        if task_id is not None and not task_alive(task_id):
            return False
        await step_updates(1)
    if ok:
        reached = await wait_for_motion_reached(
            CTRL.q_cmd.copy(),
            label=label,
            mode=mode,
            seconds_eff=max(0.25, frames / 60.0),
        )
        ok = bool(reached)
    return bool(ok and (task_id is None or task_alive(task_id)))


def maybe_prepare_swing_rebase_for_segment(q_goal, label="segment"):
    q_goal = clip_command_near(q_goal, reference=CTRL.q_cmd)
    idx = CTRL.name_to_idx["swing"]
    lo, hi = FINAL_LIMITS_RAD["swing"]
    margin = deg_to_rad(SWING_REBASE_MARGIN_DEG)
    eps = deg_to_rad(SWING_REBASE_EPS_DEG)
    target = float(q_goal[idx])

    if lo <= target <= hi:
        return q_goal, False

    try:
        q_real = get_real_joint_positions()
        real = float(q_real[idx])
    except Exception:
        real = float(CTRL.clip_action_limits(CTRL.q_cmd.copy())[idx])

    cmd_action = float(CTRL.clip_action_limits(CTRL.q_cmd.copy())[idx])

    if target < lo and (real <= lo + margin or cmd_action <= lo + margin):
        q_state = CTRL.clip_action_limits(CTRL.q_cmd.copy())
        q_state[idx] = hi - eps
        if set_robot_joint_state(q_state, "SWING PRE-REBASE LO_TO_HI"):
            CTRL.q_cmd = q_state.copy()
            CTRL.q_safe = q_state.copy()
            q_new = clip_command_near(q_goal, reference=CTRL.q_cmd)
            info_print(
                f"[SWING PRE-REBASE] {label}: low->high real={rad_to_deg(real):.2f}deg "
                f"cmd={rad_to_deg(cmd_action):.2f}deg old_target={rad_to_deg(target):.2f}deg "
                f"state={rad_to_deg(q_state[idx]):.2f}deg new_target={rad_to_deg(q_new[idx]):.2f}deg"
            )
            return q_new, True
        return q_goal, False

    if target > hi and (real >= hi - margin or cmd_action >= hi - margin):
        q_state = CTRL.clip_action_limits(CTRL.q_cmd.copy())
        q_state[idx] = lo + eps
        if set_robot_joint_state(q_state, "SWING PRE-REBASE HI_TO_LO"):
            CTRL.q_cmd = q_state.copy()
            CTRL.q_safe = q_state.copy()
            q_new = clip_command_near(q_goal, reference=CTRL.q_cmd)
            info_print(
                f"[SWING PRE-REBASE] {label}: high->low real={rad_to_deg(real):.2f}deg "
                f"cmd={rad_to_deg(cmd_action):.2f}deg old_target={rad_to_deg(target):.2f}deg "
                f"state={rad_to_deg(q_state[idx]):.2f}deg new_target={rad_to_deg(q_new[idx]):.2f}deg"
            )
            return q_new, True
        return q_goal, False

    return q_goal, False


def swing_edge_pose_before_rebase(q_goal, label="segment"):
    q_goal = clip_command_near(q_goal, reference=CTRL.q_cmd)
    idx = CTRL.name_to_idx["swing"]
    lo, hi = FINAL_LIMITS_RAD["swing"]
    margin = deg_to_rad(SWING_REBASE_MARGIN_DEG)
    eps = deg_to_rad(SWING_REBASE_EPS_DEG)
    target = float(q_goal[idx])

    if lo <= target <= hi:
        return None

    q_action_now = CTRL.clip_action_limits(CTRL.q_cmd.copy())
    current = float(q_action_now[idx])

    if target > hi and current < hi - margin:
        q_edge = q_goal.copy()
        q_edge[idx] = hi - eps
        info_print(
            f"[SWING EDGE] {label}: high current={rad_to_deg(current):.2f}deg "
            f"target={rad_to_deg(target):.2f}deg edge={rad_to_deg(q_edge[idx]):.2f}deg"
        )
        return q_edge

    if target < lo and current > lo + margin:
        q_edge = q_goal.copy()
        q_edge[idx] = lo + eps
        info_print(
            f"[SWING EDGE] {label}: low current={rad_to_deg(current):.2f}deg "
            f"target={rad_to_deg(target):.2f}deg edge={rad_to_deg(q_edge[idx]):.2f}deg"
        )
        return q_edge

    return None


def maybe_rebase_swing_for_bounded_joint(q_target):
    q = np.array(q_target, dtype=np.float32).copy()
    idx = CTRL.name_to_idx["swing"]
    lo, hi = FINAL_LIMITS_RAD["swing"]
    margin = deg_to_rad(SWING_REBASE_MARGIN_DEG)
    eps = deg_to_rad(SWING_REBASE_EPS_DEG)
    target = float(q[idx])

    if lo <= target <= hi:
        return q

    try:
        q_real = get_real_joint_positions()
        real = float(q_real[idx])
    except Exception:
        real = float(CTRL.q_cmd[idx])

    if target < lo and real <= lo + margin:
        q_state = CTRL.clip_action_limits(CTRL.q_cmd.copy())
        q_state[idx] = hi - eps
        if set_robot_joint_state(q_state, "SWING REBASE LO_TO_HI"):
            rebased_target = swing_target_near(target, q_state[idx])
            q[idx] = rebased_target
            CTRL.q_cmd[idx] = q_state[idx]
            info_print(
                f"[SWING REBASE] low->high real={rad_to_deg(real):.2f}deg "
                f"state={rad_to_deg(q_state[idx]):.2f}deg target={rad_to_deg(target):.2f}deg "
                f"new_target={rad_to_deg(q[idx]):.2f}deg"
            )
        return q

    if target > hi and real >= hi - margin:
        q_state = CTRL.clip_action_limits(CTRL.q_cmd.copy())
        q_state[idx] = lo + eps
        if set_robot_joint_state(q_state, "SWING REBASE HI_TO_LO"):
            rebased_target = swing_target_near(target, q_state[idx])
            q[idx] = rebased_target
            CTRL.q_cmd[idx] = q_state[idx]
            info_print(
                f"[SWING REBASE] high->low real={rad_to_deg(real):.2f}deg "
                f"state={rad_to_deg(q_state[idx]):.2f}deg target={rad_to_deg(target):.2f}deg "
                f"new_target={rad_to_deg(q[idx]):.2f}deg"
            )
        return q

    return q


def clip_command_near(q, reference=None):
    q = CTRL.clip_limits(q)
    if reference is None:
        reference = CTRL.q_cmd
    swing_idx = CTRL.name_to_idx["swing"]
    q[swing_idx] = swing_target_near(q[swing_idx], reference[swing_idx])
    q[swing_idx] = normalize_swing_cmd(q[swing_idx])
    return q


def interpolate_q_shortest(q0, q1, s):
    q0 = np.array(q0, dtype=np.float32)
    q1 = clip_command_near(q1, reference=q0)
    q = (1.0 - float(s)) * q0 + float(s) * q1
    swing_idx = CTRL.name_to_idx["swing"]
    q[swing_idx] = float(q0[swing_idx]) + float(s) * swing_delta(q1[swing_idx], q0[swing_idx])
    q[swing_idx] = normalize_swing_cmd(q[swing_idx])
    return CTRL.clip_limits(q)


def rotate_xy(v, angle):
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([
        c * float(v[0]) - s * float(v[1]),
        s * float(v[0]) + c * float(v[1]),
    ], dtype=np.float32)


def safe_norm(v, default=None):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        if default is None:
            return np.zeros_like(v, dtype=np.float32)
        return np.array(default, dtype=np.float32)
    return np.array(v, dtype=np.float32) / n


def point_to_2d(point, root, radial_xy):
    d = np.array(point, dtype=np.float32) - np.array(root, dtype=np.float32)
    x = float(d[0] * radial_xy[0] + d[1] * radial_xy[1])
    z = float(d[2])
    return np.array([x, z], dtype=np.float32)


def ik_end_effector_pos(end_effector="mid"):
    if end_effector == "tip":
        return bucket_tip_pos()
    if end_effector == "load":
        return bucket_load_pos()
    if end_effector == "pour":
        return bucket_pour_pos()
    return bucket_mid_pos()


def get_ik_world_points(end_effector="mid"):
    return {
        "swing": get_joint_anchor_world("swing"),
        "boom": get_joint_anchor_world("boom"),
        "arm": get_joint_anchor_world("arm"),
        "bucket": get_joint_anchor_world("bucket"),
        "end": ik_end_effector_pos(end_effector),
    }


def chain_angles_2d(points_2d):
    angles = []
    for i in range(len(points_2d) - 1):
        v = np.array(points_2d[i + 1], dtype=np.float32) - np.array(points_2d[i], dtype=np.float32)
        angles.append(math.atan2(float(v[1]), float(v[0])))
    return angles


def current_planar_chain(end_effector="mid"):
    pts = get_ik_world_points(end_effector=end_effector)
    boom = pts["boom"]
    arm = pts["arm"]
    bucket = pts["bucket"]
    end = pts["end"]

    if boom is None or arm is None or bucket is None or end is None:
        return None

    radial = safe_norm((arm - boom)[:2], default=(1.0, 0.0))
    p2 = [
        np.array([0.0, 0.0], dtype=np.float32),
        point_to_2d(arm, boom, radial),
        point_to_2d(bucket, boom, radial),
        point_to_2d(end, boom, radial),
    ]
    lengths = [
        float(np.linalg.norm(p2[1] - p2[0])),
        float(np.linalg.norm(p2[2] - p2[1])),
        float(np.linalg.norm(p2[3] - p2[2])),
    ]
    if min(lengths) < IK_MIN_SEGMENT_LENGTH:
        return None

    return {
        "world": pts,
        "root": boom,
        "radial": radial,
        "points_2d": p2,
        "lengths": lengths,
        "angles": chain_angles_2d(p2),
    }


def build_ik_offsets(q, chain, signs):
    a1, a2, a3 = chain["angles"]
    signs = np.array(signs, dtype=np.float32)
    return np.array([
        wrap_angle(a1 - signs[0] * float(q[CTRL.name_to_idx["boom"]])),
        wrap_angle((a2 - a1) - signs[1] * float(q[CTRL.name_to_idx["arm"]])),
        wrap_angle((a3 - a2) - signs[2] * float(q[CTRL.name_to_idx["bucket"]])),
    ], dtype=np.float32)


def ik_model_part(model=None, end_effector="mid"):
    if model is None:
        model = IK_MODEL
    if model is None:
        return None
    if "effectors" in model:
        part = model["effectors"].get(end_effector)
        if part is not None:
            return part
        return model["effectors"].get("mid")
    return model


def chain_angles_from_q(q, model=None, end_effector="mid"):
    part = ik_model_part(model=model, end_effector=end_effector)
    if part is None:
        return None

    signs = np.array((model or IK_MODEL)["signs"], dtype=np.float32)
    offsets = np.array(part["offsets"], dtype=np.float32)

    a1 = float(offsets[0] + signs[0] * float(q[CTRL.name_to_idx["boom"]]))
    a2 = float(a1 + offsets[1] + signs[1] * float(q[CTRL.name_to_idx["arm"]]))
    a3 = float(a2 + offsets[2] + signs[2] * float(q[CTRL.name_to_idx["bucket"]]))
    return [a1, a2, a3]


def bucket_joint_for_world_angle(q, world_angle_rad, model=None, end_effector="load"):
    if model is None:
        model = IK_MODEL
    part = ik_model_part(model=model, end_effector=end_effector)
    if part is None or model is None:
        return None

    angles = chain_angles_from_q(q, model=model, end_effector=end_effector)
    if angles is None:
        return None

    signs = np.array(model["signs"], dtype=np.float32)
    offsets = np.array(part["offsets"], dtype=np.float32)
    bucket = float(signs[2]) * wrap_angle(
        (float(world_angle_rad) - float(angles[1])) - float(offsets[2])
    )
    return {
        "bucket": bucket,
        "parent_angle": float(angles[1]),
        "offset": float(offsets[2]),
        "sign": float(signs[2]),
    }


def nearest_bucket_level_world_angle(reference_rad):
    base = deg_to_rad(BUCKET_LIFT_LEVEL_WORLD_DEG)
    candidates = [base + float(k) * math.pi for k in range(-3, 4)]
    return min(candidates, key=lambda a: abs(wrap_angle(float(a) - float(reference_rad))))


def bucket_mouth_raise_for_world_angle(q_reference, world_angle_rad, end_effector="load"):
    if q_reference is None:
        return None
    q = np.array(q_reference, dtype=np.float32).copy()
    calc = bucket_joint_for_world_angle(q, world_angle_rad, end_effector=end_effector)
    if calc is None:
        return None
    bucket_idx = CTRL.name_to_idx["bucket"]
    raw_bucket = float(calc["bucket"])
    q[bucket_idx] = raw_bucket
    q = CTRL.clip_limits(q)
    limited = abs(wrap_angle(float(q[bucket_idx]) - raw_bucket)) > deg_to_rad(0.25)
    load = predicted_end_world_point(q, end_effector="load", reference_q=q_reference)
    pour = predicted_end_world_point(q, end_effector="pour", reference_q=q_reference)
    if load is None or pour is None:
        return None
    load = np.array(load, dtype=np.float32).reshape(-1)[:3]
    pour = np.array(pour, dtype=np.float32).reshape(-1)[:3]
    return {
        "q": q,
        "load": load,
        "pour": pour,
        "raise_z": float(pour[2] - load[2]),
        "limited": bool(limited),
        "bucket": float(q[bucket_idx]),
    }


def nearest_bucket_carry_world_angle(reference_rad, q_reference=None, end_effector="load", tilt_deg=None):
    level = nearest_bucket_level_world_angle(reference_rad)
    tilt = deg_to_rad(BUCKET_CARRY_HOLD_TILT_DEG if tilt_deg is None else float(tilt_deg))
    candidates = [level + tilt, level - tilt, level]
    if q_reference is None:
        return candidates[0]

    best = None
    for angle in candidates:
        report = bucket_mouth_raise_for_world_angle(q_reference, angle, end_effector=end_effector)
        if report is None:
            continue
        raise_z = float(report["raise_z"])
        tilt_err = abs(abs(wrap_angle(float(angle) - float(level))) - tilt)
        retain_penalty = max(0.0, BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z - raise_z) * 12.0
        limit_penalty = 2.0 if report.get("limited") else 0.0
        cost = retain_penalty + tilt_err + 0.05 * abs(wrap_angle(float(angle) - float(reference_rad))) + limit_penalty
        row = {
            "angle": float(angle),
            "cost": float(cost),
            "raise_z": raise_z,
            "limited": bool(report.get("limited")),
        }
        if best is None or row["cost"] < best["cost"]:
            best = row

    if best is None:
        return candidates[0]
    return float(best["angle"])


def planar_points_from_angles(angles, lengths):
    p0 = np.array([0.0, 0.0], dtype=np.float32)
    pts = [p0]
    p = p0.copy()
    for angle, length in zip(angles, lengths):
        p = p + np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float32) * float(length)
        pts.append(p.copy())
    return pts


def planar_points_from_q(q, lengths, model=None, end_effector="mid"):
    angles = chain_angles_from_q(q, model=model, end_effector=end_effector)
    if angles is None:
        return None
    return planar_points_from_angles(angles, lengths)


def two_link_ik_2d(target, l1, l2, seed_angles):
    target = np.array(target, dtype=np.float32)
    r = float(np.linalg.norm(target))
    r_safe = max(r, 1e-6)
    max_r = max(1e-6, float(l1 + l2) - 1e-6)
    min_r = max(0.0, abs(float(l1 - l2)) + 1e-6)

    if r > max_r:
        target = target * (max_r / r_safe)
        r = max_r
    elif r < min_r:
        target = target * (min_r / r_safe)
        r = min_r

    c = (r * r - float(l1) * float(l1) - float(l2) * float(l2)) / (2.0 * float(l1) * float(l2))
    c = clamp(c, -1.0, 1.0)

    base = math.atan2(float(target[1]), float(target[0]))
    solutions = []

    for rel in [math.acos(c), -math.acos(c)]:
        a1 = base - math.atan2(float(l2) * math.sin(rel), float(l1) + float(l2) * math.cos(rel))
        a2 = a1 + rel
        cost = abs(wrap_angle(a1 - seed_angles[0])) + abs(wrap_angle(a2 - seed_angles[1]))
        solutions.append((cost, a1, a2))

    solutions.sort(key=lambda x: x[0])
    return solutions


def q_from_chain_angles(swing_goal, angles, model=None, end_effector="mid"):
    if model is None:
        model = IK_MODEL
    part = ik_model_part(model=model, end_effector=end_effector)
    signs = np.array(model["signs"], dtype=np.float32)
    offsets = np.array(part["offsets"], dtype=np.float32)

    q = np.zeros(4, dtype=np.float32)
    q[CTRL.name_to_idx["swing"]] = float(swing_goal)
    q[CTRL.name_to_idx["boom"]] = signs[0] * wrap_angle(float(angles[0]) - float(offsets[0]))
    q[CTRL.name_to_idx["arm"]] = signs[1] * wrap_angle((float(angles[1]) - float(angles[0])) - float(offsets[1]))
    q[CTRL.name_to_idx["bucket"]] = signs[2] * wrap_angle((float(angles[2]) - float(angles[1])) - float(offsets[2]))
    return CTRL.clip_limits(q)


def predicted_planar_error(q, target_2d, lengths, model=None, end_effector="mid"):
    pts = planar_points_from_q(q, lengths, model=model, end_effector=end_effector)
    if pts is None:
        return 1e6
    return float(np.linalg.norm(pts[-1] - np.array(target_2d, dtype=np.float32)))


def refine_planar_ik_candidate(q_start, target_2d, lengths, end_effector="mid"):
    q = CTRL.clip_limits(q_start)
    target = np.array(target_2d, dtype=np.float32)
    active = [
        CTRL.name_to_idx["boom"],
        CTRL.name_to_idx["arm"],
        CTRL.name_to_idx["bucket"],
    ]
    swing_idx = CTRL.name_to_idx["swing"]
    swing_value = float(q[swing_idx])

    for _ in range(IK_REFINE_ITERS):
        pts = planar_points_from_q(q, lengths, end_effector=end_effector)
        if pts is None:
            break
        p = np.array(pts[-1], dtype=np.float32)
        err = target - p
        if float(np.linalg.norm(err)) < 0.01:
            break

        jac = np.zeros((2, len(active)), dtype=np.float32)
        eps = 1e-3
        for col, idx in enumerate(active):
            q_eps = q.copy()
            q_eps[idx] += eps
            q_eps = CTRL.clip_limits(q_eps)
            q_eps[swing_idx] = swing_value
            p_eps = planar_points_from_q(q_eps, lengths, end_effector=end_effector)
            if p_eps is None:
                continue
            jac[:, col] = (np.array(p_eps[-1], dtype=np.float32) - p) / eps

        jj_t = jac @ jac.T
        try:
            step = jac.T @ np.linalg.solve(
                jj_t + float(IK_REFINE_LAMBDA) * np.eye(2, dtype=np.float32),
                err,
            )
        except Exception:
            break

        step = np.clip(step, -IK_REFINE_MAX_STEP, IK_REFINE_MAX_STEP)
        for value, idx in zip(step, active):
            q[idx] += float(value)
        q = CTRL.clip_limits(q)
        q[swing_idx] = swing_value

    return CTRL.clip_limits(q)


def predicted_min_world_z(q, lengths, boom_root_z, model=None, end_effector="mid", include_end=True):
    pts = planar_points_from_q(q, lengths, model=model, end_effector=end_effector)
    if pts is None:
        return -1e6
    used = pts if include_end else pts[:-1]
    return float(boom_root_z + min(float(p[1]) for p in used))


def predicted_end_world_z(q, lengths, boom_root_z, model=None, end_effector="mid"):
    pts = planar_points_from_q(q, lengths, model=model, end_effector=end_effector)
    if pts is None:
        return -1e6
    return float(boom_root_z + float(pts[-1][1]))


def default_ik_model():
    chain = current_planar_chain("mid")
    if chain is None:
        return None

    signs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    effectors = {}
    for end_effector in ["mid", "tip", "load", "pour"]:
        c = current_planar_chain(end_effector)
        if c is None:
            continue
        effectors[end_effector] = {
            "offsets": build_ik_offsets(CTRL.q_cmd, c, signs),
            "lengths": np.array(c["lengths"], dtype=np.float32),
        }
    if "mid" not in effectors:
        return None

    return {
        "signs": signs,
        "effectors": effectors,
        "calibrated": False,
    }


def solve_priority_ik_to_target(
    target_world,
    q_seed=None,
    preferred_bucket_rad=None,
    preferred_end_angle_rad=None,
    bucket_motion_weight=None,
    bucket_preference_weight=None,
    min_world_z=None,
    accept_err=None,
    end_effector="mid",
    allow_end_below=False,
    min_end_z=None,
    phase_mode=None,
    end_angle_tolerance_rad=None,
    use_refinement=False,
    bucket_candidate_span_deg=None,
    bucket_candidate_count=None,
    return_candidates=False,
    max_solutions=1,
    soft_accept_err=None,
):
    global IK_MODEL

    if IK_MODEL is None:
        IK_MODEL = default_ik_model()
    if IK_MODEL is None:
        return None, "IK model unavailable"

    if q_seed is None:
        update_q_cmd_from_real()
        q_now = CTRL.q_cmd.copy()
    else:
        q_now = CTRL.clip_limits(q_seed)

    has_pose_preference = preferred_bucket_rad is not None or preferred_end_angle_rad is not None
    if bucket_motion_weight is None:
        bucket_motion_weight = 3.5 if not has_pose_preference else 0.8
    if bucket_preference_weight is None:
        bucket_preference_weight = 0.0 if not has_pose_preference else 3.0
    if min_world_z is None:
        min_world_z = GROUND_TOP_Z + IK_FOLLOW_MIN_CLEARANCE
    if accept_err is None:
        accept_err = IK_ACCEPT_ERR

    pts = get_ik_world_points()
    swing_anchor = pts["swing"]
    boom_now = pts["boom"]
    if swing_anchor is None or boom_now is None:
        return None, "missing swing/boom anchor"

    target = np.array(target_world, dtype=np.float32)
    dx = float(target[0] - swing_anchor[0])
    dy = float(target[1] - swing_anchor[1])
    raw_swing_goal = math.atan2(dy, dx)
    swing_now = float(q_now[CTRL.name_to_idx["swing"]])
    swing_goal_unbounded = swing_target_near(raw_swing_goal, swing_now)
    swing_goal = normalize_swing_cmd(swing_goal_unbounded)

    boom_offset_xy = boom_now[:2] - swing_anchor[:2]
    boom_goal_xy = swing_anchor[:2] + rotate_xy(boom_offset_xy, swing_goal - float(q_now[CTRL.name_to_idx["swing"]]))
    boom_root = np.array([boom_goal_xy[0], boom_goal_xy[1], float(boom_now[2])], dtype=np.float32)

    radial = safe_norm(np.array([math.cos(swing_goal), math.sin(swing_goal)], dtype=np.float32), default=(1.0, 0.0))
    target_2d = point_to_2d(target, boom_root, radial)

    part = ik_model_part(end_effector=end_effector)
    if part is None:
        return None, f"missing IK model for end_effector={end_effector}"

    chain_now = current_planar_chain(end_effector)
    fallback_lengths = [] if chain_now is None else chain_now["lengths"]
    lengths = np.array(part.get("lengths", fallback_lengths), dtype=np.float32)
    if len(lengths) != 3 or float(np.min(lengths)) < IK_MIN_SEGMENT_LENGTH:
        if chain_now is None:
            return None, "current planar chain unavailable and IK model lengths invalid"
        lengths = np.array(chain_now["lengths"], dtype=np.float32)

    seed_angles = chain_angles_from_q(q_now, end_effector=end_effector)
    if seed_angles is None and chain_now is not None:
        seed_angles = chain_now["angles"]
    if seed_angles is None:
        return None, "seed chain angles unavailable"

    current_a3 = float(seed_angles[2])
    candidate_angles = [current_a3]
    if bucket_candidate_span_deg is None:
        bucket_candidate_span_deg = IK_BUCKET_CANDIDATE_SPAN_DEG
    if bucket_candidate_count is None:
        bucket_candidate_count = IK_BUCKET_CANDIDATE_COUNT
    bucket_candidate_count = max(1, int(bucket_candidate_count))
    span = deg_to_rad(float(bucket_candidate_span_deg))
    if bucket_candidate_count > 1:
        for a in np.linspace(current_a3 - span, current_a3 + span, bucket_candidate_count):
            candidate_angles.append(float(a))

    if preferred_bucket_rad is not None:
        preferred_rel = float(preferred_bucket_rad)
        candidate_angles.append(float(seed_angles[1] + part["offsets"][2] + IK_MODEL["signs"][2] * preferred_rel))
    if preferred_end_angle_rad is not None:
        preferred_abs = float(preferred_end_angle_rad)
        candidate_angles.append(preferred_abs)
        if bucket_candidate_count > 1:
            local_span = deg_to_rad(20.0)
            for a in np.linspace(preferred_abs - local_span, preferred_abs + local_span, 7):
                candidate_angles.append(float(a))

    best = None
    solution_rows = []
    reject_counts = {}
    total_reach = float(np.sum(lengths))
    target_dist = float(np.linalg.norm(target_2d))

    def reject(reason):
        key = str(reason)
        reject_counts[key] = int(reject_counts.get(key, 0)) + 1

    for a3 in candidate_angles:
        wrist_target = target_2d - np.array([math.cos(a3), math.sin(a3)], dtype=np.float32) * float(lengths[2])

        for _, a1, a2 in two_link_ik_2d(wrist_target, lengths[0], lengths[1], seed_angles):
            angles = [a1, a2, a3]
            q_candidate = q_from_chain_angles(swing_goal, angles, end_effector=end_effector)
            if use_refinement:
                q_candidate = refine_planar_ik_candidate(q_candidate, target_2d, lengths, end_effector=end_effector)
            actual_angles = chain_angles_from_q(q_candidate, end_effector=end_effector)
            actual_a3 = float(actual_angles[2]) if actual_angles is not None else float(a3)
            if preferred_end_angle_rad is not None and end_angle_tolerance_rad is not None:
                end_angle_err = abs(wrap_angle(float(actual_a3 - preferred_end_angle_rad)))
                if end_angle_err > float(end_angle_tolerance_rad):
                    reject(f"end angle off {rad_to_deg(end_angle_err):.2f} deg")
                    continue

            err_after_clip = predicted_planar_error(q_candidate, target_2d, lengths, end_effector=end_effector)
            min_z = predicted_min_world_z(
                q_candidate,
                lengths,
                float(boom_root[2]),
                end_effector=end_effector,
                include_end=not allow_end_below,
            )
            end_z = predicted_end_world_z(q_candidate, lengths, float(boom_root[2]), end_effector=end_effector)
            if min_z < float(min_world_z):
                reject("body below clearance")
                continue
            if min_end_z is not None and end_z < float(min_end_z):
                reject("end too deep")
                continue

            phase_report = None
            if phase_mode is not None:
                phase_report = predicted_phase_ground_report(q_candidate, phase_mode)
                if phase_report.get("tip_z") is None:
                    reject("missing predicted phase report")
                    continue
                phase_ok, phase_reason = phase_ground_ok(phase_mode, phase_report)
                if not phase_ok:
                    reject(phase_reason)
                    continue

            dq = np.array([wrap_angle(float(q_candidate[i] - q_now[i])) for i in range(4)], dtype=np.float32)
            motion_cost = float(np.sum(IK_COST_WEIGHTS * np.abs(dq)))
            bucket_change = abs(wrap_angle(float(q_candidate[CTRL.name_to_idx["bucket"]] - q_now[CTRL.name_to_idx["bucket"]])))
            bucket_pref_cost = 0.0
            if preferred_bucket_rad is not None:
                bucket_pref_cost = abs(wrap_angle(float(q_candidate[CTRL.name_to_idx["bucket"]] - preferred_bucket_rad))) * 1.2
            if preferred_end_angle_rad is not None:
                bucket_pref_cost += abs(wrap_angle(float(actual_a3 - preferred_end_angle_rad))) * 1.8

            cost = 140.0 * err_after_clip + motion_cost + float(bucket_motion_weight) * bucket_change + float(bucket_preference_weight) * bucket_pref_cost
            row = {
                "cost": float(cost),
                "q": q_candidate.copy(),
                "planar_err": float(err_after_clip),
                "min_z": float(min_z),
                "end_z": float(end_z),
                "angles": angles,
                "end_angle": float(actual_a3),
                "phase_report": phase_report,
            }
            solution_rows.append(row)

            if best is None or cost < best["cost"]:
                best = row

    if best is None:
        details = ", ".join(
            f"{k}:{v}" for k, v in sorted(reject_counts.items(), key=lambda item: -item[1])[:4]
        )
        if not details:
            details = "none"
        return None, f"no IK candidate above z={float(min_world_z):.2f}; rejects={details}"

    def info_from_row(row):
        q_row = row["q"]
        return {
            "reachable": target_dist <= total_reach + 1e-4,
            "planar_err": row["planar_err"],
            "target_2d": target_2d,
            "solved_points": planar_points_from_q(q_row, lengths, end_effector=end_effector),
            "cost": row["cost"],
            "min_z": row["min_z"],
            "end_z": row["end_z"],
            "end_angle": row["end_angle"],
            "end_effector": end_effector,
            "phase_report": row.get("phase_report"),
            "target_dist": target_dist,
            "total_reach": total_reach,
            "raw_swing_goal": raw_swing_goal,
            "swing_goal": swing_goal,
            "swing_goal_unbounded": swing_goal_unbounded,
        }

    if return_candidates:
        err_limit = float(soft_accept_err if soft_accept_err is not None else accept_err)
        rows = [row for row in solution_rows if float(row.get("planar_err", 1e9)) <= err_limit]
        rows.sort(key=lambda row: float(row.get("cost", 1e9)))
        rows = rows[: max(1, int(max_solutions))]
        if not rows:
            return [], f"best IK error too high: planar={best['planar_err']:.3f} m"
        return [(row["q"].copy(), info_from_row(row)) for row in rows], "ok"

    if best["planar_err"] > float(accept_err):
        return None, f"best IK error too high: planar={best['planar_err']:.3f} m"

    q_goal = best["q"]

    info = info_from_row(best)
    return q_goal, info


def target_radius_from_swing_center(target_xyz):
    center_xy = get_swing_xy_center()
    target_xy = np.array([float(target_xyz[0]), float(target_xyz[1])], dtype=np.float32)
    return float(np.linalg.norm(target_xy - center_xy))


def validate_dig_target(target_xyz, hard_block=False):
    """
    不再用固定 5.5m 硬阻止。
    """
    r = target_radius_from_swing_center(target_xyz)
    if r < DIG_MIN_RADIUS:
        return False, f"target too close: r={r:.2f} m"

    if float(target_xyz[2]) < GROUND_TOP_Z - 0.05:
        return False, "target below hard floor"

    return True, f"ok r={r:.2f} m"


def target_to_swing_angle(target_xyz):
    """
    用 swing joint center 计算回转角。
    """
    center_xy = get_swing_xy_center()

    dx = float(target_xyz[0] - center_xy[0])
    dy = float(target_xyz[1] - center_xy[1])

    raw = math.atan2(dy, dx)

    return raw


def dig_direction_unit(target_xyz):
    """
    从目标点指向 swing center。
    """
    center_xy = get_swing_xy_center()
    txy = np.array([float(target_xyz[0]), float(target_xyz[1])], dtype=np.float32)

    v = center_xy - txy
    n = float(np.linalg.norm(v))

    if n < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    return v / n


def offset_xy(point, direction_xy, amount, z=None):
    p = np.array(point, dtype=np.float32).copy()
    p[0] += float(direction_xy[0]) * float(amount)
    p[1] += float(direction_xy[1]) * float(amount)
    if z is not None:
        p[2] = float(z)
    return p


def adaptive_dig_plan_candidates(target_xyz):
    return ik_calculation.adaptive_dig_plan_candidates(runtime_module(), target_xyz)


def solve_dig_pose(
    label,
    point,
    bucket_deg,
    duration,
    q_seed,
    bucket_world_deg=None,
    ik_effector="tip",
    accept_err=0.38,
    bucket_motion_weight=0.45,
    bucket_preference_weight=None,
):
    if is_cutting_phase(label):
        min_end_z = GROUND_TOP_Z - DIG_MAX_TIP_DEPTH
        min_body_z = GROUND_TOP_Z - 0.02
    elif is_curl_phase(label):
        min_end_z = GROUND_TOP_Z - DIG_MAX_CURL_DEPTH
        min_body_z = GROUND_TOP_Z - 0.02
    else:
        min_end_z = GROUND_TOP_Z + 0.02
        min_body_z = GROUND_TOP_Z + IK_DIG_MIN_CLEARANCE

    carry_angle_phase = ("lift" in label) or ("carry" in label) or ("unload" in label and bucket_world_deg is not None)
    end_angle_tol_deg = BUCKET_CARRY_HOLD_TOL_DEG if carry_angle_phase else BUCKET_LIFT_LEVEL_TOL_DEG
    q_goal, info = solve_priority_ik_to_target(
        point,
        q_seed=q_seed,
        preferred_bucket_rad=None if (bucket_deg is None or bucket_world_deg is not None) else deg_to_rad(bucket_deg),
        preferred_end_angle_rad=None if bucket_world_deg is None else deg_to_rad(bucket_world_deg),
        bucket_motion_weight=bucket_motion_weight,
        bucket_preference_weight=(0.0 if bucket_world_deg is not None else 4.0) if bucket_preference_weight is None else bucket_preference_weight,
        min_world_z=min_body_z,
        accept_err=accept_err,
        end_effector=ik_effector,
        allow_end_below=is_cutting_phase(label) or is_curl_phase(label),
        min_end_z=min_end_z,
        phase_mode=label,
        end_angle_tolerance_rad=None,
        use_refinement=True,
        bucket_candidate_span_deg=DIG_IK_BUCKET_CANDIDATE_SPAN_DEG,
        bucket_candidate_count=DIG_IK_BUCKET_CANDIDATE_COUNT,
    )

    if q_goal is None:
        set_target_color(TARGET_COLOR_UNREACHABLE)
        update_status(f"[DIG PLAN WARN] {label}: {info}", force=True)
        return None

    level_calc = None
    if bucket_world_deg is not None:
        level_calc = bucket_joint_for_world_angle(q_goal, deg_to_rad(bucket_world_deg), end_effector=ik_effector)
        if level_calc is not None:
            q_goal[CTRL.name_to_idx["bucket"]] = level_calc["bucket"]
            q_goal = CTRL.clip_limits(q_goal)
            actual_angles = chain_angles_from_q(q_goal, end_effector=ik_effector)
            if actual_angles is not None:
                info["end_angle"] = float(actual_angles[2])
                level_err_rad = abs(wrap_angle(float(actual_angles[2]) - deg_to_rad(bucket_world_deg)))
                if level_err_rad > deg_to_rad(end_angle_tol_deg):
                    update_status(
                        f"[DIG PLAN DIAG] {label}: bucket world angle error {rad_to_deg(level_err_rad):.2f}deg "
                        f"after joint limits; executing candidates and scoring actual sand",
                        force=True,
                    )
                info["world_angle_err_deg"] = float(rad_to_deg(level_err_rad))

    raw_swing_goal = info.get("raw_swing_goal")
    swing_goal = info.get("swing_goal")
    raw_swing_goal_deg = None if raw_swing_goal is None else rad_to_deg(raw_swing_goal)
    swing_goal_deg = None if swing_goal is None else rad_to_deg(swing_goal)
    info_print(
        f"[DIG IK] {label}: target=({point[0]:.2f},{point[1]:.2f},{point[2]:.2f}) "
        f"bucket={fmt_optional(bucket_deg)}deg bucket_world={fmt_optional(bucket_world_deg)}deg "
        f"swing_goal={fmt_optional(swing_goal_deg)}deg raw_swing={fmt_optional(raw_swing_goal_deg)}deg "
        f"end={ik_effector} planar_err={info['planar_err']:.3f} "
        f"end_world_angle={rad_to_deg(info['end_angle']):.1f}deg "
        f"body_min_z={info['min_z']:.3f} tip_z={info['end_z']:.3f} "
        f"reach={info['target_dist']:.3f}/{info['total_reach']:.3f}"
    )
    if level_calc is not None:
        info_print(
            f"[DIG WORLD ANGLE] {label}: bucket_solved={rad_to_deg(q_goal[CTRL.name_to_idx['bucket']]):.2f}deg "
            f"world={bucket_world_deg:.2f}deg parent={rad_to_deg(level_calc['parent_angle']):.2f}deg "
            f"offset={rad_to_deg(level_calc['offset']):.2f}deg sign={level_calc['sign']:+.0f} "
            f"actual_world={rad_to_deg(info['end_angle']):.2f}deg"
        )
    report = info.get("phase_report")
    if report is not None:
        _, reason = phase_ground_ok(label, report)
        info_print("[DIG TARGET GUARD] " + format_ground_report(label, report, reason))
    info_print(
        f"[DIG Q] {label}: swing={rad_to_deg(q_goal[0]):.2f} "
        f"boom={rad_to_deg(q_goal[1]):.2f} arm={rad_to_deg(q_goal[2]):.2f} "
        f"bucket={rad_to_deg(q_goal[3]):.2f}"
    )
    return label, q_goal, duration


def solve_dig_pose_candidates(
    label,
    point,
    bucket_deg,
    duration,
    q_seed,
    bucket_world_deg=None,
    ik_effector="tip",
    accept_err=0.38,
    soft_accept_err=None,
    bucket_motion_weight=0.45,
    bucket_preference_weight=None,
    max_solutions=DIG_PLAN_TOPK_IK,
):
    if is_cutting_phase(label):
        min_end_z = GROUND_TOP_Z - DIG_MAX_TIP_DEPTH
        min_body_z = GROUND_TOP_Z - 0.02
    elif is_curl_phase(label):
        min_end_z = GROUND_TOP_Z - DIG_MAX_CURL_DEPTH
        min_body_z = GROUND_TOP_Z - 0.02
    else:
        min_end_z = GROUND_TOP_Z + 0.02
        min_body_z = GROUND_TOP_Z + IK_DIG_MIN_CLEARANCE

    carry_angle_phase = ("lift" in label) or ("carry" in label) or ("unload" in label and bucket_world_deg is not None)
    end_angle_tol_deg = BUCKET_CARRY_HOLD_TOL_DEG if carry_angle_phase else BUCKET_LIFT_LEVEL_TOL_DEG
    candidate_rows, reason = solve_priority_ik_to_target(
        point,
        q_seed=q_seed,
        preferred_bucket_rad=None if (bucket_deg is None or bucket_world_deg is not None) else deg_to_rad(bucket_deg),
        preferred_end_angle_rad=None if bucket_world_deg is None else deg_to_rad(bucket_world_deg),
        bucket_motion_weight=bucket_motion_weight,
        bucket_preference_weight=(0.0 if bucket_world_deg is not None else 4.0) if bucket_preference_weight is None else bucket_preference_weight,
        min_world_z=min_body_z,
        accept_err=accept_err,
        end_effector=ik_effector,
        allow_end_below=is_cutting_phase(label) or is_curl_phase(label),
        min_end_z=min_end_z,
        phase_mode=label,
        end_angle_tolerance_rad=None,
        use_refinement=True,
        bucket_candidate_span_deg=DIG_IK_BUCKET_CANDIDATE_SPAN_DEG,
        bucket_candidate_count=DIG_IK_BUCKET_CANDIDATE_COUNT,
        return_candidates=True,
        max_solutions=max_solutions,
        soft_accept_err=soft_accept_err,
    )

    if not candidate_rows:
        return [], str(reason)

    rows = []
    for q_goal, info in candidate_rows:
        q_goal = np.array(q_goal, dtype=np.float32).copy()
        level_calc = None
        if bucket_world_deg is not None:
            level_calc = bucket_joint_for_world_angle(q_goal, deg_to_rad(bucket_world_deg), end_effector=ik_effector)
            if level_calc is None:
                continue
            q_goal[CTRL.name_to_idx["bucket"]] = level_calc["bucket"]
            q_goal = CTRL.clip_limits(q_goal)
            actual_angles = chain_angles_from_q(q_goal, end_effector=ik_effector)
            if actual_angles is not None:
                info = dict(info)
                info["end_angle"] = float(actual_angles[2])
                level_err_rad = abs(wrap_angle(float(actual_angles[2]) - deg_to_rad(bucket_world_deg)))
                info["world_angle_err_deg"] = float(rad_to_deg(level_err_rad))

        rows.append(
            {
                "phase": label,
                "q_goal": q_goal.copy(),
                "duration": float(duration),
                "target_point": np.array(point, dtype=np.float32).copy(),
                "bucket_deg": None if bucket_deg is None else float(bucket_deg),
                "bucket_world_deg": None if bucket_world_deg is None else float(bucket_world_deg),
                "effector": ik_effector,
                "info": dict(info),
                "level_calc": level_calc,
                "item": (label, q_goal.copy(), float(duration)),
            }
        )

    if not rows:
        return [], f"{label} no usable IK candidates after bucket world-angle reconstruction"
    return rows, "ok"


def path_end_effector_for_mode(mode):
    m = str(mode).lower()
    if "dump" in m:
        return "pour"
    if "lift" in m or "carry" in m or "unload" in m:
        return "load"
    return "tip"


def solve_clearance_pose(point, q_seed, end_effector, clearance_z):
    p = np.array(point, dtype=np.float32).copy()
    p[2] = float(clearance_z)
    return solve_priority_ik_to_target(
        p,
        q_seed=q_seed,
        preferred_bucket_rad=float(q_seed[CTRL.name_to_idx["bucket"]]),
        preferred_end_angle_rad=None,
        bucket_motion_weight=1.2,
        bucket_preference_weight=1.4,
        min_world_z=GROUND_TOP_Z + PATH_CLEARANCE_BODY_Z,
        accept_err=0.55,
        end_effector=end_effector,
        allow_end_below=False,
        min_end_z=GROUND_TOP_Z + PATH_CLEARANCE_END_Z,
        phase_mode="clearance",
        use_refinement=True,
        bucket_candidate_span_deg=DIG_IK_BUCKET_CANDIDATE_SPAN_DEG,
        bucket_candidate_count=DIG_IK_BUCKET_CANDIDATE_COUNT,
    )


def route_segments_ok(q_start, route, q_goal, mode):
    q_prev = q_start
    for idx, q_next in enumerate(route):
        ok, kind, reason, sample, report = path_segment_check(q_prev, q_next, "clearance")
        if not ok:
            text = path_block_report_text("clearance", kind, report, reason)
            return False, f"segment {idx + 1} blocked at {sample}/{PATH_CHECK_SAMPLES}: {text}"
        q_prev = q_next

    ok, kind, reason, sample, report = path_segment_check(q_prev, q_goal, mode)
    if not ok:
        text = path_block_report_text(mode, kind, report, reason)
        return False, f"final segment blocked at {sample}/{PATH_CHECK_SAMPLES}: {text}"

    return True, "ok"


def find_clearance_route(q_start, q_goal, mode, label):
    end_effector = path_end_effector_for_mode(mode)
    p_start = predicted_end_world_point(q_start, end_effector=end_effector, reference_q=q_start)
    p_goal = predicted_end_world_point(q_goal, end_effector=end_effector, reference_q=q_start)

    if p_start is None:
        p_start = ik_end_effector_pos(end_effector)
    if p_start is None or p_goal is None:
        return None, "missing predicted start/goal end point"

    last_reason = "no candidate tried"
    obstacle_top = obstacle_top_z_for_segment(p_start, p_goal)

    for height in PATH_CLEARANCE_HEIGHTS:
        clearance_z = max(float(p_start[2]), float(p_goal[2]), GROUND_TOP_Z + float(height))
        if obstacle_top is not None:
            clearance_z = max(clearance_z, float(obstacle_top) + PATH_OBSTACLE_OVER_CLEARANCE_Z)

        for frac in PATH_CLEARANCE_FRACTIONS:
            p_via = (1.0 - float(frac)) * np.array(p_start, dtype=np.float32) + float(frac) * np.array(p_goal, dtype=np.float32)
            p_via[2] = clearance_z

            q_via, info = solve_clearance_pose(p_via, q_start, end_effector, clearance_z)

            if q_via is None:
                last_reason = f"clearance IK failed: {info}"
                continue

            ok_route, route_reason = route_segments_ok(q_start, [q_via], q_goal, mode)
            if not ok_route:
                last_reason = route_reason
                continue

            info_print(
                f"[PATH VIA FOUND] {label}: end={end_effector} frac={float(frac):.2f} "
                f"z={clearance_z:.2f} planar_err={info['planar_err']:.3f} "
                f"q=({rad_to_deg(q_via[0]):.2f},{rad_to_deg(q_via[1]):.2f},"
                f"{rad_to_deg(q_via[2]):.2f},{rad_to_deg(q_via[3]):.2f})"
            )
            return [q_via], "ok"

        q_up_start, info_start = solve_clearance_pose(p_start, q_start, end_effector, clearance_z)
        if q_up_start is None:
            last_reason = f"start lift IK failed: {info_start}"
            continue

        q_up_goal, info_goal = solve_clearance_pose(p_goal, q_up_start, end_effector, clearance_z)
        if q_up_goal is None:
            last_reason = f"goal lift IK failed: {info_goal}"
            continue

        route = [q_up_start, q_up_goal]
        ok_route, route_reason = route_segments_ok(q_start, route, q_goal, mode)
        if ok_route:
            info_print(
                f"[PATH ROUTE FOUND] {label}: end={end_effector} type=lift_cross_drop "
                f"z={clearance_z:.2f} waypoints={len(route)}"
            )
            return route, "ok"
        last_reason = route_reason

        direction_xy = np.array(p_goal[:2], dtype=np.float32) - np.array(p_start[:2], dtype=np.float32)
        perp = safe_norm(np.array([-float(direction_xy[1]), float(direction_xy[0])], dtype=np.float32), default=(1.0, 0.0))
        midpoint = 0.5 * (np.array(p_start, dtype=np.float32) + np.array(p_goal, dtype=np.float32))
        for side_offset in PATH_ROUTE_SIDE_OFFSETS:
            for sign in [-1.0, 1.0]:
                p_side = midpoint.copy()
                p_side[:2] += float(sign) * float(side_offset) * perp
                p_side[2] = clearance_z
                q_side, info_side = solve_clearance_pose(p_side, q_up_start, end_effector, clearance_z)
                if q_side is None:
                    last_reason = f"side route IK failed: {info_side}"
                    continue
                route = [q_up_start, q_side, q_up_goal]
                ok_route, route_reason = route_segments_ok(q_start, route, q_goal, mode)
                if ok_route:
                    info_print(
                        f"[PATH ROUTE FOUND] {label}: end={end_effector} type=lift_side_cross_drop "
                        f"z={clearance_z:.2f} side={float(sign) * float(side_offset):.2f} waypoints={len(route)}"
                    )
                    return route, "ok"
                last_reason = route_reason

    return None, last_reason


def pre_dig_needs_swing_align(q_goal, label, mode):
    text = f"{label} {mode}".lower()
    skip_tokens = [
        "coord_align",
        "travel_coord",
        "clearance",
        "unload",
        "dump",
        "follow",
        "manual",
        "calibrate",
    ]
    if any(token in text for token in skip_tokens):
        return False, 0.0

    dig_tokens = [
        "pre_dig",
        "approach",
        "insert",
        "pull",
        "cut",
        "curl",
        "lift",
        "carry",
    ]
    if not any(token in text for token in dig_tokens):
        return False, 0.0

    q_start = CTRL.q_cmd.copy()
    swing_idx = CTRL.name_to_idx["swing"]
    swing_err = abs(rad_to_deg(swing_delta(q_goal[swing_idx], q_start[swing_idx])))
    return swing_err > PRE_DIG_SWING_ALIGN_THRESHOLD_DEG, swing_err


async def prepare_pre_dig_swing_align(q_goal, label="", task_id=None, mode="pre_dig"):
    needed, swing_err_deg = pre_dig_needs_swing_align(q_goal, label, mode)
    if not needed:
        return True

    q_start = CTRL.q_cmd.copy()
    swing_idx = CTRL.name_to_idx["swing"]
    total_swing_delta = swing_delta(q_goal[swing_idx], q_start[swing_idx])

    q_travel = q_start.copy()
    for joint_name, deg in PRE_DIG_TRAVEL_DEG.items():
        q_travel[CTRL.name_to_idx[joint_name]] = deg_to_rad(deg)
    q_travel[swing_idx] = q_start[swing_idx] + float(PRE_DIG_TRAVEL_SWING_FRACTION) * total_swing_delta
    q_travel = clip_command_near(q_travel, reference=q_start)

    q_align = q_travel.copy()
    q_align[swing_idx] = q_goal[swing_idx]
    for joint_name in ["boom", "arm", "bucket"]:
        idx = CTRL.name_to_idx[joint_name]
        q_align[idx] = float(q_travel[idx]) + float(PRE_DIG_ALIGN_JOINT_BLEND) * float(q_goal[idx] - q_travel[idx])
    q_align = clip_command_near(q_align, reference=q_travel)

    swing_align_delta = abs(rad_to_deg(swing_delta(q_align[swing_idx], q_travel[swing_idx])))
    swing_seconds = max(
        PRE_DIG_SWING_ALIGN_MIN_SECONDS,
        min(PRE_DIG_SWING_ALIGN_MAX_SECONDS, swing_align_delta / max(15.0, rad_to_deg(DQ_MAX["swing"] * get_speed_multiplier())) + 0.35),
    )

    info_print(
        f"[PRE DIG COORD ALIGN] {label}: requested_swing_delta={swing_err_deg:.2f}deg "
        f"travel_swing_fraction={PRE_DIG_TRAVEL_SWING_FRACTION:.2f} "
        f"align_joint_blend={PRE_DIG_ALIGN_JOINT_BLEND:.2f} "
        f"travel_q=({rad_to_deg(q_travel[0]):.2f},{rad_to_deg(q_travel[1]):.2f},"
        f"{rad_to_deg(q_travel[2]):.2f},{rad_to_deg(q_travel[3]):.2f}) "
        f"align_q=({rad_to_deg(q_align[0]):.2f},{rad_to_deg(q_align[1]):.2f},"
        f"{rad_to_deg(q_align[2]):.2f},{rad_to_deg(q_align[3]):.2f})"
    )

    ok = await move_to_profile(
        q_travel,
        seconds=PRE_DIG_TRAVEL_SECONDS,
        label=f"{label}_travel_pose",
        task_id=task_id,
        mode="pre_dig_travel_coord",
    )
    if not ok or (task_id is not None and not task_alive(task_id)):
        update_status(f"[PRE DIG COORD ALIGN BLOCKED] {label}: travel pose failed", force=True)
        return False

    ok = await move_to_profile(
        q_align,
        seconds=swing_seconds,
        label=f"{label}_coord_align",
        task_id=task_id,
        mode="pre_dig_coord_align",
    )
    if not ok or (task_id is not None and not task_alive(task_id)):
        update_status(f"[PRE DIG COORD ALIGN BLOCKED] {label}: coordinated align failed", force=True)
        return False

    return True


async def move_to_profile_with_clearance(q_goal, seconds=1.0, label="", task_id=None, mode="auto"):
    q_goal = clip_command_near(q_goal, reference=CTRL.q_cmd)
    if not await prepare_pre_dig_swing_align(q_goal, label=label, task_id=task_id, mode=mode):
        return False
    q_goal = clip_command_near(q_goal, reference=CTRL.q_cmd)

    q_edge = swing_edge_pose_before_rebase(q_goal, label=label)
    if q_edge is not None:
        swing_idx = CTRL.name_to_idx["swing"]
        q_action_now = CTRL.clip_action_limits(CTRL.q_cmd.copy())
        edge_delta = abs(swing_delta(q_edge[swing_idx], q_action_now[swing_idx]))
        edge_speed = max(0.25, DQ_MAX["swing"] * get_speed_multiplier())
        edge_seconds = max(0.35, min(2.50, edge_delta / edge_speed + 0.20))
        success = await move_to_profile_with_clearance(
            q_edge,
            seconds=edge_seconds,
            label=f"{label}_swing_edge",
            task_id=task_id,
            mode=mode,
        )
        if not success or (task_id is not None and not task_alive(task_id)):
            update_status(f"[SWING EDGE BLOCKED] {label}", force=True)
            return False
        q_goal = clip_command_near(q_goal, reference=CTRL.q_cmd)

    q_goal, rebased = maybe_prepare_swing_rebase_for_segment(q_goal, label=label)
    if rebased:
        await step_updates(2)

    q_start = CTRL.q_cmd.copy()
    phase_ok, phase_reason, phase_sample, phase_report = path_phase_check(q_start, q_goal, mode)
    obstacle_ok, obstacle_reason, obstacle_sample, obstacle_report = path_obstacle_check(q_start, q_goal, mode)

    if phase_ok and obstacle_ok:
        info_print(f"[PATH DIRECT] {label}: mode={mode} samples={PATH_CHECK_SAMPLES} ok=True")
        return await move_to_profile(q_goal, seconds=seconds, label=label, task_id=task_id, mode=mode)

    if not phase_ok:
        info_print(
            f"[PATH DIRECT BLOCKED] {label}: sample={phase_sample}/{PATH_CHECK_SAMPLES} "
            + format_ground_report(mode, phase_report, phase_reason)
        )
    if not obstacle_ok:
        info_print(
            f"[PATH OBSTACLE DETECTED] {label}: sample={obstacle_sample}/{PATH_CHECK_SAMPLES} "
            + format_obstacle_report(mode, obstacle_report, obstacle_reason)
        )

    route, route_reason = find_clearance_route(q_start, q_goal, mode, label)
    if route is None:
        if not obstacle_ok:
            info_print(
                f"[PATH OBSTACLE NO ROUTE] {label}: predicted obstacle but no route found; "
                f"{route_reason}; executing guarded direct motion and relying on live FREEZE diagnostics"
            )
            update_status(f"[PATH OBSTACLE NO ROUTE] {label}: executing guarded direct motion", force=True)
            return await move_to_profile(q_goal, seconds=seconds, label=label, task_id=task_id, mode=mode)
        info_print(
            f"[PATH PREDICTED CONTACT IGNORED] {label}: route_failed={route_reason}; "
            "executing direct command and using live freeze/contact diagnostics"
        )
        return await move_to_profile(q_goal, seconds=seconds, label=label, task_id=task_id, mode=mode)

    via_seconds = max(0.30, min(PATH_CLEARANCE_DURATION, float(seconds) * 0.55))
    for i, q_route in enumerate(route):
        success = await move_to_profile(
            q_route,
            seconds=via_seconds,
            label=f"{label}_route_{i + 1}",
            task_id=task_id,
            mode=f"clearance_{mode}",
        )
        if not success or (task_id is not None and not task_alive(task_id)):
            update_status(f"[PATH BLOCKED] {label}: route waypoint {i + 1} move failed", force=True)
            return False

    return await move_to_profile(q_goal, seconds=seconds, label=label, task_id=task_id, mode=mode)


def dig_plan_specs_from_candidate(target_xyz, candidate):
    target = np.array(target_xyz, dtype=np.float32).copy()
    # Target Z is a material/sand target. Clamp only against the hard floor;
    # sand is deformable and should be entered during cut phases.
    target[2] = max(float(target[2]), GROUND_TOP_Z)
    surface_z = candidate.get("surface_z", None)
    if surface_z is None:
        surface_z = sand_surface_z_at_xy(float(target[0]), float(target[1]))
    if surface_z is None:
        surface_z = max(float(target[2]) + 0.35, GROUND_TOP_Z + 0.35)
    surface_z = max(float(surface_z), float(target[2]) + 0.05)
    target_depth = max(0.0, float(surface_z) - float(target[2]))
    inward = dig_direction_unit(target)
    if float(np.linalg.norm(inward)) < 1e-6:
        inward = np.array([-1.0, 0.0], dtype=np.float32)
    outward = -inward

    pre_clearance = max(0.28, float(candidate.get("pre_z", 0.60)) - min(0.50, target_depth))
    contact_clearance = max(0.015, min(0.08, float(candidate.get("contact_z", 0.03))))
    pre = offset_xy(target, outward, float(candidate.get("approach_offset", 0.25)), float(surface_z) + pre_clearance)
    contact = offset_xy(target, outward, 0.02, float(surface_z) + contact_clearance)
    insert = offset_xy(target, outward, 0.00, float(target[2]) - float(candidate.get("insert_depth", 0.08)))
    mid_cut = offset_xy(target, inward, float(candidate.get("mid_pull", 0.35)), float(target[2]) - float(candidate.get("mid_depth", 0.18)))
    exit_cut = offset_xy(target, inward, float(candidate.get("exit_pull", 0.55)), float(target[2]) - float(candidate.get("exit_depth", 0.05)))
    curl = offset_xy(target, inward, float(candidate.get("exit_pull", 0.55)), float(target[2]) + float(candidate.get("curl_z", 0.08)))
    lift_z = float(exit_cut[2]) + float(candidate.get("lift_height", 0.70))
    lift_z = max(lift_z, float(target[2]) + float(candidate.get("lift_above_target", 0.50)))
    lift_z = max(lift_z, float(candidate.get("min_lift_z", GROUND_TOP_Z + 1.05)))
    lift = offset_xy(exit_cut, inward, 0.00, lift_z)
    unload = unload_bin_dump_point(
        height_delta=float(candidate.get("unload_height_delta", 0.0)),
        xy_offset=candidate.get("unload_xy_offset", None),
    )
    bucket_cut_deg = float(candidate.get("bucket_cut", -60.0))
    bucket_mid_cut_deg = max(bucket_cut_deg, float(candidate.get("bucket_mid_cut", -54.0)))
    bucket_approach_world = float(candidate.get("bucket_attack_world", BUCKET_DIG_APPROACH_WORLD_DEG))
    bucket_insert_world = float(candidate.get("bucket_cut_world", BUCKET_DIG_INSERT_WORLD_DEG))
    bucket_mid_world = float(candidate.get("bucket_mid_cut_world", BUCKET_DIG_PULL_WORLD_DEG))
    bucket_exit_world = float(candidate.get("bucket_exit_world", BUCKET_DIG_EXIT_WORLD_DEG))

    return [
        ("pre_dig", pre, float(candidate.get("bucket_travel", -25.0)), None, "tip", 1.1, True),
        ("approach_contact", contact, float(candidate.get("bucket_attack", -50.0)), bucket_approach_world, "tip", 1.0, True),
        ("insert_cut", insert, bucket_cut_deg, bucket_insert_world, "tip", 0.9, True),
        ("pull_mid_cut", mid_cut, bucket_mid_cut_deg, bucket_mid_world, "tip", 2.0, True),
        ("pull_exit_cut", exit_cut, float(candidate.get("bucket_exit", -70.0)), bucket_exit_world, "tip", 1.5, True),
        ("curl_to_hold_material", curl, float(candidate.get("bucket_curl", -105.0)), None, "tip", 0.9, True),
        ("lift_carry", lift, None, "carry", "load", 1.2, True),
        ("unload_to_bin", unload, None, "carry", "load", 1.35, True),
    ]


def plan_dig_sequence_candidate(target_xyz, candidate, deadline=None):
    specs = dig_plan_specs_from_candidate(target_xyz, candidate)
    beams = [
        {
            "q": CTRL.q_cmd.copy(),
            "seq": [],
            "points": [],
            "stages": [],
            "cost": 0.0,
            "weighted_angle": 0.0,
            "estimated_time": 0.0,
        }
    ]
    last_reason = "no candidate tried"
    best_partial = beams[0]

    def resolve_bucket_world(label, bucket_deg, bucket_world_deg, q_seed, ik_effector):
        if bucket_world_deg == "hold":
            return rad_to_deg(q_seed[CTRL.name_to_idx["bucket"]]), None, "ok"
        if bucket_world_deg not in ("level", "carry"):
            return bucket_deg, bucket_world_deg, "ok"
        reference_angles = chain_angles_from_q(q_seed, end_effector=ik_effector)
        if reference_angles is None:
            return bucket_deg, None, f"{label} cannot read bucket carry reference"
        if bucket_world_deg == "carry":
            bucket_world_rad = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_seed, end_effector=ik_effector)
        else:
            bucket_world_rad = nearest_bucket_level_world_angle(reference_angles[2])
        return bucket_deg, rad_to_deg(bucket_world_rad), "ok"

    for label, point, bucket_deg, bucket_world_deg, ik_effector, duration, required in specs:
        if deadline is not None and time.time() > float(deadline):
            return None, list(best_partial.get("points", [])), {
                "id": str(candidate.get("id", "candidate")),
                "planned": False,
                "failed_stage": label,
                "failure_reason": "planning budget exceeded",
                "planned_prefix": len(best_partial.get("seq", [])),
                "best_partial_cost": float(best_partial.get("cost", 0.0)),
                "best_partial_q_deg": q_deg_values(best_partial.get("q", CTRL.q_cmd), wrap_swing_for_display=True),
                "stages": list(best_partial.get("stages", [])),
            }
        new_beams = []
        fail_reasons = []

        for beam in beams:
            if deadline is not None and time.time() > float(deadline):
                fail_reasons.append("planning budget exceeded")
                break
            q_seed = np.array(beam["q"], dtype=np.float32).copy()
            if label == "unload_to_bin":
                dump_deg = float(candidate.get("unload_dump_deg", BUCKET_UNLOAD_DUMP_DEG))
                q_dump, dump_info = plan_dump_pose_to_bin(
                    q_seed=q_seed,
                    dump_deg=dump_deg,
                    label=f"plan_{candidate.get('id', 'candidate')}_{label}",
                    log=False,
                    max_correction_iters=2,
                    allow_unaligned=True,
                )
                if q_dump is None:
                    reference_angles = chain_angles_from_q(q_seed, end_effector="load")
                    carry_world = None
                    if reference_angles is not None:
                        carry_world = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_seed, end_effector="load")
                    fallback_pose = solve_dig_pose(
                        label,
                        unload_bin_dump_point(),
                        None,
                        duration,
                        q_seed,
                        bucket_world_deg=None if carry_world is None else rad_to_deg(carry_world),
                        ik_effector="load",
                        accept_err=0.85,
                        bucket_motion_weight=0.65,
                        bucket_preference_weight=0.0,
                    )
                    fallback_info = "ok"
                    q_pre_dump = None
                    if fallback_pose is None:
                        fallback_info = "no fallback pose"
                    elif isinstance(fallback_pose, tuple) and len(fallback_pose) >= 3:
                        _fallback_label, q_pre_dump, _fallback_duration = fallback_pose[:3]
                    elif isinstance(fallback_pose, tuple) and len(fallback_pose) >= 2:
                        q_pre_dump, fallback_info = fallback_pose[:2]
                    else:
                        q_pre_dump = fallback_pose
                    if q_pre_dump is None:
                        fail_reasons.append(f"{label} dump-ready pose failed: {dump_info}; carry fallback failed: {fallback_info}")
                        continue
                    q_pre_dump = clip_command_near(q_pre_dump, reference=q_seed)
                    q_dump = bucket_only_dump_pose(q_pre_dump, dump_deg)
                    drop = unload_drop_report(q=q_dump, reference_q=q_pre_dump)
                    info_print(
                        "[DIG PLAN UNLOAD FALLBACK]",
                        f"candidate={candidate.get('id', 'candidate')}",
                        f"reason={dump_info}",
                        f"fallback_drop_xy_err={fmt_optional(drop.get('xy_err'))}",
                        f"inside_xy={drop.get('inside_xy')}",
                        f"close_xy={drop.get('close_xy')}",
                    )
                else:
                    q_pre_dump = np.array(q_dump, dtype=np.float32).copy()
                    bucket_idx = CTRL.name_to_idx["bucket"]
                    reference_angles = chain_angles_from_q(q_seed, end_effector="load")
                    carry_calc = None
                    if reference_angles is not None:
                        carry_world = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_seed, end_effector="load")
                        carry_calc = bucket_joint_for_world_angle(q_pre_dump, carry_world, end_effector="load")
                    if carry_calc is not None:
                        q_pre_dump[bucket_idx] = carry_calc["bucket"]
                    else:
                        q_pre_dump[bucket_idx] = float(q_seed[bucket_idx])
                    q_pre_dump = clip_command_near(q_pre_dump, reference=q_seed)
                    drop = unload_drop_report(q=q_dump, reference_q=q_pre_dump)

                landing = unload_bin_landing_point()
                drop_ready = bool(drop.get("ok", False) and drop.get("close_xy", False))
                if not drop_ready:
                    info_print(
                        "[DIG PLAN UNLOAD DIAG]",
                        f"candidate={candidate.get('id', 'candidate')}",
                        "drop_alignment=diagnostic_only",
                        f"{label} dump-ready pose failed: predicted drop not aligned: "
                        f"xy_err={fmt_optional(drop.get('xy_err'))} "
                        f"inside_xy={drop.get('inside_xy')} above_wall={drop.get('above_wall')} "
                        f"close_xy={drop.get('close_xy')} "
                        f"source_clearance={fmt_optional(drop.get('source_clearance'))}",
                    )
                motion = plan_joint_motion_metrics(q_pre_dump, q_seed, duration)
                path_penalty, path_detail = plan_path_penalty(q_seed, q_pre_dump, label)
                xy_err = float(drop.get("xy_err", 1.0) or 1.0)
                clearance_short = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - float(drop.get("source_clearance", 0.0) or 0.0))
                outside_penalty = 0.0 if bool(drop.get("inside_xy", False)) else 8.0
                close_penalty = 0.0 if bool(drop.get("close_xy", False)) else 6.0
                unload_penalty = float(DIG_PLAN_UNLOAD_XY_COST) * xy_err + 18.0 * clearance_short + outside_penalty + close_penalty
                total_cost = float(beam["cost"]) + motion["cost"] + path_penalty + unload_penalty
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(landing, 3),
                    "q_goal_rad": vec_list(q_pre_dump, 4),
                    "q_dump_rad": vec_list(q_dump, 4),
                    "q_goal_deg": q_deg_values(q_pre_dump, wrap_swing_for_display=True),
                    "q_dump_deg": q_deg_values(q_dump, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": "landing",
                    "drop": compact_unload_drop(drop),
                    "drop_alignment_ready": bool(drop_ready),
                    "drop_alignment_policy": "diagnostic_only_execute_then_score",
                    "motion": motion,
                    "path": path_detail,
                    "stage_cost": float(motion["cost"] + path_penalty + unload_penalty),
                }
                new_beams.append(
                    {
                        "q": q_pre_dump.copy(),
                        "seq": beam["seq"] + [(label, q_pre_dump.copy(), float(duration))],
                        "points": beam["points"] + [np.array(landing, dtype=np.float32).copy()],
                        "stages": beam["stages"] + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + float(motion["estimated_time"]),
                    }
                )
                continue

            if label == "curl_to_hold_material":
                q_goal = q_seed.copy()
                bucket_idx = CTRL.name_to_idx["bucket"]
                q_goal[bucket_idx] = deg_to_rad(float(bucket_deg if bucket_deg is not None else candidate.get("bucket_curl", -105.0)))
                q_goal = clip_command_near(q_goal, reference=q_seed)
                motion = plan_joint_motion_metrics(q_goal, q_seed, duration)
                path_penalty, path_detail = plan_path_penalty(q_seed, q_goal, label)
                target_point = predicted_end_world_point(q_goal, end_effector="tip", reference_q=q_seed)
                if target_point is None:
                    target_point = point
                total_cost = float(beam["cost"]) + motion["cost"] + path_penalty
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(target_point, 3),
                    "q_goal_rad": vec_list(q_goal, 4),
                    "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": "bucket_only_seal",
                    "seal_bucket_first": True,
                    "bucket_target_deg": float(rad_to_deg(q_goal[bucket_idx])),
                    "motion": motion,
                    "path": path_detail,
                    "stage_cost": float(motion["cost"] + path_penalty),
                }
                new_beams.append(
                    {
                        "q": q_goal.copy(),
                        "seq": beam["seq"] + [(label, q_goal.copy(), float(duration))],
                        "points": beam["points"] + [np.array(target_point, dtype=np.float32).copy()],
                        "stages": beam["stages"] + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + float(motion["estimated_time"]),
                    }
                )
                continue

            bucket_deg_use, bucket_world_use, bucket_reason = resolve_bucket_world(label, bucket_deg, bucket_world_deg, q_seed, ik_effector)
            if bucket_reason != "ok":
                fail_reasons.append(bucket_reason)
                continue

            if "lift" in label or "carry" in label:
                accept_err = 0.55
                soft_accept = DIG_PLAN_LIFT_SOFT_ACCEPT_ERR
                motion_weight = 0.65
                preference_weight = 0.0 if bucket_world_use is not None else 2.0
            elif is_cutting_phase(label) or is_curl_phase(label):
                accept_err = 0.38
                soft_accept = DIG_PLAN_CUT_SOFT_ACCEPT_ERR
                motion_weight = 0.55
                preference_weight = 1.8 if bucket_world_use is not None else None
            else:
                accept_err = 0.40
                soft_accept = 0.52
                motion_weight = 0.55
                preference_weight = 1.4 if bucket_world_use is not None else None

            pose_rows, pose_reason = solve_dig_pose_candidates(
                label,
                point,
                bucket_deg_use,
                duration,
                q_seed,
                bucket_world_deg=bucket_world_use,
                ik_effector=ik_effector,
                accept_err=accept_err,
                soft_accept_err=soft_accept,
                bucket_motion_weight=motion_weight,
                bucket_preference_weight=preference_weight,
                max_solutions=DIG_PLAN_TOPK_IK,
            )
            if not pose_rows:
                fail_reasons.append(f"{label}: {pose_reason}")
                continue

            for pose in pose_rows:
                q_goal = np.array(pose["q_goal"], dtype=np.float32).copy()
                info = pose.get("info", {})
                motion = plan_joint_motion_metrics(q_goal, q_seed, duration)
                path_penalty, path_detail = plan_path_penalty(q_seed, q_goal, label)
                ik_penalty = float(DIG_PLAN_IK_ERR_COST) * float(info.get("planar_err", 0.0) or 0.0)
                angle_penalty = 0.35 * max(0.0, float(info.get("world_angle_err_deg", 0.0) or 0.0))
                total_cost = float(beam["cost"]) + motion["cost"] + path_penalty + ik_penalty + angle_penalty
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(point, 3),
                    "q_goal_rad": vec_list(q_goal, 4),
                    "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": ik_effector,
                    "planar_err": float(info.get("planar_err", 0.0) or 0.0),
                    "world_angle_err_deg": float(info.get("world_angle_err_deg", 0.0) or 0.0),
                    "end_z": float(info.get("end_z", 0.0) or 0.0),
                    "min_z": float(info.get("min_z", 0.0) or 0.0),
                    "motion": motion,
                    "path": path_detail,
                    "stage_cost": float(motion["cost"] + path_penalty + ik_penalty + angle_penalty),
                }
                if bucket_world_use is not None:
                    stage_row["bucket_world_deg"] = float(bucket_world_use)
                new_beams.append(
                    {
                        "q": q_goal.copy(),
                        "seq": beam["seq"] + [(label, q_goal.copy(), float(duration))],
                        "points": beam["points"] + [np.array(point, dtype=np.float32).copy()],
                        "stages": beam["stages"] + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + float(motion["estimated_time"]),
                    }
                )

        if not new_beams:
            prefix_beams = sorted(beams, key=lambda x: float(x.get("cost", 1.0e9)))
            best_partial = prefix_beams[0] if prefix_beams else best_partial
            last_reason = "; ".join(fail_reasons[:4]) if fail_reasons else f"{label} produced no beam"
            return None, list(best_partial.get("points", [])), {
                "id": str(candidate.get("id", "candidate")),
                "planned": False,
                "failed_stage": label,
                "failure_reason": last_reason,
                "planned_prefix": len(best_partial.get("seq", [])),
                "best_partial_cost": float(best_partial.get("cost", 0.0)),
                "best_partial_q_deg": q_deg_values(best_partial.get("q", CTRL.q_cmd), wrap_swing_for_display=True),
                "stages": list(best_partial.get("stages", [])),
            }

        new_beams.sort(key=lambda x: float(x.get("cost", 1.0e9)))
        beams = new_beams[: max(1, int(DIG_PLAN_BEAM_SIZE))]
        best_partial = beams[0]

    best = sorted(beams, key=lambda x: float(x.get("cost", 1.0e9)))[0]
    unload_stage = None
    for stage in best.get("stages", []):
        if str(stage.get("phase", "")).startswith("unload"):
            unload_stage = stage
    unload_landing_point = best["points"][-1] if best["points"] else unload_bin_landing_point()
    unload_release_point = unload_bin_dump_point()
    if isinstance(unload_stage, dict):
        drop = unload_stage.get("drop", {})
        if isinstance(drop, dict) and drop.get("release") is not None:
            unload_release_point = np.array(drop.get("release"), dtype=np.float32).reshape(-1)[:3]
        if unload_stage.get("target_point") is not None:
            unload_landing_point = np.array(unload_stage.get("target_point"), dtype=np.float32).reshape(-1)[:3]
    return best["seq"], best["points"], {
        "id": str(candidate.get("id", "candidate")),
        "planned": True,
        "steps": len(best["seq"]),
        "stages": best["stages"],
        "unload_point_xyz": vec_list(unload_release_point, 3),
        "unload_release_xyz": vec_list(unload_release_point, 3),
        "unload_landing_xyz": vec_list(unload_landing_point, 3),
        "planner_cost": float(best["cost"]),
        "weighted_angle": float(best["weighted_angle"]),
        "estimated_time": float(best["estimated_time"]),
        "beam_size": int(DIG_PLAN_BEAM_SIZE),
    }


def evaluate_dig_plan_candidate(seq, points, candidate):
    if not seq:
        return -1.0e9, "no_sequence", []

    q_prev = CTRL.q_cmd.copy()
    penalties = 0.0
    reports = []
    for stage_name, q_goal, duration in seq:
        phase_ok, phase_reason, phase_sample, phase_report = path_phase_check(q_prev, q_goal, stage_name)
        obstacle_ok, obstacle_reason, obstacle_sample, obstacle_report = path_obstacle_check(q_prev, q_goal, stage_name)
        deltas = q_delta_abs_deg(q_goal, q_prev)
        motion_cost = 0.025 * float(sum(deltas))
        penalties += motion_cost
        stage_report = {
            "phase": stage_name,
            "phase_ok": bool(phase_ok),
            "obstacle_ok": bool(obstacle_ok),
            "phase_reason": str(phase_reason),
            "obstacle_reason": str(obstacle_reason),
            "phase_sample": int(phase_sample),
            "obstacle_sample": int(obstacle_sample),
            "joint_delta_deg": deltas,
            "motion_cost": motion_cost,
        }
        if not phase_ok:
            penalties += 10.0
            stage_report["phase_report"] = phase_report
        if not obstacle_ok:
            penalties += 18.0
            stage_report["obstacle_report"] = obstacle_report
        reports.append(stage_report)
        q_prev = np.array(q_goal, dtype=np.float32).copy()

    unload_point = np.array(points[-1], dtype=np.float32) if points else unload_bin_dump_point()
    dump_deg = float(candidate.get("unload_dump_deg", BUCKET_UNLOAD_DUMP_DEG))
    q_dump, dump_info = plan_dump_pose_to_bin(
        q_seed=np.array(seq[-1][1], dtype=np.float32),
        dump_deg=dump_deg,
        label=f"score_{candidate.get('id', 'candidate')}",
        log=False,
        allow_unaligned=True,
    )
    if q_dump is None:
        alignment_cost = 45.0
        dump_reason = str(dump_info)
    else:
        drop = unload_drop_report(q=q_dump, reference_q=seq[-1][1])
        if drop.get("landing") is None:
            alignment_cost = 45.0
            dump_reason = "missing predicted drop landing"
        else:
            xy_err = float(drop.get("xy_err", 999.0))
            clearance_short = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - float(drop.get("source_clearance", 0.0)))
            outside_penalty = 0.0 if bool(drop.get("inside_xy", False)) else 8.0
            alignment_cost = 28.0 * xy_err + 18.0 * clearance_short + outside_penalty
            dump_reason = (
                f"drop_xy_err={xy_err:.3f}; release_clearance={float(drop.get('source_clearance', 0.0)):.3f}; "
                f"inside_xy={drop.get('inside_xy')}"
            )
    penalties += alignment_cost

    lift_bonus = min(6.0, max(0.0, float(candidate.get("lift_height", 0.0)) * 3.0))
    score = 100.0 + lift_bonus - penalties
    reason = f"penalty={penalties:.2f}; alignment_cost={alignment_cost:.2f}; {dump_reason}; lift_bonus={lift_bonus:.2f}"
    return float(score), reason, reports


def build_shared_dig_plan_object(target_xyz, seq, points, chosen_row):
    chosen_row = chosen_row if isinstance(chosen_row, dict) else {}
    seq = [] if seq is None else list(seq)
    points = [] if points is None else list(points)
    stages = auto_collect_plan_summary(seq)
    trace_points = STATE.get("dig_plan_trace_points", []) or []
    unload_landing = chosen_row.get("unload_landing_xyz") or vec_list(unload_bin_landing_point(), 3)
    unload_release = (
        chosen_row.get("unload_release_xyz")
        or chosen_row.get("unload_point_xyz")
        or vec_list(unload_bin_dump_point(), 3)
    )
    plan_id = f"plan_{int(time.time() * 1000)}_{stable_json_hash([vec_list(target_xyz, 3), unload_landing, unload_release])}"
    total_duration = float(sum(float(item[2]) for item in seq)) if seq else 0.0
    plan = {
        "plan_id": plan_id,
        "episode_id": str(STATE.get("dataset_episode_uid", "")),
        "planner_version": PLANNER_VERSION,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "config_hash": current_config_hash(),
        "created_at": time.time(),
        "chosen_dig_target": vec_list(target_xyz, 3),
        "chosen_unload_landing_point": vec_list(unload_landing, 3),
        "chosen_unload_release_point": vec_list(unload_release, 3),
        "stage_sequence": stages,
        "semantic_target_points": [vec_list(p, 3) for p in points],
        "planned_path_points": [vec_list(p, 3) for p in trace_points],
        "trace_stage_breaks": STATE.get("dig_plan_trace_stage_breaks", []),
        "candidate_scores": [compact_plan_candidate(x, include_stages=False) for x in STATE.get("last_dig_plan_candidates", [])],
        "auto_dig_target_scores": STATE.get("last_auto_dig_target_scores", []),
        "auto_unload_scores": STATE.get("last_auto_unload_scores", []),
        "chosen_candidate": compact_plan_candidate(chosen_row, include_stages=True),
        "total_plan_cost": None if chosen_row.get("planner_cost") is None else float(chosen_row.get("planner_cost")),
        "rank_cost": None if chosen_row.get("rank_cost") is None else float(chosen_row.get("rank_cost")),
        "score": None if chosen_row.get("score") is None else float(chosen_row.get("score")),
        "estimated_duration": float(chosen_row.get("estimated_time", total_duration) or total_duration),
        "stage_count": len(seq),
        "debug": {
            "build_ms": float(STATE.get("dig_plan_last_build_ms", 0.0)),
            "planning_version": int(STATE.get("dig_plan_planning_version", 0)),
            "start_q_deg": q_deg_values(STATE.get("dig_plan_start_q", CTRL.q_cmd), wrap_swing_for_display=True),
        },
    }
    STATE["current_dig_plan"] = plan
    STATE["active_unload_landing_point"] = np.array(unload_landing, dtype=np.float32).reshape(-1)[:3].copy()
    STATE["active_unload_release_point"] = np.array(unload_release, dtype=np.float32).reshape(-1)[:3].copy()
    return plan


def _planned_stage_q(row, key):
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        q = np.array(value, dtype=np.float32).reshape(-1)[:4]
        if len(q) < 4:
            return None
        return q.copy()
    except Exception:
        return None


def planned_unload_stage_detail(stage_index=None, stage_name=None):
    candidate = STATE.get("dig_plan_candidate")
    stages = candidate.get("stages", []) if isinstance(candidate, dict) else []
    if not stages:
        return None

    selected = None
    try:
        idx = int(stage_index) if stage_index is not None else None
    except Exception:
        idx = None
    if idx is not None and 0 <= idx < len(stages):
        row = stages[idx]
        if "unload" in str(row.get("phase", "")).lower():
            selected = row

    if selected is None:
        name = "" if stage_name is None else str(stage_name)
        for row in stages:
            phase = str(row.get("phase", ""))
            if "unload" not in phase.lower():
                continue
            if not name or phase == name or name.startswith(phase):
                selected = row
                break

    if selected is None:
        return None

    q_goal = _planned_stage_q(selected, "q_goal_rad")
    q_dump = _planned_stage_q(selected, "q_dump_rad")
    if q_goal is None and q_dump is None:
        return None
    return {
        "phase": str(selected.get("phase", "")),
        "q_goal": None if q_goal is None else q_goal.copy(),
        "q_dump": None if q_dump is None else q_dump.copy(),
        "drop": selected.get("drop", {}),
        "target_point": selected.get("target_point"),
        "duration": selected.get("duration"),
    }


def plan_dig_sequence_from_target(target_xyz):
    STATE["last_dig_plan_candidates"] = []
    STATE["dig_plan_candidate"] = None
    STATE["dig_plan_best_failure"] = None
    best = None
    best_failure = None

    candidates = adaptive_dig_plan_candidates(target_xyz)
    deadline = time.time() + float(DIG_PLAN_MAX_BUILD_SECONDS)
    for candidate in candidates:
        if time.time() > deadline:
            info_print(
                "[DIG PLAN TIMEOUT]",
                f"budget={DIG_PLAN_MAX_BUILD_SECONDS:.2f}s",
                f"evaluated={len(STATE.get('last_dig_plan_candidates', []))}",
                f"best_ready={best is not None}",
            )
            break
        seq, points, detail = plan_dig_sequence_candidate(target_xyz, candidate, deadline=deadline)
        row = dict(detail)
        row["candidate"] = dict(candidate)
        if seq:
            legacy_score, score_reason, reports = evaluate_dig_plan_candidate(seq, points, candidate)
            planner_cost = float(row.get("planner_cost", 1.0e9))
            row["rank_cost"] = planner_cost
            row["score"] = float(1000.0 - planner_cost)
            row["legacy_score"] = float(legacy_score)
            row["score_reason"] = (
                f"planner_cost={planner_cost:.2f}; "
                f"weighted_angle={float(row.get('weighted_angle', 0.0)):.2f}; "
                f"estimated_time={float(row.get('estimated_time', 0.0)):.2f}; "
                + score_reason
            )
            row["path_reports"] = reports
            if best is None or planner_cost < best["rank_cost"]:
                best = {
                    "score": float(row["score"]),
                    "rank_cost": planner_cost,
                    "sequence": seq,
                    "points": points,
                    "candidate": dict(candidate),
                    "row": row,
                }
        else:
            row["score"] = -1.0e9
            row["score_reason"] = row.get("failure_reason", "planning_failed")
            info_print(
                f"[DIG PLAN CANDIDATE FAIL] {row.get('id')}: "
                f"stage={row.get('failed_stage')} prefix={row.get('planned_prefix', 0)} "
                f"reason={row.get('failure_reason')}"
            )
            if best_failure is None:
                best_failure = row
            else:
                old_prefix = int(best_failure.get("planned_prefix", 0) or 0)
                new_prefix = int(row.get("planned_prefix", 0) or 0)
                if new_prefix > old_prefix:
                    best_failure = row
                elif new_prefix == old_prefix and float(row.get("best_partial_cost", 1.0e9)) < float(best_failure.get("best_partial_cost", 1.0e9)):
                    best_failure = row
        STATE["last_dig_plan_candidates"].append(row)

    if best is None:
        STATE["dig_plan_best_failure"] = best_failure
        update_status("[DIG PLAN BLOCKED] all candidate plans failed", force=True)
        return None

    chosen = dict(best["row"])
    chosen["selected"] = True
    STATE["dig_plan_candidate"] = chosen
    STATE["dig_plan_points"] = [np.array(p, dtype=np.float32).copy() for p in best["points"]]
    cache_dig_plan_trace_points(best["sequence"], start_q=STATE.get("dig_plan_start_q", CTRL.q_cmd.copy()))
    shared_plan = build_shared_dig_plan_object(target_xyz, best["sequence"], best["points"], chosen)
    info_print(
        f"[DIG PLAN SELECT] plan={shared_plan.get('plan_id')} candidate={chosen.get('id')} score={best['score']:.2f} "
        f"rank_cost={best['rank_cost']:.2f} "
        f"landing={shared_plan.get('chosen_unload_landing_point')} "
        f"release={shared_plan.get('chosen_unload_release_point')} "
        f"reason={chosen.get('score_reason')}"
    )
    marker_point = chosen.get("unload_landing_xyz") or chosen.get("unload_point_xyz")
    log_unload_context("dig_plan_selected", target_xyz=target_xyz, unload_point=marker_point)
    ensure_unload_marker(marker_point, label="dig_plan_selected")
    return best["sequence"]


def build_dig_plan_from_current_target(force_status=True):
    if bool(STATE.get("dig_plan_planning_active", False)):
        seq_existing = STATE.get("dig_plan_sequence", None)
        if seq_existing is not None and dig_plan_target_matches_current():
            return seq_existing
        update_status("[DIG PLAN] planner is already running; wait for current build", force=force_status)
        return None

    STATE["dig_plan_planning_active"] = True
    STATE["dig_plan_planning_version"] = int(STATE.get("dig_plan_planning_version", 0)) + 1
    STATE["dig_plan_planning_source"] = "manual_or_sync"
    build_t0 = time.time()
    target = get_target_pos()
    target[2] = max(float(target[2]), GROUND_TOP_Z)

    try:
        ok, reason = validate_dig_target(target, hard_block=False)
        if not ok:
            set_target_color(TARGET_COLOR_UNREACHABLE)
            update_status("[DIG BLOCKED] " + reason, force=True)
            return None

        STATE["dig_plan_points"] = None
        STATE["dig_plan_trace_points"] = []
        STATE["dig_plan_trace_stage_breaks"] = []
        STATE["last_dig_plan_candidates"] = []
        STATE["dig_plan_candidate"] = None
        STATE["dig_plan_best_failure"] = None
        STATE["current_dig_plan"] = None
        STATE["dig_plan_start_q"] = CTRL.q_cmd.copy()
        seq = plan_dig_sequence_from_target(target)
        if not seq:
            best_failure = STATE.get("dig_plan_best_failure")
            STATE["dig_plan_points"] = None
            STATE["dig_plan_trace_points"] = []
            STATE["dig_plan_trace_stage_breaks"] = []
            STATE["dig_plan_sequence"] = None
            STATE["dig_plan_candidate"] = None
            STATE["current_dig_plan"] = None
            STATE["trace_planned_bucket_points"] = []
            STATE["trace_render_dirty"] = True
            STATE["trace_render_signature"] = None
            set_target_color(TARGET_COLOR_UNREACHABLE)
            if isinstance(best_failure, dict):
                update_status(
                    f"[DIG BLOCKED] no IK dig step could be planned. "
                    f"best_prefix={best_failure.get('planned_prefix', 0)} "
                    f"failed_stage={best_failure.get('failed_stage')} "
                    f"reason={best_failure.get('failure_reason')}",
                    force=True,
                )
            else:
                update_status("[DIG BLOCKED] no IK dig step could be planned.", force=True)
            return None

        STATE["dig_plan_target"] = target.copy()
        STATE["dig_plan_sequence"] = seq
        STATE["dig_plan_step_index"] = 0

        if force_status:
            set_target_color(TARGET_COLOR_REACHABLE)
            update_status(
                f"[DIG PLAN] {len(seq)} steps for target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})",
                force=True,
            )

        if current_trace_mode() == 2:
            draw_trace(force=True)

        return seq
    finally:
        STATE["dig_plan_last_build_ms"] = 1000.0 * max(0.0, time.time() - build_t0)
        STATE["dig_plan_planning_active"] = False


async def build_dig_plan_from_current_target_task(force_status=True):
    if bool(STATE.get("dig_plan_planning_active", False)):
        update_status("[DIG PLAN] planner already running", force=True)
        return None
    update_status("[DIG PLAN] planning...", force=True)
    await step_updates(1)
    seq = build_dig_plan_from_current_target(force_status=force_status)
    await step_updates(1)
    if seq:
        info_print(
            "[DIG PLAN READY]",
            f"steps={len(seq)}",
            f"build_ms={float(STATE.get('dig_plan_last_build_ms', 0.0)):.1f}",
            f"trace_points={len(STATE.get('dig_plan_trace_points', []) or [])}",
        )
    else:
        best_failure = STATE.get("dig_plan_best_failure")
        if isinstance(best_failure, dict):
            info_print(
                "[DIG PLAN FAILED]",
                f"failed_stage={best_failure.get('failed_stage')}",
                f"prefix={best_failure.get('planned_prefix', 0)}",
                f"reason={best_failure.get('failure_reason')}",
                f"build_ms={float(STATE.get('dig_plan_last_build_ms', 0.0)):.1f}",
            )
    return seq


def reset_dig_plan():
    STATE["dig_plan_target"] = None
    STATE["dig_plan_sequence"] = None
    STATE["dig_plan_points"] = None
    STATE["dig_plan_start_q"] = None
    STATE["dig_plan_trace_points"] = []
    STATE["dig_plan_trace_stage_breaks"] = []
    STATE["dig_plan_best_failure"] = None
    STATE["last_dig_plan_candidates"] = []
    STATE["dig_plan_candidate"] = None
    STATE["current_dig_plan"] = None
    STATE["dig_plan_step_index"] = 0
    STATE["active_plan_stage_index"] = -1
    STATE["active_unload_landing_point"] = None
    STATE["active_unload_release_point"] = None
    STATE["trace_planned_bucket_points"] = []
    STATE["trace_active_motion"] = None
    STATE["trace_render_dirty"] = True
    STATE["trace_render_signature"] = None
    ensure_unload_marker(label="reset_plan")
    if current_trace_mode() == 2:
        draw_trace(force=True)
    update_status("[DIG PLAN] reset", force=True)


def get_or_build_dig_plan():
    seq = STATE.get("dig_plan_sequence", None)
    if seq is None:
        seq = build_dig_plan_from_current_target(force_status=True)
    return seq


def plan_unload_from_current():
    point = unload_bin_dump_point()
    log_unload_context("manual_unload", target_xyz=get_target_pos() if TARGET_PATH else None, unload_point=point)
    ensure_unload_marker(point, label="manual_unload")
    q_seed = CTRL.q_cmd.copy()
    dump_deg = unload_dump_target_deg()
    q_dump, dump_info = plan_dump_pose_to_bin(
        q_seed=q_seed,
        dump_deg=dump_deg,
        label="manual_unload_to_bin",
        log=True,
        allow_unaligned=True,
    )
    if q_dump is None:
        update_status(f"[UNLOAD BLOCKED] could not plan dump landing above unload bin: {dump_info}", force=True)
        return None
    q_pre_dump = np.array(q_dump, dtype=np.float32).copy()
    bucket_idx = CTRL.name_to_idx["bucket"]
    reference_angles = chain_angles_from_q(q_seed, end_effector="load")
    carry_calc = None
    if reference_angles is not None:
        carry_world = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_seed, end_effector="load")
        carry_calc = bucket_joint_for_world_angle(q_pre_dump, carry_world, end_effector="load")
    if carry_calc is not None:
        q_pre_dump[bucket_idx] = carry_calc["bucket"]
    else:
        q_pre_dump[bucket_idx] = float(q_seed[bucket_idx])
    q_pre_dump = clip_command_near(q_pre_dump, reference=q_seed)
    info_print(
        f"[UNLOAD PLAN] manual_unload_to_bin: landing_target={vec_list(unload_bin_landing_point(), 3)} "
        f"release_target={dump_info.get('release_target')} q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)} "
        f"q_dump={q_deg_values(q_dump, wrap_swing_for_display=True)}"
    )
    return ("unload_to_bin", q_pre_dump, 1.35)


def unload_dump_target_deg():
    candidate = STATE.get("dig_plan_candidate")
    dump_deg = BUCKET_UNLOAD_DUMP_DEG
    if isinstance(candidate, dict):
        try:
            dump_deg = float((candidate.get("candidate") or candidate).get("unload_dump_deg", dump_deg))
        except Exception:
            dump_deg = BUCKET_UNLOAD_DUMP_DEG
    return float(dump_deg)


def plan_dump_pose_to_bin(q_seed=None, dump_deg=None, label="unload_dump", log=True, max_correction_iters=None, allow_unaligned=False):
    if q_seed is None:
        q_seed = CTRL.q_cmd.copy()
    else:
        q_seed = np.array(q_seed, dtype=np.float32).copy()
    if dump_deg is None:
        dump_deg = unload_dump_target_deg()

    bucket_idx = CTRL.name_to_idx["bucket"]
    q_seed_dump = q_seed.copy()
    q_seed_dump[bucket_idx] = deg_to_rad(float(dump_deg))
    q_seed_dump = clip_command_near(q_seed_dump, reference=q_seed)
    ctx = task_scene_context()
    drop_target = unload_bin_landing_point(ctx=ctx)
    nominal_release_target = unload_bin_dump_point(ctx=ctx)
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
    wall_z = float(bin_z_range[1]) if len(bin_z_range) >= 2 else GROUND_TOP_Z
    min_release_z = max(float(nominal_release_target[2]), wall_z + UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z, GROUND_TOP_Z + UNLOAD_TARGET_MIN_Z)
    pour_target, initial_drift = unload_release_target_for_landing(
        drop_target,
        q_estimate=q_seed_dump,
        q_start=q_seed,
        release_z=min_release_z,
        wall_z=wall_z,
    )
    best = None
    last_reason = "no dump candidate evaluated"

    correction_iters = max(1, int(UNLOAD_DROP_IK_CORRECTION_ITERS if max_correction_iters is None else max_correction_iters))
    for attempt in range(correction_iters):
        q_dump, info = solve_priority_ik_to_target(
            pour_target,
            q_seed=q_seed_dump,
            preferred_bucket_rad=deg_to_rad(float(dump_deg)),
            preferred_end_angle_rad=None,
            bucket_motion_weight=0.35,
            bucket_preference_weight=8.0,
            min_world_z=GROUND_TOP_Z + IK_DIG_MIN_CLEARANCE,
            accept_err=UNLOAD_DUMP_ACCEPT_ERR,
            end_effector="pour",
            allow_end_below=False,
            min_end_z=max(GROUND_TOP_Z + 0.02, min_release_z - 0.12),
            phase_mode="unload_dump",
            use_refinement=True,
            bucket_candidate_span_deg=95.0,
            bucket_candidate_count=25,
        )

        if q_dump is None:
            last_reason = str(info)
            break

        q_ik = clip_command_near(q_dump, reference=q_seed)
        ik_bucket_err = abs(rad_to_deg(wrap_angle(float(q_ik[bucket_idx]) - deg_to_rad(float(dump_deg)))))
        q_dump = q_ik.copy()
        q_dump[bucket_idx] = deg_to_rad(float(dump_deg))
        q_dump = clip_command_near(q_dump, reference=q_seed)
        bucket_err = abs(rad_to_deg(wrap_angle(float(q_dump[bucket_idx]) - deg_to_rad(float(dump_deg)))))
        drop = unload_drop_report(q=q_dump, reference_q=q_seed)
        xy_err = float(drop.get("xy_err", 999.0))
        source_clearance = float(drop.get("source_clearance", -999.0))
        source_penalty = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - source_clearance)
        bucket_penalty = max(0.0, bucket_err - UNLOAD_DUMP_BUCKET_TOL_DEG)
        ik_bucket_penalty = max(0.0, ik_bucket_err - UNLOAD_DUMP_BUCKET_TOL_DEG)
        planar_err = float(info.get("planar_err", 0.0)) if isinstance(info, dict) else 0.0
        cost = 100.0 * xy_err + 35.0 * source_penalty + 0.6 * bucket_penalty + 0.15 * ik_bucket_penalty + 5.0 * planar_err
        row = {
            "q": q_dump,
            "info": dict(info) if isinstance(info, dict) else {"reason": str(info)},
            "drop": drop,
            "bucket_err": float(bucket_err),
            "ik_bucket_err": float(ik_bucket_err),
            "cost": float(cost),
            "attempt": int(attempt),
            "pour_target": pour_target.copy(),
        }
        if best is None or row["cost"] < best["cost"]:
            best = row

        if bool(drop.get("ok", False)) and bool(drop.get("close_xy", False)):
            row["info"]["drop"] = drop
            row["info"]["drop_target"] = vec_list(drop_target, 3)
            row["info"]["pour_target"] = vec_list(pour_target, 3)
            row["info"]["release_target"] = vec_list(pour_target, 3)
            row["info"]["dump_bucket_target_deg"] = float(dump_deg)
            row["info"]["dump_bucket_err_deg"] = float(bucket_err)
            row["info"]["ik_bucket_err_deg"] = float(ik_bucket_err)
            if log:
                raw_swing = row["info"].get("raw_swing_goal")
                swing_goal = row["info"].get("swing_goal")
                info_print(
                    f"[UNLOAD DUMP PLAN] {label}: "
                    f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(pour_target, 3)} "
                    f"landing={drop.get('landing')} release={drop.get('release')} "
                    f"xy_err={fmt_optional(drop.get('xy_err'))} source_clearance={fmt_optional(drop.get('source_clearance'))} "
                    f"dump_deg={float(dump_deg):.2f} bucket_err={bucket_err:.2f}deg "
                    f"ik_bucket_err={ik_bucket_err:.2f}deg "
                    f"attempt={attempt} drift={fmt_optional(drop.get('drift_distance'))} "
                    f"drift_xy={drop.get('drift_xy')} "
                    f"swing_goal={fmt_optional(None if swing_goal is None else rad_to_deg(swing_goal))}deg "
                    f"raw_swing={fmt_optional(None if raw_swing is None else rad_to_deg(raw_swing))}deg "
                    f"planar_err={fmt_optional(row['info'].get('planar_err'))} "
                    f"q={q_deg_values(q_dump, wrap_swing_for_display=True)}"
                )
            return q_dump, row["info"]

        if drop.get("dx") is None or drop.get("dy") is None:
            last_reason = "drop report missing xy error"
            break

        err_xy = np.array([float(drop.get("dx", 0.0)), float(drop.get("dy", 0.0))], dtype=np.float32)
        err_norm = float(np.linalg.norm(err_xy))
        if err_norm < 0.01:
            last_reason = "drop xy is close but failed unload landing checks"
            break
        if err_norm > UNLOAD_DROP_MAX_XY_CORRECTION:
            err_xy *= float(UNLOAD_DROP_MAX_XY_CORRECTION / err_norm)
        pour_target[0] -= float(err_xy[0])
        pour_target[1] -= float(err_xy[1])
        pour_target[2] = min_release_z
        last_reason = (
            f"drop correction attempt={attempt} xy_err={xy_err:.3f} "
            f"bucket_err={bucket_err:.2f}deg source_clearance={source_clearance:.3f}"
        )

    if (
        best is not None
        and bool(best.get("drop", {}).get("ok", False))
        and bool(best.get("drop", {}).get("close_xy", False))
    ):
        drop = best["drop"]
        best["info"]["drop"] = drop
        best["info"]["drop_target"] = vec_list(drop_target, 3)
        best["info"]["pour_target"] = vec_list(best.get("pour_target"), 3)
        best["info"]["release_target"] = vec_list(best.get("pour_target"), 3)
        best["info"]["dump_bucket_target_deg"] = float(dump_deg)
        best["info"]["dump_bucket_err_deg"] = float(best.get("bucket_err", 0.0))
        best["info"]["ik_bucket_err_deg"] = float(best.get("ik_bucket_err", 0.0))
        if log:
            info_print(
                f"[UNLOAD DUMP PLAN] {label}: accepted_inside_bin_fallback "
                f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(best.get('pour_target'), 3)} "
                f"landing={drop.get('landing')} "
                f"xy_err={fmt_optional(drop.get('xy_err'))} close_xy={drop.get('close_xy')} "
                f"source_clearance={fmt_optional(drop.get('source_clearance'))} "
                f"drift={fmt_optional(drop.get('drift_distance'))} drift_xy={drop.get('drift_xy')} "
                f"q={q_deg_values(best['q'], wrap_swing_for_display=True)}"
            )
        return best["q"], best["info"]

    if best is not None:
        drop = best["drop"]
        reason = (
            f"predicted drop not aligned: xy_err={fmt_optional(drop.get('xy_err'))} "
            f"inside_xy={drop.get('inside_xy')} above_wall={drop.get('above_wall')} "
            f"close_xy={drop.get('close_xy')} bucket_err={best['bucket_err']:.2f}deg "
            f"ik_bucket_err={best.get('ik_bucket_err', 0.0):.2f}deg "
            f"source_clearance={fmt_optional(drop.get('source_clearance'))}"
        )
        if allow_unaligned:
            best["info"]["drop"] = drop
            best["info"]["drop_target"] = vec_list(drop_target, 3)
            best["info"]["pour_target"] = vec_list(best.get("pour_target"), 3)
            best["info"]["release_target"] = vec_list(best.get("pour_target"), 3)
            best["info"]["dump_bucket_target_deg"] = float(dump_deg)
            best["info"]["dump_bucket_err_deg"] = float(best.get("bucket_err", 0.0))
            best["info"]["ik_bucket_err_deg"] = float(best.get("ik_bucket_err", 0.0))
            best["info"]["drop_alignment_ready"] = False
            best["info"]["drop_alignment_policy"] = "diagnostic_only_execute_then_score"
            best["info"]["drop_alignment_reason"] = reason
            if log:
                info_print(
                    f"[UNLOAD DUMP PLAN DIAG] {label}: using_best_effort_unaligned_pose "
                    f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(best.get('pour_target'), 3)} "
                    f"landing={drop.get('landing')} release={drop.get('release')} "
                    f"reason={reason} "
                    f"drift={fmt_optional(drop.get('drift_distance'))} drift_xy={drop.get('drift_xy')} "
                    f"q={q_deg_values(best['q'], wrap_swing_for_display=True)}"
                )
            return best["q"], best["info"]
    else:
        reason = last_reason
    if log:
        info_print(
            f"[UNLOAD DUMP PLAN FAIL] {label}: "
            f"landing_target={vec_list(drop_target, 3)} "
            f"initial_release_target={vec_list(pour_target, 3)} "
            f"initial_drift={fmt_optional(initial_drift.get('drift_distance'))} "
            f"reason={reason}"
        )
    return None, reason


async def execute_unload_sequence(stage_name, q_goal, duration, task_id=None):
    ready, reason, _detail = await wait_for_articulation_action_ready(
        f"{stage_name}_unload_start",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_STAGE_MAX_WAIT_FRAMES,
        record_failure=True,
    )
    if not ready:
        update_status(f"[DIG EXEC FAILED] {stage_name}: action_channel_not_ready; {reason}", force=True)
        return False
    q_start = sync_motion_start_q(stage_name)
    q_goal = clip_command_near(q_goal, reference=q_start)
    swing_idx = CTRL.name_to_idx["swing"]
    swing_delta_deg = abs(rad_to_deg(swing_delta(q_goal[swing_idx], q_start[swing_idx])))
    joint_delta = q_delta_abs_deg(q_goal, q_start)
    info_print(
        f"[UNLOAD ROUTE] {stage_name}: mode=direct_cached_plan "
        f"swing_delta={swing_delta_deg:.2f}deg "
        f"joint_delta={joint_delta} "
        f"q_start={q_deg_values(q_start, wrap_swing_for_display=True)} "
        f"q_goal={q_deg_values(q_goal, wrap_swing_for_display=True)}"
    )
    try:
        trace_points = cache_active_stage_trace_points(
            stage_name,
            q_start,
            q_goal,
            stage_index=STATE.get("active_plan_stage_index", None),
            include_remaining=True,
        )
        if current_trace_mode() == 2:
            draw_trace(force=True)
        info_print(
            "[PLAN EXEC TRACE]",
            f"stage={stage_name}",
            f"points={len(trace_points)}",
            "source=active_command_remaining",
        )
    except Exception as e:
        info_print("[WARN] [PLAN EXEC TRACE] unload cache failed:", stage_name, type(e).__name__, e)

    try:
        phase_ok, phase_reason, phase_sample, phase_report = path_phase_check(q_start, q_goal, stage_name)
        obstacle_ok, obstacle_reason, obstacle_sample, obstacle_report = path_obstacle_check(q_start, q_goal, stage_name)
        if phase_ok and obstacle_ok:
            info_print(f"[PLAN EXEC DIRECT] stage={stage_name} samples={PATH_CHECK_SAMPLES} ok=True")
        else:
            detail = []
            if not phase_ok:
                detail.append(
                    f"phase sample={phase_sample}/{PATH_CHECK_SAMPLES} "
                    + format_ground_report(stage_name, phase_report, phase_reason)
                )
            if not obstacle_ok:
                detail.append(
                    f"obstacle sample={obstacle_sample}/{PATH_CHECK_SAMPLES} "
                    + format_obstacle_report(stage_name, obstacle_report, obstacle_reason)
                )
            info_print(
                "[PLAN EXEC DIRECT WARN]",
                f"stage={stage_name}",
                "; ".join(detail),
                "following cached plan without hidden route injection",
            )
    except Exception as e:
        info_print("[WARN] [PLAN EXEC DIRECT] unload path precheck failed:", stage_name, type(e).__name__, e)

    success = await move_to_profile(
        q_goal,
        seconds=duration,
        label=stage_name,
        task_id=task_id,
        mode=stage_name,
        q_start_override=q_start,
    )
    if not success or (task_id is not None and not task_alive(task_id)):
        return False
    await step_updates(2)
    if task_id is not None and not task_alive(task_id):
        return False
    return verify_unload_arrival(stage_name, q_goal)


def bucket_only_dump_pose(q_reference, dump_deg):
    q = np.array(q_reference, dtype=np.float32).copy()
    q[CTRL.name_to_idx["bucket"]] = deg_to_rad(float(dump_deg))
    return clip_command_near(q, reference=q_reference)


def non_bucket_delta_deg(q_a, q_b):
    q_a = np.array(q_a, dtype=np.float32).reshape(-1)
    q_b = np.array(q_b, dtype=np.float32).reshape(-1)
    deltas = q_delta_abs_deg(q_a, q_b)
    return [
        float(deltas[CTRL.name_to_idx[name]])
        for name in ["swing", "boom", "arm"]
    ]


def current_real_q_near(reference_q=None):
    q_ref = CTRL.q_cmd.copy() if reference_q is None else np.array(reference_q, dtype=np.float32).copy()
    return q_real_near_command(get_real_joint_positions(), q_ref)


def bucket_only_dump_ready(stage_name, dump_deg, label="before_dump"):
    try:
        q_real = current_real_q_near()
    except Exception as e:
        info_print(f"[WARN] [UNLOAD READY] {stage_name}: cannot read real joints: {type(e).__name__}: {e}")
        return False, None, None, {"ok": False, "reason": "real_joint_read_failed"}

    q_dump = bucket_only_dump_pose(q_real, dump_deg)
    drop = log_unload_drop(f"{label}_bucket_only_landing", q=q_dump, reference_q=q_real)
    align = log_unload_alignment(f"{label}_bucket_only_pour", q=q_dump, effector="pour", reference_q=q_real)
    info_print(
        f"[UNLOAD READY] {stage_name}: label={label} "
        f"bucket_only_ok={drop.get('ok')} "
        f"inside_xy={drop.get('inside_xy')} close_xy={drop.get('close_xy')} above_wall={drop.get('above_wall')} "
        f"xy_err={fmt_optional(drop.get('xy_err'))} "
        f"source_clearance={fmt_optional(drop.get('source_clearance'))} "
        f"pour_inside={align.get('inside_xy')} pour_above={align.get('above_wall')} "
        f"real_q={q_deg_values(q_real, wrap_swing_for_display=True)} "
        f"dump_q={q_deg_values(q_dump, wrap_swing_for_display=True)}"
    )
    return bool(drop.get("ok", False) and drop.get("close_xy", False)), q_real, q_dump, drop


async def wait_for_dump_settle(stage_name, task_id=None):
    last_bin_count = None
    stable_samples = 0
    max_frames = int(UNLOAD_DUMP_SETTLE_MAX_FRAMES)
    min_frames = int(UNLOAD_DUMP_SETTLE_MIN_FRAMES)
    sample_frames = max(1, int(UNLOAD_DUMP_SETTLE_SAMPLE_FRAMES))

    for frame in range(max_frames):
        if task_id is not None and not task_alive(task_id):
            return False
        await step_updates(1)
        if (frame + 1) % sample_frames != 0 and (frame + 1) < max_frames:
            continue
        metrics = record_phase_metrics("after_dump_settle_probe")
        bin_count = int(metrics.get("bin_from_pile_count", 0)) if isinstance(metrics, dict) else 0
        if last_bin_count is not None and abs(bin_count - int(last_bin_count)) <= 1:
            stable_samples += 1
        else:
            stable_samples = 0
        last_bin_count = bin_count
        if (frame + 1) >= min_frames and stable_samples >= int(UNLOAD_DUMP_STABLE_SAMPLES):
            break

    record_phase_metrics("after_dump_settle")
    return True


async def dump_bucket_at_target(stage_name, task_id=None, planned_q_dump=None):
    dump_deg = unload_dump_target_deg()
    record_phase_metrics("before_dump")

    q_dump = None
    q_real = None
    ready = False
    dump_source = "dynamic_bucket_only"

    if planned_q_dump is not None:
        try:
            q_seed = CTRL.q_cmd.copy()
            q_dump = clip_command_near(np.array(planned_q_dump, dtype=np.float32).reshape(-1)[:4].copy(), reference=q_seed)
            try:
                q_real = current_real_q_near(q_seed)
            except Exception:
                q_real = q_seed.copy()
            actual_gate = log_actual_unload_position("before_planned_dump")
            drop = log_unload_drop("planned_cached_dump_landing", q=q_dump, reference_q=q_seed)
            log_unload_alignment("planned_cached_dump_pour", q=q_dump, effector="pour", reference_q=q_seed)
            ready = bool(actual_gate.get("ok", False) and drop.get("ok", False) and drop.get("close_xy", False))
            dump_source = "cached_planned_q_dump"
            info_print(
                f"[UNLOAD DUMP PLAN MATCH] {stage_name}: "
                f"ready={ready} source={dump_source} "
                f"q_dump={q_deg_values(q_dump, wrap_swing_for_display=True)} "
                f"drop_xy_err={fmt_optional(drop.get('xy_err'))} "
                f"inside_xy={drop.get('inside_xy')} close_xy={drop.get('close_xy')} "
                f"actual_gate={actual_gate.get('ok')}"
            )
            if not ready:
                update_status(
                    f"[UNLOAD DIAG] {stage_name}: cached planned dump predicted off target; executing and scoring actual particles; "
                    f"drop_xy_err={fmt_optional(drop.get('xy_err'))} inside_xy={drop.get('inside_xy')} "
                    f"close_xy={drop.get('close_xy')} actual_gate={actual_gate.get('ok')}",
                    force=True,
                )
                ready = True
        except Exception as e:
            info_print(f"[WARN] [UNLOAD DUMP PLAN MATCH] {stage_name}: cached q_dump failed {type(e).__name__}: {e}")
            ready = False
            q_dump = None

    if q_dump is None:
        actual_gate = log_actual_unload_position("before_dump")
        ready, q_real, q_dump, drop = bucket_only_dump_ready(stage_name, dump_deg, label="before_dump")
        if ready and not bool(actual_gate.get("ok", False)):
            info_print(
                f"[UNLOAD READY BLOCKED] {stage_name}: "
                "bucket_only landing is ok, but actual landing gate disagrees"
            )
            ready = False

    if not ready:
        q_seed = CTRL.q_cmd.copy()
        q_plan, dump_info = plan_dump_pose_to_bin(
            q_seed=q_seed,
            dump_deg=dump_deg,
            label=stage_name,
            log=True,
            allow_unaligned=True,
        )
        if q_plan is None:
            update_status(f"[UNLOAD BLOCKED] {stage_name}: dump pose failed: {dump_info}", force=True)
            set_execution_failure_reason(f"execution_failed/unload_dump_pose_failed:{stage_name}:{dump_info}")
            return False

        bucket_idx = CTRL.name_to_idx["bucket"]
        q_pre_dump = q_plan.copy()
        if q_real is not None:
            q_pre_dump[bucket_idx] = float(q_real[bucket_idx])
        else:
            q_pre_dump[bucket_idx] = float(q_seed[bucket_idx])
        q_pre_dump = clip_command_near(q_pre_dump, reference=q_seed)

        base_delta = non_bucket_delta_deg(q_pre_dump, q_seed)
        max_base_delta = max(base_delta) if base_delta else 0.0
        align_seconds = max(
            UNLOAD_PRE_DUMP_ALIGN_MIN_SECONDS,
            min(UNLOAD_PRE_DUMP_ALIGN_MAX_SECONDS, 0.25 + max_base_delta / 45.0),
        )
        info_print(
            f"[UNLOAD PRE-DUMP ALIGN] {stage_name}: "
            f"reason=bucket_only_drop_not_ready "
            f"non_bucket_delta={base_delta} "
            f"seconds={align_seconds:.2f} "
            f"q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}"
        )

        if max_base_delta > UNLOAD_PRE_DUMP_NON_BUCKET_TOL_DEG:
            try:
                trace_points = cache_active_stage_trace_points(
                    f"{stage_name}_pre_dump_align",
                    q_seed,
                    q_pre_dump,
                    stage_index=STATE.get("active_plan_stage_index", None),
                    include_remaining=False,
                )
                if current_trace_mode() == 2:
                    draw_trace(force=True)
                info_print(
                    "[PLAN EXEC TRACE]",
                    f"stage={stage_name}_pre_dump_align",
                    f"points={len(trace_points)}",
                    "source=active_command_segment",
                )
            except Exception as e:
                info_print("[WARN] [PLAN EXEC TRACE] pre-dump cache failed:", stage_name, type(e).__name__, e)

            success = await move_to_profile(
                q_pre_dump,
                seconds=align_seconds,
                label=f"{stage_name}_pre_dump_align",
                task_id=task_id,
                mode="unload_pre_dump_align",
            )
            if not success or (task_id is not None and not task_alive(task_id)):
                update_status(f"[UNLOAD BLOCKED] {stage_name}: pre-dump align failed", force=True)
                set_execution_failure_reason(f"execution_failed/unload_pre_dump_align_failed:{stage_name}")
                return False
            await step_updates(2)
            if task_id is not None and not task_alive(task_id):
                return False
            if not verify_unload_arrival(f"{stage_name}_pre_dump_align", q_pre_dump):
                return False
        else:
            info_print(f"[UNLOAD PRE-DUMP ALIGN] {stage_name}: already within non-bucket tolerance")

        actual_gate = log_actual_unload_position("after_pre_dump_align")
        if not bool(actual_gate.get("ok", False)):
            update_status(
                f"[UNLOAD DIAG] {stage_name}: predicted landing is not aligned after pre-dump align; executing and scoring actual particles",
                force=True,
            )

        ready, q_real, q_dump, drop = bucket_only_dump_ready(stage_name, dump_deg, label="after_pre_dump_align")
        if not ready:
            update_status(
                f"[UNLOAD DIAG] {stage_name}: bucket-only dump prediction is not aligned; executing and scoring actual particles; "
                f"xy_err={fmt_optional(drop.get('xy_err'))} inside_xy={drop.get('inside_xy')} "
                f"above_wall={drop.get('above_wall')}",
                force=True,
            )

    actual_gate = log_actual_unload_position("before_bucket_only_dump")
    if not bool(actual_gate.get("ok", False)):
        update_status(
            f"[UNLOAD DIAG] {stage_name}: predicted landing left unload target before dump; executing and scoring actual particles",
            force=True,
        )

    log_unload_alignment("planned_dump_pour", q=q_dump, effector="pour", reference_q=q_real)
    drop = log_unload_drop("planned_dump_landing", q=q_dump, reference_q=q_real)
    if not bool(drop.get("ok", False) and drop.get("close_xy", False)):
        update_status(
            f"[UNLOAD DIAG] {stage_name}: predicted bucket sand landing not aligned with unload target; executing and scoring actual particles; "
            f"xy_err={fmt_optional(drop.get('xy_err'))} close_xy={drop.get('close_xy')}",
            force=True,
        )

    info_print(
        f"[UNLOAD DUMP] {stage_name}: mode={dump_source} bucket_target={dump_deg:.1f}deg "
        f"q_dump={q_deg_values(q_dump, wrap_swing_for_display=True)}"
    )
    try:
        trace_points = cache_active_stage_trace_points(
            f"{stage_name}_dump_pose",
            CTRL.q_cmd.copy(),
            q_dump,
            stage_index=STATE.get("active_plan_stage_index", None),
            include_remaining=False,
        )
        if current_trace_mode() == 2:
            draw_trace(force=True)
        info_print(
            "[PLAN EXEC TRACE]",
            f"stage={stage_name}_dump_pose",
            f"points={len(trace_points)}",
            "source=active_command_segment",
        )
    except Exception as e:
        info_print("[WARN] [PLAN EXEC TRACE] dump cache failed:", stage_name, type(e).__name__, e)

    success = await move_to_profile(
        q_dump,
        seconds=UNLOAD_DUMP_SECONDS,
        label=f"{stage_name}_dump_pose",
        task_id=task_id,
        mode="unload_dump",
    )
    if not success or (task_id is not None and not task_alive(task_id)):
        update_status(f"[UNLOAD BLOCKED] {stage_name}: bucket dump failed", force=True)
        set_execution_failure_reason(f"execution_failed/unload_bucket_dump_failed:{stage_name}")
        return False

    record_phase_metrics("after_dump_motion")
    log_unload_alignment("after_dump_pour", effector="pour")
    log_unload_drop("after_dump_landing")
    if not await wait_for_dump_settle(stage_name, task_id=task_id):
        return False
    notify_sand_site_step_done(stage_name)
    return True


async def execute_unload_to_bin_from_current():
    STATE["follow"] = False
    item = plan_unload_from_current()
    if item is None:
        return

    stage_name, q_goal, duration = item
    task_id = start_task("dig_step_unload_to_bin")
    update_status("[DIG STEP 8] unload_to_bin", force=True)
    success = await ik_movement.move_unload_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
    if not success or not task_alive(task_id):
        update_status(execution_failure_status_text("unload_to_bin"), force=True)
        return
    if not await dump_bucket_at_target(stage_name, task_id=task_id):
        return
    log_phase_ground("[DIG GUARD AFTER]", stage_name)
    update_status("[DIG STEP DONE] unload_to_bin", force=True)


async def execute_dig_plan_step(step_index=None):
    STATE["follow"] = False
    seq = STATE.get("dig_plan_sequence", None)
    if seq is None:
        seq = await build_dig_plan_from_current_target_task(force_status=True)
    if not seq:
        return

    if step_index is None:
        step_index = int(STATE.get("dig_plan_step_index", 0))

    if step_index == 7 and step_index >= len(seq):
        await execute_unload_to_bin_from_current()
        return

    if step_index < 0 or step_index >= len(seq):
        update_status("[DIG STEP] no remaining step. Build/Reset plan to test again.", force=True)
        return

    stage_name, q_goal, duration = seq[step_index]
    STATE["active_plan_stage_index"] = int(step_index)
    task_id = start_task(f"dig_step_{stage_name}")

    log_phase_ground("[DIG GUARD CURRENT]", stage_name)
    ok, reason, report = log_predicted_phase_ground("[DIG GUARD TARGET]", stage_name, q_goal)
    if not ok:
        msg = format_ground_report(stage_name, report, reason)
        update_status(
            f"[DIG PREDICTED CONTACT] {stage_name} target: {msg}; executing with live freeze guard",
            force=True,
        )

    update_status(f"[DIG STEP {step_index + 1}/{len(seq)}] {stage_name}", force=True)
    if "unload" in stage_name:
        unload_detail = planned_unload_stage_detail(stage_index=step_index, stage_name=stage_name)
        planned_q_dump = None if not unload_detail else unload_detail.get("q_dump")
        success = await ik_movement.move_unload_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
        if not success or not task_alive(task_id):
            update_status(execution_failure_status_text(stage_name), force=True)
            return
        if int(STATE.get("dig_plan_step_index", 0)) <= step_index:
            STATE["dig_plan_step_index"] = step_index + 1
        if not await dump_bucket_at_target(stage_name, task_id=task_id, planned_q_dump=planned_q_dump):
            return
        log_phase_ground("[DIG GUARD AFTER]", stage_name)
        update_status(f"[DIG STEP DONE] {stage_name}", force=True)
        return

    success = await ik_movement.move_planned_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
    if not success or not task_alive(task_id):
        update_status(execution_failure_status_text(stage_name), force=True)
        return

    if int(STATE.get("dig_plan_step_index", 0)) <= step_index:
        STATE["dig_plan_step_index"] = step_index + 1

    if "unload" in stage_name:
        if not await dump_bucket_at_target(stage_name, task_id=task_id):
            return
    else:
        notify_sand_site_step_done(stage_name)
        if stage_name == "curl_to_hold_material":
            record_phase_metrics("after_dig")
        elif stage_name == "lift_carry":
            record_phase_metrics("after_lift")

    log_phase_ground("[DIG GUARD AFTER]", stage_name)
    update_status(f"[DIG STEP DONE] {stage_name}", force=True)


async def execute_dig_target_ball(rebuild_plan=True, task_name="dig_target_ball"):
    STATE["follow"] = False
    if rebuild_plan:
        seq = await build_dig_plan_from_current_target_task(force_status=True)
    else:
        seq = STATE.get("dig_plan_sequence", None)
        if seq is None:
            seq = await build_dig_plan_from_current_target_task(force_status=True)
    if not seq:
        return False

    task_id = start_task(task_name)

    for stage_index, (stage_name, q_goal, duration) in enumerate(seq):
        STATE["active_plan_stage_index"] = int(stage_index)
        if not task_alive(task_id):
            return False

        log_phase_ground("[DIG GUARD CURRENT]", stage_name)
        ok, reason, report = log_predicted_phase_ground("[DIG GUARD TARGET]", stage_name, q_goal)
        if not ok:
            msg = format_ground_report(stage_name, report, reason)
            update_status(
                f"[DIG PREDICTED CONTACT] {stage_name} target: {msg}; executing with live freeze guard",
                force=True,
            )

        update_status(f"[DIG] {stage_name}", force=True)
        if "unload" in stage_name:
            unload_detail = planned_unload_stage_detail(stage_index=stage_index, stage_name=stage_name)
            planned_q_dump = None if not unload_detail else unload_detail.get("q_dump")
            success = await ik_movement.move_unload_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
            if not success or not task_alive(task_id):
                update_status(execution_failure_status_text(stage_name), force=True)
                return False
            STATE["dig_plan_step_index"] = int(STATE.get("dig_plan_step_index", 0)) + 1
            if not await dump_bucket_at_target(stage_name, task_id=task_id, planned_q_dump=planned_q_dump):
                return False
            log_phase_ground("[DIG GUARD AFTER]", stage_name)
            continue

        success = await ik_movement.move_planned_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
        if not success or not task_alive(task_id):
            update_status(execution_failure_status_text(stage_name), force=True)
            return False
        STATE["dig_plan_step_index"] = int(STATE.get("dig_plan_step_index", 0)) + 1
        if "unload" in stage_name:
            if not await dump_bucket_at_target(stage_name, task_id=task_id):
                return False
        else:
            notify_sand_site_step_done(stage_name)
            if stage_name == "curl_to_hold_material":
                record_phase_metrics("after_dig")
            elif stage_name == "lift_carry":
                record_phase_metrics("after_lift")
        log_phase_ground("[DIG GUARD AFTER]", stage_name)

    if task_alive(task_id):
        update_status("[DIG FINISHED] direct home pose set", force=True)
        was_recording = bool(STATE.get("dataset_recording", False))
        STATE["dataset_recording"] = False
        try:
            await set_joint_pose_direct_and_settle(
                safe_home_q(),
                label=f"{task_name}_home_after_plan",
                mode="plan_end_home_direct",
                settle_frames=DIRECT_PLAN_END_HOME_SETTLE_FRAMES,
                task_id=task_id,
            )
        finally:
            STATE["dataset_recording"] = was_recording
    update_status("[DIG FINISHED] bucket curled, lifted, unloaded, and homed.", force=True)
    return True


async def calibrate_ik():
    global IK_MODEL

    update_status("Calibrating planar IK...", force=True)

    update_q_cmd_from_real()
    q0 = CTRL.q_cmd.copy()
    base_chain = current_planar_chain()
    if base_chain is None:
        update_status("[IK CALIBRATE FAILED] cannot read current chain.", force=True)
        return

    base_angles = base_chain["angles"]
    h = IK_CALIBRATION_STEP
    signs = []

    probes = [
        ("boom", 0),
        ("arm", 1),
        ("bucket", 2),
    ]

    for joint_name, angle_idx in probes:
        idx = CTRL.name_to_idx[joint_name]
        q_probe = q0.copy()
        q_probe[idx] += h

        await move_to(q_probe, 0.55, mode="calibrate")
        await step_updates(24)

        chain = current_planar_chain()
        if chain is None:
            signs.append(1.0)
            info_print("[WARN] IK sign probe failed:", joint_name)
        else:
            angles = chain["angles"]
            if angle_idx == 0:
                delta = wrap_angle(angles[0] - base_angles[0])
            elif angle_idx == 1:
                delta = wrap_angle((angles[1] - angles[0]) - (base_angles[1] - base_angles[0]))
            else:
                delta = wrap_angle((angles[2] - angles[1]) - (base_angles[2] - base_angles[1]))

            sign = 1.0 if delta >= 0.0 else -1.0
            signs.append(sign)
            info_print(f"[IK CAL] {joint_name}: delta={delta:.6f} rad for +{h:.3f}, sign={sign:+.0f}")

        await move_to(q0, 0.55, mode="calibrate")
        await step_updates(24)

    await move_to(q0, 0.65, mode="calibrate")
    await step_updates(30)
    update_q_cmd_from_real()
    q_rest = CTRL.q_cmd.copy()
    chain0 = current_planar_chain("mid")
    if chain0 is None:
        update_status("[IK CALIBRATE FAILED] cannot restore chain.", force=True)
        return

    signs = np.array(signs, dtype=np.float32)
    effectors = {}
    for end_effector in ["mid", "tip", "load", "pour"]:
        c = current_planar_chain(end_effector)
        if c is None:
            info_print("[WARN] IK effector calibration failed:", end_effector)
            continue
        effectors[end_effector] = {
            "offsets": build_ik_offsets(q_rest, c, signs),
            "lengths": np.array(c["lengths"], dtype=np.float32),
        }

    if "mid" not in effectors or "tip" not in effectors:
        update_status("[IK CALIBRATE FAILED] missing mid/tip effector geometry.", force=True)
        return

    IK_MODEL = {
        "signs": signs,
        "effectors": effectors,
        "calibrated": True,
    }

    CTRL.q_cmd = q_rest.copy()
    CTRL.send_action(q_rest, mode="calibrate_restore")

    info_print("[IK CAL] signs:", [float(x) for x in IK_MODEL["signs"]])
    for end_effector, part in IK_MODEL["effectors"].items():
        info_print(f"[IK CAL] {end_effector}.lengths:", [float(x) for x in part["lengths"]])
        info_print(f"[IK CAL] {end_effector}.offsets:", [float(x) for x in part["offsets"]])
    update_status("Planar IK calibrated.", force=True)


def follow_step():
    now = time.time()
    if now - STATE.get("last_follow_time", 0.0) < STATE.get("follow_interval", 1.0 / 60.0):
        return

    p_bucket = bucket_mid_pos()
    p_target = get_target_pos()
    p_target[2] = max(float(p_target[2]), TARGET_MIN_Z)

    err = p_target - p_bucket
    dist = float(np.linalg.norm(err))

    if dist < 0.05:
        set_target_color(TARGET_COLOR_REACHABLE)
        update_status(f"Follow reached. error={dist:.3f} m", force=False)
        STATE["last_follow_time"] = now
        return

    try:
        q_goal, info = solve_priority_ik_to_target(
            p_target,
            min_world_z=GROUND_TOP_Z + IK_FOLLOW_MIN_CLEARANCE,
            accept_err=IK_ACCEPT_ERR,
            end_effector="mid",
            allow_end_below=False,
            min_end_z=GROUND_TOP_Z + IK_FOLLOW_MIN_CLEARANCE,
            phase_mode="follow",
        )
    except Exception as e:
        set_target_color(TARGET_COLOR_UNREACHABLE)
        update_status("IK solve failed: " + str(e), force=False)
        STATE["last_follow_time"] = now
        return

    if q_goal is None:
        set_target_color(TARGET_COLOR_UNREACHABLE)
        update_status("IK solve failed: " + str(info), force=False)
        STATE["last_follow_time"] = now
        return

    speed_multiplier = float(STATE.get("speed_multiplier", 1.0))
    dq = q_goal - CTRL.q_cmd
    dq[CTRL.name_to_idx["swing"]] = swing_delta(
        q_goal[CTRL.name_to_idx["swing"]],
        CTRL.q_cmd[CTRL.name_to_idx["swing"]],
    )
    max_dq = IK_MAX_DQ * speed_multiplier
    dq = np.clip(dq, -max_dq, max_dq)

    q_goal = CTRL.q_cmd + dq.astype(np.float32)
    q_goal[CTRL.name_to_idx["swing"]] = normalize_swing_cmd(q_goal[CTRL.name_to_idx["swing"]])
    STATE["trace_active_motion"] = {
        "label": "follow",
        "mode": "follow",
        "q_goal": q_goal.copy(),
        "expires_at": now + 0.5,
    }
    if current_trace_mode() == 2 and not STATE.get("dig_plan_sequence"):
        draw_trace(force=True)
    ok, reason = CTRL.apply_target(q_goal, mode="follow")

    if ok:
        set_target_color(TARGET_COLOR_REACHABLE)
        update_status(
            f"IK Follow error={dist:.3f} m planar={info['planar_err']:.3f} "
            f"body_min_z={info['min_z']:.3f} end_z={info['end_z']:.3f} "
            f"end={info['end_effector']} reachable={info['reachable']}",
            force=False,
        )
    else:
        set_target_color(TARGET_COLOR_UNREACHABLE)
        update_status(f"IK Follow blocked: {reason}, error={dist:.3f} m", force=False)

    STATE["last_follow_time"] = now


# ============================================================
# UI
# ============================================================

def sync_sliders_from_real_q(force=False):
    if not SLIDER_MODELS:
        return

    if (not force) and manual_ui_sync_is_held():
        return
    now = time.time()
    if (
        not force
        and now - float(STATE.get("last_slider_sync_time", 0.0) or 0.0) < float(STATE.get("slider_sync_interval", 0.10))
    ):
        return

    q_real = get_real_joint_positions()

    STATE["ui_syncing"] = True
    try:
        for name, idx in CTRL.name_to_idx.items():
            if name in SLIDER_MODELS:
                value_deg = rad_to_deg(float(q_real[idx]))
                if name == "swing":
                    value_deg = rad_to_deg(wrap_angle(float(q_real[idx])))
                SLIDER_MODELS[name].set_value(value_deg)
        STATE["last_slider_sync_time"] = now
    finally:
        STATE["ui_syncing"] = False


def build_ui():
    global WINDOW, STATUS_LABEL
    auto_count_model = ui.SimpleIntModel(int(AUTO_COLLECT_DEFAULT_COUNT))

    def toggle_follow():
        if not STATE["follow"]:
            start_task("follow")
            STATE["follow"] = True
        else:
            cancel_active_task("follow off")
            STATE["follow"] = False
        update_status(f"Follow = {STATE['follow']}", force=True)

    def trace_off():
        set_trace_mode(0)

    def trace_mode_1():
        set_trace_mode(1)

    def trace_mode_2():
        set_trace_mode(2, reset_real=True)

    def request_calib():
        STATE["request_calibrate"] = True
        update_status("IK calibration requested", force=True)

    def ik_one_step():
        STATE["follow"] = False
        start_task("ik_one_step")
        STATE["last_follow_time"] = 0.0
        follow_step()

    def home():
        cancel_active_task("home pressed")
        q_home = safe_home_q()
        set_manual_joint_target(q_home, reason="safe_home")
        update_status(
            f"Safe home target queued: boom={HOME_SAFE_DEG['boom']:.1f} arm={HOME_SAFE_DEG['arm']:.1f} bucket={HOME_SAFE_DEG['bucket']:.1f}",
            force=True,
        )

    def print_state():
        CTRL.print_state()
        update_status("State printed", force=True)

    def toggle_render_mode_from_ui():
        toggle_excavator_render_mode()

    def set_log_normal_from_ui():
        set_log_mode("normal")

    def toggle_debug_from_ui():
        toggle_debug_log()

    def toggle_quiet_from_ui():
        toggle_quiet_log()

    def print_log_state_from_ui():
        print_log_state()
        update_status(f"[LOG MODE] {current_log_mode()} state printed", force=True)

    def auto_collect_count_from_ui():
        try:
            count = int(auto_count_model.as_int)
        except Exception:
            try:
                count = int(auto_count_model.get_value_as_int())
            except Exception:
                count = int(round(safe_float(getattr(auto_count_model, "as_float", AUTO_COLLECT_DEFAULT_COUNT), AUTO_COLLECT_DEFAULT_COUNT)))
        count = max(1, min(10000, count))
        try:
            auto_count_model.set_value(int(count))
        except Exception:
            pass
        return count

    def start_auto_collect_from_ui():
        count = auto_collect_count_from_ui()
        update_status(f"[AUTO DATASET] start requested count={count}", force=True)
        request_auto_collect(count)

    def open_auto_collect_dir_from_ui():
        path = str(STATE.get("auto_collect_run_dir", "") or AUTO_COLLECT_DATASET_ROOT)
        try:
            os.makedirs(path, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(path)
                update_status(f"[AUTO DATASET] opened dir: {path}", force=True)
            else:
                update_status(f"[AUTO DATASET] dir: {path}", force=True)
        except Exception as e:
            info_print("[WARN] [AUTO DATASET] open dir failed:", type(e).__name__, e)
            update_status(f"[AUTO DATASET] open dir failed: {type(e).__name__}", force=True)

    def stop_all():
        if STATE.get("auto_collect_active", False):
            stop_auto_collect()
        STATE["follow"] = False
        STATE["manual_joint_active"] = False
        STATE["manual_joint_target"] = None
        STATE["trace_active_motion"] = None
        STATE["trace_no_plan_notice_shown"] = True
        STATE["trace_no_plan_notice_time"] = time.time()
        cancel_registered_task("planner", reason="stop_motion")
        cancel_registered_task("motion", reason="stop_motion")
        cancel_registered_task("replay", reason="stop_motion")
        update_status("Motion stopped", force=True)

    def stop_loop():
        if STATE.get("auto_collect_active", False):
            stop_auto_collect()
        STATE["running"] = False
        STATE["follow"] = False
        STATE["manual_joint_active"] = False
        STATE["manual_joint_target"] = None
        STATE["trace_active_motion"] = None
        STATE["trace_no_plan_notice_shown"] = True
        STATE["trace_no_plan_notice_time"] = time.time()
        STATE["robot_state_reads_enabled"] = False
        STATE["robot_state_shutdown"] = True
        cancel_registered_tasks(reason="stop_ui_loop", keep={"main_loop"})
        update_status("Loop stopping", force=True)

    def build_dig_plan_button():
        register_async_task("planner", build_dig_plan_from_current_target_task(force_status=True), replace=True)

    def reset_dig_plan_button():
        reset_dig_plan()

    def run_next_dig_step_button():
        register_async_task("motion", execute_dig_plan_step(), replace=True)

    def run_dig_step_button(step_index):
        register_async_task("motion", execute_dig_plan_step(step_index), replace=True)

    def use_selected_unload_mesh_from_ui():
        set_manual_unload_from_selected_mesh()

    def use_selected_unload_point_from_ui():
        set_manual_unload_from_selected_point()

    def clear_unload_override_from_ui():
        clear_manual_unload_override()

    WINDOW = ui.Window("Excavator Slider Control v3", width=620, height=760)

    with WINDOW.frame:
        with ui.VStack(spacing=8):
            ui.Label("Excavator Slider Control v3")
            STATUS_LABEL = ui.Label("Ready", width=590)

            with ui.ScrollingFrame(height=690):
                with ui.VStack(spacing=8):
                    with ui.HStack(spacing=6):
                        ui.Label("Log", width=80)
                        ui.Button("Normal", width=82, clicked_fn=set_log_normal_from_ui)
                        ui.Button("Debug ON/OFF", width=118, clicked_fn=toggle_debug_from_ui)
                        ui.Button("Quiet ON/OFF", width=118, clicked_fn=toggle_quiet_from_ui)
                        ui.Button("State", width=70, clicked_fn=print_log_state_from_ui)

                    ui.Separator()
                    ui.Label("Joint targets, degrees")

                    def on_manual_joint_slider_changed(model=None):
                        if STATE.get("ui_syncing", False):
                            return

                        hold_manual_ui_sync()
                        cancel_active_task("manual joint slider changed")

                        q = np.array([
                            deg_to_rad(SLIDER_MODELS["swing"].as_float),
                            deg_to_rad(SLIDER_MODELS["boom"].as_float),
                            deg_to_rad(SLIDER_MODELS["arm"].as_float),
                            deg_to_rad(SLIDER_MODELS["bucket"].as_float),
                        ], dtype=np.float32)

                        set_manual_joint_target(q, reason="slider")

                    for name in ["swing", "boom", "arm", "bucket"]:
                        lo, hi = FINAL_LIMITS_DEG[name]
                        model = ui.SimpleFloatModel(0.0)
                        SLIDER_MODELS[name] = model

                        with ui.HStack(spacing=6):
                            ui.Label(name, width=80)
                            ui.FloatSlider(model=model, min=lo, max=hi, width=360)
                            ui.FloatField(model=model, width=90)

                        model.add_value_changed_fn(on_manual_joint_slider_changed)

                    with ui.HStack(spacing=6):
                        ui.Label("Manual joints", width=170)
                        ui.Button("Home", width=82, clicked_fn=home)
                        ui.Button("Print State", width=104, clicked_fn=print_state)
                        ui.Button("Render Mode", width=112, clicked_fn=toggle_render_mode_from_ui)

                    ui.Separator()
                    ui.Label("Auto Dataset (primary pipeline)")
                    with ui.HStack(spacing=6):
                        ui.Label("Target trainable", width=135)
                        ui.Label("Count", width=50)
                        ui.IntField(model=auto_count_model, width=70)
                        ui.Button("Start", width=62, clicked_fn=start_auto_collect_from_ui)
                        ui.Button("Stop", width=62, clicked_fn=stop_auto_collect)
                        ui.Button("Dir", width=52, clicked_fn=open_auto_collect_dir_from_ui)
                        ui.Button("Replay", width=74, clicked_fn=request_replay_latest_record)

                    with ui.HStack(spacing=6):
                        speed_model = ui.SimpleFloatModel(float(STATE.get("speed_multiplier", 1.0)))
                        ui.Label("Speed x", width=80)
                        ui.FloatSlider(model=speed_model, min=SPEED_MULTIPLIER_MIN, max=SPEED_MULTIPLIER_MAX, width=360)
                        ui.FloatField(model=speed_model, width=90)

                        def update_speed_multiplier(model=None):
                            sm = safe_float(speed_model.as_float, 1.0)
                            sm = max(SPEED_MULTIPLIER_MIN, min(SPEED_MULTIPLIER_MAX, sm))
                            STATE["speed_multiplier"] = sm
                            apply_speed_to_physx_joint_limits()
                            update_status(f"Speed multiplier = {sm:.2f}", force=True)

                        speed_model.add_value_changed_fn(update_speed_multiplier)

                    ui.Separator()
                    with ui.HStack(spacing=10):
                        with ui.VStack(width=292, spacing=4):
                            ui.Label("Target ball position")
                            p = get_target_pos()
                            for axis, val in [("x", p[0]), ("y", p[1]), ("z", p[2])]:
                                model = ui.SimpleFloatModel(float(val))
                                TARGET_MODELS[axis] = model
                                with ui.HStack(spacing=4):
                                    ui.Label(axis.upper(), width=20)
                                    ui.FloatSlider(model=model, min=-15.0, max=15.0, width=170)
                                    ui.FloatField(model=model, width=70)

                        with ui.VStack(width=292, spacing=4):
                            ui.Label("Unload point / Z range")
                            unload_p = np.array(STATE.get("manual_unload_point") if STATE.get("manual_unload_point") is not None else unload_bin_landing_point(), dtype=np.float32).reshape(-1)[:3]
                            unload_z_range = STATE.get("manual_unload_z_range")
                            if unload_z_range is not None:
                                try:
                                    unload_z_range = np.array(unload_z_range, dtype=np.float32).reshape(-1)[:2]
                                except Exception:
                                    unload_z_range = None
                            if unload_z_range is None or len(unload_z_range) < 2:
                                unload_z_range = np.array([float(GROUND_TOP_Z), float(unload_p[2])], dtype=np.float32)
                            unload_z_min = min(float(unload_z_range[0]), float(unload_z_range[1]))
                            unload_z_max = max(float(unload_z_range[0]), float(unload_z_range[1]))
                            for axis, val in [("x", unload_p[0]), ("y", unload_p[1])]:
                                model = ui.SimpleFloatModel(float(val))
                                UNLOAD_MODELS[axis] = model
                                with ui.HStack(spacing=4):
                                    ui.Label(axis.upper(), width=20)
                                    ui.FloatSlider(model=model, min=-15.0, max=15.0, width=170)
                                    ui.FloatField(model=model, width=70)
                            for key, label, val in [("z_min", "Z Min", unload_z_min), ("z_max", "Z Max", unload_z_max)]:
                                model = ui.SimpleFloatModel(float(val))
                                UNLOAD_MODELS[key] = model
                                with ui.HStack(spacing=4):
                                    ui.Label(label, width=48)
                                    ui.FloatSlider(model=model, min=-2.0, max=8.0, width=142)
                                    ui.FloatField(model=model, width=70)
                            radius_model = ui.SimpleFloatModel(float(manual_unload_radius()))
                            UNLOAD_MODELS["r"] = radius_model
                            with ui.HStack(spacing=4):
                                ui.Label("R", width=20)
                                ui.FloatSlider(model=radius_model, min=0.05, max=4.0, width=170)
                                ui.FloatField(model=radius_model, width=70)
                            shrink_model = ui.SimpleFloatModel(float(manual_unload_mesh_shrink_d()))
                            UNLOAD_MODELS["d"] = shrink_model
                            with ui.HStack(spacing=4):
                                ui.Label("D", width=20)
                                ui.FloatSlider(model=shrink_model, min=0.0, max=2.0, width=170)
                                ui.FloatField(model=shrink_model, width=70)
                            STATE["last_unload_model_xyz"] = unload_p.copy()
                            STATE["last_unload_model_z_range"] = np.array([unload_z_min, unload_z_max], dtype=np.float32)
                            STATE["last_unload_model_radius"] = float(manual_unload_radius())
                            STATE["last_unload_model_shrink_d"] = float(manual_unload_mesh_shrink_d())
                            STATE["last_unload_sync_time"] = time.time()

                    ui.Label("Z Min/Max sets the blue unload range height; Z Max is the marker/release height.", width=590)
                    ui.Label("R applies to point/XYZ mode; D shrinks selected mesh XY footprint. Auto Dataset uses this unload target.", width=590)
                    ui.Label("Unload point setup")
                    with ui.HStack(spacing=6):
                        ui.Button("Unload Selected Mesh", width=166, clicked_fn=use_selected_unload_mesh_from_ui)
                        ui.Button("Unload Selected Point", width=166, clicked_fn=use_selected_unload_point_from_ui)
                        ui.Button("Clear Unload Override", width=160, clicked_fn=clear_unload_override_from_ui)

                    ui.Separator()
                    ui.Label("Debug Planner / Trace")

                    with ui.HStack(spacing=6):
                        ui.Button("Calibrate IK", width=112, clicked_fn=request_calib)
                        ui.Button("IK One Step", width=104, clicked_fn=ik_one_step)
                        ui.Button("Follow ON/OFF", width=118, clicked_fn=toggle_follow)
                    with ui.HStack(spacing=6):
                        ui.Button("Trace target", width=116, clicked_fn=trace_mode_1)
                        ui.Button("Trace path", width=108, clicked_fn=trace_mode_2)

                    with ui.HStack(spacing=6):
                        ui.Button(
                            "Build Dig Plan",
                            width=132,
                            clicked_fn=build_dig_plan_button,
                        )
                        ui.Button(
                            "Run Next Step",
                            width=124,
                            clicked_fn=run_next_dig_step_button,
                        )
                        ui.Button(
                            "Run All",
                            width=82,
                            clicked_fn=lambda: register_async_task("motion", execute_dig_target_ball(rebuild_plan=False), replace=True),
                        )
                        ui.Button("Reset Plan", width=100, clicked_fn=reset_dig_plan_button)

                    with ui.HStack(spacing=6):
                        ui.Button("1 Pre", width=82, clicked_fn=lambda: run_dig_step_button(0))
                        ui.Button("2 Contact", width=102, clicked_fn=lambda: run_dig_step_button(1))
                        ui.Button("3 Insert", width=92, clicked_fn=lambda: run_dig_step_button(2))
                        ui.Button("4 Mid Cut", width=102, clicked_fn=lambda: run_dig_step_button(3))

                    with ui.HStack(spacing=6):
                        ui.Button("5 Exit Cut", width=108, clicked_fn=lambda: run_dig_step_button(4))
                        ui.Button("6 Curl", width=82, clicked_fn=lambda: run_dig_step_button(5))
                        ui.Button("7 Lift", width=82, clicked_fn=lambda: run_dig_step_button(6))
                        ui.Button("8 Unload", width=98, clicked_fn=lambda: run_dig_step_button(7))

                    with ui.HStack(spacing=6):
                        ui.Button("Stop Motion", width=116, clicked_fn=stop_all)
                        ui.Button("Stop UI Loop", width=116, clicked_fn=stop_loop)

                    ui.Label("Debug only: these controls execute the same cached DigPlan used by Auto Dataset.", width=590)

    builtins._EXCAVATOR_UI_WINDOW = WINDOW
    WINDOW.visible = True
    try:
        WINDOW.focus()
    except Exception:
        pass
    info_print(
        "[UI THREAD]",
        "Excavator Slider Control v3 runs on the Isaac Kit main asyncio loop;",
        "button callbacks only enqueue named tasks;",
        "USD/PhysX work is not moved to unsafe worker threads.",
    )

# ============================================================
# Main
# ============================================================

async def main():
    STATE["robot_state_reads_enabled"] = False
    STATE["robot_state_shutdown"] = False
    disable_usd_audio_extension()
    enable_physx_gpu_runtime_settings()
    old_world = World.instance()
    if old_world is not None:
        try:
            await old_world.stop_async()
        except Exception:
            pass
        World.clear_instance()

    setup_paths()
    apply_excavator_render_mode(True, force_status=False)
    configure_joints()
    build_scene()
    if sand_site_active():
        ensure_particle_gpu_physics_scene("before_world")
        ensure_sand_site_bucket_colliders(force=True)
        rebind_sand_particles_to_physics_scene("before_world")
    print_ground_contact_diagnostics("before_world")

    world = World(stage_units_in_meters=1.0)
    if sand_site_active():
        ensure_particle_gpu_physics_scene("after_world_construct")
    await world.initialize_simulation_context_async()
    if sand_site_active():
        configure_world_gpu_physics(world, "after_world_init_before_reset")
        ensure_particle_gpu_physics_scene("after_world_init_before_reset")
        rebind_sand_particles_to_physics_scene("after_world_init_before_reset")

    global ROBOT, DOF_NAME_TO_REAL_IDX, JOINT_INDICES
    ROBOT = SingleArticulation(
        prim_path=ROBOT_ROOT,
        name="excavator_articulation"
    )
    world.scene.add(ROBOT)

    await world.reset_async()
    park_robot_on_support_ground("after_world_reset")
    if sand_site_active():
        configure_world_gpu_physics(world, "after_world_reset")
        ensure_particle_gpu_physics_scene("after_world_reset")
        ensure_sand_site_bucket_colliders(force=True)
    print_ground_contact_diagnostics("after_world_reset")
    await world.play_async()
    if sand_site_active():
        await step_updates(2)
        rebind_sand_particles_to_physics_scene("after_world_play_start")
    await step_updates(ROBOT_DROP_SETTLE_FRAMES)
    settle_robot_on_ground_if_needed("after_world_play")
    await step_updates(12)
    print_ground_contact_diagnostics("after_world_play")
    if sand_site_active() and AUTO_RESET_SAND_AFTER_WORLD_READY:
        await reset_sand_site_stably("after_world_ready")

    try:
        ROBOT.initialize()
    except Exception as e:
        info_print("[WARN] ROBOT.initialize failed:", e)

    try:
        info_print("ROBOT dof_names:", ROBOT.dof_names)
        DOF_NAME_TO_REAL_IDX = {}
        for i, dn in enumerate(ROBOT.dof_names):
            lname = dn.lower()
            for expected in DOF_ORDER:
                if expected in lname:
                    DOF_NAME_TO_REAL_IDX[expected] = i
        info_print("DOF_NAME_TO_REAL_IDX:", DOF_NAME_TO_REAL_IDX)

        missing = [name for name in DOF_ORDER if name not in DOF_NAME_TO_REAL_IDX]
        if missing:
            info_print("[WARN] Missing named DOFs:", missing)
            if len(ROBOT.dof_names) == len(DOF_ORDER):
                JOINT_INDICES = np.arange(len(DOF_ORDER), dtype=np.int32)
                info_print("JOINT_INDICES fallback:", JOINT_INDICES)
            else:
                JOINT_INDICES = None
                info_print("[ERROR] Cannot build JOINT_INDICES safely for DOFs:", ROBOT.dof_names)
        else:
            JOINT_INDICES = np.array(
                [DOF_NAME_TO_REAL_IDX[name] for name in DOF_ORDER],
                dtype=np.int32,
            )
            info_print("JOINT_INDICES:", JOINT_INDICES)
    except Exception as e:
        info_print("[WARN] Cannot read ROBOT dof_names:", e)

    STATE["last_valid_real_q"] = CTRL.q_cmd.copy()
    STATE["robot_state_reads_enabled"] = JOINT_INDICES is not None
    STATE["robot_state_shutdown"] = False
    if not STATE["robot_state_reads_enabled"]:
        info_print("[WARN] robot joint reads disabled: JOINT_INDICES not ready")

    # 关键：这里必须创建 UI
    update_q_cmd_from_real()
    CTRL.q_safe = CTRL.q_cmd.copy()
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["freeze_candidate_since"] = 0.0
    STATE["last_action_q"] = None
    STATE["last_action_unclipped_q"] = None
    STATE["last_action_time"] = 0.0
    build_ui()

    # 初始化 UI 上的 joint 显示为真实关节值
    try:
        sync_sliders_from_real_q()
    except Exception as e:
        info_print("[WARN] initial slider sync failed:", e)

    try:
        builtins._EXCAVATOR_UI_WINDOW = WINDOW
        WINDOW.visible = True
        WINDOW.focus()
    except Exception:
        pass

    update_status("Ready. Real joint state synced. UI opened.", force=True)
    print_motion_constraint_diagnostics("ready")
    if sand_site_active() and AUTO_RESET_SAND_AFTER_UI_READY:
        register_async_task("startup_sand_reset", delayed_startup_sand_reset(), replace=True)

    while STATE["running"]:
        if handle_timeline_stop_if_needed("main_loop"):
            await step_updates(max(1, int(60 / CONTROL_HZ)))
            continue

        if STATE["request_calibrate"]:
            STATE["request_calibrate"] = False
            await calibrate_ik()

        sync_target_from_sliders_live(force=False)
        sync_unload_from_sliders_live(force=False)

        apply_manual_joint_target_step()

        # 实时同步真实 joint 值到 UI
        sync_sliders_from_real_q()

        if STATE["follow"]:
            follow_step()

        if current_trace_mode() != 0:
            draw_trace(force=False)

        check_freeze_state("main_loop")

        await step_updates(max(1, int(60 / CONTROL_HZ)))

    STATE["follow"] = False
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    STATE["robot_state_reads_enabled"] = False
    STATE["robot_state_shutdown"] = True
    cancel_registered_tasks(reason="main_loop_exit", keep={"main_loop"})
    update_status("Slider UI exited.", force=True)

register_async_task("main_loop", main(), replace=True)
