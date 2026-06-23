import asyncio
import hashlib
import importlib
import importlib.util
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


def _load_optional_joint_space_planner():
    errors = []
    candidates = []
    package_name = str(__package__ or "")
    if package_name:
        candidates.append(f"{package_name}.joint_space_planner")
    candidates.append("excavator_app.joint_space_planner")

    for module_name in dict.fromkeys(candidates):
        try:
            return importlib.import_module(module_name), ""
        except Exception as exc:
            errors.append(f"{module_name}:{type(exc).__name__}:{exc}")

    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_space_planner.py")
    if os.path.isfile(module_path):
        try:
            spec = importlib.util.spec_from_file_location("excavator_app._joint_space_planner_runtime", module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module, ""
            errors.append("file_spec_unavailable")
        except Exception as exc:
            errors.append(f"file:{type(exc).__name__}:{exc}")
    else:
        errors.append(f"missing_file:{module_path}")

    return None, "; ".join(errors)


joint_space_planner, JOINT_SPACE_PLANNER_IMPORT_ERROR = _load_optional_joint_space_planner()
if joint_space_planner is None:
    print(
        "[INFO] [WARN] joint_space_planner import failed; fallback clearance routes will be used:",
        JOINT_SPACE_PLANNER_IMPORT_ERROR,
    )


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
    "planning_cancel_requested": False,
    "planning_path_penalty_cache": {},
    "planning_path_penalty_cache_hits": 0,
    "planning_path_penalty_cache_misses": 0,
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
    "dataset_last_action": None,
    "dataset_last_dq_real": None,
    "dataset_last_ddq_real": None,
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
    "dataset_initial_pile_particle_mask": None,
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
    "sand_snapshot_last_time": 0.0,
    "sand_snapshot_last": None,
    "auto_collect_episode_sand_snapshot": None,
    "auto_collect_episode_sand_snapshot_time": 0.0,
    "planning_sand_snapshot": None,
    "planning_sand_snapshot_active": False,
    "sand_perf_last": {},
    "perf_block_count": 0,
    "perf_block_last": None,
    "rigid_obstacle_cache_time": 0.0,
    "rigid_obstacle_cache": None,
    "rigid_obstacle_cache_hits": 0,
    "rigid_obstacle_cache_misses": 0,
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
MANUAL_SPEED_MULTIPLIER_CAP = 10.0
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
SAND_CONTACT_PHASES = {"insert_cut", "pull_mid_cut", "pull_exit_cut", "curl_to_hold_material", "secure_load"}
SAND_CUT_GEOMETRY_PHASES = {"insert_cut", "pull_mid_cut", "pull_exit_cut"}
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
SAND_CONTACT_SPILL_RATIO_MAX = 0.55
SAND_CONTACT_SPILL_WITHOUT_LOAD_MIN = 140
SAND_CONTACT_EXIT_LOADED_BUCKET_MIN = 300
SAND_CONTACT_EXIT_SPILL_RATIO_MAX = 0.80
CURL_HOLD_MIN_BUCKET_PARTICLES = 80
CURL_HOLD_ACCEPT_BUCKET_DEG = -110.0
CURL_HOLD_ACCEPT_MAX_ERR_DEG = 14.0
CURL_HOLD_TARGET_DEG = -120.0
BUCKET_LOADED_CLOSED_LIMIT_DEG = CURL_HOLD_TARGET_DEG
SECURE_HOLD_MAX_SPILL_PARTICLES = 240
SECURE_HOLD_MAX_SPILL_FRACTION = 0.35
SECURE_HOLD_MAX_BUCKET_LOSS_FRACTION = 0.45
SECURE_HOLD_MIN_RETAINED_FROM_CUT_FRACTION = 0.55
LIFT_CARRY_MIN_RETAINED_FROM_CUT_FRACTION = 0.55
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
AUTO_COLLECT_GLOBAL_PLAN_FAILURE_LIMIT = 4
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
AUTO_COLLECT_SCHEMA = "excavator_auto_state_action_v3"
DATASET_TRAJECTORY_FORMAT = "compact_jsonl_v4"
DATASET_DEBUG_PLAN_FILE = "plan_debug.json"
DATASET_DEBUG_TIMELINE_FILE = "debug_timeline.jsonl"
AUTO_COLLECT_MAX_ATTEMPT_MULTIPLIER = 5
PLANNER_VERSION = "dig_plan_v4_joint_space_world_debug"
QUALITY_GATE_VERSION = "quality_gate_v2_particles_no_freeze"
AUTO_PREFLIGHT_MIN_PARTICLES = 1000
AUTO_DIG_GRID_SIZE = 7
AUTO_DIG_TOPK_TARGETS = 8
AUTO_DIG_CORE_NORM_MAX = 0.62
AUTO_DIG_DENSITY_RADIUS = 0.34
AUTO_DIG_MIN_LOCAL_PARTICLES = 24
AUTO_DIG_MIN_SWEPT_PARTICLES = 24
AUTO_DIG_RING_RADII = [0.0, 0.18, 0.36, 0.55]
AUTO_DIG_RING_POINTS = [1, 6, 8, 10]
AUTO_DIG_DEPTH_PRIORITY = [0.22, 0.18, 0.14, 0.10]
AUTO_DIG_SWEEP_RADIUS = 0.30
AUTO_DIG_FULL_PLAN_TOPK_PER_RING = 4
AUTO_DIG_SCORE_WEIGHTS = {
    "reach": 32.0,
    "fill": 26.0,
    "swept_density": 22.0,
    "depth": 12.0,
    "center": 10.0,
    "motion": 6.0,
    "approach": 8.0,
}
AUTO_UNLOAD_GRID_SIZE = 5
# Keep the selected landing cell far enough from bin walls for the bucket body,
# not just the falling sand point.
AUTO_UNLOAD_WALL_MARGIN = 0.60
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
SAND_SNAPSHOT_MAX_AGE = 0.35
SAND_SNAPSHOT_GRID_RES = 64
SAND_SNAPSHOT_GRID_MAX_RES = 96
SAND_SNAPSHOT_CELL_RADIUS_LIMIT = 10
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
SAND_BUCKET_DIAG_EXPAND_LOCAL = np.array([0.40, 0.30, 0.45], dtype=np.float32)
SAND_SOURCE_FALLBACK_MIN_REGION_COUNT = 32
SAND_SOURCE_FALLBACK_MIN_RATIO = 0.60
SAND_PILE_CENTER = AUTO_COLLECT_TARGET_CENTER.copy()
SAND_PILE_RADIUS_X = 1.18
SAND_PILE_RADIUS_Y = 1.18
SAND_PILE_Z_MIN = -0.05
SAND_PILE_Z_MAX = 2.20
SAND_SETTLED_MIN_FRACTION_IN_FOOTPRINT = 0.55
SAND_SETTLED_HIGH_AIR_FRACTION_MAX = 0.08
SAND_SETTLED_Z_MARGIN = 0.75
SAND_SETTLED_SURFACE_MARGIN = 0.35
SAND_SURFACE_QUERY_RADIUS = 0.24
FRONT_EDGE_BODY_DEPTH_SOFT_MARGIN = 0.045
FRONT_EDGE_BODY_DEPTH_HARD_MARGIN = 0.145
FRONT_EDGE_MIN_TIP_DEPTH = 0.020
EXIT_BODY_DEPTH_SOFT_MARGIN = 0.10
EXIT_BODY_DEPTH_HARD_MARGIN = 0.26
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

# Planner-only limits are narrower than the authored joint limits. They keep
# route generation away from poses that are technically inside USD limits but
# repeatedly fail actuator tracking in simulation, such as deep arm tuck near
# -95 deg during obstacle avoidance.
PATH_EFFECTIVE_LIMITS_DEG = {
    "boom": (-70.0, 74.0),
    "arm": (-89.0, 92.0),
    "bucket": (-118.0, 88.0),
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
BUCKET_CARRY_MAX_DUMP_BRANCH_DEG = 35.0
BUCKET_CARRY_MAX_ADJUST_DEG = 65.0
BUCKET_CARRY_SOFT_ADJUST_DEG = 42.0
BUCKET_CARRY_TRANSITIONAL_MIN_POUR_Z = -0.28
BUCKET_CARRY_SPILL_RISK_COST = 140.0
BUCKET_DIG_APPROACH_WORLD_DEG = -52.0
BUCKET_DIG_INSERT_WORLD_DEG = -78.0
BUCKET_DIG_PULL_WORLD_DEG = -92.0
BUCKET_DIG_EXIT_WORLD_DEG = -96.0
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
UNLOAD_DROP_SCATTER_MARGIN_XY = 0.58
UNLOAD_FORCE_CENTER_HIGH_RELEASE = True
UNLOAD_CENTER_RELEASE_XY_TOL = 0.16
UNLOAD_CENTER_RELEASE_SOFT_XY_TOL = 0.50
UNLOAD_CENTER_RELEASE_CORRECTION_GAIN = 0.45
UNLOAD_RELEASE_SOURCE_BLEND = 0.50
UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z = 0.18
UNLOAD_PREFERRED_RELEASE_ABOVE_WALL_Z = 1.65
UNLOAD_DROP_GRAVITY = 9.81
UNLOAD_DROP_ROLL_OFFSET_BASE = 0.24
UNLOAD_DROP_ROLL_OFFSET_PER_M = 0.10
UNLOAD_DROP_TANGENTIAL_GAIN = 0.36
UNLOAD_DROP_MAX_ROLL_OFFSET = 1.25
UNLOAD_DROP_MAX_XY_CORRECTION = 1.35
UNLOAD_DUMP_STEP_DEG = 10.0
UNLOAD_DUMP_STEP_SECONDS = 0.12
UNLOAD_DUMP_STOP_BUCKET_FRACTION = 0.15
UNLOAD_DUMP_FLOW_CENTER_BUCKET_DEG = -25.0
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
        "approach_offset": 0.34,
        "pre_z": 0.46,
        "contact_z": 0.055,
        "insert_depth": 0.08,
        "mid_pull": 0.34,
        "mid_depth": 0.14,
        "exit_pull": 0.56,
        "exit_depth": 0.04,
        "exit_lift_z": 0.08,
        "curl_z": 0.34,
        "lift_height": 0.70,
        "unload_height_delta": 0.00,
        "unload_dump_deg": 82.0,
        "bucket_attack_world": -52.0,
        "bucket_cut_world": -78.0,
        "bucket_mid_cut_world": -92.0,
        "bucket_exit_world": -96.0,
        "bucket_curl": CURL_HOLD_TARGET_DEG,
        "curl_boom_lift_deg": 4.5,
    },
    {
        "id": "high_lift",
        "approach_offset": 0.42,
        "pre_z": 0.62,
        "contact_z": 0.075,
        "insert_depth": 0.07,
        "mid_pull": 0.38,
        "mid_depth": 0.12,
        "exit_pull": 0.62,
        "exit_depth": 0.03,
        "exit_lift_z": 0.10,
        "curl_z": 0.40,
        "lift_height": 1.05,
        "unload_height_delta": 0.18,
        "unload_dump_deg": 86.0,
        "bucket_attack_world": -48.0,
        "bucket_cut_world": -74.0,
        "bucket_mid_cut_world": -90.0,
        "bucket_exit_world": -94.0,
        "bucket_curl": CURL_HOLD_TARGET_DEG,
        "curl_boom_lift_deg": 5.5,
    },
    {
        "id": "front_bite",
        "approach_offset": 0.46,
        "pre_z": 0.52,
        "contact_z": 0.06,
        "insert_depth": 0.10,
        "mid_pull": 0.30,
        "mid_depth": 0.16,
        "exit_pull": 0.50,
        "exit_depth": 0.05,
        "exit_lift_z": 0.08,
        "curl_z": 0.36,
        "lift_height": 0.88,
        "unload_height_delta": 0.10,
        "unload_dump_deg": 88.0,
        "bucket_attack_world": -58.0,
        "bucket_cut_world": -84.0,
        "bucket_mid_cut_world": -96.0,
        "bucket_exit_world": -98.0,
        "bucket_curl": CURL_HOLD_TARGET_DEG,
        "curl_boom_lift_deg": 4.0,
    },
    {
        "id": "shallow_long",
        "approach_offset": 0.54,
        "pre_z": 0.58,
        "contact_z": 0.08,
        "insert_depth": 0.06,
        "mid_pull": 0.48,
        "mid_depth": 0.10,
        "exit_pull": 0.72,
        "exit_depth": 0.02,
        "exit_lift_z": 0.12,
        "curl_z": 0.44,
        "lift_height": 1.15,
        "unload_height_delta": 0.22,
        "unload_dump_deg": 86.0,
        "bucket_attack_world": -45.0,
        "bucket_cut_world": -70.0,
        "bucket_mid_cut_world": -86.0,
        "bucket_exit_world": -92.0,
        "bucket_curl": CURL_HOLD_TARGET_DEG,
        "curl_boom_lift_deg": 6.0,
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
UI_STATUS_MAX_CHARS = 72

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
    # Speed is handled in software only. Joint angle limits are authored by
    # ensure_joint_limits_are_valid() before the articulation is created.
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


def invalidate_active_task(reason=""):
    STATE["active_task_id"] = int(STATE.get("active_task_id", 0)) + 1
    STATE["active_task_name"] = "idle"
    STATE["sand_contact_stage"] = ""
    STATE["sand_contact_stage_started"] = 0.0
    if reason:
        info_print("[TASK INVALIDATE]", f"reason={reason}")


def motion_cancel_requested(task_id=None):
    if not bool(STATE.get("running", False)):
        return True
    if task_id is not None and not task_alive(task_id):
        return True
    if bool(STATE.get("auto_collect_stop_requested", False)) and str(STATE.get("dig_plan_planning_source", "")) == "auto_collect":
        return True
    if bool(STATE.get("planning_cancel_requested", False)) and str(STATE.get("dig_plan_planning_source", "")) == "auto_collect":
        return True
    return False


def cancel_registered_task(name, reason=""):
    tasks = STATE.setdefault("async_tasks", {})
    task = tasks.get(str(name))
    if str(name) in ("planner", "auto_collect"):
        STATE["planning_cancel_requested"] = True
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
    sand_floor_z = float(raw.get("sand_floor_z", GROUND_TOP_Z))
    sand_fill_height = float(raw.get("sand_fill_height", max(0.25, SAND_PILE_Z_MAX - SAND_PILE_Z_MIN)))

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
        "sand_floor_z": float(sand_floor_z),
        "sand_fill_height": max(0.05, float(sand_fill_height)),
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


def update_sphere_marker(path, point, radius, color):
    if point is None:
        hide_debug_prim(path)
        return None
    p = np.array(point, dtype=np.float32).reshape(-1)[:3]
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        prim = make_sphere(
            path,
            translate=(float(p[0]), float(p[1]), float(p[2])),
            radius=float(radius),
            color=color,
            collision=False,
        )
    else:
        set_xform(prim, translate=(float(p[0]), float(p[1]), float(p[2])))
        set_color(prim, color)
        disable_collision(prim, "debug_marker")
    try:
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    except Exception:
        pass
    return prim


def hide_debug_prim(path):
    try:
        prim = get_prim(path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            return True
    except Exception:
        pass
    return False


def update_debug_line(path, points, color, width=0.035):
    pts = []
    for point in points or []:
        if point is None:
            continue
        p = np.array(point, dtype=np.float32).reshape(-1)[:3]
        pts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))
    if len(pts) < 2:
        hide_debug_prim(path)
        return None
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
        curve.GetPointsAttr().Set(pts) if curve.GetPointsAttr().IsValid() else curve.CreatePointsAttr(pts)
        counts = [len(pts)]
        curve.GetCurveVertexCountsAttr().Set(counts) if curve.GetCurveVertexCountsAttr().IsValid() else curve.CreateCurveVertexCountsAttr(counts)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        try:
            curve.GetBasisAttr().Clear()
        except Exception:
            pass
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateWidthsAttr([float(width)])
    except Exception:
        pass
    prim = curve.GetPrim()
    set_color(prim, color)
    disable_collision(prim, "debug_line")
    try:
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    except Exception:
        pass
    return prim


def update_debug_rect_loop(path, center_xy, half_xy, z, color, width=0.025):
    try:
        c = np.array(center_xy, dtype=np.float32).reshape(-1)[:2]
        h = np.maximum(np.array(half_xy, dtype=np.float32).reshape(-1)[:2], np.array([0.02, 0.02], dtype=np.float32))
        zz = float(z)
        pts = [
            [float(c[0] - h[0]), float(c[1] - h[1]), zz],
            [float(c[0] + h[0]), float(c[1] - h[1]), zz],
            [float(c[0] + h[0]), float(c[1] + h[1]), zz],
            [float(c[0] - h[0]), float(c[1] + h[1]), zz],
            [float(c[0] - h[0]), float(c[1] - h[1]), zz],
        ]
        return update_debug_line(path, pts, color, width=width)
    except Exception:
        hide_debug_prim(path)
        return None


def draw_unload_dump_debug(
    label,
    landing_target=None,
    release_target=None,
    q_seed=None,
    q_seed_dump=None,
    best_drop=None,
    reason="",
    best_reachable_point=None,
    rejected_segment=None,
):
    if CONTROL_ROOT is None:
        return
    root = f"{CONTROL_ROOT}/LoadedRouteDebug"
    if not get_prim(root).IsValid():
        UsdGeom.Xform.Define(stage, root)
    ctx = None
    wall_z = None
    overpass_z = None
    preferred_z = None
    try:
        ctx = task_scene_context()
        bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
        bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
        bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
        wall_z = float(bin_z_range[1]) if len(bin_z_range) >= 2 else float(GROUND_TOP_Z)
        overpass_z = wall_z + float(UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z)
        preferred_z = preferred_unload_release_z(ctx=ctx, wall_z=wall_z)
        update_debug_rect_loop(
            f"{root}/WallOverpassClearanceOutline",
            bin_center[:2],
            bin_half,
            overpass_z,
            (1.0, 0.92, 0.05),
            width=0.035,
        )
        update_debug_rect_loop(
            f"{root}/PreferredHighReleaseOutline",
            bin_center[:2],
            bin_half,
            preferred_z,
            (0.10, 0.95, 1.0),
            width=0.028,
        )
    except Exception:
        hide_debug_prim(f"{root}/WallOverpassClearanceOutline")
        hide_debug_prim(f"{root}/PreferredHighReleaseOutline")
    landing = None if landing_target is None else np.array(landing_target, dtype=np.float32).reshape(-1)[:3]
    release = None if release_target is None else np.array(release_target, dtype=np.float32).reshape(-1)[:3]
    current_drop = None
    try:
        q_eval = q_seed_dump if q_seed_dump is not None else q_seed
        if q_eval is not None:
            current_drop = unload_drop_report(q=q_eval, reference_q=q_seed)
    except Exception:
        current_drop = None
    current_release = None
    current_landing = None
    drop_for_log = best_drop if isinstance(best_drop, dict) else current_drop
    if isinstance(drop_for_log, dict):
        current_release = drop_for_log.get("release")
        current_landing = drop_for_log.get("landing")

    update_sphere_marker(f"{root}/LandingTarget", landing, 0.105, (0.05, 1.0, 0.25))
    update_sphere_marker(f"{root}/RequiredRelease", release, 0.090, (0.95, 0.10, 1.0))
    update_sphere_marker(f"{root}/PredictedRelease", current_release, 0.080, (1.0, 0.55, 0.05))
    update_sphere_marker(f"{root}/PredictedLanding", current_landing, 0.080, (1.0, 0.05, 0.05))
    update_sphere_marker(f"{root}/BestReachableCarry", best_reachable_point, 0.075, (0.20, 0.55, 1.0))
    update_debug_line(f"{root}/RequiredDropLine", [release, landing], (0.05, 1.0, 0.25), width=0.040)
    update_debug_line(f"{root}/PredictedDropLine", [current_release, current_landing], (1.0, 0.35, 0.05), width=0.050)
    update_debug_line(f"{root}/ReleaseErrorLine", [current_release, release], (0.95, 0.10, 1.0), width=0.026)
    update_debug_line(f"{root}/LandingErrorLine", [current_landing, landing], (1.0, 0.05, 0.05), width=0.030)
    if overpass_z is not None:
        release_floor = None if release is None else [float(release[0]), float(release[1]), float(overpass_z)]
        current_release_floor = None if current_release is None else [
            float(np.array(current_release, dtype=np.float32).reshape(-1)[0]),
            float(np.array(current_release, dtype=np.float32).reshape(-1)[1]),
            float(overpass_z),
        ]
        landing_floor = None if landing is None else [float(landing[0]), float(landing[1]), float(overpass_z)]
        update_debug_line(f"{root}/RequiredReleaseHeightLine", [release_floor, release], (0.95, 0.10, 1.0), width=0.022)
        update_debug_line(f"{root}/PredictedReleaseHeightLine", [current_release_floor, current_release], (1.0, 0.55, 0.05), width=0.020)
        update_debug_line(f"{root}/LandingTargetHeightLine", [landing_floor, landing], (0.05, 1.0, 0.25), width=0.020)
    else:
        hide_debug_prim(f"{root}/RequiredReleaseHeightLine")
        hide_debug_prim(f"{root}/PredictedReleaseHeightLine")
        hide_debug_prim(f"{root}/LandingTargetHeightLine")
    if rejected_segment is not None:
        try:
            a, b = rejected_segment
            update_debug_line(f"{root}/RejectedWallSegment", [a, b], (1.0, 0.05, 0.05), width=0.070)
        except Exception:
            pass
    else:
        hide_debug_prim(f"{root}/RejectedWallSegment")

    now = time.time()
    if now - float(STATE.get("last_unload_debug_legend_time", 0.0) or 0.0) > 4.0:
        STATE["last_unload_debug_legend_time"] = now
        info_print(
            "[UNLOAD DEBUG LEGEND]",
            f"path={root}",
            "green=landing_target/required_drop",
            "magenta=required_high_release",
            "orange=predicted_release/drop",
            "red=predicted_landing/error",
            "blue=best_reachable_carry",
            "yellow=wall_overpass_clearance",
            "cyan=preferred_release_height",
        )

    if isinstance(drop_for_log, dict):
        info_print(
            "[UNLOAD DUMP DEBUG]",
            f"label={label}",
            f"path={root}",
            f"reason={reason}",
            f"landing_target={vec_list(landing, 3)}",
            f"required_release={vec_list(release, 3)}",
            f"predicted_release={drop_for_log.get('release')}",
            f"predicted_landing={drop_for_log.get('landing')}",
            f"xy_err={fmt_optional(drop_for_log.get('xy_err'))}",
            f"inside_xy={drop_for_log.get('inside_xy')}",
            f"above_wall={drop_for_log.get('above_wall')}",
            f"scatter_xy_ok={drop_for_log.get('scatter_xy_ok')}",
            f"execution_ok={drop_for_log.get('execution_ok')}",
            f"acceptance={drop_for_log.get('landing_acceptance')}",
            f"release_centered_ok={drop_for_log.get('release_centered_ok')}",
            f"release_xy_err={fmt_optional(drop_for_log.get('release_xy_err'))}",
            f"overflow={fmt_optional(drop_for_log.get('bin_overflow_xy'))}",
            f"source_clearance={fmt_optional(drop_for_log.get('source_clearance'))}",
            f"wall_z={fmt_optional(wall_z)}",
            f"overpass_z={fmt_optional(overpass_z)}",
            f"preferred_release_z={fmt_optional(preferred_z)}",
            f"drift={fmt_optional(drop_for_log.get('drift_distance'))}",
            f"best_reachable={vec_list(best_reachable_point, 3)}",
            f"q_seed={q_deg_values(q_seed, wrap_swing_for_display=True) if q_seed is not None else None}",
            f"q_seed_dump={q_deg_values(q_seed_dump, wrap_swing_for_display=True) if q_seed_dump is not None else None}",
        )


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


def bucket_unload_release_point_world(q=None, reference_q=None):
    q_eval = CTRL.q_cmd.copy() if q is None else np.array(q, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = CTRL.q_cmd.copy() if reference_q is None else np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    load = bucket_point_world("load", q=q_eval, reference_q=q_ref)
    pour = bucket_point_world("pour", q=q_eval, reference_q=q_ref)
    if pour is None:
        return None
    pour = np.array(pour, dtype=np.float32).reshape(-1)[:3]
    if load is None:
        return pour
    load = np.array(load, dtype=np.float32).reshape(-1)[:3]
    blend = clamp(float(UNLOAD_RELEASE_SOURCE_BLEND), 0.0, 1.0)
    return load * (1.0 - blend) + pour * blend


def pour_target_for_unload_release_source(release_target, q_estimate=None, q_start=None):
    release = np.array(release_target, dtype=np.float32).reshape(-1)[:3].copy()
    q_eval = CTRL.q_cmd.copy() if q_estimate is None else np.array(q_estimate, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = q_eval.copy() if q_start is None else np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    source = bucket_unload_release_point_world(q=q_eval, reference_q=q_ref)
    pour = bucket_point_world("pour", q=q_eval, reference_q=q_ref)
    if source is None or pour is None:
        return release
    source = np.array(source, dtype=np.float32).reshape(-1)[:3]
    pour = np.array(pour, dtype=np.float32).reshape(-1)[:3]
    return release + (pour - source)


def unload_drop_drift_model(q=None, reference_q=None, release=None, load=None, wall_z=None):
    q_eval = CTRL.q_cmd.copy() if q is None else np.array(q, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = CTRL.q_cmd.copy() if reference_q is None else np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    forward_xy = bucket_dump_forward_xy(q_eval, reference_q=q_ref)

    release_arr = None if release is None else np.array(release, dtype=np.float32).reshape(-1)[:3]
    load_arr = None if load is None else np.array(load, dtype=np.float32).reshape(-1)[:3]
    if release_arr is None:
        release_arr = bucket_unload_release_point_world(q=q_eval, reference_q=q_ref)
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


def preferred_unload_release_z(ctx=None, wall_z=None):
    ctx = task_scene_context() if ctx is None else ctx
    wall = float(GROUND_TOP_Z if wall_z is None else wall_z)
    try:
        bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
        if len(bin_z_range) >= 2:
            wall = float(bin_z_range[1])
    except Exception:
        pass
    try:
        nominal = float(np.array(unload_bin_dump_point(ctx=ctx), dtype=np.float32).reshape(-1)[2])
    except Exception:
        nominal = float(GROUND_TOP_Z + UNLOAD_TARGET_MIN_Z)
    return max(
        float(nominal),
        float(wall) + float(UNLOAD_PREFERRED_RELEASE_ABOVE_WALL_Z),
        float(GROUND_TOP_Z) + float(UNLOAD_TARGET_MIN_Z),
    )


def predict_unload_drop(q=None, reference_q=None):
    ctx = task_scene_context()
    q_ref = CTRL.q_cmd if reference_q is None else reference_q
    release = bucket_unload_release_point_world(q=q, reference_q=q_ref)
    load = bucket_point_world("load", q=q, reference_q=q_ref)
    pour = bucket_point_world("pour", q=q, reference_q=q_ref)
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
        "pour": None if pour is None else np.array(pour, dtype=np.float32).reshape(-1)[:3],
        "release_source": "bucket_opening_center",
        "release_source_blend": float(clamp(float(UNLOAD_RELEASE_SOURCE_BLEND), 0.0, 1.0)),
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
    scatter_margin = max(0.0, float(UNLOAD_DROP_SCATTER_MARGIN_XY))
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
    release_dx = float(release[0] - target[0])
    release_dy = float(release[1] - target[1])
    release_xy_err = float(math.sqrt(release_dx * release_dx + release_dy * release_dy))
    bin_dx = float(landing[0] - bin_center[0])
    bin_dy = float(landing[1] - bin_center[1])
    inside_xy = abs(bin_dx) <= float(safe_half[0]) and abs(bin_dy) <= float(safe_half[1])
    above_wall = float(drop["source_clearance"]) >= UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z
    close_xy = xy_err <= UNLOAD_DROP_XY_TOL
    overflow_x = max(0.0, abs(bin_dx) - float(safe_half[0]))
    overflow_y = max(0.0, abs(bin_dy) - float(safe_half[1]))
    overflow_xy = float(math.sqrt(overflow_x * overflow_x + overflow_y * overflow_y))
    scatter_center_tol = float(np.linalg.norm(safe_half + scatter_margin))
    scatter_xy_ok = bool(above_wall and overflow_xy <= scatter_margin and xy_err <= scatter_center_tol)
    release_center_tol = max(float(UNLOAD_CENTER_RELEASE_XY_TOL), 0.12)
    release_centered_ok = bool(above_wall and release_xy_err <= release_center_tol)
    execution_ok = bool(above_wall and (inside_xy or scatter_xy_ok or close_xy or release_centered_ok))
    if close_xy:
        acceptance = "close_xy"
    elif release_centered_ok:
        acceptance = "center_high_release"
    elif inside_xy:
        acceptance = "inside_bin"
    elif scatter_xy_ok:
        acceptance = "scatter_margin"
    else:
        acceptance = "outside_scatter_margin"

    ok = bool(inside_xy and above_wall)
    return {
        "ok": ok,
        "reason": "ok" if ok else "landing_outside_bin_or_too_low",
        "execution_ok": bool(execution_ok),
        "landing_acceptance": acceptance,
        "inside_xy": bool(inside_xy),
        "above_wall": bool(above_wall),
        "close_xy": bool(close_xy),
        "scatter_xy_ok": bool(scatter_xy_ok),
        "scatter_margin_xy": float(scatter_margin),
        "release_centered_ok": bool(release_centered_ok),
        "release_center_tol": float(release_center_tol),
        "release_dx": float(release_dx),
        "release_dy": float(release_dy),
        "release_xy_err": float(release_xy_err),
        "bin_overflow_xy": float(overflow_xy),
        "bin_overflow_x": float(overflow_x),
        "bin_overflow_y": float(overflow_y),
        "scatter_center_tol": float(scatter_center_tol),
        "target": vec_list(target, 3),
        "landing": vec_list(landing, 3),
        "release": vec_list(release, 3),
        "load": vec_list(drop.get("load"), 3),
        "pour": vec_list(drop.get("pour"), 3),
        "release_source": str(drop.get("release_source", "bucket_opening_center")),
        "release_source_blend": float(drop.get("release_source_blend", UNLOAD_RELEASE_SOURCE_BLEND)),
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
        f"scatter_xy_ok={report.get('scatter_xy_ok')} "
        f"execution_ok={report.get('execution_ok')} "
        f"acceptance={report.get('landing_acceptance')} "
        f"release_centered_ok={report.get('release_centered_ok')} "
        f"release_xy_err={fmt_optional(report.get('release_xy_err'))} "
        f"release_source={report.get('release_source')} "
        f"overflow={fmt_optional(report.get('bin_overflow_xy'))} "
        f"scatter_margin={fmt_optional(report.get('scatter_margin_xy'))} "
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
        unload_drop_execution_ready(drop_report)
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
        f"drop_scatter_xy_ok={drop.get('scatter_xy_ok')} "
        f"drop_acceptance={drop.get('landing_acceptance')} "
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
    drop_ok = unload_drop_execution_ready(drop)
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
        "bucket_from_initial": int(metrics.get("bucket_from_initial_count", metrics.get("bucket_from_pile_count", 0))),
        "bin": int(metrics.get("bin_count", 0)),
        "bin_from_pile": int(metrics.get("bin_from_pile_count", 0)),
        "bin_from_initial": int(metrics.get("bin_from_initial_count", metrics.get("bin_from_pile_count", 0))),
        "spill_from_pile": int(metrics.get("spill_from_pile_count", 0)),
        "spill_from_initial": int(metrics.get("spill_from_initial_count", metrics.get("spill_from_pile_count", 0))),
        "source_tracking": str(metrics.get("source_tracking", "unknown")),
        "source_tracking_notes": str(metrics.get("source_tracking_notes", "")),
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
        "execution_ok": bool(report.get("execution_ok", False)),
        "reason": str(report.get("reason", "")),
        "landing_acceptance": str(report.get("landing_acceptance", "")),
        "inside_xy": bool(report.get("inside_xy", False)),
        "above_wall": bool(report.get("above_wall", False)),
        "close_xy": bool(report.get("close_xy", False)),
        "scatter_xy_ok": bool(report.get("scatter_xy_ok", False)),
        "release_centered_ok": bool(report.get("release_centered_ok", False)),
        "release_xy_err": None if report.get("release_xy_err") is None else float(report.get("release_xy_err")),
        "release_center_tol": None if report.get("release_center_tol") is None else float(report.get("release_center_tol")),
        "bin_overflow_xy": None if report.get("bin_overflow_xy") is None else float(report.get("bin_overflow_xy")),
        "scatter_margin_xy": None if report.get("scatter_margin_xy") is None else float(report.get("scatter_margin_xy")),
        "dx": None if report.get("dx") is None else float(report.get("dx")),
        "dy": None if report.get("dy") is None else float(report.get("dy")),
        "xy_err": None if report.get("xy_err") is None else float(report.get("xy_err")),
        "source_clearance": None if report.get("source_clearance") is None else float(report.get("source_clearance")),
        "target": vec_list(report.get("target"), 3),
        "landing": vec_list(report.get("landing"), 3),
        "release": vec_list(report.get("release"), 3),
        "load": vec_list(report.get("load"), 3),
        "forward_xy": vec_list(report.get("forward_xy"), 2),
        "drift_xy": vec_list(report.get("drift_xy"), 2),
        "drift_distance": None if report.get("drift_distance") is None else float(report.get("drift_distance")),
        "roll_offset": None if report.get("roll_offset") is None else float(report.get("roll_offset")),
        "tangential_offset": None if report.get("tangential_offset") is None else float(report.get("tangential_offset")),
        "fall_time": None if report.get("fall_time") is None else float(report.get("fall_time")),
        "height_above_wall": None if report.get("height_above_wall") is None else float(report.get("height_above_wall")),
        "lip_span_xy": None if report.get("lip_span_xy") is None else float(report.get("lip_span_xy")),
        "bucket_delta_deg": None if report.get("bucket_delta_deg") is None else float(report.get("bucket_delta_deg")),
        "bin_center": vec_list(report.get("bin_center"), 3),
        "safe_half": vec_list(report.get("safe_half"), 2),
        "source": str(report.get("source", "")),
    }


def unload_drop_execution_ready(report):
    if not isinstance(report, dict):
        return False
    return bool(
        report.get("above_wall", False)
        and (
            report.get("close_xy", False)
            or report.get("inside_xy", False)
            or report.get("scatter_xy_ok", False)
            or report.get("release_centered_ok", False)
            or report.get("execution_ok", False)
        )
    )


def stage_constraint_summary(row):
    if not isinstance(row, dict):
        return {}
    motion = row.get("motion") if isinstance(row.get("motion"), dict) else {}
    path = row.get("path") if isinstance(row.get("path"), dict) else {}
    front = row.get("front_edge") if isinstance(row.get("front_edge"), dict) else {}
    hold = row.get("material_hold") if isinstance(row.get("material_hold"), dict) else {}
    drop = row.get("drop") if isinstance(row.get("drop"), dict) else {}
    clearance = row.get("clearance_route") if isinstance(row.get("clearance_route"), dict) else {}

    out = {
        "stage_cost": None if row.get("stage_cost") is None else float(row.get("stage_cost")),
        "motion_cost": None if motion.get("cost") is None else float(motion.get("cost")),
        "weighted_angle": None if motion.get("weighted_angle") is None else float(motion.get("weighted_angle")),
        "estimated_time": None if motion.get("estimated_time") is None else float(motion.get("estimated_time")),
        "joint_delta_deg": motion.get("joint_delta_deg", []),
        "path_penalty": None if path.get("path_penalty") is None else float(path.get("path_penalty")),
        "phase_ok": bool(path.get("phase_ok", True)),
        "obstacle_ok": bool(path.get("obstacle_ok", True)),
        "route_inserted": bool(path.get("route_inserted", clearance.get("inserted", False))),
        "route_waypoints": int(path.get("route_waypoints", clearance.get("waypoints", 0)) or 0),
        "phase_reason": str(path.get("phase_reason", "")),
        "obstacle_reason": str(path.get("obstacle_reason", "")),
    }
    if front:
        out["front_edge"] = {
            "ok": bool(front.get("ok", False)),
            "reason": str(front.get("reason", "")),
            "tip_depth": front.get("tip_depth"),
            "bucket_mid_depth": front.get("bucket_mid_depth"),
            "pour_depth": front.get("pour_depth"),
            "load_depth": front.get("load_depth"),
            "surface_source": front.get("surface_source"),
        }
    if hold:
        carry_report = hold.get("carry_report") if isinstance(hold.get("carry_report"), dict) else {}
        carry_retains = hold.get(
            "carry_retains_material",
            hold.get("retains_material", carry_report.get("retains_material", False)),
        )
        out["material_hold"] = {
            "bucket_closed_ok": bool(hold.get("bucket_closed_ok", True)),
            "carry_retains_material": bool(carry_retains),
            "pour_above_load_z": carry_report.get("pour_above_load_z", hold.get("pour_above_load_z")),
            "target_world_deg": carry_report.get("target_world_deg", hold.get("target_world_deg")),
            "actual_world_deg": carry_report.get("actual_world_deg", hold.get("actual_world_deg")),
            "world_err_deg": carry_report.get("world_err_deg", hold.get("world_err_deg")),
        }
    if drop:
        out["unload_drop"] = {
            "ok": bool(drop.get("ok", False)),
            "inside_xy": bool(drop.get("inside_xy", False)),
            "close_xy": bool(drop.get("close_xy", False)),
            "above_wall": bool(drop.get("above_wall", False)),
            "xy_err": drop.get("xy_err"),
            "source_clearance": drop.get("source_clearance"),
            "drift_distance": drop.get("drift_distance"),
            "fall_time": drop.get("fall_time"),
        }
    flags = []
    if not out["phase_ok"]:
        flags.append("phase_path_risk")
    if not out["obstacle_ok"]:
        flags.append("rigid_obstacle_risk")
    if out.get("front_edge") and not out["front_edge"].get("ok", False):
        flags.append("bad_cut_front_edge")
    if out.get("material_hold") and not out["material_hold"].get("carry_retains_material", False):
        flags.append("carry_spill_risk")
    if out.get("unload_drop") and not out["unload_drop"].get("ok", False):
        flags.append("unload_drop_risk")
    out["flags"] = flags
    return out


def compact_plan_stage(row):
    if not isinstance(row, dict):
        return {}
    out = {
        "phase": str(row.get("phase", "")),
        "planned": bool(row.get("planned", False)),
        "required": bool(row.get("required", False)),
        "target": vec_list(row.get("target_point"), 3),
    }
    out["collision_context"] = phase_collision_context(out["phase"])
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
    if row.get("route_source"):
        out["route_source"] = str(row.get("route_source"))
    if row.get("route_reason"):
        out["route_reason"] = str(row.get("route_reason"))
    if row.get("route_index") is not None:
        out["route_index"] = int(row.get("route_index"))
    if row.get("route_count") is not None:
        out["route_count"] = int(row.get("route_count"))
    if row.get("clearance_route") is not None:
        out["clearance_route"] = row.get("clearance_route")
    if row.get("path") is not None:
        path = row.get("path")
        if isinstance(path, dict):
            out["path"] = {
                "phase_ok": bool(path.get("phase_ok", True)),
                "obstacle_ok": bool(path.get("obstacle_ok", True)),
                "phase_reason": str(path.get("phase_reason", "")),
                "obstacle_reason": str(path.get("obstacle_reason", "")),
                "route_inserted": bool(path.get("route_inserted", False)),
                "route_waypoints": int(path.get("route_waypoints", 0) or 0),
                "route_reason": str(path.get("route_reason", "")),
            }
    constraints = stage_constraint_summary(row)
    if constraints:
        out["constraint_summary"] = constraints
    return out


def route_diagnostics_from_stages(stages):
    rows = []
    if not isinstance(stages, list):
        return rows
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        phase = str(stage.get("phase", ""))
        is_route = phase.startswith("clearance_route")
        clearance = stage.get("clearance_route") if isinstance(stage.get("clearance_route"), dict) else None
        if not is_route and not clearance:
            continue
        row = {
            "stage_index": int(idx),
            "phase": phase,
            "route_source": str(stage.get("route_source", "")),
            "route_reason": str(stage.get("route_reason", "")),
            "route_index": None if stage.get("route_index") is None else int(stage.get("route_index")),
            "route_count": None if stage.get("route_count") is None else int(stage.get("route_count")),
            "target_point": vec_list(stage.get("target_point"), 3),
            "q_goal_deg": stage.get("q_goal_deg"),
            "duration": None if stage.get("duration") is None else float(stage.get("duration")),
            "motion": stage.get("motion", {}),
            "path": stage.get("path", {}),
        }
        if clearance:
            row["clearance_route"] = clearance
        rows.append(row)
    return rows


def unload_ballistics_from_stages(stages):
    if not isinstance(stages, list):
        return {}
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        phase = str(stage.get("phase", ""))
        if not phase.startswith("unload"):
            continue
        drop = stage.get("drop")
        if not isinstance(drop, dict):
            continue
        out = dict(drop)
        out["stage_index"] = int(idx)
        out["phase"] = phase
        out["drop_alignment_ready"] = bool(stage.get("drop_alignment_ready", False))
        out["drop_alignment_policy"] = str(stage.get("drop_alignment_policy", ""))
        return out
    return {}


def compact_plan_candidate(row, include_stages=False):
    if not isinstance(row, dict):
        return {}
    stages = row.get("stages", [])
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
        "dig_primitive": row.get("dig_primitive", {}),
        "route_diagnostics": row.get("route_diagnostics", []),
        "unload_ballistics": row.get("unload_ballistics", {}) or unload_ballistics_from_stages(stages),
        "fsm_contract": row.get("fsm_contract", {}),
    }
    if include_stages:
        out["stages"] = [compact_plan_stage(x) for x in stages]
    return out


def compact_dig_primitive_params(candidate):
    candidate = candidate if isinstance(candidate, dict) else {}
    keys = [
        "id",
        "surface_z",
        "depth_candidate",
        "approach_offset",
        "pre_z",
        "contact_z",
        "insert_depth",
        "mid_pull",
        "mid_depth",
        "exit_pull",
        "exit_depth",
        "exit_lift_z",
        "curl_z",
        "lift_height",
        "unload_height_delta",
        "unload_dump_deg",
        "bucket_attack_world",
        "bucket_cut_world",
        "bucket_mid_cut_world",
        "bucket_exit_world",
        "bucket_curl",
        "curl_boom_lift_deg",
    ]
    out = {}
    for key in keys:
        if key in candidate:
            value = candidate.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                out[key] = float(value) if key != "id" else str(value)
            else:
                out[key] = value
    return out


DIG_PLAN_REQUIRED_PHASE_ORDER = [
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "pull_exit_cut",
    "curl_to_hold_material",
    "secure_load",
    "lift_carry",
    "unload_to_bin",
]


def dig_plan_semantic_phase_name(phase):
    text = str(phase).lower()
    if text.startswith("clearance_route"):
        return "clearance_route"
    idx = dataset_phase_index(text)
    if 0 <= idx < len(DATASET_PHASE_NAMES):
        return DATASET_PHASE_NAMES[idx]
    return text


def validate_dig_plan_contract(seq=None, points=None, stages=None, trace_points=None):
    seq = [] if seq is None else list(seq)
    points = [] if points is None else list(points)
    stages = [] if stages is None else list(stages)
    trace_points = [] if trace_points is None else list(trace_points)

    if stages:
        raw_phases = [str(stage.get("phase", "")) for stage in stages if isinstance(stage, dict)]
    else:
        raw_phases = [str(item[0]) for item in seq if isinstance(item, (list, tuple)) and len(item) >= 1]

    semantic = [dig_plan_semantic_phase_name(phase) for phase in raw_phases]
    main_semantic = [phase for phase in semantic if phase != "clearance_route"]
    unknown = [
        phase
        for phase, sem in zip(raw_phases, semantic)
        if sem not in DATASET_PHASE_NAMES and sem != "clearance_route"
    ]

    reasons = []
    warnings = []
    if not seq:
        reasons.append("empty_sequence")
    if not main_semantic:
        reasons.append("empty_semantic_sequence")

    cursor = 0
    missing = []
    out_of_order = []
    for required in DIG_PLAN_REQUIRED_PHASE_ORDER:
        try:
            found = main_semantic.index(required, cursor)
            cursor = found + 1
        except ValueError:
            missing.append(required)
    if missing:
        reasons.append("missing_required_phases:" + ",".join(missing))

    required_positions = {
        phase: main_semantic.index(phase)
        for phase in DIG_PLAN_REQUIRED_PHASE_ORDER
        if phase in main_semantic
    }
    for a, b in zip(DIG_PLAN_REQUIRED_PHASE_ORDER[:-1], DIG_PLAN_REQUIRED_PHASE_ORDER[1:]):
        if a in required_positions and b in required_positions and required_positions[a] > required_positions[b]:
            out_of_order.append(f"{a}>{b}")
    if out_of_order:
        reasons.append("out_of_order:" + ",".join(out_of_order))

    if unknown:
        reasons.append("unknown_phases:" + ",".join(unknown[:6]))

    route_count = sum(1 for phase in semantic if phase == "clearance_route")
    unload_stage = None
    for stage in stages:
        if isinstance(stage, dict) and str(stage.get("phase", "")).startswith("unload"):
            unload_stage = stage
            break
    unload_dump_embedded = bool(isinstance(unload_stage, dict) and unload_stage.get("q_dump_deg") is not None)
    if not unload_dump_embedded:
        warnings.append("unload_dump_not_embedded_in_unload_stage")

    if points and seq and len(points) != len(seq):
        warnings.append(f"points_sequence_count_mismatch:{len(points)}!={len(seq)}")
    if not trace_points:
        warnings.append("empty_trace_points")

    blocked_direct_paths = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        path = stage.get("path")
        if not isinstance(path, dict):
            continue
        phase_ok = bool(path.get("phase_ok", True))
        obstacle_ok = bool(path.get("obstacle_ok", True))
        route_inserted = bool(path.get("route_inserted", False))
        if (not phase_ok or not obstacle_ok) and not route_inserted:
            blocked_direct_paths.append(
                {
                    "stage_index": int(idx),
                    "phase": str(stage.get("phase", "")),
                    "phase_ok": phase_ok,
                    "obstacle_ok": obstacle_ok,
                    "phase_reason": str(path.get("phase_reason", "")),
                    "obstacle_reason": str(path.get("obstacle_reason", "")),
                }
            )
    if blocked_direct_paths:
        warnings.append(f"blocked_direct_paths_without_route:{len(blocked_direct_paths)}")

    constraint_flags = []
    constraint_rows = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        summary = stage.get("constraint_summary")
        if not isinstance(summary, dict):
            summary = stage_constraint_summary(stage)
        flags = summary.get("flags", []) if isinstance(summary, dict) else []
        if flags:
            constraint_flags.extend(str(flag) for flag in flags)
            constraint_rows.append(
                {
                    "stage_index": int(idx),
                    "phase": str(stage.get("phase", "")),
                    "flags": [str(flag) for flag in flags],
                    "stage_cost": summary.get("stage_cost"),
                    "weighted_angle": summary.get("weighted_angle"),
                    "path_penalty": summary.get("path_penalty"),
                }
            )

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
        "required_order": list(DIG_PLAN_REQUIRED_PHASE_ORDER),
        "raw_phases": raw_phases,
        "semantic_phases": semantic,
        "main_semantic_phases": main_semantic,
        "missing": missing,
        "unknown": unknown,
        "route_count": int(route_count),
        "unload_dump_embedded": bool(unload_dump_embedded),
        "stage_count": int(len(seq)),
        "semantic_stage_count": int(len(main_semantic)),
        "point_count": int(len(points)),
        "trace_point_count": int(len(trace_points)),
        "blocked_direct_paths": blocked_direct_paths[:6],
        "constraint_flags": sorted(set(constraint_flags)),
        "constraint_flagged_stages": constraint_rows[:10],
    }


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

DATASET_PHASE_NAMES = [
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "pull_exit_cut",
    "curl_to_hold_material",
    "secure_load",
    "lift_carry",
    "unload_to_bin",
    "unload_dump",
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
        "auto_collect_candidate_plan_seconds": AUTO_COLLECT_CANDIDATE_PLAN_SECONDS,
        "auto_collect_candidate_hard_budget_grace_seconds": AUTO_COLLECT_CANDIDATE_HARD_BUDGET_GRACE_SECONDS,
        "auto_collect_find_plan_max_seconds": AUTO_COLLECT_FIND_PLAN_MAX_SECONDS,
        "planning_path_penalty_cache_max": PLANNING_PATH_PENALTY_CACHE_MAX,
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
            "auto_dig_min_swept_particles": AUTO_DIG_MIN_SWEPT_PARTICLES,
            "auto_dig_ring_radii": list(AUTO_DIG_RING_RADII),
        "auto_dig_ring_points": list(AUTO_DIG_RING_POINTS),
        "auto_dig_depth_priority": list(AUTO_DIG_DEPTH_PRIORITY),
        "auto_dig_sweep_radius": AUTO_DIG_SWEEP_RADIUS,
        "auto_dig_full_plan_topk_per_ring": AUTO_DIG_FULL_PLAN_TOPK_PER_RING,
        "auto_dig_score_weights": dict(AUTO_DIG_SCORE_WEIGHTS),
        "auto_unload_grid_size": AUTO_UNLOAD_GRID_SIZE,
        "path_rrt_max_iters": PATH_RRT_MAX_ITERS,
        "path_rrt_step_deg": PATH_RRT_STEP_DEG,
        "path_rrt_goal_bias": PATH_RRT_GOAL_BIAS,
        "path_rrt_joint_weights": list(PATH_RRT_JOINT_WEIGHTS),
        "path_rrt_smooth_rounds": PATH_RRT_SMOOTH_ROUNDS,
        "path_rrt_smooth_alpha": PATH_RRT_SMOOTH_ALPHA,
        "path_rrt_smooth_bend_weight": PATH_RRT_SMOOTH_BEND_WEIGHT,
        "path_rrt_smooth_min_improvement": PATH_RRT_SMOOTH_MIN_IMPROVEMENT,
        "path_link_collision_segment_samples": PATH_LINK_COLLISION_SEGMENT_SAMPLES,
        "path_link_collision_radius_m": PATH_LINK_COLLISION_RADIUS_M,
        "collision_world_policy": "rigid_hard_avoid__sand_soft_contact",
        "rigid_obstacle_paths": [str(x) for x in PATH_RIGID_OBSTACLE_PATHS],
        "obstacle_exclude_tokens": [str(x) for x in PATH_OBSTACLE_EXCLUDE_TOKENS],
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
        "global_plan_failure_limit": AUTO_COLLECT_GLOBAL_PLAN_FAILURE_LIMIT,
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
    api = get_sand_site_api()
    fn = api.get("particle_positions_fn") if isinstance(api, dict) else None
    if callable(fn):
        try:
            points = fn()
            if points is not None:
                arr = np.asarray(points, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
                    return np.ascontiguousarray(arr[:, :3], dtype=np.float32)
        except Exception:
            pass

    prim = sand_particle_prim()
    if prim is None:
        return None
    try:
        attr = prim.GetAttribute("points")
        points = attr.Get() if attr.IsValid() else None
        if points is None or len(points) == 0:
            return None
        try:
            arr = np.asarray(points, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return np.ascontiguousarray(arr[:, :3], dtype=np.float32)
        except Exception:
            pass
        return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)
    except Exception as e:
        info_print("[WARN] sand particle read failed:", type(e).__name__, e)
        return None


def invalidate_sand_runtime_caches(reason=""):
    STATE["sand_snapshot_last"] = None
    STATE["sand_snapshot_last_time"] = 0.0
    STATE["sand_metrics_last"] = None
    STATE["sand_metrics_last_time"] = 0.0
    if reason:
        STATE["sand_cache_invalidated_reason"] = str(reason)


def build_sand_spatial_index(points, ctx=None):
    if points is None or len(points) == 0:
        return None
    p = np.asarray(points, dtype=np.float32)
    center, radius, _floor_z, _fill_height, _z_min, _z_expected_max = sand_pile_geometry_from_context(ctx)
    x_min = float(center[0] - radius[0])
    x_max = float(center[0] + radius[0])
    y_min = float(center[1] - radius[1])
    y_max = float(center[1] + radius[1])
    if x_max <= x_min or y_max <= y_min:
        return None
    res = max(8, min(int(SAND_SNAPSHOT_GRID_MAX_RES), int(SAND_SNAPSHOT_GRID_RES)))
    ix = np.clip(((p[:, 0] - x_min) / max(1.0e-6, x_max - x_min) * res).astype(np.int32), 0, res - 1)
    iy = np.clip(((p[:, 1] - y_min) / max(1.0e-6, y_max - y_min) * res).astype(np.int32), 0, res - 1)
    flat = iy * res + ix
    order = np.argsort(flat, kind="mergesort")
    sorted_flat = flat[order]
    unique, starts, counts = np.unique(sorted_flat, return_index=True, return_counts=True)
    cell_slices = {int(cell): (int(start), int(start + count)) for cell, start, count in zip(unique, starts, counts)}
    counts_grid = np.zeros((res, res), dtype=np.int32)
    z_max_grid = np.full((res, res), np.nan, dtype=np.float32)
    np.add.at(counts_grid, (iy, ix), 1)
    for gx, gy, gz in zip(ix, iy, p[:, 2]):
        old = z_max_grid[gy, gx]
        if np.isnan(old) or float(gz) > float(old):
            z_max_grid[gy, gx] = float(gz)
    return {
        "points": p,
        "bbox": (x_min, x_max, y_min, y_max),
        "res": int(res),
        "flat": flat,
        "order": order,
        "sorted_flat": sorted_flat,
        "cell_slices": cell_slices,
        "counts_grid": counts_grid,
        "z_max_grid": z_max_grid,
    }


def sand_snapshot_local_indices(snapshot, x, y, radius):
    if not isinstance(snapshot, dict):
        return np.zeros(0, dtype=np.int64)
    index = snapshot.get("spatial_index")
    if not isinstance(index, dict):
        return np.zeros(0, dtype=np.int64)
    p = index.get("points")
    if p is None or len(p) == 0:
        return np.zeros(0, dtype=np.int64)
    x_min, x_max, y_min, y_max = index["bbox"]
    res = int(index["res"])
    if x_max <= x_min or y_max <= y_min:
        return np.zeros(0, dtype=np.int64)
    gx = int(np.clip((float(x) - x_min) / max(1.0e-6, x_max - x_min) * res, 0, res - 1))
    gy = int(np.clip((float(y) - y_min) / max(1.0e-6, y_max - y_min) * res, 0, res - 1))
    cell_w = max(1.0e-6, (x_max - x_min) / float(res))
    cell_h = max(1.0e-6, (y_max - y_min) / float(res))
    rx = min(int(SAND_SNAPSHOT_CELL_RADIUS_LIMIT), int(math.ceil(float(radius) / cell_w)) + 1)
    ry = min(int(SAND_SNAPSHOT_CELL_RADIUS_LIMIT), int(math.ceil(float(radius) / cell_h)) + 1)
    chunks = []
    order = index["order"]
    cell_slices = index["cell_slices"]
    for cy in range(max(0, gy - ry), min(res - 1, gy + ry) + 1):
        row = cy * res
        for cx in range(max(0, gx - rx), min(res - 1, gx + rx) + 1):
            sl = cell_slices.get(int(row + cx))
            if sl is None:
                continue
            chunks.append(order[sl[0]:sl[1]])
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def sand_snapshot_surface_height(snapshot, x, y, radius=None):
    if not isinstance(snapshot, dict):
        return None
    settled = snapshot.get("settled_points")
    if settled is None or len(settled) == 0:
        return None
    r = float(radius) if radius is not None else float(SAND_SURFACE_QUERY_RADIUS)
    idx = sand_snapshot_local_indices(snapshot, x, y, r)
    if idx is None or len(idx) == 0:
        return None
    p = settled[idx]
    d2 = (p[:, 0] - float(x)) ** 2 + (p[:, 1] - float(y)) ** 2
    near = p[d2 <= r * r]
    if len(near) == 0:
        return None
    _center, _radius, floor_z, _fill_height, _z_min, z_expected_max = sand_pile_geometry_from_context(snapshot.get("ctx"))
    z = float(np.percentile(near[:, 2], 90.0))
    return float(max(float(floor_z) + 0.02, min(z, float(z_expected_max))))


def sand_snapshot_density_count(snapshot, x, y, radius):
    if not isinstance(snapshot, dict):
        return 0
    settled = snapshot.get("settled_points")
    if settled is None or len(settled) == 0:
        return 0
    r = float(radius)
    idx = sand_snapshot_local_indices(snapshot, x, y, r)
    if idx is None or len(idx) == 0:
        return 0
    p = settled[idx]
    d2 = (p[:, 0] - float(x)) ** 2 + (p[:, 1] - float(y)) ** 2
    return int(np.count_nonzero(d2 <= r * r))


def get_sand_snapshot(force=False, label="", max_age=None):
    now = time.time()
    max_age = float(SAND_SNAPSHOT_MAX_AGE if max_age is None else max_age)
    if not force and bool(STATE.get("planning_sand_snapshot_active", False)):
        planning_snapshot = STATE.get("planning_sand_snapshot")
        if isinstance(planning_snapshot, dict):
            created_at = float(planning_snapshot.get("created_at", 0.0) or 0.0)
            if now - created_at <= max_age:
                return planning_snapshot
    cached = STATE.get("sand_snapshot_last")
    if (
        not force
        and isinstance(cached, dict)
        and now - float(STATE.get("sand_snapshot_last_time", 0.0) or 0.0) <= max_age
    ):
        return cached

    t0 = time.perf_counter()
    ctx = task_scene_context()
    points = sand_particle_positions()
    t_read = time.perf_counter()
    if points is None or len(points) == 0:
        snapshot = {
            "available": False,
            "reason": "missing_particle_points",
            "label": str(label),
            "created_at": now,
            "ctx": ctx,
            "points": None,
            "settled_points": None,
            "settle": {"ok": False, "reason": "missing_particles", "n": 0},
            "spatial_index": None,
            "perf_ms": {"read": (t_read - t0) * 1000.0, "total": (time.perf_counter() - t0) * 1000.0},
        }
        STATE["sand_snapshot_last"] = snapshot
        STATE["sand_snapshot_last_time"] = now
        return snapshot

    settle = sand_settle_status(points=points, ctx=ctx)
    settled_points = filter_settled_sand_particles(points, ctx=ctx) if bool(settle.get("ok", False)) else None
    t_filter = time.perf_counter()
    index = build_sand_spatial_index(settled_points, ctx=ctx) if settled_points is not None and len(settled_points) > 0 else None
    t_index = time.perf_counter()
    snapshot = {
        "available": True,
        "reason": "ok",
        "label": str(label),
        "created_at": now,
        "ctx": ctx,
        "points": np.asarray(points, dtype=np.float32),
        "particle_count": int(len(points)),
        "settle": settle,
        "settled_points": settled_points,
        "settled_count": int(0 if settled_points is None else len(settled_points)),
        "spatial_index": index,
        "perf_ms": {
            "read": (t_read - t0) * 1000.0,
            "filter": (t_filter - t_read) * 1000.0,
            "index": (t_index - t_filter) * 1000.0,
            "total": (t_index - t0) * 1000.0,
        },
    }
    STATE["sand_snapshot_last"] = snapshot
    STATE["sand_snapshot_last_time"] = now
    STATE["sand_perf_last"] = dict(snapshot["perf_ms"])
    return snapshot


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


async def wait_for_sand_settled_on_ground(label="sand_settle"):
    label = str(label)
    max_frames = max(int(SAND_RESET_SETTLE_MIN_FRAMES), int(SAND_RESET_SETTLE_MAX_FRAMES))
    window = max(15, int(SAND_RESET_STABLE_WINDOW_FRAMES))
    elapsed = 0
    stable_windows = 0
    last_status = None
    while elapsed < max_frames:
        if not timeline_allows_background_work():
            handle_timeline_stop_if_needed(label)
            return False, {"aborted": True, "reason": "timeline_stopped_or_runtime_stopped"}
        await step_updates(window)
        elapsed += window
        status = sand_settle_status()
        last_status = status
        if bool(status.get("ok", False)):
            stable_windows += 1
        else:
            stable_windows = 0
        info_print(
            "[SAND SETTLED CHECK]",
            f"label={label}",
            f"frames={elapsed}",
            f"ok={status.get('ok')}",
            f"stable_windows={stable_windows}",
            f"reason={status.get('reason')}",
            f"n={status.get('n')}",
            f"footprint={status.get('footprint_count')}",
            f"z_p90={fmt_optional(status.get('z_p90'))}",
            f"z_max={fmt_optional(status.get('z_max'))}",
            f"fill={fmt_optional(status.get('fill_height'))}",
        )
        if stable_windows >= 2:
            return True, status
    return False, last_status


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
                    ok, stats = await wait_for_sand_settled_on_ground(f"{label}_native_settle{attempt}")
                    if ok:
                        stats = dict(stats or {})
                        stats["native_stable_reset"] = True
                        break
                    if attempt < max_attempts:
                        info_print("[SAND RESET RETRY]", f"label={label}", f"attempt={attempt}", f"reason={stats}")
                        await step_updates(30)
                        continue
                if not native_ok and attempt < max_attempts:
                    info_print("[SAND RESET RETRY]", f"label={label}", f"attempt={attempt}", "reason=native_stable_reset_failed")
                    await step_updates(30)
                    continue
            else:
                info_print("[SAND RESET NATIVE]", f"label={label}", f"attempt={attempt}/{max_attempts}", "calling=sand_site.reset")
                reset_fn()

            ok, stats = await wait_for_sand_particles_stable(f"{label}_attempt{attempt}")
            if ok:
                settled_ok, settled_stats = await wait_for_sand_settled_on_ground(f"{label}_settled_attempt{attempt}")
                ok = bool(settled_ok)
                stats = dict(settled_stats or stats or {})
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
        invalidate_sand_runtime_caches(f"sand_reset:{label}")
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


def initial_pile_mask_for_points(points, fallback_pile_mask=None):
    n = 0 if points is None else int(len(points))
    if n <= 0:
        return np.zeros(0, dtype=bool)
    initial_mask = STATE.get("dataset_initial_pile_particle_mask")
    if isinstance(initial_mask, np.ndarray) and len(initial_mask) == n:
        return np.array(initial_mask, dtype=bool, copy=False)

    initial_ids = STATE.get("dataset_initial_pile_particle_ids")
    if initial_ids is not None:
        mask = np.zeros(n, dtype=bool)
        try:
            idx = np.array(list(initial_ids), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < n)]
            mask[idx] = True
            return mask
        except Exception:
            pass

    if fallback_pile_mask is not None and len(fallback_pile_mask) == n:
        return np.array(fallback_pile_mask, dtype=bool, copy=True)
    return np.zeros(n, dtype=bool)


def bucket_particle_diagnostic(label, points=None, force_log=True):
    points = sand_particle_positions() if points is None else points
    if points is None or len(points) == 0:
        row = {"available": False, "reason": "missing_particle_points", "label": str(label)}
        info_print("[BUCKET SAND DIAG]", f"label={label}", "available=False", "reason=missing_particle_points", force_log=force_log)
        return row

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    local = project_points_to_link_local(points, BUCKET_LINK)
    if local is None or len(local) != len(points):
        row = {"available": False, "reason": "bucket_local_projection_failed", "label": str(label), "particles": int(len(points))}
        info_print(
            "[BUCKET SAND DIAG]",
            f"label={label}",
            "available=False",
            "reason=bucket_local_projection_failed",
            f"particles={len(points)}",
            force_log=force_log,
        )
        return row

    ctx = task_scene_context()
    _, _, _, _, z_min, z_expected_max = sand_pile_geometry_from_context(ctx)
    pile_mask = sand_pile_xy_mask(points, ctx) & (points[:, 2] >= z_min) & (points[:, 2] <= z_expected_max)
    initial_mask = initial_pile_mask_for_points(points, fallback_pile_mask=pile_mask)
    strict_mask = np.all(local >= SAND_BUCKET_LOCAL_MIN.reshape(1, 3), axis=1) & np.all(
        local <= SAND_BUCKET_LOCAL_MAX.reshape(1, 3),
        axis=1,
    )
    expand = SAND_BUCKET_DIAG_EXPAND_LOCAL.reshape(3)
    expanded_min = SAND_BUCKET_LOCAL_MIN - expand
    expanded_max = SAND_BUCKET_LOCAL_MAX + expand
    expanded_mask = np.all(local >= expanded_min.reshape(1, 3), axis=1) & np.all(
        local <= expanded_max.reshape(1, 3),
        axis=1,
    )

    strict_from_pile = initial_mask & strict_mask
    expanded_from_pile = initial_mask & expanded_mask
    expanded_idx = np.nonzero(expanded_from_pile)[0]
    strict_idx = np.nonzero(strict_from_pile)[0]
    expanded_all_idx = np.nonzero(expanded_mask)[0]
    strict_all_idx = np.nonzero(strict_mask)[0]
    expanded_local_min = None
    expanded_local_max = None
    expanded_all_local_min = None
    expanded_all_local_max = None
    expanded_world_center = None
    strict_world_center = None
    expanded_all_world_center = None
    strict_all_world_center = None
    if len(expanded_idx) > 0:
        expanded_local = local[expanded_idx]
        expanded_local_min = vec_list(np.min(expanded_local, axis=0), 3)
        expanded_local_max = vec_list(np.max(expanded_local, axis=0), 3)
        expanded_world_center = vec_list(np.mean(points[expanded_idx], axis=0), 3)
    if len(strict_idx) > 0:
        strict_world_center = vec_list(np.mean(points[strict_idx], axis=0), 3)
    if len(expanded_all_idx) > 0:
        expanded_all_local = local[expanded_all_idx]
        expanded_all_local_min = vec_list(np.min(expanded_all_local, axis=0), 3)
        expanded_all_local_max = vec_list(np.max(expanded_all_local, axis=0), 3)
        expanded_all_world_center = vec_list(np.mean(points[expanded_all_idx], axis=0), 3)
    if len(strict_all_idx) > 0:
        strict_all_world_center = vec_list(np.mean(points[strict_all_idx], axis=0), 3)

    row = {
        "available": True,
        "label": str(label),
        "particles": int(len(points)),
        "strict_total": int(np.count_nonzero(strict_mask)),
        "strict_from_pile": int(np.count_nonzero(strict_from_pile)),
        "expanded_total": int(np.count_nonzero(expanded_mask)),
        "expanded_from_pile": int(np.count_nonzero(expanded_from_pile)),
        "expanded_local_min": expanded_local_min,
        "expanded_local_max": expanded_local_max,
        "expanded_all_local_min": expanded_all_local_min,
        "expanded_all_local_max": expanded_all_local_max,
        "expanded_world_center": expanded_world_center,
        "strict_world_center": strict_world_center,
        "expanded_all_world_center": expanded_all_world_center,
        "strict_all_world_center": strict_all_world_center,
        "strict_box_min": vec_list(SAND_BUCKET_LOCAL_MIN, 3),
        "strict_box_max": vec_list(SAND_BUCKET_LOCAL_MAX, 3),
        "expanded_box_min": vec_list(expanded_min, 3),
        "expanded_box_max": vec_list(expanded_max, 3),
        "q_real_deg": q_deg_values(get_real_joint_positions(), wrap_swing_for_display=True),
    }
    info_print(
        "[BUCKET SAND DIAG]",
        f"label={label}",
        f"particles={row['particles']}",
        f"strict_total={row['strict_total']}",
        f"strict_from_pile={row['strict_from_pile']}",
        f"expanded_total={row['expanded_total']}",
        f"expanded_from_pile={row['expanded_from_pile']}",
        f"expanded_local_min={row['expanded_local_min']}",
        f"expanded_local_max={row['expanded_local_max']}",
        f"expanded_all_local_min={row['expanded_all_local_min']}",
        f"expanded_all_local_max={row['expanded_all_local_max']}",
        f"strict_center={row['strict_world_center']}",
        f"expanded_center={row['expanded_world_center']}",
        f"strict_all_center={row['strict_all_world_center']}",
        f"expanded_all_center={row['expanded_all_world_center']}",
        f"q_real={row['q_real_deg']}",
        force_log=force_log,
    )
    return row


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


def sand_pile_geometry_from_context(ctx=None):
    ctx = task_scene_context() if ctx is None else ctx
    center = np.array(ctx.get("pile_center", SAND_PILE_CENTER), dtype=np.float32).reshape(-1)[:3]
    radius = np.array(ctx.get("pile_radius", [SAND_PILE_RADIUS_X, SAND_PILE_RADIUS_Y]), dtype=np.float32).reshape(-1)[:2]
    radius = np.maximum(radius, np.array([0.05, 0.05], dtype=np.float32))
    floor_z = float(ctx.get("sand_floor_z", GROUND_TOP_Z))
    fill_height = max(0.05, float(ctx.get("sand_fill_height", SAND_PILE_Z_MAX - SAND_PILE_Z_MIN)))
    z_min = min(float(SAND_PILE_Z_MIN), floor_z - 0.10)
    z_expected_max = floor_z + fill_height + float(SAND_SETTLED_Z_MARGIN)
    return center, radius, floor_z, fill_height, z_min, z_expected_max


def sand_pile_xy_mask(points, ctx=None):
    n = 0 if points is None else int(len(points))
    if points is None or n == 0:
        return np.zeros(n, dtype=bool)
    center, radius, _, _, _, _ = sand_pile_geometry_from_context(ctx)
    dx = (points[:, 0] - center[0]) / max(1e-5, float(radius[0]))
    dy = (points[:, 1] - center[1]) / max(1e-5, float(radius[1]))
    return dx * dx + dy * dy <= 1.0


def sand_settle_status(points=None, ctx=None):
    points = sand_particle_positions() if points is None else points
    if points is None or len(points) == 0:
        return {"ok": False, "reason": "missing_particles", "n": 0}
    p = np.array(points, dtype=np.float32)
    ctx = task_scene_context() if ctx is None else ctx
    _, _, floor_z, fill_height, z_min, z_expected_max = sand_pile_geometry_from_context(ctx)
    xy_mask = sand_pile_xy_mask(p, ctx)
    footprint = p[xy_mask]
    if len(footprint) == 0:
        return {
            "ok": False,
            "reason": "no_particles_in_sand_footprint",
            "n": int(len(p)),
            "footprint_count": 0,
        }
    high_air_z = floor_z + fill_height + float(SAND_SETTLED_Z_MARGIN)
    settled_z_max = floor_z + fill_height + float(SAND_SETTLED_SURFACE_MARGIN)
    in_expected = footprint[(footprint[:, 2] >= z_min) & (footprint[:, 2] <= high_air_z)]
    footprint_fraction = float(len(footprint)) / max(1.0, float(len(p)))
    high_air_fraction = float(np.count_nonzero(footprint[:, 2] > high_air_z)) / max(1.0, float(len(footprint)))
    p50 = float(np.percentile(footprint[:, 2], 50.0))
    p90 = float(np.percentile(footprint[:, 2], 90.0))
    p95 = float(np.percentile(footprint[:, 2], 95.0))
    z_max = float(np.max(footprint[:, 2]))
    ok = (
        footprint_fraction >= float(SAND_SETTLED_MIN_FRACTION_IN_FOOTPRINT)
        and high_air_fraction <= float(SAND_SETTLED_HIGH_AIR_FRACTION_MAX)
        and p90 <= settled_z_max
        and len(in_expected) >= int(AUTO_PREFLIGHT_MIN_PARTICLES)
    )
    reason = "ok" if ok else (
        f"not_settled footprint_fraction={footprint_fraction:.3f} "
        f"high_air_fraction={high_air_fraction:.3f} p90={p90:.3f} "
        f"limit={settled_z_max:.3f}"
    )
    return {
        "ok": bool(ok),
        "reason": reason,
        "n": int(len(p)),
        "footprint_count": int(len(footprint)),
        "expected_count": int(len(in_expected)),
        "footprint_fraction": footprint_fraction,
        "high_air_fraction": high_air_fraction,
        "floor_z": float(floor_z),
        "fill_height": float(fill_height),
        "z_expected_max": float(z_expected_max),
        "z_p50": p50,
        "z_p90": p90,
        "z_p95": p95,
        "z_max": z_max,
    }


def filter_settled_sand_particles(points, ctx=None):
    if points is None or len(points) == 0:
        return None
    p = np.array(points, dtype=np.float32)
    ctx = task_scene_context() if ctx is None else ctx
    _, _, _, _, z_min, z_expected_max = sand_pile_geometry_from_context(ctx)
    mask = sand_pile_xy_mask(p, ctx) & (p[:, 2] >= z_min) & (p[:, 2] <= z_expected_max)
    return p[mask]


def sand_region_masks(points):
    n = 0 if points is None else int(len(points))
    empty = np.zeros(n, dtype=bool)
    if points is None or n == 0:
        return empty, empty, empty

    ctx = task_scene_context()
    _, _, _, _, z_min, z_expected_max = sand_pile_geometry_from_context(ctx)
    pile_mask = sand_pile_xy_mask(points, ctx) & (points[:, 2] >= z_min) & (points[:, 2] <= z_expected_max)

    bucket_local = project_points_to_link_local(points, BUCKET_LINK)
    if bucket_local is None:
        bucket_mask = empty.copy()
    else:
        bucket_mask = np.all(bucket_local >= SAND_BUCKET_LOCAL_MIN.reshape(1, 3), axis=1) & np.all(
            bucket_local <= SAND_BUCKET_LOCAL_MAX.reshape(1, 3), axis=1
        )

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
    snapshot = get_sand_snapshot(force=True, label="capture_initial_pile")
    points = snapshot.get("points") if isinstance(snapshot, dict) else sand_particle_positions()
    pile_mask, _, _ = sand_region_masks(points)
    ids = particle_ids_from_mask(pile_mask)
    STATE["dataset_initial_pile_particle_ids"] = ids
    STATE["dataset_initial_pile_particle_mask"] = np.array(pile_mask, dtype=bool).copy()
    STATE["dataset_initial_pile_particle_count"] = len(ids)
    return ids


def sand_metrics_current(force=False, snapshot=None):
    now = time.time()
    t0 = time.perf_counter()
    if (
        not force
        and STATE.get("sand_metrics_last") is not None
        and now - float(STATE.get("sand_metrics_last_time", 0.0)) < SAND_METRICS_INTERVAL
    ):
        return dict(STATE["sand_metrics_last"])

    if isinstance(snapshot, dict):
        points = snapshot.get("points")
    else:
        points = sand_particle_positions()
    if points is None:
        metrics = {
            "available": False,
            "particle_count": 0,
            "reason": "missing_particle_points",
        }
        STATE["sand_metrics_last"] = metrics
        STATE["sand_metrics_last_time"] = now
        perf = dict(STATE.get("sand_perf_last", {}) or {})
        perf["metrics_ms"] = (time.perf_counter() - t0) * 1000.0
        perf["metrics_particles"] = 0
        STATE["sand_perf_last"] = perf
        return metrics

    pile_mask, bucket_mask, bin_mask = sand_region_masks(points)

    initial_ids = STATE.get("dataset_initial_pile_particle_ids")
    initial_mask = STATE.get("dataset_initial_pile_particle_mask")
    if isinstance(initial_mask, np.ndarray) and len(initial_mask) == len(points):
        initial_mask = np.array(initial_mask, dtype=bool, copy=False)
    elif initial_ids is not None:
        initial_mask = np.zeros(len(points), dtype=bool)
        try:
            idx = np.array(list(initial_ids), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(points))]
            initial_mask[idx] = True
        except Exception:
            initial_mask = np.array(pile_mask, dtype=bool).copy()
    else:
        initial_mask = np.array(pile_mask, dtype=bool).copy()
        initial_ids = particle_ids_from_mask(pile_mask)

    from_pile_bucket_mask = initial_mask & bucket_mask
    from_pile_bin_mask = initial_mask & bin_mask
    from_pile_pile_mask = initial_mask & pile_mask
    from_pile_spill_mask = initial_mask & ~(bucket_mask | bin_mask | pile_mask)
    pile_count = int(np.count_nonzero(pile_mask))
    bucket_count = int(np.count_nonzero(bucket_mask))
    bin_count = int(np.count_nonzero(bin_mask))
    initial_count = int(np.count_nonzero(initial_mask))
    raw_from_pile_bucket_count = int(np.count_nonzero(from_pile_bucket_mask))
    raw_from_pile_bin_count = int(np.count_nonzero(from_pile_bin_mask))
    raw_from_pile_pile_count = int(np.count_nonzero(from_pile_pile_mask))
    raw_from_pile_spill_count = int(np.count_nonzero(from_pile_spill_mask))

    source_tracking = "initial_mask"
    fallback_notes = []

    def effective_region_count(region_name, raw_count, region_count):
        nonlocal source_tracking
        raw_count = int(raw_count)
        region_count = int(region_count)
        if region_count < int(SAND_SOURCE_FALLBACK_MIN_REGION_COUNT):
            return raw_count
        ratio = float(raw_count) / max(1.0, float(region_count))
        if ratio >= float(SAND_SOURCE_FALLBACK_MIN_RATIO):
            return raw_count
        source_tracking = "region_fallback"
        fallback_notes.append(f"{region_name}:raw={raw_count}/region={region_count}/ratio={ratio:.2f}")
        return region_count

    from_pile_bucket_count = effective_region_count("bucket", raw_from_pile_bucket_count, bucket_count)
    from_pile_bin_count = effective_region_count("bin", raw_from_pile_bin_count, bin_count)
    from_pile_pile_count = raw_from_pile_pile_count
    if source_tracking == "region_fallback":
        occupied = bucket_mask | bin_mask | pile_mask
        from_pile_spill_count = int(max(0, len(points) - int(np.count_nonzero(occupied))))
    else:
        from_pile_spill_count = raw_from_pile_spill_count
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
        "initial_pile_count": int(initial_count),
        "pile_count": int(pile_count),
        "bucket_count": int(bucket_count),
        "bucket_from_pile_count": int(from_pile_bucket_count),
        "bucket_from_initial_count": int(raw_from_pile_bucket_count),
        "bin_count": int(bin_count),
        "bin_from_pile_count": int(from_pile_bin_count),
        "bin_from_initial_count": int(raw_from_pile_bin_count),
        "pile_from_initial_count": int(from_pile_pile_count),
        "spill_from_pile_count": int(from_pile_spill_count),
        "spill_from_initial_count": int(raw_from_pile_spill_count),
        "source_tracking": str(source_tracking),
        "source_tracking_notes": "; ".join(fallback_notes),
        "bucket_from_pile_mass": float(from_pile_bucket_count * mass),
        "bin_from_pile_mass": float(from_pile_bin_count * mass),
        "spill_from_pile_mass": float(from_pile_spill_count * mass),
    }
    STATE["sand_metrics_last"] = metrics
    STATE["sand_metrics_last_time"] = now
    perf = dict(STATE.get("sand_perf_last", {}) or {})
    perf["metrics_ms"] = (time.perf_counter() - t0) * 1000.0
    perf["metrics_particles"] = int(len(points))
    STATE["sand_perf_last"] = perf
    return dict(metrics)


def is_sand_contact_phase(mode):
    m = str(mode).lower()
    return any(phase in m for phase in SAND_CONTACT_PHASES)


def is_sand_cut_geometry_phase(mode):
    m = str(mode).lower()
    return any(phase in m for phase in SAND_CUT_GEOMETRY_PHASES)


def phase_collision_context(mode):
    m = str(mode).lower()
    if is_sand_contact_phase(m) or "curl_to_hold_material" in m:
        return {
            "phase_class": "material_interaction",
            "rigid_policy": "hard_avoid",
            "sand_policy": "soft_contact_progress",
            "strict_path_precheck": False,
        }
    if "dump" in m:
        return {
            "phase_class": "unload_release",
            "rigid_policy": "hard_avoid",
            "sand_policy": "ballistic_release_scored",
            "strict_path_precheck": False,
        }
    return {
        "phase_class": "rigid_free_space",
        "rigid_policy": "hard_avoid",
        "sand_policy": "ignored_for_collision",
        "strict_path_precheck": True,
    }


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


def secure_hold_spill_report(stage_name, report, bucket_loaded=None):
    stage_l = str(stage_name).lower()
    if "curl_to_hold_material" not in stage_l and "secure_load" not in stage_l:
        return {
            "ok": True,
            "reason": "not_secure_hold_stage",
        }
    report = report if isinstance(report, dict) else {}
    bucket_total = int(report.get("total_bucket_delta", 0) or 0)
    spill_total = max(0, int(report.get("total_spill_delta", 0) or 0))
    current_bucket = int(report.get("bucket", 0) or 0)
    if bucket_loaded is None:
        bucket_loaded = current_bucket
    bucket_loaded = max(0, int(bucket_loaded or 0))
    bucket_loss = max(0, -bucket_total)
    spill_limit = max(
        int(SECURE_HOLD_MAX_SPILL_PARTICLES),
        int(float(bucket_loaded) * float(SECURE_HOLD_MAX_SPILL_FRACTION)),
    )
    loss_limit = max(
        int(CURL_HOLD_MIN_BUCKET_PARTICLES),
        int(float(max(bucket_loaded, current_bucket)) * float(SECURE_HOLD_MAX_BUCKET_LOSS_FRACTION)),
    )
    moved = max(1, max(0, bucket_total) + spill_total)
    spill_ratio = float(spill_total) / float(moved)
    ok = spill_total <= spill_limit and bucket_loss <= loss_limit
    cut_report = secure_phase_delta_report(current_metrics=None)
    cut_ok = True
    if isinstance(cut_report, dict) and cut_report.get("baseline") == "after_cut":
        cut_ok = bool(cut_report.get("ok", True))
        if not cut_ok:
            ok = False
    reason = "ok" if ok else (
        f"secure_spill_or_loss_too_high spill={spill_total}/{spill_limit} "
        f"bucket_loss={bucket_loss}/{loss_limit} spill_ratio={spill_ratio:.2f}"
    )
    if not ok and not cut_ok:
        reason = str(cut_report.get("reason", reason))
    return {
        "ok": bool(ok),
        "reason": reason,
        "bucket_total_delta": int(bucket_total),
        "bucket_loss": int(bucket_loss),
        "bucket_loss_limit": int(loss_limit),
        "spill_total_delta": int(spill_total),
        "spill_limit": int(spill_limit),
        "spill_ratio": float(spill_ratio),
        "bucket_loaded": int(bucket_loaded),
        "current_bucket": int(current_bucket),
        "cut_baseline": cut_report,
    }


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
    if "curl_to_hold_material" in stage_name or "secure_load" in stage_name:
        bucket_idx = CTRL.name_to_idx.get("bucket", 3)
        bucket_real_deg = rad_to_deg(float(q_real[bucket_idx]))
        retain_report = carry_material_report_for_q(q_real, end_effector="load")
        retains_material = bool(retain_report.get("retains_material", False))
        metrics = sand_metrics_current(force=True)
        bucket_loaded = int(metrics.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics, dict) else 0
        bucket_err = float(err_deg[bucket_idx]) if len(err_deg) > bucket_idx else float(max_err)
        real_loaded_hold = real_loaded_secure_hold_allowed(retain_report, loaded_count=bucket_loaded)
        if (
            bucket_loaded >= CURL_HOLD_MIN_BUCKET_PARTICLES
            and (retains_material or real_loaded_hold)
            and bucket_err <= CURL_HOLD_ACCEPT_MAX_ERR_DEG
        ):
            hold_spill = secure_hold_spill_report(stage_name, report, bucket_loaded=bucket_loaded)
            if not bool(hold_spill.get("ok", False)):
                info_print(
                    "[SECURE HOLD WAIT]",
                    f"stage={stage_name}",
                    hold_spill.get("reason", "secure hold spill gate failed"),
                    f"bucket_loaded={bucket_loaded}",
                )
                return False
            info_print(
                "[CURL HOLD DONE]",
                f"stage={stage_name}",
                f"elapsed={elapsed:.2f}s",
                f"bucket_real={bucket_real_deg:.2f}deg",
                f"bucket_err={bucket_err:.2f}deg",
                f"bucket_loaded={bucket_loaded}",
                f"pour_above_load_z={fmt_optional(retain_report.get('pour_above_load_z'))}",
                "reason=loaded_bucket_retaining_geometry" if retains_material else "reason=real_loaded_bucket_hold",
            )
            return True

    bad_cut, _bad_reason = sand_contact_bad_cut_geometry(stage_name, report)
    if bad_cut:
        return False
    bucket_total = int(report.get("total_bucket_delta", 0))
    pile_total = int(report.get("total_pile_delta", 0))
    spill_total = int(report.get("total_spill_delta", 0))
    current_bucket = int(report.get("bucket", 0) or 0)
    moved_material = max(1, bucket_total + spill_total)
    spill_ratio = float(spill_total) / float(moved_material)
    exit_loaded = "pull_exit_cut" in str(stage_name) and current_bucket >= CURL_HOLD_MIN_BUCKET_PARTICLES
    material_progress = (
        bucket_total >= SAND_CONTACT_BUCKET_PROGRESS_MIN
        or (pile_total >= SAND_CONTACT_PILE_PROGRESS_MIN and spill_ratio <= SAND_CONTACT_SPILL_RATIO_MAX)
    )
    tip_progress = float(report.get("total_tip_delta", 0.0) or 0.0) >= SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M
    q_close_enough = float(max_err) <= SAND_CONTACT_Q_LAG_ACCEPT_DEG
    if material_progress or (exit_loaded and (tip_progress or q_close_enough)) or (tip_progress and q_close_enough) or q_close_enough:
        info_print(
            "[SAND CONTACT DONE]",
            f"stage={stage_name}",
            f"elapsed={elapsed:.2f}s",
            f"max_err={max_err:.2f}deg",
            f"bucket_now={current_bucket}",
            f"bucket_total=+{int(report.get('total_bucket_delta', 0))}",
            f"pile_total=-{int(report.get('total_pile_delta', 0))}",
            f"tip_total={float(report.get('total_tip_delta', 0.0) or 0.0):.3f}m",
            "reason=loaded_exit_or_material_progress",
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
    current_bucket = int(report.get("bucket", 0) or 0)
    tip_total = float(report.get("total_tip_delta", 0.0) or 0.0)
    if "curl_to_hold_material" in str(stage_name) or "secure_load" in str(stage_name):
        try:
            bucket_idx = CTRL.name_to_idx.get("bucket", 3)
            bucket_real_deg = rad_to_deg(float(q_real[bucket_idx])) if q_real is not None else 999.0
            bucket_err = abs(rad_to_deg(wrap_angle(float(q_cmd[bucket_idx]) - float(q_real[bucket_idx])))) if q_cmd is not None and q_real is not None else max_err
            retain_report = carry_material_report_for_q(q_real if q_real is not None else q_cmd, end_effector="load")
            retains_material = bool(retain_report.get("retains_material", False))
        except Exception:
            bucket_real_deg = 999.0
            bucket_err = max_err
            retain_report = {}
            retains_material = False
        metrics = sand_metrics_current(force=True)
        bucket_loaded = int(metrics.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics, dict) else 0
        real_loaded_hold = real_loaded_secure_hold_allowed(retain_report, loaded_count=bucket_loaded)
        if (
            bucket_loaded >= CURL_HOLD_MIN_BUCKET_PARTICLES
            and (retains_material or real_loaded_hold)
            and bucket_err <= CURL_HOLD_ACCEPT_MAX_ERR_DEG
        ):
            hold_spill = secure_hold_spill_report(stage_name, report, bucket_loaded=bucket_loaded)
            if not bool(hold_spill.get("ok", False)):
                return False, str(hold_spill.get("reason", "secure hold spill gate failed"))
            return True, (
                f"loaded_bucket_retaining_geometry elapsed={elapsed:.2f}s "
                f"bucket_real={bucket_real_deg:.2f}deg bucket_err={bucket_err:.2f}deg "
                f"bucket_loaded={bucket_loaded} "
                f"pour_above_load_z={fmt_optional(retain_report.get('pour_above_load_z'))} "
                f"hold_source={'geometry' if retains_material else 'real_loaded'}"
            )

    bad_cut, _bad_reason = sand_contact_bad_cut_geometry(stage_name, report)
    if bad_cut:
        return False, ""
    moved_material = max(1, bucket_total + spill_total)
    spill_ratio = float(spill_total) / float(moved_material)
    exit_loaded = "pull_exit_cut" in str(stage_name) and current_bucket >= CURL_HOLD_MIN_BUCKET_PARTICLES

    if (
        bucket_total >= SAND_CONTACT_ADVANCE_BUCKET_MIN
        or (pile_total >= SAND_CONTACT_ADVANCE_PILE_MIN and spill_ratio <= SAND_CONTACT_SPILL_RATIO_MAX)
        or (tip_total >= SAND_CONTACT_ADVANCE_TIP_MIN_M and bucket_total >= SAND_CONTACT_BUCKET_PROGRESS_MIN)
        or (exit_loaded and tip_total >= SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M)
    ):
        return True, (
            f"loaded_exit_or_material_progress elapsed={elapsed:.2f}s bucket_now={current_bucket} bucket={bucket_total} "
            f"pile={pile_total} tip={tip_total:.3f}m spill={spill_total} spill_ratio={spill_ratio:.2f} "
            f"max_err={max_err:.2f}deg boom_err={boom_err:.2f}deg"
        )

    if elapsed >= SAND_CONTACT_MAX_STAGE_WALL_SECONDS and (
        bucket_total >= SAND_CONTACT_ADVANCE_FALLBACK_BUCKET_MIN
        or (pile_total >= SAND_CONTACT_ADVANCE_FALLBACK_PILE_MIN and spill_ratio <= SAND_CONTACT_SPILL_RATIO_MAX)
        or (tip_total >= SAND_CONTACT_ACCEPT_TIP_PROGRESS_MIN_M and bucket_total >= SAND_CONTACT_BUCKET_PROGRESS_MIN)
    ):
        return True, (
            f"stage_time_cap elapsed={elapsed:.2f}s bucket={bucket_total} "
            f"pile={pile_total} tip={tip_total:.3f}m spill={spill_total} spill_ratio={spill_ratio:.2f} "
            f"max_err={max_err:.2f}deg boom_err={boom_err:.2f}deg"
        )

    return False, ""


def sand_contact_bad_cut_geometry(stage_name, report):
    if not is_sand_contact_phase(stage_name) or not isinstance(report, dict):
        return False, ""
    if not is_sand_cut_geometry_phase(stage_name):
        return False, ""
    bucket_total = int(report.get("total_bucket_delta", 0) or 0)
    pile_total = int(report.get("total_pile_delta", 0) or 0)
    spill_total = int(report.get("total_spill_delta", 0) or 0)
    current_bucket = int(report.get("bucket", 0) or 0)
    if current_bucket >= max(int(CURL_HOLD_MIN_BUCKET_PARTICLES), int(SAND_CONTACT_BUCKET_PROGRESS_MIN)):
        return False, ""
    if spill_total < SAND_CONTACT_SPILL_WITHOUT_LOAD_MIN:
        return False, ""
    moved_material = max(1, bucket_total + spill_total)
    spill_ratio = float(spill_total) / float(moved_material)
    if "pull_exit_cut" in str(stage_name):
        if current_bucket >= CURL_HOLD_MIN_BUCKET_PARTICLES:
            return False, ""
    if bucket_total <= max(8, int(SAND_CONTACT_BUCKET_PROGRESS_MIN * 0.25)) and spill_ratio > SAND_CONTACT_SPILL_RATIO_MAX:
        return True, (
            f"spilling_without_loading bucket={bucket_total} bucket_now={current_bucket} pile={pile_total} "
            f"spill={spill_total} spill_ratio={spill_ratio:.2f}"
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
    dq = dataset_q_delta(q_cmd, q_prev)
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
    dq = dataset_q_delta(q_real, q_prev)
    STATE["dataset_last_q_real"] = q_real.copy()
    return [float(x) for x in (dq / dt)]


def dataset_motion_derivatives(q_cmd, q_real, now):
    q_cmd = np.array(q_cmd, dtype=np.float32).reshape(-1)[:4]
    q_real = np.array(q_real, dtype=np.float32).reshape(-1)[:4]
    t_prev = float(STATE.get("dataset_last_sample_time", 0.0) or 0.0)
    dt = max(1.0e-4, float(now) - t_prev) if t_prev > 0.0 else 0.0

    last_q_cmd = STATE.get("dataset_last_q_cmd")
    last_q_real = STATE.get("dataset_last_q_real")
    last_action = STATE.get("dataset_last_action")
    last_dq_real = STATE.get("dataset_last_dq_real")
    last_ddq_real = STATE.get("dataset_last_ddq_real")

    if last_q_cmd is None or dt <= 0.0:
        action = np.zeros(4, dtype=np.float32)
    else:
        action = dataset_q_delta(q_cmd, np.array(last_q_cmd, dtype=np.float32)) / dt

    if last_q_real is None or dt <= 0.0:
        dq_real = np.zeros(4, dtype=np.float32)
    else:
        dq_real = dataset_q_delta(q_real, np.array(last_q_real, dtype=np.float32)) / dt

    if last_dq_real is None or dt <= 0.0:
        ddq_real = np.zeros(4, dtype=np.float32)
    else:
        ddq_real = (dq_real - np.array(last_dq_real, dtype=np.float32).reshape(-1)[:4]) / dt

    if last_action is None or dt <= 0.0:
        action_ddq = np.zeros(4, dtype=np.float32)
    else:
        action_ddq = (action - np.array(last_action, dtype=np.float32).reshape(-1)[:4]) / dt

    if last_ddq_real is None or dt <= 0.0:
        jerk_real = np.zeros(4, dtype=np.float32)
    else:
        jerk_real = (ddq_real - np.array(last_ddq_real, dtype=np.float32).reshape(-1)[:4]) / dt

    STATE["dataset_last_q_cmd"] = q_cmd.copy()
    STATE["dataset_last_q_real"] = q_real.copy()
    STATE["dataset_last_action"] = action.copy()
    STATE["dataset_last_dq_real"] = dq_real.copy()
    STATE["dataset_last_ddq_real"] = ddq_real.copy()

    return {
        "dt": float(dt),
        "action": [float(x) for x in action],
        "dq_real": [float(x) for x in dq_real],
        "ddq_real": [float(x) for x in ddq_real],
        "action_ddq": [float(x) for x in action_ddq],
        "jerk_real": [float(x) for x in jerk_real],
        "speed_l2": float(np.linalg.norm(dq_real)),
        "accel_l2": float(np.linalg.norm(ddq_real)),
        "jerk_l2": float(np.linalg.norm(jerk_real)),
        "command_accel_l2": float(np.linalg.norm(action_ddq)),
        "energy_proxy": float(np.dot(np.abs(action), np.abs(action))),
    }


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


def dataset_phase_index(phase):
    text = str(phase).lower()
    if "pre_dig" in text or "travel" in text or "align" in text:
        return 0
    if "approach_contact" in text or ("approach" in text and "contact" in text):
        return 1
    if "insert" in text:
        return 2
    if "pull_mid" in text:
        return 3
    if "pull_exit" in text:
        return 4
    if "curl" in text:
        return 5
    if "secure_load" in text or text == "secure" or "secure" in text:
        return 6
    if "lift_carry" in text or "lift" in text or "carry" in text:
        return 7
    if "unload_dump" in text or ("dump" in text and "unload" in text):
        return 9
    if "unload_to_bin" in text or "unload" in text or "dump" in text:
        return 8
    return -1


def dataset_phase_features(phase):
    idx = dataset_phase_index(phase)
    one_hot = [0] * len(DATASET_PHASE_NAMES)
    if 0 <= idx < len(one_hot):
        one_hot[idx] = 1
    return {
        "name": DATASET_PHASE_NAMES[idx] if 0 <= idx < len(DATASET_PHASE_NAMES) else str(phase),
        "index": int(idx),
        "one_hot": one_hot,
        "context": phase_collision_context(phase),
    }


def point_aabb_signed_distance(point, mn, mx):
    p = np.array(point, dtype=np.float32).reshape(-1)[:3]
    lo = np.array(mn, dtype=np.float32).reshape(-1)[:3]
    hi = np.array(mx, dtype=np.float32).reshape(-1)[:3]
    outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    outside_dist = float(np.linalg.norm(outside))
    if outside_dist > 0.0:
        return outside_dist
    inside_margin = np.minimum(p - lo, hi - p)
    return -float(np.min(inside_margin))


def dataset_rigid_clearance_summary():
    obstacles = rigid_obstacle_bboxes()
    points = [
        ("tip", bucket_tip_pos()),
        ("load", bucket_load_pos()),
        ("pour", bucket_pour_pos()),
        ("mid", bucket_mid_pos()),
    ]
    best = {
        "available": bool(obstacles),
        "min_m": None,
        "point": "",
        "obstacle": "",
        "inside": False,
        "obstacle_count": int(len(obstacles)),
    }
    if not obstacles:
        return best
    best_dist = None
    best_point = ""
    best_obstacle = ""
    for point_name, point in points:
        if point is None:
            continue
        for obstacle in obstacles:
            try:
                dist = point_aabb_signed_distance(point, obstacle["min"], obstacle["max"])
            except Exception:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = float(dist)
                best_point = point_name
                best_obstacle = str(obstacle.get("path", ""))
    if best_dist is not None:
        best.update({
            "min_m": float(best_dist),
            "point": best_point,
            "obstacle": best_obstacle,
            "inside": bool(best_dist < 0.0),
        })
    return best


def dataset_local_height_patch(target=None, radius=0.42, grid=3):
    if target is None:
        return {
            "available": False,
            "grid": int(grid),
            "radius": float(radius),
            "values": [],
            "source": "missing_target",
        }
    try:
        p = np.array(target, dtype=np.float32).reshape(-1)[:3]
    except Exception:
        return {
            "available": False,
            "grid": int(grid),
            "radius": float(radius),
            "values": [],
            "source": "bad_target",
        }
    grid = max(1, int(grid))
    radius = max(0.05, float(radius))
    snapshot = get_sand_snapshot(force=False, label="dataset_height_patch", max_age=0.75)
    values = []
    sources = []
    offsets = np.linspace(-radius, radius, grid)
    for dy in offsets:
        row = []
        for dx in offsets:
            x = float(p[0] + dx)
            y = float(p[1] + dy)
            z = sand_snapshot_surface_height(snapshot, x, y)
            source = "snapshot"
            if z is None:
                try:
                    z, source = sand_surface_query_at_xy(x, y)
                except Exception:
                    z = None
                    source = "missing"
            row.append(None if z is None else round(float(z), 4))
            sources.append(str(source))
        values.append(row)
    available = any(v is not None for row in values for v in row)
    return {
        "available": bool(available),
        "grid": int(grid),
        "radius": float(radius),
        "center": vec_list(p, 3),
        "values": values,
        "source": "snapshot" if any(s == "snapshot" for s in sources) else (sources[0] if sources else "missing"),
    }


def dataset_environment_summary(target=None):
    ctx = task_scene_context()
    api = get_sand_site_api()
    if target is None:
        target = get_target_pos() if TARGET_PATH else None
    surface_z = None
    surface_source = ""
    if target is not None:
        try:
            surface_z, surface_source = sand_surface_query_at_xy(float(target[0]), float(target[1]))
        except Exception:
            surface_z = None
            surface_source = "query_failed"
    return {
        "dig_target": vec_list(target, 3),
        "sand_surface_z": None if surface_z is None else float(surface_z),
        "sand_surface_source": str(surface_source),
        "local_height_patch": dataset_local_height_patch(target=target, radius=0.42, grid=3),
        "soil": {
            "sand_site_active": bool(api is not None and sand_site_active()),
            "fidelity": None if api is None else api.get("sand_fidelity"),
            "particle_mass": sand_particle_mass(),
            "particle_count": int((STATE.get("sand_snapshot_last") or {}).get("particle_count", 0))
            if isinstance(STATE.get("sand_snapshot_last"), dict)
            else 0,
        },
        "unload_landing": vec_list(STATE.get("active_unload_landing_point"), 3),
        "unload_release": vec_list(STATE.get("active_unload_release_point"), 3),
        "bin_center": vec_list(ctx.get("unload_bin_center"), 3),
        "bin_half_size": vec_list(ctx.get("unload_bin_half_size"), 2),
        "bin_z_range": vec_list(ctx.get("unload_bin_z_range"), 2),
        "bin_aabb": {
            "center": vec_list(ctx.get("unload_bin_center"), 3),
            "half_size_xy": vec_list(ctx.get("unload_bin_half_size"), 2),
            "z_range": vec_list(ctx.get("unload_bin_z_range"), 2),
        },
        "rigid_obstacle_count": int(len(rigid_obstacle_bboxes())),
    }


def dataset_contact_flags(phase, sand_metrics, rigid_clearance):
    sand_metrics = sand_metrics if isinstance(sand_metrics, dict) else {}
    bucket_from_pile = int(sand_metrics.get("bucket_from_pile_count", 0) or 0)
    bucket_total = int(sand_metrics.get("bucket_count", 0) or 0)
    bin_from_pile = int(sand_metrics.get("bin_from_pile_count", 0) or 0)
    spill_from_pile = int(sand_metrics.get("spill_from_pile_count", 0) or 0)
    rigid_inside = bool(isinstance(rigid_clearance, dict) and rigid_clearance.get("inside", False))
    return {
        "bucket_soil_phase": bool(is_sand_contact_phase(phase) or is_curl_phase(phase)),
        "bucket_has_pile_sand": bool(bucket_from_pile > 0),
        "bucket_has_any_sand": bool(bucket_total > 0),
        "bin_has_pile_sand": bool(bin_from_pile > 0),
        "spill_from_pile": bool(spill_from_pile > 0),
        "bucket_rigid_overlap": rigid_inside,
    }


def dataset_cost_summary(q_cmd, q_real, action, sand_metrics, rigid_clearance, dynamics=None):
    err = dataset_joint_error(q_cmd, q_real)
    err_deg = [abs(rad_to_deg(x)) for x in err]
    action_arr = np.array(action, dtype=np.float32).reshape(-1) if action is not None else np.zeros(4, dtype=np.float32)
    bucket_from_pile = int(sand_metrics.get("bucket_from_pile_count", 0) or 0) if isinstance(sand_metrics, dict) else 0
    spill_from_pile = int(sand_metrics.get("spill_from_pile_count", 0) or 0) if isinstance(sand_metrics, dict) else 0
    load_total = max(1.0, float(bucket_from_pile + spill_from_pile))
    dynamics = dynamics if isinstance(dynamics, dict) else {}
    return {
        "joint_error_max_deg": float(max(err_deg) if err_deg else 0.0),
        "joint_error_l2_rad": float(np.linalg.norm(np.array(err, dtype=np.float32))),
        "action_l2": float(np.linalg.norm(action_arr)),
        "speed_l2": float(dynamics.get("speed_l2", 0.0) or 0.0),
        "accel_l2": float(dynamics.get("accel_l2", 0.0) or 0.0),
        "jerk_l2": float(dynamics.get("jerk_l2", 0.0) or 0.0),
        "command_accel_l2": float(dynamics.get("command_accel_l2", 0.0) or 0.0),
        "energy_proxy": float(dynamics.get("energy_proxy", 0.0) or 0.0),
        "clearance_to_rigid_m": None if not isinstance(rigid_clearance, dict) else rigid_clearance.get("min_m"),
        "spill_ratio_instant": float(spill_from_pile / load_total),
        "bucket_from_pile": int(bucket_from_pile),
        "spill_from_pile": int(spill_from_pile),
    }


def dataset_constraint_flags(cost, contact, phase_features):
    flags = []
    if isinstance(contact, dict) and contact.get("bucket_rigid_overlap", False):
        flags.append("rigid_overlap")
    if isinstance(cost, dict) and float(cost.get("joint_error_max_deg", 0.0) or 0.0) > 12.0:
        flags.append("joint_tracking_error")
    context = phase_features.get("context", {}) if isinstance(phase_features, dict) else {}
    if context.get("phase_class") == "rigid_free_space" and isinstance(cost, dict):
        clearance = cost.get("clearance_to_rigid_m")
        if clearance is not None and float(clearance) < 0.04:
            flags.append("low_rigid_clearance")
    if isinstance(cost, dict) and float(cost.get("spill_ratio_instant", 0.0) or 0.0) > QUALITY_MAX_SPILL_RATIO:
        flags.append("high_spill_ratio")
    return flags


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

        dynamics = dataset_motion_derivatives(q_cmd, q_real, now)
        action = dynamics["action"]
        joint_velocity = dynamics["dq_real"]
        joint_acceleration = dynamics["ddq_real"]
        sand_metrics = sand_metrics_current(force=force)
        update_episode_quality_trackers(sand_metrics, phase, q_cmd=q_cmd, q_real=q_real, action=action)
        q_goal = STATE.get("dataset_current_q_goal")
        target = get_target_pos() if TARGET_PATH else None
        phase_features = dataset_phase_features(phase)
        rigid_clearance = dataset_rigid_clearance_summary()
        contact_flags = dataset_contact_flags(phase, sand_metrics, rigid_clearance)
        env_summary = dataset_environment_summary(target)
        cost_summary = dataset_cost_summary(q_cmd, q_real, action, sand_metrics, rigid_clearance, dynamics=dynamics)
        constraint_flags = dataset_constraint_flags(cost_summary, contact_flags, phase_features)
        sample_index = int(STATE.get("dataset_samples", 0)) - int(STATE.get("dataset_episode_sample_start", 0))
        sample = {
            "v": DATASET_TRAJECTORY_FORMAT,
            "ep": int(STATE.get("dataset_episode_id", 0)),
            "id": str(STATE.get("dataset_episode_uid", "")),
            "i": int(sample_index),
            "t": float(now) - float(STATE.get("dataset_episode_start_time", now)),
            "phase": str(phase),
            "phase.index": phase_features["index"],
            "phase.one_hot": phase_features["one_hot"],
            "phase.context": phase_features["context"],
            "label": str(label),
            "obs.state": dataset_observation_state(q_real=q_real),
            "obs.q": vec_list(q_real, 4),
            "obs.dq": joint_velocity,
            "obs.ddq": joint_acceleration,
            "obs.q_cmd": vec_list(q_cmd, 4),
            "obs.q_err": dataset_joint_error(q_cmd, q_real),
            "action": action,
            "action.ddq": dynamics["action_ddq"],
            "goal.q": vec_list(q_goal, 4),
            "target": vec_list(target, 3),
            "bucket.tip": vec_list(bucket_tip_pos(), 3),
            "bucket.load": vec_list(bucket_load_pos(), 3),
            "bucket.pour": vec_list(bucket_pour_pos(), 3),
            "sand": compact_sand_metrics(sand_metrics),
            "contact": contact_flags,
            "env": env_summary,
            "cost": cost_summary,
            "label.flags": constraint_flags,
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
            include_sand=str(event) in {"phase_metrics", "episode_end", "execution_failure"},
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
            "phase_names": DATASET_PHASE_NAMES,
            "trajectory_fields": {
                "obs.state": DATASET_STATE_NAMES,
                "obs.q": DOF_ORDER,
                "obs.dq": DOF_ORDER,
                "obs.ddq": DOF_ORDER,
                "obs.q_cmd": DOF_ORDER,
                "obs.q_err": DOF_ORDER,
                "action": DATASET_ACTION_NAMES,
                "action.ddq": DATASET_ACTION_NAMES,
                "goal.q": DOF_ORDER,
                "phase.index": "index into phase_names; -1 means unknown/debug",
                "phase.one_hot": DATASET_PHASE_NAMES,
                "phase.context": ["phase_class", "rigid_policy", "sand_policy", "strict_path_precheck"],
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
                "contact": [
                    "bucket_soil_phase",
                    "bucket_has_pile_sand",
                    "bucket_has_any_sand",
                    "bin_has_pile_sand",
                    "spill_from_pile",
                    "bucket_rigid_overlap",
                ],
                "env": [
                    "dig_target",
                    "sand_surface_z",
                    "sand_surface_source",
                    "local_height_patch",
                    "soil",
                    "unload_landing",
                    "unload_release",
                    "bin_center",
                    "bin_half_size",
                    "bin_z_range",
                    "bin_aabb",
                    "rigid_obstacle_count",
                ],
                "cost": [
                    "joint_error_max_deg",
                    "joint_error_l2_rad",
                    "action_l2",
                    "clearance_to_rigid_m",
                    "spill_ratio_instant",
                    "bucket_from_pile",
                    "spill_from_pile",
                ],
                "label.flags": [
                    "rigid_overlap",
                    "joint_tracking_error",
                    "low_rigid_clearance",
                    "high_spill_ratio",
                ],
                "events.stage_audit.constraint_summary": [
                    "stage_cost",
                    "motion_cost",
                    "weighted_angle",
                    "estimated_time",
                    "path_penalty",
                    "phase_ok",
                    "obstacle_ok",
                    "route_inserted",
                    "front_edge",
                    "material_hold",
                    "unload_drop",
                    "flags",
                ],
            },
            "trainable_index": "trainable_episodes.jsonl",
            "successful_index": "successful_episodes.jsonl",
            "rejected_index": "rejected_episodes.jsonl",
            "failed_index": "failed_episodes.jsonl",
            "diagnostic_index": "diagnostic_episodes.jsonl",
            "planning_diagnostics_index": "planning_diagnostics.jsonl",
            "segment_indices": {
                "dig": "segment_dig.jsonl",
                "dig_secure": "segment_dig_secure.jsonl",
                "lift_carry": "segment_lift_carry.jsonl",
                "unload": "segment_unload.jsonl",
            },
            "debug_timeline_index": DATASET_DEBUG_TIMELINE_FILE,
            "trajectory_format": DATASET_TRAJECTORY_FORMAT,
            "scene_context": compact_scene_context(),
            "target_center": vec_list(AUTO_COLLECT_TARGET_CENTER, 3),
            "target_radius_x": AUTO_COLLECT_TARGET_RADIUS_X,
            "target_radius_y": AUTO_COLLECT_TARGET_RADIUS_Y,
            "target_depths": AUTO_COLLECT_TARGET_DEPTHS,
            "target_strategy": "center_first_ring_full_plan",
            "target_ring_radii": list(AUTO_DIG_RING_RADII),
            "target_ring_points": list(AUTO_DIG_RING_POINTS),
            "target_depth_priority": list(AUTO_DIG_DEPTH_PRIORITY),
            "target_sweep_radius": AUTO_DIG_SWEEP_RADIUS,
            "target_full_plan_topk_per_ring": AUTO_DIG_FULL_PLAN_TOPK_PER_RING,
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
        "segment_dig.jsonl",
        "segment_dig_secure.jsonl",
        "segment_lift_carry.jsonl",
        "segment_unload.jsonl",
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
    segment_dig_count = jsonl_line_count(os.path.join(run_dir, "segment_dig.jsonl"))
    segment_dig_secure_count = jsonl_line_count(os.path.join(run_dir, "segment_dig_secure.jsonl"))
    segment_lift_carry_count = jsonl_line_count(os.path.join(run_dir, "segment_lift_carry.jsonl"))
    segment_unload_count = jsonl_line_count(os.path.join(run_dir, "segment_unload.jsonl"))
    indexed_attempt_rows = int(episodes_count) + int(planning_count) + int(diagnostic_count)
    attempts = max(int(STATE.get("auto_collect_attempts", 0)), episodes_count, indexed_attempt_rows)
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
            "segments": {
                "dig": segment_dig_count,
                "dig_secure": segment_dig_secure_count,
                "lift_carry": segment_lift_carry_count,
                "unload": segment_unload_count,
            },
            "last_result": STATE.get("auto_collect_last_result", ""),
            "run_dir": run_dir,
            "trainable_index": os.path.join(run_dir, "trainable_episodes.jsonl"),
            "successful_index": os.path.join(run_dir, "successful_episodes.jsonl"),
            "rejected_index": os.path.join(run_dir, "rejected_episodes.jsonl"),
            "failed_index": os.path.join(run_dir, "failed_episodes.jsonl"),
            "diagnostic_index": os.path.join(run_dir, "diagnostic_episodes.jsonl"),
            "planning_diagnostics_index": os.path.join(run_dir, "planning_diagnostics.jsonl"),
            "segment_indices": {
                "dig": os.path.join(run_dir, "segment_dig.jsonl"),
                "dig_secure": os.path.join(run_dir, "segment_dig_secure.jsonl"),
                "lift_carry": os.path.join(run_dir, "segment_lift_carry.jsonl"),
                "unload": os.path.join(run_dir, "segment_unload.jsonl"),
            },
            "debug_timeline_index": os.path.join(run_dir, DATASET_DEBUG_TIMELINE_FILE),
            "scene_context": compact_scene_context(),
            "sand_reset_policy": AUTO_COLLECT_SAND_RESET_POLICY,
            "sand_reset_done": bool(STATE.get("auto_collect_sand_reset_done", False)),
            "sand_site_stable_reset_done": bool(STATE.get("sand_site_stable_reset_done", False)),
            "sand_site_last_reset_label": STATE.get("sand_site_last_reset_label", ""),
            "auto_reset_sand_after_ui_ready": AUTO_RESET_SAND_AFTER_UI_READY,
            "perf_last": dict(STATE.get("sand_perf_last", {}) or {}),
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
            if stage.get("q_release_align_rad") is not None:
                row["q_release_align_rad"] = vec_list(stage.get("q_release_align_rad"), 4)
            if stage.get("q_release_align_deg") is not None:
                row["q_release_align_deg"] = stage.get("q_release_align_deg")
            if stage.get("drop") is not None:
                row["drop"] = stage.get("drop")
            for key in [
                "motion",
                "path",
                "front_edge",
                "material_hold",
                "clearance_route",
                "stage_cost",
                "drop_alignment_ready",
                "drop_alignment_policy",
                "release_alignment_bucket_deg",
                "planar_err",
                "world_angle_err_deg",
                "bucket_world_deg",
            ]:
                if stage.get(key) is not None:
                    row[key] = stage.get(key)
            row["constraint_summary"] = stage_constraint_summary(stage)
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
                f"spill_from_pile={int(metrics.get('spill_from_pile_count', 0))} "
                f"source_tracking={metrics.get('source_tracking', 'unknown')}"
            ),
            data=row,
        )
    return metrics


def stage_contract_summary(stage_name, stage_index=None, q_goal=None, duration=None):
    stage_name = str(stage_name)
    plan = STATE.get("current_dig_plan")
    plan = plan if isinstance(plan, dict) else {}
    seq = plan.get("stage_sequence", [])
    stage_row = {}
    if isinstance(seq, list) and stage_index is not None:
        try:
            idx = int(stage_index)
            if 0 <= idx < len(seq) and isinstance(seq[idx], dict):
                stage_row = dict(seq[idx])
        except Exception:
            stage_row = {}
    if not stage_row and isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict) and str(item.get("phase", "")) == stage_name:
                stage_row = dict(item)
                break

    out = {
        "plan_id": str(plan.get("plan_id", "")),
        "planner_version": str(plan.get("planner_version", PLANNER_VERSION)),
        "stage_index": None if stage_index is None else int(stage_index),
        "stage_name": stage_name,
        "phase_context": phase_collision_context(stage_name),
        "duration": None if duration is None else float(duration),
    }
    if q_goal is not None:
        out["q_goal_deg"] = q_deg_values(q_goal, wrap_swing_for_display=True)
    if stage_row:
        out["planned"] = bool(stage_row.get("planned", True))
        out["required"] = bool(stage_row.get("required", False))
        out["target"] = vec_list(stage_row.get("target"), 3)
        if stage_row.get("q_goal_deg") is not None:
            out["plan_q_goal_deg"] = stage_row.get("q_goal_deg")
        if stage_row.get("q_dump_deg") is not None:
            out["plan_q_dump_deg"] = stage_row.get("q_dump_deg")
        if stage_row.get("path") is not None:
            out["path"] = stage_row.get("path")
        if stage_row.get("clearance_route") is not None:
            out["clearance_route"] = stage_row.get("clearance_route")
        if stage_row.get("drop") is not None:
            out["unload_drop"] = stage_row.get("drop")
        constraints = stage_row.get("constraint_summary")
        if constraints is None:
            constraints = stage_constraint_summary(stage_row)
        if constraints:
            out["constraint_summary"] = constraints
    if stage_name.startswith("unload") and isinstance(plan.get("unload_ballistics"), dict):
        out["unload_ballistics"] = plan.get("unload_ballistics")
    return out


def record_stage_audit(stage_name, stage_index, result, reason="", q_goal=None, duration=None, data=None, include_sand=False):
    try:
        row = stage_contract_summary(stage_name, stage_index=stage_index, q_goal=q_goal, duration=duration)
        row["result"] = str(result)
        if reason:
            row["reason"] = debug_short_string(reason, 360)
        try:
            q_real = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
        except Exception:
            q_real = None
        row["q_cmd_deg"] = q_deg_values(CTRL.q_cmd, wrap_swing_for_display=True)
        if q_real is not None:
            row["q_real_deg"] = q_deg_values(q_real, wrap_swing_for_display=True)
        if q_goal is not None and q_real is not None:
            try:
                row["goal_error_deg"] = q_delta_abs_deg(q_goal, q_real)
            except Exception:
                pass
        if include_sand:
            row["sand"] = debug_compact_sand_counts()
        if isinstance(data, dict):
            row.update(data)
        if bool(STATE.get("dataset_recording", False)):
            dataset_record_event(
                "stage_audit",
                f"{stage_name}:{result}" + (f":{reason}" if reason else ""),
                data=row,
            )
        else:
            debug_timeline_record(
                "STAGE_AUDIT",
                stage=stage_name,
                result=str(result),
                reason=reason,
                data=row,
                q_cmd=q_goal if q_goal is not None else CTRL.q_cmd,
                q_real=q_real,
                include_sand=include_sand,
            )
    except Exception as e:
        info_print("[WARN] stage audit failed:", stage_name, result, type(e).__name__, e)


def auto_collect_preflight_report(target_successes=None):
    run_dir = ensure_auto_collect_run_dir()
    ctx = task_scene_context()
    t0 = time.perf_counter()
    snapshot = get_sand_snapshot(force=True, label="auto_preflight")
    if isinstance(snapshot, dict):
        STATE["auto_collect_episode_sand_snapshot"] = snapshot
        STATE["auto_collect_episode_sand_snapshot_time"] = float(time.time())
    metrics = sand_metrics_current(force=True, snapshot=snapshot)
    particles = snapshot.get("points") if isinstance(snapshot, dict) else None
    particle_count = int(len(particles)) if particles is not None else int(metrics.get("particle_count", 0) or 0)
    settle = snapshot.get("settle") if isinstance(snapshot, dict) and isinstance(snapshot.get("settle"), dict) else sand_settle_status(points=particles, ctx=ctx)
    preflight_snapshot_ms = (time.perf_counter() - t0) * 1000.0
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

    ik_ready = ik_model_is_valid()
    checks = {
        "sand_site": api is not None and bool(sand_site_active()),
        "particle_system": sand_particle_prim() is not None,
        "particle_count": particle_count >= int(AUTO_PREFLIGHT_MIN_PARTICLES),
        "sand_settled": bool(settle.get("ok", False)),
        "unload_bin": np.all(unload_inner > 0.05),
        "bucket_collider": bucket_collider_ready,
        "robot": ROBOT is not None and len(DOF_NAME_TO_REAL_IDX) >= 4,
        "action_channel": action_ready,
        "ik": ik_ready,
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
        "sand_settled": "preflight_failed/sand_not_settled",
        "unload_bin": "preflight_failed/unload_bin_missing",
        "bucket_collider": "preflight_failed/bucket_collider_not_ready",
        "robot": "preflight_failed/robot_not_ready",
        "action_channel": "preflight_failed/action_channel_not_ready",
        "ik": "preflight_failed/ik_calibration_invalid" if IK_MODEL is not None else "preflight_failed/ik_not_ready",
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
            "ok": checks["sand_site"] and checks["particle_system"] and checks["particle_count"] and checks["sand_settled"],
            "center": vec_list(pile_center, 3),
            "radius": vec_list(pile_radius, 2),
            "particles": particle_count,
            "min_particles": int(AUTO_PREFLIGHT_MIN_PARTICLES),
            "settled": bool(settle.get("ok", False)),
            "settle_reason": str(settle.get("reason", "")),
            "settle": settle,
            "status": dataset_sand_status(),
            "snapshot_ms": float(preflight_snapshot_ms),
            "snapshot_perf_ms": dict(snapshot.get("perf_ms", {}) if isinstance(snapshot, dict) else {}),
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
            "ik_report": dict(STATE.get("ik_calibration_report", {}) or {}),
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
        f"settled={report['sand'].get('settled')} settle_reason={report['sand'].get('settle_reason')}",
        f"snapshot_ms={preflight_snapshot_ms:.1f}",
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


def truncate_text(value, limit=240):
    text = str(value)
    limit = max(16, int(limit))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compact_target_score_row(row):
    row = row if isinstance(row, dict) else {}
    keys = [
        "target_xyz",
        "ring_index",
        "center_distance",
        "surface_z",
        "depth_candidate",
        "swept_density_count",
        "density_count",
        "effective_density_count",
        "density_score",
        "swept_density_score",
        "depth_score",
        "fill_potential_score",
        "motion_score",
        "center_score",
        "approach_score",
        "approach_quality",
        "score_components",
        "score",
        "planned",
        "failed_stage",
        "reason",
        "failure_reason",
    ]
    return {key: row.get(key) for key in keys if key in row}


def compact_unload_score_row(row):
    row = row if isinstance(row, dict) else {}
    keys = ["landing", "cell_height", "cell_count", "center_norm", "motion_dist", "score", "route_hint"]
    return {key: row.get(key) for key in keys if key in row}


def compact_auto_plan_attempts(plan_attempts, limit=4):
    if not isinstance(plan_attempts, list):
        return []
    rows = []
    for item in plan_attempts[: max(0, int(limit))]:
        item = item if isinstance(item, dict) else {}
        rows.append(
            {
                "retry": item.get("retry"),
                "ring_index": item.get("ring_index"),
                "target_xyz": item.get("target_xyz"),
                "selected_unload_landing_xyz": item.get("selected_unload_landing_xyz"),
                "planned": bool(item.get("planned", False)),
                "steps": item.get("steps"),
                "failed_stage": item.get("failed_stage"),
                "failure_reason": truncate_text(item.get("failure_reason", ""), 280),
                "build_ms": item.get("build_ms"),
                "wall_ms": item.get("wall_ms"),
                "candidate_count": item.get("candidate_count"),
                "planning_world": item.get("planning_world", {}),
                "target_score": compact_target_score_row(item.get("target_score", {})),
                "best_failure": compact_plan_candidate(item.get("best_failure", {}), include_stages=False),
            }
        )
    return rows


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
        "prepare_gate_report": STATE.get("auto_collect_prepare_gate_report", {}),
        "planning_world": planning_world_snapshot(force=False),
        "best_failure": compact_plan_candidate(best_failure, include_stages=True),
        "chosen_plan": compact_plan_candidate(chosen_plan, include_stages=True),
        "shared_plan": {
            "plan_id": shared_plan.get("plan_id", "") if isinstance(shared_plan, dict) else "",
            "cost": shared_plan.get("total_plan_cost") if isinstance(shared_plan, dict) else None,
            "stage_count": shared_plan.get("stage_count") if isinstance(shared_plan, dict) else None,
            "unload_ballistics": shared_plan.get("unload_ballistics", {}) if isinstance(shared_plan, dict) else {},
        },
        "plan_attempts": compact_auto_plan_attempts(plan_attempts, limit=4),
        "dig_target_candidates": [
            compact_target_score_row(x)
            for x in (STATE.get("last_auto_dig_target_scores", []) or [])[:12]
        ],
        "unload_flat_fill_candidates": [
            compact_unload_score_row(x)
            for x in (STATE.get("last_auto_unload_scores", []) or [])[:10]
        ],
        "perf_last": dict(STATE.get("sand_perf_last", {}) or {}),
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
        if raw_reason.startswith((
            "prepare_failed/",
            "preflight_failed/",
            "planning_failed/",
            "execution_failed/",
            "quality_rejected/",
        )):
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


def phase_metric_bucket_from_pile(phase_metrics, phase_names):
    if not isinstance(phase_metrics, dict):
        return 0
    names = [str(x).lower() for x in phase_names]
    best = 0
    for label, row in phase_metrics.items():
        label_l = str(label).lower()
        if not any(name in label_l for name in names):
            continue
        sand = row.get("sand") if isinstance(row, dict) else {}
        if isinstance(sand, dict):
            best = max(best, int(sand.get("bucket_from_pile", 0) or 0))
    return int(best)


def auto_collect_write_segment_indices(run_dir, meta, index_row, score_report):
    if not run_dir:
        return
    counts = score_report.get("counts", {}) if isinstance(score_report, dict) else {}
    phase_metrics = score_report.get("phase_metrics", {}) if isinstance(score_report, dict) else {}
    dig_bucket = max(
        int(counts.get("max_bucket_from_pile_particles", 0) or 0),
        phase_metric_bucket_from_pile(phase_metrics, ["insert_cut", "pull_mid_cut", "pull_exit_cut", "after_cut"]),
    )
    secure_bucket = phase_metric_bucket_from_pile(phase_metrics, ["curl_to_hold_material", "secure_load"])
    lift_bucket = int(counts.get("lift_bucket_from_pile_particles", 0) or 0)
    final_bin = int(counts.get("final_bin_from_pile_particles", 0) or 0)
    final_spill = int(counts.get("final_spill_from_pile_particles", 0) or 0)
    episode_reason = str(index_row.get("reason", "") or "")
    secure_gate_failed = (
        "secure_not_retaining_material" in episode_reason
        or "carry_spill_risk" in episode_reason
        or "secure_material_loss" in episode_reason
    )
    base = {
        "episode_id": index_row.get("episode_id"),
        "episode_index": index_row.get("episode_index"),
        "created_at": time.time(),
        "full_chain_success": bool(index_row.get("full_chain_success", False)),
        "episode_status": index_row.get("status"),
        "episode_reason": index_row.get("reason"),
        "planner_version": index_row.get("planner_version"),
        "config_hash": index_row.get("config_hash"),
        "target_xyz": index_row.get("target_xyz"),
        "unload_landing_xyz": index_row.get("unload_landing_xyz"),
        "unload_release_xyz": index_row.get("unload_release_xyz"),
        "initial_pose_id": index_row.get("initial_pose_id", ""),
        "plan_debug": index_row.get("plan_debug", ""),
        "trajectory": index_row.get("trajectory", ""),
        "events": index_row.get("events", ""),
        "score_path": index_row.get("score_path", ""),
        "meta": index_row.get("meta", ""),
        "score": index_row.get("score"),
        "freeze_count": index_row.get("freeze_count"),
    }

    def append_segment(filename, segment, count, threshold, extra=None):
        if int(count) < int(threshold):
            return
        bucket_field = int(count) if str(segment) in ("dig", "dig_secure") else int(secure_bucket)
        row = dict(base)
        row.update({
            "segment": str(segment),
            "segment_success": True,
            "segment_count": int(count),
            "threshold": int(threshold),
            "bucket_from_pile": bucket_field,
            "lift_bucket_from_pile": int(lift_bucket),
            "final_bin_from_pile": int(final_bin),
            "final_spill_from_pile": int(final_spill),
        })
        if isinstance(extra, dict):
            row.update(extra)
        append_jsonl(os.path.join(run_dir, filename), row)

    append_segment(
        "segment_dig.jsonl",
        "dig",
        dig_bucket,
        QUALITY_MIN_BUCKET_PARTICLES,
        {"success_gate": "bucket_from_pile_after_cut"},
    )
    if not secure_gate_failed:
        append_segment(
            "segment_dig_secure.jsonl",
            "dig_secure",
            secure_bucket,
            QUALITY_MIN_BUCKET_PARTICLES,
            {"success_gate": "bucket_from_pile_after_curl_or_secure_and_retains_material"},
        )
    append_segment(
        "segment_lift_carry.jsonl",
        "lift_carry",
        lift_bucket,
        QUALITY_MIN_BUCKET_PARTICLES,
        {"success_gate": "bucket_from_pile_after_lift"},
    )
    append_segment(
        "segment_unload.jsonl",
        "unload",
        final_bin,
        QUALITY_MIN_DUMP_PARTICLES,
        {"success_gate": "bin_from_pile_after_dump"},
    )


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
    STATE["dataset_last_action"] = None
    STATE["dataset_last_dq_real"] = None
    STATE["dataset_last_ddq_real"] = None
    STATE["dataset_current_q_goal"] = None
    STATE["last_execution_failure_reason"] = ""
    STATE["sand_metrics_last_time"] = 0.0
    STATE["sand_metrics_last"] = None
    STATE["dataset_initial_pile_particle_ids"] = None
    STATE["dataset_initial_pile_particle_mask"] = None
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
    chosen_dig_primitive = chosen_plan_compact.get("dig_primitive", {})
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
            "dig_primitive": chosen_dig_primitive,
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
        "dig_primitive": chosen_dig_primitive,
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
        "phase_names": DATASET_PHASE_NAMES,
        "trajectory_fields_added_v3": [
            "phase.index",
            "phase.one_hot",
            "phase.context",
            "contact",
            "env",
                "cost",
                "label.flags",
                "obs.ddq",
                "action.ddq",
            ],
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
        "full_chain_success": bool(success),
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
    auto_collect_write_segment_indices(run_dir, meta, index_row, score_report)
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
    ctx = task_scene_context()
    center = np.array(ctx["pile_center"], dtype=np.float32)
    pile_radius = np.array(ctx["pile_radius"], dtype=np.float32)
    rx = min(float(AUTO_COLLECT_TARGET_RADIUS_X), max(0.12, float(pile_radius[0]) * 0.52))
    ry = min(float(AUTO_COLLECT_TARGET_RADIUS_Y), max(0.12, float(pile_radius[1]) * 0.52))
    depths = auto_dig_depth_candidates()
    retry = max(0, int(retry_index))
    ring_index = min(retry // max(1, len(depths)), len(AUTO_DIG_RING_RADII) - 1)
    radius_norm = float(AUTO_DIG_RING_RADII[ring_index])
    if ring_index == 0:
        x = float(center[0])
        y = float(center[1])
    else:
        n = max(1, int(AUTO_DIG_RING_POINTS[min(ring_index, len(AUTO_DIG_RING_POINTS) - 1)]))
        phase = 2.0 * math.pi * (((max(1, int(attempt_index)) - 1) * 0.3819660112501051) % 1.0)
        angle = phase + 2.0 * math.pi * float(retry % n) / float(n)
        x = float(center[0]) + rx * radius_norm * math.cos(angle)
        y = float(center[1]) + ry * radius_norm * math.sin(angle)

    snapshot = get_sand_snapshot(force=False, label="auto_sample_target", max_age=1.0)
    surface_z = sand_surface_height_for_auto_target(x, y, snapshot=snapshot)
    depth = float(depths[retry % max(1, len(depths))])
    z = auto_dig_target_z_from_surface(surface_z, depth)
    return np.array([x, y, z], dtype=np.float32)


def auto_dig_target_z_from_surface(surface_z, depth):
    surface_z = float(surface_z)
    depth = max(0.04, float(depth))
    dynamic_max_z = max(float(AUTO_COLLECT_TARGET_MAX_Z), surface_z + 0.05)
    return float(max(AUTO_COLLECT_TARGET_MIN_Z, min(dynamic_max_z, surface_z - depth)))


def sand_surface_height_for_auto_target(x, y, particles=None, snapshot=None):
    ctx = task_scene_context()
    _, _, floor_z, fill_height, _, z_expected_max = sand_pile_geometry_from_context(ctx)
    z_snapshot = sand_snapshot_surface_height(snapshot, x, y)
    if z_snapshot is not None:
        return float(z_snapshot)
    if particles is not None and len(particles) > 0:
        p = filter_settled_sand_particles(particles, ctx=ctx)
        if p is None or len(p) == 0:
            return float(floor_z + 0.10)
        d = np.linalg.norm(p[:, :2] - np.array([[float(x), float(y)]], dtype=np.float32), axis=1)
        near = p[d <= float(SAND_SURFACE_QUERY_RADIUS)]
        if len(near) > 0:
            # Prefer the real settled particle surface. The sand-site height_fn is a reference
            # shape used for generation and can describe empty air after particles settle.
            return float(max(float(floor_z) + 0.02, min(float(np.percentile(near[:, 2], 90.0)), float(z_expected_max))))
        return float(floor_z + 0.10)
    return float(max(float(floor_z) + 0.10, min(float(floor_z) + 0.18, float(z_expected_max))))


def auto_dig_depth_candidates():
    values = []
    for depth in list(AUTO_DIG_DEPTH_PRIORITY) + sorted([float(x) for x in AUTO_COLLECT_TARGET_DEPTHS], reverse=True):
        depth = float(depth)
        if depth > 0.24:
            continue
        if not any(abs(depth - old) < 1.0e-5 for old in values):
            values.append(depth)
    return values or [0.22, 0.18, 0.14, 0.10]


def auto_dig_swept_density_count(target, surface_z, particles=None, snapshot=None):
    if isinstance(snapshot, dict):
        particles = snapshot.get("settled_points")
    if particles is None or len(particles) == 0:
        return 0
    target = np.array(target, dtype=np.float32).reshape(-1)[:3]
    p = np.asarray(particles, dtype=np.float32)
    if p is None or len(p) == 0:
        return 0
    inward = dig_direction_unit(target)
    if float(np.linalg.norm(inward)) < 1.0e-6:
        inward = np.array([-1.0, 0.0], dtype=np.float32)
    a = target[:2] - inward * 0.04
    b = target[:2] + inward * 0.72
    ab = b - a
    ab2 = max(1.0e-6, float(np.dot(ab, ab)))
    if isinstance(snapshot, dict):
        center = 0.5 * (a + b)
        search_radius = float(np.linalg.norm(ab)) * 0.5 + max(0.08, float(AUTO_DIG_SWEEP_RADIUS))
        idx = sand_snapshot_local_indices(snapshot, float(center[0]), float(center[1]), search_radius)
        if idx is None or len(idx) == 0:
            return 0
        p = p[idx]

    rel = p[:, :2] - a.reshape(1, 2)
    t = np.clip((rel @ ab.reshape(2, 1)).reshape(-1) / ab2, 0.0, 1.0)
    closest = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)
    d2 = np.sum((p[:, :2] - closest) ** 2, axis=1)
    radius = max(0.08, float(AUTO_DIG_SWEEP_RADIUS))
    z_min = float(target[2]) - 0.20
    z_max = float(surface_z) + 0.24
    mask = (d2 <= radius * radius) & (p[:, 2] >= z_min) & (p[:, 2] <= z_max)
    return int(np.count_nonzero(mask))


def auto_dig_approach_quality(target, surface_z, snapshot=None):
    target = np.array(target, dtype=np.float32).reshape(-1)[:3]
    inward = dig_direction_unit(target)
    if float(np.linalg.norm(inward)) < 1.0e-6:
        inward = np.array([-1.0, 0.0], dtype=np.float32)
    outward = -inward
    probe_dist = 0.28
    p_outer = target[:2] + outward * probe_dist
    p_inner = target[:2] + inward * probe_dist
    outer_z = sand_snapshot_surface_height(snapshot, float(p_outer[0]), float(p_outer[1])) if isinstance(snapshot, dict) else None
    inner_z = sand_snapshot_surface_height(snapshot, float(p_inner[0]), float(p_inner[1])) if isinstance(snapshot, dict) else None
    if outer_z is None:
        outer_z = float(surface_z)
    if inner_z is None:
        inner_z = float(surface_z)
    slope_rise = float(inner_z) - float(outer_z)
    # Prefer a modest rise into the pile. Flat is acceptable; steep local slopes
    # and falling surfaces are less reliable for front-edge cutting.
    slope_score = clamp01(1.0 - abs(slope_rise - 0.08) / 0.42)
    falling_penalty = clamp01(max(0.0, -slope_rise) / 0.30)
    approach_score = clamp01(slope_score * (1.0 - 0.45 * falling_penalty))
    return {
        "approach_score": float(approach_score),
        "slope_rise_m": float(slope_rise),
        "outer_surface_z": float(outer_z),
        "inner_surface_z": float(inner_z),
    }


def auto_dig_fast_reach_check(target_world, q_seed=None, end_effector="tip"):
    target = np.array(target_world, dtype=np.float32).reshape(-1)[:3]
    q = CTRL.q_cmd.copy() if q_seed is None else CTRL.clip_limits(q_seed)
    pts = get_ik_world_points()
    swing_anchor = pts.get("swing") if isinstance(pts, dict) else None
    boom_now = pts.get("boom") if isinstance(pts, dict) else None
    if swing_anchor is None or boom_now is None:
        return False, "missing swing/boom anchor", {}

    part = ik_model_part(end_effector=end_effector)
    if part is None:
        part = ik_model_part(end_effector="mid")
    chain_now = current_planar_chain(end_effector)
    fallback_lengths = [] if chain_now is None else chain_now.get("lengths", [])
    lengths = np.array((part or {}).get("lengths", fallback_lengths), dtype=np.float32).reshape(-1)
    if len(lengths) != 3 or not np.all(np.isfinite(lengths)) or float(np.min(lengths)) < IK_MIN_SEGMENT_LENGTH:
        return False, "invalid fast reach lengths", {}

    dx = float(target[0] - swing_anchor[0])
    dy = float(target[1] - swing_anchor[1])
    raw_swing_goal = math.atan2(dy, dx)
    swing_now = float(q[CTRL.name_to_idx["swing"]])
    swing_goal = normalize_swing_cmd(swing_target_near(raw_swing_goal, swing_now))
    boom_offset_xy = boom_now[:2] - swing_anchor[:2]
    boom_goal_xy = swing_anchor[:2] + rotate_xy(boom_offset_xy, swing_goal - swing_now)
    boom_root = np.array([boom_goal_xy[0], boom_goal_xy[1], float(boom_now[2])], dtype=np.float32)
    radial = safe_norm(np.array([math.cos(swing_goal), math.sin(swing_goal)], dtype=np.float32), default=(1.0, 0.0))
    target_2d = point_to_2d(target, boom_root, radial)
    planar_dist = float(np.linalg.norm(target_2d))
    total_reach = float(np.sum(lengths))
    min_fold = max(0.0, float(np.max(lengths) - (np.sum(lengths) - np.max(lengths))))
    reach_margin = 0.35
    ok = bool(planar_dist <= total_reach + reach_margin and planar_dist >= max(0.0, min_fold - reach_margin))
    reason = "fast_reach_ok" if ok else f"fast_reach_out_of_range:{planar_dist:.3f}/{total_reach:.3f}"
    return ok, reason, {
        "planar_dist": float(planar_dist),
        "total_reach": float(total_reach),
        "min_fold": float(min_fold),
        "swing_goal_deg": float(rad_to_deg(swing_goal)),
    }


def auto_collect_rank_dig_targets(attempt_index):
    rank_t0 = time.perf_counter()
    ctx = task_scene_context()
    center = np.array(ctx["pile_center"], dtype=np.float32).reshape(-1)[:3]
    pile_radius = np.array(ctx["pile_radius"], dtype=np.float32).reshape(-1)[:2]
    rx = min(float(AUTO_COLLECT_TARGET_RADIUS_X), max(0.12, float(pile_radius[0]) * 0.56))
    ry = min(float(AUTO_COLLECT_TARGET_RADIUS_Y), max(0.12, float(pile_radius[1]) * 0.56))
    snapshot = None
    cached_snapshot = STATE.get("auto_collect_episode_sand_snapshot")
    cached_age = time.time() - float(STATE.get("auto_collect_episode_sand_snapshot_time", 0.0) or 0.0)
    if isinstance(cached_snapshot, dict) and cached_age <= float(AUTO_COLLECT_PLANNING_SNAPSHOT_MAX_AGE):
        snapshot = cached_snapshot
    if snapshot is None:
        snapshot = get_sand_snapshot(force=False, label="auto_rank_targets", max_age=AUTO_COLLECT_PLANNING_SNAPSHOT_MAX_AGE)
    settled_status = snapshot.get("settle", {}) if isinstance(snapshot, dict) else {}
    settled_particles = snapshot.get("settled_points") if isinstance(snapshot, dict) else None
    q_ref = CTRL.q_cmd.copy()
    depths = auto_dig_depth_candidates()
    phase = 2.0 * math.pi * (((max(1, int(attempt_index)) - 1) * 0.3819660112501051) % 1.0)
    rows = []
    candidate_budget = int(sum((1 if int(i) == 0 else int(AUTO_DIG_RING_POINTS[min(int(i), len(AUTO_DIG_RING_POINTS) - 1)])) for i, _r in enumerate(AUTO_DIG_RING_RADII)) * max(1, len(depths)))
    info_print(
        "[AUTO DIG TARGET RANK]",
        f"start attempt={int(attempt_index)}",
        f"candidate_budget={candidate_budget}",
        f"particles={0 if settled_particles is None else len(settled_particles)}",
        f"snapshot_ms={fmt_optional((snapshot.get('perf_ms') or {}).get('total') if isinstance(snapshot, dict) else None)}",
    )

    for ring_index, radius_norm in enumerate(AUTO_DIG_RING_RADII):
        point_count = 1 if ring_index == 0 else int(AUTO_DIG_RING_POINTS[min(ring_index, len(AUTO_DIG_RING_POINTS) - 1)])
        for angle_index in range(max(1, point_count)):
            if ring_index == 0:
                angle = 0.0
                x = float(center[0])
                y = float(center[1])
                norm = 0.0
            else:
                angle = phase + 2.0 * math.pi * float(angle_index) / float(point_count)
                x = float(center[0]) + rx * float(radius_norm) * math.cos(angle)
                y = float(center[1]) + ry * float(radius_norm) * math.sin(angle)
                norm = float(radius_norm)
            surface_z = sand_surface_height_for_auto_target(x, y, particles=settled_particles, snapshot=snapshot)
            for depth in depths:
                z = auto_dig_target_z_from_surface(surface_z, depth)
                target = np.array([x, y, z], dtype=np.float32)
                center_distance = float(np.linalg.norm(target[:2] - center[:2]))
                ok, reason = validate_dig_target(target, hard_block=False)
                base_row = {
                    "target_xyz": vec_list(target, 3),
                    "ring_index": int(ring_index),
                    "ring_radius_norm": float(radius_norm),
                    "angle_index": int(angle_index),
                    "angle_rad": float(angle),
                    "center_distance": center_distance,
                    "surface_z": float(surface_z),
                    "depth_candidate": float(depth),
                    "target_depth": float(surface_z - z),
                    "full_plan_ok": None,
                    "failed_stage": "",
                    "failure_reason": "",
                }
                if not ok:
                    row = dict(base_row)
                    row.update({"planned": False, "score": -1.0e9, "reason": reason})
                    rows.append(row)
                    continue

                density_count = 0
                if settled_particles is not None and len(settled_particles) > 0:
                    density_count = sand_snapshot_density_count(snapshot, x, y, float(AUTO_DIG_DENSITY_RADIUS))
                    if density_count <= 0:
                        dxy = np.linalg.norm(settled_particles[:, :2] - np.array([[x, y]], dtype=np.float32), axis=1)
                        density_count = int(np.count_nonzero(dxy <= float(AUTO_DIG_DENSITY_RADIUS)))
                swept_density_count = auto_dig_swept_density_count(target, surface_z, particles=settled_particles, snapshot=snapshot)
                effective_density_count = max(int(density_count), int(swept_density_count))
                min_local = int(AUTO_DIG_MIN_LOCAL_PARTICLES)
                min_swept = int(AUTO_DIG_MIN_SWEPT_PARTICLES)
                if settled_particles is not None and len(settled_particles) > 0 and (
                    effective_density_count < min_local or int(swept_density_count) < min_swept
                ):
                    if int(swept_density_count) < min_swept:
                        reject_reason = f"low_swept_sand_density:{swept_density_count}"
                    else:
                        reject_reason = f"low_local_sand_density:{effective_density_count}"
                    row = dict(base_row)
                    row.update(
                        {
                            "density_count": density_count,
                            "swept_density_count": swept_density_count,
                            "effective_density_count": effective_density_count,
                            "planned": False,
                            "score": -1.0e9,
                            "reason": reject_reason,
                        }
                    )
                    rows.append(row)
                    continue

                density_score = clamp01(effective_density_count / 320.0)
                swept_density_score = clamp01(swept_density_count / 320.0)
                depth_score = clamp01((surface_z - z) / max(0.01, max(AUTO_COLLECT_TARGET_DEPTHS)))
                center_score = clamp01(1.0 - float(norm) / max(0.01, max(AUTO_DIG_RING_RADII)))
                fill_potential_score = clamp01(0.58 * swept_density_score + 0.42 * depth_score)
                approach_quality = auto_dig_approach_quality(target, surface_z, snapshot=snapshot)
                approach_score = float(approach_quality.get("approach_score", 0.0))
                swing_goal = target_to_swing_angle(target)
                swing_delta_deg_abs = abs(rad_to_deg(swing_delta(swing_goal, q_ref[CTRL.name_to_idx["swing"]])))
                motion_score = clamp01(1.0 - swing_delta_deg_abs / 180.0)

                ik_ok, ik_reason, reach_info = auto_dig_fast_reach_check(target, q_seed=q_ref, end_effector="tip")

                reach_score = 1.0 if ik_ok else 0.0
                weights = AUTO_DIG_SCORE_WEIGHTS
                score_components = {
                    "reach": float(weights["reach"] * reach_score),
                    "fill": float(weights["fill"] * fill_potential_score),
                    "swept_density": float(weights["swept_density"] * swept_density_score),
                    "depth": float(weights["depth"] * depth_score),
                    "center": float(weights["center"] * center_score),
                    "motion": float(weights["motion"] * motion_score),
                    "approach": float(weights["approach"] * approach_score),
                }
                score = float(sum(score_components.values()))
                row = dict(base_row)
                row.update(
                    {
                        "density_count": density_count,
                        "swept_density_count": swept_density_count,
                        "effective_density_count": effective_density_count,
                        "density_score": float(density_score),
                        "swept_density_score": float(swept_density_score),
                        "depth_score": float(depth_score),
                        "fill_potential_score": float(fill_potential_score),
                        "motion_score": float(motion_score),
                        "center_score": float(center_score),
                        "approach_score": float(approach_score),
                        "approach_quality": approach_quality,
                        "fast_reach": reach_info,
                        "reach_score": float(reach_score),
                        "score_components": score_components,
                        "score": float(score),
                        "planned": bool(ik_ok),
                        "reason": ik_reason,
                    }
                )
                rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row.get("ring_index", 999)),
            0 if bool(row.get("planned", False)) else 1,
            -float(row.get("score", -1.0e9)),
            float(row.get("center_distance", 1.0e9)),
            -float(row.get("target_depth", 0.0) or 0.0),
        )
    )
    usable = [row for row in rows if bool(row.get("planned", False))]
    rank_ms = (time.perf_counter() - rank_t0) * 1000.0
    perf = dict(STATE.get("sand_perf_last", {}) or {})
    perf["target_rank_ms"] = float(rank_ms)
    perf["target_candidates"] = int(len(rows))
    perf["target_usable"] = int(len(usable))
    STATE["sand_perf_last"] = perf
    if not usable:
        STATE["last_auto_dig_target_scores"] = rows
        info_print(
            "[AUTO DIG TARGET SELECT]",
            f"candidates={len(rows)}",
            "usable=0",
            "selected_for_plan=0",
            "reason=no_candidate_with_swept_sand",
            f"target_rank_ms={rank_ms:.1f}",
        )
        return []
    STATE["last_auto_dig_target_scores"] = rows
    selected_usable = []
    per_ring_limit = max(1, int(AUTO_DIG_FULL_PLAN_TOPK_PER_RING))
    for ring_index in sorted({int(row.get("ring_index", 999)) for row in usable}):
        ring_rows = [row for row in usable if int(row.get("ring_index", 999)) == ring_index]
        selected_usable.extend(ring_rows[:per_ring_limit])
    info_print(
        "[AUTO DIG TARGET SELECT]",
        f"candidates={len(rows)}",
        f"usable={len(usable)}",
        f"selected_for_plan={len(selected_usable)}",
        f"best={usable[0].get('target_xyz')}",
        f"ring_index={usable[0].get('ring_index')}",
        f"center_distance={fmt_optional(usable[0].get('center_distance'))}",
        f"score={fmt_optional(usable[0].get('score'))}",
        f"depth={fmt_optional(usable[0].get('target_depth'))}",
        f"swept_density={usable[0].get('swept_density_count')}",
        f"target_rank_ms={rank_ms:.1f}",
        f"snapshot_ms={fmt_optional((snapshot.get('perf_ms') or {}).get('total') if isinstance(snapshot, dict) else None)}",
        f"reason={usable[0].get('reason')}",
    )
    return selected_usable[: max(1, len(AUTO_DIG_RING_RADII) * per_ring_limit)]


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
    STATE["auto_collect_prepare_failure_reason"] = ""
    gate_report = {"started_at": time.time(), "gates": []}
    STATE["auto_collect_prepare_gate_report"] = gate_report

    def record_gate(name, ok, reason="ok", detail=None):
        row = {
            "gate": str(name),
            "ok": bool(ok),
            "reason": str(reason or ("ok" if ok else "failed")),
        }
        if isinstance(detail, dict):
            row["detail"] = detail
        elif detail is not None:
            row["detail"] = {"value": str(detail)}
        gate_report["gates"].append(row)
        STATE["auto_collect_prepare_gate_report"] = gate_report
        return bool(ok)

    def fail_prepare(reason, gate="", detail=None, q_cmd=None):
        if gate:
            record_gate(gate, False, reason, detail=detail)
        gate_report["finished_at"] = time.time()
        gate_report["ok"] = False
        gate_report["reason"] = str(reason)
        STATE["auto_collect_prepare_failure_reason"] = str(reason)
        STATE["auto_collect_prepare_gate_report"] = gate_report
        info_print("[AUTO DATASET PREP FAILED]", reason)
        debug_timeline_record(
            "AUTO_PREP",
            result="failed",
            reason=str(reason),
            data={"gate_report": gate_report},
            q_cmd=q_cmd,
            include_sand=True,
        )
        STATE["dataset_recording"] = was_recording
        return False

    reset_dig_plan()
    if not simulation_timeline_is_playing():
        ensure_timeline_playing("auto_collect_prepare_start")
        await step_updates(2)
    timeline_ok = simulation_timeline_is_playing()
    record_gate("timeline", timeline_ok, "ok" if timeline_ok else "prepare_failed/timeline_not_playing")
    if not timeline_ok or handle_timeline_stop_if_needed("auto_collect_prepare_start"):
        return fail_prepare("prepare_failed/timeline_not_playing", gate="timeline")

    action_ready, action_reason, action_detail = await wait_for_articulation_action_ready(
        "auto_collect_prepare_action_channel",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_MAX_WAIT_FRAMES,
        record_failure=False,
    )
    record_gate(
        "action_channel",
        action_ready,
        "ok" if action_ready else "prepare_failed/action_channel_not_ready",
        detail={"reason": action_reason, "action_detail": action_detail},
    )
    if not action_ready:
        return fail_prepare(
            f"prepare_failed/action_channel_not_ready:{action_reason}",
            gate="action_channel",
            detail=action_detail,
        )

    physics_ok = object_physics_view_state(ROBOT) is not False
    record_gate("physics_view", physics_ok, "ok" if physics_ok else "prepare_failed/physics_view_missing")
    if not physics_ok:
        return fail_prepare("prepare_failed/physics_view_missing", gate="physics_view")

    if not ik_model_is_valid():
        update_status("[AUTO DATASET] calibrating IK", force=True)
        ok_ik = await calibrate_ik()
        if not ok_ik or not ik_model_is_valid():
            reason = "prepare_failed/ik_invalid"
            update_status(f"[AUTO DATASET] {reason}", force=True)
            return fail_prepare(
                reason,
                gate="ik_calibrated",
                detail={"ik_report": dict(STATE.get("ik_calibration_report", {}) or {})},
            )
    record_gate(
        "ik_calibrated",
        True,
        "ok",
        detail={"ik_report": dict(STATE.get("ik_calibration_report", {}) or {})},
    )

    task_id = start_task("auto_collect_prepare")
    task_ok = bool(task_alive(task_id))
    record_gate(
        "task_state",
        task_ok,
        "ok" if task_ok else "prepare_failed/task_state_desync",
        detail={"task_id": task_id, "active_task": STATE.get("active_task_name", "")},
    )
    if not task_ok:
        return fail_prepare(
            "prepare_failed/task_state_desync",
            gate="task_state",
            detail={"task_id": task_id, "active_task": STATE.get("active_task_name", "")},
        )
    q_home = safe_home_q()
    ok = await set_joint_pose_direct_and_settle(
        q_home,
        label="auto_collect_home",
        mode="auto_collect_home_direct",
        settle_frames=DIRECT_HOME_SETTLE_FRAMES,
        task_id=None,
    )
    if not ok:
        action_ready = robot_articulation_action_ready()
        reason = (
            "prepare_failed/home_direct_failed:"
            f"action_ready={action_ready}; "
            f"timeline={simulation_timeline_is_playing()}; "
            f"running={STATE.get('running', False)}; "
            f"active_task={STATE.get('active_task_name', '')}; "
            f"task_alive={task_alive(task_id)}; "
            f"q_home_deg={q_deg_values(q_home, wrap_swing_for_display=True)}"
        )
        return fail_prepare(reason, gate="home_pose", q_cmd=q_home)
    record_gate("home_pose", True, "ok", detail={"q_home_deg": q_deg_values(q_home, wrap_swing_for_display=True)})
    await step_updates(AUTO_COLLECT_PRE_RESET_SETTLE_FRAMES)
    if handle_timeline_stop_if_needed("auto_collect_prepare_after_home"):
        return fail_prepare("prepare_failed/timeline_stopped_after_home", gate="timeline")

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
                return fail_prepare("prepare_failed/timeline_stopped_before_sand_reset", gate="timeline")
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
            record_gate(
                "sand_settled",
                bool(reset_ok),
                "ok" if reset_ok else "prepare_failed/sand_not_settled",
                detail={"reset_attempted": True},
            )
            if not reset_ok:
                return fail_prepare("prepare_failed/sand_not_settled", gate="sand_settled")
        except Exception as e:
            info_print("[WARN] [AUTO DATASET] sand reset failed:", type(e).__name__, e)
            return fail_prepare(
                f"prepare_failed/sand_reset_exception:{type(e).__name__}:{e}",
                gate="sand_settled",
            )
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
        settle_snapshot = get_sand_snapshot(force=False, label="auto_prepare_sand_gate", max_age=1.0)
        settle = (
            settle_snapshot.get("settle")
            if isinstance(settle_snapshot, dict) and isinstance(settle_snapshot.get("settle"), dict)
            else sand_settle_status(points=settle_snapshot.get("points") if isinstance(settle_snapshot, dict) else None)
        )
        sand_ok = bool(settle.get("ok", False))
        record_gate(
            "sand_settled",
            sand_ok,
            "ok" if sand_ok else "prepare_failed/sand_not_settled",
            detail={"settle": settle},
        )
        if not sand_ok:
            return fail_prepare("prepare_failed/sand_not_settled", gate="sand_settled", detail=settle)

    STATE["dataset_recording"] = was_recording
    if handle_timeline_stop_if_needed("auto_collect_prepare_done"):
        return fail_prepare("prepare_failed/timeline_stopped_after_prepare", gate="timeline")
    if not bool(STATE.get("auto_collect_active", False)):
        return fail_prepare("prepare_failed/auto_collect_inactive_after_prepare", gate="task_state")
    gate_report["finished_at"] = time.time()
    gate_report["ok"] = True
    gate_report["reason"] = "ok"
    STATE["auto_collect_prepare_gate_report"] = gate_report
    return True


async def auto_collect_find_plan(attempt_index):
    return await auto_dataset_collect.find_plan(runtime_module(), attempt_index)


async def auto_collect_one_episode():
    attempt = int(STATE.get("auto_collect_attempts", 0)) + 1
    STATE["auto_collect_attempts"] = attempt
    update_status(f"[AUTO DATASET] episode {attempt} prepare", force=True)

    prepared = await auto_collect_prepare_environment()
    if not prepared:
        prepare_reason = str(
            STATE.pop("auto_collect_prepare_failure_reason", "")
            or "prepare_failed/prepare_environment_failed"
        )
        gate_report = STATE.get("auto_collect_prepare_gate_report", {}) or {}
        target = auto_collect_sample_target(attempt, 0)
        plan_attempts = [{
            "prepared": False,
            "failed_gate": str(gate_report.get("reason", prepare_reason)) if isinstance(gate_report, dict) else prepare_reason,
            "prepare_reason": prepare_reason,
            "prepare_gate_report": gate_report,
        }]
        info_print(
            "[AUTO DATASET ATTEMPT]",
            f"attempt={attempt}",
            f"result={prepare_reason}",
            "executed=False",
        )
        auto_collect_record_planning_diagnostic(
            attempt,
            target,
            plan_attempts,
            prepare_reason,
            initial_info=None,
        )
        return False

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
        active_name = str(STATE.get("active_task_name", "") or "")
        if active_name.startswith("auto_collect"):
            invalidate_active_task("auto_collect_loop_end")
        STATE["auto_collect_active"] = False
        STATE["auto_collect_stop_requested"] = False
        STATE["dataset_recording"] = previous_dataset["dataset_recording"]
        STATE["dataset_path"] = previous_dataset["dataset_path"]
        STATE["dataset_event_path"] = previous_dataset["dataset_event_path"]
        STATE["dataset_meta_path"] = previous_dataset["dataset_meta_path"]
        STATE["dataset_sand_metrics_path"] = previous_dataset["dataset_sand_metrics_path"]
        STATE["auto_collect_episode_sand_snapshot"] = None
        STATE["auto_collect_episode_sand_snapshot_time"] = 0.0
        clear_planning_runtime_caches("auto_collect_loop_end")
        auto_collect_write_run_summary()
        update_status(auto_collect_status_text(), force=True)


def request_auto_collect(count):
    if bool(STATE.get("auto_collect_active", False)):
        update_status("[AUTO DATASET] already running", force=True)
        return
    STATE["planning_cancel_requested"] = False
    task = register_async_task("auto_collect", auto_collect_loop(max(1, int(count))), replace=True)
    STATE["auto_collect_task"] = task


def stop_auto_collect():
    STATE["auto_collect_stop_requested"] = True
    STATE["trace_active_motion"] = None
    STATE["trace_no_plan_notice_shown"] = True
    STATE["trace_no_plan_notice_time"] = time.time()
    cancel_active_task("auto dataset stop requested")
    invalidate_active_task("auto_collect_stop_requested")
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


def dataset_q_delta(q_a, q_b):
    q_a = np.array(q_a, dtype=np.float32).reshape(-1)[:4]
    q_b = np.array(q_b, dtype=np.float32).reshape(-1)[:4]
    dq = q_a - q_b
    try:
        swing_idx = CTRL.name_to_idx["swing"]
        dq[swing_idx] = swing_delta(q_a[swing_idx], q_b[swing_idx])
    except Exception:
        pass
    return dq.astype(np.float32, copy=False)


def plan_joint_motion_metrics(q_to, q_from, duration=0.0):
    return ik_calculation.joint_motion_metrics(runtime_module(), q_to, q_from, duration=duration)


def reload_ik_calculation_module(reason=""):
    global ik_calculation
    module_name = getattr(ik_calculation, "__name__", "excavator_app.ik_calculation")
    try:
        if module_name in sys.modules:
            ik_calculation = importlib.reload(sys.modules[module_name])
        else:
            ik_calculation = importlib.import_module(module_name)
        info_print(
            "[MODULE RELOAD] ik_calculation",
            f"reason={reason}",
            f"module={module_name}",
            f"file={getattr(ik_calculation, '__file__', '')}",
        )
        return True
    except Exception as e:
        info_print("[WARN] [MODULE RELOAD] ik_calculation failed:", f"reason={reason}", type(e).__name__, e)
        return False


def plan_path_penalty_cache_key(q_start, q_goal, mode):
    try:
        qa = np.round(np.array(q_start, dtype=np.float32).reshape(-1)[:4], 4)
        qb = np.round(np.array(q_goal, dtype=np.float32).reshape(-1)[:4], 4)
        return (str(mode), tuple(float(x) for x in qa), tuple(float(x) for x in qb))
    except Exception:
        return None


def path_penalty_cacheable(detail):
    if not isinstance(detail, dict):
        return False
    reason_text = " ".join(
        [
            str(detail.get("phase_reason", "")),
            str(detail.get("obstacle_reason", "")),
        ]
    ).lower()
    return "planning budget exceeded" not in reason_text and "planning_deadline" not in reason_text


def compute_path_penalty_uncached(q_start, q_goal, mode, deadline=None):
    try:
        return ik_calculation.path_penalty(runtime_module(), q_start, q_goal, mode, deadline=deadline)
    except TypeError as e:
        if "deadline" not in str(e):
            raise
        info_print(
            "[WARN] [MODULE STALE] ik_calculation.path_penalty missing deadline; attempting reload",
            f"error={e}",
        )
        if reload_ik_calculation_module(reason="path_penalty_deadline_signature"):
            try:
                return ik_calculation.path_penalty(runtime_module(), q_start, q_goal, mode, deadline=deadline)
            except TypeError as e2:
                if "deadline" not in str(e2):
                    raise
                info_print(
                    "[WARN] [MODULE STALE] ik_calculation.path_penalty still missing deadline after reload; using fallback",
                    f"error={e2}",
                )
        return ik_calculation.path_penalty(runtime_module(), q_start, q_goal, mode)


def plan_path_penalty(q_start, q_goal, mode, deadline=None):
    cache_key = plan_path_penalty_cache_key(q_start, q_goal, mode)
    cache = STATE.setdefault("planning_path_penalty_cache", {})
    if bool(STATE.get("dig_plan_planning_active", False)) and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            STATE["planning_path_penalty_cache_hits"] = int(STATE.get("planning_path_penalty_cache_hits", 0)) + 1
            return float(cached[0]), dict(cached[1])

    penalty, detail = compute_path_penalty_uncached(q_start, q_goal, mode, deadline=deadline)
    STATE["planning_path_penalty_cache_misses"] = int(STATE.get("planning_path_penalty_cache_misses", 0)) + 1
    if (
        bool(STATE.get("dig_plan_planning_active", False))
        and cache_key is not None
        and not planning_deadline_exceeded(deadline)
        and path_penalty_cacheable(detail)
    ):
        if len(cache) >= int(PLANNING_PATH_PENALTY_CACHE_MAX):
            try:
                cache.pop(next(iter(cache)))
            except Exception:
                cache.clear()
        cache[cache_key] = (float(penalty), dict(detail))
    return float(penalty), detail


def planning_deadline_exceeded(deadline):
    source = str(STATE.get("dig_plan_planning_source", ""))
    if bool(STATE.get("planning_cancel_requested", False)) and source != "loaded_route_test":
        return True
    if bool(STATE.get("auto_collect_stop_requested", False)) and str(STATE.get("dig_plan_planning_source", "")) == "auto_collect":
        return True
    try:
        perf_deadline = STATE.get("dig_plan_active_perf_deadline", None)
        if perf_deadline is not None and time.perf_counter() > float(perf_deadline):
            return True
    except Exception:
        pass
    return deadline is not None and time.time() > float(deadline)


def child_planning_deadline(parent_deadline, max_seconds, min_seconds=0.05):
    if parent_deadline is None:
        return None
    try:
        now = time.time()
        parent_deadline = float(parent_deadline)
        remaining = max(0.0, parent_deadline - now)
        if remaining <= 0.0:
            return now
        window = max(float(min_seconds), min(float(max_seconds), remaining))
        return min(parent_deadline, now + window)
    except Exception:
        return parent_deadline


def perf_block_record(label, elapsed_ms, data=None, threshold_ms=None):
    try:
        elapsed_ms = float(elapsed_ms)
    except Exception:
        return
    threshold = float(PLANNER_SYNC_BLOCK_WARN_MS if threshold_ms is None else threshold_ms)
    if elapsed_ms < threshold:
        return
    payload = dict(data or {})
    payload["elapsed_ms"] = float(elapsed_ms)
    payload["threshold_ms"] = float(threshold)
    payload["planning_active"] = bool(STATE.get("dig_plan_planning_active", False))
    payload["auto_collect_active"] = bool(STATE.get("auto_collect_active", False))
    payload["stop_requested"] = bool(STATE.get("auto_collect_stop_requested", False))
    payload["path_penalty_cache_hits"] = int(STATE.get("planning_path_penalty_cache_hits", 0))
    payload["path_penalty_cache_misses"] = int(STATE.get("planning_path_penalty_cache_misses", 0))
    STATE["perf_block_count"] = int(STATE.get("perf_block_count", 0) or 0) + 1
    STATE["perf_block_last"] = {"label": str(label), **payload}
    info_print(
        "[PERF BLOCK]",
        f"label={label}",
        f"elapsed_ms={elapsed_ms:.1f}",
        f"threshold_ms={threshold:.1f}",
        f"planning={payload['planning_active']}",
        f"auto={payload['auto_collect_active']}",
        f"cache={payload['path_penalty_cache_hits']}/{payload['path_penalty_cache_misses']}",
    )
    try:
        debug_timeline_record(
            "PERF_BLOCK",
            stage=str(label),
            result="slow_sync",
            data=payload,
            include_sand=False,
        )
    except Exception:
        pass


def clear_planning_runtime_caches(reason=""):
    STATE["planning_path_penalty_cache"] = {}
    STATE["planning_path_penalty_cache_hits"] = 0
    STATE["planning_path_penalty_cache_misses"] = 0
    STATE["planning_swing_corridor_cache"] = {}
    STATE["planning_swing_corridor_cache_hits"] = 0
    STATE["planning_swing_corridor_cache_misses"] = 0
    STATE["planning_sand_snapshot"] = None
    STATE["planning_sand_snapshot_active"] = False
    if reason:
        STATE["planning_cache_clear_reason"] = str(reason)


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


def freeze_contact_detail(mode="freeze", light=False):
    details = []
    if light:
        try:
            bucket_min = bbox_min_z(BUCKET_LINK)
            if bucket_min is not None and float(bucket_min) < GROUND_TOP_Z - 0.02:
                details.append(f"bucket_below_ground={float(bucket_min):.3f}")
                if float(bucket_min) < GROUND_TOP_Z - MANUAL_RECOVERY_BUCKET_PENETRATION_Z:
                    details.append("bucket_penetration_locks_swing=True")
        except Exception as e:
            details.append(f"bucket_ground_check_failed={type(e).__name__}")
        if sand_site_active():
            details.append("sand_site_active=True")
        if not details:
            details.append("light_check_no_below_ground_bucket")
        return "; ".join(details)

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
    include_sand = bool(STATE.get("dataset_recording", False)) and not mode.lower().startswith("manual")
    debug_timeline_record(
        "FREEZE",
        stage=mode,
        result="stalled",
        reason=f"{reason}; {extra}",
        q_cmd=q_cmd,
        q_real=q_real,
        include_sand=include_sand,
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
    if is_sand_contact_phase(action_mode):
        STATE["freeze_candidate_since"] = 0.0
        try:
            q_cmd_contact = np.array(q_cmd, dtype=np.float32)
            q_real_contact = q_real_near_command(get_real_joint_positions(), q_cmd_contact)
            update_sand_contact_progress(
                action_mode,
                q_cmd=q_cmd_contact,
                q_real=q_real_contact,
                force=False,
                log=True,
            )
        except Exception:
            pass
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
        f"{freeze_contact_detail(label, light=action_mode.startswith('manual'))}"
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

        desired_lo, desired_hi = DESIRED_LIMITS_DEG[name]
        if LIMIT_POLICY == "override":
            needs_override = (
                not valid
                or abs(float(lo) - float(desired_lo)) > 1.0e-3
                or abs(float(hi) - float(desired_hi)) > 1.0e-3
            )
            if needs_override:
                lo_attr.Set(float(desired_lo))
                hi_attr.Set(float(desired_hi))
                info_print(
                    f"[JOINT LIMIT OVERRIDE] {name}: imported=({raw_lo},{raw_hi}) "
                    f"set=({desired_lo:.3f},{desired_hi:.3f})"
                )
            else:
                info_print(f"[JOINT LIMIT OK] {name}: lower={lo:.3f} upper={hi:.3f}")
            continue

        if valid:
            info_print(f"[JOINT LIMIT OK] {name}: lower={lo:.3f} upper={hi:.3f}")
            continue

        lo_attr.Set(float(desired_lo))
        hi_attr.Set(float(desired_hi))
        info_print(
            f"[JOINT LIMIT FIX] {name}: invalid imported lower/upper=({raw_lo},{raw_hi}) "
            f"set=({desired_lo:.3f},{desired_hi:.3f})"
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

        dump_motion = is_unload_dump_motion(mode=mode)
        q = (
            clip_unload_dump_command(q, reference=self.q_cmd)
            if dump_motion
            else clip_command_near(q, reference=self.q_cmd)
        )
        q = maybe_rebase_swing_for_bounded_joint(q)
        q_action = self.clip_action_limits(q)
        if dump_motion:
            bucket_idx = self.name_to_idx.get("bucket", 3)
            bucket_cmd = float(q[bucket_idx])
            bucket_action = float(q_action[bucket_idx])
            if abs(bucket_action - bucket_cmd) > deg_to_rad(0.25):
                info_print(
                    "[UNLOAD DUMP ACTION CLIP]",
                    f"cmd={rad_to_deg(bucket_cmd):.2f}deg",
                    f"action={rad_to_deg(bucket_action):.2f}deg",
                    force_log=True,
                )
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
        q = (
            clip_unload_dump_command(q_target, reference=self.q_cmd)
            if is_unload_dump_motion(mode=mode)
            else clip_command_near(q_target, reference=self.q_cmd)
        )
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
        "lift_carry" not in level_mode
        and (
            ("unload_to_bin" in level_mode)
            or ("clearance_route_post" in level_mode)
            or ("carry" in level_mode and "unload" not in level_mode)
        )
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
                if mode_requires_loaded_carry_bucket(mode):
                    q = force_loaded_carry_bucket_q(q, reference=q, label=mode)
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


def current_dig_plan_contract_status():
    plan = STATE.get("current_dig_plan")
    if not isinstance(plan, dict):
        return True, {}
    contract = plan.get("fsm_contract")
    if not isinstance(contract, dict):
        return True, {}
    return bool(contract.get("ok", True)), contract


def block_invalid_dig_plan_contract(task_label="dig_plan"):
    ok, contract = current_dig_plan_contract_status()
    if ok:
        return False
    plan = STATE.get("current_dig_plan")
    if (
        isinstance(plan, dict)
        and bool(plan.get("staged_execution", False))
        and bool(plan.get("staged_prefix_ready", False))
    ):
        missing = contract.get("missing", []) if isinstance(contract, dict) else []
        info_print(
            "[DIG PLAN CONTRACT STAGED]",
            f"{task_label}: allowing executable dig prefix",
            f"missing_later_phases={missing}",
        )
        return False
    reasons = contract.get("reasons", []) if isinstance(contract, dict) else []
    reason_text = "planning_failed/fsm_contract_invalid:" + ";".join(str(x) for x in reasons[:4])
    set_execution_failure_reason(reason_text)
    update_status(f"[DIG PLAN CONTRACT FAILED] {task_label}: {reason_text}", force=True)
    debug_timeline_record(
        "PLAN_CONTRACT",
        stage=str(task_label),
        result="failed",
        reason=reason_text,
        data=contract,
        include_sand=False,
    )
    return True


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
    if is_sand_contact_phase(str(label or mode)):
        report = STATE.get("sand_contact_last_report")
        report = report if isinstance(report, dict) else {}
        reason = (
            f"execution_failed/no_material_progress:{label}:"
            f"bucket_total={int(report.get('total_bucket_delta', 0) or 0)};"
            f"pile_total={int(report.get('total_pile_delta', 0) or 0)};"
            f"spill_total={int(report.get('total_spill_delta', 0) or 0)};"
            f"tip_total={float(report.get('total_tip_delta', 0.0) or 0.0):.3f};"
            f"progress_age={float(report.get('progress_age', 0.0) or 0.0):.2f};"
            f"last_reach_detail={last_detail}"
        )
        set_execution_failure_reason(reason)
        debug_timeline_record(
            "SAND_CONTACT_NO_PROGRESS",
            stage=str(label or mode),
            result="failed",
            reason=reason,
            data=report,
            include_sand=True,
        )
        return False
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
    is_dump_motion = is_unload_dump_motion(mode=mode, label=label)
    q1 = clip_unload_dump_command(q_goal, reference=q0) if is_dump_motion else clip_command_near(q_goal, reference=q0)
    if is_dump_motion:
        bucket_idx = CTRL.name_to_idx.get("bucket", 3)
        info_print(
            "[UNLOAD DUMP COMMAND]",
            f"label={label}",
            f"mode={mode}",
            f"bucket_start={rad_to_deg(float(q0[bucket_idx])):.2f}deg",
            f"bucket_target={rad_to_deg(float(q1[bucket_idx])):.2f}deg",
            f"bucket_delta={rad_to_deg(float(q1[bucket_idx] - q0[bucket_idx])):.2f}deg",
            force_log=True,
        )
    if "lift_carry" in str(label or mode).lower():
        bucket_idx = CTRL.name_to_idx.get("bucket", 3)
        q1[bucket_idx] = float(q0[bucket_idx])
        q1 = CTRL.clip_limits(q1)
    if mode_requires_loaded_carry_bucket(mode, label):
        q1 = force_loaded_carry_bucket_q(q1, reference=q0, label=label or mode)
    q_final_cmd = q1.copy()
    contact_stage_name = str(label or mode)
    if is_sand_contact_phase(contact_stage_name):
        start_sand_contact_stage(contact_stage_name)

    sm = get_speed_multiplier()
    apply_speed_to_physx_joint_limits()

    seconds_eff = estimate_stage_motion_seconds(q0, q1, requested_seconds=seconds)
    if mode_requires_loaded_carry_bucket(mode, label):
        loaded_motion_text = f"{mode} {label}".lower()
        loaded_floor = (
            float(LOADED_ROUTE_FINAL_STAGE_SECONDS)
            if "unload_to_bin" in loaded_motion_text and "clearance_route_post" not in loaded_motion_text
            else float(LOADED_ROUTE_MIN_STAGE_SECONDS)
        )
        seconds_eff = max(float(seconds_eff), loaded_floor)
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
    stage_text = f"{label} {mode}".lower()
    auto_carry_bucket = (
        "lift_carry" not in stage_text
        and (
            ("unload_to_bin" in level_mode)
            or ("clearance_route_post" in level_mode)
            or ("carry" in level_mode and "unload" not in level_mode)
        )
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
    carry_loaded_limit_logged = False

    for i in range(steps):
        if motion_cancel_requested(task_id):
            update_status(f"[MOVE STOPPED] {label}", force=True)
            return False

        u = float(i + 1) / steps
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        q = interpolate_q_motion(q0, q1, s, mode=mode, label=label)
        if carry_bucket_world_rad is not None:
            carry_calc = bucket_joint_for_world_angle(q, carry_bucket_world_rad, end_effector="load")
            if carry_calc is not None:
                raw_bucket = float(carry_calc["bucket"])
                raw_bucket_deg = rad_to_deg(raw_bucket)
                skip_auto_carry_adjust = False
                if bucket_is_dump_branch_for_carry(raw_bucket_deg):
                    carry_limited = True
                    skip_auto_carry_adjust = True
                    if not carry_loaded_limit_logged:
                        carry_loaded_limit_logged = True
                        info_print(
                            "[LOADED BUCKET LIMIT]",
                            f"stage={label}",
                            f"requested={raw_bucket_deg:.2f}deg",
                            f"max_carry_dump_branch={float(BUCKET_CARRY_MAX_DUMP_BRANCH_DEG):.2f}deg",
                            "reason=avoid_dump_branch_during_carry",
                        )
                # Keep the planned/interpolated bucket command when the world
                # angle branch would open the bucket. Dumping is allowed only in
                # dump_bucket_at_target(), not while carrying.
                if not skip_auto_carry_adjust:
                    q[CTRL.name_to_idx["bucket"]] = carry_calc["bucket"]
                    q = CTRL.clip_limits(q)
                    q, loaded_limited, old_bucket_deg = apply_loaded_bucket_closed_limit(q, label=label)
                    if mode_requires_loaded_carry_bucket(mode, label):
                        before_force_deg = rad_to_deg(float(q[CTRL.name_to_idx["bucket"]]))
                        q = force_loaded_carry_bucket_q(q, reference=q, label=label or mode)
                        after_force_deg = rad_to_deg(float(q[CTRL.name_to_idx["bucket"]]))
                        if abs(after_force_deg - before_force_deg) > 0.25:
                            carry_limited = True
                            if not carry_loaded_limit_logged:
                                carry_loaded_limit_logged = True
                                info_print(
                                    "[LOADED BUCKET LIMIT]",
                                    f"stage={label}",
                                    f"requested={before_force_deg:.2f}deg",
                                    f"forced_to={after_force_deg:.2f}deg",
                                    "reason=loaded_carry_contract",
                                )
                    if loaded_limited:
                        carry_limited = True
                        if not carry_loaded_limit_logged:
                            carry_loaded_limit_logged = True
                            info_print(
                                "[LOADED BUCKET LIMIT]",
                                f"stage={label}",
                                f"requested={old_bucket_deg:.2f}deg",
                                f"limited_to={float(BUCKET_LOADED_CLOSED_LIMIT_DEG):.2f}deg",
                                "reason=loaded_carry_executable_limit",
                            )
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
        if motion_cancel_requested(task_id):
            update_status(f"[MOVE STOPPED] {label}", force=True)
            return False
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
            bad_cut, bad_reason = sand_contact_bad_cut_geometry(contact_stage_name, contact_report)
            if bad_cut:
                reason_text = f"execution_failed/bad_cut_geometry:{contact_stage_name}:{bad_reason}"
                set_execution_failure_reason(reason_text)
                info_print(
                    "[DIG EXEC FAILED]",
                    f"label={label}",
                    "reason=bad_cut_geometry",
                    bad_reason,
                )
                debug_timeline_record(
                    "BAD_CUT_GEOMETRY",
                    stage=contact_stage_name,
                    result="failed",
                    reason=bad_reason,
                    q_cmd=CTRL.q_cmd.copy(),
                    q_real=q_contact_real,
                    data={
                        "bucket_total": int(contact_report.get("total_bucket_delta", 0) or 0),
                        "pile_total": int(contact_report.get("total_pile_delta", 0) or 0),
                        "spill_total": int(contact_report.get("total_spill_delta", 0) or 0),
                    },
                    include_sand=True,
                )
                return False
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
        if not is_sand_contact_phase(contact_stage_name):
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
        dataset_record_event(
            "sand_contact_advance",
            f"stage={label}; mode={mode}; reason={contact_advance_reason}",
        )
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
IK_CALIBRATION_MIN_DELTA_RAD = 0.018
IK_CALIBRATION_REACH_TOL_DEG = 2.0
IK_CALIBRATION_RESTORE_TOL_DEG = 2.0
IK_CALIBRATION_SAFE_SETTLE_FRAMES = 24
IK_SANITY_MAX_POS_ERR_M = 0.45
IK_MIN_SEGMENT_LENGTH = 1e-4
IK_BUCKET_CANDIDATE_SPAN_DEG = 70.0
IK_BUCKET_CANDIDATE_COUNT = 17
DIG_IK_BUCKET_CANDIDATE_SPAN_DEG = 170.0
DIG_IK_BUCKET_CANDIDATE_COUNT = 17
IK_COST_WEIGHTS = np.array([0.35, 0.45, 0.55, 4.50], dtype=np.float32)
DIG_PLAN_BEAM_SIZE = 3
DIG_PLAN_TOPK_IK = 2
DIG_PLAN_MAX_CANDIDATES = 10
DIG_PLAN_PATH_CHECK_SAMPLES = 6
DIG_PLAN_MAX_BUILD_SECONDS = 10.0
PLANNER_SYNC_BLOCK_WARN_MS = 250.0
AUTO_COLLECT_FIND_PLAN_MAX_SECONDS = 30.0
AUTO_COLLECT_CANDIDATE_PLAN_SECONDS = 10.0
AUTO_COLLECT_CANDIDATE_HARD_BUDGET_GRACE_SECONDS = 0.50
AUTO_COLLECT_MAX_FULL_PLAN_ATTEMPTS = 2
AUTO_COLLECT_PLANNING_SNAPSHOT_MAX_AGE = 6.0
PLANNING_PATH_PENALTY_CACHE_MAX = 256
PLANNING_SWING_CORRIDOR_CACHE_MAX = 128
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
IK_GOAL_OBSTACLE_COST = 260.0
DIG_MAX_TIP_DEPTH = 0.28
DIG_MAX_CURL_DEPTH = 0.14
DIG_MAX_BUCKET_BODY_DEPTH = 0.10
DIG_ARM_MIN_CLEARANCE = 0.03
PATH_CHECK_SAMPLES = 24
PATH_CLEARANCE_BODY_Z = 0.06
PATH_CLEARANCE_END_Z = 0.18
PATH_CLEARANCE_HEIGHTS = [0.22, 0.38, 0.60, 0.85, 1.15, 1.55, 2.05, 2.75, 3.45]
PATH_FAST_SIDE_HEIGHTS = [0.38, 0.75, 1.10]
PATH_FAST_SIDE_PADDING_XY = 0.48
PATH_FAST_CORNER_EXTRA_PADS_XY = [0.35, 0.75, 1.15]
PATH_FAST_SIDE_MAX_BLOCKERS = 3
PATH_CLEARANCE_FRACTIONS = [0.25, 0.40, 0.60, 0.75]
PATH_CLEARANCE_DURATION = 0.85
PATH_OBSTACLE_MARGIN_XY = 0.18
PATH_OBSTACLE_MARGIN_Z = 0.10
PATH_OBSTACLE_OVER_CLEARANCE_Z = 0.45
UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z = 0.055
UNLOAD_BIN_WALL_EXEC_EXTRA_CLEARANCE_Z = 0.28
PATH_OBSTACLE_CACHE_SECONDS = 2.50
PATH_OBSTACLE_MESH_PROXY_MIN_AREA = 1.0e-4
PATH_OBSTACLE_MESH_PROXY_MAX_FACES = 32
PATH_ROUTE_SIDE_OFFSETS = [0.65, 1.10, 1.65, 2.25]
PATH_ROUTE_PLANNING_SAMPLE_COUNT = 6
PATH_RRT_MAX_ITERS = 260
PATH_RRT_MAX_ITERS_WITH_DEADLINE = 70
PATH_RRT_STEP_DEG = 13.0
PATH_RRT_GOAL_BIAS = 0.18
PATH_RRT_JOINT_WEIGHTS = [1.15, 1.0, 0.9, 0.7]
PATH_RRT_SMOOTH_ROUNDS = 24
PATH_RRT_SMOOTH_ROUNDS_WITH_DEADLINE = 4
PATH_RRT_CONNECT_STEPS_WITH_DEADLINE = 12
PATH_RRT_SMOOTH_ALPHA = 0.55
PATH_RRT_SMOOTH_BEND_WEIGHT = 0.35
PATH_RRT_SMOOTH_MIN_IMPROVEMENT = 1.0e-4
PATH_LINK_COLLISION_SEGMENT_SAMPLES = 4
PATH_LINK_COLLISION_RADIUS_M = 0.10
LOADED_ROUTE_TEST_PLAN_BUDGET_SECONDS = 30.0
LOADED_ROUTE_MIN_STAGE_SECONDS = 2.40
LOADED_ROUTE_FINAL_STAGE_SECONDS = 2.80
LOADED_ROUTE_RUNTIME_FAST_EXEC = True
LOADED_ROUTE_STAGE_PARTICLE_DIAGNOSTICS = False
PATH_DETERMINISTIC_ROUTE_POSES_DEG = [
    {"boom": 72.0, "arm": -88.0, "bucket": -56.0},
    {"boom": 72.0, "arm": -88.0, "bucket": -46.0},
    {"boom": 64.0, "arm": -76.0, "bucket": -34.0},
    {"boom": 56.0, "arm": -62.0, "bucket": -20.0},
]
PATH_DETERMINISTIC_APPROACH_LIFTS_DEG = [8.0, 14.0, 22.0, 32.0]
PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG = [-4.0, -10.0, -18.0, 4.0]
PATH_DETERMINISTIC_SWING_DETOURS_DEG = [0.0, 85.0, -85.0, 120.0, -120.0, 55.0, -55.0, 28.0, -28.0]
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


def active_loaded_route_fast_exec(stage_name=None):
    if not bool(LOADED_ROUTE_RUNTIME_FAST_EXEC):
        return False
    if str(STATE.get("active_task_name", "")) != "loaded_unload_route_test":
        return False
    if stage_name is None:
        return True
    text = str(stage_name)
    return text.startswith("clearance_route_post") or "unload" in text
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
    "/World/truck",
    "/truck",
]
PATH_VISUAL_OBSTACLE_PATHS = {"/World/truck", "/truck"}
PATH_OBSTACLE_EXCLUDE_TOKENS = [
    "/controlrig",
    "/realsandparticles",
    "/particlesystem",
    "/target",
    "/tracepath",
    "rangeguide",
    "selectedrange",
    "parkingground",
    "groundplane",
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


def strict_path_precheck_phase(mode):
    m = str(mode).lower()
    if is_sand_contact_phase(m) or is_curl_phase(m):
        return False
    if "dump" in m:
        return False
    return True


def is_calibrate_phase(mode):
    return "calibrate" in str(mode).lower()


def sand_surface_query_at_xy(x, y):
    api = get_sand_site_api()
    ctx = task_scene_context()
    center = np.array(ctx.get("pile_center", [0.0, 0.0, GROUND_TOP_Z]), dtype=np.float32).reshape(-1)[:3]
    radius = np.array(ctx.get("pile_radius", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
    if len(radius) >= 2 and float(radius[0]) > 0.0 and float(radius[1]) > 0.0:
        if abs(float(x) - float(center[0])) > float(radius[0]) or abs(float(y) - float(center[1])) > float(radius[1]):
            return None, "outside"

    snapshot = STATE.get("planning_sand_snapshot") if bool(STATE.get("planning_sand_snapshot_active", False)) else None
    if isinstance(snapshot, dict):
        z_snapshot = sand_snapshot_surface_height(snapshot, x, y)
        if z_snapshot is not None:
            return float(z_snapshot), "planning_particle_snapshot"

    snapshot = get_sand_snapshot(force=False, label="surface_query", max_age=0.75)
    z_snapshot = sand_snapshot_surface_height(snapshot, x, y)
    if z_snapshot is not None:
        return float(z_snapshot), "particle_snapshot"

    fn = api.get("particle_surface_height_fn") if isinstance(api, dict) else None
    if callable(fn):
        try:
            z = float(fn(float(x), float(y)))
            if math.isfinite(z):
                return z, "particle_fn"
        except Exception:
            pass

    particles = sand_particle_positions()
    settled = sand_settle_status(points=particles, ctx=ctx)
    settled_particles = filter_settled_sand_particles(particles, ctx=ctx) if bool(settled.get("ok", False)) else None
    if settled_particles is not None and len(settled_particles) > 0:
        dxy = np.linalg.norm(
            settled_particles[:, :2] - np.array([[float(x), float(y)]], dtype=np.float32),
            axis=1,
        )
        near = settled_particles[dxy <= float(SAND_SURFACE_QUERY_RADIUS)]
        if len(near) > 0:
            _, _, floor_z, _fill_height, _expected_min, z_expected_max = sand_pile_geometry_from_context(ctx)
            z = float(np.percentile(near[:, 2], 90.0))
            z = max(float(floor_z) + 0.02, min(z, float(z_expected_max)))
            return z, "particle_p90"

    try:
        _sx, _sy, floor_z, _fill_height, _expected_min, z_expected_max = sand_pile_geometry_from_context(ctx)
        return float(max(float(floor_z) + 0.10, min(float(floor_z) + 0.18, float(z_expected_max)))), "fallback_floor_no_particle_surface"
    except Exception:
        if len(center) >= 3:
            return float(center[2]), "fallback_center_no_particle_surface"
    return None, "missing_particle_surface"


def sand_surface_z_at_xy(x, y):
    z, _source = sand_surface_query_at_xy(x, y)
    return z


def sand_depth_for_point(point):
    if point is None:
        return None, None, "missing_point"
    try:
        p = np.array(point, dtype=np.float32).reshape(-1)
        if len(p) < 3:
            return None, None, "invalid_point"
        surface_z, source = sand_surface_query_at_xy(float(p[0]), float(p[1]))
        if surface_z is None:
            return None, None, str(source)
        return max(0.0, float(surface_z) - float(p[2])), float(surface_z), str(source)
    except Exception:
        return None, None, "error"


def phase_ground_report(mode):
    tip = bucket_tip_pos()
    load = bucket_load_pos()
    pour = bucket_pour_pos()
    bucket_mid = bucket_mid_pos()
    tip_sand_depth, tip_sand_surface, tip_sand_source = sand_depth_for_point(tip)
    load_sand_depth, load_sand_surface, load_sand_source = sand_depth_for_point(load)
    pour_sand_depth, pour_sand_surface, pour_sand_source = sand_depth_for_point(pour)
    mid_sand_depth, mid_sand_surface, mid_sand_source = sand_depth_for_point(bucket_mid)
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
        "tip_sand_surface_source": tip_sand_source,
        "load_sand_surface_source": load_sand_source,
        "pour_sand_surface_source": pour_sand_source,
        "bucket_mid_sand_surface_source": mid_sand_source,
    }


def min_existing(values):
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return min(valid)


def predicted_chain_world_z(q, end_effector="tip", reference_q=None):
    pts = predicted_chain_world_points(q, end_effector=end_effector, reference_q=reference_q)
    if pts is None:
        return None
    return [float(p[2]) for p in pts]


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


def predicted_phase_ground_report(q, mode, reference_q=None):
    tip_pts = predicted_chain_world_points(q, "tip", reference_q=reference_q)
    mid_pts = predicted_chain_world_points(q, "mid", reference_q=reference_q)
    load_pts = predicted_chain_world_points(q, "load", reference_q=reference_q)
    pour_pts = predicted_chain_world_points(q, "pour", reference_q=reference_q)

    def point_at(points, idx):
        if points is None or len(points) <= idx:
            return None
        return points[idx]

    def z_at(points, idx):
        p = point_at(points, idx)
        if p is None:
            return None
        return float(p[2])

    boom_root_z = z_at(tip_pts, 0)
    arm_joint_z = z_at(tip_pts, 1)
    bucket_joint_z = z_at(tip_pts, 2)
    tip_z = z_at(tip_pts, 3)
    mid_z = z_at(mid_pts, 3)
    load_z = z_at(load_pts, 3)
    pour_z = z_at(pour_pts, 3)
    tip_point = point_at(tip_pts, 3)
    mid_point = point_at(mid_pts, 3)
    load_point = point_at(load_pts, 3)
    pour_point = point_at(pour_pts, 3)
    tip_sand_depth, tip_sand_surface, tip_sand_source = sand_depth_for_point(tip_point)
    load_sand_depth, load_sand_surface, load_sand_source = sand_depth_for_point(load_point)
    pour_sand_depth, pour_sand_surface, pour_sand_source = sand_depth_for_point(pour_point)
    mid_sand_depth, mid_sand_surface, mid_sand_source = sand_depth_for_point(mid_point)

    base_path = LINK_PATHS.get("base_link")

    return {
        "mode": str(mode),
        "reference_q_deg": None if reference_q is None else q_deg_values(reference_q, wrap_swing_for_display=True),
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
        "tip_sand_surface_source": tip_sand_source,
        "load_sand_surface_source": load_sand_source,
        "pour_sand_surface_source": pour_sand_source,
        "bucket_mid_sand_surface_source": mid_sand_source,
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
    ref_q = report.get("reference_q_deg") if isinstance(report, dict) else None
    ref_text = "" if ref_q is None else f" ref_q={ref_q}"
    surface_source = report.get("tip_sand_surface_source") if isinstance(report, dict) else None
    source_text = "" if surface_source is None else f" surface_source={surface_source}"
    return (
        f"phase={mode} kind={phase_kind} {reason}; "
        f"tip_z={fmt_optional(report.get('tip_z'))} tip_hard_depth={fmt_optional(tip_hard_depth)} tip_sand_depth={fmt_optional(report.get('tip_sand_depth'))} "
        f"load_z={fmt_optional(report.get('load_z'))} load_hard_depth={fmt_optional(load_hard_depth)} load_sand_depth={fmt_optional(report.get('load_sand_depth'))} "
        f"pour_z={fmt_optional(report.get('pour_z'))} pour_hard_depth={fmt_optional(pour_hard_depth)} pour_sand_depth={fmt_optional(report.get('pour_sand_depth'))} "
        f"bucket_mid_z={fmt_optional(report.get('bucket_mid_z'))} mid_hard_depth={fmt_optional(mid_hard_depth)} mid_sand_depth={fmt_optional(report.get('bucket_mid_sand_depth'))} "
        f"sand_surface_z={fmt_optional(report.get('tip_sand_surface_z'))} "
        f"{source_text} "
        f"bucket_min={fmt_optional(report.get('bucket_min'))} "
        f"arm_min={fmt_optional(report.get('arm_min'))} boom_min={fmt_optional(report.get('boom_min'))}"
        f"{ref_text}"
    )


def log_phase_ground(prefix, mode):
    report = phase_ground_report(mode)
    info_print(f"{prefix} " + format_ground_report(mode, report, "check"))


def log_predicted_phase_ground(prefix, mode, q, reference_q=None):
    report = predicted_phase_ground_report(q, mode, reference_q=reference_q)
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


def cut_front_edge_quality(mode, report):
    if not is_cutting_phase(mode):
        return True, "ok", 0.0
    if not isinstance(report, dict):
        return False, "missing cut quality report", 100.0
    tip_depth = report.get("tip_sand_depth")
    if tip_depth is None:
        return False, "missing tip sand depth", 100.0
    tip_depth = float(tip_depth)
    body_depths = [
        float(v)
        for v in [
            report.get("bucket_mid_sand_depth"),
            report.get("pour_sand_depth"),
            report.get("load_sand_depth"),
        ]
        if v is not None
    ]
    if not body_depths:
        return True, "ok", 0.0
    deepest_body = max(body_depths)
    body_over_tip = float(deepest_body - tip_depth)
    mode_l = str(mode).lower()
    if "pull_exit_cut" in mode_l:
        if deepest_body > float(EXIT_BODY_DEPTH_HARD_MARGIN):
            return False, f"exit bucket body still too deep: body={deepest_body:.3f}", 90.0
        exit_penalty = max(0.0, deepest_body - float(EXIT_BODY_DEPTH_SOFT_MARGIN)) * 18.0
        return True, f"exit_ok tip={tip_depth:.3f} body={deepest_body:.3f}", float(exit_penalty)
    if tip_depth < float(FRONT_EDGE_MIN_TIP_DEPTH):
        return False, f"front tip not engaged enough: tip={tip_depth:.3f}", 80.0
    if body_over_tip > float(FRONT_EDGE_BODY_DEPTH_HARD_MARGIN):
        return False, f"bucket body much deeper than front tip: tip={tip_depth:.3f} body={deepest_body:.3f}", 100.0
    attack_penalty = max(0.0, body_over_tip - float(FRONT_EDGE_BODY_DEPTH_SOFT_MARGIN)) * 28.0
    return True, f"front_edge_ok tip={tip_depth:.3f} body={deepest_body:.3f} body_over_tip={body_over_tip:.3f}", float(attack_penalty)


def path_phase_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None):
    t0 = time.perf_counter()
    samples = max(2, int(samples))
    last_report = None

    try:
        for i in range(1, samples + 1):
            if planning_deadline_exceeded(deadline):
                return False, "planning budget exceeded", i, last_report
            s = float(i) / float(samples)
            q = interpolate_q_shortest(q_start, q_goal, s)
            report = predicted_phase_ground_report(q, mode, reference_q=q_start)
            last_report = report

            if report.get("tip_z") is None:
                return False, "missing predicted phase report", i, report

            ok, reason = phase_ground_ok(mode, report)
            if not ok:
                return False, reason, i, report

        return True, "ok", samples, last_report
    finally:
        perf = dict(STATE.get("sand_perf_last", {}) or {})
        total = float(perf.get("path_check_ms", 0.0) or 0.0)
        count = int(perf.get("path_check_count", 0) or 0)
        perf["path_check_ms"] = total + (time.perf_counter() - t0) * 1000.0
        perf["path_check_count"] = count + 1
        STATE["sand_perf_last"] = perf


def obstacle_path_excluded(path):
    text = str(path).lower()
    try:
        robot_base = str(ROBOT_BASE or "").lower()
        if robot_base and text.startswith(robot_base):
            return True
    except Exception:
        pass
    for token in PATH_OBSTACLE_EXCLUDE_TOKENS:
        if str(token).lower() in text:
            return True
    return False


def append_obstacle_bbox(bboxes, seen_paths, path, source="collision_api", collision_required=True):
    path = str(path)
    if not path or path in seen_paths or obstacle_path_excluded(path):
        return
    prim = get_prim(path)
    if not prim or not prim.IsValid():
        return
    if collision_required and not prim_collision_enabled(prim):
        return
    mn, mx = bbox_min_max(path)
    if mn is None or mx is None:
        return
    if not bbox_values_are_valid(mn, mx):
        return
    size = np.array(mx, dtype=np.float32) - np.array(mn, dtype=np.float32)
    if float(np.max(size)) <= 1.0e-4:
        return
    footprint = None
    footprint_source_vertices = 0
    proxy = "aabb"
    try:
        xy_points = mesh_world_xy_points_under(prim)
        hull = convex_hull_xy(xy_points) if xy_points is not None and len(xy_points) >= 3 else None
        if hull is not None and abs(polygon_signed_area(hull)) >= float(PATH_OBSTACLE_MESH_PROXY_MIN_AREA):
            footprint_source_vertices = int(len(hull))
            limited = visual_polygon_xy(hull, PATH_OBSTACLE_MESH_PROXY_MAX_FACES)
            if limited is not None and len(limited) >= 3 and abs(polygon_signed_area(limited)) >= float(PATH_OBSTACLE_MESH_PROXY_MIN_AREA):
                footprint = np.array(limited, dtype=np.float32).reshape(-1, 2)
                proxy = "mesh_footprint_prism"
    except Exception:
        footprint = None
        footprint_source_vertices = 0
        proxy = "aabb"
    seen_paths.add(path)
    row = {
        "path": path,
        "min": np.array(mn, dtype=np.float32),
        "max": np.array(mx, dtype=np.float32),
        "collision_required": bool(collision_required),
        "source": str(source),
        "proxy": proxy,
    }
    if footprint is not None:
        row["footprint_xy"] = footprint
        row["footprint_vertices"] = int(len(footprint))
        row["footprint_faces"] = int(len(footprint))
        row["footprint_source_vertices"] = int(footprint_source_vertices)
        row["footprint_max_faces"] = int(PATH_OBSTACLE_MESH_PROXY_MAX_FACES)
    bboxes.append(row)


def collect_collision_api_obstacles(bboxes, seen_paths):
    try:
        prim_iter = stage.Traverse()
    except Exception:
        return
    for prim in prim_iter:
        try:
            if not prim or not prim.IsValid():
                continue
            path = str(prim.GetPath())
            if obstacle_path_excluded(path):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim_collision_enabled(prim):
                continue
            append_obstacle_bbox(
                bboxes,
                seen_paths,
                path,
                source="collision_api",
                collision_required=False,
            )
        except Exception:
            continue


def rigid_obstacle_bboxes(force=False):
    now = time.time()
    if not force:
        cached = STATE.get("rigid_obstacle_cache")
        cached_time = float(STATE.get("rigid_obstacle_cache_time", 0.0) or 0.0)
        if cached is not None and (
            bool(STATE.get("dig_plan_planning_active", False))
            or bool(STATE.get("auto_collect_active", False))
        ):
            STATE["rigid_obstacle_cache_hits"] = int(STATE.get("rigid_obstacle_cache_hits", 0)) + 1
            return cached
        if cached is not None and now - cached_time <= PATH_OBSTACLE_CACHE_SECONDS:
            STATE["rigid_obstacle_cache_hits"] = int(STATE.get("rigid_obstacle_cache_hits", 0)) + 1
            return cached

    STATE["rigid_obstacle_cache_misses"] = int(STATE.get("rigid_obstacle_cache_misses", 0)) + 1
    roots = ["/SandSite", "/World/SandSite"]
    bboxes = []
    seen_paths = set()
    collect_collision_api_obstacles(bboxes, seen_paths)
    for base_path in PATH_RIGID_OBSTACLE_PATHS:
        if str(base_path).startswith("/SandSite"):
            suffix = base_path[len("/SandSite"):]
            candidate_paths = [f"{root}{suffix}" if suffix.startswith("/") else f"{root}/{suffix}" for root in roots]
            collision_required = True
            source = "configured_collision_path"
        else:
            candidate_paths = [str(base_path)]
            collision_required = str(base_path) not in PATH_VISUAL_OBSTACLE_PATHS
            source = "visual_bbox_fallback" if not collision_required else "configured_collision_path"
        for path in candidate_paths:
            if source == "visual_bbox_fallback" and any(str(p).startswith(str(path).rstrip("/") + "/") for p in seen_paths):
                continue
            append_obstacle_bbox(
                bboxes,
                seen_paths,
                path,
                source=source,
                collision_required=collision_required,
            )
    STATE["rigid_obstacle_cache"] = bboxes
    STATE["rigid_obstacle_cache_time"] = now
    return bboxes


def clear_rigid_obstacle_cache(reason=""):
    STATE["rigid_obstacle_cache"] = None
    STATE["rigid_obstacle_cache_time"] = 0.0
    STATE["rigid_obstacle_cache_hits"] = 0
    STATE["rigid_obstacle_cache_misses"] = 0
    if reason:
        info_print("[OBSTACLE CACHE CLEAR]", f"reason={reason}")


def store_excavator_runtime_api():
    builtins._EXCAVATOR_RUNTIME = {
        "clear_rigid_obstacle_cache": clear_rigid_obstacle_cache,
    }


store_excavator_runtime_api()


def compact_obstacle_bbox(row):
    if not isinstance(row, dict):
        return {}
    mn = np.array(row.get("min", []), dtype=np.float32).reshape(-1)
    mx = np.array(row.get("max", []), dtype=np.float32).reshape(-1)
    if len(mn) < 3 or len(mx) < 3:
        return {"path": str(row.get("path", "")), "valid": False}
    center = 0.5 * (mn[:3] + mx[:3])
    size = mx[:3] - mn[:3]
    return {
        "path": str(row.get("path", "")),
        "source": str(row.get("source", "")),
        "proxy": str(row.get("proxy", "aabb")),
        "footprint_vertices": int(row.get("footprint_vertices", 0) or 0),
        "footprint_faces": int(row.get("footprint_faces", 0) or 0),
        "footprint_source_vertices": int(row.get("footprint_source_vertices", 0) or 0),
        "footprint_max_faces": int(row.get("footprint_max_faces", 0) or 0),
        "collision_required": bool(row.get("collision_required", False)),
        "min": debug_round_vec(mn[:3], 3),
        "max": debug_round_vec(mx[:3], 3),
        "center": debug_round_vec(center, 3),
        "size": debug_round_vec(size, 3),
    }


def planning_world_snapshot(force=False, max_obstacles=64):
    obstacles = rigid_obstacle_bboxes(force=force)
    rows = [compact_obstacle_bbox(x) for x in obstacles[:max(0, int(max_obstacles))]]
    particle_like = []
    for row in rows:
        path = str(row.get("path", "")).lower()
        if any(token in path for token in ("/realsandparticles", "/particlesystem", "sandparticles")):
            particle_like.append(row.get("path", ""))
    return {
        "collision_world_policy": "rigid_hard_avoid__sand_soft_contact",
        "rigid_obstacle_count": int(len(obstacles)),
        "reported_obstacle_count": int(len(rows)),
        "obstacles_truncated": bool(len(obstacles) > len(rows)),
        "rigid_obstacles": rows,
        "configured_rigid_paths": [str(x) for x in PATH_RIGID_OBSTACLE_PATHS],
        "visual_bbox_fallback_paths": [str(x) for x in sorted(PATH_VISUAL_OBSTACLE_PATHS)],
        "excluded_path_tokens": [str(x) for x in PATH_OBSTACLE_EXCLUDE_TOKENS],
        "sand_particles_in_rigid_world": particle_like,
        "sand_particles_treated_as_soft_contact": bool(not particle_like),
    }


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


def expanded_bbox_arrays(mn, mx, margin_xy=0.0, margin_z=0.0, radius=0.0):
    lo = np.array(mn, dtype=np.float32).copy()
    hi = np.array(mx, dtype=np.float32).copy()
    r = max(0.0, float(radius))
    lo[0] -= float(margin_xy) + r
    lo[1] -= float(margin_xy) + r
    hi[0] += float(margin_xy) + r
    hi[1] += float(margin_xy) + r
    lo[2] -= float(margin_z) + r
    hi[2] += float(margin_z) + r
    return lo, hi


def segment_intersects_expanded_bbox(a, b, mn, mx, margin_xy=0.0, margin_z=0.0, radius=0.0):
    p0 = np.array(a, dtype=np.float32).reshape(-1)[:3]
    p1 = np.array(b, dtype=np.float32).reshape(-1)[:3]
    lo, hi = expanded_bbox_arrays(mn, mx, margin_xy=margin_xy, margin_z=margin_z, radius=radius)
    d = p1 - p0
    tmin = 0.0
    tmax = 1.0
    for axis in range(3):
        if abs(float(d[axis])) < 1.0e-8:
            if float(p0[axis]) < float(lo[axis]) or float(p0[axis]) > float(hi[axis]):
                return False, None
            continue
        inv = 1.0 / float(d[axis])
        t1 = (float(lo[axis]) - float(p0[axis])) * inv
        t2 = (float(hi[axis]) - float(p0[axis])) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False, None
    hit_t = max(0.0, min(1.0, tmin))
    hit_point = p0 + hit_t * d
    return True, hit_point


def point_in_convex_polygon_xy(point_xy, poly, eps=1.0e-7):
    p = np.array(point_xy, dtype=np.float64).reshape(-1)[:2]
    poly = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        return False
    if polygon_signed_area(poly) < 0.0:
        poly = poly[::-1].copy()
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        e = b - a
        if float(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) < -float(eps):
            return False
    return True


def orient2d(a, b, c):
    a = np.array(a, dtype=np.float64).reshape(-1)[:2]
    b = np.array(b, dtype=np.float64).reshape(-1)[:2]
    c = np.array(c, dtype=np.float64).reshape(-1)[:2]
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect_xy(a0, a1, b0, b1, eps=1.0e-8):
    a0 = np.array(a0, dtype=np.float64).reshape(-1)[:2]
    a1 = np.array(a1, dtype=np.float64).reshape(-1)[:2]
    b0 = np.array(b0, dtype=np.float64).reshape(-1)[:2]
    b1 = np.array(b1, dtype=np.float64).reshape(-1)[:2]
    o1 = orient2d(a0, a1, b0)
    o2 = orient2d(a0, a1, b1)
    o3 = orient2d(b0, b1, a0)
    o4 = orient2d(b0, b1, a1)

    def on_segment(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - eps <= float(q[0]) <= max(float(p[0]), float(r[0])) + eps
            and min(float(p[1]), float(r[1])) - eps <= float(q[1]) <= max(float(p[1]), float(r[1])) + eps
        )

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a0, b0, a1):
        return True
    if abs(o2) <= eps and on_segment(a0, b1, a1):
        return True
    if abs(o3) <= eps and on_segment(b0, a0, b1):
        return True
    if abs(o4) <= eps and on_segment(b0, a1, b1):
        return True
    return False


def point_segment_distance_xy(point, a, b):
    p = np.array(point, dtype=np.float64).reshape(-1)[:2]
    a = np.array(a, dtype=np.float64).reshape(-1)[:2]
    b = np.array(b, dtype=np.float64).reshape(-1)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_segment_distance_xy(a0, a1, b0, b1):
    if segments_intersect_xy(a0, a1, b0, b1):
        return 0.0
    return min(
        point_segment_distance_xy(a0, b0, b1),
        point_segment_distance_xy(a1, b0, b1),
        point_segment_distance_xy(b0, a0, a1),
        point_segment_distance_xy(b1, a0, a1),
    )


def point_in_polygon_with_margin_xy(point_xy, poly, margin_xy=0.0):
    poly = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return False
    if point_in_convex_polygon_xy(point_xy, poly):
        return True
    margin = max(0.0, float(margin_xy))
    if margin <= 1.0e-6:
        return False
    for i in range(poly.shape[0]):
        if point_segment_distance_xy(point_xy, poly[i], poly[(i + 1) % poly.shape[0]]) <= margin:
            return True
    return False


def segment_intersects_polygon_with_margin_xy(a_xy, b_xy, poly, margin_xy=0.0):
    poly = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return False
    if point_in_convex_polygon_xy(a_xy, poly) or point_in_convex_polygon_xy(b_xy, poly):
        return True
    for i in range(poly.shape[0]):
        edge_a = poly[i]
        edge_b = poly[(i + 1) % poly.shape[0]]
        if segments_intersect_xy(a_xy, b_xy, edge_a, edge_b):
            return True
    margin = max(0.0, float(margin_xy))
    if margin <= 1.0e-6:
        return False
    for i in range(poly.shape[0]):
        if segment_segment_distance_xy(a_xy, b_xy, poly[i], poly[(i + 1) % poly.shape[0]]) <= margin:
            return True
    return False


def segment_z_overlap_subsegment(a, b, z_min, z_max, margin_z=0.0, radius=0.0):
    p0 = np.array(a, dtype=np.float32).reshape(-1)[:3]
    p1 = np.array(b, dtype=np.float32).reshape(-1)[:3]
    lo = float(z_min) - float(margin_z) - max(0.0, float(radius))
    hi = float(z_max) + float(margin_z) + max(0.0, float(radius))
    dz = float(p1[2] - p0[2])
    if abs(dz) < 1.0e-8:
        if float(p0[2]) < lo or float(p0[2]) > hi:
            return None
        return p0.copy(), p1.copy()
    t0 = (lo - float(p0[2])) / dz
    t1 = (hi - float(p0[2])) / dz
    if t0 > t1:
        t0, t1 = t1, t0
    t0 = max(0.0, float(t0))
    t1 = min(1.0, float(t1))
    if t0 > t1:
        return None
    return p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)


def point_inside_obstacle_proxy(point, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0):
    poly = obstacle.get("footprint_xy") if isinstance(obstacle, dict) else None
    if poly is None:
        return point_inside_expanded_bbox(
            point,
            obstacle["min"],
            obstacle["max"],
            margin_xy=margin_xy,
            margin_z=margin_z,
        )
    p = np.array(point, dtype=np.float32).reshape(-1)[:3]
    mn = obstacle["min"]
    mx = obstacle["max"]
    z_lo = float(mn[2]) - float(margin_z) - max(0.0, float(radius))
    z_hi = float(mx[2]) + float(margin_z) + max(0.0, float(radius))
    if float(p[2]) < z_lo or float(p[2]) > z_hi:
        return False
    return point_in_polygon_with_margin_xy(p[:2], poly, margin_xy=max(0.0, float(margin_xy) + float(radius)))


def segment_intersects_obstacle_proxy(a, b, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0):
    poly = obstacle.get("footprint_xy") if isinstance(obstacle, dict) else None
    if poly is None:
        return segment_intersects_expanded_bbox(
            a,
            b,
            obstacle["min"],
            obstacle["max"],
            margin_xy=margin_xy,
            margin_z=margin_z,
            radius=radius,
        )
    sub = segment_z_overlap_subsegment(
        a,
        b,
        float(obstacle["min"][2]),
        float(obstacle["max"][2]),
        margin_z=margin_z,
        radius=radius,
    )
    if sub is None:
        return False, None
    a_sub, b_sub = sub
    hit = segment_intersects_polygon_with_margin_xy(
        a_sub[:2],
        b_sub[:2],
        poly,
        margin_xy=max(0.0, float(margin_xy) + float(radius)),
    )
    if not hit:
        return False, None
    return True, 0.5 * (a_sub + b_sub)


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
        segment_samples = max(1, int(PATH_LINK_COLLISION_SEGMENT_SAMPLES))
        for a, b in zip(chain_points[:-1], chain_points[1:]):
            if a is None or b is None:
                continue
            pa = np.array(a, dtype=np.float32)
            pb = np.array(b, dtype=np.float32)
            for j in range(1, segment_samples):
                s = float(j) / float(segment_samples)
                points.append((1.0 - s) * pa + s * pb)

    unique = []
    seen = set()
    for p in points:
        key = tuple(round(float(v), 3) for v in p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def predicted_obstacle_check_segments(q, reference_q=None):
    segments = []
    for end_effector in ["tip", "mid", "load", "pour"]:
        chain_points = predicted_chain_world_points(q, end_effector=end_effector, reference_q=reference_q)
        if chain_points is None or len(chain_points) < 2:
            continue
        for idx, (a, b) in enumerate(zip(chain_points[:-1], chain_points[1:])):
            if a is None or b is None:
                continue
            pa = np.array(a, dtype=np.float32).reshape(-1)[:3]
            pb = np.array(b, dtype=np.float32).reshape(-1)[:3]
            if float(np.linalg.norm(pb - pa)) < 1.0e-5:
                continue
            if idx == 0:
                link_name = "boom"
            elif idx == 1:
                link_name = "arm"
            else:
                link_name = f"bucket_{end_effector}"
            segments.append((link_name, pa, pb))

    unique = []
    seen = set()
    for link_name, pa, pb in segments:
        key = (
            link_name,
            tuple(round(float(v), 3) for v in pa),
            tuple(round(float(v), 3) for v in pb),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((link_name, pa, pb))
    return unique


def format_obstacle_report(mode, report, reason):
    point = report.get("point") if isinstance(report, dict) else None
    segment = report.get("segment") if isinstance(report, dict) else None
    link_name = report.get("link_name") if isinstance(report, dict) else None
    bbox_min = report.get("bbox_min") if isinstance(report, dict) else None
    bbox_max = report.get("bbox_max") if isinstance(report, dict) else None
    point_text = "None" if point is None else f"({float(point[0]):.3f},{float(point[1]):.3f},{float(point[2]):.3f})"
    segment_text = ""
    if segment is not None:
        try:
            a, b = segment
            segment_text = (
                f" segment=({float(a[0]):.3f},{float(a[1]):.3f},{float(a[2]):.3f})"
                f"->({float(b[0]):.3f},{float(b[1]):.3f},{float(b[2]):.3f})"
            )
        except Exception:
            segment_text = " segment=unavailable"
    link_text = "" if not link_name else f" link={link_name}"
    min_text = "None" if bbox_min is None else f"({float(bbox_min[0]):.3f},{float(bbox_min[1]):.3f},{float(bbox_min[2]):.3f})"
    max_text = "None" if bbox_max is None else f"({float(bbox_max[0]):.3f},{float(bbox_max[1]):.3f},{float(bbox_max[2]):.3f})"
    obstacle = report.get("obstacle", "unknown") if isinstance(report, dict) else "unknown"
    source = report.get("obstacle_source", "") if isinstance(report, dict) else ""
    source_text = "" if not source else f" source={source}"
    proxy = report.get("obstacle_proxy", "") if isinstance(report, dict) else ""
    proxy_text = "" if not proxy else f" proxy={proxy}"
    faces = report.get("obstacle_footprint_faces", None) if isinstance(report, dict) else None
    source_vertices = report.get("obstacle_footprint_source_vertices", None) if isinstance(report, dict) else None
    footprint_text = ""
    try:
        if faces is not None and int(faces) > 0:
            footprint_text = f" footprint_faces={int(faces)} source_vertices={int(source_vertices or 0)}"
    except Exception:
        footprint_text = ""
    return (
        f"phase={mode} kind=rigid_obstacle {reason}; "
        f"obstacle={obstacle}{source_text}{proxy_text}{footprint_text}{link_text} point={point_text}{segment_text} "
        f"bbox_min={min_text} bbox_max={max_text}"
    )


def obstacle_aabb_overlaps_segment(pa, pb, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0):
    try:
        mn = np.array(obstacle["min"], dtype=np.float32).reshape(-1)[:3]
        mx = np.array(obstacle["max"], dtype=np.float32).reshape(-1)[:3]
        a = np.array(pa, dtype=np.float32).reshape(-1)[:3]
        b = np.array(pb, dtype=np.float32).reshape(-1)[:3]
    except Exception:
        return True
    pad_xy = max(0.0, float(margin_xy) + float(radius))
    pad_z = max(0.0, float(margin_z) + float(radius))
    seg_min = np.minimum(a, b) - np.array([pad_xy, pad_xy, pad_z], dtype=np.float32)
    seg_max = np.maximum(a, b) + np.array([pad_xy, pad_xy, pad_z], dtype=np.float32)
    return bool(
        float(seg_max[0]) >= float(mn[0])
        and float(seg_min[0]) <= float(mx[0])
        and float(seg_max[1]) >= float(mn[1])
        and float(seg_min[1]) <= float(mx[1])
        and float(seg_max[2]) >= float(mn[2])
        and float(seg_min[2]) <= float(mx[2])
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
            if point_inside_obstacle_proxy(
                point,
                obstacle,
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


def is_unload_bin_wall_obstacle(obstacle):
    path = str((obstacle or {}).get("path", "")).lower()
    return ("unloadbin" in path or "unload_bin" in path) and "wall" in path


def unload_bin_wall_overpass_allowed(mode, obstacle, *points):
    if "unload_to_bin" not in str(mode).lower():
        return False
    if not is_unload_bin_wall_obstacle(obstacle):
        return False
    try:
        wall_top = float(np.array(obstacle.get("max"), dtype=np.float32).reshape(-1)[2])
    except Exception:
        return False
    min_z = None
    for point in points:
        if point is None:
            continue
        try:
            z = float(np.array(point, dtype=np.float32).reshape(-1)[2])
        except Exception:
            return False
        min_z = z if min_z is None else min(min_z, z)
    if min_z is None:
        return False
    return float(min_z) >= wall_top + float(UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z)


def path_obstacle_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None):
    t0 = time.perf_counter()
    obstacles = rigid_obstacle_bboxes()
    try:
        if not obstacles:
            return True, "ok", samples, None

        samples = max(2, int(samples))
        for i in range(1, samples + 1):
            if planning_deadline_exceeded(deadline):
                return False, "planning budget exceeded", i, None
            s = float(i) / float(samples)
            q = interpolate_q_shortest(q_start, q_goal, s)
            segments = predicted_obstacle_check_segments(q, reference_q=q_start)
            for link_name, pa, pb in segments:
                for obstacle in obstacles:
                    if planning_deadline_exceeded(deadline):
                        return False, "planning budget exceeded", i, None
                    if unload_bin_wall_overpass_allowed(mode, obstacle, pa, pb):
                        continue
                    if not obstacle_aabb_overlaps_segment(
                        pa,
                        pb,
                        obstacle,
                        margin_xy=PATH_OBSTACLE_MARGIN_XY,
                        margin_z=PATH_OBSTACLE_MARGIN_Z,
                        radius=PATH_LINK_COLLISION_RADIUS_M,
                    ):
                        continue
                    hit, hit_point = segment_intersects_obstacle_proxy(
                        pa,
                        pb,
                        obstacle,
                        margin_xy=PATH_OBSTACLE_MARGIN_XY,
                        margin_z=PATH_OBSTACLE_MARGIN_Z,
                        radius=PATH_LINK_COLLISION_RADIUS_M,
                    )
                    if hit:
                        report = {
                            "mode": str(mode),
                            "obstacle": obstacle["path"],
                            "obstacle_source": obstacle.get("source", ""),
                            "obstacle_proxy": obstacle.get("proxy", "aabb"),
                            "obstacle_footprint_faces": obstacle.get("footprint_faces", 0),
                            "obstacle_footprint_source_vertices": obstacle.get("footprint_source_vertices", 0),
                            "link_name": link_name,
                            "point": hit_point,
                            "segment": (pa, pb),
                            "bbox_min": obstacle["min"],
                            "bbox_max": obstacle["max"],
                        }
                        return False, "predicted rigid link sweep contact", i, report

            # Segment sweeps already include all chain endpoints. Keep point-only
            # checks as a fallback for unusual missing-segment predictions instead
            # of repeating a full second obstacle pass for every path sample.
            if not segments:
                points = predicted_obstacle_check_points(q, reference_q=q_start)
                for p in points:
                    for obstacle in obstacles:
                        if planning_deadline_exceeded(deadline):
                            return False, "planning budget exceeded", i, None
                        if unload_bin_wall_overpass_allowed(mode, obstacle, p):
                            continue
                        if not obstacle_aabb_overlaps_segment(
                            p,
                            p,
                            obstacle,
                            margin_xy=PATH_OBSTACLE_MARGIN_XY,
                            margin_z=PATH_OBSTACLE_MARGIN_Z,
                            radius=0.0,
                        ):
                            continue
                        if point_inside_obstacle_proxy(
                            p,
                            obstacle,
                            margin_xy=PATH_OBSTACLE_MARGIN_XY,
                            margin_z=PATH_OBSTACLE_MARGIN_Z,
                        ):
                            report = {
                                "mode": str(mode),
                                "obstacle": obstacle["path"],
                                "obstacle_source": obstacle.get("source", ""),
                                "obstacle_proxy": obstacle.get("proxy", "aabb"),
                                "obstacle_footprint_faces": obstacle.get("footprint_faces", 0),
                                "obstacle_footprint_source_vertices": obstacle.get("footprint_source_vertices", 0),
                                "point": p,
                                "bbox_min": obstacle["min"],
                                "bbox_max": obstacle["max"],
                            }
                            return False, "predicted rigid obstacle contact", i, report

        return True, "ok", samples, None
    finally:
        perf = dict(STATE.get("sand_perf_last", {}) or {})
        perf["obstacle_check_ms"] = float(perf.get("obstacle_check_ms", 0.0) or 0.0) + (
            time.perf_counter() - t0
        ) * 1000.0
        perf["obstacle_check_count"] = int(perf.get("obstacle_check_count", 0) or 0) + 1
        STATE["sand_perf_last"] = perf


def path_segment_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None):
    ok, reason, sample, report = path_phase_check(q_start, q_goal, mode, samples=samples, deadline=deadline)
    if not ok:
        return False, "phase", reason, sample, report

    ok, reason, sample, report = path_obstacle_check(q_start, q_goal, mode, samples=samples, deadline=deadline)
    if not ok:
        return False, "obstacle", reason, sample, report

    return True, "ok", "ok", samples, report


def unload_bin_wall_clearance_required_z(ctx=None):
    ctx = task_scene_context() if ctx is None else ctx
    try:
        bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
        wall_top = float(bin_z_range[1]) if len(bin_z_range) >= 2 else float(GROUND_TOP_Z)
    except Exception:
        wall_top = float(GROUND_TOP_Z)
    return wall_top, wall_top + float(UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z)


def unload_goal_pose_collision_report(q_goal, reference_q=None, mode="unload_to_bin"):
    obstacles = rigid_obstacle_bboxes()
    if not obstacles:
        return None
    q_goal = np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = q_goal if reference_q is None else np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    segments = predicted_obstacle_check_segments(q_goal, reference_q=q_ref)
    for link_name, pa, pb in segments:
        for obstacle in obstacles:
            if unload_bin_wall_overpass_allowed(mode, obstacle, pa, pb):
                continue
            if not obstacle_aabb_overlaps_segment(
                pa,
                pb,
                obstacle,
                margin_xy=PATH_OBSTACLE_MARGIN_XY,
                margin_z=PATH_OBSTACLE_MARGIN_Z,
                radius=PATH_LINK_COLLISION_RADIUS_M,
            ):
                continue
            hit, hit_point = segment_intersects_obstacle_proxy(
                pa,
                pb,
                obstacle,
                margin_xy=PATH_OBSTACLE_MARGIN_XY,
                margin_z=PATH_OBSTACLE_MARGIN_Z,
                radius=PATH_LINK_COLLISION_RADIUS_M,
            )
            if not hit:
                continue
            wall_top = None
            required_z = None
            min_segment_z = None
            if is_unload_bin_wall_obstacle(obstacle):
                try:
                    wall_top = float(np.array(obstacle.get("max"), dtype=np.float32).reshape(-1)[2])
                    required_z = wall_top + float(UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z)
                    min_segment_z = min(float(pa[2]), float(pb[2]))
                except Exception:
                    pass
            return {
                "mode": str(mode),
                "obstacle": obstacle.get("path", ""),
                "obstacle_source": obstacle.get("source", ""),
                "obstacle_proxy": obstacle.get("proxy", "aabb"),
                "obstacle_footprint_faces": obstacle.get("footprint_faces", 0),
                "obstacle_footprint_source_vertices": obstacle.get("footprint_source_vertices", 0),
                "link_name": link_name,
                "point": hit_point,
                "segment": (pa, pb),
                "bbox_min": obstacle.get("min"),
                "bbox_max": obstacle.get("max"),
                "is_unload_bin_wall": bool(is_unload_bin_wall_obstacle(obstacle)),
                "wall_top_z": wall_top,
                "required_clearance_z": required_z,
                "min_segment_z": min_segment_z,
            }
    return None


def unload_collision_report_from_path_obstacle(report, mode="unload_to_bin"):
    if not isinstance(report, dict):
        return None
    obstacle_path = str(report.get("obstacle", ""))
    obstacle = {
        "path": obstacle_path,
        "source": report.get("obstacle_source", ""),
        "proxy": report.get("obstacle_proxy", "aabb"),
        "footprint_faces": report.get("obstacle_footprint_faces", 0),
        "footprint_source_vertices": report.get("obstacle_footprint_source_vertices", 0),
        "min": report.get("bbox_min"),
        "max": report.get("bbox_max"),
    }
    wall_top = None
    required_z = None
    min_segment_z = None
    if is_unload_bin_wall_obstacle(obstacle):
        try:
            wall_top = float(np.array(obstacle.get("max"), dtype=np.float32).reshape(-1)[2])
            required_z = wall_top + float(UNLOAD_BIN_WALL_OVERPASS_CLEARANCE_Z)
        except Exception:
            wall_top = None
            required_z = None
    try:
        segment = report.get("segment")
        if segment is not None:
            a, b = segment
            min_segment_z = min(float(np.array(a, dtype=np.float32).reshape(-1)[2]), float(np.array(b, dtype=np.float32).reshape(-1)[2]))
    except Exception:
        min_segment_z = None
    out = {
        "mode": str(mode),
        "obstacle": obstacle_path,
        "obstacle_source": obstacle.get("source", ""),
        "obstacle_proxy": obstacle.get("proxy", "aabb"),
        "obstacle_footprint_faces": obstacle.get("footprint_faces", 0),
        "obstacle_footprint_source_vertices": obstacle.get("footprint_source_vertices", 0),
        "link_name": report.get("link_name", ""),
        "point": report.get("point"),
        "segment": report.get("segment"),
        "bbox_min": report.get("bbox_min"),
        "bbox_max": report.get("bbox_max"),
        "is_unload_bin_wall": bool(is_unload_bin_wall_obstacle(obstacle)),
        "wall_top_z": wall_top,
        "required_clearance_z": required_z,
        "min_segment_z": min_segment_z,
    }
    return out


def unload_goal_reachability_report(q_seed, landing_target=None, label="", deadline=None):
    ctx = task_scene_context()
    landing = unload_bin_landing_point(ctx=ctx) if landing_target is None else np.array(landing_target, dtype=np.float32).reshape(-1)[:3]
    bin_center = np.array(ctx["unload_bin_center"], dtype=np.float32).reshape(-1)[:3]
    bin_half = np.array(ctx["unload_bin_half_size"], dtype=np.float32).reshape(-1)[:2]
    safe_half = np.maximum(bin_half - UNLOAD_BIN_SAFE_XY_MARGIN, np.array([0.02, 0.02], dtype=np.float32))
    wall_top, required_z = unload_bin_wall_clearance_required_z(ctx)
    q_seed = np.array(q_seed, dtype=np.float32).reshape(-1)[:4].copy()
    swing_idx = CTRL.name_to_idx["swing"]
    boom_idx = CTRL.name_to_idx["boom"]
    arm_idx = CTRL.name_to_idx["arm"]
    target_xy = np.array([float(landing[0]), float(landing[1])], dtype=np.float32)
    try:
        boom_anchor = get_joint_anchor_world("boom")
        if boom_anchor is None:
            boom_anchor = np.zeros(3, dtype=np.float32)
        bearing = math.atan2(float(target_xy[1] - boom_anchor[1]), float(target_xy[0] - boom_anchor[0]))
    except Exception:
        bearing = float(q_seed[swing_idx])

    def unique_values(values, ndigits=6):
        out = []
        seen = set()
        for value in values:
            try:
                v = float(value)
            except Exception:
                continue
            key = round(v, ndigits)
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    boom_lo, boom_hi = planner_effective_joint_bounds_rad("boom")
    arm_lo, arm_hi = planner_effective_joint_bounds_rad("arm")
    swing_values = unique_values(
        [
            bearing,
            float(q_seed[swing_idx]),
            bearing + deg_to_rad(8.0),
            bearing - deg_to_rad(8.0),
        ]
    )
    boom_values = unique_values(
        list(np.linspace(float(boom_lo), float(boom_hi), 6))
        + [float(q_seed[boom_idx]), float(q_seed[boom_idx]) + deg_to_rad(10.0), float(q_seed[boom_idx]) + deg_to_rad(22.0)]
    )
    arm_values = unique_values(
        list(np.linspace(float(arm_lo), float(arm_hi), 7))
        + [float(q_seed[arm_idx]), float(q_seed[arm_idx]) - deg_to_rad(8.0), float(q_seed[arm_idx]) - deg_to_rad(18.0)]
    )
    xy_accept = max(float(np.max(safe_half)) + 0.45, 0.65)
    rows = []
    accepted = []
    sample_count = 0
    for swing in swing_values:
        if planning_deadline_exceeded(deadline):
            break
        for boom in boom_values:
            if planning_deadline_exceeded(deadline):
                break
            for arm in arm_values:
                if planning_deadline_exceeded(deadline):
                    break
                sample_count += 1
                q = q_seed.copy()
                q[swing_idx] = float(swing)
                q[boom_idx] = float(boom)
                q[arm_idx] = float(arm)
                q = clip_route_command_near(q, reference=q_seed)
                try:
                    q_carry, carry_report = carry_hold_adjusted_q(
                        q,
                        q_reference=q_seed,
                        end_effector="load",
                        max_bucket_adjust_deg=105.0,
                    )
                except Exception:
                    q_carry, carry_report = q.copy(), {"ok": False, "reason": "carry_adjust_exception"}
                tip = predicted_end_world_point(q_carry, end_effector="tip", reference_q=q_seed)
                load = predicted_end_world_point(q_carry, end_effector="load", reference_q=q_seed)
                pour = predicted_end_world_point(q_carry, end_effector="pour", reference_q=q_seed)
                pts = [p for p in [tip, load, pour] if p is not None]
                if len(pts) < 2:
                    continue
                pts = [np.array(p, dtype=np.float32).reshape(-1)[:3] for p in pts]
                xy_err = min(float(np.linalg.norm(p[:2] - target_xy)) for p in pts)
                min_bucket_z = min(float(p[2]) for p in pts)
                max_bucket_z = max(float(p[2]) for p in pts)
                tip_z = None if tip is None else float(np.array(tip, dtype=np.float32).reshape(-1)[2])
                row = {
                    "q": q_carry.copy(),
                    "carry_report": carry_report if isinstance(carry_report, dict) else {"reason": str(carry_report)},
                    "xy_err": float(xy_err),
                    "tip_z": tip_z,
                    "load_z": None if load is None else float(np.array(load, dtype=np.float32).reshape(-1)[2]),
                    "pour_z": None if pour is None else float(np.array(pour, dtype=np.float32).reshape(-1)[2]),
                    "min_bucket_z": float(min_bucket_z),
                    "max_bucket_z": float(max_bucket_z),
                    "best_point": max(pts, key=lambda p: float(p[2])),
                }
                rows.append(row)
                if xy_err <= xy_accept:
                    accepted.append(row)
    pool = accepted if accepted else rows
    best = None
    if pool:
        best = sorted(pool, key=lambda r: (-float(r.get("min_bucket_z", -999.0)), float(r.get("xy_err", 999.0))))[0]
    max_tip_z = None
    max_min_bucket_z = None
    if accepted:
        tip_values = [float(r["tip_z"]) for r in accepted if r.get("tip_z") is not None]
        min_values = [float(r["min_bucket_z"]) for r in accepted if r.get("min_bucket_z") is not None]
        if tip_values:
            max_tip_z = max(tip_values)
        if min_values:
            max_min_bucket_z = max(min_values)
    elif best is not None:
        max_tip_z = best.get("tip_z")
        max_min_bucket_z = best.get("min_bucket_z")
    can_clear = bool(max_min_bucket_z is not None and float(max_min_bucket_z) >= float(required_z))
    report = {
        "ok": bool(best is not None),
        "reason": "ok" if best is not None else "no_reachability_sample",
        "label": str(label),
        "sample_count": int(sample_count),
        "accepted_count": int(len(accepted)),
        "xy_accept": float(xy_accept),
        "bin_center": vec_list(bin_center, 3),
        "bin_safe_half": vec_list(safe_half, 2),
        "bin_radius": float(np.linalg.norm(bin_center[:2])),
        "landing_target": vec_list(landing, 3),
        "wall_top_z": float(wall_top),
        "required_clearance_z": float(required_z),
        "max_reachable_tip_z": None if max_tip_z is None else float(max_tip_z),
        "max_reachable_bucket_min_z": None if max_min_bucket_z is None else float(max_min_bucket_z),
        "reach_margin": None if max_min_bucket_z is None else float(max_min_bucket_z) - float(required_z),
        "can_clear_wall": bool(can_clear),
        "best_q_rad": None if best is None else vec_list(best["q"], 4),
        "best_q_deg": None if best is None else q_deg_values(best["q"], wrap_swing_for_display=True),
        "best_xy_err": None if best is None else float(best.get("xy_err", 999.0)),
        "best_point": None if best is None else vec_list(best.get("best_point"), 3),
        "best_carry_report": {} if best is None else best.get("carry_report", {}),
    }
    info_print(
        "[UNLOAD REACHABILITY]",
        f"label={label}",
        f"samples={sample_count}",
        f"accepted={len(accepted)}",
        f"bin_radius={fmt_optional(report.get('bin_radius'))}",
        f"required_z={fmt_optional(required_z)}",
        f"max_tip_z={fmt_optional(report.get('max_reachable_tip_z'))}",
        f"max_bucket_min_z={fmt_optional(report.get('max_reachable_bucket_min_z'))}",
        f"margin={fmt_optional(report.get('reach_margin'))}",
        f"can_clear_wall={report.get('can_clear_wall')}",
        f"best_q={report.get('best_q_deg')}",
        force_log=True,
    )
    return report


def unload_pose_bucket_clearance_report(q_pose, reference_q=None):
    q_pose = np.array(q_pose, dtype=np.float32).reshape(-1)[:4].copy()
    if reference_q is None:
        reference_q = q_pose
    points = []
    point_names = []
    for effector in ("tip", "mid", "load", "pour"):
        try:
            p = predicted_end_world_point(q_pose, end_effector=effector, reference_q=reference_q)
        except Exception:
            p = None
        if p is None:
            continue
        points.append(np.array(p, dtype=np.float32).reshape(-1)[:3])
        point_names.append(effector)
    if not points:
        return {"ok": False, "reason": "no_predicted_bucket_points"}
    z_values = [float(p[2]) for p in points]
    min_i = int(np.argmin(np.array(z_values, dtype=np.float32)))
    max_i = int(np.argmax(np.array(z_values, dtype=np.float32)))
    wall_top, required_z = unload_bin_wall_clearance_required_z()
    return {
        "ok": True,
        "reason": "ok",
        "wall_top_z": float(wall_top),
        "required_clearance_z": float(required_z),
        "exec_required_clearance_z": float(required_z) + float(UNLOAD_BIN_WALL_EXEC_EXTRA_CLEARANCE_Z),
        "min_bucket_z": float(z_values[min_i]),
        "max_bucket_z": float(z_values[max_i]),
        "min_point": point_names[min_i],
        "max_point": point_names[max_i],
        "margin": float(z_values[min_i]) - float(required_z),
        "exec_margin": float(z_values[min_i]) - (float(required_z) + float(UNLOAD_BIN_WALL_EXEC_EXTRA_CLEARANCE_Z)),
        "points": {name: vec_list(point, 3) for name, point in zip(point_names, points)},
    }


def validate_unload_goal(q_start, q_pre_dump, q_dump=None, dump_info=None, label="unload_goal", deadline=None):
    ctx = task_scene_context()
    landing = unload_bin_landing_point(ctx=ctx)
    release = None
    if isinstance(dump_info, dict):
        release = dump_info.get("release_target") or dump_info.get("pour_target")
    if release is None:
        release = unload_bin_dump_point(ctx=ctx)
    drop = unload_drop_report(q=q_dump, reference_q=q_start) if q_dump is not None else {}
    collision = unload_goal_pose_collision_report(q_pre_dump, reference_q=q_start, mode="unload_to_bin")
    ok = collision is None
    if ok:
        wall_top, required_z = unload_bin_wall_clearance_required_z(ctx)
        reach = {
            "ok": True,
            "reason": "skipped_goal_collision_free",
            "label": str(label),
            "wall_top_z": float(wall_top),
            "required_clearance_z": float(required_z),
            "can_clear_wall": True,
        }
    else:
        reach_deadline = child_planning_deadline(deadline, 0.90, min_seconds=0.20) if deadline is not None else None
        reach = unload_goal_reachability_report(q_start, landing_target=landing, label=label, deadline=reach_deadline)
    if ok:
        reason = "ok"
    elif bool((collision or {}).get("is_unload_bin_wall", False)) and not bool(reach.get("can_clear_wall", False)):
        reason = "unload_goal_unreachable_by_reach"
    else:
        reason = "unload_goal_invalid"
    rejected_segment = None
    if isinstance(collision, dict):
        rejected_segment = collision.get("segment")
    draw_unload_dump_debug(
        label,
        landing_target=landing,
        release_target=release,
        q_seed=q_start,
        q_seed_dump=q_dump,
        best_drop=drop,
        reason=reason,
        best_reachable_point=reach.get("best_point"),
        rejected_segment=rejected_segment,
    )
    info_print(
        "[UNLOAD GOAL VALIDATE]",
        f"label={label}",
        f"ok={ok}",
        f"reason={reason}",
        f"q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}",
        f"drop_inside={drop.get('inside_xy') if isinstance(drop, dict) else None}",
        f"drop_close={drop.get('close_xy') if isinstance(drop, dict) else None}",
        f"collision_obstacle={(collision or {}).get('obstacle') if isinstance(collision, dict) else None}",
        f"collision_link={(collision or {}).get('link_name') if isinstance(collision, dict) else None}",
        f"min_segment_z={fmt_optional((collision or {}).get('min_segment_z') if isinstance(collision, dict) else None)}",
        f"wall_top={fmt_optional((collision or {}).get('wall_top_z') if isinstance(collision, dict) else None)}",
        f"required_z={fmt_optional((collision or {}).get('required_clearance_z') if isinstance(collision, dict) else reach.get('required_clearance_z'))}",
        f"reach_margin={fmt_optional(reach.get('reach_margin'))}",
        force_log=True,
    )
    if not ok:
        info_print(
            "[UNLOAD GOAL FAIL FAST]",
            f"label={label}",
            f"reason={reason}",
            f"can_clear_wall={reach.get('can_clear_wall')}",
            f"best_q={reach.get('best_q_deg')}",
            force_log=True,
        )
    return {
        "ok": bool(ok),
        "reason": reason,
        "drop": drop,
        "collision": collision,
        "reachability": reach,
        "can_retry_high": bool((not ok) and reach.get("can_clear_wall", False)),
    }


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
        if overlap and obstacle.get("footprint_xy") is not None:
            overlap = segment_intersects_polygon_with_margin_xy(
                a[:2],
                b[:2],
                obstacle["footprint_xy"],
                margin_xy=PATH_OBSTACLE_MARGIN_XY + PATH_LINK_COLLISION_RADIUS_M,
            )
        if overlap:
            top = float(mx[2]) if top is None else max(top, float(mx[2]))
    return top


def obstacle_bboxes_for_segment_xy(p_start, p_goal):
    obstacles = rigid_obstacle_bboxes()
    if not obstacles or p_start is None or p_goal is None:
        return []

    a = np.array(p_start, dtype=np.float32)
    b = np.array(p_goal, dtype=np.float32)
    seg_min = np.minimum(a[:2], b[:2]) - PATH_OBSTACLE_MARGIN_XY
    seg_max = np.maximum(a[:2], b[:2]) + PATH_OBSTACLE_MARGIN_XY
    rows = []
    for obstacle in obstacles:
        mn = obstacle["min"]
        mx = obstacle["max"]
        overlap = (
            float(seg_max[0]) >= float(mn[0] - PATH_OBSTACLE_MARGIN_XY)
            and float(seg_min[0]) <= float(mx[0] + PATH_OBSTACLE_MARGIN_XY)
            and float(seg_max[1]) >= float(mn[1] - PATH_OBSTACLE_MARGIN_XY)
            and float(seg_min[1]) <= float(mx[1] + PATH_OBSTACLE_MARGIN_XY)
        )
        if overlap and obstacle.get("footprint_xy") is not None:
            overlap = segment_intersects_polygon_with_margin_xy(
                a[:2],
                b[:2],
                obstacle["footprint_xy"],
                margin_xy=PATH_OBSTACLE_MARGIN_XY + PATH_LINK_COLLISION_RADIUS_M,
            )
        if not overlap:
            continue
        size_xy = float(max(abs(float(mx[0] - mn[0])), abs(float(mx[1] - mn[1]))))
        rows.append((size_xy, obstacle))
    rows.sort(key=lambda item: float(item[0]), reverse=True)
    return [item[1] for item in rows]


def obstacle_corridor_report(p_start, p_goal, obstacle=None, link_name="planned_end"):
    if obstacle is None:
        blockers = obstacle_bboxes_for_segment_xy(p_start, p_goal)
        obstacle = blockers[0] if blockers else None
    if obstacle is None:
        return None
    a = np.array(p_start, dtype=np.float32).reshape(-1)[:3]
    b = np.array(p_goal, dtype=np.float32).reshape(-1)[:3]
    return {
        "mode": "corridor",
        "obstacle": obstacle.get("path", ""),
        "obstacle_source": obstacle.get("source", ""),
        "obstacle_proxy": obstacle.get("proxy", "aabb"),
        "obstacle_footprint_faces": obstacle.get("footprint_faces", 0),
        "obstacle_footprint_source_vertices": obstacle.get("footprint_source_vertices", 0),
        "link_name": str(link_name),
        "point": 0.5 * (a + b),
        "segment": (a, b),
        "bbox_min": obstacle.get("min"),
        "bbox_max": obstacle.get("max"),
    }


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


def is_unload_dump_motion(mode="", label=""):
    text = f"{mode} {label}".lower()
    return "unload_dump" in text or ("dump_pose" in text and "unload" in text)


def bucket_dump_branch_target(target_rad, current_rad=None):
    """Pick a bucket joint branch that actually opens the bucket for dump."""
    idx = CTRL.name_to_idx.get("bucket", 3)
    name = CTRL.dof_names[idx] if idx < len(CTRL.dof_names) else "bucket"
    lo, hi = FINAL_LIMITS_RAD.get(name, (deg_to_rad(-360.0), deg_to_rad(360.0)))
    target = float(target_rad)
    current = float(CTRL.q_cmd[idx] if current_rad is None else current_rad)
    candidates = [target + (2.0 * math.pi * k) for k in range(-3, 4)]
    valid = [c for c in candidates if float(lo) - 1e-6 <= c <= float(hi) + 1e-6]
    if valid:
        opening = [c for c in valid if c >= current - deg_to_rad(1.0)]
        if opening:
            return float(min(opening, key=lambda c: abs(c - current)))
        return float(min(valid, key=lambda c: abs(c - current)))
    clipped = min(max(target, float(lo)), float(hi))
    info_print(
        "[UNLOAD DUMP TARGET LIMIT]",
        f"target={rad_to_deg(target):.2f}deg",
        f"current={rad_to_deg(current):.2f}deg",
        f"limits=({rad_to_deg(float(lo)):.2f},{rad_to_deg(float(hi)):.2f})deg",
        f"using={rad_to_deg(float(clipped)):.2f}deg",
    )
    return float(clipped)


def clip_unload_dump_command(q, reference=None):
    q_raw = np.array(q, dtype=np.float32).copy()
    if reference is None:
        reference = CTRL.q_cmd
    reference = np.array(reference, dtype=np.float32).copy()
    q = CTRL.clip_limits(q_raw.copy())
    swing_idx = CTRL.name_to_idx["swing"]
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    q[swing_idx] = swing_target_near(q_raw[swing_idx], reference[swing_idx])
    q[swing_idx] = normalize_swing_cmd(q[swing_idx])
    q[bucket_idx] = bucket_dump_branch_target(q_raw[bucket_idx], current_rad=reference[bucket_idx])
    return CTRL.clip_limits(q)


def planner_effective_joint_bounds_rad(name):
    try:
        lo, hi = FINAL_LIMITS_RAD[name]
        lo = float(lo)
        hi = float(hi)
    except Exception:
        lo, hi = deg_to_rad(DESIRED_LIMITS_DEG.get(name, (-180.0, 180.0))[0]), deg_to_rad(DESIRED_LIMITS_DEG.get(name, (-180.0, 180.0))[1])
    if name in PATH_EFFECTIVE_LIMITS_DEG:
        eff_lo, eff_hi = PATH_EFFECTIVE_LIMITS_DEG[name]
        lo = max(lo, deg_to_rad(float(eff_lo)))
        hi = min(hi, deg_to_rad(float(eff_hi)))
        if lo >= hi:
            lo, hi = FINAL_LIMITS_RAD.get(name, (lo, hi))
    return float(lo), float(hi)


def clip_route_command_near(q, reference=None):
    q = clip_command_near(q, reference=reference)
    for name in ["boom", "arm", "bucket"]:
        idx = CTRL.name_to_idx.get(name)
        if idx is None:
            continue
        lo, hi = planner_effective_joint_bounds_rad(name)
        q[idx] = min(max(float(q[idx]), float(lo)), float(hi))
    return clip_command_near(q, reference=reference)


def interpolate_q_shortest(q0, q1, s):
    q0 = np.array(q0, dtype=np.float32)
    q1 = clip_command_near(q1, reference=q0)
    q = (1.0 - float(s)) * q0 + float(s) * q1
    swing_idx = CTRL.name_to_idx["swing"]
    q[swing_idx] = float(q0[swing_idx]) + float(s) * swing_delta(q1[swing_idx], q0[swing_idx])
    q[swing_idx] = normalize_swing_cmd(q[swing_idx])
    return CTRL.clip_limits(q)


def interpolate_q_motion(q0, q1, s, mode="", label=""):
    if not is_unload_dump_motion(mode=mode, label=label):
        return interpolate_q_shortest(q0, q1, s)
    q0 = np.array(q0, dtype=np.float32)
    q1 = clip_unload_dump_command(q1, reference=q0)
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


def bucket_carry_world_angle_candidates(reference_rad, tilt_deg=None):
    """Candidate bucket world angles for carrying material.

    Bucket orientation has multiple 180/360-degree equivalents in the planar
    model, but only one branch keeps the pour edge above the load point for the
    actual bucket geometry. Enumerate those branches and let FK pick the
    physically retaining one.
    """
    tilt = deg_to_rad(BUCKET_CARRY_HOLD_TILT_DEG if tilt_deg is None else float(tilt_deg))
    base = deg_to_rad(BUCKET_LIFT_LEVEL_WORLD_DEG)
    offsets = [tilt, -tilt, 0.0, 0.5 * tilt, -0.5 * tilt, 1.5 * tilt, -1.5 * tilt]
    candidates = []
    seen = set()
    for k in range(-3, 4):
        for branch in (0.0, math.pi):
            level = base + branch + 2.0 * math.pi * float(k)
            for offset in offsets:
                angle = float(level + offset)
                key = round(wrap_angle(angle), 6)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(angle)
    candidates.sort(key=lambda a: abs(wrap_angle(float(a) - float(reference_rad))))
    return candidates


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
    candidates = bucket_carry_world_angle_candidates(reference_rad, tilt_deg=tilt_deg)
    if q_reference is None:
        return min(candidates, key=lambda a: abs(wrap_angle(float(a) - float(reference_rad))))

    best = None
    fallback = None
    for angle in candidates:
        report = bucket_mouth_raise_for_world_angle(q_reference, angle, end_effector=end_effector)
        if report is None:
            continue
        raise_z = float(report["raise_z"])
        tilt_err = abs(abs(wrap_angle(float(angle) - float(level))) - tilt)
        retain_short = max(0.0, BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z - raise_z)
        retain_penalty = retain_short * 80.0
        limit_penalty = 25.0 if report.get("limited") else 0.0
        bucket_motion = abs(wrap_angle(float(report.get("bucket", 0.0)) - float(q_reference[CTRL.name_to_idx.get("bucket", 3)])))
        bucket_delta_deg = abs(rad_to_deg(bucket_motion))
        flip_penalty = 5000.0 if bucket_delta_deg > float(BUCKET_CARRY_MAX_ADJUST_DEG) else 0.0
        soft_penalty = max(0.0, bucket_delta_deg - float(BUCKET_CARRY_SOFT_ADJUST_DEG)) * 0.12
        cost = (
            retain_penalty
            + 0.35 * tilt_err
            + 0.04 * abs(wrap_angle(float(angle) - float(reference_rad)))
            + 0.08 * bucket_motion
            + soft_penalty
            + flip_penalty
            + limit_penalty
        )
        row = {
            "angle": float(angle),
            "cost": float(cost),
            "raise_z": raise_z,
            "limited": bool(report.get("limited")),
            "bucket_delta_deg": float(bucket_delta_deg),
            "bucket_flip_rejected": bool(bucket_delta_deg > float(BUCKET_CARRY_MAX_ADJUST_DEG)),
            "retains_material": bool(raise_z >= BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z),
        }
        if row["bucket_flip_rejected"]:
            if fallback is None or row["cost"] < fallback["cost"]:
                fallback = row
            continue
        if best is None or (
            (bool(row["retains_material"]) and not bool(best.get("retains_material", False)))
            or (bool(row["retains_material"]) == bool(best.get("retains_material", False)) and row["cost"] < best["cost"])
        ):
            best = row

    if best is None:
        if fallback is not None:
            return float(fallback["angle"])
        return min(candidates, key=lambda a: abs(wrap_angle(float(a) - float(reference_rad))))
    return float(best["angle"])


def carry_hold_adjusted_q(q_pose, q_reference=None, end_effector="load", max_bucket_adjust_deg=None):
    q_pose = np.array(q_pose, dtype=np.float32).reshape(-1)[:4].copy()
    q_reference = q_pose if q_reference is None else np.array(q_reference, dtype=np.float32).reshape(-1)[:4].copy()
    max_adjust_deg = float(BUCKET_CARRY_MAX_ADJUST_DEG if max_bucket_adjust_deg is None else max_bucket_adjust_deg)
    pose_angles = chain_angles_from_q(q_pose, end_effector=end_effector)
    if pose_angles is None:
        return q_pose, {
            "ok": False,
            "reason": "missing_pose_angles",
            "end_effector": str(end_effector),
        }
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    best = None
    rows = []
    for carry_world in bucket_carry_world_angle_candidates(float(pose_angles[2])):
        calc = bucket_joint_for_world_angle(q_pose, carry_world, end_effector=end_effector)
        if calc is None:
            continue
        raw_bucket = float(calc["bucket"])
        q_out = q_pose.copy()
        q_out[bucket_idx] = raw_bucket
        q_out = clip_command_near(q_out, reference=q_pose)
        q_out, loaded_bucket_limited, _old_bucket_deg = apply_loaded_bucket_closed_limit(
            q_out,
            label="carry_hold_adjusted_q",
        )
        limited = (
            abs(wrap_angle(float(q_out[bucket_idx]) - raw_bucket)) > deg_to_rad(0.25)
            or loaded_bucket_limited
        )
        report = bucket_mouth_raise_for_world_angle(q_out, carry_world, end_effector=end_effector)
        raise_z = None if report is None else float(report.get("raise_z", 0.0))
        goal_angles = chain_angles_from_q(q_out, end_effector=end_effector)
        goal_world = None if goal_angles is None else float(goal_angles[2])
        bucket_deg = rad_to_deg(float(q_out[bucket_idx]))
        dump_branch = bucket_is_dump_branch_for_carry(bucket_deg)
        retains = bool(raise_z is not None and raise_z >= BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z and not dump_branch)
        q_delta = q_delta_abs_deg(q_out, q_reference)
        motion_cost = float(np.sum(np.array(DIG_PLAN_MOTION_WEIGHTS, dtype=np.float32) * np.array(q_delta, dtype=np.float32)))
        bucket_delta_deg = abs(rad_to_deg(wrap_angle(float(q_out[bucket_idx]) - float(q_reference[bucket_idx]))))
        bucket_flip_rejected = bool(bucket_delta_deg > float(max_adjust_deg))
        bucket_soft_penalty = max(0.0, bucket_delta_deg - float(BUCKET_CARRY_SOFT_ADJUST_DEG)) * 6.0
        retain_short = max(0.0, BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z - float(raise_z if raise_z is not None else -1.0))
        level = nearest_bucket_level_world_angle(float(pose_angles[2]))
        tilt = deg_to_rad(BUCKET_CARRY_HOLD_TILT_DEG)
        tilt_err = abs(abs(wrap_angle(float(carry_world) - float(level))) - tilt)
        score = (
            (0.0 if retains else 1000.0)
            + (5000.0 if bucket_flip_rejected else 0.0)
            + (8000.0 if dump_branch else 0.0)
            + 120.0 * retain_short
            + 0.45 * motion_cost
            + bucket_soft_penalty
            + 0.8 * rad_to_deg(tilt_err)
            + (80.0 if limited else 0.0)
        )
        row = {
            "q": q_out.copy(),
            "raw_bucket": raw_bucket,
            "carry_world": float(carry_world),
            "goal_world": goal_world,
            "raise_z": raise_z,
            "retains_material": retains,
            "loaded_carry_joint_ok": bool(bucket_joint_in_loaded_carry_state(bucket_deg)),
            "dump_branch_for_carry": bool(dump_branch),
            "limited": bool(limited),
            "loaded_bucket_limited": loaded_bucket_limited,
            "motion_cost": motion_cost,
            "bucket_delta_deg": float(bucket_delta_deg),
            "bucket_flip_rejected": bucket_flip_rejected,
            "score": float(score),
        }
        rows.append(row)
        if bucket_flip_rejected or dump_branch:
            continue
        if best is None or row["score"] < best["score"]:
            best = row

    if best is None:
        return q_pose, {
            "ok": False,
            "reason": "bucket_carry_requires_large_bucket_flip",
            "end_effector": str(end_effector),
            "max_bucket_adjust_deg": float(max_adjust_deg),
            "candidate_count": len(rows),
            "candidate_preview": [
                {
                    "world_deg": rad_to_deg(row["carry_world"]),
                    "bucket_deg": rad_to_deg(row["q"][bucket_idx]),
                    "bucket_delta_deg": row.get("bucket_delta_deg"),
                    "raise_z": row["raise_z"],
                    "retains": row["retains_material"],
                    "dump_branch_for_carry": row.get("dump_branch_for_carry", False),
                    "limited": row["limited"],
                    "loaded_bucket_limited": row.get("loaded_bucket_limited", False),
                    "flip_rejected": row.get("bucket_flip_rejected", False),
                    "score": row["score"],
                }
                for row in sorted(rows, key=lambda r: float(r.get("score", 1.0e9)))[:6]
            ],
        }
    q_out = best["q"].copy()
    carry_world = float(best["carry_world"])
    goal_world = best.get("goal_world")
    raise_z = best.get("raise_z")
    return q_out, {
        "ok": True,
        "reason": "ok",
        "end_effector": str(end_effector),
        "target_world_deg": rad_to_deg(carry_world),
        "actual_world_deg": None if goal_world is None else rad_to_deg(goal_world),
        "world_err_deg": None if goal_world is None else rad_to_deg(abs(wrap_angle(goal_world - carry_world))),
        "bucket_deg": rad_to_deg(q_out[bucket_idx]),
        "loaded_carry_joint_ok": bool(bucket_joint_in_loaded_carry_state(rad_to_deg(q_out[bucket_idx]))),
        "loaded_carry_target_deg": float(CURL_HOLD_TARGET_DEG),
        "loaded_carry_accept_deg": float(CURL_HOLD_ACCEPT_BUCKET_DEG),
        "raw_bucket_deg": rad_to_deg(float(best.get("raw_bucket", q_out[bucket_idx]))),
        "limited": bool(best.get("limited", False)),
        "loaded_bucket_limited": bool(best.get("loaded_bucket_limited", False)),
        "dump_branch_for_carry": bool(best.get("dump_branch_for_carry", False)),
        "max_carry_dump_branch_deg": float(BUCKET_CARRY_MAX_DUMP_BRANCH_DEG),
        "loaded_bucket_limit_deg": float(BUCKET_LOADED_CLOSED_LIMIT_DEG),
        "max_bucket_adjust_deg": float(max_adjust_deg),
        "pour_above_load_z": raise_z,
        "min_pour_above_load_z": float(BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z),
        "retains_material": bool(best.get("retains_material", False)),
        "candidate_count": len(rows),
        "candidate_preview": [
            {
                "world_deg": rad_to_deg(row["carry_world"]),
                "bucket_deg": rad_to_deg(row["q"][bucket_idx]),
                "bucket_delta_deg": row.get("bucket_delta_deg"),
                "raise_z": row["raise_z"],
                "retains": row["retains_material"],
                "dump_branch_for_carry": row.get("dump_branch_for_carry", False),
                "limited": row["limited"],
                "loaded_bucket_limited": row.get("loaded_bucket_limited", False),
                "flip_rejected": row.get("bucket_flip_rejected", False),
                "score": row["score"],
            }
            for row in sorted(rows, key=lambda r: float(r.get("score", 1.0e9)))[:6]
        ],
    }


def carry_report_pour_above_load(carry_report):
    if not isinstance(carry_report, dict):
        return None
    value = carry_report.get("pour_above_load_z")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def carry_report_allows_transitional_load(carry_report):
    if bool((carry_report or {}).get("retains_material", False)):
        return True
    if bool((carry_report or {}).get("transitional_load", False)):
        return True
    pour_above = carry_report_pour_above_load(carry_report)
    if pour_above is None:
        return False
    return pour_above >= float(BUCKET_CARRY_TRANSITIONAL_MIN_POUR_Z)


def loaded_transitional_hold_allowed(carry_report, loaded_count=0):
    try:
        loaded = int(loaded_count or 0)
    except Exception:
        loaded = 0
    if loaded < int(CURL_HOLD_MIN_BUCKET_PARTICLES):
        return False
    if not isinstance(carry_report, dict) or not bool(carry_report.get("ok", False)):
        return False
    # Once the real particle monitor proves the bucket already contains a
    # meaningful load, pour_above_load_z is no longer a reliable hard veto:
    # the mesh proxy can say "spill risk" while particles are visibly held.
    # Keep the true hard safety guard: do not proceed if the bucket is on a
    # dump branch. The later secure/lift material gates still reject real loss.
    if bool(carry_report.get("dump_branch_for_carry", False)):
        return False
    return True


def real_loaded_secure_hold_allowed(carry_report, loaded_count=0):
    if not loaded_transitional_hold_allowed(carry_report, loaded_count=loaded_count):
        return False
    if not bool((carry_report or {}).get("loaded_carry_joint_ok", False)):
        return False
    return True


def carry_spill_risk_penalty(carry_report):
    if bool((carry_report or {}).get("retains_material", False)):
        return 0.0
    if carry_report_pour_above_load(carry_report) is None and bool((carry_report or {}).get("transitional_load", False)):
        return float(BUCKET_CARRY_SPILL_RISK_COST)
    pour_above = carry_report_pour_above_load(carry_report)
    if pour_above is None:
        return float(BUCKET_CARRY_SPILL_RISK_COST)
    shortfall = max(0.0, float(BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z) - float(pour_above))
    return float(BUCKET_CARRY_SPILL_RISK_COST) * shortfall


def apply_loaded_bucket_closed_limit(q, label=""):
    q = np.array(q, dtype=np.float32).reshape(-1)[:4].copy()
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    limit_rad = deg_to_rad(float(BUCKET_LOADED_CLOSED_LIMIT_DEG))
    if float(q[bucket_idx]) < limit_rad:
        old_deg = rad_to_deg(float(q[bucket_idx]))
        raw_q = CTRL.clip_limits(q)
        limited_q = np.array(raw_q, dtype=np.float32).copy()
        limited_q[bucket_idx] = limit_rad
        limited_q = CTRL.clip_limits(limited_q)
        # The loaded limit is a safety clamp, not a license to destroy a valid
        # carry pose. If the requested angle is still inside hard joint limits
        # and is the only geometry that keeps the pour edge above the load
        # region, preserve it and let the quality gate score the real result.
        try:
            raw_report = carry_material_report_for_q(raw_q, end_effector="load")
            limited_report = carry_material_report_for_q(limited_q, end_effector="load")
            if (
                bool(raw_report.get("retains_material", False))
                and not bool(limited_report.get("retains_material", False))
            ):
                return raw_q, False, old_deg
        except Exception:
            pass
        return limited_q, True, old_deg
    return q, False, None


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
        "valid": False,
    }


def ik_model_is_valid(model=None):
    model = IK_MODEL if model is None else model
    if not isinstance(model, dict):
        return False
    if not bool(model.get("calibrated", False)) or not bool(model.get("valid", False)):
        return False
    signs = model.get("signs", None)
    effectors = model.get("effectors", None)
    if signs is None or len(signs) != 3 or not isinstance(effectors, dict):
        return False
    for key in ["mid", "tip"]:
        part = effectors.get(key)
        if not isinstance(part, dict):
            return False
        lengths = np.array(part.get("lengths", []), dtype=np.float32)
        offsets = np.array(part.get("offsets", []), dtype=np.float32)
        if len(lengths) != 3 or len(offsets) != 3:
            return False
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(offsets)):
            return False
        if float(np.min(lengths)) < IK_MIN_SEGMENT_LENGTH:
            return False
    return True


def validate_ik_model_against_current_pose(model, q_reference):
    errors = {}
    for end_effector, actual_fn in [
        ("mid", bucket_mid_pos),
        ("tip", bucket_tip_pos),
        ("load", bucket_load_pos),
        ("pour", bucket_pour_pos),
    ]:
        if end_effector not in model.get("effectors", {}):
            continue
        pred = predicted_end_world_point(q_reference, end_effector=end_effector, reference_q=q_reference)
        try:
            actual = actual_fn()
        except Exception:
            actual = None
        if pred is None or actual is None:
            errors[end_effector] = None
            continue
        errors[end_effector] = float(np.linalg.norm(np.array(pred, dtype=np.float32) - np.array(actual, dtype=np.float32)))
    finite = [v for v in errors.values() if v is not None and math.isfinite(float(v))]
    if not finite:
        return False, "no effector sanity samples", errors
    worst = max(float(v) for v in finite)
    if worst > IK_SANITY_MAX_POS_ERR_M:
        return False, f"effector FK sanity error {worst:.3f}m", errors
    return True, f"ok worst={worst:.3f}m", errors


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
    deadline=None,
    score_goal_obstacle=True,
):
    global IK_MODEL

    if IK_MODEL is None:
        IK_MODEL = default_ik_model()
    if IK_MODEL is None:
        return None, "IK model unavailable"
    if planning_deadline_exceeded(deadline):
        return ([], "planning budget exceeded") if return_candidates else (None, "planning budget exceeded")

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

    unique_candidate_angles = []
    seen_candidate_angles = set()
    for angle in candidate_angles:
        key = round(float(wrap_angle(angle)), 4)
        if key in seen_candidate_angles:
            continue
        seen_candidate_angles.add(key)
        unique_candidate_angles.append(float(angle))
    candidate_angles = unique_candidate_angles

    best = None
    solution_rows = []
    reject_counts = {}
    total_reach = float(np.sum(lengths))
    target_dist = float(np.linalg.norm(target_2d))

    def reject(reason):
        key = str(reason)
        reject_counts[key] = int(reject_counts.get(key, 0)) + 1

    for a3 in candidate_angles:
        if planning_deadline_exceeded(deadline):
            if return_candidates:
                break
            return None, "planning budget exceeded"
        wrist_target = target_2d - np.array([math.cos(a3), math.sin(a3)], dtype=np.float32) * float(lengths[2])

        for _, a1, a2 in two_link_ik_2d(wrist_target, lengths[0], lengths[1], seed_angles):
            if planning_deadline_exceeded(deadline):
                if return_candidates:
                    break
                return None, "planning budget exceeded"
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
                phase_report = predicted_phase_ground_report(q_candidate, phase_mode, reference_q=q_now)
                if phase_report.get("tip_z") is None:
                    reject("missing predicted phase report")
                    continue
                phase_ok, phase_reason = phase_ground_ok(phase_mode, phase_report)
                if not phase_ok:
                    reject(phase_reason)
                    continue

            goal_obstacle_ok = True
            goal_obstacle_reason = "ok"
            goal_obstacle_report = None
            if phase_mode is not None and bool(score_goal_obstacle):
                try:
                    goal_obstacle_ok, goal_obstacle_reason, _goal_sample, goal_obstacle_report = path_obstacle_check(
                        q_candidate,
                        q_candidate,
                        phase_mode,
                        samples=2,
                        deadline=deadline,
                    )
                except Exception as exc:
                    goal_obstacle_ok = True
                    goal_obstacle_reason = f"obstacle_check_error:{type(exc).__name__}:{exc}"
                    goal_obstacle_report = None

            dq = np.array([wrap_angle(float(q_candidate[i] - q_now[i])) for i in range(4)], dtype=np.float32)
            motion_cost = float(np.sum(IK_COST_WEIGHTS * np.abs(dq)))
            bucket_change = abs(wrap_angle(float(q_candidate[CTRL.name_to_idx["bucket"]] - q_now[CTRL.name_to_idx["bucket"]])))
            bucket_pref_cost = 0.0
            if preferred_bucket_rad is not None:
                bucket_pref_cost = abs(wrap_angle(float(q_candidate[CTRL.name_to_idx["bucket"]] - preferred_bucket_rad))) * 1.2
            if preferred_end_angle_rad is not None:
                bucket_pref_cost += abs(wrap_angle(float(actual_a3 - preferred_end_angle_rad))) * 1.8

            goal_obstacle_cost = 0.0 if goal_obstacle_ok else float(IK_GOAL_OBSTACLE_COST)
            cost = (
                140.0 * err_after_clip
                + motion_cost
                + float(bucket_motion_weight) * bucket_change
                + float(bucket_preference_weight) * bucket_pref_cost
                + goal_obstacle_cost
            )
            row = {
                "cost": float(cost),
                "q": q_candidate.copy(),
                "planar_err": float(err_after_clip),
                "min_z": float(min_z),
                "end_z": float(end_z),
                "angles": angles,
                "end_angle": float(actual_a3),
                "phase_report": phase_report,
                "goal_obstacle_ok": bool(goal_obstacle_ok),
                "goal_obstacle_reason": str(goal_obstacle_reason),
                "goal_obstacle_report": goal_obstacle_report,
            }
            solution_rows.append(row)

            if best is None or cost < best["cost"]:
                best = row
        if planning_deadline_exceeded(deadline):
            break

    if best is None:
        if planning_deadline_exceeded(deadline):
            return ([], "planning budget exceeded") if return_candidates else (None, "planning budget exceeded")
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
            "goal_obstacle_ok": bool(row.get("goal_obstacle_ok", True)),
            "goal_obstacle_reason": str(row.get("goal_obstacle_reason", "")),
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
    deadline=None,
):
    if planning_deadline_exceeded(deadline):
        return None
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
        bucket_candidate_count=(min(DIG_IK_BUCKET_CANDIDATE_COUNT, 9) if deadline is not None else DIG_IK_BUCKET_CANDIDATE_COUNT),
        deadline=deadline,
        score_goal_obstacle=(deadline is None),
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
    deadline=None,
):
    if planning_deadline_exceeded(deadline):
        return [], "planning budget exceeded"
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
    fast_pre_dig = (str(label).lower() == "pre_dig") and (deadline is not None)
    ik_bucket_candidate_count = 3 if fast_pre_dig else (
        min(DIG_IK_BUCKET_CANDIDATE_COUNT, 9) if deadline is not None else DIG_IK_BUCKET_CANDIDATE_COUNT
    )
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
        use_refinement=not fast_pre_dig,
        bucket_candidate_span_deg=DIG_IK_BUCKET_CANDIDATE_SPAN_DEG,
        bucket_candidate_count=ik_bucket_candidate_count,
        return_candidates=True,
        max_solutions=min(int(max_solutions), 2) if fast_pre_dig else max_solutions,
        soft_accept_err=soft_accept_err,
        deadline=deadline,
        score_goal_obstacle=False,
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
    if "clearance" in m:
        return "load"
    if "lift" in m or "carry" in m or "unload" in m:
        return "load"
    return "tip"


def solve_clearance_pose(point, q_seed, end_effector, clearance_z, deadline=None):
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
        bucket_candidate_count=(min(DIG_IK_BUCKET_CANDIDATE_COUNT, 9) if deadline is not None else DIG_IK_BUCKET_CANDIDATE_COUNT),
        deadline=deadline,
        score_goal_obstacle=(deadline is None),
    )


def route_segments_ok(q_start, route, q_goal, mode, samples=None, deadline=None):
    sample_count = PATH_CHECK_SAMPLES if samples is None else max(2, int(samples))
    carry_locked = mode_requires_loaded_carry_bucket(mode)
    check_mode = mode if carry_locked else "clearance"
    q_prev = force_loaded_carry_bucket_q(q_start, reference=q_start, label="route_segments_start") if carry_locked else q_start
    for idx, q_next in enumerate(route):
        if planning_deadline_exceeded(deadline):
            return False, "planning budget exceeded"
        if carry_locked:
            q_next = force_loaded_carry_bucket_q(q_next, reference=q_prev, label="route_segments_waypoint")
        ok, kind, reason, sample, report = path_segment_check(
            q_prev, q_next, check_mode, samples=sample_count, deadline=deadline
        )
        if not ok:
            text = path_block_report_text(check_mode, kind, report, reason)
            return False, f"segment {idx + 1} blocked at {sample}/{sample_count}: {text}"
        q_prev = q_next

    if planning_deadline_exceeded(deadline):
        return False, "planning budget exceeded"
    if carry_locked:
        q_goal = force_loaded_carry_bucket_q(q_goal, reference=q_prev, label="route_segments_goal")
    ok, kind, reason, sample, report = path_segment_check(q_prev, q_goal, mode, samples=sample_count, deadline=deadline)
    if not ok:
        text = path_block_report_text(mode, kind, report, reason)
        return False, f"final segment blocked at {sample}/{sample_count}: {text}"

    return True, "ok"


def clearance_route_cost(q_start, route, q_goal, duration=0.0, clearance_z=0.0, side_offset=0.0):
    q_prev = np.array(q_start, dtype=np.float32).copy()
    total = 0.0
    total_angle = 0.0
    for q_next in list(route) + [q_goal]:
        motion = plan_joint_motion_metrics(q_next, q_prev, duration=duration)
        total += float(motion.get("cost", 0.0))
        total_angle += float(motion.get("weighted_angle", 0.0))
        q_prev = np.array(q_next, dtype=np.float32).copy()
    total += 0.16 * max(0.0, float(clearance_z) - GROUND_TOP_Z)
    total += 0.08 * abs(float(side_offset))
    total += 1.8 * max(0, len(route) - 1)
    return float(total), float(total_angle)


def q_with_joint_degrees(reference_q, joint_degrees, swing_value=None):
    q = np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    for name, value_deg in dict(joint_degrees or {}).items():
        if name not in CTRL.name_to_idx:
            continue
        q[CTRL.name_to_idx[name]] = deg_to_rad(float(value_deg))
    if swing_value is not None and "swing" in CTRL.name_to_idx:
        q[CTRL.name_to_idx["swing"]] = float(swing_value)
    return clip_command_near(q, reference=reference_q)


def q_with_swing_near(reference_q, target_swing):
    q = np.array(reference_q, dtype=np.float32).reshape(-1)[:4].copy()
    swing_idx = CTRL.name_to_idx["swing"]
    q[swing_idx] = float(q[swing_idx]) + float(swing_delta(target_swing, q[swing_idx]))
    return clip_command_near(q, reference=reference_q)


def q_goal_raised_approach(q_goal, q_reference, lift_deg, arm_delta_deg, bucket_blend=0.55):
    q = np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy()
    boom_idx = CTRL.name_to_idx["boom"]
    arm_idx = CTRL.name_to_idx["arm"]
    bucket_idx = CTRL.name_to_idx["bucket"]

    goal_boom = rad_to_deg(q_goal[boom_idx])
    goal_arm = rad_to_deg(q_goal[arm_idx])
    q[boom_idx] = deg_to_rad(goal_boom + float(lift_deg))
    q[arm_idx] = deg_to_rad(goal_arm + float(arm_delta_deg))
    q[bucket_idx] = float(q_goal[bucket_idx]) + float(bucket_blend) * float(q_reference[bucket_idx] - q_goal[bucket_idx])
    return clip_command_near(q, reference=q_reference)


def swing_corridor_cache_key(q_start, q_goal, mode, samples):
    try:
        qa = np.round(np.array(q_start, dtype=np.float32).reshape(-1)[:4], 3)
        qb = np.round(np.array(q_goal, dtype=np.float32).reshape(-1)[:4], 3)
        world_version = int(STATE.get("planning_world_snapshot_version", 0) or 0)
        return (
            str(mode),
            int(samples),
            int(world_version),
            tuple(float(x) for x in qa),
            tuple(float(x) for x in qb),
        )
    except Exception:
        return None


def swing_corridor_summary(q_start, q_goal, mode, samples=25, deadline=None):
    q_start = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    q_goal = clip_route_command_near(q_goal, reference=q_start)
    sample_count = max(9, int(samples))
    cache_key = swing_corridor_cache_key(q_start, q_goal, mode, sample_count)
    cache = STATE.setdefault("planning_swing_corridor_cache", {})
    if bool(STATE.get("dig_plan_planning_active", False)) and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            STATE["planning_swing_corridor_cache_hits"] = int(STATE.get("planning_swing_corridor_cache_hits", 0)) + 1
            STATE["last_swing_corridor"] = dict(cached)
            return dict(cached)
    STATE["planning_swing_corridor_cache_misses"] = int(STATE.get("planning_swing_corridor_cache_misses", 0)) + 1
    swing_idx = CTRL.name_to_idx.get("swing", 0)
    base_pose = PATH_DETERMINISTIC_ROUTE_POSES_DEG[0] if PATH_DETERMINISTIC_ROUTE_POSES_DEG else {}
    q_probe_base = clip_route_command_near(q_with_joint_degrees(q_start, base_pose), reference=q_start)
    if mode_requires_loaded_carry_bucket(mode):
        q_probe_base = force_loaded_carry_bucket_q(q_probe_base, reference=q_start, label="swing_corridor_probe")
    center = float(q_start[swing_idx]) + 0.5 * float(swing_delta(float(q_goal[swing_idx]), float(q_start[swing_idx])))
    span = math.radians(220.0)
    rows = []
    free_runs = []
    current_run = None
    for i in range(sample_count):
        if planning_deadline_exceeded(deadline):
            break
        frac = 0.0 if sample_count <= 1 else float(i) / float(sample_count - 1)
        swing = center - 0.5 * span + span * frac
        q_probe = q_probe_base.copy()
        q_probe[swing_idx] = normalize_swing_cmd(swing)
        ok, kind, reason, _sample, _report = path_segment_check(q_probe, q_probe, mode, samples=1, deadline=deadline)
        deg = rad_to_deg(float(q_probe[swing_idx]))
        rows.append({
            "swing_deg": round(float(deg), 2),
            "ok": bool(ok),
            "kind": str(kind),
            "reason": debug_short_string(reason, 120),
        })
        if ok:
            if current_run is None:
                current_run = [deg, deg]
            else:
                current_run[1] = deg
        elif current_run is not None:
            free_runs.append([round(float(current_run[0]), 2), round(float(current_run[1]), 2)])
            current_run = None
    if current_run is not None:
        free_runs.append([round(float(current_run[0]), 2), round(float(current_run[1]), 2)])
    summary = {
        "mode": str(mode),
        "samples": len(rows),
        "free_count": sum(1 for row in rows if row.get("ok")),
        "blocked_count": sum(1 for row in rows if not row.get("ok")),
        "free_intervals_deg": free_runs,
        "preview": rows[:8],
    }
    STATE["last_swing_corridor"] = summary
    info_print(
        "[SWING CORRIDOR]",
        f"mode={mode}",
        f"free={summary['free_count']}/{summary['samples']}",
        f"intervals={summary['free_intervals_deg'][:4]}",
    )
    debug_timeline_record(
        "SWING_CORRIDOR",
        result="ok" if summary["free_count"] > 0 else "blocked",
        reason=f"free={summary['free_count']}/{summary['samples']}",
        data=summary,
        include_sand=False,
    )
    if (
        bool(STATE.get("dig_plan_planning_active", False))
        and cache_key is not None
        and not planning_deadline_exceeded(deadline)
    ):
        if len(cache) >= int(PLANNING_SWING_CORRIDOR_CACHE_MAX):
            try:
                cache.pop(next(iter(cache)))
            except Exception:
                cache.clear()
        cache[cache_key] = dict(summary)
    return summary


def find_clearance_route(q_start, q_goal, mode, label, deadline=None, samples=None):
    end_effector = path_end_effector_for_mode(mode)
    route_verbose = "staged_unload" in str(label).lower() or "unload_to_bin" in str(mode).lower()
    if planning_deadline_exceeded(deadline):
        if route_verbose:
            info_print(
                "[CLEARANCE ROUTE SKIP]",
                f"label={label}",
                "reason=planning budget exceeded before route start",
                force_log=True,
            )
        return None, "planning budget exceeded before route start"
    carry_locked_route = mode_requires_loaded_carry_bucket(mode, label)
    if carry_locked_route:
        q_start = force_loaded_carry_bucket_q(q_start, reference=q_start, label=f"{label}_route_start")
        q_goal = force_loaded_carry_bucket_q(q_goal, reference=q_start, label=f"{label}_route_goal")
    p_start = predicted_end_world_point(q_start, end_effector=end_effector, reference_q=q_start)
    p_goal = predicted_end_world_point(q_goal, end_effector=end_effector, reference_q=q_start)

    if p_start is None:
        p_start = ik_end_effector_pos(end_effector)
    if p_start is None or p_goal is None:
        return None, "missing predicted start/goal end point"

    last_reason = "no candidate tried"
    obstacle_top = obstacle_top_z_for_segment(p_start, p_goal)
    blockers = obstacle_bboxes_for_segment_xy(p_start, p_goal)
    candidates = []
    sample_count = PATH_CHECK_SAMPLES if samples is None else max(2, int(samples))
    mode_text = f"{mode} {label}".lower()
    pre_dig_route = "pre_dig" in mode_text
    corridor_deadline = child_planning_deadline(deadline, 0.25, min_seconds=0.05) if deadline is not None else None
    corridor = swing_corridor_summary(
        q_start,
        q_goal,
        mode,
        samples=17 if deadline is not None else 25,
        deadline=corridor_deadline,
    )
    corridor_intervals = list((corridor or {}).get("free_intervals_deg", []) or [])
    if route_verbose:
        info_print(
            "[CLEARANCE ROUTE START]",
            f"label={label}",
            f"mode={mode}",
            f"end={end_effector}",
            f"carry_locked={carry_locked_route}",
            f"deadline={'none' if deadline is None else fmt_optional(max(0.0, float(deadline) - time.time())) + 's'}",
            f"p_start={vec_list(p_start, 3)}",
            f"p_goal={vec_list(p_goal, 3)}",
            f"blockers={len(blockers)}",
            f"corridor={corridor_intervals}",
            force_log=True,
        )

    def budget_expired():
        return planning_deadline_exceeded(deadline)

    def swing_corridor_distance_deg(q_pose):
        if not corridor_intervals:
            return 0.0
        try:
            swing_deg = rad_to_deg(float(np.array(q_pose, dtype=np.float32).reshape(-1)[CTRL.name_to_idx["swing"]]))
        except Exception:
            return 0.0
        best = None
        for interval in corridor_intervals:
            try:
                lo = float(interval[0])
                hi = float(interval[1])
            except Exception:
                continue
            if lo <= swing_deg <= hi:
                return 0.0
            dist = min(abs(swing_deg - lo), abs(swing_deg - hi))
            best = dist if best is None else min(best, dist)
        return 0.0 if best is None else float(best)

    def add_route(route, route_type, clearance_z, detail="", side_offset=0.0):
        nonlocal last_reason
        if budget_expired():
            last_reason = "planning budget exceeded"
            return
        route_clipped = []
        q_ref = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
        for q_raw in route or []:
            q_next = clip_route_command_near(q_raw, reference=q_ref)
            if carry_locked_route:
                q_next = force_loaded_carry_bucket_q(q_next, reference=q_ref, label=f"{label}_route_waypoint")
            route_clipped.append(q_next.copy())
            q_ref = q_next.copy()
        route = route_clipped
        ok_route, route_reason = route_segments_ok(q_start, route, q_goal, mode, samples=sample_count, deadline=deadline)
        if not ok_route:
            last_reason = route_reason
            return
        cost, weighted_angle = clearance_route_cost(
            q_start,
            route,
            q_goal,
            duration=PATH_CLEARANCE_DURATION,
            clearance_z=clearance_z,
            side_offset=side_offset,
        )
        corridor_miss = max([swing_corridor_distance_deg(q) for q in [q_start] + list(route) + [q_goal]] or [0.0])
        if carry_locked_route and corridor_miss > 0.0:
            cost += 0.65 * float(corridor_miss)
            detail = f"{detail} corridor_miss={float(corridor_miss):.1f}deg"
        candidates.append({
            "route": [np.array(q, dtype=np.float32).copy() for q in route],
            "type": str(route_type),
            "z": float(clearance_z),
            "cost": float(cost),
            "weighted_angle": float(weighted_angle),
            "detail": str(detail),
            "side_offset": float(side_offset),
        })

    def choose_best_candidate():
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: (float(row["cost"]), float(row["weighted_angle"]), len(row["route"])))[0]

    def try_joint_rrt_route():
        nonlocal last_reason
        if route_verbose:
            info_print(
                "[CLEARANCE ROUTE RRT START]",
                f"label={label}",
                f"remaining={'none' if deadline is None else fmt_optional(max(0.0, float(deadline) - time.time())) + 's'}",
                force_log=True,
            )
        if joint_space_planner is None:
            last_reason = "joint_space_planner unavailable; deterministic routes only"
            if route_verbose:
                info_print("[CLEARANCE ROUTE RRT SKIP]", f"label={label}", last_reason, force_log=True)
            return
        if budget_expired():
            last_reason = "planning budget exceeded"
            if route_verbose:
                info_print("[CLEARANCE ROUTE RRT SKIP]", f"label={label}", last_reason, force_log=True)
            return
        rrt_deadline = None
        if deadline is not None:
            now = time.time()
            remaining = max(0.0, float(deadline) - now)
            if pre_dig_route:
                rrt_deadline = child_planning_deadline(deadline, max(0.20, min(0.85, 0.28 * remaining)), min_seconds=0.10)
            else:
                rrt_deadline = child_planning_deadline(deadline, max(0.25, min(1.20, 0.45 * remaining)), min_seconds=0.12)
        rrt_result = joint_space_planner.plan_joint_space_route(
            runtime_module(),
            q_start,
            q_goal,
            mode=mode,
            label=label,
            deadline=rrt_deadline,
            samples=sample_count,
        )
        if bool(rrt_result.get("ok", False)):
            route = [np.array(q, dtype=np.float32).copy() for q in rrt_result.get("waypoints", [])]
            rrt_stats = rrt_result.get("stats", {}) if isinstance(rrt_result.get("stats", {}), dict) else {}
            add_route(
                route,
                "joint_rrt_connect",
                max(float(p_start[2]), float(p_goal[2])),
                detail=(
                    f"reason={rrt_result.get('reason')} "
                    f"iters={rrt_result.get('iterations')} nodes={rrt_result.get('nodes')} "
                    f"trapped={rrt_stats.get('trapped', 0)} advanced={rrt_stats.get('advanced', 0)} "
                    f"smooth={rrt_stats.get('smooth_accepts', 0)}/{rrt_stats.get('smooth_rejects', 0)} "
                    f"smooth_cost={float(rrt_stats.get('cost_before_smooth', 0.0)):.2f}->{float(rrt_stats.get('cost_after_smooth', 0.0)):.2f} "
                    f"carry_world={fmt_optional(rrt_result.get('carry_world_deg'))}deg "
                    f"elapsed_ms={float(rrt_result.get('elapsed_ms', 0.0)):.1f}"
                ),
            )
        else:
            rrt_stats = rrt_result.get("stats", {}) if isinstance(rrt_result.get("stats", {}), dict) else {}
            last_reason = (
                f"joint_rrt_connect failed: {rrt_result.get('reason', 'unknown')} "
                f"trapped={rrt_stats.get('trapped', 0)} advanced={rrt_stats.get('advanced', 0)} "
                f"last={rrt_stats.get('last_reason', '')}"
            )
            if route_verbose:
                info_print("[CLEARANCE ROUTE RRT FAILED]", f"label={label}", last_reason, force_log=True)

    def add_deterministic_joint_routes():
        nonlocal last_reason
        q_start_arr = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
        q_goal_arr = np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy()
        swing_idx = CTRL.name_to_idx["swing"]
        goal_swing = float(q_goal_arr[swing_idx])
        base_clearance_z = max(float(p_start[2]), float(p_goal[2]), GROUND_TOP_Z + 1.25)
        if obstacle_top is not None:
            base_clearance_z = max(base_clearance_z, float(obstacle_top) + PATH_OBSTACLE_OVER_CLEARANCE_Z)
        if deadline is None:
            candidate_limit = 999999
        elif pre_dig_route:
            candidate_limit = 4
        else:
            candidate_limit = 8

        tried = 0
        detours = PATH_DETERMINISTIC_SWING_DETOURS_DEG if deadline is None else PATH_DETERMINISTIC_SWING_DETOURS_DEG[:7]
        if route_verbose:
            info_print(
                "[CLEARANCE ROUTE DETERMINISTIC START]",
                f"label={label}",
                f"candidate_limit={candidate_limit}",
                f"poses={len(PATH_DETERMINISTIC_ROUTE_POSES_DEG)}",
                f"detours={len(detours)}",
                force_log=True,
            )
        approach_pairs = [
            (PATH_DETERMINISTIC_APPROACH_LIFTS_DEG[0], PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG[0]),
            (PATH_DETERMINISTIC_APPROACH_LIFTS_DEG[1], PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG[1]),
            (PATH_DETERMINISTIC_APPROACH_LIFTS_DEG[2], PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG[2]),
            (PATH_DETERMINISTIC_APPROACH_LIFTS_DEG[3], PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG[2]),
            (PATH_DETERMINISTIC_APPROACH_LIFTS_DEG[1], PATH_DETERMINISTIC_APPROACH_ARM_DELTAS_DEG[3]),
        ]
        for pose_idx, pose in enumerate(PATH_DETERMINISTIC_ROUTE_POSES_DEG):
            if budget_expired():
                break
            q_clear_start = q_with_joint_degrees(q_start_arr, pose)
            q_clear_goal = q_with_swing_near(q_clear_start, goal_swing)
            q_clear_start = clip_command_near(q_clear_start, reference=q_start_arr)
            q_clear_goal = clip_command_near(q_clear_goal, reference=q_clear_start)

            add_route(
                [q_clear_start, q_clear_goal],
                "joint_tuck_swing",
                base_clearance_z,
                detail=f"pose={pose_idx} direct_swing_clearance",
            )
            tried += 1
            if pre_dig_route and candidates:
                return
            if len(candidates) >= candidate_limit:
                return

            for detour_deg in detours:
                if budget_expired():
                    break
                if abs(float(detour_deg)) < 1.0e-4:
                    continue
                q_detour = q_clear_start.copy()
                q_detour[swing_idx] = (
                    float(q_clear_start[swing_idx])
                    + float(swing_delta(goal_swing, q_clear_start[swing_idx]))
                    + deg_to_rad(float(detour_deg))
                )
                q_detour = clip_command_near(q_detour, reference=q_clear_start)
                add_route(
                    [q_clear_start, q_detour, q_clear_goal],
                    "joint_tuck_swing_detour",
                    base_clearance_z,
                    detail=f"pose={pose_idx} detour={float(detour_deg):.1f}deg",
                    side_offset=float(detour_deg) / 45.0,
                )
                tried += 1
                if pre_dig_route and candidates:
                    return
                if len(candidates) >= candidate_limit:
                    return

            for lift_deg, arm_delta_deg in approach_pairs:
                if budget_expired():
                    break
                q_approach = q_goal_raised_approach(
                    q_goal_arr,
                    q_clear_goal,
                    lift_deg=float(lift_deg),
                    arm_delta_deg=float(arm_delta_deg),
                    bucket_blend=0.45,
                )
                add_route(
                    [q_clear_start, q_clear_goal, q_approach],
                    "joint_tuck_swing_high_approach",
                    base_clearance_z,
                    detail=(
                        f"pose={pose_idx} lift={float(lift_deg):.1f}deg "
                        f"arm_delta={float(arm_delta_deg):.1f}deg"
                    ),
                )
                tried += 1
                if pre_dig_route and candidates:
                    return
                if len(candidates) >= candidate_limit:
                    return

        if not candidates and tried > 0:
            last_reason = f"deterministic joint clearance tried={tried} last={last_reason}"
        if route_verbose:
            info_print(
                "[CLEARANCE ROUTE DETERMINISTIC DONE]",
                f"label={label}",
                f"tried={tried}",
                f"candidates={len(candidates)}",
                f"last={last_reason}",
                force_log=True,
            )

    add_deterministic_joint_routes()
    best = choose_best_candidate()
    if best is not None:
        info_print(
            f"[PATH ROUTE SELECTED] {label}: end={end_effector} type={best['type']} "
            f"z={best['z']:.2f} cost={best['cost']:.2f} weighted_angle={best['weighted_angle']:.2f} "
            f"waypoints={len(best['route'])} {best['detail']}"
        )
        return best["route"], str(best["type"])

    midpoint = 0.5 * (np.array(p_start, dtype=np.float32) + np.array(p_goal, dtype=np.float32))

    def info_planar_err(info):
        if isinstance(info, dict):
            try:
                return float(info.get("planar_err", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def solve_clearance_waypoints(points, q_seed, clearance_z):
        route = []
        infos = []
        q_cursor = np.array(q_seed, dtype=np.float32).copy()
        for point in points:
            if budget_expired():
                return None, "planning budget exceeded", infos
            q_next, info_next = solve_clearance_pose(point, q_cursor, end_effector, clearance_z, deadline=deadline)
            if q_next is None:
                return None, info_next, infos
            route.append(np.array(q_next, dtype=np.float32).copy())
            infos.append(info_next)
            q_cursor = np.array(q_next, dtype=np.float32).copy()
        return route, "ok", infos

    def obstacle_corner_route_points(obstacle, clearance_z):
        mn = obstacle["min"]
        mx = obstacle["max"]
        start_xy = np.array(p_start[:2], dtype=np.float32)
        goal_xy = np.array(p_goal[:2], dtype=np.float32)
        out = []
        pad_values = [
            float(PATH_FAST_SIDE_PADDING_XY) + float(extra_pad)
            for extra_pad in PATH_FAST_CORNER_EXTRA_PADS_XY
        ]
        for pad in pad_values:
            left_x = float(mn[0]) - float(PATH_OBSTACLE_MARGIN_XY) - pad
            right_x = float(mx[0]) + float(PATH_OBSTACLE_MARGIN_XY) + pad
            bottom_y = float(mn[1]) - float(PATH_OBSTACLE_MARGIN_XY) - pad
            top_y = float(mx[1]) + float(PATH_OBSTACLE_MARGIN_XY) + pad
            x_sides = [("bbox_left_corner", left_x), ("bbox_right_corner", right_x)]
            y_sides = [("north", top_y), ("south", bottom_y)]
            x_sides = sorted(
                x_sides,
                key=lambda item: abs(float(start_xy[0]) - item[1]) + abs(float(goal_xy[0]) - item[1]),
            )
            y_sides = sorted(
                y_sides,
                key=lambda item: abs(float(start_xy[1]) - item[1]) + abs(float(goal_xy[1]) - item[1]),
            )
            for x_name, side_x in x_sides:
                for y_name, side_y in y_sides:
                    corner = np.array([float(side_x), float(side_y), float(clearance_z)], dtype=np.float32)
                    side_goal = np.array([float(side_x), float(goal_xy[1]), float(clearance_z)], dtype=np.float32)
                    if np.linalg.norm(side_goal[:2] - corner[:2]) < 0.35:
                        route_points = [corner]
                    else:
                        route_points = [corner, side_goal]
                    route_len = float(np.linalg.norm(start_xy - corner[:2]))
                    for a, b in zip(route_points[:-1], route_points[1:]):
                        route_len += float(np.linalg.norm(a[:2] - b[:2]))
                    route_len += float(np.linalg.norm(route_points[-1][:2] - goal_xy))
                    out.append(
                        {
                            "type": f"{x_name}_{y_name}",
                            "points": route_points,
                            "pad": float(pad),
                            "side_offset": float(side_x - midpoint[0]),
                            "route_len": float(route_len),
                            "detail": (
                                f"bbox={obstacle.get('path', '')} "
                                f"corner=({float(side_x):.2f},{float(side_y):.2f}) "
                                f"pad={float(pad):.2f}"
                            ),
                        }
                    )
        return sorted(out, key=lambda row: (float(row["route_len"]), abs(float(row["side_offset"])), float(row["pad"])))

    fast_heights = PATH_FAST_SIDE_HEIGHTS if deadline is None else PATH_FAST_SIDE_HEIGHTS[:2]
    fast_blocker_limit = int(PATH_FAST_SIDE_MAX_BLOCKERS) if deadline is None else 1
    fast_corner_limit = 10 if deadline is None else 4
    if route_verbose:
        info_print(
            "[CLEARANCE ROUTE CORNER START]",
            f"label={label}",
            f"heights={fast_heights}",
            f"blocker_limit={fast_blocker_limit}",
            f"corner_limit={fast_corner_limit}",
            force_log=True,
        )
    for height in fast_heights:
        if budget_expired():
            break
        clearance_z = max(float(p_start[2]), float(p_goal[2]), GROUND_TOP_Z + float(height))
        for obstacle in blockers[: max(1, fast_blocker_limit)]:
            for route_candidate in obstacle_corner_route_points(obstacle, clearance_z)[:fast_corner_limit]:
                if budget_expired():
                    break
                route, fail_info, infos = solve_clearance_waypoints(
                    route_candidate["points"],
                    q_start,
                    clearance_z,
                )
                if route is None:
                    last_reason = f"{route_candidate['type']} route IK failed: {fail_info}"
                    continue
                add_route(
                    route,
                    route_candidate["type"],
                    clearance_z,
                    detail=(
                        f"{route_candidate['detail']} "
                        f"waypoints={len(route)} "
                        f"planar_err={max([info_planar_err(info) for info in infos] or [0.0]):.3f}"
                    ),
                    side_offset=float(route_candidate["side_offset"]),
                )
        best = choose_best_candidate()
        if best is not None:
            info_print(
                f"[PATH ROUTE SELECTED] {label}: end={end_effector} type={best['type']} "
                f"z={best['z']:.2f} cost={best['cost']:.2f} weighted_angle={best['weighted_angle']:.2f} "
                f"waypoints={len(best['route'])} {best['detail']}"
            )
            return best["route"], str(best["type"])

    fallback_heights = PATH_CLEARANCE_HEIGHTS if deadline is None else PATH_CLEARANCE_HEIGHTS[:3]
    fallback_fractions = PATH_CLEARANCE_FRACTIONS if deadline is None else [0.50]
    fallback_side_offsets = PATH_ROUTE_SIDE_OFFSETS if deadline is None else PATH_ROUTE_SIDE_OFFSETS[:1]
    if route_verbose:
        info_print(
            "[CLEARANCE ROUTE FALLBACK START]",
            f"label={label}",
            f"heights={fallback_heights}",
            f"fractions={fallback_fractions}",
            f"side_offsets={fallback_side_offsets}",
            f"remaining={'none' if deadline is None else fmt_optional(max(0.0, float(deadline) - time.time())) + 's'}",
            force_log=True,
        )
    for height in fallback_heights:
        if budget_expired():
            break
        base_clearance_z = max(float(p_start[2]), float(p_goal[2]), GROUND_TOP_Z + float(height))
        clearance_z = base_clearance_z
        if obstacle_top is not None:
            clearance_z = max(clearance_z, float(obstacle_top) + PATH_OBSTACLE_OVER_CLEARANCE_Z)

        for frac in fallback_fractions:
            if budget_expired():
                break
            p_via = (1.0 - float(frac)) * np.array(p_start, dtype=np.float32) + float(frac) * np.array(p_goal, dtype=np.float32)
            p_via[2] = clearance_z

            q_via, info = solve_clearance_pose(p_via, q_start, end_effector, clearance_z, deadline=deadline)

            if q_via is None:
                last_reason = f"clearance IK failed: {info}"
                continue

            add_route(
                [q_via],
                "over_via",
                clearance_z,
                detail=f"frac={float(frac):.2f} planar_err={info['planar_err']:.3f}",
            )

        q_up_start, info_start = solve_clearance_pose(p_start, q_start, end_effector, clearance_z, deadline=deadline)
        if q_up_start is None:
            last_reason = f"start lift IK failed: {info_start}"
            continue

        q_up_goal, info_goal = solve_clearance_pose(p_goal, q_up_start, end_effector, clearance_z, deadline=deadline)
        if q_up_goal is None:
            last_reason = f"goal lift IK failed: {info_goal}"
            continue

        route = [q_up_start, q_up_goal]
        add_route(route, "lift_cross_drop", clearance_z, detail=f"waypoints={len(route)}")

        direction_xy = np.array(p_goal[:2], dtype=np.float32) - np.array(p_start[:2], dtype=np.float32)
        perp = safe_norm(np.array([-float(direction_xy[1]), float(direction_xy[0])], dtype=np.float32), default=(1.0, 0.0))
        for side_offset in fallback_side_offsets:
            for sign in [-1.0, 1.0]:
                if budget_expired():
                    break
                p_side = midpoint.copy()
                p_side[:2] += float(sign) * float(side_offset) * perp
                p_side[2] = base_clearance_z
                q_side, info_side = solve_clearance_pose(p_side, q_up_start, end_effector, base_clearance_z, deadline=deadline)
                if q_side is None:
                    last_reason = f"side route IK failed: {info_side}"
                    continue
                route = [q_up_start, q_side, q_up_goal]
                add_route(
                    route,
                    "lift_side_cross_drop",
                    clearance_z,
                    detail=f"side={float(sign) * float(side_offset):.2f} waypoints={len(route)}",
                    side_offset=float(sign) * float(side_offset),
                )

    best = choose_best_candidate()
    if best is not None:
        info_print(
            f"[PATH ROUTE SELECTED] {label}: end={end_effector} type={best['type']} "
            f"z={best['z']:.2f} cost={best['cost']:.2f} weighted_angle={best['weighted_angle']:.2f} "
            f"waypoints={len(best['route'])} {best['detail']}"
        )
        return best["route"], str(best["type"])

    try_joint_rrt_route()
    best = choose_best_candidate()
    if best is not None:
        info_print(
            f"[PATH ROUTE SELECTED] {label}: end={end_effector} type={best['type']} "
            f"z={best['z']:.2f} cost={best['cost']:.2f} weighted_angle={best['weighted_angle']:.2f} "
            f"waypoints={len(best['route'])} {best['detail']}"
        )
        return best["route"], str(best["type"])

    if route_verbose:
        info_print("[CLEARANCE ROUTE FAILED]", f"label={label}", f"reason={last_reason}", force_log=True)
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


async def move_to_profile_with_clearance(q_goal, seconds=1.0, label="", task_id=None, mode="auto", q_start_override=None):
    if q_start_override is not None:
        try:
            q_reference = CTRL.clip_limits(np.array(q_start_override, dtype=np.float32).reshape(-1)[:4].copy())
        except Exception:
            q_reference = CTRL.q_cmd.copy()
    else:
        q_reference = CTRL.q_cmd.copy()
    q_goal = clip_command_near(q_goal, reference=q_reference)
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
            reason_text = f"execution_failed/path_precheck_failed:{label}:no_collision_free_route:{route_reason}"
            set_execution_failure_reason(reason_text)
            info_print(
                f"[PATH OBSTACLE NO ROUTE] {label}: predicted obstacle but no route found; "
                f"{route_reason}; hard_stop=True"
            )
            update_status(f"[PATH OBSTACLE NO ROUTE] {label}: {route_reason}", force=True)
            debug_timeline_record(
                "PATH_OBSTACLE_NO_ROUTE",
                stage=label,
                result="failed",
                reason=reason_text,
                q_cmd=q_goal,
                q_real=get_real_joint_positions(),
                include_sand=True,
            )
            return False
        if strict_path_precheck_phase(mode) and not phase_ok:
            reason_text = f"execution_failed/path_precheck_failed:{label}:no_clearance_route:{route_reason}"
            set_execution_failure_reason(reason_text)
            info_print(f"[PATH PHASE NO ROUTE] {label}: {route_reason}; hard_stop=True")
            update_status(f"[PATH PHASE NO ROUTE] {label}: {route_reason}", force=True)
            return False
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
    target_depth = max(0.0, min(0.24, float(surface_z) - float(target[2])))
    inward = dig_direction_unit(target)
    if float(np.linalg.norm(inward)) < 1e-6:
        inward = np.array([-1.0, 0.0], dtype=np.float32)
    outward = -inward

    def cut_z(depth, min_clearance=0.035):
        depth = max(0.0, float(depth))
        return max(GROUND_TOP_Z + float(min_clearance), float(surface_z) - depth)

    cut_depth = max(0.08, min(0.22, target_depth))
    insert_depth = max(0.035, min(float(candidate.get("insert_depth", 0.08)), cut_depth * 0.55))
    mid_depth = max(insert_depth + 0.025, min(float(candidate.get("mid_depth", 0.14)), cut_depth))
    exit_depth = max(0.015, min(float(candidate.get("exit_depth", 0.04)), insert_depth))

    pre_clearance = max(0.26, float(candidate.get("pre_z", 0.46)) - min(0.24, target_depth))
    contact_clearance = max(0.015, min(0.08, float(candidate.get("contact_z", 0.03))))
    pre = offset_xy(target, outward, float(candidate.get("approach_offset", 0.25)), float(surface_z) + pre_clearance)
    contact = offset_xy(target, outward, max(0.16, float(candidate.get("approach_offset", 0.25)) * 0.70), float(surface_z) + contact_clearance)
    insert = offset_xy(target, outward, max(0.10, float(candidate.get("approach_offset", 0.25)) * 0.42), cut_z(insert_depth))
    mid_cut = offset_xy(target, inward, float(candidate.get("mid_pull", 0.35)), cut_z(mid_depth))
    exit_lift_z = max(0.03, float(candidate.get("exit_lift_z", 0.08)))
    exit_cut_z = max(cut_z(exit_depth), float(surface_z) + exit_lift_z)
    exit_cut = offset_xy(target, inward, float(candidate.get("exit_pull", 0.55)), exit_cut_z)
    curl = offset_xy(exit_cut, inward, 0.02, max(float(surface_z) + 0.22, float(exit_cut[2]) + float(candidate.get("curl_z", 0.18))))
    secure = offset_xy(curl, inward, 0.02, max(float(curl[2]) + 0.18, float(surface_z) + 0.52))
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
        ("curl_to_hold_material", curl, float(candidate.get("bucket_curl", CURL_HOLD_TARGET_DEG)), None, "tip", 0.9, True),
        ("secure_load", secure, None, "hold", "load", 0.85, True),
        ("lift_carry", lift, None, "hold", "load", 1.2, True),
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

    def budget_failure(label, partial=None, reasons=None):
        partial = partial if isinstance(partial, dict) else best_partial
        reason_text = "; ".join([str(x) for x in (reasons or []) if str(x)]) or "planning budget exceeded"
        if "planning budget exceeded" not in reason_text:
            reason_text = "planning budget exceeded; " + reason_text
        partial_stages = list(partial.get("stages", []))
        return None, list(partial.get("points", [])), {
            "id": str(candidate.get("id", "candidate")),
            "dig_primitive": compact_dig_primitive_params(candidate),
            "planned": False,
            "failed_stage": label,
            "failure_reason": reason_text,
            "planned_prefix": len(partial.get("seq", [])),
            "best_partial_cost": float(partial.get("cost", 0.0)),
            "best_partial_q_deg": q_deg_values(partial.get("q", CTRL.q_cmd), wrap_swing_for_display=True),
            "stages": partial_stages,
            "route_diagnostics": route_diagnostics_from_stages(partial_stages),
            "unload_ballistics": unload_ballistics_from_stages(partial_stages),
            "fsm_contract": validate_dig_plan_contract(
                seq=partial.get("seq", []),
                points=partial.get("points", []),
                stages=partial_stages,
                trace_points=[],
            ),
        }

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

    def routed_stage_components(q_seed, q_goal, label, duration, target_point, effector):
        if planning_deadline_exceeded(deadline):
            return None, "planning budget exceeded"
        pre_dig_primary_route = str(label).lower() == "pre_dig"
        pre_dig_obstacle_context = False
        pre_dig_corridor_report = None
        pre_dig_route_deadline = deadline
        if pre_dig_primary_route:
            try:
                route_effector = path_end_effector_for_mode(label)
                p0 = predicted_end_world_point(q_seed, end_effector=route_effector, reference_q=q_seed)
                p1 = predicted_end_world_point(q_goal, end_effector=route_effector, reference_q=q_seed)
                if p0 is not None and p1 is not None:
                    blockers = obstacle_bboxes_for_segment_xy(p0, p1)
                    pre_dig_obstacle_context = bool(blockers)
                    if blockers:
                        pre_dig_corridor_report = obstacle_corridor_report(
                            p0,
                            p1,
                            obstacle=blockers[0],
                            link_name=route_effector,
                        )
            except Exception:
                pre_dig_obstacle_context = False
        direct_phase_ok, direct_phase_reason, direct_phase_sample, direct_phase_report = path_phase_check(
            q_seed, q_goal, label, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
        )
        if planning_deadline_exceeded(deadline):
            return None, "planning budget exceeded"
        if pre_dig_primary_route and pre_dig_obstacle_context:
            direct_obstacle_ok = False
            direct_obstacle_reason = "pre_dig corridor intersects rigid obstacle; route required before direct sweep"
            direct_obstacle_sample = 0
            direct_obstacle_report = pre_dig_corridor_report
            if deadline is not None:
                now = time.time()
                remaining = max(0.0, float(deadline) - now)
                pre_dig_route_deadline = child_planning_deadline(
                    deadline,
                    max(0.35, min(1.10, 0.45 * remaining)),
                    min_seconds=0.12,
                )
        else:
            direct_obstacle_ok, direct_obstacle_reason, direct_obstacle_sample, direct_obstacle_report = path_obstacle_check(
                q_seed, q_goal, label, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
            )
        route_required = not (direct_phase_ok and direct_obstacle_ok)
        route_preferred = bool(pre_dig_primary_route and pre_dig_obstacle_context)
        route_waypoints = []
        route_reason = "direct_ok"
        if route_required or route_preferred:
            route_waypoints, route_reason = find_clearance_route(
                q_seed,
                q_goal,
                label,
                f"plan_{candidate.get('id', 'candidate')}_{label}",
                deadline=pre_dig_route_deadline,
                samples=PATH_ROUTE_PLANNING_SAMPLE_COUNT,
            )
            if route_waypoints is None:
                if route_preferred and not route_required and direct_phase_ok and direct_obstacle_ok:
                    route_waypoints = []
                    route_reason = f"direct_ok; preferred pre_dig route unavailable: {route_reason}"
                else:
                    detail = []
                    if not direct_phase_ok:
                        detail.append(
                            f"phase sample={direct_phase_sample}/{PATH_CHECK_SAMPLES} "
                            + format_ground_report(label, direct_phase_report, direct_phase_reason)
                        )
                    if not direct_obstacle_ok:
                        detail.append(
                            f"obstacle sample={direct_obstacle_sample}/{PATH_CHECK_SAMPLES} "
                            + format_obstacle_report(label, direct_obstacle_report, direct_obstacle_reason)
                        )
                    return None, f"{label}: no collision-free route: {route_reason}; " + "; ".join(detail)
            else:
                route_required = True
                route_reason = str(route_reason)
        route_seq = []
        route_points = []
        route_stages = []
        route_cost = 0.0
        route_weighted_angle = 0.0
        route_estimated_time = 0.0
        q_motion_seed = np.array(q_seed, dtype=np.float32).copy()
        route_end_effector = path_end_effector_for_mode(label)
        path_eval_deadline = pre_dig_route_deadline if route_preferred else deadline
        route_prevalidated = bool(route_waypoints)
        for route_idx, q_route_raw in enumerate(route_waypoints or []):
            if planning_deadline_exceeded(path_eval_deadline):
                return None, "planning budget exceeded"
            q_route = np.array(q_route_raw, dtype=np.float32).copy()
            route_label = f"clearance_route_{route_idx + 1}"
            route_duration = max(0.45, min(PATH_CLEARANCE_DURATION, float(duration) * 0.65))
            route_motion = plan_joint_motion_metrics(q_route, q_motion_seed, route_duration)
            route_duration = max(route_duration, float(route_motion.get("estimated_time", route_duration) or route_duration))
            route_motion = plan_joint_motion_metrics(q_route, q_motion_seed, route_duration)
            if route_prevalidated:
                route_path_penalty = 0.0
                route_path_detail = {
                    "phase_ok": True,
                    "obstacle_ok": True,
                    "phase_reason": "prevalidated_by_clearance_route",
                    "obstacle_reason": "prevalidated_by_clearance_route",
                    "phase_sample": int(PATH_ROUTE_PLANNING_SAMPLE_COUNT),
                    "obstacle_sample": int(PATH_ROUTE_PLANNING_SAMPLE_COUNT),
                    "path_penalty": 0.0,
                    "path_check_reused": True,
                    "route_prevalidated": True,
                }
            else:
                route_path_penalty, route_path_detail = plan_path_penalty(
                    q_motion_seed, q_route, "clearance", deadline=path_eval_deadline
                )
                if planning_deadline_exceeded(path_eval_deadline):
                    return None, "planning budget exceeded"
                if not bool(route_path_detail.get("phase_ok", True)) or not bool(route_path_detail.get("obstacle_ok", True)):
                    return None, (
                        f"{label}: clearance route segment still invalid; "
                        f"phase={route_path_detail.get('phase_reason')} obstacle={route_path_detail.get('obstacle_reason')}"
                    )
            route_target = predicted_end_world_point(q_route, end_effector=route_end_effector, reference_q=q_motion_seed)
            if route_target is None:
                route_target = np.array(target_point, dtype=np.float32).copy()
            route_stage_cost = float(route_motion["cost"] + route_path_penalty)
            route_seq.append((route_label, q_route.copy(), float(route_duration)))
            route_points.append(np.array(route_target, dtype=np.float32).copy())
            route_stages.append(
                {
                    "phase": route_label,
                    "planned": True,
                    "required": True,
                    "target_point": vec_list(route_target, 3),
                    "q_goal_rad": vec_list(q_route, 4),
                    "q_goal_deg": q_deg_values(q_route, wrap_swing_for_display=True),
                    "duration": float(route_duration),
                    "effector": route_end_effector,
                    "route_source": "obstacle_clearance",
                    "route_reason": str(route_reason),
                    "route_index": int(route_idx + 1),
                    "route_count": int(len(route_waypoints or [])),
                    "motion": route_motion,
                    "path": route_path_detail,
                    "stage_cost": route_stage_cost,
                }
            )
            route_cost += route_stage_cost
            route_weighted_angle += float(route_motion["weighted_angle"])
            route_estimated_time += float(route_motion["estimated_time"])
            q_motion_seed = q_route.copy()

        if planning_deadline_exceeded(path_eval_deadline):
            return None, "planning budget exceeded"
        if route_prevalidated:
            path_penalty = 0.0
            path_detail = {
                "phase_ok": True,
                "obstacle_ok": True,
                "phase_reason": "prevalidated_by_clearance_route",
                "obstacle_reason": "prevalidated_by_clearance_route",
                "phase_sample": int(PATH_ROUTE_PLANNING_SAMPLE_COUNT),
                "obstacle_sample": int(PATH_ROUTE_PLANNING_SAMPLE_COUNT),
                "path_penalty": 0.0,
                "path_check_reused": True,
                "route_prevalidated": True,
                "route_inserted": True,
                "route_waypoints": len(route_waypoints or []),
                "route_reason": str(route_reason),
                "direct_phase_ok": bool(direct_phase_ok),
                "direct_phase_reason": str(direct_phase_reason),
                "direct_obstacle_ok": bool(direct_obstacle_ok),
                "direct_obstacle_reason": str(direct_obstacle_reason),
            }
        elif (
            not route_required
            and bool(direct_phase_ok)
            and bool(direct_obstacle_ok)
            and not planning_deadline_exceeded(path_eval_deadline)
        ):
            path_penalty = 0.0
            path_detail = {
                "phase_ok": True,
                "obstacle_ok": True,
                "phase_reason": str(direct_phase_reason),
                "obstacle_reason": str(direct_obstacle_reason),
                "phase_sample": int(direct_phase_sample),
                "obstacle_sample": int(direct_obstacle_sample),
                "path_penalty": 0.0,
                "path_check_reused": True,
            }
        else:
            path_penalty, path_detail = plan_path_penalty(q_motion_seed, q_goal, label, deadline=path_eval_deadline)
        if planning_deadline_exceeded(path_eval_deadline):
            return None, "planning budget exceeded"
        if route_required:
            path_detail["route_inserted"] = bool(route_waypoints)
            path_detail["route_waypoints"] = len(route_waypoints or [])
            path_detail["route_reason"] = str(route_reason)
            path_detail["direct_phase_ok"] = bool(direct_phase_ok)
            path_detail["direct_phase_reason"] = str(direct_phase_reason)
            path_detail["direct_obstacle_ok"] = bool(direct_obstacle_ok)
            path_detail["direct_obstacle_reason"] = str(direct_obstacle_reason)
        if not bool(path_detail.get("obstacle_ok", True)):
            return None, f"{label}: final segment still hits rigid obstacle: {path_detail.get('obstacle_reason')}"
        if strict_path_precheck_phase(label) and not bool(path_detail.get("phase_ok", True)):
            return None, f"{label}: final segment violates ground/path guard: {path_detail.get('phase_reason')}"
        return {
            "route_seq": route_seq,
            "route_points": route_points,
            "route_stages": route_stages,
            "route_cost": float(route_cost),
            "route_weighted_angle": float(route_weighted_angle),
            "route_estimated_time": float(route_estimated_time),
            "q_motion_seed": q_motion_seed.copy(),
            "path_penalty": float(path_penalty),
            "path_detail": path_detail,
        }, "ok"

    for label, point, bucket_deg, bucket_world_deg, ik_effector, duration, required in specs:
        if planning_deadline_exceeded(deadline):
            return budget_failure(label)
        new_beams = []
        fail_reasons = []

        for beam in beams:
            if planning_deadline_exceeded(deadline):
                fail_reasons.append("planning budget exceeded")
                break
            q_seed = np.array(beam["q"], dtype=np.float32).copy()
            if label == "unload_to_bin":
                dump_deg = float(candidate.get("unload_dump_deg", BUCKET_UNLOAD_DUMP_DEG))
                q_release_align, dump_info = plan_dump_pose_to_bin(
                    q_seed=q_seed,
                    dump_deg=dump_deg,
                    label=f"plan_{candidate.get('id', 'candidate')}_{label}",
                    log=False,
                    max_correction_iters=2,
                    allow_unaligned=True,
                    deadline=deadline,
                )
                q_dump_final = None
                if q_release_align is None:
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
                        deadline=deadline,
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
                    q_release_align = bucket_only_dump_pose(q_pre_dump, unload_release_alignment_bucket_deg(dump_deg))
                    q_dump_final = bucket_only_dump_pose(q_pre_dump, dump_deg)
                    drop = unload_drop_report(q=q_release_align, reference_q=q_pre_dump)
                    info_print(
                        "[DIG PLAN UNLOAD FALLBACK]",
                        f"candidate={candidate.get('id', 'candidate')}",
                        f"reason={dump_info}",
                        f"fallback_drop_xy_err={fmt_optional(drop.get('xy_err'))}",
                        f"inside_xy={drop.get('inside_xy')}",
                        f"close_xy={drop.get('close_xy')}",
                    )
                else:
                    q_pre_dump = np.array(q_release_align, dtype=np.float32).copy()
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
                    q_dump_final = bucket_only_dump_pose(q_pre_dump, dump_deg)
                    drop = unload_drop_report(q=q_release_align, reference_q=q_pre_dump)

                landing = unload_bin_landing_point()
                drop_ready = unload_drop_execution_ready(drop)
                if not drop_ready:
                    info_print(
                        "[DIG PLAN UNLOAD DIAG]",
                        f"candidate={candidate.get('id', 'candidate')}",
                        "drop_alignment=diagnostic_only",
                        f"{label} dump-ready pose failed: predicted drop not aligned: "
                        f"xy_err={fmt_optional(drop.get('xy_err'))} "
                        f"inside_xy={drop.get('inside_xy')} above_wall={drop.get('above_wall')} "
                        f"close_xy={drop.get('close_xy')} "
                        f"scatter_xy_ok={drop.get('scatter_xy_ok')} "
                        f"acceptance={drop.get('landing_acceptance')} "
                        f"source_clearance={fmt_optional(drop.get('source_clearance'))}",
                    )

                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                direct_phase_ok, direct_phase_reason, direct_phase_sample, direct_phase_report = path_phase_check(
                    q_seed, q_pre_dump, label, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
                )
                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                direct_obstacle_ok, direct_obstacle_reason, direct_obstacle_sample, direct_obstacle_report = path_obstacle_check(
                    q_seed, q_pre_dump, label, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
                )
                route_waypoints = []
                route_reason = "direct_ok"
                route_required = not (direct_phase_ok and direct_obstacle_ok)
                if route_required:
                    route_waypoints, route_reason = find_clearance_route(
                        q_seed,
                        q_pre_dump,
                        label,
                        f"plan_{candidate.get('id', 'candidate')}_{label}",
                        deadline=deadline,
                        samples=PATH_ROUTE_PLANNING_SAMPLE_COUNT,
                    )
                    if route_waypoints is None:
                        detail = []
                        if not direct_phase_ok:
                            detail.append(
                                f"phase sample={direct_phase_sample}/{PATH_CHECK_SAMPLES} "
                                + format_ground_report(label, direct_phase_report, direct_phase_reason)
                            )
                        if not direct_obstacle_ok:
                            detail.append(
                                f"obstacle sample={direct_obstacle_sample}/{PATH_CHECK_SAMPLES} "
                                + format_obstacle_report(label, direct_obstacle_report, direct_obstacle_reason)
                            )
                        fail_reasons.append(
                            f"{label}: no collision-free route: {route_reason}; " + "; ".join(detail)
                        )
                        continue

                route_seq = []
                route_points = []
                route_stages = []
                route_cost = 0.0
                route_weighted_angle = 0.0
                route_estimated_time = 0.0
                q_motion_seed = q_seed.copy()
                route_end_effector = path_end_effector_for_mode(label)
                for route_idx, q_route_raw in enumerate(route_waypoints or []):
                    if planning_deadline_exceeded(deadline):
                        fail_reasons.append("planning budget exceeded")
                        break
                    q_route = np.array(q_route_raw, dtype=np.float32).copy()
                    route_label = f"clearance_route_{route_idx + 1}"
                    route_duration = max(0.45, min(PATH_CLEARANCE_DURATION, float(duration) * 0.65))
                    route_motion = plan_joint_motion_metrics(q_route, q_motion_seed, route_duration)
                    route_duration = max(route_duration, float(route_motion.get("estimated_time", route_duration) or route_duration))
                    route_motion = plan_joint_motion_metrics(q_route, q_motion_seed, route_duration)
                    route_path_penalty, route_path_detail = plan_path_penalty(
                        q_motion_seed, q_route, "clearance", deadline=deadline
                    )
                    if planning_deadline_exceeded(deadline):
                        fail_reasons.append("planning budget exceeded")
                        break
                    route_target = predicted_end_world_point(q_route, end_effector=route_end_effector, reference_q=q_motion_seed)
                    if route_target is None:
                        route_target = np.array(landing, dtype=np.float32).copy()
                    route_stage_cost = float(route_motion["cost"] + route_path_penalty)
                    route_seq.append((route_label, q_route.copy(), float(route_duration)))
                    route_points.append(np.array(route_target, dtype=np.float32).copy())
                    route_stages.append(
                        {
                            "phase": route_label,
                            "planned": True,
                            "required": True,
                            "target_point": vec_list(route_target, 3),
                            "q_goal_rad": vec_list(q_route, 4),
                            "q_goal_deg": q_deg_values(q_route, wrap_swing_for_display=True),
                            "duration": float(route_duration),
                            "effector": route_end_effector,
                            "route_source": "obstacle_clearance",
                            "route_reason": str(route_reason),
                            "route_index": int(route_idx + 1),
                            "route_count": int(len(route_waypoints or [])),
                            "motion": route_motion,
                            "path": route_path_detail,
                            "stage_cost": route_stage_cost,
                        }
                    )
                    route_cost += route_stage_cost
                    route_weighted_angle += float(route_motion["weighted_angle"])
                    route_estimated_time += float(route_motion["estimated_time"])
                    q_motion_seed = q_route.copy()
                if planning_deadline_exceeded(deadline):
                    break

                motion = plan_joint_motion_metrics(q_pre_dump, q_motion_seed, duration)
                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                if not route_required and bool(direct_phase_ok) and bool(direct_obstacle_ok):
                    path_penalty = 0.0
                    path_detail = {
                        "phase_ok": True,
                        "obstacle_ok": True,
                        "phase_reason": str(direct_phase_reason),
                        "obstacle_reason": str(direct_obstacle_reason),
                        "phase_sample": int(direct_phase_sample),
                        "obstacle_sample": int(direct_obstacle_sample),
                        "path_penalty": 0.0,
                        "path_check_reused": True,
                    }
                else:
                    path_penalty, path_detail = plan_path_penalty(q_motion_seed, q_pre_dump, label, deadline=deadline)
                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                if route_required:
                    path_detail["route_inserted"] = bool(route_waypoints)
                    path_detail["route_waypoints"] = len(route_waypoints or [])
                    path_detail["route_reason"] = str(route_reason)
                    path_detail["direct_phase_ok"] = bool(direct_phase_ok)
                    path_detail["direct_phase_reason"] = str(direct_phase_reason)
                    path_detail["direct_obstacle_ok"] = bool(direct_obstacle_ok)
                    path_detail["direct_obstacle_reason"] = str(direct_obstacle_reason)
                xy_err = float(drop.get("xy_err", 1.0) or 1.0)
                overflow_xy = float(drop.get("bin_overflow_xy", xy_err) or 0.0)
                effective_xy_err = overflow_xy if unload_drop_execution_ready(drop) else xy_err
                clearance_short = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - float(drop.get("source_clearance", 0.0) or 0.0))
                outside_penalty = 0.0 if bool(drop.get("inside_xy", False)) else 8.0
                scatter_penalty = 0.0 if unload_drop_execution_ready(drop) else 10.0
                close_penalty = 0.0 if bool(drop.get("close_xy", False) or drop.get("scatter_xy_ok", False)) else 4.0
                height_bonus = min(2.5, max(0.0, float(drop.get("source_clearance", 0.0) or 0.0) - float(UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z))) * 3.0
                unload_penalty = (
                    float(DIG_PLAN_UNLOAD_XY_COST) * effective_xy_err
                    + 18.0 * clearance_short
                    + outside_penalty
                    + close_penalty
                    + scatter_penalty
                    - height_bonus
                )
                total_cost = float(beam["cost"]) + route_cost + motion["cost"] + path_penalty + unload_penalty
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(landing, 3),
                    "q_goal_rad": vec_list(q_pre_dump, 4),
                    "q_dump_rad": vec_list(q_dump_final, 4),
                    "q_release_align_rad": vec_list(q_release_align, 4),
                    "q_goal_deg": q_deg_values(q_pre_dump, wrap_swing_for_display=True),
                    "q_dump_deg": q_deg_values(q_dump_final, wrap_swing_for_display=True),
                    "q_release_align_deg": q_deg_values(q_release_align, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": "landing",
                    "drop": compact_unload_drop(drop),
                    "drop_alignment_ready": bool(drop_ready),
                    "drop_alignment_policy": str(drop.get("landing_acceptance", "scatter_tolerant_execute_then_score")),
                    "clearance_route": {
                        "inserted": bool(route_waypoints),
                        "required": bool(route_required),
                        "waypoints": len(route_waypoints or []),
                        "reason": str(route_reason),
                    },
                    "motion": motion,
                    "path": path_detail,
                    "stage_cost": float(motion["cost"] + path_penalty + unload_penalty),
                }
                new_beams.append(
                    {
                        "q": q_pre_dump.copy(),
                        "seq": beam["seq"] + route_seq + [(label, q_pre_dump.copy(), float(duration))],
                        "points": beam["points"] + route_points + [np.array(landing, dtype=np.float32).copy()],
                        "stages": beam["stages"] + route_stages + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + route_weighted_angle + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + route_estimated_time + float(motion["estimated_time"]),
                    }
                )
                continue

            if label == "curl_to_hold_material":
                bucket_idx = CTRL.name_to_idx["bucket"]
                boom_idx = CTRL.name_to_idx.get("boom", 1)
                target_bucket_deg = float(bucket_deg if bucket_deg is not None else candidate.get("bucket_curl", CURL_HOLD_TARGET_DEG))
                target_bucket_rad = deg_to_rad(target_bucket_deg)
                curl_candidates = []

                def add_curl_candidate(q_raw, source, reason="", extra_lift_deg=0.0, pose_info=None):
                    if planning_deadline_exceeded(deadline):
                        return
                    if q_raw is None:
                        return
                    q_candidate = np.array(q_raw, dtype=np.float32).reshape(-1)[:4].copy()
                    if extra_lift_deg:
                        q_candidate[boom_idx] = float(q_candidate[boom_idx]) + deg_to_rad(float(extra_lift_deg))
                    q_candidate[bucket_idx] = target_bucket_rad
                    q_candidate = clip_command_near(q_candidate, reference=q_seed)
                    bucket_after = float(rad_to_deg(q_candidate[bucket_idx]))
                    closed_deficit = max(0.0, bucket_after - float(CURL_HOLD_ACCEPT_BUCKET_DEG))
                    if deadline is not None:
                        phase_ok, phase_reason, phase_sample, _phase_report = path_phase_check(
                            q_seed, q_candidate, label, samples=2, deadline=deadline
                        )
                        obstacle_ok, obstacle_reason, obstacle_sample, _obstacle_report = path_obstacle_check(
                            q_seed, q_candidate, label, samples=2, deadline=deadline
                        )
                        motion = plan_joint_motion_metrics(q_candidate, q_seed, duration)
                        path_penalty = 0.0
                        if not phase_ok:
                            path_penalty += float(DIG_PLAN_PATH_SOFT_PENALTY)
                        if not obstacle_ok:
                            path_penalty += float(DIG_PLAN_OBSTACLE_SOFT_PENALTY)
                        path_detail = {
                            "phase_ok": bool(phase_ok),
                            "obstacle_ok": bool(obstacle_ok),
                            "phase_reason": str(phase_reason),
                            "obstacle_reason": str(obstacle_reason),
                            "phase_sample": int(phase_sample),
                            "obstacle_sample": int(obstacle_sample),
                            "route_inserted": False,
                            "route_reason": "auto_budget_direct_material_hold",
                            "path_penalty": path_penalty,
                            "budget_fast_path": True,
                        }
                        route_cost = 0.0
                        route_weighted_angle = 0.0
                        route_estimated_time = 0.0
                        route_seq = []
                        route_points = []
                        route_stages = []
                        routed_ok = bool(phase_ok and obstacle_ok)
                    else:
                        routed, route_reason = routed_stage_components(q_seed, q_candidate, label, duration, point, "tip")
                        if routed is None:
                            motion = plan_joint_motion_metrics(q_candidate, q_seed, duration)
                            path_penalty = float(DIG_PLAN_OBSTACLE_SOFT_PENALTY + DIG_PLAN_PATH_SOFT_PENALTY)
                            path_detail = {
                                "phase_ok": False,
                                "obstacle_ok": False,
                                "phase_reason": str(route_reason),
                                "obstacle_reason": str(route_reason),
                                "route_inserted": False,
                                "route_reason": str(route_reason),
                                "path_penalty": path_penalty,
                            }
                            route_cost = 0.0
                            route_weighted_angle = 0.0
                            route_estimated_time = 0.0
                            route_seq = []
                            route_points = []
                            route_stages = []
                            routed_ok = False
                        else:
                            motion = plan_joint_motion_metrics(q_candidate, routed["q_motion_seed"], duration)
                            path_penalty = float(routed["path_penalty"])
                            path_detail = routed["path_detail"]
                            route_cost = float(routed["route_cost"])
                            route_weighted_angle = float(routed["route_weighted_angle"])
                            route_estimated_time = float(routed["route_estimated_time"])
                            route_seq = routed["route_seq"]
                            route_points = routed["route_points"]
                            route_stages = routed["route_stages"]
                            routed_ok = True
                    curl_report = predicted_phase_ground_report(q_candidate, label, reference_q=q_seed)
                    curl_ok, curl_reason = phase_ground_ok(label, curl_report)
                    curl_hold_q, curl_hold_report = carry_hold_adjusted_q(q_candidate, q_candidate, end_effector="load")
                    curl_retains = bool((curl_hold_report or {}).get("retains_material", False))
                    retain_penalty = 0.0 if curl_retains else 10.0
                    closed_penalty = 0.0 if curl_retains else 80.0 * closed_deficit
                    ground_penalty = 0.0 if curl_ok else 120.0
                    tip_point = predicted_end_world_point(q_candidate, end_effector="tip", reference_q=q_seed)
                    if tip_point is None:
                        tip_point = np.array(point, dtype=np.float32).copy()
                    curl_candidates.append(
                        {
                            "q": q_candidate.copy(),
                            "source": str(source),
                            "reason": str(reason),
                            "extra_lift_deg": float(extra_lift_deg),
                            "pose_info": dict(pose_info or {}),
                            "bucket_deg": bucket_after,
                            "joint_closed_ok_deprecated": bool(closed_deficit <= 0.05),
                            "closed_ok": bool(curl_retains),
                            "retains": bool(curl_retains),
                            "motion": motion,
                            "path_detail": path_detail,
                            "path_penalty": float(path_penalty),
                            "route_cost": float(route_cost),
                            "route_weighted_angle": float(route_weighted_angle),
                            "route_estimated_time": float(route_estimated_time),
                            "route_seq": route_seq,
                            "route_points": route_points,
                            "route_stages": route_stages,
                            "ground_report": curl_report,
                            "ground_ok": bool(curl_ok and routed_ok),
                            "ground_reason": str(curl_reason if routed_ok else path_detail.get("route_reason", "route_failed")),
                            "hold_report": curl_hold_report,
                            "retain_penalty": float(retain_penalty),
                            "closed_penalty": float(closed_penalty),
                            "ground_penalty": float(ground_penalty),
                            "target_point": np.array(tip_point, dtype=np.float32).copy(),
                            "score": float(route_cost + motion["cost"] + path_penalty + retain_penalty + closed_penalty + ground_penalty),
                        }
                    )

                # Curl is a material-retention action: bucket closure is the hard
                # constraint, while exact tip position is only a soft/diagnostic target.
                add_curl_candidate(q_seed, "bucket_close_only", "preserve_pull_exit_pose", extra_lift_deg=0.0)
                add_curl_candidate(
                    q_seed,
                    "bucket_close_with_boom_lift",
                    "seal_bucket_then_small_lift",
                    extra_lift_deg=float(candidate.get("curl_boom_lift_deg", 4.5)),
                )
                try:
                    curl_rows, curl_reason = solve_dig_pose_candidates(
                        label,
                        point,
                        target_bucket_deg,
                        duration,
                        q_seed,
                        bucket_world_deg=None,
                        ik_effector="tip",
                        accept_err=0.62,
                        soft_accept_err=0.90,
                        bucket_motion_weight=0.50,
                        bucket_preference_weight=8.0,
                        max_solutions=2,
                        deadline=deadline,
                    )
                except Exception as e:
                    curl_rows = []
                    curl_reason = f"{type(e).__name__}:{e}"
                for row in curl_rows or []:
                    if planning_deadline_exceeded(deadline):
                        break
                    q_row = np.array(row.get("q_goal"), dtype=np.float32).copy()
                    row_info = dict(row.get("info", {}) or {})
                    row_info["source"] = "curl_tip_ik_bucket_forced_closed"
                    row_info["raw_ik_bucket_deg"] = float(rad_to_deg(q_row[bucket_idx]))
                    add_curl_candidate(q_row, "curl_tip_ik_bucket_forced_closed", str(curl_reason), pose_info=row_info)

                valid_curl_candidates = [
                    row for row in curl_candidates
                    if row["ground_ok"] and row["retains"]
                ]
                if not valid_curl_candidates:
                    if planning_deadline_exceeded(deadline):
                        fail_reasons.append("planning budget exceeded")
                        break
                    if curl_candidates:
                        best_diag = sorted(curl_candidates, key=lambda row: row["score"])[0]
                        fail_reasons.append(
                            f"{label}: no geometrically retaining safe curl pose; "
                            f"best_source={best_diag['source']} bucket={best_diag['bucket_deg']:.2f}deg "
                            f"joint_closed={best_diag.get('joint_closed_ok_deprecated')} retains={best_diag['retains']} ground={best_diag['ground_ok']} "
                            f"ground_reason={best_diag['ground_reason']}"
                        )
                    else:
                        fail_reasons.append(f"{label}: no curl candidates; {curl_reason}")
                    continue

                best_curl = sorted(
                    valid_curl_candidates,
                    key=lambda row: (
                        0 if row["retains"] else 1,
                        float(row["score"]),
                        sum(q_delta_abs_deg(row["q"], q_seed)),
                    ),
                )[0]
                q_goal = best_curl["q"].copy()
                curl_bucket_deg = float(best_curl["bucket_deg"])
                curl_closed_ok = bool(best_curl["retains"])
                curl_hold_report = best_curl["hold_report"]
                curl_retains = bool(best_curl["retains"])
                if not curl_closed_ok:
                    fail_reasons.append(
                        f"{label}: bucket geometry does not retain material; bucket={curl_bucket_deg:.2f}deg "
                        f"pour_above_load_z={fmt_optional((curl_hold_report or {}).get('pour_above_load_z'))}"
                    )
                    continue
                motion = best_curl["motion"]
                path_penalty = float(best_curl["path_penalty"])
                path_detail = best_curl["path_detail"]
                route_seq = list(best_curl.get("route_seq", []))
                route_points = list(best_curl.get("route_points", []))
                route_stages = list(best_curl.get("route_stages", []))
                route_cost = float(best_curl.get("route_cost", 0.0) or 0.0)
                route_weighted_angle = float(best_curl.get("route_weighted_angle", 0.0) or 0.0)
                route_estimated_time = float(best_curl.get("route_estimated_time", 0.0) or 0.0)
                curl_report = best_curl["ground_report"]
                curl_ok = bool(best_curl["ground_ok"])
                curl_reason = str(best_curl["ground_reason"])
                if not curl_ok:
                    fail_reasons.append(f"{label}: {curl_reason}")
                    continue
                retain_penalty = float(best_curl["retain_penalty"])
                target_point = best_curl["target_point"]
                total_cost = float(beam["cost"]) + route_cost + motion["cost"] + path_penalty + retain_penalty
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(target_point, 3),
                    "q_goal_rad": vec_list(q_goal, 4),
                    "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": "bucket_closure_primary",
                    "seal_bucket_first": True,
                    "bucket_target_deg": curl_bucket_deg,
                    "material_hold": {
                        "bucket_closed_ok": bool(curl_closed_ok),
                        "carry_retains_material": bool(curl_retains),
                        "carry_report": curl_hold_report,
                        "penalty": float(retain_penalty),
                    },
                    "curl_pose": {
                        "source": best_curl["source"],
                        "reason": best_curl["reason"],
                        "extra_lift_deg": best_curl["extra_lift_deg"],
                        "target_bucket_deg": target_bucket_deg,
                        "candidate_count": len(curl_candidates),
                        "pose_info": best_curl["pose_info"],
                    },
                    "ground": {
                        "ok": bool(curl_ok),
                        "reason": str(curl_reason),
                        "tip_depth": curl_report.get("tip_sand_depth"),
                        "bucket_mid_depth": curl_report.get("bucket_mid_sand_depth"),
                        "pour_depth": curl_report.get("pour_sand_depth"),
                        "load_depth": curl_report.get("load_sand_depth"),
                        "surface_source": curl_report.get("tip_sand_surface_source"),
                    },
                    "motion": motion,
                    "path": path_detail,
                    "clearance_route": {
                        "inserted": bool(route_seq),
                        "required": bool(route_seq),
                        "waypoints": len(route_seq),
                    },
                    "stage_cost": float(route_cost + motion["cost"] + path_penalty + retain_penalty),
                }
                new_beams.append(
                    {
                        "q": q_goal.copy(),
                        "seq": beam["seq"] + route_seq + [(label, q_goal.copy(), float(duration))],
                        "points": beam["points"] + route_points + [np.array(target_point, dtype=np.float32).copy()],
                        "stages": beam["stages"] + route_stages + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + route_weighted_angle + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + route_estimated_time + float(motion["estimated_time"]),
                    }
                )
                continue

            if label == "secure_load":
                boom_idx = CTRL.name_to_idx.get("boom", 1)
                arm_idx = CTRL.name_to_idx.get("arm", 2)
                bucket_idx = CTRL.name_to_idx.get("bucket", 3)
                secure_rows = []

                for boom_lift_deg, arm_retract_deg in [
                    (3.0, 0.0),
                    (4.5, -1.5),
                    (6.0, -2.5),
                    (7.5, -3.0),
                    (2.0, 1.0),
                ]:
                    if planning_deadline_exceeded(deadline):
                        secure_rows.append({"ok": False, "reason": "planning budget exceeded"})
                        break
                    q_secure = q_seed.copy()
                    q_secure[boom_idx] = float(q_secure[boom_idx]) + deg_to_rad(float(boom_lift_deg))
                    q_secure[arm_idx] = float(q_secure[arm_idx]) + deg_to_rad(float(arm_retract_deg))
                    q_secure = clip_command_near(q_secure, reference=q_seed)
                    q_secure_base = q_secure.copy()
                    q_secure, carry_report = carry_hold_adjusted_q(q_secure, q_seed, end_effector="load")
                    if not bool((carry_report or {}).get("ok", False)):
                        secure_rows.append({
                            "ok": False,
                            "reason": f"carry hold failed: {(carry_report or {}).get('reason', 'unknown')}",
                            "q": q_secure.copy(),
                            "carry_report": carry_report,
                        })
                        continue
                    if not bool((carry_report or {}).get("retains_material", False)):
                        secure_rows.append({
                            "ok": False,
                            "reason": f"carry would spill; pour_above_load_z={fmt_optional((carry_report or {}).get('pour_above_load_z'))}",
                            "q": q_secure.copy(),
                            "carry_report": carry_report,
                        })
                        continue
                    secure_report = predicted_phase_ground_report(q_secure, label, reference_q=q_seed)
                    secure_ok, secure_reason = phase_ground_ok(label, secure_report)
                    if not secure_ok:
                        secure_rows.append({
                            "ok": False,
                            "reason": str(secure_reason),
                            "q": q_secure.copy(),
                            "carry_report": carry_report,
                        })
                        continue
                    if deadline is not None:
                        phase_ok, phase_reason, phase_sample, _phase_report = path_phase_check(
                            q_seed, q_secure, label, samples=2, deadline=deadline
                        )
                        obstacle_ok, obstacle_reason, obstacle_sample, _obstacle_report = path_obstacle_check(
                            q_seed, q_secure, label, samples=2, deadline=deadline
                        )
                        if not (phase_ok and obstacle_ok):
                            secure_rows.append({
                                "ok": False,
                                "reason": (
                                    "auto_budget_direct_secure_load_failed: "
                                    f"phase={phase_reason} obstacle={obstacle_reason}"
                                ),
                                "q": q_secure.copy(),
                                "carry_report": carry_report,
                            })
                            continue
                        routed = {
                            "q_motion_seed": q_seed.copy(),
                            "route_seq": [],
                            "route_points": [],
                            "route_stages": [],
                            "route_cost": 0.0,
                            "route_weighted_angle": 0.0,
                            "route_estimated_time": 0.0,
                            "path_penalty": 0.0,
                            "path_detail": {
                                "phase_ok": True,
                                "obstacle_ok": True,
                                "phase_reason": str(phase_reason),
                                "obstacle_reason": str(obstacle_reason),
                                "phase_sample": int(phase_sample),
                                "obstacle_sample": int(obstacle_sample),
                                "route_inserted": False,
                                "route_reason": "auto_budget_direct_secure_load",
                                "path_penalty": 0.0,
                                "budget_fast_path": True,
                            },
                        }
                    else:
                        routed, route_reason = routed_stage_components(q_seed, q_secure, label, duration, point, ik_effector)
                        if routed is None:
                            secure_rows.append({
                                "ok": False,
                                "reason": str(route_reason),
                                "q": q_secure.copy(),
                                "carry_report": carry_report,
                            })
                            continue
                    motion = plan_joint_motion_metrics(q_secure, routed["q_motion_seed"], duration)
                    target_point = predicted_end_world_point(q_secure, end_effector="load", reference_q=routed["q_motion_seed"])
                    if target_point is None:
                        target_point = np.array(point, dtype=np.float32).copy()
                    score = (
                        float(routed["route_cost"])
                        + float(motion["cost"])
                        + float(routed["path_penalty"])
                        + carry_spill_risk_penalty(carry_report)
                        - 12.0 * max(0.0, float((carry_report or {}).get("pour_above_load_z", 0.0) or 0.0))
                    )
                    secure_rows.append(
                        {
                            "ok": True,
                            "score": float(score),
                            "q": q_secure.copy(),
                            "boom_lift_deg": float(boom_lift_deg),
                            "arm_retract_deg": float(arm_retract_deg),
                            "carry_report": carry_report,
                            "ground_report": secure_report,
                            "routed": routed,
                            "motion": motion,
                            "target_point": np.array(target_point, dtype=np.float32).copy(),
                        }
                    )

                valid_secure = [row for row in secure_rows if bool(row.get("ok", False))]
                if not valid_secure:
                    if planning_deadline_exceeded(deadline):
                        fail_reasons.append("planning budget exceeded")
                        break
                    best_diag = sorted(
                        secure_rows,
                        key=lambda row: float((row.get("carry_report") or {}).get("pour_above_load_z", -999.0) or -999.0),
                        reverse=True,
                    )[0] if secure_rows else {}
                    fail_reasons.append(f"{label}: no retaining secure-load pose; {best_diag.get('reason', 'no candidates')}")
                    continue

                best_secure = sorted(valid_secure, key=lambda row: float(row["score"]))[0]
                q_goal = best_secure["q"].copy()
                routed = best_secure["routed"]
                motion = best_secure["motion"]
                carry_report = best_secure["carry_report"]
                stage_row = {
                    "phase": label,
                    "planned": True,
                    "required": bool(required),
                    "target_point": vec_list(best_secure["target_point"], 3),
                    "q_goal_rad": vec_list(q_goal, 4),
                    "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
                    "duration": float(duration),
                    "effector": "load",
                    "secure_load": {
                        "boom_lift_deg": float(best_secure["boom_lift_deg"]),
                        "arm_retract_deg": float(best_secure["arm_retract_deg"]),
                        "candidate_count": len(secure_rows),
                    },
                    "material_hold": carry_report,
                    "ground": {
                        "ok": True,
                        "reason": "ok",
                        "tip_depth": best_secure["ground_report"].get("tip_sand_depth"),
                        "bucket_mid_depth": best_secure["ground_report"].get("bucket_mid_sand_depth"),
                        "pour_depth": best_secure["ground_report"].get("pour_sand_depth"),
                        "load_depth": best_secure["ground_report"].get("load_sand_depth"),
                        "surface_source": best_secure["ground_report"].get("tip_sand_surface_source"),
                    },
                    "motion": motion,
                    "path": routed["path_detail"],
                    "clearance_route": {
                        "inserted": bool(routed["route_seq"]),
                        "required": bool(routed["route_seq"]),
                        "waypoints": len(routed["route_seq"]),
                    },
                    "stage_cost": float(routed["route_cost"] + motion["cost"] + routed["path_penalty"]),
                }
                new_beams.append(
                    {
                        "q": q_goal.copy(),
                        "seq": beam["seq"] + routed["route_seq"] + [(label, q_goal.copy(), float(duration))],
                        "points": beam["points"] + routed["route_points"] + [best_secure["target_point"].copy()],
                        "stages": beam["stages"] + routed["route_stages"] + [stage_row],
                        "cost": float(beam["cost"]) + float(routed["route_cost"]) + float(motion["cost"]) + float(routed["path_penalty"]),
                        "weighted_angle": float(beam["weighted_angle"]) + float(routed["route_weighted_angle"]) + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + float(routed["route_estimated_time"]) + float(motion["estimated_time"]),
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
                deadline=deadline,
            )
            if not pose_rows:
                fail_reasons.append(f"{label}: {pose_reason}")
                continue

            for pose in pose_rows:
                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                q_goal = np.array(pose["q_goal"], dtype=np.float32).copy()
                carry_report = None
                if label == "lift_carry":
                    q_goal, carry_report = carry_hold_adjusted_q(q_goal, q_seed, end_effector="load")
                    if not bool((carry_report or {}).get("ok", False)):
                        fail_reasons.append(f"{label}: carry hold failed: {(carry_report or {}).get('reason', 'unknown')}")
                        continue
                    if not bool((carry_report or {}).get("retains_material", False)):
                        fail_reasons.append(
                            f"{label}: carry angle would spill material; "
                            f"pour_above_load_z={fmt_optional((carry_report or {}).get('pour_above_load_z'))}"
                        )
                        continue
                info = pose.get("info", {})
                routed, route_reason = routed_stage_components(q_seed, q_goal, label, duration, point, ik_effector)
                if routed is None:
                    fail_reasons.append(str(route_reason))
                    continue
                if planning_deadline_exceeded(deadline):
                    fail_reasons.append("planning budget exceeded")
                    break
                motion = plan_joint_motion_metrics(q_goal, routed["q_motion_seed"], duration)
                path_penalty = float(routed["path_penalty"])
                path_detail = routed["path_detail"]
                front_report = predicted_phase_ground_report(q_goal, label, reference_q=q_seed)
                front_ok, front_reason, front_penalty = cut_front_edge_quality(label, front_report)
                if not front_ok:
                    fail_reasons.append(f"{label}: {front_reason}")
                    continue
                ik_penalty = float(DIG_PLAN_IK_ERR_COST) * float(info.get("planar_err", 0.0) or 0.0)
                angle_penalty = 0.35 * max(0.0, float(info.get("world_angle_err_deg", 0.0) or 0.0))
                total_cost = (
                    float(beam["cost"])
                    + float(routed["route_cost"])
                    + motion["cost"]
                    + path_penalty
                    + ik_penalty
                    + angle_penalty
                    + front_penalty
                    + carry_spill_risk_penalty(carry_report)
                )
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
                    "front_edge": {
                        "ok": bool(front_ok),
                        "reason": str(front_reason),
                        "tip_depth": front_report.get("tip_sand_depth"),
                        "bucket_mid_depth": front_report.get("bucket_mid_sand_depth"),
                        "pour_depth": front_report.get("pour_sand_depth"),
                        "load_depth": front_report.get("load_sand_depth"),
                        "surface_source": front_report.get("tip_sand_surface_source"),
                    },
                    "material_hold": {} if carry_report is None else carry_report,
                    "clearance_route": {
                        "inserted": bool(routed["route_seq"]),
                        "required": bool(routed["route_seq"]),
                        "waypoints": len(routed["route_seq"]),
                    },
                    "stage_cost": float(
                        routed["route_cost"]
                        + motion["cost"]
                        + path_penalty
                        + ik_penalty
                        + angle_penalty
                        + front_penalty
                        + carry_spill_risk_penalty(carry_report)
                    ),
                }
                if bucket_world_use is not None:
                    stage_row["bucket_world_deg"] = float(bucket_world_use)
                new_beams.append(
                    {
                        "q": q_goal.copy(),
                        "seq": beam["seq"] + routed["route_seq"] + [(label, q_goal.copy(), float(duration))],
                        "points": beam["points"] + routed["route_points"] + [np.array(point, dtype=np.float32).copy()],
                        "stages": beam["stages"] + routed["route_stages"] + [stage_row],
                        "cost": total_cost,
                        "weighted_angle": float(beam["weighted_angle"]) + float(routed["route_weighted_angle"]) + float(motion["weighted_angle"]),
                        "estimated_time": float(beam["estimated_time"]) + float(routed["route_estimated_time"]) + float(motion["estimated_time"]),
                    }
                )

        if planning_deadline_exceeded(deadline):
            if new_beams and label == specs[-1][0]:
                info_print(
                    "[DIG PLAN DEADLINE GRACE]",
                    f"candidate={candidate.get('id', 'candidate')}",
                    f"final_stage={label}",
                    f"beams={len(new_beams)}",
                    "accepting_complete_plan=True",
                )
            else:
                partial = sorted(new_beams, key=lambda x: float(x.get("cost", 1.0e9)))[0] if new_beams else best_partial
                return budget_failure(label, partial=partial, reasons=fail_reasons)

        if not new_beams:
            prefix_beams = sorted(beams, key=lambda x: float(x.get("cost", 1.0e9)))
            best_partial = prefix_beams[0] if prefix_beams else best_partial
            last_reason = "; ".join(fail_reasons[:4]) if fail_reasons else f"{label} produced no beam"
            partial_stages = list(best_partial.get("stages", []))
            return None, list(best_partial.get("points", [])), {
                "id": str(candidate.get("id", "candidate")),
                "dig_primitive": compact_dig_primitive_params(candidate),
                "planned": False,
                "failed_stage": label,
                "failure_reason": last_reason,
                "planned_prefix": len(best_partial.get("seq", [])),
                "best_partial_cost": float(best_partial.get("cost", 0.0)),
                "best_partial_q_deg": q_deg_values(best_partial.get("q", CTRL.q_cmd), wrap_swing_for_display=True),
                "stages": partial_stages,
                "route_diagnostics": route_diagnostics_from_stages(partial_stages),
                "unload_ballistics": unload_ballistics_from_stages(partial_stages),
                "fsm_contract": validate_dig_plan_contract(
                    seq=best_partial.get("seq", []),
                    points=best_partial.get("points", []),
                    stages=partial_stages,
                    trace_points=[],
                ),
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
    best_stages = list(best.get("stages", []))
    fsm_contract = validate_dig_plan_contract(
        seq=best.get("seq", []),
        points=best.get("points", []),
        stages=best_stages,
        trace_points=[],
    )
    return best["seq"], best["points"], {
        "id": str(candidate.get("id", "candidate")),
        "dig_primitive": compact_dig_primitive_params(candidate),
        "planned": True,
        "steps": len(best["seq"]),
        "stages": best_stages,
        "unload_point_xyz": vec_list(unload_release_point, 3),
        "unload_release_xyz": vec_list(unload_release_point, 3),
        "unload_landing_xyz": vec_list(unload_landing_point, 3),
        "planner_cost": float(best["cost"]),
        "weighted_angle": float(best["weighted_angle"]),
        "estimated_time": float(best["estimated_time"]),
        "beam_size": int(DIG_PLAN_BEAM_SIZE),
        "route_diagnostics": route_diagnostics_from_stages(best_stages),
        "unload_ballistics": unload_ballistics_from_stages(best_stages),
        "fsm_contract": fsm_contract,
    }


def evaluate_dig_plan_candidate(seq, points, candidate, stages=None):
    if not seq:
        return -1.0e9, "no_sequence", []

    q_prev = CTRL.q_cmd.copy()
    penalties = 0.0
    reports = []
    stage_rows = [row for row in (stages or []) if isinstance(row, dict) and row.get("q_goal_rad") is not None]
    if stage_rows:
        iterable = []
        for row in stage_rows:
            try:
                q_goal = np.array(row.get("q_goal_rad"), dtype=np.float32).reshape(-1)[:4]
                if len(q_goal) < 4:
                    continue
            except Exception:
                continue
            iterable.append((str(row.get("phase", "")), q_goal.copy(), float(row.get("duration", 0.0) or 0.0), row))
    else:
        iterable = [(stage_name, np.array(q_goal, dtype=np.float32).copy(), float(duration), None) for stage_name, q_goal, duration in seq]

    for stage_name, q_goal, duration, stage_row in iterable:
        if isinstance(stage_row, dict):
            path = stage_row.get("path") if isinstance(stage_row.get("path"), dict) else {}
            motion = stage_row.get("motion") if isinstance(stage_row.get("motion"), dict) else {}
            phase_ok = bool(path.get("phase_ok", True))
            obstacle_ok = bool(path.get("obstacle_ok", True))
            phase_reason = str(path.get("phase_reason", "cached"))
            obstacle_reason = str(path.get("obstacle_reason", "cached"))
            phase_sample = int(path.get("phase_sample", -1) or -1)
            obstacle_sample = int(path.get("obstacle_sample", -1) or -1)
            phase_report = None
            obstacle_report = None
            deltas = motion.get("joint_delta_deg")
            if not isinstance(deltas, (list, tuple)) or not deltas:
                deltas = q_delta_abs_deg(q_goal, q_prev)
            motion_cost = float(motion.get("cost", 0.025 * float(sum(float(x) for x in deltas))) or 0.0)
        else:
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
            "joint_delta_deg": [float(x) for x in deltas],
            "motion_cost": motion_cost,
            "source": "cached_stage" if isinstance(stage_row, dict) else "recomputed",
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
            drop_ready = unload_drop_execution_ready(drop)
            overflow_xy = float(drop.get("bin_overflow_xy", xy_err) or 0.0)
            effective_xy_err = overflow_xy if drop_ready else xy_err
            clearance_short = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - float(drop.get("source_clearance", 0.0)))
            outside_penalty = 0.0 if bool(drop.get("inside_xy", False)) else 8.0
            scatter_penalty = 0.0 if drop_ready else 10.0
            height_bonus = min(2.5, max(0.0, float(drop.get("source_clearance", 0.0)) - float(UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z))) * 3.0
            alignment_cost = 28.0 * effective_xy_err + 18.0 * clearance_short + outside_penalty + scatter_penalty - height_bonus
            dump_reason = (
                f"drop_xy_err={xy_err:.3f}; release_clearance={float(drop.get('source_clearance', 0.0)):.3f}; "
                f"inside_xy={drop.get('inside_xy')}; scatter_xy_ok={drop.get('scatter_xy_ok')}; "
                f"acceptance={drop.get('landing_acceptance')}"
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
    fsm_contract = validate_dig_plan_contract(
        seq=seq,
        points=points,
        stages=chosen_row.get("stages", []),
        trace_points=trace_points,
    )
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
        "dig_primitive": chosen_row.get("dig_primitive", {}),
        "route_diagnostics": route_diagnostics_from_stages(chosen_row.get("stages", [])),
        "unload_ballistics": chosen_row.get("unload_ballistics", {}) or unload_ballistics_from_stages(chosen_row.get("stages", [])),
        "fsm_contract": fsm_contract,
        "staged_execution": bool(chosen_row.get("staged_execution", False)),
        "staged_prefix_ready": bool(chosen_row.get("staged_prefix_ready", False)),
        "staged_post_secure_load_pending": bool(chosen_row.get("staged_post_secure_load_pending", False)),
        "staged_source_failure": chosen_row.get("staged_source_failure", {}),
        "total_plan_cost": None if chosen_row.get("planner_cost") is None else float(chosen_row.get("planner_cost")),
        "rank_cost": None if chosen_row.get("rank_cost") is None else float(chosen_row.get("rank_cost")),
        "score": None if chosen_row.get("score") is None else float(chosen_row.get("score")),
        "estimated_duration": float(chosen_row.get("estimated_time", total_duration) or total_duration),
        "stage_count": len(seq),
        "debug": {
            "build_ms": float(STATE.get("dig_plan_last_build_ms", 0.0)),
            "planning_version": int(STATE.get("dig_plan_planning_version", 0)),
            "start_q_deg": q_deg_values(STATE.get("dig_plan_start_q", CTRL.q_cmd), wrap_swing_for_display=True),
            "rigid_collision_model": {
                "type": "link_segment_vs_expanded_aabb",
                "link_radius_m": float(PATH_LINK_COLLISION_RADIUS_M),
                "segment_samples": int(PATH_LINK_COLLISION_SEGMENT_SAMPLES),
                "obstacle_margin_xy": float(PATH_OBSTACLE_MARGIN_XY),
                "obstacle_margin_z": float(PATH_OBSTACLE_MARGIN_Z),
            },
            "joint_space_route_planner": {
                "type": "rrt_connect_with_shortcut_fallback_bbox",
                "max_iters": int(PATH_RRT_MAX_ITERS),
                "step_deg": float(PATH_RRT_STEP_DEG),
                "goal_bias": float(PATH_RRT_GOAL_BIAS),
                "joint_weights": list(PATH_RRT_JOINT_WEIGHTS),
                "smooth_rounds": int(PATH_RRT_SMOOTH_ROUNDS),
                "smooth_alpha": float(PATH_RRT_SMOOTH_ALPHA),
                "smooth_bend_weight": float(PATH_RRT_SMOOTH_BEND_WEIGHT),
                "smooth_min_improvement": float(PATH_RRT_SMOOTH_MIN_IMPROVEMENT),
            },
            "planning_world": planning_world_snapshot(force=False),
            "path_penalty_cache": {
                "hits": int(STATE.get("planning_path_penalty_cache_hits", 0)),
                "misses": int(STATE.get("planning_path_penalty_cache_misses", 0)),
                "entries": len(STATE.get("planning_path_penalty_cache", {}) or {}),
            },
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
    q_release_align = _planned_stage_q(selected, "q_release_align_rad")
    if q_goal is None and q_dump is None and q_release_align is None:
        return None
    return {
        "phase": str(selected.get("phase", "")),
        "q_goal": None if q_goal is None else q_goal.copy(),
        "q_dump": None if q_dump is None else q_dump.copy(),
        "q_release_align": None if q_release_align is None else q_release_align.copy(),
        "drop": selected.get("drop", {}),
        "target_point": selected.get("target_point"),
        "duration": selected.get("duration"),
    }


def dig_plan_staged_prefix_requirements():
    out = []
    for phase in DIG_PLAN_REQUIRED_PHASE_ORDER:
        out.append(phase)
        if phase == "pull_exit_cut":
            break
    return out


def seq_points_from_plan_stages(stages, fallback_point=None):
    seq = []
    points = []
    fallback = None if fallback_point is None else np.array(fallback_point, dtype=np.float32).reshape(-1)[:3].copy()
    for stage in stages or []:
        if not isinstance(stage, dict):
            continue
        q_goal = _planned_stage_q(stage, "q_goal_rad")
        if q_goal is None:
            continue
        phase = str(stage.get("phase", ""))
        if not phase:
            continue
        try:
            duration = float(stage.get("duration", 0.8) or 0.8)
        except Exception:
            duration = 0.8
        seq.append((phase, q_goal.copy(), max(0.08, duration)))
        point = stage.get("target_point")
        try:
            p = np.array(point, dtype=np.float32).reshape(-1)[:3].copy()
            if len(p) < 3:
                p = None
        except Exception:
            p = None
        if p is None:
            p = predicted_end_world_point(q_goal, end_effector=path_end_effector_for_mode(phase), reference_q=CTRL.q_cmd)
        if p is None:
            p = fallback.copy() if fallback is not None else np.zeros(3, dtype=np.float32)
        points.append(np.array(p, dtype=np.float32).reshape(-1)[:3].copy())
    return seq, points


def staged_prefix_stages_from_failure(failure_row):
    if not isinstance(failure_row, dict):
        return [], "missing_failure_row"
    stages = [stage for stage in failure_row.get("stages", []) if isinstance(stage, dict)]
    if not stages:
        return [], "missing_partial_stages"

    prefix = []
    main_semantic = []
    for stage in stages:
        prefix.append(dict(stage))
        sem = dig_plan_semantic_phase_name(str(stage.get("phase", "")))
        if sem != "clearance_route":
            main_semantic.append(sem)
        if sem == "secure_load":
            break

    required = dig_plan_staged_prefix_requirements()
    cursor = 0
    missing = []
    for phase in required:
        try:
            found = main_semantic.index(phase, cursor)
            cursor = found + 1
        except ValueError:
            missing.append(phase)
    if missing:
        return [], "missing_staged_prefix_phases:" + ",".join(missing)
    return prefix, "ok"


def install_staged_prefix_plan_from_failure(target_xyz, failure_row):
    prefix_stages, reason = staged_prefix_stages_from_failure(failure_row)
    if not prefix_stages:
        info_print("[DIG PLAN STAGED SKIP]", reason)
        return None

    seq, points = seq_points_from_plan_stages(prefix_stages, fallback_point=target_xyz)
    if not seq:
        info_print("[DIG PLAN STAGED SKIP] empty staged sequence")
        return None

    chosen = dict(failure_row)
    chosen["planned"] = True
    chosen["selected"] = True
    chosen["staged_execution"] = True
    chosen["staged_prefix_ready"] = True
    semantic = [dig_plan_semantic_phase_name(str(stage.get("phase", ""))) for stage in prefix_stages]
    prefix_has_secure = "secure_load" in semantic
    chosen["staged_post_secure_load_pending"] = bool(prefix_has_secure)
    chosen["staged_post_dig_secure_pending"] = bool((not prefix_has_secure) and ("pull_exit_cut" in semantic))
    chosen["staged_prefix_terminal_phase"] = str(next((phase for phase in reversed(semantic) if phase != "clearance_route"), ""))
    chosen["staged_source_failure"] = {
        "failed_stage": str(failure_row.get("failed_stage", "")),
        "failure_reason": str(failure_row.get("failure_reason", "")),
        "planned_prefix": int(failure_row.get("planned_prefix", len(seq)) or len(seq)),
        "prefix_terminal_phase": chosen["staged_prefix_terminal_phase"],
    }
    chosen["steps"] = len(seq)
    chosen["stages"] = prefix_stages
    chosen["unload_landing_xyz"] = vec_list(unload_bin_landing_point(), 3)
    chosen["unload_release_xyz"] = vec_list(unload_bin_dump_point(), 3)
    chosen["unload_point_xyz"] = chosen["unload_release_xyz"]
    chosen["planner_cost"] = float(failure_row.get("best_partial_cost", 0.0) or 0.0)
    chosen["rank_cost"] = chosen["planner_cost"]
    chosen["score"] = float(1000.0 - chosen["planner_cost"])
    chosen["score_reason"] = (
        "staged_prefix_from_best_partial; "
        f"source_failed_stage={chosen['staged_source_failure']['failed_stage']}; "
        f"source_reason={chosen['staged_source_failure']['failure_reason']}"
    )
    chosen["route_diagnostics"] = route_diagnostics_from_stages(prefix_stages)
    chosen["unload_ballistics"] = {}

    STATE["dig_plan_candidate"] = chosen
    STATE["dig_plan_points"] = [np.array(p, dtype=np.float32).copy() for p in points]
    cache_dig_plan_trace_points(seq, start_q=STATE.get("dig_plan_start_q", CTRL.q_cmd.copy()))
    plan = build_shared_dig_plan_object(target_xyz, seq, points, chosen)
    plan["staged_execution"] = True
    plan["staged_prefix_ready"] = True
    plan["staged_post_secure_load_pending"] = bool(prefix_has_secure)
    plan["staged_post_dig_secure_pending"] = bool((not prefix_has_secure) and ("pull_exit_cut" in semantic))
    plan["staged_prefix_terminal_phase"] = chosen["staged_prefix_terminal_phase"]
    plan["staged_source_failure"] = dict(chosen["staged_source_failure"])
    info_print(
        "[DIG PLAN STAGED]",
        f"steps={len(seq)}",
        f"prefix_through={chosen['staged_prefix_terminal_phase']}",
        f"source_failed_stage={chosen['staged_source_failure']['failed_stage']}",
        f"reason={chosen['staged_source_failure']['failure_reason']}",
    )
    return seq


def make_stage_row_from_q(phase, q_goal, q_from, duration, target_point=None, extra=None, deadline=None):
    q_goal = np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy()
    q_from = np.array(q_from, dtype=np.float32).reshape(-1)[:4].copy()
    duration = max(0.08, float(duration))
    motion = plan_joint_motion_metrics(q_goal, q_from, duration)
    try:
        phase_ok, phase_reason, phase_sample, _phase_report = path_phase_check(
            q_from, q_goal, phase, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
        )
        obstacle_ok, obstacle_reason, obstacle_sample, _obstacle_report = path_obstacle_check(
            q_from, q_goal, phase, samples=DIG_PLAN_PATH_CHECK_SAMPLES, deadline=deadline
        )
    except Exception as e:
        phase_ok, phase_reason, phase_sample = True, "unchecked:" + type(e).__name__, -1
        obstacle_ok, obstacle_reason, obstacle_sample = True, "unchecked:" + type(e).__name__, -1
    if target_point is None:
        target_point = predicted_end_world_point(q_goal, end_effector=path_end_effector_for_mode(phase), reference_q=q_from)
    if target_point is None:
        target_point = np.zeros(3, dtype=np.float32)
    row = {
        "phase": str(phase),
        "planned": True,
        "required": True,
        "target_point": vec_list(target_point, 3),
        "q_goal_rad": vec_list(q_goal, 4),
        "q_goal_deg": q_deg_values(q_goal, wrap_swing_for_display=True),
        "duration": float(duration),
        "effector": path_end_effector_for_mode(phase),
        "motion": motion,
        "path": {
            "phase_ok": bool(phase_ok),
            "obstacle_ok": bool(obstacle_ok),
            "phase_reason": str(phase_reason),
            "obstacle_reason": str(obstacle_reason),
            "phase_sample": int(phase_sample),
            "obstacle_sample": int(obstacle_sample),
            "path_penalty": 0.0,
            "staged_runtime_plan": True,
        },
        "stage_cost": float(motion.get("cost", 0.0) or 0.0),
    }
    if isinstance(extra, dict):
        row.update(extra)
    return row


def bucket_is_dump_branch_for_carry(bucket_deg):
    try:
        return float(bucket_deg) > float(BUCKET_CARRY_MAX_DUMP_BRANCH_DEG)
    except Exception:
        return False


def bucket_joint_in_loaded_carry_state(bucket_deg):
    try:
        return float(bucket_deg) <= float(CURL_HOLD_ACCEPT_BUCKET_DEG)
    except Exception:
        return False


def set_bucket_loaded_carry_joint(q_pose, reference=None):
    q = np.array(q_pose, dtype=np.float32).reshape(-1)[:4].copy()
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    q[bucket_idx] = deg_to_rad(float(CURL_HOLD_TARGET_DEG))
    return clip_command_near(q, reference=(q_pose if reference is None else reference))


def force_loaded_carry_bucket_q(q_pose, reference=None, label=""):
    """Project a pose to a material-carrying bucket orientation.

    Carry is a bucket-world orientation problem, not a fixed joint-angle
    contract: boom/arm motion changes the bucket frame, so route planning should
    keep the bucket mouth in a retaining orientation relative to gravity and
    only fall back to the old closed-joint clamp when no relative carry solution
    is available.
    """
    q = np.array(q_pose, dtype=np.float32).reshape(-1)[:4].copy()
    q_ref = q.copy() if reference is None else np.array(reference, dtype=np.float32).reshape(-1)[:4].copy()
    try:
        q_adjusted, carry_report = carry_hold_adjusted_q(
            q,
            q_reference=q_ref,
            end_effector="load",
            max_bucket_adjust_deg=105.0,
        )
        if bool((carry_report or {}).get("ok", False)):
            return q_adjusted.copy()
    except Exception:
        pass
    q = clip_command_near(q, reference=q_ref)
    q, _limited, _old_bucket_deg = apply_loaded_bucket_closed_limit(q, label=label)
    return q


def mode_requires_loaded_carry_bucket(mode, label=""):
    text = f"{mode} {label}".lower()
    return (
        "unload_to_bin" in text
        or "clearance_route_post" in text
        or "staged_unload" in text
        or "high_carry" in text
    )


def phase_metric_sand_counts(name):
    phase_metrics = STATE.get("dataset_phase_metrics")
    if not isinstance(phase_metrics, dict):
        return None
    row = phase_metrics.get(str(name), {})
    if not isinstance(row, dict):
        return None
    sand = row.get("sand", {})
    if not isinstance(sand, dict):
        return None
    return sand


def secure_material_baseline_sand():
    # after_cut is the physical state immediately before curl/secure. Using
    # after_dig hides catastrophic loss that happens during the curl itself.
    for name in ("after_cut", "after_dig"):
        sand = phase_metric_sand_counts(name)
        if sand is not None:
            return str(name), sand
    return "", None


def carry_material_report_for_q(q_pose, end_effector="load"):
    q_pose = CTRL.clip_limits(np.array(q_pose, dtype=np.float32).reshape(-1)[:4].copy())
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    angles = chain_angles_from_q(q_pose, end_effector=end_effector)
    load = predicted_end_world_point(q_pose, end_effector="load", reference_q=q_pose)
    pour = predicted_end_world_point(q_pose, end_effector="pour", reference_q=q_pose)
    bucket_deg = rad_to_deg(float(q_pose[bucket_idx]))
    dump_branch = bucket_is_dump_branch_for_carry(bucket_deg)
    if angles is None or load is None or pour is None:
        return {
            "ok": False,
            "reason": "missing_carry_geometry",
            "end_effector": str(end_effector),
            "retains_material": False,
            "bucket_deg": bucket_deg,
            "dump_branch_for_carry": bool(dump_branch),
        }
    load = np.array(load, dtype=np.float32).reshape(-1)[:3]
    pour = np.array(pour, dtype=np.float32).reshape(-1)[:3]
    pour_above = float(pour[2] - load[2])
    height_ok = bool(pour_above >= float(BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z))
    loaded_joint_ok = bucket_joint_in_loaded_carry_state(bucket_deg)
    retains = bool(height_ok and loaded_joint_ok and not dump_branch)
    if dump_branch:
        reason = "bucket_in_dump_branch_for_carry"
    elif not loaded_joint_ok:
        reason = "bucket_not_loaded_carry_joint"
    elif not height_ok:
        reason = "pour_edge_below_carry_window"
    else:
        reason = "ok"
    return {
        "ok": True,
        "reason": reason,
        "end_effector": str(end_effector),
        "retains_material": bool(retains),
        "pour_above_load_z": pour_above,
        "min_pour_above_load_z": float(BUCKET_CARRY_MIN_POUR_ABOVE_LOAD_Z),
        "bucket_deg": bucket_deg,
        "loaded_carry_joint_ok": bool(loaded_joint_ok),
        "loaded_carry_target_deg": float(CURL_HOLD_TARGET_DEG),
        "loaded_carry_accept_deg": float(CURL_HOLD_ACCEPT_BUCKET_DEG),
        "dump_branch_for_carry": bool(dump_branch),
        "max_carry_dump_branch_deg": float(BUCKET_CARRY_MAX_DUMP_BRANCH_DEG),
        "actual_world_deg": rad_to_deg(float(angles[2])),
        "load": vec_list(load, 3),
        "pour": vec_list(pour, 3),
    }


def secure_phase_delta_report(current_metrics=None):
    current_metrics = sand_metrics_current(force=True) if current_metrics is None else current_metrics
    current_metrics = current_metrics if isinstance(current_metrics, dict) else {}
    current_bucket = int(current_metrics.get("bucket_from_pile_count", 0) or 0)
    current_spill = int(current_metrics.get("spill_from_pile_count", 0) or 0)
    baseline_name, baseline_sand = secure_material_baseline_sand()
    baseline_sand = baseline_sand if isinstance(baseline_sand, dict) else {}
    start_bucket = int(baseline_sand.get("bucket_from_pile", current_bucket) or 0)
    start_spill = int(baseline_sand.get("spill_from_pile", current_spill) or 0)
    bucket_loss = max(0, start_bucket - current_bucket)
    spill_delta = max(0, current_spill - start_spill)
    spill_limit = max(
        int(SECURE_HOLD_MAX_SPILL_PARTICLES),
        int(float(max(current_bucket, start_bucket, 1)) * float(SECURE_HOLD_MAX_SPILL_FRACTION)),
    )
    loss_limit = max(
        int(CURL_HOLD_MIN_BUCKET_PARTICLES),
        int(float(max(start_bucket, current_bucket, 1)) * float(SECURE_HOLD_MAX_BUCKET_LOSS_FRACTION)),
    )
    retained_fraction = 1.0 if start_bucket <= 0 else float(current_bucket) / float(max(1, start_bucket))
    min_retained_fraction = float(SECURE_HOLD_MIN_RETAINED_FROM_CUT_FRACTION)
    fraction_ok = bool(start_bucket <= 0 or retained_fraction >= min_retained_fraction)
    ok = spill_delta <= spill_limit and bucket_loss <= loss_limit and fraction_ok
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else (
            f"secure_material_loss spill_delta={spill_delta}/{spill_limit} "
            f"bucket_loss={bucket_loss}/{loss_limit} retained={retained_fraction:.2f}/{min_retained_fraction:.2f}"
        ),
        "baseline": str(baseline_name),
        "bucket_before": int(start_bucket),
        "bucket_after": int(current_bucket),
        "bucket_loss": int(bucket_loss),
        "bucket_loss_limit": int(loss_limit),
        "retained_fraction": float(retained_fraction),
        "min_retained_fraction": float(min_retained_fraction),
        "spill_before": int(start_spill),
        "spill_after": int(current_spill),
        "spill_delta": int(spill_delta),
        "spill_limit": int(spill_limit),
    }


def secure_post_gate_report(q_start, current_metrics=None):
    q_start = CTRL.clip_limits(np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy())
    carry_report = carry_material_report_for_q(q_start, end_effector="load")
    delta_report = secure_phase_delta_report(current_metrics=current_metrics)
    bucket_after = int(delta_report.get("bucket_after", 0) or 0)
    geometry_retains = bool(carry_report.get("ok", False)) and bool(carry_report.get("retains_material", False))
    real_loaded_hold = real_loaded_secure_hold_allowed(carry_report, loaded_count=bucket_after)
    carry_ok = bool(geometry_retains or real_loaded_hold)
    delta_ok = bool(delta_report.get("ok", False))
    ok = carry_ok and delta_ok
    if ok:
        reason = "ok"
    elif not delta_ok:
        reason = str(delta_report.get("reason", "secure_material_loss"))
    elif real_loaded_hold:
        reason = "ok_real_loaded_hold"
    else:
        reason = str(carry_report.get("reason", "current_pose_not_retaining_material"))
    return {
        "ok": bool(ok),
        "reason": reason,
        "q_secure_deg": q_deg_values(q_start, wrap_swing_for_display=True),
        "carry_report": carry_report,
        "material_delta": delta_report,
        "spill_gate_ok": bool(delta_report.get("ok", False)),
        "carry_gate_ok": bool(carry_ok),
        "geometry_retains_material": bool(geometry_retains),
        "real_loaded_hold_allowed": bool(real_loaded_hold),
    }


def post_lift_material_gate_report(current_metrics=None, q_pose=None):
    current_metrics = sand_metrics_current(force=True) if current_metrics is None else current_metrics
    current_metrics = current_metrics if isinstance(current_metrics, dict) else {}
    current_bucket = int(current_metrics.get("bucket_from_pile_count", 0) or 0)
    current_spill = int(current_metrics.get("spill_from_pile_count", 0) or 0)
    try:
        q_check = get_real_joint_positions() if q_pose is None else q_pose
        q_check = CTRL.clip_limits(np.array(q_check, dtype=np.float32).reshape(-1)[:4].copy())
    except Exception:
        q_check = CTRL.q_cmd.copy()
    carry_report = carry_material_report_for_q(q_check, end_effector="load")
    carry_ok = bool(real_loaded_secure_hold_allowed(carry_report, loaded_count=current_bucket))
    baseline_name, baseline_sand = secure_material_baseline_sand()
    baseline_sand = baseline_sand if isinstance(baseline_sand, dict) else {}
    start_bucket = int(baseline_sand.get("bucket_from_pile", current_bucket) or 0)
    start_spill = int(baseline_sand.get("spill_from_pile", current_spill) or 0)
    retained_fraction = 1.0 if start_bucket <= 0 else float(current_bucket) / float(max(1, start_bucket))
    min_bucket = max(
        int(CURL_HOLD_MIN_BUCKET_PARTICLES),
        int(float(max(start_bucket, 1)) * float(LIFT_CARRY_MIN_RETAINED_FROM_CUT_FRACTION)),
    )
    material_ok = bool(current_bucket >= min_bucket)
    ok = bool(material_ok and carry_ok)
    if ok:
        reason = "ok"
    elif not material_ok:
        reason = (
            f"lift_lost_material bucket={current_bucket}/{min_bucket} "
            f"retained={retained_fraction:.2f}/{float(LIFT_CARRY_MIN_RETAINED_FROM_CUT_FRACTION):.2f}"
        )
    else:
        reason = "lift_bucket_not_carry_safe:" + str((carry_report or {}).get("reason", "unknown"))
    return {
        "ok": bool(ok),
        "reason": reason,
        "baseline": str(baseline_name),
        "bucket_before": int(start_bucket),
        "bucket_after": int(current_bucket),
        "bucket_min": int(min_bucket),
        "material_ok": bool(material_ok),
        "carry_ok": bool(carry_ok),
        "q_lift_real_deg": q_deg_values(q_check, wrap_swing_for_display=True),
        "carry_report": carry_report,
        "retained_fraction": float(retained_fraction),
        "min_retained_fraction": float(LIFT_CARRY_MIN_RETAINED_FROM_CUT_FRACTION),
        "spill_before": int(start_spill),
        "spill_after": int(current_spill),
        "spill_delta": int(max(0, current_spill - start_spill)),
    }


def staged_carry_safe_projection_candidates(q_start):
    q_start = CTRL.clip_limits(np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy())
    boom_idx = CTRL.name_to_idx["boom"]
    arm_idx = CTRL.name_to_idx["arm"]
    try:
        metrics_now = sand_metrics_current(force=False)
        loaded_now = int(metrics_now.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics_now, dict) else 0
    except Exception:
        loaded_now = 0
    rows = []
    for boom_lift_deg, arm_retract_deg in [
        (0.0, 0.0),
        (3.0, -1.0),
        (6.0, -2.5),
        (9.0, -4.0),
        (12.0, -5.5),
        (16.0, -7.0),
        (22.0, -9.5),
        (28.0, -12.0),
        (34.0, -14.0),
    ]:
        q = q_start.copy()
        q[boom_idx] = float(q[boom_idx]) + deg_to_rad(float(boom_lift_deg))
        q[arm_idx] = float(q[arm_idx]) + deg_to_rad(float(arm_retract_deg))
        q = clip_command_near(q, reference=q_start)
        q, carry_report = carry_hold_adjusted_q(q, q_reference=q_start, end_effector="load")
        q_forced = force_loaded_carry_bucket_q(q, reference=q_start, label="secure_carry_safe_projection")
        forced_report = carry_material_report_for_q(q_forced, end_effector="load")
        if bool((forced_report or {}).get("retains_material", False)):
            q = q_forced.copy()
            carry_report = dict(forced_report)
            carry_report["projection_source"] = "forced_loaded_carry_joint"
        elif isinstance(carry_report, dict):
            carry_report = dict(carry_report)
            carry_report["forced_loaded_carry_report"] = forced_report
        actual_report = carry_material_report_for_q(q, end_effector="load")
        retains_material = bool((carry_report or {}).get("retains_material", False)) and bool(
            actual_report.get("retains_material", False)
        )
        real_loaded_hold = real_loaded_secure_hold_allowed(actual_report, loaded_count=loaded_now) or real_loaded_secure_hold_allowed(
            carry_report,
            loaded_count=loaded_now,
        )
        if not bool((carry_report or {}).get("ok", False)):
            rows.append({
                "ok": False,
                "reason": f"carry_hold_failed:{(carry_report or {}).get('reason', 'unknown')}",
                "carry_report": carry_report,
                "actual_report": actual_report,
            })
            continue
        if not (retains_material or real_loaded_hold):
            rows.append({
                "ok": False,
                "reason": f"carry_would_spill:{actual_report.get('reason', carry_report.get('reason', 'unknown'))}",
                "carry_report": carry_report,
                "actual_report": actual_report,
            })
            continue
        ok, kind, reason, sample, report = path_segment_check(
            q_start,
            q,
            "secure_carry_safe",
            samples=max(3, min(DIG_PLAN_PATH_CHECK_SAMPLES, 5)),
        )
        if not ok:
            rows.append({
                "ok": False,
                "reason": f"{kind}:{reason}",
                "sample": sample,
                "report": report,
                "carry_report": carry_report,
                "actual_report": actual_report,
            })
            continue
        duration = estimate_stage_motion_seconds(q_start, q, requested_seconds=0.75)
        motion = plan_joint_motion_metrics(q, q_start, duration)
        score = float(motion.get("cost", 0.0) or 0.0)
        if real_loaded_hold and not retains_material:
            score += carry_spill_risk_penalty(actual_report)
        rows.append({
            "ok": True,
            "q": q.copy(),
            "duration": float(duration),
            "score": float(score),
            "target_point": predicted_end_world_point(q, end_effector="load", reference_q=q_start),
            "carry_report": carry_report,
            "actual_report": actual_report,
            "retains_material": bool(retains_material),
            "real_loaded_hold_allowed": bool(real_loaded_hold),
            "motion": motion,
            "boom_lift_deg": float(boom_lift_deg),
            "arm_retract_deg": float(arm_retract_deg),
        })
    return rows


def staged_lift_candidates(q_start):
    rows = []
    boom_idx = CTRL.name_to_idx["boom"]
    arm_idx = CTRL.name_to_idx["arm"]
    bucket_idx = CTRL.name_to_idx["bucket"]
    q_start = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    try:
        metrics_now = sand_metrics_current(force=False)
        loaded_now = int(metrics_now.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics_now, dict) else 0
    except Exception:
        loaded_now = 0
    for boom_lift_deg, arm_retract_deg in [(8.0, -3.0), (14.0, -6.0), (20.0, -9.0), (26.0, -12.0), (32.0, -14.0)]:
        q = q_start.copy()
        q[boom_idx] = float(q[boom_idx]) + deg_to_rad(boom_lift_deg)
        q[arm_idx] = float(q[arm_idx]) + deg_to_rad(arm_retract_deg)
        q[bucket_idx] = float(q_start[bucket_idx])
        q = clip_command_near(q, reference=q_start)
        preserved_report = carry_material_report_for_q(q, end_effector="load")
        preserve_loaded_bucket = bool(loaded_transitional_hold_allowed(preserved_report, loaded_count=loaded_now))
        if preserve_loaded_bucket:
            carry_report = dict(preserved_report)
            carry_report["transitional_load"] = True
            carry_report["lift_preserves_bucket"] = True
        else:
            q, carry_report = carry_hold_adjusted_q(q, q_reference=q_start, end_effector="load")
            q_forced = force_loaded_carry_bucket_q(q, reference=q_start, label="lift_carry_candidate")
            forced_report = carry_material_report_for_q(q_forced, end_effector="load")
            if bool((forced_report or {}).get("retains_material", False)):
                q = q_forced.copy()
                carry_report = dict(forced_report)
                carry_report["projection_source"] = "forced_loaded_carry_joint"
            elif isinstance(carry_report, dict):
                carry_report = dict(carry_report)
                carry_report["forced_loaded_carry_report"] = forced_report
        if not bool((carry_report or {}).get("ok", False)):
            continue
        retains_material = bool((carry_report or {}).get("retains_material", False))
        transitional_material_hold = bool(loaded_transitional_hold_allowed(carry_report, loaded_count=loaded_now))
        if not (retains_material or transitional_material_hold):
            rows.append({
                "ok": False,
                "q": q,
                "reason": (
                    "carry_would_spill_before_lift:"
                    f"pour_above_load_z={fmt_optional((carry_report or {}).get('pour_above_load_z'))}"
                ),
                "carry_report": carry_report,
                "transitional_material_hold": bool(transitional_material_hold),
                "loaded_now": int(loaded_now),
            })
            continue
        ok, kind, reason, sample, report = path_segment_check(
            q_start, q, "lift_carry", samples=DIG_PLAN_PATH_CHECK_SAMPLES
        )
        if not ok:
            rows.append({
                "ok": False,
                "q": q,
                "reason": f"{kind}:{reason}",
                "sample": sample,
                "report": report,
                "carry_report": carry_report,
            })
            continue
        duration = estimate_stage_motion_seconds(q_start, q, requested_seconds=1.0)
        motion = plan_joint_motion_metrics(q, q_start, duration)
        target_point = predicted_end_world_point(q, end_effector="load", reference_q=q_start)
        score = (
            float(motion.get("cost", 0.0) or 0.0)
            + carry_spill_risk_penalty(carry_report)
            + (8.0 if preserve_loaded_bucket else 0.0)
            + (24.0 if transitional_material_hold and not retains_material and not preserve_loaded_bucket else 0.0)
        )
        rows.append({
            "ok": True,
            "q": q.copy(),
            "duration": float(duration),
            "score": float(score),
            "target_point": target_point,
            "carry_report": carry_report,
            "retains_material": bool(retains_material),
            "transitional_material_hold": bool(transitional_material_hold and not retains_material),
            "preserve_loaded_bucket": bool(preserve_loaded_bucket),
            "loaded_now": int(loaded_now),
            "boom_lift_deg": float(boom_lift_deg),
            "arm_retract_deg": float(arm_retract_deg),
        })
    return rows


def staged_high_carry_unload_fallback(q_lift, q_pre_dump, deadline=None):
    boom_idx = CTRL.name_to_idx["boom"]
    arm_idx = CTRL.name_to_idx["arm"]
    bucket_idx = CTRL.name_to_idx["bucket"]
    q_lift = np.array(q_lift, dtype=np.float32).reshape(-1)[:4].copy()
    q_pre_dump = np.array(q_pre_dump, dtype=np.float32).reshape(-1)[:4].copy()
    try:
        metrics_now = sand_metrics_current(force=False)
        loaded_now = int(metrics_now.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics_now, dict) else 0
    except Exception:
        loaded_now = 0
    failures = []
    fallback_deadline = time.time() + 4.0
    for boom_extra_deg, arm_extra_deg in [(10.0, -4.0), (18.0, -7.0), (26.0, -10.0), (34.0, -12.0), (42.0, -14.0)]:
        if planning_deadline_exceeded(fallback_deadline):
            break
        q_high = q_lift.copy()
        q_high[boom_idx] = float(q_high[boom_idx]) + deg_to_rad(boom_extra_deg)
        q_high[arm_idx] = float(q_high[arm_idx]) + deg_to_rad(arm_extra_deg)
        q_high[bucket_idx] = float(q_lift[bucket_idx])
        q_high = clip_command_near(q_high, reference=q_lift)
        q_high, carry_report = carry_hold_adjusted_q(q_high, q_reference=q_lift, end_effector="load")
        q_high_forced = force_loaded_carry_bucket_q(q_high, reference=q_lift, label="staged_high_carry_route")
        forced_report = carry_material_report_for_q(q_high_forced, end_effector="load")
        if bool((forced_report or {}).get("retains_material", False)):
            q_high = q_high_forced.copy()
            carry_report = dict(forced_report)
            carry_report["projection_source"] = "forced_loaded_carry_joint"
        elif isinstance(carry_report, dict):
            carry_report = dict(carry_report)
            carry_report["forced_loaded_carry_report"] = forced_report
        if not bool((carry_report or {}).get("ok", False)):
            failures.append(f"carry_hold_failed:{(carry_report or {}).get('reason', 'unknown')}")
            continue
        retains_material = bool((carry_report or {}).get("retains_material", False))
        transitional_hold = bool(real_loaded_secure_hold_allowed(carry_report, loaded_count=loaded_now))
        if not (retains_material or transitional_hold):
            failures.append(
                "high_carry_would_spill:"
                f"pour_above_load_z={fmt_optional((carry_report or {}).get('pour_above_load_z'))}"
            )
            continue
        q_high, _limited, _old_bucket = apply_loaded_bucket_closed_limit(q_high, label="staged_high_carry_route")

        q_pre = q_pre_dump.copy()
        reference_angles = chain_angles_from_q(q_high, end_effector="load")
        if reference_angles is not None:
            carry_world = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_high, end_effector="load")
            carry_calc = bucket_joint_for_world_angle(q_pre, carry_world, end_effector="load")
            if carry_calc is not None:
                q_pre[bucket_idx] = float(carry_calc["bucket"])
        q_pre = clip_command_near(q_pre, reference=q_high)
        q_pre = force_loaded_carry_bucket_q(q_pre, reference=q_high, label="staged_high_carry_pre_dump")

        ok1, kind1, reason1, sample1, report1 = path_segment_check(
            q_lift,
            q_high,
            "lift_carry",
            samples=max(4, min(DIG_PLAN_PATH_CHECK_SAMPLES, 8)),
            deadline=fallback_deadline,
        )
        if not ok1:
            failures.append(f"high_lift:{kind1}:{reason1}; sample={sample1}")
            continue
        ok2, kind2, reason2, sample2, report2 = path_segment_check(
            q_high,
            q_pre,
            "unload_to_bin",
            samples=max(4, min(DIG_PLAN_PATH_CHECK_SAMPLES, 8)),
            deadline=fallback_deadline,
        )
        if not ok2:
            route2, route2_reason = find_clearance_route(
                q_high,
                q_pre,
                "unload_to_bin",
                "staged_high_carry_unload_fallback",
                deadline=fallback_deadline,
                samples=max(5, min(PATH_ROUTE_PLANNING_SAMPLE_COUNT, 9)),
            )
            if route2 is None:
                failures.append(
                    f"high_to_unload:{kind2}:{reason2}; route={route2_reason}; sample={sample2}"
                )
                continue
            return {
                "route": [q_high.copy()] + [np.array(q, dtype=np.float32).copy() for q in route2],
                "q_pre_dump": q_pre.copy(),
                "reason": (
                    "high_carry_fallback_with_route "
                    f"boom_extra={boom_extra_deg:.1f}deg arm_extra={arm_extra_deg:.1f}deg "
                    f"route={route2_reason}"
                ),
                "carry_report": carry_report,
                "reports": [report1, report2],
            }, "ok"
        return {
            "route": [q_high.copy()],
            "q_pre_dump": q_pre.copy(),
            "reason": (
                "high_carry_fallback "
                f"boom_extra={boom_extra_deg:.1f}deg arm_extra={arm_extra_deg:.1f}deg"
            ),
            "carry_report": carry_report,
            "reports": [report1, report2],
        }, "ok"
    return None, "; ".join(failures[:4]) or "no high carry fallback route"


def staged_dig_secure_candidates(q_start, loaded_count_hint=None):
    q_start = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    boom_idx = CTRL.name_to_idx.get("boom", 1)
    arm_idx = CTRL.name_to_idx.get("arm", 2)
    try:
        metrics_now = sand_metrics_current(force=False)
        loaded_now = int(metrics_now.get("bucket_from_pile_count", 0) or 0) if isinstance(metrics_now, dict) else 0
    except Exception:
        loaded_now = 0
    try:
        loaded_hint = int(loaded_count_hint or 0)
    except Exception:
        loaded_hint = 0
    loaded_count_for_secure = max(int(loaded_now), int(loaded_hint))

    secure_rows = []
    pose_offsets = [
        (0.0, 0.0),
        (2.0, 0.0),
        (3.5, -1.0),
        (5.0, -2.0),
        (7.0, -3.0),
        (9.0, -4.0),
        (12.0, -5.5),
        (16.0, -7.0),
        (21.0, -9.0),
        (27.0, -11.5),
        (34.0, -14.0),
        (42.0, -16.0),
        (52.0, -18.0),
        (62.0, -20.0),
        (6.0, 1.0),
        (10.0, 1.5),
    ]

    def add_secure_candidate(q_seed_base, source, boom_lift_deg=0.0, arm_retract_deg=0.0):
        q_base = np.array(q_seed_base, dtype=np.float32).reshape(-1)[:4].copy()
        q_base[boom_idx] = float(q_base[boom_idx]) + deg_to_rad(float(boom_lift_deg))
        q_base[arm_idx] = float(q_base[arm_idx]) + deg_to_rad(float(arm_retract_deg))
        q_base = clip_command_near(q_base, reference=q_start)

        q_adjusted, adjusted_carry_report = carry_hold_adjusted_q(
            q_base,
            q_reference=q_start,
            end_effector="load",
            max_bucket_adjust_deg=105.0,
        )
        adjusted_actual_report = carry_material_report_for_q(q_adjusted, end_effector="load")

        q_forced = set_bucket_loaded_carry_joint(q_base, reference=q_start)
        q_forced, forced_limited, forced_old_bucket_deg = apply_loaded_bucket_closed_limit(
            q_forced,
            label="staged_secure_force_closed",
        )
        forced_report = carry_material_report_for_q(q_forced, end_effector="load")
        forced_loaded_hold = bool(real_loaded_secure_hold_allowed(forced_report, loaded_count=loaded_count_for_secure))
        adjusted_loaded_hold = bool(
            real_loaded_secure_hold_allowed(adjusted_actual_report, loaded_count=loaded_count_for_secure)
            or real_loaded_secure_hold_allowed(adjusted_carry_report, loaded_count=loaded_count_for_secure)
        )
        forced_joint_safe = bool((forced_report or {}).get("ok", False)) and bool(
            (forced_report or {}).get("loaded_carry_joint_ok", False)
        ) and not bool((forced_report or {}).get("dump_branch_for_carry", False))
        prefer_forced_closed = bool(
            loaded_count_for_secure >= int(CURL_HOLD_MIN_BUCKET_PARTICLES)
            and forced_joint_safe
        )

        if prefer_forced_closed:
            q_secure = q_forced.copy()
            carry_report = dict(forced_report)
            carry_report.update({
                "secure_source": "forced_loaded_closed",
                "forced_bucket_closed": True,
                "forced_bucket_limited": bool(forced_limited),
                "forced_old_bucket_deg": forced_old_bucket_deg,
                "adjusted_carry_report": adjusted_carry_report,
                "adjusted_actual_report": adjusted_actual_report,
            })
            actual_report = dict(forced_report)
            secure_source = "forced_loaded_closed"
        else:
            q_secure = np.array(q_adjusted, dtype=np.float32).reshape(-1)[:4].copy()
            carry_report = dict(adjusted_carry_report or {})
            carry_report.update({
                "secure_source": "carry_hold_adjusted",
                "forced_bucket_closed": False,
                "forced_report": forced_report,
                "forced_bucket_limited": bool(forced_limited),
                "forced_old_bucket_deg": forced_old_bucket_deg,
            })
            actual_report = dict(adjusted_actual_report or {})
            secure_source = "carry_hold_adjusted"

        if not bool((carry_report or {}).get("ok", False)) and not prefer_forced_closed:
            secure_rows.append({
                "ok": False,
                "source": str(source),
                "reason": f"carry_hold_failed:{(carry_report or {}).get('reason', 'unknown')}",
                "carry_report": carry_report,
                "actual_report": actual_report,
                "forced_report": forced_report,
            })
            return
        retains_material = bool((carry_report or {}).get("retains_material", False)) and bool(
            actual_report.get("retains_material", False)
        )
        transitional_material_hold = bool(
            forced_loaded_hold
            or adjusted_loaded_hold
            or real_loaded_secure_hold_allowed(actual_report, loaded_count=loaded_count_for_secure)
            or real_loaded_secure_hold_allowed(carry_report, loaded_count=loaded_count_for_secure)
        )
        if not (retains_material or transitional_material_hold):
            secure_rows.append({
                "ok": False,
                "source": str(source),
                "reason": (
                    "secure_carry_would_spill:"
                    f"secure_source={secure_source} "
                    f"q_secure_bucket={rad_to_deg(float(q_secure[bucket_idx])):.2f}deg "
                    f"forced_bucket={rad_to_deg(float(q_forced[bucket_idx])):.2f}deg "
                    f"pour_above_load_z={fmt_optional(actual_report.get('pour_above_load_z', (carry_report or {}).get('pour_above_load_z')))}"
                ),
                "carry_report": carry_report,
                "actual_report": actual_report,
                "forced_report": forced_report,
                "transitional_material_hold": bool(transitional_material_hold),
                "forced_loaded_hold": bool(forced_loaded_hold),
                "adjusted_loaded_hold": bool(adjusted_loaded_hold),
            })
            return

        # Close the bucket first, but do not hold boom/arm fixed if that would
        # drive the bucket through hard ground or into a rigid obstacle. Try a
        # small lattice of synchronized boom/arm corrections before rejecting
        # the secure pose.
        curl_attempts = []
        selected_curl = None
        for boom_fraction, arm_fraction in [
            (0.25, 0.10),
            (0.35, 0.20),
            (0.45, 0.25),
            (0.55, 0.35),
            (0.65, 0.45),
            (0.75, 0.55),
            (0.85, 0.65),
            (0.95, 0.75),
            (1.00, 0.85),
            (1.00, 1.00),
        ]:
            q_curl = q_start.copy()
            q_curl[bucket_idx] = float(q_secure[bucket_idx])
            q_curl[boom_idx] = float(q_start[boom_idx]) + float(boom_fraction) * float(q_secure[boom_idx] - q_start[boom_idx])
            q_curl[arm_idx] = float(q_start[arm_idx]) + float(arm_fraction) * float(q_secure[arm_idx] - q_start[arm_idx])
            q_curl = clip_command_near(q_curl, reference=q_start)
            curl_bucket_deg = float(rad_to_deg(q_curl[bucket_idx]))

            curl_report = predicted_phase_ground_report(q_curl, "curl_to_hold_material", reference_q=q_start)
            curl_ok, curl_reason = phase_ground_ok("curl_to_hold_material", curl_report)
            if not curl_ok:
                curl_attempts.append({
                    "ok": False,
                    "reason": f"curl_ground:{curl_reason}",
                    "boom_fraction": float(boom_fraction),
                    "arm_fraction": float(arm_fraction),
                    "curl_report": curl_report,
                })
                continue
            curl_path_ok, curl_kind, curl_path_reason, curl_sample, curl_path_report = path_segment_check(
                q_start,
                q_curl,
                "curl_to_hold_material",
                samples=2,
            )
            if not curl_path_ok:
                curl_attempts.append({
                    "ok": False,
                    "reason": f"curl_{curl_kind}:{curl_path_reason}",
                    "sample": curl_sample,
                    "report": curl_path_report,
                    "boom_fraction": float(boom_fraction),
                    "arm_fraction": float(arm_fraction),
                    "curl_report": curl_report,
                })
                continue
            selected_curl = {
                "q": q_curl.copy(),
                "bucket_deg": float(curl_bucket_deg),
                "report": curl_report,
                "boom_fraction": float(boom_fraction),
                "arm_fraction": float(arm_fraction),
                "attempts": curl_attempts,
            }
            break
        if selected_curl is None:
            best_reason = "; ".join(str(row.get("reason", "")) for row in curl_attempts[:3]) or "curl no executable boom/arm adjustment"
            secure_rows.append({
                "ok": False,
                "source": str(source),
                "reason": best_reason,
                "carry_report": carry_report,
                "actual_report": actual_report,
                "forced_report": forced_report,
                "curl_attempts": curl_attempts[:5],
                "secure_source": str(secure_source),
                "q_secure_bucket_deg": float(rad_to_deg(q_secure[bucket_idx])),
                "forced_bucket_deg": float(rad_to_deg(q_forced[bucket_idx])),
            })
            return
        q_curl = selected_curl["q"].copy()
        curl_bucket_deg = float(selected_curl["bucket_deg"])
        curl_report = selected_curl["report"]

        secure_report = predicted_phase_ground_report(q_secure, "secure_load", reference_q=q_curl)
        secure_ok, secure_reason = phase_ground_ok("secure_load", secure_report)
        if not secure_ok:
            secure_rows.append({
                "ok": False,
                "source": str(source),
                "reason": (
                    f"secure_ground:{secure_reason}; "
                    f"secure_source={secure_source} "
                    f"q_secure_bucket={rad_to_deg(float(q_secure[bucket_idx])):.2f}deg "
                    f"forced_bucket={rad_to_deg(float(q_forced[bucket_idx])):.2f}deg"
                ),
                "carry_report": carry_report,
                "actual_report": actual_report,
                "forced_report": forced_report,
                "ground_report": secure_report,
                "secure_source": str(secure_source),
            })
            return
        ok, kind, reason, sample, report = path_segment_check(
            q_curl,
            q_secure,
            "secure_load",
            samples=2,
        )
        if not ok:
            secure_rows.append({
                "ok": False,
                "source": str(source),
                "reason": f"{kind}:{reason}",
                "sample": sample,
                "report": report,
                "carry_report": carry_report,
                "actual_report": actual_report,
                "forced_report": forced_report,
                "secure_source": str(secure_source),
                "q_secure_bucket_deg": float(rad_to_deg(q_secure[bucket_idx])),
            })
            return
        curl_duration = estimate_stage_motion_seconds(q_start, q_curl, requested_seconds=0.90)
        secure_duration = estimate_stage_motion_seconds(q_curl, q_secure, requested_seconds=0.85)
        curl_motion = plan_joint_motion_metrics(q_curl, q_start, curl_duration)
        secure_motion = plan_joint_motion_metrics(q_secure, q_curl, secure_duration)
        target_point = predicted_end_world_point(q_secure, end_effector="load", reference_q=q_curl)
        score = (
            float(curl_motion.get("cost", 0.0) or 0.0)
            + float(secure_motion.get("cost", 0.0) or 0.0)
            + carry_spill_risk_penalty(carry_report)
            - 16.0 * max(0.0, float(actual_report.get("pour_above_load_z", (carry_report or {}).get("pour_above_load_z", 0.0)) or 0.0))
        )
        secure_rows.append({
            "ok": True,
            "source": str(source),
            "retains_material": bool(retains_material),
            "transitional_material_hold": bool(transitional_material_hold and not retains_material),
            "loaded_count_for_secure": int(loaded_count_for_secure),
            "score": float(score),
            "q_curl": q_curl.copy(),
            "q": q_secure.copy(),
            "curl_duration": float(curl_duration),
            "duration": float(secure_duration),
            "target_point": target_point,
            "curl_motion": curl_motion,
            "motion": secure_motion,
            "carry_report": carry_report,
            "actual_report": actual_report,
            "forced_report": forced_report,
            "secure_source": str(secure_source),
            "forced_bucket_closed": bool(prefer_forced_closed),
            "q_secure_bucket_deg": float(rad_to_deg(q_secure[bucket_idx])),
            "forced_bucket_deg": float(rad_to_deg(q_forced[bucket_idx])),
            "curl_report": curl_report,
            "ground_report": secure_report,
            "curl_bucket_deg": float(curl_bucket_deg),
            "curl_joint_closed_ok_deprecated": bool(curl_bucket_deg <= float(CURL_HOLD_ACCEPT_BUCKET_DEG) + 0.25),
            "curl_boom_fraction": float(selected_curl.get("boom_fraction", 0.0)),
            "curl_arm_fraction": float(selected_curl.get("arm_fraction", 0.0)),
            "boom_lift_deg": float(boom_lift_deg),
            "arm_retract_deg": float(arm_retract_deg),
        })

    for boom_lift_deg, arm_retract_deg in pose_offsets:
        add_secure_candidate(
            q_start,
            "carry_projection_from_pull_exit",
            boom_lift_deg=boom_lift_deg,
            arm_retract_deg=arm_retract_deg,
        )

    valid_secure = [row for row in secure_rows if bool(row.get("ok", False))]
    if not valid_secure:
        try:
            cut_metrics = sand_metrics_current(force=True)
            cut_bucket = int(cut_metrics.get("bucket_from_pile_count", 0) or 0) if isinstance(cut_metrics, dict) else 0
        except Exception:
            cut_bucket = 0
        try:
            hint_bucket = int(loaded_count_hint or 0)
        except Exception:
            hint_bucket = 0
        cut_bucket = max(cut_bucket, hint_bucket)
        transitional_report = carry_material_report_for_q(q_start, end_effector="load")
        best_diag = sorted(
            secure_rows,
            key=lambda row: float((row.get("actual_report") or row.get("carry_report") or {}).get("pour_above_load_z", -999.0) or -999.0),
            reverse=True,
        )[0] if secure_rows else {}
        return [], (
            "secure_load no retaining pose: "
            f"{best_diag.get('reason', 'no candidates')}; "
            f"loaded_count={int(cut_bucket)} "
            f"transitional={bool(loaded_transitional_hold_allowed(transitional_report, loaded_count=cut_bucket))} "
            f"current_bucket={rad_to_deg(float(q_start[bucket_idx])):.2f}deg "
            f"target_bucket={float(CURL_HOLD_TARGET_DEG):.2f}deg"
        )

    best_secure = sorted(valid_secure, key=lambda row: float(row.get("score", 1.0e9)))[0]
    q_curl = np.array(best_secure["q_curl"], dtype=np.float32).copy()
    q_secure = np.array(best_secure["q"], dtype=np.float32).copy()
    curl_duration = float(best_secure.get("curl_duration", 0.90))
    curl_target = predicted_end_world_point(q_curl, end_effector="tip", reference_q=q_start)
    curl_report = best_secure.get("curl_report") or {}
    curl_row = make_stage_row_from_q(
        "curl_to_hold_material",
        q_curl,
        q_start,
        curl_duration,
        target_point=curl_target,
        extra={
            "staged_runtime_plan": True,
            "staged_append_source": "post_pull_exit",
            "seal_bucket_first": True,
            "material_hold": {
                "ok": True,
                "reason": "curl_bucket_toward_retaining_secure_pose"
                if bool(best_secure.get("retains_material", False))
                else "curl_bucket_toward_real_loaded_secure_pose",
                "bucket_deg": float(rad_to_deg(q_curl[bucket_idx])),
                "joint_closed_ok_deprecated": bool(best_secure.get("curl_joint_closed_ok_deprecated", False)),
                "target_secure_bucket_deg": float(rad_to_deg(q_secure[bucket_idx])),
                "retaining_secure_report": best_secure.get("actual_report", best_secure.get("carry_report", {})),
                "real_loaded_hold_allowed": bool(best_secure.get("transitional_material_hold", False)),
                "loaded_count_for_secure": int(best_secure.get("loaded_count_for_secure", 0) or 0),
                "curl_boom_fraction": float(best_secure.get("curl_boom_fraction", 0.0) or 0.0),
                "curl_arm_fraction": float(best_secure.get("curl_arm_fraction", 0.0) or 0.0),
            },
            "ground": {
                "ok": True,
                "reason": "ok",
                "tip_depth": curl_report.get("tip_sand_depth"),
                "bucket_mid_depth": curl_report.get("bucket_mid_sand_depth"),
                "pour_depth": curl_report.get("pour_sand_depth"),
                "load_depth": curl_report.get("load_sand_depth"),
                "surface_source": curl_report.get("tip_sand_surface_source"),
            },
        },
    )
    secure_row = make_stage_row_from_q(
        "secure_load",
        q_secure,
        q_curl,
        float(best_secure.get("duration", 0.85)),
        target_point=best_secure.get("target_point"),
        extra={
            "effector": "load",
            "staged_runtime_plan": True,
            "staged_append_source": "post_pull_exit",
            "secure_load": {
                "boom_lift_deg": float(best_secure.get("boom_lift_deg", 0.0)),
                "arm_retract_deg": float(best_secure.get("arm_retract_deg", 0.0)),
                "candidate_count": len(secure_rows),
                "source": str(best_secure.get("source", "")),
                "retains_material": bool(best_secure.get("retains_material", False)),
                "transitional_material_hold": bool(best_secure.get("transitional_material_hold", False)),
            },
            "material_hold": best_secure.get("actual_report", best_secure.get("carry_report", {})),
            "carry_projection": best_secure.get("carry_report", {}),
            "ground": {
                "ok": True,
                "reason": "ok",
                "tip_depth": (best_secure.get("ground_report") or {}).get("tip_sand_depth"),
                "bucket_mid_depth": (best_secure.get("ground_report") or {}).get("bucket_mid_sand_depth"),
                "pour_depth": (best_secure.get("ground_report") or {}).get("pour_sand_depth"),
                "load_depth": (best_secure.get("ground_report") or {}).get("load_sand_depth"),
                "surface_source": (best_secure.get("ground_report") or {}).get("tip_sand_surface_source"),
            },
        },
    )
    return [
        ("curl_to_hold_material", q_curl.copy(), float(curl_duration), curl_target, curl_row),
        ("secure_load", q_secure.copy(), float(best_secure.get("duration", 0.85)), best_secure.get("target_point"), secure_row),
    ], "ok"


def append_staged_post_dig_secure_plan(task_label="dig_target_ball"):
    plan = STATE.get("current_dig_plan")
    if not (isinstance(plan, dict) and bool(plan.get("staged_execution", False))):
        return True
    if bool(plan.get("staged_post_dig_secure_appended", False)):
        return True
    if not bool(plan.get("staged_post_dig_secure_pending", False)):
        return True

    seq = STATE.get("dig_plan_sequence")
    points = STATE.get("dig_plan_points")
    candidate = STATE.get("dig_plan_candidate")
    if not isinstance(seq, list) or not isinstance(points, list) or not isinstance(candidate, dict):
        set_execution_failure_reason("execution_failed/staged_plan_state_missing")
        return False

    q_start = sync_motion_start_q("staged_post_dig_secure")
    cut_metrics = record_phase_metrics("after_cut", q_cmd=q_start, q_real=q_start)
    cut_sand = cut_metrics.get("sand", {}) if isinstance(cut_metrics, dict) else {}
    cut_bucket = int(cut_sand.get("bucket_from_pile", 0) or 0) if isinstance(cut_sand, dict) else 0
    rows, reason = staged_dig_secure_candidates(q_start, loaded_count_hint=cut_bucket)
    if not rows:
        if cut_bucket >= int(CURL_HOLD_MIN_BUCKET_PARTICLES):
            set_execution_failure_reason("quality_rejected/secure_not_retaining_material:" + str(reason))
        else:
            set_execution_failure_reason("planning_failed/staged_secure_unreachable:" + str(reason))
        info_print("[DIG PLAN STAGED FAILED]", "stage=post_pull_exit_secure", reason)
        return False

    stages = list(candidate.get("stages", []) or [])
    for phase, q_goal, duration, point, row in rows:
        seq.append((phase, np.array(q_goal, dtype=np.float32).copy(), float(duration)))
        if point is None:
            point = predicted_end_world_point(q_goal, end_effector=path_end_effector_for_mode(phase), reference_q=q_start)
        if point is None:
            point = np.zeros(3, dtype=np.float32)
        points.append(np.array(point, dtype=np.float32).reshape(-1)[:3].copy())
        stages.append(row)

    candidate["stages"] = stages
    candidate["steps"] = len(seq)
    candidate["planned"] = True
    candidate["staged_post_dig_secure_pending"] = False
    candidate["staged_post_dig_secure_appended"] = True
    candidate["staged_post_secure_load_pending"] = True
    candidate["route_diagnostics"] = route_diagnostics_from_stages(stages)
    candidate["unload_ballistics"] = unload_ballistics_from_stages(stages)
    STATE["dig_plan_candidate"] = candidate
    STATE["dig_plan_points"] = points
    cache_dig_plan_trace_points(seq, start_q=STATE.get("dig_plan_start_q", CTRL.q_cmd.copy()))
    new_plan = build_shared_dig_plan_object(STATE.get("dig_plan_target", get_target_pos()), seq, points, candidate)
    new_plan["staged_execution"] = True
    new_plan["staged_prefix_ready"] = True
    new_plan["staged_post_dig_secure_pending"] = False
    new_plan["staged_post_dig_secure_appended"] = True
    new_plan["staged_post_secure_load_pending"] = True
    new_plan["staged_prefix_terminal_phase"] = "secure_load"
    if current_trace_mode() == 2:
        draw_trace(force=True)
    info_print(
        "[DIG PLAN STAGED APPEND]",
        "added=" + ",".join(str(row[0]) for row in rows),
        f"steps={len(seq)}",
        f"source={task_label}",
        f"reason={reason}",
        f"after_cut_bucket={cut_bucket}",
    )
    return True


def append_staged_post_secure_load_plan(task_label="dig_target_ball"):
    plan = STATE.get("current_dig_plan")
    if not (isinstance(plan, dict) and bool(plan.get("staged_execution", False))):
        return True
    if bool(plan.get("staged_post_secure_load_appended", False)):
        return True

    seq = STATE.get("dig_plan_sequence")
    points = STATE.get("dig_plan_points")
    candidate = STATE.get("dig_plan_candidate")
    if not isinstance(seq, list) or not isinstance(points, list) or not isinstance(candidate, dict):
        set_execution_failure_reason("execution_failed/staged_plan_state_missing")
        return False

    q_start = sync_motion_start_q("staged_post_secure_load")
    loaded_route_test = (
        str(task_label) == "loaded_unload_route_test"
        or str(candidate.get("debug_task", "")) == "loaded_unload_route_test"
        or str(candidate.get("id", "")) == "loaded_unload_route_test"
    )
    post_lift_reentry = bool(candidate.get("staged_lift_before_bucket_safe_appended", False)) and not bool(
        candidate.get("staged_post_secure_load_appended", False)
    )
    if loaded_route_test:
        secure_gate = {
            "ok": True,
            "reason": "loaded_route_test_current_pose_assumed_carry_safe",
            "q_secure_deg": q_deg_values(q_start, wrap_swing_for_display=True),
            "spill_gate_ok": True,
            "carry_gate_ok": True,
            "loaded_route_test": True,
            "material_gate_bypassed": True,
        }
    elif post_lift_reentry:
        lift_metrics = record_phase_metrics("after_lift")
        lift_gate = post_lift_material_gate_report(current_metrics=lift_metrics, q_pose=q_start)
        candidate["post_lift_gate"] = lift_gate
        if not bool(lift_gate.get("ok", False)):
            reason = str(lift_gate.get("reason", "lift_lost_material"))
            candidate["post_secure_projection"] = {
                "ok": False,
                "reason": reason,
                "stage": "post_lift_reentry",
            }
            STATE["dig_plan_candidate"] = candidate
            set_execution_failure_reason("quality_rejected/lift_lost_material:" + reason)
            info_print("[POST LIFT GATE FAILED]", reason)
            return False
        secure_gate = {
            "ok": True,
            "reason": "post_lift_ready_for_unload",
            "q_secure_deg": q_deg_values(q_start, wrap_swing_for_display=True),
            "spill_gate_ok": True,
            "carry_gate_ok": True,
            "post_lift_reentry": True,
            "lift_material_gate": lift_gate,
        }
    else:
        secure_metrics = record_phase_metrics("after_secure_load")
        secure_gate = secure_post_gate_report(q_start, current_metrics=secure_metrics)
    candidate["post_secure_gate"] = secure_gate
    STATE["dig_plan_candidate"] = candidate
    debug_timeline_record(
        "SECURE_GATE",
        stage="post_lift" if post_lift_reentry else "secure_load",
        result="ok" if bool(secure_gate.get("ok", False)) else "project_required",
        reason=str(secure_gate.get("reason", "")),
        q_cmd=q_start,
        q_real=q_start,
        data=secure_gate,
        include_sand=not loaded_route_test,
    )
    carry_safe_seq = []
    carry_safe_points = []
    carry_safe_stages = []
    q_lift_start = q_start.copy()
    if not bool(secure_gate.get("ok", False)):
        if not bool(secure_gate.get("spill_gate_ok", False)):
            reason = str(secure_gate.get("reason", "secure_material_loss"))
            candidate["post_secure_projection"] = {
                "ok": False,
                "reason": reason,
                "skipped": True,
                "skip_reason": "secure_material_loss_not_recoverable_by_projection",
            }
            STATE["dig_plan_candidate"] = candidate
            set_execution_failure_reason("quality_rejected/secure_not_retaining_material:" + reason)
            info_print("[SECURE GATE FAILED]", reason)
            debug_timeline_record(
                "CARRY_SAFE_PROJECT",
                stage="secure_carry_safe",
                result="skipped",
                reason=reason,
                q_cmd=q_start,
                q_real=q_start,
                data={"secure_gate": secure_gate},
                include_sand=True,
            )
            return False
    if not bool(secure_gate.get("ok", False)):
        if bool(candidate.get("staged_secure_carry_safe_appended", False)):
            reason = "secure_carry_safe_still_not_retaining:" + str(
                secure_gate.get("reason", "current pose does not retain material")
            )
            candidate["post_secure_projection"] = {
                "ok": False,
                "reason": reason,
                "secure_gate": secure_gate,
                "already_appended": True,
            }
            STATE["dig_plan_candidate"] = candidate
            set_execution_failure_reason("quality_rejected/secure_not_retaining_material:" + reason)
            info_print("[SECURE GATE FAILED]", reason)
            debug_timeline_record(
                "CARRY_SAFE_PROJECT",
                stage="secure_carry_safe",
                result="failed_after_execution",
                reason=reason,
                q_cmd=q_start,
                q_real=q_start,
                data={"secure_gate": secure_gate},
                include_sand=True,
            )
            return False
        projection_rows = staged_carry_safe_projection_candidates(q_start)
        valid_projection = [row for row in projection_rows if bool(row.get("ok", False))]
        if not valid_projection:
            reason = "; ".join(str(row.get("reason", "")) for row in projection_rows[:4]) or str(
                secure_gate.get("reason", "current pose does not retain material")
            )
            candidate["post_secure_projection"] = {
                "ok": False,
                "reason": reason,
                "candidate_count": int(len(projection_rows)),
            }
            STATE["dig_plan_candidate"] = candidate
            set_execution_failure_reason("quality_rejected/secure_not_retaining_material:" + reason)
            info_print("[SECURE GATE FAILED]", reason)
            debug_timeline_record(
                "CARRY_SAFE_PROJECT",
                stage="secure_carry_safe",
                result="failed",
                reason=reason,
                q_cmd=q_start,
                q_real=q_start,
                data={
                    "secure_gate": secure_gate,
                    "projection_candidates": [
                        {
                            "ok": bool(row.get("ok", False)),
                            "reason": str(row.get("reason", "")),
                            "carry_report": row.get("carry_report", {}),
                            "actual_report": row.get("actual_report", {}),
                        }
                        for row in projection_rows[:6]
                    ],
                },
                include_sand=True,
            )
            return False

        projection = sorted(valid_projection, key=lambda row: float(row.get("score", 1.0e9)))[0]
        q_carry_safe = np.array(projection["q"], dtype=np.float32).copy()
        q_lift_start = q_carry_safe.copy()
        projection_duration = float(projection.get("duration", 0.75) or 0.75)
        projection_target = projection.get("target_point")
        carry_safe_row = make_stage_row_from_q(
            "secure_carry_safe",
            q_carry_safe,
            q_start,
            projection_duration,
            target_point=projection_target,
            extra={
                "effector": "load",
                "staged_runtime_plan": True,
                "secure_gate": secure_gate,
                "carry_safe_projection": {
                    "ok": True,
                    "reason": "projected_to_retaining_pose",
                    "boom_lift_deg": float(projection.get("boom_lift_deg", 0.0)),
                    "arm_retract_deg": float(projection.get("arm_retract_deg", 0.0)),
                    "candidate_count": int(len(projection_rows)),
                },
                "material_hold": projection.get("carry_report", {}),
            },
        )
        carry_safe_seq.append(("secure_carry_safe", q_carry_safe.copy(), projection_duration))
        if projection_target is None:
            projection_target = predicted_end_world_point(q_carry_safe, end_effector="load", reference_q=q_start)
        carry_safe_points.append(np.array(projection_target if projection_target is not None else unload_bin_landing_point(), dtype=np.float32).reshape(-1)[:3].copy())
        carry_safe_stages.append(carry_safe_row)
        candidate["post_secure_projection"] = {
            "ok": True,
            "q_carry_safe_deg": q_deg_values(q_carry_safe, wrap_swing_for_display=True),
            "carry_report": projection.get("carry_report", {}),
            "actual_report": projection.get("actual_report", {}),
        }
        STATE["dig_plan_candidate"] = candidate
        debug_timeline_record(
            "CARRY_SAFE_PROJECT",
            stage="secure_carry_safe",
            result="ok",
            reason="projected_to_retaining_pose",
            q_cmd=q_carry_safe,
            q_real=q_start,
            data=candidate["post_secure_projection"],
            include_sand=True,
        )
        info_print(
            "[CARRY SAFE PROJECT]",
            f"q_start={q_deg_values(q_start, wrap_swing_for_display=True)}",
            f"q_safe={q_deg_values(q_carry_safe, wrap_swing_for_display=True)}",
            f"pour_above_load_z={fmt_optional((projection.get('carry_report') or {}).get('pour_above_load_z'))}",
        )

        # Enforce the physical order: first execute the bucket carry-safe
        # projection, then re-enter this function from the executed
        # secure_carry_safe stage to plan lift/carry/unload from the real
        # post-projection joint state.
        seq.extend(carry_safe_seq)
        points.extend(carry_safe_points)
        stages = list(candidate.get("stages", []) or [])
        stages.extend(carry_safe_stages)
        candidate["stages"] = stages
        candidate["steps"] = len(seq)
        candidate["planned"] = True
        candidate["staged_secure_carry_safe_appended"] = True
        candidate["staged_post_secure_load_pending"] = True
        candidate["staged_post_secure_load_appended"] = False
        candidate["staged_prefix_terminal_phase"] = "secure_carry_safe"
        candidate["route_diagnostics"] = route_diagnostics_from_stages(stages)
        candidate["unload_ballistics"] = unload_ballistics_from_stages(stages)
        STATE["dig_plan_candidate"] = candidate
        STATE["dig_plan_points"] = points
        cache_dig_plan_trace_points(seq, start_q=STATE.get("dig_plan_start_q", CTRL.q_cmd.copy()))
        new_plan = build_shared_dig_plan_object(STATE.get("dig_plan_target", get_target_pos()), seq, points, candidate)
        new_plan["staged_execution"] = True
        new_plan["staged_prefix_ready"] = True
        new_plan["staged_post_secure_load_pending"] = True
        new_plan["staged_post_secure_load_appended"] = False
        new_plan["staged_prefix_terminal_phase"] = "secure_carry_safe"
        if current_trace_mode() == 2:
            draw_trace(force=True)
        info_print(
            "[DIG PLAN STAGED APPEND]",
            "added=secure_carry_safe",
            f"steps={len(seq)}",
            f"source={task_label}",
            "next=plan_lift_after_real_secure_carry_safe",
        )
        return True

    lift = None
    lift_row = None
    lift_duration = 0.0
    q_lift = q_lift_start.copy()
    skip_lift_stage = bool(post_lift_reentry or loaded_route_test)
    if not skip_lift_stage:
        lift_rows = staged_lift_candidates(q_lift_start)
        valid_lift = [row for row in lift_rows if bool(row.get("ok", False))]
        if not valid_lift:
            reason = "; ".join(str(row.get("reason", "")) for row in lift_rows[:3]) or "no lift candidate"
            set_execution_failure_reason("planning_failed/staged_lift_unreachable:" + reason)
            info_print("[DIG PLAN STAGED FAILED]", "stage=lift_carry", reason)
            return False

        lift = sorted(valid_lift, key=lambda row: float(row.get("score", 1.0e9)))[0]
        q_lift = np.array(lift["q"], dtype=np.float32).copy()
        lift_duration = float(lift.get("duration", 1.1) or 1.1)
        lift_row = make_stage_row_from_q(
            "lift_carry",
            q_lift,
            q_lift_start,
            lift_duration,
            target_point=lift.get("target_point"),
            extra={
                "material_hold": lift.get("carry_report", {}),
                "staged_runtime_plan": True,
                "secure_load_runtime_append": {
                    "source_task": str(task_label),
                    "boom_lift_deg": float(lift.get("boom_lift_deg", 0.0)),
                    "arm_retract_deg": float(lift.get("arm_retract_deg", 0.0)),
                    "preserve_loaded_bucket": bool(lift.get("preserve_loaded_bucket", False)),
                    "deferred_bucket_safe_until_after_lift": False,
                },
            },
        )

    deadline = (
        time.time() + float(LOADED_ROUTE_TEST_PLAN_BUDGET_SECONDS)
        if loaded_route_test
        else time.time() + 10.0
    )
    if loaded_route_test:
        info_print(
            "[LOADED ROUTE TEST OBSERVE]",
            f"budget={float(LOADED_ROUTE_TEST_PLAN_BUDGET_SECONDS):.1f}s",
            "mode=finite_search_with_debug",
            "reason=avoid_main_thread_candidate_explosion",
            f"cleared_stale_cancel={bool(STATE.get('loaded_route_test_cleared_stale_cancel', False))}",
            force_log=True,
        )
    info_print(
        "[POST SECURE PLAN START]",
        f"loaded_route_test={loaded_route_test}",
        f"q_start={q_deg_values(q_lift, wrap_swing_for_display=True)}",
        f"landing={vec_list(unload_bin_landing_point(), 3)}",
        f"dump_deg={fmt_optional(unload_dump_target_deg())}",
        force_log=loaded_route_test,
    )
    dump_deadline = child_planning_deadline(deadline, 6.0, min_seconds=2.0) if loaded_route_test else deadline
    if loaded_route_test:
        info_print(
            "[POST SECURE PLAN DUMP BUDGET]",
            f"dump_budget={fmt_optional(max(0.0, float(dump_deadline) - time.time()))}s",
            f"route_reserved={fmt_optional(max(0.0, float(deadline) - float(dump_deadline)))}s",
            force_log=True,
        )
    q_release_align, dump_info = plan_dump_pose_to_bin(
        q_seed=q_lift,
        dump_deg=unload_dump_target_deg(),
        label="staged_unload_to_bin",
        log=True,
        allow_unaligned=True,
        max_correction_iters=3 if loaded_route_test else None,
        deadline=dump_deadline,
        goal_obstacle_check=not loaded_route_test,
        bucket_candidate_count_override=5 if loaded_route_test else None,
    )
    if q_release_align is None:
        set_execution_failure_reason("planning_failed/staged_unload_dump_pose:" + str(dump_info))
        info_print("[DIG PLAN STAGED FAILED]", "stage=unload_to_bin", dump_info)
        return False

    bucket_idx = CTRL.name_to_idx["bucket"]

    def compute_pre_dump_carry_pose(q_dump_pose, q_reference, compute_label):
        q_pre = np.array(q_dump_pose, dtype=np.float32).copy()
        reference_angles = chain_angles_from_q(q_reference, end_effector="load")
        carry_calc = None
        if reference_angles is not None:
            carry_world = nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_reference, end_effector="load")
            carry_calc = bucket_joint_for_world_angle(q_pre, carry_world, end_effector="load")
        if carry_calc is not None:
            q_pre[bucket_idx] = carry_calc["bucket"]
        else:
            q_pre[bucket_idx] = float(q_reference[bucket_idx])
        q_pre = clip_command_near(q_pre, reference=q_reference)
        old_bucket_deg = rad_to_deg(float(q_pre[bucket_idx]))
        q_pre = force_loaded_carry_bucket_q(q_pre, reference=q_reference, label=compute_label)
        new_bucket_deg = rad_to_deg(float(q_pre[bucket_idx]))
        carry_report = carry_material_report_for_q(q_pre, end_effector="load")
        if abs(new_bucket_deg - old_bucket_deg) > 0.25:
            info_print(
                "[LOADED BUCKET CARRY]",
                f"stage={compute_label}",
                f"requested={old_bucket_deg:.2f}deg",
                f"adjusted_to={new_bucket_deg:.2f}deg",
                f"pour_above_load_z={fmt_optional((carry_report or {}).get('pour_above_load_z'))}",
                "reason=relative_loaded_carry_pose",
            )
        return q_pre, carry_report

    q_pre_dump, pre_dump_carry_report = compute_pre_dump_carry_pose(q_release_align, q_lift, "staged_unload_to_bin")

    def exec_clearance_for(q_pose):
        return unload_pose_bucket_clearance_report(q_pose, reference_q=q_lift)

    def needs_higher_exec_pose(clearance):
        if not loaded_route_test:
            return False
        if not isinstance(clearance, dict) or not bool(clearance.get("ok", False)):
            return False
        return float(clearance.get("exec_margin", 0.0) or 0.0) < 0.0

    unload_goal_validation = validate_unload_goal(
        q_lift,
        q_pre_dump,
        q_dump=q_release_align,
        dump_info=dump_info,
        label="staged_unload_to_bin",
        deadline=deadline,
    )
    candidate["unload_goal_validation"] = unload_goal_validation
    STATE["dig_plan_candidate"] = candidate
    unload_exec_clearance = exec_clearance_for(q_pre_dump)
    needs_exec_high_retry = needs_higher_exec_pose(unload_exec_clearance)
    if needs_exec_high_retry:
        info_print(
            "[UNLOAD GOAL RETRY HIGH]",
            "label=staged_unload_to_bin",
            "reason=insufficient_execute_clearance",
            f"min_bucket_z={fmt_optional(unload_exec_clearance.get('min_bucket_z'))}",
            f"required_exec_z={fmt_optional(unload_exec_clearance.get('exec_required_clearance_z'))}",
            f"exec_margin={fmt_optional(unload_exec_clearance.get('exec_margin'))}",
            force_log=True,
        )
    if (
        (not bool(unload_goal_validation.get("ok", False)) and bool(unload_goal_validation.get("can_retry_high", False)))
        or needs_exec_high_retry
    ):
        _, required_z = unload_bin_wall_clearance_required_z()
        reach = unload_goal_validation.get("reachability", {}) if isinstance(unload_goal_validation.get("reachability"), dict) else {}
        max_bucket_min_z = reach.get("max_reachable_bucket_min_z")
        current_release_z = None
        try:
            if isinstance(dump_info, dict):
                current_release = dump_info.get("release_target") or dump_info.get("pour_target")
                if current_release is not None:
                    current_release_z = float(np.array(current_release, dtype=np.float32).reshape(-1)[2])
        except Exception:
            current_release_z = None
        if current_release_z is None:
            try:
                current_release_z = float(np.array(unload_bin_dump_point(), dtype=np.float32).reshape(-1)[2])
            except Exception:
                current_release_z = float(required_z) + 0.75
        high_base_z = max(float(current_release_z), float(required_z) + 0.55)
        retry_zs = [
            float(high_base_z) + 0.25,
            float(high_base_z) + 0.55,
            float(high_base_z) + 0.90,
            float(high_base_z) + 1.25,
        ]
        try:
            if max_bucket_min_z is not None:
                retry_zs.append(max(float(high_base_z) + 0.20, float(max_bucket_min_z) - 0.08))
        except Exception:
            pass
        retry_zs = sorted({round(float(z), 3) for z in retry_zs})
        retry_ok = False
        best_retry_ok = None
        for retry_idx, retry_z in enumerate(retry_zs):
            if planning_deadline_exceeded(deadline):
                break
            info_print(
                "[UNLOAD GOAL RETRY HIGH]",
                f"label=staged_unload_to_bin",
                f"attempt={retry_idx + 1}/{len(retry_zs)}",
                f"release_z={retry_z:.3f}",
                f"required_z={required_z:.3f}",
                force_log=loaded_route_test,
            )
            q_retry_align, retry_info = plan_dump_pose_to_bin(
                q_seed=q_lift,
                dump_deg=unload_dump_target_deg(),
                label=f"staged_unload_to_bin_high_{retry_idx + 1}",
                log=True,
                allow_unaligned=True,
                max_correction_iters=3 if loaded_route_test else 1,
                deadline=child_planning_deadline(deadline, 1.8, min_seconds=0.35),
                goal_obstacle_check=False,
                bucket_candidate_count_override=5 if loaded_route_test else None,
                release_z_override=retry_z,
            )
            if q_retry_align is None:
                continue
            q_pre_retry, retry_carry_report = compute_pre_dump_carry_pose(
                q_retry_align,
                q_lift,
                f"staged_unload_to_bin_high_{retry_idx + 1}",
            )
            retry_validation = validate_unload_goal(
                q_lift,
                q_pre_retry,
                q_dump=q_retry_align,
                dump_info=retry_info,
                label=f"staged_unload_to_bin_high_{retry_idx + 1}",
                deadline=deadline,
            )
            if bool(retry_validation.get("ok", False)):
                retry_exec_clearance = exec_clearance_for(q_pre_retry)
                if best_retry_ok is None or float(retry_exec_clearance.get("exec_margin", -999.0) or -999.0) > float(
                    best_retry_ok.get("exec_clearance", {}).get("exec_margin", -999.0) or -999.0
                ):
                    best_retry_ok = {
                        "q_release_align": q_retry_align,
                        "dump_info": retry_info,
                        "q_pre_dump": q_pre_retry,
                        "carry_report": retry_carry_report,
                        "validation": retry_validation,
                        "exec_clearance": retry_exec_clearance,
                        "release_z": retry_z,
                    }
                if needs_higher_exec_pose(retry_exec_clearance):
                    info_print(
                        "[UNLOAD GOAL RETRY HIGH LOW_MARGIN]",
                        f"release_z={retry_z:.3f}",
                        f"min_bucket_z={fmt_optional(retry_exec_clearance.get('min_bucket_z'))}",
                        f"required_exec_z={fmt_optional(retry_exec_clearance.get('exec_required_clearance_z'))}",
                        f"exec_margin={fmt_optional(retry_exec_clearance.get('exec_margin'))}",
                        f"q_pre_dump={q_deg_values(q_pre_retry, wrap_swing_for_display=True)}",
                        force_log=loaded_route_test,
                    )
                    continue
                q_release_align = q_retry_align
                dump_info = retry_info
                q_pre_dump = q_pre_retry
                pre_dump_carry_report = retry_carry_report
                unload_goal_validation = retry_validation
                unload_exec_clearance = retry_exec_clearance
                candidate["unload_goal_validation"] = retry_validation
                STATE["dig_plan_candidate"] = candidate
                retry_ok = True
                info_print(
                    "[UNLOAD GOAL RETRY HIGH OK]",
                    f"release_z={retry_z:.3f}",
                    f"q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}",
                    force_log=True,
            )
            break
        if not retry_ok and best_retry_ok is not None:
            q_release_align = np.array(best_retry_ok["q_release_align"], dtype=np.float32).copy()
            dump_info = best_retry_ok["dump_info"]
            q_pre_dump = np.array(best_retry_ok["q_pre_dump"], dtype=np.float32).copy()
            pre_dump_carry_report = best_retry_ok["carry_report"]
            unload_goal_validation = best_retry_ok["validation"]
            unload_exec_clearance = best_retry_ok["exec_clearance"]
            candidate["unload_goal_validation"] = unload_goal_validation
            STATE["dig_plan_candidate"] = candidate
            info_print(
                "[UNLOAD GOAL RETRY HIGH BEST_AVAILABLE]",
                f"release_z={fmt_optional(best_retry_ok.get('release_z'))}",
                f"exec_margin={fmt_optional(unload_exec_clearance.get('exec_margin'))}",
                f"q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}",
                force_log=True,
            )
        if not retry_ok:
            candidate["unload_goal_validation"] = unload_goal_validation
            STATE["dig_plan_candidate"] = candidate

    if not bool(unload_goal_validation.get("ok", False)):
        fail_reason = str(unload_goal_validation.get("reason", "unload_goal_invalid"))
        reach = unload_goal_validation.get("reachability", {}) if isinstance(unload_goal_validation.get("reachability"), dict) else {}
        set_execution_failure_reason(
            "planning_failed/staged_unload_goal:"
            + fail_reason
            + f"; reach_margin={fmt_optional(reach.get('reach_margin'))}"
        )
        info_print(
            "[DIG PLAN STAGED FAILED]",
            "stage=unload_to_bin",
            f"reason={fail_reason}",
            f"reach_margin={fmt_optional(reach.get('reach_margin'))}",
            f"best_q={reach.get('best_q_deg')}",
            force_log=True,
        )
        return False

    q_dump = bucket_only_dump_pose(q_pre_dump, unload_dump_target_deg())
    final_drop = dump_info.get("drop") if isinstance(dump_info, dict) else None
    if not isinstance(final_drop, dict):
        final_drop = unload_drop_report(q=q_release_align, reference_q=q_lift)
    if isinstance(final_drop, dict) and not unload_drop_execution_ready(final_drop):
        reason = (
            "planning_failed/staged_unload_drop_unaligned:"
            f"inside_xy={final_drop.get('inside_xy')} "
            f"above_wall={final_drop.get('above_wall')} "
            f"scatter_xy_ok={final_drop.get('scatter_xy_ok')} "
            f"acceptance={final_drop.get('landing_acceptance')} "
            f"xy_err={fmt_optional(final_drop.get('xy_err'))}"
        )
        set_execution_failure_reason(reason)
        info_print(
            "[DIG PLAN STAGED FAILED]",
            "stage=unload_to_bin",
            "reason=drop_not_inside_bin",
            f"inside_xy={final_drop.get('inside_xy')}",
            f"above_wall={final_drop.get('above_wall')}",
            f"scatter_xy_ok={final_drop.get('scatter_xy_ok')}",
            f"acceptance={final_drop.get('landing_acceptance')}",
            f"xy_err={fmt_optional(final_drop.get('xy_err'))}",
            force_log=True,
        )
        return False
    unload_exec_clearance = exec_clearance_for(q_pre_dump)
    info_print(
        "[UNLOAD EXEC GOAL]",
        f"q_carry={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}",
        f"q_release_align={q_deg_values(q_release_align, wrap_swing_for_display=True)}",
        f"q_dump={q_deg_values(q_dump, wrap_swing_for_display=True)}",
        f"release_align_deg={unload_release_alignment_bucket_deg(unload_dump_target_deg()):.2f}",
        f"min_bucket_z={fmt_optional(unload_exec_clearance.get('min_bucket_z'))}",
        f"exec_required_z={fmt_optional(unload_exec_clearance.get('exec_required_clearance_z'))}",
        f"exec_margin={fmt_optional(unload_exec_clearance.get('exec_margin'))}",
        "dump_policy=bucket_only_after_arrival",
        force_log=True,
    )

    direct_samples = 8 if loaded_route_test else DIG_PLAN_PATH_CHECK_SAMPLES
    route_samples = 3 if loaded_route_test else PATH_ROUTE_PLANNING_SAMPLE_COUNT
    direct_ok, kind, reason, sample, report = path_segment_check(
        q_lift, q_pre_dump, "unload_to_bin", samples=direct_samples, deadline=deadline
    )
    info_print(
        "[POST SECURE PLAN DIRECT CHECK]",
        f"loaded_route_test={loaded_route_test}",
        f"direct_ok={direct_ok}",
        f"samples={direct_samples}",
        f"kind={kind}",
        f"reason={reason}",
        f"q_pre_dump={q_deg_values(q_pre_dump, wrap_swing_for_display=True)}",
        force_log=loaded_route_test,
    )
    route_seq = []
    route_points = []
    route_stages = []
    q_route_seed = q_lift.copy()
    if not direct_ok:
        info_print(
            "[POST SECURE PLAN ROUTE START]",
            f"loaded_route_test={loaded_route_test}",
            f"kind={kind}",
            f"reason={reason}",
            f"samples={route_samples}",
            f"remaining={fmt_optional(max(0.0, float(deadline) - time.time()))}s",
            force_log=loaded_route_test,
        )
        route, route_reason = find_clearance_route(
            q_lift,
            q_pre_dump,
            "unload_to_bin",
            "staged_unload_to_bin",
            deadline=deadline,
            samples=route_samples,
        )
        if route is None:
            fallback, fallback_reason = staged_high_carry_unload_fallback(q_lift, q_pre_dump, deadline=deadline)
            if fallback is not None:
                route = fallback.get("route", [])
                route_reason = (
                    f"{fallback.get('reason', 'high_carry_fallback')}; "
                    f"original_route={route_reason}; original={kind}:{reason}; sample={sample}"
                )
                q_pre_dump = np.array(fallback.get("q_pre_dump", q_pre_dump), dtype=np.float32).copy()
                deadline = max(deadline, time.time() + 5.0) if loaded_route_test else time.time() + 5.0
                info_print(
                    "[DIG PLAN STAGED FALLBACK]",
                    "stage=unload_to_bin",
                    "method=high_carry_route",
                    f"route_waypoints={len(route)}",
                    f"reason={route_reason}",
                )
            else:
                set_execution_failure_reason(
                    f"planning_failed/staged_unload_route:{kind}:{reason}; route={route_reason}; "
                    f"fallback={fallback_reason}; sample={sample}"
                )
                info_print(
                    "[DIG PLAN STAGED FAILED]",
                    "stage=unload_to_bin",
                    f"kind={kind}",
                    f"reason={reason}",
                    f"route={route_reason}",
                    f"fallback={fallback_reason}",
                )
                return False
        for route_idx, q_route_raw in enumerate(route):
            q_route = np.array(q_route_raw, dtype=np.float32).copy()
            route_label = f"clearance_route_post_{route_idx + 1}"
            route_duration = estimate_stage_motion_seconds(
                q_route_seed,
                q_route,
                requested_seconds=LOADED_ROUTE_MIN_STAGE_SECONDS if loaded_route_test else 0.55,
            )
            if loaded_route_test:
                route_duration = max(float(route_duration), float(LOADED_ROUTE_MIN_STAGE_SECONDS))
            route_target = predicted_end_world_point(q_route, end_effector=path_end_effector_for_mode("unload_to_bin"), reference_q=q_route_seed)
            route_row = make_stage_row_from_q(
                route_label,
                q_route,
                q_route_seed,
                route_duration,
                target_point=route_target,
                extra={
                    "route_source": "staged_post_secure_load",
                    "route_reason": str(route_reason),
                    "route_index": int(route_idx + 1),
                    "route_count": int(len(route)),
                    "clearance_route": {
                        "inserted": True,
                        "required": True,
                        "waypoints": int(len(route)),
                        "reason": str(route_reason),
                    },
                },
                deadline=deadline,
            )
            route_seq.append((route_label, q_route.copy(), float(route_duration)))
            route_points.append(np.array(route_target if route_target is not None else unload_bin_landing_point(), dtype=np.float32).copy())
            route_stages.append(route_row)
            q_route_seed = q_route.copy()

    q_release_align = bucket_only_dump_pose(q_pre_dump, unload_release_alignment_bucket_deg(unload_dump_target_deg()))
    q_dump = bucket_only_dump_pose(q_pre_dump, unload_dump_target_deg())
    unload_duration = estimate_stage_motion_seconds(
        q_route_seed,
        q_pre_dump,
        requested_seconds=LOADED_ROUTE_FINAL_STAGE_SECONDS if loaded_route_test else 1.2,
    )
    if loaded_route_test:
        unload_duration = max(float(unload_duration), float(LOADED_ROUTE_FINAL_STAGE_SECONDS))
    drop = final_drop if isinstance(final_drop, dict) else unload_drop_report(q=q_release_align, reference_q=q_lift)
    unload_row = make_stage_row_from_q(
        "unload_to_bin",
        q_pre_dump,
        q_route_seed,
        unload_duration,
        target_point=unload_bin_landing_point(),
        extra={
            "q_dump_rad": vec_list(q_dump, 4),
            "q_release_align_rad": vec_list(q_release_align, 4),
            "q_dump_deg": q_deg_values(q_dump, wrap_swing_for_display=True),
            "q_release_align_deg": q_deg_values(q_release_align, wrap_swing_for_display=True),
            "release_alignment_bucket_deg": unload_release_alignment_bucket_deg(unload_dump_target_deg()),
            "effector": "landing",
            "drop": compact_unload_drop(drop),
            "drop_alignment_ready": unload_drop_execution_ready(drop),
            "drop_alignment_policy": str(drop.get("landing_acceptance", "staged_runtime_execute_then_score")),
            "clearance_route": {
                "inserted": bool(route_seq),
                "required": bool(route_seq or not direct_ok),
                "waypoints": int(len(route_seq)),
                "reason": "direct_ok" if direct_ok else "staged_post_secure_load_route",
            },
            "staged_runtime_plan": True,
        },
        deadline=deadline,
    )

    seq.extend(carry_safe_seq)
    points.extend(carry_safe_points)
    if not skip_lift_stage:
        seq.extend([("lift_carry", q_lift.copy(), lift_duration)])
        lift_target = lift.get("target_point") if isinstance(lift, dict) else None
        points.append(
            np.array(
                lift_target if lift_target is not None else unload_bin_landing_point(),
                dtype=np.float32,
            ).reshape(-1)[:3].copy()
        )
    seq.extend(route_seq)
    points.extend(route_points)
    seq.append(("unload_to_bin", q_pre_dump.copy(), unload_duration))
    points.append(np.array(unload_bin_landing_point(), dtype=np.float32).reshape(-1)[:3].copy())

    stages = list(candidate.get("stages", []) or [])
    stages.extend(carry_safe_stages)
    if lift_row is not None:
        stages.append(lift_row)
    stages.extend(route_stages)
    stages.append(unload_row)
    candidate["stages"] = stages
    candidate["steps"] = len(seq)
    candidate["planned"] = True
    candidate["staged_post_secure_load_pending"] = False
    candidate["staged_post_secure_load_appended"] = True
    candidate["unload_landing_xyz"] = vec_list(unload_bin_landing_point(), 3)
    release_target = None
    if isinstance(dump_info, dict):
        release_target = dump_info.get("release_target")
    if release_target is None:
        release_target = unload_bin_dump_point()
    candidate["unload_release_xyz"] = vec_list(release_target, 3)
    candidate["unload_point_xyz"] = candidate["unload_release_xyz"]
    candidate["route_diagnostics"] = route_diagnostics_from_stages(stages)
    candidate["unload_ballistics"] = unload_ballistics_from_stages(stages)
    STATE["dig_plan_candidate"] = candidate
    STATE["dig_plan_points"] = points
    cache_dig_plan_trace_points(seq, start_q=STATE.get("dig_plan_start_q", CTRL.q_cmd.copy()))
    new_plan = build_shared_dig_plan_object(STATE.get("dig_plan_target", get_target_pos()), seq, points, candidate)
    new_plan["staged_execution"] = True
    new_plan["staged_prefix_ready"] = True
    new_plan["staged_post_secure_load_pending"] = False
    new_plan["staged_post_secure_load_appended"] = True
    if current_trace_mode() == 2:
        draw_trace(force=True)
    info_print(
        "[DIG PLAN STAGED APPEND]",
        "added=unload_to_bin" if skip_lift_stage else "added=lift_carry,unload_to_bin",
        f"route_waypoints={len(route_seq)}",
        f"drop_xy_err={fmt_optional(drop.get('xy_err'))}",
        f"inside_xy={drop.get('inside_xy')}",
        f"loaded_route_test={loaded_route_test}",
    )
    return True


def should_append_staged_post_secure_load_after_stage(stage_name):
    semantic = dig_plan_semantic_phase_name(stage_name)
    if semantic in ("secure_load", "secure_carry_safe"):
        return True
    return False


def plan_dig_sequence_from_target(target_xyz, max_seconds=None):
    STATE["last_dig_plan_candidates"] = []
    STATE["dig_plan_candidate"] = None
    STATE["dig_plan_best_failure"] = None
    best = None
    best_failure = None

    candidates = adaptive_dig_plan_candidates(target_xyz)
    budget_source = max_seconds if max_seconds is not None else STATE.get("dig_plan_build_budget_seconds", DIG_PLAN_MAX_BUILD_SECONDS)
    budget_seconds = float(budget_source or DIG_PLAN_MAX_BUILD_SECONDS)
    deadline = time.time() + budget_seconds
    previous_perf_deadline = STATE.get("dig_plan_active_perf_deadline")
    STATE["dig_plan_active_perf_deadline"] = time.perf_counter() + budget_seconds
    try:
        for candidate in candidates:
            if planning_deadline_exceeded(deadline):
                info_print(
                    "[DIG PLAN TIMEOUT]",
                    f"budget={budget_seconds:.2f}s",
                    f"evaluated={len(STATE.get('last_dig_plan_candidates', []))}",
                    f"best_ready={best is not None}",
                )
                break
            candidate_t0 = time.perf_counter()
            seq, points, detail = plan_dig_sequence_candidate(target_xyz, candidate, deadline=deadline)
            candidate_ms = 1000.0 * max(0.0, time.perf_counter() - candidate_t0)
            perf_block_record(
                f"plan_candidate:{candidate.get('id', 'candidate')}",
                candidate_ms,
                data={
                    "candidate": str(candidate.get("id", "candidate")),
                    "planned": bool(seq),
                    "failed_stage": "" if seq else str((detail or {}).get("failed_stage", "")),
                    "prefix": 0 if seq else int((detail or {}).get("planned_prefix", 0) or 0),
                    "budget_seconds": float(budget_seconds),
                },
            )
            row = dict(detail)
            row["candidate"] = dict(candidate)
            if seq:
                if planning_deadline_exceeded(deadline):
                    legacy_score = 0.0
                    score_reason = "post_plan_eval_skipped_due_budget"
                    reports = []
                else:
                    legacy_score, score_reason, reports = evaluate_dig_plan_candidate(seq, points, candidate, stages=row.get("stages"))
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
    finally:
        STATE["dig_plan_active_perf_deadline"] = previous_perf_deadline

    if best is None:
        STATE["dig_plan_best_failure"] = best_failure
        staged_seq = install_staged_prefix_plan_from_failure(target_xyz, best_failure)
        if staged_seq:
            plan = STATE.get("current_dig_plan")
            terminal = ""
            if isinstance(plan, dict):
                terminal = str(plan.get("staged_prefix_terminal_phase", "") or "")
            update_status(
                f"[DIG PLAN STAGED] executable prefix ready through {terminal or 'partial dig'}",
                force=True,
            )
            return staged_seq
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


def build_dig_plan_from_current_target(force_status=True, max_seconds=None):
    if bool(STATE.get("dig_plan_planning_active", False)):
        seq_existing = STATE.get("dig_plan_sequence", None)
        if seq_existing is not None and dig_plan_target_matches_current():
            return seq_existing
        update_status("[DIG PLAN] planner is already running; wait for current build", force=force_status)
        return None

    STATE["planning_cancel_requested"] = False
    STATE["planning_path_penalty_cache"] = {}
    STATE["planning_path_penalty_cache_hits"] = 0
    STATE["planning_path_penalty_cache_misses"] = 0
    STATE["dig_plan_planning_active"] = True
    STATE["dig_plan_planning_version"] = int(STATE.get("dig_plan_planning_version", 0)) + 1
    STATE["dig_plan_planning_source"] = "auto_collect" if bool(STATE.get("auto_collect_active", False)) else "manual_or_sync"
    build_t0 = time.time()
    previous_planning_snapshot = STATE.get("planning_sand_snapshot")
    previous_planning_snapshot_active = bool(STATE.get("planning_sand_snapshot_active", False))
    target = get_target_pos()
    target[2] = max(float(target[2]), GROUND_TOP_Z)

    try:
        try:
            snapshot_for_plan = None
            cached_snapshot = STATE.get("auto_collect_episode_sand_snapshot")
            cached_age = time.time() - float(STATE.get("auto_collect_episode_sand_snapshot_time", 0.0) or 0.0)
            if isinstance(cached_snapshot, dict) and cached_age <= float(AUTO_COLLECT_PLANNING_SNAPSHOT_MAX_AGE):
                snapshot_for_plan = cached_snapshot
            if snapshot_for_plan is None:
                snapshot_for_plan = get_sand_snapshot(force=False, label="dig_plan_build", max_age=3.0)
            STATE["planning_sand_snapshot"] = snapshot_for_plan
            STATE["planning_sand_snapshot_active"] = True
        except Exception as e:
            STATE["planning_sand_snapshot"] = None
            STATE["planning_sand_snapshot_active"] = False
            info_print("[WARN] [DIG PLAN SNAPSHOT] failed:", type(e).__name__, e)

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
        seq = plan_dig_sequence_from_target(target, max_seconds=max_seconds)
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
        perf_block_record(
            "build_dig_plan_from_current_target",
            STATE["dig_plan_last_build_ms"],
            data={
                "target": vec_list(target, 3),
                "candidate_count": len(STATE.get("last_dig_plan_candidates", []) or []),
                "plan_ready": STATE.get("dig_plan_sequence") is not None,
            },
        )
        STATE["planning_sand_snapshot"] = previous_planning_snapshot
        STATE["planning_sand_snapshot_active"] = previous_planning_snapshot_active
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
    clear_planning_runtime_caches("reset_dig_plan")
    ensure_unload_marker(label="reset_plan")
    if current_trace_mode() == 2:
        draw_trace(force=True)
    update_status("[DIG PLAN] reset", force=True)


def get_or_build_dig_plan():
    seq = STATE.get("dig_plan_sequence", None)
    if seq is None:
        seq = build_dig_plan_from_current_target(force_status=True)
    return seq


def install_loaded_unload_route_test_plan_from_current():
    if bool(STATE.get("dig_plan_planning_active", False)):
        update_status("[LOADED ROUTE TEST] planner already running", force=True)
        return False
    q_start = sync_motion_start_q("loaded_unload_route_test_start")
    target = get_target_pos() if TARGET_PATH else bucket_load_pos()
    if target is None:
        target = unload_bin_landing_point()
    target = np.array(target, dtype=np.float32).reshape(-1)[:3].copy()
    landing = unload_bin_landing_point()
    release = unload_bin_dump_point()
    ensure_unload_marker(landing, label="loaded_unload_route_test")

    STATE["dig_plan_start_q"] = q_start.copy()
    STATE["dig_plan_target"] = target.copy()
    STATE["dig_plan_points"] = []
    STATE["dig_plan_sequence"] = []
    STATE["dig_plan_step_index"] = 0
    STATE["dig_plan_trace_points"] = []
    STATE["dig_plan_trace_stage_breaks"] = []
    STATE["trace_planned_bucket_points"] = []
    candidate = {
        "id": "loaded_unload_route_test",
        "planned": True,
        "selected": True,
        "stages": [],
        "steps": 0,
        "staged_execution": True,
        "staged_prefix_ready": True,
        "staged_prefix_terminal_phase": "secure_load",
        "staged_post_secure_load_pending": True,
        "staged_post_secure_load_appended": False,
        "unload_landing_xyz": vec_list(landing, 3),
        "unload_release_xyz": vec_list(release, 3),
        "unload_point_xyz": vec_list(release, 3),
        "unload_dump_deg": float(BUCKET_UNLOAD_DUMP_DEG),
        "planner_cost": 0.0,
        "rank_cost": 0.0,
        "score": 0.0,
        "estimated_time": 0.0,
        "debug_task": "loaded_unload_route_test",
    }
    STATE["dig_plan_candidate"] = candidate
    build_shared_dig_plan_object(target, STATE["dig_plan_sequence"], STATE["dig_plan_points"], candidate)
    previous_active = bool(STATE.get("dig_plan_planning_active", False))
    previous_source = STATE.get("dig_plan_planning_source", "")
    previous_perf_deadline = STATE.get("dig_plan_active_perf_deadline")
    previous_cache = STATE.get("planning_path_penalty_cache")
    previous_swing_cache = STATE.get("planning_swing_corridor_cache")
    stale_planning_cancel = bool(STATE.get("planning_cancel_requested", False))
    STATE["planning_cancel_requested"] = False
    STATE["loaded_route_test_cleared_stale_cancel"] = stale_planning_cancel
    STATE["planning_path_penalty_cache"] = {}
    STATE["planning_path_penalty_cache_hits"] = 0
    STATE["planning_path_penalty_cache_misses"] = 0
    STATE["planning_swing_corridor_cache"] = {}
    STATE["planning_swing_corridor_cache_hits"] = 0
    STATE["planning_swing_corridor_cache_misses"] = 0
    STATE["dig_plan_planning_active"] = True
    STATE["dig_plan_planning_source"] = "loaded_route_test"
    STATE["dig_plan_active_perf_deadline"] = None
    try:
        t_obstacles = time.perf_counter()
        hit_before = int(STATE.get("rigid_obstacle_cache_hits", 0))
        miss_before = int(STATE.get("rigid_obstacle_cache_misses", 0))
        obstacles = rigid_obstacle_bboxes(force=False)
        hit_delta = int(STATE.get("rigid_obstacle_cache_hits", 0)) - hit_before
        miss_delta = int(STATE.get("rigid_obstacle_cache_misses", 0)) - miss_before
        info_print(
            "[LOADED ROUTE TEST SNAPSHOT]",
            f"rigid_obstacles={len(obstacles)}",
            f"obstacle_snapshot_ms={(time.perf_counter() - t_obstacles) * 1000.0:.1f}",
            f"cache_hit={bool(hit_delta > 0)}",
            f"cache_miss={bool(miss_delta > 0)}",
            force_log=True,
        )
        ok = append_staged_post_secure_load_plan(task_label="loaded_unload_route_test")
    finally:
        info_print(
            "[LOADED ROUTE TEST CACHE]",
            f"path_hits={int(STATE.get('planning_path_penalty_cache_hits', 0))}",
            f"path_misses={int(STATE.get('planning_path_penalty_cache_misses', 0))}",
            f"swing_hits={int(STATE.get('planning_swing_corridor_cache_hits', 0))}",
            f"swing_misses={int(STATE.get('planning_swing_corridor_cache_misses', 0))}",
            force_log=True,
        )
        STATE["dig_plan_planning_active"] = previous_active
        STATE["dig_plan_planning_source"] = previous_source
        STATE["dig_plan_active_perf_deadline"] = previous_perf_deadline
        STATE["planning_path_penalty_cache"] = previous_cache if isinstance(previous_cache, dict) else {}
        STATE["planning_swing_corridor_cache"] = previous_swing_cache if isinstance(previous_swing_cache, dict) else {}
    seq = STATE.get("dig_plan_sequence")
    if not ok or not isinstance(seq, list) or not seq:
        reason = str(STATE.get("last_execution_failure_reason", "") or "loaded route test plan failed")
        update_status(f"[LOADED ROUTE TEST] plan failed: {reason}", force=True)
        return False
    if current_trace_mode() == 2:
        draw_trace(force=True)
    update_status(f"[LOADED ROUTE TEST] plan ready: {len(seq)} stages", force=True)
    planned_landing = candidate.get("unload_landing_xyz", vec_list(landing, 3))
    planned_release = candidate.get("unload_release_xyz", vec_list(release, 3))
    planned_drop = None
    for row in reversed(candidate.get("stages", []) or []):
        if isinstance(row, dict) and str(row.get("phase", "")) == "unload_to_bin":
            planned_drop = row.get("drop") if isinstance(row.get("drop"), dict) else None
            break
    if isinstance(planned_drop, dict):
        planned_landing = planned_drop.get("landing") or planned_landing
        planned_release = planned_drop.get("release") or planned_release
    info_print(
        "[LOADED ROUTE TEST PLAN]",
        f"start_q={q_deg_values(q_start, wrap_swing_for_display=True)}",
        f"stages={[str(item[0]) for item in seq]}",
        f"landing_target={vec_list(landing, 3)}",
        f"planned_landing={planned_landing}",
        f"planned_release={planned_release}",
        f"release_target={candidate.get('unload_release_xyz')}",
        f"acceptance={planned_drop.get('landing_acceptance') if isinstance(planned_drop, dict) else None}",
        f"release_xy_err={fmt_optional(planned_drop.get('release_xy_err') if isinstance(planned_drop, dict) else None)}",
        f"drop_xy_err={fmt_optional(planned_drop.get('xy_err') if isinstance(planned_drop, dict) else None)}",
    )
    return True


async def execute_loaded_unload_route_test_from_current():
    if STATE.get("auto_collect_active", False):
        stop_auto_collect()
    invalidate_active_task("loaded_route_test_start")
    STATE["follow"] = False
    STATE["manual_joint_active"] = False
    STATE["manual_joint_target"] = None
    try:
        bucket_min = float(bbox_min_z(BUCKET_LINK))
    except Exception:
        bucket_min = 0.0
    if bucket_min < float(GROUND_TOP_Z) - 0.05:
        reason = f"bucket_below_hard_ground={bucket_min:.3f}; lift boom/arm before loaded route test"
        update_status(f"[LOADED ROUTE TEST] blocked: {reason}", force=True)
        info_print("[LOADED ROUTE TEST BLOCKED]", reason)
        return False
    metrics = record_phase_metrics("loaded_route_test_start")
    bucket_particle_diagnostic("loaded_route_test_start")
    info_print(
        "[LOADED ROUTE TEST]",
        "sand_unchanged=True",
        f"metrics_bucket={metrics.get('bucket_from_pile_count') if isinstance(metrics, dict) else None}",
    )

    update_status("[LOADED ROUTE TEST] planning loaded unload route...", force=True)
    await step_updates(1)
    if not install_loaded_unload_route_test_plan_from_current():
        return False
    await step_updates(1)
    return await execute_dig_target_ball(
        rebuild_plan=False,
        task_name="loaded_unload_route_test",
        return_home=False,
    )


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


def unload_release_alignment_bucket_deg(dump_deg=None):
    final_deg = unload_dump_target_deg() if dump_deg is None else float(dump_deg)
    align_deg = float(UNLOAD_DUMP_FLOW_CENTER_BUCKET_DEG)
    # If a future dump target is less open than the release-alignment angle, use the reachable final target.
    if final_deg < align_deg:
        align_deg = final_deg
    return float(align_deg)


def plan_dump_pose_to_bin(
    q_seed=None,
    dump_deg=None,
    label="unload_dump",
    log=True,
    max_correction_iters=None,
    allow_unaligned=False,
    deadline=None,
    goal_obstacle_check=True,
    bucket_candidate_count_override=None,
    release_z_override=None,
):
    if q_seed is None:
        q_seed = CTRL.q_cmd.copy()
    else:
        q_seed = np.array(q_seed, dtype=np.float32).copy()
    if dump_deg is None:
        dump_deg = unload_dump_target_deg()
    final_dump_deg = float(dump_deg)
    release_alignment_bucket_deg = unload_release_alignment_bucket_deg(final_dump_deg)

    bucket_idx = CTRL.name_to_idx["bucket"]
    q_seed_dump = q_seed.copy()
    q_seed_dump[bucket_idx] = deg_to_rad(float(release_alignment_bucket_deg))
    q_seed_dump = clip_command_near(q_seed_dump, reference=q_seed)
    ctx = task_scene_context()
    drop_target = unload_bin_landing_point(ctx=ctx)
    nominal_release_target = unload_bin_dump_point(ctx=ctx)
    bin_z_range = np.array(ctx["unload_bin_z_range"], dtype=np.float32).reshape(-1)
    wall_z = float(bin_z_range[1]) if len(bin_z_range) >= 2 else GROUND_TOP_Z
    preferred_release_z = preferred_unload_release_z(ctx=ctx, wall_z=wall_z)
    min_release_z = max(
        float(nominal_release_target[2]),
        float(wall_z) + float(UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z),
        float(GROUND_TOP_Z) + float(UNLOAD_TARGET_MIN_Z),
        float(preferred_release_z),
    )
    if release_z_override is not None:
        try:
            min_release_z = max(float(min_release_z), float(release_z_override))
        except Exception:
            pass
    center_release_mode = bool(UNLOAD_FORCE_CENTER_HIGH_RELEASE)
    if center_release_mode:
        release_target = np.array(drop_target, dtype=np.float32).reshape(-1)[:3].copy()
        release_target[2] = float(min_release_z)
        # In center-release mode the controlled quantity is the bucket opening center.
        # A precomputed pour->opening offset is brittle across IK poses, so use the
        # desired opening center directly and close the residual with release_xy feedback.
        pour_target = release_target.copy()
        load = bucket_point_world("load", q=q_seed_dump, reference_q=q_seed)
        initial_drift = unload_drop_drift_model(
            q=q_seed_dump,
            reference_q=q_seed,
            release=release_target,
            load=load,
            wall_z=wall_z,
        )
    else:
        release_target, initial_drift = unload_release_target_for_landing(
            drop_target,
            q_estimate=q_seed_dump,
            q_start=q_seed,
            release_z=min_release_z,
            wall_z=wall_z,
        )
        pour_target = pour_target_for_unload_release_source(release_target, q_estimate=q_seed_dump, q_start=q_seed)
    initial_release_target = np.array(release_target, dtype=np.float32).copy()
    initial_pour_target = np.array(pour_target, dtype=np.float32).copy()
    seed_drop = unload_drop_report(q=q_seed_dump, reference_q=q_seed)
    seed_center_ready = (
        (not center_release_mode)
        or (
            bool(seed_drop.get("release_centered_ok", False))
            and float(seed_drop.get("release_xy_err", 999.0) or 999.0) <= float(UNLOAD_CENTER_RELEASE_XY_TOL)
        )
        or (
            center_release_mode
            and unload_drop_execution_ready(seed_drop)
            and bool(seed_drop.get("close_xy", False))
            and float(seed_drop.get("release_xy_err", 999.0) or 999.0) <= float(UNLOAD_CENTER_RELEASE_SOFT_XY_TOL)
        )
    )
    if allow_unaligned and unload_drop_execution_ready(seed_drop) and seed_center_ready:
        info = {
            "reason": "current_pose_dump_already_inside_bin",
            "drop": seed_drop,
            "drop_target": vec_list(drop_target, 3),
            "pour_target": vec_list(seed_drop.get("release"), 3),
            "release_target": vec_list(seed_drop.get("release"), 3),
            "release_source": seed_drop.get("release_source", "bucket_opening_center"),
            "dump_bucket_target_deg": float(final_dump_deg),
            "release_alignment_bucket_deg": float(release_alignment_bucket_deg),
            "q_release_align_rad": vec_list(q_seed_dump, 4),
            "q_release_align_deg": q_deg_values(q_seed_dump, wrap_swing_for_display=True),
            "dump_bucket_err_deg": 0.0,
            "ik_bucket_err_deg": 0.0,
            "drop_alignment_ready": True,
            "drop_alignment_policy": str(seed_drop.get("landing_acceptance", "current_pose_fast_path")),
        }
        if log:
            info_print(
                f"[UNLOAD DUMP PLAN] {label}: current_pose_fast_path "
                f"landing_target={vec_list(drop_target, 3)} release={seed_drop.get('release')} "
                f"landing={seed_drop.get('landing')} xy_err={fmt_optional(seed_drop.get('xy_err'))} "
                f"inside_xy={seed_drop.get('inside_xy')} close_xy={seed_drop.get('close_xy')} "
                f"scatter_xy_ok={seed_drop.get('scatter_xy_ok')} acceptance={seed_drop.get('landing_acceptance')} "
                f"source_clearance={fmt_optional(seed_drop.get('source_clearance'))} "
                f"q={q_deg_values(q_seed_dump, wrap_swing_for_display=True)}"
            )
        draw_unload_dump_debug(
            label,
            landing_target=drop_target,
            release_target=seed_drop.get("release"),
            q_seed=q_seed,
            q_seed_dump=q_seed_dump,
            best_drop=seed_drop,
            reason="current_pose_fast_path",
        )
        return q_seed_dump, info
    if log:
        info_print(
            f"[UNLOAD DUMP PLAN START] {label}: "
            f"deadline={'none' if deadline is None else fmt_optional(max(0.0, float(deadline) - time.time())) + 's'} "
            f"landing_target={vec_list(drop_target, 3)} "
            f"required_release={vec_list(initial_release_target, 3)} "
            f"ik_pour_target={vec_list(initial_pour_target, 3)} "
            f"release_policy={'opening_center_high_no_drift_compensation' if center_release_mode else 'drift_compensated'} "
            f"ik_target_policy={'direct_opening_center_iterative' if center_release_mode else 'pour_offset_from_release'} "
            f"release_source=bucket_opening_center "
            f"release_align_deg={release_alignment_bucket_deg:.2f} "
            f"final_dump_deg={final_dump_deg:.2f} "
            f"preferred_release_z={fmt_optional(preferred_release_z)} "
            f"wall_z={fmt_optional(wall_z)} "
            f"current_release={seed_drop.get('release')} "
            f"current_landing={seed_drop.get('landing')} "
            f"current_xy_err={fmt_optional(seed_drop.get('xy_err'))} "
            f"inside_xy={seed_drop.get('inside_xy')} "
            f"above_wall={seed_drop.get('above_wall')} "
            f"drift={fmt_optional(seed_drop.get('drift_distance'))} "
            f"q_seed={q_deg_values(q_seed, wrap_swing_for_display=True)} "
            f"q_seed_dump={q_deg_values(q_seed_dump, wrap_swing_for_display=True)}",
            force_log=True,
        )
    draw_unload_dump_debug(
        label,
        landing_target=drop_target,
        release_target=initial_release_target,
        q_seed=q_seed,
        q_seed_dump=q_seed_dump,
        best_drop=seed_drop,
        reason="dump_plan_start",
    )
    best = None
    last_reason = "no dump candidate evaluated"

    correction_iters = max(1, int(UNLOAD_DROP_IK_CORRECTION_ITERS if max_correction_iters is None else max_correction_iters))
    if deadline is not None:
        correction_iters = min(correction_iters, 3 if center_release_mode else 1)
    for attempt in range(correction_iters):
        if planning_deadline_exceeded(deadline):
            draw_unload_dump_debug(
                label,
                landing_target=drop_target,
                release_target=initial_release_target,
                q_seed=q_seed,
                q_seed_dump=q_seed_dump,
                reason="planning budget exceeded before dump IK",
            )
            return None, "planning budget exceeded"
        if log:
            info_print(
                f"[UNLOAD DUMP PLAN TRY] {label}: "
                f"attempt={attempt + 1}/{correction_iters} "
                f"release_target={vec_list(release_target, 3)} "
                f"ik_pour_target={vec_list(pour_target, 3)} "
                f"remaining={'none' if deadline is None else fmt_optional(max(0.0, float(deadline) - time.time())) + 's'}",
                force_log=True,
            )
        q_dump, info = solve_priority_ik_to_target(
            pour_target,
            q_seed=q_seed_dump,
            preferred_bucket_rad=deg_to_rad(float(release_alignment_bucket_deg)),
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
            bucket_candidate_count=(
                int(bucket_candidate_count_override)
                if bucket_candidate_count_override is not None
                else (9 if deadline is not None else 25)
            ),
            deadline=deadline,
            score_goal_obstacle=bool(goal_obstacle_check),
        )

        if q_dump is None:
            last_reason = str(info)
            if log:
                info_print(
                    f"[UNLOAD DUMP PLAN TRY FAILED] {label}: "
                    f"attempt={attempt + 1}/{correction_iters} reason={last_reason}",
                    force_log=True,
                )
            break

        q_ik = clip_command_near(q_dump, reference=q_seed)
        ik_bucket_err = abs(rad_to_deg(wrap_angle(float(q_ik[bucket_idx]) - deg_to_rad(float(release_alignment_bucket_deg)))))
        q_dump = q_ik.copy()
        q_dump[bucket_idx] = deg_to_rad(float(release_alignment_bucket_deg))
        q_dump = clip_command_near(q_dump, reference=q_seed)
        bucket_err = abs(rad_to_deg(wrap_angle(float(q_dump[bucket_idx]) - deg_to_rad(float(release_alignment_bucket_deg)))))
        drop = unload_drop_report(q=q_dump, reference_q=q_seed)
        xy_err = float(drop.get("xy_err", 999.0))
        source_clearance = float(drop.get("source_clearance", -999.0))
        drop_ready = unload_drop_execution_ready(drop)
        center_ready = (
            (not center_release_mode)
            or (
                bool(drop.get("release_centered_ok", False))
                and float(drop.get("release_xy_err", 999.0) or 999.0) <= float(UNLOAD_CENTER_RELEASE_XY_TOL)
            )
        )
        overflow_xy = float(drop.get("bin_overflow_xy", xy_err) or 0.0)
        release_xy_err = float(drop.get("release_xy_err", xy_err) or 0.0)
        center_soft_ready = bool(
            center_ready
            or (
                center_release_mode
                and drop_ready
                and release_xy_err <= float(UNLOAD_CENTER_RELEASE_SOFT_XY_TOL)
            )
        )
        effective_xy_err = release_xy_err if center_release_mode else (overflow_xy if drop_ready else xy_err)
        source_penalty = max(0.0, UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z - source_clearance)
        bucket_penalty = max(0.0, bucket_err - UNLOAD_DUMP_BUCKET_TOL_DEG)
        ik_bucket_penalty = max(0.0, ik_bucket_err - UNLOAD_DUMP_BUCKET_TOL_DEG)
        planar_err = float(info.get("planar_err", 0.0)) if isinstance(info, dict) else 0.0
        height_bonus = min(2.5, max(0.0, source_clearance - float(UNLOAD_DROP_SOURCE_MIN_CLEARANCE_Z))) * 7.5
        acceptance_penalty = 0.0 if (drop_ready and center_soft_ready) else 35.0
        cost = (
            65.0 * effective_xy_err
            + 35.0 * source_penalty
            + 0.6 * bucket_penalty
            + 0.15 * ik_bucket_penalty
            + 5.0 * planar_err
            + acceptance_penalty
            - height_bonus
        )
        row = {
            "q": q_dump,
            "info": dict(info) if isinstance(info, dict) else {"reason": str(info)},
            "drop": drop,
            "bucket_err": float(bucket_err),
            "ik_bucket_err": float(ik_bucket_err),
            "cost": float(cost),
            "attempt": int(attempt),
            "release_target": release_target.copy(),
            "pour_target": pour_target.copy(),
            "center_ready": bool(center_ready),
            "center_soft_ready": bool(center_soft_ready),
        }
        if best is None or row["cost"] < best["cost"]:
            best = row
        if log:
            info_print(
                f"[UNLOAD DUMP PLAN TRY RESULT] {label}: "
                f"attempt={attempt + 1}/{correction_iters} "
                f"xy_err={fmt_optional(drop.get('xy_err'))} "
                f"inside_xy={drop.get('inside_xy')} "
                f"above_wall={drop.get('above_wall')} "
                f"close_xy={drop.get('close_xy')} "
                f"scatter_xy_ok={drop.get('scatter_xy_ok')} "
                f"release_centered_ok={drop.get('release_centered_ok')} "
                f"release_xy_err={fmt_optional(drop.get('release_xy_err'))} "
                f"center_required={center_release_mode} "
                f"center_ready={center_ready} "
                f"center_soft_ready={center_soft_ready} "
                f"release_align_deg={release_alignment_bucket_deg:.2f} "
                f"final_dump_deg={final_dump_deg:.2f} "
                f"acceptance={drop.get('landing_acceptance')} "
                f"source_clearance={fmt_optional(drop.get('source_clearance'))} "
                f"bucket_err={bucket_err:.2f}deg "
                f"cost={cost:.2f} "
                f"q={q_deg_values(q_dump, wrap_swing_for_display=True)}",
                force_log=True,
            )

        if drop_ready and center_soft_ready:
            row["info"]["drop"] = drop
            row["info"]["drop_target"] = vec_list(drop_target, 3)
            row["info"]["pour_target"] = vec_list(pour_target, 3)
            row["info"]["release_target"] = vec_list(release_target, 3)
            row["info"]["release_source"] = drop.get("release_source", "bucket_opening_center")
            row["info"]["dump_bucket_target_deg"] = float(final_dump_deg)
            row["info"]["release_alignment_bucket_deg"] = float(release_alignment_bucket_deg)
            row["info"]["q_release_align_rad"] = vec_list(q_dump, 4)
            row["info"]["q_release_align_deg"] = q_deg_values(q_dump, wrap_swing_for_display=True)
            row["info"]["dump_bucket_err_deg"] = float(bucket_err)
            row["info"]["ik_bucket_err_deg"] = float(ik_bucket_err)
            row["info"]["drop_alignment_ready"] = True
            row["info"]["drop_alignment_policy"] = str(
                drop.get(
                    "landing_acceptance",
                    "center_soft_close_xy_high_release" if center_release_mode and not center_ready else "scatter_tolerant_high_release",
                )
            )
            if log:
                raw_swing = row["info"].get("raw_swing_goal")
                swing_goal = row["info"].get("swing_goal")
                info_print(
                    f"[UNLOAD DUMP PLAN] {label}: "
                    f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(release_target, 3)} "
                    f"ik_pour_target={vec_list(pour_target, 3)} "
                    f"landing={drop.get('landing')} release={drop.get('release')} "
                    f"xy_err={fmt_optional(drop.get('xy_err'))} source_clearance={fmt_optional(drop.get('source_clearance'))} "
                    f"acceptance={drop.get('landing_acceptance')} scatter_xy_ok={drop.get('scatter_xy_ok')} "
                    f"release_align_deg={release_alignment_bucket_deg:.2f} final_dump_deg={final_dump_deg:.2f} "
                    f"bucket_err={bucket_err:.2f}deg "
                    f"ik_bucket_err={ik_bucket_err:.2f}deg "
                    f"attempt={attempt} drift={fmt_optional(drop.get('drift_distance'))} "
                    f"drift_xy={drop.get('drift_xy')} "
                    f"swing_goal={fmt_optional(None if swing_goal is None else rad_to_deg(swing_goal))}deg "
                    f"raw_swing={fmt_optional(None if raw_swing is None else rad_to_deg(raw_swing))}deg "
                    f"planar_err={fmt_optional(row['info'].get('planar_err'))} "
                    f"q={q_deg_values(q_dump, wrap_swing_for_display=True)}"
                )
            return q_dump, row["info"]

        if center_release_mode:
            correction_drop = drop
            correction_q = q_dump
            correction_target = release_target.copy()
            try:
                best_drop = best.get("drop", {}) if isinstance(best, dict) else {}
                best_err = float(best_drop.get("release_xy_err", 999.0) or 999.0)
                row_err = float(drop.get("release_xy_err", 999.0) or 999.0)
                if best_err + 1e-4 < row_err:
                    correction_drop = best_drop
                    correction_q = np.array(best.get("q", q_dump), dtype=np.float32).copy()
                    correction_target = np.array(best.get("release_target", release_target), dtype=np.float32).copy()
            except Exception:
                correction_drop = drop
                correction_q = q_dump
                correction_target = release_target.copy()

            err_xy = np.array(
                [float(correction_drop.get("release_dx", 0.0)), float(correction_drop.get("release_dy", 0.0))],
                dtype=np.float32,
            )
            err_norm = float(np.linalg.norm(err_xy))
            last_reason = (
                f"center opening release failed: release_xy_err={fmt_optional(correction_drop.get('release_xy_err'))} "
                f"release_centered_ok={correction_drop.get('release_centered_ok')} "
                f"xy_err={fmt_optional(correction_drop.get('xy_err'))} "
                f"acceptance={correction_drop.get('landing_acceptance')}"
            )
            if err_norm < 0.01:
                break
            if err_norm > UNLOAD_DROP_MAX_XY_CORRECTION:
                err_xy *= float(UNLOAD_DROP_MAX_XY_CORRECTION / err_norm)
            gain = clamp(float(UNLOAD_CENTER_RELEASE_CORRECTION_GAIN), 0.05, 1.0)
            release_target = correction_target.copy()
            release_target[0] -= float(gain * err_xy[0])
            release_target[1] -= float(gain * err_xy[1])
            release_target[2] = min_release_z
            pour_target = release_target.copy()
            q_seed_dump = correction_q.copy()
            continue

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
        release_target[0] -= float(err_xy[0])
        release_target[1] -= float(err_xy[1])
        release_target[2] = min_release_z
        q_seed_dump = q_dump.copy()
        last_reason = (
            f"drop correction attempt={attempt} xy_err={xy_err:.3f} "
            f"bucket_err={bucket_err:.2f}deg source_clearance={source_clearance:.3f}"
        )

    if (
        best is not None
        and unload_drop_execution_ready(best.get("drop", {}))
        and (not center_release_mode or bool(best.get("center_soft_ready", False)))
    ):
        drop = best["drop"]
        best["info"]["drop"] = drop
        best["info"]["drop_target"] = vec_list(drop_target, 3)
        best["info"]["pour_target"] = vec_list(best.get("pour_target"), 3)
        best["info"]["release_target"] = vec_list(best.get("release_target"), 3)
        best["info"]["release_source"] = drop.get("release_source", "bucket_opening_center")
        best["info"]["dump_bucket_target_deg"] = float(final_dump_deg)
        best["info"]["release_alignment_bucket_deg"] = float(release_alignment_bucket_deg)
        best["info"]["q_release_align_rad"] = vec_list(best["q"], 4)
        best["info"]["q_release_align_deg"] = q_deg_values(best["q"], wrap_swing_for_display=True)
        best["info"]["dump_bucket_err_deg"] = float(best.get("bucket_err", 0.0))
        best["info"]["ik_bucket_err_deg"] = float(best.get("ik_bucket_err", 0.0))
        best["info"]["drop_alignment_ready"] = True
        best["info"]["drop_alignment_policy"] = str(drop.get("landing_acceptance", "scatter_tolerant_high_release"))
        if log:
            info_print(
                f"[UNLOAD DUMP PLAN] {label}: accepted_inside_bin_fallback "
                f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(best.get('release_target'), 3)} "
                f"ik_pour_target={vec_list(best.get('pour_target'), 3)} "
                f"landing={drop.get('landing')} "
                f"xy_err={fmt_optional(drop.get('xy_err'))} close_xy={drop.get('close_xy')} "
                f"scatter_xy_ok={drop.get('scatter_xy_ok')} acceptance={drop.get('landing_acceptance')} "
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
            f"scatter_xy_ok={drop.get('scatter_xy_ok')} acceptance={drop.get('landing_acceptance')} "
            f"source_clearance={fmt_optional(drop.get('source_clearance'))}"
        )
        if allow_unaligned and unload_drop_execution_ready(drop) and (not center_release_mode or bool(best.get("center_soft_ready", False))):
            best["info"]["drop"] = drop
            best["info"]["drop_target"] = vec_list(drop_target, 3)
            best["info"]["pour_target"] = vec_list(best.get("pour_target"), 3)
            best["info"]["release_target"] = vec_list(best.get("release_target"), 3)
            best["info"]["release_source"] = drop.get("release_source", "bucket_opening_center")
            best["info"]["dump_bucket_target_deg"] = float(final_dump_deg)
            best["info"]["release_alignment_bucket_deg"] = float(release_alignment_bucket_deg)
            best["info"]["q_release_align_rad"] = vec_list(best["q"], 4)
            best["info"]["q_release_align_deg"] = q_deg_values(best["q"], wrap_swing_for_display=True)
            best["info"]["dump_bucket_err_deg"] = float(best.get("bucket_err", 0.0))
            best["info"]["ik_bucket_err_deg"] = float(best.get("ik_bucket_err", 0.0))
            best["info"]["drop_alignment_ready"] = True
            best["info"]["drop_alignment_policy"] = str(drop.get("landing_acceptance", "scatter_tolerant_execute_then_score"))
            best["info"]["drop_alignment_reason"] = reason
            if log:
                info_print(
                    f"[UNLOAD DUMP PLAN] {label}: using_scatter_tolerant_high_release_pose "
                    f"landing_target={vec_list(drop_target, 3)} release_target={vec_list(best.get('release_target'), 3)} "
                    f"ik_pour_target={vec_list(best.get('pour_target'), 3)} "
                    f"landing={drop.get('landing')} release={drop.get('release')} "
                    f"reason={reason} "
                    f"drift={fmt_optional(drop.get('drift_distance'))} drift_xy={drop.get('drift_xy')} "
                    f"q={q_deg_values(best['q'], wrap_swing_for_display=True)}"
                )
            draw_unload_dump_debug(
                label,
                landing_target=drop_target,
                release_target=best.get("release_target"),
                q_seed=q_seed,
                q_seed_dump=q_seed_dump,
                best_drop=drop,
                reason=reason,
            )
            return best["q"], best["info"]
    else:
        reason = last_reason
    draw_unload_dump_debug(
        label,
        landing_target=drop_target,
        release_target=initial_release_target,
        q_seed=q_seed,
        q_seed_dump=q_seed_dump,
        best_drop=best.get("drop") if isinstance(best, dict) else None,
        reason=reason,
    )
    if log:
        info_print(
            f"[UNLOAD DUMP PLAN FAIL] {label}: "
            f"landing_target={vec_list(drop_target, 3)} "
            f"initial_release_target={vec_list(initial_release_target, 3)} "
            f"final_release_target={vec_list(release_target, 3)} "
            f"final_ik_pour_target={vec_list(pour_target, 3)} "
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
    q_goal_raw = q_goal.copy()
    q_edge = swing_edge_pose_before_rebase(q_goal_raw, label=stage_name)
    if q_edge is not None:
        swing_idx = CTRL.name_to_idx["swing"]
        edge_delta = abs(float(swing_delta(q_edge[swing_idx], q_start[swing_idx])))
        edge_speed = max(0.20, float(DQ_MAX["swing"]) * max(0.05, get_speed_multiplier()))
        edge_seconds = max(0.55, min(max(3.20, float(duration)), edge_delta / edge_speed + 0.25))
        info_print(
            "[UNLOAD SWING EDGE]",
            f"stage={stage_name}",
            f"q_start={q_deg_values(q_start, wrap_swing_for_display=True)}",
            f"q_edge={q_deg_values(q_edge, wrap_swing_for_display=True)}",
            f"q_goal_raw={q_deg_values(q_goal_raw, wrap_swing_for_display=True)}",
            f"seconds={edge_seconds:.2f}",
            force_log=True,
        )
        edge_ok = await move_to_profile(
            q_edge,
            seconds=edge_seconds,
            label=f"{stage_name}_swing_edge",
            task_id=task_id,
            mode=stage_name,
            q_start_override=q_start,
        )
        if not edge_ok or (task_id is not None and not task_alive(task_id)):
            update_status(f"[UNLOAD BLOCKED] {stage_name}: swing edge move failed", force=True)
            set_execution_failure_reason(f"execution_failed/unload_swing_edge_failed:{stage_name}")
            return False
        await step_updates(2)
        q_goal, rebased = maybe_prepare_swing_rebase_for_segment(q_goal_raw, label=stage_name)
        if rebased:
            await step_updates(2)
        q_start = sync_motion_start_q(f"{stage_name}_after_swing_edge")
        q_goal = clip_command_near(q_goal_raw, reference=q_start)
    else:
        q_goal, rebased = maybe_prepare_swing_rebase_for_segment(q_goal_raw, label=stage_name)
        if rebased:
            await step_updates(2)
            q_start = sync_motion_start_q(f"{stage_name}_after_swing_rebase")
            q_goal = clip_command_near(q_goal_raw, reference=q_start)
    swing_idx = CTRL.name_to_idx["swing"]
    swing_delta_deg = abs(rad_to_deg(swing_delta(q_goal[swing_idx], q_start[swing_idx])))
    joint_delta = q_delta_abs_deg(q_goal, q_start)
    info_print(
        f"[UNLOAD ROUTE] {stage_name}: mode=direct_cached_plan "
        "contract=dig_plan_no_runtime_replan "
        f"swing_delta={swing_delta_deg:.2f}deg "
        f"joint_delta={joint_delta} "
        f"q_start={q_deg_values(q_start, wrap_swing_for_display=True)} "
        f"q_goal={q_deg_values(q_goal, wrap_swing_for_display=True)}"
    )
    fast_loaded_route = active_loaded_route_fast_exec(stage_name)
    if fast_loaded_route:
        info_print(
            "[PLAN EXEC FAST]",
            f"stage={stage_name}",
            "source=planned_loaded_route",
            "trace_redraw=False",
            "precheck=skipped",
            force_log=True,
        )
    else:
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
                "source=active_stage_diagnostic",
                f"blue_trace_source={STATE.get('trace_plan_source', '')}",
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
                detail_text = "; ".join(detail)
                strict_needs_route = strict_path_precheck_phase(stage_name)
                if strict_needs_route:
                    reason_text = f"execution_failed/path_precheck_failed:{stage_name}:{detail_text}"
                    set_execution_failure_reason(reason_text)
                    info_print(
                        "[PLAN EXEC PRECHECK FAILED]",
                        f"stage={stage_name}",
                        detail_text,
                        "hard_stop=True",
                    )
                    update_status(f"[DIG EXEC FAILED] {stage_name}: path_precheck_failed", force=True)
                    try:
                        debug_timeline_record(
                            "PATH_PRECHECK_FAIL",
                            stage=stage_name,
                            result="failed",
                            reason=reason_text,
                            q_cmd=q_goal,
                            q_real=get_real_joint_positions(),
                            data={
                                "phase_ok": bool(phase_ok),
                                "phase_reason": str(phase_reason),
                                "phase_sample": int(phase_sample),
                                "obstacle_ok": bool(obstacle_ok),
                                "obstacle_reason": str(obstacle_reason),
                                "obstacle_sample": int(obstacle_sample),
                            },
                            include_sand=True,
                        )
                    except Exception:
                        pass
                    return False
                else:
                    info_print(
                        "[PLAN EXEC DIRECT WARN]",
                        f"stage={stage_name}",
                        detail_text,
                        "sand_contact_or_curl_allowance=True",
                    )
        except Exception as e:
            if strict_path_precheck_phase(stage_name):
                reason_text = f"execution_failed/path_precheck_failed:{stage_name}:{type(e).__name__}:{e}"
                set_execution_failure_reason(reason_text)
                info_print("[PLAN EXEC PRECHECK FAILED]", f"stage={stage_name}", reason_text, "hard_stop=True")
                return False
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
    return clip_unload_dump_command(q, reference=q_reference)


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
    return unload_drop_execution_ready(drop), q_real, q_dump, drop


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


async def execute_unload_bucket_dump_motion(q_dump, stage_name, task_id=None):
    ready, reason, _detail = await wait_for_articulation_action_ready(
        f"{stage_name}_bucket_dump_start",
        min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=ACTION_READY_STAGE_MAX_WAIT_FRAMES,
        record_failure=True,
    )
    if not ready:
        update_status(f"[UNLOAD BLOCKED] {stage_name}: action_channel_not_ready; {reason}", force=True)
        set_execution_failure_reason(f"execution_failed/unload_dump_action_not_ready:{stage_name}:{reason}")
        return False

    q0 = sync_motion_start_q(f"{stage_name}_bucket_dump")
    q1 = clip_unload_dump_command(q_dump, reference=q0)
    bucket_idx = CTRL.name_to_idx.get("bucket", 3)
    target_bucket = float(q1[bucket_idx])
    step_rad = deg_to_rad(max(2.0, float(UNLOAD_DUMP_STEP_DEG)))
    max_steps = max(8, int(math.ceil(abs(target_bucket - float(q0[bucket_idx])) / max(1e-4, step_rad))) + 4)
    wait_frames = max(3, int(float(UNLOAD_DUMP_STEP_SECONDS) * 60))
    start_metrics = record_phase_metrics("before_dump_direct")
    start_bucket_count = int(start_metrics.get("bucket_from_pile_count", 0)) if isinstance(start_metrics, dict) else 0
    start_bin_count = int(start_metrics.get("bin_from_pile_count", 0)) if isinstance(start_metrics, dict) else 0
    if str(STATE.get("active_task_name", "")) == "loaded_unload_route_test":
        bucket_particle_diagnostic("before_dump_direct")
    info_print(
        "[UNLOAD DUMP DIRECT START]",
        f"stage={stage_name}",
        "policy=progressive_open_until_release",
        f"steps={max_steps}",
        f"step_deg={rad_to_deg(step_rad):.2f}",
        f"bucket_start={rad_to_deg(float(q0[bucket_idx])):.2f}deg",
        f"bucket_target={rad_to_deg(target_bucket):.2f}deg",
        f"bucket_start_particles={start_bucket_count}",
        f"bin_start_particles={start_bin_count}",
        force_log=True,
    )

    q_final = q0.copy()
    last_metrics = start_metrics
    for i in range(max_steps):
        if motion_cancel_requested(task_id):
            update_status(f"[MOVE STOPPED] {stage_name}_dump_pose", force=True)
            return False
        try:
            q_real = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
        except Exception:
            q_real = CTRL.q_cmd.copy()
        real_bucket = float(q_real[bucket_idx])
        if real_bucket >= target_bucket - deg_to_rad(1.5):
            q_final = q_real.copy()
            break
        q = q_real.copy()
        q[bucket_idx] = min(target_bucket, real_bucket + step_rad)
        q = clip_unload_dump_command(q, reference=CTRL.q_cmd)
        q_final = q.copy()
        ok, send_reason = CTRL.send_action(q, mode="unload_dump")
        if not ok:
            update_status(f"[UNLOAD BLOCKED] {stage_name}: bucket dump action failed; {send_reason}", force=True)
            set_execution_failure_reason(f"execution_failed/unload_dump_action_failed:{stage_name}:{send_reason}")
            return False
        await step_updates(wait_frames)
        last_metrics = record_phase_metrics("during_dump_direct")
        bucket_now = int(last_metrics.get("bucket_from_pile_count", 0)) if isinstance(last_metrics, dict) else 0
        bin_now = int(last_metrics.get("bin_from_pile_count", 0)) if isinstance(last_metrics, dict) else 0
        info_print(
            "[UNLOAD DUMP DIRECT SAMPLE]",
            f"stage={stage_name}",
            f"i={i + 1}/{max_steps}",
            f"bucket_cmd={rad_to_deg(float(q[bucket_idx])):.2f}deg",
            f"bucket_real={rad_to_deg(real_bucket):.2f}deg",
            f"bucket_delta={bucket_now - start_bucket_count}",
            f"bin_delta={bin_now - start_bin_count}",
            force_log=True,
        )
        stop_bucket_count = max(0, int(start_bucket_count * float(UNLOAD_DUMP_STOP_BUCKET_FRACTION)))
        if (
            bin_now - start_bin_count >= max(QUALITY_MIN_DUMP_PARTICLES, 8)
            and bucket_now <= stop_bucket_count
            and float(q[bucket_idx]) >= target_bucket - deg_to_rad(8.0)
        ):
            q_final = q_real.copy()
            break

    STATE["dataset_current_q_goal"] = q_final.copy()
    try:
        q_hold = q_real_near_command(get_real_joint_positions(), q_final)
        CTRL.send_action(q_hold, mode="unload_dump_hold_real")
        q_final = q_hold.copy()
    except Exception:
        pass
    await step_updates(4)
    try:
        q_real = q_real_near_command(get_real_joint_positions(), q_final)
        err_deg = abs(rad_to_deg(float(q_final[bucket_idx] - q_real[bucket_idx])))
        end_metrics = record_phase_metrics("after_dump_direct")
        bucket_end = int(end_metrics.get("bucket_from_pile_count", 0)) if isinstance(end_metrics, dict) else 0
        bin_end = int(end_metrics.get("bin_from_pile_count", 0)) if isinstance(end_metrics, dict) else 0
        if str(STATE.get("active_task_name", "")) == "loaded_unload_route_test":
            bucket_particle_diagnostic("after_dump_direct")
        info_print(
            "[UNLOAD DUMP DIRECT DONE]",
            f"stage={stage_name}",
            f"bucket_cmd={rad_to_deg(float(q_final[bucket_idx])):.2f}deg",
            f"bucket_real={rad_to_deg(float(q_real[bucket_idx])):.2f}deg",
            f"bucket_err={err_deg:.2f}deg",
            f"bucket_delta={bucket_end - start_bucket_count}",
            f"bin_delta={bin_end - start_bin_count}",
            force_log=True,
        )
    except Exception as e:
        info_print("[WARN] [UNLOAD DUMP DIRECT DONE] cannot read real bucket:", type(e).__name__, e)
    return True


async def dump_bucket_at_target(stage_name, task_id=None, planned_q_dump=None, planned_q_release_align=None):
    dump_deg = unload_dump_target_deg()
    record_phase_metrics("before_dump")

    q_dump = None
    q_real = None
    q_release_align = None
    ready = False
    dump_source = "dynamic_bucket_only"

    if planned_q_dump is not None:
        try:
            q_seed = CTRL.q_cmd.copy()
            q_dump = clip_command_near(np.array(planned_q_dump, dtype=np.float32).reshape(-1)[:4].copy(), reference=q_seed)
            if planned_q_release_align is not None:
                q_release_align = clip_command_near(
                    np.array(planned_q_release_align, dtype=np.float32).reshape(-1)[:4].copy(),
                    reference=q_seed,
                )
            try:
                q_real = current_real_q_near(q_seed)
            except Exception:
                q_real = q_seed.copy()
            actual_gate = log_actual_unload_position("before_planned_dump")
            drop = log_unload_drop("planned_cached_dump_landing", q=q_dump, reference_q=q_seed)
            release_drop = None
            if q_release_align is not None:
                release_drop = log_unload_drop("planned_release_align_landing", q=q_release_align, reference_q=q_seed)
            log_unload_alignment("planned_cached_dump_pour", q=q_dump, effector="pour", reference_q=q_seed)
            readiness_drop = release_drop if isinstance(release_drop, dict) else drop
            ready = bool(actual_gate.get("ok", False) and unload_drop_execution_ready(readiness_drop))
            dump_source = "cached_planned_q_dump"
            info_print(
                f"[UNLOAD DUMP PLAN MATCH] {stage_name}: "
                f"ready={ready} source={dump_source} "
                f"q_dump={q_deg_values(q_dump, wrap_swing_for_display=True)} "
                f"q_release_align={q_deg_values(q_release_align, wrap_swing_for_display=True) if q_release_align is not None else None} "
                f"drop_xy_err={fmt_optional(drop.get('xy_err'))} "
                f"release_xy_err={fmt_optional((release_drop or {}).get('release_xy_err'))} "
                f"inside_xy={drop.get('inside_xy')} close_xy={drop.get('close_xy')} "
                f"scatter_xy_ok={drop.get('scatter_xy_ok')} acceptance={drop.get('landing_acceptance')} "
                f"readiness_acceptance={(readiness_drop or {}).get('landing_acceptance')} "
                f"actual_gate={actual_gate.get('ok')}"
            )
            if not ready:
                update_status(
                    f"[UNLOAD DIAG] {stage_name}: cached planned dump predicted off target; executing and scoring actual particles; "
                    f"drop_xy_err={fmt_optional(drop.get('xy_err'))} inside_xy={drop.get('inside_xy')} "
                    f"close_xy={drop.get('close_xy')} scatter_xy_ok={drop.get('scatter_xy_ok')} "
                    f"acceptance={drop.get('landing_acceptance')} actual_gate={actual_gate.get('ok')}",
                    force=True,
                )
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
        q_release_align = np.array(q_plan, dtype=np.float32).reshape(-1)[:4].copy()
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

        q_release_align = bucket_only_dump_pose(q_pre_dump, unload_release_alignment_bucket_deg(dump_deg))
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

    if q_release_align is not None:
        log_unload_drop("planned_release_flow_landing", q=q_release_align, reference_q=q_real)
    log_unload_alignment("planned_dump_pour", q=q_dump, effector="pour", reference_q=q_real)
    drop = log_unload_drop("planned_dump_landing", q=q_dump, reference_q=q_real)
    readiness_drop = (
        unload_drop_report(q=q_release_align, reference_q=q_real)
        if q_release_align is not None
        else drop
    )
    if not unload_drop_execution_ready(readiness_drop):
        update_status(
            f"[UNLOAD DIAG] {stage_name}: release-flow opening center is not aligned; executing and scoring actual particles; "
            f"release_xy_err={fmt_optional(readiness_drop.get('release_xy_err'))} "
            f"xy_err={fmt_optional(readiness_drop.get('xy_err'))} close_xy={readiness_drop.get('close_xy')} "
            f"scatter_xy_ok={readiness_drop.get('scatter_xy_ok')} acceptance={readiness_drop.get('landing_acceptance')}",
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

    success = await execute_unload_bucket_dump_motion(q_dump, stage_name, task_id=task_id)
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

    if block_invalid_dig_plan_contract("debug_step"):
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

    record_stage_audit(
        stage_name,
        step_index,
        "start",
        q_goal=q_goal,
        duration=duration,
        include_sand=is_sand_contact_phase(stage_name) or "unload" in stage_name,
    )
    log_phase_ground("[DIG GUARD CURRENT]", stage_name)
    try:
        guard_reference_q = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
    except Exception:
        guard_reference_q = CTRL.q_cmd.copy()
    ok, reason, report = log_predicted_phase_ground(
        "[DIG GUARD TARGET]",
        stage_name,
        q_goal,
        reference_q=guard_reference_q,
    )
    if not ok:
        msg = format_ground_report(stage_name, report, reason)
        update_status(
            f"[DIG PREDICTED CONTACT] {stage_name} target: {msg}; executing with live freeze guard",
            force=True,
        )
        record_stage_audit(
            stage_name,
            step_index,
            "predicted_contact",
            reason=msg,
            q_goal=q_goal,
            duration=duration,
            data={"predicted_ground_ok": False},
            include_sand=is_sand_contact_phase(stage_name),
        )

    update_status(f"[DIG STEP {step_index + 1}/{len(seq)}] {stage_name}", force=True)
    if "unload" in stage_name:
        unload_detail = planned_unload_stage_detail(stage_index=step_index, stage_name=stage_name)
        planned_q_dump = None if not unload_detail else unload_detail.get("q_dump")
        success = await ik_movement.move_unload_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
        if not success or not task_alive(task_id):
            update_status(execution_failure_status_text(stage_name), force=True)
            record_stage_audit(
                stage_name,
                step_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "unload_stage_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            return
        if int(STATE.get("dig_plan_step_index", 0)) <= step_index:
            STATE["dig_plan_step_index"] = step_index + 1
        if not await dump_bucket_at_target(stage_name, task_id=task_id, planned_q_dump=planned_q_dump):
            record_stage_audit(
                stage_name,
                step_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "dump_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            return
        log_phase_ground("[DIG GUARD AFTER]", stage_name)
        record_stage_audit(
            stage_name,
            step_index,
            "done",
            q_goal=q_goal,
            duration=duration,
            include_sand=True,
        )
        update_status(f"[DIG STEP DONE] {stage_name}", force=True)
        return

    success = await ik_movement.move_planned_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
    if not success or not task_alive(task_id):
        update_status(execution_failure_status_text(stage_name), force=True)
        record_stage_audit(
            stage_name,
            step_index,
            "failed",
            reason=str(STATE.get("last_execution_failure_reason", "") or "stage_failed"),
            q_goal=q_goal,
            duration=duration,
            include_sand=is_sand_contact_phase(stage_name),
        )
        return

    if int(STATE.get("dig_plan_step_index", 0)) <= step_index:
        STATE["dig_plan_step_index"] = step_index + 1

    if "unload" in stage_name:
        if not await dump_bucket_at_target(stage_name, task_id=task_id):
            record_stage_audit(
                stage_name,
                step_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "dump_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            return
    else:
        notify_sand_site_step_done(stage_name)
        if stage_name == "curl_to_hold_material":
            record_phase_metrics("after_dig")
        elif stage_name == "lift_carry":
            lift_metrics = record_phase_metrics("after_lift")
            lift_gate = post_lift_material_gate_report(current_metrics=lift_metrics, q_pose=get_real_joint_positions())
            debug_timeline_record(
                "LIFT_MATERIAL_GATE",
                stage=stage_name,
                result="ok" if bool(lift_gate.get("ok", False)) else "failed",
                reason=str(lift_gate.get("reason", "")),
                q_cmd=CTRL.q_cmd.copy(),
                q_real=get_real_joint_positions(),
                data=lift_gate,
                include_sand=True,
            )
            if not bool(lift_gate.get("ok", False)):
                reason_text = f"quality_rejected/lift_lost_material:{lift_gate.get('reason', '')}"
                set_execution_failure_reason(reason_text)
                record_stage_audit(
                    stage_name,
                    step_index,
                    "failed",
                    reason=reason_text,
                    q_goal=q_goal,
                    duration=duration,
                    data={"lift_material_gate": lift_gate},
                    include_sand=True,
                )
                return

    log_phase_ground("[DIG GUARD AFTER]", stage_name)
    record_stage_audit(
        stage_name,
        step_index,
        "done",
        q_goal=q_goal,
        duration=duration,
        include_sand=is_sand_contact_phase(stage_name) or stage_name in ("curl_to_hold_material", "lift_carry"),
    )
    if dig_plan_semantic_phase_name(stage_name) == "pull_exit_cut":
        if not append_staged_post_dig_secure_plan(task_label="debug_step"):
            update_status(execution_failure_status_text(stage_name), force=True)
            record_stage_audit(
                stage_name,
                step_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "staged_post_dig_secure_plan_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            return
    if should_append_staged_post_secure_load_after_stage(stage_name):
        if not append_staged_post_secure_load_plan(task_label="debug_step"):
            update_status(execution_failure_status_text(stage_name), force=True)
            record_stage_audit(
                stage_name,
                step_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "staged_post_secure_load_plan_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            return
    update_status(f"[DIG STEP DONE] {stage_name}", force=True)


async def execute_dig_target_ball(rebuild_plan=True, task_name="dig_target_ball", return_home=True):
    STATE["follow"] = False
    if rebuild_plan:
        seq = await build_dig_plan_from_current_target_task(force_status=True)
    else:
        seq = STATE.get("dig_plan_sequence", None)
        if seq is None:
            seq = await build_dig_plan_from_current_target_task(force_status=True)
    if not seq:
        return False

    if block_invalid_dig_plan_contract(task_name):
        return False

    task_id = start_task(task_name)
    loaded_route_diag = str(task_name) == "loaded_unload_route_test"

    stage_index = 0
    while stage_index < len(seq):
        stage_name, q_goal, duration = seq[stage_index]
        STATE["active_plan_stage_index"] = int(stage_index)
        if not task_alive(task_id):
            return False

        record_stage_audit(
            stage_name,
            stage_index,
            "start",
            q_goal=q_goal,
            duration=duration,
            include_sand=is_sand_contact_phase(stage_name) or "unload" in stage_name,
        )
        log_phase_ground("[DIG GUARD CURRENT]", stage_name)
        try:
            guard_reference_q = q_real_near_command(get_real_joint_positions(), CTRL.q_cmd)
        except Exception:
            guard_reference_q = CTRL.q_cmd.copy()
        ok, reason, report = log_predicted_phase_ground(
            "[DIG GUARD TARGET]",
            stage_name,
            q_goal,
            reference_q=guard_reference_q,
        )
        if not ok:
            msg = format_ground_report(stage_name, report, reason)
            update_status(
                f"[DIG PREDICTED CONTACT] {stage_name} target: {msg}; executing with live freeze guard",
                force=True,
            )
            record_stage_audit(
                stage_name,
                stage_index,
                "predicted_contact",
                reason=msg,
                q_goal=q_goal,
                duration=duration,
                data={"predicted_ground_ok": False},
                include_sand=is_sand_contact_phase(stage_name),
            )

        update_status(f"[DIG] {stage_name}", force=True)
        if (
            loaded_route_diag
            and bool(LOADED_ROUTE_STAGE_PARTICLE_DIAGNOSTICS)
            and mode_requires_loaded_carry_bucket(stage_name, stage_name)
        ):
            bucket_particle_diagnostic(f"before_{stage_name}")
        if "unload" in stage_name:
            unload_detail = planned_unload_stage_detail(stage_index=stage_index, stage_name=stage_name)
            planned_q_dump = None if not unload_detail else unload_detail.get("q_dump")
            planned_q_release_align = None if not unload_detail else unload_detail.get("q_release_align")
            success = await ik_movement.move_unload_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
            if not success or not task_alive(task_id):
                update_status(execution_failure_status_text(stage_name), force=True)
                record_stage_audit(
                    stage_name,
                    stage_index,
                    "failed",
                    reason=str(STATE.get("last_execution_failure_reason", "") or "unload_stage_failed"),
                    q_goal=q_goal,
                    duration=duration,
                    include_sand=True,
                )
                return False
            STATE["dig_plan_step_index"] = int(STATE.get("dig_plan_step_index", 0)) + 1
            if loaded_route_diag and bool(LOADED_ROUTE_STAGE_PARTICLE_DIAGNOSTICS):
                bucket_particle_diagnostic(f"after_{stage_name}_arrival")
            if not await dump_bucket_at_target(
                stage_name,
                task_id=task_id,
                planned_q_dump=planned_q_dump,
                planned_q_release_align=planned_q_release_align,
            ):
                record_stage_audit(
                    stage_name,
                    stage_index,
                    "failed",
                    reason=str(STATE.get("last_execution_failure_reason", "") or "dump_failed"),
                    q_goal=q_goal,
                    duration=duration,
                    include_sand=True,
                )
                return False
            log_phase_ground("[DIG GUARD AFTER]", stage_name)
            record_stage_audit(
                stage_name,
                stage_index,
                "done",
                q_goal=q_goal,
                duration=duration,
                include_sand=True,
            )
            stage_index += 1
            continue

        success = await ik_movement.move_planned_stage(runtime_module(), stage_name, q_goal, duration, task_id=task_id)
        if not success or not task_alive(task_id):
            update_status(execution_failure_status_text(stage_name), force=True)
            record_stage_audit(
                stage_name,
                stage_index,
                "failed",
                reason=str(STATE.get("last_execution_failure_reason", "") or "stage_failed"),
                q_goal=q_goal,
                duration=duration,
                include_sand=is_sand_contact_phase(stage_name),
            )
            return False
        if (
            loaded_route_diag
            and bool(LOADED_ROUTE_STAGE_PARTICLE_DIAGNOSTICS)
            and mode_requires_loaded_carry_bucket(stage_name, stage_name)
        ):
            bucket_particle_diagnostic(f"after_{stage_name}")
        STATE["dig_plan_step_index"] = int(STATE.get("dig_plan_step_index", 0)) + 1
        if "unload" in stage_name:
            unload_detail = planned_unload_stage_detail(stage_index=stage_index, stage_name=stage_name)
            planned_q_dump = None if not unload_detail else unload_detail.get("q_dump")
            planned_q_release_align = None if not unload_detail else unload_detail.get("q_release_align")
            if not await dump_bucket_at_target(
                stage_name,
                task_id=task_id,
                planned_q_dump=planned_q_dump,
                planned_q_release_align=planned_q_release_align,
            ):
                record_stage_audit(
                    stage_name,
                    stage_index,
                    "failed",
                    reason=str(STATE.get("last_execution_failure_reason", "") or "dump_failed"),
                    q_goal=q_goal,
                    duration=duration,
                    include_sand=True,
                )
                return False
        else:
            notify_sand_site_step_done(stage_name)
            if stage_name == "curl_to_hold_material":
                record_phase_metrics("after_dig")
            elif stage_name == "lift_carry":
                lift_metrics = record_phase_metrics("after_lift")
                lift_gate = post_lift_material_gate_report(current_metrics=lift_metrics, q_pose=get_real_joint_positions())
                debug_timeline_record(
                    "LIFT_MATERIAL_GATE",
                    stage=stage_name,
                    result="ok" if bool(lift_gate.get("ok", False)) else "failed",
                    reason=str(lift_gate.get("reason", "")),
                    q_cmd=CTRL.q_cmd.copy(),
                    q_real=get_real_joint_positions(),
                    data=lift_gate,
                    include_sand=True,
                )
                if not bool(lift_gate.get("ok", False)):
                    reason_text = f"quality_rejected/lift_lost_material:{lift_gate.get('reason', '')}"
                    set_execution_failure_reason(reason_text)
                    record_stage_audit(
                        stage_name,
                        stage_index,
                        "failed",
                        reason=reason_text,
                        q_goal=q_goal,
                        duration=duration,
                        data={"lift_material_gate": lift_gate},
                        include_sand=True,
                    )
                    return False
        log_phase_ground("[DIG GUARD AFTER]", stage_name)
        record_stage_audit(
            stage_name,
            stage_index,
            "done",
            q_goal=q_goal,
            duration=duration,
            include_sand=is_sand_contact_phase(stage_name) or stage_name in ("curl_to_hold_material", "lift_carry"),
        )
        if dig_plan_semantic_phase_name(stage_name) == "pull_exit_cut":
            if not append_staged_post_dig_secure_plan(task_label=task_name):
                update_status(execution_failure_status_text(stage_name), force=True)
                record_stage_audit(
                    stage_name,
                    stage_index,
                    "failed",
                    reason=str(STATE.get("last_execution_failure_reason", "") or "staged_post_dig_secure_plan_failed"),
                    q_goal=q_goal,
                    duration=duration,
                    include_sand=True,
                )
                return False
        if should_append_staged_post_secure_load_after_stage(stage_name):
            if not append_staged_post_secure_load_plan(task_label=task_name):
                update_status(execution_failure_status_text(stage_name), force=True)
                record_stage_audit(
                    stage_name,
                    stage_index,
                    "failed",
                    reason=str(STATE.get("last_execution_failure_reason", "") or "staged_post_secure_load_plan_failed"),
                    q_goal=q_goal,
                    duration=duration,
                    include_sand=True,
                )
                return False
        stage_index += 1

    if return_home and task_alive(task_id):
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
    if return_home:
        update_status("[DIG FINISHED] bucket curled, lifted, unloaded, and homed.", force=True)
    else:
        update_status("[LOADED ROUTE TEST FINISHED] unloaded; holding current pose.", force=True)
    return True


async def calibrate_ik():
    global IK_MODEL

    update_status("Calibrating planar IK...", force=True)
    IK_MODEL = None
    STATE["ik_calibration_valid"] = False
    STATE["ik_calibration_report"] = {"status": "running"}

    async def fail(reason, data=None):
        IK_MODEL_STATE = {"status": "failed", "reason": str(reason)}
        if isinstance(data, dict):
            IK_MODEL_STATE.update(data)
        STATE["ik_calibration_valid"] = False
        STATE["ik_calibration_report"] = IK_MODEL_STATE
        update_status(f"[IK CALIBRATE FAILED] {reason}", force=True)
        debug_timeline_record("IK_CALIBRATE", result="failed", reason=str(reason), data=IK_MODEL_STATE, include_sand=False)
        return False

    safe_ok = await set_joint_pose_direct_and_settle(
        safe_home_q(),
        label="ik_calibration_safe_pose",
        mode="calibrate_safe_pose",
        settle_frames=IK_CALIBRATION_SAFE_SETTLE_FRAMES,
    )
    if not safe_ok:
        return await fail("safe pose not reached")

    update_q_cmd_from_real()
    q0 = CTRL.q_cmd.copy()
    base_chain = current_planar_chain()
    if base_chain is None:
        return await fail("cannot read current chain")

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
        reached, reach_detail, _blocked, _swing_err, max_err, _q_real = motion_reach_report(q_probe)
        if (not reached) and float(max_err) > IK_CALIBRATION_REACH_TOL_DEG:
            return await fail(
                f"{joint_name} probe not reached",
                {
                    "joint": joint_name,
                    "max_err_deg": round(float(max_err), 3),
                    "detail": reach_detail,
                },
            )

        chain = current_planar_chain()
        if chain is None:
            return await fail(f"{joint_name} sign probe failed")
        else:
            angles = chain["angles"]
            if angle_idx == 0:
                delta = wrap_angle(angles[0] - base_angles[0])
            elif angle_idx == 1:
                delta = wrap_angle((angles[1] - angles[0]) - (base_angles[1] - base_angles[0]))
            else:
                delta = wrap_angle((angles[2] - angles[1]) - (base_angles[2] - base_angles[1]))

            if abs(float(delta)) < IK_CALIBRATION_MIN_DELTA_RAD:
                return await fail(
                    f"{joint_name} probe response too small",
                    {
                        "joint": joint_name,
                        "delta_rad": float(delta),
                        "min_delta_rad": float(IK_CALIBRATION_MIN_DELTA_RAD),
                        "requested_step_rad": float(h),
                    },
                )
            sign = 1.0 if delta >= 0.0 else -1.0
            signs.append(sign)
            info_print(f"[IK CAL] {joint_name}: delta={delta:.6f} rad for +{h:.3f}, sign={sign:+.0f}")

        await move_to(q0, 0.55, mode="calibrate")
        await step_updates(24)
        restored, restore_detail, _blocked, _swing_err, restore_max_err, _q_real = motion_reach_report(q0)
        if (not restored) and float(restore_max_err) > IK_CALIBRATION_RESTORE_TOL_DEG:
            return await fail(
                f"{joint_name} restore pose not reached",
                {
                    "joint": joint_name,
                    "max_err_deg": round(float(restore_max_err), 3),
                    "detail": restore_detail,
                },
            )

    await move_to(q0, 0.65, mode="calibrate")
    await step_updates(30)
    update_q_cmd_from_real()
    q_rest = CTRL.q_cmd.copy()
    chain0 = current_planar_chain("mid")
    if chain0 is None:
        return await fail("cannot restore chain")

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
        return await fail("missing mid/tip effector geometry")

    IK_MODEL = {
        "signs": signs,
        "effectors": effectors,
        "calibrated": True,
        "valid": False,
    }
    sanity_ok, sanity_reason, sanity_errors = validate_ik_model_against_current_pose(IK_MODEL, q_rest)
    if not sanity_ok:
        IK_MODEL = None
        return await fail(
            sanity_reason,
            {
                "effector_errors_m": {
                    k: None if v is None else round(float(v), 4)
                    for k, v in sanity_errors.items()
                }
            },
        )
    IK_MODEL["valid"] = True
    IK_MODEL["validation"] = {
        "sanity": sanity_reason,
        "effector_errors_m": {
            k: None if v is None else round(float(v), 4)
            for k, v in sanity_errors.items()
        },
    }

    CTRL.q_cmd = q_rest.copy()
    CTRL.send_action(q_rest, mode="calibrate_restore")

    info_print("[IK CAL] signs:", [float(x) for x in IK_MODEL["signs"]])
    for end_effector, part in IK_MODEL["effectors"].items():
        info_print(f"[IK CAL] {end_effector}.lengths:", [float(x) for x in part["lengths"]])
        info_print(f"[IK CAL] {end_effector}.offsets:", [float(x) for x in part["offsets"]])
    STATE["ik_calibration_valid"] = True
    STATE["ik_calibration_report"] = {
        "status": "ok",
        "signs": [float(x) for x in IK_MODEL["signs"]],
        "validation": IK_MODEL.get("validation", {}),
    }
    debug_timeline_record("IK_CALIBRATE", result="ok", reason="ok", data=STATE["ik_calibration_report"], include_sand=False)
    update_status("Planar IK calibrated.", force=True)
    return True


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

    def run_loaded_unload_route_test_button():
        register_async_task("motion", execute_loaded_unload_route_test_from_current(), replace=True)

    def use_selected_unload_mesh_from_ui():
        set_manual_unload_from_selected_mesh()

    def use_selected_unload_point_from_ui():
        set_manual_unload_from_selected_point()

    def clear_unload_override_from_ui():
        clear_manual_unload_override()

    WINDOW = ui.Window("Excavator Slider Control v3", width=620, height=760)

    with WINDOW.frame:
        with ui.VStack(spacing=6):
            with ui.VStack(height=44, spacing=2):
                ui.Label("Excavator Slider Control v3", height=20)
                STATUS_LABEL = ui.Label(ui_short_text("Ready", 72), width=590, height=20)

            with ui.ScrollingFrame(height=ui.Fraction(1)):
                with ui.VStack(spacing=8):
                    with ui.HStack(spacing=6, height=26):
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

                        with ui.HStack(spacing=6, height=24):
                            ui.Label(name, width=80)
                            ui.FloatSlider(model=model, min=lo, max=hi, width=360)
                            ui.FloatField(model=model, width=90)

                        model.add_value_changed_fn(on_manual_joint_slider_changed)

                    with ui.HStack(spacing=6, height=26):
                        ui.Label("Manual joints", width=170)
                        ui.Button("Home", width=82, clicked_fn=home)
                        ui.Button("Print State", width=104, clicked_fn=print_state)
                        ui.Button("Render Mode", width=112, clicked_fn=toggle_render_mode_from_ui)

                    ui.Separator()
                    ui.Label("Auto Dataset (primary pipeline)")
                    with ui.HStack(spacing=6, height=26):
                        ui.Label("Target trainable", width=135)
                        ui.Label("Count", width=50)
                        ui.IntField(model=auto_count_model, width=70)
                        ui.Button("Start", width=62, clicked_fn=start_auto_collect_from_ui)
                        ui.Button("Stop", width=62, clicked_fn=stop_auto_collect)
                        ui.Button("Dir", width=52, clicked_fn=open_auto_collect_dir_from_ui)
                        ui.Button("Replay", width=74, clicked_fn=request_replay_latest_record)

                    with ui.HStack(spacing=6, height=24):
                        speed_model = ui.SimpleFloatModel(float(STATE.get("speed_multiplier", 1.0)))
                        ui.Label("Safe speed x", width=96)
                        ui.FloatSlider(model=speed_model, min=SPEED_MULTIPLIER_MIN, max=SPEED_MULTIPLIER_MAX, width=360)
                        ui.FloatField(model=speed_model, width=90)

                        def update_speed_multiplier(model=None):
                            sm = safe_float(speed_model.as_float, 1.0)
                            sm = max(SPEED_MULTIPLIER_MIN, min(SPEED_MULTIPLIER_MAX, sm))
                            STATE["speed_multiplier"] = sm
                            apply_speed_to_physx_joint_limits()
                            update_status(f"Safe speed multiplier = {sm:.2f}", force=True)

                        speed_model.add_value_changed_fn(update_speed_multiplier)

                    ui.Separator()
                    with ui.HStack(spacing=10):
                        with ui.VStack(width=292, spacing=4):
                            ui.Label("Target ball position")
                            p = get_target_pos()
                            for axis, val in [("x", p[0]), ("y", p[1]), ("z", p[2])]:
                                model = ui.SimpleFloatModel(float(val))
                                TARGET_MODELS[axis] = model
                                with ui.HStack(spacing=4, height=24):
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
                                with ui.HStack(spacing=4, height=24):
                                    ui.Label(axis.upper(), width=20)
                                    ui.FloatSlider(model=model, min=-15.0, max=15.0, width=170)
                                    ui.FloatField(model=model, width=70)
                            for key, label, val in [("z_min", "Z Min", unload_z_min), ("z_max", "Z Max", unload_z_max)]:
                                model = ui.SimpleFloatModel(float(val))
                                UNLOAD_MODELS[key] = model
                                with ui.HStack(spacing=4, height=24):
                                    ui.Label(label, width=48)
                                    ui.FloatSlider(model=model, min=-2.0, max=8.0, width=142)
                                    ui.FloatField(model=model, width=70)
                            radius_model = ui.SimpleFloatModel(float(manual_unload_radius()))
                            UNLOAD_MODELS["r"] = radius_model
                            with ui.HStack(spacing=4, height=24):
                                ui.Label("R", width=20)
                                ui.FloatSlider(model=radius_model, min=0.05, max=4.0, width=170)
                                ui.FloatField(model=radius_model, width=70)
                            shrink_model = ui.SimpleFloatModel(float(manual_unload_mesh_shrink_d()))
                            UNLOAD_MODELS["d"] = shrink_model
                            with ui.HStack(spacing=4, height=24):
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
                    with ui.HStack(spacing=6, height=26):
                        ui.Button("Unload Selected Mesh", width=166, clicked_fn=use_selected_unload_mesh_from_ui)
                        ui.Button("Unload Selected Point", width=166, clicked_fn=use_selected_unload_point_from_ui)
                        ui.Button("Clear Unload Override", width=160, clicked_fn=clear_unload_override_from_ui)

                    ui.Separator()
                    ui.Label("Debug Planner / Trace")

                    with ui.HStack(spacing=6, height=26):
                        ui.Button("Calibrate IK", width=112, clicked_fn=request_calib)
                        ui.Button("IK One Step", width=104, clicked_fn=ik_one_step)
                        ui.Button("Follow ON/OFF", width=118, clicked_fn=toggle_follow)
                    with ui.HStack(spacing=6, height=26):
                        ui.Button("Trace target", width=116, clicked_fn=trace_mode_1)
                        ui.Button("Trace path", width=108, clicked_fn=trace_mode_2)
                        ui.Button("Loaded Route Test", width=156, clicked_fn=run_loaded_unload_route_test_button)

                    with ui.HStack(spacing=6, height=26):
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

                    with ui.HStack(spacing=6, height=26):
                        ui.Button("1 Pre", width=82, clicked_fn=lambda: run_dig_step_button(0))
                        ui.Button("2 Contact", width=102, clicked_fn=lambda: run_dig_step_button(1))
                        ui.Button("3 Insert", width=92, clicked_fn=lambda: run_dig_step_button(2))
                        ui.Button("4 Mid Cut", width=102, clicked_fn=lambda: run_dig_step_button(3))

                    with ui.HStack(spacing=6, height=26):
                        ui.Button("5 Exit Cut", width=108, clicked_fn=lambda: run_dig_step_button(4))
                        ui.Button("6 Curl", width=82, clicked_fn=lambda: run_dig_step_button(5))
                        ui.Button("7 Lift", width=82, clicked_fn=lambda: run_dig_step_button(6))
                        ui.Button("8 Unload", width=98, clicked_fn=lambda: run_dig_step_button(7))

                    with ui.HStack(spacing=6, height=26):
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
