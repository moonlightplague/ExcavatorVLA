"""Shared training/deployment contract for excavator VLA observations."""

import math
from typing import Mapping, Optional, Sequence


BASE_STATE_NAMES_14D = [
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

CANONICAL_PHASE_NAMES = [
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "curl_to_hold_material",
    "pull_exit_cut",
    "secure_load",
    "lift_carry",
    "loaded_transit",
    "unload_to_bin",
]

LEGACY_STATE_NAMES_28D_V3 = BASE_STATE_NAMES_14D + [
    "swing_velocity",
    "boom_velocity",
    "arm_velocity",
    "bucket_velocity",
    "phase_index",
    "dig_target_local_x",
    "dig_target_local_y",
    "dig_target_local_z",
    "unload_landing_local_x",
    "unload_landing_local_y",
    "unload_landing_local_z",
    "truck_heading_relative_sin",
    "truck_heading_relative_cos",
    "bucket_load_rate",
]

STATE_NAMES_28D = [
    "swing_position",
    "boom_position",
    "arm_position",
    "bucket_position",
    "swing_velocity",
    "boom_velocity",
    "arm_velocity",
    "bucket_velocity",
    "swing_tracking_error",
    "boom_tracking_error",
    "arm_tracking_error",
    "bucket_tracking_error",
    "previous_swing_action",
    "previous_boom_action",
    "previous_arm_action",
    "previous_bucket_action",
    "bucket_pour_axis_forward",
    "bucket_pour_axis_up",
    "dig_target_from_tip_forward",
    "dig_target_from_tip_left",
    "dig_target_from_tip_up",
    "unload_from_load_forward",
    "unload_from_load_left",
    "unload_from_load_up",
    "truck_heading_relative_sin",
    "truck_heading_relative_cos",
    "bucket_fill_fraction",
    "bucket_fill_rate_fraction_per_s",
]

EFFORT_NAMES_4D = [
    "swing_measured_effort",
    "boom_measured_effort",
    "arm_measured_effort",
    "bucket_measured_effort",
]

ACTION_NAMES_4D = [
    "swing_cmd_velocity",
    "boom_cmd_velocity",
    "arm_cmd_velocity",
    "bucket_cmd_velocity",
]

SCHEMA_VERSION = "excavator_state_v4_28d_plus_4effort_categorical_phase10"
BUCKET_FILL_CAPACITY_PARTICLES = 6400.0
BUCKET_FILL_RATE_TAU_SECONDS = 0.40
BUCKET_FILL_FRACTION_MAX = 1.50
BUCKET_FILL_RATE_ABS_MAX = 5.0


class ObservationContractError(ValueError):
    """Raised when a live deployment observation cannot satisfy the model schema."""


def _finite_vector(value: object, size: int, label: str) -> list:
    try:
        vector = [float(item) for item in value]
    except Exception as exc:
        raise ObservationContractError(f"{label} is not a numeric vector: {exc}") from exc
    if len(vector) != int(size):
        raise ObservationContractError(f"{label} must contain {size} values, got {len(vector)}")
    if not all(math.isfinite(item) for item in vector):
        raise ObservationContractError(f"{label} contains NaN or Inf: {vector}")
    return vector


def wrap_angle_rad(value: float) -> float:
    angle = float(value)
    if not math.isfinite(angle):
        raise ObservationContractError(f"angle is not finite: {value!r}")
    return math.atan2(math.sin(angle), math.cos(angle))


def joint_delta(
    current_q: Sequence[float],
    previous_q: Sequence[float],
) -> list:
    current = _finite_vector(current_q, 4, "current_q")
    previous = _finite_vector(previous_q, 4, "previous_q")
    delta = [current[index] - previous[index] for index in range(4)]
    delta[0] = wrap_angle_rad(delta[0])
    return delta


def canonical_phase_index(phase: object) -> int:
    if isinstance(phase, (int, float)) and not isinstance(phase, bool):
        value = float(phase)
        index = int(round(value))
        if math.isfinite(value) and abs(value - index) <= 1.0e-6 and 0 <= index < len(CANONICAL_PHASE_NAMES):
            return index

    text = str(phase or "").strip().lower()
    if text == "loaded_transit" or "clearance_route_post" in text or "staged_unload" in text or "high_carry" in text:
        return 8
    if "unload_to_bin" in text or "unload_pre_dump_align" in text or "unload" in text or "dump" in text:
        return 9
    if "pre_dig" in text or "clearance_route" in text or "travel" in text or "align" in text:
        return 0
    if "approach_contact" in text or ("approach" in text and "contact" in text):
        return 1
    if "insert" in text:
        return 2
    if "pull_mid" in text:
        return 3
    if "curl" in text:
        return 4
    if "pull_exit" in text:
        return 5
    if "secure" in text:
        return 6
    if "lift" in text or "carry" in text:
        return 7
    raise ObservationContractError(f"unknown deployment phase: {phase!r}")


def point_in_initial_heading_frame(
    point_xyz: Sequence[float],
    origin_xy: Sequence[float],
    heading_rad: float,
) -> list:
    point = _finite_vector(point_xyz, 3, "point_xyz")
    origin = _finite_vector(origin_xy, 2, "origin_xy")
    return point_delta_in_heading_frame(
        point,
        [origin[0], origin[1], 0.0],
        heading_rad,
    )


def point_delta_in_heading_frame(
    point_xyz: Sequence[float],
    reference_xyz: Sequence[float],
    heading_rad: float,
) -> list:
    point = _finite_vector(point_xyz, 3, "point_xyz")
    reference = _finite_vector(reference_xyz, 3, "reference_xyz")
    heading = float(heading_rad)
    if not math.isfinite(heading):
        raise ObservationContractError(f"heading_rad is not finite: {heading_rad!r}")
    dx = point[0] - reference[0]
    dy = point[1] - reference[1]
    c = math.cos(heading)
    s = math.sin(heading)
    return [
        c * dx + s * dy,
        -s * dx + c * dy,
        point[2] - reference[2],
    ]


def bucket_pour_axis_in_heading_frame(
    bucket_load_world_xyz: Sequence[float],
    bucket_pour_world_xyz: Sequence[float],
    heading_rad: float,
) -> list:
    axis = point_delta_in_heading_frame(
        bucket_pour_world_xyz,
        bucket_load_world_xyz,
        heading_rad,
    )
    norm = math.hypot(float(axis[0]), float(axis[2]))
    if norm <= 1.0e-8:
        raise ObservationContractError("bucket load-to-pour axis is degenerate")
    return [float(axis[0]) / norm, float(axis[2]) / norm]


def joint_velocity_from_samples(
    current_q: Sequence[float],
    previous_q: Sequence[float],
    dt: float,
) -> list:
    delta_time = float(dt)
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ObservationContractError(f"joint velocity dt must be positive and finite: {dt!r}")
    return _finite_vector(
        [value / delta_time for value in joint_delta(current_q, previous_q)],
        4,
        "joint_velocity_4d",
    )


def bucket_fill_fraction(
    bucket_load_particles: float,
    capacity_particles: float = BUCKET_FILL_CAPACITY_PARTICLES,
) -> float:
    load = float(bucket_load_particles)
    capacity = float(capacity_particles)
    if not math.isfinite(load) or not math.isfinite(capacity) or capacity <= 0.0:
        raise ObservationContractError(
            f"invalid bucket fill values: load={bucket_load_particles!r} capacity={capacity_particles!r}"
        )
    return min(BUCKET_FILL_FRACTION_MAX, max(0.0, load / capacity))


def causal_bucket_fill_rate(
    current_fill_fraction: float,
    previous_fill_fraction: Optional[float],
    previous_smoothed_rate: Optional[float],
    dt: float,
    tau_seconds: float = BUCKET_FILL_RATE_TAU_SECONDS,
) -> float:
    current = float(current_fill_fraction)
    if previous_fill_fraction is None:
        return 0.0
    delta_time = float(dt)
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        return 0.0
    raw_rate = (current - float(previous_fill_fraction)) / delta_time
    previous_rate = 0.0 if previous_smoothed_rate is None else float(previous_smoothed_rate)
    tau = max(1.0e-6, float(tau_seconds))
    alpha = 1.0 - math.exp(-delta_time / tau)
    smoothed = previous_rate + alpha * (raw_rate - previous_rate)
    return min(BUCKET_FILL_RATE_ABS_MAX, max(-BUCKET_FILL_RATE_ABS_MAX, smoothed))


def source_tracked_bucket_load(
    bucket_count: int,
    bucket_from_initial_count: int,
    fallback_min_region_count: int = 32,
    fallback_min_source_ratio: float = 0.60,
) -> dict:
    """Apply the dataset's initial-pile source fallback to a closed-volume count."""
    total = max(0, int(bucket_count))
    from_initial = max(0, min(total, int(bucket_from_initial_count)))
    min_count = max(0, int(fallback_min_region_count))
    min_ratio = float(fallback_min_source_ratio)
    if not math.isfinite(min_ratio) or not 0.0 <= min_ratio <= 1.0:
        raise ObservationContractError(
            f"fallback_min_source_ratio must be in [0, 1], got {min_ratio!r}"
        )
    ratio = float(from_initial) / max(1.0, float(total))
    if total >= min_count and ratio < min_ratio:
        return {
            "count": total,
            "bucket_count": total,
            "bucket_from_initial_count": from_initial,
            "source_ratio": ratio,
            "source_tracking": "bucket_region_fallback",
        }
    return {
        "count": from_initial,
        "bucket_count": total,
        "bucket_from_initial_count": from_initial,
        "source_ratio": ratio,
        "source_tracking": "initial_mask",
    }


class DeploymentPhaseEstimator:
    """Infer the dataset's monotonic 10-stage phase from live simulator state."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self.previous_load_z = None
        self.stage_entry_bucket_joint = None
        self.max_bucket_load = 0.0
        self.last_transition_reason = "episode_start"

    def update(
        self,
        dt: float,
        joint_positions_4d: Sequence[float],
        bucket_tip_world_xyz: Sequence[float],
        bucket_load_world_xyz: Sequence[float],
        bucket_load_estimate: float,
        bucket_load_rate: float,
        dig_target_world_xyz: Sequence[float],
        unload_landing_world_xyz: Sequence[float],
    ) -> dict:
        delta_time = float(dt)
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ObservationContractError(
                f"phase estimator dt must be positive and finite: {dt!r}"
            )
        joints = _finite_vector(joint_positions_4d, 4, "joint_positions_4d")
        tip = _finite_vector(bucket_tip_world_xyz, 3, "bucket_tip_world_xyz")
        load_point = _finite_vector(
            bucket_load_world_xyz,
            3,
            "bucket_load_world_xyz",
        )
        dig_target = _finite_vector(
            dig_target_world_xyz,
            3,
            "dig_target_world_xyz",
        )
        unload_landing = _finite_vector(
            unload_landing_world_xyz,
            3,
            "unload_landing_world_xyz",
        )
        load_count = float(bucket_load_estimate)
        load_rate = float(bucket_load_rate)
        if not math.isfinite(load_count) or not math.isfinite(load_rate):
            raise ObservationContractError(
                "phase estimator bucket load/count rate must be finite"
            )

        self.phase_elapsed += delta_time
        self.max_bucket_load = max(self.max_bucket_load, load_count)
        tip_dx = tip[0] - dig_target[0]
        tip_dy = tip[1] - dig_target[1]
        tip_dz = tip[2] - dig_target[2]
        dig_tip_xy = math.hypot(tip_dx, tip_dy)
        dig_tip_distance = math.sqrt(
            tip_dx * tip_dx + tip_dy * tip_dy + tip_dz * tip_dz
        )
        unload_dx = load_point[0] - unload_landing[0]
        unload_dy = load_point[1] - unload_landing[1]
        unload_dz = load_point[2] - unload_landing[2]
        unload_load_xy = math.hypot(unload_dx, unload_dy)
        unload_load_distance = math.sqrt(
            unload_dx * unload_dx
            + unload_dy * unload_dy
            + unload_dz * unload_dz
        )
        if self.previous_load_z is None:
            load_vertical_velocity = 0.0
        else:
            load_vertical_velocity = (
                load_point[2] - float(self.previous_load_z)
            ) / delta_time
        self.previous_load_z = load_point[2]

        enough_load = load_count >= 16.0
        strong_load = load_count >= 32.0
        transition_reason = ""
        phase = int(self.phase_index)
        if phase == 0 and dig_tip_distance <= 1.25:
            transition_reason = "tip_entered_dig_approach"
        elif (
            phase == 1
            and dig_tip_xy <= 0.65
            and tip[2] <= dig_target[2] + 0.55
        ):
            transition_reason = "tip_reached_contact_neighborhood"
        elif (
            phase == 2
            and self.phase_elapsed >= 0.30
            and (
                strong_load
                or (
                    dig_tip_xy <= 0.65
                    and tip[2] <= dig_target[2] + 0.25
                )
            )
        ):
            transition_reason = "insert_depth_or_load_reached"
        elif (
            phase == 3
            and self.phase_elapsed >= 0.45
            and (
                strong_load
                or load_rate > 5.0
                or (
                    dig_tip_xy <= 0.80
                    and tip[2] <= dig_target[2] + 0.40
                )
            )
        ):
            transition_reason = "pull_mid_progress_reached"
        elif phase == 4 and self.phase_elapsed >= 0.35 and enough_load:
            entry_bucket = (
                joints[3]
                if self.stage_entry_bucket_joint is None
                else float(self.stage_entry_bucket_joint)
            )
            closed_delta = entry_bucket - joints[3]
            if (
                closed_delta >= math.radians(12.0)
                or load_vertical_velocity > 0.025
                or self.phase_elapsed >= 1.50
            ):
                transition_reason = "loaded_bucket_curled_or_rising"
        elif (
            phase == 5
            and self.phase_elapsed >= 0.30
            and enough_load
            and load_point[2] >= dig_target[2] + 0.35
        ):
            transition_reason = "bucket_exited_cut"
        elif (
            phase == 6
            and self.phase_elapsed >= 0.30
            and enough_load
            and load_point[2] >= dig_target[2] + 0.65
        ):
            transition_reason = "secure_load_above_surface"
        elif (
            phase == 7
            and self.phase_elapsed >= 0.40
            and enough_load
            and load_point[2] >= dig_target[2] + 1.00
        ):
            transition_reason = "carry_height_reached"
        elif (
            phase == 8
            and self.phase_elapsed >= 0.40
            and (
                unload_load_xy <= 1.80
                or unload_load_distance <= 2.25
            )
        ):
            transition_reason = "bucket_reached_unload_neighborhood"

        if transition_reason and self.phase_index < len(CANONICAL_PHASE_NAMES) - 1:
            self.phase_index += 1
            self.phase_elapsed = 0.0
            self.stage_entry_bucket_joint = joints[3]
            self.last_transition_reason = transition_reason

        return {
            "phase_index": int(self.phase_index),
            "phase_name": CANONICAL_PHASE_NAMES[int(self.phase_index)],
            "phase_source": "simulator_auto_fsm",
            "phase_elapsed": float(self.phase_elapsed),
            "transition_reason": str(
                transition_reason or self.last_transition_reason
            ),
            "dig_tip_distance": float(dig_tip_distance),
            "dig_tip_xy": float(dig_tip_xy),
            "unload_load_distance": float(unload_load_distance),
            "unload_load_xy": float(unload_load_xy),
            "load_height_above_dig_target": float(
                load_point[2] - dig_target[2]
            ),
            "load_vertical_velocity": float(load_vertical_velocity),
            "bucket_load_estimate": float(load_count),
            "bucket_load_rate": float(load_rate),
        }


def build_legacy_state_28d_v3(
    base_state_14d: Sequence[float],
    joint_velocity_4d: Sequence[float],
    phase: object,
    dig_target_world_xyz: Sequence[float],
    unload_landing_world_xyz: Sequence[float],
    initial_origin_xy: Sequence[float],
    initial_heading_rad: float,
    truck_yaw_rad: float,
    bucket_load_rate: float,
) -> list:
    base_state = _finite_vector(base_state_14d, 14, "base_state_14d")
    joint_velocity = _finite_vector(joint_velocity_4d, 4, "joint_velocity_4d")
    dig_local = point_in_initial_heading_frame(
        dig_target_world_xyz,
        initial_origin_xy,
        initial_heading_rad,
    )
    unload_local = point_in_initial_heading_frame(
        unload_landing_world_xyz,
        initial_origin_xy,
        initial_heading_rad,
    )
    relative_yaw = wrap_angle_rad(float(truck_yaw_rad) - float(initial_heading_rad))
    state = (
        base_state
        + joint_velocity
        + [float(canonical_phase_index(phase))]
        + dig_local
        + unload_local
        + [math.sin(relative_yaw), math.cos(relative_yaw)]
        + [float(bucket_load_rate)]
    )
    return _finite_vector(
        state,
        len(LEGACY_STATE_NAMES_28D_V3),
        "legacy_observation.state",
    )


def build_state_28d(
    joint_positions_4d: Sequence[float],
    joint_velocity_4d: Sequence[float],
    joint_tracking_error_4d: Sequence[float],
    previous_action_4d: Sequence[float],
    bucket_tip_world_xyz: Sequence[float],
    bucket_load_world_xyz: Sequence[float],
    bucket_pour_world_xyz: Sequence[float],
    dig_target_world_xyz: Sequence[float],
    unload_landing_world_xyz: Sequence[float],
    upper_heading_rad: float,
    truck_yaw_rad: float,
    bucket_fill_fraction_value: float,
    bucket_fill_rate_fraction_per_s: float,
) -> list:
    q = _finite_vector(joint_positions_4d, 4, "joint_positions_4d")
    dq = _finite_vector(joint_velocity_4d, 4, "joint_velocity_4d")
    q_error = _finite_vector(joint_tracking_error_4d, 4, "joint_tracking_error_4d")
    q_error[0] = wrap_angle_rad(q_error[0])
    previous_action = _finite_vector(previous_action_4d, 4, "previous_action_4d")
    tip = _finite_vector(bucket_tip_world_xyz, 3, "bucket_tip_world_xyz")
    load = _finite_vector(bucket_load_world_xyz, 3, "bucket_load_world_xyz")
    pour = _finite_vector(bucket_pour_world_xyz, 3, "bucket_pour_world_xyz")
    upper_heading = float(upper_heading_rad)
    truck_yaw = float(truck_yaw_rad)
    fill_fraction_value = float(bucket_fill_fraction_value)
    fill_rate = float(bucket_fill_rate_fraction_per_s)
    if not all(math.isfinite(value) for value in (upper_heading, truck_yaw, fill_fraction_value, fill_rate)):
        raise ObservationContractError("heading or bucket fill feature is not finite")

    pour_axis = bucket_pour_axis_in_heading_frame(load, pour, upper_heading)
    dig_delta = point_delta_in_heading_frame(
        dig_target_world_xyz,
        tip,
        upper_heading,
    )
    unload_delta = point_delta_in_heading_frame(
        unload_landing_world_xyz,
        load,
        upper_heading,
    )
    relative_yaw = wrap_angle_rad(truck_yaw - upper_heading)
    state = (
        q
        + dq
        + q_error
        + previous_action
        + pour_axis
        + dig_delta
        + unload_delta
        + [math.sin(relative_yaw), math.cos(relative_yaw)]
        + [fill_fraction_value, fill_rate]
    )
    return _finite_vector(state, len(STATE_NAMES_28D), "observation.state")


def schema_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "observation.state": {
            "shape": [len(STATE_NAMES_28D)],
            "names": list(STATE_NAMES_28D),
        },
        "observation.effort": {
            "shape": [len(EFFORT_NAMES_4D)],
            "names": list(EFFORT_NAMES_4D),
        },
        "observation.stage_current_id": {
            "dtype": "int64",
            "shape": [1],
            "names": ["stage_current_id"],
            "class_names": list(CANONICAL_PHASE_NAMES),
            "role": "categorical_supervision_not_policy_input",
        },
        "action": {
            "shape": [len(ACTION_NAMES_4D)],
            "names": list(ACTION_NAMES_4D),
            "unit": "rad/s",
            "semantics": "next_command_velocity",
        },
        "effective_robot_observation_dim": len(STATE_NAMES_28D) + len(EFFORT_NAMES_4D),
    }


def validate_payload(payload: Mapping[str, object]) -> dict:
    if not isinstance(payload, Mapping):
        raise ObservationContractError("deployment observation payload must be a mapping")
    state = _finite_vector(payload.get("observation_state_28d"), len(STATE_NAMES_28D), "observation_state_28d")
    effort = _finite_vector(payload.get("observation_effort"), len(EFFORT_NAMES_4D), "observation_effort")
    phase_raw = payload.get("observation_stage_current_id")
    phase_index = None if phase_raw is None else canonical_phase_index(phase_raw)
    return {
        "observation_state_28d": state,
        "observation_effort": effort,
        "observation_stage_current_id": phase_index,
    }


def optional_context_value(context: Optional[Mapping[str, object]], *names: str):
    if not isinstance(context, Mapping):
        return None
    for name in names:
        value = context.get(name)
        if value is not None:
            return value
    return None
