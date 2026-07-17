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

STATE_NAMES_28D = BASE_STATE_NAMES_14D + [
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

SCHEMA_VERSION = "excavator_state_v3_28d_plus_4effort_phase_index10"


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
    heading = float(heading_rad)
    if not math.isfinite(heading):
        raise ObservationContractError(f"heading_rad is not finite: {heading_rad!r}")
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    c = math.cos(heading)
    s = math.sin(heading)
    return [
        c * dx + s * dy,
        -s * dx + c * dy,
        point[2],
    ]


def joint_velocity_from_samples(
    current_q: Sequence[float],
    previous_q: Sequence[float],
    dt: float,
) -> list:
    current = _finite_vector(current_q, 4, "current_q")
    previous = _finite_vector(previous_q, 4, "previous_q")
    delta_time = float(dt)
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ObservationContractError(f"joint velocity dt must be positive and finite: {dt!r}")
    delta = [current[index] - previous[index] for index in range(4)]
    delta[0] = math.atan2(math.sin(delta[0]), math.cos(delta[0]))
    return _finite_vector(
        [value / delta_time for value in delta],
        4,
        "joint_velocity_4d",
    )


def build_state_28d(
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
    phase_index = canonical_phase_index(phase)
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
    truck_yaw = float(truck_yaw_rad)
    load_rate = float(bucket_load_rate)
    if not math.isfinite(truck_yaw):
        raise ObservationContractError(f"truck_yaw_rad is not finite: {truck_yaw_rad!r}")
    if not math.isfinite(load_rate):
        raise ObservationContractError(f"bucket_load_rate is not finite: {bucket_load_rate!r}")

    relative_yaw = truck_yaw - float(initial_heading_rad)
    state = (
        base_state
        + joint_velocity
        + [float(phase_index)]
        + dig_local
        + unload_local
        + [math.sin(relative_yaw), math.cos(relative_yaw)]
        + [load_rate]
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
        "action": {
            "shape": [len(ACTION_NAMES_4D)],
            "names": list(ACTION_NAMES_4D),
            "unit": "rad/s",
        },
        "effective_robot_observation_dim": len(STATE_NAMES_28D) + len(EFFORT_NAMES_4D),
    }


def validate_payload(payload: Mapping[str, object]) -> dict:
    if not isinstance(payload, Mapping):
        raise ObservationContractError("deployment observation payload must be a mapping")
    state = _finite_vector(payload.get("observation_state_28d"), len(STATE_NAMES_28D), "observation_state_28d")
    effort = _finite_vector(payload.get("observation_effort"), len(EFFORT_NAMES_4D), "observation_effort")
    return {
        "observation_state_28d": state,
        "observation_effort": effort,
    }


def optional_context_value(context: Optional[Mapping[str, object]], *names: str):
    if not isinstance(context, Mapping):
        return None
    for name in names:
        value = context.get(name)
        if value is not None:
            return value
    return None
