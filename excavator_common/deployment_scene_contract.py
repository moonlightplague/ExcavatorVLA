"""Deterministic sand/truck scene profile for simulator deployment."""

import math
from typing import Sequence


# The sand runtime and training generator are authored around this center.
AUTHORED_SAND_CENTER_XY = (0.0, 6.7)

# Training data uses these prims, not the legacy /World/DumpTruck path.
TRUCK_ROOT_PATH = "/World/truck"
TRUCK_BED_COLLISION_PATH = (
    "/World/truck/DumpBedCollision/dump_bed_collision"
)

# Verified successful training scene:
# .dashboard_success/episodes/260712_055131_ep000001
FIXED_TRUCK_TRANSLATION_XYZ = (
    -5.110382080078125,
    -9.374096870422363,
    0.0,
)
FIXED_TRUCK_YAW_DEG = -93.37492370605469
FIXED_UNLOAD_LANDING_XYZ = (
    -6.891574859619141,
    -7.851065635681152,
    4.222683429718018,
)

# World-origin workspace gates used by dataset scene randomization.
TRAINING_SAND_RADIUS_RANGE_M = (5.8, 7.35)
TRAINING_UNLOAD_RADIUS_RANGE_M = (8.0, 10.8)


def _finite_vector(value: Sequence[float], size: int, label: str) -> tuple:
    result = tuple(float(item) for item in value)
    if len(result) != size:
        raise ValueError(f"{label} must contain {size} values, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf: {result}")
    return result


def planar_radius(xy: Sequence[float]) -> float:
    x, y = _finite_vector(xy, 2, "xy")
    return math.hypot(x, y)


def wrapped_yaw_error_deg(actual_deg: float, expected_deg: float) -> float:
    """Return the smallest absolute difference between two yaw angles."""
    actual = float(actual_deg)
    expected = float(expected_deg)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError("yaw angles must be finite")
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def validate_fixed_scene_profile() -> dict:
    """Validate the fixed profile against the generator's world workspace."""
    sand_xy = _finite_vector(
        AUTHORED_SAND_CENTER_XY,
        2,
        "AUTHORED_SAND_CENTER_XY",
    )
    truck_xyz = _finite_vector(
        FIXED_TRUCK_TRANSLATION_XYZ,
        3,
        "FIXED_TRUCK_TRANSLATION_XYZ",
    )
    unload_xyz = _finite_vector(
        FIXED_UNLOAD_LANDING_XYZ,
        3,
        "FIXED_UNLOAD_LANDING_XYZ",
    )
    yaw_deg = float(FIXED_TRUCK_YAW_DEG)
    if not math.isfinite(yaw_deg):
        raise ValueError("FIXED_TRUCK_YAW_DEG must be finite")

    sand_radius = planar_radius(sand_xy)
    unload_radius = planar_radius(unload_xyz[:2])
    if not (
        TRAINING_SAND_RADIUS_RANGE_M[0]
        <= sand_radius
        <= TRAINING_SAND_RADIUS_RANGE_M[1]
    ):
        raise ValueError(
            "fixed sand center is outside the training workspace: "
            f"radius={sand_radius:.6f}"
        )
    if not (
        TRAINING_UNLOAD_RADIUS_RANGE_M[0]
        <= unload_radius
        <= TRAINING_UNLOAD_RADIUS_RANGE_M[1]
    ):
        raise ValueError(
            "fixed unload landing is outside the training workspace: "
            f"radius={unload_radius:.6f}"
        )

    return {
        "sand_center_xy": sand_xy,
        "sand_world_radius_m": sand_radius,
        "truck_root_path": TRUCK_ROOT_PATH,
        "truck_bed_collision_path": TRUCK_BED_COLLISION_PATH,
        "truck_translation_xyz": truck_xyz,
        "truck_yaw_deg": yaw_deg,
        "unload_landing_xyz": unload_xyz,
        "unload_world_radius_m": unload_radius,
    }
