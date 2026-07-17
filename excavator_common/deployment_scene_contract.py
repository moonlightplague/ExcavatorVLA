"""Dataset-compatible sand/truck scene profiles for deployment."""

import math
import random
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
TRAINING_SAND_ANGLE_RANGE_DEG = (82.0, 98.0)
TRAINING_UNLOAD_ANGLE_RANGE_DEG = (-170.0, -120.0)
TRAINING_SAND_AMOUNT_RANGE = (3.0, 15.0)
TRAINING_MIN_SAND_UNLOAD_DISTANCE_M = 6.0
TRAINING_MIN_ROBOT_TRUCK_DISTANCE_M = 3.5
TRAINING_TRUCK_SIDE_YAW_RANGE_DEG = 90.0
RANDOM_SCENE_MAX_TRIES = 96


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


def _rotate_xy(vector: Sequence[float], yaw_deg: float) -> tuple:
    x, y = _finite_vector(vector, 2, "vector")
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * x - sine * y,
        sine * x + cosine * y,
    )


def _sample_polar_xy(rng, radius_range, angle_range_deg) -> tuple:
    radius = rng.uniform(*sorted(float(value) for value in radius_range))
    angle_deg = rng.uniform(*sorted(float(value) for value in angle_range_deg))
    angle_rad = math.radians(angle_deg)
    return (
        (radius * math.cos(angle_rad), radius * math.sin(angle_rad)),
        radius,
        angle_deg,
    )


def sample_random_scene_profile(
    seed: int,
    robot_xy: Sequence[float],
    truck_center_xy: Sequence[float],
    truck_dump_center_xy: Sequence[float],
    truck_yaw_deg: float,
) -> dict:
    """Sample the deployment scene from dataset-collection distributions."""
    robot_xy = _finite_vector(robot_xy, 2, "robot_xy")
    truck_center_xy = _finite_vector(
        truck_center_xy,
        2,
        "truck_center_xy",
    )
    truck_dump_center_xy = _finite_vector(
        truck_dump_center_xy,
        2,
        "truck_dump_center_xy",
    )
    baseline_yaw_deg = float(truck_yaw_deg)
    if not math.isfinite(baseline_yaw_deg):
        raise ValueError("truck_yaw_deg must be finite")

    world_dump_offset = (
        truck_dump_center_xy[0] - truck_center_xy[0],
        truck_dump_center_xy[1] - truck_center_xy[1],
    )
    local_dump_offset = _rotate_xy(world_dump_offset, -baseline_yaw_deg)
    if math.hypot(*local_dump_offset) < 1.0e-4:
        raise ValueError("truck dump-bed offset is too small to determine yaw")

    rng = random.Random(int(seed))
    last_reason = "not_sampled"
    for sample_try in range(1, RANDOM_SCENE_MAX_TRIES + 1):
        sand_xy, sand_radius, sand_angle_deg = _sample_polar_xy(
            rng,
            TRAINING_SAND_RADIUS_RANGE_M,
            TRAINING_SAND_ANGLE_RANGE_DEG,
        )
        unload_xy, unload_radius, unload_angle_deg = _sample_polar_xy(
            rng,
            TRAINING_UNLOAD_RADIUS_RANGE_M,
            TRAINING_UNLOAD_ANGLE_RANGE_DEG,
        )
        sand_amount = rng.uniform(*TRAINING_SAND_AMOUNT_RANGE)

        rear_to_robot = (
            robot_xy[0] - unload_xy[0],
            robot_xy[1] - unload_xy[1],
        )
        desired_rear_angle_deg = math.degrees(
            math.atan2(rear_to_robot[1], rear_to_robot[0])
        )
        local_rear_angle_deg = math.degrees(
            math.atan2(local_dump_offset[1], local_dump_offset[0])
        )
        side_offset_deg = rng.uniform(
            -TRAINING_TRUCK_SIDE_YAW_RANGE_DEG,
            TRAINING_TRUCK_SIDE_YAW_RANGE_DEG,
        )
        sampled_truck_yaw_deg = (
            desired_rear_angle_deg
            - local_rear_angle_deg
            + side_offset_deg
            + 180.0
        ) % 360.0 - 180.0
        rotated_dump_offset = _rotate_xy(
            local_dump_offset,
            sampled_truck_yaw_deg,
        )
        sampled_truck_center_xy = (
            unload_xy[0] - rotated_dump_offset[0],
            unload_xy[1] - rotated_dump_offset[1],
        )

        sand_unload_distance = math.dist(sand_xy, unload_xy)
        robot_truck_distance = math.dist(
            robot_xy,
            sampled_truck_center_xy,
        )
        if sand_unload_distance < TRAINING_MIN_SAND_UNLOAD_DISTANCE_M:
            last_reason = "sand_unload_distance"
            continue
        if robot_truck_distance < TRAINING_MIN_ROBOT_TRUCK_DISTANCE_M:
            last_reason = "robot_truck_distance"
            continue

        return {
            "seed": int(seed),
            "sample_try": sample_try,
            "sand_xy": sand_xy,
            "sand_radius_m": sand_radius,
            "sand_angle_deg": sand_angle_deg,
            "sand_amount_multiplier": sand_amount,
            "unload_xy": unload_xy,
            "unload_radius_m": unload_radius,
            "unload_angle_deg": unload_angle_deg,
            "truck_center_xy": sampled_truck_center_xy,
            "truck_yaw_deg": sampled_truck_yaw_deg,
            "truck_side_offset_deg": side_offset_deg,
            "sand_unload_distance_m": sand_unload_distance,
            "robot_truck_distance_m": robot_truck_distance,
        }

    raise ValueError(
        "could not sample a legal randomized scene after "
        f"{RANDOM_SCENE_MAX_TRIES} tries: {last_reason}"
    )


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
