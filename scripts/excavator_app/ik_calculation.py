import math
import numpy as np


def floor_safe_effector_target_z(
    model,
    end_effector,
    world_angle_rad,
    point_min_z,
    margin=0.0,
):
    """Return the lowest target Z that keeps bucket geometry above its limits."""
    effectors = (model or {}).get("effectors", {})
    target_part = effectors.get(str(end_effector))
    if not isinstance(target_part, dict):
        return None, {"reason": f"missing_effector:{end_effector}"}

    target_lengths = np.asarray(target_part.get("lengths", []), dtype=np.float64).reshape(-1)
    target_offsets = np.asarray(target_part.get("offsets", []), dtype=np.float64).reshape(-1)
    if len(target_lengths) != 3 or len(target_offsets) != 3:
        return None, {"reason": f"invalid_effector_model:{end_effector}"}

    target_angle = float(world_angle_rad)
    target_terminal_z = float(target_lengths[2]) * math.sin(target_angle)
    requirements = []

    for name, min_z in dict(point_min_z or {}).items():
        if min_z is None:
            continue
        if str(name) == "bucket_joint":
            delta_z = -target_terminal_z
        else:
            part = effectors.get(str(name))
            if not isinstance(part, dict):
                continue
            lengths = np.asarray(part.get("lengths", []), dtype=np.float64).reshape(-1)
            offsets = np.asarray(part.get("offsets", []), dtype=np.float64).reshape(-1)
            if len(lengths) != 3 or len(offsets) != 3:
                continue
            point_angle = target_angle + float(offsets[2] - target_offsets[2])
            point_terminal_z = float(lengths[2]) * math.sin(point_angle)
            delta_z = point_terminal_z - target_terminal_z
        requirements.append(
            {
                "point": str(name),
                "min_z": float(min_z),
                "delta_from_target_z": float(delta_z),
                "required_target_z": float(min_z) - float(delta_z) + float(margin),
            }
        )

    if not requirements:
        return None, {"reason": "no_usable_floor_limits"}
    limiting = max(requirements, key=lambda row: float(row["required_target_z"]))
    return float(limiting["required_target_z"]), {
        "reason": "ok",
        "end_effector": str(end_effector),
        "world_angle_deg": math.degrees(target_angle),
        "margin": float(margin),
        "limiting_point": str(limiting["point"]),
        "required_target_z": float(limiting["required_target_z"]),
        "requirements": requirements,
    }


def floor_safe_vertical_correction(point_z, point_min_z, margin=0.0):
    """Return the upward correction required by a fresh FK floor report."""
    requirements = []
    measured = dict(point_z or {})
    for name, min_z in dict(point_min_z or {}).items():
        value = measured.get(str(name))
        if value is None or min_z is None:
            continue
        deficit = float(min_z) - float(value)
        if deficit <= 0.0:
            continue
        requirements.append(
            {
                "point": str(name),
                "actual_z": float(value),
                "min_z": float(min_z),
                "deficit": float(deficit),
            }
        )
    if not requirements:
        return 0.0, {
            "reason": "already_safe",
            "margin": float(margin),
            "requirements": [],
        }
    limiting = max(requirements, key=lambda row: float(row["deficit"]))
    correction = float(limiting["deficit"]) + max(0.0, float(margin))
    return correction, {
        "reason": "correction_required",
        "margin": float(margin),
        "limiting_point": str(limiting["point"]),
        "correction_z": float(correction),
        "requirements": requirements,
    }


def joint_motion_metrics(rt, q_to, q_from, duration=0.0):
    deltas = np.array(rt.q_delta_abs_deg(q_to, q_from), dtype=np.float32)
    weighted_angle = float(np.sum(rt.DIG_PLAN_MOTION_WEIGHTS * deltas))
    est_times = []
    for i, name in enumerate(rt.DOF_ORDER):
        speed = max(1e-4, rt.rad_to_deg(float(rt.DQ_MAX.get(name, 1.0)) * rt.get_speed_multiplier()))
        est_times.append(float(deltas[i]) / speed)
    estimated_time = max(float(duration), max(est_times) if est_times else 0.0)
    cost = (
        float(rt.DIG_PLAN_ANGLE_COST_WEIGHT) * weighted_angle
        + float(rt.DIG_PLAN_TIME_COST_WEIGHT) * estimated_time
    )
    return {
        "joint_delta_deg": [float(x) for x in deltas],
        "weighted_angle": weighted_angle,
        "estimated_time": float(estimated_time),
        "cost": float(cost),
    }


def path_penalty(rt, q_start, q_goal, mode, deadline=None):
    penalty = 0.0
    detail = {
        "phase_ok": True,
        "obstacle_ok": True,
        "phase_reason": "ok",
        "obstacle_reason": "ok",
    }
    try:
        samples = max(3, int(getattr(rt, "DIG_PLAN_PATH_CHECK_SAMPLES", rt.PATH_CHECK_SAMPLES)))
        phase_ok, phase_reason, phase_sample, phase_report = rt.path_phase_check(
            q_start, q_goal, mode, samples=samples, deadline=deadline
        )
        detail.update(
            {
                "phase_ok": bool(phase_ok),
                "phase_reason": str(phase_reason),
                "phase_sample": int(phase_sample),
            }
        )
        if not phase_ok:
            penalty += float(rt.DIG_PLAN_PATH_SOFT_PENALTY)
            detail["phase_report"] = phase_report
    except Exception as e:
        detail["phase_ok"] = False
        detail["phase_reason"] = f"phase_check_failed:{type(e).__name__}:{e}"
        penalty += float(rt.DIG_PLAN_PATH_SOFT_PENALTY)

    try:
        samples = max(3, int(getattr(rt, "DIG_PLAN_PATH_CHECK_SAMPLES", rt.PATH_CHECK_SAMPLES)))
        obstacle_ok, obstacle_reason, obstacle_sample, obstacle_report = rt.path_obstacle_check(
            q_start, q_goal, mode, samples=samples, deadline=deadline
        )
        detail.update(
            {
                "obstacle_ok": bool(obstacle_ok),
                "obstacle_reason": str(obstacle_reason),
                "obstacle_sample": int(obstacle_sample),
            }
        )
        if not obstacle_ok:
            penalty += float(rt.DIG_PLAN_OBSTACLE_SOFT_PENALTY)
            detail["obstacle_report"] = obstacle_report
    except Exception as e:
        detail["obstacle_ok"] = False
        detail["obstacle_reason"] = f"obstacle_check_failed:{type(e).__name__}:{e}"
        penalty += float(rt.DIG_PLAN_OBSTACLE_SOFT_PENALTY)

    detail["path_penalty"] = float(penalty)
    return float(penalty), detail


def adaptive_dig_plan_candidates(rt, target_xyz):
    target = np.array(target_xyz, dtype=np.float32).reshape(-1)[:3]
    ctx = rt.task_scene_context()
    pile_center = np.array(ctx.get("pile_center", [target[0], target[1], rt.GROUND_TOP_Z]), dtype=np.float32).reshape(-1)[:3]
    surface_z = None
    try:
        surface_z = rt.sand_surface_z_at_xy(float(target[0]), float(target[1]))
    except Exception:
        surface_z = None
    if surface_z is None:
        surface_z = max(float(pile_center[2]), float(target[2]) + 0.22, rt.GROUND_TOP_Z + 0.35)
    surface_z = max(float(surface_z), float(target[2]) + 0.05)
    requested_depth = max(0.04, min(0.34, surface_z - float(target[2])))

    candidates = [dict(x) for x in rt.DIG_PLAN_CANDIDATES]
    approach_values = [0.18, 0.28, 0.40]
    pull_pairs = [(0.26, 0.46), (0.36, 0.60), (0.48, 0.76)]
    depth_scales = [0.65, 1.0]
    lift_values = [0.46, 0.68, 0.92]
    idx = 0
    for approach in approach_values:
        for mid_pull, exit_pull in pull_pairs:
            for depth_scale in depth_scales:
                if idx >= rt.DIG_PLAN_MAX_CANDIDATES - len(rt.DIG_PLAN_CANDIDATES):
                    break
                insert_depth = max(0.10, min(0.26, requested_depth * depth_scale))
                mid_depth = max(insert_depth + 0.06, min(0.38, requested_depth * (1.20 + 0.30 * depth_scale)))
                exit_depth = max(0.04, min(0.12, insert_depth * 0.50))
                lift_height = lift_values[idx % len(lift_values)]
                angle_bias = -4.0 if depth_scale > 0.9 else 3.0
                candidates.append(
                    {
                        "id": f"adaptive_{idx:02d}",
                        "surface_z": float(surface_z),
                        "approach_offset": approach,
                        "pre_z": max(0.46, min(0.86, 0.38 + requested_depth + 0.16 * (idx % 2))),
                        "contact_z": max(0.02, min(0.10, requested_depth * 0.28)),
                        "insert_depth": insert_depth,
                        "mid_pull": mid_pull,
                        "mid_depth": mid_depth,
                        "exit_pull": exit_pull,
                        "exit_depth": exit_depth,
                        "exit_lift_z": max(0.08, min(0.14, requested_depth * 0.32 + 0.04)),
                        "curl_z": max(0.28, min(0.46, requested_depth * 0.72 + 0.18)),
                        "lift_height": lift_height,
                        "min_lift_z": max(rt.GROUND_TOP_Z + 1.05, float(target[2]) + 0.48),
                        "lift_above_target": max(0.48, lift_height),
                        "unload_height_delta": 0.0 if lift_height < 0.80 else 0.10,
                        "unload_dump_deg": 84.0 if idx % 2 == 0 else 88.0,
                        "bucket_travel": -24.0,
                        "bucket_attack": -48.0,
                        "bucket_cut": -58.0 - 4.0 * depth_scale,
                        "bucket_attack_world": float(rt.BUCKET_DIG_APPROACH_WORLD_DEG + 0.5 * angle_bias),
                        "bucket_cut_world": float(rt.BUCKET_DIG_INSERT_WORLD_DEG + angle_bias),
                        "bucket_mid_cut_world": float(rt.BUCKET_DIG_PULL_WORLD_DEG + 0.6 * angle_bias),
                        "bucket_exit_world": float(rt.BUCKET_DIG_EXIT_WORLD_DEG + 0.35 * angle_bias),
                        "bucket_exit": -70.0,
                        "bucket_curl": float(rt.CURL_HOLD_TARGET_DEG),
                        "curl_boom_lift_deg": max(4.0, min(6.0, 4.0 + 2.0 * depth_scale)),
                    }
                )
                idx += 1

    unique = []
    seen = set()
    for row in candidates:
        key = (
            round(float(row.get("approach_offset", 0.0)), 3),
            round(float(row.get("insert_depth", 0.0)), 3),
            round(float(row.get("mid_pull", 0.0)), 3),
            round(float(row.get("exit_pull", 0.0)), 3),
            round(float(row.get("lift_height", 0.0)), 3),
            round(float(row.get("unload_dump_deg", 0.0)), 1),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= rt.DIG_PLAN_MAX_CANDIDATES:
            break
    return unique
