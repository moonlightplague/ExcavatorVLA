"""Pure-Python helpers shared by excavator path planning code."""


def greedy_prune_waypoints(waypoints, route_is_valid, minimum_count=1):
    """Remove waypoints only when the supplied full-route validator still passes."""
    route = list(waypoints or [])
    original_count = len(route)
    minimum_count = max(0, int(minimum_count))
    checks = 0

    changed = True
    while changed and len(route) > minimum_count:
        changed = False
        for index in range(len(route)):
            candidate = route[:index] + route[index + 1 :]
            checks += 1
            if bool(route_is_valid(candidate)):
                route = candidate
                changed = True
                break

    return route, {
        "original_count": original_count,
        "final_count": len(route),
        "removed_count": original_count - len(route),
        "validation_checks": checks,
    }


def route_efficiency_score(
    estimated_time_s,
    weighted_angle_deg,
    waypoint_count,
    angle_seconds_per_degree=0.0025,
    waypoint_stop_penalty_s=0.12,
):
    """Combine execution time, joint travel, and waypoint stops in seconds-equivalent units."""
    return (
        max(0.0, float(estimated_time_s))
        + max(0.0, float(weighted_angle_deg)) * max(0.0, float(angle_seconds_per_degree))
        + max(0, int(waypoint_count)) * max(0.0, float(waypoint_stop_penalty_s))
    )


def height_floor_report(load_margin_m, tip_margin_m, tolerance_m=0.02):
    """Classify carry-height margins while tolerating small kinematic noise."""
    load_margin_m = float(load_margin_m)
    tip_margin_m = float(tip_margin_m)
    tolerance_m = max(0.0, float(tolerance_m))
    minimum_margin_m = min(load_margin_m, tip_margin_m)
    ok = minimum_margin_m >= -tolerance_m
    within_tolerance = ok and minimum_margin_m < 0.0
    return {
        "ok": bool(ok),
        "within_tolerance": bool(within_tolerance),
        "warning": "loaded_carry_height_near_floor" if within_tolerance else "",
        "reason": "ok_with_tolerance" if within_tolerance else ("ok" if ok else "loaded_carry_height_drop"),
        "load_margin": load_margin_m,
        "tip_margin": tip_margin_m,
        "minimum_margin": minimum_margin_m,
        "tolerance_m": tolerance_m,
    }


def loaded_lift_recovery_report(
    loaded_count,
    minimum_loaded_count,
    transitional_hold,
    loaded_carry_joint_ok,
    dump_branch,
    carry_score_before,
    carry_score_after,
    score_warning_threshold=3.0,
):
    """Allow a loaded lift trial while treating approximate carry score as advisory."""
    score_drop = max(0.0, float(carry_score_before) - float(carry_score_after))
    allowed = bool(
        int(loaded_count) >= int(minimum_loaded_count)
        and transitional_hold
        and loaded_carry_joint_ok
        and not dump_branch
    )
    return {
        "allowed": allowed,
        "score_drop": score_drop,
        "score_warning": bool(allowed and score_drop > max(0.0, float(score_warning_threshold))),
        "score_warning_threshold": max(0.0, float(score_warning_threshold)),
    }
