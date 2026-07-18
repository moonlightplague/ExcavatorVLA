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
