import numpy as np


def _copy_points(points, limit):
    return [np.array(p, dtype=np.float32).reshape(-1)[:3].copy() for p in points[:limit]]


def _extend_segment(points, segment):
    if not segment:
        return
    if points:
        segment = segment[1:]
    points.extend(segment)


def _append_planned_unload_dump(rt, points, breaks, q_cursor, stage_name, stage_index=None, source="planned"):
    name = str(stage_name).lower()
    if "unload" not in name or "dump_pose" in name or "planned_dump" in name:
        return q_cursor
    detail = rt.planned_unload_stage_detail(stage_index=stage_index, stage_name=stage_name)
    if not detail:
        return q_cursor
    q_dump = detail.get("q_dump")
    if q_dump is None:
        return q_cursor
    q_dump = rt.clip_command_near(np.array(q_dump, dtype=np.float32).reshape(-1)[:4].copy(), reference=q_cursor)
    segment, q_end = rt.planned_bucket_segment_points(q_cursor, q_dump, mode="unload_dump", samples=12)
    _extend_segment(points, segment)
    breaks.append({"phase": f"{stage_name}_planned_dump", "point_count": len(points), "source": source})
    return q_end.copy() if q_end is not None else q_dump.copy()


def cache_dig_plan_trace_points(rt, seq=None, start_q=None):
    seq = rt.STATE.get("dig_plan_sequence", None) if seq is None else seq
    if not seq:
        rt.STATE["dig_plan_trace_points"] = []
        rt.STATE["dig_plan_trace_stage_breaks"] = []
        return []

    if start_q is None:
        start_q = rt.STATE.get("dig_plan_start_q", None)
    q_cursor = rt.CTRL.q_cmd.copy() if start_q is None else np.array(start_q, dtype=np.float32).copy()
    points = []
    breaks = []
    for stage_index, (stage_name, q_goal, _duration) in enumerate(seq):
        segment, q_cursor_next = rt.planned_bucket_segment_points(q_cursor, q_goal, mode=stage_name)
        _extend_segment(points, segment)
        breaks.append({"phase": str(stage_name), "point_count": len(points)})
        if q_cursor_next is not None:
            q_cursor = q_cursor_next.copy()
        q_cursor = _append_planned_unload_dump(
            rt,
            points,
            breaks,
            q_cursor,
            stage_name,
            stage_index=stage_index,
            source="dig_plan_full",
        )
        if len(points) >= rt.TRACE_PLAN_MAX_POINTS:
            break

    points = points[: rt.TRACE_PLAN_MAX_POINTS]
    rt.STATE["dig_plan_trace_points"] = [np.array(p, dtype=np.float32).copy() for p in points]
    rt.STATE["dig_plan_trace_stage_breaks"] = breaks
    rt.STATE["trace_planned_bucket_points"] = rt.STATE["dig_plan_trace_points"]
    rt.STATE["trace_plan_source"] = "dig_plan_full"
    rt.STATE["trace_render_dirty"] = True
    rt.STATE["trace_no_plan_notice_shown"] = False
    return rt.STATE["dig_plan_trace_points"]


def cache_active_stage_trace_points(rt, stage_name, q_start, q_goal, stage_index=None, include_remaining=True):
    """Cache the execution segment for diagnostics without replacing blue plan trace."""
    if q_start is None or q_goal is None:
        return []

    q_cursor = np.array(q_start, dtype=np.float32).reshape(-1)[:4].copy()
    q_stage_goal = rt.clip_command_near(np.array(q_goal, dtype=np.float32).reshape(-1)[:4].copy(), reference=q_cursor)
    points = []
    breaks = []

    segment, q_cursor_next = rt.planned_bucket_segment_points(q_cursor, q_stage_goal, mode=stage_name)
    if segment:
        points.extend(segment)
    breaks.append({"phase": str(stage_name), "point_count": len(points), "source": "active"})
    if q_cursor_next is not None:
        q_cursor = q_cursor_next.copy()
    else:
        q_cursor = q_stage_goal.copy()
    q_cursor = _append_planned_unload_dump(
        rt,
        points,
        breaks,
        q_cursor,
        stage_name,
        stage_index=stage_index,
        source="active",
    )

    seq = rt.STATE.get("dig_plan_sequence", None)
    if include_remaining and seq:
        if stage_index is None:
            stage_index = rt.STATE.get("active_plan_stage_index", rt.STATE.get("dig_plan_step_index", 0))
        try:
            start_index = max(0, int(stage_index) + 1)
        except Exception:
            start_index = 0

        for next_index, (next_stage, next_goal, _duration) in enumerate(list(seq)[start_index:], start=start_index):
            if len(points) >= rt.TRACE_PLAN_MAX_POINTS:
                break
            segment, q_cursor_next = rt.planned_bucket_segment_points(q_cursor, next_goal, mode=next_stage)
            _extend_segment(points, segment)
            breaks.append({"phase": str(next_stage), "point_count": len(points), "source": "remaining"})
            if q_cursor_next is not None:
                q_cursor = q_cursor_next.copy()
            q_cursor = _append_planned_unload_dump(
                rt,
                points,
                breaks,
                q_cursor,
                next_stage,
                stage_index=next_index,
                source="remaining",
            )

    points = points[: rt.TRACE_PLAN_MAX_POINTS]
    rt.STATE["active_stage_trace_points"] = _copy_points(points, rt.TRACE_PLAN_MAX_POINTS)
    rt.STATE["active_stage_trace_stage_breaks"] = breaks
    rt.STATE["active_stage_trace_source"] = "active_command_remaining"
    return rt.STATE["active_stage_trace_points"]


def planned_bucket_points_from_dig_plan(rt):
    cached = rt.STATE.get("dig_plan_trace_points", [])
    if cached:
        return cached[: rt.TRACE_PLAN_MAX_POINTS]

    seq = rt.STATE.get("dig_plan_sequence", None)
    if not seq:
        return []

    start_q = rt.STATE.get("dig_plan_start_q", None)
    q_cursor = rt.CTRL.q_cmd.copy() if start_q is None else np.array(start_q, dtype=np.float32).copy()
    points = []
    for stage_index, (stage_name, q_goal, _duration) in enumerate(seq):
        segment, q_cursor_next = rt.planned_bucket_segment_points(q_cursor, q_goal, mode=stage_name)
        _extend_segment(points, segment)
        if q_cursor_next is not None:
            q_cursor = q_cursor_next.copy()
        q_cursor = _append_planned_unload_dump(
            rt,
            points,
            [],
            q_cursor,
            stage_name,
            stage_index=stage_index,
            source="dig_plan_full",
        )
        if len(points) >= rt.TRACE_PLAN_MAX_POINTS:
            return points[: rt.TRACE_PLAN_MAX_POINTS]

    return points[: rt.TRACE_PLAN_MAX_POINTS]
