async def move_planned_stage(rt, stage_name, q_goal, duration, task_id=None):
    ready, reason, _detail = await rt.wait_for_articulation_action_ready(
        f"{stage_name}_trace_start",
        min_stable_frames=rt.ACTION_READY_MIN_STABLE_FRAMES,
        max_frames=rt.ACTION_READY_STAGE_MAX_WAIT_FRAMES,
        record_failure=True,
    )
    if not ready:
        rt.update_status(f"[DIG EXEC FAILED] {stage_name}: action_channel_not_ready; {reason}", force=True)
        return False

    q_start = rt.sync_motion_start_q(stage_name)
    q_goal = rt.clip_command_near(q_goal, reference=q_start)
    try:
        trace_points = rt.cache_active_stage_trace_points(
            stage_name,
            q_start,
            q_goal,
            stage_index=rt.STATE.get("active_plan_stage_index", None),
            include_remaining=True,
        )
        if rt.current_trace_mode() == 2:
            rt.draw_trace(force=True)
        rt.info_print(
            "[PLAN EXEC TRACE]",
            f"stage={stage_name}",
            f"points={len(trace_points)}",
            "source=active_command_remaining",
            f"q_start={rt.q_deg_values(q_start, wrap_swing_for_display=True)}",
            f"q_goal={rt.q_deg_values(q_goal, wrap_swing_for_display=True)}",
        )
    except Exception as e:
        rt.info_print("[WARN] [PLAN EXEC TRACE] cache failed:", stage_name, type(e).__name__, e)

    try:
        phase_ok, phase_reason, phase_sample, phase_report = rt.path_phase_check(q_start, q_goal, stage_name)
        obstacle_ok, obstacle_reason, obstacle_sample, obstacle_report = rt.path_obstacle_check(q_start, q_goal, stage_name)
        if phase_ok and obstacle_ok:
            rt.info_print(
                "[PLAN EXEC DIRECT]",
                f"stage={stage_name}",
                f"samples={rt.PATH_CHECK_SAMPLES}",
                "ok=True",
                f"q_start={rt.q_deg_values(q_start, wrap_swing_for_display=True)}",
                f"q_goal={rt.q_deg_values(q_goal, wrap_swing_for_display=True)}",
            )
        else:
            detail = []
            if not phase_ok:
                detail.append(
                    f"phase sample={phase_sample}/{rt.PATH_CHECK_SAMPLES} "
                    + rt.format_ground_report(stage_name, phase_report, phase_reason)
                )
            if not obstacle_ok:
                detail.append(
                    f"obstacle sample={obstacle_sample}/{rt.PATH_CHECK_SAMPLES} "
                    + rt.format_obstacle_report(stage_name, obstacle_report, obstacle_reason)
                )
            rt.info_print(
                "[PLAN EXEC DIRECT WARN]",
                f"stage={stage_name}",
                "; ".join(detail),
                "following cached plan without hidden route injection",
            )
    except Exception as e:
        rt.info_print("[WARN] [PLAN EXEC DIRECT] path precheck failed:", stage_name, type(e).__name__, e)

    return await rt.move_to_profile(
        q_goal,
        seconds=duration,
        label=stage_name,
        task_id=task_id,
        mode=stage_name,
        q_start_override=q_start,
    )


async def move_unload_stage(rt, stage_name, q_goal, duration, task_id=None):
    return await rt.execute_unload_sequence(stage_name, q_goal, duration, task_id=task_id)
