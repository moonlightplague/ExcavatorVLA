import copy


PLAN_STATE_KEYS = [
    "current_dig_plan",
    "dig_plan_sequence",
    "dig_plan_candidate",
    "dig_plan_points",
    "dig_plan_trace_points",
    "dig_plan_trace_stage_breaks",
    "trace_planned_bucket_points",
    "trace_plan_source",
    "trace_render_dirty",
    "trace_render_signature",
    "dig_plan_target",
    "dig_plan_start_q",
    "dig_plan_step_index",
    "last_dig_plan_candidates",
    "dig_plan_best_failure",
    "dig_plan_last_build_ms",
]

TRANSIENT_PREPARE_FAILURE_MARKERS = (
    "prepare_failed/initial_robot_truck_overlap",
    "prepare_failed/home_direct_failed",
    "prepare_failed/initial_pose_pre_reset_failed",
    "prepare_failed/sand_not_settled",
)

PERSISTENT_PREPARE_FAILURE_MARKERS = (
    "prepare_failed/timeline_not_playing",
    "prepare_failed/timeline_stopped",
    "prepare_failed/action_channel_not_ready",
    "prepare_failed/physics_view_missing",
    "prepare_failed/ik_invalid",
    "prepare_failed/task_state_desync",
    "prepare_failed/sand_reset_exception",
    "prepare_failed/auto_collect_inactive_after_prepare",
)


def _snapshot_plan_state(rt):
    return {key: copy.deepcopy(rt.STATE.get(key)) for key in PLAN_STATE_KEYS}


def _restore_plan_state(rt, snapshot):
    for key, value in snapshot.items():
        rt.STATE[key] = copy.deepcopy(value)


def _clear_executable_plan_state(rt):
    for key in (
        "current_dig_plan",
        "dig_plan_sequence",
        "dig_plan_candidate",
        "dig_plan_points",
        "dig_plan_trace_points",
        "dig_plan_trace_stage_breaks",
        "trace_planned_bucket_points",
        "trace_plan_source",
        "trace_render_signature",
        "dig_plan_target",
        "dig_plan_start_q",
        "dig_plan_step_index",
    ):
        rt.STATE[key] = None
    rt.STATE["trace_render_dirty"] = True


def _group_targets_by_ring(rows):
    groups = []
    current_ring = None
    current_rows = []
    for row in rows:
        ring = int(row.get("ring_index", 999))
        if current_ring is None:
            current_ring = ring
        if ring != current_ring:
            groups.append((current_ring, current_rows))
            current_ring = ring
            current_rows = []
        current_rows.append(row)
    if current_rows:
        groups.append((current_ring, current_rows))
    return groups


def _planning_failure_signature(row):
    stage = str(row.get("failed_stage", ""))
    reason = str(row.get("failure_reason", ""))
    if "planning budget exceeded" in reason and ("no candidate tried" in reason or "planning_deadline" in reason):
        reason = "planning_budget_exceeded"
    elif "bucket body" in reason and "front tip" in reason:
        reason = "front_edge_body_over_tip"
    elif "no collision-free route" in reason:
        reason = "no_collision_free_route"
    elif "route IK failed" in reason:
        reason = "route_ik_failed"
    elif "planning budget exceeded" in reason:
        reason = "planning_budget_exceeded"
    elif ":" in reason:
        reason = reason.split(":", 1)[0]
    return f"{stage}:{reason}"


def _is_route_budget_signature(signature):
    return "planning_budget_exceeded" in str(signature)


async def find_plan(rt, attempt_index):
    plan_attempts = []
    target = rt.auto_collect_sample_target(attempt_index, 0)

    try:
        unload_landing, unload_scores = rt.choose_unload_landing_point_for_flat_fill()
    except Exception as e:
        unload_landing = rt.unload_bin_landing_point()
        unload_scores = []
        rt.STATE["active_unload_landing_point"] = unload_landing
        rt.STATE["last_auto_unload_scores"] = unload_scores
        rt.info_print("[WARN] [AUTO UNLOAD SELECT] failed:", type(e).__name__, e)

    try:
        ranked_targets = rt.auto_collect_rank_dig_targets(attempt_index)
    except Exception as e:
        ranked_targets = []
        rt.STATE["last_auto_dig_target_scores"] = []
        rt.info_print("[WARN] [AUTO DIG TARGET SELECT] failed:", type(e).__name__, e)

    try:
        # Static scene obstacles such as /World/truck are expensive to traverse.
        # Reuse the runtime cache here; the lower-level planner refreshes it when stale.
        rt.rigid_obstacle_bboxes(force=False)
    except Exception as e:
        rt.info_print("[WARN] [AUTO PLAN OBSTACLE SNAPSHOT] failed:", type(e).__name__, e)

    if not ranked_targets:
        rt.info_print("[AUTO DIG TARGET SELECT]", "no usable sand-sweep target; planning will be diagnostic only")
        return target, None, [
            {
                "retry": 0,
                "target_xyz": rt.vec_list(target, 3),
                "planned": False,
                "steps": 0,
                "failed_stage": "target_selection",
                "failure_reason": "planning_failed/no_swept_sand_target",
                "target_score": {
                    "target_xyz": rt.vec_list(target, 3),
                    "planned": False,
                    "reason": "no_swept_sand_target",
                },
                "unload_score_top": unload_scores[: min(10, len(unload_scores))] if isinstance(unload_scores, list) else [],
            }
        ]

    max_full_plan_attempts = max(
        1,
        int(getattr(rt, "AUTO_COLLECT_MAX_FULL_PLAN_ATTEMPTS", max(1, rt.AUTO_COLLECT_MAX_PLAN_RETRIES))),
    )
    per_ring_limit = max(1, int(getattr(rt, "AUTO_DIG_FULL_PLAN_TOPK_PER_RING", rt.AUTO_COLLECT_MAX_PLAN_RETRIES)))
    per_ring_limit = min(per_ring_limit, max_full_plan_attempts)
    find_plan_deadline = rt.time.time() + float(getattr(rt, "AUTO_COLLECT_FIND_PLAN_MAX_SECONDS", 28.0))
    retry = 0
    global_failure_signature = None
    global_failure_count = 0
    global_failure_limit = max(2, int(getattr(rt, "AUTO_COLLECT_GLOBAL_PLAN_FAILURE_LIMIT", 4)))
    for ring_index, ring_rows_all in _group_targets_by_ring(ranked_targets):
        ring_rows = ring_rows_all[:per_ring_limit]
        ring_successes = []
        first_success_time = None
        min_successes_per_ring = max(1, int(getattr(rt, "AUTO_COLLECT_MIN_SUCCESS_PER_RING", 1)))
        continue_after_first_s = max(
            0.0,
            float(getattr(rt, "AUTO_COLLECT_CONTINUE_AFTER_FIRST_SUCCESS_SECONDS", 0.0)),
        )
        rt.info_print(
            "[AUTO DIG TARGET RING]",
            f"ring_index={ring_index}",
            f"candidates={len(ring_rows_all)}",
            f"planning_top={len(ring_rows)}",
            f"min_successes={min_successes_per_ring}",
            f"continue_after_first={continue_after_first_s:.1f}s",
        )
        repeated_failure_signature = None
        repeated_failure_count = 0
        def _return_best_ring_success(reason):
            if not ring_successes:
                return None
            best = sorted(ring_successes, key=lambda x: (float(x["cost"]), -float(x["target_score"])))[0]
            _restore_plan_state(rt, best["snapshot"])
            rt.set_target_models_from_xyz(best["target"])
            best["row"]["selected"] = True
            rt.info_print(
                "[AUTO DIG TARGET SELECT]",
                f"ring_index={best['row'].get('ring_index')} selected=True",
                f"target={rt.vec_list(best['target'], 3)}",
                f"plan_cost={best['cost']:.3f}",
                f"reason={reason}",
            )
            return best["target"], best["seq"], plan_attempts

        for row in ring_rows:
            if rt.STATE.get("auto_collect_stop_requested", False) or not rt.STATE.get("running", False):
                rt.info_print("[AUTO DIG TARGET PLAN STOP]", "reason=stop_requested_or_runtime_stopped")
                _clear_executable_plan_state(rt)
                return target, None, plan_attempts
            if rt.time.time() > find_plan_deadline:
                best_result = _return_best_ring_success("find_plan_budget_reached_after_success")
                if best_result is not None:
                    return best_result
                rt.info_print(
                    "[AUTO DIG TARGET PLAN TIMEOUT]",
                    f"elapsed_budget={float(getattr(rt, 'AUTO_COLLECT_FIND_PLAN_MAX_SECONDS', 28.0)):.1f}s",
                    f"attempts={retry}",
                    "reason=find_plan_budget_exceeded",
                )
                _clear_executable_plan_state(rt)
                return target, None, plan_attempts
            if retry >= max_full_plan_attempts:
                best_result = _return_best_ring_success("full_plan_attempt_limit_after_success")
                if best_result is not None:
                    return best_result
                rt.info_print(
                    "[AUTO DIG TARGET PLAN LIMIT]",
                    f"attempts={retry}",
                    f"limit={max_full_plan_attempts}",
                    "reason=max_full_plan_attempts",
                )
                _clear_executable_plan_state(rt)
                return target, None, plan_attempts
            target = rt.np.array(row.get("target_xyz"), dtype=rt.np.float32).reshape(-1)[:3]
            rt.STATE["active_unload_landing_point"] = rt.np.array(unload_landing, dtype=rt.np.float32).reshape(-1)[:3]
            rt.set_target_models_from_xyz(target)
            await rt.step_updates(1)
            plan_t0 = rt.time.perf_counter()
            rt.info_print(
                "[AUTO DIG TARGET PLAN START]",
                f"retry={retry}",
                f"ring_index={row.get('ring_index')}",
                f"target={rt.vec_list(target, 3)}",
                f"depth={row.get('depth_candidate')}",
                f"swept_density={row.get('swept_density_count')}",
                f"budget={float(getattr(rt, 'AUTO_COLLECT_CANDIDATE_PLAN_SECONDS', 4.0)):.2f}s",
            )
            seq = rt.build_dig_plan_from_current_target(
                force_status=True,
                max_seconds=float(getattr(rt, "AUTO_COLLECT_CANDIDATE_PLAN_SECONDS", 4.0)),
            )
            plan_elapsed_ms = 1000.0 * max(0.0, rt.time.perf_counter() - plan_t0)
            rt.perf_block_record(
                "auto_find_plan_candidate",
                plan_elapsed_ms,
                data={
                    "retry": int(retry),
                    "ring_index": int(row.get("ring_index", ring_index)),
                    "target": rt.vec_list(target, 3),
                },
            )
            shared_plan = rt.STATE.get("current_dig_plan")
            chosen_plan = rt.STATE.get("dig_plan_candidate")
            best_failure = rt.STATE.get("dig_plan_best_failure")
            pre_sample_ok = True
            pre_sample_reason = "ok"
            pre_sample_detail = {}
            if seq:
                gate = getattr(rt, "auto_collect_plan_pre_sample_gate", None)
                if callable(gate):
                    pre_sample_ok, pre_sample_reason, pre_sample_detail = gate(seq, target=target)
                if not pre_sample_ok:
                    rt.info_print(
                        "[AUTO PRE-SAMPLE GATE]",
                        f"retry={retry}",
                        f"ring_index={row.get('ring_index')}",
                        "result=reject_plan_before_recording",
                        f"reason={pre_sample_reason}",
                    )
                    seq = None
                    row["pre_sample_gate"] = {
                        "ok": False,
                        "reason": str(pre_sample_reason),
                        "detail": pre_sample_detail,
                    }
                    row["failed_stage"] = "pre_sample_gate"
                    row["failure_reason"] = str(pre_sample_reason)
                    rt.STATE["dig_plan_best_failure"] = dict(row)
                    rt.STATE["dig_plan_sequence"] = None
                    rt.STATE["current_dig_plan"] = None
            row["full_plan_ok"] = bool(seq)
            if seq:
                row["failed_stage"] = ""
                row["failure_reason"] = "ok"
                row["pre_sample_gate"] = {
                    "ok": True,
                    "reason": str(pre_sample_reason),
                    "detail": pre_sample_detail,
                }
                plan_cost = (
                    (chosen_plan or {}).get("rank_cost")
                    if isinstance(chosen_plan, dict) and (chosen_plan or {}).get("rank_cost") is not None
                    else (
                        shared_plan.get("total_plan_cost")
                        if isinstance(shared_plan, dict) and shared_plan.get("total_plan_cost") is not None
                        else (chosen_plan or {}).get("planner_cost", 1.0e9)
                    )
                )
                row["full_plan_cost"] = float(plan_cost)
            else:
                if str(row.get("failed_stage", "")) == "pre_sample_gate":
                    row["failed_stage"] = "pre_sample_gate"
                    row["failure_reason"] = str(row.get("failure_reason", pre_sample_reason))
                elif isinstance(best_failure, dict):
                    row["failed_stage"] = str(best_failure.get("failed_stage", "unknown"))
                    row["failure_reason"] = str(best_failure.get("failure_reason", "planning_failed"))
                else:
                    row["failed_stage"] = "unknown"
                    row["failure_reason"] = "planning_failed/no_sequence"
                plan_cost = 1.0e9

            world_summary = rt.planning_world_snapshot(False, 12) or {}
            attempt_row = {
                "retry": retry,
                "ring_index": int(row.get("ring_index", ring_index)),
                "center_distance": row.get("center_distance"),
                "depth_candidate": row.get("depth_candidate"),
                "swept_density_count": row.get("swept_density_count"),
                "full_plan_ok": bool(seq),
                "failed_stage": row.get("failed_stage", ""),
                "failure_reason": row.get("failure_reason", ""),
                "target_xyz": rt.vec_list(target, 3),
                "unload_point_xyz": rt.vec_list(rt.unload_bin_dump_point(), 3),
                "unload_landing_xyz": rt.vec_list(rt.unload_bin_landing_point(), 3),
                "selected_unload_landing_xyz": rt.vec_list(unload_landing, 3),
                "target_score": row,
                "unload_score_top": unload_scores[: min(10, len(unload_scores))] if isinstance(unload_scores, list) else [],
                "planned": seq is not None,
                "steps": 0 if seq is None else len(seq),
                "candidate_count": len(rt.STATE.get("last_dig_plan_candidates", [])),
                "planning_world": {
                    "rigid_obstacle_count": int(world_summary.get("rigid_obstacle_count", 0)),
                    "sand_soft_contact": bool(world_summary.get("sand_particles_treated_as_soft_contact", True)),
                },
                "chosen_plan": rt.compact_plan_candidate(chosen_plan, include_stages=False),
                "shared_plan": {
                    "plan_id": shared_plan.get("plan_id", "") if isinstance(shared_plan, dict) else "",
                    "cost": shared_plan.get("total_plan_cost") if isinstance(shared_plan, dict) else None,
                    "estimated_duration": shared_plan.get("estimated_duration") if isinstance(shared_plan, dict) else None,
                    "stage_count": shared_plan.get("stage_count") if isinstance(shared_plan, dict) else None,
                    "unload_ballistics": shared_plan.get("unload_ballistics", {}) if isinstance(shared_plan, dict) else {},
                },
                "best_failure": rt.compact_plan_candidate(best_failure, include_stages=True),
                "build_ms": float(rt.STATE.get("dig_plan_last_build_ms", 0.0)),
                "wall_ms": float(plan_elapsed_ms),
                "candidates": [
                    rt.compact_plan_candidate(x, include_stages=False)
                    for x in rt.STATE.get("last_dig_plan_candidates", [])
                ],
            }
            plan_attempts.append(attempt_row)
            rt.info_print(
                "[AUTO DIG TARGET PLAN]",
                f"retry={retry}",
                f"ring_index={row.get('ring_index')}",
                f"target={rt.vec_list(target, 3)}",
                f"depth={row.get('depth_candidate')}",
                f"swept_density={row.get('swept_density_count')}",
                f"full_plan_ok={bool(seq)}",
                f"cost={plan_cost if seq else 'None'}",
                f"build_ms={float(rt.STATE.get('dig_plan_last_build_ms', 0.0)):.1f}",
                f"wall_ms={plan_elapsed_ms:.1f}",
                f"failed_stage={row.get('failed_stage', '')}",
                f"reason={row.get('failure_reason', row.get('reason', ''))}",
            )
            if seq:
                now_success = rt.time.time()
                if first_success_time is None:
                    first_success_time = now_success
                repeated_failure_signature = None
                repeated_failure_count = 0
                ring_successes.append(
                    {
                        "target": target.copy(),
                        "seq": copy.deepcopy(seq),
                        "snapshot": _snapshot_plan_state(rt),
                        "cost": float(plan_cost),
                        "target_score": float(row.get("score", 0.0) or 0.0),
                        "row": row,
                    }
                )
                rt.info_print(
                    "[AUTO DIG TARGET ACCEPT]",
                    f"retry={retry}",
                    f"ring_index={row.get('ring_index')}",
                    "reason=candidate_success_best_of_k_pending",
                    f"successes={len(ring_successes)}/{min_successes_per_ring}",
                    f"elapsed_after_first={0.0 if first_success_time is None else now_success - first_success_time:.2f}s",
                    f"wall_ms={plan_elapsed_ms:.1f}",
                )
                enough_successes = len(ring_successes) >= min_successes_per_ring
                enough_after_first = (
                    len(ring_successes) > 0
                    and continue_after_first_s <= 0.0
                ) or (
                    first_success_time is not None
                    and now_success - first_success_time >= continue_after_first_s
                )
                retry += 1
                if (
                    enough_successes
                    or enough_after_first
                    or retry >= max_full_plan_attempts
                    or rt.time.time() > find_plan_deadline
                ):
                    reason = (
                        "best_of_k_min_successes"
                        if enough_successes
                        else (
                            "best_of_k_time_window"
                            if enough_after_first
                            else (
                                "best_of_k_attempt_limit"
                                if retry >= max_full_plan_attempts
                                else "best_of_k_find_plan_budget"
                            )
                        )
                    )
                    return _return_best_ring_success(reason)
                await rt.step_updates(2)
                continue
            retry += 1
            budget_s = float(getattr(rt, "AUTO_COLLECT_CANDIDATE_PLAN_SECONDS", 4.0))
            hard_grace_s = float(getattr(rt, "AUTO_COLLECT_CANDIDATE_HARD_BUDGET_GRACE_SECONDS", 0.75))
            if plan_elapsed_ms > 1000.0 * max(0.1, budget_s + hard_grace_s):
                rt.info_print(
                    "[AUTO DIG TARGET HARD BUDGET]",
                    f"retry={retry}",
                    f"wall_ms={plan_elapsed_ms:.1f}",
                    f"budget={budget_s:.2f}s",
                    f"grace={hard_grace_s:.2f}s",
                    "reason=single_candidate_over_budget_stop_more_planning",
                )
                _clear_executable_plan_state(rt)
                return target, None, plan_attempts
            if not seq:
                signature = _planning_failure_signature(row)
                if signature == global_failure_signature:
                    global_failure_count += 1
                else:
                    global_failure_signature = signature
                    global_failure_count = 1
                if signature == repeated_failure_signature:
                    repeated_failure_count += 1
                else:
                    repeated_failure_signature = signature
                    repeated_failure_count = 1
                pre_dig_route_signature = signature in (
                    "pre_dig:planning_budget_exceeded",
                    "pre_dig:no_collision_free_route",
                )
                signature_global_limit = global_failure_limit
                signature_ring_limit = 2
                if pre_dig_route_signature:
                    signature_global_limit = 2 if signature.endswith("no_collision_free_route") else 3
                    signature_ring_limit = 1
                elif _is_route_budget_signature(signature):
                    signature_global_limit = 2
                    signature_ring_limit = 1
                if global_failure_count >= signature_global_limit:
                    rt.info_print(
                        "[AUTO DIG TARGET GLOBAL PRUNE]",
                        f"signature={signature}",
                        f"repeated={global_failure_count}",
                        f"limit={signature_global_limit}",
                        "stop_planning_attempts=True",
                    )
                    _clear_executable_plan_state(rt)
                    return target, None, plan_attempts
                if repeated_failure_count >= signature_ring_limit:
                    rt.info_print(
                        "[AUTO DIG TARGET RING PRUNE]",
                        f"ring_index={ring_index}",
                        f"signature={signature}",
                        f"repeated={repeated_failure_count}",
                        f"limit={signature_ring_limit}",
                        "moving_to_next_ring=True",
                    )
                    break
            await rt.step_updates(2)

        if ring_successes:
            best = sorted(ring_successes, key=lambda x: (float(x["cost"]), -float(x["target_score"])))[0]
            _restore_plan_state(rt, best["snapshot"])
            rt.set_target_models_from_xyz(best["target"])
            await rt.step_updates(1)
            best["row"]["selected"] = True
            rt.info_print(
                "[AUTO DIG TARGET SELECT]",
                f"ring_index={best['row'].get('ring_index')} selected=True",
                f"target={rt.vec_list(best['target'], 3)}",
                f"center_distance={best['row'].get('center_distance')}",
                f"depth={best['row'].get('depth_candidate')}",
                f"swept_density={best['row'].get('swept_density_count')}",
                f"plan_cost={best['cost']:.3f}",
            )
            return best["target"], best["seq"], plan_attempts

        rt.info_print(
            "[AUTO DIG TARGET RING FAILED]",
            f"ring_index={ring_index}",
            f"attempts={len(ring_rows)}",
            "moving_to_next_ring=True",
        )
    _clear_executable_plan_state(rt)
    return target, None, plan_attempts


def prepare_failure_streak_policy(last_result):
    text = str(last_result or "")
    if "prepare_failed" not in text:
        return "reset"
    if any(marker in text for marker in PERSISTENT_PREPARE_FAILURE_MARKERS):
        return "count"
    if any(marker in text for marker in TRANSIENT_PREPARE_FAILURE_MARKERS):
        return "retry"
    return "count"


def update_prepare_failure_streak(rt, current_count):
    policy = prepare_failure_streak_policy(
        rt.STATE.get("auto_collect_last_result", "")
    )
    if policy == "count":
        return int(current_count) + 1
    return 0
