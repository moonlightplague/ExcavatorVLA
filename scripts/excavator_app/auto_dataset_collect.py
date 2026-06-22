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


def _snapshot_plan_state(rt):
    return {key: copy.deepcopy(rt.STATE.get(key)) for key in PLAN_STATE_KEYS}


def _restore_plan_state(rt, snapshot):
    for key, value in snapshot.items():
        rt.STATE[key] = copy.deepcopy(value)


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
    if "bucket body" in reason and "front tip" in reason:
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
        rt.rigid_obstacle_bboxes(force=True)
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

    per_ring_limit = max(1, int(getattr(rt, "AUTO_DIG_FULL_PLAN_TOPK_PER_RING", rt.AUTO_COLLECT_MAX_PLAN_RETRIES)))
    retry = 0
    global_failure_signature = None
    global_failure_count = 0
    global_failure_limit = max(2, int(getattr(rt, "AUTO_COLLECT_GLOBAL_PLAN_FAILURE_LIMIT", 4)))
    for ring_index, ring_rows_all in _group_targets_by_ring(ranked_targets):
        ring_rows = ring_rows_all[:per_ring_limit]
        ring_successes = []
        rt.info_print(
            "[AUTO DIG TARGET RING]",
            f"ring_index={ring_index}",
            f"candidates={len(ring_rows_all)}",
            f"planning_top={len(ring_rows)}",
        )
        repeated_failure_signature = None
        repeated_failure_count = 0
        for row in ring_rows:
            target = rt.np.array(row.get("target_xyz"), dtype=rt.np.float32).reshape(-1)[:3]
            rt.STATE["active_unload_landing_point"] = rt.np.array(unload_landing, dtype=rt.np.float32).reshape(-1)[:3]
            rt.set_target_models_from_xyz(target)
            await rt.step_updates(1)
            plan_t0 = rt.time.perf_counter()
            seq = rt.build_dig_plan_from_current_target(force_status=True)
            plan_elapsed_ms = 1000.0 * max(0.0, rt.time.perf_counter() - plan_t0)
            shared_plan = rt.STATE.get("current_dig_plan")
            chosen_plan = rt.STATE.get("dig_plan_candidate")
            best_failure = rt.STATE.get("dig_plan_best_failure")
            row["full_plan_ok"] = bool(seq)
            if seq:
                row["failed_stage"] = ""
                row["failure_reason"] = "ok"
                plan_cost = (
                    shared_plan.get("total_plan_cost")
                    if isinstance(shared_plan, dict) and shared_plan.get("total_plan_cost") is not None
                    else (chosen_plan or {}).get("rank_cost", (chosen_plan or {}).get("planner_cost", 1.0e9))
                )
                row["full_plan_cost"] = float(plan_cost)
            else:
                if isinstance(best_failure, dict):
                    row["failed_stage"] = str(best_failure.get("failed_stage", "unknown"))
                    row["failure_reason"] = str(best_failure.get("failure_reason", "planning_failed"))
                else:
                    row["failed_stage"] = "unknown"
                    row["failure_reason"] = "planning_failed/no_sequence"
                plan_cost = 1.0e9

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
                "chosen_plan": rt.compact_plan_candidate(chosen_plan, include_stages=False),
                "shared_plan": {
                    "plan_id": shared_plan.get("plan_id", "") if isinstance(shared_plan, dict) else "",
                    "cost": shared_plan.get("total_plan_cost") if isinstance(shared_plan, dict) else None,
                    "estimated_duration": shared_plan.get("estimated_duration") if isinstance(shared_plan, dict) else None,
                    "stage_count": shared_plan.get("stage_count") if isinstance(shared_plan, dict) else None,
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
            retry += 1
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
                if global_failure_count >= global_failure_limit:
                    rt.info_print(
                        "[AUTO DIG TARGET GLOBAL PRUNE]",
                        f"signature={signature}",
                        f"repeated={global_failure_count}",
                        "stop_planning_attempts=True",
                    )
                    return target, None, plan_attempts
                if repeated_failure_count >= 2:
                    rt.info_print(
                        "[AUTO DIG TARGET RING PRUNE]",
                        f"ring_index={ring_index}",
                        f"signature={signature}",
                        f"repeated={repeated_failure_count}",
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
    return target, None, plan_attempts


def update_prepare_failure_streak(rt, current_count):
    last_result = str(rt.STATE.get("auto_collect_last_result", ""))
    if "prepare_failed" in last_result:
        return int(current_count) + 1
    return 0
