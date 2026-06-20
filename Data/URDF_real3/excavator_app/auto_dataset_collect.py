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

    if not ranked_targets:
        ranked_targets = [
            {
                "target_xyz": rt.vec_list(rt.auto_collect_sample_target(attempt_index, 0), 3),
                "score": 0.0,
                "planned": True,
                "reason": "fallback_random_after_rank_failed",
            }
        ]

    max_retries = max(1, int(rt.AUTO_COLLECT_MAX_PLAN_RETRIES))
    candidate_rows = ranked_targets[: max_retries]
    for retry, row in enumerate(candidate_rows):
        target = rt.np.array(row.get("target_xyz"), dtype=rt.np.float32).reshape(-1)[:3]
        rt.STATE["active_unload_landing_point"] = rt.np.array(unload_landing, dtype=rt.np.float32).reshape(-1)[:3]
        rt.set_target_models_from_xyz(target)
        await rt.step_updates(1)
        seq = rt.build_dig_plan_from_current_target(force_status=True)
        shared_plan = rt.STATE.get("current_dig_plan")
        plan_attempts.append(
            {
                "retry": retry,
                "target_xyz": rt.vec_list(target, 3),
                "unload_point_xyz": rt.vec_list(rt.unload_bin_dump_point(), 3),
                "unload_landing_xyz": rt.vec_list(rt.unload_bin_landing_point(), 3),
                "selected_unload_landing_xyz": rt.vec_list(unload_landing, 3),
                "target_score": row,
                "unload_score_top": unload_scores[: min(10, len(unload_scores))] if isinstance(unload_scores, list) else [],
                "planned": seq is not None,
                "steps": 0 if seq is None else len(seq),
                "candidate_count": len(rt.STATE.get("last_dig_plan_candidates", [])),
                "chosen_plan": rt.compact_plan_candidate(rt.STATE.get("dig_plan_candidate"), include_stages=False),
                "shared_plan": {
                    "plan_id": shared_plan.get("plan_id", "") if isinstance(shared_plan, dict) else "",
                    "cost": shared_plan.get("total_plan_cost") if isinstance(shared_plan, dict) else None,
                    "estimated_duration": shared_plan.get("estimated_duration") if isinstance(shared_plan, dict) else None,
                    "stage_count": shared_plan.get("stage_count") if isinstance(shared_plan, dict) else None,
                },
                "best_failure": rt.compact_plan_candidate(rt.STATE.get("dig_plan_best_failure"), include_stages=True),
                "build_ms": float(rt.STATE.get("dig_plan_last_build_ms", 0.0)),
                "candidates": [
                    rt.compact_plan_candidate(x, include_stages=False)
                    for x in rt.STATE.get("last_dig_plan_candidates", [])
                ],
            }
        )
        if seq:
            return target, seq, plan_attempts
        await rt.step_updates(2)
    return target, None, plan_attempts


def update_prepare_failure_streak(rt, current_count):
    last_result = str(rt.STATE.get("auto_collect_last_result", ""))
    if "prepare_failed" in last_result:
        return int(current_count) + 1
    return 0
