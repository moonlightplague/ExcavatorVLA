import math
import time

import numpy as np


def _as_q(q):
    return np.array(q, dtype=np.float32).reshape(-1)[:4].copy()


def _delta(rt, q_to, q_from):
    q_to = _as_q(q_to)
    q_from = _as_q(q_from)
    out = q_to - q_from
    swing_idx = rt.CTRL.name_to_idx.get("swing", 0)
    out[swing_idx] = float(rt.swing_delta(q_to[swing_idx], q_from[swing_idx]))
    return out


def _distance(rt, q_a, q_b, weights):
    d = _delta(rt, q_a, q_b)
    return float(np.linalg.norm(np.array(weights, dtype=np.float32) * d))


def _nearest(rt, tree, q, weights):
    best_idx = 0
    best_dist = float("inf")
    for idx, node in enumerate(tree["nodes"]):
        dist = _distance(rt, node["q"], q, weights)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx, best_dist


def _path(tree, idx):
    out = []
    while idx is not None:
        node = tree["nodes"][idx]
        out.append(node["q"].copy())
        idx = node["parent"]
    out.reverse()
    return out


def _clip_near(rt, q, reference):
    try:
        if hasattr(rt, "clip_route_command_near"):
            return rt.clip_route_command_near(_as_q(q), reference=_as_q(reference))
        return rt.clip_command_near(_as_q(q), reference=_as_q(reference))
    except Exception:
        q = _as_q(q)
        try:
            return rt.CTRL.clip_limits(q)
        except Exception:
            return q


def _carry_world_angle(rt, q_start, q_goal, mode):
    text = str(mode).lower()
    if not any(token in text for token in ("carry", "unload", "clearance")):
        return None
    try:
        goal_angles = rt.chain_angles_from_q(q_goal, end_effector="load")
        start_angles = rt.chain_angles_from_q(q_start, end_effector="load")
        reference_angles = goal_angles if goal_angles is not None else start_angles
        if reference_angles is None:
            return None
        return float(rt.nearest_bucket_carry_world_angle(reference_angles[2], q_reference=q_goal, end_effector="load"))
    except Exception:
        return None


def _project_carry_bucket(rt, q, reference, carry_world_rad):
    q = _clip_near(rt, q, reference)
    if carry_world_rad is None:
        return q
    try:
        calc = rt.bucket_joint_for_world_angle(q, carry_world_rad, end_effector="load")
        if calc is None:
            return q
        bucket_idx = rt.CTRL.name_to_idx.get("bucket", 3)
        q[bucket_idx] = float(calc["bucket"])
        return _clip_near(rt, q, reference)
    except Exception:
        return q


def _steer(rt, q_from, q_to, max_step, carry_world_rad=None):
    q_from = _as_q(q_from)
    q_to = _clip_near(rt, q_to, q_from)
    d = _delta(rt, q_to, q_from)
    norm = float(np.linalg.norm(d))
    if norm <= max(1.0e-6, float(max_step)):
        return _project_carry_bucket(rt, q_to, q_from, carry_world_rad), True
    q_next = q_from + d * (float(max_step) / norm)
    return _project_carry_bucket(rt, q_next, q_from, carry_world_rad), False


def _edge_ok(rt, q_from, q_to, mode, samples, carry_world_rad=None, deadline=None):
    if deadline is not None and time.time() > float(deadline):
        return False, "planning_deadline"
    if carry_world_rad is not None:
        q_prev = _project_carry_bucket(rt, q_from, q_from, carry_world_rad)
        q_to = _project_carry_bucket(rt, q_to, q_prev, carry_world_rad)
        for i in range(1, max(2, int(samples)) + 1):
            if deadline is not None and time.time() > float(deadline):
                return False, "planning_deadline"
            s = float(i) / float(max(2, int(samples)))
            q = rt.interpolate_q_shortest(q_from, q_to, s)
            q = _project_carry_bucket(rt, q, q_prev, carry_world_rad)
            ok, kind, reason, sample, report = rt.path_segment_check(q_prev, q, mode, samples=2, deadline=deadline)
            if not ok:
                try:
                    text = rt.path_block_report_text(mode, kind, report, reason)
                except Exception:
                    text = str(reason)
                return False, f"carry_{kind}:{i}/{samples}:{sample}/2:{text}"
            q_prev = q.copy()
        return True, "ok"

    ok, kind, reason, sample, report = rt.path_segment_check(q_from, q_to, mode, samples=samples, deadline=deadline)
    if ok:
        return True, "ok"
    try:
        text = rt.path_block_report_text(mode, kind, report, reason)
    except Exception:
        text = str(reason)
    return False, f"{kind}:{sample}/{samples}:{text}"


def _joint_bounds(rt, q_start, q_goal):
    q_start = _as_q(q_start)
    q_goal = _clip_near(rt, q_goal, q_start)
    bounds = []
    for i, name in enumerate(rt.DOF_ORDER):
        if name == "swing":
            lo = min(float(q_start[i]), float(q_goal[i])) - math.radians(130.0)
            hi = max(float(q_start[i]), float(q_goal[i])) + math.radians(130.0)
            bounds.append((lo, hi))
            continue
        try:
            if hasattr(rt, "planner_effective_joint_bounds_rad"):
                lo, hi = rt.planner_effective_joint_bounds_rad(name)
            else:
                lo, hi = rt.FINAL_LIMITS_RAD[name]
            lo = float(lo)
            hi = float(hi)
        except Exception:
            center = 0.5 * (float(q_start[i]) + float(q_goal[i]))
            lo = center - math.radians(95.0)
            hi = center + math.radians(95.0)
        bounds.append((lo, hi))
    return bounds


def _sample(rt, rng, q_start, q_goal, bounds, goal_bias, carry_world_rad=None):
    if float(rng.random()) < float(goal_bias):
        return _project_carry_bucket(rt, q_goal, q_start, carry_world_rad)
    q = np.zeros(4, dtype=np.float32)
    for i, (lo, hi) in enumerate(bounds):
        q[i] = float(rng.uniform(float(lo), float(hi)))
    swing_idx = rt.CTRL.name_to_idx.get("swing", 0)
    if float(rng.random()) < 0.35:
        base = q_goal if float(rng.random()) < 0.5 else q_start
        q[swing_idx] = float(base[swing_idx]) + float(rng.uniform(-math.pi, math.pi))
    return _project_carry_bucket(rt, q, q_start, carry_world_rad)


def _extend(rt, tree, q_target, mode, max_step, samples, weights, carry_world_rad=None, deadline=None):
    if deadline is not None and time.time() > float(deadline):
        return "trapped", None, "planning_deadline"
    nearest_idx, _dist = _nearest(rt, tree, q_target, weights)
    q_near = tree["nodes"][nearest_idx]["q"]
    q_next, reached = _steer(rt, q_near, q_target, max_step, carry_world_rad=carry_world_rad)
    if np.linalg.norm(_delta(rt, q_next, q_near)) < 1.0e-5:
        return "trapped", nearest_idx, "zero_step"
    ok, reason = _edge_ok(rt, q_near, q_next, mode, samples, carry_world_rad=carry_world_rad, deadline=deadline)
    if not ok:
        return "trapped", nearest_idx, reason
    tree["nodes"].append({"q": q_next.copy(), "parent": nearest_idx})
    return "reached" if reached else "advanced", len(tree["nodes"]) - 1, "ok"


def _connect(rt, tree, q_target, mode, max_step, samples, weights, carry_world_rad=None, deadline=None):
    last_idx = None
    last_reason = "not_started"
    max_steps = 64
    if deadline is not None:
        max_steps = max(6, int(getattr(rt, "PATH_RRT_CONNECT_STEPS_WITH_DEADLINE", 24)))
    for _ in range(max_steps):
        if deadline is not None and time.time() > float(deadline):
            return "trapped", last_idx, "planning_deadline"
        status, idx, reason = _extend(
            rt,
            tree,
            q_target,
            mode,
            max_step,
            samples,
            weights,
            carry_world_rad=carry_world_rad,
            deadline=deadline,
        )
        last_idx = idx
        last_reason = reason
        if status != "advanced":
            return status, idx, reason
    return "advanced", last_idx, last_reason


def _path_cost(rt, path):
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        try:
            total += float(rt.plan_joint_motion_metrics(b, a, duration=0.0).get("weighted_angle", 0.0))
        except Exception:
            total += float(np.linalg.norm(_delta(rt, b, a)))
    return float(total)


def _local_cost(rt, q_prev, q_mid, q_next, weights, bend_weight):
    weights = np.array(weights, dtype=np.float32)
    d0 = weights * _delta(rt, q_mid, q_prev)
    d1 = weights * _delta(rt, q_next, q_mid)
    length_cost = float(np.linalg.norm(d0) + np.linalg.norm(d1))
    bend_cost = float(np.linalg.norm(d1 - d0))
    return length_cost + float(bend_weight) * bend_cost


def _q_deg_list(rt, q):
    try:
        return rt.q_deg_values(q, wrap_swing_for_display=True)
    except Exception:
        return [float(math.degrees(x)) for x in _as_q(q)]


def _shortcut(rt, path, mode, samples, deadline, rng, carry_world_rad=None):
    if len(path) <= 3:
        return path
    out = [p.copy() for p in path]
    max_rounds = min(80, max(12, 4 * len(out)))
    if deadline is not None:
        max_rounds = min(max_rounds, 18)
    for _ in range(max_rounds):
        if deadline is not None and time.time() > float(deadline):
            break
        if len(out) <= 3:
            break
        i = int(rng.integers(0, len(out) - 2))
        j = int(rng.integers(i + 2, len(out)))
        ok, _reason = _edge_ok(rt, out[i], out[j], mode, samples, carry_world_rad=carry_world_rad, deadline=deadline)
        if ok:
            out = out[: i + 1] + out[j:]
    return out


def _elastic_smooth(rt, path, mode, samples, deadline, rng, carry_world_rad=None, weights=None):
    if len(path) <= 2:
        return path, {
            "smooth_rounds": 0,
            "smooth_accepts": 0,
            "smooth_rejects": 0,
            "cost_before_smooth": _path_cost(rt, path),
            "cost_after_smooth": _path_cost(rt, path),
        }
    out = [p.copy() for p in path]
    weights = np.array(weights if weights is not None else [1.15, 1.0, 0.9, 0.7], dtype=np.float32)
    rounds = int(getattr(rt, "PATH_RRT_SMOOTH_ROUNDS", 24))
    if deadline is not None:
        rounds = min(rounds, int(getattr(rt, "PATH_RRT_SMOOTH_ROUNDS_WITH_DEADLINE", 8)))
    alpha = float(getattr(rt, "PATH_RRT_SMOOTH_ALPHA", 0.55))
    bend_weight = float(getattr(rt, "PATH_RRT_SMOOTH_BEND_WEIGHT", 0.35))
    min_improvement = float(getattr(rt, "PATH_RRT_SMOOTH_MIN_IMPROVEMENT", 1.0e-4))
    rounds = max(0, rounds)
    alpha = max(0.05, min(0.95, alpha))
    accepts = 0
    rejects = 0
    cost_before = _path_cost(rt, out)

    for _ in range(rounds):
        if deadline is not None and time.time() > float(deadline):
            break
        if len(out) <= 2:
            break
        order = list(range(1, len(out) - 1))
        rng.shuffle(order)
        for idx in order:
            if deadline is not None and time.time() > float(deadline):
                break
            q_prev = out[idx - 1]
            q_curr = out[idx]
            q_next = out[idx + 1]
            q_mid = q_prev + 0.5 * _delta(rt, q_next, q_prev)
            proposal = q_curr + alpha * _delta(rt, q_mid, q_curr)
            proposal = _project_carry_bucket(rt, proposal, q_curr, carry_world_rad)
            if float(np.linalg.norm(_delta(rt, proposal, q_curr))) < 1.0e-5:
                rejects += 1
                continue
            before = _local_cost(rt, q_prev, q_curr, q_next, weights, bend_weight)
            after = _local_cost(rt, q_prev, proposal, q_next, weights, bend_weight)
            if after > before - min_improvement:
                rejects += 1
                continue
            ok_a, _reason_a = _edge_ok(rt, q_prev, proposal, mode, samples, carry_world_rad=carry_world_rad, deadline=deadline)
            if not ok_a:
                rejects += 1
                continue
            ok_b, _reason_b = _edge_ok(rt, proposal, q_next, mode, samples, carry_world_rad=carry_world_rad, deadline=deadline)
            if not ok_b:
                rejects += 1
                continue
            out[idx] = proposal.copy()
            accepts += 1

    return out, {
        "smooth_rounds": int(rounds),
        "smooth_accepts": int(accepts),
        "smooth_rejects": int(rejects),
        "cost_before_smooth": float(cost_before),
        "cost_after_smooth": _path_cost(rt, out),
    }


def plan_joint_space_route(
    rt,
    q_start,
    q_goal,
    mode="clearance",
    label="joint_rrt",
    deadline=None,
    samples=None,
):
    start_t = time.time()
    q_start = _as_q(q_start)
    q_goal = _clip_near(rt, q_goal, q_start)
    carry_world_rad = _carry_world_angle(rt, q_start, q_goal, mode)
    q_start = _project_carry_bucket(rt, q_start, q_start, carry_world_rad)
    q_goal = _project_carry_bucket(rt, q_goal, q_start, carry_world_rad)
    sample_count = max(2, int(samples if samples is not None else getattr(rt, "PATH_ROUTE_PLANNING_SAMPLE_COUNT", 8)))
    max_iters = int(getattr(rt, "PATH_RRT_MAX_ITERS", 220))
    if deadline is not None:
        max_iters = min(max_iters, int(getattr(rt, "PATH_RRT_MAX_ITERS_WITH_DEADLINE", 120)))
    max_step = math.radians(float(getattr(rt, "PATH_RRT_STEP_DEG", 13.0)))
    goal_bias = float(getattr(rt, "PATH_RRT_GOAL_BIAS", 0.18))
    weights = np.array(getattr(rt, "PATH_RRT_JOINT_WEIGHTS", [1.15, 1.0, 0.9, 0.7]), dtype=np.float32)
    seed_base = int(abs(hash(str(label))) % 1000003)
    rng = np.random.default_rng(seed_base + int(time.time() * 10.0) % 1000003)
    stats = {
        "direct_reason": "",
        "trapped": 0,
        "advanced": 0,
        "reached": 0,
        "connect_reached": 0,
        "last_reason": "",
    }

    direct_ok, direct_reason = _edge_ok(
        rt, q_start, q_goal, mode, sample_count, carry_world_rad=carry_world_rad, deadline=deadline
    )
    stats["direct_reason"] = str(direct_reason)
    if direct_ok:
        return {
            "ok": True,
            "path": [q_start.copy(), q_goal.copy()],
            "waypoints": [],
            "reason": "direct_ok",
            "iterations": 0,
            "nodes": 2,
            "cost": _path_cost(rt, [q_start, q_goal]),
            "carry_world_deg": None if carry_world_rad is None else float(rt.rad_to_deg(carry_world_rad)),
            "q_start_deg": _q_deg_list(rt, q_start),
            "q_goal_deg": _q_deg_list(rt, q_goal),
            "stats": stats,
            "elapsed_ms": 1000.0 * (time.time() - start_t),
        }

    q_ok, q_reason = _edge_ok(
        rt, q_start, q_start, mode, sample_count, carry_world_rad=carry_world_rad, deadline=deadline
    )
    if not q_ok:
        stats["last_reason"] = str(q_reason)
        return {
            "ok": False,
            "reason": f"start_invalid:{q_reason}",
            "q_start_deg": _q_deg_list(rt, q_start),
            "q_goal_deg": _q_deg_list(rt, q_goal),
            "stats": stats,
            "elapsed_ms": 1000.0 * (time.time() - start_t),
        }
    q_ok, q_reason = _edge_ok(
        rt, q_goal, q_goal, mode, sample_count, carry_world_rad=carry_world_rad, deadline=deadline
    )
    if not q_ok:
        stats["last_reason"] = str(q_reason)
        return {
            "ok": False,
            "reason": f"goal_invalid:{q_reason}",
            "q_start_deg": _q_deg_list(rt, q_start),
            "q_goal_deg": _q_deg_list(rt, q_goal),
            "stats": stats,
            "elapsed_ms": 1000.0 * (time.time() - start_t),
        }

    bounds = _joint_bounds(rt, q_start, q_goal)
    tree_start = {"nodes": [{"q": q_start.copy(), "parent": None}]}
    tree_goal = {"nodes": [{"q": q_goal.copy(), "parent": None}]}
    last_reason = direct_reason

    for iteration in range(max_iters):
        if deadline is not None and time.time() > float(deadline):
            break
        active_start = iteration % 2 == 0
        tree_a = tree_start if active_start else tree_goal
        tree_b = tree_goal if active_start else tree_start
        q_rand = _sample(rt, rng, q_start, q_goal, bounds, goal_bias, carry_world_rad=carry_world_rad)
        status, idx_a, reason = _extend(
            rt,
            tree_a,
            q_rand,
            mode,
            max_step,
            sample_count,
            weights,
            carry_world_rad=carry_world_rad,
            deadline=deadline,
        )
        last_reason = reason
        if status == "trapped":
            stats["trapped"] += 1
            continue
        if status == "advanced":
            stats["advanced"] += 1
        elif status == "reached":
            stats["reached"] += 1
        q_new = tree_a["nodes"][idx_a]["q"]
        status_b, idx_b, reason_b = _connect(
            rt,
            tree_b,
            q_new,
            mode,
            max_step,
            sample_count,
            weights,
            carry_world_rad=carry_world_rad,
            deadline=deadline,
        )
        last_reason = reason_b
        if status_b == "reached":
            stats["connect_reached"] += 1
            if active_start:
                start_path = _path(tree_start, idx_a)
                goal_path = _path(tree_goal, idx_b)
            else:
                start_path = _path(tree_start, idx_b)
                goal_path = _path(tree_goal, idx_a)
            full = start_path + list(reversed(goal_path))[1:]
            full = _shortcut(rt, full, mode, sample_count, deadline, rng, carry_world_rad=carry_world_rad)
            full, smooth_stats = _elastic_smooth(
                rt,
                full,
                mode,
                sample_count,
                deadline,
                rng,
                carry_world_rad=carry_world_rad,
                weights=weights,
            )
            stats.update(smooth_stats)
            return {
                "ok": True,
                "path": [p.copy() for p in full],
                "waypoints": [p.copy() for p in full[1:-1]],
                "reason": "rrt_connect",
                "iterations": int(iteration + 1),
                "nodes": int(len(tree_start["nodes"]) + len(tree_goal["nodes"])),
                "cost": _path_cost(rt, full),
                "carry_world_deg": None if carry_world_rad is None else float(rt.rad_to_deg(carry_world_rad)),
                "q_start_deg": _q_deg_list(rt, q_start),
                "q_goal_deg": _q_deg_list(rt, q_goal),
                "stats": stats,
                "elapsed_ms": 1000.0 * (time.time() - start_t),
            }

    stats["last_reason"] = str(last_reason)
    return {
        "ok": False,
        "reason": f"rrt_failed:{last_reason}",
        "iterations": int(max_iters),
        "nodes": int(len(tree_start["nodes"]) + len(tree_goal["nodes"])),
        "carry_world_deg": None if carry_world_rad is None else float(rt.rad_to_deg(carry_world_rad)),
        "q_start_deg": _q_deg_list(rt, q_start),
        "q_goal_deg": _q_deg_list(rt, q_goal),
        "stats": stats,
        "elapsed_ms": 1000.0 * (time.time() - start_t),
    }
