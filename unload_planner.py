"""Exact opening-center IK and preflight of carry and dump endpoints."""
from __future__ import annotations

import numpy as np
from collections import Counter


def damped_ik(position, target, seeds, lower, upper, iterations=28, tolerance=.025):
    """Bounded finite-difference IK; deterministic and independent of SciPy."""
    target = np.asarray(target, dtype=float)
    lower, upper = np.asarray(lower), np.asarray(upper)
    solutions = []
    for seed in seeds:
        q = np.clip(np.asarray(seed, dtype=float), lower, upper)
        for _ in range(iterations):
            point = np.asarray(position(q), dtype=float)
            residual = target - point
            error = float(np.linalg.norm(residual))
            if error < tolerance:
                if not any(np.linalg.norm(q - old[0]) < .02 for old in solutions):
                    solutions.append((q.copy(), error))
                break
            eps = 1e-4
            jacobian = np.column_stack([(np.asarray(position(q + np.eye(len(q))[axis] * eps)) - point) / eps
                                        for axis in range(len(q))])
            step = np.linalg.solve(jacobian.T @ jacobian + np.eye(len(q)) * .002,
                                   jacobian.T @ residual)
            step *= min(1., .30 / max(1e-9, float(np.max(np.abs(step)))))
            improved = False
            for scale in (1., .5, .2):
                candidate = np.clip(q + step * scale, lower, upper)
                if np.linalg.norm(target - position(candidate)) < error - 1e-7:
                    q, improved = candidate, True
                    break
            if not improved:
                break
    return solutions


class UnloadPlanner:
    def __init__(self, rt):
        self.rt = rt
        self.cache = {}

    def reset(self):
        self.cache.clear()

    def solve(self, landing, seed, release_height=None, *, dump_degrees=88., min_release_z=None):
        if release_height is None:
            wall_z = float(self.rt.task_scene_context()["unload_bin_z_range"][1])
            minimum = .65 if min_release_z is None else max(.65, float(min_release_z) - wall_z)
            for height in (minimum, minimum + .5, minimum + 1.):
                q, detail = self.solve(landing, seed, height, dump_degrees=dump_degrees)
                if q is not None:
                    return q, detail
            return None, "no_collision_free_centered_dump_pose"
        rt = self.rt
        landing = np.asarray(landing, dtype=float)
        seed = np.asarray(seed, dtype=np.float32)
        alignment_degrees = float(rt.unload_release_alignment_bucket_deg(dump_degrees))
        key = (*np.round(landing, 4), round(float(release_height), 3), alignment_degrees)
        if key in self.cache:
            return self.cache[key]
        indices = [rt.CTRL.name_to_idx[name] for name in ("swing", "boom", "arm")]
        bucket = rt.CTRL.name_to_idx["bucket"]
        bearing = float(rt.target_to_swing_angle(landing))
        target = landing.copy()
        target[2] = float(rt.task_scene_context()["unload_bin_z_range"][1]) + release_height
        low = [bearing - .65, *[rt.planner_effective_joint_bounds_rad(name)[0] for name in ("boom", "arm")]]
        high = [bearing + .65, *[rt.planner_effective_joint_bounds_rad(name)[1] for name in ("boom", "arm")]]
        q_dump = seed.copy()
        # Aim at the opening angle where material starts flowing. The final
        # empty-bucket angle has a different opening-center position.
        q_dump[bucket] = rt.deg_to_rad(alignment_degrees)
        def full(values):
            q = q_dump.copy()
            q[indices] = values
            return q
        def position(values):
            return np.asarray(rt.bucket_unload_release_point_world(q=full(values), reference_q=seed), dtype=float)
        seeds = [[bearing, np.radians(boom), np.radians(arm)]
                 for boom, arm in ((45, -65), (65, -85), (25, -25), (30, -60), (60, -40))]
        solutions = damped_ik(position, target, seeds, low, high)
        accepted = []
        rejects = Counter()
        for values, error in solutions:
            q = full(values)
            carry = q.copy()
            loaded_seed = seed.copy()
            loaded_seed[bucket] = rt.deg_to_rad(rt.CURL_HOLD_TARGET_DEG)
            carry[bucket] = loaded_seed[bucket]
            carry = rt.force_loaded_carry_bucket_q(carry, reference=loaded_seed, label="repeat_unload_preflight")
            closed, _report = rt.gravity_carry_closed_for_q(carry)
            if not closed:
                rejects["carry_not_closed"] += 1
                if not accepted and rejects["carry_not_closed"] == 1:
                    rt.info_print("[REPEAT DUMP CARRY REJECT]", f"release_q_deg={np.degrees(q).tolist()}",
                                  f"carry_q_deg={np.degrees(carry).tolist()}",
                                  f"world_error={_report.get('world_level_err_deg')}",
                                  f"reason={_report.get('reason')}", force_log=True)
                continue
            collision = rt.unload_goal_pose_collision_report(carry, reference_q=seed, mode="unload_to_bin")
            if collision is not None:
                rejects["carry_collision"] += 1
                rt.info_print("[REPEAT DUMP COLLISION REJECT]", str(collision), force_log=True)
                continue
            # The runtime report reads its target from STATE, including during
            # preflight of cells that have not been selected yet.
            missing = object()
            previous = rt.STATE.get("active_unload_landing_point", missing)
            rt.STATE["active_unload_landing_point"] = landing.copy()
            try:
                drop = rt.unload_drop_report(q=q, reference_q=seed)
            finally:
                if previous is missing:
                    rt.STATE.pop("active_unload_landing_point", None)
                else:
                    rt.STATE["active_unload_landing_point"] = previous
            if not rt.unload_drop_execution_ready(drop) or float(drop.get("release_xy_err", 999)) > .10:
                rejects["drop_not_ready"] += 1
                continue
            info = {"drop": drop, "drop_target": landing.tolist(), "release_target": target.tolist(),
                    "pour_target": rt.bucket_point_world("pour", q=q, reference_q=seed).tolist(),
                    "q_release_align_rad": q.tolist(), "dump_bucket_target_deg": dump_degrees,
                    "release_alignment_bucket_deg": float(np.degrees(q[bucket])),
                    "drop_alignment_ready": True, "drop_alignment_policy": "exact_opening_center_v1",
                    "planar_err": error, "release_source": "bucket_opening_center"}
            accepted.append((q, info))
        result = accepted[0] if accepted else (None, "no_collision_free_centered_dump_pose")
        rt.info_print("[REPEAT DUMP PREFLIGHT]", f"xy={landing[:2].tolist()}",
                      f"height={release_height}", f"solutions={len(solutions)}",
                      f"accepted={len(accepted)}", f"rejects={dict(rejects)}", force_log=True)
        self.cache[key] = result
        return result
