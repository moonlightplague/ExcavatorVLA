"""Small runtime adapter: reuse upstream physics, planner, recorder and gates.

Installed after the upstream bootstrap loads the runtime, before collection.
Runtime changes are confined to this process. The shared exporter separately
recognizes the adapter's explicit physical-return supervision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
import time

import numpy as np

from baseline.bed_fill import BedFillPlanner, FillConfig
from baseline.episode_metrics import RepeatEpisodeMetrics
from baseline.motion_geometry import BucketFloor, RigidBucketKinematics
from baseline.unload_planner import UnloadPlanner, damped_ik


@dataclass(frozen=True)
class RepeatConfig:
    scoops: int = 3
    return_plan_seconds: float = 12.0
    settle_seconds: float = 1.0
    fill: FillConfig = field(default_factory=FillConfig)
    dig_target_spacing_m: float = .28
    min_dig_height_m: float = .15
    min_curl_height_m: float = 1.05

    def __post_init__(self):
        if not 2 <= self.scoops <= 64:
            raise ValueError("Repeated episodes require 2 to 64 scoops.")
        if not np.isfinite(self.return_plan_seconds) or self.return_plan_seconds <= 0:
            raise ValueError("Return planning budget must be finite and positive.")
        if not np.isfinite(self.settle_seconds) or self.settle_seconds < .1:
            raise ValueError("Dump settling must be at least 0.1 simulation seconds.")
        for name in ("dig_target_spacing_m", "min_dig_height_m", "min_curl_height_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class RepeatController:
    def __init__(self, rt, config):
        self.rt, self.config = rt, config
        # Endpoint-only 3.5 cm clearance leaves the opening-to-pull rotation
        # grazing the floor between endpoints. Reserve headroom for that arc so
        # the planner can select a direct cut instead of a free-space detour.
        self.floor = BucketFloor(rt, clearance=.10)
        self.kinematics = RigidBucketKinematics(rt)
        self.unloader = UnloadPlanner(rt)
        self.reset()

    def reset(self):
        self.bed = None
        self.cycles = []
        self.before = None
        self.selected = None
        self.meta = None
        self.metrics = None
        self.attempt_started = time.perf_counter()
        self.floor.reset()
        self.kinematics.reset()
        self.unloader.reset()

    def ensure_bed(self):
        if self.bed is None:
            rt = self.rt
            ctx = rt.task_scene_context()
            margin = max(self.config.fill.wall_margin_m,
                         float(rt.UNLOAD_BIN_SAFE_XY_MARGIN), float(rt.AUTO_UNLOAD_WALL_MARGIN))
            polygon = rt.unload_context_polygon_xy(ctx, safe_margin=0)
            safe = rt.unload_context_polygon_xy(ctx, safe_margin=margin)
            # The collection scene must provide the actual (possibly rotated)
            # truck polygon. An AABB fallback could place soil outside the bed.
            if polygon is None or safe is None:
                raise RuntimeError("Truck bed has no valid inset collision polygon.")
            z = ctx["unload_bin_z_range"]
            self.bed = BedFillPlanner(polygon, safe, z[0], z[1], self.config.fill)
        return self.bed

    def choose_landing(self):
        rt = self.rt
        bed = self.ensure_bed()
        rows, self.before = bed.rank(rt.sand_particle_positions(), rt.bucket_pour_pos())
        allowed = []
        for row in rows:
            status = rt.unload_landing_radius_status(
                row["landing"][:2], random_truck=None,
                soft_margin=rt.AUTO_SCENE_UNLOAD_LANDING_RADIUS_SOFT_MARGIN,
            )
            row.update(radius_ok=bool(status.get("ok")), radius_m=status.get("radius_m"))
            if row["radius_ok"]:
                allowed.append(row)
        if getattr(rt, "IK_MODEL", None) is not None:
            reachable = []
            for row in allowed:
                q_dump, detail = self.unloader.solve(row["landing"], rt.get_real_joint_positions())
                row["dump_preflight_ok"] = q_dump is not None
                if q_dump is not None:
                    row["dump_release_error_m"] = float(detail["drop"].get("release_xy_err", 0))
                    reachable.append(row)
            allowed = reachable
        if not allowed:
            raise RuntimeError("No safe truck-bed cell is inside the excavator reach window.")
        self.selected = allowed[0]
        landing = np.asarray(self.selected["landing"], dtype=np.float32)
        rt.STATE["active_unload_landing_point"] = landing.copy()
        rt.STATE["last_auto_unload_scores"] = allowed
        rt.info_print("[REPEAT FILL]", f"cell={self.selected['cell_id']}",
                      f"landing={landing.tolist()}", f"height={self.selected['cell_height']:.3f}",
                      f"particles={self.selected['cell_count']}", force_log=True)
        return landing, allowed

    def curl_candidates(self, q_seed, target, deadline=None):
        """Close while raising the bucket clear of the full rotation arc."""
        rt = self.rt
        seed = np.asarray(q_seed, dtype=np.float32)
        indices = [rt.CTRL.name_to_idx[name] for name in ("swing", "boom", "arm")]
        bucket = rt.CTRL.name_to_idx["bucket"]
        closed = seed.copy()
        closed[bucket] = rt.deg_to_rad(rt.CURL_HOLD_TARGET_DEG)
        def full(values):
            q = closed.copy()
            q[indices] = values
            return q
        def position(values):
            return np.asarray(rt.bucket_point_world("load", q=full(values), reference_q=seed), dtype=float)
        swing = float(seed[indices[0]])
        lower = [swing - .15, *[rt.planner_effective_joint_bounds_rad(name)[0] for name in ("boom", "arm")]]
        upper = [swing + .15, *[rt.planner_effective_joint_bounds_rad(name)[1] for name in ("boom", "arm")]]
        seeds = [seed[indices], [swing, np.radians(20.), np.radians(-70.)],
                 [swing, np.radians(40.), np.radians(-85.)]]
        target = np.asarray(target, dtype=float).copy()
        base_z = max(float(target[2]), float(rt.GROUND_TOP_Z) + self.config.min_curl_height_m)
        rows = []
        for extra_z in np.arange(.15, .76, .15):
            if deadline is not None and rt.time.time() >= deadline:
                break
            target[2] = base_z + extra_z
            for values, error in damped_ik(position, target, seeds, lower, upper):
                q = full(values)
                if rt.gravity_carry_closed_for_q(q)[0]:
                    phase_ok = rt.path_phase_check(seed, q, "curl_to_hold_material",
                                                   samples=rt.PATH_CHECK_SAMPLES, deadline=deadline)[0]
                    obstacle_ok = phase_ok and rt.path_obstacle_check(seed, q, "curl_to_hold_material",
                                                                     samples=rt.PATH_CHECK_SAMPLES, deadline=deadline)[0]
                    if phase_ok and obstacle_ok:
                        rows.append({"q": q, "info": {"load_target": target.tolist(), "error_m": error}})
            if len(rows) >= 3:
                break
        if deadline is not None:
            rt.info_print("[REPEAT CURL IK]", f"target={np.asarray(target).tolist()}",
                          f"solutions={len(rows)}", f"heights={[row['info']['load_target'][2] for row in rows]}",
                          force_log=True)
        return rows

    def adjust_plan_point(self, label, point, q_seed):
        point = np.asarray(point, dtype=np.float32).copy()
        if label in ("pull_exit_cut", "lift_carry"):
            load = self.rt.bucket_point_world("load", q=q_seed, reference_q=q_seed)
            margin = .04 if label == "pull_exit_cut" else .20
            point[2] = max(float(point[2]), float(load[2]) + margin)
        return point

    async def return_arm(self, q_goal):
        rt = self.rt
        q_start = rt.sync_motion_start_q("repeat_return_pre_dig")
        rt.clear_planning_runtime_caches("repeat_return_actual_pose")
        deadline = rt.time.time() + self.config.return_plan_seconds
        direct = rt.path_phase_check(q_start, q_goal, "pre_dig", samples=rt.PATH_CHECK_SAMPLES, deadline=deadline)[0]
        direct = direct and rt.path_obstacle_check(q_start, q_goal, "pre_dig", samples=rt.PATH_CHECK_SAMPLES, deadline=deadline)[0]
        if direct:
            waypoints, reason = [], "direct_ok"
        else:
            waypoints, reason = rt.find_clearance_route(
                q_start, q_goal, "pre_dig", "repeat_return_pre_dig", deadline=deadline,
                samples=rt.PATH_CHECK_SAMPLES,
            )
        if waypoints is None:
            rt.set_execution_failure_reason(f"execution_failed/repeat_return_no_route:{reason}")
            return False
        task_id = rt.start_task("repeat_return_pre_dig")
        rt.dataset_record_event("repeat_return_start", data={
            "scoop_index": rt.STATE.get("dataset_scoop_index", 0),
            "q_start_rad": np.asarray(q_start).tolist(), "q_goal_rad": np.asarray(q_goal).tolist(),
            "waypoints": len(waypoints), "route_reason": str(reason),
        })
        seconds = 0.0
        previous = q_start
        for q in [*waypoints, q_goal]:
            seconds += float(rt.plan_joint_motion_metrics(q, previous, duration=0).get("estimated_time", 1.0))
            previous = q
        rt.info_print("[REPEAT RETURN]", f"waypoints={len(waypoints)}",
                      f"planned_seconds={seconds:.2f}", force_log=True)
        ok = await rt.move_to_profile(
            q_goal, seconds=max(1.0, seconds + float(rt.MOVE_DURATION_MARGIN_SECONDS)),
            label="repeat_return_pre_dig", mode="pre_dig", task_id=task_id,
            path_waypoints=waypoints,
        )
        rt.dataset_record_event("repeat_return_end", data={
            "success": bool(ok), "q_real_rad": np.asarray(rt.get_real_joint_positions()).tolist(),
        })
        return bool(ok)

    async def execute_cycle(self, original, rebuild_plan=True, task_name="dig_target_ball", return_home=True):
        if self.metrics is not None:
            self.metrics.start_cycle(self.rt.STATE.get("dataset_scoop_index", 0))
        success = False
        try:
            success = await self._execute_cycle(original, rebuild_plan, task_name, return_home)
            return success
        finally:
            if self.metrics is not None:
                reason = "" if success else (self.rt.STATE.get("last_execution_failure_reason") or "execution_failed")
                row = self.metrics.finish_cycle(success, reason)
                if self.cycles and self.cycles[-1]["scoop_index"] == row["scoop_index"]:
                    self.cycles[-1]["metrics"] = row
            if self.cycles and self.cycles[-1]["scoop_index"] == int(self.rt.STATE.get("dataset_scoop_index", 0)):
                cycle = self.cycles[-1]
                directory = Path(self.rt.STATE["dataset_episode_dir"])
                self.rt.append_jsonl_dataset(str(directory / "repeat_cycles.jsonl"), cycle)
                self.rt.dataset_record_event("repeat_cycle_end", data=cycle)

    async def _execute_cycle(self, original, rebuild_plan=True, task_name="dig_target_ball", return_home=True):
        rt = self.rt
        sequence = rt.STATE.get("dig_plan_sequence") or []
        q_return = next((np.asarray(q).copy() for name, q, _duration in sequence if name == "pre_dig"), None)
        if q_return is None:
            rt.set_execution_failure_reason("execution_failed/repeat_missing_pre_dig_return_pose")
            return False
        selected = dict(self.selected or {})
        before = self.before
        episode_start = int(rt.STATE.get("dataset_episode_sample_start", 0))
        sample_start = int(rt.STATE.get("dataset_samples", 0)) - episode_start
        success = await original(rebuild_plan=rebuild_plan, task_name=task_name, return_home=False)
        if not success:
            return False
        # Keep the recording clock running while soil lands and while returning.
        for _ in range(max(1, int(np.ceil(self.config.settle_seconds / float(rt.CONTROL_DT))))):
            await rt.step_updates(rt.control_step_frames())
        after_dump = self.ensure_bed().measure(rt.sand_particle_positions())
        return_start = int(rt.STATE.get("dataset_samples", 0)) - episode_start
        returned = await self.return_arm(q_return)
        after = self.ensure_bed().measure(rt.sand_particle_positions())
        cycle = {
            "scoop_index": int(rt.STATE.get("dataset_scoop_index", 0)),
            "landing": selected, "bed_before": before, "bed_after_dump": after_dump,
            "bed_after_return": after, "returned": returned,
            "q_return_rad": q_return.tolist(),
            "sample_start": sample_start, "return_start": return_start,
            "sample_end": int(rt.STATE.get("dataset_samples", 0)) - episode_start,
        }
        self.cycles.append(cycle)
        if returned:
            self.bed.commit(selected["landing"])
        return returned

    def task(self, count=None):
        return (f"Dig {self.config.scoops} consecutive scoops from the visible sand pile into the truck. "
                "Spread the sand across low areas of the truck bed to make the load as flat as possible. "
                "Return the arm to the pile after every unload, including the last, without resetting the scene.")


def install(rt, config):
    """Install one explicit repeated-task policy on the stock collector."""
    if getattr(rt, "repeat_baseline", None) is not None:
        raise RuntimeError("Repeat baseline is already installed.")
    controller = RepeatController(rt, config)
    rt.repeat_baseline = controller
    # The upstream adaptive mode rejects the technical cap and may stop at an
    # unrelated truck particle target. This baseline has an explicit fixed count.
    rt.MULTI_SCOOP_ADAPTIVE_STOP_ENABLED = False
    rt.MULTI_SCOOPS_MIN_PER_EPISODE = config.scoops
    rt.MULTI_SCOOPS_PER_EPISODE = config.scoops
    # The stock search ends 0.37 m from the pile center. After the first cut,
    # this repeatedly closes the bucket in the same low crater. Search more of
    # the actual source polygon, prefer fresh cuts, and leave a floor clearance.
    # Existing sand-boundary, density, reach and physical quality gates still run.
    # Ring radii are normalized by ~0.67 m in this scene, while previous-cut
    # spacing is in meters. Keep dense nearby cuts available; physical mesh
    # clearance, rather than a large spatial exclusion, protects the floor.
    rt.AUTO_DIG_RING_RADII = [0., .45, .65, .90, 1.15, 1.40]
    rt.AUTO_DIG_RING_POINTS = [1, 8, 10, 12, 16, 20]
    rt.MULTI_SCOOP_TARGET_MIN_SPACING_M = config.dig_target_spacing_m
    rt.AUTO_COLLECT_TARGET_MIN_Z = max(float(rt.AUTO_COLLECT_TARGET_MIN_Z),
                                      float(rt.GROUND_TOP_Z) + config.min_dig_height_m)
    rt.AUTO_COLLECT_CANDIDATE_PLAN_SECONDS = max(float(rt.AUTO_COLLECT_CANDIDATE_PLAN_SECONDS), 12.)
    rt.AUTO_COLLECT_FIND_PLAN_MAX_SECONDS = max(float(rt.AUTO_COLLECT_FIND_PLAN_MAX_SECONDS), 60.)
    rt.STATE.update(dataset_scoops_target=config.scoops, dataset_scoops_min=config.scoops,
                    dataset_scoop_adaptive_stop_enabled=False)
    rt.choose_unload_landing_point_for_flat_fill = controller.choose_landing
    rt.multi_scoop_generic_task_text = controller.task
    original_episode = rt.auto_collect_one_episode
    original_begin = rt.auto_collect_begin_episode
    original_finish = rt.auto_collect_finish_episode
    original_execute = rt.execute_dig_target_ball
    original_initial_pose = rt.auto_collect_initial_pose_for_attempt
    original_find_plan = rt.auto_collect_find_plan
    original_specs = rt.dig_plan_specs_from_candidate
    original_floor_target = rt.dig_floor_safe_target_z
    original_ground_report = rt.predicted_phase_ground_report
    original_ground_ok = rt.phase_ground_ok
    original_ground_correction = rt.predicted_phase_ground_vertical_correction
    original_front_quality = rt.cut_front_edge_quality
    original_dump_plan = rt.plan_dump_pose_to_bin
    original_chain_points = rt.predicted_chain_world_points
    original_spatial_filter = rt.multi_scoop_filter_new_spatial_targets
    original_path_phase_check = rt.path_phase_check
    original_path_obstacle_check = rt.path_obstacle_check
    original_clearance_route = rt.find_clearance_route
    original_record_sample = rt.dataset_record_sample

    @wraps(original_clearance_route)
    def clearance_route(q_start, q_goal, mode, label, deadline=None, samples=None):
        # A generic free-space route inside a cut lifts out of the pile and
        # reinserts the bucket (sometimes via identical 6.5 m waypoints). Reject
        # that IK candidate so the planner searches for a safe continuous cut.
        # Ground/obstacle checks still run on every candidate and at execution.
        if str(mode) in {"approach_contact", "insert_cut", "pull_mid_cut",
                         "curl_to_hold_material", "pull_exit_cut"}:
            return None, "repeat_cut_requires_direct_collision_checked_path"
        return original_clearance_route(q_start, q_goal, mode, label, deadline=deadline, samples=samples)

    @wraps(original_record_sample)
    def record_sample(*args, **kwargs):
        recorded = original_record_sample(*args, **kwargs)
        if recorded and controller.metrics is not None:
            controller.metrics.observe()
        return recorded

    @wraps(original_spatial_filter)
    def spatial_filter(ranked_targets):
        fresh, report = original_spatial_filter(ranked_targets)
        if not fresh and ranked_targets:
            report = {**report, "repeat_target_fallback": True,
                      "preferred_spatially_exhausted": True, "spatially_exhausted": False}
            return list(ranked_targets), report
        return fresh, report

    def carry_samples(mode, samples):
        # Planning with six samples missed a truck-wall collision at 21/24.
        # Keep the dense check on loaded transit; digging has its own contact
        # checks and candidate budget.
        if "unload" in str(mode) or "clearance_route_post" in str(mode) or str(mode) == "lift_carry":
            return max(int(samples), int(rt.PATH_CHECK_SAMPLES))
        return samples

    @wraps(original_path_phase_check)
    def path_phase_check(q_start, q_goal, mode, samples=rt.PATH_CHECK_SAMPLES, deadline=None):
        return original_path_phase_check(q_start, q_goal, mode, samples=carry_samples(mode, samples), deadline=deadline)

    @wraps(original_path_obstacle_check)
    def path_obstacle_check(q_start, q_goal, mode, samples=rt.PATH_CHECK_SAMPLES, deadline=None):
        return original_path_obstacle_check(q_start, q_goal, mode, samples=carry_samples(mode, samples), deadline=deadline)

    @wraps(original_chain_points)
    def chain_points(q, end_effector="mid", reference_q=None):
        if getattr(rt, "IK_MODEL", None) is None or end_effector not in ("mid", "tip", "load", "pour"):
            return original_chain_points(q, end_effector=end_effector, reference_q=reference_q)
        return controller.kinematics.points(q, end_effector)

    @wraps(original_floor_target)
    def floor_target(effector, angle_deg):
        height, detail = original_floor_target(effector, angle_deg)
        if getattr(rt, "IK_MODEL", None) is None:
            return height, detail
        minimum = controller.floor.target_floor(effector, angle_deg)
        return max(height, minimum), {**detail, "repeat_mesh_floor_target_z": minimum}

    @wraps(original_ground_report)
    def ground_report(q, mode, reference_q=None):
        report = original_ground_report(q, mode, reference_q=reference_q)
        if getattr(rt, "IK_MODEL", None) is not None:
            report["repeat_mesh_min_z"] = controller.floor.minimum_z(q)
            report["repeat_mesh_q_rad"] = np.asarray(q).tolist()
        return report

    @wraps(original_ground_ok)
    def ground_ok(mode, report):
        ok, reason = original_ground_ok(mode, report)
        height = report.get("repeat_mesh_min_z") if isinstance(report, dict) else None
        if ok and height is not None and height < float(rt.GROUND_TOP_Z) + .005:
            return False, f"physical_bucket_mesh_below_floor:{height:.3f}"
        return ok, reason

    @wraps(original_ground_correction)
    def ground_correction(mode, report, margin=.02):
        amount, detail = original_ground_correction(mode, report, margin=margin)
        height = report.get("repeat_mesh_min_z") if isinstance(report, dict) else None
        if height is not None:
            mesh_amount = max(0., float(rt.GROUND_TOP_Z) + controller.floor.clearance - float(height))
            if mesh_amount > float(amount):
                return mesh_amount, {**detail, "limiting_point": "bucket_collision_mesh",
                                     "repeat_mesh_min_z": float(height)}
        return amount, detail

    @wraps(original_front_quality)
    def front_quality(mode, report):
        ok, reason, penalty = original_front_quality(mode, report)
        if ok or "not engaged enough" not in reason or not isinstance(report, dict):
            return ok, reason, penalty
        q = report.get("repeat_mesh_q_rad")
        if q is None:
            return ok, reason, penalty
        contact = controller.floor.sand_contact(q)
        report["repeat_mesh_sand_contact"] = contact
        if contact["depth_m"] >= float(rt.FRONT_EDGE_MIN_TIP_DEPTH):
            return True, f"physical_cut_contact depth={contact['depth_m']:.3f}", 8.
        return ok, reason, penalty

    @wraps(original_dump_plan)
    def dump_plan(q_seed=None, dump_deg=None, *args, **kwargs):
        if getattr(rt, "IK_MODEL", None) is None:
            return original_dump_plan(q_seed, dump_deg, *args, **kwargs)
        seed = rt.get_real_joint_positions() if q_seed is None else q_seed
        q, detail = controller.unloader.solve(
            rt.unload_bin_landing_point(), seed,
            dump_degrees=rt.unload_dump_target_deg() if dump_deg is None else dump_deg,
            min_release_z=kwargs.get("release_z_override"),
        )
        if q is not None:
            import copy
            detail = copy.deepcopy(detail)
            detail["drop"] = rt.unload_drop_report(q=q, reference_q=seed)
            if kwargs.get("log", True):
                rt.info_print("[REPEAT EXACT DUMP]", f"release_error={detail['drop'].get('release_xy_err')}", force_log=True)
            return q.copy(), detail
        return None, detail

    def curl_candidate_ok(row):
        # The stock curl selector prioritizes a bucket-only rotation even when
        # the Cartesian curl target was raised. Enforce the same closed-posture
        # requirement used during physical execution on the selected endpoint.
        height = (row.get("ground_report") or {}).get("load_z")
        if height is None or not np.isfinite(height) or float(height) < float(rt.GROUND_TOP_Z) + config.min_curl_height_m - .02:
            return False
        closed, _report = rt.gravity_carry_closed_for_q(row["q"])
        return bool(closed)

    rt.DIG_CURL_CANDIDATE_FILTER = curl_candidate_ok
    rt.DIG_CURL_POSE_CANDIDATES = controller.curl_candidates
    rt.DIG_PLAN_POINT_ADJUSTER = controller.adjust_plan_point

    @wraps(original_specs)
    def dig_specs(target_xyz, candidate):
        specs = original_specs(target_xyz, candidate)
        # A depleted source can put the stock low curl below the height needed
        # for the bucket's full rotation arc. Raise the Cartesian curl/escape
        # waypoints before IK and collision planning, rather than changing joint
        # commands after planning or accepting a bucket that failed to close.
        floor = float(rt.GROUND_TOP_Z)
        curl_z = next((float(spec[1][2]) for spec in specs if spec[0] == "curl_to_hold_material"), floor)
        curl_z = max(curl_z, floor + config.min_curl_height_m)
        minimum_z = {"curl_to_hold_material": curl_z, "pull_exit_cut": curl_z + .04,
                     "secure_load": curl_z + .22, "lift_carry": curl_z + .50}
        adjusted = []
        for spec in specs:
            name, point, *parameters = spec
            point = np.asarray(point, dtype=np.float32).copy()
            if name == "approach_contact" and parameters[1] is not None and getattr(rt, "IK_MODEL", None) is not None:
                point[2] = max(float(point[2]), controller.floor.target_floor(parameters[2], parameters[1]))
            if name in minimum_z:
                point[2] = max(float(point[2]), minimum_z[name])
            if name == "pull_exit_cut":
                # Curl has already sealed the bucket. Reusing the digging
                # attack angle here can select the opposite elbow branch and
                # tip the load out before secure_load. Plan a gravity-retaining
                # escape instead.
                parameters[0], parameters[1] = None, "carry"
            adjusted.append((name, point, *parameters))
        candidate["computed_curl_z"] = curl_z
        candidate["computed_exit_cut_z"] = max(float(candidate.get("computed_exit_cut_z", floor)), curl_z + .04)
        return adjusted

    @wraps(original_initial_pose)
    def initial_pose(attempt_index):
        profile = rt.AUTO_SCENE_FIXED_PROFILE
        if isinstance(profile, dict) and profile.get("initial_pose_deg"):
            return original_initial_pose(attempt_index)
        # Reset only before recording. Facing the source avoids the stock pose
        # family starting inside the truck or crossing it on the first swing.
        target = rt.task_scene_context()["pile_center"]
        swing = rt.target_to_swing_angle(target)
        pose = {"swing": float(np.degrees(swing)), "boom": 46., "arm": -62., "bucket": -20.}
        q = rt.q_from_pose_deg(pose, reference=rt.CTRL.q_cmd)
        return {"id": "repeat_pile_facing", "pose_deg": pose, "q": q.copy(),
                "q_deg": rt.q_deg_values(q, wrap_swing_for_display=True)}

    @wraps(original_find_plan)
    async def find_plan(attempt_index, ranked_targets=None):
        # The stock helper catches unload-selection errors and falls back to the
        # bed center. Validate outside that catch so a bad bed/reach is rejected.
        try:
            if getattr(rt, "IK_MODEL", None) is not None:
                # Pre-sample checks can run before the scene/initial pose is
                # applied. Anchor to the measured frames used for this dig.
                controller.kinematics.reset()
                controller.floor.reset()
                controller.unloader.reset()
                controller.kinematics.ensure()
                controller.floor.ensure()
                ref = controller.kinematics.reference
                snapshot = {"q": ref["q"].tolist(), "signs": ref["signs"].tolist(),
                            "radial": ref["radial"].tolist(),
                            "points": {k: v.tolist() for k, v in ref["points"].items()},
                            "ends": {k: v.tolist() for k, v in ref["ends"].items()},
                            "scene": rt.task_scene_context(),
                            "bounds": {name: list(rt.planner_effective_joint_bounds_rad(name)) for name in ("boom", "arm")},
                            "floor": {"angle": controller.floor.reference["angle"],
                                      "support": controller.floor.reference["support"].tolist(),
                                      "angles": controller.floor.reference["angles"],
                                      "effectors": {name: points.tolist() for name, points in
                                                    controller.floor.reference["effectors"].items()}}}
                rt.write_json_file(str(Path(rt.STATE["auto_collect_run_dir"]) / f"repeat_kinematics_{attempt_index:06d}.json"), snapshot)
            controller.choose_landing()
        except (ValueError, RuntimeError) as exc:
            rt.STATE["dig_plan_best_failure"] = {
                "failed_stage": "repeat_bed_fill", "failure_reason": str(exc),
            }
            return None, None, [{"failed_stage": "repeat_bed_fill", "failure_reason": str(exc)}]
        target, sequence, attempts = await original_find_plan(attempt_index, ranked_targets=ranked_targets)
        all_targets = rt.STATE.get("multi_scoop_ranked_targets_all") or []
        if sequence is None and all_targets and rt.STATE.get("dataset_scoop_target_history"):
            # Upstream only retries denser, previously used areas when adaptive
            # stopping is enabled. This fixed-count task still needs that
            # planning fallback; spacing must not imply material exhaustion.
            rt.dataset_record_event("repeat_dig_revisit", data={
                "scoop_index": (rt.STATE.get("multi_scoop_planning_index")
                                if rt.STATE.get("multi_scoop_planning_index") is not None
                                else int(rt.STATE.get("dataset_scoop_index", 0)) + 1),
                "reason": "fresh_targets_failed_planning", "candidates": len(all_targets),
            })
            spacing = rt.MULTI_SCOOP_TARGET_MIN_SPACING_M
            rt.MULTI_SCOOP_TARGET_MIN_SPACING_M = 0.
            try:
                target, sequence, retried = await original_find_plan(attempt_index, ranked_targets=all_targets)
                attempts = [*attempts, *retried]
            finally:
                rt.MULTI_SCOOP_TARGET_MIN_SPACING_M = spacing
        return target, sequence, attempts

    @wraps(original_episode)
    async def episode():
        controller.reset()
        return await original_episode()

    @wraps(original_begin)
    def begin(*args, **kwargs):
        meta = original_begin(*args, **kwargs)
        controller.meta = meta
        meta["repeat_baseline"] = {"version": "repeat_flat_fill_v2", "config": asdict(config),
                                   "stop_policy": "fixed_successful_cycles",
                                   "cut_policy": "continuous_collision_checked_no_clearance_detours",
                                   "mesh_floor_clearance_m": controller.floor.clearance,
                                   "return_policy": "recorded_physical_collision_checked",
                                   "ground_top_z": float(rt.GROUND_TOP_Z),
                                   "soil_backend": rt.os.environ.get("EXCAVATOR_SOIL_BACKEND", "physx")}
        controller.metrics = RepeatEpisodeMetrics(rt, controller.attempt_started)
        meta["repeat_metrics_path"] = str(Path(rt.STATE["dataset_episode_dir"]) / "repeat_metrics.json")
        rt.write_json_file(rt.STATE["dataset_meta_path"], meta)
        return meta

    @wraps(original_execute)
    async def execute(*args, **kwargs):
        return await controller.execute_cycle(original_execute, *args, **kwargs)

    @wraps(original_finish)
    def finish(meta, success, reason):
        cycles = controller.cycles
        meta["repeat_cycles"] = cycles
        meta["repeat_returns_completed"] = sum(bool(cycle["returned"]) for cycle in cycles)
        if success and (len(cycles) != config.scoops or not all(cycle["returned"] for cycle in cycles)):
            success, reason = False, "quality_rejected/repeat_incomplete_physical_returns"
        if success:
            rt.multi_scoop_set_stop("requested_repeat_cycles_completed", True, {"cycles": len(cycles)})
        result = original_finish(meta, success, reason)
        if controller.metrics is not None:
            bed = controller.ensure_bed().measure(rt.sand_particle_positions())
            metrics = controller.metrics.finalize(meta, bed)
            meta["repeat_metrics"] = metrics
            rt.write_json_file(meta["repeat_metrics_path"], metrics)
            rt.write_json_file(rt.STATE["dataset_meta_path"], meta)
            rt.append_jsonl_dataset(str(Path(rt.STATE["auto_collect_run_dir"]) / "repeat_metrics.jsonl"),
                                    {"episode_id": meta.get("episode_id"),
                                     "episode_index": meta.get("episode_index"), **metrics})
        return result

    rt.auto_collect_one_episode = episode
    rt.auto_collect_initial_pose_for_attempt = initial_pose
    rt.auto_collect_find_plan = find_plan
    rt.dig_plan_specs_from_candidate = dig_specs
    rt.dig_floor_safe_target_z = floor_target
    rt.predicted_phase_ground_report = ground_report
    rt.predicted_chain_world_points = chain_points
    rt.phase_ground_ok = ground_ok
    rt.predicted_phase_ground_vertical_correction = ground_correction
    rt.cut_front_edge_quality = front_quality
    rt.multi_scoop_filter_new_spatial_targets = spatial_filter
    rt.path_phase_check = path_phase_check
    rt.path_obstacle_check = path_obstacle_check
    rt.find_clearance_route = clearance_route
    rt.dataset_record_sample = record_sample
    rt.plan_dump_pose_to_bin = dump_plan
    rt.auto_collect_begin_episode = begin
    rt.execute_dig_target_ball = execute
    rt.auto_collect_finish_episode = finish
    rt.info_print("[REPEAT BASELINE]", f"scoops={config.scoops}",
                  "physical_return=True", "measured_flat_fill=True", force_log=True)
    return controller
