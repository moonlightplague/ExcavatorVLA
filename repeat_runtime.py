"""Small runtime adapter: reuse upstream physics, planner, recorder and gates.

Installed after the upstream bootstrap loads the runtime, before collection.
Runtime changes are confined to this process. The shared exporter separately
recognizes the adapter's explicit physical-return supervision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path

import numpy as np

from baseline.bed_fill import BedFillPlanner, FillConfig


@dataclass(frozen=True)
class RepeatConfig:
    scoops: int = 3
    return_plan_seconds: float = 12.0
    settle_seconds: float = 1.0
    fill: FillConfig = field(default_factory=FillConfig)
    dig_target_spacing_m: float = .55
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
        self.reset()

    def reset(self):
        self.bed = None
        self.cycles = []
        self.before = None
        self.selected = None
        self.meta = None

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

    async def return_arm(self, q_goal):
        rt = self.rt
        q_start = rt.sync_motion_start_q("repeat_return_pre_dig")
        rt.clear_planning_runtime_caches("repeat_return_actual_pose")
        deadline = rt.time.time() + self.config.return_plan_seconds
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
        directory = Path(rt.STATE["dataset_episode_dir"])
        rt.append_jsonl_dataset(str(directory / "repeat_cycles.jsonl"), cycle)
        rt.dataset_record_event("repeat_cycle_end", data=cycle)
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
    rt.AUTO_DIG_RING_RADII = [0., .45, .90, 1.15, 1.40]
    rt.AUTO_DIG_RING_POINTS = [1, 8, 12, 16, 20]
    rt.MULTI_SCOOP_TARGET_MIN_SPACING_M = config.dig_target_spacing_m
    rt.AUTO_COLLECT_TARGET_MIN_Z = max(float(rt.AUTO_COLLECT_TARGET_MIN_Z),
                                      float(rt.GROUND_TOP_Z) + config.min_dig_height_m)
    rt.AUTO_COLLECT_CANDIDATE_PLAN_SECONDS = max(float(rt.AUTO_COLLECT_CANDIDATE_PLAN_SECONDS), 8.)
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
            if name in minimum_z:
                point[2] = max(float(point[2]), minimum_z[name])
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
            controller.choose_landing()
        except (ValueError, RuntimeError) as exc:
            rt.STATE["dig_plan_best_failure"] = {
                "failed_stage": "repeat_bed_fill", "failure_reason": str(exc),
            }
            return None, None, [{"failed_stage": "repeat_bed_fill", "failure_reason": str(exc)}]
        return await original_find_plan(attempt_index, ranked_targets=ranked_targets)

    @wraps(original_episode)
    async def episode():
        controller.reset()
        return await original_episode()

    @wraps(original_begin)
    def begin(*args, **kwargs):
        meta = original_begin(*args, **kwargs)
        controller.meta = meta
        meta["repeat_baseline"] = {"version": "repeat_flat_fill_v1", "config": asdict(config),
                                   "stop_policy": "fixed_successful_cycles",
                                   "return_policy": "recorded_physical_collision_checked",
                                   "ground_top_z": float(rt.GROUND_TOP_Z),
                                   "soil_backend": rt.os.environ.get("EXCAVATOR_SOIL_BACKEND", "physx")}
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
        return original_finish(meta, success, reason)

    rt.auto_collect_one_episode = episode
    rt.auto_collect_initial_pose_for_attempt = initial_pose
    rt.auto_collect_find_plan = find_plan
    rt.dig_plan_specs_from_candidate = dig_specs
    rt.auto_collect_begin_episode = begin
    rt.execute_dig_target_ball = execute
    rt.auto_collect_finish_episode = finish
    rt.info_print("[REPEAT BASELINE]", f"scoops={config.scoops}",
                  "physical_return=True", "measured_flat_fill=True", force_log=True)
    return controller
