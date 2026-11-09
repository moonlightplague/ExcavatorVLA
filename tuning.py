"""Explicit workspace and quality policies for long Newton MPM episodes."""
from __future__ import annotations

import math


def configure_workspace(rt, profile, radius_min=6.7, radius_max=7.0, approach="side"):
    if profile == "broad":
        return
    if profile != "reachable":
        raise ValueError(f"Unknown repeat workspace: {profile}")
    # Sample different scenes, while reserving reach for a centered opening and
    # a loaded carry above the truck walls. Do not retry or discard failed seeds.
    rt.AUTO_SCENE_TRUCK_RANDOM_RADIUS_RANGE = (8.0, 10.8)
    rt.AUTO_SCENE_TRUCK_RANDOM_ANGLE_DEG_RANGE = (-170.0, -120.0)
    rt.AUTO_SCENE_TRUCK_REAR_TO_ROBOT_YAW_RANGE_DEG = 90.0 if approach == "side" else 25.0
    rt.AUTO_SCENE_ROBOT_RANDOM_YAW_DEG_RANGE = (-15.0, 15.0)
    rt.AUTO_SCENE_SAND_RANDOM_RADIUS_RANGE = (6.0, 6.7)
    rt.AUTO_SCENE_SAND_RANDOM_ANGLE_DEG_RANGE = (84.0, 96.0)

    def sample_unload(rng, _radius_range, _angle_range):
        import numpy as np
        center = np.asarray(rt.get_swing_xy_center(), dtype=float)[:2]
        radius = math.sqrt(float(rng.uniform(radius_min**2, radius_max**2)))
        angle = math.radians(float(rng.uniform(-100., -80.)))
        xy = np.asarray(center + radius*np.array([math.cos(angle), math.sin(angle)]), dtype=np.float32)
        # Existing scene legality uses world-space polar coordinates. Preserve
        # that contract while sampling the actual arm workspace around its pivot.
        return xy, rt.auto_scene_xy_radius(xy), rt.auto_scene_xy_angle_deg(xy)

    rt.auto_scene_sample_truck_unload_xy = sample_unload

    def landing_status(xy, random_truck=None, soft_margin=0.):
        import numpy as np
        radius = float(np.linalg.norm(np.asarray(xy)[:2] - np.asarray(rt.get_swing_xy_center())[:2]))
        margin = max(.5, float(soft_margin))
        lo, hi = radius_min-margin, radius_max+margin
        return dict(radius_m=radius, min_m=lo, max_m=hi, ok=lo <= radius <= hi,
                    source="repeat_measured_swing_joint")

    rt.unload_landing_radius_status = landing_status
    if approach == "side":
        def side_yaw(rng, unload_xy, _angle, baseline):
            previous = rt.AUTO_SCENE_TRUCK_REAR_TO_ROBOT_YAW_RANGE_DEG
            rt.AUTO_SCENE_TRUCK_REAR_TO_ROBOT_YAW_RANGE_DEG = 0.
            try:
                centered = rt.auto_scene_truck_rear_to_robot_yaw_for_unload(rng, unload_xy, baseline)
            finally:
                rt.AUTO_SCENE_TRUCK_REAR_TO_ROBOT_YAW_RANGE_DEG = previous
            offset = float(rng.uniform(70., 85.)) * (-1 if rng.random() < .5 else 1)
            return rt.wrap_deg_180(centered + offset)
        rt.auto_scene_sample_truck_yaw_for_unload = side_yaw


def install_quality_policy(rt, config):
    snapshot = getattr(rt, "quality_gate_config_snapshot", None)
    if snapshot is not None:
        def quality_snapshot():
            return {**snapshot(), "multi_scoop_min_score": rt.MULTI_SCOOP_MIN_SCORE,
                    "multi_scoop_max_spill_ratio": rt.MULTI_SCOOP_MAX_SPILL_RATIO,
                    "repeat_acceptance_profile": config.acceptance_profile,
                    "repeat_episode_min_mean_score": config.episode_min_score,
                    "repeat_episode_max_weighted_spill_ratio": config.episode_max_spill_ratio}
        rt.quality_gate_config_snapshot = quality_snapshot
    if config.acceptance_profile == "strict":
        return
    original = rt.multi_scoop_cycle_report

    def cycle_report(execution_success, reason=""):
        event = rt.dataset_record_event
        def strict_event(name, *args, **kwargs):
            return event("repeat_strict_scoop_end" if name == "scoop_end" else name, *args, **kwargs)
        rt.dataset_record_event = strict_event
        try:
            report = original(execution_success, reason)
        finally:
            rt.dataset_record_event = event
        report["strict_success"] = bool(report["success"])
        report["strict_reason"] = report.get("reason", "")
        failures = [item for item in report.get("reason", "").split(";") if item]
        controller = getattr(rt, "repeat_baseline", None)
        if controller is not None:
            cycles = controller.cycles
            measured = cycles[-1].get("metrics") if cycles and cycles[-1].get("scoop_index") == report["scoop_index"] else None
            if measured is None or any(measured.get(k) is None for k in
                    ("payload_particles", "scooped_particles", "spillage_particles")):
                failures.append("source_cycle_metrics_unavailable")
            else:
                captured, delivered, spilled = (int(measured[k]) for k in
                    ("scooped_particles", "payload_particles", "spillage_particles"))
                ratio = spilled/max(1, captured)
                source_score = (35*min(1., captured/25) + 30*min(1., delivered/max(1,captured))
                    + 15*min(1., report.get("lift_bucket_from_pile_particles", 0)/max(1,captured))
                    + 5 + (10 if execution_success else 0) - 20*ratio)
                report.update(source_captured_particles=captured, source_delivered_particles=delivered,
                              source_spillage_particles=spilled, source_spill_ratio=ratio, source_score=source_score)
                if captured < 5:
                    failures.append(f"low_source_capture:{captured}")
                if delivered < 3:
                    failures.append(f"low_source_delivery:{delivered}")
                elif any(item.startswith("low_bin_gain:") for item in failures):
                    # New source IDs reached the truck even if an older deposit
                    # moved out. Keep that net loss for the episode efficiency gate.
                    failures = [item if not item.startswith("low_bin_gain:") else
                                "net_truck_"+item for item in failures]
        warnings = [item for item in failures if item.startswith(("high_spill_ratio:", "score_below_minimum:", "net_truck_low_bin_gain:"))]
        failures = [item for item in failures if item not in warnings]
        # Efficiency is evaluated over the complete episode. Actual capture,
        # positive delivery, execution and freeze checks still apply per scoop.
        if not all(math.isfinite(float(report[key])) for key in ("score", "spill_ratio")):
            failures.append("nonfinite_cycle_quality")
        if not execution_success and not failures:
            failures.append(str(reason or "execution_failed"))
        report.update(success=not failures, reason=";".join(failures),
                      quality_warnings=warnings, acceptance_profile="episode")
        rt.STATE["dataset_scoops_completed"] = sum(bool(row.get("success"))
            for row in rt.STATE["dataset_scoop_results"])
        event("scoop_end", f"scoop_index={report['scoop_index']}; success={report['success']}; "
              f"strict_success={report['strict_success']}; score={report['score']:.1f}; "
              f"new_delivery={report.get('source_delivered_particles')}; "
              f"net_bin_gain={report.get('bin_gain_from_pile_particles')}; reason={report['reason']}",
              data={key:report[key] for key in ("success", "strict_success", "strict_reason", "quality_warnings", "acceptance_profile")})
        rt.info_print("[REPEAT QUALITY]", f"scoop={report['scoop_index'] + 1}",
                      f"accepted={report['success']} strict={report['strict_success']}",
                      f"score={report.get('source_score', report['score']):.1f} spill={report.get('source_spill_ratio', report['spill_ratio']):.3f}",
                      f"warnings={warnings}", force_log=True)
        return report

    rt.multi_scoop_cycle_report = cycle_report


def configure_seed_sequence(rt, start):
    if start is None:
        return
    if getattr(rt, "AUTO_SCENE_FIXED_PROFILE", None) is not None or getattr(rt, "AUTO_SCENE_REPLAY_SEED", None) is not None:
        raise ValueError("Consecutive scene seeds cannot be combined with fixed-layout replay")
    original = rt.auto_scene_seed_context

    def seed_context(attempt):
        return {**original(attempt), "scene_seed": start + int(attempt) - 1,
                "scene_seed_source": "repeat_explicit_sequence", "scene_seed_namespace": f"repeat-{start}"}

    rt.auto_scene_seed_context = seed_context


def install_workspace_probe(rt):
    """Record preparation/first-plan feasibility without executing training actions."""
    from pathlib import Path
    original = rt.auto_collect_find_plan

    async def probe(attempt, ranked_targets=None):
        target, sequence, details = await original(attempt, ranked_targets=ranked_targets)
        selection = rt.repeat_baseline.selected
        row = dict(attempt=attempt, scene_seed=rt.auto_scene_seed_context(attempt)["scene_seed"],
                   feasible=bool(sequence), selection=selection,
                   scene=rt.STATE.get("auto_scene_last_randomization"),
                   kinematics=rt.repeat_baseline.kinematics.reference,
                   failure=rt.STATE.get("dig_plan_best_failure") if not sequence else None)
        rt.append_jsonl_dataset(str(Path(rt.STATE["auto_collect_run_dir"])/"workspace_probe.jsonl"),
                                rt.json_sanitize(row))
        rt.info_print("[REPEAT WORKSPACE PROBE]", f"seed={row['scene_seed']} feasible={bool(sequence)}", force_log=True)
        return target, None, details

    rt.auto_collect_find_plan = probe


def install_source_identity(rt):
    """The Newton controller contains only material points born in this source.

    Avalanche during initial settling can put most of those IDs outside the
    authored dig polygon. Spatial membership is not their material provenance.
    Exclude any soil already in the bucket/truck when the episode starts.
    """
    original = getattr(rt, "capture_initial_pile_particle_ids", None)
    if original is None:
        return

    def capture():
        import numpy as np
        if rt.os.environ.get("EXCAVATOR_SOIL_BACKEND") != "newton_mpm":
            return original()
        points = rt.sand_particle_positions()
        if points is None or not len(points) or not np.isfinite(points).all():
            raise ValueError("Newton source identity requires finite material-point positions")
        _, bucket, truck = rt.sand_region_masks(points)
        mask = ~(np.asarray(bucket, dtype=bool) | np.asarray(truck, dtype=bool))
        ids = set(np.flatnonzero(mask).tolist())
        rt.STATE.update(dataset_initial_pile_particle_ids=ids,
                        dataset_initial_pile_particle_mask=mask,
                        dataset_initial_pile_particle_count=len(ids),
                        repeat_source_identity="newton_source_material_ids_excluding_initial_bucket_and_truck")
        return ids

    rt.capture_initial_pile_particle_ids = capture


def episode_quality(results, config, final_transfer=None):
    source_ids = bool(results) and all("source_score" in row for row in results)
    lift = sum(int(row.get("source_captured_particles" if source_ids else "lift_bucket_from_pile_particles", 0)) for row in results)
    spill = sum(int(row.get("source_spillage_particles" if source_ids else "spill_from_lift_particles", 0)) for row in results)
    mean_score = sum(float(row.get("source_score" if source_ids else "score", 0)) for row in results) / max(1, len(results))
    ratio = spill / max(1, lift)
    failures = []
    if len(results) != config.scoops or not all(row.get("success") for row in results):
        failures.append("incomplete_successful_cycles")
    if not math.isfinite(ratio) or ratio > config.episode_max_spill_ratio:
        failures.append(f"episode_spill_ratio:{ratio:.3f}")
    if not math.isfinite(mean_score) or mean_score < config.episode_min_score:
        failures.append(f"episode_mean_score:{mean_score:.1f}")
    final_ratio = (final_transfer or {}).get("spillage_fraction")
    if source_ids and (final_ratio is None or not math.isfinite(float(final_ratio))
                       or float(final_ratio) > config.episode_max_spill_ratio):
        failures.append(f"final_source_spill_ratio:{final_ratio}")
    result = dict(success=not failures, reason=";".join(failures),
                mean_scoop_score=mean_score, weighted_spill_ratio=ratio,
                min_mean_score=config.episode_min_score, max_spill_ratio=config.episode_max_spill_ratio,
                strict_scoops=sum(bool(row.get("strict_success", row.get("success"))) for row in results))
    if source_ids:
        result.update(measurement="source_material_ids", final_source_spill_ratio=final_ratio)
    return result
