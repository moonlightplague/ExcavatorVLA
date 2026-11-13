"""Keep planning density checks in the same mass units across MPM resolutions.

These are search/preflight thresholds, not episode or export quality gates.
The reference is the previously validated fine, 32k-budget collection source.
"""
from __future__ import annotations

import math
from functools import wraps
import numpy as np


PLANNING_COUNTS = ("AUTO_PREFLIGHT_MIN_PARTICLES", "AUTO_DIG_MIN_LOCAL_PARTICLES",
                   "AUTO_DIG_MIN_SWEPT_PARTICLES")
CONTROL_MIN_COUNTS = (
    'SAND_CONTACT_BUCKET_PROGRESS_MIN', 'SAND_CONTACT_PILE_PROGRESS_MIN',
    'SAND_CONTACT_ADVANCE_BUCKET_MIN', 'SAND_CONTACT_ADVANCE_PILE_MIN',
    'SAND_CONTACT_ADVANCE_FALLBACK_BUCKET_MIN', 'SAND_CONTACT_ADVANCE_FALLBACK_PILE_MIN',
    'SAND_CONTACT_SPILL_WITHOUT_LOAD_MIN', 'SAND_CONTACT_EXIT_LOADED_BUCKET_MIN',
    'CURL_HOLD_MIN_BUCKET_PARTICLES', 'LIFT_CARRY_MIN_ABSOLUTE_BUCKET_PARTICLES',
    'LIFT_CARRY_HIGH_LOAD_RECOVERY_PARTICLES')
CONTROL_MAX_COUNTS = ('SECURE_HOLD_MAX_SPILL_PARTICLES',)


def accessible_swept_count(points, target, surface_z, inward, radius=.30):
    """Material centers the blade can reach, excluding the sheet below its tip.

    Match the upstream 76 cm capsule and upper-surface margin. Unlike its
    generic density probe, do not include a 20 cm layer below the cut target.
    This estimate never substitutes for measured source capture or delivery.
    """
    if points is None or not len(points):
        return 0
    points = np.asarray(points, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    inward = np.asarray(inward, dtype=np.float32)
    a, b = target[:2] - inward * .04, target[:2] + inward * .72
    ab = b - a
    t = np.clip((points[:, :2] - a) @ ab / max(1.e-6, float(ab @ ab)), 0., 1.)
    closest = a + t[:, None] * ab
    horizontal = np.sum((points[:, :2] - closest)**2, axis=1) <= max(.08, float(radius))**2
    above_blade = (points[:, 2] >= float(target[2])) & (points[:, 2] <= float(surface_z) + .24)
    return int(np.count_nonzero(horizontal & above_blade))


def rank_material_targets(rows, point_mass):
    """Prefer current cut inventory over ring order; keep every feasible candidate.

    This is a planning estimate, never a replacement for measured delivery.
    Upstream spatial-history, full IK and collision checks still follow.
    """
    if not math.isfinite(point_mass) or point_mass <= 0:
        raise ValueError("Target ranking requires a finite, positive material-point mass")
    for row in rows:
        count = row.get("swept_density_count")
        row["repeat_estimated_swept_mass_kg"] = None if count is None else float(count) * point_mass
    return sorted(rows, key=lambda row: (
        -float(row.get("repeat_estimated_swept_mass_kg") or 0.),
        -float(row.get("plan_surface_z") or 0.),
        -float(row.get("score") or 0.)))


def scale_planning_counts(counts, reference_mass, point_mass):
    if not all(math.isfinite(v) and v > 0 for v in (reference_mass, point_mass)):
        raise ValueError("Planning requires finite, positive material-point masses")
    scale = reference_mass / point_mass
    return {name: max(1, math.ceil(value * scale)) for name, value in counts.items()}


def complete_material_targets(rows, point_mass):
    """Try distinct cuts in mass order, preserving their original ring metadata."""
    seen, result = set(), []
    for row in rank_material_targets(rows, point_mass):
        key = tuple(round(float(v), 4) for v in row['target_xyz'])
        if key not in seen:
            seen.add(key)
            row['repeat_material_search_order'] = 'complete_unique_mass_priority_v2'
            result.append(row)
    return result


def install_material_search_order(rt):
    """Keep the first feasible mass-ranked cut through native group selection.

    Native grouping preserves input order, but compares several successes by
    motion cost within each group. Only our marked coarse rows use single-cut
    groups. Every original planner, pre-sample and execution check still runs.
    """
    planner = getattr(rt, 'auto_dataset_collect', None)
    original = getattr(planner, '_group_targets_by_ring', None)
    if original is None:
        return
    @wraps(original)
    def groups(rows):
        if rows and all(row.get('repeat_material_search_order') == 'complete_unique_mass_priority_v2'
                        for row in rows):
            return [(int(row.get('ring_index', 999)), [row]) for row in rows]
        return original(rows)
    planner._group_targets_by_ring = groups


def configure_coarse_planning(rt):
    getter = getattr(rt, "get_sand_site_api", None)
    api = getter() if getter else {}
    diagnostics = api.get("mpm_diagnostics") if api else None
    state = diagnostics() if callable(diagnostics) else {}
    if state.get("profile") != "coarse":
        return
    if not state.get("active") or state.get("error"):
        raise RuntimeError("Coarse planning requires active, healthy Newton MPM")
    from excavator_app import newton_mpm_soil as soil

    controller = soil.current()
    if controller is None:
        raise RuntimeError("Coarse planning has no live material source")
    key = (id(controller), state["points"], state["metric_point_mass_kg"])
    if getattr(rt, "_repeat_planning_resolution_key", None) == key:
        return
    original = getattr(rt, "_repeat_original_planning_counts", None)
    if original is None:
        original = {name: getattr(rt, name) for name in PLANNING_COUNTS}
        rt._repeat_original_planning_counts = original
    runtime = controller.runtime
    _, _, fine_count = soil.choose_resolution(runtime, 32000, spacing=.075, voxel=.10)
    legacy_count = len(runtime["sand_particle_points"]()[0])
    reference_count = min(fine_count, legacy_count)
    total_mass = float(state["points"]) * float(state["point_mass_kg"])
    reference_mass = total_mass / reference_count
    point_mass = float(state["metric_point_mass_kg"])
    counts = scale_planning_counts(original, reference_mass, point_mass)
    for name, value in counts.items():
        setattr(rt, name, value)
    from baseline.coarse_source import RING_RADII, RING_POINTS, SOURCE_RADIUS, TARGET_RADIUS, SURFACE_RADIUS, PICKUP_RADIUS
    rt.AUTO_DIG_RING_RADII = list(RING_RADII)
    rt.AUTO_DIG_RING_POINTS = list(RING_POINTS)
    controller = getattr(rt, 'repeat_baseline', None)
    if controller is not None:
        # Planning headroom only. The 5 mm mesh guard and path checks remain.
        controller.floor.clearance = .05
    control = None
    if getattr(getattr(controller, 'config', None), 'acceptance_profile', None) == 'coarse_mass':
        # These contact/progress/loaded controls were authored for the native
        # granular source. Preserve their kg meaning using its real grain mass;
        # do not require hundreds of much larger MPM quadrature points.
        native_mass = float(runtime['PARTICLE_MASS'])
        original_control = getattr(rt, '_repeat_original_control_counts', None)
        if original_control is None:
            original_control = {name: getattr(rt, name) for name in (*CONTROL_MIN_COUNTS, *CONTROL_MAX_COUNTS)
                                if hasattr(rt, name)}
            rt._repeat_original_control_counts = original_control
        resolved_control = scale_planning_counts({k:v for k,v in original_control.items()
                                                if k in CONTROL_MIN_COUNTS}, native_mass, point_mass)
        resolved_control.update({k:max(0, math.floor(v * native_mass / point_mass))
                                 for k,v in original_control.items() if k in CONTROL_MAX_COUNTS})
        for name, value in resolved_control.items():
            setattr(rt, name, value)
        control = dict(policy='native_material_mass_equivalent_controls_v1',
            reference_point_mass_kg=native_mass, original_counts=original_control,
            resolved_counts=resolved_control, angular_and_collision_limits_changed=False)
    report = dict(policy="mass_equivalent_fine32k_planning_v1",
                  target_priority="complete_unique_mass_priority_v2",
                  swept_estimator="material_centers_at_or_above_tip_target_v1",
                  source_radius_m=SOURCE_RADIUS, target_radius_m=TARGET_RADIUS,
                  surface_radius_m=SURFACE_RADIUS, pickup_radius_m=list(PICKUP_RADIUS),
                  mesh_planning_clearance_m=.05,
                  material_controls=control,
                  reference_metric_points=reference_count,
                  reference_point_mass_kg=reference_mass, metric_point_mass_kg=point_mass,
                  original_counts=original, resolved_counts=counts,
                  episode_quality_thresholds_changed=False)
    if rt.STATE.get("repeat_planning_resolution") != report:
        rt.info_print("[REPEAT MPM PLANNING]", str(report), force_log=True)
    rt.STATE["repeat_planning_resolution"] = report
    rt._repeat_planning_resolution_key = key
