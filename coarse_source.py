"""Plan against current Newton source material after its initial avalanche.

The authored pile box describes generation, not the settled soil boundary.
These process-local hooks leave the simulator, source IDs and quality gates
unchanged. Fine MPM and PhysX continue through the upstream functions.
"""
from __future__ import annotations

from functools import wraps
import numpy as np

from baseline.soil_planning import accessible_swept_count


SOURCE_RADIUS = 3.
TARGET_RADIUS = 2.85
SURFACE_RADIUS = 3.75
PICKUP_RADIUS = (7.2, 8.4)
RING_RADII = [0., .45, .65, .90, 1.15, 1.40, 1.8, 2.2, 2.6, 3., 3.4, 3.8]
RING_POINTS = [1, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44]


def current_source_points(points, center, bucket, truck, source_mask=None):
    """Select measured material, never move it or synthesize a pile surface."""
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite Newton source material")
    keep = np.linalg.norm(points[:, :2] - np.asarray(center, dtype=np.float32)[:2], axis=1) <= SOURCE_RADIUS
    keep &= ~(np.asarray(bucket, dtype=bool) | np.asarray(truck, dtype=bool))
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != (len(points),):
            raise ValueError("Newton source identity and snapshot point counts disagree")
        keep &= source_mask
    return points[keep]


def install_current_source_planning(rt):
    required = ("get_sand_snapshot", "auto_dig_target_sand_region_check",
                "sand_surface_query_at_xy", "auto_dig_swept_density_count")
    if not all(callable(getattr(rt, name, None)) for name in required):
        return
    original_snapshot, original_region, original_surface, original_sweep = (
        getattr(rt, name) for name in required)

    def coarse():
        return bool(rt.STATE.get("repeat_planning_resolution"))

    @wraps(original_snapshot)
    def source_snapshot(*args, **kwargs):
        snapshot = original_snapshot(*args, **kwargs)
        if not coarse() or snapshot.get("repeat_avalanche_source"):
            return snapshot
        from excavator_app import newton_mpm_soil
        soil = newton_mpm_soil.current()
        if soil is None or soil.closed or soil.error:
            return snapshot
        points = snapshot.get("points")
        if points is None or not len(points):
            return snapshot
        ctx = rt.task_scene_context()
        center = np.asarray(ctx["pile_center"], dtype=np.float32)
        _, bucket, truck = rt.sand_region_masks(points)
        source = current_source_points(points, center, bucket, truck,
                                       rt.STATE.get("dataset_initial_pile_particle_mask"))
        expanded = {**ctx, "pile_radius": [SOURCE_RADIUS, SOURCE_RADIUS]}
        settle = rt.sand_settle_status(points=source, ctx=expanded)
        settled = bool(settle.get("ok"))
        result = {**snapshot, "ctx": expanded, "settle": settle,
                  "settled_points": source if settled else None,
                  "settled_count": len(source) if settled else 0,
                  "spatial_index": rt.build_sand_spatial_index(source, ctx=expanded)
                      if settled and len(source) else None,
                  "repeat_avalanche_source": True}
        rt.STATE["sand_snapshot_last"] = result
        rt.STATE["repeat_avalanche_extent"] = dict(radius_m=SOURCE_RADIUS,
            source_points=len(source), authored_pile_center=center.tolist(),
            source="current_Newton_birth_material_excluding_bucket_and_truck")
        return result

    @wraps(original_region)
    def source_region(x, y, snapshot=None, ctx=None, real_surface_z=None):
        if not coarse():
            return original_region(x, y, snapshot=snapshot, ctx=ctx, real_surface_z=real_surface_z)
        radius = rt.target_radius_from_swing_center([x, y, float(rt.GROUND_TOP_Z)])
        if not PICKUP_RADIUS[0] <= radius <= PICKUP_RADIUS[1]:
            return False, "outside_loaded_pickup_search_window", {"measured_swing_radius_m": radius}
        ok, reason, detail = original_region(x, y, snapshot=snapshot, ctx=ctx, real_surface_z=real_surface_z)
        if ok or reason != "outside_sand_polygon" or not (snapshot or {}).get("repeat_avalanche_source"):
            return ok, reason, detail
        center = np.asarray(rt.task_scene_context()["pile_center"], dtype=float)[:2]
        if np.linalg.norm(np.array([x, y]) - center) > TARGET_RADIUS:
            return False, "outside_measured_source_search_radius", detail
        surface = real_surface_z if real_surface_z is not None else rt.sand_snapshot_surface_height(snapshot, x, y)
        local = rt.sand_snapshot_density_count(snapshot, x, y, float(rt.AUTO_DIG_DENSITY_RADIUS))
        if surface is None or local < int(rt.AUTO_DIG_MIN_LOCAL_PARTICLES):
            return False, "no_actual_avalanched_source_at_target", detail
        return True, "actual_avalanched_source_material", {**detail,
            "source_extent": "current_birth_material_within_3m",
            "local_particle_count": local, "real_surface_z": float(surface)}

    @wraps(original_surface)
    def source_surface(x, y):
        if coarse():
            center = np.asarray(rt.task_scene_context()["pile_center"], dtype=float)[:2]
            if np.linalg.norm(np.array([x, y]) - center) <= SURFACE_RADIUS:
                snapshot = rt.get_sand_snapshot(force=False, label="actual_source_surface", max_age=.75)
                if snapshot.get("repeat_avalanche_source"):
                    surface = rt.sand_snapshot_surface_height(snapshot, x, y)
                    if surface is not None:
                        return float(surface), "actual_Newton_source_snapshot"
                    # Existing contact checks reject the explicit fallback as
                    # evidence of soil. Bare ground has zero sand depth.
                    return float(rt.GROUND_TOP_Z), "fallback_floor_no_current_source_material"
        return original_surface(x, y)

    @wraps(original_sweep)
    def source_sweep(target, surface_z, particles=None, snapshot=None):
        if not coarse():
            return original_sweep(target, surface_z, particles=particles, snapshot=snapshot)
        if isinstance(snapshot, dict):
            particles = snapshot.get("settled_points")
        return accessible_swept_count(particles, target, surface_z, rt.dig_direction_unit(target),
                                      rt.AUTO_DIG_SWEEP_RADIUS)

    rt.get_sand_snapshot = source_snapshot
    rt.auto_dig_target_sand_region_check = source_region
    rt.sand_surface_query_at_xy = source_surface
    rt.auto_dig_swept_density_count = source_sweep
