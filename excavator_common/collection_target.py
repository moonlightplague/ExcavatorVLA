"""Below-surface digging targets using the collection cut-path surface recipe."""
import math


# Keep these aligned with excavator_runtime.py's AUTO_DIG_TARGET_* constants.
CENTER_SURFACE_RADIUS_M = 0.24
PATH_SURFACE_RADIUS_M = 0.14
PATH_SURFACE_PROBES_M = (-0.34, -0.18, 0.0, 0.24, 0.46, 0.68)
CUT_DEPTH_M = 0.58
EXTRA_BITE_M = 0.12
TARGET_METHOD = 'collection_local_cut_surface_depth_v1'
SAND_INITIALIZATION_METHOD = 'temporary_rectangular_walls_during_settle_v1'


def dig_target_from_particles(points, center_xy, robot_xy, floor_z=0.0, np_module=None):
    """Return a target and provenance for collection's deepest configured cut.

    Collection observes the 90th percentile in small surface probes, takes a
    low percentile along the cut path, and subtracts cut depth plus uncertainty.
    A percentile of *all particle heights* is not a digging target.
    """
    if np_module is None:
        import numpy as np_module
    np = np_module
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    origin = np.asarray(robot_xy, dtype=np.float64)
    floor = float(floor_z)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or center.shape != (2,) or origin.shape != (2,)
            or not np.isfinite(points).all() or not np.isfinite(center).all()
            or not np.isfinite(origin).all() or not math.isfinite(floor)):
        raise ValueError('Dig target requires finite particle positions, XY centers and floor')
    points = points[points[:, 2] >= floor - 0.10]
    inward = origin - center
    distance = float(np.linalg.norm(inward))
    inward = inward / distance if distance >= 1e-6 else np.asarray([-1.0, 0.0])

    def surface_at(xy, radius):
        near = points[np.sum(np.square(points[:, :2] - xy), axis=1) <= radius * radius]
        return float(np.percentile(near[:, 2], 90.0)) if len(near) else None

    center_surface = surface_at(center, CENTER_SURFACE_RADIUS_M)
    probes = [dict(offset_m=offset,
                   surface_z=surface_at(center + inward * offset, PATH_SURFACE_RADIUS_M))
              for offset in PATH_SURFACE_PROBES_M]
    surfaces = [probe['surface_z'] for probe in probes if probe['surface_z'] is not None]
    if center_surface is None or not surfaces:
        raise ValueError('No sand particles near the digging target surface probes')
    plan_surface = max(floor + 0.08, center_surface - 0.95,
                       min(center_surface, float(np.percentile(surfaces, 25.0))))
    spread = max(0.0, float(np.percentile(surfaces, 90.0) - np.percentile(surfaces, 25.0)))
    margin = min(0.22, max(0.0, (spread - 0.03) * 0.50)
                 + (0.04 if len(surfaces) < 3 else 0.0))
    depth = CUT_DEPTH_M + EXTRA_BITE_M + margin
    target_z = max(floor + 0.04, plan_surface - depth)
    target = [float(center[0]), float(center[1]), float(target_z)]
    return target, dict(method=TARGET_METHOD, center_surface_z=center_surface,
                        plan_surface_z=plan_surface, planned_depth_m=depth,
                        surface_uncertainty_margin_m=margin, probes=probes)
