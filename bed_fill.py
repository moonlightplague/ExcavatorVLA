"""Measured, truck-aligned low-fill placement without Isaac Sim imports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def inside_polygon(xy, polygon):
    """Ray crossing, including the boundary, for either polygon winding."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    inside = np.zeros(len(xy), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        relative = xy - a
        cross = relative[:, 0] * edge[1] - relative[:, 1] * edge[0]
        boundary |= (np.abs(cross) <= 1e-7) & (relative @ edge >= -1e-7) & (
            relative @ edge <= edge @ edge + 1e-7
        )
        if abs(edge[1]) > 1e-12:
            crossing = ((a[1] > xy[:, 1]) != (b[1] > xy[:, 1])) & (
                xy[:, 0] < a[0] + (xy[:, 1] - a[1]) * edge[0] / edge[1]
            )
            inside ^= crossing
    return inside | boundary


@dataclass(frozen=True)
class FillConfig:
    long_cells: int = 5
    short_cells: int = 3
    wall_margin_m: float = 0.60
    revisit_penalty_m: float = 0.35
    footprint_radius_m: float = 0.65

    def __post_init__(self):
        if not 1 <= self.long_cells <= 20 or not 1 <= self.short_cells <= 20:
            raise ValueError("Bed grid dimensions must be between 1 and 20.")
        for name in ("wall_margin_m", "revisit_penalty_m", "footprint_radius_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class BedFillPlanner:
    """Equal-area cells in the bed frame, recomputed from actual soil positions.

    The inset polygon is used for landing safety, while the complete polygon is
    used for measurements. Empty cells have floor height. No particles are moved
    or fabricated by this planner. Visit penalties break near-empty ties and
    encourage broad coverage; actual surface height remains the main cost.
    """

    def __init__(self, polygon, safe_polygon, floor_z, wall_z, config=FillConfig()):
        self.config = config
        self.polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
        self.safe_polygon = np.asarray(safe_polygon, dtype=float).reshape(-1, 2)
        self.floor_z, self.wall_z = float(floor_z), float(wall_z)
        if (len(self.polygon) < 3 or len(self.safe_polygon) < 3
                or not np.isfinite(self.polygon).all()
                or not np.isfinite(self.safe_polygon).all()
                or not np.isfinite([self.floor_z, self.wall_z]).all()
                or self.wall_z <= self.floor_z):
            raise ValueError("Invalid truck-bed geometry.")
        edges = np.roll(self.polygon, -1, axis=0) - self.polygon
        major = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        if np.linalg.norm(major) < 1e-6:
            raise ValueError("Degenerate truck-bed polygon.")
        major = major / np.linalg.norm(major)
        # Keep cell IDs stable when USD happens to reverse the polygon winding.
        if major[np.argmax(np.abs(major))] < 0:
            major = -major
        self.axes = np.column_stack((major, [-major[1], major[0]]))
        local = self.polygon @ self.axes
        self.low, self.high = local.min(axis=0), local.max(axis=0)
        self.shape = np.array([config.long_cells, config.short_cells])
        self.cell_size = (self.high - self.low) / self.shape
        if np.any(self.cell_size < 1e-5):
            raise ValueError("Degenerate truck-bed cell size.")
        xs = self.low[0] + (np.arange(config.long_cells) + .5) * self.cell_size[0]
        ys = self.low[1] + (np.arange(config.short_cells) + .5) * self.cell_size[1]
        centers = np.array([(x, y) for x in xs for y in ys]) @ self.axes.T
        centers = centers[inside_polygon(centers, self.polygon)]
        # Keep measurement and landing cells aligned. Uniformly resampling the
        # smaller inset can map several landings into the same measurement cell.
        # Instead project each original cell center into the safe polygon.
        safe_center = self.safe_polygon.mean(axis=0)
        projected = []
        for xy in centers:
            if not inside_polygon([xy], self.safe_polygon)[0]:
                nearest = []
                for a, b in zip(self.safe_polygon, np.roll(self.safe_polygon, -1, axis=0)):
                    edge = b - a
                    t = np.clip(np.dot(xy - a, edge) / max(1e-12, np.dot(edge, edge)), 0, 1)
                    nearest.append(a + t * edge)
                xy = min(nearest, key=lambda point: np.linalg.norm(point - xy))
                xy = xy + 1e-5 * (safe_center - xy)
            if not any(np.linalg.norm(xy - old) < 1e-6 for old in projected):
                projected.append(xy)
        self.centers = np.asarray(projected).reshape(-1, 2)
        if not len(self.centers):
            raise ValueError("No landing cells fit inside the safe truck bed.")
        self.visits = []

    def measure(self, points):
        points = np.asarray(points if points is not None else [], dtype=float).reshape(-1, 3)
        keep = np.isfinite(points).all(axis=1)
        keep &= (points[:, 2] >= self.floor_z - .05) & (points[:, 2] <= self.wall_z + .25)
        keep &= inside_polygon(points[:, :2], self.polygon)
        points = points[keep]
        ids = np.clip(((points[:, :2] @ self.axes - self.low) / self.cell_size).astype(int),
                      0, self.shape - 1)
        flat = ids[:, 0] * self.shape[1] + ids[:, 1]
        total = int(np.prod(self.shape))
        counts = np.bincount(flat, minlength=total)
        heights = np.full(total, self.floor_z)
        for index in np.flatnonzero(counts):
            heights[index] = max(self.floor_z, float(np.percentile(points[flat == index, 2], 90)))
        depth = heights - self.floor_z
        valid_centers = np.array([
            (x + .5, y + .5) for x in range(self.shape[0]) for y in range(self.shape[1])
        ]) * self.cell_size + self.low
        valid = inside_polygon(valid_centers @ self.axes.T, self.polygon)
        return {
            "particle_count": int(len(points)),
            "cell_counts": counts.tolist(),
            "cell_surface_z_m": heights.tolist(),
            "occupied_fraction": float(np.mean(counts[valid] > 0)),
            "surface_std_m": float(np.std(depth[valid])),
            "surface_range_m": float(np.ptp(depth[valid])),
            "mean_depth_m": float(np.mean(depth[valid])),
            "grid_shape": self.shape.tolist(),
            "axes_world_xy": self.axes.tolist(),
            "floor_z_m": self.floor_z,
            "measurement": "p90_surface_including_empty_cells_at_floor",
        }

    def rank(self, points, bucket_xy=None):
        measurement = self.measure(points)
        counts = np.asarray(measurement["cell_counts"])
        heights = np.asarray(measurement["cell_surface_z_m"])
        bucket_xy = np.mean(self.centers, axis=0) if bucket_xy is None else np.asarray(bucket_xy)[:2]
        rows = []
        for index, xy in enumerate(self.centers):
            cell = np.clip(((xy @ self.axes - self.low) / self.cell_size).astype(int), 0, self.shape - 1)
            flat = int(cell[0] * self.shape[1] + cell[1])
            distance = float(np.linalg.norm(xy - bucket_xy))
            center_norm = float(np.linalg.norm((xy @ self.axes - .5 * (self.low + self.high))
                                               / (.5 * (self.high - self.low))))
            revisit = sum(np.exp(-.5 * (np.linalg.norm(xy - old)
                                       / self.config.footprint_radius_m) ** 2) for old in self.visits)
            score = (float(heights[flat] - self.floor_z)
                     + .15 * float(counts[flat]) / max(1, int(counts.max()))
                     + self.config.revisit_penalty_m * revisit
                     + .08 * center_norm + .005 * distance)
            rows.append({
                "landing": [float(xy[0]), float(xy[1]), self.wall_z + .06],
                "cell_id": flat, "candidate_id": index,
                "cell_height": float(heights[flat]), "cell_count": int(counts[flat]),
                "motion_dist": distance, "center_norm": center_norm,
                "revisit_cost": float(revisit), "score": score,
                "source": "repeat_measured_low_fill",
            })
        return sorted(rows, key=lambda row: (row["score"], row["candidate_id"])), measurement

    def commit(self, landing):
        self.visits.append(np.asarray(landing, dtype=float)[:2].copy())
