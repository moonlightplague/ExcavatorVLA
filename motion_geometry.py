"""Rigid bucket floor clearance using the authored mesh, not tip markers."""
from __future__ import annotations

import numpy as np


def hull_xz(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=float).reshape(-1, 2))))
    if len(points) < 3:
        return np.asarray(points)
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    halves = []
    for sequence in (points, points[::-1]):
        half = []
        for point in sequence:
            while len(half) > 1 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.extend(half[:-1])
    return np.asarray(halves)


def rotated_min_z(points_xz, angle):
    points = np.asarray(points_xz)
    return float(np.min(points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)))


class RigidBucketKinematics:
    """Exact 3D serial-chain FK, retaining the markers' lateral offsets.

    The legacy planar FK discards 0.30 m of lateral pour-marker offset and
    0.18 m at the tip. Those are larger than the unload/floor tolerances.
    Anchor to measured joint frames and use the calibrated joint signs.
    """
    def __init__(self, rt):
        self.rt = rt
        self.reference = None

    def reset(self):
        self.reference = None

    def ensure(self):
        if self.reference is not None:
            return
        rt = self.rt
        points = rt.get_ik_world_points("load")
        radial = np.asarray(points["arm"])[:2] - np.asarray(points["boom"])[:2]
        radial /= np.linalg.norm(radial)
        self.reference = {
            "q": np.asarray(rt.get_real_joint_positions(), dtype=float).copy(),
            "signs": np.asarray(rt.IK_MODEL["signs"]), "radial": radial,
            "points": {name: np.asarray(points[name], dtype=float).copy()
                       for name in ("swing", "boom", "arm", "bucket")},
            "ends": {name: np.asarray(rt.bucket_point_world(name), dtype=float).copy()
                     for name in ("tip", "mid", "load", "pour")},
        }

    def points(self, q, effector):
        self.ensure()
        ref, indices = self.reference, self.rt.CTRL.name_to_idx
        q = np.asarray(q)
        swing = float(q[indices["swing"]] - ref["q"][indices["swing"]])
        angles = np.array([q[indices[name]] - ref["q"][indices[name]]
                           for name in ("boom", "arm", "bucket")]) * ref["signs"]
        radial = ref["radial"]
        tangent = np.array([-radial[1], radial[0]])
        rotation = np.array([[np.cos(swing), -np.sin(swing)], [np.sin(swing), np.cos(swing)]])
        target_radial, target_tangent = rotation @ radial, rotation @ tangent
        def rotate(vector, pitch):
            x, y, z = vector[:2] @ radial, vector[:2] @ tangent, vector[2]
            rx, rz = np.cos(pitch) * x - np.sin(pitch) * z, np.sin(pitch) * x + np.cos(pitch) * z
            return np.array([*(target_radial * rx + target_tangent * y), rz])
        points = ref["points"]
        boom = points["swing"] + rotate(points["boom"] - points["swing"], 0.)
        arm = boom + rotate(points["arm"] - points["boom"], angles[0])
        bucket = arm + rotate(points["bucket"] - points["arm"], angles[0] + angles[1])
        end = bucket + rotate(ref["ends"][effector] - points["bucket"], float(np.sum(angles)))
        return [boom, arm, bucket, end]


class BucketFloor:
    def __init__(self, rt, clearance=.035):
        self.rt, self.clearance = rt, clearance
        self.reference = None

    def reset(self):
        self.reference = None

    def ensure(self):
        if self.reference is not None:
            return
        rt = self.rt
        chunks, errors = [], []
        # The FBX display mesh and the load-volume marker are not colliders.
        # Including them raises the cutting floor by about 0.3 m and makes
        # later cuts skim above the depleted pile.
        for prim in rt.bucket_collision_mesh_candidates():
            if not rt.prim_collision_enabled(prim):
                continue
            source, _paths, failures = rt.bucket_load_source_mesh_local(str(prim.GetPath()))
            errors.extend(failures)
            if source is not None:
                chunks.append(source[0])
        if not chunks:
            raise RuntimeError(f"Cannot read physical bucket mesh: {errors}")
        vertices = np.vstack(chunks)
        world = rt.transform_local_points_to_world(rt.BUCKET_LINK, vertices)
        q = np.asarray(rt.get_real_joint_positions()).copy()
        joint = np.asarray(rt.get_joint_anchor_world("bucket"))
        chain = rt.current_planar_chain("load")
        radial = np.asarray(chain["radial"])[:2]
        radial = radial / np.linalg.norm(radial)
        relative = np.asarray(world) - joint
        tangent = np.array([-radial[1], radial[0]])
        local = np.column_stack((relative[:, :2] @ radial, relative[:, :2] @ tangent, relative[:, 2]))
        support = hull_xz(local[:, [0, 2]])
        # Keep real vertices on both sides of the bucket's support silhouette.
        # Named tip/load markers sit inside this mesh and can miss thin sand.
        contacts = []
        for point in support:
            matches = local[np.linalg.norm(local[:, [0, 2]] - point, axis=1) < 1e-5]
            contacts.extend([matches[np.argmin(matches[:, 1])], matches[np.argmax(matches[:, 1])]])
        angles, effectors = {}, {}
        for name in ("tip", "load", "mid", "pour"):
            angles[name] = float(rt.chain_angles_from_q(q, end_effector=name)[2])
            point = np.asarray(rt.bucket_point_world(name))
            delta = np.asarray(world) - point
            effectors[name] = hull_xz(np.column_stack((delta[:, :2] @ radial, delta[:, 2])))
        self.reference = {"q": q, "angle": angles["load"], "support": support,
                          "angles": angles, "effectors": effectors, "radial": radial,
                          "contacts": np.unique(np.asarray(contacts), axis=0)}
        rt.info_print("[REPEAT BUCKET MESH]", f"vertices={len(vertices)}",
                      f"floor_min_z={float(np.min(world[:, 2])):.3f}", force_log=True)

    def minimum_z(self, q):
        self.ensure()
        rt = self.rt
        angle = float(rt.chain_angles_from_q(q, end_effector="load")[2])
        joint = rt.predicted_chain_world_points(q, end_effector="load")[-2]
        return float(joint[2]) + rotated_min_z(self.reference["support"], angle - self.reference["angle"])

    def target_floor(self, effector, angle_deg):
        self.ensure()
        ref = self.reference
        offset = rotated_min_z(ref["effectors"][effector],
                               np.radians(angle_deg) - ref["angles"][effector])
        return float(self.rt.GROUND_TOP_Z) + self.clearance - offset

    def contact_points(self, q):
        self.ensure()
        rt, ref = self.rt, self.reference
        angle = float(rt.chain_angles_from_q(q, end_effector="load")[2]) - ref["angle"]
        points = ref["contacts"]
        x = points[:, 0] * np.cos(angle) - points[:, 2] * np.sin(angle)
        z = points[:, 0] * np.sin(angle) + points[:, 2] * np.cos(angle)
        swing_index = rt.CTRL.name_to_idx["swing"]
        yaw = float(q[swing_index] - ref["q"][swing_index])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        radial = rotation @ ref["radial"]
        tangent = np.array([-radial[1], radial[0]])
        joint = rt.predicted_chain_world_points(q, end_effector="load")[-2]
        return np.column_stack((x[:, None] * radial + points[:, 1, None] * tangent, z)) + joint

    def sand_contact(self, q):
        best = {"depth_m": 0., "point": None, "surface_source": "none"}
        for point in self.contact_points(q):
            if point[2] < float(self.rt.GROUND_TOP_Z) + .005:
                continue
            depth, _surface, source = self.rt.sand_depth_for_point(point)
            # An empty surface query has a conservative floor fallback. It is
            # not evidence of soil contact and must not validate a cut.
            if depth is not None and not str(source).startswith("fallback") and float(depth) > best["depth_m"]:
                best = {"depth_m": float(depth), "point": point.tolist(), "surface_source": str(source)}
        return best

