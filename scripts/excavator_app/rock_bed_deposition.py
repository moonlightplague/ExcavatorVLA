"""Local settling of coarse MPM material points in a stationary dump bed."""
import math
import time
from dataclasses import dataclass

import numpy as np


class BedSurface:
    """Cache vertical intersections with the actual, world-space bed shell."""

    def __init__(self, vertices, indices, diameter):
        self.low = vertices.min(axis=0)
        self.high = vertices.max(axis=0)
        # Bound the one-time surface cache even for very small display overrides.
        self.pitch = max(diameter / 4, max(self.high[:2]-self.low[:2]) / 384)
        self.origin = self.low[:2] - self.pitch
        shape = np.ceil((self.high[:2] - self.origin) / self.pitch).astype(int) + 2
        top = np.full(shape, -np.inf)
        up = np.zeros(shape)
        for a, b, c in vertices[np.asarray(indices).reshape(-1, 3)]:
            u, v = b-a, c-a
            normal = np.cross(u, v)
            if abs(normal[2]) < 1.e-10:
                continue
            xy = np.array([a[:2], b[:2], c[:2]])
            lo = np.ceil((xy.min(axis=0)-self.origin)/self.pitch).astype(int)
            hi = np.floor((xy.max(axis=0)-self.origin)/self.pitch).astype(int) + 1
            ix, iy = np.ogrid[lo[0]:hi[0], lo[1]:hi[1]]
            dx = self.origin[0] + ix*self.pitch - a[0]
            dy = self.origin[1] + iy*self.pitch - a[1]
            s = (dx*v[1]-dy*v[0])/normal[2]
            t = (u[0]*dy-u[1]*dx)/normal[2]
            z = a[2] + s*u[2] + t*v[2]
            region = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
            take = (s >= -1.e-8) & (t >= -1.e-8) & (s+t <= 1.00000001) & (z > top[region])
            top[region][take] = z[take]
            up[region][take] = normal[2] / np.linalg.norm(normal)

        # Keep the footprint of a whole rock inside the loadable surface,
        # including at the open end. Exclude the narrow rim and steep trim.
        valid = np.isfinite(top) & (up > .34)
        safe = valid.copy()
        for di, dj in ((2,0), (-2,0), (0,2), (0,-2), (1,1), (-1,1), (1,-1), (-1,-1)):
            safe &= np.roll(valid, (di,dj), axis=(0,1))
        safe[:2] = safe[-2:] = False
        safe[:, :2] = safe[:, -2:] = False
        self.safe = safe
        # The existing coarse collision floor carries centers 7 cm above the
        # visible floor. Account for the surface normal on the bed ramps.
        self.z = np.where(valid, top + .07 / np.maximum(up, .34), 0.)

    def heights(self, xy):
        uv = (xy-self.origin) / self.pitch
        ij = np.floor(uv).astype(int)
        i, j = np.clip(ij, 0, np.asarray(self.z.shape)-2).T
        valid = (ij[:,0] == i) & (ij[:,1] == j)
        valid &= self.safe[i,j] & self.safe[i+1,j] & self.safe[i,j+1] & self.safe[i+1,j+1]
        s, t = (uv-ij).T
        z = (1-s)*((1-t)*self.z[i,j]+t*self.z[i,j+1]) + s*((1-t)*self.z[i+1,j]+t*self.z[i+1,j+1])
        return np.where(valid, z, np.nan)


@dataclass
class Landing:
    start: np.ndarray
    target: np.ndarray
    velocity: np.ndarray
    begin: float
    duration: float
    supports: tuple


def make_landing_kernel(wp):
    @wp.kernel
    def guide(ids: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.vec3),
              targets: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
              times: wp.array(dtype=wp.vec2), now: float, q: wp.array(dtype=wp.vec3),
              qd: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.mat33),
              grains: wp.array(dtype=wp.vec3)):
        k = wp.tid()
        i = ids[k]
        duration = times[k][1]
        u = wp.clamp((now-times[k][0])/duration, 0.0, 1.0)
        goal = targets[k]
        pos = goal
        vel = wp.vec3(0.0)
        if u < 1.0:
            delta = goal-starts[k]
            blend = u*u*(3.0-2.0*u)
            pos = starts[k]+blend*delta + (duration*u*(1.0-u)*(1.0-u))*velocities[k]
            vel = (6.0*u*(1.0-u)/duration)*delta + ((1.0-u)*(1.0-3.0*u))*velocities[k]
        q[i] = pos
        qd[i] = vel
        grad[i] = wp.mat33(0.0)
        grains[i] = pos
    return guide


class RockBedDeposition:
    def __init__(self, vertices, indices, diameter):
        self.surface = BedSurface(vertices, indices, diameter)
        self.diameter = diameter
        self.landings = {}
        self.cells = {}
        self.ids = np.empty(0, dtype=np.int32)
        self.previous = None
        self.gpu = None
        self.kernel = None
        self.elapsed = 0.
        self.admitted = self.released = 0
        rings = [np.zeros((1,2))]
        for ring, (size, radius) in enumerate(((8,.6), (12,1.), (16,1.45), (20,1.95))):
            a = np.arange(size)*math.tau/size + .25*ring
            rings.append(radius*np.column_stack((np.cos(a), np.sin(a))))
        self.offsets = diameter*np.concatenate(rings)
        a = np.arange(8)*math.tau/8
        self.refine = diameter*np.column_stack((np.cos(a), np.sin(a)))

    def cell(self, xy):
        return tuple(np.floor(xy/self.diameter).astype(int))

    def neighbors(self, xy):
        i, j = self.cell(xy)
        return [idx for x in range(i-3,i+4) for y in range(j-3,j+4)
                for idx in self.cells.get((x,y), ())]

    def find_landing(self, idx, xy, neighbors):
        d = self.diameter
        points = np.array([self.landings[n].target for n in neighbors]).reshape(-1,3)
        angle = (idx*.6180339887498949 % 1.)*math.tau
        rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        def lowest(candidates):
            distance = np.linalg.norm(candidates-xy, axis=1)
            heights = self.surface.heights(candidates)
            if len(points):
                delta = candidates[:,None,:]-points[None,:,:2]
                sq = np.einsum('ijk,ijk->ij', delta, delta)
                support = np.where(sq < d*d, points[:,2]+np.sqrt(np.maximum(0., d*d-sq)), -np.inf)
                heights = np.maximum(heights, support.max(axis=1))
            cost = heights + .45*distance
            valid = np.isfinite(cost) & (distance <= 2*d + 1.e-7) & (heights+d/2 < self.surface.high[2])
            if not valid.any():
                return None, math.inf
            k = np.argmin(np.where(valid,cost,np.inf))
            return np.r_[candidates[k], heights[k]], cost[k]
        target, score = lowest(xy + self.offsets@rot)
        if target is None:
            return None, ()
        fine = self.refine@rot
        for scale in (.28, .12, .05, .02):
            other, value = lowest(target[:2]+scale*fine)
            if value < score:
                target, score = other, value
        supports = tuple(n for n,p in zip(neighbors,points)
                         if p[2]<target[2] and np.linalg.norm(target-p)<=d+.005)
        return target, supports

    def advance(self, points, dt, now, bucket_pose, bucket_rotation, bucket_bounds):
        """Reserve local sites only for descending, released points near the bed."""
        old = self.previous
        self.previous = points
        if old is None:
            return
        low, high = self.surface.low, self.surface.high
        d = self.diameter
        def near_bucket(q):
            local = (q-bucket_pose)@bucket_rotation
            return np.all((local >= bucket_bounds[0]-d/2) & (local <= bucket_bounds[1]+d/2), axis=1)
        changed = False
        if self.ids.size:
            removed = set(self.ids[near_bucket(points[self.ids])].tolist())
            if removed:
                # Entries are in deposition order, so supports precede dependents.
                for idx,a in self.landings.items():
                    if any(s in removed for s in a.supports):
                        removed.add(idx)
                for idx in removed:
                    landing = self.landings.pop(idx)
                    self.cells[self.cell(landing.target[:2])].remove(idx)
                self.released += len(removed)
                changed = True
        near = ((points[:,0] >= low[0]) & (points[:,0] <= high[0]) &
                (points[:,1] >= low[1]) & (points[:,1] <= high[1]) &
                (points[:,2] > low[2]) & (points[:,2] <= high[2]))
        near[self.ids] = False
        ids = np.flatnonzero(near)
        if ids.size:
            ids = ids[~near_bucket(points[ids])]
            floor = self.surface.heights(points[ids,:2])
            vz = (points[ids,2]-old[ids,2])/dt
            # Approximate time to the floor sets the within-tick arrival order.
            order = np.argsort((points[ids,2]-floor)/np.maximum(-vz,.05), kind='stable')
            for j in order:
                idx = int(ids[j])
                if not np.isfinite(floor[j]) or vz[j] >= -.015 or points[idx,2] < floor[j]-.045:
                    continue
                p = points[idx]
                neighbors = self.neighbors(p[:2])
                top = max([floor[j]]+[self.landings[n].target[2] for n in neighbors])
                if p[2] > top + np.clip(-vz[j]*.12, 2*d, 5*d):
                    continue
                target, supports = self.find_landing(idx, p[:2], neighbors)
                if target is None or near_bucket(target[None,:])[0]:
                    continue
                drop = max(0., p[2]-target[2])
                fall_time = 2*drop/(math.sqrt(vz[j]**2+19.62*drop)-vz[j])
                duration = max(.15, fall_time, np.linalg.norm(target[:2]-p[:2])/1.8)
                for n in supports:
                    base = self.landings[n]
                    duration = max(duration, base.begin+base.duration-now+.015)
                # Bound the Hermite tangent to avoid overshooting the destination.
                slope = (target-p)/duration
                v = (p-old[idx])/dt
                v = np.where(v*slope > 0, np.sign(slope)*np.minimum(abs(v),3*abs(slope)), 0.)
                self.landings[idx] = Landing(p.copy(), target, v, now, duration, supports)
                self.cells.setdefault(self.cell(target[:2]),set()).add(idx)
                self.admitted += 1
                changed = True
        if changed:
            self.ids = np.fromiter(self.landings, dtype=np.int32, count=len(self.landings))
            self.gpu = None

    def apply(self, soil, dt, bucket_pose, bucket_rotation, bucket_bounds):
        start = time.perf_counter()
        self.advance(soil.positions, dt, soil.sim_time, bucket_pose, bucket_rotation, bucket_bounds)
        if self.ids.size:
            wp = soil.wp
            if self.gpu is None:
                entries = list(self.landings.values())
                def device_array(data, dtype):
                    return wp.array(data, dtype=dtype, device=soil.device)
                self.gpu = [device_array(self.ids, wp.int32),
                            device_array([a.start for a in entries], wp.vec3),
                            device_array([a.target for a in entries], wp.vec3),
                            device_array([a.velocity for a in entries], wp.vec3),
                            device_array([(a.begin,a.duration) for a in entries], wp.vec2)]
            if self.kernel is None:
                self.kernel = make_landing_kernel(wp)
            wp.launch(self.kernel, dim=len(self.ids), inputs=[
                *self.gpu, soil.sim_time+dt, soil.state.particle_q, soil.state.particle_qd,
                soil.state.mpm.particle_qd_grad, soil.grains], device=soil.device)
        self.elapsed += time.perf_counter()-start
