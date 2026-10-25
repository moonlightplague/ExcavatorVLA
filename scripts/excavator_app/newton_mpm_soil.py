"""Newton dry granular soil inside the existing Isaac/PhysX excavator scene.

Only soil is integrated by Newton. Existing PhysX links supply collision poses
and velocities; MPM contact impulses are returned to those links before the
PhysX step. Material points (stable IDs) remain the collection/scoring API;
passive render grains are never counted as excavated material.
"""
from __future__ import annotations

import builtins
import math
import os
import time
from pathlib import Path

import numpy as np


def enabled():
    backend = os.environ.get("EXCAVATOR_SOIL_BACKEND", "newton_mpm").strip().lower()
    if backend not in ("newton_mpm", "physx"):
        raise ValueError(f"Unknown EXCAVATOR_SOIL_BACKEND: {backend!r}")
    return backend == "newton_mpm"


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def triangulate(counts, indices, left_handed=False):
    """USD polygon fan, preserving orientation and the open bucket mouth."""
    if sum(counts) != len(indices) or any(int(count) < 3 for count in counts):
        raise ValueError("Invalid USD face indices")
    triangles, start = [], 0
    for count in counts:
        face = indices[start:start+int(count)]
        for k in range(1, int(count)-1):
            triangle = [face[0], face[k], face[k+1]]
            triangles.append(triangle[::-1] if left_handed else triangle)
        start += int(count)
    if start != len(indices):
        raise ValueError("Invalid USD face indices")
    return np.asarray(triangles, dtype=np.int32).reshape(-1)


def material_layout(runtime, spacing):
    """Count the complete source before allocating any material-point arrays."""
    x0, x1, y0, y1 = runtime["current_sand_footprint_bbox_xy"]()
    nx, ny = np.maximum(1, np.ceil(np.array([x1-x0, y1-y0])/spacing).astype(int))
    dx, dy = (x1-x0)/nx, (y1-y0)/ny
    xx, yy = np.meshgrid(x0+(np.arange(nx)+0.5)*dx, y0+(np.arange(ny)+0.5)*dy)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    xy = xy[runtime["is_inside_diggable_xy_batch"](xy[:, 0], xy[:, 1])]
    floor = float(runtime["SAND_FLOOR_Z"])
    height = runtime["initial_sand_height_xy_batch"](xy[:, 0], xy[:, 1])
    layers = np.maximum(0, np.floor((height-floor)/spacing).astype(np.int64))
    return xy, layers, float(dx), float(dy), floor


def choose_resolution(runtime, max_count=40000, spacing=.075, voxel=.10):
    """Coarsen the whole source uniformly, keeping mass and the entire footprint.

    The requested spacing/grid are minima; both grow by the same factor when
    necessary. A smaller source may explicitly use finer values for regression.
    """
    if not 0 < max_count <= 40000:
        raise ValueError('MPM material-point budget must be between 1 and 40000')
    if not math.isfinite(voxel) or not math.isfinite(spacing) or not 0 < spacing <= voxel:
        raise ValueError('MPM spacing/grid must be finite, positive and spacing <= grid')
    for _ in range(32):
        count = int(material_layout(runtime, spacing)[1].sum())
        if count <= 0:
            raise RuntimeError('Soil source has no sampled volume at the requested resolution')
        if count <= max_count:
            return spacing, voxel, count
        factor = max(1.01, (count/max_count)**(1/3)*1.005)
        spacing, voxel = spacing*factor, voxel*factor
    raise RuntimeError('Could not fit the complete soil source into the material-point budget')


def sample_material(runtime, spacing, max_count):
    """Stratified volume sampling; never truncate one side of a pile to budget."""
    xy, layers, dx, dy, floor = material_layout(runtime, spacing)
    count = int(layers.sum())
    if count <= 0 or count > max_count:
        raise RuntimeError(f"MPM needs {count:,} material points at {spacing:.5f}m spacing; "
                           f"limit={max_count:,}. Adjust sand amount or explicit "
                           "EXCAVATOR_MPM_MAX_PARTICLES; grid is not silently coarsened.")
    rows = np.repeat(np.arange(len(xy)), layers)
    starts = np.cumsum(layers)-layers
    z_index = np.arange(count)-np.repeat(starts, layers)
    xyz = np.column_stack((xy[rows], floor+(z_index+0.5)*spacing))
    rng = np.random.default_rng(int(os.environ.get("EXCAVATOR_MPM_SEED", "4107")))
    xyz += rng.uniform(-0.4, 0.4, size=xyz.shape)*np.array([dx, dy, spacing])
    volume = float(dx*dy*spacing)
    return np.asarray(xyz, dtype=np.float32), volume


def aggregate_wrenches(impulses, positions, collider_ids, body_ids, origins, body_count):
    """Sum *rigid-side* impulses and angular impulses about link origins."""
    wrench = np.zeros((body_count, 6), dtype=np.float64)
    valid = (collider_ids >= 0) & (collider_ids < len(body_ids))
    body = body_ids[collider_ids[valid]]
    keep = body >= 0
    body, impulse, points = body[keep], impulses[valid][keep], positions[valid][keep]
    np.add.at(wrench[:, :3], body, impulse)
    np.add.at(wrench[:, 3:], body, np.cross(points-origins[body], impulse))
    return wrench


def mpm_substeps(dt):
    """Fixed 120 Hz MPM: two substeps per normal 60 Hz PhysX tick.

    Deliberately independent of outlier particle speeds. Collider projection
    and an explicit escape guard handle/check penetration instead of global CFL.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f'Invalid PhysX dt: {dt}')
    return max(1, math.ceil(dt*120-1e-6))


def make_gpu_kernels(wp):
    """Import/compile only after Isaac has enabled its bundled Warp extension."""
    @wp.kernel
    def contact_wrench(impulses: wp.array(dtype=wp.vec3), positions: wp.array(dtype=wp.vec3),
                       ids: wp.array(dtype=wp.int32), body_ids: wp.array(dtype=wp.int32),
                       poses: wp.array(dtype=wp.transform),
                       output: wp.array(dtype=wp.float32, ndim=2)):
        i = wp.tid()
        collider = ids[i]
        if collider >= 0 and collider < body_ids.shape[0]:
            body = body_ids[collider]
            if body >= 0:
                impulse = impulses[i]
                torque = wp.cross(positions[i]-wp.transform_get_translation(poses[body]), impulse)
                for axis in range(3):
                    wp.atomic_add(output, body, axis, impulse[axis])
                    wp.atomic_add(output, body, axis+3, torque[axis])

    @wp.kernel
    def max_speed(velocities: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.float32)):
        wp.atomic_max(output, 0, wp.length(velocities[wp.tid()]))

    return contact_wrench, max_speed


def material_density(runtime, point_count, point_volume, legacy_count):
    """Refine quadrature without silently changing the original total mass."""
    override = os.environ.get('EXCAVATOR_MPM_DENSITY', '').strip()
    if override:
        density, source = float(override), 'explicit_density'
    elif 'PARTICLE_MASS' in runtime:
        density = float(runtime['PARTICLE_MASS'])*legacy_count/(point_count*point_volume)
        source = 'preserve_legacy_total_mass'
    else:
        density, source = 1600., 'standalone_test_density'
    if not math.isfinite(density) or density <= 0:
        raise ValueError('MPM density must be finite and positive')
    return density, source


def current():
    return getattr(builtins, "_EXCAVATOR_MPM_SOIL", None)


def close():
    controller = current()
    if controller is not None:
        controller.close()
        builtins._EXCAVATOR_MPM_SOIL = None


def create(runtime, root):
    close()
    try:
        controller = NewtonSoil(runtime, root)
    except Exception as exc:
        runtime["STATE"].update(real_sand_enabled=False,
                                real_sand_error=f"Newton MPM: {type(exc).__name__}: {exc}")
        raise  # fail visibly; never run collection against a static heightfield
    builtins._EXCAVATOR_MPM_SOIL = controller
    return controller.points.GetPrim()


def augment_api(runtime, api):
    if not enabled():
        return
    api["soil_backend"] = "newton_mpm"
    api["mpm_diagnostics"] = lambda: current().diagnostics() if current() else {"active": False}
    api["cache_mpm_initial_state"] = lambda: current().cache_initial_state() if current() else False
    api["reset_mpm_initial_state"] = lambda: current().restore_initial_state() if current() else False
    controller = current()
    if controller is None:
        return
    controller.publish_status()
    api["particle_material_config"] = dict(controller.material)
    api["derived_particle_config"].update(
        mass=controller.metric_mass, radius=controller.radius,
        spacing_xy=controller.spacing, spacing_z=controller.spacing,
        estimated_particle_count=len(controller.metric_indices), grid_size=controller.voxel,
        backend="newton_mpm", render_grains_are_physics=False,
    )


def build_ui(runtime):
    """Soil-only UI; do not present ignored PhysX fluid parameters as MPM knobs."""
    import omni.ui as ui
    window = ui.Window("Sand Site Control", width=510, height=260)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Newton MPM | dry frictional soil", height=24)
            controller = current()
            description = (f"Grid: {controller.voxel*100:g} cm | friction: 0.68 | "
                           f"density: {controller.material['density_kg_m3']:.1f} kg/m3"
                           if controller else "Grid: 10 cm minimum | <=40000 material points | mass: legacy scene")
            ui.Label(description, word_wrap=True)
            ui.Label("Cohesion, tension and viscosity: 0. Material points drive metrics; fine grains are display only.", word_wrap=True)
            ui.Label("Reset applies the current sand source and amount.", word_wrap=True)
            with ui.HStack(height=30):
                ui.Button("Reset Sand", clicked_fn=lambda: runtime['request_sand_reset']('mpm_ui'))
                ui.Button("Clean Sand", clicked_fn=runtime['clear_real_particle_sand'])
                ui.Button("Status", clicked_fn=lambda: print('[MPM STATUS]', current().diagnostics() if current() else 'inactive', flush=True))
    builtins._SAND_SITE_UI_WINDOW = window
    return window


class NewtonSoil:
    def __init__(self, runtime, root):
        if os.environ.get('EXCAVATOR_MPM_DEBUG_STACK') == '1':
            import faulthandler
            import signal
            faulthandler.enable()
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.warp.core")
        enable_extension("isaacsim.pip.newton")
        import warp as wp
        import newton
        import omni.usd
        import omni.physx
        from newton.solvers import SolverImplicitMPM
        from pxr import UsdGeom, Gf, Vt

        self.wp, self.newton = wp, newton
        self.runtime, self.root = runtime, str(root)
        self.stage = omni.usd.get_context().get_stage()
        self.physx = omni.physx.get_physx_interface()
        self.subscription = self.force_subscription = self.update_subscription = None
        self.closed, self.error, self.steps, self.sim_time = False, "", 0, 0.0
        self.step_wall_seconds, self.peak_force = 0.0, 0.0
        self.physics_wall_seconds = self.render_wall_seconds = 0.0
        self.render_updates = self.host_contact_values = 0
        self.last_force = np.empty((0, 6), dtype=np.float32)
        self.sleeping, self.quiet_checks, self.solver_substeps = False, 0, 0
        self.max_particle_speed = float('inf')
        self.sleep_speed = float(os.environ.get('EXCAVATOR_MPM_SLEEP_SPEED', '0.01'))
        if not math.isfinite(self.sleep_speed) or self.sleep_speed < 0:
            raise ValueError('MPM sleep speed must be finite and nonnegative (0 disables sleep)')
        self.sim_view = self.body_view = None
        self.max_points = int(os.environ.get('EXCAVATOR_MPM_MAX_PARTICLES', '40000'))
        requested_voxel = float(os.environ.get('EXCAVATOR_MPM_VOXEL_SIZE', '.10'))
        requested_spacing = float(os.environ.get('EXCAVATOR_MPM_PARTICLE_SPACING', str(.75*requested_voxel)))
        self.spacing, self.voxel, _ = choose_resolution(runtime, self.max_points,
                                                       requested_spacing, requested_voxel)
        self.positions, volume = sample_material(runtime, self.spacing, self.max_points)
        # USD display diameter only; never used by the model or contact solver.
        self.render_diameter = float(os.environ.get('EXCAVATOR_MPM_RENDER_DIAMETER', '.012'))
        if not math.isfinite(self.render_diameter) or self.render_diameter <= 0:
            raise ValueError('MPM render diameter must be finite and positive')
        # Preserve the existing count-based collection/evaluation scale when
        # refining the MPM quadrature. A deterministic uniform subset of real
        # material IDs represents the full mass; never count passive grains.
        legacy_count = len(runtime['sand_particle_points']()[0])
        density, density_source = material_density(runtime, len(self.positions), volume, legacy_count)
        self.mass, self.radius = density*volume, volume**(1/3)/2
        metric_count = min(len(self.positions), legacy_count)
        if metric_count <= 0:
            raise RuntimeError('Legacy collection/scoring source is empty')
        self.metric_indices = np.sort(np.random.default_rng(4108).choice(
            len(self.positions), size=metric_count, replace=False))
        self.metric_mass = self.mass*len(self.positions)/metric_count
        self.material = {"backend": "newton_mpm", "friction": 0.68, "young_modulus": 1e15,
                         "yield_pressure": 1e12, "yield_stress": 0.0,
                         "tensile_yield_ratio": 0.0, "dilatancy": 0.0,
                         "viscosity": 0.0, "density_kg_m3": density,
                         "density_source": density_source}
        # The old temporary rectangular walls remain as invisible, noncolliding
        # source guides until their existing cleanup call. They cannot prop up soil.
        self._disable_source_walls()
        device = os.environ.get("EXCAVATOR_MPM_DEVICE")
        if device is None:
            import carb
            gpu = carb.settings.get_settings().get("/physics/cudaDevice")
            device = f"cuda:{max(0, int(gpu or 0))}"
        self.device = device
        self.wrench_kernel, self.speed_kernel = make_gpu_kernels(wp)
        wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[2]/".mpm_warp_cache")
        newton.use_coord_layout_targets = True
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0, 0, -9.81))
            SolverImplicitMPM.register_custom_attributes(builder)
            self.body_paths, self.collision_paths, self.body_bounds = [], [], {}
            self._import_colliders(builder)
            n = len(self.positions)
            attrs = {"mpm:"+key: [value]*n for key, value in self.material.items()
                     if key in ("friction", "young_modulus", "yield_pressure", "yield_stress",
                                "tensile_yield_ratio", "dilatancy", "viscosity")}
            builder.add_particles(pos=self.positions, vel=np.zeros_like(self.positions),
                                  mass=[self.mass]*n, radius=[self.radius]*n,
                                  custom_attributes=attrs)
            self.model = builder.finalize(device=device)
            # Finite mass/inertia in MPM's contact compliance prevents an
            # infinite-mass proxy launching a light real PhysX body on impact.
            self._attach_view()
            config = SolverImplicitMPM.Config()
            config.voxel_size, config.max_iterations, config.tolerance = self.voxel, 150, 1e-4
            self.solver = SolverImplicitMPM(self.model, config=config)
            self.state, self.previous = self.model.state(), self.model.state()
            self.collider_model = self.model
            self._setup_colliders()
            self.last_force = np.zeros((self.model.body_count, 6), dtype=np.float32)
            self.force_origins = self.state.body_q.numpy()[:, :3].copy()
            self.applied_impulse = np.zeros_like(self.last_force)
            self.gpu_wrench = wp.zeros((self.model.body_count, 6), dtype=wp.float32, device=device)
            self.gpu_max_speed = wp.zeros(1, dtype=wp.float32, device=device)
            # Display samples do not affect the material-point budget or scores.
            grains_per_point = max(1, min(8, 160000//n))
            self.grains = self.solver.sample_render_grains(self.state, grains_per_point)
            self.collider_body_ids = self.solver.collider_body_index.numpy()
        path = f"{root}/{runtime['PARTICLE_POINTS_PATH_SUFFIX']}"
        self.points = UsdGeom.Points.Define(self.stage, path)
        self.points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(self.metric_positions))
        self.points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(metric_count, self.radius*2, dtype=np.float32)))
        self.points.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        # Use a sibling Xform, not children of a Points Gprim (imaging may prune
        # Gprim children). The existing system subtree is also in the standalone
        # creation whitelist; it has no PhysX particle-system schema.
        render_root = f"{root}/{runtime.get('PARTICLE_SYSTEM_PATH_SUFFIX', 'ParticleSystem')}"
        UsdGeom.Xform.Define(self.stage, render_root)
        # Native USD Points use Isaac's particle renderer without a million
        # instance transforms/BVH instances. These are the same passive grains
        # and sphere diameters as the standalone visualization.
        self.render = UsdGeom.Points.Define(self.stage, render_root+"/FineGrains")
        self.render.CreateWidthsAttr(Vt.FloatArray.FromNumpy(
            np.full(self.grains.size, self.render_diameter, dtype=np.float32)))
        self.render.CreateDisplayColorAttr([Gf.Vec3f(0.47, 0.27, 0.10)])
        self.render.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        self.render.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(self.grains.flatten().numpy()))
        self.points.CreateDisplayColorAttr([Gf.Vec3f(0.47, 0.27, 0.10)])
        self.dirty = False
        self.publish_status()
        try:
            # Attach outside the pre-step callback: PhysX cannot discover its
            # currently attached stage while entering a simulation callback.
            self._attach_view()
            self.force_subscription = self.physx.subscribe_physics_on_step_events(self._on_pre_step, True, 0)
            self.subscription = self.physx.subscribe_physics_on_step_events(self._on_step, False, 0)
            import omni.kit.app
            self.update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="ExcavatorNewtonSoilRender")
        except Exception:
            self.close()
            raise
        print(f"[MPM READY] grid={self.voxel} spacing={self.spacing} budget={self.max_points} "
              f"substeps=2@60Hz contact_reduce=gpu points={n} metric_points={metric_count} render_grains={self.grains.size} "
              f"mass_kg={n*self.mass:.3f} bodies={self.body_paths} colliders={len(self.collision_paths)} "
              f"device={device}", flush=True)

    def _source_spec(self):
        rt = self.runtime
        x0, x1, y0, y1 = rt['current_sand_footprint_bbox_xy']()
        center = np.array([rt.get('SAND_CENTER_X', (x0+x1)/2),
                           rt.get('SAND_CENTER_Y', (y0+y1)/2), rt['SAND_FLOOR_Z']])
        polygon_fn = rt.get('current_sand_footprint_polygon_xy')
        polygon = np.asarray(polygon_fn() if polygon_fn else
                             [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=float)
        spec = np.r_[rt.get('SANDBOX_FILL_HEIGHT', 0.), rt.get('SAND_AMOUNT_MULTIPLIER', 1.),
                     center[2], (polygon-center[:2]).ravel()]
        return center, spec

    def cache_initial_state(self):
        """Cache the actual settled material, including plastic state and grains."""
        with self.wp.ScopedDevice(self.device):
            template = self.model.state()
            self.initial_state = {}
            def copy_arrays(target, source, prefix=()):
                for name, value in vars(target).items():
                    path = prefix+(name,)
                    if isinstance(value, self.wp.array):
                        value.assign(getattr(source, name))
                        self.initial_state[path] = value
                    elif isinstance(value, self.newton.Model.AttributeNamespace):
                        copy_arrays(value, getattr(source, name), path)
            # Solver-added grid-sized buffers (e.g. collider_ids) are numerical
            # caches, not physical state; model.state() defines the fields to
            # preserve and the solver will rebuild those caches after reset.
            copy_arrays(template, self.state)
            self.initial_grains = self.wp.clone(self.grains)
        self.initial_positions = self.positions.copy()
        self.initial_center, self.initial_spec = self._source_spec()
        print('[MPM INITIAL STATE CACHED]', len(self.positions), flush=True)
        return True

    def restore_initial_state(self):
        """Fast diagnostic episode reset for an unchanged soil specification.

        Relocate a complete settled initial state and rebuild only the small
        collider model. This stays within the original bridge's 120s timeout;
        changed amount/source shape falls back to full reconstruction.
        """
        if not hasattr(self, 'initial_state'):
            return False
        center, spec = self._source_spec()
        if spec.shape != self.initial_spec.shape or not np.allclose(spec, self.initial_spec, atol=2e-4, rtol=0):
            return False
        started = time.perf_counter()
        self._disable_source_walls()
        old_paths = list(self.body_paths)
        with self.wp.ScopedDevice(self.device):
            builder = self.newton.ModelBuilder(up_axis=self.newton.Axis.Z, gravity=(0,0,-9.81))
            self.body_paths, self.collision_paths, self.body_bounds = [], [], {}
            self._import_colliders(builder)
            if self.body_paths != old_paths:
                return False
            self.collider_model = builder.finalize(device=self.device)
            self._setup_colliders()
            for path, saved in self.initial_state.items():
                target = self.state
                for name in path[:-1]:
                    target = getattr(target, name)
                getattr(target, path[-1]).assign(saved)
            delta = np.asarray(center-self.initial_center, dtype=np.float32)
            self.positions = self.initial_positions+delta
            self.state.particle_q.assign(self.positions)
            self.grains.assign(self.initial_grains.numpy()+delta)
            self.state.body_q.assign(self.body_view.get_transforms().numpy()[self.model_to_view])
            self.state.body_qd.assign(self.body_view.get_velocities().numpy()[self.model_to_view])
            # Clear numerical warm starts without erasing the restored physical
            # stress, plastic volume or deformation (flags=0 is intentional).
            self.solver.reset(self.state, flags=0)
        self.last_force.fill(0)
        self.applied_impulse.fill(0)
        self.force_origins = self.state.body_q.numpy()[:, :3].copy()
        self.error, self.sleeping, self.quiet_checks = '', False, 0
        self.steps, self.sim_time, self.solver_substeps, self.peak_force = 0, 0., 0, 0.
        self.physics_wall_seconds = self.render_wall_seconds = 0.0
        self.render_updates = self.host_contact_values = 0
        self.soil_bounds = self.positions.min(axis=0), self.positions.max(axis=0)
        self.dirty = True
        self._on_update(None)
        self.publish_status()
        print(f'[MPM INITIAL STATE RESTORED] delta={delta.tolist()} '
              f'seconds={time.perf_counter()-started:.3f}', flush=True)
        return True

    def _disable_source_walls(self):
        from pxr import UsdGeom, UsdPhysics, Usd
        walls = self.stage.GetPrimAtPath(self.root+"/SandRetainingWalls")
        if walls.IsValid():
            UsdGeom.Imageable(walls).MakeInvisible()
            for prim in Usd.PrimRange(walls):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    def _import_colliders(self, builder):
        from pxr import UsdGeom, UsdPhysics, Usd, Gf
        wp, newton = self.wp, self.newton
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        bodies = {}
        self.static_meshes, self.static_margins = [], []
        imported_ground = False
        # The URDF keeps most collision meshes inside instance proxies. A
        # plain Stage.Traverse() would silently omit the boom/arm/base links.
        for prim in Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if path.startswith(self.root+"/") and '/CircularSandRetainingWall/' not in path:
                continue  # old heightfield, debug geometry and temporary soil walls
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if prim.GetTypeName() in ('Xform', 'Scope'):
                continue  # organizational collision scopes have no surface
            owner = prim
            while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            owner_path = str(owner.GetPath()) if owner and owner.HasAPI(UsdPhysics.RigidBodyAPI) else None
            world = cache.GetLocalToWorldTransform(prim)
            if prim.GetTypeName() == "Plane":
                # Scene ground is horizontal; never replace an inclined plane
                # by a horizontal fake floor.
                normal = world.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
                if abs(normal[2]) < 0.999:
                    raise RuntimeError(f"Unsupported nonhorizontal ground plane: {path}")
                builder.add_ground_plane(height=float(world.ExtractTranslation()[2]),
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5), label=path)
                plane = newton.Mesh.create_plane(width=1000., length=1000.)
                vertices = np.asarray(plane.vertices).copy()
                vertices[:, 2] += float(world.ExtractTranslation()[2])
                self._add_static_mesh(vertices, plane.indices, margin=0.)
                imported_ground = True
                self.collision_paths.append(path)
                continue
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                raw = mesh.GetPointsAttr().Get()
                if raw is None or len(raw) == 0:
                    continue
                vertices = np.asarray(raw, dtype=np.float64)
                indices = triangulate(mesh.GetFaceVertexCountsAttr().Get(),
                    mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetOrientationAttr().Get() == "leftHanded")
            elif prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
                cube = newton.Mesh.create_box(half, half, half)
                vertices, indices = cube.vertices, cube.indices
            else:
                raise RuntimeError(f"Unsupported active soil collider {path}: {prim.GetTypeName()}")
            if len(indices) == 0:
                continue
            if owner_path:
                if owner_path not in bodies:
                    transform = Gf.Transform(cache.GetLocalToWorldTransform(owner))
                    q = transform.GetRotation().GetQuat()
                    p = transform.GetTranslation()
                    bodies[owner_path] = builder.add_body(xform=wp.transform(
                        wp.vec3(*p), wp.quat(*q.GetImaginary(), q.GetReal())),
                        label=owner_path, mass=1.0, inertia=wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0),
                        is_kinematic=False, lock_inertia=True)
                    self.body_paths.append(owner_path)
                body = bodies[owner_path]
                pose = Gf.Transform(cache.GetLocalToWorldTransform(owner))
                # Body pose is rigid. Bake all authored scale into vertices.
                rigid_world = Gf.Matrix4d().SetRotate(pose.GetRotation())
                rigid_world.SetTranslateOnly(pose.GetTranslation())
                relative = world*rigid_world.GetInverse()
            else:
                body, relative = -1, world
            matrix = np.asarray(relative, dtype=np.float64)
            vertices = np.asarray(vertices)@matrix[:3, :3] + matrix[3, :3]
            if body >= 0:
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                if body in self.body_bounds:
                    low = np.minimum(low, self.body_bounds[body][0])
                    high = np.maximum(high, self.body_bounds[body][1])
                self.body_bounds[body] = low, high
            if np.linalg.det(matrix[:3, :3]) < 0:
                indices = np.asarray(indices).reshape(-1, 3)[:, ::-1].reshape(-1)
            if body < 0:
                # Keep each static surface separate. Merging open ground and
                # closed truck/parking meshes corrupts signed-distance queries.
                self._add_static_mesh(vertices, indices, margin=0.)
            builder.add_shape_mesh(body=body, mesh=newton.Mesh(
                np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32), compute_inertia=False),
                cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5), label=path)
            self.collision_paths.append(path)
        if not imported_ground:
            raise RuntimeError("No scene ground Plane found; refusing to invent a floor")
        if not any("bucket" in path.lower() for path in self.collision_paths):
            raise RuntimeError("No active bucket triangle mesh found for soil contact")

    def _add_static_mesh(self, vertices, indices, margin):
        points = self.wp.array(np.asarray(vertices, dtype=np.float32), dtype=self.wp.vec3, device=self.device)
        triangles = self.wp.array(np.asarray(indices, dtype=np.int32), dtype=self.wp.int32, device=self.device)
        self.static_meshes.append(self.wp.Mesh(points, triangles, self.wp.zeros_like(points)))
        self.static_margins.append(margin)

    def _setup_colliders(self):
        bodies = len(self.body_paths)
        count = len(self.static_meshes)+bodies
        self.solver.setup_collider(model=self.collider_model,
            collider_meshes=[*self.static_meshes, *([None]*bodies)],
            collider_body_ids=[*([None]*len(self.static_meshes)), *range(bodies)],
            collider_friction=[0.5]*count, collider_adhesion=[0.]*count,
            collider_margins=[*self.static_margins, *([min(.0125, 0.5*self.voxel)]*bodies)],
            body_com=self.model.body_com, body_mass=self.model.body_mass,
            body_inv_inertia=self.model.body_inv_inertia)
        self.collider_body_ids = self.solver.collider_body_index.numpy()

    def _attach_view(self):
        if not self.body_paths:
            return
        if self.body_view is not None and self.body_view.check():
            return
        import omni.physics.tensors as tensors
        import omni.usd
        # Warp supports both CPU and GPU PhysX pipelines; numpy is CPU-only.
        self.sim_view = tensors.create_simulation_view("warp",
            stage_id=omni.usd.get_context().get_stage_id(), backend="physx")
        self.sim_view.set_subspace_roots("/")
        self.body_view = self.sim_view.create_rigid_body_view(self.body_paths)
        paths = list(self.body_view.prim_paths)
        if set(paths) != set(self.body_paths):
            raise RuntimeError(f"MPM/PhysX body mismatch: {paths} vs {self.body_paths}")
        self.view_to_model = np.array([self.body_paths.index(path) for path in paths])
        self.model_to_view = np.argsort(self.view_to_model)
        # Both APIs report linear velocity at the COM. Match the collider's
        # local COM to PhysX, otherwise rotating links get an extra omega x r.
        coms = self.body_view.get_coms().numpy()[self.model_to_view, :3]
        self.local_com = np.asarray(coms, dtype=np.float32)
        self.model.body_com.assign(self.local_com)
        masses = self.body_view.get_masses().numpy()[self.model_to_view].reshape(-1)
        self.inv_mass = self.body_view.get_inv_masses().numpy()[self.model_to_view].reshape(-1)
        inertia = self.body_view.get_inertias().numpy()[self.model_to_view].reshape(-1, 3, 3)
        self.inv_inertia = self.body_view.get_inv_inertias().numpy()[self.model_to_view].reshape(-1, 3, 3)
        self.model.body_mass.assign(np.asarray(np.where(self.inv_mass > 0, masses, 0), dtype=np.float32))
        self.model.body_inv_mass.assign(np.asarray(self.inv_mass, dtype=np.float32))
        self.model.body_inertia.assign(np.asarray(inertia, dtype=np.float32))
        self.model.body_inv_inertia.assign(np.asarray(self.inv_inertia, dtype=np.float32))
        self.view_indices = self.wp.array(np.arange(len(paths), dtype=np.uint32),
            dtype=self.wp.uint32, device=self.sim_view.device)

    def _on_step(self, dt):
        if self.closed or self.error:
            return
        try:
            self.advance(float(dt))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.runtime["STATE"].update(real_sand_enabled=False, real_sand_error=self.error, last_reset_healthy=False)
            import omni.timeline
            omni.timeline.get_timeline_interface().pause()
            import traceback
            traceback.print_exc()
            print(f"[MPM FATAL] Physics paused: {self.error}", flush=True)

    def _on_pre_step(self, dt):
        """Lagged coupling: apply the previous completed MPM contact solve."""
        if self.closed or self.error or not len(self.body_paths):
            return
        self.applied_impulse = self.last_force*float(dt)
        if not np.any(self.last_force):
            return
        order = self.view_to_model
        force = self.wp.array(self.last_force[order, :3].copy(), dtype=self.wp.float32, device=self.sim_view.device)
        torque = self.wp.array(self.last_force[order, 3:].copy(), dtype=self.wp.float32, device=self.sim_view.device)
        position = self.wp.array(self.force_origins[order].copy(), dtype=self.wp.float32, device=self.sim_view.device)
        self.body_view.apply_forces_and_torques_at_position(force, torque, position, self.view_indices, True)

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid PhysX dt: {dt}")
        started = time.perf_counter()
        self._attach_view()
        body_count = len(self.body_paths)
        poses = self.state.body_q.numpy()
        velocities = np.zeros((body_count, 6), dtype=np.float32)
        if body_count:
            poses = self.body_view.get_transforms().numpy()[self.model_to_view]
            velocities = self.body_view.get_velocities().numpy()[self.model_to_view]
            # PhysX has advanced gravity, motors and constraints, including our
            # last contact impulse. Remove that old impulse to obtain the free
            # velocity for the replacement MPM solve (Newton's official two-way
            # example uses the same lagged/subtract-old-force construction).
            for body in range(body_count):
                rot = rotation(poses[body, 3:])
                com = poses[body, :3] + rot@self.local_com[body]
                angular = self.applied_impulse[body, 3:] + np.cross(
                    self.force_origins[body]-com, self.applied_impulse[body, :3])
                velocities[body, :3] -= self.inv_mass[body]*self.applied_impulse[body, :3]
                velocities[body, 3:] -= rot@self.inv_inertia[body]@rot.T@angular
        if self.sleeping and self._body_near_soil(poses, velocities, dt):
            self.sleeping, self.quiet_checks = False, 0
            print('[MPM WAKE] rigid body approaching resting soil', flush=True)
        if self.sleeping:
            self.steps += 1
            self.sim_time += dt
            self.last_force.fill(0)
            self.step_wall_seconds = time.perf_counter()-started
            self.publish_status()
            return
        total = np.zeros((body_count, 6), dtype=np.float64)
        steps = mpm_substeps(dt)
        if steps > 64:
            raise RuntimeError(f'Unsafe MPM advection speed requires {steps} substeps')
        h = dt/steps
        with self.wp.ScopedDevice(self.device):
            for _ in range(steps):
                if body_count:
                    self.state.body_q.assign(poses)
                    self.state.body_qd.assign(velocities)
                self.previous.particle_q.assign(self.state.particle_q)
                self.previous.mpm.particle_transform.assign(self.state.mpm.particle_transform)
                self.solver.step(self.state, self.state, control=None, contacts=None, dt=h)
                # Fixed substeps can advect a fast outlier beyond a narrow
                # distance-query band. Project against actual imported mesh
                # colliders using a wider band, without adding global substeps.
                self.solver.project_outside(self.state, self.state, h,
                                            gap=max(.25, math.sqrt(3)*self.voxel))
                self.solver.update_particle_frames(self.previous, self.state, h)
                self.solver.update_render_grains(self.previous, self.state, self.grains, h)
                impulses, positions, ids = self.solver.collect_collider_impulses(self.state)
                self.gpu_wrench.zero_()
                self.wp.launch(self.wrench_kernel, dim=ids.size,
                    inputs=[impulses, positions, ids, self.solver.collider_body_index,
                            self.state.body_q, self.gpu_wrench], device=self.device)
                # Only 6 values per real body cross to the PhysX adapter. Grid
                # contact arrays remain on the GPU; no NumPy grid reduction.
                impulse = self.gpu_wrench.numpy().astype(np.float64)
                self.host_contact_values += body_count*6
                total += impulse
                # Predict the finite proxy's velocity over the remaining MPM
                # substeps. PhysX receives the same total impulse exactly once.
                for body in range(body_count):
                    rot = rotation(poses[body, 3:])
                    angular_impulse = impulse[body, 3:] - np.cross(rot@self.local_com[body], impulse[body, :3])
                    velocities[body, :3] += self.inv_mass[body]*impulse[body, :3]
                    velocities[body, 3:] += rot@self.inv_inertia[body]@rot.T@angular_impulse
                self.solver_substeps += 1
            self.positions = self.state.particle_q.numpy()
        if not np.isfinite(self.positions).all() or not np.isfinite(total).all():
            raise RuntimeError("Nonfinite MPM state/contact force")
        if self.positions[:, 2].min() < float(self.runtime['SAND_FLOOR_Z'])-3*self.voxel:
            raise RuntimeError('MPM ground collision escape; refusing to collect invalid soil')
        self.last_force = np.asarray(total/dt, dtype=np.float32)
        self.force_origins = poses[:, :3].copy()
        if body_count:
            self.peak_force = max(self.peak_force, float(np.max(np.linalg.norm(total[:, :3]/dt, axis=1))))
        self.steps += 1
        self.sim_time += dt
        self.step_wall_seconds = time.perf_counter()-started
        self.physics_wall_seconds += self.step_wall_seconds
        self.dirty = True
        if self.steps % 15 == 0:
            self.gpu_max_speed.zero_()
            self.wp.launch(self.speed_kernel, dim=self.model.particle_count,
                inputs=[self.state.particle_qd, self.gpu_max_speed], device=self.device)
            self.max_particle_speed = float(self.gpu_max_speed.numpy()[0])
            self.soil_bounds = self.positions.min(axis=0), self.positions.max(axis=0)
            quiet = self.sleep_speed > 0 and self.max_particle_speed < self.sleep_speed
            self.quiet_checks = self.quiet_checks+1 if quiet else 0
            if self.quiet_checks >= 3 and not self._body_near_soil(poses, velocities, dt):
                self.sleeping = True
                print(f'[MPM SLEEP] all material points quiet: vmax={self.max_particle_speed:.6g} m/s', flush=True)
        if self.steps == 1 or self.steps % 120 == 0:
            print(f"[MPM STEP] steps={self.steps} t={self.sim_time:.3f} "
                  f"z=({self.positions[:,2].min():.4f},{self.positions[:,2].max():.4f}) "
                  f"peak_link_force_N={self.peak_force:.3f} vmax={self.max_particle_speed:.6g} "
                  f"wall_s={self.step_wall_seconds:.3f}", flush=True)

    def _body_near_soil(self, poses, velocities, dt):
        """Conservative swept AABBs; carried material also keeps its body awake."""
        if not hasattr(self, 'soil_bounds'):
            return True
        soil_low, soil_high = self.soil_bounds
        for body, (low, high) in self.body_bounds.items():
            corners = np.array([[x,y,z] for x in (low[0],high[0])
                                for y in (low[1],high[1]) for z in (low[2],high[2])])
            world = corners@rotation(poses[body,3:]).T + poses[body,:3]
            margin = (3*self.voxel + np.linalg.norm(velocities[body,:3])*dt
                      + np.linalg.norm(velocities[body,3:])*np.linalg.norm(high-low)*dt)
            if np.all(world.max(axis=0)+margin >= soil_low) and np.all(world.min(axis=0)-margin <= soil_high):
                return True
        return False

    def _on_update(self, _event):
        if self.closed or not self.dirty:
            return
        from pxr import Vt
        started = time.perf_counter()
        self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.metric_positions))
        self.render.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.grains.flatten().numpy()))
        self.dirty = False
        self.render_wall_seconds += time.perf_counter()-started
        self.render_updates += 1
        self.publish_status()

    def publish_status(self):
        self.runtime["STATE"].update(
            soil_backend="newton_mpm", real_sand_enabled=not bool(self.error) and not self.closed,
            real_sand_particle_count=len(self.metric_indices), estimated_particle_count=len(self.metric_indices),
            particle_mass=self.metric_mass, particle_radius=self.radius,
            particle_spacing_xy=self.spacing, particle_spacing_z=self.spacing,
            mpm_grid_size=self.voxel, mpm_steps=self.steps, mpm_time=self.sim_time,
            mpm_peak_force_N=self.peak_force, mpm_render_grains=int(self.grains.size),
            mpm_render_diameter_m=self.render_diameter,
            mpm_material_points=len(self.positions), mpm_material_point_mass=self.mass,
            mpm_sleeping=self.sleeping, mpm_solver_substeps=self.solver_substeps,
            mpm_material=dict(self.material), real_sand_error=self.error,
        )

    def diagnostics(self):
        return {"active": not self.closed, "error": self.error, "steps": self.steps,
                "time": self.sim_time, "grid_m": self.voxel, "device": self.device,
                "points": len(self.positions), "point_mass_kg": self.mass,
                "spacing_m": self.spacing, "point_budget": self.max_points,
                "substep_mode": "fixed_120hz", "contact_reduction": "gpu",
                "host_contact_values": self.host_contact_values,
                "physics_wall_seconds": self.physics_wall_seconds,
                "render_update_wall_seconds": self.render_wall_seconds,
                "render_updates": self.render_updates,
                "metric_points": len(self.metric_indices), "metric_point_mass_kg": self.metric_mass,
                "sleeping": self.sleeping, "solver_substeps": self.solver_substeps,
                "max_particle_speed": self.max_particle_speed,
                "render_grains": int(self.grains.size), "peak_force_N": self.peak_force,
                "render_grain_diameter_m": self.render_diameter,
                "bodies": list(self.body_paths), "colliders": list(self.collision_paths),
                "last_link_wrenches": self.last_force.tolist(), "step_wall_seconds": self.step_wall_seconds}

    @property
    def metric_positions(self):
        return self.positions[self.metric_indices]

    def close(self):
        self.closed = True
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None
        if self.force_subscription is not None:
            self.force_subscription.unsubscribe()
            self.force_subscription = None
        if self.update_subscription is not None:
            self.update_subscription.unsubscribe()
            self.update_subscription = None
        self.body_view = self.sim_view = None
