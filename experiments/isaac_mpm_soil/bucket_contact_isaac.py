"""Small dynamic open scoop + Newton rigid/MPM two-way coupling, viewed in Isaac.

Based on bundled Newton example_mujoco_mpm_coupled_solver.py and
example_mpm_grain_rendering.py. No excavator scene or PhysX contact is used.
"""
import argparse
import math
import os
import time
from pathlib import Path


def dig_target(t):
    """Smooth targets for a force-driven scoop; positions are never imposed."""
    start = (-1.0, 0.0, 0.10, 0.0)
    stages = [
        (1.2, start, "settle"),
        (2.0, (-0.65, 0.0, 0.08, 0.0), "approach"),
        (3.8, (0.05, 0.0, 0.065, 0.0), "cut"),
        (5.2, (0.20, 0.0, 0.20, -0.95), "curl"),
        (6.6, (0.20, 0.0, 0.65, -0.95), "lift"),
        (8.0, (0.20, 0.0, 0.65, -0.95), "hold"),
    ]
    previous_time, previous = 0.0, start
    for end, target, phase in stages:
        if t <= end:
            duration = end - previous_time
            u = max(0.0, min(1.0, (t - previous_time) / duration))
            blend = u * u * (3 - 2 * u)
            rate = 6 * u * (1 - u) / duration
            position = tuple(a + blend * (b - a) for a, b in zip(previous, target))
            velocity = tuple(rate * (b - a) for a, b in zip(previous, target))
            return position, velocity, phase
        previous_time, previous = end, target
    return previous, (0.0, 0.0, 0.0, 0.0), "hold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold", action="store_true",
                        help="Continue physics after the motion until the GUI closes")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--dig", action="store_true", help="Force-driven cut, curl and lift instead of free fall")
    parser.add_argument("--diagnostics", action="store_true", help="Print material/render checks to stdout only")
    parser.add_argument("--voxel-size", type=float, default=0.025,
                        help="MPM grid edge in metres; use 0.05 for the coarse-grid baseline")
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 8.0 if args.dig else 4.0
    if args.seconds <= 0 or not math.isfinite(args.seconds) or (args.hold and args.headless):
        parser.error("seconds must be positive; hold requires GUI")
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("voxel-size must be finite and positive")
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parent / ".warp_cache"))
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": args.headless, "width": 1280, "height": 800,
        "renderer": "RayTracedLighting", "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
        "extra_args": ["--/app/settings/persistent=0", "--/app/file/ignoreUnsavedStage=true",
                       "--/renderer/multiGpu/enabled=false",
                       "--/app/window/title=" + ("Newton MPM Simple Dig" if args.dig else "Newton MPM Bucket Contact")],
    })
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.warp.core")
        enable_extension("isaacsim.pip.newton")
        for _ in range(3):
            app.update()
        import newton
        import warp as wp
        from newton.solvers import SolverImplicitMPM, SolverMuJoCo
        from newton.solvers.experimental.coupled import SolverCoupledProxy
        from pxr import Gf, UsdGeom, UsdLux, Vt
        from omni.kit.viewport.utility import get_active_viewport

        wp.set_device("cuda:0")
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0, 0, -9.81))
        SolverImplicitMPM.register_custom_attributes(builder)
        # One dynamic compound rigid body: bottom, back and two side plates.
        # Positive X is the open mouth; the top remains open as well.
        initial_position = dig_target(0)[0][:3] if args.dig else (0.10, 0, 1.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(*initial_position), wp.quat_identity()), label="TestBucket")
        plates = [
            ((0, 0, 0), (0.22, 0.20, 0.02)),
            ((-0.20, 0, 0.13), (0.02, 0.20, 0.11)),
            ((0, -0.18, 0.13), (0.22, 0.02, 0.11)),
            ((0, 0.18, 0.13), (0.22, 0.02, 0.11)),
        ]
        for index, (center, halfsize) in enumerate(plates):
            builder.add_shape_box(body=body,
                                  xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                                  hx=halfsize[0], hy=halfsize[1], hz=halfsize[2],
                                  cfg=newton.ModelBuilder.ShapeConfig(density=700, mu=0.5),
                                  label=f"BucketPlate{index}")
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        voxel = args.voxel_size
        # Keep the same initial material points, jitter, masses and visual
        # grain size when comparing grids. Only the solver grid is refined.
        sampling_voxel = 0.05
        grain_radius = sampling_voxel / 24
        extent = np.array([0.5, 0.5, 0.7])
        counts = np.ceil(extent / (sampling_voxel / 3)).astype(int)
        spacing = extent / counts
        origin = np.array([-0.25, -0.25, 0.08]) + spacing / 2
        builder.add_particle_grid(
            pos=wp.vec3(*origin), rot=wp.quat_identity(), vel=wp.vec3(0),
            dim_x=int(counts[0]), dim_y=int(counts[1]), dim_z=int(counts[2]),
            cell_x=float(spacing[0]), cell_y=float(spacing[1]), cell_z=float(spacing[2]),
            mass=float(1600 * np.prod(spacing)), jitter=float(min(spacing)),
            radius_mean=float(np.prod(spacing)**(1 / 3) / 2),
            custom_attributes={"mpm:friction": 0.68, "mpm:young_modulus": 1e15,
                               "mpm:yield_pressure": 1e12, "mpm:yield_stress": 0.0,
                               "mpm:tensile_yield_ratio": 0.0, "mpm:dilatancy": 0.0},
        )
        model = builder.finalize(device="cuda:0")
        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = voxel
        mpm_config.max_iterations = 150
        mpm_config.tolerance = 1e-4
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupledProxy.Entry(name="rigid", solver=lambda v: SolverMuJoCo(
                    model=v, use_mujoco_contacts=False, njmax=100),
                    bodies=[body], joints=list(range(model.joint_count)), substeps=2),
                SolverCoupledProxy.Entry(name="mpm", solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                                         particles=list(range(model.particle_count)), in_place=True),
            ],
            coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(
                source="rigid", destination="mpm", bodies=[body], mass_scale=1.0,
                mode="lagged", collision_pipeline=lambda _model: None)], iterations=1),
        )
        mpm = coupled.solver("mpm")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        coupled.sync_entry_states(state)
        previous = model.state()
        local = coupled.entry_state("mpm")
        grains = mpm.sample_render_grains(local, 8)
        pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
        contacts = pipeline.contacts()
        control = model.control()
        bucket_mass = float(model.body_mass.numpy()[body])
        bucket_com = model.body_com.numpy()[body]
        particle_mass = 1600 * float(np.prod(spacing))

        def rotation_matrix(q):
            x, y, z, w = q
            return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

        def apply_drive(t):
            target, speed, phase = dig_target(t)
            pose = state.body_q.numpy()[body]
            velocity = state.body_qd.numpy()[body]
            pitch = target[3]
            target_q = np.array([0, math.sin(pitch/2), 0, math.cos(pitch/2)])
            current_q = pose[3:7]
            current_rotation = rotation_matrix(current_q)
            target_rotation = rotation_matrix(target_q)
            com_now = pose[:3] + current_rotation @ bucket_com
            com_target = np.array(target[:3]) + target_rotation @ bucket_com
            angular_target = np.array([0, speed[3], 0])
            velocity_target = np.array(speed[:3]) + np.cross(angular_target, target_rotation @ bucket_com)
            # Bounded external wrench acts at the COM. Soil can slow or deflect
            # the bucket; it is not a kinematic object or a pose teleport.
            drive_force = 4000*(com_target-com_now) + 400*(velocity_target-velocity[:3])
            drive_force[2] += bucket_mass * 9.81
            drive_force *= min(1.0, 1200.0 / max(float(np.linalg.norm(drive_force)), 1e-9))
            conjugate = np.array([-current_q[0], -current_q[1], -current_q[2], current_q[3]])
            error_vec = target_q[3]*conjugate[:3] + conjugate[3]*target_q[:3] + np.cross(target_q[:3], conjugate[:3])
            error_w = target_q[3]*conjugate[3] - np.dot(target_q[:3], conjugate[:3])
            rotation_error = 2 * (1 if error_w >= 0 else -1) * error_vec
            drive_torque = 80*rotation_error + 8*(angular_target-velocity[3:])
            drive_torque *= min(1.0, 40.0 / max(float(np.linalg.norm(drive_torque)), 1e-9))
            wrench = np.zeros((model.body_count, 6), dtype=np.float32)
            wrench[body, :3] = drive_force
            wrench[body, 3:] = drive_torque
            state.body_f.assign(wrench)
            return phase, float(np.linalg.norm(com_target-com_now))

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
        ground.CreatePointsAttr([(-3,-3,0),(3,-3,0),(3,3,0),(-3,3,0)])
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0,1,2,3])
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.24,0.27,0.30)])
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000)
        instancer = UsdGeom.PointInstancer.Define(stage, "/World/Soil")
        grain = UsdGeom.Sphere.Define(stage, "/World/Soil/Grain")
        grain.CreateRadiusAttr(grain_radius)
        grain.CreateDisplayColorAttr([Gf.Vec3f(0.47,0.27,0.10)])
        instancer.CreatePrototypesRel().SetTargets([grain.GetPath()])
        instancer.CreateProtoIndicesAttr([0] * grains.size)
        instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(grains.flatten().numpy()))
        bucket = UsdGeom.Xform.Define(stage, "/World/Bucket")
        position_op = bucket.AddTranslateOp()
        orientation_op = bucket.AddOrientOp()
        for index, (center, halfsize) in enumerate(plates):
            plate = UsdGeom.Cube.Define(stage, f"/World/Bucket/Plate{index}")
            plate.CreateSizeAttr(1.0)
            plate.AddTranslateOp().Set(Gf.Vec3d(*center))
            plate.AddScaleOp().Set(Gf.Vec3f(*(2 * v for v in halfsize)))
            plate.CreateDisplayColorAttr([Gf.Vec3f(0.85,0.55,0.06)])
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01,100))
        camera.CreateFocalLengthAttr(30)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.8,-2.3,1.6), Gf.Vec3d(0,0,0.35), Gf.Vec3d(0,0,1)).GetInverse())
        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = camera.GetPath()
        def draw_bucket():
            pose = state.body_q.numpy()[body]
            position_op.Set(Gf.Vec3d(*map(float, pose[:3])))
            orientation_op.Set(Gf.Quatf(float(pose[6]), Gf.Vec3f(*map(float, pose[3:6]))))
            return pose
        draw_bucket()
        label = None
        controls = {"paused": False, "material_view": False}
        if not args.headless:
            import omni.ui as ui
            window = ui.Window("Newton rigid + MPM", width=350, height=180)
            def toggle_pause():
                controls["paused"] = not controls["paused"]
                pause_button.text = "Resume physics" if controls["paused"] else "Pause physics"
            def toggle_material_view():
                controls["material_view"] = not controls["material_view"]
                material_button.text = "Show fine grains" if controls["material_view"] else "Show physical material points"
                count = model.particle_count if controls["material_view"] else grains.size
                instancer.GetProtoIndicesAttr().Set([0] * count)
                grain.GetRadiusAttr().Set(float(min(spacing) / 4) if controls["material_view"] else grain_radius)
            with window.frame:
                with ui.VStack(spacing=6):
                    label = ui.Label("Dynamic open bucket\nStarting...", word_wrap=True)
                    pause_button = ui.Button("Pause physics", clicked_fn=toggle_pause, height=30)
                    material_button = ui.Button("Show physical material points", clicked_fn=toggle_material_view, height=30)
        print(f"BUCKET_READY mode={'dig' if args.dig else 'drop'} grid_m={voxel:.4f} mass_kg={bucket_mass:.3f} material_points={model.particle_count} grains={grains.size}", flush=True)
        for _ in range(10):
            app.update()
        peak_force = 0.0
        start = time.perf_counter()
        dt = 1 / 120
        motion_frames = math.ceil(args.seconds * 60)
        frame = 0
        while app.is_running() and (args.hold or frame < motion_frames):
            if controls["paused"]:
                if label:
                    label.text = f"PHYSICS PAUSED | t={frame/60:.2f}s\nClick Resume physics to continue."
                app.update()
                time.sleep(1/60)
                continue
            for substep in range(2):
                local = coupled.entry_state("mpm")
                previous.particle_q.assign(local.particle_q)
                previous.mpm.particle_transform.assign(local.mpm.particle_transform)
                state.clear_forces()
                phase, drive_error = apply_drive((frame*2+substep)*dt) if args.dig else ("drop", 0.0)
                pipeline.collide(state, contacts)
                coupled.step(state, state, control, contacts, dt)
                local = coupled.entry_state("mpm")
                mpm.update_particle_frames(previous, local, dt)
                mpm.update_render_grains(previous, local, grains, dt)
                state.mpm.particle_transform.assign(local.mpm.particle_transform)
                impulses, _positions, ids = mpm.collect_collider_impulses(local)
                impulse_values, collider_ids = impulses.numpy(), ids.numpy()
                body_mapping = mpm.collider_body_index.numpy()
                valid = (collider_ids >= 0) & (collider_ids < len(body_mapping))
                selected = np.zeros(len(collider_ids), dtype=bool)
                selected[valid] = body_mapping[collider_ids[valid]] >= 0
                force = impulse_values[selected].sum(axis=0) / dt
                peak_force = max(peak_force, float(np.linalg.norm(force)))
            pose = draw_bucket()
            sand = state.particle_q.numpy()
            if not (np.isfinite(pose).all() and np.isfinite(sand).all()):
                raise RuntimeError("Non-finite coupled state")
            local_sand = (sand - pose[:3]) @ rotation_matrix(pose[3:7])
            inside = ((local_sand[:,0] > -0.18) & (local_sand[:,0] < 0.20) &
                      (np.abs(local_sand[:,1]) < 0.16) &
                      (local_sand[:,2] > 0.025) & (local_sand[:,2] < 0.24))
            carried_mass = float(np.count_nonzero(inside) * particle_mass)
            grain_positions = grains.numpy()
            display_positions = sand if controls["material_view"] else grain_positions.reshape(-1, 3)
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(display_positions))
            if label:
                label.text = f"PHYSICS RUNNING | {phase.upper()} | t={(frame+1)/60:.2f}s\nGrid: {voxel*100:g} cm | Soil force Z: {force[2]:.1f} N\nSoil inside (estimate): {carried_mass:.2f} kg"
            app.update()
            if frame % (30 if frame < motion_frames else 120) == (29 if frame < motion_frames else 119):
                print(f"BUCKET_STEP t={(frame+1)/60:.2f} phase={phase} x={pose[0]:.4f} z={pose[2]:.4f} soil_force_z={force[2]:.3f} peak_force={peak_force:.3f} carried_kg={carried_mass:.3f} drive_error_m={drive_error:.4f}", flush=True)
                if args.diagnostics and frame >= 359:
                    # Compare passive display samples to their owning material
                    # points. No particles are deleted or moved by this check.
                    offsets = np.linalg.norm(grain_positions - sand[:, None, :], axis=2)
                    velocities = state.particle_qd.numpy()
                    ground_soil = (~inside) & (sand[:, 2] < 0.45)
                    point_z = sand[ground_soil, 2]
                    render_z = grain_positions[ground_soil, :, 2].ravel()
                    top = ground_soil & (sand[:, 2] > np.quantile(point_z, 0.99))
                    print(f"BUCKET_DIAG t={(frame+1)/60:.2f} grid_field={getattr(local, 'velocity_field', None) is not None} root_local_max_m={np.max(np.abs(sand-local.particle_q.numpy())):.6g} render_offset_max_m={offsets.max():.6f} material_z_q99_max={np.quantile(point_z, .99):.5f},{point_z.max():.5f} render_z_q99_max={np.quantile(render_z, .99):.5f},{render_z.max():.5f} top_speed_max={np.linalg.norm(velocities[top],axis=1).max():.5f}", flush=True)
            frame += 1
            if frame == motion_frames:
                print("BUCKET_ACTION_DONE physics_continues=" + str(args.hold), flush=True)
        if peak_force < 1.0:
            raise RuntimeError("No measurable soil reaction reached the dynamic bucket")
        print(f"BUCKET_DONE peak_soil_force_N={peak_force:.3f} wall_seconds={time.perf_counter()-start:.1f}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
