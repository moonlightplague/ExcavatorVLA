"""Standalone Isaac display + bundled Newton MPM. No scene/backend replacement.

Run through Isaac's python.sh (see run.sh), not system Python.
"""
import argparse
import csv
import json
import math
import os
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=positive, default=10.0)
    parser.add_argument("--voxel-size", type=positive, default=0.035)
    parser.add_argument("--width", type=positive, default=0.5)
    parser.add_argument("--height", type=positive, default=0.7)
    parser.add_argument("--drop-height", type=positive, default=0.08)
    parser.add_argument("--density", type=positive, default=1600)
    parser.add_argument("--friction", type=positive, default=0.68)
    parser.add_argument("--ground-friction", type=positive, default=0.5)
    parser.add_argument("--young-modulus", type=positive, default=1e15)
    parser.add_argument("--shear-strength", type=nonnegative, default=0.0,
                        help="Cohesive shear yield stress in Pa; use 0 for the dry friction-only comparison")
    parser.add_argument("--dilatancy", type=nonnegative, default=0.0,
                        help="MPM dilatancy factor, in [0, 1]")
    parser.add_argument("--particles-per-cell", type=int, default=3,
                        help="Material samples per grid-cell axis (official granular example uses 3)")
    parser.add_argument("--grains-per-particle", type=int, default=8,
                        help="Passive display grains per material sample (official grain example uses 8)")
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--hold", action="store_true", help="Keep GUI open at the final state")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--no-record", action="store_true", help="Display only; do not save reports, data or screenshots")
    args = parser.parse_args()
    if args.substeps < 1:
        parser.error("--substeps must be positive")
    if args.dilatancy > 1:
        parser.error("--dilatancy must be in [0, 1]")
    if args.particles_per_cell < 1 or args.grains_per_particle < 1:
        parser.error("particle and grain sampling counts must be positive")
    if args.headless and args.hold:
        parser.error("--hold requires the GUI")
    output = args.results or Path(__file__).resolve().parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not args.no_record:
        output.mkdir(parents=True, exist_ok=False)
    # Cache only this new experiment's kernels, leaving installed packages alone.
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parent / ".warp_cache"))
    report = {"status": "starting", "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "mode": "bundled Newton SolverImplicitMPM + Isaac USD PointInstancer; not native USD backend MPM"}

    def save():
        if not args.no_record:
            (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    print("SOIL_DISPLAY_ONLY" if args.no_record else f"SOIL_RESULTS {output}", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": args.headless, "width": 1280, "height": 800,
        "renderer": "RayTracedLighting", "active_gpu": 0, "physics_gpu": 0,
        "multi_gpu": False,
        "extra_args": ["--/app/settings/persistent=0", "--/app/file/ignoreUnsavedStage=true",
                       "--/renderer/multiGpu/enabled=false", "--/app/window/title=Isaac MPM Soil Drop"],
    })
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        # Loading the package extensions only; the native backend is not enabled.
        enable_extension("omni.warp.core")
        enable_extension("isaacsim.pip.newton")
        for _ in range(3):
            app.update()
        import warp as wp
        import newton
        from newton.solvers import SolverImplicitMPM
        from pxr import Gf, UsdGeom, UsdLux, Vt
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        wp.set_device("cuda:0")
        dimensions = np.array([args.width, args.width, args.height], dtype=float)
        counts = np.ceil(dimensions / (args.voxel_size / args.particles_per_cell)).astype(int)
        particle_count = int(np.prod(counts))
        if particle_count > 250000:
            raise ValueError(f"{particle_count} particles exceeds this basic test's 250000 limit")
        render_count = particle_count * args.grains_per_particle
        if render_count > 2000000:
            raise ValueError(f"{render_count} display grains exceeds this basic test's 2000000 limit")
        spacing = dimensions / counts
        origin = np.array([-args.width / 2, -args.width / 2, args.drop_height]) + spacing / 2
        mass = args.density * float(np.prod(spacing))
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0, 0, -9.81))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(*origin), rot=wp.quat_identity(), vel=wp.vec3(0),
            dim_x=int(counts[0]), dim_y=int(counts[1]), dim_z=int(counts[2]),
            cell_x=float(spacing[0]), cell_y=float(spacing[1]), cell_z=float(spacing[2]),
            mass=mass, jitter=float(min(spacing)),
            radius_mean=float(np.prod(spacing) ** (1 / 3) / 2),
        )
        ground_config = newton.ModelBuilder.ShapeConfig(mu=args.ground_friction)
        builder.add_ground_plane(cfg=ground_config)
        model = builder.finalize(device="cuda:0")
        material = {"young_modulus": args.young_modulus, "poisson_ratio": 0.3,
                    "friction": args.friction, "yield_pressure": 1e12,
                    "yield_stress": args.shear_strength, "tensile_yield_ratio": 0.0, "viscosity": 0.0,
                    "hardening": 0.0, "dilatancy": args.dilatancy}
        for name, value in material.items():
            getattr(model.mpm, name).fill_(value)
        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size
        config.max_iterations = 250
        config.tolerance = 1e-4
        solver = SolverImplicitMPM(model, config=config)
        current, following = model.state(), model.state()
        initial = current.particle_q.numpy().copy()
        # Follow Newton example_mpm_grain_rendering.py: these finer samples are
        # passive render grains, not additional independently colliding particles.
        grains = solver.sample_render_grains(current, args.grains_per_particle)
        initial_display = grains.flatten().numpy()

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        floor = UsdGeom.Mesh.Define(stage, "/World/GroundDisplay")
        floor.CreatePointsAttr([(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)])
        floor.CreateFaceVertexCountsAttr([4])
        floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        floor.CreateDoubleSidedAttr(True)
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.24, 0.27, 0.30)])
        UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(700)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2200)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -35))
        particles = UsdGeom.PointInstancer.Define(stage, "/World/Soil")
        grain = UsdGeom.Sphere.Define(stage, "/World/Soil/Grain")
        grain.CreateRadiusAttr(args.voxel_size / (3 * args.grains_per_particle))
        grain.CreateDisplayColorAttr([Gf.Vec3f(0.47, 0.27, 0.10)])
        particles.CreatePrototypesRel().SetTargets([grain.GetPath()])
        particles.CreateProtoIndicesAttr([0] * render_count)
        particles.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(initial_display))
        # The spheres are render geometry only, not independent rigid-body grains.
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        camera.CreateFocalLengthAttr(30)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.9, -2.5, 1.6), Gf.Vec3d(0, 0, 0.35), Gf.Vec3d(0, 0, 1)).GetInverse())
        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = camera.GetPath()
        report.update(status="ready", newton_version=version("newton"), warp_version=wp.__version__,
                      device=str(model.device), particles=particle_count, display_grains=render_count,
                      total_mass_kg=particle_count * mass, material=material,
                      spacing_m=spacing.tolist(), native_newton_backend_enabled=False)
        save()
        print("SOIL_READY " + json.dumps(report), flush=True)

        def screenshot(name):
            if args.no_record or not viewport:
                return False
            target = output / name
            capture = capture_viewport_to_file(viewport, str(target))
            for _ in range(40):
                app.update()
                if target.exists() and target.stat().st_size > 0:
                    return True
            return False

        for _ in range(20):
            app.update()
        report["initial_screenshot"] = screenshot("initial.png")
        if not args.no_record:
            np.savez_compressed(output / "initial.npz", positions=initial)
        frame_count = int(math.ceil(args.seconds * 60))
        dt = 1 / (60 * args.substeps)
        records = []
        start = time.perf_counter()
        stream_context = nullcontext(None) if args.no_record else (output / "metrics.csv").open("w", newline="", encoding="utf-8")
        with stream_context as stream:
            writer = None
            if stream is not None:
                writer = csv.DictWriter(stream, fieldnames=["sim_seconds", "wall_seconds", "z_min_m",
                                        "height_p99_m", "radius_p95_m", "speed_p95_m_s", "speed_max_m_s"])
                writer.writeheader()
            completed = 0
            for frame in range(frame_count):
                if not app.is_running():
                    break
                tick = time.perf_counter()
                for _ in range(args.substeps):
                    current.clear_forces()
                    solver.step(current, following, None, None, dt)
                    solver.project_outside(following, following, dt)
                    solver.update_particle_frames(current, following, dt)
                    solver.update_render_grains(current, following, grains, dt)
                    current, following = following, current
                positions = current.particle_q.numpy()
                if not np.isfinite(positions).all():
                    raise RuntimeError("Non-finite MPM positions")
                display_positions = grains.flatten().numpy()
                if not np.isfinite(display_positions).all():
                    raise RuntimeError("Non-finite display grain positions")
                particles.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(display_positions))
                app.update()
                completed = frame + 1
                if completed % 30 == 0 or completed == frame_count:
                    speed = np.linalg.norm(current.particle_qd.numpy(), axis=1)
                    if not np.isfinite(speed).all():
                        raise RuntimeError("Non-finite MPM velocities")
                    record = dict(sim_seconds=completed / 60, wall_seconds=time.perf_counter() - start,
                                  z_min_m=float(positions[:, 2].min()),
                                  height_p99_m=float(np.percentile(positions[:, 2], 99)),
                                  radius_p95_m=float(np.percentile(np.linalg.norm(positions[:, :2], axis=1), 95)),
                                  speed_p95_m_s=float(np.percentile(speed, 95)), speed_max_m_s=float(speed.max()))
                    if writer is not None:
                        writer.writerow(record)
                        stream.flush()
                    records.append(record)
                    print("SOIL_STEP " + json.dumps(record), flush=True)
                    if record["z_min_m"] < -args.voxel_size:
                        raise RuntimeError("Ground penetration exceeds one grid cell")
                if not args.headless:
                    time.sleep(max(0, 1 / 60 - (time.perf_counter() - tick)))
        final = current.particle_q.numpy()
        if not args.no_record:
            np.savez_compressed(output / "final.npz", positions=final, velocities=current.particle_qd.numpy())
        report.update(status="completed" if completed == frame_count else "window_closed_early",
                      sim_seconds=completed / 60, wall_seconds=time.perf_counter() - start,
                      initial_mean_z_m=float(initial[:, 2].mean()), final_mean_z_m=float(final[:, 2].mean()),
                      samples=records, usd_instance_count=len(particles.GetPositionsAttr().Get()))
        report["final_screenshot"] = screenshot("final.png") if app.is_running() else False
        if not args.no_record:
            stage.GetRootLayer().Export(str(output / "final_scene.usda"))
        save()
        print("SOIL_DONE " + json.dumps(report), flush=True)
        if args.hold:
            while app.is_running():
                app.update()
                time.sleep(1 / 60)
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        save()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
