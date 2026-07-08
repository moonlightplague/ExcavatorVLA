import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


OUT_DIR = Path(os.environ.get("CAMERA_DIAG_OUT_DIR", "/tmp/linux_camera_diag")).resolve()
SCENE_PATH = os.environ.get("CAMERA_DIAG_SCENE", "")
HEADLESS = os.environ.get("CAMERA_DIAG_HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off")
RENDERER = os.environ.get("CAMERA_DIAG_RENDERER", "RayTracedLighting")
ACTIVE_GPU = int(os.environ.get("CAMERA_DIAG_ACTIVE_GPU", "0") or 0)
PHYSICS_GPU = int(os.environ.get("CAMERA_DIAG_PHYSICS_GPU", "0") or 0)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_PATHS = {
    "0": "/World/URDF_real3/arm_link/Camera_0",
    "1": "/World/URDF_real3/swing_link/Camera_1",
    "2": "/World/URDF_real3/swing_link/Camera_2",
}

RESULTS = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "out_dir": str(OUT_DIR),
    "scene_path": SCENE_PATH,
    "headless": HEADLESS,
    "renderer": RENDERER,
    "active_gpu": ACTIVE_GPU,
    "physics_gpu": PHYSICS_GPU,
    "python": {
        "executable": sys.executable,
        "version": sys.version,
        "path": sys.path[:20],
    },
    "tests": {},
    "errors": [],
}


simulation_app = SimulationApp(
    {
        "headless": HEADLESS,
        "width": 512,
        "height": 512,
        "renderer": RENDERER,
        "multi_gpu": False,
        "active_gpu": ACTIVE_GPU,
        "physics_gpu": PHYSICS_GPU,
        "extra_args": [
            "--/renderer/multiGpu/autoEnable=0",
            "--/omni/replicator/captureOnPlay=false",
            "--/app/asyncRendering=false",
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/app/player/useFixedTimeStepping=true",
            "--/rtx/post/dlss/execMode=2",
            "--/omni.importer.onshape.enabled=false",
        ],
    }
)

import carb
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdLux


def log(*args):
    print("[CAMERA_DIAG]", *args, flush=True)


def sanitize(value):
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(name, data):
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize(data), f, indent=2, ensure_ascii=False)
    return str(path)


def write_ppm(path, rgb):
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"bad rgb shape: {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    height, width = arr.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def image_stats(data):
    if data is None:
        return {"valid_shape": False, "shape": None, "dtype": None, "size": None, "reason": "none"}

    arr = np.asarray(data)
    out = {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "valid_shape": bool(arr.ndim == 3 and arr.shape[-1] >= 3 and arr.size > 0),
    }
    if not out["valid_shape"]:
        out.update(
            {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "std": 0.0,
                "nonzero_fraction": 0.0,
                "black_like": True,
                "reason": f"bad_shape:{list(arr.shape)}",
            }
        )
        return out

    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        max0 = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max0 <= 1.5:
            rgb_eval = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_eval = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_eval = rgb

    max_value = int(np.nanmax(rgb_eval)) if rgb_eval.size else 0
    mean_value = float(np.nanmean(rgb_eval)) if rgb_eval.size else 0.0
    out.update(
        {
            "min": int(np.nanmin(rgb_eval)) if rgb_eval.size else 0,
            "max": max_value,
            "mean": mean_value,
            "std": float(np.nanstd(rgb_eval)) if rgb_eval.size else 0.0,
            "nonzero_fraction": float(np.count_nonzero(rgb_eval)) / float(rgb_eval.size) if rgb_eval.size else 0.0,
            "black_like": bool(max_value <= 2 and mean_value <= 0.5),
        }
    )
    return out


def collect_settings():
    settings = carb.settings.get_settings()
    keys = [
        "/omni/replicator/captureOnPlay",
        "/app/asyncRendering",
        "/exts/isaacsim.core.throttling/enable_async",
        "/app/player/useFixedTimeStepping",
        "/rtx/post/dlss/execMode",
        "/renderer/activeGpu",
        "/physics/cudaDevice",
        "/app/renderer/resolution/width",
        "/app/renderer/resolution/height",
        "/app/window/width",
        "/app/window/height",
        "/app/useFabricSceneDelegate",
    ]
    out = {}
    for key in keys:
        try:
            out[key] = settings.get(key)
        except Exception as exc:
            out[key] = f"{type(exc).__name__}:{exc}"
    return out


def collect_extensions():
    out = {}
    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        for ext in [
            "omni.replicator.core",
            "omni.syntheticdata",
            "isaacsim.sensors.camera",
            "isaacsim.replicator.writers",
            "omni.kit.viewport.utility",
            "omni.importer.onshape",
        ]:
            try:
                out[ext] = bool(manager.is_extension_enabled(ext))
            except Exception as exc:
                out[ext] = f"{type(exc).__name__}:{exc}"
    except Exception as exc:
        out["extension_manager_error"] = f"{type(exc).__name__}:{exc}"
    return out


def collect_stage_summary():
    stage = omni.usd.get_context().get_stage()
    out = {"stage": bool(stage)}
    if stage is None:
        return out
    counts = {}
    cameras = []
    lights = []
    try:
        for prim in stage.Traverse():
            typ = str(prim.GetTypeName())
            counts[typ] = counts.get(typ, 0) + 1
            if prim.IsA(UsdGeom.Camera):
                cameras.append(str(prim.GetPath()))
            if prim.IsA(UsdLux.Light):
                lights.append(str(prim.GetPath()))
    except Exception as exc:
        out["traverse_error"] = f"{type(exc).__name__}:{exc}"
    out["type_counts_top"] = sorted(counts.items(), key=lambda kv: -kv[1])[:30]
    out["camera_paths"] = cameras[:50]
    out["light_paths"] = lights[:50]
    return out


def define_look_at_camera(stage, path, eye, target):
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    mat = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    camera.AddTransformOp().Set(mat)
    return camera


def camera_world_matrix(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path) if stage else None
    if not prim or not prim.IsValid():
        return None
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def camera_world_pose(path):
    mat = camera_world_matrix(path)
    if mat is None:
        return {"available": False, "path": path}
    try:
        t = mat.ExtractTranslation()
        rows = [[float(mat[r][c]) for c in range(4)] for r in range(4)]
        return {
            "available": True,
            "path": path,
            "position": [float(t[0]), float(t[1]), float(t[2])],
            "world_transform": rows,
        }
    except Exception as exc:
        return {"available": False, "path": path, "reason": f"{type(exc).__name__}:{exc}"}


def add_probe_cube_in_front(camera_path, name, distance=2.0):
    stage = omni.usd.get_context().get_stage()
    mat = camera_world_matrix(camera_path)
    if stage is None or mat is None:
        return {"ok": False, "reason": "stage_or_camera_missing"}
    try:
        pos = mat.Transform(Gf.Vec3d(0.0, 0.0, -float(distance)))
        cube_path = f"/World/CameraDiag/ProbeCube_{name}"
        UsdGeom.Xform.Define(stage, "/World/CameraDiag")
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(0.45)
        cube.AddTranslateOp().Set(pos)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.2)])
        return {"ok": True, "path": cube_path, "position": [float(pos[0]), float(pos[1]), float(pos[2])]}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}:{exc}"}


def add_basic_light_and_markers():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage")
    UsdGeom.Xform.Define(stage, "/World/CameraDiag")
    dome = UsdLux.DomeLight.Define(stage, "/World/CameraDiag/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    cube = UsdGeom.Cube.Define(stage, "/World/CameraDiag/OriginBrightCube")
    cube.CreateSizeAttr(0.8)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.8))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
    return {"dome": "/World/CameraDiag/DomeLight", "cube": "/World/CameraDiag/OriginBrightCube"}


async def app_updates(count):
    app = omni.kit.app.get_app()
    for _ in range(max(0, int(count))):
        await app.next_update_async()


async def try_step_async(label, rt_subframes=16, delta_time=0.0, timeout=20.0, wait_for_render=True):
    row = {
        "label": label,
        "rt_subframes": int(rt_subframes),
        "delta_time": delta_time,
        "timeout": float(timeout),
        "wait_for_render": bool(wait_for_render),
        "ok": False,
        "error": "",
        "elapsed_s": None,
    }
    t0 = time.perf_counter()
    try:
        try:
            pending = rep.orchestrator.step_async(
                delta_time=delta_time,
                pause_timeline=False,
                rt_subframes=int(rt_subframes),
                wait_for_render=bool(wait_for_render),
            )
        except TypeError:
            pending = rep.orchestrator.step_async(
                delta_time=delta_time,
                pause_timeline=False,
                rt_subframes=int(rt_subframes),
            )
        await asyncio.wait_for(pending, timeout=float(timeout))
        row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
        RESULTS["errors"].append(
            {
                "where": f"step_async:{label}",
                "error": row["error"],
                "trace": traceback.format_exc(limit=8),
            }
        )
    finally:
        row["elapsed_s"] = time.perf_counter() - t0
    return row


async def create_render_product_and_annotator(path, name):
    try:
        render_product = rep.create.render_product(path, resolution=(256, 256), name=f"diag_rp_{name}")
    except TypeError:
        render_product = rep.create.render_product(path, resolution=(256, 256))
    try:
        render_product.hydra_texture.set_updates_enabled(True)
    except Exception:
        pass
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return render_product, annotator


async def run_single_camera_capture(test_name, camera_path, attempts=10):
    test = {
        "camera_path": camera_path,
        "camera_pose": camera_world_pose(camera_path),
        "attempts": [],
        "success": False,
        "ppm": "",
    }

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_path) if stage else None
    test["camera_valid"] = bool(prim and prim.IsValid())
    test["camera_type"] = str(prim.GetTypeName()) if prim and prim.IsValid() else None
    if not test["camera_valid"]:
        test["reason"] = "camera_prim_missing"
        return test

    render_product, annotator = await create_render_product_and_annotator(camera_path, test_name)
    test["render_product"] = str(getattr(render_product, "path", str(render_product)))

    await app_updates(10)
    modes = [
        {"delta_time": 0.0, "wait_for_render": True, "rt_subframes": 16},
        {"delta_time": None, "wait_for_render": True, "rt_subframes": 16},
        {"delta_time": 0.0, "wait_for_render": False, "rt_subframes": 16},
        {"delta_time": 0.0, "wait_for_render": True, "rt_subframes": 32},
    ]

    for attempt in range(int(attempts)):
        mode = modes[attempt % len(modes)]
        try:
            render_product.hydra_texture.set_updates_enabled(True)
        except Exception:
            pass
        step_row = await try_step_async(
            f"{test_name}_{attempt}",
            rt_subframes=mode["rt_subframes"],
            delta_time=mode["delta_time"],
            wait_for_render=mode["wait_for_render"],
            timeout=20.0,
        )
        await app_updates(1)
        data = annotator.get_data()
        stats = image_stats(data)
        test["attempts"].append({"i": attempt, "step": step_row, "rgb_stats": stats})
        log(test_name, "attempt", attempt, "step_ok=", step_row["ok"], "stats=", stats)

        if stats.get("valid_shape") and not stats.get("black_like"):
            out = OUT_DIR / f"{test_name}.ppm"
            write_ppm(out, data)
            test["success"] = True
            test["ppm"] = str(out)
            return test

    return test


async def test_clean_scene():
    log("test_clean_scene start")
    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    await app_updates(20)

    stage = ctx.get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    define_look_at_camera(stage, "/World/TestCamera", eye=(0.0, -4.0, 2.0), target=(0.0, 0.0, 0.5))

    await app_updates(30)
    return await run_single_camera_capture("clean_scene", "/World/TestCamera", attempts=12)


async def open_loaded_stage():
    if not SCENE_PATH:
        raise RuntimeError("CAMERA_DIAG_SCENE is empty")
    log("opening scene", SCENE_PATH)
    await omni.usd.get_context().open_stage_async(SCENE_PATH)
    await app_updates(180)
    return collect_stage_summary()


async def test_loaded_stage():
    log("test_loaded_stage start")
    stage_summary = await open_loaded_stage()
    marker_info = add_basic_light_and_markers()
    probe_cubes = {}
    for name, path in CAMERA_PATHS.items():
        probe_cubes[name] = add_probe_cube_in_front(path, name, distance=2.0)
    await app_updates(60)

    out = {
        "stage_summary": stage_summary,
        "marker_info": marker_info,
        "probe_cubes": probe_cubes,
        "camera_tests": {},
    }
    for name, path in CAMERA_PATHS.items():
        out["camera_tests"][name] = await run_single_camera_capture(f"loaded_camera_{name}", path, attempts=18)
    return out


async def main_async():
    try:
        settings = carb.settings.get_settings()
        settings.set("/omni/replicator/captureOnPlay", False)
        settings.set("/app/asyncRendering", False)
        settings.set("/exts/isaacsim.core.throttling/enable_async", False)
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/rtx/post/dlss/execMode", 2)

        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass

        RESULTS["settings_initial"] = collect_settings()
        RESULTS["extensions_initial"] = collect_extensions()

        try:
            timeline = omni.timeline.get_timeline_interface()
            if timeline and not timeline.is_playing():
                timeline.play()
            RESULTS["timeline_play_called"] = True
        except Exception as exc:
            RESULTS["timeline_play_called"] = f"{type(exc).__name__}:{exc}"

        await app_updates(10)
        RESULTS["tests"]["clean_scene"] = await test_clean_scene()

        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass
        RESULTS["tests"]["loaded_stage"] = await test_loaded_stage()
        RESULTS["settings_final"] = collect_settings()
        RESULTS["extensions_final"] = collect_extensions()

    except Exception as exc:
        RESULTS["fatal_error"] = f"{type(exc).__name__}:{exc}"
        RESULTS["fatal_trace"] = traceback.format_exc()
        log("FATAL", RESULTS["fatal_error"])
        print(RESULTS["fatal_trace"], flush=True)
    finally:
        save_json("camera_diag_results.json", RESULTS)
        log("saved", OUT_DIR / "camera_diag_results.json")


task = asyncio.ensure_future(main_async())
while not task.done():
    simulation_app.update()

for _ in range(5):
    simulation_app.update()
simulation_app.close()

