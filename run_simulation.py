#!/usr/bin/env python3
"""
ExcavatorVLA Standalone Launch Script
=====================================

This script launches Isaac Sim with the excavator scene and starts the TCP bridge server.

Important:
- This version avoids isaacsim.sensors.camera.Camera.initialize().
- RGB images are captured through the active viewport using capture_viewport_to_buffer().
- This avoids the omni.syntheticdata RGB annotator bug:
  TypeError: Unable to write from unknown dtype, kind=f, size=0
"""

import argparse
import sys
import os
import math
import ctypes
import importlib
import importlib.util
import builtins
import time
from concurrent.futures import ThreadPoolExecutor


def _read_bridge_measured_effort(
    articulation,
    joint_indices,
):
    import numpy as np
    """
    Return measured effort in the exact order used
    by the effort18 training dataset:

      [swing, boom, arm, bucket]
    """
    effort = np.asarray(
        articulation.get_measured_joint_efforts(
            joint_indices=joint_indices,
        ),
        dtype=np.float32,
    ).reshape(-1)

    if effort.size < 4:
        raise RuntimeError(
            "Expected at least four measured "
            "joint-effort values, but got "
            f"shape {effort.shape}: "
            f"{effort.tolist()}"
        )

    effort = effort[:4]

    if not np.all(
        np.isfinite(
            effort
        )
    ):
        raise RuntimeError(
            "Measured joint effort contains "
            "NaN or Inf: "
            f"{effort.tolist()}"
        )

    return effort


# This standalone pipeline owns sand creation.
# excavator_runtime.py must not schedule any later sand reset.
os.environ["EXCAVATOR_SINGLE_SAND_CREATE"] = "1"
os.environ["EXCAVATOR_AUTO_RESET_SAND_AFTER_WORLD_READY"] = "0"
os.environ["EXCAVATOR_AUTO_RESET_SAND_AFTER_UI_READY"] = "0"

try:
    from PIL import Image
except Exception:
    Image = None

# Resolve project-local imports deterministically.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for import_root in [
    SCRIPT_DIR,
    os.environ.get("EXCAVATOR_PROJECT_ROOT", ""),
]:
    if import_root:
        import_root = os.path.abspath(import_root)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)


def _add_import_roots(project_root):
    import_roots = [
        project_root,
        os.path.join(project_root, "scripts"),
    ]

    for import_root in reversed(import_roots):
        import_root = os.path.abspath(import_root)

        while import_root in sys.path:
            try:
                sys.path.remove(import_root)
            except ValueError:
                break

        sys.path.insert(0, import_root)


def _clear_runtime_module_cache():
    prefixes = (
        "excavator_app",
    )
    cleared = []

    for name in sorted(list(sys.modules.keys())):
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            sys.modules.pop(name, None)
            cleared.append(name)

    importlib.invalidate_caches()

    if cleared:
        print(
            "[INFO] [RUN SIMULATION CACHE CLEAR]",
            f"modules={cleared}",
            flush=True,
        )


PROJECT_DIR = os.path.abspath(
    os.environ.get("EXCAVATOR_PROJECT_ROOT", SCRIPT_DIR)
)
_add_import_roots(PROJECT_DIR)
_clear_runtime_module_cache()

from excavator_common.deployment_contract import (
    ACTION_NAMES_4D,
    CAMERA_KEYS,
    CANONICAL_DOF_NAMES,
    PHYSICS_HZ,
    PROTOCOL_VERSION,
    PhysicsTickScheduler,
    canonical_values,
    resolve_canonical_dof_indices,
    validate_client_contract,
)

print(
    "[INFO] Run simulation project root:",
    PROJECT_DIR,
    flush=True,
)
print(
    "[INFO] Run simulation entry script:",
    os.path.abspath(__file__),
    flush=True,
)

# Scene and robot paths
SCENE_USD_PATH = os.path.join(PROJECT_DIR, "assets/usd/excavator_scene.usd")
ROBOT_PRIM_PATH = "/World/URDF_real3"
SAND_RUNTIME_PATH = os.path.join(PROJECT_DIR, "scripts", "excavator_app", "sand_site_runtime.py")

# The excavator starts with zero yaw and its arm facing world +X.
# World +Y is therefore slightly left of the arm.
SAND_INITIAL_CENTER = (-0.5, 9.2)

def _load_sand_runtime():
    """
    Load only sand_site_runtime.py under a standalone module name.
    The bootstrap/runtime pipeline is not imported.
    """
    module_name = "_excavator_sand_runtime_standalone"
    runtime_path = os.path.realpath(SAND_RUNTIME_PATH)

    _add_import_roots(PROJECT_DIR)
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        module_name,
        runtime_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load sand runtime: {runtime_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    print(
        "[SAND] Loading standalone runtime:",
        runtime_path,
        flush=True,
    )

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    loaded_path = os.path.realpath(
        getattr(module, "__file__", "")
    )

    if loaded_path != runtime_path:
        raise ImportError(
            "Loaded the wrong sand runtime file: "
            f"expected={runtime_path}, loaded={loaded_path}"
        )

    return module

CAMERA_PRIM_PATHS = {
    "0": "/World/URDF_real3/arm_link/Camera_0",
    "1": "/World/URDF_real3/swing_link/Camera_1",
    "2": "/World/URDF_real3/swing_link/Camera_2",
}

# The visible Isaac Sim viewport remains on this camera while a separate
# off-screen capture viewport cycles through all three model cameras.
DISPLAY_CAMERA_NAME = "0"
CAMERA_PRIM_PATH = CAMERA_PRIM_PATHS[DISPLAY_CAMERA_NAME]


# TCP Bridge settings
HOST = "0.0.0.0"
PORT = 5555

# Viewport capture settings.
# Render at the same 1024x1024 resolution used during dataset collection,
# then downsample to the 256x256 SmolVLA input resolution.
CAPTURE_WIDTH = 1024
CAPTURE_HEIGHT = 1024
MODEL_IMAGE_WIDTH = 256
MODEL_IMAGE_HEIGHT = 256
CAPTURE_WAIT_FRAMES = 5
PENDING_VIEWPORT_CAPTURE_HELPERS = []
LAST_CAPTURE_TIMING_MS = {}
CAPTURE_RESIZE_EXECUTOR = ThreadPoolExecutor(max_workers=3)


# ---------------------------------------------------------------------
# PyCapsule -> numpy helpers for capture_viewport_to_buffer
# ---------------------------------------------------------------------
PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]


def capsule_to_numpy_rgba(capsule, buffer_size, width, height, np_module):
    """
    Convert viewport capture PyCapsule buffer to a numpy RGBA array.

    capture_viewport_to_buffer callback gives:
        capsule, buffer_size, width, height, format

    For TextureFormat.RGBA8_UNORM:
        buffer_size = width * height * 4
    """
    name = PyCapsule_GetName(capsule)
    ptr = PyCapsule_GetPointer(capsule, name)

    if ptr is None or ptr == 0:
        raise RuntimeError("Failed to get pointer from PyCapsule")

    array_type = ctypes.c_uint8 * buffer_size
    c_array = array_type.from_address(ptr)
    arr = np_module.ctypeslib.as_array(c_array)

    channels = buffer_size // (width * height)
    arr = arr.reshape((height, width, channels))

    # Must copy because the original buffer may become invalid after callback.
    return arr.copy()


def cleanup_viewport_capture_helpers(force=False):
    kept = []
    for holder in PENDING_VIEWPORT_CAPTURE_HELPERS:
        if not holder.get("done") and not force:
            kept.append(holder)
            continue
        holder.clear()
    PENDING_VIEWPORT_CAPTURE_HELPERS[:] = kept



def resize_rgb_for_model(rgb, np_module):
    """Downsample a rendered RGB frame to the SmolVLA input resolution."""
    arr = np_module.asarray(rgb)

    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"Unexpected RGB shape before resize: {arr.shape}")

    arr = np_module.ascontiguousarray(arr[:, :, :3])

    if (
        int(arr.shape[1]) == MODEL_IMAGE_WIDTH
        and int(arr.shape[0]) == MODEL_IMAGE_HEIGHT
    ):
        return arr

    if Image is not None:
        image = Image.fromarray(arr)
        resampling = getattr(
            getattr(Image, "Resampling", Image),
            "LANCZOS",
            1,
        )
        image = image.resize(
            (MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT),
            resampling,
        )
        return np_module.ascontiguousarray(
            np_module.asarray(image, dtype=np_module.uint8)
        )

    y_idx = np_module.linspace(
        0,
        max(0, arr.shape[0] - 1),
        MODEL_IMAGE_HEIGHT,
    ).astype(np_module.int32)
    x_idx = np_module.linspace(
        0,
        max(0, arr.shape[1] - 1),
        MODEL_IMAGE_WIDTH,
    ).astype(np_module.int32)

    return np_module.ascontiguousarray(
        arr[y_idx][:, x_idx, :3]
    )


def capture_rgb_from_persistent_viewports(
    capture_views,
    simulation_app,
    capture_viewport_to_buffer,
    np_module,
    wait_frames=CAPTURE_WAIT_FRAMES,
):
    """Capture all fixed-camera viewports in one render window.

    Each camera owns a persistent viewport, so there is no camera switching or
    settle delay.  All callbacks are submitted before Kit is advanced.
    """
    global LAST_CAPTURE_TIMING_MS

    capture_start = time.perf_counter()
    results = {}
    holders = []

    for camera_name, entry in capture_views.items():
        result = {"done": False, "rgb": None, "error": ""}
        results[camera_name] = result
        window = entry.get("window")
        if window is not None:
            try:
                window.visible = True
            except Exception:
                pass

        def on_capture(capsule, buffer_size, width, height, fmt, _result=result):
            try:
                rgba = capsule_to_numpy_rgba(
                    capsule,
                    buffer_size,
                    width,
                    height,
                    np_module,
                )
                _result["rgb"] = np_module.ascontiguousarray(rgba[:, :, :3])
            except Exception as exc:
                _result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                _result["done"] = True

        holder = {
            "done": False,
            "helper": capture_viewport_to_buffer(entry["viewport"], on_capture),
        }
        holders.append(holder)
        PENDING_VIEWPORT_CAPTURE_HELPERS.append(holder)

    render_updates = 0
    for _ in range(max(1, int(wait_frames))):
        # The bridge pauses the timeline before deployment capture, so this
        # renders all persistent viewports without advancing simulation time.
        simulation_app.update()
        render_updates += 1
        if all(result["done"] for result in results.values()):
            break

    for holder in holders:
        holder["done"] = True
    cleanup_viewport_capture_helpers(force=False)

    raw_frames = {}
    for camera_name, result in results.items():
        entry = capture_views[camera_name]
        window = entry.get("window")
        if window is not None:
            try:
                window.visible = False
            except Exception:
                pass
        if result["rgb"] is None:
            print(
                f"[WARN] persistent viewport capture failed [{camera_name}]: "
                f"{result.get('error') or 'timeout'}",
                flush=True,
            )
            raw_frames[camera_name] = np_module.zeros(
                (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3),
                dtype=np_module.uint8,
            )
        else:
            raw_frames[camera_name] = result["rgb"]

    capture_done = time.perf_counter()
    futures = {
        name: CAPTURE_RESIZE_EXECUTOR.submit(resize_rgb_for_model, rgb, np_module)
        for name, rgb in raw_frames.items()
    }
    resized = {name: future.result() for name, future in futures.items()}
    resize_done = time.perf_counter()

    LAST_CAPTURE_TIMING_MS = {
        "camera_render_copy": (capture_done - capture_start) * 1000.0,
        "camera_resize": (resize_done - capture_done) * 1000.0,
        "camera_render_updates": int(render_updates),
    }
    return resized



_JOINT_LIMITS_PRINTED = False


def _print_joint_limits_once(
    articulation,
):
    """
    Print articulation joint names, current positions,
    and USD lower/upper limits once.

    The first four controlled joints are expected to be:

      0 = swing
      1 = boom
      2 = arm
      3 = bucket
    """
    global _JOINT_LIMITS_PRINTED

    if _JOINT_LIMITS_PRINTED:
        return

    import numpy as np

    limits = None
    source = None

    # --------------------------------------------------------
    # API 1
    # --------------------------------------------------------

    if hasattr(
        articulation,
        "get_dof_limits",
    ):
        try:
            limits = np.asarray(
                articulation.get_dof_limits(),
                dtype=np.float32,
            )

            source = (
                "articulation."
                "get_dof_limits()"
            )

        except Exception as exc:
            print(
                "[JOINT LIMITS] "
                "get_dof_limits failed:",
                repr(exc),
            )

    # --------------------------------------------------------
    # API 2
    # --------------------------------------------------------

    if limits is None:

        try:
            view = (
                articulation
                ._articulation_view
            )

            limits = np.asarray(
                view.get_dof_limits(),
                dtype=np.float32,
            )

            source = (
                "_articulation_view."
                "get_dof_limits()"
            )

        except Exception as exc:
            print(
                "[JOINT LIMITS] "
                "articulation view failed:",
                repr(exc),
            )

    # --------------------------------------------------------
    # Remove a leading batch dimension when present:
    #
    #   [1, num_dofs, 2]
    #       ->
    #   [num_dofs, 2]
    # --------------------------------------------------------

    if (
        limits is not None
        and limits.ndim == 3
        and limits.shape[0] == 1
    ):
        limits = limits[0]

    # --------------------------------------------------------
    # Joint names
    # --------------------------------------------------------

    names = None

    for attribute_name in (
        "dof_names",
        "joint_names",
    ):
        try:
            value = getattr(
                articulation,
                attribute_name,
            )

            if value is not None:
                names = list(value)
                break

        except Exception:
            pass

    # --------------------------------------------------------
    # Current joint positions
    # --------------------------------------------------------

    try:
        q = np.asarray(
            articulation
            .get_joint_positions(),
            dtype=np.float32,
        ).reshape(-1)

    except Exception as exc:
        q = None

        print(
            "[JOINT LIMITS] "
            "failed to read q:",
            repr(exc),
        )

    print()
    print(
        "=" * 72
    )

    print(
        "[JOINT LIMITS] source:",
        source,
    )

    print(
        "[JOINT LIMITS] names:",
        names,
    )

    print(
        "[JOINT LIMITS] current q:",
        (
            q.tolist()
            if q is not None
            else None
        ),
    )

    print(
        "[JOINT LIMITS] raw limits:"
    )

    print(
        limits
    )

    if (
        limits is not None
        and limits.ndim == 2
        and limits.shape[1] >= 2
    ):

        print()

        print(
            "[JOINT LIMITS] "
            "first four controlled joints:"
        )

        labels = [
            "swing",
            "boom",
            "arm",
            "bucket",
        ]

        count = min(
            4,
            limits.shape[0],
        )

        for index in range(count):

            lower = float(
                limits[index, 0]
            )

            upper = float(
                limits[index, 1]
            )

            current = (
                float(q[index])
                if (
                    q is not None
                    and index < q.size
                )
                else float("nan")
            )

            distance_to_lower = (
                current - lower
            )

            distance_to_upper = (
                upper - current
            )

            name = (
                names[index]
                if (
                    names is not None
                    and index < len(names)
                )
                else labels[index]
            )

            print(
                f"  index={index} "
                f"name={name} "
                f"q={current:.6f} "
                f"lower={lower:.6f} "
                f"upper={upper:.6f} "
                f"to_lower={distance_to_lower:.6f} "
                f"to_upper={distance_to_upper:.6f}"
            )

    print(
        "=" * 72
    )

    print()

    _JOINT_LIMITS_PRINTED = True


def main(args):
    """Main entry point for standalone launch."""

    # Import Isaac Sim modules after Isaac Sim Python env is set up.
    from isaacsim import SimulationApp
    import carb

    # Create simulation app.
    # Keep headless False because viewport capture needs an active viewport.
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1024,
        "height": 1024,
        "renderer": str(args.renderer),
    })

    # Import remaining Isaac Sim modules after SimulationApp is created.
    import omni.usd
    import omni.kit.app
    import omni.timeline

    from pxr import Usd, UsdGeom, Gf, UsdLux, UsdPhysics, PhysxSchema, Sdf

    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction

    from omni.kit.viewport.utility import (
        get_active_viewport,
        capture_viewport_to_buffer,
        create_viewport_window,
    )

    def wait_for_stage_loading_complete(
        label,
        timeout_updates=3600,
        stable_updates=30,
    ):
        """
        Wait until all referenced USD assets have finished loading.

        The stable-update window prevents sand creation from beginning
        immediately after the loading flag first becomes false.
        """
        context = omni.usd.get_context()
        stable_count = 0
        last_status = None

        print(
            f"[SCENE] Waiting for stage loading before {label}...",
            flush=True,
        )

        for update_index in range(max(1, int(timeout_updates))):
            status = None
            is_loading = False

            try:
                status = context.get_stage_loading_status()
                last_status = status

                if isinstance(status, (tuple, list)):
                    if status and isinstance(status[0], bool):
                        is_loading = bool(status[0])
                    elif len(status) >= 3:
                        try:
                            is_loading = int(status[1]) < int(status[2])
                        except Exception:
                            is_loading = False
                elif isinstance(status, bool):
                    is_loading = status
            except Exception:
                is_loading = False

            world.step(render=True)
            simulation_app.update()

            if is_loading:
                stable_count = 0
            else:
                stable_count += 1

            if stable_count >= max(1, int(stable_updates)):
                print(
                    f"[SCENE] Stage loading complete before {label}:",
                    f"status={last_status}",
                    f"stable_updates={stable_count}",
                    flush=True,
                )
                return last_status

        raise TimeoutError(
            f"Stage did not become stable before {label}; "
            f"last_status={last_status}"
        )



    def hide_sand_source_guides():
        """Hide visual range/debug guides while preserving physical sand."""
        hidden = []

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower_path = path.lower()
            type_name = str(prim.GetTypeName())

            is_visual_guide = (
                "sand" in lower_path
                and (
                    "guide" in lower_path
                    or "range" in lower_path
                    or "selection" in lower_path
                    or "debug" in lower_path
                )
            )

            if not is_visual_guide:
                continue

            if type_name not in (
                "BasisCurves",
                "Mesh",
                "Points",
                "Xform",
            ):
                continue

            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(path)
            except Exception:
                pass

        print(
            "[SAND] Hidden visual guides:",
            hidden,
            flush=True,
        )
        return hidden

    # Import bridge server components.
    import base64
    import json
    import socket
    import struct
    import threading
    import zlib
    import numpy as np
    import queue

    # Import asset converter for GLB conversion.
    # This import passed your test, so we keep it.
    # import omni.kit.asset_converter as asset_converter

    print("=" * 60)
    print("ExcavatorVLA Standalone Launcher")
    print("=" * 60)
    print(f"Scene USD: {SCENE_USD_PATH}")
    print(f"Robot Prim: {ROBOT_PRIM_PATH}")
    print("Camera Prims for Viewport Capture:")
    for camera_name, camera_path in CAMERA_PRIM_PATHS.items():
        print(f"  {camera_name}: {camera_path}")
    print(f"TCP Server: {HOST}:{PORT}")
    print(f"Sand amount: {args.sand_amount} (0=disabled, 1-10=multiplier)")
    print("=" * 60)

    # Get stage.
    stage = get_current_stage()

    def enable_gpu_particle_physics():
        settings = carb.settings.get_settings()

        bool_settings = [
            "/physics/physx/useGpu",
            "/physics/physx/useGPU",
            "/physics/physx/useGpuDynamics",
            "/physics/physx/enableGPUDynamics",
            "/physics/physx/enableGpuDynamics",
            "/physics/physx/gpuDynamicsEnabled",
            "/persistent/physics/physx/useGpu",
            "/persistent/physics/physx/useGPU",
            "/persistent/physics/physx/useGpuDynamics",
            "/persistent/physics/physx/enableGPUDynamics",
            "/persistent/physics/physx/enableGpuDynamics",
            "/persistent/physics/physx/gpuDynamicsEnabled",
        ]

        for key in bool_settings:
            try:
                settings.set_bool(key, True)
            except Exception:
                pass

        try:
            settings.set_string(
                "/physics/physx/broadphaseType",
                "GPU",
            )
        except Exception:
            pass

        try:
            settings.set_string(
                "/persistent/physics/physx/broadphaseType",
                "GPU",
            )
        except Exception:
            pass

        physics_scenes = []

        for prim in stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                physics_scenes.append(prim)

        if not physics_scenes:
            scene = UsdPhysics.Scene.Define(
                stage,
                Sdf.Path("/physicsScene"),
            )
            physics_scenes.append(scene.GetPrim())

        for scene_prim in physics_scenes:
            scene = UsdPhysics.Scene(scene_prim)

            scene.CreateGravityDirectionAttr().Set(
                Gf.Vec3f(0.0, 0.0, -1.0)
            )
            scene.CreateGravityMagnitudeAttr().Set(9.81)

            try:
                physx_api = PhysxSchema.PhysxSceneAPI.Apply(
                    scene_prim
                )
            except Exception:
                physx_api = PhysxSchema.PhysxSceneAPI(
                    scene_prim
                )

            try:
                physx_api.CreateEnableGPUDynamicsAttr().Set(True)
            except Exception:
                pass

            try:
                physx_api.CreateBroadphaseTypeAttr().Set("GPU")
            except Exception:
                pass

            try:
                physx_api.CreateSolverTypeAttr().Set("TGS")
            except Exception:
                pass

            try:
                physx_api.CreateTimeStepsPerSecondAttr().Set(60)
            except Exception:
                pass

            raw_attributes = [
                (
                    "physxScene:enableGPUDynamics",
                    True,
                    Sdf.ValueTypeNames.Bool,
                ),
                (
                    "physxScene:enableGpuDynamics",
                    True,
                    Sdf.ValueTypeNames.Bool,
                ),
                (
                    "physxScene:gpuDynamicsEnabled",
                    True,
                    Sdf.ValueTypeNames.Bool,
                ),
                (
                    "physxScene:broadphaseType",
                    "GPU",
                    Sdf.ValueTypeNames.Token,
                ),
                (
                    "physxScene:solverType",
                    "TGS",
                    Sdf.ValueTypeNames.Token,
                ),
            ]

            for name, value, value_type in raw_attributes:
                attr = scene_prim.GetAttribute(name)

                if not attr.IsValid():
                    attr = scene_prim.CreateAttribute(
                        name,
                        value_type,
                    )

                attr.Set(value)

            print(
                "[PHYSX] GPU dynamics enabled:",
                scene_prim.GetPath(),
                flush=True,
            )

    # Check if scene file exists.
    if not os.path.exists(SCENE_USD_PATH):
        print(f"[ERROR] Scene file not found: {SCENE_USD_PATH}")
        print("Please ensure the excavator_scene.usd file exists.")
        simulation_app.close()
        return

    # Load the USD scene.
    add_reference_to_stage(usd_path=SCENE_USD_PATH, prim_path="/World")
    print(f"[INFO] Loaded scene from: {SCENE_USD_PATH}")

    # Check if robot prim exists.
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        print(f"[ERROR] Robot prim not found: {ROBOT_PRIM_PATH}")
        print("The scene may not contain the expected robot.")
        simulation_app.close()
        return

    print(f"[INFO] Robot prim found: {ROBOT_PRIM_PATH}")

    # Create World.
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

    enable_gpu_particle_physics()

    # Add ground plane.
    world.scene.add_default_ground_plane(
        z_position=0,
        name="ground_plane",
        prim_path="/World/GroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
    )
    print("[INFO] Added ground plane")

    # Add lighting.
    light_path = "/World/MainLight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.CreateIntensityAttr(1000.0)
        light.CreateAngleAttr(0.53)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        print(f"[INFO] Added light at: {light_path}")

    # Add a dome light as extra illumination.
    dome_light_path = "/World/DomeLight"
    if not stage.GetPrimAtPath(dome_light_path).IsValid():
        dome = UsdLux.DomeLight.Define(stage, dome_light_path)
        dome.CreateIntensityAttr(500.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        print(f"[INFO] Added dome light at: {dome_light_path}")

    # Initialize robot articulation.
    robot = SingleArticulation(
        prim_path=ROBOT_PRIM_PATH,
        name="excavator",
    )
    world.scene.add(robot)

    # Load dump truck using GLB converter.
    # TRUCK_GLB_PATH = "/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.glb"
    # TRUCK_USD_PATH = "/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.usd"

    # if os.path.exists(TRUCK_GLB_PATH):
    #     truck_prim_path = "/World/DumpTruck"

    #     # Convert GLB to USD if USD doesn't exist.
    #     if not os.path.exists(TRUCK_USD_PATH):
    #         print(f"[INFO] Converting GLB to USD: {TRUCK_GLB_PATH}")

    #         context = asset_converter.AssetConverterContext()
    #         context.ignore_materials = False
    #         context.export_preview_surface = True
    #         context.use_meter_as_world_unit = True

    #         converter_instance = asset_converter.get_instance()
    #         task = converter_instance.create_converter_task(
    #             TRUCK_GLB_PATH,
    #             TRUCK_USD_PATH,
    #             None,
    #             context,
    #         )

    #         while not task.is_finished():
    #             omni.kit.app.get_app().update()

    #         if task.get_status() != asset_converter.Status.SUCCESS:
    #             print(f"[ERROR] Failed to convert GLB: {task.get_status()}")
    #         else:
    #             print(f"[INFO] GLB converted successfully to: {TRUCK_USD_PATH}")

    #     # Load converted USD file.
    #     if os.path.exists(TRUCK_USD_PATH):
    #         if not stage.GetPrimAtPath(truck_prim_path).IsValid():
    #             truck_prim = stage.DefinePrim(truck_prim_path, "Xform")

    #             xform = UsdGeom.Xformable(truck_prim)
    #             xform.AddTranslateOp().Set(Gf.Vec3d(4.14439, 6.72012, 1.0))
    #             xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    #             xform.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 100.0))

    #             refs = truck_prim.GetReferences()
    #             refs.AddReference(assetPath=TRUCK_USD_PATH)

    #             print(f"[INFO] Loaded truck model at: {truck_prim_path}")
    #         else:
    #             print(f"[INFO] Truck already exists at: {truck_prim_path}")
    #     else:
    #         print(f"[WARN] Truck USD file not found: {TRUCK_USD_PATH}")
    # else:
    #     print(f"[WARN] Truck GLB file not found: {TRUCK_GLB_PATH}")

    # Start physics directly, without a world-level reset.
    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        timeline.play()

    for _ in range(30):
        world.step(render=True)
        simulation_app.update()

    robot.initialize()

    # Sand center is around (0.0, 6.7).
    ROBOT_INITIAL_POS = np.array(
        [-9.2, 6.7, 1.243],
        dtype=np.float32,
    )
    ROBOT_INITIAL_ORI = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    try:
        robot.set_world_pose(
            position=ROBOT_INITIAL_POS,
            orientation=ROBOT_INITIAL_ORI,
        )
        for _ in range(10):
            world.step(render=True)
            simulation_app.update()

        print("[INFO] Robot world pose:", robot.get_world_pose(), flush=True)
    except Exception as e:
        print("[ERROR] Failed to set robot pose:", repr(e), flush=True)


    print("[INFO] World initialized")
    _print_joint_limits_once(robot)
    print("[INFO] Robot joint positions:", robot.get_joint_positions())
    try:
        print("[INFO] DOF names:", robot.dof_names, flush=True)
    except Exception as e:
        print("[WARN] Could not get robot.dof_names:", repr(e), flush=True)

    try:
        print("[INFO] Joint names:", robot.joint_names, flush=True)
    except Exception as e:
        print("[WARN] Could not get robot.joint_names:", repr(e), flush=True)

    raw_dof_names = [str(name) for name in robot.dof_names]
    canonical_to_raw = resolve_canonical_dof_indices(raw_dof_names)
    canonical_joint_indices = np.asarray(canonical_to_raw, dtype=np.int32)
    print(
        "[BRIDGE CONTRACT] DOF mapping:",
        f"raw_dof_names={raw_dof_names}",
        f"canonical_names={list(CANONICAL_DOF_NAMES)}",
        f"canonical_to_raw={list(canonical_to_raw)}",
        flush=True,
    )

    def read_canonical_joint_positions():
        raw = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        return raw, np.asarray(
            canonical_values(raw, canonical_to_raw),
            dtype=np.float32,
        )

    def read_canonical_joint_velocities():
        raw = np.asarray(robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
        return raw, np.asarray(
            canonical_values(raw, canonical_to_raw),
            dtype=np.float32,
        )

    canonical_joint_limits = None
    try:
        raw_limits = np.asarray(robot.get_dof_limits(), dtype=np.float32)
        if raw_limits.ndim == 3:
            raw_limits = raw_limits[0]
        if raw_limits.ndim == 2 and raw_limits.shape[1] >= 2:
            canonical_joint_limits = raw_limits[canonical_joint_indices, :2].copy()
            print(
                "[BRIDGE CONTRACT] canonical_joint_limits_rad=",
                canonical_joint_limits.tolist(),
                flush=True,
            )
    except Exception as exc:
        print("[WARN] Could not resolve canonical joint limits:", repr(exc), flush=True)

    # -----------------------------------------------------------------
    # Viewport setup
    #
    # The visible viewport stays on one fixed camera. A second off-screen
    # viewport is used only for the three-camera model input capture.
    # -----------------------------------------------------------------
    viewport = get_active_viewport()
    if viewport is None:
        print("[ERROR] No active viewport found. Viewport capture cannot run.")
        simulation_app.close()
        return

    # Use only the three excavator-mounted USD cameras.
    active_camera_paths = {}
    for camera_name, camera_path in CAMERA_PRIM_PATHS.items():
        if stage.GetPrimAtPath(camera_path).IsValid():
            active_camera_paths[camera_name] = camera_path
        else:
            print(f"[WARN] Camera prim not found: {camera_path}")

    if not active_camera_paths:
        print("[ERROR] No configured Camera_0/1/2 prims found. Viewport capture cannot run.")
        simulation_app.close()
        return

    display_camera_name = (
        DISPLAY_CAMERA_NAME
        if DISPLAY_CAMERA_NAME in active_camera_paths
        else next(iter(active_camera_paths))
    )
    display_camera_path = active_camera_paths[display_camera_name]
    viewport.camera_path = display_camera_path

    capture_views = {}
    for capture_index, (camera_name, camera_path) in enumerate(active_camera_paths.items()):
        capture_window = create_viewport_window(
            name=f"SmolVLA Capture Viewport {camera_name}",
            usd_context_name="",
            width=CAPTURE_WIDTH,
            height=CAPTURE_HEIGHT,
            position_x=-3000 - (capture_index * (CAPTURE_WIDTH + 20)),
            position_y=-3000,
            camera_path=Sdf.Path(camera_path),
        )
        if capture_window is None:
            raise RuntimeError(
                f"Failed to create persistent capture viewport for camera {camera_name}."
            )
        capture_api = capture_window.viewport_api
        capture_api.camera_path = camera_path
        try:
            capture_window.visible = False
        except Exception:
            pass
        capture_views[camera_name] = {
            "window": capture_window,
            "viewport": capture_api,
            "camera_path": camera_path,
        }

    print("[INFO] Model capture cameras:", flush=True)
    for camera_name, camera_path in active_camera_paths.items():
        print(f"  {camera_name}: {camera_path}", flush=True)

    print(
        "[INFO] Visible viewport fixed to:",
        f"{display_camera_name}: {display_camera_path}",
        flush=True,
    )
    print(
        "[INFO] Persistent off-screen capture viewports created:",
        list(capture_views),
        flush=True,
    )

    if not timeline.is_playing():
        timeline.play()
    print("[INFO] Timeline playing:", timeline.is_playing(), flush=True)

    # Finish all scene construction and referenced-asset loading before sand.
    for _ in range(30):
        world.step(render=True)
        simulation_app.update()

    wait_for_stage_loading_complete(
        label="camera warmup",
        timeout_updates=3600,
        stable_updates=60,
    )

    print("[INFO] Warming up viewport rendering...")
    for i in range(30):
        world.step(render=True)
        simulation_app.update()

    # Test viewport capture from every active camera.
    test_rgbs = capture_rgb_from_persistent_viewports(
        capture_views=capture_views,
        simulation_app=simulation_app,
        capture_viewport_to_buffer=capture_viewport_to_buffer,
        np_module=np,
        wait_frames=CAPTURE_WAIT_FRAMES,
    )

    viewport.camera_path = display_camera_path

    for _ in range(2):
        world.step(render=True)
        simulation_app.update()

    for camera_name, test_rgb in test_rgbs.items():
        print(
            f"[INFO] Initial viewport capture [{camera_name}]:",
            "shape=", test_rgb.shape,
            "dtype=", test_rgb.dtype,
            "min=", int(test_rgb.min()),
            "max=", int(test_rgb.max()),
            "mean=", float(test_rgb.mean()),
            flush=True,
        )

        if float(test_rgb.mean()) == 0.0:
            print(
                f"[WARN] Initial RGB capture from {camera_name} is all black. "
                "The capture pipeline works, but the selected camera may be looking at a dark/empty view.",
                flush=True,
            )

    # All non-sand scene work is complete at this point.
    wait_for_stage_loading_complete(
        label="final sand creation",
        timeout_updates=3600,
        stable_updates=120,
    )

    def inspect_existing_particle_sand():
        """
        Detect particle sand already authored in the loaded USD scene.

        Runtime STATE cannot detect this automatically, so checking only
        real_sand_enabled may incorrectly trigger a second generation.
        """
        sand_root = "/World/SandSite"
        candidates = []

        root_prim = stage.GetPrimAtPath(sand_root)
        if not root_prim.IsValid():
            return {
                "exists": False,
                "particle_count": 0,
                "points_path": "",
                "particle_system_path": "",
            }

        particle_system_path = ""
        for prim in stage.Traverse():
            path = str(prim.GetPath())

            if not (
                path == sand_root
                or path.startswith(sand_root + "/")
            ):
                continue

            if prim.GetTypeName() == "PhysxParticleSystem":
                particle_system_path = path
            elif "particlesystem" in path.lower():
                particle_system_path = particle_system_path or path

            if prim.IsA(UsdGeom.Points):
                try:
                    points = UsdGeom.Points(prim).GetPointsAttr().Get()
                    count = len(points) if points is not None else 0
                except Exception:
                    count = 0

                candidates.append((count, path))

        candidates.sort(reverse=True)

        if candidates and candidates[0][0] > 0:
            particle_count, points_path = candidates[0]
            return {
                "exists": True,
                "particle_count": int(particle_count),
                "points_path": points_path,
                "particle_system_path": particle_system_path,
            }

        return {
            "exists": False,
            "particle_count": 0,
            "points_path": "",
            "particle_system_path": particle_system_path,
        }


    def synchronize_runtime_with_existing_sand(
        sand_module,
        existing_sand,
    ):
        """
        Make sand_site_runtime status reflect the particle set loaded from USD.
        This does not create, clear, or rebuild any prim.
        """
        state = getattr(sand_module, "STATE", None)

        if isinstance(state, dict):
            state["real_sand_enabled"] = True
            state["real_sand_particle_count"] = int(
                existing_sand["particle_count"]
            )
            state["real_sand_error"] = ""
            state["real_sand_points_path"] = str(
                existing_sand["points_path"]
            )
            state["particle_system_path"] = str(
                existing_sand["particle_system_path"]
            )

        store_runtime_api = getattr(
            sand_module,
            "store_runtime_api",
            None,
        )
        if callable(store_runtime_api):
            store_runtime_api()

        print(
            "[SAND] Reusing USD-authored particle sand:",
            f"points={existing_sand['points_path']}",
            f"particles={existing_sand['particle_count']}",
            f"system={existing_sand['particle_system_path']}",
            flush=True,
        )


    existing_sand_before_runtime = inspect_existing_particle_sand()

    print(
        "[SAND] USD sand before runtime import:",
        existing_sand_before_runtime,
        flush=True,
    )

    # -----------------------------------------------------------------
    # Interactive PhysX sand
    #
    # IMPORTANT: this is intentionally the final scene-creation stage.
    # Do not add or reference any USD scene objects after this block.
    # -----------------------------------------------------------------
    sand_amount = int(args.sand_amount)

    if not os.path.isfile(SAND_RUNTIME_PATH):
        raise FileNotFoundError(
            f"Sand runtime not found: {SAND_RUNTIME_PATH}"
        )

    print(
        f"[SAND] Loading runtime: {SAND_RUNTIME_PATH}",
        flush=True,
    )

    enable_gpu_particle_physics()

    for _ in range(5):
        simulation_app.update()

    builtins._SAND_SITE_STARTUP_CONFIG = {
        "amount": sand_amount,
        "show_ui": False,
        "auto_create": False,
        "parameter_mode": "soft_dig",
    }

    for stale_name in (
        "_SAND_SITE",
        "_CONFIGURE_SAND_STARTUP",
    ):
        if hasattr(builtins, stale_name):
            delattr(builtins, stale_name)

    try:
        sand_module = _load_sand_runtime()
    finally:
        if hasattr(builtins, "_SAND_SITE_STARTUP_CONFIG"):
            del builtins._SAND_SITE_STARTUP_CONFIG

    sand_api = getattr(builtins, "_SAND_SITE", None)

    if not isinstance(sand_api, dict):
        raise RuntimeError(
            "sand_site_runtime.py did not create builtins._SAND_SITE"
        )

    set_parameter_mode = (
        sand_api.get("set_sand_parameter_mode")
        or sand_api.get("apply_sand_parameter_mode")
        or getattr(
            sand_module,
            "apply_sand_parameter_mode",
            None,
        )
    )

    if not callable(set_parameter_mode):
        raise RuntimeError(
            "Sand runtime does not expose a parameter-mode setter"
        )

    mode_result = set_parameter_mode(
        "soft_dig",
        announce=True,
    )

    sand_api = getattr(builtins, "_SAND_SITE", sand_api)
    get_status = sand_api.get("get_status")
    sand_status = get_status() if callable(get_status) else {}

    actual_sand_mode = str(
        sand_status.get(
            "sand_parameter_mode",
            getattr(
                sand_module,
                "SAND_PARAMETER_MODE",
                "",
            ),
        )
    ).strip().lower()

    if actual_sand_mode != "soft_dig":
        raise RuntimeError(
            f"Expected sand mode soft_dig, got {actual_sand_mode!r}"
        )

    print(
        "[SAND] Parameter mode verified before creation:",
        actual_sand_mode,
        mode_result,
        flush=True,
    )

    apply_auto_scene_parameters = (
        sand_api.get("apply_auto_scene_parameters")
        or getattr(
            sand_module,
            "apply_auto_scene_parameters",
            None,
        )
    )

    if not callable(apply_auto_scene_parameters):
        raise RuntimeError(
            "Sand runtime does not expose apply_auto_scene_parameters"
        )

    sand_position_result = apply_auto_scene_parameters(
        sand_center_xy=SAND_INITIAL_CENTER,
        rebuild=False,
    )

    sand_api = getattr(builtins, "_SAND_SITE", sand_api)

    print(
        "[SAND] Initial center configured:",
        SAND_INITIAL_CENTER,
        sand_position_result,
        flush=True,
    )

    if sand_amount > 0:
        existing_sand_after_runtime = (
            inspect_existing_particle_sand()
        )

        if existing_sand_after_runtime["exists"]:
            synchronize_runtime_with_existing_sand(
                sand_module,
                existing_sand_after_runtime,
            )

        sand_api = getattr(
            builtins,
            "_SAND_SITE",
            sand_api,
        )
        get_status = sand_api.get("get_status")
        status_before_create = (
            get_status() if callable(get_status) else {}
        )

        already_enabled = bool(
            status_before_create.get(
                "real_sand_enabled",
                False,
            )
        )
        existing_particle_count = int(
            status_before_create.get(
                "real_sand_particle_count",
                0,
            )
            or 0
        )

        print(
            "[SAND] Status immediately after runtime import:",
            f"enabled={already_enabled}",
            f"particles={existing_particle_count}",
            flush=True,
        )

        creation_guard = getattr(
            builtins,
            "_EXCAVATOR_SAND_CREATED_ONCE",
            None,
        )

        if (
            existing_sand_after_runtime["exists"]
            and existing_sand_after_runtime["particle_count"] > 0
        ):
            builtins._EXCAVATOR_SAND_CREATED_ONCE = {
                "source": "usd_scene",
                "particle_count": int(
                    existing_sand_after_runtime[
                        "particle_count"
                    ]
                ),
                "points_path": str(
                    existing_sand_after_runtime[
                        "points_path"
                    ]
                ),
            }
            print(
                "[SAND] USD scene already contains particle sand; "
                "explicit creation is disabled",
                flush=True,
            )
        elif already_enabled and existing_particle_count > 0:
            builtins._EXCAVATOR_SAND_CREATED_ONCE = {
                "source": "runtime_import",
                "particle_count": existing_particle_count,
            }
            print(
                "[SAND] Runtime already created particle sand; "
                "skipping explicit creation",
                flush=True,
            )
        elif creation_guard is not None:
            raise RuntimeError(
                "Sand creation was already attempted in this process: "
                f"{creation_guard}"
            )
        else:
            create_once = (
                getattr(
                    sand_module,
                    "create_real_sand_only",
                    None,
                )
                or sand_api.get("create_real_sand_only")
            )

            if not callable(create_once):
                raise RuntimeError(
                    "Sand runtime did not create particles and does not "
                    "expose create_real_sand_only"
                )

            print(
                "[SAND] Runtime did not create particles; "
                "creating them once explicitly...",
                flush=True,
            )

            final_existing_check = (
                inspect_existing_particle_sand()
            )

            if final_existing_check["exists"]:
                raise RuntimeError(
                    "Refusing to create particle sand because the "
                    "USD scene already contains particles: "
                    f"{final_existing_check}"
                )

            builtins._EXCAVATOR_SAND_CREATED_ONCE = {
                "source": "run_simulation",
                "state": "creating",
            }

            creation_result = create_once()

            builtins._EXCAVATOR_SAND_CREATED_ONCE = {
                "source": "run_simulation",
                "state": "created",
                "result": str(creation_result),
            }

            if hasattr(creation_result, "__await__"):
                raise RuntimeError(
                    "create_real_sand_only must be synchronous "
                    "in run_simulation"
                )

            print(
                "[SAND] Explicit creation call returned:",
                creation_result,
                flush=True,
            )

        print(
            "[SAND] Waiting for particle creation to finish...",
            flush=True,
        )

        creation_complete = False
        last_creation_status = {}

        for _ in range(1800):
            sand_api = getattr(
                builtins,
                "_SAND_SITE",
                sand_api,
            )
            get_status = sand_api.get("get_status")
            last_creation_status = (
                get_status() if callable(get_status) else {}
            )

            enabled = bool(
                last_creation_status.get(
                    "real_sand_enabled",
                    False,
                )
            )
            particle_count = int(
                last_creation_status.get(
                    "real_sand_particle_count",
                    0,
                )
                or 0
            )
            error = str(
                last_creation_status.get(
                    "real_sand_error",
                    "",
                )
                or ""
            )

            if error and not enabled:
                raise RuntimeError(
                    f"Particle sand creation failed: {error}"
                )

            if enabled and particle_count > 0:
                creation_complete = True
                break

            world.step(render=True)
            simulation_app.update()

        if not creation_complete:
            raise TimeoutError(
                "Particle sand creation did not finish; "
                f"last_status={last_creation_status}"
            )

        print(
            "[SAND] Particle creation complete:",
            f"particles={last_creation_status.get('real_sand_particle_count')}",
            flush=True,
        )

    if sand_amount == 0:
        print(
            "[SAND] Disabled by --sand-amount=0",
            flush=True,
        )
    else:
        print(
            "[SAND] Settling interactive particles...",
            flush=True,
        )

        for _ in range(max(1, int(args.sand_settle_frames))):
            world.step(render=True)
            simulation_app.update()

        sand_api = getattr(
            builtins,
            "_SAND_SITE",
            sand_api,
        )

        get_status = sand_api.get("get_status")
        status = get_status() if callable(get_status) else {}

        print(
            "[SAND] Ready:",
            f"enabled={status.get('real_sand_enabled')}",
            f"particles={status.get('real_sand_particle_count')}",
            f"error={status.get('real_sand_error', '')}",
            flush=True,
        )


    # The standalone launcher intentionally does not import the full excavator
    # runtime.  This is the same closed proxy profile used by that runtime when
    # an authored bucket-volume mesh is unavailable.
    bucket_load_profile_xz = np.asarray(
        [
            [-0.08, 0.43],
            [-0.08, -0.08],
            [0.18, -0.30],
            [0.78, -0.24],
            [0.90, 0.12],
            [0.54, 0.50],
        ],
        dtype=np.float32,
    )
    bucket_load_y_min = -0.50
    bucket_load_y_max = 0.50

    def points_in_polygon_2d(points_xz, polygon_xz):
        pts = np.asarray(points_xz, dtype=np.float32).reshape(-1, 2)
        poly = np.asarray(polygon_xz, dtype=np.float32).reshape(-1, 2)
        inside = np.zeros(len(pts), dtype=bool)
        if len(pts) == 0 or len(poly) < 3:
            return inside
        x = pts[:, 0]
        z = pts[:, 1]
        xj, zj = float(poly[-1, 0]), float(poly[-1, 1])
        for vertex in poly:
            xi, zi = float(vertex[0]), float(vertex[1])
            crosses = ((zi > z) != (zj > z)) & (
                x < (xj - xi) * (z - zi) / (zj - zi + 1.0e-9) + xi
            )
            inside ^= crosses
            xj, zj = xi, zi
        return inside

    def estimate_bucket_load_particles():
        start = time.perf_counter()
        result = {
            "count": 0,
            "source": "closed_proxy_profile",
            "candidate_count": 0,
            "particle_count": 0,
            "elapsed_ms": 0.0,
        }
        try:
            runtime_api = getattr(builtins, "_SAND_SITE", sand_api)
            positions_fn = runtime_api.get("particle_positions_fn") if isinstance(runtime_api, dict) else None
            if not callable(positions_fn):
                result["source"] = "particle_positions_unavailable"
                return result
            points = np.asarray(positions_fn(), dtype=np.float32).reshape(-1, 3)
            result["particle_count"] = int(len(points))
            if len(points) == 0:
                return result

            bucket_prim = stage.GetPrimAtPath("/World/URDF_real3/bucket_link")
            if not bucket_prim.IsValid():
                result["source"] = "bucket_prim_unavailable"
                return result
            world_xf = UsdGeom.Xformable(bucket_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            inverse_xf = world_xf.GetInverse()

            local_min = np.asarray(
                [bucket_load_profile_xz[:, 0].min(), bucket_load_y_min, bucket_load_profile_xz[:, 1].min()],
                dtype=np.float32,
            )
            local_max = np.asarray(
                [bucket_load_profile_xz[:, 0].max(), bucket_load_y_max, bucket_load_profile_xz[:, 1].max()],
                dtype=np.float32,
            )
            world_corners = []
            for x_value in (local_min[0], local_max[0]):
                for y_value in (local_min[1], local_max[1]):
                    for z_value in (local_min[2], local_max[2]):
                        p = world_xf.Transform(
                            Gf.Vec3d(float(x_value), float(y_value), float(z_value))
                        )
                        world_corners.append([float(p[0]), float(p[1]), float(p[2])])
            world_corners = np.asarray(world_corners, dtype=np.float32)
            world_min = world_corners.min(axis=0) - 0.02
            world_max = world_corners.max(axis=0) + 0.02
            candidate_mask = np.all(points >= world_min, axis=1) & np.all(points <= world_max, axis=1)
            candidates = points[candidate_mask]
            result["candidate_count"] = int(len(candidates))
            if len(candidates) == 0:
                return result

            local = np.empty_like(candidates)
            for index, point in enumerate(candidates):
                transformed = inverse_xf.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                local[index] = [
                    float(transformed[0]),
                    float(transformed[1]),
                    float(transformed[2]),
                ]
            inside_y = (local[:, 1] >= bucket_load_y_min) & (local[:, 1] <= bucket_load_y_max)
            inside_xz = points_in_polygon_2d(local[:, [0, 2]], bucket_load_profile_xz)
            result["count"] = int(np.count_nonzero(inside_y & inside_xz))
            return result
        except Exception as exc:
            result["source"] = f"error:{type(exc).__name__}:{exc}"
            return result
        finally:
            result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0

    # TCP Bridge Server functions.
    # No USD prims, references, or other scene objects may be
    # created after the sand block above.
    command_queue = queue.Queue()
    response_queue = queue.Queue()

    bridge_stop = threading.Event()
    bridge_ready = threading.Event()
    bridge_network_state = {"error": "", "connected": False, "connection_id": 0}

    def recv_exact(sock, size):
        chunks = []
        remaining = int(size)
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_json_socket(sock):
        size = struct.unpack("!I", recv_exact(sock, 4))[0]
        if size <= 0 or size > 128 * 1024 * 1024:
            raise ValueError(f"invalid bridge message size: {size}")
        return json.loads(recv_exact(sock, size).decode("utf-8"))

    def write_json_socket(sock, obj):
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        sock.sendall(struct.pack("!I", len(data)) + data)

    def bridge_network_worker():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((HOST, PORT))
                server_socket.listen(1)
                server_socket.settimeout(0.5)
                bridge_ready.set()
                print(f"[bridge] TCP server listening on {HOST}:{PORT}", flush=True)
                while not bridge_stop.is_set():
                    try:
                        client_socket, peer = server_socket.accept()
                    except socket.timeout:
                        continue
                    print(f"[bridge] Client connected: {peer}", flush=True)
                    bridge_network_state["connected"] = True
                    bridge_network_state["connection_id"] += 1
                    connection_id = int(bridge_network_state["connection_id"])
                    try:
                        with client_socket:
                            while not bridge_stop.is_set():
                                message = read_json_socket(client_socket)
                                if not isinstance(message, dict):
                                    raise ValueError("bridge message must be a JSON object")
                                message["_bridge_connection_id"] = connection_id
                                command_queue.put(message)
                                reply = response_queue.get()
                                write_json_socket(client_socket, reply)
                    except (ConnectionError, OSError) as exc:
                        print(f"[bridge] Client disconnected: {peer}: {exc}", flush=True)
                    except Exception as exc:
                        print(f"[bridge] Client error: {peer}: {repr(exc)}", flush=True)
                    finally:
                        bridge_network_state["connected"] = False
        except Exception as exc:
            bridge_network_state["error"] = repr(exc)
            bridge_ready.set()
            print(f"[bridge] Server failed: {repr(exc)}", flush=True)

    bridge_thread = threading.Thread(
        target=bridge_network_worker,
        name="ExcavatorBridgeNetwork",
        daemon=True,
    )
    bridge_thread.start()
    if not bridge_ready.wait(timeout=5.0):
        raise RuntimeError("Timed out starting TCP bridge server")
    if bridge_network_state["error"]:
        raise RuntimeError(f"TCP bridge startup failed: {bridge_network_state['error']}")
    builtins._EXCAVATOR_BRIDGE_RUNTIME = {
        "owner": os.path.abspath(__file__),
        "protocol_version": PROTOCOL_VERSION,
        "host": HOST,
        "port": PORT,
        "timing_mode": "deterministic_lockstep",
        "raw_dof_names": list(raw_dof_names),
        "canonical_dof_names": list(CANONICAL_DOF_NAMES),
        "canonical_to_raw": list(canonical_to_raw),
        "camera_paths": dict(active_camera_paths),
        "camera_mode": "per_camera_persistent_viewport",
    }

    # From this point onward the bridge owns simulation time.  Keeping the
    # timeline paused also lets viewport captures and UI updates render without
    # advancing physics while the VLA client is running inference.
    timeline.pause()

    print(
        "[INFO] Simulation running in bridge-lockstep mode: "
        "each command advances exactly 1/training_fps s. Press Ctrl+C to exit."
    )

    # Run simulation loop.
    PHYSICS_DT = 1.0 / PHYSICS_HZ
    active_contract = None
    active_connection_id = None
    tick_scheduler = None
    last_idle_ui_update = 0.0
    idle_ui_interval = 1.0 / max(0.1, float(args.idle_ui_hz))
    while simulation_app.is_running():
        if not command_queue.empty():
            cmd = command_queue.get()

            if cmd.get("type") == "handshake":
                try:
                    active_contract = validate_client_contract(cmd)
                    active_connection_id = int(cmd["_bridge_connection_id"])
                    tick_scheduler = PhysicsTickScheduler(active_contract["training_fps"])
                    reply = {
                        "type": "handshake_ack",
                        "ok": True,
                        "protocol_version": PROTOCOL_VERSION,
                        "physics_hz": PHYSICS_HZ,
                        "training_fps": active_contract["training_fps"],
                        "timing_mode": "deterministic_lockstep",
                        "raw_dof_names": raw_dof_names,
                        "canonical_dof_names": list(CANONICAL_DOF_NAMES),
                        "canonical_to_raw": list(canonical_to_raw),
                        "action_names": list(ACTION_NAMES_4D),
                        "camera_keys": list(CAMERA_KEYS),
                        "normalization_hash": active_contract.get("normalization_hash", ""),
                    }
                    print("[BRIDGE CONTRACT] accepted:", reply, flush=True)
                except Exception as exc:
                    active_contract = None
                    active_connection_id = None
                    tick_scheduler = None
                    reply = {
                        "type": "handshake_ack",
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print("[BRIDGE CONTRACT] rejected:", reply["error"], flush=True)
                response_queue.put(reply)
                continue

            if active_contract is None or tick_scheduler is None:
                response_queue.put({
                    "type": "error",
                    "error": "A valid protocol-v2 handshake is required before motion.",
                })
                continue
            if int(cmd.get("_bridge_connection_id", -1)) != active_connection_id:
                response_queue.put({
                    "type": "error",
                    "error": "This TCP connection has not completed the deployment handshake.",
                })
                continue
            if "ticks" in cmd:
                response_queue.put({
                    "type": "error",
                    "error": "ticks is not accepted in protocol v2; cadence comes from training_fps.",
                })
                continue

            bridge_step_start = time.perf_counter()
            bridge_step_ticks = tick_scheduler.next_ticks()
            bridge_step_seconds = float(bridge_step_ticks) * PHYSICS_DT

            if cmd.get("joint_velocities") is not None and cmd.get("joint_positions") is None:
                _, q_target = read_canonical_joint_positions()
                vel = np.asarray(cmd["joint_velocities"], dtype=np.float32).reshape(-1)
                if vel.shape != (4,) or not np.all(np.isfinite(vel)):
                    response_queue.put({
                        "type": "error",
                        "error": f"Expected four finite canonical joint velocities, got {vel.tolist()}",
                    })
                    continue
            else:
                response_queue.put({
                    "type": "error",
                    "error": "Protocol v2 deployment accepts joint_velocities only.",
                })
                continue

            physics_start = time.perf_counter()
            timeline.play()
            try:
                for _ in range(bridge_step_ticks):
                    q_target = q_target + vel * PHYSICS_DT
                    if canonical_joint_limits is not None:
                        q_target = np.clip(
                            q_target,
                            canonical_joint_limits[:, 0],
                            canonical_joint_limits[:, 1],
                        )
                    robot.apply_action(
                        ArticulationAction(
                            joint_positions=q_target.astype(np.float32),
                            joint_indices=canonical_joint_indices,
                        )
                    )
                    world.step(render=True)
                    simulation_app.update()
            finally:
                timeline.pause()
            physics_done = time.perf_counter()

            q_raw, q = read_canonical_joint_positions()
            qd_raw, qd = read_canonical_joint_velocities()

            viewport.camera_path = display_camera_path

            rgb_by_camera = capture_rgb_from_persistent_viewports(
                capture_views=capture_views,
                simulation_app=simulation_app,
                capture_viewport_to_buffer=capture_viewport_to_buffer,
                np_module=np,
                wait_frames=CAPTURE_WAIT_FRAMES,
            )

            viewport.camera_path = display_camera_path

            encode_start = time.perf_counter()
            encoded_cameras = {}
            for camera_name, rgb in rgb_by_camera.items():
                rgb = np.asarray(rgb)

                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                if rgb.ndim == 3 and rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]

                rgb_compressed = zlib.compress(rgb.tobytes(), level=1)
                encoded_cameras[camera_name] = {
                    "camera_path": active_camera_paths[camera_name],
                    "rgb_shape": list(rgb.shape),
                    "rgb_dtype": str(rgb.dtype),
                    "rgb_zlib_b64": base64.b64encode(rgb_compressed).decode("ascii"),
                }
            encode_done = time.perf_counter()

            primary_camera = (
                "front" if "front" in encoded_cameras else next(iter(encoded_cameras))
            )
            primary_rgb = encoded_cameras[primary_camera]

            # Compute full 14D observation_state matching dataset schema:
            # [base_x, base_y, base_yaw, swing, boom, arm, bucket,
            #  bucket_load_estimate, tip_x, tip_y, tip_z, load_x, load_y, load_z]
            try:
                base_pos, base_ori = robot.get_world_pose()
                base_x = float(base_pos[0])
                base_y = float(base_pos[1])
                qw = float(base_ori[3])
                qx = float(base_ori[0])
                qy = float(base_ori[1])
                qz = float(base_ori[2])
                base_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            except Exception:
                base_x, base_y, base_yaw = 0.0, 0.0, 0.0

            joint_positions = np.asarray(q, dtype=np.float32).reshape(-1)[:4]
            bucket_load_metrics = estimate_bucket_load_particles()

            tip_xyz = [0.0, 0.0, 0.0]
            load_xyz = [0.0, 0.0, 0.0]
            try:
                bucket_prim = stage.GetPrimAtPath("/World/URDF_real3/bucket_link")
                if bucket_prim.IsValid():
                    xform = UsdGeom.Xformable(bucket_prim)
                    world_xf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    tip_local = Gf.Vec3d(0.75, 0.0, -0.18)
                    load_local = Gf.Vec3d(0.35, 0.0, 0.08)
                    tip_world = world_xf.Transform(tip_local)
                    load_world = world_xf.Transform(load_local)
                    tip_xyz = [float(tip_world[0]), float(tip_world[1]), float(tip_world[2])]
                    load_xyz = [float(load_world[0]), float(load_world[1]), float(load_world[2])]
            except Exception:
                pass

            observation_state = [
                base_x,
                base_y,
                base_yaw,
                float(joint_positions[0]),
                float(joint_positions[1]),
                float(joint_positions[2]),
                float(joint_positions[3]),
                float(bucket_load_metrics["count"]),
                tip_xyz[0],
                tip_xyz[1],
                tip_xyz[2],
                load_xyz[0],
                load_xyz[1],
                load_xyz[2],
            ]

            task_text = "Dig soil from the marked area and dump it into the target container."
            try:
                sand_x = float(getattr(sand_module, "SAND_CENTER_X", 0.0))
                sand_y = float(getattr(sand_module, "SAND_CENTER_Y", 6.7))
                sand_xy = (sand_x, sand_y)
                truck_prim = stage.GetPrimAtPath("/World/DumpTruck")
                if truck_prim.IsValid():
                    truck_xf = UsdGeom.Xformable(truck_prim)
                    truck_pos = truck_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                    unload_xy = (float(truck_pos[0]), float(truck_pos[1]))
                else:
                    unload_xy = None

                robot_xy = (base_x, base_y)
                robot_yaw = base_yaw

                def _dir_label(point_xy, origin_xy):
                    if point_xy is None or origin_xy is None:
                        return "nearby"
                    dx = point_xy[0] - origin_xy[0]
                    dy = point_xy[1] - origin_xy[1]
                    dist = abs(dx) + abs(dy)
                    if dist < 1e-6:
                        return "nearby"
                    fwd = robot_yaw + math.pi * 0.5
                    angle = math.atan2(dy, dx) - fwd
                    labels = ["front", "front-left", "left", "rear-left",
                              "rear", "rear-right", "right", "front-right"]
                    idx = int(math.floor((math.degrees(angle) + 22.5) % 360.0 / 45.0))
                    return labels[idx % len(labels)]

                sand_dir = _dir_label(sand_xy, robot_xy)
                unload_dir = _dir_label(unload_xy, robot_xy)
                sand_s = f"({sand_x:.2f}, {sand_y:.2f})"
                unload_s = f"({unload_xy[0]:.2f}, {unload_xy[1]:.2f})" if unload_xy else "the target container"
                task_text = (
                    f"Dig soil from the sand pile near {sand_s}, {sand_dir} of the excavator, "
                    f"and dump it into the truck bed near {unload_s}, {unload_dir} of the excavator."
                )
            except Exception:
                pass

            reply = {
                "joint_positions": q.tolist(),
                "joint_velocities": qd.tolist(),
                "raw_joint_positions": q_raw.tolist(),
                "raw_joint_velocities": qd_raw.tolist(),
                "observation_state": observation_state,
                "observation_effort": _read_bridge_measured_effort(
                    robot,
                    canonical_joint_indices,
                ).tolist(),
                "bucket_load_metrics": bucket_load_metrics,
                "task_text": task_text,
                "primary_camera": primary_camera,
                "rgb_shape": primary_rgb["rgb_shape"],
                "rgb_dtype": primary_rgb["rgb_dtype"],
                "rgb_zlib_b64": primary_rgb["rgb_zlib_b64"],
                "cameras": encoded_cameras,
                "bridge_timing": {
                    "physics_ms": (physics_done - physics_start) * 1000.0,
                    **dict(LAST_CAPTURE_TIMING_MS),
                    "encode_ms": (encode_done - encode_start) * 1000.0,
                    "bucket_load_ms": float(bucket_load_metrics.get("elapsed_ms", 0.0)),
                    "total_server_ms": (time.perf_counter() - bridge_step_start) * 1000.0,
                    "physics_ticks": int(bridge_step_ticks),
                    "simulated_seconds": float(bridge_step_seconds),
                    "training_fps": float(active_contract["training_fps"]),
                },
            }

            response_queue.put(reply)

        else:
            # Network I/O runs on a dedicated thread.  Keep the paused UI
            # responsive at a low rate instead of continuously RayTracing while
            # the policy uses the GPU.
            now = time.perf_counter()
            if now - last_idle_ui_update >= idle_ui_interval:
                simulation_app.update()
                last_idle_ui_update = now
            else:
                time.sleep(min(0.002, idle_ui_interval))

    bridge_stop.set()
    cleanup_viewport_capture_helpers(force=True)
    CAPTURE_RESIZE_EXECUTOR.shutdown(wait=True)
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ExcavatorVLA simulator with interactive PhysX sand and SmolVLA TCP bridge"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Accepted for compatibility; viewport capture still requires headless=False.",
    )
    parser.add_argument(
        "--sand-amount",
        type=int,
        default=1,
        choices=range(0, 11),
        metavar="0-10",
        help="0 disables sand; 1-10 controls the sand amount multiplier.",
    )
    parser.add_argument(
        "--sand-settle-frames",
        type=int,
        default=240,
        help="Simulation frames used to settle generated particles before the TCP bridge starts.",
    )
    parser.add_argument(
        "--renderer",
        default="RayTracedLighting",
        help="Isaac renderer used when policy observations are captured.",
    )
    parser.add_argument(
        "--idle-ui-hz",
        type=float,
        default=5.0,
        help="Paused UI refresh rate while waiting for policy inference; camera viewports stay hidden.",
    )
    args = parser.parse_args()

    if args.headless:
        print(
            "[WARN] --headless was passed, but viewport capture requires "
            "SimulationApp(headless=False)."
        )

    if args.sand_settle_frames < 1:
        parser.error("--sand-settle-frames must be at least 1")
    if args.idle_ui_hz <= 0.0:
        parser.error("--idle-ui-hz must be positive")

    main(args)
