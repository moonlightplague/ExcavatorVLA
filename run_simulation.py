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
    EFFORT_NAMES_4D,
    OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
    OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
    PHYSICS_HZ,
    PROTOCOL_VERSION,
    STATE_NAMES_27D,
    STATE_NAMES_28D,
    PhysicsTickScheduler,
    canonical_values,
    resolve_canonical_dof_indices,
    validate_client_contract,
)
from excavator_common import vla_observation_contract
from excavator_common import deployment_scene_contract

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

SAND_AUTHORED_CENTER = deployment_scene_contract.AUTHORED_SAND_CENTER_XY
TRUCK_ROOT_PATH = deployment_scene_contract.TRUCK_ROOT_PATH
TRUCK_BED_COLLISION_PATH = (
    deployment_scene_contract.TRUCK_BED_COLLISION_PATH
)
_ACTIVE_SAND_POSE_RESTORE = None

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
            raise RuntimeError(
                f"Persistent viewport capture failed [{camera_name}]: "
                f"{result.get('error') or 'timeout'}"
            )
        rgb = np_module.asarray(result["rgb"], dtype=np_module.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise RuntimeError(
                f"Persistent viewport [{camera_name}] returned invalid RGB shape {rgb.shape}"
            )
        if not np_module.any(rgb[:, :, :3]):
            raise RuntimeError(
                f"Persistent viewport [{camera_name}] returned an all-zero RGB frame"
            )
        raw_frames[camera_name] = rgb[:, :, :3]

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
    global _ACTIVE_SAND_POSE_RESTORE

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

    fixed_scene_profile = (
        deployment_scene_contract.validate_fixed_scene_profile()
    )

    def set_prim_translation_and_yaw(
        prim,
        translation_xyz,
        yaw_deg,
    ):
        """Set truck translation/yaw without discarding authored scale ops."""
        if prim is None or not prim.IsValid():
            raise RuntimeError("Cannot transform an invalid USD prim")
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = list(xformable.GetOrderedXformOps())
        translate_op = None
        rotate_op = None
        rotate_types = {
            UsdGeom.XformOp.TypeRotateZ,
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
            UsdGeom.XformOp.TypeOrient,
        }
        for op in ordered_ops:
            op_type = op.GetOpType()
            if (
                translate_op is None
                and op_type == UsdGeom.XformOp.TypeTranslate
            ):
                translate_op = op
            if rotate_op is None and op_type in rotate_types:
                rotate_op = op

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translation = Gf.Vec3d(
            float(translation_xyz[0]),
            float(translation_xyz[1]),
            float(translation_xyz[2]),
        )
        try:
            translate_op.Set(translation)
        except Exception:
            translate_op.Set(Gf.Vec3f(*translation))

        if rotate_op is None:
            rotate_op = xformable.AddRotateXYZOp()
        rotate_type = rotate_op.GetOpType()
        yaw = float(yaw_deg)
        if rotate_type == UsdGeom.XformOp.TypeRotateZ:
            rotate_op.Set(yaw)
        elif rotate_type == UsdGeom.XformOp.TypeOrient:
            half_yaw = math.radians(yaw) * 0.5
            try:
                rotate_op.Set(
                    Gf.Quatd(
                        math.cos(half_yaw),
                        Gf.Vec3d(0.0, 0.0, math.sin(half_yaw)),
                    )
                )
            except Exception:
                rotate_op.Set(
                    Gf.Quatf(
                        math.cos(half_yaw),
                        Gf.Vec3f(0.0, 0.0, math.sin(half_yaw)),
                    )
                )
        elif rotate_type in (
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        ):
            current = rotate_op.Get(Usd.TimeCode.Default())
            rotate_xyz = (
                float(current[0]) if current is not None else 0.0,
                float(current[1]) if current is not None else 0.0,
                yaw,
            )
            try:
                rotate_op.Set(Gf.Vec3d(*rotate_xyz))
            except Exception:
                rotate_op.Set(Gf.Vec3f(*rotate_xyz))
        else:
            raise RuntimeError(
                "Unsupported truck rotation op: "
                f"{rotate_type} on {prim.GetPath()}"
            )

        all_ops = list(xformable.GetOrderedXformOps())
        normalized_order = []
        for op in all_ops:
            if (
                op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                and op not in normalized_order
            ):
                normalized_order.append(op)
        if rotate_op not in normalized_order:
            normalized_order.append(rotate_op)
        for op in all_ops:
            if (
                op.GetOpType() == UsdGeom.XformOp.TypeScale
                and op not in normalized_order
            ):
                normalized_order.append(op)
        for op in all_ops:
            if op not in normalized_order:
                normalized_order.append(op)
        xformable.SetXformOpOrder(normalized_order)

    def prim_world_position(prim):
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        position = matrix.ExtractTranslation()
        return np.asarray(
            [float(position[0]), float(position[1]), float(position[2])],
            dtype=np.float64,
        )

    def prim_local_yaw_z_deg(prim):
        """Read truck yaw with the same local-op semantics as collection."""
        if prim is None or not prim.IsValid():
            raise RuntimeError("Cannot read yaw from an invalid USD prim")
        vector_rotate_types = {
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        }
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            op_type = op.GetOpType()
            if op_type == UsdGeom.XformOp.TypeRotateZ:
                return float(op.Get(Usd.TimeCode.Default()))
            if op_type in vector_rotate_types:
                value = op.Get(Usd.TimeCode.Default())
                if value is None or len(value) < 3:
                    break
                return float(value[2])
            if op_type == UsdGeom.XformOp.TypeOrient:
                value = op.Get(Usd.TimeCode.Default())
                if value is None:
                    break
                imaginary = value.GetImaginary()
                return math.degrees(
                    2.0
                    * math.atan2(
                        float(imaginary[2]),
                        float(value.GetReal()),
                    )
                )
        raise RuntimeError(
            f"No supported local rotation op on truck prim {prim.GetPath()}"
        )

    def apply_fixed_training_truck_pose():
        truck_prim = stage.GetPrimAtPath(TRUCK_ROOT_PATH)
        if not truck_prim.IsValid():
            raise RuntimeError(
                f"Training truck root is unavailable: {TRUCK_ROOT_PATH}"
            )
        bed_prim = stage.GetPrimAtPath(TRUCK_BED_COLLISION_PATH)
        if not bed_prim.IsValid():
            raise RuntimeError(
                "Training dump-bed collision mesh is unavailable: "
                f"{TRUCK_BED_COLLISION_PATH}"
            )
        set_prim_translation_and_yaw(
            truck_prim,
            fixed_scene_profile["truck_translation_xyz"],
            fixed_scene_profile["truck_yaw_deg"],
        )
        actual_position = prim_world_position(truck_prim)
        actual_local_yaw_deg = prim_local_yaw_z_deg(truck_prim)
        expected_position = np.asarray(
            fixed_scene_profile["truck_translation_xyz"],
            dtype=np.float64,
        )
        position_error = float(np.linalg.norm(actual_position - expected_position))
        yaw_error_deg = deployment_scene_contract.wrapped_yaw_error_deg(
            actual_local_yaw_deg,
            fixed_scene_profile["truck_yaw_deg"],
        )
        if position_error > 0.02 or yaw_error_deg > 0.25:
            raise RuntimeError(
                "Truck pose did not match the fixed training profile: "
                f"position_error={position_error:.6f}, "
                f"local_yaw_error_deg={yaw_error_deg:.6f}"
            )
        print(
            "[SCENE CONTRACT] Fixed training truck pose applied:",
            f"root={TRUCK_ROOT_PATH}",
            f"bed={TRUCK_BED_COLLISION_PATH}",
            f"position={actual_position.tolist()}",
            f"local_yaw_deg={actual_local_yaw_deg:.6f}",
            f"unload_landing={list(fixed_scene_profile['unload_landing_xyz'])}",
            flush=True,
        )
        return truck_prim, bed_prim

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
        """Hide sand BBox/debug visuals while preserving particle physics."""
        hidden = []
        preserved = []
        tokens = (
            "bbox", "bounding_box", "boundingbox", "bounding",
            "bounds", "aabb", "range", "guide", "selection", "debug",
        )

        def contains_particle_geometry(root_prim):
            try:
                for descendant in Usd.PrimRange(root_prim):
                    if descendant == root_prim:
                        continue
                    if (
                        str(descendant.GetTypeName()) == "PhysxParticleSystem"
                        or descendant.IsA(UsdGeom.Points)
                    ):
                        return True
            except Exception:
                pass
            return False

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower_path = path.lower()
            type_name = str(prim.GetTypeName())

            if "sand" not in lower_path:
                continue
            if not any(token in lower_path for token in tokens):
                continue
            if type_name == "PhysxParticleSystem" or prim.IsA(UsdGeom.Points):
                preserved.append(path)
                continue
            if type_name == "Xform" and contains_particle_geometry(prim):
                preserved.append(path)
                continue

            try:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                    hidden.append(path)
            except Exception as exc:
                print(
                    "[SAND] Failed to hide visual helper:",
                    path,
                    repr(exc),
                    flush=True,
                )

        print("[SAND] Hidden guide/BBox visuals:", hidden, flush=True)
        if preserved:
            print(
                "[SAND] Preserved particle-related prims:",
                preserved,
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

    missing_camera_names = sorted(set(CAMERA_PRIM_PATHS) - set(active_camera_paths))
    if missing_camera_names:
        print(
            "[ERROR] Required Camera_0/1/2 prims are incomplete: "
            f"missing={missing_camera_names}, available={sorted(active_camera_paths)}"
        )
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

    truck_pose_timeline_was_playing = timeline.is_playing()
    if truck_pose_timeline_was_playing:
        timeline.pause()
        simulation_app.update()
    apply_fixed_training_truck_pose()
    for _ in range(3):
        simulation_app.update()
    if truck_pose_timeline_was_playing:
        timeline.play()

    def elevate_excavator_for_sand_init(sand_floor_z, sand_fill_height):
        """Move the complete articulation above the sand initialization zone."""
        robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not robot_prim.IsValid():
            raise RuntimeError(
                f"Cannot elevate robot; prim is missing: {ROBOT_PRIM_PATH}"
            )
        original_position, original_orientation = robot.get_world_pose()
        original_position = np.asarray(
            original_position,
            dtype=np.float32,
        ).copy()
        original_orientation = np.asarray(
            original_orientation,
            dtype=np.float32,
        ).copy()
        original_joint_positions = np.asarray(
            robot.get_joint_positions(),
            dtype=np.float32,
        ).copy()
        original_joint_velocities = np.asarray(
            robot.get_joint_velocities(),
            dtype=np.float32,
        ).copy()

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )
        robot_range = bbox_cache.ComputeWorldBound(
            robot_prim
        ).ComputeAlignedRange()
        if robot_range.IsEmpty():
            raise RuntimeError("Cannot elevate robot; world bound is empty")
        original_min_z = float(robot_range.GetMin()[2])
        target_min_z = (
            float(sand_floor_z) + float(sand_fill_height) + 1.0
        )
        lift_z = max(2.0, target_min_z - original_min_z)
        elevated_position = original_position.copy()
        elevated_position[2] += float(lift_z)

        try:
            robot.set_world_pose(
                position=elevated_position,
                orientation=original_orientation,
            )
            robot.set_joint_velocities(
                np.zeros_like(original_joint_velocities)
            )
            for _ in range(3):
                simulation_app.update()

            actual_position, _ = robot.get_world_pose()
            actual_position = np.asarray(actual_position, dtype=np.float32)
            position_error = float(
                np.linalg.norm(actual_position - elevated_position)
            )
            elevated_bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.render,
                    UsdGeom.Tokens.proxy,
                ],
                useExtentsHint=True,
            )
            elevated_range = elevated_bbox_cache.ComputeWorldBound(
                robot_prim
            ).ComputeAlignedRange()
            elevated_min_z = float(elevated_range.GetMin()[2])
            if (
                position_error > 0.02
                or elevated_min_z < target_min_z - 0.05
            ):
                raise RuntimeError(
                    "Excavator elevation did not take effect: "
                    f"position_error={position_error:.6f}, "
                    f"elevated_min_z={elevated_min_z:.6f}, "
                    f"target_min_z={target_min_z:.6f}"
                )
        except Exception:
            robot.set_world_pose(
                position=original_position,
                orientation=original_orientation,
            )
            robot.set_joint_positions(original_joint_positions)
            robot.set_joint_velocities(original_joint_velocities)
            simulation_app.update()
            raise
        print(
            "[SAND INIT] Excavator elevated:",
            f"original_position={original_position.tolist()}",
            f"elevated_position={actual_position.tolist()}",
            f"original_min_z={original_min_z:.6f}",
            f"elevated_min_z={elevated_min_z:.6f}",
            f"target_min_z={target_min_z:.6f}",
            f"lift_z={lift_z:.6f}",
            flush=True,
        )
        return {
            "position": original_position,
            "orientation": original_orientation,
            "joint_positions": original_joint_positions,
            "joint_velocities": original_joint_velocities,
        }

    def restore_excavator_pose_after_sand_init(snapshot):
        """Restore the exact base and articulation state saved before lifting."""
        robot.set_world_pose(
            position=snapshot["position"],
            orientation=snapshot["orientation"],
        )
        robot.set_joint_positions(snapshot["joint_positions"])
        robot.set_joint_velocities(snapshot["joint_velocities"])
        for _ in range(3):
            simulation_app.update()

        actual_position, actual_orientation = robot.get_world_pose()
        actual_position = np.asarray(actual_position, dtype=np.float32)
        actual_orientation = np.asarray(
            actual_orientation,
            dtype=np.float32,
        )
        position_error = float(
            np.linalg.norm(actual_position - snapshot["position"])
        )
        actual_orientation /= max(
            1.0e-12,
            float(np.linalg.norm(actual_orientation)),
        )
        expected_orientation = np.asarray(
            snapshot["orientation"],
            dtype=np.float32,
        )
        expected_orientation /= max(
            1.0e-12,
            float(np.linalg.norm(expected_orientation)),
        )
        orientation_dot = float(
            np.clip(
                abs(np.dot(actual_orientation, expected_orientation)),
                0.0,
                1.0,
            )
        )
        orientation_error_deg = math.degrees(
            2.0 * math.acos(orientation_dot)
        )
        actual_joint_positions = np.asarray(
            robot.get_joint_positions(),
            dtype=np.float32,
        )
        joint_error = float(
            np.max(
                np.abs(
                    actual_joint_positions - snapshot["joint_positions"]
                )
            )
        )
        if (
            position_error > 0.02
            or orientation_error_deg > 0.1
            or joint_error > 1.0e-3
        ):
            raise RuntimeError(
                "Failed to restore excavator after sand init: "
                f"position_error={position_error:.6f}, "
                f"orientation_error_deg={orientation_error_deg:.6f}, "
                f"joint_error={joint_error:.6f}"
            )
        print(
            "[SAND INIT] Excavator pose restored:",
            f"position={actual_position.tolist()}",
            f"position_error={position_error:.6f}",
            f"orientation_error_deg={orientation_error_deg:.6f}",
            f"joint_error={joint_error:.6f}",
            flush=True,
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
    excavator_pose_snapshot = None

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

    actual_sand_center = (
        float(getattr(sand_module, "SAND_CENTER_X", float("nan"))),
        float(getattr(sand_module, "SAND_CENTER_Y", float("nan"))),
    )
    if not np.allclose(
        np.asarray(actual_sand_center, dtype=np.float64),
        np.asarray(SAND_AUTHORED_CENTER, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-4,
    ):
        raise RuntimeError(
            "Sand runtime center differs from the authored training center: "
            f"actual={actual_sand_center}, expected={SAND_AUTHORED_CENTER}"
        )
    print(
        "[SCENE CONTRACT] Authored sand center retained:",
        f"center={actual_sand_center}",
        f"world_radius_m={fixed_scene_profile['sand_world_radius_m']:.6f}",
        flush=True,
    )

    if sand_amount > 0:
        excavator_pose_snapshot = elevate_excavator_for_sand_init(
            sand_floor_z=float(
                getattr(sand_module, "SAND_FLOOR_Z", 0.0)
            ),
            sand_fill_height=float(
                getattr(sand_module, "SANDBOX_FILL_HEIGHT", 3.0)
            ),
        )
        _ACTIVE_SAND_POSE_RESTORE = lambda snapshot=(
            excavator_pose_snapshot
        ): restore_excavator_pose_after_sand_init(snapshot)

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

        restore_excavator_pose_after_sand_init(
            excavator_pose_snapshot
        )
        _ACTIVE_SAND_POSE_RESTORE = None
        excavator_pose_snapshot = None


    # Hide sand BBox/range/debug visuals after particle settling.
    hidden_sand_visuals = hide_sand_source_guides()

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
    bucket_volume_mesh_path = (
        "/World/URDF_real3/bucket_link/bucket_cut/node_/mesh_"
    )
    bucket_source_tracker = {
        "available": False,
        "mask": None,
        "particle_count": 0,
        "source_count": 0,
        "reason": "not_captured",
    }

    def mesh_faces_from_usd(mesh):
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        faces = []
        offset = 0
        for count in counts:
            count = int(count)
            if count >= 3 and offset + count <= len(indices):
                faces.append(
                    [int(indices[offset + index]) for index in range(count)]
                )
            offset += max(0, count)
        return faces

    def triangulate_faces(faces):
        triangles = []
        for face in faces:
            if len(face) < 3:
                continue
            for index in range(1, len(face) - 1):
                triangles.append([face[0], face[index], face[index + 1]])
        return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

    def closed_mesh_boundary_edge_count(faces):
        counts = {}
        for face in faces:
            for index, first in enumerate(face):
                second = face[(index + 1) % len(face)]
                key = tuple(sorted((int(first), int(second))))
                counts[key] = int(counts.get(key, 0)) + 1
        return sum(1 for count in counts.values() if count != 2)

    def authored_bucket_volume_topology():
        result = {
            "available": False,
            "source": "",
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "triangles": np.zeros((0, 3), dtype=np.int32),
            "local_min": None,
            "local_max": None,
            "reason": "",
        }
        try:
            mesh_prim = stage.GetPrimAtPath(bucket_volume_mesh_path)
            bucket_prim = stage.GetPrimAtPath("/World/URDF_real3/bucket_link")
            if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
                result["reason"] = f"missing_mesh:{bucket_volume_mesh_path}"
                return result
            if not bucket_prim.IsValid():
                result["reason"] = "missing_bucket_link"
                return result
            mesh = UsdGeom.Mesh(mesh_prim)
            raw_points = list(mesh.GetPointsAttr().Get() or [])
            faces = mesh_faces_from_usd(mesh)
            if len(raw_points) < 4 or not faces:
                result["reason"] = (
                    f"invalid_topology:vertices={len(raw_points)};faces={len(faces)}"
                )
                return result
            boundary_edges = closed_mesh_boundary_edge_count(faces)
            if boundary_edges:
                result["reason"] = f"mesh_not_closed:boundary_edges={boundary_edges}"
                return result
            mesh_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            bucket_world = UsdGeom.Xformable(bucket_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            bucket_world_inverse = bucket_world.GetInverse()
            vertices = []
            for point in raw_points:
                world_point = mesh_world.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                local_point = bucket_world_inverse.Transform(world_point)
                vertices.append(
                    [
                        float(local_point[0]),
                        float(local_point[1]),
                        float(local_point[2]),
                    ]
                )
            vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            triangles = triangulate_faces(faces)
            if len(triangles) == 0:
                result["reason"] = "mesh_has_no_triangles"
                return result
            result.update(
                {
                    "available": True,
                    "source": f"authored_closed_mesh:{bucket_volume_mesh_path}",
                    "vertices": vertices,
                    "triangles": triangles,
                    "local_min": vertices.min(axis=0),
                    "local_max": vertices.max(axis=0),
                    "reason": "ok",
                }
            )
            return result
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}:{exc}"
            return result

    bucket_volume_topology = authored_bucket_volume_topology()
    print(
        "[BRIDGE CONTRACT] bucket volume:",
        f"available={bucket_volume_topology['available']}",
        f"source={bucket_volume_topology['source']}",
        f"reason={bucket_volume_topology['reason']}",
        f"vertices={len(bucket_volume_topology['vertices'])}",
        f"triangles={len(bucket_volume_topology['triangles'])}",
        flush=True,
    )

    def points_in_closed_bucket_mesh(local_points):
        points = np.asarray(local_points, dtype=np.float32).reshape(-1, 3)
        vertices = np.asarray(
            bucket_volume_topology["vertices"],
            dtype=np.float32,
        ).reshape(-1, 3)
        triangles = np.asarray(
            bucket_volume_topology["triangles"],
            dtype=np.int32,
        ).reshape(-1, 3)
        if len(points) == 0 or len(vertices) < 4 or len(triangles) == 0:
            return np.zeros(len(points), dtype=bool)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        counts = np.zeros(len(points), dtype=np.int32)
        epsilon = 1.0e-6
        for triangle in triangles:
            v0 = vertices[int(triangle[0])]
            edge1 = vertices[int(triangle[1])] - v0
            edge2 = vertices[int(triangle[2])] - v0
            h = np.cross(direction, edge2)
            determinant = float(np.dot(edge1, h))
            if abs(determinant) < epsilon:
                continue
            inverse = 1.0 / determinant
            relative = points - v0.reshape(1, 3)
            u = inverse * np.einsum("ij,j->i", relative, h)
            candidate = (u >= -epsilon) & (u <= 1.0 + epsilon)
            if not np.any(candidate):
                continue
            q = np.cross(relative, edge1.reshape(1, 3))
            v = inverse * np.einsum("ij,j->i", q, direction)
            distance = inverse * np.einsum("j,ij->i", edge2, q)
            hit = (
                candidate
                & (v >= -epsilon)
                & ((u + v) <= 1.0 + epsilon)
                & (distance > epsilon)
            )
            counts[hit] += 1
        return (counts % 2) == 1

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

    def capture_initial_pile_source_mask():
        result = {
            "available": False,
            "mask": None,
            "particle_count": 0,
            "source_count": 0,
            "reason": "",
        }
        try:
            runtime_api = getattr(builtins, "_SAND_SITE", sand_api)
            positions_fn = (
                runtime_api.get("particle_positions_fn")
                if isinstance(runtime_api, dict)
                else None
            )
            if not callable(positions_fn):
                result["reason"] = "particle_positions_unavailable"
                return result
            points = np.asarray(
                positions_fn(),
                dtype=np.float32,
            ).reshape(-1, 3)
            result["particle_count"] = int(len(points))
            if len(points) == 0:
                result["reason"] = "missing_particles"
                return result

            context = (
                runtime_api.get("scene_context")
                if isinstance(runtime_api, dict)
                else None
            )
            if callable(context):
                context = context()
            if not isinstance(context, dict):
                result["reason"] = "sand_scene_context_unavailable"
                return result
            center = np.asarray(
                context.get("pile_center"),
                dtype=np.float32,
            ).reshape(-1)[:3]
            radius = np.asarray(
                context.get("diggable_radius"),
                dtype=np.float32,
            ).reshape(-1)[:2]
            if center.shape != (3,) or radius.shape != (2,):
                result["reason"] = "invalid_pile_geometry"
                return result
            radius = np.maximum(
                radius,
                np.asarray([0.05, 0.05], dtype=np.float32),
            )
            floor_z = float(context.get("sand_floor_z", center[2]))
            fill_height = max(
                0.05,
                float(context.get("sand_fill_height", 0.25)),
            )
            z_min = min(-0.05, floor_z - 0.10)
            z_expected_max = floor_z + fill_height + 0.75
            dx = (points[:, 0] - center[0]) / float(radius[0])
            dy = (points[:, 1] - center[1]) / float(radius[1])
            pile_mask = (
                (dx * dx + dy * dy <= 1.0)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_expected_max)
            )
            source_count = int(np.count_nonzero(pile_mask))
            if source_count <= 0:
                result["reason"] = "initial_pile_mask_empty"
                return result
            result.update(
                {
                    "available": True,
                    "mask": np.asarray(pile_mask, dtype=bool),
                    "source_count": source_count,
                    "reason": "ok",
                }
            )
            return result
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}:{exc}"
            return result

    def estimate_bucket_load_particles(source_tracking_required=False):
        start = time.perf_counter()
        result = {
            "count": 0,
            "source": (
                bucket_volume_topology["source"]
                if bucket_volume_topology["available"]
                else "closed_proxy_profile"
            ),
            "authored_volume_available": bool(bucket_volume_topology["available"]),
            "authored_volume_reason": str(bucket_volume_topology["reason"]),
            "candidate_count": 0,
            "particle_count": 0,
            "bucket_count": 0,
            "bucket_from_pile_count": 0,
            "bucket_from_initial_count": 0,
            "source_tracking": (
                "initial_mask"
                if source_tracking_required
                else "all_particles_legacy"
            ),
            "source_tracking_valid": not bool(source_tracking_required),
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

            if bucket_volume_topology["available"]:
                local_min = np.asarray(
                    bucket_volume_topology["local_min"],
                    dtype=np.float32,
                )
                local_max = np.asarray(
                    bucket_volume_topology["local_max"],
                    dtype=np.float32,
                )
            else:
                local_min = np.asarray(
                    [
                        bucket_load_profile_xz[:, 0].min(),
                        bucket_load_y_min,
                        bucket_load_profile_xz[:, 1].min(),
                    ],
                    dtype=np.float32,
                )
                local_max = np.asarray(
                    [
                        bucket_load_profile_xz[:, 0].max(),
                        bucket_load_y_max,
                        bucket_load_profile_xz[:, 1].max(),
                    ],
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
            candidate_indices = np.nonzero(candidate_mask)[0]
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
            if bucket_volume_topology["available"]:
                inside = points_in_closed_bucket_mesh(local)
            else:
                inside_y = (
                    (local[:, 1] >= bucket_load_y_min)
                    & (local[:, 1] <= bucket_load_y_max)
                )
                inside_xz = points_in_polygon_2d(
                    local[:, [0, 2]],
                    bucket_load_profile_xz,
                )
                inside = inside_y & inside_xz
            bucket_count = int(np.count_nonzero(inside))
            result["bucket_count"] = bucket_count
            if source_tracking_required:
                initial_mask = bucket_source_tracker.get("mask")
                if (
                    not bool(bucket_source_tracker.get("available", False))
                    or not isinstance(initial_mask, np.ndarray)
                    or len(initial_mask) != len(points)
                ):
                    result["source_tracking"] = "initial_mask_unavailable"
                    result["source_tracking_valid"] = False
                    result["source_tracking_reason"] = str(
                        bucket_source_tracker.get(
                            "reason",
                            "particle_count_changed",
                        )
                    )
                    return result
                inside_candidate_indices = candidate_indices[
                    np.nonzero(inside)[0]
                ]
                raw_from_pile = int(
                    np.count_nonzero(initial_mask[inside_candidate_indices])
                )
                source_result = (
                    vla_observation_contract.source_tracked_bucket_load(
                        bucket_count=bucket_count,
                        bucket_from_initial_count=raw_from_pile,
                    )
                )
                from_pile = int(source_result["count"])
                result.update(
                    {
                        "count": int(from_pile),
                        "bucket_from_pile_count": int(from_pile),
                        "bucket_from_initial_count": int(raw_from_pile),
                        "source_tracking": str(
                            source_result["source_tracking"]
                        ),
                        "source_ratio": float(
                            source_result["source_ratio"]
                        ),
                        "source_tracking_valid": True,
                    }
                )
            else:
                result.update(
                    {
                        "count": bucket_count,
                        "bucket_from_pile_count": bucket_count,
                        "bucket_from_initial_count": bucket_count,
                    }
                )
            return result
        except Exception as exc:
            result["source"] = f"error:{type(exc).__name__}:{exc}"
            return result
        finally:
            result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0

    def quaternion_yaw_wxyz(orientation):
        qw = float(orientation[0])
        qx = float(orientation[1])
        qy = float(orientation[2])
        qz = float(orientation[3])
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def read_robot_base_pose():
        position, orientation = robot.get_world_pose()
        return (
            float(position[0]),
            float(position[1]),
            quaternion_yaw_wxyz(orientation),
        )

    def read_truck_yaw_rad():
        truck_prim = stage.GetPrimAtPath(TRUCK_ROOT_PATH)
        if not truck_prim.IsValid():
            raise RuntimeError(
                f"truck prim {TRUCK_ROOT_PATH} is unavailable"
            )
        return math.radians(prim_local_yaw_z_deg(truck_prim))

    def resolve_state27_dig_target():
        center_x = float(
            getattr(sand_module, "SAND_CENTER_X", SAND_AUTHORED_CENTER[0])
        )
        center_y = float(
            getattr(sand_module, "SAND_CENTER_Y", SAND_AUTHORED_CENTER[1])
        )
        target_z = 0.35
        source = "configured_sand_center"
        positions_fn = (
            sand_api.get("particle_positions_fn")
            if isinstance(sand_api, dict)
            else None
        )
        if callable(positions_fn):
            try:
                points = np.asarray(
                    positions_fn(),
                    dtype=np.float32,
                ).reshape(-1, 3)
                if len(points):
                    radius_squared = (
                        np.square(points[:, 0] - center_x)
                        + np.square(points[:, 1] - center_y)
                    )
                    local_points = points[radius_squared <= 1.0]
                    if len(local_points) < 32:
                        local_points = points
                    target_z = float(
                        np.percentile(local_points[:, 2], 65.0)
                    )
                    source = "sand_particle_percentile65"
            except Exception as exc:
                print(
                    "[STATE27] Dig target particle lookup failed:",
                    repr(exc),
                    flush=True,
                )
        return np.asarray(
            [center_x, center_y, target_z],
            dtype=np.float32,
        ), source

    def resolve_state27_unload_target():
        bed_prim = stage.GetPrimAtPath(TRUCK_BED_COLLISION_PATH)
        if not bed_prim.IsValid():
            raise RuntimeError(
                "Training dump-bed collision mesh is unavailable: "
                f"{TRUCK_BED_COLLISION_PATH}"
            )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )
        aligned_range = bbox_cache.ComputeWorldBound(
            bed_prim
        ).ComputeAlignedRange()
        if aligned_range.IsEmpty():
            raise RuntimeError(
                "Training dump-bed collision mesh has an empty world bound: "
                f"{TRUCK_BED_COLLISION_PATH}"
            )
        minimum_value = aligned_range.GetMin()
        maximum_value = aligned_range.GetMax()
        minimum = np.asarray(
            [float(minimum_value[i]) for i in range(3)],
            dtype=np.float64,
        )
        maximum = np.asarray(
            [float(maximum_value[i]) for i in range(3)],
            dtype=np.float64,
        )
        landing = np.asarray(
            fixed_scene_profile["unload_landing_xyz"],
            dtype=np.float64,
        )
        xy_inside = bool(
            np.all(landing[:2] >= minimum[:2] - 0.05)
            and np.all(landing[:2] <= maximum[:2] + 0.05)
        )
        z_above_bed = float(landing[2] - maximum[2])
        if not xy_inside or not (-0.05 <= z_above_bed <= 0.25):
            raise RuntimeError(
                "Fixed unload landing does not match the transformed dump bed: "
                f"landing={landing.tolist()}, min={minimum.tolist()}, "
                f"max={maximum.tolist()}, z_above_bed={z_above_bed:.6f}"
            )
        return (
            landing.astype(np.float32),
            "fixed_training_landing_validated_against_dump_bed",
        )

    state27_dig_target_world, state27_dig_target_source = (
        resolve_state27_dig_target()
    )
    state27_unload_target_world, state27_unload_target_source = (
        resolve_state27_unload_target()
    )
    state27_truck_yaw = read_truck_yaw_rad()
    startup_base_x, startup_base_y, startup_base_yaw = (
        read_robot_base_pose()
    )
    _, startup_q = read_canonical_joint_positions()
    startup_heading = startup_base_yaw + float(startup_q[0])
    startup_origin = (startup_base_x, startup_base_y)
    startup_dig_local = (
        vla_observation_contract.point_in_initial_heading_frame(
            state27_dig_target_world,
            startup_origin,
            startup_heading,
        )
    )
    startup_unload_local = (
        vla_observation_contract.point_in_initial_heading_frame(
            state27_unload_target_world,
            startup_origin,
            startup_heading,
        )
    )
    startup_dig_radius = math.hypot(
        float(startup_dig_local[0]),
        float(startup_dig_local[1]),
    )
    startup_unload_radius = math.hypot(
        float(startup_unload_local[0]),
        float(startup_unload_local[1]),
    )
    dig_local_in_training_range = 7.313 <= startup_dig_radius <= 9.146
    unload_local_in_training_range = (
        3.953 <= startup_unload_radius <= 10.518
    )
    print(
        "[STATE27] Fixed environment features:",
        f"dig_target={state27_dig_target_world.tolist()}",
        f"dig_source={state27_dig_target_source}",
        f"unload_target={state27_unload_target_world.tolist()}",
        f"unload_source={state27_unload_target_source}",
        f"truck_yaw={state27_truck_yaw:.6f}",
        f"initial_origin={list(startup_origin)}",
        f"initial_heading={startup_heading:.6f}",
        f"dig_local_radius={startup_dig_radius:.6f}",
        f"dig_local_in_training_range={dig_local_in_training_range}",
        f"unload_local_radius={startup_unload_radius:.6f}",
        f"unload_local_in_training_range={unload_local_in_training_range}",
        flush=True,
    )
    if not dig_local_in_training_range or not unload_local_in_training_range:
        print(
            "[WARN] [SCENE CONTRACT] Fixed sand/truck world placement is "
            "training-valid, but the unchanged robot pose makes a target "
            "local radius fall outside the observed training range.",
            flush=True,
        )

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
        "observation_state_schema": OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
        "observation_state_names": list(STATE_NAMES_27D),
        "observation_state_dim": len(STATE_NAMES_27D),
        "sand_hidden_visuals": list(hidden_sand_visuals),
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
    active_observation_context = {}
    active_phase_estimator = None
    deployment_previous_q = None
    deployment_previous_load = None
    deployment_elapsed_seconds = 0.0
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
                    active_observation_context = dict(
                        active_contract.get("observation_context") or {}
                    )
                    active_phase_estimator = None
                    deployment_previous_q = None
                    deployment_previous_load = None
                    deployment_elapsed_seconds = 0.0
                    if active_contract["observation_schema"] in (
                        OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
                        OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
                    ):
                        base_x, base_y, base_yaw = read_robot_base_pose()
                        _, initial_q = read_canonical_joint_positions()
                        active_observation_context.setdefault(
                            "initial_origin_xy",
                            [base_x, base_y],
                        )
                        active_observation_context.setdefault(
                            "initial_heading_rad",
                            float(base_yaw) + float(initial_q[0]),
                        )
                        active_observation_context.setdefault(
                            "truck_yaw_rad",
                            read_truck_yaw_rad(),
                        )
                    if (
                        active_contract["observation_schema"]
                        == OBSERVATION_SCHEMA_28D_PLUS_EFFORT
                    ):
                        if not bucket_volume_topology["available"]:
                            raise ValueError(
                                "28D deployment requires the authored bucket_cut "
                                f"closed mesh: {bucket_volume_topology['reason']}"
                            )
                        bucket_source_tracker.clear()
                        bucket_source_tracker.update(
                            capture_initial_pile_source_mask()
                        )
                        if not bool(bucket_source_tracker["available"]):
                            raise ValueError(
                                "28D deployment cannot reproduce the dataset's "
                                "initial-pile bucket-load semantics: "
                                f"{bucket_source_tracker['reason']}"
                            )
                        active_phase_estimator = (
                            vla_observation_contract.DeploymentPhaseEstimator()
                        )
                    reply = {
                        "type": "handshake_ack",
                        "ok": True,
                        "protocol_version": PROTOCOL_VERSION,
                        "physics_hz": PHYSICS_HZ,
                        "training_fps": active_contract["training_fps"],
                        "observation_schema": active_contract["observation_schema"],
                        "phase_mode": str(
                            active_observation_context.get("phase_mode") or ""
                        ),
                        "state_names": list(active_contract["state_names"]),
                        "effort_names": list(active_contract.get("effort_names") or []),
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
                    active_phase_estimator = None
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
            context_update = cmd.get("observation_context")
            if context_update is not None:
                if not isinstance(context_update, dict):
                    response_queue.put({
                        "type": "error",
                        "error": "observation_context must be a JSON object.",
                    })
                    continue
                immutable_change = None
                for key, value in context_update.items():
                    if (
                        key in ("phase_name", "phase_index", "phase")
                        or value is None
                    ):
                        continue
                    if (
                        key in active_observation_context
                        and active_observation_context[key] != value
                    ):
                        immutable_change = key
                        break
                if immutable_change is not None:
                    response_queue.put({
                        "type": "error",
                        "error": (
                            "Episode observation context is immutable after handshake; "
                            f"attempted change={immutable_change}"
                        ),
                    })
                    continue
                active_observation_context.update(
                    {
                        str(key): value
                        for key, value in context_update.items()
                        if value is not None
                    }
                )

            bridge_step_start = time.perf_counter()
            bridge_step_ticks = tick_scheduler.next_ticks()
            bridge_step_seconds = float(bridge_step_ticks) * PHYSICS_DT
            deployment_elapsed_seconds += bridge_step_seconds

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
                    # simulation_app.update()
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

            # Compute the 14D base state. The negotiated deployment contract
            # extends this to either the legacy 18D state (with effort) or the
            # 28D state plus a separate 4D effort vector.
            try:
                base_x, base_y, base_yaw = read_robot_base_pose()
            except Exception:
                base_x, base_y, base_yaw = 0.0, 0.0, 0.0

            joint_positions = np.asarray(q, dtype=np.float32).reshape(-1)[:4]
            use_dataset_bucket_source_tracking = bool(
                active_contract["observation_schema"]
                == OBSERVATION_SCHEMA_28D_PLUS_EFFORT
            )
            bucket_load_metrics = estimate_bucket_load_particles(
                source_tracking_required=use_dataset_bucket_source_tracking,
            )

            tip_xyz = [0.0, 0.0, 0.0]
            load_xyz = [0.0, 0.0, 0.0]
            try:
                bucket_prim = stage.GetPrimAtPath(
                    "/World/URDF_real3/bucket_link"
                )
                if bucket_prim.IsValid():
                    world_xf = UsdGeom.Xformable(
                        bucket_prim
                    ).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    tip_world = world_xf.Transform(
                        Gf.Vec3d(0.75, 0.0, -0.18)
                    )
                    load_world = world_xf.Transform(
                        Gf.Vec3d(0.35, 0.0, 0.08)
                    )
                    tip_xyz = [
                        float(tip_world[0]),
                        float(tip_world[1]),
                        float(tip_world[2]),
                    ]
                    load_xyz = [
                        float(load_world[0]),
                        float(load_world[1]),
                        float(load_world[2]),
                    ]
            except Exception as exc:
                print(
                    "[WARN] Failed to compute bucket points:",
                    repr(exc),
                    flush=True,
                )

            base_state_14d = [
                base_x,
                base_y,
                base_yaw,
                float(joint_positions[0]),
                float(joint_positions[1]),
                float(joint_positions[2]),
                float(joint_positions[3]),
                float(bucket_load_metrics["count"]),
                float(tip_xyz[0]),
                float(tip_xyz[1]),
                float(tip_xyz[2]),
                float(load_xyz[0]),
                float(load_xyz[1]),
                float(load_xyz[2]),
            ]
            observation_effort = _read_bridge_measured_effort(
                robot,
                canonical_joint_indices,
            ).tolist()

            if deployment_previous_q is None:
                deployment_joint_velocity = [0.0, 0.0, 0.0, 0.0]
            else:
                deployment_joint_velocity = (
                    vla_observation_contract.joint_velocity_from_samples(
                        joint_positions,
                        deployment_previous_q,
                        bridge_step_seconds,
                    )
                )
            deployment_previous_q = joint_positions.copy()

            current_bucket_load = float(bucket_load_metrics["count"])
            if deployment_previous_load is None:
                deployment_bucket_load_rate = 0.0
            else:
                deployment_bucket_load_rate = (
                    current_bucket_load - float(deployment_previous_load)
                ) / max(1.0e-6, bridge_step_seconds)
            deployment_previous_load = current_bucket_load

            if (
                active_contract["observation_schema"]
                == OBSERVATION_SCHEMA_27D_PLUS_EFFORT
            ):
                joint_velocities_state = np.asarray(
                    qd,
                    dtype=np.float32,
                ).reshape(-1)
                if joint_velocities_state.size < 4:
                    raise RuntimeError(
                        "State27 requires four joint velocities, got "
                        f"{joint_velocities_state.shape}"
                    )
                initial_origin_xy = active_observation_context.get(
                    "initial_origin_xy",
                    (base_x, base_y),
                )
                initial_heading_rad = float(
                    active_observation_context.get(
                        "initial_heading_rad",
                        base_yaw,
                    )
                )
                dig_target_local = (
                    vla_observation_contract.point_in_initial_heading_frame(
                        state27_dig_target_world,
                        initial_origin_xy,
                        initial_heading_rad,
                    )
                )
                unload_target_local = (
                    vla_observation_contract.point_in_initial_heading_frame(
                        state27_unload_target_world,
                        initial_origin_xy,
                        initial_heading_rad,
                    )
                )
                relative_truck_yaw = (
                    float(state27_truck_yaw) - initial_heading_rad
                )
                observation_state = (
                    list(base_state_14d)
                    + joint_velocities_state[:4].astype(float).tolist()
                    + list(dig_target_local)
                    + list(unload_target_local)
                    + [
                        math.sin(relative_truck_yaw),
                        math.cos(relative_truck_yaw),
                        float(deployment_bucket_load_rate),
                    ]
                )
                if len(observation_state) != len(STATE_NAMES_27D):
                    raise RuntimeError(
                        "State27 construction produced "
                        f"{len(observation_state)} values"
                    )
                if not np.all(
                    np.isfinite(
                        np.asarray(observation_state, dtype=np.float32)
                    )
                ):
                    raise RuntimeError(
                        f"State27 contains NaN/Inf: {observation_state}"
                    )
            else:
                observation_state = list(base_state_14d)

            task_text = "Dig soil from the marked area and dump it into the target container."
            try:
                sand_x = float(getattr(sand_module, "SAND_CENTER_X", 0.0))
                sand_y = float(getattr(sand_module, "SAND_CENTER_Y", 6.7))
                sand_xy = (sand_x, sand_y)
                truck_prim = stage.GetPrimAtPath(TRUCK_ROOT_PATH)
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

            observation_state_28d = None
            observation_32d_ready = False
            observation_32d_error = ""
            observation_phase_report = {}
            observation_phase_source = ""
            if (
                active_contract["observation_schema"]
                == OBSERVATION_SCHEMA_28D_PLUS_EFFORT
            ):
                try:
                    if not bool(
                        bucket_load_metrics.get(
                            "source_tracking_valid",
                            False,
                        )
                    ):
                        raise vla_observation_contract.ObservationContractError(
                            "bucket load initial-pile source tracking is invalid: "
                            f"{bucket_load_metrics.get('source_tracking_reason', '')}"
                        )
                    task_text = str(
                        active_observation_context.get("task_text") or ""
                    ).strip()
                    if not task_text:
                        raise vla_observation_contract.ObservationContractError(
                            "observation_context.task_text is missing"
                        )
                    dig_target_xyz = active_observation_context.get(
                        "dig_target_xyz"
                    )
                    unload_landing_xyz = active_observation_context.get(
                        "unload_landing_xyz"
                    )
                    phase_mode = str(
                        active_observation_context.get("phase_mode") or "auto"
                    ).strip().lower()
                    if phase_mode == "external":
                        phase = active_observation_context.get("phase_index")
                        if phase is None:
                            phase = active_observation_context.get("phase_name")
                        if phase is None:
                            phase = active_observation_context.get("phase")
                        observation_phase_source = "external_supervisor"
                        observation_phase_report = {
                            "phase_index": int(
                                vla_observation_contract.canonical_phase_index(
                                    phase
                                )
                            ),
                            "phase_name": (
                                vla_observation_contract.CANONICAL_PHASE_NAMES[
                                    vla_observation_contract.canonical_phase_index(
                                        phase
                                    )
                                ]
                            ),
                            "phase_source": observation_phase_source,
                        }
                    elif phase_mode == "auto":
                        if active_phase_estimator is None:
                            raise vla_observation_contract.ObservationContractError(
                                "simulator phase estimator is unavailable"
                            )
                        observation_phase_report = (
                            active_phase_estimator.update(
                                dt=bridge_step_seconds,
                                joint_positions_4d=joint_positions,
                                bucket_tip_world_xyz=tip_xyz,
                                bucket_load_world_xyz=load_xyz,
                                bucket_load_estimate=current_bucket_load,
                                bucket_load_rate=deployment_bucket_load_rate,
                                dig_target_world_xyz=dig_target_xyz,
                                unload_landing_world_xyz=unload_landing_xyz,
                            )
                        )
                        phase = observation_phase_report["phase_index"]
                        observation_phase_source = str(
                            observation_phase_report["phase_source"]
                        )
                    else:
                        raise vla_observation_contract.ObservationContractError(
                            f"unsupported phase_mode: {phase_mode!r}"
                        )
                    observation_state_28d = (
                        vla_observation_contract.build_state_28d(
                            base_state_14d=base_state_14d,
                            joint_velocity_4d=deployment_joint_velocity,
                            phase=phase,
                            dig_target_world_xyz=dig_target_xyz,
                            unload_landing_world_xyz=unload_landing_xyz,
                            initial_origin_xy=active_observation_context.get(
                                "initial_origin_xy"
                            ),
                            initial_heading_rad=active_observation_context.get(
                                "initial_heading_rad"
                            ),
                            truck_yaw_rad=active_observation_context.get(
                                "truck_yaw_rad"
                            ),
                            bucket_load_rate=deployment_bucket_load_rate,
                        )
                    )
                    observation_32d_ready = True
                except Exception as exc:
                    observation_32d_error = f"{type(exc).__name__}: {exc}"

            reply_observation_context = dict(active_observation_context)
            if observation_phase_report:
                reply_observation_context.update(
                    {
                        "resolved_phase_index": int(
                            observation_phase_report["phase_index"]
                        ),
                        "resolved_phase_name": str(
                            observation_phase_report["phase_name"]
                        ),
                        "resolved_phase_source": str(
                            observation_phase_report["phase_source"]
                        ),
                    }
                )
            reply = {
                "joint_positions": q.tolist(),
                "joint_velocities": qd.tolist(),
                "raw_joint_positions": q_raw.tolist(),
                "raw_joint_velocities": qd_raw.tolist(),
                "observation_state": observation_state,
                "observation_state_schema": active_contract[
                    "observation_schema"
                ],
                "observation_state_names": list(
                    active_contract["state_names"]
                ),
                "observation_state_dim": len(observation_state),
                "observation_effort": observation_effort,
                "observation_state_28d": observation_state_28d,
                "observation_32d_ready": observation_32d_ready,
                "observation_32d_error": observation_32d_error,
                "observation_contract": (
                    vla_observation_contract.schema_payload()
                    if active_contract["observation_schema"]
                    == OBSERVATION_SCHEMA_28D_PLUS_EFFORT
                    else {
                        "schema_version": active_contract[
                            "observation_schema"
                        ],
                        "observation.state": {
                            "shape": [
                                len(active_contract["state_names"])
                            ],
                            "names": list(active_contract["state_names"]),
                        },
                        "observation.effort": {
                            "shape": [
                                len(active_contract["effort_names"])
                            ],
                            "names": list(active_contract["effort_names"]),
                        },
                    }
                ),
                "observation_context": reply_observation_context,
                "phase_report": dict(observation_phase_report),
                "phase_source": str(observation_phase_source),
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

    try:
        main(args)
    finally:
        pending_pose_restore = _ACTIVE_SAND_POSE_RESTORE
        _ACTIVE_SAND_POSE_RESTORE = None
        if callable(pending_pose_restore):
            try:
                pending_pose_restore()
            except Exception as exc:
                print(
                    "[ERROR] Emergency excavator pose restoration failed:",
                    repr(exc),
                    flush=True,
                )
