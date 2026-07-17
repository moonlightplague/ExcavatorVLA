"""
TCP bridge for the full sand-site runtime.

Run this inside Isaac Sim after main.py has loaded the complete scene:

    import runpy
    runpy.run_path("/isaac-sim/ExcavatorVLA/main.py", run_name="__main__")
    runpy.run_path("/isaac-sim/ExcavatorVLA/scripts/bridge_test/sand_site_tcp_bridge_server.py", run_name="__main__")

This script intentionally does not create a World, reload the USD stage, add a
truck, add an unload bin, or reset the simulation. It attaches to the already
running excavator_app.excavator_runtime instance and exposes observations/actions
for SmolVLA clients.
"""

import asyncio
import ctypes
import os
import sys

import numpy as np
import omni.kit.app
import omni.usd

from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _candidate in [
    os.environ.get("EXCAVATOR_PROJECT_ROOT", ""),
    PROJECT_ROOT,
    "/isaac-sim/ExcavatorVLA",
    os.getcwd(),
]:
    if _candidate and os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from excavator_common.bridge_protocol import (  # noqa: E402
    async_read_json,
    async_write_json,
    encode_rgb_payload,
)
from excavator_common import vla_observation_contract  # noqa: E402


HOST = os.environ.get("EXCAVATOR_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("EXCAVATOR_BRIDGE_PORT", "5555") or 5555)
CAPTURE_WAIT_FRAMES = int(os.environ.get("EXCAVATOR_BRIDGE_CAPTURE_WAIT_FRAMES", "8") or 8)
CAPTURE_SETTLE_FRAMES = int(os.environ.get("EXCAVATOR_BRIDGE_CAPTURE_SETTLE_FRAMES", "2") or 2)

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]

_server = None
_capture_helpers = []


def _runtime_module():
    module = sys.modules.get("excavator_app.excavator_runtime")
    if module is None:
        raise RuntimeError("excavator_app.excavator_runtime is not loaded. Run main.py first.")
    return module


def _robot():
    rt = _runtime_module()
    robot = getattr(rt, "ROBOT", None)
    if robot is None:
        raise RuntimeError("excavator runtime ROBOT is not ready yet.")
    return robot


def _canonical_articulation_action(cmd):
    fields = [
        ("joint_positions", "joint_positions"),
        ("joint_velocities", "joint_velocities"),
        ("joint_efforts", "joint_efforts"),
    ]
    provided = [(key, arg_name) for key, arg_name in fields if cmd.get(key) is not None]
    if len(provided) > 1:
        raise ValueError(
            "Send only one of joint_positions, joint_velocities, joint_efforts per command."
        )
    if not provided:
        return None

    rt = _runtime_module()
    indices = np.asarray(getattr(rt, "JOINT_INDICES", None), dtype=np.int32).reshape(-1)
    if len(indices) != 4:
        raise RuntimeError(f"canonical joint index mapping is unavailable: {indices.tolist()}")
    key, arg_name = provided[0]
    values = np.asarray(cmd[key], dtype=np.float32).reshape(-1)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain four finite canonical values")
    return ArticulationAction(
        joint_indices=indices,
        **{arg_name: values},
    )


def _camera_paths():
    rt = _runtime_module()
    stage = omni.usd.get_context().get_stage()
    paths = {}

    specs_fn = getattr(rt, "dataset_camera_specs", None)
    if callable(specs_fn):
        for spec in specs_fn():
            name = str(spec.get("name", ""))
            path = str(spec.get("path", ""))
            if name and path and stage.GetPrimAtPath(path).IsValid():
                paths[name] = path

    if paths:
        return paths

    fallback = {
        "0": "/World/URDF_real3/arm_link/Camera_0",
        "1": "/World/URDF_real3/swing_link/Camera_1",
        "2": "/World/URDF_real3/swing_link/Camera_2",
    }
    return {name: path for name, path in fallback.items() if stage.GetPrimAtPath(path).IsValid()}


def _capsule_to_rgb(capsule, buffer_size, width, height):
    name = PyCapsule_GetName(capsule)
    ptr = PyCapsule_GetPointer(capsule, name)
    if not ptr:
        raise RuntimeError("capture_viewport_to_buffer returned an empty buffer pointer")

    array_type = ctypes.c_uint8 * int(buffer_size)
    arr = np.ctypeslib.as_array(array_type.from_address(ptr))
    channels = int(buffer_size) // (int(width) * int(height))
    rgba = arr.reshape((int(height), int(width), channels)).copy()
    return rgba[:, :, :3].copy()


async def _next_frames(count):
    app = omni.kit.app.get_app()
    for _ in range(max(0, int(count))):
        await app.next_update_async()


async def _capture_viewport_rgb(viewport):
    result = {"done": False, "rgb": None, "error": None}
    helper_holder = {"done": False, "helper": None}

    def _on_capture(capsule, buffer_size, width, height, _fmt):
        try:
            result["rgb"] = _capsule_to_rgb(capsule, buffer_size, width, height)
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            result["done"] = True
            helper_holder["done"] = True

    helper_holder["helper"] = capture_viewport_to_buffer(viewport, _on_capture)
    _capture_helpers.append(helper_holder)

    for _ in range(max(1, CAPTURE_WAIT_FRAMES)):
        await _next_frames(1)
        if result["done"]:
            break

    _capture_helpers[:] = [holder for holder in _capture_helpers if not holder.get("done")]

    if result["rgb"] is None:
        detail = result["error"] or "capture callback did not produce an RGB frame"
        raise RuntimeError(f"viewport capture failed: {detail}")

    rgb = np.asarray(result["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise RuntimeError(f"viewport capture returned an invalid RGB shape: {rgb.shape}")
    if not np.any(rgb):
        raise RuntimeError("viewport capture returned an all-zero RGB frame")
    return rgb


async def _capture_cameras(viewport, camera_paths):
    rgbs = {}
    for name, path in camera_paths.items():
        viewport.camera_path = path
        await _next_frames(CAPTURE_SETTLE_FRAMES)
        rgbs[name] = await _capture_viewport_rgb(viewport)
    return rgbs


async def _handle_client(reader, writer):
    peer = writer.get_extra_info("peername")
    print("[sand-site-bridge] client connected:", peer, flush=True)

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport found. Open Isaac Sim with GUI rendering enabled.")

    camera_paths = _camera_paths()
    missing_cameras = sorted({"0", "1", "2"} - set(camera_paths))
    if missing_cameras:
        raise RuntimeError(
            "Dataset camera prims are incomplete; "
            f"missing={missing_cameras}, available={sorted(camera_paths)}"
        )

    print("[sand-site-bridge] cameras:", camera_paths, flush=True)

    try:
        while True:
            cmd = await async_read_json(reader)

            rt = _runtime_module()
            context = cmd.get("observation_context")
            reset_context = bool(cmd.get("reset_observation_context", False))
            if context is not None or reset_context:
                rt.deployment_observation_context_update(
                    context=context,
                    reset=reset_context,
                )

            action = _canonical_articulation_action(cmd)
            if action is not None:
                _robot().apply_action(action)

            ticks = max(1, min(int(cmd.get("ticks", 4) or 4), 120))
            await _next_frames(ticks)

            q = np.asarray(rt.get_real_joint_positions(), dtype=np.float32)
            qd = np.asarray(rt.deployment_canonical_joint_velocity(), dtype=np.float32)
            bucket_metrics = rt.bucket_load_fast_current(force=False)
            observation_state_14d = rt.dataset_observation_state(
                q_real=q,
                bucket_load_metrics=bucket_metrics,
            )
            effort_report = rt.dataset_joint_effort_observation()
            observation_effort = (
                effort_report.get("observation.effort")
                if isinstance(effort_report, dict)
                else None
            )

            rgbs = await _capture_cameras(viewport, camera_paths)
            primary_name = "1" if "1" in rgbs else next(iter(rgbs))
            primary_rgb = rgbs[primary_name]
            cameras = {
                name: encode_rgb_payload(rgb, np_module=np, camera_path=camera_paths.get(name, ""))
                for name, rgb in rgbs.items()
            }

            reply = {
                "joint_positions": q.tolist(),
                "joint_velocities": qd.tolist(),
                "observation_state": observation_state_14d,
                "observation_effort": observation_effort,
                "observation_contract": vla_observation_contract.schema_payload(),
                "primary_camera": primary_name,
                "cameras": cameras,
            }
            try:
                reply.update(rt.deployment_vla_observation_payload())
            except Exception as exc:
                reply["observation_state_28d"] = None
                reply["observation_32d_ready"] = False
                reply["observation_32d_error"] = f"{type(exc).__name__}: {exc}"
            else:
                reply["observation_32d_ready"] = True
            if not str(reply.get("task_text", "") or "").strip():
                reply["task_text"] = str(
                    (rt.STATE.get("deployment_observation_context") or {}).get("task_text")
                    or rt.STATE.get("dataset_task_text", "")
                    or ""
                )
            reply.update(encode_rgb_payload(primary_rgb, np_module=np, camera_path=camera_paths[primary_name]))
            await async_write_json(writer, reply)

    except asyncio.IncompleteReadError:
        print("[sand-site-bridge] client disconnected:", peer, flush=True)
    except Exception as exc:
        print("[sand-site-bridge] error:", repr(exc), flush=True)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    global _server

    _robot()
    if _server is not None:
        _server.close()
        await _server.wait_closed()

    _server = await asyncio.start_server(_handle_client, HOST, PORT)
    print(f"[sand-site-bridge] listening on {HOST}:{PORT}", flush=True)
    print("[sand-site-bridge] connect SmolVLA client to this port.", flush=True)


asyncio.ensure_future(main())
