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
import math
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
    make_articulation_action,
)


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


def _direction_label_from_xy(point_xy, origin_xy, robot_yaw_rad):
    if point_xy is None or origin_xy is None or len(point_xy) < 2 or len(origin_xy) < 2:
        return "nearby"
    dx = float(point_xy[0]) - float(origin_xy[0])
    dy = float(point_xy[1]) - float(origin_xy[1])
    if abs(dx) + abs(dy) < 1.0e-6:
        return "nearby"
    forward_angle = (float(robot_yaw_rad or 0.0) + math.pi * 0.5)
    angle = math.atan2(dy, dx) - forward_angle
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    labels = [
        "front",
        "front-left",
        "left",
        "rear-left",
        "rear",
        "rear-right",
        "right",
        "front-right",
    ]
    index = int(math.floor(((math.degrees(angle) + 22.5) % 360.0) / 45.0))
    return labels[index % len(labels)]


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
        if result["error"]:
            print("[sand-site-bridge] viewport capture failed:", result["error"], flush=True)
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return result["rgb"]


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
    if not camera_paths:
        raise RuntimeError("No dataset camera prims found. Run main.py and wait until the runtime is ready.")

    print("[sand-site-bridge] cameras:", camera_paths, flush=True)

    try:
        while True:
            cmd = await async_read_json(reader)

            action = make_articulation_action(cmd, ArticulationAction, np)
            if action is not None:
                _robot().apply_action(action)

            ticks = max(1, min(int(cmd.get("ticks", 4) or 4), 120))
            await _next_frames(ticks)

            robot = _robot()
            q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            qd = np.asarray(robot.get_joint_velocities(), dtype=np.float32)

            rt = _runtime_module()
            obs_state = None
            task_text = ""
            try:
                obs_fn = getattr(rt, "dataset_observation_state", None)
                if callable(obs_fn):
                    obs_state = obs_fn(q_real=q)
                if obs_state is not None and len(obs_state) >= 3:
                    robot_xy = [float(obs_state[0]), float(obs_state[1])]
                    robot_yaw = float(obs_state[2])
                    sand_module = sys.modules.get("excavator_app.sand_site_runtime")
                    if sand_module is not None:
                        sand_xy = [float(sand_module.SAND_CENTER_X), float(sand_module.SAND_CENTER_Y)]
                    else:
                        sand_xy = None
                    try:
                        unload = rt.unload_bin_dump_point()
                        unload_xy = [float(unload[0]), float(unload[1])]
                    except Exception:
                        unload_xy = None
                    sand_dir = _direction_label_from_xy(sand_xy, robot_xy, robot_yaw)
                    unload_dir = _direction_label_from_xy(unload_xy, robot_xy, robot_yaw)
                    sand_str = f"({sand_xy[0]:.2f}, {sand_xy[1]:.2f})" if sand_xy else "the marked area"
                    unload_str = f"({unload_xy[0]:.2f}, {unload_xy[1]:.2f})" if unload_xy else "the target container"
                    task_text = (
                        f"Dig soil from the sand pile near {sand_str}, {sand_dir} of the excavator, "
                        f"and dump it into the truck bed near {unload_str}, {unload_dir} of the excavator."
                    )
            except Exception as exc:
                print("[sand-site-bridge] observation_state/task_text compute failed:", repr(exc), flush=True)

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
                "primary_camera": primary_name,
                "cameras": cameras,
            }
            if obs_state is not None:
                reply["observation_state"] = obs_state
            if task_text:
                reply["task_text"] = task_text
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
