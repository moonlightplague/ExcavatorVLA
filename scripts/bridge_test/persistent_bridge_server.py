#!/usr/bin/env python3
"""
Persistent-velocity TCP bridge for ExcavatorVLA.

Paste/run inside Isaac Sim after the scene and robot are ready.
Every physics step (60 Hz) the last received joint velocity is applied
as a position target.  The client only sends velocity updates and
immediately receives the freshest observation.

Usage (inside Isaac Sim Script Editor or via runpy):
    import runpy
    runpy.run_path("/isaac-sim/ExcavatorVLA/scripts/bridge_test/persistent_bridge_server.py",
                   run_name="__main__")
"""

import asyncio
import ctypes
import json
import math
import os
import struct
import sys
import time

import numpy as np
import omni.kit.app
import omni.usd

from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for candidate in [
    os.environ.get("EXCAVATOR_PROJECT_ROOT", ""),
    PROJECT_ROOT,
    "/isaac-sim/ExcavatorVLA",
]:
    if candidate and os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from excavator_common.bridge_protocol import encode_rgb_payload  # noqa: E402


HOST = os.environ.get("EXCAVATOR_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("EXCAVATOR_BRIDGE_PORT", "5555") or 5555)
PHYSICS_DT = 1.0 / 60.0
CAPTURE_WAIT_FRAMES = int(os.environ.get("EXCAVATOR_BRIDGE_CAPTURE_WAIT_FRAMES", "12") or 12)


PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]


_persistent_vel = np.zeros(4, dtype=np.float32)
_latest_obs = None
_obs_ready = asyncio.Event()
_step_start_time = 0.0


def _get_robot():
    for name in list(sys.modules.keys()):
        mod = sys.modules[name]
        robot = getattr(mod, "robot", None)
        if robot is not None:
            return robot
    from isaacsim.core.prims import SingleArticulation
    stage = omni.usd.get_context().get_stage()
    robot_path = "/World/URDF_real3"
    if stage.GetPrimAtPath(robot_path).IsValid():
        return SingleArticulation(prim_path=robot_path, name="excavator")
    raise RuntimeError("No robot found.")


def _get_stage():
    return omni.usd.get_context().get_stage()


def _get_camera_paths():
    stage = _get_stage()
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
        raise RuntimeError("empty capture buffer")
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
    result = {"done": False, "rgb": np.zeros((1, 1, 3), dtype=np.uint8)}

    def _on_capture(capsule, buffer_size, width, height, _fmt):
        try:
            result["rgb"] = _capsule_to_rgb(capsule, buffer_size, width, height)
        except Exception:
            pass
        result["done"] = True

    capture_viewport_to_buffer(viewport, _on_capture)
    for _ in range(max(1, CAPTURE_WAIT_FRAMES)):
        await _next_frames(1)
        if result["done"]:
            break
    return result["rgb"]


async def _capture_cameras(viewport, camera_paths):
    rgbs = {}
    for name, path in camera_paths.items():
        viewport.camera_path = path
        await _next_frames(2)
        rgbs[name] = await _capture_viewport_rgb(viewport)
    return rgbs


def _compute_observation_state(robot, stage):
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

    q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)[:4]

    tip_xyz = [0.0, 0.0, 0.0]
    load_xyz = [0.0, 0.0, 0.0]
    try:
        from pxr import Usd, UsdGeom, Gf
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

    return [
        base_x, base_y, base_yaw,
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        0.0,
        tip_xyz[0], tip_xyz[1], tip_xyz[2],
        load_xyz[0], load_xyz[1], load_xyz[2],
    ]


async def _sim_step(robot, viewport, camera_paths):
    """One physics tick: apply persistent velocity, step, capture observation."""
    global _latest_obs, _obs_ready, _step_start_time

    v = _persistent_vel.copy()
    q_now = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)[:4]
    q_target = q_now + v * PHYSICS_DT
    robot.apply_action(ArticulationAction(joint_positions=q_target.astype(np.float32)))

    await _next_frames(1)

    stage = _get_stage()
    obs_state = _compute_observation_state(robot, stage)
    q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    qd = np.asarray(robot.get_joint_velocities(), dtype=np.float32)

    rgbs = await _capture_cameras(viewport, camera_paths)
    cameras = {name: encode_rgb_payload(rgb, np_module=np, camera_path=camera_paths.get(name, ""))
               for name, rgb in rgbs.items()}
    primary_name = "1" if "1" in rgbs else next(iter(rgbs))
    primary_payload = encode_rgb_payload(rgbs[primary_name], np_module=np, camera_path=camera_paths[primary_name])

    _latest_obs = {
        "joint_positions": q.tolist(),
        "joint_velocities": qd.tolist(),
        "observation_state": obs_state,
        "primary_camera": primary_name,
        "cameras": cameras,
    }
    _latest_obs.update(primary_payload)
    _obs_ready.set()
    _step_start_time = time.perf_counter()


async def _handle_client(reader, writer):
    global _persistent_vel, _obs_ready, _latest_obs

    peer = writer.get_extra_info("peername")
    print(f"[pbridge] connected: {peer}", flush=True)

    async def _recv():
        header = await reader.readexactly(4)
        n = struct.unpack("!I", header)[0]
        return json.loads((await reader.readexactly(n)).decode("utf-8"))

    async def _send(obj):
        data = json.dumps(obj).encode("utf-8")
        writer.write(struct.pack("!I", len(data)))
        writer.write(data)
        await writer.drain()

    try:
        while True:
            cmd = await _recv()
            vel = np.asarray(cmd.get("joint_velocities", []), dtype=np.float32).reshape(-1)
            if vel.shape[0] >= 4:
                _persistent_vel[0] = float(vel[0])
                _persistent_vel[1] = float(vel[1])
                _persistent_vel[2] = float(vel[2])
                _persistent_vel[3] = float(vel[3])

            await _obs_ready.wait()
            obs = _latest_obs
            _obs_ready.clear()
            await _send(obs)

    except asyncio.IncompleteReadError:
        print(f"[pbridge] disconnected: {peer}", flush=True)
    except Exception as exc:
        print(f"[pbridge] error: {repr(exc)}", flush=True)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _sim_loop(robot, viewport, camera_paths):
    print("[pbridge] sim loop running", flush=True)
    while True:
        await _sim_step(robot, viewport, camera_paths)
        await asyncio.sleep(0)


async def _main():
    robot = _get_robot()
    camera_paths = _get_camera_paths()
    if not camera_paths:
        raise RuntimeError("No camera prims found.")
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport.")
    print(f"[pbridge] cameras: {camera_paths}", flush=True)

    _server = await asyncio.start_server(_handle_client, HOST, PORT)
    print(f"[pbridge] listening on {HOST}:{PORT}", flush=True)
    print("[pbridge] connect persistent client to this port.", flush=True)

    await _sim_loop(robot, viewport, camera_paths)


asyncio.ensure_future(_main())
