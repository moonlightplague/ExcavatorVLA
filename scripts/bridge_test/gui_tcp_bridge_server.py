# 02_gui_tcp_bridge_server.py
#
# Paste this into Isaac Sim: Window > Script Editor, then Run.
# It starts an asyncio TCP server inside the running Isaac Sim GUI.
#
# External Python sends JSON:
#   {"ticks": 4}
#   {"ticks": 4, "joint_positions": [0.0, 0.3, -0.2, 0.4]}
#   {"ticks": 4, "joint_velocities": [0.0, 0.1, 0.0, 0.0]}
#   {"ticks": 4, "joint_efforts": [0.0, 1000.0, 0.0, 0.0]}
#
# Server returns JSON:
#   joint_positions, joint_velocities, rgb_shape, rgb_zlib_b64

import asyncio
import base64
import json
import struct
import zlib

import numpy as np
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera


ROBOT_PRIM_PATH = "/World/URDF_real3"
CAMERA_PARENT_PATH = "/World/URDF_real3/swing_link"
CAMERA_PRIM_PATH = CAMERA_PARENT_PATH + "/front_camera"

HOST = "0.0.0.0"
PORT = 5555


_robot = None
_camera = None
_world = None
_isaac_bridge_server = None


def ensure_camera_prim(stage, camera_path):
    prim = stage.GetPrimAtPath(camera_path)
    if prim.IsValid():
        return

    parent_path = str(camera_path).rsplit("/", 1)[0]
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        raise RuntimeError(f"Camera parent does not exist: {parent_path}")

    cam = UsdGeom.Camera.Define(stage, camera_path)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)

    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, 1.2))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))


async def read_json(reader):
    header = await reader.readexactly(4)
    n = struct.unpack("!I", header)[0]
    data = await reader.readexactly(n)
    return json.loads(data.decode("utf-8"))


async def write_json(writer, obj):
    data = json.dumps(obj).encode("utf-8")
    writer.write(struct.pack("!I", len(data)))
    writer.write(data)
    await writer.drain()


def make_action_from_command(cmd):
    has_pos = "joint_positions" in cmd and cmd["joint_positions"] is not None
    has_vel = "joint_velocities" in cmd and cmd["joint_velocities"] is not None
    has_eff = "joint_efforts" in cmd and cmd["joint_efforts"] is not None

    # Do not mix methods for the same command.
    if sum([has_pos, has_vel, has_eff]) > 1:
        raise ValueError("Send only one of joint_positions, joint_velocities, joint_efforts per command.")

    if has_pos:
        return ArticulationAction(
            joint_positions=np.asarray(cmd["joint_positions"], dtype=np.float32)
        )
    if has_vel:
        return ArticulationAction(
            joint_velocities=np.asarray(cmd["joint_velocities"], dtype=np.float32)
        )
    if has_eff:
        return ArticulationAction(
            joint_efforts=np.asarray(cmd["joint_efforts"], dtype=np.float32)
        )

    return None


async def handle_client(reader, writer):
    global _robot, _camera

    peer = writer.get_extra_info("peername")
    print("[bridge] client connected:", peer)

    try:
        while True:
            cmd = await read_json(reader)

            action = make_action_from_command(cmd)
            if action is not None:
                _robot.apply_action(action)

            ticks = int(cmd.get("ticks", 4))
            ticks = max(1, min(ticks, 120))

            for _ in range(ticks):
                await omni.kit.app.get_app().next_update_async()

            q = np.asarray(_robot.get_joint_positions(), dtype=np.float32)
            qd = np.asarray(_robot.get_joint_velocities(), dtype=np.float32)

            rgb = _camera.get_rgb()
            if rgb is None:
                rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            rgb = np.asarray(rgb)

            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]

            rgb_compressed = zlib.compress(rgb.tobytes(), level=1)

            reply = {
                "joint_positions": q.tolist(),
                "joint_velocities": qd.tolist(),
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": str(rgb.dtype),
                "rgb_zlib_b64": base64.b64encode(rgb_compressed).decode("ascii"),
            }

            await write_json(writer, reply)

    except asyncio.IncompleteReadError:
        print("[bridge] client disconnected:", peer)
    except Exception as exc:
        print("[bridge] error:", repr(exc))
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    global _robot, _camera, _world, _isaac_bridge_server

    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        print("[ERROR] Robot prim path does not exist:", ROBOT_PRIM_PATH)
        print("Change ROBOT_PRIM_PATH at the top of this script.")
        return

    ensure_camera_prim(stage, CAMERA_PRIM_PATH)

    _world = World.instance()
    if _world is None:
        _world = World()

    await _world.initialize_simulation_context_async()

    _robot = SingleArticulation(
        prim_path=ROBOT_PRIM_PATH,
        name="excavator_bridge_robot",
    )

    _camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        resolution=(640, 480),
        frequency=30,
    )

    await _world.reset_async()
    _robot.initialize()
    _camera.initialize()
    await _world.play_async()

    for _ in range(30):
        await omni.kit.app.get_app().next_update_async()

    print("[bridge] robot q:", _robot.get_joint_positions())
    print("[bridge] camera:", CAMERA_PRIM_PATH)

    # If script is re-run, close old server first.
    if _isaac_bridge_server is not None:
        _isaac_bridge_server.close()
        await _isaac_bridge_server.wait_closed()

    _isaac_bridge_server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"[bridge] listening on {HOST}:{PORT}")
    print("[bridge] Now run 03_external_client_policy.py from normal Python / WSL.")


asyncio.ensure_future(main())
