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
import ctypes

from excavator_common.bridge_protocol import (
    async_read_json,
    async_write_json,
    encode_rgb_payload,
    make_articulation_action,
    validate_bridge_command,
)
from excavator_common.paths import (
    default_scene_path,
    default_truck_usd_path,
    env_path,
    resolve_existing_path,
)

# Add the project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR

# Scene and robot paths
SCENE_USD_PATH = default_scene_path(PROJECT_DIR)
ROBOT_PRIM_PATH = "/World/URDF_real3"

# Use the three existing robot-mounted dataset cameras in the USD scene.
# These names intentionally match the VLA dataset keys: observation.images.0/1/2.
CAMERA_SWING_PARENT_PATH = "/World/URDF_real3/swing_link"
CAMERA_ARM_PARENT_PATH = "/World/URDF_real3/arm_link"
CAMERA_PRIM_PATHS = {
    "0": CAMERA_ARM_PARENT_PATH + "/Camera_0",
    "1": CAMERA_SWING_PARENT_PATH + "/Camera_1",
    "2": CAMERA_SWING_PARENT_PATH + "/Camera_2",
}
CAMERA_PRIM_PATH = CAMERA_PRIM_PATHS["1"]

# TCP Bridge settings
HOST = "0.0.0.0"
PORT = 5555

# Viewport capture settings
CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600
CAPTURE_WAIT_FRAMES = 5
PENDING_VIEWPORT_CAPTURE_HELPERS = []


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


def capture_rgb_from_viewport(viewport, world, simulation_app, capture_viewport_to_buffer, np_module,
                              width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
                              wait_frames=CAPTURE_WAIT_FRAMES):
    """
    Capture RGB image from the active viewport.

    This replaces:
        camera.initialize()
        camera.get_rgb()

    It avoids Isaac Sim Camera RGB annotator / syntheticdata bug.
    """
    result = {"done": False, "rgb": None}

    helper_holder = {"done": False, "helper": None}

    def on_capture(capsule, buffer_size, w, h, fmt):
        try:
            rgba = capsule_to_numpy_rgba(capsule, buffer_size, w, h, np_module)

            if rgba.ndim != 3 or rgba.shape[-1] < 3:
                raise RuntimeError(f"Unexpected viewport buffer shape: {rgba.shape}")

            rgb = rgba[:, :, :3].copy()
            result["rgb"] = rgb

        except Exception as e:
            print(f"[WARN] viewport capture failed: {repr(e)}", flush=True)

        finally:
            result["done"] = True
            helper_holder["done"] = True

    # Request one viewport capture.
    helper_holder["helper"] = capture_viewport_to_buffer(viewport, on_capture)
    PENDING_VIEWPORT_CAPTURE_HELPERS.append(helper_holder)

    # Pump frames until callback finishes.
    for _ in range(wait_frames):
        world.step(render=True)
        simulation_app.update()

        if result["done"]:
            break

    cleanup_viewport_capture_helpers(force=False)

    if result["rgb"] is None:
        print("[WARN] viewport capture returned None, using black image", flush=True)
        return np_module.zeros((height, width, 3), dtype=np_module.uint8)

    rgb = result["rgb"]

    if rgb.dtype != np_module.uint8:
        rgb = np_module.clip(rgb, 0, 255).astype(np_module.uint8)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb


def capture_rgb_from_cameras(viewport, camera_paths, world, simulation_app, capture_viewport_to_buffer, np_module,
                             width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
                             wait_frames=CAPTURE_WAIT_FRAMES,
                             settle_frames=2):
    rgb_by_camera = {}

    for camera_name, camera_path in camera_paths.items():
        viewport.camera_path = camera_path

        for _ in range(settle_frames):
            world.step(render=True)
            simulation_app.update()

        rgb_by_camera[camera_name] = capture_rgb_from_viewport(
            viewport=viewport,
            world=world,
            simulation_app=simulation_app,
            capture_viewport_to_buffer=capture_viewport_to_buffer,
            np_module=np_module,
            width=width,
            height=height,
            wait_frames=wait_frames,
        )

    return rgb_by_camera


def main(args=None):
    """Main entry point for standalone launch."""
    if args is None:
        args = argparse.Namespace(
            scene=SCENE_USD_PATH,
            truck_usd=env_path("EXCAVATOR_TRUCK_USD", default_truck_usd_path(PROJECT_DIR)),
            truck_glb=env_path("EXCAVATOR_TRUCK_GLB", ""),
            force_test_pose=False,
        )

    # Import Isaac Sim modules after Isaac Sim Python env is set up.
    from isaacsim import SimulationApp

    # Create simulation app.
    # Keep headless False because viewport capture needs an active viewport.
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1920,
        "height": 1080,
        "renderer": "RayTracedLighting",
    })

    # Import remaining Isaac Sim modules after SimulationApp is created.
    import omni.usd
    import omni.kit.app
    import omni.timeline

    from pxr import UsdGeom, Gf, UsdLux

    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction

    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer

    # Import bridge server components.
    import asyncio
    import numpy as np
    import queue

    # Import asset converter for GLB conversion.
    # This import passed your test, so we keep it.
    import omni.kit.asset_converter as asset_converter

    print("=" * 60)
    print("ExcavatorVLA Standalone Launcher")
    print("=" * 60)
    scene_usd_path = resolve_existing_path(args.scene, root=PROJECT_DIR)
    truck_usd_path = resolve_existing_path(args.truck_usd, root=PROJECT_DIR) if args.truck_usd else ""
    truck_glb_path = resolve_existing_path(args.truck_glb, root=PROJECT_DIR) if args.truck_glb else ""
    print(f"Scene USD: {scene_usd_path}")
    print(f"Robot Prim: {ROBOT_PRIM_PATH}")
    print(f"Truck USD: {truck_usd_path or '[disabled]'}")
    print(f"Truck GLB: {truck_glb_path or '[not set]'}")
    print("Camera Prims for Viewport Capture:")
    for camera_name, camera_path in CAMERA_PRIM_PATHS.items():
        print(f"  {camera_name}: {camera_path}")
    print(f"TCP Server: {HOST}:{PORT}")
    print("=" * 60)

    # Get stage.
    stage = get_current_stage()

    # Check if scene file exists.
    if not os.path.exists(scene_usd_path):
        print(f"[ERROR] Scene file not found: {scene_usd_path}")
        print("Please ensure the excavator_scene.usd file exists.")
        simulation_app.close()
        return

    # Load the USD scene.
    add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World")
    print(f"[INFO] Loaded scene from: {scene_usd_path}")

    # Check if robot prim exists.
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        print(f"[ERROR] Robot prim not found: {ROBOT_PRIM_PATH}")
        print("The scene may not contain the expected robot.")
        simulation_app.close()
        return

    print(f"[INFO] Robot prim found: {ROBOT_PRIM_PATH}")


    # Create World.
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

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

    # Load dump truck from checked-in USD by default. External GLB conversion is optional.
    truck_prim_path = "/World/DumpTruck"
    if truck_glb_path and truck_usd_path and os.path.exists(truck_glb_path) and not os.path.exists(truck_usd_path):
        print(f"[INFO] Converting GLB to USD: {truck_glb_path}")

        context = asset_converter.AssetConverterContext()
        context.ignore_materials = False
        context.export_preview_surface = True
        context.use_meter_as_world_unit = True

        converter_instance = asset_converter.get_instance()
        task = converter_instance.create_converter_task(
            truck_glb_path,
            truck_usd_path,
            None,
            context,
        )

        while not task.is_finished():
            omni.kit.app.get_app().update()

        if task.get_status() != asset_converter.Status.SUCCESS:
            print(f"[ERROR] Failed to convert GLB: {task.get_status()}")
        else:
            print(f"[INFO] GLB converted successfully to: {truck_usd_path}")

    if truck_usd_path and os.path.exists(truck_usd_path):
        if not stage.GetPrimAtPath(truck_prim_path).IsValid():
            truck_prim = stage.DefinePrim(truck_prim_path, "Xform")

            xform = UsdGeom.Xformable(truck_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(4.14439, 6.72012, 1.0))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            xform.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 100.0))

            refs = truck_prim.GetReferences()
            refs.AddReference(assetPath=truck_usd_path)

            print(f"[INFO] Loaded truck model at: {truck_prim_path}")
        else:
            print(f"[INFO] Truck already exists at: {truck_prim_path}")
    else:
        print(f"[WARN] Truck USD file not found or disabled: {truck_usd_path or '[disabled]'}")

    # Reset world and initialize robot.
    world.reset()
    robot.initialize()

    if args.force_test_pose:
        robot_initial_pos_after_reset = np.array([-9.2, 6.7, 1.243], dtype=np.float32)
        robot_initial_ori_after_reset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        try:
            robot.set_world_pose(
                position=robot_initial_pos_after_reset,
                orientation=robot_initial_ori_after_reset,
            )
            for _ in range(10):
                world.step(render=True)
                simulation_app.update()

            print("[INFO] FORCE robot world pose after reset:", robot.get_world_pose(), flush=True)
        except Exception as e:
            print("[ERROR] Failed to force robot pose after reset:", repr(e), flush=True)


    print("[INFO] World initialized")
    print("[INFO] Robot joint positions:", robot.get_joint_positions())
    try:
        print("[INFO] DOF names:", robot.dof_names, flush=True)
    except Exception as e:
        print("[WARN] Could not get robot.dof_names:", repr(e), flush=True)

    try:
        print("[INFO] Joint names:", robot.joint_names, flush=True)
    except Exception as e:
        print("[WARN] Could not get robot.joint_names:", repr(e), flush=True)

    # -----------------------------------------------------------------
    # Viewport capture setup
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

    active_capture_camera_name = next(iter(active_camera_paths))
    active_capture_camera_path = active_camera_paths[active_capture_camera_name]
    viewport.camera_path = active_capture_camera_path

    print("[INFO] Active viewport capture cameras:", flush=True)
    for camera_name, camera_path in active_camera_paths.items():
        print(f"  {camera_name}: {camera_path}", flush=True)
    print(f"[INFO] Initial viewport camera set to: {active_capture_camera_path}", flush=True)

    omni.timeline.get_timeline_interface().play()

    print("[INFO] Warming up viewport rendering...")
    for i in range(30):
        world.step(render=True)
        simulation_app.update()

    # Test viewport capture from every active camera.
    test_rgbs = capture_rgb_from_cameras(
        viewport=viewport,
        camera_paths=active_camera_paths,
        world=world,
        simulation_app=simulation_app,
        capture_viewport_to_buffer=capture_viewport_to_buffer,
        np_module=np,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
        wait_frames=CAPTURE_WAIT_FRAMES,
    )

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

    # TCP Bridge Server functions.
    command_queue = queue.Queue()
    response_queue = queue.Queue()

    def make_action_from_command(cmd):
        return make_articulation_action(cmd, ArticulationAction, np)

    async def handle_client(reader, writer):
        peer = writer.get_extra_info("peername")
        print(f"[bridge] Client connected: {peer}")

        try:
            while True:
                cmd = await async_read_json(reader)

                command_queue.put(cmd)

                while response_queue.empty():
                    await asyncio.sleep(0.001)

                reply = response_queue.get()
                await async_write_json(writer, reply)

        except asyncio.IncompleteReadError:
            print(f"[bridge] Client disconnected: {peer}")
        except Exception as exc:
            print(f"[bridge] Error: {repr(exc)}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    _server = None

    async def start_bridge_server():
        nonlocal _server
        _server = await asyncio.start_server(handle_client, HOST, PORT)
        print(f"[bridge] TCP server listening on {HOST}:{PORT}")
        print("[bridge] You can now connect with the external client or GUI client.")

        async with _server:
            await _server.serve_forever()

    asyncio.ensure_future(start_bridge_server())

    print("[INFO] Simulation running. Press Ctrl+C to exit.")

    # Run simulation loop.
    while simulation_app.is_running():
        if not command_queue.empty():
            cmd = command_queue.get()
            validate_bridge_command(cmd)

            action = make_action_from_command(cmd)
            if action is not None:
                robot.apply_action(action)

            ticks = int(cmd.get("ticks", 4))
            ticks = max(1, min(ticks, 120))

            for _ in range(ticks):
                world.step(render=True)
                simulation_app.update()

            q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            qd = np.asarray(robot.get_joint_velocities(), dtype=np.float32)

            rgb_by_camera = capture_rgb_from_cameras(
                viewport=viewport,
                camera_paths=active_camera_paths,
                world=world,
                simulation_app=simulation_app,
                capture_viewport_to_buffer=capture_viewport_to_buffer,
                np_module=np,
                width=CAPTURE_WIDTH,
                height=CAPTURE_HEIGHT,
                wait_frames=CAPTURE_WAIT_FRAMES,
            )

            encoded_cameras = {}
            for camera_name, rgb in rgb_by_camera.items():
                encoded_cameras[camera_name] = encode_rgb_payload(
                    rgb,
                    np_module=np,
                    camera_path=active_camera_paths[camera_name],
                )

            primary_camera = (
                "front" if "front" in encoded_cameras else next(iter(encoded_cameras))
            )
            primary_rgb = encoded_cameras[primary_camera]

            reply = {
                "joint_positions": q.tolist(),
                "joint_velocities": qd.tolist(),
                "primary_camera": primary_camera,
                "rgb_shape": primary_rgb["rgb_shape"],
                "rgb_dtype": primary_rgb["rgb_dtype"],
                "rgb_zlib_b64": primary_rgb["rgb_zlib_b64"],
                "cameras": encoded_cameras,
            }

            response_queue.put(reply)

        else:
            world.step(render=True)
            simulation_app.update()

    cleanup_viewport_capture_helpers(force=True)
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExcavatorVLA Standalone Launcher")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--scene", default=SCENE_USD_PATH, help="USD scene path. Defaults to the repo original scene.")
    parser.add_argument(
        "--truck-usd",
        default=env_path("EXCAVATOR_TRUCK_USD", default_truck_usd_path(PROJECT_DIR)),
        help="Dump truck USD path. Defaults to EXCAVATOR_TRUCK_USD or assets/fbx/truck/truck.usd.",
    )
    parser.add_argument(
        "--truck-glb",
        default=env_path("EXCAVATOR_TRUCK_GLB", ""),
        help="Optional dump truck GLB path to convert when --truck-usd does not exist.",
    )
    parser.add_argument(
        "--force-test-pose",
        action="store_true",
        help="Force the old debug robot pose after reset. Off by default.",
    )
    args = parser.parse_args()

    if args.headless:
        print("[WARN] --headless was passed, but this version uses viewport capture and needs headless=False.")

    main(args)
