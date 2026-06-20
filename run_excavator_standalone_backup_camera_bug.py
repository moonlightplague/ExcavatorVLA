#!/usr/bin/env python3
"""
ExcavatorVLA Standalone Launch Script
=====================================

This script launches Isaac Sim with the excavator scene and starts the TCP bridge server.

Usage:
    /isaac-sim/python.sh run_excavator_standalone.py

Or run directly:
    python run_excavator_standalone.py (after sourcing setup_python_env.sh)
"""

import argparse
import sys
import os

# Add the project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR

# Scene and robot paths
SCENE_USD_PATH = os.path.join(PROJECT_DIR, "assets/usd/excavator_scene.usd")
ROBOT_PRIM_PATH = "/World/URDF_real3"
CAMERA_PARENT_PATH = "/World/URDF_real3/swing_link"
CAMERA_PRIM_PATH = CAMERA_PARENT_PATH + "/Camera"

# TCP Bridge settings
HOST = "0.0.0.0"
PORT = 5555


def main():
    """Main entry point for standalone launch."""
    
    # Import Isaac Sim modules (must be done after Isaac Sim Python env is set up)
    from isaacsim import SimulationApp
    
    # Create simulation app
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1920,
        "height": 1080,
    })
    
    # Import remaining Isaac Sim modules after SimulationApp is created
    import omni.usd
    import omni.kit.app
    from pxr import UsdGeom, Gf, UsdLux
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    
    # Import bridge server components
    import asyncio
    import base64
    import json
    import struct
    import zlib
    import numpy as np
    import queue
    
    from isaacsim.core.utils.types import ArticulationAction
    
    # Import asset converter for GLB conversion
    import omni.kit.asset_converter as asset_converter
    
    print("=" * 60)
    print("ExcavatorVLA Standalone Launcher")
    print("=" * 60)
    print(f"Scene USD: {SCENE_USD_PATH}")
    print(f"Robot Prim: {ROBOT_PRIM_PATH}")
    print(f"TCP Server: {HOST}:{PORT}")
    print("=" * 60)
    
    # Get stage
    stage = get_current_stage()
    
    # Check if scene file exists
    if not os.path.exists(SCENE_USD_PATH):
        print(f"[ERROR] Scene file not found: {SCENE_USD_PATH}")
        print("Please ensure the excavator_scene.usd file exists.")
        simulation_app.close()
        return
    
    # Load the USD scene
    add_reference_to_stage(usd_path=SCENE_USD_PATH, prim_path="/World")
    print(f"[INFO] Loaded scene from: {SCENE_USD_PATH}")
    
    # Check if robot prim exists
    if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        print(f"[ERROR] Robot prim not found: {ROBOT_PRIM_PATH}")
        print("The scene may not contain the expected robot.")
        simulation_app.close()
        return
    
    print(f"[INFO] Robot prim found: {ROBOT_PRIM_PATH}")
    
    # Create World
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    
    # Add ground plane
    world.scene.add_default_ground_plane(
        z_position=0,
        name="ground_plane",
        prim_path="/World/GroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
    )
    print("[INFO] Added ground plane")
    
    # Add lighting
    light_path = "/World/MainLight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.CreateIntensityAttr(500.0)
        light.CreateAngleAttr(0.53)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        print(f"[INFO] Added light at: {light_path}")
    
    # Set viewport camera position to view the robot
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport:
        # Create a camera for viewing
        view_camera_path = "/World/ViewCamera"
        if not stage.GetPrimAtPath(view_camera_path).IsValid():
            view_cam = UsdGeom.Camera.Define(stage, view_camera_path)
            view_cam.CreateFocalLengthAttr(24.0)
            
            # Set camera transform
            xform = UsdGeom.Xformable(view_cam.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(8.0, 8.0, 5.0))
            # Look towards origin (rotate to point at target)
            xform.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, -45.0, 0.0))
            
            # Set this camera as the viewport camera
            viewport.camera_path = view_camera_path
            print(f"[INFO] Set viewport camera to: {view_camera_path}")
    
    # Initialize robot articulation
    robot = SingleArticulation(
        prim_path=ROBOT_PRIM_PATH,
        name="excavator",
    )
    world.scene.add(robot)
    
    # Load dump truck using GLB converter
    TRUCK_GLB_PATH = "/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.glb"
    TRUCK_USD_PATH = "/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.usd"
    
    if os.path.exists(TRUCK_GLB_PATH):
        truck_prim_path = "/World/DumpTruck"
        
        # Convert GLB to USD if USD doesn't exist
        if not os.path.exists(TRUCK_USD_PATH):
            print(f"[INFO] Converting GLB to USD: {TRUCK_GLB_PATH}")
            
            # Create converter context
            context = asset_converter.AssetConverterContext()
            context.ignore_materials = False
            context.export_preview_surface = True
            context.use_meter_as_world_unit = True
            
            # Create converter task
            converter_instance = asset_converter.get_instance()
            task = converter_instance.create_converter_task(TRUCK_GLB_PATH, TRUCK_USD_PATH, None, context)
            
            # Wait for conversion to complete
            import omni.kit.app
            while not task.is_finished():
                omni.kit.app.get_app().update()
            
            if task.get_status() != asset_converter.Status.SUCCESS:
                print(f"[ERROR] Failed to convert GLB: {task.get_status()}")
            else:
                print(f"[INFO] GLB converted successfully to: {TRUCK_USD_PATH}")
        
        # Load the converted USD file
        if os.path.exists(TRUCK_USD_PATH):
            if not stage.GetPrimAtPath(truck_prim_path).IsValid():
                # Define a new Xform prim for the truck
                truck_prim = stage.DefinePrim(truck_prim_path, "Xform")
                
                # Set transform for the truck
                xform = UsdGeom.Xformable(truck_prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(4.14439, 6.72012, 1.0))
                xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                xform.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 100.0))
                
                # Reference the converted USD file
                refs = truck_prim.GetReferences()
                refs.AddReference(assetPath=TRUCK_USD_PATH)
                
                print(f"[INFO] Loaded truck model at: {truck_prim_path}")
            else:
                print(f"[INFO] Truck already exists at: {truck_prim_path}")
        else:
            print(f"[WARN] Truck USD file not found: {TRUCK_USD_PATH}")
    else:
        print(f"[WARN] Truck GLB file not found: {TRUCK_GLB_PATH}")
    
    # Ensure camera exists
    def ensure_camera_prim(stage, camera_path):
        prim = stage.GetPrimAtPath(camera_path)
        if prim.IsValid():
            return
        
        parent_path = str(camera_path).rsplit("/", 1)[0]
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            print(f"[WARN] Camera parent does not exist: {parent_path}")
            return
        
        cam = UsdGeom.Camera.Define(stage, camera_path)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, 1.2))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))
        print(f"[INFO] Created camera at: {camera_path}")
    
    ensure_camera_prim(stage, CAMERA_PRIM_PATH)
    
    # Create camera sensor
    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        resolution=(640, 480),
        frequency=30,
    )
    
    # Reset world
    world.reset()
    robot.initialize()
    camera.initialize()
    
    print("[INFO] World initialized")
    print("[INFO] Robot joint positions:", robot.get_joint_positions())
    
    # Wait for camera to stabilize - important!
    print("[INFO] Waiting for camera to stabilize...")
    for i in range(30):
        world.step(render=True)
        rgb = camera.get_rgb()
        if rgb is not None:
            print(f"[INFO] Camera ready after {i+1} steps")
            break
    else:
        print("[WARN] Camera still not producing images after warmup")
    
    # TCP Bridge Server functions
    # Command queue for thread-safe communication
    command_queue = queue.Queue()
    response_queue = queue.Queue()
    
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
        peer = writer.get_extra_info("peername")
        print(f"[bridge] Client connected: {peer}")
        
        try:
            while True:
                cmd = await read_json(reader)
                
                # Put command in queue and wait for response
                command_queue.put(cmd)
                
                # Wait for response from main loop
                while response_queue.empty():
                    await asyncio.sleep(0.001)
                
                reply = response_queue.get()
                await write_json(writer, reply)
        
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
    
    # Global server reference
    _server = None
    
    async def start_bridge_server():
        global _server
        _server = await asyncio.start_server(handle_client, HOST, PORT)
        print(f"[bridge] TCP server listening on {HOST}:{PORT}")
        print("[bridge] You can now connect with the external client or GUI client.")
        
        async with _server:
            await _server.serve_forever()
    
    # Start TCP server using Isaac Sim's async engine
    asyncio.ensure_future(start_bridge_server())
    
    print("[INFO] Simulation running. Press Ctrl+C to exit.")
    
    # Run simulation loop
    while simulation_app.is_running():
        # Process commands from queue
        if not command_queue.empty():
            cmd = command_queue.get()
            
            action = make_action_from_command(cmd)
            if action is not None:
                robot.apply_action(action)
            
            ticks = int(cmd.get("ticks", 4))
            ticks = max(1, min(ticks, 120))
            
            for _ in range(ticks):
                world.step(render=True)
            
            q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            qd = np.asarray(robot.get_joint_velocities(), dtype=np.float32)
            
            rgb = camera.get_rgb()
            if rgb is None:
                print("[DEBUG] camera.get_rgb() is None")
                rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            else:
                rgb_arr = np.asarray(rgb)
                print(
                    "[DEBUG] rgb shape:", rgb_arr.shape,
                    "dtype:", rgb_arr.dtype,
                    "min:", rgb_arr.min(),
                    "max:", rgb_arr.max(),
                    "mean:", rgb_arr.mean()
                )
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
            
            response_queue.put(reply)
        else:
            world.step(render=True)
    
    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExcavatorVLA Standalone Launcher")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    
    if args.headless:
        print("[INFO] Running in headless mode")
    
    main()