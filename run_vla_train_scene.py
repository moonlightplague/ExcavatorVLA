#!/usr/bin/env python3
"""
Launch the full sand + truck + excavator runtime without using main.py.

Typical VLA training scene:
    /isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.0 --bridge

Headless auto collect:
    /isaac-sim/python.sh run_vla_train_scene.py --headless --auto-collect --success-count 20 --max-attempts 120
"""

import argparse
import importlib
import os
import platform
import runpy
import sys
import time

try:
    importlib.import_module("nest_asyncio").apply()
except Exception:
    pass

from excavator_common import paths


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = paths.find_project_root(start=__file__)


def add_import_roots(project_root):
    for import_root in [project_root, os.path.join(project_root, "scripts")]:
        import_root = os.path.abspath(import_root)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)


def env_set_if_value(name, value):
    if value is None:
        return
    text = str(value).strip()
    if text:
        os.environ[str(name)] = text


def parse_args():
    parser = argparse.ArgumentParser(description="Launch ExcavatorVLA full sand-site runtime for VLA training.")
    parser.add_argument("--scene", default=paths.default_scene_path(PROJECT_ROOT), help="USD scene containing the truck and excavator.")
    parser.add_argument("--sand-amount", type=float, default=None, help="Initial sand amount multiplier, clamped by runtime limits.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless. Intended for auto collect, not viewport bridge.")
    parser.add_argument("--with-ui", action="store_true", help="Show Excavator/Sand control windows. Off by default for this launcher.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument(
        "--graphics-api",
        choices=("auto", "vulkan", "d3d12"),
        default="auto",
        help="Renderer graphics API. auto uses D3D12 for Windows headless auto-collect and Isaac defaults elsewhere.",
    )
    parser.add_argument("--active-gpu", type=int, default=0, help="Renderer GPU index for Isaac SimulationApp.")
    parser.add_argument("--physics-gpu", type=int, default=0, help="Physics CUDA device index for Isaac SimulationApp.")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable Isaac multi-GPU rendering.")

    bridge_group = parser.add_mutually_exclusive_group()
    bridge_group.add_argument("--bridge", dest="bridge", action="store_true", help="Start TCP bridge for SmolVLA clients.")
    bridge_group.add_argument("--no-bridge", dest="bridge", action="store_false", help="Do not start TCP bridge.")
    parser.set_defaults(bridge=None)
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=int, default=5555)

    parser.add_argument("--auto-collect", action="store_true", help="Start auto dataset collection after runtime is ready.")
    parser.add_argument("--success-count", type=int, default=10, help="Target number of successful/trainable episodes.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Maximum episode attempts. 0 uses runtime default multiplier.")
    parser.add_argument(
        "--random-sand-amount",
        action="store_true",
        help="Allow auto collect to randomize sand amount even when --sand-amount is provided.",
    )
    parser.add_argument("--dataset-root", default="", help="Override EXCAVATOR_DATASET_ROOT.")
    parser.add_argument("--disable-export", action="store_true", help="Disable automatic LeRobot v3 export after auto collect.")
    parser.add_argument("--export-python", default="", help="Python executable used for LeRobot export subprocess.")
    parser.add_argument("--wait-runtime-seconds", type=float, default=180.0)
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument("--wait-export", dest="wait_export", action="store_true", help="Wait for LeRobot export task before closing in auto-collect mode.")
    export_group.add_argument("--no-wait-export", dest="wait_export", action="store_false", help="Close after auto collect without waiting for export.")
    parser.set_defaults(wait_export=True)
    parser.add_argument("--keep-running-after-auto-collect", action="store_true")
    return parser.parse_args()


def open_stage(simulation_app, scene_path):
    import omni.usd

    scene_path = paths.resolve_existing_path(scene_path, root=PROJECT_ROOT)
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene USD not found: {scene_path}")

    opened = omni.usd.get_context().open_stage(scene_path)
    print(f"[INFO] Opened stage: {scene_path} result={opened}", flush=True)
    for _ in range(10):
        simulation_app.update()
    return scene_path


def runtime_is_ready(rt):
    robot = getattr(rt, "ROBOT", None)
    indices = getattr(rt, "JOINT_INDICES", None)
    return robot is not None and indices is not None and bool(rt.STATE.get("robot_state_reads_enabled", False))


def wait_runtime_ready(simulation_app, rt, timeout_s):
    started = time.time()
    last_print = 0.0
    while simulation_app.is_running():
        simulation_app.update()
        if runtime_is_ready(rt):
            suppress_headless_log_noise()
            print("[INFO] Runtime ready: robot articulation and joint indices are initialized.", flush=True)
            return

        task = (rt.STATE.get("async_tasks") or {}).get("main_loop")
        if task is not None and hasattr(task, "done") and task.done():
            raise RuntimeError("Runtime main_loop finished before robot became ready.")

        now = time.time()
        if now - last_print > 5.0:
            last_print = now
            print("[INFO] Waiting for runtime readiness...", flush=True)
        if now - started > float(timeout_s):
            raise TimeoutError(f"Runtime did not become ready within {timeout_s:.1f}s.")


def start_bridge():
    env_set_if_value("EXCAVATOR_BRIDGE_HOST", os.environ.get("EXCAVATOR_BRIDGE_HOST", "0.0.0.0"))
    bridge_path = os.path.join(PROJECT_ROOT, "scripts", "bridge_test", "sand_site_tcp_bridge_server.py")
    runpy.run_path(bridge_path, run_name="__main__")
    print("[INFO] Sand-site TCP bridge scheduled.", flush=True)


def wait_task_done(simulation_app, task, label):
    if task is None or not hasattr(task, "done"):
        return
    while simulation_app.is_running() and not task.done():
        simulation_app.update()
    if task.done():
        try:
            exc = task.exception()
        except Exception:
            exc = None
        if exc is not None:
            raise RuntimeError(f"{label} failed: {type(exc).__name__}: {exc}")


def suppress_headless_log_noise():
    try:
        import carb
        import carb.logging

        logging = carb.logging.acquire_logging()
        logging.set_level_threshold(carb.logging.LEVEL_ERROR)
        for source in [
            "isaacsim.core.simulation_manager",
            "isaacsim.core.simulation_manager.plugin",
            "isaacsim.core.simulation_manager.impl.simulation_manager",
        ]:
            logging.set_log_enabled_for_source(source, False)
            logging.set_level_threshold_for_source(source, carb.logging.LEVEL_ERROR)
    except Exception:
        pass


def request_runtime_shutdown(simulation_app, rt, timeout_s=10.0):
    if rt is None:
        return
    try:
        rt.STATE["running"] = False
        rt.STATE["auto_collect_stop_requested"] = True
    except Exception:
        pass

    started = time.time()
    timeout_s = min(float(timeout_s), 3.0)
    while simulation_app.is_running():
        tasks = getattr(rt, "STATE", {}).get("async_tasks", {}) or {}
        main_task = tasks.get("main_loop")
        if main_task is None or not hasattr(main_task, "done") or main_task.done():
            break
        simulation_app.update()
        if time.time() - started > float(timeout_s):
            try:
                main_task.cancel()
            except Exception:
                pass
            break

    try:
        world = getattr(rt, "World", None)
        world_instance = world.instance() if world is not None else None
        if world_instance is not None:
            stop = getattr(world_instance, "stop", None)
            if callable(stop):
                stop()
    except Exception:
        pass
    try:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass
    try:
        cancel_tasks = getattr(rt, "cancel_registered_tasks", None)
        if callable(cancel_tasks):
            cancel_tasks(reason="launcher_shutdown", keep={"main_loop"})
    except Exception:
        pass

    for _ in range(2):
        try:
            simulation_app.update()
        except Exception:
            break


def run_auto_collect(simulation_app, rt, success_count, max_attempts, wait_export):
    suppress_headless_log_noise()
    success_count = max(1, int(success_count))
    max_attempts_arg = None if int(max_attempts or 0) <= 0 else int(max_attempts)
    print(
        "[INFO] Starting auto collect:",
        f"success_count={success_count}",
        f"max_attempts={max_attempts_arg if max_attempts_arg is not None else 'runtime_default'}",
        flush=True,
    )
    rt.request_auto_collect(success_count, max_attempts=max_attempts_arg)

    started = False
    last_print = 0.0
    while simulation_app.is_running():
        simulation_app.update()
        active = bool(rt.STATE.get("auto_collect_active", False))
        task = rt.STATE.get("auto_collect_task")
        if active or task is not None:
            started = True

        now = time.time()
        if now - last_print > 5.0:
            last_print = now
            print(
                "[AUTO COLLECT]",
                f"active={active}",
                f"attempts={rt.STATE.get('auto_collect_attempts', 0)}",
                f"success={rt.STATE.get('auto_collect_successes', 0)}",
                f"target={rt.STATE.get('auto_collect_requested', success_count)}",
                f"max_attempts={rt.STATE.get('auto_collect_max_attempts_requested', 0)}",
                f"run_dir={rt.STATE.get('auto_collect_run_dir', '')}",
                flush=True,
            )

        if started and task is not None and hasattr(task, "done") and task.done() and not active:
            wait_task_done(simulation_app, task, "auto_collect")
            break

    if wait_export:
        export_task = (rt.STATE.get("async_tasks") or {}).get("lerobot_v3_export")
        wait_task_done(simulation_app, export_task, "lerobot_v3_export")

    print(
        "[INFO] Auto collect finished:",
        f"attempts={rt.STATE.get('auto_collect_attempts', 0)}",
        f"success={rt.STATE.get('auto_collect_successes', 0)}",
        f"run_dir={rt.STATE.get('auto_collect_run_dir', '')}",
        flush=True,
    )


def main():
    args = parse_args()
    add_import_roots(PROJECT_ROOT)
    rt = None

    bridge_enabled = args.bridge if args.bridge is not None else (not args.auto_collect and not args.headless)
    if args.headless and bridge_enabled:
        raise SystemExit("--bridge requires a GUI viewport. Use non-headless mode or pass --no-bridge.")

    os.environ["EXCAVATOR_PROJECT_ROOT"] = PROJECT_ROOT
    if args.headless:
        os.environ["EXCAVATOR_HEADLESS"] = "1"
    if args.headless or not args.with_ui:
        os.environ["EXCAVATOR_NO_UI"] = "1"
    if args.sand_amount is not None:
        os.environ["EXCAVATOR_SAND_AMOUNT"] = str(args.sand_amount)
    if args.dataset_root:
        os.environ["EXCAVATOR_DATASET_ROOT"] = paths.resolve_existing_path(args.dataset_root, root=PROJECT_ROOT)
    if args.disable_export:
        os.environ["EXCAVATOR_AUTO_EXPORT_LEROBOT_V3"] = "0"
    if args.export_python:
        os.environ["EXCAVATOR_LEROBOT_V3_EXPORT_PYTHON"] = args.export_python
    os.environ["EXCAVATOR_BRIDGE_HOST"] = str(args.bridge_host)
    os.environ["EXCAVATOR_BRIDGE_PORT"] = str(args.bridge_port)

    from isaacsim import SimulationApp

    graphics_api = str(args.graphics_api or "auto").lower()
    extra_args = ["--/renderer/multiGpu/autoEnable=0"]
    if graphics_api == "auto" and platform.system().lower().startswith("win") and args.headless and args.auto_collect:
        graphics_api = "d3d12"
    if graphics_api == "d3d12":
        extra_args.append("--/app/vulkan=false")
    elif graphics_api == "vulkan":
        extra_args.append("--/app/vulkan=true")

    os.environ["EXCAVATOR_GRAPHICS_API"] = str(graphics_api)
    os.environ["EXCAVATOR_RENDERER"] = str(args.renderer)
    os.environ["EXCAVATOR_ACTIVE_GPU"] = str(int(args.active_gpu))
    os.environ["EXCAVATOR_PHYSICS_GPU"] = str(int(args.physics_gpu))
    os.environ["EXCAVATOR_MULTI_GPU"] = "1" if bool(args.multi_gpu) else "0"

    print(
        "[INFO] Isaac renderer config:",
        f"renderer={args.renderer}",
        f"graphics_api={graphics_api}",
        f"multi_gpu={bool(args.multi_gpu)}",
        f"active_gpu={int(args.active_gpu)}",
        f"physics_gpu={int(args.physics_gpu)}",
        f"extra_args={extra_args}",
        flush=True,
    )

    simulation_app = SimulationApp({
        "headless": bool(args.headless),
        "width": int(args.width),
        "height": int(args.height),
        "renderer": str(args.renderer),
        "multi_gpu": bool(args.multi_gpu),
        "active_gpu": int(args.active_gpu),
        "physics_gpu": int(args.physics_gpu),
        "extra_args": extra_args,
    })
    if args.headless and args.auto_collect:
        suppress_headless_log_noise()

    try:
        open_stage(simulation_app, args.scene)

        from excavator_app.bootstrap import run_excavator_with_sand

        rt = run_excavator_with_sand()
        wait_runtime_ready(simulation_app, rt, args.wait_runtime_seconds)
        if args.sand_amount is not None and not args.random_sand_amount:
            rt.STATE["auto_scene_random_sand_amount_enabled"] = False
            print("[INFO] Auto collect sand amount randomization disabled because --sand-amount was provided.", flush=True)

        if bridge_enabled:
            start_bridge()

        if args.auto_collect:
            run_auto_collect(
                simulation_app,
                rt,
                success_count=args.success_count,
                max_attempts=args.max_attempts,
                wait_export=args.wait_export,
            )
            if not args.keep_running_after_auto_collect:
                request_runtime_shutdown(simulation_app, rt)
                return

        print("[INFO] Runtime is running. Press Ctrl+C to exit.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.", flush=True)
    finally:
        request_runtime_shutdown(simulation_app, rt)
        try:
            simulation_app.close(
                wait_for_replicator=False,
                skip_cleanup=bool(args.headless and args.auto_collect),
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
