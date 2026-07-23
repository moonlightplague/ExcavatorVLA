#!/usr/bin/env python3
"""
Launch the full sand + truck + excavator runtime without using main.py.

Typical VLA training scene:
    /isaac-sim/python.sh run_vla_train_scene.py --sand-amount 1.0 --bridge

Headless auto collect:
    /isaac-sim/python.sh run_vla_train_scene.py --headless --auto-collect --success-count 20 --max-attempts 120

Fixed deployment-scene replay:
    /isaac-sim/python.sh run_vla_train_scene.py --headless --auto-collect --success-count 5 --scene-seed 2 --sand-settle-frames 240
"""

import argparse
import asyncio
import importlib
import json
import math
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
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=None,
        help=(
            "Reproduce the deployment scene for this seed once, then reuse its exact "
            "sand/truck pose and amount for every auto-collect attempt."
        ),
    )
    parser.add_argument(
        "--fixed-scene-profile",
        default=None,
        help=(
            "JSON file containing the final sand/truck scene applied by the "
            "deployment simulator. This bypasses scene sampling and truck "
            "baseline reconstruction."
        ),
    )
    parser.add_argument(
        "--sand-settle-frames",
        type=int,
        default=None,
        help="Fix auto-collect sand settling to exactly this many simulation frames.",
    )
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
    parser.add_argument(
        "--log-mode",
        choices=("data", "data_multi", "data_multi_recovery", "debug", "profile"),
        default="",
        help=(
            "Runtime mode: data collects one scoop per episode; data_multi collects "
            "multiple consecutive scoops in one v2.0 episode; data_multi_recovery "
            "adds controlled recoverable deviations and recovery supervision."
        ),
    )
    parser.add_argument(
        "--scoops-per-episode",
        type=int,
        default=1,
        help=(
            "Optional quality floor for data_multi modes, not a fixed scoop count. "
            "Collection continues until no effective dig target remains (default: 1)."
        ),
    )
    parser.add_argument(
        "--max-scoops-per-episode",
        type=int,
        default=64,
        help=(
            "Technical runaway guard for adaptive multi-scoop collection. Reaching this "
            "limit is rejected rather than treated as natural completion (default: 64)."
        ),
    )
    parser.add_argument(
        "--stable-camera-render",
        action="store_true",
        help=(
            "Opt in to deterministic/stable render settings for camera-quality diagnostics. "
            "Off by default because these global Kit settings can slow auto-collect execution."
        ),
    )
    parser.add_argument(
        "--camera-capture-resolution",
        default="",
        help=(
            "Optional viewport capture resolution before downsampling, e.g. 256 or 1024. "
            "Unset keeps runtime default."
        ),
    )
    parser.add_argument(
        "--fast-sampled-replay",
        action="store_true",
        help=(
            "Enable planned-time sampled replay for the free-space pre_dig stage only. "
            "Loaded transit, lift, contact, dump, and settle remain full PhysX."
        ),
    )
    parser.add_argument(
        "--fast-replay-sample-hz",
        type=float,
        default=10.0,
        help="Planned-time sample rate for --fast-sampled-replay (default: 10 Hz).",
    )
    parser.add_argument(
        "--attempt-motion-speed",
        type=float,
        default=None,
        help="Physical auto-attempt free-space motion scale. Default runtime policy is 2.0x with lower contact/carry caps.",
    )
    parser.add_argument(
        "--dataset-hz",
        type=float,
        default=None,
        help="State and camera target sampling frequency in simulation time. Default runtime policy is 10 Hz.",
    )
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
        profiled_simulation_update(simulation_app, None, f"wait_task:{label}")
    if task.done():
        if hasattr(task, "cancelled") and task.cancelled():
            print(f"[WARN] Task cancelled: {label}", flush=True)
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            print(f"[WARN] Task cancelled: {label}", flush=True)
            return
        if exc is not None:
            raise RuntimeError(f"{label} failed: {type(exc).__name__}: {exc}")


def launcher_update_profile_threshold_ms():
    raw = os.environ.get("EXCAVATOR_LAUNCHER_UPDATE_PROFILE_THRESHOLD_MS", "")
    if raw != "":
        try:
            return max(0.0, float(raw))
        except Exception:
            return 0.0
    mode = str(os.environ.get("EXCAVATOR_LOG_MODE", "data") or "data").strip().lower()
    if mode == "profile":
        return 400.0
    if mode == "debug":
        return 1500.0
    return 0.0


def stable_camera_render_requested(args=None):
    raw = os.environ.get("EXCAVATOR_STABLE_CAMERA_RENDER", "")
    if raw != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(args, "stable_camera_render", False))


def apply_stable_dataset_render_settings(reason="startup", enabled=False):
    """Reduce temporal render variance in captured training frames."""
    if not bool(enabled):
        print(
            "[INFO] Stable dataset render settings skipped:",
            f"reason={reason}",
            "enabled=False",
            "note=use --stable-camera-render or EXCAVATOR_STABLE_CAMERA_RENDER=1 to enable",
            flush=True,
        )
        return {}
    try:
        import carb
    except Exception as exc:
        print(f"[WARN] Stable render settings skipped: {type(exc).__name__}: {exc}", flush=True)
        return {}
    settings = carb.settings.get_settings()
    applied = {}

    def set_value(key, value):
        try:
            settings.set(key, value)
            applied[str(key)] = value
        except Exception as exc:
            applied[str(key)] = f"{type(exc).__name__}: {exc}"

    def set_bool(key, value):
        try:
            settings.set_bool(key, bool(value))
            applied[str(key)] = bool(value)
        except Exception:
            set_value(key, bool(value))

    def set_int(key, value):
        try:
            settings.set_int(key, int(value))
            applied[str(key)] = int(value)
        except Exception:
            set_value(key, int(value))

    def set_float(key, value):
        try:
            settings.set_float(key, float(value))
            applied[str(key)] = float(value)
        except Exception:
            set_value(key, float(value))

    # Keep the camera pipeline on a deterministic Kit clock and avoid temporal
    # post effects that can make static background pixels shimmer frame-to-frame.
    set_bool("/omni/replicator/captureOnPlay", False)
    set_bool("/app/asyncRendering", False)
    set_bool("/exts/isaacsim.core.throttling/enable_async", False)
    set_bool("/app/player/useFixedTimeStepping", True)

    # DLSS mode 2 is the stable/off-style mode used by the camera diagnostics.
    set_int("/rtx/post/dlss/execMode", 2)
    set_int("/rtx/post/aa/op", 0)
    set_bool("/rtx/post/motionblur/enabled", False)
    set_bool("/rtx/post/tonemap/autoExposure/enabled", False)
    set_bool("/rtx/post/histogram/enabled", False)
    set_float("/rtx/post/tonemap/exposure", 0.0)
    set_float("/rtx/post/tonemap/whitepoint", 1.0)

    # Disable adaptive render resolution variants where available.
    set_bool("/rtx-transient/resourcemanager/enableTextureStreaming", False)
    set_bool("/rtx/post/dlss/autoExposure", False)
    set_bool("/rtx/post/dlss/autoScale", False)

    print(
        "[INFO] Stable dataset render settings applied:",
        f"reason={reason}",
        f"count={len(applied)}",
        flush=True,
    )
    return applied


def runtime_update_snapshot(rt):
    if rt is None:
        return {}
    try:
        state = getattr(rt, "STATE", {}) or {}
        progress = dict(state.get("auto_collect_progress", {}) or {})
        step_recent = list(state.get("step_updates_recent", []) or [])
        debug_last = dict(state.get("debug_profile_last", {}) or {})
        sample_spans = dict(state.get("dataset_record_sample_spans", {}) or {})
        return {
            "active_task": str(state.get("active_task_name", "") or ""),
            "stage": str(progress.get("stage", "") or ""),
            "result": str(progress.get("result", "") or ""),
            "progress_age_s": round(max(0.0, time.time() - float(progress.get("updated_at", time.time()) or time.time())), 3)
            if progress else 0.0,
            "attempt": int(progress.get("attempt", state.get("auto_collect_attempts", 0)) or 0),
            "run_dir": str(state.get("auto_collect_run_dir", "") or ""),
            "auto_collect_active": bool(state.get("auto_collect_active", False)),
            "dataset_recording": bool(state.get("dataset_recording", False)),
            "dataset_samples": int(state.get("dataset_samples", 0) or 0),
            "dataset_episode": str(state.get("dataset_episode_uid", "") or ""),
            "camera_backend": str(state.get("dataset_camera_backend", "") or ""),
            "camera_available": bool(state.get("dataset_camera_available", False)),
            "debug_visuals": bool(state.get("debug_visuals_visible", False)),
            "trace_mode": int(state.get("trace_mode", 0) or 0),
            "last_step_caller": str(step_recent[-1].get("caller", "") if step_recent else ""),
            "last_step_frame_ms": float(step_recent[-1].get("per_frame_ms", 0.0) if step_recent else 0.0),
            "last_profile_label": str(debug_last.get("label", "") or ""),
            "last_profile_ms": float(debug_last.get("elapsed_ms", 0.0) or 0.0),
            "sample_spans": sample_spans,
        }
    except Exception as exc:
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}


def record_launcher_update_profile(rt, label, elapsed_ms, threshold_ms, before, after):
    if rt is None:
        return
    try:
        state = getattr(rt, "STATE", {}) or {}
        profile = state.get("launcher_update_profile")
        if not isinstance(profile, dict):
            profile = {
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "slow_count": 0,
                "threshold_ms": float(threshold_ms),
                "by_context": {},
            }
        profile["count"] = int(profile.get("count", 0) or 0) + 1
        profile["total_ms"] = float(profile.get("total_ms", 0.0) or 0.0) + float(elapsed_ms)
        profile["avg_ms"] = profile["total_ms"] / max(1, int(profile.get("count", 0) or 0))
        profile["max_ms"] = max(float(profile.get("max_ms", 0.0) or 0.0), float(elapsed_ms))
        if elapsed_ms >= threshold_ms:
            profile["slow_count"] = int(profile.get("slow_count", 0) or 0) + 1

        after = after if isinstance(after, dict) else {}
        context_key = "|".join(
            [
                str(label or "launcher"),
                str(after.get("stage", "")),
                str(after.get("result", "")),
                str(after.get("active_task", "")),
                f"recording={bool(after.get('dataset_recording', False))}",
                f"viz={bool(after.get('debug_visuals', False))}",
                str(after.get("camera_backend", "")),
            ]
        )
        by_context = profile.get("by_context")
        if not isinstance(by_context, dict):
            by_context = {}
        row = by_context.get(context_key)
        if not isinstance(row, dict):
            row = {
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "slow_count": 0,
                "label": str(label or ""),
                "stage": str(after.get("stage", "")),
                "result": str(after.get("result", "")),
                "active_task": str(after.get("active_task", "")),
                "dataset_recording": bool(after.get("dataset_recording", False)),
                "debug_visuals": bool(after.get("debug_visuals", False)),
                "camera_backend": str(after.get("camera_backend", "")),
            }
        row["count"] = int(row.get("count", 0) or 0) + 1
        row["total_ms"] = float(row.get("total_ms", 0.0) or 0.0) + float(elapsed_ms)
        row["avg_ms"] = row["total_ms"] / max(1, int(row.get("count", 0) or 0))
        row["max_ms"] = max(float(row.get("max_ms", 0.0) or 0.0), float(elapsed_ms))
        if elapsed_ms >= threshold_ms:
            row["slow_count"] = int(row.get("slow_count", 0) or 0) + 1
        by_context[context_key] = row
        profile["by_context"] = by_context
        state["launcher_update_profile"] = profile

        if elapsed_ms >= threshold_ms:
            entry = {
                "t": time.time(),
                "label": str(label or ""),
                "elapsed_ms": round(float(elapsed_ms), 3),
                "threshold_ms": round(float(threshold_ms), 3),
                "before": before,
                "after": after,
            }
            recent = state.get("launcher_update_recent")
            if not isinstance(recent, list):
                recent = []
            recent.append(entry)
            state["launcher_update_recent"] = recent[-64:]

            now = time.time()
            last_print = float(state.get("launcher_update_last_print_time", 0.0) or 0.0)
            slow_count = int(profile.get("slow_count", 0) or 0)
            if slow_count <= 5 or elapsed_ms >= threshold_ms * 4.0 or now - last_print >= 5.0:
                state["launcher_update_last_print_time"] = now
                print(
                    "[LAUNCHER UPDATE SLOW]",
                    f"label={label}",
                    f"elapsed_ms={elapsed_ms:.1f}",
                    f"stage={after.get('stage', '')}",
                    f"result={after.get('result', '')}",
                    f"active_task={after.get('active_task', '')}",
                    f"recording={after.get('dataset_recording', False)}",
                    f"samples={after.get('dataset_samples', 0)}",
                    f"viz={after.get('debug_visuals', False)}",
                    f"last_step={after.get('last_step_caller', '')}:{after.get('last_step_frame_ms', 0.0):.1f}ms",
                    f"last_profile={after.get('last_profile_label', '')}:{after.get('last_profile_ms', 0.0):.1f}ms",
                    flush=True,
                )
    except Exception:
        pass


def profiled_simulation_update(simulation_app, rt=None, label="launcher"):
    threshold_ms = launcher_update_profile_threshold_ms()
    started = time.perf_counter() if threshold_ms > 0.0 else 0.0
    before = runtime_update_snapshot(rt) if threshold_ms > 0.0 else {}
    simulation_app.update()
    if threshold_ms <= 0.0:
        return
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after = runtime_update_snapshot(rt)
    record_launcher_update_profile(rt, label, elapsed_ms, threshold_ms, before, after)
    if elapsed_ms <= threshold_ms:
        return


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
    last_status_key = None
    heartbeat_interval = float(os.environ.get("EXCAVATOR_AUTO_COLLECT_HEARTBEAT_SECONDS", "0") or 0)
    last_heartbeat = time.time()
    while simulation_app.is_running():
        profiled_simulation_update(simulation_app, rt, "auto_collect")
        active = bool(rt.STATE.get("auto_collect_active", False))
        task = rt.STATE.get("auto_collect_task")
        if active or task is not None:
            started = True

        now = time.time()
        status_key = (
            bool(active),
            int(rt.STATE.get("auto_collect_successes", 0) or 0),
            int(rt.STATE.get("auto_collect_rejections", 0) or 0),
            int(rt.STATE.get("auto_collect_failures", 0) or 0),
            str(rt.STATE.get("auto_collect_last_result", "") or ""),
        )
        heartbeat_due = heartbeat_interval > 0.0 and now - last_heartbeat > heartbeat_interval
        if status_key != last_status_key or heartbeat_due:
            if heartbeat_due:
                last_heartbeat = now
            last_status_key = status_key
            print(
                "[AUTO COLLECT]",
                f"active={active}",
                f"attempts={rt.STATE.get('auto_collect_attempts', 0)}",
                f"success={rt.STATE.get('auto_collect_successes', 0)}",
                f"rejected={rt.STATE.get('auto_collect_rejections', 0)}",
                f"fail={rt.STATE.get('auto_collect_failures', 0)}",
                f"target={rt.STATE.get('auto_collect_requested', success_count)}",
                f"max_attempts={rt.STATE.get('auto_collect_max_attempts_requested', 0)}",
                f"last={rt.STATE.get('auto_collect_last_result', '')}",
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
    run_dir = str(rt.STATE.get("auto_collect_run_dir", "") or "")
    summary = {}
    summary_path = os.path.join(run_dir, "summary.json") if run_dir else ""
    if summary_path and os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                summary = loaded
        except Exception as exc:
            print(
                "[WARN] Auto collect result could not read summary:",
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    requested = max(
        success_count,
        int(summary.get("requested", 0) or 0),
        int(rt.STATE.get("auto_collect_requested", 0) or 0),
    )
    successes = max(
        int(summary.get("successes", 0) or 0),
        int(rt.STATE.get("auto_collect_successes", 0) or 0),
    )
    exit_reason = str(
        summary.get("loop_exit_reason")
        or rt.STATE.get("auto_collect_loop_exit_reason", "")
        or ""
    )
    cancelled = bool(
        summary.get("loop_cancelled", False)
        or rt.STATE.get("auto_collect_loop_cancelled", False)
    )
    result = {
        "schema": "excavator_auto_collect_result_v1",
        "completed": bool(
            successes >= requested
            and exit_reason == "target_successes_reached"
            and not cancelled
        ),
        "requested": int(requested),
        "successes": int(successes),
        "attempts": int(
            max(
                int(summary.get("attempts", 0) or 0),
                int(rt.STATE.get("auto_collect_attempts", 0) or 0),
            )
        ),
        "loop_exit_reason": exit_reason,
        "loop_cancelled": cancelled,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "finished_at": time.time(),
    }
    result_path = str(os.environ.get("EXCAVATOR_AUTO_COLLECT_RESULT_FILE", "") or "").strip()
    if result_path:
        result_path = os.path.abspath(os.path.expanduser(result_path))
        try:
            os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
            temporary_path = f"{result_path}.tmp.{os.getpid()}"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
            os.replace(temporary_path, result_path)
            print(
                "[INFO] Auto collect result written:",
                f"completed={result['completed']}",
                f"path={result_path}",
                flush=True,
            )
        except Exception as exc:
            print(
                "[ERROR] Auto collect result write failed:",
                f"{type(exc).__name__}: {exc}",
                f"path={result_path}",
                flush=True,
            )
    return result


def main():
    args = parse_args()
    add_import_roots(PROJECT_ROOT)
    rt = None
    stable_render_enabled = stable_camera_render_requested(args)

    bridge_enabled = args.bridge if args.bridge is not None else (not args.auto_collect and not args.headless)
    if args.headless and bridge_enabled:
        raise SystemExit("--bridge requires a GUI viewport. Use non-headless mode or pass --no-bridge.")
    if (
        args.scene_seed is not None
        or args.fixed_scene_profile is not None
    ) and not args.auto_collect:
        raise SystemExit(
            "--scene-seed/--fixed-scene-profile require --auto-collect."
        )
    if args.sand_settle_frames is not None and int(args.sand_settle_frames) < 1:
        raise SystemExit("--sand-settle-frames must be at least 1.")

    os.environ["EXCAVATOR_PROJECT_ROOT"] = PROJECT_ROOT
    if args.headless:
        os.environ["EXCAVATOR_HEADLESS"] = "1"
    if args.headless or not args.with_ui:
        os.environ["EXCAVATOR_NO_UI"] = "1"
    if args.sand_amount is not None:
        os.environ["EXCAVATOR_SAND_AMOUNT"] = str(args.sand_amount)
    if args.fixed_scene_profile is not None:
        profile_path = paths.resolve_existing_path(
            args.fixed_scene_profile,
            root=PROJECT_ROOT,
        )
        if not os.path.isfile(profile_path):
            raise SystemExit(
                f"--fixed-scene-profile does not exist: {profile_path}"
            )
        with open(profile_path, "r", encoding="utf-8") as handle:
            fixed_profile = json.load(handle)
        if not isinstance(fixed_profile, dict):
            raise SystemExit("--fixed-scene-profile must contain a JSON object.")
        required_vectors = {
            "robot_world_position_xyz": 3,
            "robot_world_orientation_wxyz": 4,
            "sand_xy": 2,
            "truck_translation_xyz": 3,
            "unload_landing_xyz": 3,
        }
        for key, size in required_vectors.items():
            value = fixed_profile.get(key)
            if not isinstance(value, (list, tuple)) or len(value) < size:
                raise SystemExit(
                    f"--fixed-scene-profile missing {key}[{size}]"
                )
            if not all(math.isfinite(float(item)) for item in value[:size]):
                raise SystemExit(
                    f"--fixed-scene-profile has non-finite {key}"
                )
        selected_unload_landing = fixed_profile.get(
            "selected_unload_landing_xyz"
        )
        if selected_unload_landing is not None:
            if (
                not isinstance(selected_unload_landing, (list, tuple))
                or len(selected_unload_landing) < 3
                or not all(
                    math.isfinite(float(item))
                    for item in selected_unload_landing[:3]
                )
            ):
                raise SystemExit(
                    "--fixed-scene-profile selected_unload_landing_xyz "
                    "must contain three finite values."
                )
        for key in ("sand_amount_multiplier", "truck_yaw_deg"):
            if not math.isfinite(float(fixed_profile.get(key, float("nan")))):
                raise SystemExit(
                    f"--fixed-scene-profile missing finite {key}"
                )
        initial_pose = fixed_profile.get("initial_pose_deg")
        if initial_pose is not None:
            if not isinstance(initial_pose, dict):
                raise SystemExit(
                    "--fixed-scene-profile initial_pose_deg must be an object."
                )
            for joint_name in ("swing", "boom", "arm", "bucket"):
                if not math.isfinite(
                    float(initial_pose.get(joint_name, float("nan")))
                ):
                    raise SystemExit(
                        "--fixed-scene-profile initial_pose_deg missing finite "
                        f"{joint_name}"
                    )
        os.environ["EXCAVATOR_AUTO_SCENE_FIXED_PROFILE_JSON"] = json.dumps(
            fixed_profile,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        os.environ["EXCAVATOR_AUTO_SCENE_FIXED_PROFILE_PATH"] = profile_path
    if args.scene_seed is not None:
        os.environ["EXCAVATOR_AUTO_SCENE_REPLAY_SEED"] = str(int(args.scene_seed))
    if args.scene_seed is not None or args.fixed_scene_profile is not None:
        os.environ["EXCAVATOR_RANDOM_TRUCK"] = "0"
        os.environ["EXCAVATOR_RANDOM_TRUCK_YAW"] = "0"
        os.environ["EXCAVATOR_RANDOM_ROBOT_YAW"] = "0"
        os.environ["EXCAVATOR_RANDOM_SAND_XY"] = "0"
        os.environ["EXCAVATOR_RANDOM_SAND_AMOUNT"] = "0"
    if args.sand_settle_frames is not None:
        settle_frames = int(args.sand_settle_frames)
        os.environ["EXCAVATOR_SAND_RESET_SETTLE_MIN_FRAMES"] = str(settle_frames)
        os.environ["EXCAVATOR_SAND_RESET_SETTLE_MAX_FRAMES"] = str(settle_frames)
        os.environ["EXCAVATOR_SAND_RESET_SETTLE_ENFORCE_MIN"] = "1"
    if args.dataset_root:
        os.environ["EXCAVATOR_DATASET_ROOT"] = paths.resolve_existing_path(args.dataset_root, root=PROJECT_ROOT)
    if args.disable_export:
        os.environ["EXCAVATOR_AUTO_EXPORT_LEROBOT_V3"] = "0"
    if args.export_python:
        os.environ["EXCAVATOR_LEROBOT_V3_EXPORT_PYTHON"] = args.export_python
    # Do not inherit a stale EXCAVATOR_LOG_MODE=profile/debug from the shell.
    # Auto collection should default to the fastest data mode unless the launch
    # command explicitly requests diagnostics.
    selected_log_mode = str(args.log_mode or "data")
    os.environ["EXCAVATOR_LOG_MODE"] = selected_log_mode
    multi_scoop_mode = selected_log_mode in ("data_multi", "data_multi_recovery")
    os.environ["EXCAVATOR_MULTI_SCOOP"] = "1" if multi_scoop_mode else "0"
    os.environ["EXCAVATOR_MULTI_SCOOP_RECOVERY"] = (
        "1" if selected_log_mode == "data_multi_recovery" else "0"
    )
    multi_scoop_min = max(1, int(args.scoops_per_episode)) if multi_scoop_mode else 1
    multi_scoop_max = (
        max(multi_scoop_min + 1, int(args.max_scoops_per_episode))
        if multi_scoop_mode
        else 1
    )
    os.environ["EXCAVATOR_MULTI_SCOOP_ADAPTIVE_STOP"] = "1" if multi_scoop_mode else "0"
    os.environ["EXCAVATOR_MULTI_SCOOPS_MIN_PER_EPISODE"] = str(multi_scoop_min)
    os.environ["EXCAVATOR_MULTI_SCOOPS_MAX_PER_EPISODE"] = str(multi_scoop_max)
    # Compatibility alias for older runtime/tools that still read the fixed-count name.
    os.environ["EXCAVATOR_MULTI_SCOOPS_PER_EPISODE"] = str(multi_scoop_min)
    os.environ["EXCAVATOR_BRIDGE_HOST"] = str(args.bridge_host)
    os.environ["EXCAVATOR_BRIDGE_PORT"] = str(args.bridge_port)
    os.environ["EXCAVATOR_FAST_SAMPLED_REPLAY"] = "1" if bool(args.fast_sampled_replay) else "0"
    os.environ["EXCAVATOR_FAST_REPLAY_PRE_DIG"] = "1"
    os.environ["EXCAVATOR_FAST_REPLAY_SAMPLE_HZ"] = str(max(1.0, float(args.fast_replay_sample_hz)))
    if args.attempt_motion_speed is not None:
        env_set_if_value(
            "EXCAVATOR_AUTO_ATTEMPT_MOTION_SPEED_SCALE",
            max(0.1, float(args.attempt_motion_speed)),
        )
    if args.dataset_hz is not None:
        dataset_hz = max(1.0, float(args.dataset_hz))
        if bool(args.auto_collect) and abs(dataset_hz - 10.0) > 1.0e-6:
            raise ValueError(
                f"--auto-collect requires original 10 Hz source data; got --dataset-hz={dataset_hz:g}"
            )
        env_set_if_value("EXCAVATOR_DATASET_SAMPLE_INTERVAL", 1.0 / dataset_hz)
        env_set_if_value("EXCAVATOR_DATASET_CAMERA_FREQUENCY", int(round(dataset_hz)))
        env_set_if_value("EXCAVATOR_DATASET_CAMERA_BACKGROUND_INTERVAL_S", 1.0 / dataset_hz)
    if bool(args.auto_collect) and bool(args.fast_sampled_replay):
        raise ValueError(
            "--fast-sampled-replay is incompatible with original physical 10 Hz auto-collect data"
        )
    env_set_if_value("EXCAVATOR_DATASET_CAMERA_CAPTURE_RESOLUTION", args.camera_capture_resolution)

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
    apply_stable_dataset_render_settings(reason="post_simulation_app", enabled=stable_render_enabled)
    if args.headless and args.auto_collect:
        suppress_headless_log_noise()

    try:
        open_stage(simulation_app, args.scene)
        apply_stable_dataset_render_settings(reason="post_open_stage", enabled=stable_render_enabled)

        from excavator_app.bootstrap import run_excavator_with_sand

        rt = run_excavator_with_sand()
        try:
            rt.set_log_mode(selected_log_mode, announce=bool(args.log_mode))
        except Exception:
            pass
        wait_runtime_ready(simulation_app, rt, args.wait_runtime_seconds)
        stable_settings = apply_stable_dataset_render_settings(reason="runtime_ready", enabled=stable_render_enabled)
        try:
            rt.STATE["dataset_stable_render_settings"] = stable_settings
            rt.STATE["dataset_stable_render_settings_enabled"] = bool(stable_render_enabled)
        except Exception:
            pass
        if args.scene_seed is not None or args.fixed_scene_profile is not None:
            rt.STATE["auto_scene_random_truck_enabled"] = False
            rt.STATE["auto_scene_random_truck_yaw_enabled"] = False
            rt.STATE["auto_scene_random_robot_yaw_enabled"] = False
            rt.STATE["auto_scene_random_sand_xy_enabled"] = False
            rt.STATE["auto_scene_random_sand_amount_enabled"] = False
            print(
                "[INFO] Fixed auto-collect scene replay enabled:",
                f"scene_seed={args.scene_seed if args.scene_seed is not None else 'profile'}",
                f"profile={args.fixed_scene_profile or 'derived_from_seed'}",
                f"sand_settle_frames={args.sand_settle_frames or 'runtime-default'}",
                "robot_yaw=authored",
                flush=True,
            )
        elif args.sand_amount is not None and not args.random_sand_amount:
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
