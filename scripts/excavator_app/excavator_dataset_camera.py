import asyncio
import inspect
import os
import time

import numpy as np
import omni.usd
try:
    import omni.replicator.core as rep
except Exception:
    rep = None

from pxr import Usd, UsdGeom, Gf, Sdf
try:
    from PIL import Image
except Exception:
    Image = None


CAMERA_MODULE_VERSION = "dataset_camera_explicit_replicator_rgb_v13_offscreen_replicator_async"
SYNC_STEP_ERROR_TEXT = "Synchronous call to `step`"


def replicator_tick_enabled(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_REPLICATOR_TICK", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_replicator_tick_enabled", None)
    if state_value is not None:
        return bool(state_value)
    return True


def syntheticdata_wait_enabled(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_SYNTHETICDATA_WAIT", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(rt.STATE.get("dataset_camera_syntheticdata_wait_enabled", False))


def camera_tick_timeout_seconds(rt):
    return max(0.1, _state_float(rt, "dataset_camera_tick_timeout_s", "EXCAVATOR_CAMERA_TICK_TIMEOUT_S", 12.0))


def camera_wait_for_render_enabled(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_WAIT_FOR_RENDER", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_wait_for_render", None)
    if state_value is not None:
        return bool(state_value)
    return True


def camera_delta_time(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_DELTA_TIME", "")
    if value == "":
        value = rt.STATE.get("dataset_camera_delta_time", 0.0)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("none", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return 0.0


def camera_rt_subframes(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_RT_SUBFRAMES", "")
    if value == "":
        value = rt.STATE.get("dataset_camera_rt_subframes", 16)
    try:
        return max(1, int(value))
    except Exception:
        return 16


def reset_replicator_tick_state(rt, reason=""):
    rt.STATE["dataset_camera_replicator_step_disabled"] = False
    rt.STATE["dataset_camera_replicator_step_disable_logged"] = False
    rt.STATE["dataset_camera_replicator_step_disable_reason"] = ""
    rt.STATE["dataset_camera_replicator_step_disable_detail"] = ""
    rt.STATE["dataset_camera_tick_pending"] = False
    rt.STATE["dataset_camera_last_tick_reset_reason"] = str(reason)


def mark_replicator_step_disabled(rt, reason, detail=""):
    reason = str(reason or "replicator_step_disabled")
    detail = str(detail or "")
    # Keep this as diagnostic state only. Production capture must fail fast
    # when explicit Replicator RGB rendering fails.
    rt.STATE["dataset_camera_replicator_step_disabled"] = False
    rt.STATE["dataset_camera_replicator_step_disable_reason"] = reason
    rt.STATE["dataset_camera_replicator_step_disable_detail"] = detail
    return reason


def disabled_step_context(rt):
    return {
        "replicator_step_disabled": bool(rt.STATE.get("dataset_camera_replicator_step_disabled", False)),
        "disable_reason": str(rt.STATE.get("dataset_camera_replicator_step_disable_reason", "") or ""),
        "disable_detail": str(rt.STATE.get("dataset_camera_replicator_step_disable_detail", "") or ""),
        "last_tick_reset_reason": str(rt.STATE.get("dataset_camera_last_tick_reset_reason", "") or ""),
    }


def image_extension(rt):
    requested = str(rt.STATE.get("dataset_camera_image_format", "ppm") or "ppm").strip().lower()
    if requested == "png":
        return "png" if Image is not None else "ppm"
    return "ppm"


def resolution(rt):
    value = rt.STATE.get("dataset_camera_resolution", rt.DATASET_CAMERA_DEFAULT_RESOLUTION)
    try:
        width = int(value[0])
        height = int(value[1])
    except Exception:
        width, height = rt.DATASET_CAMERA_DEFAULT_RESOLUTION
    return [max(32, width), max(32, height)]


def _state_float(rt, key, env_key, default_value):
    value = os.environ.get(env_key, "")
    if value == "":
        value = rt.STATE.get(key, default_value)
    try:
        return float(value)
    except Exception:
        return float(default_value)


def camera_black_mean_threshold(rt):
    return max(0.0, _state_float(rt, "dataset_camera_black_mean_threshold", "EXCAVATOR_DATASET_CAMERA_BLACK_MEAN", 0.5))


def camera_black_max_threshold(rt):
    return max(0.0, _state_float(rt, "dataset_camera_black_max_threshold", "EXCAVATOR_DATASET_CAMERA_BLACK_MAX", 2.0))


def coerce_rgb_uint8(rgb):
    if rgb is None:
        return None, "rgb_none", {}
    try:
        arr = np.asarray(rgb)
    except Exception as exc:
        return None, f"rgb_array_error:{type(exc).__name__}:{exc}", {}
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None, f"rgb_bad_shape:{list(arr.shape)}", {}
    arr = arr[:, :, :3]
    source_dtype = str(arr.dtype)
    source_min = None
    source_max = None
    if arr.size:
        try:
            source_min = float(np.nanmin(arr))
            source_max = float(np.nanmax(arr))
        except Exception:
            source_min = None
            source_max = None
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if source_max is not None and source_max <= 1.5 and (source_min is None or source_min >= -0.01):
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    stats = rgb_stats(arr)
    stats["source_dtype"] = source_dtype
    if source_min is not None:
        stats["source_min"] = source_min
    if source_max is not None:
        stats["source_max"] = source_max
    return arr, "ok", stats


def resize_rgb_to_resolution(rgb, target_resolution):
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None, f"resize_bad_shape:{list(arr.shape)}"
    target_w = int(target_resolution[0])
    target_h = int(target_resolution[1])
    if int(arr.shape[1]) == target_w and int(arr.shape[0]) == target_h:
        return np.ascontiguousarray(arr[:, :, :3]), "ok"
    arr = np.ascontiguousarray(arr[:, :, :3])
    if Image is not None:
        try:
            image = Image.fromarray(arr)
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8), "pil_resize"
        except Exception as exc:
            return None, f"pil_resize_failed:{type(exc).__name__}:{exc}"
    try:
        y_idx = np.linspace(0, max(0, arr.shape[0] - 1), target_h).astype(np.int32)
        x_idx = np.linspace(0, max(0, arr.shape[1] - 1), target_w).astype(np.int32)
        return np.ascontiguousarray(arr[y_idx][:, x_idx, :3]), "nearest_resize"
    except Exception as exc:
        return None, f"nearest_resize_failed:{type(exc).__name__}:{exc}"


def rgb_stats(rgb):
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3 or arr.size == 0:
        return {"shape": [int(x) for x in arr.shape], "mean": 0.0, "max": 0}
    arr = arr[:, :, :3]
    flat = arr.reshape(-1, 3)
    step = max(1, int(flat.shape[0] // 4096))
    sample = flat[::step]
    return {
        "shape": [int(x) for x in arr.shape],
        "mean": float(np.mean(sample)) if sample.size else 0.0,
        "max": int(np.max(sample)) if sample.size else 0,
        "min": int(np.min(sample)) if sample.size else 0,
    }


def validate_rgb_content(rt, rgb, stats=None):
    if rgb is None:
        return False, "rgb_none"
    if stats is None:
        stats = rgb_stats(rgb)
    mean_value = float(stats.get("mean", 0.0) or 0.0)
    max_value = float(stats.get("max", 0.0) or 0.0)
    if max_value <= camera_black_max_threshold(rt) and mean_value <= camera_black_mean_threshold(rt):
        return False, f"black_frame:mean={mean_value:.3f}:max={max_value:.0f}"
    return True, "ok"


def camera_render_product_paths(rt):
    render_products = rt.STATE.get("dataset_camera_render_products")
    if isinstance(render_products, dict) and render_products:
        paths = []
        seen = set()
        for name in rt.DATASET_CAMERA_NAMES:
            path = render_product_path(render_products.get(str(name)))
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths
    objects = rt.STATE.get("dataset_camera_objects")
    if not isinstance(objects, dict):
        return []
    paths = []
    seen = set()
    for name in rt.DATASET_CAMERA_NAMES:
        cam = objects.get(str(name))
        if cam is None:
            continue
        path = ""
        try:
            getter = getattr(cam, "get_render_product_path", None)
            if callable(getter):
                path = str(getter() or "")
            if not path:
                path = str(getattr(cam, "_render_product_path", "") or "")
        except Exception:
            path = ""
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def render_product_path(render_product):
    if render_product is None:
        return ""
    try:
        return str(render_product.path)
    except Exception:
        pass
    try:
        return str(render_product.get_output_prims()["renderProduct"][0])
    except Exception:
        pass
    return str(render_product)


def simulation_render_product_path(render_product):
    """Path used by SyntheticData NEW_FRAME events for a render product."""
    if render_product is None:
        return ""
    try:
        hydra_texture = getattr(render_product, "hydra_texture", None)
        getter = getattr(hydra_texture, "get_render_product_path", None)
        if callable(getter):
            path = str(getter() or "")
            if path and not path.startswith("/"):
                path = "/Render/RenderProduct_" + path
            if path:
                return path
    except Exception:
        pass
    path = render_product_path(render_product)
    if path.startswith("/Render/OmniverseKit/HydraTextures/"):
        name = path.rsplit("/", 1)[-1]
        if name:
            return "/Render/RenderProduct_" + name
    return path


def camera_sim_render_product_paths(rt):
    render_products = rt.STATE.get("dataset_camera_render_products")
    if not isinstance(render_products, dict):
        return []
    paths = []
    seen = set()
    for name in rt.DATASET_CAMERA_NAMES:
        path = simulation_render_product_path(render_products.get(str(name)))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def set_render_products_updates_enabled(rt, enabled=True):
    render_products = rt.STATE.get("dataset_camera_render_products")
    if not isinstance(render_products, dict):
        return 0
    count = 0
    for render_product in list(render_products.values()):
        try:
            hydra_texture = getattr(render_product, "hydra_texture", None)
            setter = getattr(hydra_texture, "set_updates_enabled", None)
            if callable(setter):
                setter(bool(enabled))
                count += 1
        except Exception:
            continue
    rt.STATE["dataset_camera_render_product_updates_enabled"] = bool(enabled)
    rt.STATE["dataset_camera_render_product_updates_enabled_count"] = int(count)
    return count


def get_rgb_annotator():
    if rep is None:
        raise RuntimeError("replicator_unavailable")
    registry = getattr(rep, "AnnotatorRegistry", None)
    if registry is not None and callable(getattr(registry, "get_annotator", None)):
        return registry.get_annotator("rgb")
    annotators = getattr(rep, "annotators", None)
    getter = getattr(annotators, "get", None)
    if callable(getter):
        return getter("rgb")
    raise RuntimeError("rgb_annotator_registry_unavailable")


def attach_annotator_to_render_product(annotator, render_product):
    path = render_product_path(render_product)
    attempts = []
    if path:
        attempts.append([path])
    attempts.append([render_product])
    attempts.append(render_product)
    last_error = None
    for target in attempts:
        try:
            annotator.attach(target)
            return True, "ok"
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        return False, f"{type(last_error).__name__}:{last_error}"
    return False, "attach_target_missing"


def disable_capture_on_play(rt):
    if rep is None:
        return {"ok": False, "reason": "replicator_unavailable", "captureOnPlay": None}
    try:
        rep.orchestrator.set_capture_on_play(False)
        return {"ok": True, "reason": "set_false_ok", "captureOnPlay": False}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}:{exc}", "captureOnPlay": None}


def camera_fingerprint(rt, status=None):
    status = status if isinstance(status, dict) else rt.STATE.get("dataset_camera_last_status", {})
    if not isinstance(status, dict):
        status = {}
    views = status.get("views", {}) if isinstance(status.get("views", {}), dict) else {}
    result = {
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "backend": backend(rt),
        "captureOnPlay": (rt.STATE.get("dataset_camera_capture_on_play_status") or {}).get("captureOnPlay"),
        "captureOnPlay_status": rt.STATE.get("dataset_camera_capture_on_play_status"),
        "replicator_available": bool(rep is not None),
        "wait_for_render": bool(camera_wait_for_render_enabled(rt)),
        "delta_time": camera_delta_time(rt),
        "tick_timeout_s": float(camera_tick_timeout_seconds(rt)),
        "rt_subframes": int(camera_rt_subframes(rt)),
        "replicator_step": disabled_step_context(rt),
        "render_tick": dict(rt.STATE.get("dataset_camera_render_tick_status", {}) or {}),
        "required_cameras": [str(x) for x in rt.DATASET_CAMERA_NAMES],
        "views": {},
    }
    for name in rt.DATASET_CAMERA_NAMES:
        name = str(name)
        view = views.get(name, {}) if isinstance(views.get(name, {}), dict) else {}
        render_product_obj = None
        render_products = rt.STATE.get("dataset_camera_render_products")
        if isinstance(render_products, dict):
            render_product_obj = render_products.get(name)
        result["views"][name] = {
            "prim": str(view.get("path") or view.get("prim_path") or ""),
            "valid": bool(view.get("available", False)),
            "render_product": str(view.get("render_product", "")),
            "sim_render_product": simulation_render_product_path(render_product_obj),
            "annotator": str(view.get("rgb_annotator", "")),
            "reason": str(view.get("reason", "")),
        }
    return result


def log_camera_fingerprint(rt, status=None, force=False):
    fp = camera_fingerprint(rt, status=status)
    key = repr(fp)
    if (not force) and rt.STATE.get("dataset_camera_last_fingerprint_key") == key:
        return fp
    rt.STATE["dataset_camera_last_fingerprint_key"] = key
    pieces = [
        "[DATASET CAMERA FINGERPRINT]",
        f"module={fp.get('module_version')}",
        f"backend={fp.get('backend')}",
        f"captureOnPlay={fp.get('captureOnPlay')}",
        f"captureOnPlay_status={(fp.get('captureOnPlay_status') or {}).get('reason')}",
        f"replicator={fp.get('replicator_available')}",
        f"wait_for_render={fp.get('wait_for_render')}",
        f"rt_subframes={fp.get('rt_subframes')}",
        f"tick_timeout_s={fp.get('tick_timeout_s'):.2f}",
    ]
    for name in rt.DATASET_CAMERA_NAMES:
        view = (fp.get("views") or {}).get(str(name), {})
        pieces.append(
            f"{name}:prim={view.get('prim')} valid={view.get('valid')} "
            f"rp={view.get('render_product')} annotator={view.get('annotator')} reason={view.get('reason')}"
        )
    rt.info_print(*pieces)
    return fp


def format_rgb_stats(views):
    chunks = []
    views = views if isinstance(views, dict) else {}
    for name in sorted(str(x) for x in views.keys()):
        view = views.get(name, {}) if isinstance(views.get(name, {}), dict) else {}
        stats = view.get("rgb_stats", {}) if isinstance(view.get("rgb_stats", {}), dict) else {}
        shape = stats.get("shape") or view.get("shape")
        mean_value = float(stats.get("mean", 0.0) or 0.0)
        max_value = float(stats.get("max", 0.0) or 0.0)
        dtype = view.get("dtype") or stats.get("source_dtype", "")
        chunks.append(f"{name}:shape={shape} dtype={dtype} mean={mean_value:.3f} max={max_value:.0f}")
    return " ".join(chunks)


def read_rgb_frames(rt):
    annotators = rt.STATE.get("dataset_camera_rgb_annotators")
    if not isinstance(annotators, dict):
        return None, {"ok": False, "reason": "annotators_missing", "views": {}}
    frames = {}
    views = {}
    failures = []
    for name in rt.DATASET_CAMERA_NAMES:
        name = str(name)
        annotator = annotators.get(name)
        view_payload = {"available": False, "name": name, "reason": ""}
        if annotator is None:
            reason = "annotator_missing"
            view_payload["reason"] = reason
            views[name] = view_payload
            failures.append(f"{name}:{reason}")
            continue
        try:
            raw = annotator.get_data()
            rgb, reason, stats = coerce_rgb_uint8(raw)
            view_payload["rgb_stats"] = stats
            if rgb is None:
                view_payload["reason"] = reason
                views[name] = view_payload
                failures.append(f"{name}:{reason}")
                continue
            valid, valid_reason = validate_rgb_content(rt, rgb, stats)
            if not valid:
                if str(valid_reason).startswith("black_frame"):
                    rt.STATE["dataset_camera_black_rejected"] = int(rt.STATE.get("dataset_camera_black_rejected", 0) or 0) + 1
                view_payload["reason"] = valid_reason
                views[name] = view_payload
                failures.append(f"{name}:{valid_reason}")
                continue
            view_payload.update(
                {
                    "available": True,
                    "reason": "ok",
                    "shape": [int(x) for x in rgb.shape],
                    "dtype": str(rgb.dtype),
                    "capture_backend": backend(rt),
                }
            )
            frames[name] = np.ascontiguousarray(rgb[:, :, :3])
            views[name] = view_payload
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            view_payload["reason"] = reason
            views[name] = view_payload
            failures.append(f"{name}:{reason}")
    if failures:
        return None, {"ok": False, "reason": ",".join(failures), "views": views}
    return frames, {"ok": True, "reason": "ok", "views": views}


def resolve_syntheticdata_sensors():
    errors = []
    try:
        import omni.syntheticdata as syntheticdata

        sensors = getattr(syntheticdata, "sensors", None)
        if sensors is not None and callable(getattr(sensors, "next_render_simulation_async", None)):
            return sensors, "omni.syntheticdata.sensors_attr"
        errors.append("omni.syntheticdata.sensors_attr:missing_next_render_simulation_async")
    except Exception as exc:
        errors.append(f"omni.syntheticdata:{type(exc).__name__}:{exc}")
    try:
        from omni.syntheticdata import sensors

        if callable(getattr(sensors, "next_render_simulation_async", None)):
            return sensors, "from_omni.syntheticdata_import_sensors"
        errors.append("from_omni.syntheticdata_import_sensors:missing_next_render_simulation_async")
    except Exception as exc:
        errors.append(f"from_omni.syntheticdata_import_sensors:{type(exc).__name__}:{exc}")
    try:
        import omni.syntheticdata.sensors as sensors

        if callable(getattr(sensors, "next_render_simulation_async", None)):
            return sensors, "omni.syntheticdata.sensors_module"
        errors.append("omni.syntheticdata.sensors_module:missing_next_render_simulation_async")
    except Exception as exc:
        errors.append(f"omni.syntheticdata.sensors_module:{type(exc).__name__}:{exc}")
    return None, "syntheticdata_unavailable:" + " | ".join(errors)


async def tick_camera_render_products_async(rt, frames=0):
    paths = camera_sim_render_product_paths(rt)
    hydra_paths = camera_render_product_paths(rt)
    if not paths:
        rt.STATE["dataset_camera_render_tick_status"] = {
            "ok": False,
            "reason": "no_sim_render_products",
            "paths": [],
            "hydra_paths": hydra_paths,
        }
        return False, "no_render_products"
    sd_sensors, source = resolve_syntheticdata_sensors()
    if sd_sensors is None:
        reason = source
        rt.STATE["dataset_camera_render_tick_status"] = {
            "ok": False,
            "reason": reason,
            "paths": paths,
            "hydra_paths": hydra_paths,
        }
        return False, reason
    failures = []
    frame_offset = max(0, int(frames))
    for path in paths:
        try:
            await sd_sensors.next_render_simulation_async(path, frame_offset)
        except Exception as exc:
            failures.append(f"{path}:{type(exc).__name__}:{exc}")
    if failures:
        reason = ";".join(failures)
        rt.STATE["dataset_camera_render_tick_status"] = {
            "ok": False,
            "reason": reason,
            "paths": paths,
            "hydra_paths": hydra_paths,
            "frame_offset": int(frame_offset),
        }
        now = time.time()
        if now - float(rt.STATE.get("dataset_camera_last_render_tick_error_time", 0.0)) > 2.0:
            rt.STATE["dataset_camera_last_render_tick_error_time"] = now
            rt.info_print("[WARN] dataset camera render tick failed:", reason)
        return False, reason
    reason = f"rendered:{len(paths)}"
    rt.STATE["dataset_camera_render_tick_status"] = {
        "ok": True,
        "reason": reason,
        "paths": paths,
        "hydra_paths": hydra_paths,
        "source": source,
        "frame_offset": int(frame_offset),
    }
    return True, reason


def _replicator_step_async(rt, rt_subframes=1, wait_for_render=None, delta_time=None):
    if rep is None:
        return None
    step_async = getattr(rep.orchestrator, "step_async", None)
    if not callable(step_async):
        return None
    if wait_for_render is None:
        wait_for_render = camera_wait_for_render_enabled(rt)
    if delta_time is None:
        delta_time = camera_delta_time(rt)
    kwargs = {
        "rt_subframes": int(rt_subframes),
        "delta_time": None if delta_time is None else float(delta_time),
        "pause_timeline": False,
        "wait_for_render": bool(wait_for_render),
    }
    try:
        return step_async(**kwargs)
    except TypeError:
        kwargs.pop("wait_for_render", None)
        return step_async(**kwargs)


async def global_tick_async(rt):
    if not replicator_tick_enabled(rt):
        await rt.step_updates(1)
        rt.STATE["dataset_camera_render_tick_status"] = {
            "ok": True,
            "reason": "kit_update",
            "paths": camera_render_product_paths(rt),
            "source": "kit_update",
        }
        if syntheticdata_wait_enabled(rt):
            await tick_camera_render_products_async(rt, frames=0)
        return True
    if rep is None:
        rt.info_print("[WARN] camera tick failed:", "replicator_unavailable")
        return False
    install_replicator_simtime_guard(rt)
    try:
        rt.STATE["dataset_camera_tick_pending"] = False
        rt.STATE["dataset_camera_capture_on_play_status"] = disable_capture_on_play(rt)
        wait_for_render = camera_wait_for_render_enabled(rt)
        rt_subframes = camera_rt_subframes(rt)
        delta_time = camera_delta_time(rt)
        updates_enabled_count = set_render_products_updates_enabled(rt, True)
        pending = _replicator_step_async(
            rt,
            rt_subframes=rt_subframes,
            wait_for_render=wait_for_render,
            delta_time=delta_time,
        )
        if inspect.isawaitable(pending):
            try:
                await asyncio.wait_for(pending, timeout=camera_tick_timeout_seconds(rt))
            except asyncio.TimeoutError:
                detail = (
                    f"timeout_s={camera_tick_timeout_seconds(rt):.3f};"
                    f"wait_for_render={bool(wait_for_render)};"
                    f"rt_subframes={int(rt_subframes)};delta_time={delta_time}"
                )
                mark_replicator_step_disabled(rt, "replicator_step_async_timeout", detail)
                rt.STATE["dataset_camera_render_tick_status"] = {
                    "ok": False,
                    "reason": "replicator_step_async_timeout",
                    "paths": camera_render_product_paths(rt),
                    "source": "replicator_step_async",
                    "timeout_s": camera_tick_timeout_seconds(rt),
                    "wait_for_render": bool(wait_for_render),
                    "delta_time": delta_time,
                    "rt_subframes": int(rt_subframes),
                    "updates_enabled_count": int(updates_enabled_count),
                }
                now = time.time()
                if now - float(rt.STATE.get("dataset_camera_last_tick_timeout_log_time", 0.0)) > 2.0:
                    rt.STATE["dataset_camera_last_tick_timeout_log_time"] = now
                    rt.info_print(
                        "[WARN] camera async tick timeout; explicit Replicator RGB capture failed fast",
                        f"timeout={camera_tick_timeout_seconds(rt):.2f}s",
                    )
                return False
            await rt.step_updates(1)
            rt.STATE["dataset_camera_render_tick_status"] = {
                "ok": True,
                "reason": "replicator_step_async",
                "paths": camera_render_product_paths(rt),
                "source": "replicator_step_async",
                "wait_for_render": bool(wait_for_render),
                "delta_time": delta_time,
                "rt_subframes": int(rt_subframes),
                "updates_enabled_count": int(updates_enabled_count),
            }
        elif pending is None:
            mark_replicator_step_disabled(rt, "replicator_step_async_missing", "rep.orchestrator.step_async unavailable")
            rt.STATE["dataset_camera_render_tick_status"] = {
                "ok": False,
                "reason": "replicator_step_async_missing",
                "paths": camera_render_product_paths(rt),
                "source": "replicator_step_async",
            }
            return False
        if syntheticdata_wait_enabled(rt):
            await tick_camera_render_products_async(rt, frames=0)
        return True
    except Exception as e:
        if SYNC_STEP_ERROR_TEXT in str(e):
            mark_replicator_step_disabled(rt, "replicator_step_async_rejected", f"{type(e).__name__}:{e}")
            if not bool(rt.STATE.get("dataset_camera_replicator_step_disable_logged", False)):
                rt.STATE["dataset_camera_replicator_step_disable_logged"] = True
                rt.info_print(
                    "[WARN] camera async tick disabled:",
                    type(e).__name__,
                    "Kit rejected Replicator step_async; RGB frames will not be marked ready until async render succeeds",
                )
            rt.STATE["dataset_camera_render_tick_status"] = {
                "ok": False,
                "reason": "replicator_step_async_rejected",
                "paths": camera_render_product_paths(rt),
                "source": "replicator_step_async",
                "detail": f"{type(e).__name__}:{e}",
            }
            return False
        rt.info_print("[WARN] camera tick failed:", type(e).__name__, e)
        return False


def install_replicator_simtime_guard(rt):
    if bool(rt.STATE.get("replicator_simtime_guard_installed", False)):
        return True
    try:
        import omni.graph.core as og
    except Exception:
        return False
    original_set = getattr(og.AttributeValueHelper, "set", None)
    if not callable(original_set):
        return False
    if bool(getattr(original_set, "_excavator_replicator_guard", False)):
        rt.STATE["replicator_simtime_guard_installed"] = True
        return True

    def guarded_set(self, new_value, *args, **kwargs):
        try:
            return original_set(self, new_value, *args, **kwargs)
        except TypeError as exc:
            text = str(exc)
            if "Unable to write from unknown dtype" not in text:
                raise
            try:
                values = list(new_value)
            except Exception:
                raise
            if all(isinstance(item, tuple) and len(item) == 2 for item in values):
                try:
                    return original_set(self, np.asarray(values, dtype=np.int64).reshape(-1), *args, **kwargs)
                except TypeError:
                    return None
            try:
                arr = np.asarray(values)
                if arr.dtype.kind in ("i", "u"):
                    return original_set(self, np.asarray(values, dtype=np.int64), *args, **kwargs)
                if arr.dtype.kind == "f":
                    return original_set(self, np.asarray(values, dtype=np.float64), *args, **kwargs)
            except Exception:
                pass
            raise

    guarded_set._excavator_replicator_guard = True
    og.AttributeValueHelper.set = guarded_set
    rt.STATE["replicator_simtime_guard_installed"] = True
    return True


def warmup_graph(rt, frames=5):
    rt.STATE["dataset_camera_render_tick_status"] = {
        "ok": False,
        "reason": "warmup_graph_deprecated_use_warmup_for_episode",
        "paths": camera_render_product_paths(rt),
        "source": "sync_noop",
    }


def backend(rt):
    rt.STATE["dataset_camera_backend"] = "replicator_rgb"
    return "replicator_rgb"


def specs(rt):
    swing_parent = rt.LINK_PATHS.get("swing_link") or (f"{rt.ROBOT_BASE}/swing_link" if rt.ROBOT_BASE else "")
    arm_parent = rt.LINK_PATHS.get("arm_link") or (f"{rt.ROBOT_BASE}/arm_link" if rt.ROBOT_BASE else "")
    return [
        {
            "name": "0",
            "meaning": "arm-tip top-down camera",
            "parent": arm_parent,
            "path": f"{arm_parent}/Camera_0" if arm_parent else "",
            "translate": [0.0, 0.0, 0.45],
            "rotate_xyz_deg": [0.0, -60.0, 0.0],
        },
        {
            "name": "1",
            "meaning": "original main camera on swing",
            "parent": swing_parent,
            "path": f"{swing_parent}/Camera_1" if swing_parent else "",
            "translate": [0.0, -1.2, 1.4],
            "rotate_xyz_deg": [65.0, 0.0, 0.0],
        },
        {
            "name": "2",
            "meaning": "swing-mounted overhead panorama camera",
            "parent": swing_parent,
            "path": f"{swing_parent}/Camera_2" if swing_parent else "",
            "translate": [0.0, 0.0, 3.0],
            "rotate_xyz_deg": [0.0, -70.0, 0.0],
        },
    ]


def is_camera_prim(prim):
    if not prim or not prim.IsValid():
        return False
    try:
        return bool(prim.IsA(UsdGeom.Camera))
    except Exception:
        return str(prim.GetTypeName()) == "Camera"


def ensure_prim(rt, stage_obj, spec):
    path = str(spec.get("path", ""))
    parent = str(spec.get("parent", ""))
    if not path or not parent:
        return None, "missing_camera_path_or_parent"
    parent_prim = stage_obj.GetPrimAtPath(parent)
    if not parent_prim or not parent_prim.IsValid():
        return None, f"missing_parent:{parent}"
    prim = stage_obj.GetPrimAtPath(path)
    if not is_camera_prim(prim):
        try:
            camera = UsdGeom.Camera.Define(stage_obj, Sdf.Path(path))
            prim = camera.GetPrim()
            xform = UsdGeom.XformCommonAPI(prim)
            translate = spec.get("translate", [0.0, 0.0, 0.0])
            rotate_xyz = spec.get("rotate_xyz_deg", [0.0, 0.0, 0.0])
            xform.SetTranslate(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
            xform.SetRotate(Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            camera.GetFocalLengthAttr().Set(18.0)
            camera.GetHorizontalApertureAttr().Set(20.955)
            camera.GetVerticalApertureAttr().Set(15.2908)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
            return prim, "created"
        except Exception as exc:
            return None, f"configured_camera_missing:{path}:{type(exc).__name__}:{exc}"
    return prim, "ok"


def prim_metadata(rt, path):
    prim = rt.get_prim(path)
    if not prim or not prim.IsValid():
        return {"available": False, "prim_path": str(path)}
    meta = {"available": True, "prim_path": str(path)}
    try:
        cam = UsdGeom.Camera(prim)
        for key, attr_name in [
            ("focal_length", "focalLength"),
            ("horizontal_aperture", "horizontalAperture"),
            ("vertical_aperture", "verticalAperture"),
        ]:
            attr = cam.GetPrim().GetAttribute(attr_name)
            value = attr.Get() if attr else None
            if value is not None:
                meta[key] = float(value)
        clip_attr = cam.GetPrim().GetAttribute("clippingRange")
        clip = clip_attr.Get() if clip_attr else None
        if clip is not None:
            meta["clipping_range"] = [float(clip[0]), float(clip[1])]
    except Exception as exc:
        meta["metadata_error"] = f"{type(exc).__name__}:{exc}"
    return meta


def world_pose(rt, path):
    prim = rt.get_prim(path)
    if not prim or not prim.IsValid():
        return {"available": False, "prim_path": str(path)}
    try:
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = mat.ExtractTranslation()
        rows = []
        for r in range(4):
            rows.append([float(mat[r][c]) for c in range(4)])
        return {
            "available": True,
            "prim_path": str(path),
            "position": [float(translation[0]), float(translation[1]), float(translation[2])],
            "world_transform": rows,
        }
    except Exception as exc:
        return {"available": False, "prim_path": str(path), "reason": f"{type(exc).__name__}:{exc}"}


def save_rgb_image(rt, path, rgb, ensure_dir=True):
    if ensure_dir:
        rt.ensure_parent_dir(path)
    rgb, reason, _stats = coerce_rgb_uint8(rgb)
    if rgb is None:
        raise ValueError(f"invalid_rgb:{reason}")
    if Image is not None and str(path).lower().endswith(".png"):
        compress_level = int(rt.STATE.get("dataset_camera_png_compress_level", 3) or 3)
        compress_level = max(0, min(9, compress_level))
        Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).save(
            str(path),
            format="PNG",
            optimize=bool(rt.STATE.get("dataset_camera_png_optimize", False)),
            compress_level=compress_level,
        )
        return "png"
    with open(str(path), "wb") as f:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(rgb[:, :, :3]).tobytes())
    return "ppm"


def runtime_ready(rt):
    if not bool(rt.STATE.get("running", False)):
        return False, "runtime_not_running"
    if not rt.simulation_timeline_is_playing():
        return False, "timeline_not_playing"
    try:
        ctx = omni.usd.get_context()
        if ctx is None:
            return False, "usd_context_missing"
        stage_obj = ctx.get_stage()
        if stage_obj is None:
            return False, "stage_missing"
        if hasattr(ctx, "get_stage_id"):
            stage_id = ctx.get_stage_id()
            try:
                if int(stage_id) < 0:
                    return False, f"stage_id_not_ready:{stage_id}"
            except Exception:
                pass
    except Exception as exc:
        return False, f"stage_not_ready:{type(exc).__name__}:{exc}"
    return True, "ok"


def shutdown(rt, reason="shutdown"):
    closed = 0
    annotators = rt.STATE.get("dataset_camera_rgb_annotators")
    if isinstance(annotators, dict):
        for _name, annotator in list(annotators.items()):
            method = getattr(annotator, "detach", None)
            if callable(method):
                try:
                    method()
                    closed += 1
                except Exception:
                    pass
    render_products = rt.STATE.get("dataset_camera_render_products")
    if isinstance(render_products, dict):
        for _name, render_product in list(render_products.items()):
            method = getattr(render_product, "destroy", None)
            if callable(method):
                try:
                    method()
                    closed += 1
                except Exception:
                    pass
    objects = rt.STATE.get("dataset_camera_objects")
    if isinstance(objects, dict):
        for _name, cam in list(objects.items()):
            for method_name in ("destroy", "cleanup", "stop", "pause"):
                method = getattr(cam, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    closed += 1
                    break
                except Exception:
                    continue
    rt.STATE["dataset_camera_objects"] = {}
    rt.STATE["dataset_camera_render_products"] = {}
    rt.STATE["dataset_camera_rgb_annotators"] = {}
    rt.STATE["dataset_camera_initialized"] = False
    rt.STATE["dataset_camera_init_attempted"] = False
    rt.STATE["dataset_camera_last_status"] = {"enabled": bool(rt.STATE.get("dataset_camera_enabled", True)), "available": False, "reason": reason, "closed": closed}
    return closed


def initialize(rt, force=False):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        rt.STATE["dataset_camera_last_status"] = {"enabled": False, "reason": "disabled"}
        return False
    current_backend = backend(rt)
    if rep is None:
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": "replicator_unavailable", "backend": current_backend}
        return False
    ready, ready_reason = runtime_ready(rt)
    if not ready:
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": ready_reason, "backend": current_backend}
        return False
    if bool(rt.STATE.get("dataset_camera_initialized", False)) and not force:
        annotators = rt.STATE.get("dataset_camera_rgb_annotators")
        if isinstance(annotators, dict) and all(str(name) in annotators for name in rt.DATASET_CAMERA_NAMES):
            return True

    stage_obj = omni.usd.get_context().get_stage()
    cam_resolution = resolution(rt)
    render_products = {}
    annotators = {}
    install_replicator_simtime_guard(rt)
    capture_status = disable_capture_on_play(rt)
    rt.STATE["dataset_camera_capture_on_play_status"] = capture_status
    status = {
        "enabled": True,
        "schema": rt.DATASET_CAMERA_SCHEMA,
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "backend": current_backend,
        "captureOnPlay": capture_status.get("captureOnPlay"),
        "captureOnPlay_status": capture_status,
        "replicator_tick_enabled": bool(replicator_tick_enabled(rt)),
        "wait_for_render": bool(camera_wait_for_render_enabled(rt)),
        "delta_time": camera_delta_time(rt),
        "rt_subframes": int(camera_rt_subframes(rt)),
        "tick_timeout_s": float(camera_tick_timeout_seconds(rt)),
        "resolution": cam_resolution,
        "image_format": image_extension(rt),
        "views": {},
    }
    for spec in specs(rt):
        name = str(spec.get("name", ""))
        path = str(spec.get("path", ""))
        view_status = dict(spec)
        view_status.pop("translate", None)
        view_status.pop("rotate_xyz_deg", None)
        try:
            prim, reason = ensure_prim(rt, stage_obj, spec)
            if prim is None:
                view_status.update({"available": False, "reason": reason})
                status["views"][name] = view_status
                continue
            if not is_camera_prim(prim):
                view_status.update({"available": False, "reason": f"not_camera_prim:{prim.GetTypeName()}", "prim_path": path})
                status["views"][name] = view_status
                continue
            try:
                render_product = rep.create.render_product(
                    path,
                    resolution=(int(cam_resolution[0]), int(cam_resolution[1])),
                    name=f"excavator_dataset_rp_{name}",
                )
            except TypeError:
                render_product = rep.create.render_product(
                    path,
                    resolution=(int(cam_resolution[0]), int(cam_resolution[1])),
                )
            try:
                hydra_texture = getattr(render_product, "hydra_texture", None)
                setter = getattr(hydra_texture, "set_updates_enabled", None)
                if callable(setter):
                    setter(True)
            except Exception:
                pass
            rgb_annotator = get_rgb_annotator()
            attached, attach_reason = attach_annotator_to_render_product(rgb_annotator, render_product)
            if not attached:
                raise RuntimeError(f"rgb_annotator_attach_failed:{attach_reason}")
            render_products[name] = render_product
            annotators[name] = rgb_annotator
            view_status.update(prim_metadata(rt, path))
            view_status.update(
                {
                    "available": True,
                    "reason": "ok",
                    "render_product": render_product_path(render_product),
                    "sim_render_product": simulation_render_product_path(render_product),
                    "rgb_annotator": "explicit",
                    "rgb_annotator_attach": attach_reason,
                    "rgb_annotator_ok": True,
                }
            )
        except Exception as exc:
            view_status.update({"available": False, "reason": f"{type(exc).__name__}:{exc}", "prim_path": path})
        status["views"][name] = view_status

    missing = [str(name) for name in rt.DATASET_CAMERA_NAMES if str(name) not in annotators]
    status["available"] = not bool(missing)
    if missing:
        status["reason"] = "missing_required_cameras:" + ",".join(missing)
    rt.STATE["dataset_camera_objects"] = {}
    rt.STATE["dataset_camera_render_products"] = render_products
    rt.STATE["dataset_camera_rgb_annotators"] = annotators
    status["render_product_updates_enabled_count"] = int(set_render_products_updates_enabled(rt, True))
    rt.STATE["dataset_camera_initialized"] = not bool(missing)
    rt.STATE["dataset_camera_init_attempted"] = True
    rt.STATE["dataset_camera_last_status"] = status
    log_camera_fingerprint(rt, status=status, force=force)
    if not missing:
        rt.info_print(
            "[DATASET CAMERA]",
            f"module={CAMERA_MODULE_VERSION}",
            "backend=replicator_rgb",
            f"replicator_tick={replicator_tick_enabled(rt)}",
            f"views={list(annotators.keys())}",
            f"resolution={cam_resolution}",
            f"format={image_extension(rt)}",
        )
    else:
        rt.info_print("[WARN] [DATASET CAMERA] missing required camera views", status)
    return not bool(missing)


def episode_metadata(rt):
    status = rt.STATE.get("dataset_camera_last_status")
    if not isinstance(status, dict) or not status:
        status = {
            "enabled": bool(rt.STATE.get("dataset_camera_enabled", True)),
            "schema": rt.DATASET_CAMERA_SCHEMA,
            "resolution": resolution(rt),
            "image_format": image_extension(rt),
            "views": {},
        }
        for spec in specs(rt):
            name = str(spec.get("name", ""))
            path = str(spec.get("path", ""))
            view = dict(spec)
            view.update(prim_metadata(rt, path))
            status["views"][name] = view
    return status


def episode_summary(rt, meta=None):
    meta = meta if isinstance(meta, dict) else {}
    episode_dir = str(rt.STATE.get("dataset_episode_dir", "") or "")
    extension = image_extension(rt)
    image_counts = {}
    for name in rt.DATASET_CAMERA_NAMES:
        name = str(name)
        count = 0
        image_dir = os.path.join(episode_dir, "images", name) if episode_dir else ""
        try:
            if image_dir and os.path.isdir(image_dir):
                suffix = "." + extension.lower().lstrip(".")
                count = sum(1 for item in os.listdir(image_dir) if str(item).lower().endswith(suffix))
        except Exception:
            count = 0
        image_counts[name] = int(count)
    metrics = meta.get("final_metrics", {}) if isinstance(meta.get("final_metrics", {}), dict) else {}
    samples = metrics.get("samples")
    if samples is None:
        samples = int(rt.STATE.get("dataset_samples", 0) or 0) - int(rt.STATE.get("dataset_episode_sample_start", 0) or 0)
    summary = {
        "module_version": CAMERA_MODULE_VERSION,
        "backend": backend(rt),
        "samples": int(max(0, samples or 0)),
        "image_counts": image_counts,
        "dropped_incomplete": int(rt.STATE.get("dataset_camera_dropped_incomplete_samples", 0) or 0),
        "black_rejected": int(rt.STATE.get("dataset_camera_black_rejected", 0) or 0),
        "require_complete_samples": bool(rt.STATE.get("dataset_camera_require_complete_samples", True)),
        "warmup": dict(rt.STATE.get("dataset_camera_warmup_status", {}) or {}),
        "fingerprint": camera_fingerprint(rt),
    }
    counts_match = all(int(image_counts.get(str(name), 0) or 0) == int(summary["samples"]) for name in rt.DATASET_CAMERA_NAMES)
    summary["counts_match_samples"] = bool(counts_match)
    return summary


def log_episode_summary(rt, meta=None):
    summary = episode_summary(rt, meta=meta)
    counts = summary.get("image_counts", {})
    rt.info_print(
        "[DATASET CAMERA EPISODE SUMMARY]",
        f"backend={summary.get('backend')}",
        f"samples={summary.get('samples')}",
        f"image_counts.0={counts.get('0', 0)}",
        f"image_counts.1={counts.get('1', 0)}",
        f"image_counts.2={counts.get('2', 0)}",
        f"dropped_incomplete={summary.get('dropped_incomplete')}",
        f"black_rejected={summary.get('black_rejected')}",
        f"counts_match_samples={summary.get('counts_match_samples')}",
    )
    return summary


def episode_cache(rt, reset=False):
    episode_dir = str(rt.STATE.get("dataset_episode_dir", "") or "")
    image_dir = str(rt.STATE.get("dataset_image_dir", "") or "")
    extension = image_extension(rt)
    cam_resolution = resolution(rt)
    key = (
        episode_dir,
        image_dir,
        extension,
        tuple(int(x) for x in cam_resolution),
        tuple(str(x) for x in rt.DATASET_CAMERA_NAMES),
    )
    cache = rt.STATE.get("dataset_camera_episode_cache")
    if (
        not reset
        and isinstance(cache, dict)
        and cache.get("key") == key
        and isinstance(cache.get("views"), dict)
    ):
        return cache

    camera_specs = specs(rt)
    spec_by_name = {str(item.get("name", "")): item for item in camera_specs}
    views = {}
    if image_dir:
        for name in rt.DATASET_CAMERA_NAMES:
            name = str(name)
            abs_dir = os.path.join(image_dir, name)
            try:
                os.makedirs(abs_dir, exist_ok=True)
            except Exception:
                pass
            rel_dir = ""
            if episode_dir:
                try:
                    rel_dir = os.path.relpath(abs_dir, episode_dir).replace(os.sep, "/")
                except Exception:
                    rel_dir = f"images/{name}"
            else:
                rel_dir = f"images/{name}"
            spec = spec_by_name.get(name, {})
            views[name] = {
                "name": name,
                "spec": spec,
                "prim_path": str(spec.get("path", "")),
                "abs_dir": abs_dir,
                "rel_dir": rel_dir,
            }
    cache = {
        "key": key,
        "episode_dir": episode_dir,
        "image_dir": image_dir,
        "extension": extension,
        "resolution": cam_resolution,
        "views": views,
    }
    rt.STATE["dataset_camera_episode_cache"] = cache
    return cache


def sample_requires_complete_images(rt, sample_index):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        return False
    if not bool(rt.STATE.get("dataset_camera_require_complete_samples", True)):
        return False
    stride = max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1))
    return int(sample_index) % stride == 0


def payload_complete(rt, payload, sample_index):
    if not sample_requires_complete_images(rt, sample_index):
        return True, "not_required"
    if not isinstance(payload, dict):
        return False, "payload_missing"
    missing = []
    for name in rt.DATASET_CAMERA_NAMES:
        key = f"observation.images.{name}"
        view = {}
        camera_info = payload.get("observation.camera", {})
        if isinstance(camera_info, dict):
            views = camera_info.get("views", {})
            if isinstance(views, dict):
                view = views.get(str(name), {}) if isinstance(views.get(str(name), {}), dict) else {}
        if not payload.get(key) or not bool(view.get("available", False)):
            missing.append(str(name))
    if missing:
        camera_info = payload.get("observation.camera", {})
        reason = ""
        if isinstance(camera_info, dict):
            reason = str(camera_info.get("reason", "") or "")
            if not reason:
                view_reasons = []
                views = camera_info.get("views", {})
                if isinstance(views, dict):
                    for name in missing:
                        view = views.get(str(name), {})
                        if isinstance(view, dict):
                            view_reasons.append(f"{name}:{view.get('reason', 'missing')}")
                reason = ",".join(view_reasons)
        return False, reason or f"missing_camera_images:{','.join(missing)}"
    return True, "ok"


def rgb_ready(rt):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        return True, "disabled"
    if not rt.simulation_timeline_is_playing():
        return False, "timeline_not_playing"
    if not initialize(rt, force=False):
        status = rt.STATE.get("dataset_camera_last_status", {})
        reason = status.get("reason", "camera_unavailable") if isinstance(status, dict) else "camera_unavailable"
        return False, str(reason)
    frames, meta = read_rgb_frames(rt)
    if frames is None:
        return False, str(meta.get("reason", "rgb_invalid") if isinstance(meta, dict) else "rgb_invalid")
    return True, "ok"


async def warmup_for_episode(rt, label="episode"):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        rt.STATE["dataset_camera_warmup_status"] = {"ok": True, "reason": "disabled", "label": str(label)}
        return True
    if not bool(rt.STATE.get("dataset_camera_require_complete_samples", True)):
        rt.STATE["dataset_camera_warmup_status"] = {"ok": True, "reason": "complete_samples_not_required", "label": str(label)}
        return True
    reset_replicator_tick_state(rt, reason=f"warmup_start:{label}")
    max_frames = max(0, int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12))
    min_frames = max(0, int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3))
    ready_required = max(1, int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2))
    ready_streak = 0
    last_reason = "not_checked"
    if max_frames <= 0:
        max_frames = max(min_frames, ready_required)
    initialize(rt, force=False)
    for frame in range(max_frames):
        tick_ok = await global_tick_async(rt)
        if not tick_ok:
            status = rt.STATE.get("dataset_camera_render_tick_status", {}) or {}
            last_reason = str(status.get("reason", "replicator_tick_failed"))
            ready = False
            if last_reason in ("replicator_step_async_timeout", "replicator_step_async_missing", "replicator_step_async_rejected"):
                break
        else:
            ready, reason = rgb_ready(rt)
            last_reason = reason
        if ready:
            ready_streak += 1
        else:
            ready_streak = 0
        if frame + 1 >= min_frames and ready_streak >= ready_required:
            _frames, frame_meta = read_rgb_frames(rt)
            view_stats = frame_meta.get("views", {}) if isinstance(frame_meta, dict) else {}
            status = {
                "ok": True,
                "reason": "ok",
                "label": str(label),
                "frames": int(frame + 1),
                "ready_streak": int(ready_streak),
                "render_products": camera_render_product_paths(rt),
                "sim_render_products": camera_sim_render_product_paths(rt),
                "render_tick": dict(rt.STATE.get("dataset_camera_render_tick_status", {}) or {}),
                "replicator_step": disabled_step_context(rt),
                "views": view_stats,
            }
            rt.STATE["dataset_camera_warmup_status"] = status
            rt.info_print(
                "[DATASET CAMERA WARMUP OK]",
                "backend=replicator_rgb",
                f"label={label}",
                f"ok=True",
                f"frames={frame + 1}",
                f"ready_streak={ready_streak}",
                f"render_tick={(status.get('render_tick') or {}).get('reason')}",
                f"replicator_step={(status.get('replicator_step') or {}).get('disable_reason')}",
                f"wait_for_render={camera_wait_for_render_enabled(rt)}",
                format_rgb_stats(view_stats),
            )
            return True
    _frames, frame_meta = read_rgb_frames(rt)
    view_stats = frame_meta.get("views", {}) if isinstance(frame_meta, dict) else {}
    status = {
        "ok": False,
        "reason": str(last_reason),
        "label": str(label),
        "frames": int(max_frames),
        "ready_streak": int(ready_streak),
        "render_products": camera_render_product_paths(rt),
        "sim_render_products": camera_sim_render_product_paths(rt),
        "render_tick": dict(rt.STATE.get("dataset_camera_render_tick_status", {}) or {}),
        "replicator_step": disabled_step_context(rt),
        "views": view_stats,
    }
    rt.STATE["dataset_camera_warmup_status"] = status
    rt.info_print(
        "[WARN] [DATASET CAMERA WARMUP]",
        f"label={label}",
        "ok=False",
        f"frames={max_frames}",
        f"ready_streak={ready_streak}",
        f"reason={last_reason}",
        f"render_tick={(status.get('render_tick') or {}).get('reason')}",
        f"replicator_step={(status.get('replicator_step') or {}).get('disable_reason')}",
        format_rgb_stats(view_stats),
    )
    return False


def capture_observations(rt, sample_index):
    payload = {
        "observation.images.0": None,
        "observation.images.1": None,
        "observation.images.2": None,
        "observation.camera": {
            "schema": rt.DATASET_CAMERA_SCHEMA,
            "available": False,
            "frame_index": int(sample_index),
            "views": {},
        },
    }
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        payload["observation.camera"]["reason"] = "disabled"
        return payload
    stride = max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1))
    if int(sample_index) % stride != 0:
        payload["observation.camera"]["reason"] = "stride_skipped"
        return payload
    current_backend = backend(rt)
    if not initialize(rt, force=False):
        status = rt.STATE.get("dataset_camera_last_status", {})
        payload["observation.camera"]["reason"] = status.get("reason", "camera_unavailable") if isinstance(status, dict) else "camera_unavailable"
        return payload
    payload["observation.camera"]["backend"] = current_backend

    episode_dir = str(rt.STATE.get("dataset_episode_dir", "") or "")
    image_dir = str(rt.STATE.get("dataset_image_dir", "") or "")
    if not episode_dir or not image_dir:
        payload["observation.camera"]["reason"] = "missing_episode_image_dir"
        return payload

    camera_cache = episode_cache(rt, reset=False)
    extension = str(camera_cache.get("extension", image_extension(rt)))
    view_cache = camera_cache.get("views", {}) if isinstance(camera_cache, dict) else {}

    frames, frame_meta = read_rgb_frames(rt)
    if frames is None:
        payload["observation.camera"]["reason"] = str(frame_meta.get("reason", "rgb_invalid") if isinstance(frame_meta, dict) else "rgb_invalid")
        payload["observation.camera"]["views"] = frame_meta.get("views", {}) if isinstance(frame_meta, dict) else {}
        return payload
    capture_backend = str(frame_meta.get("backend", current_backend) if isinstance(frame_meta, dict) else current_backend)
    payload["observation.camera"]["backend"] = capture_backend

    for name in rt.DATASET_CAMERA_NAMES:
        name = str(name)
        rgb = frames.get(name)
        meta_views = frame_meta.get("views", {}) if isinstance(frame_meta, dict) else {}
        view_payload = dict(meta_views.get(name, {}) if isinstance(meta_views.get(name, {}), dict) else {})
        view_payload.setdefault("available", False)
        view_payload.setdefault("name", name)
        view_payload.setdefault("path", None)
        view_payload.setdefault("shape", None)
        view_payload.setdefault("dtype", None)
        cached_view = view_cache.get(str(name), {}) if isinstance(view_cache, dict) else {}
        prim_path = str(cached_view.get("prim_path", ""))
        view_payload["prim_path"] = prim_path
        view_payload["pose"] = world_pose(rt, prim_path)
        try:
            if rgb is None:
                view_payload["reason"] = "rgb_frame_missing_after_validation"
                payload["observation.camera"]["views"][name] = view_payload
                continue
            filename = f"{int(sample_index):06d}.{extension}"
            abs_dir = str(cached_view.get("abs_dir", os.path.join(image_dir, name)))
            rel_dir = str(cached_view.get("rel_dir", f"images/{name}"))
            abs_path = os.path.join(abs_dir, filename)
            rel_path = f"{rel_dir}/{filename}".replace("\\", "/")
            fmt = extension
            image_job = {
                "kind": "image",
                "path": abs_path,
                "rgb": np.ascontiguousarray(rgb[:, :, :3]).copy(),
                "ensure_dir": False,
            }
            if not rt.dataset_writer_enqueue(image_job):
                fmt = save_rgb_image(rt, abs_path, rgb, ensure_dir=False)
            payload[f"observation.images.{name}"] = rel_path
            view_payload.update(
                {
                    "available": True,
                    "path": rel_path,
                    "shape": [int(x) for x in rgb.shape],
                    "dtype": str(rgb.dtype),
                    "format": fmt,
                    "capture_backend": capture_backend,
                }
            )
        except Exception as exc:
            view_payload["reason"] = f"{type(exc).__name__}:{exc}"
            now = time.time()
            if now - float(rt.STATE.get("dataset_camera_last_error_time", 0.0)) > 2.0:
                rt.STATE["dataset_camera_last_error_time"] = now
                rt.info_print("[WARN] dataset camera capture failed:", name, type(exc).__name__, exc)
        payload["observation.camera"]["views"][name] = view_payload
    payload["observation.camera"]["available"] = all(
        bool(payload["observation.camera"]["views"].get(str(name), {}).get("available", False))
        for name in rt.DATASET_CAMERA_NAMES
    )
    if payload["observation.camera"]["available"]:
        payload["observation.camera"]["reason"] = "ok"
    payload["observation.camera"]["image_format"] = extension
    payload["observation.camera"]["resolution"] = resolution(rt)
    return payload


async def capture_observations_async(rt, sample_index):
    if bool(rt.STATE.get("dataset_camera_enabled", True)):
        stride = max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1))
        should_tick = int(sample_index) % stride == 0
        if should_tick and initialize(rt, force=False):
            tick_ok = await global_tick_async(rt)
            if not tick_ok:
                status = dict(rt.STATE.get("dataset_camera_render_tick_status", {}) or {})
                return {
                    "observation.images.0": None,
                    "observation.images.1": None,
                    "observation.images.2": None,
                    "observation.camera": {
                        "schema": rt.DATASET_CAMERA_SCHEMA,
                        "available": False,
                        "frame_index": int(sample_index),
                        "backend": backend(rt),
                        "reason": str(status.get("reason", "render_tick_failed")),
                        "render_tick": status,
                        "views": {},
                    },
                }
    return capture_observations(rt, sample_index)


def config_snapshot(rt):
    current_backend = backend(rt)
    active_gpu_raw = str(os.environ.get("EXCAVATOR_ACTIVE_GPU", "") or "")
    physics_gpu_raw = str(os.environ.get("EXCAVATOR_PHYSICS_GPU", "") or "")
    multi_gpu_raw = str(os.environ.get("EXCAVATOR_MULTI_GPU", "") or "")
    try:
        active_gpu_value = int(active_gpu_raw) if active_gpu_raw != "" else None
    except Exception:
        active_gpu_value = active_gpu_raw or None
    try:
        physics_gpu_value = int(physics_gpu_raw) if physics_gpu_raw != "" else None
    except Exception:
        physics_gpu_value = physics_gpu_raw or None
    multi_gpu_value = None
    if multi_gpu_raw != "":
        multi_gpu_value = multi_gpu_raw.strip().lower() in ("1", "true", "yes", "on")
    renderer_launch = {
        "graphics_api": str(os.environ.get("EXCAVATOR_GRAPHICS_API", "unknown") or "unknown"),
        "renderer": str(os.environ.get("EXCAVATOR_RENDERER", "unknown") or "unknown"),
        "active_gpu": active_gpu_value,
        "physics_gpu": physics_gpu_value,
        "multi_gpu": multi_gpu_value,
        "notes": "Graphics API is launch-time only; it must be set before SimulationApp creates the renderer.",
    }
    return {
        "schema": rt.DATASET_CAMERA_SCHEMA,
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "enabled": bool(rt.STATE.get("dataset_camera_enabled", True)),
        "available": bool(rep is not None),
        "backend": current_backend,
        "renderer_launch": renderer_launch,
        "captureOnPlay": (rt.STATE.get("dataset_camera_capture_on_play_status") or {}).get("captureOnPlay"),
        "captureOnPlay_status": rt.STATE.get("dataset_camera_capture_on_play_status"),
        "backend_available": {
            "replicator": bool(rep is not None),
            "replicator_tick": bool(replicator_tick_enabled(rt)),
            "syntheticdata_wait": bool(syntheticdata_wait_enabled(rt)),
            "wait_for_render": bool(camera_wait_for_render_enabled(rt)),
            "module_version": CAMERA_MODULE_VERSION,
        },
        "resolution": resolution(rt),
        "frequency": int(rt.STATE.get("dataset_camera_frequency", 10) or 10),
        "sample_stride": max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1)),
        "require_complete_samples": bool(rt.STATE.get("dataset_camera_require_complete_samples", True)),
        "warmup_frames": int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3),
        "warmup_ready_frames": int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2),
        "warmup_max_frames": int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12),
        "replicator_tick": bool(replicator_tick_enabled(rt)),
        "syntheticdata_wait": bool(syntheticdata_wait_enabled(rt)),
        "wait_for_render": bool(camera_wait_for_render_enabled(rt)),
        "delta_time": camera_delta_time(rt),
        "rt_subframes": int(camera_rt_subframes(rt)),
        "tick_timeout_s": camera_tick_timeout_seconds(rt),
        "replicator_step": disabled_step_context(rt),
        "render_tick": dict(rt.STATE.get("dataset_camera_render_tick_status", {}) or {}),
        "black_frame_guard": {
            "mean_threshold": camera_black_mean_threshold(rt),
            "max_threshold": camera_black_max_threshold(rt),
            "note": "Frames at or below both thresholds are rejected and are not written as valid observations.",
        },
        "fingerprint": camera_fingerprint(rt),
        "requested_image_format": str(rt.STATE.get("dataset_camera_image_format", "ppm") or "ppm"),
        "image_format": image_extension(rt),
        "image_format_note": "Default PPM keeps RGB frame content uncompressed during collection; LeRobot export converts images/videos after the run.",
        "png_compression": {
            "compress_level": int(rt.STATE.get("dataset_camera_png_compress_level", 3) or 3),
            "optimize": bool(rt.STATE.get("dataset_camera_png_optimize", False)),
            "lossless": True,
        },
        "async_writer": {
            "enabled": bool(rt.STATE.get("dataset_async_writer_enabled", True)),
            "queue_max": int(rt.STATE.get("dataset_async_writer_queue_max", 4096) or 4096),
            "content_unchanged": True,
        },
        "views": specs(rt),
        "pil_available": bool(Image is not None),
    }
