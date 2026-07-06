import asyncio
import ctypes
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


CAMERA_MODULE_VERSION = "dataset_camera_viewport_capture_v23_capture_1024_downsample"
SYNC_STEP_ERROR_TEXT = "Synchronous call to `step`"


PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]


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
    return False


def dataset_viewport_keep_visible(rt):
    value = os.environ.get("EXCAVATOR_DATASET_VIEWPORT_KEEP_VISIBLE", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_viewport_keep_visible", None)
    if state_value is not None:
        return bool(state_value)
    return False


def camera_allow_frame_reuse(rt):
    value = os.environ.get("EXCAVATOR_DATASET_CAMERA_ALLOW_REUSE", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_allow_reuse", None)
    if state_value is not None:
        return bool(state_value)
    return False


def set_dataset_viewports_visible(rt, visible):
    holder = rt.STATE.get("dataset_viewport_capture")
    viewports = holder.get("viewports") if isinstance(holder, dict) else None
    if not isinstance(viewports, dict):
        return 0
    count = 0
    for entry in viewports.values():
        if not isinstance(entry, dict):
            continue
        window = entry.get("window")
        if window is None:
            continue
        try:
            window.visible = bool(visible)
            count += 1
        except Exception:
            pass
    rt.STATE["dataset_camera_viewport_visible"] = bool(visible)
    rt.STATE["dataset_camera_viewport_visible_count"] = int(count)
    return count


def set_dataset_viewports_capture_active(rt, active):
    visible = bool(active) or dataset_viewport_keep_visible(rt)
    return set_dataset_viewports_visible(rt, visible)


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
        value = rt.STATE.get("dataset_camera_rt_subframes", 1)
    try:
        return max(1, int(value))
    except Exception:
        return 1


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


def viewport_capture_resolution(rt):
    value = os.environ.get("EXCAVATOR_DATASET_CAMERA_CAPTURE_RESOLUTION", "")
    if not value:
        value = rt.STATE.get("dataset_camera_viewport_capture_resolution", [1024, 1024])
    try:
        if isinstance(value, str):
            text = value.strip().lower().replace("x", ",")
            parts = [part for part in text.split(",") if part.strip()]
            if len(parts) == 1:
                width = height = int(float(parts[0]))
            else:
                width = int(float(parts[0]))
                height = int(float(parts[1]))
        else:
            width = int(value[0])
            height = int(value[1])
    except Exception:
        width, height = 1024, 1024
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
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
            image = image.resize((target_w, target_h), resampling)
            return np.asarray(image, dtype=np.uint8), "pil_lanczos_resize"
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


def compact_camera_metadata_value(value, key=""):
    """Keep camera diagnostics JSON-safe without serializing image buffers."""
    if isinstance(value, np.ndarray):
        return {
            "omitted": "ndarray",
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            name = str(k)
            lower = name.lower()
            if lower in ("rgb", "rgba", "image", "images", "pixels", "pixel_data", "buffer"):
                out[name] = compact_camera_metadata_value(v, key=name)
            else:
                out[name] = compact_camera_metadata_value(v, key=name)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 128:
            return {"omitted": "large_sequence", "length": int(len(value))}
        return [compact_camera_metadata_value(v, key=key) for v in value]
    return value


def compact_camera_views(views):
    if not isinstance(views, dict):
        return {}
    return {str(k): compact_camera_metadata_value(v, key=str(k)) for k, v in views.items()}


def compact_capture_meta(meta):
    if not isinstance(meta, dict):
        return {}
    compact = compact_camera_metadata_value(meta)
    if isinstance(compact, dict):
        compact.pop("rgb", None)
    return compact


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


def normalize_rgb_resolution(rt, rgb):
    rgb, reason, stats = coerce_rgb_uint8(rgb)
    if rgb is None:
        return None, reason, stats
    target_w, target_h = resolution(rt)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if w == int(target_w) and h == int(target_h):
        arr = np.ascontiguousarray(rgb[:, :, :3])
        return arr, "ok", rgb_stats(arr)
    resized, resize_reason = resize_rgb_to_resolution(rgb, [target_w, target_h])
    if resized is None:
        return None, f"rgb_bad_resolution:{w}x{h}:target={target_w}x{target_h}:{resize_reason}", stats
    arr = np.ascontiguousarray(resized[:, :, :3])
    return arr, f"resized:{resize_reason}", rgb_stats(arr)


def capsule_to_numpy_rgb(capsule, buffer_size, width, height):
    name = PyCapsule_GetName(capsule)
    ptr = PyCapsule_GetPointer(capsule, name)
    if ptr is None or int(ptr) == 0:
        raise RuntimeError("viewport_capture_capsule_pointer_null")
    buffer_size = int(buffer_size)
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or buffer_size <= 0:
        raise RuntimeError(f"viewport_capture_bad_buffer:w={width}:h={height}:size={buffer_size}")
    channels = buffer_size // max(1, width * height)
    if channels < 3:
        raise RuntimeError(f"viewport_capture_bad_channels:{channels}")
    array_type = ctypes.c_uint8 * buffer_size
    c_array = array_type.from_address(int(ptr))
    arr = np.ctypeslib.as_array(c_array)
    arr = arr.reshape((height, width, channels)).copy()
    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def active_viewport_camera_path_snapshot():
    try:
        from omni.kit.viewport.utility import get_active_viewport

        active = get_active_viewport()
        if active is None:
            return ""
        return str(getattr(active, "camera_path", "") or "")
    except Exception:
        return ""


def viewport_api_from_window(window):
    viewport_api = getattr(window, "viewport_api", None)
    if viewport_api is None:
        viewport_api = getattr(window, "viewport", None)
    return viewport_api


def set_viewport_camera_path(viewport_api, camera_path):
    try:
        viewport_api.camera_path = Sdf.Path(str(camera_path))
    except Exception:
        viewport_api.camera_path = str(camera_path)


def configure_viewport_capture_resolution(viewport_api, window, width, height):
    width = int(width)
    height = int(height)
    status = {
        "requested": [width, height],
        "resolution_set": False,
        "texture_resolution_set": False,
        "window_size_set": False,
    }
    if window is not None:
        try:
            window.width = width
            window.height = height
            status["window_size_set"] = True
        except Exception as exc:
            status["window_size_error"] = f"{type(exc).__name__}:{exc}"
    try:
        viewport_api.resolution_scale = 1.0
    except Exception as exc:
        status["resolution_scale_error"] = f"{type(exc).__name__}:{exc}"
    try:
        viewport_api.resolution = (width, height)
        status["resolution_set"] = True
    except Exception as exc:
        status["resolution_error"] = f"{type(exc).__name__}:{exc}"
    try:
        setter = getattr(viewport_api, "set_texture_resolution", None)
        if callable(setter):
            setter((width, height))
            status["texture_resolution_set"] = True
    except Exception as exc:
        status["texture_resolution_error"] = f"{type(exc).__name__}:{exc}"
    try:
        getter = getattr(viewport_api, "get_texture_resolution", None)
        if callable(getter):
            tex = getter()
            status["actual_texture_resolution"] = [int(tex[0]), int(tex[1])]
    except Exception as exc:
        status["actual_texture_resolution_error"] = f"{type(exc).__name__}:{exc}"
    try:
        res = viewport_api.resolution
        status["actual_resolution"] = [int(res[0]), int(res[1])]
    except Exception as exc:
        status["actual_resolution_error"] = f"{type(exc).__name__}:{exc}"
    try:
        full_res = viewport_api.full_resolution
        status["full_resolution"] = [int(full_res[0]), int(full_res[1])]
    except Exception:
        pass
    return status


def viewport_holder_info(holder):
    if not isinstance(holder, dict):
        return {}
    viewports = holder.get("viewports")
    if isinstance(viewports, dict):
        return {
            "mode": str(holder.get("mode", "")),
            "count": len(viewports),
            "views": {
                str(name): {
                    "name": str(view.get("name", "")),
                    "camera_path": str(view.get("camera_path", "")),
                    "width": int(view.get("width", 0) or 0),
                    "height": int(view.get("height", 0) or 0),
                    "has_window": view.get("window") is not None,
                    "has_viewport_api": view.get("viewport_api") is not None,
                }
                for name, view in viewports.items()
                if isinstance(view, dict)
            },
        }
    return {
        "mode": str(holder.get("mode", "")),
        "name": str(holder.get("name", "")),
        "width": int(holder.get("width", 0) or 0),
        "height": int(holder.get("height", 0) or 0),
        "has_window": holder.get("window") is not None,
        "has_viewport_api": holder.get("viewport_api") is not None,
    }


def ensure_dataset_viewports(rt):
    holder = rt.STATE.get("dataset_viewport_capture")
    required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
    w, h = viewport_capture_resolution(rt)
    if isinstance(holder, dict):
        viewports = holder.get("viewports")
        if (
            int(holder.get("width", 0) or 0) == int(w)
            and int(holder.get("height", 0) or 0) == int(h)
            and isinstance(viewports, dict)
            and all(
                isinstance(viewports.get(name), dict) and viewports.get(name, {}).get("viewport_api") is not None
                for name in required
            )
        ):
            return viewports
    try:
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:
        raise RuntimeError(f"create_viewport_window_unavailable:{type(exc).__name__}:{exc}")
    viewports = {}
    for spec in specs(rt):
        name = str(spec.get("name", ""))
        if name not in required:
            continue
        camera_path = str(spec.get("path", ""))
        window_name = f"ExcavatorDatasetCaptureViewport_{name}"
        window = create_viewport_window(
            name=window_name,
            width=int(w),
            height=int(h),
        )
        viewport_api = viewport_api_from_window(window)
        if viewport_api is None:
            raise RuntimeError(f"dataset_viewport_api_unavailable:{name}")
        set_viewport_camera_path(viewport_api, camera_path)
        resolution_status = configure_viewport_capture_resolution(viewport_api, window, w, h)
        try:
            window.visible = bool(dataset_viewport_keep_visible(rt))
        except Exception:
            pass
        viewports[name] = {
            "window": window,
            "viewport_api": viewport_api,
            "name": window_name,
            "camera_path": camera_path,
            "width": int(w),
            "height": int(h),
            "resolution_status": resolution_status,
        }
    missing = [name for name in required if name not in viewports]
    if missing:
        raise RuntimeError("dataset_viewport_missing:" + ",".join(missing))
    first = viewports[required[0]]
    rt.STATE["dataset_viewport_capture"] = {
        "mode": "per_camera_persistent",
        "viewports": viewports,
        "window": first.get("window"),
        "viewport_api": first.get("viewport_api"),
        "name": "ExcavatorDatasetCaptureViewport",
        "width": int(w),
        "height": int(h),
    }
    set_dataset_viewports_capture_active(rt, False)
    return viewports


def ensure_dataset_viewport(rt):
    viewports = ensure_dataset_viewports(rt)
    first_name = str(rt.DATASET_CAMERA_NAMES[0])
    first = viewports[first_name]
    return first["viewport_api"], first.get("window")


async def capture_viewport_rgb_async(rt, viewport_api, camera_path, wait_frames=0, timeout_s=2.0, update_camera=True):
    try:
        from omni.kit.viewport.utility import capture_viewport_to_buffer, next_viewport_frame_async
    except Exception as exc:
        return None, f"viewport_capture_api_unavailable:{type(exc).__name__}:{exc}", {}

    result = {
        "done": False,
        "rgb": None,
        "error": "",
        "camera_path": str(camera_path),
    }
    if update_camera:
        set_viewport_camera_path(viewport_api, camera_path)

    for _ in range(max(0, int(wait_frames))):
        try:
            await next_viewport_frame_async(viewport_api)
        except Exception:
            await rt.step_updates(1)

    def on_capture(capsule, buffer_size, width, height, fmt=None):
        try:
            rgb = capsule_to_numpy_rgb(capsule, buffer_size, width, height)
            result["rgb"] = rgb
            result["shape"] = [int(x) for x in rgb.shape]
            result["dtype"] = str(rgb.dtype)
            result["format"] = str(fmt)
            result["width"] = int(width)
            result["height"] = int(height)
            result["buffer_size"] = int(buffer_size)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}:{exc}"
        finally:
            result["done"] = True

    try:
        helper = capture_viewport_to_buffer(viewport_api, on_capture, is_hdr=False)
    except Exception as exc:
        return None, f"capture_viewport_to_buffer_failed:{type(exc).__name__}:{exc}", result

    try:
        if hasattr(helper, "__await__"):
            await asyncio.wait_for(helper, timeout=float(timeout_s))
    except Exception:
        pass

    deadline = time.time() + float(timeout_s)
    while not bool(result["done"]) and time.time() < deadline:
        try:
            await next_viewport_frame_async(viewport_api)
        except Exception:
            await rt.step_updates(1)

    rgb = result.get("rgb")
    if rgb is None:
        return None, result["error"] or "viewport_capture_no_rgb", compact_capture_meta(result)
    return rgb, "ok", compact_capture_meta(result)


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
    viewport_info = viewport_holder_info(rt.STATE.get("dataset_viewport_capture"))
    result = {
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "backend": backend(rt),
        "captureOnPlay": (rt.STATE.get("dataset_camera_capture_on_play_status") or {}).get("captureOnPlay"),
        "captureOnPlay_status": rt.STATE.get("dataset_camera_capture_on_play_status"),
        "replicator_available": bool(rep is not None),
        "viewport_capture_available": True,
        "dataset_viewport": viewport_info,
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
            "capture_backend": str(view.get("capture_backend", "")),
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
    env = os.environ.get("EXCAVATOR_DATASET_CAMERA_BACKEND", "").strip().lower()
    if env in ("viewport_capture", "replicator_rgb"):
        selected = env
    else:
        selected = "viewport_capture"
    rt.STATE["dataset_camera_backend"] = selected
    return selected


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
    holder = rt.STATE.get("dataset_viewport_capture")
    if isinstance(holder, dict):
        windows = []
        viewports = holder.get("viewports")
        if isinstance(viewports, dict):
            for view in viewports.values():
                if isinstance(view, dict) and view.get("window") is not None:
                    windows.append(view.get("window"))
        elif holder.get("window") is not None:
            windows.append(holder.get("window"))
        seen = set()
        for window in windows:
            key = id(window)
            if key in seen:
                continue
            seen.add(key)
            for method_name in ("destroy", "cleanup", "close"):
                method = getattr(window, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    closed += 1
                    break
                except Exception:
                    continue
            try:
                window.visible = False
            except Exception:
                pass
        rt.STATE["dataset_viewport_capture"] = {}
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
    ready, ready_reason = runtime_ready(rt)
    if not ready:
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": ready_reason, "backend": current_backend}
        return False
    if current_backend == "viewport_capture":
        if bool(rt.STATE.get("dataset_camera_initialized", False)) and not force:
            status = rt.STATE.get("dataset_camera_last_status", {})
            if isinstance(status, dict) and status.get("backend") == "viewport_capture":
                return bool(status.get("available", False))
        stage_obj = omni.usd.get_context().get_stage()
        cam_resolution = resolution(rt)
        status = {
            "enabled": True,
            "schema": rt.DATASET_CAMERA_SCHEMA,
            "module_version": CAMERA_MODULE_VERSION,
            "module_file": __file__,
            "backend": current_backend,
            "resolution": cam_resolution,
            "image_format": image_extension(rt),
            "views": {},
        }
        missing = []
        for spec in specs(rt):
            name = str(spec.get("name", ""))
            path = str(spec.get("path", ""))
            view_status = dict(spec)
            view_status.pop("translate", None)
            view_status.pop("rotate_xyz_deg", None)
            try:
                prim, reason = ensure_prim(rt, stage_obj, spec)
                if prim is None or not is_camera_prim(prim):
                    view_status.update({"available": False, "reason": reason, "prim_path": path})
                    missing.append(name)
                else:
                    view_status.update(prim_metadata(rt, path))
                    view_status.update({"available": True, "reason": "ok", "capture_backend": "viewport_capture"})
            except Exception as exc:
                view_status.update({"available": False, "reason": f"{type(exc).__name__}:{exc}", "prim_path": path})
                missing.append(name)
            status["views"][name] = view_status
        try:
            ensure_dataset_viewports(rt)
            status["dataset_viewport"] = {
                "available": True,
                "name": "ExcavatorDatasetCaptureViewport",
                "mode": "per_camera_persistent",
                "count": len(rt.STATE.get("dataset_viewport_capture", {}).get("viewports", {}) or {}),
                "active_viewport_camera": active_viewport_camera_path_snapshot(),
            }
        except Exception as exc:
            status["dataset_viewport"] = {"available": False, "reason": f"{type(exc).__name__}:{exc}"}
            missing = list(set(missing + ["dataset_viewport"]))
        status["available"] = not bool(missing)
        status["reason"] = "ok" if not missing else "missing_required_cameras:" + ",".join(str(x) for x in missing)
        rt.STATE["dataset_camera_objects"] = {}
        rt.STATE["dataset_camera_render_products"] = {}
        rt.STATE["dataset_camera_rgb_annotators"] = {}
        rt.STATE["dataset_camera_initialized"] = not bool(missing)
        rt.STATE["dataset_camera_init_attempted"] = True
        rt.STATE["dataset_camera_last_status"] = status
        if not missing:
            rt.info_print(
                "[DATASET CAMERA]",
                f"module={CAMERA_MODULE_VERSION}",
                "backend=viewport_capture",
                f"views={[str(x) for x in rt.DATASET_CAMERA_NAMES]}",
                f"resolution={cam_resolution}",
                f"format={image_extension(rt)}",
            )
        else:
            rt.info_print("[WARN] [DATASET CAMERA] viewport capture unavailable", status)
        return not bool(missing)
    if rep is None:
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": "replicator_unavailable", "backend": current_backend}
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
        "written_images": {},
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


def empty_payload(rt, sample_index, reason="", views=None):
    return {
        "observation.images.0": None,
        "observation.images.1": None,
        "observation.images.2": None,
        "observation.camera": {
            "schema": rt.DATASET_CAMERA_SCHEMA,
            "available": False,
            "frame_index": int(sample_index),
            "backend": backend(rt),
            "reason": str(reason or ""),
            "views": views if isinstance(views, dict) else {},
        },
    }


def store_latest_capture(rt, frames, views, sample_index, capture_started, capture_finished, source="viewport_capture"):
    required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
    if not isinstance(frames, dict) or any(name not in frames for name in required):
        return False
    seq = int(rt.STATE.get("dataset_camera_latest_seq", 0) or 0) + 1
    safe_frames = {name: np.ascontiguousarray(frames[name][:, :, :3]).copy() for name in required}
    safe_views = compact_camera_views(
        {
            str(name): views.get(str(name), {}) if isinstance(views, dict) else {}
            for name in required
        }
    )
    record = {
        "seq": int(seq),
        "source": str(source),
        "sample_index": int(sample_index),
        "capture_started_at": float(capture_started),
        "capture_finished_at": float(capture_finished),
        "capture_elapsed_ms": float(max(0.0, capture_finished - capture_started) * 1000.0),
        "frames": safe_frames,
        "views": safe_views,
        "backend": backend(rt),
        "resolution": resolution(rt),
        "image_format": image_extension(rt),
    }
    rt.STATE["dataset_camera_latest_seq"] = int(seq)
    rt.STATE["dataset_camera_latest_capture"] = record
    rt.STATE["dataset_camera_latest_capture_time"] = float(capture_finished)
    return True


def latest_capture_payload(rt, sample_index):
    payload = empty_payload(rt, sample_index, reason="")
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        payload["observation.camera"]["reason"] = "disabled"
        return payload
    stride = max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1))
    if int(sample_index) >= 0 and int(sample_index) % stride != 0:
        payload["observation.camera"]["reason"] = "stride_skipped"
        return payload
    episode_dir = str(rt.STATE.get("dataset_episode_dir", "") or "")
    image_dir = str(rt.STATE.get("dataset_image_dir", "") or "")
    if not episode_dir or not image_dir:
        payload["observation.camera"]["reason"] = "missing_episode_image_dir"
        return payload
    latest = rt.STATE.get("dataset_camera_latest_capture")
    if not isinstance(latest, dict):
        payload["observation.camera"]["reason"] = "camera_latest_missing"
        return payload
    frames = latest.get("frames")
    views = latest.get("views")
    if not isinstance(frames, dict) or not isinstance(views, dict):
        payload["observation.camera"]["reason"] = "camera_latest_invalid"
        return payload

    required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
    missing = [name for name in required if name not in frames]
    if missing:
        payload["observation.camera"]["reason"] = "camera_latest_missing_frames:" + ",".join(missing)
        return payload

    now = time.time()
    capture_finished = float(latest.get("capture_finished_at", latest.get("capture_started_at", now)) or now)
    camera_cache = episode_cache(rt, reset=False)
    extension = str(camera_cache.get("extension", image_extension(rt)))
    view_cache = camera_cache.get("views", {}) if isinstance(camera_cache, dict) else {}
    payload["observation.camera"].update(
        {
            "available": True,
            "reason": "ok",
            "backend": str(latest.get("backend", backend(rt))),
            "image_format": extension,
            "resolution": resolution(rt),
            "capture_seq": int(latest.get("seq", 0) or 0),
            "capture_started_at": float(latest.get("capture_started_at", capture_finished) or capture_finished),
            "capture_finished_at": float(capture_finished),
            "capture_elapsed_ms": float(latest.get("capture_elapsed_ms", 0.0) or 0.0),
            "sample_timestamp": float(now),
            "frame_age_ms": float(max(0.0, now - capture_finished) * 1000.0),
            "source": str(latest.get("source", "latest_cache")),
            "capture_backoff": bool(capture_backoff_reason(rt, now=now)),
            "capture_backoff_reason": capture_backoff_reason(rt, now=now),
            "stale_frame": bool(max(0.0, now - capture_finished) > max(1.0, background_interval_seconds(rt) * 3.0)),
            "views": {},
        }
    )

    written_images = camera_cache.setdefault("written_images", {})
    capture_seq = int(latest.get("seq", 0) or 0)
    last_payload_seq = int(rt.STATE.get("dataset_camera_last_payload_capture_seq", 0) or 0)
    allow_reuse = bool(camera_allow_frame_reuse(rt))
    fresh_for_payload = capture_seq > last_payload_seq
    if (not allow_reuse) and not fresh_for_payload:
        payload["observation.camera"].update(
            {
                "available": False,
                "reason": f"camera_no_fresh_frame:capture_seq={capture_seq}:last_used={last_payload_seq}",
                "capture_seq": int(capture_seq),
                "last_payload_capture_seq": int(last_payload_seq),
                "reuse_blocked": True,
                "fresh_frame": False,
            }
        )
        return payload
    for name in required:
        rgb = np.ascontiguousarray(frames[name][:, :, :3])
        cached_view = view_cache.get(name, {}) if isinstance(view_cache, dict) else {}
        abs_dir = str(cached_view.get("abs_dir", os.path.join(image_dir, name)))
        rel_dir = str(cached_view.get("rel_dir", f"images/{name}"))
        reuse_key = f"{name}:{capture_seq}:{extension}"
        reuse_entry = written_images.get(reuse_key) if isinstance(written_images, dict) else None
        reused_image = isinstance(reuse_entry, dict) and bool(reuse_entry.get("rel_path"))
        if reused_image:
            rel_path = str(reuse_entry.get("rel_path", ""))
            fmt = str(reuse_entry.get("format", extension))
        else:
            filename = f"{int(sample_index):06d}.{extension}"
            abs_path = os.path.join(abs_dir, filename)
            rel_path = f"{rel_dir}/{filename}".replace("\\", "/")
            image_job = {
                "kind": "image",
                "path": abs_path,
                "rgb": rgb.copy(),
                "ensure_dir": False,
            }
            fmt = extension
            if not rt.dataset_writer_enqueue(image_job):
                fmt = save_rgb_image(rt, abs_path, rgb, ensure_dir=False)
            if isinstance(written_images, dict):
                written_images[reuse_key] = {
                    "rel_path": rel_path,
                    "format": fmt,
                }
        view_payload = compact_camera_metadata_value(
            views.get(name, {}) if isinstance(views.get(name, {}), dict) else {},
            key=name,
        )
        view_payload.update(
            {
                "available": True,
                "path": rel_path,
                "format": fmt,
                "capture_seq": capture_seq,
                "capture_finished_at": float(capture_finished),
                "frame_age_ms": float(max(0.0, now - capture_finished) * 1000.0),
                "image_reused": bool(reused_image),
                "fresh_frame": bool(fresh_for_payload),
            }
        )
        payload[f"observation.images.{name}"] = rel_path
        payload["observation.camera"]["views"][name] = view_payload
    payload["observation.camera"]["fresh_frame"] = bool(fresh_for_payload)
    payload["observation.camera"]["last_payload_capture_seq"] = int(last_payload_seq)
    rt.STATE["dataset_camera_last_payload_capture_seq"] = int(capture_seq)
    return payload


async def capture_observations_viewport_async(rt, sample_index, write_files=True):
    payload = empty_payload(rt, sample_index, reason="")
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        payload["observation.camera"]["reason"] = "disabled"
        return payload
    stride = max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1))
    if int(sample_index) >= 0 and int(sample_index) % stride != 0:
        payload["observation.camera"]["reason"] = "stride_skipped"
        return payload
    episode_dir = str(rt.STATE.get("dataset_episode_dir", "") or "")
    image_dir = str(rt.STATE.get("dataset_image_dir", "") or "")
    if write_files and (not episode_dir or not image_dir):
        payload["observation.camera"]["reason"] = "missing_episode_image_dir"
        return payload
    if not initialize(rt, force=False):
        status = rt.STATE.get("dataset_camera_last_status", {})
        payload["observation.camera"]["reason"] = status.get("reason", "camera_unavailable") if isinstance(status, dict) else "camera_unavailable"
        payload["observation.camera"]["views"] = status.get("views", {}) if isinstance(status, dict) else {}
        return payload
    try:
        viewports = ensure_dataset_viewports(rt)
    except Exception as exc:
        payload["observation.camera"]["reason"] = f"dataset_viewport_unavailable:{type(exc).__name__}:{exc}"
        return payload
    set_dataset_viewports_capture_active(rt, True)

    capture_started = time.time()
    before_active = active_viewport_camera_path_snapshot()
    frames = {}
    views = {}
    failures = []
    wait_frames = max(0, int(rt.STATE.get("dataset_camera_viewport_wait_frames", 0) or 0))
    timeout_s = max(0.5, float(rt.STATE.get("dataset_camera_viewport_timeout_s", 2.0) or 2.0))

    async def capture_one(spec):
        name = str(spec.get("name", ""))
        camera_path = str(spec.get("path", ""))
        view_payload = {
            "available": False,
            "name": name,
            "prim_path": camera_path,
            "capture_backend": "viewport_capture",
            "pose": world_pose(rt, camera_path),
        }
        viewport_entry = viewports.get(name) if isinstance(viewports, dict) else None
        if not isinstance(viewport_entry, dict) or viewport_entry.get("viewport_api") is None:
            reason = "dataset_viewport_missing"
            view_payload["reason"] = reason
            return name, None, view_payload, f"{name}:{reason}"
        try:
            rgb_raw, reason, raw_meta = await capture_viewport_rgb_async(
                rt,
                viewport_entry["viewport_api"],
                camera_path,
                wait_frames=wait_frames,
                timeout_s=timeout_s,
                update_camera=False,
            )
            view_payload["raw_meta"] = compact_capture_meta(raw_meta)
            if rgb_raw is None:
                view_payload["reason"] = reason
                return name, None, view_payload, f"{name}:{reason}"
            rgb, norm_reason, stats = normalize_rgb_resolution(rt, rgb_raw)
            view_payload["rgb_stats"] = stats
            view_payload["normalize_reason"] = norm_reason
            if rgb is None:
                view_payload["reason"] = norm_reason
                return name, None, view_payload, f"{name}:{norm_reason}"
            valid, valid_reason = validate_rgb_content(rt, rgb, stats)
            if not valid:
                if str(valid_reason).startswith("black_frame"):
                    rt.STATE["dataset_camera_black_rejected"] = int(rt.STATE.get("dataset_camera_black_rejected", 0) or 0) + 1
                view_payload["reason"] = valid_reason
                return name, None, view_payload, f"{name}:{valid_reason}"
            frame = np.ascontiguousarray(rgb[:, :, :3])
            view_payload.update(
                {
                    "available": True,
                    "reason": "ok",
                    "shape": [int(x) for x in rgb.shape],
                    "dtype": str(rgb.dtype),
                }
            )
            return name, frame, view_payload, ""
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            view_payload["reason"] = reason
            return name, None, view_payload, f"{name}:{reason}"

    try:
        results = await asyncio.gather(*(capture_one(spec) for spec in specs(rt)))
    finally:
        set_dataset_viewports_capture_active(rt, False)
    for name, frame, view_payload, failure in results:
        views[name] = compact_camera_metadata_value(view_payload, key=name)
        if frame is not None:
            frames[name] = frame
        if failure:
            failures.append(failure)

    after_active = active_viewport_camera_path_snapshot()
    capture_finished = time.time()
    payload["observation.camera"]["views"] = compact_camera_views(views)
    payload["observation.camera"]["active_viewport_before"] = before_active
    payload["observation.camera"]["active_viewport_after"] = after_active
    payload["observation.camera"]["active_viewport_unchanged"] = bool(before_active == after_active)

    required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
    missing = [name for name in required if name not in frames]
    if failures or missing:
        payload["observation.camera"]["reason"] = ",".join(failures or [f"missing_required_frames:{','.join(missing)}"])
        return payload

    capture_source = "viewport_capture_direct" if int(sample_index) >= 0 else "viewport_capture_warmup"
    store_latest_capture(rt, frames, views, sample_index, capture_started, capture_finished, source=capture_source)
    payload["observation.camera"]["available"] = True
    payload["observation.camera"]["reason"] = "ok"
    payload["observation.camera"]["image_format"] = image_extension(rt)
    payload["observation.camera"]["resolution"] = resolution(rt)
    payload["observation.camera"]["capture_started_at"] = float(capture_started)
    payload["observation.camera"]["capture_finished_at"] = float(capture_finished)
    payload["observation.camera"]["capture_elapsed_ms"] = float(max(0.0, capture_finished - capture_started) * 1000.0)
    if not write_files:
        return payload

    camera_cache = episode_cache(rt, reset=False)
    extension = str(camera_cache.get("extension", image_extension(rt)))
    view_cache = camera_cache.get("views", {}) if isinstance(camera_cache, dict) else {}

    for name in required:
        cached_view = view_cache.get(name, {}) if isinstance(view_cache, dict) else {}
        filename = f"{int(sample_index):06d}.{extension}"
        abs_dir = str(cached_view.get("abs_dir", os.path.join(image_dir, name)))
        rel_dir = str(cached_view.get("rel_dir", f"images/{name}"))
        abs_path = os.path.join(abs_dir, filename)
        rel_path = f"{rel_dir}/{filename}".replace("\\", "/")
        image_job = {
            "kind": "image",
            "path": abs_path,
            "rgb": frames[name].copy(),
            "ensure_dir": False,
        }
        fmt = extension
        if not rt.dataset_writer_enqueue(image_job):
            fmt = save_rgb_image(rt, abs_path, frames[name], ensure_dir=False)
        payload[f"observation.images.{name}"] = rel_path
        payload["observation.camera"]["views"][name].update(
            {
                "available": True,
                "path": rel_path,
                "format": fmt,
            }
        )
    payload["observation.camera"]["image_format"] = extension
    return payload


def background_interval_seconds(rt):
    value = rt.STATE.get("dataset_camera_background_interval_s", None)
    if value is None:
        value = os.environ.get("EXCAVATOR_DATASET_CAMERA_BACKGROUND_INTERVAL_S", "")
    if value in (None, ""):
        value = 0.50
    try:
        interval = float(value)
    except Exception:
        interval = 0.0
    if interval <= 0.0:
        interval = 0.50
    return max(0.10, float(interval))


def background_min_idle_seconds(rt):
    value = rt.STATE.get("dataset_camera_background_min_idle_s", None)
    if value is None:
        value = os.environ.get("EXCAVATOR_DATASET_CAMERA_BACKGROUND_MIN_IDLE_S", "")
    try:
        idle = float(value)
    except Exception:
        idle = 0.10
    if idle <= 0.0:
        idle = 0.20
    return max(0.02, float(idle))


def capture_block_watchdog_ms(rt):
    value = rt.STATE.get("dataset_camera_capture_block_watchdog_ms", None)
    if value is None:
        value = os.environ.get("EXCAVATOR_DATASET_CAMERA_CAPTURE_BLOCK_WATCHDOG_MS", "")
    try:
        threshold = float(value)
    except Exception:
        threshold = 0.0
    return max(0.0, float(threshold))


def capture_block_backoff_seconds(rt):
    value = rt.STATE.get("dataset_camera_capture_block_backoff_s", None)
    if value is None:
        value = os.environ.get("EXCAVATOR_DATASET_CAMERA_CAPTURE_BLOCK_BACKOFF_S", "")
    try:
        seconds = float(value)
    except Exception:
        seconds = 0.0
    return max(0.0, float(seconds))


def capture_block_backoff_max_seconds(rt):
    value = rt.STATE.get("dataset_camera_capture_block_backoff_max_s", None)
    if value is None:
        value = os.environ.get("EXCAVATOR_DATASET_CAMERA_CAPTURE_BLOCK_BACKOFF_MAX_S", "")
    try:
        seconds = float(value)
    except Exception:
        seconds = 30.0
    return max(0.0, float(seconds))


def capture_backoff_reason(rt, now=None):
    now = time.time() if now is None else float(now)
    until = float(rt.STATE.get("dataset_camera_capture_backoff_until", 0.0) or 0.0)
    if until > now:
        return f"capture_backoff:{until - now:.2f}s"
    return ""


def background_capture_enabled(rt):
    value = os.environ.get("EXCAVATOR_DATASET_CAMERA_BACKGROUND_ENABLED", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_background_enabled", None)
    if state_value is not None:
        return bool(state_value)
    return True


def opportunistic_capture_enabled(rt):
    value = os.environ.get("EXCAVATOR_DATASET_CAMERA_OPPORTUNISTIC_CAPTURE", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_opportunistic_capture_enabled", None)
    if state_value is not None:
        return bool(state_value)
    return False


def _finalize_pending_triplet(rt, triplet):
    try:
        pending = triplet.get("pending")
        if pending:
            return False
        if bool(triplet.get("done", False)):
            return True
        triplet["done"] = True
        capture_finished = time.time()
        frames = triplet.get("frames", {})
        views = triplet.get("views", {})
        failures = list(triplet.get("failures", []) or [])
        required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
        missing = [name for name in required if name not in frames]
        try:
            triplet_generation = int(triplet.get("generation", -1) or -1)
            current_generation = int(rt.STATE.get("dataset_camera_capture_generation", 0) or 0)
        except Exception:
            triplet_generation = -1
            current_generation = 0
        if triplet_generation != current_generation:
            rt.STATE["dataset_camera_stale_captures"] = int(rt.STATE.get("dataset_camera_stale_captures", 0) or 0) + 1
            triplet["failure_reason"] = "stale_capture_generation"
            current = rt.STATE.get("dataset_camera_pending_capture")
            if isinstance(current, dict) and int(current.get("seq", -1) or -1) == int(triplet.get("seq", -2) or -2):
                rt.STATE["dataset_camera_pending_capture"] = None
            set_dataset_viewports_capture_active(rt, False)
            return True
        try:
            triplet_seq = int(triplet.get("seq", 0) or 0)
            latest_seq = int(rt.STATE.get("dataset_camera_latest_seq", 0) or 0)
        except Exception:
            triplet_seq = 0
            latest_seq = 0
        if triplet_seq > 0 and latest_seq > 0 and triplet_seq < latest_seq:
            rt.STATE["dataset_camera_stale_captures"] = int(rt.STATE.get("dataset_camera_stale_captures", 0) or 0) + 1
            triplet["failure_reason"] = "stale_capture_seq"
            current = rt.STATE.get("dataset_camera_pending_capture")
            if isinstance(current, dict) and int(current.get("seq", -1) or -1) == int(triplet.get("seq", -2) or -2):
                rt.STATE["dataset_camera_pending_capture"] = None
            set_dataset_viewports_capture_active(rt, False)
            return True
        elapsed_ms = float(max(0.0, capture_finished - float(triplet.get("started_at", capture_finished) or capture_finished)) * 1000.0)
        watchdog_ms = capture_block_watchdog_ms(rt)
        if watchdog_ms > 0.0 and elapsed_ms > watchdog_ms:
            backoff_s = capture_block_backoff_seconds(rt)
            consecutive = int(rt.STATE.get("dataset_camera_blocked_capture_consecutive", 0) or 0) + 1
            rt.STATE["dataset_camera_blocked_capture_consecutive"] = int(consecutive)
            if backoff_s > 0.0:
                backoff_s = min(capture_block_backoff_max_seconds(rt), backoff_s * float(min(6, consecutive)))
            rt.STATE["dataset_camera_capture_backoff_until"] = max(
                float(rt.STATE.get("dataset_camera_capture_backoff_until", 0.0) or 0.0),
                float(capture_finished + backoff_s),
            )
            rt.STATE["dataset_camera_blocked_capture_count"] = int(rt.STATE.get("dataset_camera_blocked_capture_count", 0) or 0) + 1
            rt.STATE["dataset_camera_last_blocked_capture_ms"] = float(elapsed_ms)
            rt.STATE["dataset_camera_last_blocked_capture_seq"] = int(triplet_seq)
        else:
            rt.STATE["dataset_camera_blocked_capture_consecutive"] = 0
        if not failures and not missing:
            store_latest_capture(
                rt,
                frames,
                views,
                int(triplet.get("sample_index", -1) or -1),
                float(triplet.get("started_at", capture_finished) or capture_finished),
                float(capture_finished),
                source="viewport_capture_callback",
            )
            rt.STATE["dataset_camera_background_complete_captures"] = (
                int(rt.STATE.get("dataset_camera_background_complete_captures", 0) or 0) + 1
            )
        else:
            rt.STATE["dataset_camera_background_failed_triplets"] = (
                int(rt.STATE.get("dataset_camera_background_failed_triplets", 0) or 0) + 1
            )
            triplet["failure_reason"] = ",".join(failures or [f"missing_required_frames:{','.join(missing)}"])
        current = rt.STATE.get("dataset_camera_pending_capture")
        if isinstance(current, dict) and int(current.get("seq", -1) or -1) == int(triplet.get("seq", -2) or -2):
            rt.STATE["dataset_camera_pending_capture"] = None
        set_dataset_viewports_capture_active(rt, False)
        return True
    except Exception as exc:
        rt.STATE["dataset_camera_background_finalize_error"] = f"{type(exc).__name__}:{exc}"
        set_dataset_viewports_capture_active(rt, False)
        return False


def submit_viewport_capture_triplet(rt, sample_index=-1):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        return False, "disabled"
    now = time.time()
    backoff_reason = capture_backoff_reason(rt, now=now)
    if backoff_reason:
        return False, backoff_reason
    if not initialize(rt, force=False):
        status = rt.STATE.get("dataset_camera_last_status", {})
        return False, status.get("reason", "camera_unavailable") if isinstance(status, dict) else "camera_unavailable"
    try:
        from omni.kit.viewport.utility import capture_viewport_to_buffer
    except Exception as exc:
        return False, f"viewport_capture_api_unavailable:{type(exc).__name__}:{exc}"
    try:
        viewports = ensure_dataset_viewports(rt)
    except Exception as exc:
        return False, f"dataset_viewport_unavailable:{type(exc).__name__}:{exc}"

    timeout_s = max(0.5, float(rt.STATE.get("dataset_camera_viewport_timeout_s", 2.0) or 2.0))
    pending = rt.STATE.get("dataset_camera_pending_capture")
    if isinstance(pending, dict) and not bool(pending.get("done", False)):
        age = now - float(pending.get("started_at", now) or now)
        if age <= timeout_s:
            return False, "pending_capture_in_flight"
        rt.STATE["dataset_camera_background_expired_pending"] = (
            int(rt.STATE.get("dataset_camera_background_expired_pending", 0) or 0) + 1
        )
        rt.STATE["dataset_camera_pending_capture"] = None
        set_dataset_viewports_capture_active(rt, False)

    required = [str(x) for x in rt.DATASET_CAMERA_NAMES]
    seq = int(rt.STATE.get("dataset_camera_capture_request_seq", 0) or 0) + 1
    rt.STATE["dataset_camera_capture_request_seq"] = int(seq)
    triplet = {
        "seq": int(seq),
        "sample_index": int(sample_index),
        "started_at": float(now),
        "generation": int(rt.STATE.get("dataset_camera_capture_generation", 0) or 0),
        "frames": {},
        "views": {},
        "failures": [],
        "pending": list(required),
        "helpers": [],
        "done": False,
    }
    rt.STATE["dataset_camera_pending_capture"] = triplet
    set_dataset_viewports_capture_active(rt, True)

    def make_callback(name, camera_path):
        def on_capture(capsule, buffer_size, width, height, fmt=None):
            view_payload = {
                "available": False,
                "name": str(name),
                "prim_path": str(camera_path),
                "capture_backend": "viewport_capture_callback",
                "pose": world_pose(rt, str(camera_path)),
                "resolution_status": dict(viewports.get(str(name), {}).get("resolution_status", {}) or {})
                if isinstance(viewports, dict) and isinstance(viewports.get(str(name), {}), dict)
                else {},
            }
            try:
                rgb_raw = capsule_to_numpy_rgb(capsule, buffer_size, width, height)
                rgb, norm_reason, stats = normalize_rgb_resolution(rt, rgb_raw)
                view_payload["rgb_stats"] = stats
                view_payload["normalize_reason"] = norm_reason
                if rgb is None:
                    view_payload["reason"] = norm_reason
                    triplet["failures"].append(f"{name}:{norm_reason}")
                    return
                valid, valid_reason = validate_rgb_content(rt, rgb, stats)
                if not valid:
                    if str(valid_reason).startswith("black_frame"):
                        rt.STATE["dataset_camera_black_rejected"] = int(rt.STATE.get("dataset_camera_black_rejected", 0) or 0) + 1
                    view_payload["reason"] = valid_reason
                    triplet["failures"].append(f"{name}:{valid_reason}")
                    return
                triplet["frames"][str(name)] = np.ascontiguousarray(rgb[:, :, :3])
                view_payload.update(
                    {
                        "available": True,
                        "reason": "ok",
                        "shape": [int(x) for x in rgb.shape],
                        "dtype": str(rgb.dtype),
                        "format": str(fmt),
                    }
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                view_payload["reason"] = reason
                triplet["failures"].append(f"{name}:{reason}")
            finally:
                triplet["views"][str(name)] = view_payload
                pending_list = triplet.get("pending")
                if isinstance(pending_list, list) and str(name) in pending_list:
                    pending_list.remove(str(name))
                _finalize_pending_triplet(rt, triplet)

        return on_capture

    started_any = False
    for spec in specs(rt):
        name = str(spec.get("name", ""))
        if name not in required:
            continue
        camera_path = str(spec.get("path", ""))
        viewport_entry = viewports.get(name) if isinstance(viewports, dict) else None
        if not isinstance(viewport_entry, dict) or viewport_entry.get("viewport_api") is None:
            triplet["views"][name] = {
                "available": False,
                "name": name,
                "prim_path": camera_path,
                "reason": "dataset_viewport_missing",
            }
            triplet["failures"].append(f"{name}:dataset_viewport_missing")
            if name in triplet["pending"]:
                triplet["pending"].remove(name)
            continue
        try:
            helper = capture_viewport_to_buffer(
                viewport_entry["viewport_api"],
                make_callback(name, camera_path),
                is_hdr=False,
            )
            if hasattr(helper, "__await__"):
                task = asyncio.ensure_future(helper)

                def _consume_task_result(done_task, _name=name):
                    try:
                        done_task.result()
                    except Exception as exc:
                        reason = f"helper_error:{type(exc).__name__}:{exc}"
                        rt.STATE["dataset_camera_background_helper_error"] = reason
                        triplet["failures"].append(f"{_name}:{reason}")
                        if _name in triplet["pending"]:
                            triplet["pending"].remove(_name)
                        _finalize_pending_triplet(rt, triplet)

                task.add_done_callback(_consume_task_result)
                triplet["helpers"].append(task)
            else:
                triplet["helpers"].append(helper)
            started_any = True
        except Exception as exc:
            reason = f"capture_viewport_to_buffer_failed:{type(exc).__name__}:{exc}"
            triplet["views"][name] = {
                "available": False,
                "name": name,
                "prim_path": camera_path,
                "reason": reason,
            }
            triplet["failures"].append(f"{name}:{reason}")
            if name in triplet["pending"]:
                triplet["pending"].remove(name)
    _finalize_pending_triplet(rt, triplet)
    if not started_any:
        rt.STATE["dataset_camera_pending_capture"] = None
        set_dataset_viewports_capture_active(rt, False)
        return False, str(triplet.get("failure_reason", "no_capture_started") or "no_capture_started")
    return True, "submitted"


def maybe_submit_opportunistic_capture(rt, sample_index=-1):
    if not opportunistic_capture_enabled(rt):
        return False, "opportunistic_disabled"
    if backend(rt) != "viewport_capture":
        return False, "backend_not_viewport_capture"
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        return False, "disabled"
    now = time.time()
    last_submit = float(rt.STATE.get("dataset_camera_last_opportunistic_submit_time", 0.0) or 0.0)
    latest_time = float(rt.STATE.get("dataset_camera_latest_capture_time", 0.0) or 0.0)
    interval = background_interval_seconds(rt)
    latest_missing = latest_time <= 0.0 or not isinstance(rt.STATE.get("dataset_camera_latest_capture"), dict)
    if (not latest_missing) and now - last_submit < interval:
        return False, "interval_not_due"
    submitted, reason = submit_viewport_capture_triplet(rt, sample_index=sample_index)
    if submitted:
        rt.STATE["dataset_camera_last_opportunistic_submit_time"] = float(now)
        rt.STATE["dataset_camera_opportunistic_submissions"] = int(
            rt.STATE.get("dataset_camera_opportunistic_submissions", 0) or 0
        ) + 1
    elif reason != "pending_capture_in_flight":
        rt.STATE["dataset_camera_opportunistic_failures"] = int(
            rt.STATE.get("dataset_camera_opportunistic_failures", 0) or 0
        ) + 1
    return bool(submitted), str(reason)


async def background_capture_loop(rt, label="dataset_camera_background"):
    rt.STATE["dataset_camera_background_stop_requested"] = False
    rt.STATE["dataset_camera_background_running"] = True
    rt.STATE["dataset_camera_background_label"] = str(label)
    failures = 0
    captures = 0
    rt.info_print(
        "[DATASET CAMERA BACKGROUND]",
        "started",
        f"backend={backend(rt)}",
        f"interval_s={background_interval_seconds(rt):.3f}",
    )
    try:
        while (
            bool(rt.STATE.get("dataset_camera_enabled", True))
            and not bool(rt.STATE.get("dataset_camera_background_stop_requested", False))
            and bool(rt.STATE.get("auto_collect_active", False))
            and bool(str(rt.STATE.get("dataset_episode_uid", "") or ""))
            and bool(str(rt.STATE.get("dataset_episode_dir", "") or ""))
        ):
            started = time.time()
            submitted, reason = submit_viewport_capture_triplet(rt, sample_index=-1)
            if submitted:
                captures += 1
                rt.STATE["dataset_camera_background_submissions"] = int(captures)
                failures = 0
            elif reason != "pending_capture_in_flight":
                failures += 1
                rt.STATE["dataset_camera_background_failures"] = int(failures)
                if failures <= 3 or failures % 30 == 0:
                    rt.info_print(
                        "[WARN] [DATASET CAMERA BACKGROUND]",
                        f"capture_failed={failures}",
                        f"reason={reason}",
                    )
            elapsed = time.time() - started
            delay = max(background_min_idle_seconds(rt), background_interval_seconds(rt) - elapsed)
            if delay > 0.0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    finally:
        rt.STATE["dataset_camera_background_running"] = False
        rt.info_print(
            "[DATASET CAMERA BACKGROUND]",
            "stopped",
            f"submissions={int(rt.STATE.get('dataset_camera_background_submissions', captures) or 0)}",
            f"complete={int(rt.STATE.get('dataset_camera_background_complete_captures', 0) or 0)}",
            f"failures={int(rt.STATE.get('dataset_camera_background_failures', failures) or 0)}",
        )


def stop_background_capture(rt, reason=""):
    rt.STATE["dataset_camera_background_stop_requested"] = True
    rt.STATE["dataset_camera_background_stop_reason"] = str(reason)


def rgb_ready(rt):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        return True, "disabled"
    if not rt.simulation_timeline_is_playing():
        return False, "timeline_not_playing"
    if backend(rt) == "viewport_capture":
        status = rt.STATE.get("dataset_camera_warmup_status", {})
        if isinstance(status, dict) and bool(status.get("ok", False)):
            return True, "ok"
        return bool(initialize(rt, force=False)), str((rt.STATE.get("dataset_camera_last_status", {}) or {}).get("reason", "ok"))
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
    if backend(rt) == "viewport_capture":
        max_frames = max(1, int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12))
        min_frames = max(0, int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3))
        ready_required = max(1, int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2))
        ready_streak = 0
        last_reason = "not_checked"
        last_views = {}
        for frame in range(max_frames):
            payload = await capture_observations_viewport_async(rt, sample_index=-1, write_files=False)
            camera_info = payload.get("observation.camera", {}) if isinstance(payload, dict) else {}
            last_reason = str(camera_info.get("reason", "") or "camera_payload_missing")
            last_views = camera_info.get("views", {}) if isinstance(camera_info, dict) else {}
            if bool(camera_info.get("available", False)):
                ready_streak += 1
            else:
                ready_streak = 0
            if frame + 1 >= min_frames and ready_streak >= ready_required:
                status = {
                    "ok": True,
                    "reason": "ok",
                    "label": str(label),
                    "frames": int(frame + 1),
                    "ready_streak": int(ready_streak),
                    "backend": "viewport_capture",
                    "views": compact_camera_views(last_views),
                    "active_viewport_unchanged": bool(camera_info.get("active_viewport_unchanged", True)),
                }
                rt.STATE["dataset_camera_warmup_status"] = status
                rt.info_print(
                    "[DATASET CAMERA WARMUP OK]",
                    "backend=viewport_capture",
                    f"label={label}",
                    f"frames={frame + 1}",
                    f"ready_streak={ready_streak}",
                    format_rgb_stats(last_views),
                )
                return True
        status = {
            "ok": False,
            "reason": str(last_reason),
            "label": str(label),
            "frames": int(max_frames),
            "ready_streak": int(ready_streak),
            "backend": "viewport_capture",
            "views": compact_camera_views(last_views),
        }
        rt.STATE["dataset_camera_warmup_status"] = status
        rt.info_print(
            "[WARN] [DATASET CAMERA WARMUP]",
            "backend=viewport_capture",
            f"reason={last_reason}",
            format_rgb_stats(last_views),
        )
        return False
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
                "views": compact_camera_views(view_stats),
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
        "views": compact_camera_views(view_stats),
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
    payload = empty_payload(rt, sample_index, reason="sync_capture_uses_latest_background_frame")
    if backend(rt) == "viewport_capture":
        if opportunistic_capture_enabled(rt):
            maybe_submit_opportunistic_capture(rt, sample_index=sample_index)
        return latest_capture_payload(rt, sample_index)
    payload["observation.camera"]["views"] = {}
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
        "available": bool(current_backend == "viewport_capture" or rep is not None),
        "backend": current_backend,
        "renderer_launch": renderer_launch,
        "stable_render_settings": dict(rt.STATE.get("dataset_stable_render_settings", {}) or {}),
        "captureOnPlay": (rt.STATE.get("dataset_camera_capture_on_play_status") or {}).get("captureOnPlay"),
        "captureOnPlay_status": rt.STATE.get("dataset_camera_capture_on_play_status"),
        "backend_available": {
            "viewport_capture": True,
            "replicator": bool(rep is not None),
            "replicator_rgb": bool(rep is not None),
            "isaac_camera": False,
            "replicator_tick": bool(replicator_tick_enabled(rt)),
            "syntheticdata_wait": bool(syntheticdata_wait_enabled(rt)),
            "wait_for_render": bool(camera_wait_for_render_enabled(rt)),
            "module_version": CAMERA_MODULE_VERSION,
            "note": "viewport_capture is the production backend; replicator_rgb is retained only as an explicit diagnostic fallback.",
        },
        "resolution": resolution(rt),
        "viewport_capture_resolution": viewport_capture_resolution(rt),
        "viewport_capture_note": "Viewport texture is captured at this higher resolution and Lanczos-downsampled to resolution for saved training images.",
        "frequency": int(rt.STATE.get("dataset_camera_frequency", 10) or 10),
        "sample_stride": max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1)),
        "require_complete_samples": bool(rt.STATE.get("dataset_camera_require_complete_samples", True)),
        "warmup_frames": int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3),
        "warmup_ready_frames": int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2),
        "warmup_max_frames": int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12),
        "viewport_wait_frames": max(0, int(rt.STATE.get("dataset_camera_viewport_wait_frames", 0) or 0)),
        "viewport_timeout_s": max(0.5, float(rt.STATE.get("dataset_camera_viewport_timeout_s", 2.0) or 2.0)),
        "viewport_keep_visible": bool(dataset_viewport_keep_visible(rt)),
        "viewport_visible": bool(rt.STATE.get("dataset_camera_viewport_visible", False)),
        "viewport_visible_count": int(rt.STATE.get("dataset_camera_viewport_visible_count", 0) or 0),
        "allow_frame_reuse": bool(camera_allow_frame_reuse(rt)),
        "last_payload_capture_seq": int(rt.STATE.get("dataset_camera_last_payload_capture_seq", 0) or 0),
        "background_capture": {
            "enabled_for_auto_collect": bool(background_capture_enabled(rt)),
            "interval_s": background_interval_seconds(rt),
            "min_idle_s": background_min_idle_seconds(rt),
            "latest_seq": int(rt.STATE.get("dataset_camera_latest_seq", 0) or 0),
            "running": bool(rt.STATE.get("dataset_camera_background_running", False)),
            "opportunistic_capture": bool(opportunistic_capture_enabled(rt)),
            "opportunistic_submissions": int(rt.STATE.get("dataset_camera_opportunistic_submissions", 0) or 0),
            "note": "Throughput mode: background capture owns camera submissions at its interval; dataset samples only attach the latest complete triplet unless opportunistic capture is explicitly enabled.",
        },
        "capture_block_watchdog": {
            "threshold_ms": capture_block_watchdog_ms(rt),
            "backoff_s": capture_block_backoff_seconds(rt),
            "backoff_max_s": capture_block_backoff_max_seconds(rt),
            "blocked_count": int(rt.STATE.get("dataset_camera_blocked_capture_count", 0) or 0),
            "blocked_consecutive": int(rt.STATE.get("dataset_camera_blocked_capture_consecutive", 0) or 0),
            "last_blocked_capture_ms": float(rt.STATE.get("dataset_camera_last_blocked_capture_ms", 0.0) or 0.0),
        },
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
