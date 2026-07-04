import asyncio
import inspect
import os
import time

import numpy as np
import omni.usd
import omni.kit.app
try:
    import omni.replicator.core as rep
except Exception:
    rep = None

from pxr import Usd, UsdGeom, Gf, Sdf
try:
    from isaacsim.sensors.camera import Camera as IsaacCamera
    HAS_ISAAC_CAMERA = True
except Exception:
    IsaacCamera = None
    HAS_ISAAC_CAMERA = False
try:
    from PIL import Image
except Exception:
    Image = None


CAMERA_MODULE_VERSION = "dataset_camera_no_manual_replicator_tick_v3"
SYNC_STEP_ERROR_TEXT = "Synchronous call to `step`"


def replicator_tick_enabled(rt):
    value = os.environ.get("EXCAVATOR_CAMERA_REPLICATOR_TICK", "")
    if value != "":
        return value.strip().lower() in ("1", "true", "yes", "on")
    state_value = rt.STATE.get("dataset_camera_replicator_tick_enabled", None)
    if state_value is not None:
        return bool(state_value)
    return False


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


def _replicator_step_async(rt, rt_subframes=1):
    if rep is None:
        return None
    step_async = getattr(rep.orchestrator, "step_async", None)
    if not callable(step_async):
        return None
    return step_async(
        rt_subframes=int(rt_subframes),
        delta_time=rt.CONTROL_DT,
        pause_timeline=False,
    )


def global_tick(rt):
    if not replicator_tick_enabled(rt):
        return True
    if rep is None:
        rt.info_print("[WARN] camera tick failed:", "replicator_unavailable")
        return False
    if bool(rt.STATE.get("dataset_camera_replicator_step_disabled", False)):
        return True
    install_replicator_simtime_guard(rt)
    try:
        # This module runs inside Kit's asyncio loop. Replicator's synchronous
        # orchestrator.step() is only legal in standalone workflows, so the
        # sync API here only schedules step_async and returns immediately.
        if bool(rt.STATE.get("dataset_camera_tick_pending", False)):
            return True
        pending = _replicator_step_async(rt, rt_subframes=1)
        if inspect.isawaitable(pending):
            rt.STATE["dataset_camera_tick_pending"] = True
            task = asyncio.ensure_future(pending)

            def _clear_tick_pending(_task):
                rt.STATE["dataset_camera_tick_pending"] = False
                try:
                    _task.result()
                except Exception as exc:
                    if SYNC_STEP_ERROR_TEXT in str(exc):
                        rt.STATE["dataset_camera_replicator_step_disabled"] = True
                        if not bool(rt.STATE.get("dataset_camera_replicator_step_disable_logged", False)):
                            rt.STATE["dataset_camera_replicator_step_disable_logged"] = True
                            rt.info_print(
                                "[WARN] camera async tick disabled:",
                                type(exc).__name__,
                                "Kit rejected Replicator step_async; using IsaacCamera/Kit update frames only",
                            )
                    else:
                        rt.info_print("[WARN] camera async tick failed:", type(exc).__name__, exc)

            task.add_done_callback(_clear_tick_pending)
        elif pending is None:
            return True
        return True
    except Exception as e:
        rt.STATE["dataset_camera_tick_pending"] = False
        if SYNC_STEP_ERROR_TEXT in str(e):
            rt.STATE["dataset_camera_replicator_step_disabled"] = True
            if not bool(rt.STATE.get("dataset_camera_replicator_step_disable_logged", False)):
                rt.STATE["dataset_camera_replicator_step_disable_logged"] = True
                rt.info_print(
                    "[WARN] camera async tick disabled:",
                    type(e).__name__,
                    "Kit rejected Replicator step_async; using IsaacCamera/Kit update frames only",
                )
            return True
        rt.info_print("[WARN] camera tick failed:", type(e).__name__, e)
        return False


async def global_tick_async(rt):
    if not replicator_tick_enabled(rt):
        await rt.step_updates(1)
        return True
    if rep is None:
        rt.info_print("[WARN] camera tick failed:", "replicator_unavailable")
        return False
    if bool(rt.STATE.get("dataset_camera_replicator_step_disabled", False)):
        await rt.step_updates(1)
        return True
    install_replicator_simtime_guard(rt)
    try:
        rt.STATE["dataset_camera_tick_pending"] = False
        pending = _replicator_step_async(rt, rt_subframes=1)
        if inspect.isawaitable(pending):
            await pending
        elif pending is None:
            await rt.step_updates(1)
        return True
    except Exception as e:
        if SYNC_STEP_ERROR_TEXT in str(e):
            rt.STATE["dataset_camera_replicator_step_disabled"] = True
            if not bool(rt.STATE.get("dataset_camera_replicator_step_disable_logged", False)):
                rt.STATE["dataset_camera_replicator_step_disable_logged"] = True
                rt.info_print(
                    "[WARN] camera async tick disabled:",
                    type(e).__name__,
                    "Kit rejected Replicator step_async; using IsaacCamera/Kit update frames only",
                )
            await rt.step_updates(1)
            return True
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
    if not replicator_tick_enabled(rt):
        return
    if bool(rt.STATE.get("dataset_camera_replicator_step_disabled", False)):
        return
    for _ in range(max(1, int(frames))):
        global_tick(rt)


def backend(rt):
    rt.STATE["dataset_camera_backend"] = "isaac"
    return "isaac"


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
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
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
    objects = rt.STATE.get("dataset_camera_objects")
    closed = 0
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
    rt.STATE["dataset_camera_initialized"] = False
    rt.STATE["dataset_camera_init_attempted"] = False
    rt.STATE["dataset_camera_last_status"] = {"enabled": bool(rt.STATE.get("dataset_camera_enabled", True)), "available": False, "reason": reason, "closed": closed}
    return closed


def initialize(rt, force=False):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        rt.STATE["dataset_camera_last_status"] = {"enabled": False, "reason": "disabled"}
        return False
    current_backend = backend(rt)
    if current_backend == "isaac" and (not HAS_ISAAC_CAMERA or IsaacCamera is None):
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": "isaac_camera_api_unavailable", "backend": current_backend}
        return False
    ready, ready_reason = runtime_ready(rt)
    if not ready:
        rt.STATE["dataset_camera_last_status"] = {"enabled": True, "available": False, "reason": ready_reason, "backend": current_backend}
        return False
    if bool(rt.STATE.get("dataset_camera_initialized", False)) and not force:
        objects = rt.STATE.get("dataset_camera_objects")
        if isinstance(objects, dict) and objects:
            return True

    stage_obj = omni.usd.get_context().get_stage()
    cam_resolution = resolution(rt)
    frequency = int(rt.STATE.get("dataset_camera_frequency", 10) or 10)
    objects = {}
    install_replicator_simtime_guard(rt)
    status = {
        "enabled": True,
        "schema": rt.DATASET_CAMERA_SCHEMA,
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "backend": current_backend,
        "replicator_tick_enabled": bool(replicator_tick_enabled(rt)),
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
            rt.set_prim_visibility(prim, True)
            cam = IsaacCamera(prim_path=path, resolution=(int(cam_resolution[0]), int(cam_resolution[1])), frequency=frequency)
            cam.initialize()
            objects[name] = cam
            view_status.update(prim_metadata(rt, path))
            view_status.update({"available": True, "reason": "ok"})
        except Exception as exc:
            view_status.update({"available": False, "reason": f"{type(exc).__name__}:{exc}", "prim_path": path})
        status["views"][name] = view_status

    if objects:
        first_name = sorted(objects.keys())[0]
        first_cam = objects[first_name]
        for name in rt.DATASET_CAMERA_NAMES:
            if name not in objects:
                objects[name] = first_cam
                view_status = status["views"].get(name, {"name": name})
                view_status.update({"available": True, "reason": "shared_isaac_camera_fallback", "fallback_source": first_name})
                status["views"][name] = view_status

    rt.STATE["dataset_camera_objects"] = objects
    rt.STATE["dataset_camera_initialized"] = bool(objects)
    rt.STATE["dataset_camera_init_attempted"] = True
    rt.STATE["dataset_camera_last_status"] = status
    if objects:
        warmup_graph(rt, 5)
        rt.info_print(
            "[DATASET CAMERA]",
            f"module={CAMERA_MODULE_VERSION}",
            "backend=isaac",
            f"replicator_tick={replicator_tick_enabled(rt)}",
            f"views={list(objects.keys())}",
            f"resolution={cam_resolution}",
            f"format={image_extension(rt)}",
        )
    else:
        rt.info_print("[WARN] [DATASET CAMERA] no usable camera views", status)
    return bool(objects)


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
        if not payload.get(key):
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
    objects = rt.STATE.get("dataset_camera_objects")
    if not isinstance(objects, dict):
        return False, "camera_objects_missing"
    if objects:
        return True, "ok"
    missing = []
    for name in rt.DATASET_CAMERA_NAMES:
        cam = objects.get(str(name))
        if cam is None:
            missing.append(f"{name}:missing")
            continue
        try:
            rgb = cam.get_rgb()
            if rgb is None:
                missing.append(f"{name}:rgb_none")
        except Exception as exc:
            missing.append(f"{name}:{type(exc).__name__}")
    if missing:
        return False, ",".join(missing)
    return True, "ok"


async def warmup_for_episode(rt, label="episode"):
    if not bool(rt.STATE.get("dataset_camera_enabled", True)):
        rt.STATE["dataset_camera_warmup_status"] = {"ok": True, "reason": "disabled", "label": str(label)}
        return True
    if not bool(rt.STATE.get("dataset_camera_require_complete_samples", True)):
        rt.STATE["dataset_camera_warmup_status"] = {"ok": True, "reason": "complete_samples_not_required", "label": str(label)}
        return True
    max_frames = max(0, int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12))
    min_frames = max(0, int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3))
    ready_required = max(1, int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2))
    ready_streak = 0
    last_reason = "not_checked"
    if max_frames <= 0:
        max_frames = max(min_frames, ready_required)
    for frame in range(max_frames):
        await rt.step_updates(1)
        await global_tick_async(rt)
        ready, reason = rgb_ready(rt)
        last_reason = reason
        if ready:
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
            }
            rt.STATE["dataset_camera_warmup_status"] = status
            rt.info_print(
                "[DATASET CAMERA WARMUP]",
                f"label={label}",
                f"ok=True",
                f"frames={frame + 1}",
                f"ready_streak={ready_streak}",
            )
            return True
    status = {
        "ok": False,
        "reason": str(last_reason),
        "label": str(label),
        "frames": int(max_frames),
        "ready_streak": int(ready_streak),
    }
    rt.STATE["dataset_camera_warmup_status"] = status
    rt.info_print(
        "[WARN] [DATASET CAMERA WARMUP]",
        f"label={label}",
        "ok=False",
        f"frames={max_frames}",
        f"ready_streak={ready_streak}",
        f"reason={last_reason}",
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

    objects = rt.STATE.get("dataset_camera_objects")
    if not isinstance(objects, dict):
        objects = {}
    camera_cache = episode_cache(rt, reset=False)
    extension = str(camera_cache.get("extension", image_extension(rt)))
    cam_resolution = resolution(rt)
    any_available = False
    view_cache = camera_cache.get("views", {}) if isinstance(camera_cache, dict) else {}
    for name in rt.DATASET_CAMERA_NAMES:
        cam = objects.get(name)
        view_payload = {
            "available": False,
            "name": name,
            "path": None,
            "shape": None,
            "dtype": None,
        }
        cached_view = view_cache.get(str(name), {}) if isinstance(view_cache, dict) else {}
        prim_path = str(cached_view.get("prim_path", ""))
        view_payload["prim_path"] = prim_path
        view_payload["pose"] = world_pose(rt, prim_path)
        try:
            fallback_reason = ""
            if cam is None:
                rgb = np.zeros((int(cam_resolution[1]), int(cam_resolution[0]), 3), dtype=np.uint8)
                fallback_reason = "camera_object_fallback"
            else:
                rgb = cam.get_rgb()
                if rgb is None:
                    rgb = np.zeros((int(cam_resolution[1]), int(cam_resolution[0]), 3), dtype=np.uint8)
                    fallback_reason = "camera_rgb_fallback"
            rgb = np.asarray(rgb)
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                rgb = np.zeros((int(cam_resolution[1]), int(cam_resolution[0]), 3), dtype=np.uint8)
                fallback_reason = "camera_shape_fallback"
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
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
                    "capture_backend": current_backend,
                }
            )
            if fallback_reason:
                view_payload["reason"] = fallback_reason
            any_available = True
        except Exception as exc:
            view_payload["reason"] = f"{type(exc).__name__}:{exc}"
            now = time.time()
            if now - float(rt.STATE.get("dataset_camera_last_error_time", 0.0)) > 2.0:
                rt.STATE["dataset_camera_last_error_time"] = now
                rt.info_print("[WARN] dataset camera capture failed:", name, type(exc).__name__, exc)
        payload["observation.camera"]["views"][name] = view_payload
    payload["observation.camera"]["available"] = any_available
    payload["observation.camera"]["image_format"] = extension
    payload["observation.camera"]["resolution"] = resolution(rt)
    return payload


def config_snapshot(rt):
    current_backend = backend(rt)
    return {
        "schema": rt.DATASET_CAMERA_SCHEMA,
        "module_version": CAMERA_MODULE_VERSION,
        "module_file": __file__,
        "enabled": bool(rt.STATE.get("dataset_camera_enabled", True)),
        "available": bool(HAS_ISAAC_CAMERA),
        "backend": current_backend,
        "backend_available": {
            "isaac_camera": bool(HAS_ISAAC_CAMERA),
            "viewport_capture": False,
            "viewport_reason": "dataset_viewport_capture_disabled",
            "replicator_tick": bool(replicator_tick_enabled(rt)),
            "module_version": CAMERA_MODULE_VERSION,
        },
        "resolution": resolution(rt),
        "frequency": int(rt.STATE.get("dataset_camera_frequency", 10) or 10),
        "sample_stride": max(1, int(rt.STATE.get("dataset_camera_sample_stride", 1) or 1)),
        "require_complete_samples": bool(rt.STATE.get("dataset_camera_require_complete_samples", True)),
        "warmup_frames": int(rt.STATE.get("dataset_camera_warmup_frames", 3) or 3),
        "warmup_ready_frames": int(rt.STATE.get("dataset_camera_warmup_ready_frames", 2) or 2),
        "warmup_max_frames": int(rt.STATE.get("dataset_camera_warmup_max_frames", 12) or 12),
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
