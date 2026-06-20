import os
import shutil
import sys
from urllib.parse import unquote, urlparse


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return ""


def _looks_like_project_root(root):
    return bool(root) and (
        os.path.isfile(os.path.join(root, "scripts", "excavator_app", "bootstrap.py"))
        or os.path.isfile(os.path.join(root, "excavator_app", "bootstrap.py"))
    )


def _import_roots(root):
    roots = [root, os.path.join(root, "scripts")]
    out = []
    for candidate in roots:
        candidate = os.path.abspath(candidate)
        if candidate not in out:
            out.append(candidate)
    return out


def _zsp_dir(root):
    return os.path.join(root, "assets", "zsp")


def _zsp_stage_path(root):
    return os.path.join(_zsp_dir(root), "URDF_real3.usd")


def _url_to_path(value):
    if not value:
        return ""
    text = str(value)
    text = text.rstrip("*")
    if text.startswith("file:"):
        parsed = urlparse(text)
        path = unquote(parsed.path or "")
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return os.path.abspath(path)
    return os.path.abspath(text)


def _current_stage_file():
    try:
        import omni.usd

        context = omni.usd.get_context()
        url = ""
        try:
            url = context.get_stage_url()
        except Exception:
            url = ""
        if url:
            path = _url_to_path(url)
            if path:
                return path
        stage = context.get_stage()
        if stage is not None and stage.GetRootLayer() is not None:
            identifier = getattr(stage.GetRootLayer(), "realPath", "") or stage.GetRootLayer().identifier
            return _url_to_path(identifier)
    except Exception:
        return ""
    return ""


def _parents(path, max_depth=6):
    out = []
    cur = os.path.abspath(path) if path else ""
    if cur and os.path.isfile(cur):
        cur = os.path.dirname(cur)
    for _ in range(max_depth):
        if not cur or cur in out:
            break
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def _stage_related_roots():
    stage_file = _current_stage_file()
    roots = []
    for parent in _parents(stage_file, max_depth=7):
        roots.append(parent)
        for child_name in ["URDF_real3", "URDF_real3_code", "excavator_code", "code"]:
            roots.append(os.path.join(parent, child_name))
        try:
            for child in os.listdir(parent):
                candidate = os.path.join(parent, child)
                if _looks_like_project_root(candidate):
                    roots.append(candidate)
        except Exception:
            pass
    return roots


def _known_repo_roots():
    roots = []
    for root in [
        "/isaac-sim/ExcavatorVLA",
        "/root/isaacsim/ExcavatorVLA",
    ]:
        if os.path.isdir(root):
            roots.append(root)
    return roots


def _candidate_roots():
    roots = [
        os.environ.get("EXCAVATOR_PROJECT_ROOT", ""),
        os.environ.get("URDF_REAL3_ROOT", ""),
        _script_dir(),
        os.getcwd(),
    ]
    roots.extend(_known_repo_roots())
    roots.extend(_stage_related_roots())
    out = []
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if root not in out:
            out.append(root)
    return out


def ensure_project_root():
    for root in _candidate_roots():
        if _looks_like_project_root(root):
            for import_root in reversed(_import_roots(root)):
                if import_root not in sys.path:
                    sys.path.insert(0, import_root)
            return root
    raise RuntimeError(
        "Cannot find scripts/excavator_app/bootstrap.py. Script Editor may be running a temp copy; "
        "set EXCAVATOR_PROJECT_ROOT=/isaac-sim/ExcavatorVLA or run main_zsp.py from the repository root."
    )


def _normalize_zsp_layer_paths(project_root):
    zsp_dir = _zsp_dir(project_root)
    if not os.path.isdir(zsp_dir):
        return

    try:
        from pxr import Sdf
    except Exception as exc:
        print("[WARN] Could not import pxr.Sdf for ZSP USD path normalization:", repr(exc))
        return

    aliases = {
        "configuration/URDF_real3_base.usd": "URDF_real3_base.usd",
        "configuration/URDF_real3_physics.usd": "URDF_real3_physics.usd",
        "configuration/URDF_real3_robot.usd": "URDF_real3_robot.usd",
        "configuration/URDF_real3_sensor.usd": "URDF_real3_sensor.usd",
    }

    for filename in [
        "URDF_real3.usd",
        "URDF_real3_base.usd",
        "URDF_real3_physics.usd",
        "URDF_real3_robot.usd",
        "URDF_real3_sensor.usd",
    ]:
        layer_path = os.path.join(zsp_dir, filename)
        if not os.path.isfile(layer_path):
            continue

        try:
            layer = Sdf.Layer.FindOrOpen(layer_path)
            if layer is None:
                continue

            changed = False
            sublayers = []
            for sublayer_path in layer.subLayerPaths:
                normalized_key = str(sublayer_path).replace("\\", "/")
                new_path = aliases.get(normalized_key, sublayer_path)
                if new_path != sublayer_path:
                    changed = True
                sublayers.append(new_path)

            if changed:
                layer.subLayerPaths = sublayers
                layer.Save()
                print(f"[INFO] Normalized ZSP USD sublayers: {layer_path}")
        except Exception as exc:
            print(f"[WARN] Could not normalize ZSP layer {layer_path}: {repr(exc)}")


def _ensure_zsp_configuration_aliases(project_root):
    zsp_dir = _zsp_dir(project_root)
    configuration_dir = os.path.join(zsp_dir, "configuration")
    aliases = [
        "URDF_real3_base.usd",
        "URDF_real3_physics.usd",
        "URDF_real3_robot.usd",
        "URDF_real3_sensor.usd",
    ]
    if not os.path.isdir(zsp_dir):
        return

    os.makedirs(configuration_dir, exist_ok=True)
    for filename in aliases:
        source = os.path.join(zsp_dir, filename)
        target = os.path.join(configuration_dir, filename)
        if not os.path.isfile(source):
            print(f"[WARN] ZSP source USD missing for configuration alias: {source}")
            continue
        if os.path.exists(target):
            continue
        try:
            os.symlink(os.path.join("..", filename), target)
        except Exception:
            shutil.copy2(source, target)


def _open_zsp_stage(project_root):
    stage_path = _zsp_stage_path(project_root)
    if not os.path.isfile(stage_path):
        print(f"[WARN] ZSP stage not found: {stage_path}")
        return

    _ensure_zsp_configuration_aliases(project_root)
    _normalize_zsp_layer_paths(project_root)

    try:
        import omni.usd

        current_stage = _current_stage_file()
        if current_stage and os.path.abspath(current_stage) == os.path.abspath(stage_path):
            return

        opened = omni.usd.get_context().open_stage(stage_path)
        if opened is False:
            print(f"[WARN] Isaac Sim did not report success opening ZSP stage: {stage_path}")
        else:
            print(f"[INFO] Opened ZSP stage: {stage_path}")
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            for _ in range(5):
                app.update()
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] Could not open ZSP stage {stage_path}: {repr(exc)}")


PROJECT_ROOT = ensure_project_root()
_open_zsp_stage(PROJECT_ROOT)

from excavator_app.bootstrap import run_excavator_with_sand

run_excavator_with_sand()
