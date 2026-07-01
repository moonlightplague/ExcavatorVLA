import os
import sys
from urllib.parse import unquote, urlparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for import_root in [SCRIPT_DIR, os.environ.get("EXCAVATOR_PROJECT_ROOT", "")]:
    if import_root and import_root not in sys.path:
        sys.path.insert(0, import_root)

from excavator_common import paths


def _url_to_path(value):
    if not value:
        return ""
    text = str(value).rstrip("*")
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


def _add_import_roots(project_root):
    import_roots = [project_root, os.path.join(project_root, "scripts")]
    for import_root in reversed(import_roots):
        import_root = os.path.abspath(import_root)
        while import_root in sys.path:
            try:
                sys.path.remove(import_root)
            except ValueError:
                break
        sys.path.insert(0, import_root)


def _load_runtime_config(project_root):
    try:
        config = paths.load_config(project_root)
        print(f"[INFO] Loaded excavator config: {paths.project_path(paths.CONFIG_FILE_NAME, root=project_root)}")
        return config
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        return {}


def _validated_model_source(config):
    model_source = str(config.get("model_source", "original") or "original").strip().lower()
    if model_source not in {"original", "scene", "original-scene"}:
        raise ValueError(
            "Only the original excavator scene is supported by main.py. "
            f"Set model_source to 'original' or remove it, got: {model_source!r}"
        )
    return "original"


def _pump_kit_updates(frame_count=5):
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(max(0, int(frame_count))):
            app.update()
    except Exception:
        pass


def _open_original_stage(project_root):
    stage_path = paths.default_scene_path(project_root)
    if not os.path.isfile(stage_path):
        print(f"[WARN] Original excavator scene not found: {stage_path}")
        return

    try:
        import omni.usd

        current_stage = _current_stage_file()
        if current_stage and os.path.abspath(current_stage) == os.path.abspath(stage_path):
            print(f"[INFO] Original excavator scene already open: {stage_path}")
            return

        opened = omni.usd.get_context().open_stage(stage_path)
        if opened is False:
            print(f"[WARN] Isaac Sim did not report success opening original excavator scene: {stage_path}")
        else:
            print(f"[INFO] Opened original excavator scene: {stage_path}")
        _pump_kit_updates(5)
    except Exception as exc:
        print(f"[WARN] Could not open original excavator scene {stage_path}: {repr(exc)}")


PROJECT_ROOT = paths.find_project_root(start=__file__)
_add_import_roots(PROJECT_ROOT)
CONFIG = _load_runtime_config(PROJECT_ROOT)
MODEL_SOURCE = _validated_model_source(CONFIG)

print(f"[INFO] Excavator model source: {MODEL_SOURCE}")
print(f"[INFO] Excavator project root: {PROJECT_ROOT}")
print(f"[INFO] Excavator entry script: {os.path.abspath(__file__)}")
print(f"[INFO] Excavator config project_root: {CONFIG.get('project_root', '')}")
_open_original_stage(PROJECT_ROOT)

from excavator_app.bootstrap import run_excavator_with_sand

run_excavator_with_sand()
