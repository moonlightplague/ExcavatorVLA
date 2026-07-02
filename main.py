import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for import_root in [SCRIPT_DIR, os.environ.get("EXCAVATOR_PROJECT_ROOT", "")]:
    if import_root and import_root not in sys.path:
        sys.path.insert(0, import_root)

from excavator_common import paths


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


PROJECT_ROOT = paths.find_project_root(start=__file__)
_add_import_roots(PROJECT_ROOT)
CONFIG = _load_runtime_config(PROJECT_ROOT)
MODEL_SOURCE = _validated_model_source(CONFIG)

print(f"[INFO] Excavator model source: {MODEL_SOURCE}")
print(f"[INFO] Excavator project root: {PROJECT_ROOT}")
print(f"[INFO] Excavator entry script: {os.path.abspath(__file__)}")
print(f"[INFO] Excavator config project_root: {CONFIG.get('project_root', '')}")

from excavator_app.bootstrap import run_excavator_with_sand

run_excavator_with_sand()
