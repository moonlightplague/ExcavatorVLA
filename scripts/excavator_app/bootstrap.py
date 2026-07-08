import importlib
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_project_root(extra_root=None):
    roots = []
    if extra_root:
        roots.append(extra_root)
    for key in ["EXCAVATOR_PROJECT_ROOT", "URDF_REAL3_ROOT"]:
        value = os.environ.get(key, "")
        if value:
            roots.append(value)
    roots.append(PROJECT_ROOT)
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if os.path.isfile(os.path.join(root, "excavator_app", "bootstrap.py")):
            while root in sys.path:
                try:
                    sys.path.remove(root)
                except ValueError:
                    break
            sys.path.insert(0, root)
            return root
    return ""


def reload_runtime(module_name):
    ensure_project_root()
    importlib.invalidate_caches()
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def run_excavator_with_sand():
    root = ensure_project_root()
    print(f"[INFO] [BOOTSTRAP ROOT] root={root} bootstrap={__file__}")
    for module_name in [
        "excavator_app.ik_calculation",
        "excavator_app.auto_dataset_collect",
        "excavator_app.ik_movement",
        "excavator_app.trace_showing",
        "excavator_app.joint_space_planner",
        "excavator_app.excavator_dataset_camera",
    ]:
        reload_runtime(module_name)
    reload_runtime("excavator_app.sand_site_runtime")
    return reload_runtime("excavator_app.excavator_runtime")


def run_sand_site_only():
    ensure_project_root()
    return reload_runtime("excavator_app.sand_site_runtime")
