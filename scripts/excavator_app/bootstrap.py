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
        if os.path.isfile(os.path.join(root, "excavator_app", "bootstrap.py")) and root not in sys.path:
            sys.path.insert(0, root)


def reload_runtime(module_name):
    ensure_project_root()
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def run_excavator_with_sand():
    ensure_project_root()
    reload_runtime("excavator_app.sand_site_runtime")
    return reload_runtime("excavator_app.excavator_runtime")


def run_sand_site_only():
    ensure_project_root()
    return reload_runtime("excavator_app.sand_site_runtime")
