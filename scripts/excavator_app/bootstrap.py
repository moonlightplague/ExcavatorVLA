import importlib
import os
import sys


APP_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_PARENT) if os.path.basename(PACKAGE_PARENT) == "scripts" else PACKAGE_PARENT


def import_roots(root):
    roots = [root, os.path.join(root, "scripts")]
    out = []
    for candidate in roots:
        candidate = os.path.abspath(candidate)
        if candidate not in out:
            out.append(candidate)
    return out


def has_excavator_app(root):
    return (
        os.path.isfile(os.path.join(root, "scripts", "excavator_app", "bootstrap.py"))
        or os.path.isfile(os.path.join(root, "excavator_app", "bootstrap.py"))
    )


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
        if has_excavator_app(root):
            for import_root in reversed(import_roots(root)):
                if import_root not in sys.path:
                    sys.path.insert(0, import_root)


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
