import os
import sys
from urllib.parse import unquote, urlparse


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return ""


def _looks_like_project_root(root):
    return bool(root) and os.path.isfile(os.path.join(root, "excavator_app", "bootstrap.py"))


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


def _candidate_roots():
    roots = [
        os.environ.get("EXCAVATOR_PROJECT_ROOT", ""),
        os.environ.get("URDF_REAL3_ROOT", ""),
        _script_dir(),
        os.getcwd(),
    ]
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
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise RuntimeError(
        "Cannot find excavator_app/bootstrap.py. Open and run main.py from the "
        "extracted package folder, or set EXCAVATOR_PROJECT_ROOT to that folder."
    )


PROJECT_ROOT = ensure_project_root()

from excavator_app.bootstrap import run_excavator_with_sand

run_excavator_with_sand()
