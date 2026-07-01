import json
import os
from pathlib import Path


CONFIG_FILE_NAME = "excavator_config.json"
PROJECT_MARKERS = (
    CONFIG_FILE_NAME,
    "main.py",
    "scripts/excavator_app/bootstrap.py",
)


def _as_path(value):
    return Path(str(value)).expanduser()


def looks_like_project_root(path):
    root = _as_path(path)
    return all((root / marker).exists() for marker in PROJECT_MARKERS[:2]) and (
        root / PROJECT_MARKERS[2]
    ).exists()


def find_project_root(start=None):
    """Resolve the ExcavatorVLA project root without requiring a fixed mount path."""
    env_root = os.environ.get("EXCAVATOR_PROJECT_ROOT", "")
    candidates = []
    if env_root:
        candidates.append(_as_path(env_root))
    if start:
        start_path = _as_path(start)
        candidates.append(start_path if start_path.is_dir() else start_path.parent)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[1])

    seen = set()
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            parent = parent.resolve()
            if parent in seen:
                continue
            seen.add(parent)
            if looks_like_project_root(parent):
                return str(parent)
    raise FileNotFoundError(
        "Could not locate the ExcavatorVLA project root. Set EXCAVATOR_PROJECT_ROOT "
        "or run the script from inside the project checkout."
    )


def project_path(*parts, root=None):
    return str(Path(root or find_project_root()).joinpath(*parts))


def load_config(root=None):
    config_path = Path(root or find_project_root()) / CONFIG_FILE_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Excavator config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Excavator config must be a JSON object: {config_path}")
    return data


def env_path(name, default=""):
    value = os.environ.get(name, "")
    return value if value else str(default or "")


def default_scene_path(root=None):
    return project_path("assets", "usd", "excavator_scene.usd", root=root)


def default_truck_usd_path(root=None):
    return project_path("assets", "fbx", "truck", "truck.usd", root=root)


def resolve_existing_path(value, root=None):
    if not value:
        return ""
    path = _as_path(value)
    if not path.is_absolute():
        path = Path(root or find_project_root()) / path
    return str(path.resolve()) if path.exists() else str(path)
