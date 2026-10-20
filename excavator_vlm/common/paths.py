"""Repository locations independent of the caller's current working directory."""
from pathlib import Path

SUBMODULE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SUBMODULE.parent
CONFIGS = SUBMODULE / 'configs'
SCRIPTS = SUBMODULE / 'scripts'
