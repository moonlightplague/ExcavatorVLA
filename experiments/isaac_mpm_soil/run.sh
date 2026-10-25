#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_dir="${EXCAVATOR_ISAAC_DIR:-/root/gpufree-data/isaacsim}"
if [[ ! -x "$isaac_dir/python.sh" ]]; then
    printf 'Isaac Python not found at %s; set EXCAVATOR_ISAAC_DIR.\n' "$isaac_dir" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$isaac_dir/kit/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ALLOW_ROOT=1
exec "$isaac_dir/python.sh" -B "$experiment_dir/soil_drop_isaac.py" "$@"
