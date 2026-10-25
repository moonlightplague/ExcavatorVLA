#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_dir="${EXCAVATOR_ISAAC_DIR:-/root/gpufree-data/isaacsim}"
export LD_LIBRARY_PATH="$isaac_dir/kit/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ALLOW_ROOT=1
exec "$isaac_dir/python.sh" -B "$experiment_dir/bucket_contact_isaac.py" "$@"
