#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ISAAC_PY="${ISAAC_PY:-/isaac-sim/python.sh}"
SCENE_PATH="${SCENE_PATH:-$PROJECT_ROOT/assets/usd/excavator_scene.usd}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/output}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/linux_camera_diag_$TS"
LOG="$OUT_DIR/linux_camera_diag.log"

mkdir -p "$OUT_DIR"

exec > >(tee -a "$LOG") 2>&1

echo "========== LINUX CAMERA DIAG START =========="
echo "timestamp=$TS"
echo "script_dir=$SCRIPT_DIR"
echo "project_root=$PROJECT_ROOT"
echo "isaac_py=$ISAAC_PY"
echo "scene_path=$SCENE_PATH"
echo "out_dir=$OUT_DIR"
echo

echo "========== SYSTEM =========="
date
uname -a || true
cat /etc/os-release || true
echo

echo "========== DISK =========="
df -h || true
echo
du -h --max-depth=2 "$PROJECT_ROOT" 2>/dev/null | sort -h | tail -50 || true
echo

echo "========== GPU =========="
nvidia-smi || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free --format=csv || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
ls -l /dev/nvidia* || true
echo

echo "========== PROCESS =========="
ps -ef | grep -Ei "isaac|kit|python" | grep -v grep || true
echo

echo "========== ENV SELECTED =========="
env | sort | grep -Ei "ISAAC|OMNI|NVIDIA|CUDA|VK|VULKAN|DISPLAY|CARB|EXCAVATOR|PYTHON|LD_LIBRARY|PATH" || true
echo

echo "========== ISAAC PYTHON BASIC =========="
"$ISAAC_PY" -c "import sys, os; print('sys.executable=', sys.executable); print('sys.version=', sys.version); print('cwd=', os.getcwd())" || true
echo

echo "========== RUN PYTHON DIAG =========="
export CAMERA_DIAG_OUT_DIR="$OUT_DIR"
export CAMERA_DIAG_SCENE="$SCENE_PATH"
export CAMERA_DIAG_HEADLESS="${CAMERA_DIAG_HEADLESS:-1}"
export CAMERA_DIAG_ACTIVE_GPU="${CAMERA_DIAG_ACTIVE_GPU:-0}"
export CAMERA_DIAG_PHYSICS_GPU="${CAMERA_DIAG_PHYSICS_GPU:-0}"
export CAMERA_DIAG_RENDERER="${CAMERA_DIAG_RENDERER:-RayTracedLighting}"

"$ISAAC_PY" "$SCRIPT_DIR/linux_camera_diag.py"
RC=$?

echo
echo "========== PYTHON DIAG EXIT CODE =========="
echo "$RC"

echo
echo "========== OUTPUT TREE =========="
find "$OUT_DIR" -maxdepth 3 -type f -printf "%p %s bytes\n" | sort || true

echo
echo "========== IMAGE FILE TYPES =========="
if command -v file >/dev/null 2>&1; then
  file "$OUT_DIR"/*.ppm "$OUT_DIR"/*.png 2>/dev/null || true
fi

echo
echo "========== PACKAGE =========="
cd "$(dirname "$OUT_DIR")" || exit 1
tar -czf "$OUT_DIR.tar.gz" "$(basename "$OUT_DIR")"
echo "tarball=$OUT_DIR.tar.gz"
echo "========== LINUX CAMERA DIAG DONE =========="
exit "$RC"

