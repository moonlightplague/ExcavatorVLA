#!/bin/bash
# ExcavatorVLA Launcher Script
# ============================
# This script launches the ExcavatorVLA simulation with Isaac Sim

set -e

ISAAC_SIM_DIR="/isaac-sim"
PROJECT_DIR="${ISAAC_SIM_DIR}/ExcavatorVLA"

echo "=============================================="
echo "ExcavatorVLA Launcher"
echo "=============================================="
echo "Isaac Sim Directory: ${ISAAC_SIM_DIR}"
echo "Project Directory: ${PROJECT_DIR}"
echo "=============================================="

# Check if Isaac Sim exists
if [ ! -d "${ISAAC_SIM_DIR}" ]; then
    echo "ERROR: Isaac Sim not found at ${ISAAC_SIM_DIR}"
    exit 1
fi

# Check if project exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: ExcavatorVLA project not found at ${PROJECT_DIR}"
    exit 1
fi

# Check if scene file exists
SCENE_FILE="${PROJECT_DIR}/assets/usd/excavator_scene.usd"
if [ ! -f "${SCENE_FILE}" ]; then
    echo "WARNING: Scene file not found: ${SCENE_FILE}"
    echo "The simulation may not work correctly without the scene."
fi

# Run the standalone script
cd "${PROJECT_DIR}"
echo "Starting ExcavatorVLA simulation..."
echo ""

# Run with Isaac Sim's Python
"${ISAAC_SIM_DIR}/python.sh" "${PROJECT_DIR}/run_excavator_standalone.py" "$@"