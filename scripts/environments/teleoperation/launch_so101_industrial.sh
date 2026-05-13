#!/usr/bin/env bash
# Launch SO101 NutThread teleoperation (LeIsaac-SO101-NutThread-v0)
# Usage: bash launch_so101_industrial.sh [--teleop_device keyboard|gamepad] [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEISAAC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ISAAC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
TELEOP_SCRIPT="${SCRIPT_DIR}/teleop_se3_agent.py"

LEISAAC_ASSETS_ROOT=/workspace/leisaac/assets \
exec "${ISAAC_PYTHON}" "${TELEOP_SCRIPT}" \
    --task LeIsaac-SO101-NutThread-v0 \
    --teleop_device keyboard \
    --device cuda \
    --livestream 2 \
    --enable_cameras \
    --num_envs 1 \
    "$@"
