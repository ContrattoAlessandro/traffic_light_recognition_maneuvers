#!/usr/bin/env bash
# Usage: bash scripts/run_node.sh replace|shadow <map folder>
HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME_DIR/setup_env.sh"
exec ros2 launch tl_maneuver_perception tl_maneuver.launch.py mode:="${1:?mode: replace or shadow}" map_path:="$(realpath "${2:?map folder}")"
