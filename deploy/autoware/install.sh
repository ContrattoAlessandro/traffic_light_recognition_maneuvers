#!/usr/bin/env bash
# Offline installation of the R node. Usage: bash install.sh [Autoware workspace, default ~/autoware]
set -euo pipefail
HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOWARE_DIR="${1:-$HOME/autoware}"
ROS_SETUP=/opt/ros/humble/setup.bash

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$ROS_SETUP" ] || fail "$ROS_SETUP not found: this delivery targets ROS 2 Humble (Ubuntu 22.04)"
[ -f "$AUTOWARE_DIR/install/setup.bash" ] || fail "$AUTOWARE_DIR/install/setup.bash not found: pass the Autoware workspace as first argument"
[ "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.10" ] || fail "Python 3.10 is required (the wheels are built for it)"
python3 -m pip --version > /dev/null || fail "pip is missing: sudo apt install python3-pip"
case "$(findmnt -n -o FSTYPE --target "$HOME_DIR" || true)" in
  vfat|exfat|ntfs|fuseblk) fail "Copy this folder from the USB key to the Linux disk first, then run install.sh there";;
esac

# Autoware's setup scripts read unset variables, so they run without -u
set +u
source "$ROS_SETUP"
source "$AUTOWARE_DIR/install/setup.bash"
set -u
echo "$AUTOWARE_DIR" > "$HOME_DIR/.autoware_dir"

echo "[1/3] Checking the Autoware messages"
python3 - <<'EOF' || fail "This Autoware has no TrafficLightGroupArray (older than June 2024): see README, section Limits"
from autoware_perception_msgs.msg import TrafficLightElement, TrafficLightGroup, TrafficLightGroupArray
from autoware_planning_msgs.msg import LaneletRoute
assert hasattr(TrafficLightGroup(), "traffic_light_group_id") and hasattr(TrafficLightGroupArray(), "traffic_light_groups")
EOF

echo "[2/3] Installing the Python runtime and the CUDA libraries in vendor/ (nothing outside this folder changes)"
rm -rf "$HOME_DIR/vendor"
python3 -m pip install --quiet --no-index --find-links "$HOME_DIR/wheels" --target "$HOME_DIR/vendor" -r "$HOME_DIR/requirements.txt"

echo "[3/3] Building the ROS 2 package"
cd "$HOME_DIR/ros2_ws"
colcon build --symlink-install

set +u
source "$HOME_DIR/setup_env.sh"
set -u
python3 "$HOME_DIR/scripts/self_test.py"
echo "Installation finished. Next step: README.md, section Run."
