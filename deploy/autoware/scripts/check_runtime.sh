#!/usr/bin/env bash
# Run while AWSIM, Autoware and the node are up. Usage: bash scripts/check_runtime.sh replace|shadow
HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$(cat "$HOME_DIR/.autoware_dir")/install/setup.bash"
MODE="${1:-replace}"
FINAL=/perception/traffic_light_recognition/traffic_signals
SHADOW=/perception/traffic_light_recognition/maneuver/traffic_signals
CAMERA=/sensing/camera/traffic_light/image_raw

echo "== Camera rate (5 s). No output means AWSIM's traffic light camera is off"
timeout 5 ros2 topic hz "$CAMERA" | tail -3
echo "== Encoding and size"
timeout 5 ros2 topic echo --once --field encoding "$CAMERA" | head -1
timeout 5 ros2 topic echo --once --field width "$CAMERA" | head -1
timeout 5 ros2 topic echo --once --field height "$CAMERA" | head -1
echo "== Publishers of the planner's input"
ros2 topic info -v "$FINAL" | grep -E "Node name|Publisher count"
if [ "$MODE" = replace ]; then
  echo "Expected in replace mode: exactly one publisher, tl_maneuver_perception"
else
  echo "== Publishers of the shadow topic"
  ros2 topic info -v "$SHADOW" | grep -E "Node name|Publisher count"
fi
echo "== Route"
timeout 5 ros2 topic echo --once --field segments /planning/mission_planning/route --qos-durability transient_local --qos-reliability reliable | head -c 200; echo
echo "== Last decision of the node (needs a route and a stop line within activation_distance)"
timeout 5 ros2 topic echo --once --field data /tl_maneuver_perception/debug/decision | head -c 800; echo
