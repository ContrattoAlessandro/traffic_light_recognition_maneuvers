#!/usr/bin/env bash
# Usage: bash scripts/run_autoware.sh baseline|replace|shadow <map folder>
#   baseline, shadow: Autoware with its own traffic light recognition
#   replace:          the same Autoware with its recognition off, the R node feeds the planner
# VEHICLE_MODEL and SENSOR_MODEL override the AWSIM Labs defaults (classic AWSIM: sample_vehicle, awsim_sensor_kit).
set -eo pipefail
HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
MAP="${2:-}"
case "$MODE" in baseline|replace|shadow) ;; *) echo "mode must be baseline, replace or shadow" >&2; exit 1;; esac
[ -f "$MAP/lanelet2_map.osm" ] || { echo "$MAP/lanelet2_map.osm not found" >&2; exit 1; }
source /opt/ros/humble/setup.bash
source "$(cat "$HOME_DIR/.autoware_dir")/install/setup.bash"

EXTRA=()
if [ "$MODE" = replace ]; then
  SHARE="$(ros2 pkg prefix --share autoware_launch)/launch"
  if grep -rqs 'perception/enable_traffic_light' "$SHARE"; then
    EXTRA+=("perception/enable_traffic_light:=false")
  else
    EXTRA+=("use_traffic_light_recognition:=false")
  fi
  echo "Autoware traffic light recognition off with: ${EXTRA[*]}"
fi
exec ros2 launch autoware_launch e2e_simulator.launch.xml \
  vehicle_model:="${VEHICLE_MODEL:-awsim_labs_vehicle}" sensor_model:="${SENSOR_MODEL:-awsim_labs_sensor_kit}" \
  map_path:="$(realpath "$MAP")" launch_vehicle_interface:=true "${EXTRA[@]}"
