# Source this file only in the terminal that runs the R node: its Python and CUDA libraries stay out of Autoware.
export TL_MANEUVER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOWARE_DIR="$(cat "$TL_MANEUVER_HOME/.autoware_dir")"
source /opt/ros/humble/setup.bash
source "$AUTOWARE_DIR/install/setup.bash"
[ -f "$TL_MANEUVER_HOME/ros2_ws/install/setup.bash" ] && source "$TL_MANEUVER_HOME/ros2_ws/install/setup.bash"
V="$TL_MANEUVER_HOME/vendor"
export PYTHONPATH="$V:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$V/nvidia/cuda_runtime/lib:$V/nvidia/cuda_nvrtc/lib:$V/nvidia/cublas/lib:$V/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
unset V
