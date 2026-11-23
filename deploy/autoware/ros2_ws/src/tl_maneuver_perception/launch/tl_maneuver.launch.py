"""Start the R node.

mode:=replace  publishes on the topic the planner reads (start Autoware with its traffic light recognition off).
mode:=shadow   publishes on .../maneuver/traffic_signals: Autoware keeps driving with its own recognition.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

TOPICS = {"replace": "/perception/traffic_light_recognition/traffic_signals",
          "shadow": "/perception/traffic_light_recognition/maneuver/traffic_signals"}


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    mode = get("mode")
    if mode not in TOPICS:
        raise RuntimeError(f"mode must be replace or shadow, not {mode}")
    home = get("home")
    map_path = get("map_path")
    if os.path.isdir(map_path):
        map_path = os.path.join(map_path, "lanelet2_map.osm")
    params = os.path.join(get_package_share_directory("tl_maneuver_perception"), "config", "tl_maneuver.param.yaml")
    return [Node(package="tl_maneuver_perception", executable="tl_maneuver_node", name="tl_maneuver_perception", output="screen",
                 parameters=[params, {"model_path": os.path.join(home, "model", "r_1280_opset17.onnx"),
                                      "config_path": os.path.join(home, "model", "model_config.json"),
                                      "map_path": map_path, "output_topic": TOPICS[mode], "use_sim_time": get("use_sim_time") == "true"}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_path", description="Map folder or lanelet2_map.osm, the same map given to Autoware"),
        DeclareLaunchArgument("mode", default_value="replace"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("home", default_value=os.environ.get("TL_MANEUVER_HOME", ""), description="Folder of this delivery"),
        OpaqueFunction(function=setup),
    ])
