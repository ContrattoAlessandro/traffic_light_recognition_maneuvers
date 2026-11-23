"""ROS 2 node: traffic light camera -> R -> color of the ego's maneuver -> Autoware TrafficLightGroupArray."""
import json
import time
from typing import Optional

import numpy as np
import rclpy
from autoware_perception_msgs.msg import TrafficLightElement, TrafficLightGroup, TrafficLightGroupArray
from autoware_planning_msgs.msg import LaneletRoute
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .decision_filter import DecisionFilter
from .inference import ManeuverModel, directions
from .lanelet_map import LaneletMap
from .route_signals import RouteSignals

COLOR = {"red": TrafficLightElement.RED, "yellow": TrafficLightElement.AMBER, "green": TrafficLightElement.GREEN}


def to_rgb(msg: Image) -> np.ndarray:
    """RGB uint8 array (H, W, 3) from a sensor_msgs/Image, without cv_bridge."""
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    data = np.frombuffer(msg.data, dtype=np.uint8)
    row = msg.width * channels
    if msg.step == row:
        img = data.reshape(msg.height, msg.width, channels)
    else:
        img = data.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, channels)
    return img[..., 2::-1].copy() if msg.encoding.startswith("bgr") else img[..., :3].copy()


def group(regulation_id: int, color: str, confidence: float) -> TrafficLightGroup:
    """One signal group with one circle of the maneuver's color: the decision for the lane is already taken here."""
    element = TrafficLightElement()
    element.color = COLOR.get(color, TrafficLightElement.UNKNOWN)
    element.shape = TrafficLightElement.CIRCLE if color in COLOR else TrafficLightElement.UNKNOWN
    element.status = TrafficLightElement.SOLID_ON if color in COLOR else TrafficLightElement.UNKNOWN
    element.confidence = float(confidence)
    out = TrafficLightGroup()
    out.traffic_light_group_id = int(regulation_id)
    out.elements = [element]
    return out


class TlManeuverNode(Node):
    def __init__(self):
        super().__init__("tl_maneuver_perception")
        p = self.declare_parameters("", [
            ("model_path", ""), ("config_path", ""), ("map_path", ""), ("use_gpu", True), ("gpu_id", 0), ("threshold", -1.0),
            ("image_topic", "/sensing/camera/traffic_light/image_raw"), ("output_topic", "/perception/traffic_light_recognition/traffic_signals"),
            ("route_topic", "/planning/mission_planning/route"), ("odometry_topic", "/localization/kinematic_state"),
            ("activation_distance", 120.0), ("passed_margin", 0.5), ("go_confirm_frames", 3), ("miss_hold_frames", 5),
        ])
        v = {x.name: x.value for x in p}
        self.v = v
        if not v["model_path"] or not v["config_path"] or not v["map_path"]:
            raise RuntimeError("model_path, config_path and map_path are required")
        self.model = ManeuverModel(v["model_path"], v["config_path"], v["use_gpu"], v["gpu_id"], v["threshold"])
        if v["use_gpu"] and self.model.provider != "CUDAExecutionProvider":
            self.get_logger().error("CUDA is not available to onnxruntime: running on CPU, too slow for closed-loop driving")
        self.map = LaneletMap(v["map_path"])
        self.get_logger().info(f"{self.model.provider}, threshold {self.model.threshold}; map: {len(self.map.lanelets)} lanelets, "
                               f"{len(self.map.traffic_lights)} traffic light regulatory elements")
        self.filter = DecisionFilter(v["go_confirm_frames"], v["miss_hold_frames"])
        self.route: Optional[RouteSignals] = None
        self.pose = None
        self.active_key = None

        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaneletRoute, v["route_topic"], self.on_route, latched)
        self.create_subscription(Odometry, v["odometry_topic"], self.on_odometry, 1)
        camera = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=qos_profile_sensor_data.reliability, durability=qos_profile_sensor_data.durability)
        self.create_subscription(Image, v["image_topic"], self.on_image, camera)
        self.publisher = self.create_publisher(TrafficLightGroupArray, v["output_topic"], 1)
        self.debug = self.create_publisher(String, "~/debug/decision", 10)

    def on_route(self, msg: LaneletRoute) -> None:
        preferred = [s.preferred_primitive.id for s in msg.segments]
        segments = [[p.id for p in s.primitives] for s in msg.segments]
        try:
            self.route = RouteSignals(self.map, preferred, segments)
        except KeyError as e:
            self.route = None
            self.get_logger().error(str(e))
            return
        self.filter.reset()
        self.active_key = None
        self.get_logger().info(f"route with {len(self.route.signals)} signalized stop lines: " +
                               ", ".join(f"{s.maneuver}@{s.stop_s:.0f}m" for s in self.route.signals))

    def on_odometry(self, msg: Odometry) -> None:
        self.pose = msg.pose.pose.position

    def on_image(self, msg: Image) -> None:
        if self.route is None or self.pose is None:
            return
        start = time.perf_counter()
        margin = self.v["passed_margin"]
        active = self.route.update(self.pose.x, self.pose.y, margin, self.v["activation_distance"])
        groups = {r: group(r, "none", 0.0) for r in self.route.ahead_ids(margin)}
        info = {"active": False}
        if active is not None:
            key = frozenset(active.regulation_ids)
            if key != self.active_key:
                self.filter.reset()
                self.active_key = key
            detections = self.model.detect(to_rgb(msg))
            dirs = directions(detections)
            raw, confidence = dirs[active.maneuver]
            color = self.filter.update(raw)
            for r in active.regulation_ids:
                groups[r] = group(r, color, confidence)
            info = {"active": True, "regulations": sorted(active.regulation_ids), "maneuver": active.maneuver, "distance_m": round(active.distance, 2),
                    "raw_color": raw, "published_color": color, "directions": {k: c for k, (c, _) in dirs.items()},
                    "detections": [d._asdict() for d in detections]}
        out = TrafficLightGroupArray()
        out.stamp = msg.header.stamp
        out.traffic_light_groups = list(groups.values())
        self.publisher.publish(out)
        info.update(stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, processing_ms=round((time.perf_counter() - start) * 1e3, 1))
        self.debug.publish(String(data=json.dumps(info)))


def main():
    rclpy.init()
    node = TlManeuverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
