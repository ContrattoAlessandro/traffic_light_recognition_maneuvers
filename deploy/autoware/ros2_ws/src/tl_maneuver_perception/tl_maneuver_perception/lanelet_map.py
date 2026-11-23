"""Minimal Lanelet2 (.osm) reader: lanelets, turn directions and traffic light regulatory elements. No ROS import.

Autoware looks a signal up by the id of the traffic light regulatory element, so that id is what this module exposes.
Coordinates come from the local_x / local_y node tags when present, otherwise from lat / lon through the MGRS grid
(UTM modulo 100 km), which is what Autoware's MGRS projector gives.
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def latlon_to_mgrs_local(lat: float, lon: float) -> Tuple[float, float]:
    """WGS84 lat / lon in degrees -> x, y inside the 100 km MGRS square (Krueger series, zone from the longitude)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    zone = int((lon + 180) // 6) + 1
    lon0 = math.radians(6 * zone - 183)
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    n = f / (2 - f)
    big_a = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    c = 2 * math.sqrt(n) / (1 + n)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi, eta = math.atan2(t, math.cos(lam)), math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    sx = sum(al * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, al in enumerate(alpha, 1))
    se = sum(al * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, al in enumerate(alpha, 1))
    easting = 500000.0 + k0 * big_a * (eta + se)
    northing = k0 * big_a * (xi + sx) + (10000000.0 if lat < 0 else 0.0)
    return easting % 100000.0, northing % 100000.0


@dataclass
class Lanelet:
    id: int
    centerline: np.ndarray  # [N, 2]
    turn_direction: Optional[str] = None
    regulatory_elements: List[int] = field(default_factory=list)


@dataclass
class TrafficLightRegulation:
    id: int
    stop_line: np.ndarray  # [N, 2]


def _centerline(left: np.ndarray, right: np.ndarray, samples: int = 30) -> np.ndarray:
    def resample(line):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        t = np.linspace(0.0, s[-1], samples)
        return np.c_[np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])]

    return (resample(left) + resample(right)) / 2.0


class LaneletMap:
    def __init__(self, path: str):
        root = ET.parse(path).getroot()
        points: Dict[int, Tuple[float, float]] = {}
        for node in root.iter("node"):
            tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
            if "local_x" in tags and "local_y" in tags:
                points[int(node.get("id"))] = (float(tags["local_x"]), float(tags["local_y"]))
            else:
                points[int(node.get("id"))] = latlon_to_mgrs_local(float(node.get("lat")), float(node.get("lon")))
        ways = {int(w.get("id")): np.array([points[int(n.get("ref"))] for n in w.findall("nd")], dtype=np.float64) for w in root.iter("way")}

        self.lanelets: Dict[int, Lanelet] = {}
        self.traffic_lights: Dict[int, TrafficLightRegulation] = {}
        relations = list(root.iter("relation"))
        for rel in relations:
            tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
            if tags.get("type") == "regulatory_element" and tags.get("subtype") == "traffic_light":
                stop = [ways[int(m.get("ref"))] for m in rel.findall("member") if m.get("role") == "ref_line" and int(m.get("ref")) in ways]
                if stop:
                    self.traffic_lights[int(rel.get("id"))] = TrafficLightRegulation(int(rel.get("id")), stop[0])
        for rel in relations:
            tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
            if tags.get("type") != "lanelet":
                continue
            bound = {m.get("role"): int(m.get("ref")) for m in rel.findall("member") if m.get("type") == "way"}
            if "left" not in bound or "right" not in bound or bound["left"] not in ways or bound["right"] not in ways:
                continue
            left, right = ways[bound["left"]], ways[bound["right"]]
            if np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]) > np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0]):
                right = right[::-1]  # a Lanelet2 bound may be stored against the driving direction
            regs = [int(m.get("ref")) for m in rel.findall("member") if m.get("role") == "regulatory_element"]
            self.lanelets[int(rel.get("id"))] = Lanelet(int(rel.get("id")), _centerline(left, right), tags.get("turn_direction"),
                                                        [r for r in regs if r in self.traffic_lights])
