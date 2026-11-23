"""Which traffic light the ego vehicle must obey next, and which maneuver it will perform there. No ROS import."""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np

from .lanelet_map import LaneletMap

SAME_STOP_LINE = 2.0  # m: signals this close along the route are one stop line
TURN_LOOKAHEAD = 80.0  # m after the stop line in which the turn_direction of the route is searched
MAX_JUMP_BACK, MAX_JUMP_AHEAD = 10.0, 60.0  # m: window of the ego projection after the first localization
MAX_OFFSET = 8.0  # m: farther than this from the route inside that window, the projection restarts on the whole route


@dataclass
class RouteSignal:
    regulation_ids: Set[int]  # every regulatory element of the road segment at this stop line
    stop_s: float
    maneuver: str  # straight, left or right


@dataclass
class ActiveSignal:
    regulation_ids: Set[int]
    maneuver: str
    distance: float


class RouteSignals:
    """The route as one polyline of arc length s, with its signalized stop lines in driving order."""

    def __init__(self, lanelet_map: LaneletMap, preferred_ids: Sequence[int], segment_ids: Sequence[Sequence[int]]):
        self.points = np.zeros((0, 2))
        self.signals: List[RouteSignal] = []
        self.ego_s: Optional[float] = None
        spans, pts = [], []
        length = 0.0
        for lane_id in preferred_ids:
            if lane_id not in lanelet_map.lanelets:
                raise KeyError(f"lanelet {lane_id} of the route is not in the map file: is this the map Autoware is using?")
            c = lanelet_map.lanelets[lane_id].centerline
            if pts:
                c = np.r_[[pts[-1][-1]], c]
            pts.append(c if not spans else c[1:])
            seg = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
            spans.append((lane_id, length, length + seg))
            length += seg
        if not pts:
            return
        self.points = np.concatenate(pts)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]

        for (lane_id, start, end), segment in zip(spans, segment_ids):
            regs = lanelet_map.lanelets[lane_id].regulatory_elements
            if not regs:
                continue
            stop_mid = lanelet_map.traffic_lights[regs[0]].stop_line.mean(axis=0)
            stop_s = self._project(stop_mid, start - 40.0, end + 40.0)
            ids = {r for other in segment if other in lanelet_map.lanelets for r in lanelet_map.lanelets[other].regulatory_elements}
            ids.update(regs)
            self.signals.append(RouteSignal(ids, stop_s, self._maneuver(lanelet_map, spans, stop_s)))
        self.signals.sort(key=lambda x: x.stop_s)
        merged: List[RouteSignal] = []
        for sig in self.signals:
            if merged and sig.stop_s - merged[-1].stop_s < SAME_STOP_LINE:
                merged[-1].regulation_ids |= sig.regulation_ids
            else:
                merged.append(sig)
        self.signals = merged

    def _project(self, xy: np.ndarray, s_min: float, s_max: float) -> float:
        """Arc length of the polyline point nearest to xy, with s inside [s_min, s_max]."""
        a, b = self.points[:-1], self.points[1:]
        keep = (self.s[1:] >= s_min) & (self.s[:-1] <= s_max)
        if not keep.any():
            keep = np.ones(len(a), dtype=bool)
        a, b, s0 = a[keep], b[keep], self.s[:-1][keep]
        d = b - a
        den = np.maximum((d * d).sum(axis=1), 1e-9)
        t = np.clip(((xy - a) * d).sum(axis=1) / den, 0.0, 1.0)
        dist = np.linalg.norm(a + t[:, None] * d - xy, axis=1)
        i = int(np.argmin(dist))
        return float(s0[i] + t[i] * np.linalg.norm(d[i]))

    def _point(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.points[:, 0]), np.interp(s, self.s, self.points[:, 1])])

    @staticmethod
    def _maneuver(lanelet_map, spans, stop_s) -> str:
        for lane_id, start, end in spans:
            if end <= stop_s + 0.5 or start >= stop_s + TURN_LOOKAHEAD:
                continue
            turn = lanelet_map.lanelets[lane_id].turn_direction
            if turn in ("straight", "left", "right"):
                return turn
        return "straight"

    def update(self, x: float, y: float, passed_margin: float, activation_distance: float) -> Optional[ActiveSignal]:
        """The first stop line ahead of the ego within activation_distance; None when there is none."""
        if len(self.points) < 2:
            return None
        xy = np.array([x, y])
        if self.ego_s is not None:
            s = self._project(xy, self.ego_s - MAX_JUMP_BACK, self.ego_s + MAX_JUMP_AHEAD)
            near = np.linalg.norm(self._point(s) - xy) <= MAX_OFFSET
        if self.ego_s is None or not near:  # first pose, or the vehicle was moved (new initial pose)
            s = self._project(xy, -np.inf, np.inf)
        self.ego_s = s
        for sig in self.signals:
            distance = sig.stop_s - self.ego_s
            if distance > -passed_margin:
                return ActiveSignal(sig.regulation_ids, sig.maneuver, distance) if distance <= activation_distance else None
        return None

    def ahead_ids(self, passed_margin: float) -> Set[int]:
        """Every regulation of the stop lines not passed yet."""
        if self.ego_s is None:
            return set()
        return {r for sig in self.signals if sig.stop_s - self.ego_s > -passed_margin for r in sig.regulation_ids}
