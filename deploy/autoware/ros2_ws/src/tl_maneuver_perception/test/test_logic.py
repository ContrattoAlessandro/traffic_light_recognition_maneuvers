"""Tests of the logic without ROS or the model: python3 test/test_logic.py from the package folder."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tl_maneuver_perception.decision_filter import DecisionFilter  # noqa: E402
from tl_maneuver_perception.inference import Detection, directions, nms, rule  # noqa: E402
from tl_maneuver_perception.lanelet_map import LaneletMap, latlon_to_mgrs_local  # noqa: E402
from tl_maneuver_perception.route_signals import RouteSignals  # noqa: E402


def write_map(path):
    """Road along +x. Lanelet 10 (x 0-100, signal 500, stop line at x=95), then 11 (straight) or 12 (left), then 13, with signal 600 at x=175."""
    nodes, ways, next_node = [], [], [1]

    def line(way_id, pts, extra=""):
        refs = []
        for x, y in pts:
            n = next_node[0]
            next_node[0] += 1
            nodes.append(f"<node id='{n}' lat='0' lon='0'><tag k='local_x' v='{x}'/><tag k='local_y' v='{y}'/></node>")
            refs.append(f"<nd ref='{n}'/>")
        ways.append(f"<way id='{way_id}'>{''.join(refs)}{extra}</way>")

    def lanelet(rel, left, right, tags="", reg=None):
        member = f"<member type='relation' role='regulatory_element' ref='{reg}'/>" if reg else ""
        return (f"<relation id='{rel}'><member type='way' role='left' ref='{left}'/><member type='way' role='right' ref='{right}'/>{member}"
                f"<tag k='type' v='lanelet'/>{tags}</relation>")

    def regulation(rel, tl_way, stop_way):
        return (f"<relation id='{rel}'><member type='way' role='refers' ref='{tl_way}'/><member type='way' role='ref_line' ref='{stop_way}'/>"
                "<tag k='type' v='regulatory_element'/><tag k='subtype' v='traffic_light'/></relation>")

    line(1, [(0, 2), (100, 2)]), line(2, [(100, 0), (0, 0)])  # right bound stored against the driving direction
    line(3, [(100, 2), (130, 2)]), line(4, [(100, 0), (130, 0)])
    line(5, [(100, 2), (130, 30)]), line(6, [(100, 0), (130, 28)])
    line(7, [(130, 2), (200, 2)]), line(8, [(130, 0), (200, 0)])
    line(20, [(95, 0), (95, 2)]), line(21, [(110, 5), (110, 7)], "<tag k='type' v='traffic_light'/>")
    line(22, [(175, 0), (175, 2)]), line(23, [(190, 5), (190, 7)], "<tag k='type' v='traffic_light'/>")
    rel = [regulation(500, 21, 20), regulation(600, 23, 22), lanelet(10, 1, 2, reg=500),
           lanelet(11, 3, 4, "<tag k='turn_direction' v='straight'/>"), lanelet(12, 5, 6, "<tag k='turn_direction' v='left'/>"), lanelet(13, 7, 8, reg=600)]
    with open(path, "w") as f:
        f.write(f"<?xml version='1.0'?><osm>{''.join(nodes)}{''.join(ways)}{''.join(rel)}</osm>")


def test_rule():
    assert rule([])["straight"] == ("none", 0.0)
    r = rule([("circle", "red", 0.9), ("arrow_left", "green", 0.8)])
    assert r["straight"][0] == "red" and r["right"][0] == "red" and r["left"][0] == "green"
    r = rule([("circle", "green", 0.5), ("circle", "red", 0.9)])
    assert r["straight"] == ("red", 0.9)
    r = rule([("arrow_straight_right", "green", 0.7), ("unknown", "red", 0.6)])
    assert r["straight"][0] == "green" and r["right"][0] == "green" and r["left"][0] == "red"
    d = [Detection("circle", "red", False, 0.9, (0, 0, 1, 1)), Detection("circle", "green", True, 0.5, (0, 0, 1, 1))]
    assert directions(d)["straight"] == ("green", 0.5)  # lights outside the junction do not count


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.5, 0.9, 0.7]), 0.5).tolist() == [1, 2]


def test_filter():
    f = DecisionFilter(3, 2)
    assert [f.update(c) for c in ["green", "green", "green"]] == ["none", "none", "green"]
    assert [f.update(c) for c in ["none", "none", "none"]] == ["green", "green", "none"]
    f.reset()
    assert [f.update(c) for c in ["red", "green", "green", "yellow", "green"]] == ["red", "red", "red", "yellow", "yellow"]
    f = DecisionFilter(1, 0)
    assert [f.update(c) for c in ["green", "none", "red", "green"]] == ["green", "none", "red", "green"]


def test_route():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "map.osm")
        write_map(path)
        m = LaneletMap(path)
    assert set(m.traffic_lights) == {500, 600} and m.lanelets[10].regulatory_elements == [500] and m.lanelets[12].turn_direction == "left"
    assert abs(m.lanelets[10].centerline[0, 1] - 1.0) < 1e-6  # the reversed right bound is handled

    r = RouteSignals(m, [10, 11, 13], [[10], [11], [13]])
    assert [(s.regulation_ids, s.maneuver) for s in r.signals] == [({500}, "straight"), ({600}, "straight")]
    assert abs(r.signals[0].stop_s - 95) < 1 and abs(r.signals[1].stop_s - 175) < 1
    a = r.update(10.0, 1.0, 0.5, 120.0)
    assert a.regulation_ids == {500} and abs(a.distance - 85) < 1 and r.ahead_ids(0.5) == {500, 600}
    a = r.update(95.2, 1.0, 0.5, 120.0)
    assert a.regulation_ids == {500}  # within the margin the line still counts
    a = r.update(96.0, 1.0, 0.5, 120.0)
    assert a.regulation_ids == {600} and r.ahead_ids(0.5) == {600}
    assert r.update(176.0, 1.0, 0.5, 120.0) is None and r.ahead_ids(0.5) == set()  # 80 m in one step: found again on the whole route
    assert r.update(10.0, 1.0, 0.5, 120.0).regulation_ids == {500}  # the vehicle was moved back to the start

    r = RouteSignals(m, [10, 12], [[10], [12]])
    assert r.signals[0].maneuver == "left"
    assert RouteSignals(m, [10, 11, 13], [[10], [11], [13]]).update(10.0, 1.0, 0.5, 50.0) is None  # the stop line is farther than the activation distance

    r = RouteSignals(m, [10, 11], [[10, 13], [11]])  # the regulations of the whole road segment are included
    assert r.signals[0].regulation_ids == {500, 600}


def test_mgrs():
    x, y = latlon_to_mgrs_local(35.6895, 139.6917)
    assert 0 <= x < 100000 and 0 <= y < 100000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
