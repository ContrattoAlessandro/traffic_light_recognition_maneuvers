"""Synthetic intersection: two approach lanes, four connectors, a distant light."""
import numpy as np

from dtsu.labels import build_frame, traffic_side_vote

RED, GREEN = 1, 2


def line(p0, p1, n=21):
    return np.linspace(p0, p1, n).tolist()


def arc(start, turn_deg, radius=10.0, n=21):
    """Circular arc from start, initially heading +x, turning by turn_deg (left positive)."""
    sign = np.sign(turn_deg)
    cx, cy = start[0], start[1] + sign * radius
    t = np.radians(np.linspace(0, abs(turn_deg), n))
    xs = cx + radius * np.sin(t)
    ys = cy - sign * radius * np.cos(t)
    return np.stack([xs, ys, np.zeros(n)], 1).tolist()


def frame(extra_lights=()):
    # 0: ego approach lane (y=0) -> 2 straight, 3 right
    # 1: dedicated left lane (y=3.5) -> 4 left, 5 U-turn
    # 6: lane beyond a distant intersection (no successors)
    lanes = [
        (line([-10, 0, 0], [20, 0, 0]), False),
        (line([0, 3.5, 0], [20, 3.5, 0]), False),
        (line([20, 0, 0], [40, 0, 0]), True),
        (arc([20, 0, 0], -90), True),
        (arc([20, 3.5, 0], 90), True),
        (arc([20, 3.5, 0], 180, radius=4), True),
        (line([42, 0, 0], [49, 0, 0]), False),
    ]
    lclc = np.zeros((7, 7), int)
    lclc[0, [2, 3]] = 1
    lclc[1, [4, 5]] = 1
    lights = [("ball", RED, [0]), ("left_arrow", GREEN, [1]), ("distant", GREEN, [6])] + list(extra_lights)
    lcte = np.zeros((len(lanes), len(lights)), int)
    for j, (_, _, linked) in enumerate(lights):
        lcte[linked, j] = 1
    return {
        "segment_id": "x",
        "timestamp": 0,
        "sensor": {"ring_front_center": {"image_path": "img.jpg"}},
        "annotation": {
            "lane_centerline": [{"id": str(i), "points": p, "is_intersection_or_connector": c} for i, (p, c) in enumerate(lanes)],
            "traffic_element": [{"id": n, "category": 1, "attribute": a, "points": [[0, 0], [1, 1]]} for n, a, _ in lights],
            "topology_lclc": lclc.tolist(),
            "topology_lcte": lcte.tolist(),
        },
    }


def right_hand(rec):
    """Resolve a built frame with right-hand-traffic rules, as dtsu.labels.main does."""
    return {**rec, **rec["by_side"]["right"]}


def test_direction_assignment():
    rec = right_hand(build_frame(frame()))
    assert rec["color"] == {"straight": "red", "left": "green", "right": "none", "uturn": "green"}
    assert rec["pass"] == {"straight": "stop", "left": "go", "right": "go", "uturn": "go"}
    assert [b["relevant"] for b in rec["boxes"]] == [True, True, False]  # distant light ignored
    assert rec["conflict"] == []


def test_dedicated_right_signal():
    rec = right_hand(build_frame(frame(extra_lights=[("right_arrow", GREEN, [3])])))
    assert rec["color"]["right"] == "green"
    assert rec["color"]["straight"] == "red"


def test_left_inherits_straight_without_left_lane():
    f = frame()
    lcte = np.array(f["annotation"]["topology_lcte"])
    lcte[1, 1] = 0  # left lane no longer linked to any light
    f["annotation"]["topology_lcte"] = lcte.tolist()
    rec = right_hand(build_frame(f))
    assert rec["color"]["left"] == rec["color"]["uturn"] == "red"


def test_left_hand_traffic_mirrors_rules():
    # same lights: a dedicated light on the right-turn connector, in left-hand traffic
    # the right turn crosses oncoming traffic and U-turn follows it
    rec = build_frame(frame(extra_lights=[("right_arrow", GREEN, [3])]))
    lht = rec["by_side"]["left"]["color"]
    assert lht["right"] == "green" and lht["uturn"] == "green"
    assert lht["left"] == "none"  # the near turn needs a dedicated left-only light
    assert lht["straight"] == "red"


def test_traffic_side_vote():
    def lanes_with_oncoming(y):
        return [{"pts": np.array(line([20, y, 0], [-20, y, 0])), "connector": False}]

    assert traffic_side_vote(lanes_with_oncoming(3.5)) == 1  # oncoming on the left: right-hand traffic
    assert traffic_side_vote(lanes_with_oncoming(-3.5)) == -1


def test_junction_starting_just_behind_ego_at_stop_line():
    # ego stopped at the stop line: the straight connector starts 0.2 m behind the origin and
    # leads to a lane ending in a second junction 40 m ahead; the light belongs to this junction
    lanes = [
        (line([-0.2, 0, 0], [20, 0, 0]), True),  # straight connector the ego is entering
        (line([20, 0, 0], [40, 0, 0]), False),
        (line([40, 0, 0], [49, 0, 0]), True),  # next junction
        (arc([0.6, 3.5, 0], 90), True),  # left connector starting just ahead
    ]
    lclc = np.zeros((4, 4), int)
    lclc[0, 1] = lclc[1, 2] = 1
    lcte = np.zeros((4, 2), int)
    lcte[0, 0] = 1  # ball light -> straight connector
    lcte[3, 1] = 1  # arrow light -> left connector
    info = {
        "segment_id": "x", "timestamp": 0, "sensor": {"CAM_FRONT": {"image_path": "img.jpg"}},
        "annotation": {
            "lane_centerline": [{"id": str(i), "points": p, "is_intersection_or_connector": c} for i, (p, c) in enumerate(lanes)],
            "traffic_element": [{"id": n, "category": 1, "attribute": GREEN, "points": [[0, 0], [1, 1]]} for n in ("ball", "arrow")],
            "topology_lclc": lclc.tolist(),
            "topology_lcte": lcte.tolist(),
        },
    }
    rec = right_hand(build_frame(info))
    assert rec["color"]["straight"] == "green" and rec["color"]["left"] == "green"
