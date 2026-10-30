"""DTLD and LISA direction mapping on synthetic frames."""
from dtsu import labels_dtld, labels_lisa


def lamp(uid, pic, state, relevant=True, h=30, direction="front"):
    return {"unique_id": uid, "x": 10 * uid, "y": 0, "w": 10, "h": h,
            "attributes": {"pictogram": pic, "state": state, "direction": direction,
                           "relevance": "relevant" if relevant else "not_relevant"}}


def dtld(*labels, rule_ok=None):
    im = {"image_path": "./City/City1/2015-01-01_00-00-00/DE_X_k0.tiff", "time_stamp": 1.5, "labels": list(labels)}
    return labels_dtld.build_frame(im, rule_ok)


def test_dtld_circle_governs_all_directions():
    # German rules: no right turn on a red circle
    rec = dtld(lamp(0, "circle", "red"), lamp(1, "circle", "red"))
    assert rec["color"] == {"straight": "red", "left": "red", "right": "red", "uturn": "red"}
    assert rec["kept"] and rec["segment_id"] == "City/City1/2015-01-01_00-00-00" and rec["image"] == "DE_X_k0.jpg"


def test_dtld_non_relevant_left_arrow_at_same_intersection_sets_left():
    # ego goes straight under a green circle; the red left arrow of the next lane is not "relevant"
    rec = dtld(lamp(0, "circle", "green"), lamp(1, "arrow_left", "red", relevant=False))
    assert rec["color"] == {"straight": "green", "left": "red", "right": "green", "uturn": "red"}
    assert [b["relevant"] for b in rec["boxes"]] == [True, True]


def test_dtld_small_non_relevant_lamps_are_another_intersection():
    rec = dtld(lamp(0, "circle", "green"), lamp(1, "arrow_left", "red", relevant=False, h=10))
    assert rec["color"]["left"] == "green"
    assert [b["relevant"] for b in rec["boxes"]] == [True, False]


def test_dtld_left_lane_with_unused_circles():
    # ego in the left-turn lane: arrow relevant, circles serve straight
    rec = dtld(lamp(0, "arrow_left", "green"), lamp(1, "circle", "red", relevant=False))
    assert rec["color"] == {"straight": "red", "left": "green", "right": "red", "uturn": "green"}


def test_dtld_right_arrow_overrides_circle():
    rec = dtld(lamp(0, "circle", "red_yellow"), lamp(1, "arrow_right", "green", relevant=False),
               lamp(2, "pedestrian", "green"), lamp(3, "circle", "unknown", direction="left"))
    assert rec["color"] == {"straight": "red", "left": "red", "right": "green", "uturn": "red"}
    assert rec["pass"]["straight"] == "stop" and len(rec["boxes"]) == 2
    # left-turn-only approach: no circle, no right arrow -> no signal for right
    assert dtld(lamp(0, "arrow_left", "red"))["color"]["right"] == "none"


def test_dtld_drops_conflict_unknown_and_rule_mismatch():
    assert dtld(lamp(0, "circle", "red"), lamp(1, "circle", "green"))["conflict"] == ["straight", "left", "right", "uturn"]
    rec = dtld(lamp(0, "circle", "off"))
    assert rec["unknown"] == ["straight", "left", "right", "uturn"] and not rec["kept"]
    assert not dtld(lamp(0, "circle", "red"), rule_ok=False)["kept"]
    assert not dtld(lamp(0, "pedestrian", "red"))["kept"]  # no relevant vehicle lamp


def lisa(*lights):
    return labels_lisa.build_frame("dayClip1", "00000", "img.jpg", list(lights))


def test_lisa_ball_and_left_arrow():
    rec = lisa(("go", [0, 0, 10, 30]), ("stopLeft", [20, 0, 30, 30]))
    assert rec["color"] == {"straight": "green", "left": "red", "right": "none", "uturn": "red"}
    assert rec["kept"]


def test_lisa_size_filter_is_per_kind():
    # a near-side left-arrow pole is much taller than the far-side balls; the balls still count,
    # but a small ball of another approach does not
    rec = lisa(("stopLeft", [0, 0, 20, 90]), ("go", [50, 0, 60, 30]), ("stop", [70, 0, 75, 12]))
    assert rec["color"]["straight"] == "green" and rec["color"]["left"] == "red"
    assert [b["relevant"] for b in rec["boxes"]] == [True, True, False]


def test_lisa_drops_conflicts_and_arrow_only_frames():
    assert lisa(("go", [0, 0, 10, 30]), ("stop", [20, 0, 30, 30]))["conflict"]
    assert not lisa(("goLeft", [0, 0, 10, 30]))["kept"]
