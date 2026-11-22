"""The baseline M: joint classes, its direction decoder and the DTLD route labels it trains on."""
import torch

from dtsu.labels import assign_directions
from dtsu.labels_dtld import build_frame, to_openlane_protocol
from dtsu.moves import assign_scored, attribute_codes, box_class, frame_colors, parse_class
from dtsu.rules import LAMP_COLORS, PICTOGRAMS


def test_class_round_trip():
    assert box_class({"relevant": True, "moves": ["left", "straight"], "color": "red"}) == "left+straight|red"
    assert box_class({"relevant": False, "moves": [], "color": "green"}) == "irrelevant|green"
    assert parse_class("left+straight|red") == ({"left", "straight"}, "red")
    assert parse_class("irrelevant|green") == (None, "green")


def test_most_specific_light_wins_then_score():
    dets = [({"generic"}, "green", 0.9), ({"left"}, "red", 0.4)]
    assert assign_scored(dets) == {"straight": "green", "left": "red", "right": "none"}
    dets = [({"left"}, "red", 0.4), ({"left"}, "green", 0.8)]
    assert assign_scored(dets)["left"] == "green"


def test_right_turn_us_versus_german_rule():
    dets = [({"right", "straight"}, "red", 0.9)]
    assert assign_scored(dets, near_turn="dedicated")["right"] == "none"
    assert assign_scored(dets, near_turn="signal")["right"] == "red"


def test_decoder_agrees_with_label_builder_on_ground_truth():
    lights = {0: ("red", {"straight", "left"}), 1: ("green", {"right"}), 2: ("green", {"generic"})}
    dets = [(m, c, 1.0) for c, m in lights.values()]
    assert assign_scored(dets) == assign_directions(lights)["color"]


def test_frame_colors_thresholds_and_ignores_irrelevant():
    names = ["irrelevant|green", "straight|red"]
    det = (torch.tensor([0, 1]), torch.tensor([0.9, 0.3]), torch.tensor([[0.2, 0.2, 0.01, 0.02], [0.5, 0.5, 0.01, 0.02]]))
    assert frame_colors(det, names, 0.25)["straight"] == "red"
    assert frame_colors(det, names, 0.35)["straight"] == "none"


def test_attribute_codes_carry_color_and_relevance():
    code = attribute_codes(["irrelevant|yellow", "left|red"])
    assert (code // 6 == PICTOGRAMS.index("circle")).all()
    assert (code % 3).tolist() == [LAMP_COLORS.index("yellow"), LAMP_COLORS.index("red")]
    assert ((code // 3) % 2).tolist() == [0, 1]


def _lamp(i, picto, state, relevant, h=20):
    return {"unique_id": i, "x": 10 * i, "y": 0, "w": 8, "h": h,
            "attributes": {"direction": "front", "pictogram": picto, "state": state, "relevance": "relevant" if relevant else "not_relevant"}}


def test_dtld_route_labels_german_and_us():
    im = {"image_path": "./Berlin/Berlin1/s1/DE_x.tiff", "time_stamp": 1,
          "labels": [_lamp(0, "circle", "red", True), _lamp(1, "arrow_left", "green", False)]}  # driving straight
    rec = build_frame(im, "DE_x.jpg")
    assert rec["kept"] and rec["color"] == {"straight": "red", "left": "green", "right": "red"}  # the circle governs right
    us = to_openlane_protocol(rec)
    assert us["kept"] and us["color"] == {"straight": "red", "left": "green", "right": "none"}  # right needs its own signal
    assert build_frame(im, "DE_x.jpg", rule_ok=False)["kept"] is False
