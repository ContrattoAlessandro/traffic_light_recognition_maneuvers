"""Traffic rules that turn per-light attributes into per-maneuver colors."""
from dtsu.rules import apply_topology, class_name, ground_truth, junction_rule, parse, us_rule


def test_class_name_round_trip():
    for p, j, c in (("circle", True, "red"), ("arrow_left", False, "green"), ("?", None, "yellow")):
        assert parse(class_name(p, j, c)) == (p, j, c)


def test_junction_rule_circle_governs_every_maneuver():
    assert junction_rule([("circle", "red", 0.9)]) == {"straight": "red", "left": "red", "right": "red"}


def test_junction_rule_arrow_overrides_circle_for_its_maneuvers():
    lights = [("circle", "red", 0.9), ("arrow_left", "green", 0.5)]
    assert junction_rule(lights) == {"straight": "red", "left": "green", "right": "red"}
    lights = [("circle", "green", 0.9), ("arrow_straight_right", "red", 0.4)]
    assert junction_rule(lights) == {"straight": "red", "left": "green", "right": "red"}


def test_junction_rule_highest_score_wins_and_unknown_is_round():
    assert junction_rule([("circle", "red", 0.4), ("unknown", "green", 0.8)])["straight"] == "green"
    assert junction_rule([("arrow_left", "red", 0.9), ("arrow_left", "green", 0.3)])["left"] == "red"
    assert junction_rule([]) == {"straight": "none", "left": "none", "right": "none"}


def test_us_rule_right_needs_its_own_arrow_and_left_follows_straight():
    assert us_rule([("circle", "red", 0.9)]) == {"straight": "red", "left": "red", "right": "none"}
    assert us_rule([("circle", "red", 0.9), ("arrow_right", "green", 0.5)])["right"] == "green"
    assert us_rule([("arrow_straight", "green", 0.9), ("arrow_left", "red", 0.5)]) == {"straight": "green", "left": "red", "right": "none"}


def test_us_rule_mirrors_in_left_hand_traffic():
    assert us_rule([("circle", "red", 0.9)], side="left") == {"straight": "red", "left": "none", "right": "red"}


def test_apply_topology():
    colors = {"straight": "red", "left": "red", "right": "green"}
    assert apply_topology(colors, None) == colors
    assert apply_topology(colors, {"left"}) == {"straight": "none", "left": "red", "right": "none"}
    # left inherits straight, so it is kept while straight exists
    assert apply_topology(colors, {"straight"}) == {"straight": "red", "left": "red", "right": "none"}


def test_ground_truth():
    b = lambda p, c, j=True: {"picto": p, "color": c, "junction": j}  # noqa: E731
    assert ground_truth([b("circle", "red"), b("arrow_left", "green")]) == ({"straight": "red", "left": "green", "right": "red"}, None)
    assert ground_truth([b("circle", "red"), b("circle", "green")]) == (None, "circles disagree")
    assert ground_truth([b("arrow_left", "red"), b("arrow_left", "green")]) == (None, "arrows disagree")
    assert ground_truth([b("circle", "red", False)]) == (None, "no junction light")
