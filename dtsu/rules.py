"""Per-light attributes and the traffic rules that turn them into per-maneuver signals.

Every light carries three attributes:
- pictogram (``PICTOGRAMS``): a circle or an arrow; "other" = a pedestrian, bicycle or tram signal; "unknown" = an
  unreadable vehicle pictogram, treated as a circle;
- color (``LAMP_COLORS``): red, yellow or green (red-yellow counts as red);
- junction: whether the light belongs to the junction the ego vehicle is approaching.

A training class is the string ``"<pictogram>|<J|N|?>|<color>"``; "?" marks an attribute the source dataset does
not label, which the factorized loss ignores (``dtsu.factorized``).

Rules (applied to the junction lights of one frame; per maneuver the highest-scoring light decides):
- ``junction_rule``, German convention: an arrow sets the color of its own maneuvers, every other maneuver takes
  the circle's color, the right turn included;
- ``us_rule``, the convention of the OpenLane-V2 benchmark labels: straight takes a straight arrow or else the
  circle; the far turn (left in right-hand traffic) takes its arrow or else follows straight; the near turn is
  governed only by its own arrow, otherwise "none".
"""
from . import DIRECTIONS

# the order is fixed by the model's pictogram head: arrow_straight_right was the last class added
PICTOGRAMS = ["circle", "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left", "unknown", "other",
              "arrow_straight_right"]
LAMP_COLORS = ["red", "yellow", "green"]
ARROW_MOVES = {"arrow_left": {"left"}, "arrow_right": {"right"}, "arrow_straight": {"straight"},
               "arrow_straight_left": {"straight", "left"}, "arrow_straight_right": {"straight", "right"}}
ROUND = ("circle", "unknown")
UNKNOWN = "?"
STATE = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}


def class_name(picto, junction, color):
    """picto / color may be UNKNOWN, junction may be None (unlabelled)."""
    return f"{picto}|{UNKNOWN if junction is None else 'J' if junction else 'N'}|{color}"


def parse(name):
    """-> (picto, junction, color); junction is None when unlabelled, picto / color may be UNKNOWN."""
    p, j, c = name.split("|")
    return p, (None if j == UNKNOWN else j == "J"), c


def _best(lights):
    """Highest-scoring arrow per maneuver and highest-scoring round light, as (score, color)."""
    arrows = {}
    for p, c, s in lights:
        for m in ARROW_MOVES.get(p, ()):
            if m not in arrows or s > arrows[m][0]:
                arrows[m] = (s, c)
    circle = max(((s, c) for p, c, s in lights if p in ROUND), default=None)
    return arrows, circle


def junction_rule(lights):
    """lights: (picto, color, score) of the junction lights -> {maneuver: color}, "none" when nothing governs it."""
    arrows, circle = _best(lights)
    out = {}
    for d in DIRECTIONS:
        src = arrows.get(d) or circle
        out[d] = src[1] if src else "none"
    return out


def us_rule(lights, side="right"):
    """The benchmark's US convention, mirrored for left-hand traffic when side == "left"."""
    arrows, circle = _best(lights)
    far, near = ("left", "right") if side == "right" else ("right", "left")
    straight = arrows.get("straight") or circle
    res = {"straight": straight, far: arrows.get(far) or straight, near: arrows.get(near)}
    return {d: res[d][1] if res[d] else "none" for d in DIRECTIONS}


def apply_topology(colors, moves):
    """Blank the maneuvers no lane of the ego approach leads to (``dtsu.topology``); moves None = no junction ahead.
    Left is kept while straight exists, as the benchmark's left inherits straight."""
    if moves is None:
        return colors
    colors = dict(colors)
    if "straight" not in moves:
        colors["straight"] = "none"
    if not moves & {"left", "straight"}:
        colors["left"] = "none"
    if "right" not in moves:
        colors["right"] = "none"
    return colors


def ground_truth(boxes):
    """Junction-rule ground truth of one labelled frame: ({maneuver: color}, None) when the rule gives one answer,
    else (None, reason). boxes: dicts with picto, junction and color."""
    lights = [(b["picto"], b["color"]) for b in boxes if b["junction"]]
    if not lights:
        return None, "no junction light"
    if len({c for p, c in lights if p in ROUND}) > 1:
        return None, "circles disagree"
    arrows = {}
    for p, c in lights:
        for m in ARROW_MOVES.get(p, ()):
            arrows.setdefault(m, set()).add(c)
    if any(len(v) > 1 for v in arrows.values()):
        return None, "arrows disagree"
    return junction_rule([(p, c, 1.0) for p, c in lights]), None
