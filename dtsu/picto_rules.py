"""Exp 6 (REPLICATION_PLAN.md): DTLD lights with pictogram, junction relevance and color, and the junction rule.

Per box: pictogram (PICTOS), junction relevance (bool) and color (COLORS). The joint arm J learns one class
per "<pictogram>|<J or N>|<color>"; the factorized arm F learns the three separately (dtsu/factorized.py).

The rule turns the junction lights of one frame into per-maneuver colors: an arrow sets its own maneuvers,
every maneuver without an arrow takes the circle color (German convention: the circle also governs the
right turn), U-turn follows left, unknown pictograms count as circles.
"""
import statistics

from . import DIRECTIONS

# arrow_straight_right was appended for Exp 7 (ATLAS); the Exp 6 models have 7 pictogram outputs, the same first seven
PICTOS = ["circle", "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left", "unknown", "other", "arrow_straight_right"]
COLORS = ["red", "yellow", "green"]
ARROW_MOVES = {"arrow_left": {"left"}, "arrow_right": {"right"}, "arrow_straight": {"straight"},
               "arrow_straight_left": {"straight", "left"}, "arrow_straight_right": {"straight", "right"}}
DTLD_OTHER = {"pedestrian", "bicycle", "pedestrian_bicycle", "tram"}
DTLD_VEHICLE = set(ARROW_MOVES) | {"circle", "unknown"}
STATE = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}
JUNCTION_RATIO = 0.6  # dtsu.labels_dtld.vehicle_group


UNKNOWN = "?"  # Exp 7: an attribute the source dataset does not label (masked in the loss)


def class_name(picto, junction, color):
    """picto / color may be UNKNOWN, junction may be None (unlabelled)."""
    return f"{picto}|{UNKNOWN if junction is None else 'J' if junction else 'N'}|{color}"


def parse(name):
    """-> (picto, junction, color); junction is None when unlabelled, picto / color may be UNKNOWN."""
    p, j, c = name.split("|")
    return p, (None if j == UNKNOWN else j == "J"), c


def dtld_boxes(im):
    """Boxes of one DTLD v2.0 image: front-facing lamps with a visible color, with pictogram and junction flag."""
    front = [l for l in im["labels"] if l["attributes"]["direction"] == "front" and l["attributes"]["state"] in STATE]
    vehicle = [l for l in im["labels"] if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in DTLD_VEHICLE]
    route = [l for l in vehicle if l["attributes"]["relevance"] == "relevant"]
    h = statistics.median(l["h"] for l in route) if route else None
    out = []
    for l in front:
        a = l["attributes"]
        if a["pictogram"] not in DTLD_VEHICLE | DTLD_OTHER:
            continue
        picto = "other" if a["pictogram"] in DTLD_OTHER else a["pictogram"]
        junction = picto != "other" and route and (a["relevance"] == "relevant" or l["h"] >= JUNCTION_RATIO * h)
        out.append({"box": [l["x"], l["y"], l["x"] + l["w"], l["y"] + l["h"]], "picto": picto,
                    "junction": bool(junction), "color": STATE[a["state"]]})
    return out


def rule(lights):
    """lights: (picto, color, score) of junction lights -> per-maneuver colors ("none" when nothing governs it).
    Per maneuver the highest-scoring arrow decides, else the highest-scoring circle (unknown = circle)."""
    best = {}
    circle = max(((s, c) for p, c, s in lights if p in ("circle", "unknown")), default=None)
    for p, c, s in lights:
        for m in ARROW_MOVES.get(p, ()):
            if m not in best or s > best[m][0]:
                best[m] = (s, c)
    out = {}
    for d in ("straight", "left", "right"):
        src = best.get(d) or circle
        out[d] = src[1] if src else "none"
    out["uturn"] = out["left"]
    return {d: out[d] for d in DIRECTIONS}


def gt_status(boxes):
    """Ground truth of one frame: (per-maneuver colors, None) when the rule gives one answer, else (None, reason)."""
    lights = [(b["picto"], b["color"]) for b in boxes if b["junction"]]
    if not lights:
        return None, "no junction light"
    circles = {c for p, c in lights if p in ("circle", "unknown")}
    if len(circles) > 1:
        return None, "circles disagree"
    arrows = {}
    for p, c in lights:
        for m in ARROW_MOVES.get(p, ()):
            arrows.setdefault(m, set()).add(c)
    if any(len(v) > 1 for v in arrows.values()):
        return None, "arrows disagree"
    return rule([(p, c, 1.0) for p, c in lights]), None
