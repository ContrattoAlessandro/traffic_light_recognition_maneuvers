"""Route labels of DTLD v2.0 for the baseline M (German rules), and D*_US, the same frames under the OpenLane-V2 protocol.

DTLD marks a lamp ``relevant`` when it serves the driven route, which itself is not annotated. The lamps serving each
possible route are re-derived from the front-facing vehicle lamps of the ego intersection: the relevant lamps plus
non-relevant ones at least SIZE_RATIO x the median relevant height.
- A route with its own arrow is served by that arrow (an annotated relevant arrow first). Circles stay with it only if
  the annotator used one of those arrows; otherwise they serve another route.
- A route without an arrow is served by the circles (annotated relevant ones first).
A lamp's ``moves`` are the routes it serves: the analogue of OpenLane-V2's light-to-lane links, and the class M learns.

German labels (``build_frame``): each maneuver takes the color of its lamps; left without lamps inherits straight; the
right turn obeys the circle when there is no right arrow (``near_turn = "signal"``). red_yellow counts as red. A
maneuver whose lamps disagree, or have no color, drops the frame. Frames where the rule does not reproduce the
annotated relevant set for the manually reviewed route (``splits/dtld_route_review.json``) are dropped as well.

D*_US (``to_openlane_protocol``): the same per-lamp moves through the OpenLane-V2 rule of ``dtsu.labels`` (US rules:
right only from a dedicated right signal; the most specific lamps win; lamps without a color are ignored).
"""
import collections
import json
import os
import statistics

from . import DIRECTIONS, color_to_pass
from .labels import assign_directions
from .paths import DATASETS, LABELS, SPLITS

ARROW_ROUTES = {"arrow_left": {"left"}, "arrow_right": {"right"}, "arrow_straight": {"straight"},
                "arrow_straight_left": {"straight", "left"}}
VEHICLE_PICS = set(ARROW_ROUTES) | {"circle"}
STATE_COLOR = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}
SIZE_RATIO = 0.6


def vehicle_group(labels):
    """Front-facing vehicle lamps at the ego intersection: (relevant, non-relevant)."""
    front = [l for l in labels if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in VEHICLE_PICS]
    rel = [l for l in front if l["attributes"]["relevance"] == "relevant"]
    if not rel:
        return [], []
    h_ref = statistics.median(l["h"] for l in rel)
    nrel = [l for l in front if l["attributes"]["relevance"] != "relevant" and l["h"] >= SIZE_RATIO * h_ref]
    return rel, nrel


def serving(route, rel, nrel):
    """Lamps serving one route."""
    group = rel + nrel
    arrows = [l for l in group if route in ARROW_ROUTES.get(l["attributes"]["pictogram"], ())]
    arrows = [l for l in arrows if l in rel] or arrows
    circles = [l for l in group if l["attributes"]["pictogram"] == "circle"]
    if arrows:
        circles = [l for l in circles if l in rel] if any(l in rel for l in arrows) else []
    else:
        circles = [l for l in circles if l in rel] or circles
    return arrows + circles


def build_frame(im, image, rule_ok=None):
    """One DTLD image -> route-label record (German rules)."""
    rel, nrel = vehicle_group(im["labels"])
    lamps = {r: serving(r, rel, nrel) for r in DIRECTIONS} if rel else {}
    conflict, unknown = [], []

    def resolve(d, ls):
        colors = {STATE_COLOR[l["attributes"]["state"]] for l in ls if l["attributes"]["state"] in STATE_COLOR}
        if len(colors) > 1:
            conflict.append(d)
        elif not colors:
            unknown.append(d)
        return sorted(colors)[0] if colors else "none"

    color = {"straight": resolve("straight", lamps["straight"]) if lamps.get("straight") else "none"}
    if lamps.get("left"):
        color["left"] = resolve("left", lamps["left"])
    else:
        color["left"] = color["straight"]
        conflict += ["left"] if "straight" in conflict else []
        unknown += ["left"] if "straight" in unknown else []
    color["right"] = resolve("right", lamps["right"]) if lamps.get("right") else "none"

    moves = collections.defaultdict(set)
    for r, ls in lamps.items():
        for l in ls:
            moves[l["unique_id"]].add(r)
    boxes = [{"box": [l["x"], l["y"], l["x"] + l["w"], l["y"] + l["h"]], "color": STATE_COLOR[l["attributes"]["state"]],
              "relevant": l["unique_id"] in moves, "moves": sorted(moves.get(l["unique_id"], ()))}
             for l in im["labels"]
             if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in VEHICLE_PICS
             and l["attributes"]["state"] in STATE_COLOR]
    path = im["image_path"].replace("\\", "/").lstrip("./")
    rec = {"segment_id": "/".join(path.split("/")[:-1]), "timestamp": str(im["time_stamp"]), "image": image,
           "boxes": boxes, "n_relevant_lights": len(moves), "color": color,
           "pass": {d: color_to_pass(color[d]) for d in DIRECTIONS},
           "conflict": [d for d in DIRECTIONS if d in conflict], "unknown": [d for d in DIRECTIONS if d in unknown],
           "rule_ok": rule_ok, "traffic_side": "right", "near_turn": "signal"}
    rec["kept"] = rec["n_relevant_lights"] > 0 and not rec["conflict"] and not rec["unknown"] and rule_ok is not False
    return rec


def to_openlane_protocol(rec):
    """D*_US: the record's per-lamp moves through ``labels.assign_directions`` (US rules) and OpenLane-V2's frame filter
    (at least one relevant colored lamp, no conflict); the route-review filter is kept as label quality control."""
    light_moves = {i: (b["color"], set(b["moves"])) for i, b in enumerate(rec["boxes"]) if b["relevant"]}
    out = assign_directions(light_moves, "right")
    out = {k: ({d: v[d] for d in DIRECTIONS} if k in ("color", "pass") else v) for k, v in out.items()}
    new = {**rec, **out, "traffic_side": "right", "near_turn": "dedicated", "n_relevant_lights": len(light_moves),
           "had_unknown": bool(rec["unknown"]), "unknown": []}
    new["kept"] = bool(light_moves) and not out["conflict"] and rec["rule_ok"] is not False
    return new


def route_records(split):
    """Every frame of a DTLD split (train or test) as a route-label record, in the order of DTLD's own label file."""
    raw, jpg = os.path.join(DATASETS, "DTLD", "v2.0"), os.path.join(DATASETS, "DTLD_jpg", split)
    review = json.load(open(os.path.join(SPLITS, "dtld_route_review.json")))
    bad = set(review["rule_mismatch"][split])
    out = []
    for im in json.load(open(os.path.join(raw, f"DTLD_{split}.json")))["images"]:
        stem = os.path.basename(im["image_path"].replace("\\", "/")).rsplit(".", 1)[0]
        out.append(build_frame(im, os.path.abspath(os.path.join(jpg, stem + ".jpg")), False if stem in bad else None))
    return out


def label_path(kind, part):
    """kind "route" (German rules) or "us" (D*_US); part "fit", "tune" or "test"."""
    return os.path.join(LABELS, f"dtld_{kind}_{part}.jsonl")
