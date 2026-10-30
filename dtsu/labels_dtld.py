"""Build maneuver-level traffic-signal labels from DTLD v2.0 (Germany, right-hand traffic).

DTLD marks a lamp ``relevant`` when it serves the vehicle's planned route, which itself is not
annotated. Following the route-label rule of the traffic-light-detection project
(experiments/route_labels_20261023), the lamps serving each route are re-derived from the
front-facing vehicle lamps of the ego intersection: the relevant lamps plus non-relevant ones at
least SIZE_RATIO times the median relevant height.

- A route with its own arrow is served by that arrow (the annotator's arrow first). Circles stay
  with it only if the annotator used one of those arrows; otherwise they serve another route.
- A route without an arrow is served by the circles (annotated ones first).

Directions then follow ``dtsu.labels`` where the traffic rules agree: straight, left and right take
the color of their lamps, left without lamps inherits straight and U-turn inherits left (DTLD has no
U-turn signals). Unlike the paper's US convention (right = dedicated signal only), German right turns
obey the circle when there is no right arrow; records say so with ``near_turn = "signal"``.
red_yellow ("prepare to go") is still a stop and counts as red. Off/unknown lamps give no color: a
direction whose lamps all lack one, or whose lamps disagree, drops the frame.
"""
import argparse
import collections
import json
import os
import statistics

from . import DIRECTIONS, color_to_pass
from .labels import stats

ARROW_ROUTES = {
    "arrow_left": {"left"},
    "arrow_right": {"right"},
    "arrow_straight": {"straight"},
    "arrow_straight_left": {"straight", "left"},
}
VEHICLE_PICS = set(ARROW_ROUTES) | {"circle"}
STATE_COLOR = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}
SIZE_RATIO = 0.6  # a non-relevant lamp is at the ego intersection if h >= ratio * median relevant h


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
    """Lamps serving one route (route_labels_20261023's counterfactual rule)."""
    group = rel + nrel
    arrows = [l for l in group if route in ARROW_ROUTES.get(l["attributes"]["pictogram"], ())]
    arrows = [l for l in arrows if l in rel] or arrows
    circles = [l for l in group if l["attributes"]["pictogram"] == "circle"]
    if arrows:
        circles = [l for l in circles if l in rel] if any(l in rel for l in arrows) else []
    else:
        circles = [l for l in circles if l in rel] or circles
    return arrows + circles


def build_frame(im, rule_ok=None):
    rel, nrel = vehicle_group(im["labels"])
    lamps = {r: serving(r, rel, nrel) for r in ("straight", "left", "right")} if rel else {}
    conflict, unknown = [], []

    def resolve(d, ls):
        colors = {STATE_COLOR[l["attributes"]["state"]] for l in ls if l["attributes"]["state"] in STATE_COLOR}
        if len(colors) > 1:
            conflict.append(d)
        elif not colors:
            unknown.append(d)
        return sorted(colors)[0] if colors else "none"

    color = {}
    color["straight"] = resolve("straight", lamps["straight"]) if lamps.get("straight") else "none"
    if lamps.get("left"):
        color["left"] = resolve("left", lamps["left"])
    else:
        color["left"] = color["straight"]
        conflict += ["left"] if "straight" in conflict else []
        unknown += ["left"] if "straight" in unknown else []
    color["uturn"] = color["left"]
    conflict += ["uturn"] if "left" in conflict else []
    unknown += ["uturn"] if "left" in unknown else []
    color["right"] = resolve("right", lamps["right"]) if lamps.get("right") else "none"

    moves = collections.defaultdict(set)
    for r, ls in lamps.items():
        for l in ls:
            moves[l["unique_id"]].add(r)
    boxes = [
        {"box": [l["x"], l["y"], l["x"] + l["w"], l["y"] + l["h"]], "color": STATE_COLOR[l["attributes"]["state"]],
         "relevant": l["unique_id"] in moves, "moves": sorted(moves.get(l["unique_id"], ()))}
        for l in im["labels"]
        if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in VEHICLE_PICS
        and l["attributes"]["state"] in STATE_COLOR
    ]
    path = im["image_path"].replace("\\", "/").lstrip("./")
    rec = {
        "segment_id": "/".join(path.split("/")[:-1]),  # one session = one intersection approach
        "timestamp": str(im["time_stamp"]),
        "image": os.path.basename(path).rsplit(".", 1)[0] + ".jpg",
        "crossing": False,
        "boxes": boxes,
        "n_lights": len(boxes),
        "n_relevant_lights": len(moves),
        "color": {d: color[d] for d in DIRECTIONS},
        "pass": {d: color_to_pass(color[d]) for d in DIRECTIONS},
        "conflict": [d for d in DIRECTIONS if d in conflict],
        "unknown": [d for d in DIRECTIONS if d in unknown],
        "rule_ok": rule_ok,
        "traffic_side": "right",
        "near_turn": "signal",  # German rules: the circle also governs the right turn
    }
    rec["kept"] = rec["n_relevant_lights"] > 0 and not rec["conflict"] and not rec["unknown"] and rule_ok is not False
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True, help="DTLD v2.0 DTLD_train.json / DTLD_test.json")
    ap.add_argument("--route-frames", help="route_frames.json of route_labels_20261023; drops frames whose "
                    "annotated relevant set the rule does not reproduce for the reviewed route (rule_ok false)")
    ap.add_argument("--out", required=True, help="output .jsonl with all frames (kept flag included)")
    args = ap.parse_args()

    route = json.load(open(args.route_frames))["frames"] if args.route_frames else {}
    records = []
    for im in json.load(open(args.labels))["images"]:
        stem = os.path.basename(im["image_path"]).rsplit(".", 1)[0]
        records.append(build_frame(im, route.get(stem, {}).get("rule_ok")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in records if r["kept"]]
    print(f"frames={len(records)} kept={len(kept)} segments={len({r['segment_id'] for r in records})} "
          f"no_relevant_light={sum(r['n_relevant_lights'] == 0 for r in records)} "
          f"conflict={sum(bool(r['conflict']) for r in records)} unknown_state={sum(bool(r['unknown']) for r in records)} "
          f"rule_mismatch={sum(r['rule_ok'] is False for r in records)}")
    for d, c in stats(kept).items():
        print(f"  {d:8s} {c}")


if __name__ == "__main__":
    main()
