"""Build maneuver-level traffic-signal labels from OpenLane-V2 annotations.

For every frame we derive, for each ego maneuver d in DIRECTIONS, the color of
the traffic signal controlling it (none/red/yellow/green) and the implied
passability. The derivation follows the paper's data-construction section, with
the details it leaves open filled in as documented in REPLICATION_PLAN.md:

1. Traffic lights are traffic elements whose attribute is a color. Each light is
   linked by ``topology_lcte`` to the lane centerlines that must obey it.
2. The movements a linked lane leads into are found by walking its successors
   (``topology_lclc``) until the first intersection connector ahead of the ego
   vehicle, and classifying that connector's heading change as straight / left /
   right / U-turn. Lanes cut off by the annotation range give no movement and
   are treated as generic (they control "straight" and, by inheritance, "left").
3. Lights are kept only when they control lanes heading the ego's way, at the
   nearest intersection ahead. This drops opposite / crossing traffic and signals
   of distant intersections. When no such lane is ahead (the ego has passed the
   stop line), the approach lanes just behind the ego are used instead.
4. Directions: straight takes lights covering straight (or generic lanes); left
   takes lights covering left, else inherits straight (permissive left on the
   ball light); U-turn takes lights covering U-turn, else inherits left; right
   takes only dedicated right-turn lights, else none. Among candidates the most
   specific lights (smallest movement set) win; remaining color disagreement
   marks the frame as conflicting. These right-hand-traffic rules are applied
   everywhere by default, which matches the paper's Fig. 3; ``--traffic-side auto``
   instead detects left-hand traffic (Singapore scenes of subset_B) per segment and
   swaps the roles of left and right there.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from . import DIRECTIONS, color_to_pass

FRONT_CAMERAS = ("ring_front_center", "CAM_FRONT")  # subset_A (Argoverse 2), subset_B (nuScenes)
LIGHT_COLORS = {1: "red", 2: "green", 3: "yellow"}  # traffic_element attribute -> color
SIGN_MOVES = {4: "straight", 5: "left", 6: "right", 9: "uturn", 11: "straight", 12: "straight"}

ALIGNED_DEG = 60.0  # a lane heads the ego's way if its heading is within this of the x-axis
STRAIGHT_DEG = 30.0  # |turn| below this is straight
UTURN_DEG = 135.0  # |turn| above this is a U-turn
AHEAD_X = 0.0  # a lane counts as "ahead" if it ends beyond this x (metres)
# a connector is the junction being entered if it starts beyond this x: at the stop line the
# junction starts slightly behind the ego origin, while junctions being left start far behind
ENTERING_X = -3.0
SAME_JUNCTION_M = 10.0  # lanes whose stop line is within this of the nearest one share the junction
TRAVERSE_M = 30.0  # approach lanes ending at most this far behind the ego still apply while crossing
MAX_HOPS = 4


def _heading(points, at_end):
    """Heading in degrees of a polyline near its start or end, from ~10% of its points."""
    k = max(2, len(points) // 10)
    seg = points[-k:] if at_end else points[:k]
    v = seg[-1, :2] - seg[0, :2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def classify_turn(points):
    turn = _wrap(_heading(points, True) - _heading(points, False))
    if abs(turn) < STRAIGHT_DEG:
        return "straight"
    if abs(turn) > UTURN_DEG:
        return "uturn"
    return "left" if turn > 0 else "right"  # ego frame: y points left


def lane_movements(i, lanes, lclc, ahead_x=ENTERING_X):
    """Movements reachable from lane i through its first connector starting beyond ahead_x.

    Returns (movements, stop_x): the set of maneuvers and the x coordinate where
    the junction starts, or (set(), None) when no connector is reached.
    """
    moves, stops = set(), []
    frontier, seen = [(i, 0)], {i}
    while frontier:
        j, hops = frontier.pop()
        pts = lanes[j]["pts"]
        if lanes[j]["connector"] and pts[0, 0] > ahead_x:
            moves.add(classify_turn(pts))
            stops.append(pts[0, 0])
            continue
        if hops >= MAX_HOPS:
            continue
        for s in np.flatnonzero(lclc[j]):
            if s not in seen:
                seen.add(s)
                frontier.append((s, hops + 1))
    return moves, (min(stops) if stops else None)


def build_frame(info):
    ann = info["annotation"]
    lanes = [
        {"pts": np.asarray(l["points"], dtype=np.float64), "connector": bool(l.get("is_intersection_or_connector", False))}
        for l in ann["lane_centerline"]
    ]
    tes = ann["traffic_element"]
    n_l, n_t = len(lanes), len(tes)
    lclc = np.asarray(ann["topology_lclc"], dtype=np.int64).reshape(n_l, n_l) if n_l else np.zeros((0, 0), int)
    lcte = np.asarray(ann["topology_lcte"], dtype=np.int64).reshape(n_l, n_t) if n_l and n_t else np.zeros((n_l, n_t), int)

    lights = [j for j, t in enumerate(tes) if t["attribute"] in LIGHT_COLORS]

    # Per linked lane: movements, junction position, and whether it heads the ego's way.
    lane_info = {}
    for i in np.flatnonzero(lcte[:, lights].sum(1) > 0) if lights else []:
        moves, stop_x = lane_movements(i, lanes, lclc)
        pts = lanes[i]["pts"]
        if not moves:  # no junction within range: fall back to linked road-sign arrows
            moves = {SIGN_MOVES[tes[j]["attribute"]] for j in np.flatnonzero(lcte[i]) if tes[j]["attribute"] in SIGN_MOVES}
        if stop_x is None:
            stop_x = pts[-1, 0]
        # connectors are judged by their entry heading, other lanes by their end heading
        heading = _heading(pts, at_end=not lanes[i]["connector"])
        lane_info[i] = {
            "moves": moves or {"generic"},
            "stop_x": float(stop_x),
            "aligned": abs(_wrap(heading)) < ALIGNED_DEG,
            "ahead": pts[-1, 0] > AHEAD_X,
        }

    candidates = [i for i, li in lane_info.items() if li["aligned"] and li["ahead"]]
    if candidates:
        nearest = min(lane_info[i]["stop_x"] for i in candidates)
        relevant = {i for i in candidates if lane_info[i]["stop_x"] <= nearest + SAME_JUNCTION_M}
    else:
        # Ego has passed the stop line and is crossing the junction: the signal it crosses
        # under still applies (paper Fig. 2), via the approach lanes just behind it.
        behind = [i for i, li in lane_info.items()
                  if li["aligned"] and -TRAVERSE_M < lanes[i]["pts"][-1, 0] <= AHEAD_X]
        last = max((lanes[i]["pts"][-1, 0] for i in behind), default=None)
        relevant = {i for i in behind if lanes[i]["pts"][-1, 0] >= last - SAME_JUNCTION_M}
        for i in relevant:
            moves, _ = lane_movements(i, lanes, lclc, ahead_x=-np.inf)
            if moves:
                lane_info[i]["moves"] = moves

    light_moves = {}
    for j in lights:
        linked = [i for i in np.flatnonzero(lcte[:, j]) if i in relevant]
        if linked:
            light_moves[j] = (LIGHT_COLORS[tes[j]["attribute"]], set().union(*(lane_info[i]["moves"] for i in linked)))

    boxes = [
        {"box": [v for p in tes[j]["points"] for v in p], "color": LIGHT_COLORS[tes[j]["attribute"]], "relevant": j in light_moves,
         "moves": sorted(light_moves[j][1]) if j in light_moves else []}
        for j in lights
    ]
    camera = next(c for c in FRONT_CAMERAS if c in info["sensor"])
    return {
        "segment_id": info["segment_id"],
        "timestamp": str(info["timestamp"]),
        "image": info["sensor"][camera]["image_path"],
        "by_side": {side: assign_directions(light_moves, side) for side in ("right", "left")},
        "side_vote": traffic_side_vote(lanes),
        "crossing": not candidates and bool(light_moves),  # signal taken from approach lanes behind the ego
        "boxes": boxes,
        "n_lights": len(lights),
        "n_relevant_lights": len(light_moves),
    }


def assign_directions(light_moves, side="right"):
    """Map lights to the four maneuvers for right-hand ("right") or left-hand ("left") traffic.

    The turn across oncoming traffic ("far" turn: left in right-hand traffic) and the
    U-turn are governed by the approach signal unless a light covers them; the "near"
    turn (right in right-hand traffic) is only set by a dedicated signal.
    """
    far, near = ("left", "right") if side == "right" else ("right", "left")

    def pick(pred):
        cands = [(len(m), c) for c, m in light_moves.values() if pred(m)]
        if not cands:
            return None, False
        best = min(n for n, _ in cands)
        colors = {c for n, c in cands if n == best}
        if len(colors) > 1:
            return sorted(colors)[0], True
        return colors.pop(), False

    out = {"straight": pick(lambda m: "straight" in m or "generic" in m)}
    out[far] = pick(lambda m: far in m)
    if out[far][0] is None:
        out[far] = out["straight"]
    out["uturn"] = pick(lambda m: "uturn" in m)
    if out["uturn"][0] is None:
        out["uturn"] = out[far]
    out[near] = pick(lambda m: m <= {near})

    color = {d: out[d][0] or "none" for d in DIRECTIONS}
    return {
        "color": color,
        "pass": {d: color_to_pass(color[d]) for d in DIRECTIONS},
        "conflict": [d for d in DIRECTIONS if out[d][1]],
    }


def traffic_side_vote(lanes):
    """+1 if oncoming lanes near the ego lie to its left (right-hand traffic), -1 if to its right."""
    votes = []
    for l in lanes:
        pts = l["pts"]
        if l["connector"]:
            continue
        near = pts[(np.abs(pts[:, 0]) < 20) & (np.abs(pts[:, 1]) < 15)]
        if len(near) >= 2 and abs(_wrap(_heading(pts, at_end=False))) > 150:
            votes.append(np.sign(near[:, 1].mean()))
    return int(np.sign(sum(votes))) if votes else 0


def keep(rec):
    """Benchmark frames: a relevant signal is visible and its maneuver mapping is unambiguous."""
    return rec["n_relevant_lights"] > 0 and not rec["conflict"]


def iter_info_files(root, split, data_dict=None):
    if data_dict:
        segments = json.load(open(data_dict))[split]
        for seg, frames in sorted(segments.items()):
            for f in frames:
                yield os.path.join(root, split, seg, "info", f)
    else:
        yield from sorted(glob.glob(os.path.join(root, split, "*", "info", "*.json")))


def stats(records):
    out = {d: collections.Counter(r["color"][d] for r in records) for d in DIRECTIONS}
    return {d: dict(sorted(c.items())) for d, c in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="OpenLane-V2 subset root (contains train/ val/)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-dict", help="data_dict_subset_A.json; scans the directory when omitted")
    ap.add_argument("--out", required=True, help="output .jsonl with all frames (kept flag included)")
    # "right" reproduces the paper's Fig. 3 on subset A+B (right-turn ~97% none even with
    # Singapore scenes); "auto" is the semantically correct variant for left-hand traffic.
    ap.add_argument("--traffic-side", choices=("auto", "right"), default="right",
                    help="right: US rules everywhere (paper); auto: detect left-hand traffic per segment and mirror the rules")
    args = ap.parse_args()

    records, missing = [], 0
    for path in iter_info_files(args.root, args.split, args.data_dict):
        if not os.path.exists(path):
            missing += 1
            continue
        with open(path) as f:
            records.append(build_frame(json.load(f)))

    seg_votes = collections.defaultdict(int)
    for r in records:
        seg_votes[r["segment_id"]] += r["side_vote"]
    for r in records:
        side = "left" if args.traffic_side == "auto" and seg_votes[r["segment_id"]] < 0 else "right"
        r.update(r.pop("by_side")[side], traffic_side=side)
        del r["side_vote"]
        r["kept"] = keep(r)
    n_left = len({r["segment_id"] for r in records if r["traffic_side"] == "left"})
    print(f"segments={len(seg_votes)} left-hand-traffic segments={n_left}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in records if r["kept"]]
    n_conflict = sum(bool(r["conflict"]) for r in records)
    print(f"frames={len(records)} kept={len(kept)} no_relevant_light="
          f"{sum(r['n_relevant_lights'] == 0 for r in records)} conflict={n_conflict} missing_info={missing}")
    for d, c in stats(kept).items():
        print(f"  {d:8s} {c}")


if __name__ == "__main__":
    main()
