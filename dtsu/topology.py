"""Maneuvers the ego approach allows, from the OpenLane-V2 lane topology (HD map), without any traffic light.

This is a map prior: it uses the annotated lane centerlines of the frame, which a camera-only system does not
have. ``dtsu.rules.apply_topology`` blanks the maneuvers no lane of the ego approach leads to.
"""
import json
import os

import numpy as np

from .labels import AHEAD_X, ALIGNED_DEG, SAME_JUNCTION_M, _heading, _wrap, lane_movements


def info_path(record):
    p = record["image"]
    for cam in ("ring_front_center", "CAM_FRONT"):
        p = p.replace(f"/image/{cam}/", "/info/")
    return os.path.join(record["root"], os.path.splitext(p)[0] + ".json")


def approach_moves(path):
    """Maneuvers reachable from the ego approach at the nearest junction; None if no junction is ahead."""
    ann = json.load(open(path))["annotation"]
    lanes = [{"pts": np.asarray(l["points"], dtype=np.float64), "connector": bool(l.get("is_intersection_or_connector", False))}
             for l in ann["lane_centerline"]]
    n = len(lanes)
    lclc = np.asarray(ann["topology_lclc"], dtype=np.int64).reshape(n, n) if n else np.zeros((0, 0), int)
    cands = []
    for i, l in enumerate(lanes):
        if abs(_wrap(_heading(l["pts"], at_end=not l["connector"]))) < ALIGNED_DEG and l["pts"][-1, 0] > AHEAD_X:
            moves, stop = lane_movements(i, lanes, lclc)
            if moves:
                cands.append((stop, moves))
    if not cands:
        return None
    near = min(s for s, _ in cands)
    return set().union(*(m for s, m in cands if s <= near + SAME_JUNCTION_M))


def topology_masks(records):
    """Per OpenLane-V2 label record: the allowed maneuvers (or None)."""
    cache = {}
    for r in records:
        k = info_path(r)
        if k not in cache:
            cache[k] = approach_moves(k)
    return [cache[info_path(r)] for r in records]
