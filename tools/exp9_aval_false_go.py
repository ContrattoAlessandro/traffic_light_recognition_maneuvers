"""Exp 9 follow-up (exploratory, not pre-registered): why WF has more False-Go than F8 on OpenLane A val.

Both models are decoded exactly as in the Exp 9 guards (US rule + lane topology mask, each at its selected epoch
and threshold). A False-Go cell is a (frame, direction) whose ground truth is red and whose prediction is not red
("none" counts as go). For every cell where exactly one model makes a False-Go, the light that gave the other
model its red is traced in the failing model's detections and the cause is named:
  missed          no detection of that light above the failing model's threshold
  junction_off    detected, but junction score <= 0.5, so the rule ignores it
  color           detected as a junction light, but not red
  routed_away     red junction light, but no light sets this direction (its pictogram does not cover it)
  outvoted        red junction light, but another (non-red) light sets this direction
Cells where the other model's red came from an unmatched source are "other".
Usage: exp9_aval_false_go.py [WF|W]. Writes runs/exp9_eval/aval_false_go_{arm}.json and prints a summary.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou, box_convert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp8 as e8  # noqa: E402
import tools.exp7_checks as ck  # noqa: E402
from dtsu.picto_rules import ARROW_MOVES, COLORS, PICTOS  # noqa: E402
from tools.eval_picto import kept  # noqa: E402

DIRS = ("straight", "left", "right", "uturn")
ARMS = {"WF": ("runs/exp9_eval/WF/e4_A_val.pt", 0.35), "W": ("runs/exp9_eval/W/e1_A_val.pt", 0.25)}  # selected epoch, threshold
F8 = ("runs/exp8_bdd_F/exp8_eval/e3_A_val.pt", 0.45)
MATCH_IOU = 0.3


def us_rule_src(d, side):
    """ck.us_rule with the index (into d) of the light that sets each direction."""
    lights = [(PICTOS[int(p)], COLORS[int(c)], float(s), i)
              for i, (p, c, j, s) in enumerate(zip(d["picto"], d["color"], d["junction"], d["score"])) if j > 0.5]
    best, circle = {}, max(((s, c, i) for p, c, s, i in lights if p in ("circle", "unknown")), default=None)
    for p, c, s, i in lights:
        for m in ARROW_MOVES.get(p, ()):
            if m not in best or s > best[m][0]:
                best[m] = (s, c, i)
    far, near = ("left", "right") if side == "right" else ("right", "left")
    straight = best.get("straight") or circle
    f = best.get(far) or straight
    res = {"straight": straight, far: f, near: best.get(near), "uturn": f}
    return {k: (v[1], v[2]) if v else ("none", None) for k, v in res.items()}


def decode(d, thr, rec, mask):
    k = kept(d, thr)
    out = us_rule_src(k, rec["traffic_side"])
    color = ck.apply_topo({a: v[0] for a, v in out.items()}, mask)
    return k, {a: (color[a], out[a][1] if color[a] != "none" else None) for a in DIRS}


def cause(src_box, fail_all, fail_kept, fail_out, direction, thr):
    """Why the failing model is not red on `direction`, given the box of the other model's red light."""
    def match(boxes):
        if not len(boxes):
            return None
        iou = box_iou(box_convert(src_box[None], "cxcywh", "xyxy"), box_convert(boxes, "cxcywh", "xyxy"))[0]
        j = int(iou.argmax())
        return j if iou[j] >= MATCH_IOU else None

    j = match(fail_kept["box"])
    if j is None:
        a = match(fail_all["box"])
        return "missed", None if a is None else float(fail_all["score"][a])
    if float(fail_kept["junction"][j]) <= 0.5:
        return "junction_off", float(fail_kept["junction"][j])
    c = COLORS[int(fail_kept["color"][j])]
    if c != "red":
        return "color", f"red->{c}"
    other = fail_out[direction][1]
    if other is None:
        return "routed_away", PICTOS[int(fail_kept["picto"][j])]
    return "outvoted", f"{PICTOS[int(fail_kept['picto'][j])]} red, set by {PICTOS[int(fail_kept['picto'][other])]}/{COLORS[int(fail_kept['color'][other])]}"


def main(arm):
    SEL = {"F8": F8, arm: ARMS[arm]}
    recs, _, masks, _ = e8.openlane_set("A_val")
    dets = {m: torch.load(p, weights_only=False) for m, (p, _) in SEL.items()}
    rows = []
    for i, (r, mask) in enumerate(zip(recs, masks)):
        dec = {m: decode(dets[m][i], SEL[m][1], r, mask) for m in SEL}
        for a in DIRS:
            if r["color"][a] != "red":
                continue
            fg = {m: dec[m][1][a][0] != "red" for m in SEL}
            if fg["F8"] == fg[arm]:
                continue
            fail, ok = (arm, "F8") if fg[arm] else ("F8", arm)
            k_ok, out_ok = dec[ok]
            k_fail, out_fail = dec[fail]
            src = out_ok[a][1]
            row = {"frame": i, "segment": r["segment_id"], "image": r["image"], "direction": a, "fails": fail,
                   "fail_color": out_fail[a][0], "ok_picto": PICTOS[int(k_ok["picto"][src])] if src is not None else None}
            if src is None:
                row["cause"], row["detail"] = "other", None
            else:
                row["cause"], row["detail"] = cause(k_ok["box"][src], dets[fail][i], k_fail, out_fail, a, SEL[fail][1])
                row["box"] = [round(x, 4) for x in k_ok["box"][src].tolist()]
            rows.append(row)

    red_cells = sum(1 for r, mask in zip(recs, masks) for a in DIRS if r["color"][a] == "red")
    summary = {"red_cells": red_cells}
    for m in SEL:
        mine = [x for x in rows if x["fails"] == m]
        summary[m] = {"only_this_model_false_go": len(mine), "cause": Counter(x["cause"] for x in mine),
                      "fail_color": Counter(x["fail_color"] for x in mine), "direction": Counter(x["direction"] for x in mine),
                      "ok_picto": Counter(x["ok_picto"] for x in mine),
                      "detail": Counter(f"{x['cause']}:{x['detail']}" for x in mine if x["cause"] in ("color", "routed_away", "outvoted")),
                      "segments": len({x["segment"] for x in mine}), "frames": len({x["frame"] for x in mine}),
                      "top_segments": Counter(x["segment"] for x in mine).most_common(5)}
    net = defaultdict(int)
    for x in rows:
        net[x["cause"]] += 1 if x["fails"] == arm else -1
    summary[f"net_{arm}_minus_F8_by_cause"] = dict(net)
    os.makedirs("runs/exp9_eval", exist_ok=True)
    with open(f"runs/exp9_eval/aval_false_go_{arm}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "WF")
