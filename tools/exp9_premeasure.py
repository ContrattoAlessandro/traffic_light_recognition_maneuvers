"""Exp 9 (REPLICATION_PLAN.md): measurements taken before fixing the design. No training.

1. Pictogram and color counts of the F8 training set, the class weights (inverse frequency ** 0.5) with and
   without a horizontal flip that swaps left / right arrows.
2. F8 (epoch 3, threshold 0.45) pictogram recall by lamp height on DTLD test, ATLAS test and LISA test, matched lights.
3. F8 pictogram on the hand-labelled OpenLane-V2 val lights (annotations/openlane_pictograms_v1.json).
4. F8 color confusion of yellow lamps by set.
Uses the detection caches of tools/eval_exp8.py; writes runs/exp8_bdd_F/exp8_eval/exp9_premeasure.json.
"""
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_rfdetr_perception as perc  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS, parse  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.exp8_attributes import dtld, openlane, perception  # noqa: E402

THR = 0.45
SWAP = {"arrow_left": "arrow_right", "arrow_right": "arrow_left", "arrow_straight_left": "arrow_straight_right",
        "arrow_straight_right": "arrow_straight_left"}
HEIGHTS = (0, 10, 15, 20, 30, 1e9)


def weights(counts, power=0.5):
    """w_c ∝ n_c ** -power, scaled so the count-weighted mean is 1 (the loss scale of plain cross-entropy)."""
    n = np.array(list(counts.values()), float)
    w = n ** -power
    w *= n.sum() / (n * w).sum()
    return {k: round(float(v), 3) for k, v in zip(counts, w)}


def train_counts():
    d = json.load(open("data/rfdetr_coco_exp8/train/_annotations.coco.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    p, c = collections.Counter(), collections.Counter()
    for a in d["annotations"]:
        pi, _, ci = parse(cats[a["category_id"]])
        if pi != "?":
            p[pi] += 1
        if ci != "?":
            c[ci] += 1
    flip = collections.Counter()
    for k, n in p.items():  # flip with probability 0.5: half of each class moves to its mirror
        flip[k] += n / 2
        flip[SWAP.get(k, k)] += n / 2
    p = dict(sorted(p.items(), key=lambda x: -x[1]))
    flip = {k: round(flip[k]) for k in p}
    return {"picto": p, "color": dict(c), "picto_weights": weights(p), "picto_weights_flip": weights(flip),
            "picto_counts_flip": flip, "color_weights": weights(dict(c))}


def matched(items, preds):
    """-> rows (gt picto, pred picto, gt color, pred color, height) for lights matched above the threshold."""
    rows = []
    for it, (s, p, c, j, b) in zip(items, preds):
        ign = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
        m = perc.match(b, s, it["boxes"], ign, "center")
        for q in torch.nonzero((s > THR) & (m >= 0)).flatten().tolist():
            g = int(m[q])
            h = float(it["boxes"][g, 3] - it["boxes"][g, 1])
            rows.append((it["picto"][g], PICTOS[int(p[q])], it["color"][g], COLORS[int(c[q])], h))
    return rows


def by_height(rows):
    out = {}
    for lo, hi in zip(HEIGHTS[:-1], HEIGHTS[1:]):
        r = [x for x in rows if lo <= x[4] < hi and x[0] not in (None, "?")]
        cell = {}
        for cls in ("circle", "arrow_left", "arrow_right", "arrow_straight"):
            k = [x for x in r if x[0] == cls]
            cell[cls] = [sum(x[1] == cls for x in k), len(k)]
        arrows = [x for x in r if x[0].startswith("arrow")]
        cell["arrow_read_as_circle"] = [sum(x[1] in ("circle", "unknown") for x in arrows), len(arrows)]
        out[f"{lo:g}-{hi:g}px"] = cell
    return out


def openlane_hand():
    labels = json.load(open("annotations/openlane_pictograms_v1.json"))["labels"]
    want = {}
    for line in open("data/annotation/pictogram_v1/items.jsonl"):
        it = json.loads(line)
        if it["qc_of"] is None and it["id"] in labels:
            for bi in it["box_indices"]:
                want[(it["subset"], it["segment_id"], str(it["timestamp"]), bi)] = labels[it["id"]]
    conf = collections.Counter()
    for subset, name in (("A", "A_val"), ("B", "B_val")):
        recs = load_records(*SETS[name])
        items, preds = openlane(name)
        for r, it, (s, p, c, j, b) in zip(recs, items, preds):
            gt = {bi: lab for (sub, seg, ts, bi), lab in want.items() if sub == subset and seg == r["segment_id"] and ts == str(r["timestamp"])}
            if not gt:
                continue
            ign = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
            m = perc.match(b, s, it["boxes"], ign, "center")
            hit = {int(m[q]): q for q in torch.nonzero((s > THR) & (m >= 0)).flatten().tolist()}
            for bi, lab in gt.items():
                conf[(lab, PICTOS[int(p[hit[bi]])] if bi in hit else "not detected")] += 1
    circ = {k[1]: n for k, n in conf.items() if k[0] == "circle"}
    found = sum(n for k, n in circ.items() if k != "not detected")
    arrows = sum(n for k, n in circ.items() if k.startswith("arrow"))
    return {"confusion": {f"{a} -> {b}": n for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])},
            "circles_detected": found, "circles_called_arrow": arrows, "false_arrow_rate_on_circles": arrows / max(found, 1)}


def main():
    out = {"train": train_counts()}
    print(json.dumps(out["train"], indent=1), flush=True)
    sets = {"DTLD_test": dtld(), "ATLAS_test": perception("atlas"), "LISA_test": perception("lisa")}
    rows = {k: matched(*v) for k, v in sets.items()}
    out["picto_by_height"] = {k: by_height(r) for k, r in rows.items()}
    print(json.dumps(out["picto_by_height"], indent=1), flush=True)
    for ds in ("bdd100k", "vzc"):
        rows[ds] = matched(*perception(ds))
    yellow = {}
    for k, r in rows.items():
        y = collections.Counter(x[3] for x in r if x[2] == "yellow")
        yellow[k] = dict(y)
    out["yellow_read_as"] = yellow
    print(json.dumps(yellow), flush=True)
    out["openlane_hand_labels"] = openlane_hand()
    print(json.dumps(out["openlane_hand_labels"], indent=1), flush=True)
    with open("runs/exp8_bdd_F/exp8_eval/exp9_premeasure.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
