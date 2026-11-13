"""Per-light attribute accuracy of the Exp 8 model (F8: runs/exp8_bdd_F, its selected epoch and threshold) on every set.

Uses the detections cached by tools/eval_exp8.py (no inference). Detections above the threshold, class-agnostic NMS
0.5, matched to the annotations with the center criterion (greedy by score; lights with no color label are ignore
boxes). On the matched lights:
- color: accuracy, per-color recall, confusion;
- pictogram (where the set labels it: DTLD test, ATLAS test, LISA test): accuracy over the 8 pictograms, per-class
  recall, macro recall, arrow-vs-circle accuracy, confusion; DTLD "unknown" lamps have no true shape and are left out;
- pictogram and color both right;
- junction relevance (where the set labels it: DTLD test junction, OpenLane A / B val and VZC "relevant"):
  accuracy, recall and precision of the junction class;
plus detection AP, recall and precision (lower bound where annotation is incomplete: LISA, BDD100K, S2TLD).
95% CIs by cluster bootstrap (1,000 draws, seed 0): segments (OpenLane, DTLD), sessions (VZC), blocks of 100 frames
(LISA, S2TLD), images (ATLAS, BDD100K). Writes runs/exp8_bdd_F/exp8_eval/attributes.json.
"""
import collections
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_rfdetr_perception as perc  # noqa: E402
import tools.eval_vzc as vzc  # noqa: E402
import tools.exp7_checks as ck  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402

RUN = "runs/exp8_bdd_F"
CACHE = os.path.join(RUN, "exp8_eval")
DRAWS = 1000
SHAPE = {None: "circle", ("left",): "arrow_left", ("right",): "arrow_right", ("straight",): "arrow_straight",
         ("left", "straight"): "arrow_straight_left", ("right", "straight"): "arrow_straight_right"}


def shape_of(arrow):
    return SHAPE.get(None if arrow is None else tuple(sorted(arrow)), "?")


# ------------------------------------------------------------------ sets -> items (boxes pixels, color, picto, junction, cluster) + preds
def frame_preds(dets, size):
    """Frame-cache detections (cxcywh normalized, > 0.05) -> list of (score, picto, color, junction, xyxy pixels) after NMS."""
    w, h = size
    out = []
    for d in dets:
        b = box_convert(d["box"], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
        keep = nms(b, d["score"], 0.5) if len(d["score"]) else torch.zeros(0, dtype=torch.long)
        out.append((d["score"][keep], d["picto"][keep], d["color"][keep], (d["junction"][keep] > 0.5).long(), b[keep]))
    return out


def virtual_preds(preds):
    """Perception cache (picto * 6 + junction * 3 + color, score, xyxy) -> the same tuple layout as frame_preds."""
    return [(s, v // 6, v % 3, (v // 3) % 2, b) for v, s, b in preds]


def openlane_items(name):
    """-> (items, image size) of an OpenLane-V2 set; pictogram None, junction = the label builder's "relevant"."""
    recs = load_records(*SETS[name])
    with Image.open(os.path.join(recs[0]["root"], recs[0]["image"])) as im:
        size = im.size
    return [{"boxes": torch.tensor([b["box"] for b in r["boxes"]], dtype=torch.float32).reshape(-1, 4),
              "color": [b["color"] if b["color"] in COLORS else None for b in r["boxes"]], "picto": [None] * len(r["boxes"]),
              "junction": [bool(b["relevant"]) for b in r["boxes"]], "cluster": r["segment_id"]} for r in recs], size


def openlane(name):
    items, size = openlane_items(name)
    return items, frame_preds(torch.load(os.path.join(CACHE, f"e3_{name}.pt"), weights_only=False), size)


DTLD_SIZE = (2048, 1024)


def dtld_items():
    recs = [json.loads(l) for l in open("data/rfdetr_coco_dtld_picto/dtld_test_records.jsonl")]
    return [{"boxes": torch.tensor([b["box"] for b in r["boxes"]], dtype=torch.float32).reshape(-1, 4),
              "color": [b["color"] for b in r["boxes"]], "picto": [None if b["picto"] == "unknown" else b["picto"] for b in r["boxes"]],
              "junction": [bool(b["junction"]) for b in r["boxes"]], "cluster": r["segment_id"]} for r in recs]


def dtld():
    return dtld_items(), frame_preds(torch.load(os.path.join(CACHE, "e3_dtld_test.pt"), weights_only=False), DTLD_SIZE)


def perception_items(ds):
    raw = vzc.load_items() if ds == "vzc" else perc.LOADERS[ds]()
    items = []
    for i, it in enumerate(raw):
        n = len(it["color"])
        cluster = it["session"] if ds == "vzc" else i // 100 if ds in ("lisa", "s2tld") else i
        items.append({"path": it["path"], "boxes": it["boxes"], "color": it["color"], "group": it["group"], "cluster": cluster,
                      "picto": [shape_of(a) for a in it["arrow"]] if ds in ("atlas", "lisa") else [None] * n,
                      "junction": it["relevant"] if ds == "vzc" else [None] * n})
    return items


def perception(ds):
    return perception_items(ds), virtual_preds(torch.load(os.path.join(CACHE, f"e3_{ds}.pt"), weights_only=False))


# ------------------------------------------------------------------ per-image tallies, then sums (so the bootstrap is cheap)
def tally(items, preds, thr):
    """-> per-image arrays of counts, plus confusion lists and the AP table."""
    keys = ("gt", "det", "fp", "hit", "col_ok", "pic_n", "pic_ok", "arr_n", "arr_ok", "both_n", "both_ok",
            "jn", "j_ok", "j_tp", "j_pred", "j_gt")
    per = {k: np.zeros(len(items)) for k in keys}
    col_conf, pic_conf = np.zeros((3, 3), int), collections.Counter()
    ap_items, ap_preds = [], []
    for i, (it, (s, p, c, j, b)) in enumerate(zip(items, preds)):
        ignore = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
        m = perc.match(b, s, it["boxes"], ignore, "center")
        ap_items.append({"boxes": it["boxes"], "color": it["color"]})
        ap_preds.append((None, s, b))
        above = s > thr
        per["gt"][i] = int((~ignore).sum())
        per["det"][i] = int((above & (m != -2)).sum())
        per["fp"][i] = int((above & (m == -1)).sum())
        for q in torch.nonzero(above & (m >= 0)).flatten().tolist():
            g = int(m[q])
            gc, pc = COLORS.index(it["color"][g]), int(c[q])
            per["hit"][i] += 1
            per["col_ok"][i] += gc == pc
            col_conf[gc, pc] += 1
            gp = it["picto"][g]
            if gp is not None and gp != "?":
                pp = PICTOS[int(p[q])]
                per["pic_n"][i] += 1
                per["pic_ok"][i] += gp == pp
                per["both_n"][i] += 1
                per["both_ok"][i] += gp == pp and gc == pc
                pic_conf[(gp, pp)] += 1
                if gp != "other":
                    per["arr_n"][i] += 1
                    per["arr_ok"][i] += (gp == "circle") == (pp in ("circle", "unknown"))
            gj = it["junction"][g]
            if gj is not None:
                pj = bool(j[q])
                per["jn"][i] += 1
                per["j_ok"][i] += gj == pj
                per["j_tp"][i] += gj and pj
                per["j_pred"][i] += pj
                per["j_gt"][i] += gj
    return per, col_conf, pic_conf, (ap_items, ap_preds)


def ratio(per, num, den, w=None):
    w = np.ones(len(per[num])) if w is None else w
    d = (per[den] * w).sum()
    return float((per[num] * w).sum() / d) if d else None


METRICS = {"recall": ("hit", "gt"), "color_acc": ("col_ok", "hit"), "picto_acc": ("pic_ok", "pic_n"),
           "arrow_vs_circle_acc": ("arr_ok", "arr_n"), "picto_and_color_acc": ("both_ok", "both_n"),
           "junction_acc": ("j_ok", "jn"), "junction_recall": ("j_tp", "j_gt"), "junction_precision": ("j_tp", "j_pred")}


def summarize(items, preds, thr):
    per, col_conf, pic_conf, (ai, apr) = tally(items, preds, thr)
    out = {"images": len(items), "lights": int(per["gt"].sum()), "matched": int(per["hit"].sum()),
           "AP": ck.ap_w(ck.det_table(ai, apr), np.ones(len(items))),
           "precision": float((per["det"].sum() - per["fp"].sum()) / max(per["det"].sum(), 1))}
    clusters = np.unique([str(it["cluster"]) for it in items], return_inverse=True)[1]
    nc = clusters.max() + 1
    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, nc, nc), minlength=nc)[clusters].astype(float) for _ in range(DRAWS)]
    for name, (a, b) in METRICS.items():
        v = ratio(per, a, b)
        out[name] = v
        if v is not None:
            bs = [x for x in (ratio(per, a, b, w) for w in weights) if x is not None]
            out[name + "_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["n_picto"], out["n_junction"] = int(per["pic_n"].sum()), int(per["jn"].sum())
    out["color_recall"] = {g: float(col_conf[a, a] / col_conf[a].sum()) if col_conf[a].sum() else None for a, g in enumerate(COLORS)}
    out["color_confusion_gt_x_pred"] = {g: dict(zip(COLORS, map(int, col_conf[a]))) for a, g in enumerate(COLORS)}
    if pic_conf:
        gts = sorted({g for g, _ in pic_conf}, key=PICTOS.index)
        rec = {g: sum(n for (a, p), n in pic_conf.items() if a == g and p == g) / sum(n for (a, _), n in pic_conf.items() if a == g) for g in gts}
        out["picto_recall"] = rec
        out["picto_n"] = {g: sum(n for (a, _), n in pic_conf.items() if a == g) for g in gts}
        out["picto_macro_recall"] = float(np.mean(list(rec.values())))
        out["picto_confusion_gt_x_pred"] = {g: {p: n for (a, p), n in sorted(pic_conf.items(), key=lambda x: -x[1]) if a == g} for g in gts}
    return out


def main():
    sel = json.load(open(os.path.join(CACHE, "result.json")))["selection"]["F8"]
    assert sel["epoch"] == 3
    thr = sel["threshold"]
    sets = {"A_val": openlane("A_val"), "B_val": openlane("B_val"), "DTLD_test": dtld()}
    for ds in ("atlas", "bdd100k", "lisa", "s2tld", "vzc"):
        sets[ds] = perception(ds)
    cells = [("in", "A_val", "A_val", None), ("in", "B_val", "B_val", None), ("in", "DTLD_test", "DTLD_test", None),
             ("in", "ATLAS_test", "atlas", None), ("in", "BDD_val_day", "bdd100k", "daytime"), ("in", "BDD_val_night", "bdd100k", "night"),
             ("in", "BDD_val_all", "bdd100k", None),
             ("ood", "LISA_day", "lisa", "day"), ("ood", "LISA_night", "lisa", "night"), ("ood", "LISA_all", "lisa", None),
             ("ood", "S2TLD_720p", "s2tld", "1280x720"), ("ood", "S2TLD_1080p", "s2tld", "1920x1080"), ("ood", "S2TLD_all", "s2tld", None),
             ("ood", "VZC", "vzc", None)]
    out = {"epoch": sel["epoch"], "threshold": thr}
    for dom, name, ds, group in cells:
        items, preds = sets[ds]
        assert len(items) == len(preds), (name, len(items), len(preds))
        idx = [i for i, it in enumerate(items) if group is None or it.get("group") == group]
        r = summarize([items[i] for i in idx], [preds[i] for i in idx], thr)
        r["domain"] = dom
        out[name] = r
        f = lambda k: "  -  " if r.get(k) is None else f"{r[k]:.3f}"  # noqa: E731
        print(f"{dom:3s} {name:14s} lights {r['lights']:6d} AP {f('AP')} R {f('recall')} P {f('precision')} | color {f('color_acc')} "
              f"| picto {f('picto_acc')} (n {r['n_picto']}) macro {f('picto_macro_recall')} arrow/circle {f('arrow_vs_circle_acc')} "
              f"both {f('picto_and_color_acc')} | junction {f('junction_acc')} R {f('junction_recall')} P {f('junction_precision')}", flush=True)
    with open(os.path.join(CACHE, "attributes.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
