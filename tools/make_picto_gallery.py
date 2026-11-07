"""Manual review page of the per-maneuver targets of the Exp 6 joint model (runs/exp6_picto_J).

100 images, the model at its selected epoch and threshold (DTLD tune):
- DTLD test, 34 scored frames: 17 where the rule's prediction equals the label on every maneuver, 17 where
  it does not (random, seed 0). Prediction and label are shown.
- ATLAS test and LISA test, 33 each: random images (seed 0) with at least one annotated colored light,
  half of them with an annotated arrow. No target labels, so only the prediction is shown.
The same junction rule decodes every image (dtsu.picto_rules.rule: arrows set their own maneuvers, the
circle sets the rest including right, U-turn follows left).
Each image shows the junction lights the model detects (box in the light's color, label "pictogram | color
score") and, thin and grey, the lights it assigns to no junction (other junction, pedestrian, bike, tram).
Writes data/review_picto/*.jpg, items.json and index.html (same review page as tools/make_review_gallery.py).
"""
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.picto_rules import COLORS, PICTOS, rule  # noqa: E402
from tools.eval_picto import directions, kept, load_model  # noqa: E402
from tools.eval_picto_perception import predictions  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS  # noqa: E402
from tools.make_review_gallery import render  # noqa: E402

RUN, ARM, OUT = "runs/exp6_picto_J", "J", "data/review_picto"
DATASET_DIR = "data/rfdetr_coco_dtld_picto"
N_DTLD, N_OOD = 34, 33
SHORT = {"circle": "circle", "arrow_left": "left arrow", "arrow_right": "right arrow", "arrow_straight": "straight arrow",
         "arrow_straight_left": "straight+left arrow", "unknown": "unknown", "other": "ped/bike/tram"}


def dtld_items(sel, class_names):
    recs = [json.loads(l) for l in open(os.path.join(DATASET_DIR, "dtld_test_records.jsonl"))]
    dets = torch.load(os.path.join(RUN, "picto_eval", f"e{sel['epoch']}_test.pt"), weights_only=False)
    thr = sel["threshold"]
    ok, bad = [], []
    for i, r in enumerate(recs):
        if r["excluded"] is None and any(b["junction"] for b in r["boxes"]):
            (ok if directions(kept(dets[i], thr)) == r["color"] else bad).append(i)
    rng = random.Random(0)
    pick = [(i, True) for i in rng.sample(ok, N_DTLD // 2)] + [(i, False) for i in rng.sample(bad, N_DTLD - N_DTLD // 2)]
    items = []
    for i, good in pick:
        r, d = recs[i], kept(dets[i], thr)
        w, h = Image.open(r["image"]).size
        boxes = box_convert(d["box"], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
        dl = [({SHORT[PICTOS[int(p)]]} if j > 0.5 else None, COLORS[int(c)], float(s), b)
              for p, c, j, s, b in zip(d["picto"], d["color"], d["junction"], d["score"], boxes.tolist())]
        fid = f"dtld_{i}"
        render(r["image"], dl, os.path.join(OUT, f"{fid}.jpg"))
        items.append({"id": fid, "part": "test", "set": "DTLD_test", "auto": "correct" if good else "wrong", "rule": "junction",
                      "file": os.path.basename(r["image"]), "pred": directions(d), "gt": r["color"]})
    return items


def ood_items(ds, net, res, class_names, thr):
    pool = [it for it in LOADERS[ds]() if any(c is not None for c in it["color"])]
    arrows = [it for it in pool if any(a is not None for a, c in zip(it["arrow"], it["color"]) if c is not None)]
    rest = [it for it in pool if not any(a is not None for a, c in zip(it["arrow"], it["color"]) if c is not None)]
    rng = random.Random(0)
    sel = rng.sample(arrows, N_OOD // 2) + rng.sample(rest, N_OOD - N_OOD // 2)
    items = []
    for n, (it, (v, s, b)) in enumerate(zip(sel, predictions(net, sel, class_names, res, ARM, 0))):
        k = s > thr
        v, s, b = v[k], s[k], b[k]
        p, j, c = v // 6, (v // 3) % 2, v % 3
        dl = [({SHORT[PICTOS[int(pi)]]} if ji else None, COLORS[int(ci)], float(si), bi)
              for pi, ji, ci, si, bi in zip(p, j, c, s, b.tolist())]
        pred = rule([(PICTOS[int(pi)], COLORS[int(ci)], float(si)) for pi, ji, ci, si in zip(p, j, c, s) if ji])
        fid = f"{ds}_{n}"
        render(it["path"], dl, os.path.join(OUT, f"{fid}.jpg"))
        items.append({"id": fid, "part": "ood", "set": ds, "auto": None, "rule": "junction",
                      "file": os.path.relpath(it["path"]).replace("\\", "/"), "group": it["group"], "pred": pred, "gt": None,
                      "has_arrow": n < N_OOD // 2})
    return items


def main():
    os.makedirs(OUT, exist_ok=True)
    class_names = json.load(open(os.path.join(DATASET_DIR, "class_names.json")))
    sel = json.load(open(os.path.join(RUN, "picto_eval", "result.json")))
    items = dtld_items(sel, class_names)
    print("dtld done", flush=True)
    net, res = load_model(RUN, sel["epoch"], class_names, ARM)
    for ds in ("atlas", "lisa"):
        items += ood_items(ds, net, res, class_names, sel["threshold"])
        print(ds, "done", flush=True)
    with open(os.path.join(OUT, "items.json"), "w") as f:
        json.dump(items, f, indent=1)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_gallery.html"), encoding="utf-8").read()
    for a, b in (("/*ITEMS*/[]", json.dumps(items)), ("review_mix_v1", "review_picto_v1"),
                 ("TITLE", "Exp 6 joint model (junction lights, pictogram, color)"),
                 ("DTLD_US_test: \"DTLD test (D*_US)\"", "DTLD_test: \"DTLD test\", DTLD_US_test: \"DTLD test (D*_US)\""),
                 ("Boxes: a relevant light the model detected, in that light's color, labelled with the moves it controls and its score.",
                  "Boxes: a light the model assigns to the ego's junction, in that light's color, labelled with its pictogram, color and score."),
                 ("Thin grey boxes: lights the model calls not relevant.",
                  "Thin grey boxes: lights it assigns to no junction (another junction, or pedestrian / bike / tram signals)."),
                 ("zooms in on the relevant lights.",
                  "zooms in on the junction lights. The targets come from the junction rule: an arrow sets its own maneuvers, "
                  "the circle sets the rest (right included), U-turn follows left.")):
        assert a in page, a
        page = page.replace(a, b)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(len(items), "items ->", OUT)


if __name__ == "__main__":
    main()
