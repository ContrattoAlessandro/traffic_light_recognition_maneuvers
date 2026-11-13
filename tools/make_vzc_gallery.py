"""Manual review page of VZC-TLD: the Exp 7 model and the mixed moves model against the VZC labels, on 100 images.

Selection (seed 0, all 2,988 frames, none used to train any model), by whether each model's image decision equals
the VZC label: 25 both right, 25 only Exp 7 right, 25 only the moves model right, 25 both wrong. Selected by
outcome to show disagreements, so the page is not a sample of the error rate.
Image decision as in tools/eval_vzc.py: the highest-scoring light the model calls relevant; red or yellow = stop,
green = go, nothing = no relevant light. Each image has three views: VZC labels, Exp 7 and moves model.
Writes data/review_vzc/*.jpg, items.json and index.html (from tools/review_vzc.html).
"""
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.picto_rules import PICTOS  # noqa: E402
from tools.eval_vzc import MODELS, OUT as CACHE, decoder, load_items  # noqa: E402
from tools.make_picto_gallery import SHORT  # noqa: E402
from tools.make_review_gallery import render  # noqa: E402

OUT = "data/review_vzc"
SHOWN = {"f7": ("Exp7 F", "Exp7_F.pt"), "mix": ("mix", "mix.pt")}
PICTO_TEXT = {**SHORT, "arrow_straight_right": "straight+right arrow"}
PER_GROUP = 25
LABEL = {"RR": "stop", "RG": "go", "NoR": "none"}


def model_output(name, item, pred):
    """-> (render detections, image decision)."""
    _, _, decode = decoder(name)
    kind, _, _, thr, _, _ = MODELS[name]
    c, s, b = pred
    k = s > thr
    dets, top = [], None
    for ci, si, bi in zip(c[k].tolist(), s[k].tolist(), b[k].tolist()):
        rel, what, color = decode(ci)
        shown = set(what) if kind == "moves" and rel else {PICTO_TEXT[what]} if rel else None
        dets.append((shown, color, si, bi))
        if rel and (top is None or si > top[0]):
            top = (si, color)
    return dets, "none" if top is None else "go" if top[1] == "green" else "stop"


def label_output(item):
    dets = []
    for box, color, relevant in zip(item["boxes"].tolist(), item["color"], item["relevant"]):
        dets.append(({"relevant"} if relevant else None, color or "none", 1.0, box))
    return dets


def main():
    os.makedirs(OUT, exist_ok=True)
    items = load_items()
    preds = {k: torch.load(os.path.join(CACHE, f), weights_only=False) for k, (_, f) in SHOWN.items()}
    groups = {g: [] for g in ("both right", "only Exp 7 right", "only moves right", "both wrong")}
    decided = {}
    for i, it in enumerate(items):
        o = {k: model_output(SHOWN[k][0], it, preds[k][i]) for k in SHOWN}
        decided[i] = o
        gt = LABEL[it["label"]]
        a, b = o["f7"][1] == gt, o["mix"][1] == gt
        groups["both right" if a and b else "only Exp 7 right" if a else "only moves right" if b else "both wrong"].append(i)
    print({g: len(v) for g, v in groups.items()}, flush=True)
    rng = random.Random(0)
    out = []
    for g, pool in groups.items():
        for i in rng.sample(pool, PER_GROUP):
            it, o = items[i], decided[i]
            fid = f"vzc_{i}"
            render(it["path"], label_output(it), os.path.join(OUT, f"{fid}_label.jpg"), show_score=False)
            for k in SHOWN:
                render(it["path"], o[k][0], os.path.join(OUT, f"{fid}_{k}.jpg"))
            out.append({"id": fid, "group": g, "split": it["group"], "file": os.path.basename(it["path"]), "label": LABEL[it["label"]],
                        "f7": o["f7"][1], "mix": o["mix"][1]})
        print(g, "done", flush=True)
    with open(os.path.join(OUT, "items.json"), "w") as f:
        json.dump(out, f, indent=1)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_vzc.html"), encoding="utf-8").read()
    assert "/*ITEMS*/[]" in page
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page.replace("/*ITEMS*/[]", json.dumps(out)))
    print(len(out), "items ->", OUT)


if __name__ == "__main__":
    main()
