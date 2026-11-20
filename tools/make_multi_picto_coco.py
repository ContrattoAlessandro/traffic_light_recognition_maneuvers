"""Export the Exp 7 training set (EXPERIMENT_LOG.md): DTLD + ATLAS + OpenLane-V2 with partial labels.

Classes are "<pictogram>|<J|N|?>|<color>" (dtsu.picto_rules.class_name); "?" marks an attribute the source
does not label, which the factorized loss ignores:
- DTLD fit (every frame; the Exp 6 export data/rfdetr_coco_dtld_picto/train): pictogram, junction, color;
- ATLAS train, the 85% of sessions not held out (seed 0, the split of tools/make_pictogram_crops.py):
  pictogram and color, junction "?"; "off" lamps are left out, red_yellow -> red;
- OpenLane-V2 A fit + B fit (kept frames, as the moves models): junction = the label builder's "relevant",
  color, pictogram "?";
- with --add-bdd (Exp 8) BDD100K train, the frames with at least one light where every light has a color:
  color only, pictogram and junction "?".
valid/ (RF-DETR's own per-epoch mAP log only) = DTLD tune + OpenLane A tune + B tune. Model selection uses
the tune records directly (tools/eval_multi_picto.py). Images are referenced by absolute path.
"""
import argparse
import collections
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.data import load_records  # noqa: E402
from dtsu.picto_rules import PICTOS, STATE, UNKNOWN, class_name  # noqa: E402

TLD = os.path.abspath(os.environ.get("TLD", "../traffic-light-detection"))
EXP6 = "data/rfdetr_coco_dtld_picto"
L, R = "data/labels", "data/OpenLane-V2"
OPENLANE = {"train": [(f"{L}/tune/subset_A_train_fit.jsonl", f"{R}/subset_A"), (f"{L}/tune/subset_B_train_fit.jsonl", f"{R}/subset_B")],
            "valid": [(f"{L}/tune/subset_A_train_tune.jsonl", f"{R}/subset_A"), (f"{L}/tune/subset_B_train_tune.jsonl", f"{R}/subset_B")]}


def dtld(folder):
    out = []
    for line in open(os.path.join(EXP6, folder, "records.jsonl")):
        r = json.loads(line)
        out.append({"source": "dtld", "image": r["image"], "size": (2048, 1024),
                    "boxes": [(b["box"], class_name(b["picto"], b["junction"], b["color"])) for b in r["boxes"]]})
    return out


def atlas_train():
    frames = [json.loads(l) for l in open(os.path.join(TLD, "datasets", "lamps", "atlas_full_train.jsonl"))]
    sessions = sorted({r["session"] for r in frames})
    held = set(random.Random(0).sample(sessions, round(0.15 * len(sessions))))
    out = []
    for r in frames:
        if r["session"] in held:
            continue
        boxes = []
        for lp in r["lamps"]:
            if lp["pictogram"] not in PICTOS or lp["state"] not in STATE:
                continue
            boxes.append((lp["box"], class_name(lp["pictogram"], None, STATE[lp["state"]])))
        out.append({"source": "atlas", "image": os.path.join(TLD, r["image"]), "size": (r["width"], r["height"]), "boxes": boxes})
    return out, len(held), len(sessions)


def openlane(split):
    out, sizes = [], {}
    for lab, root in OPENLANE[split]:
        for r in load_records(lab, root):
            path = os.path.abspath(os.path.join(root, r["image"]))
            if root not in sizes:
                with Image.open(path) as im:
                    sizes[root] = im.size
            out.append({"source": "openlane_" + root[-1], "image": path, "size": sizes[root],
                        "boxes": [(b["box"], class_name(UNKNOWN, b["relevant"], b["color"])) for b in r["boxes"]]})
    return out


def bdd_train():
    """Exp 8: frames with at least one light where every light has a color (color "none" boxes fit neither target nor background)."""
    out = []
    for line in open(os.path.join(TLD, "datasets", "lamps", "bdd100k_train.jsonl")):
        r = json.loads(line)
        if r["lamps"] and all(lp["state"] in STATE for lp in r["lamps"]):
            out.append({"source": "bdd", "image": os.path.join(TLD, r["image"]), "size": (r["width"], r["height"]),
                        "boxes": [(lp["box"], class_name(UNKNOWN, None, STATE[lp["state"]])) for lp in r["lamps"]]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/rfdetr_coco_multi_picto")
    ap.add_argument("--limit", type=int, help="first N frames per source and folder (smoke tests)")
    ap.add_argument("--add-bdd", action="store_true", help="Exp 8: also BDD100K train clean frames (color only)")
    args = ap.parse_args()

    at, n_held, n_ses = atlas_train()
    print(f"ATLAS train: {len(at)} frames from {n_ses - n_held} of {n_ses} sessions")
    folders = {"train": {"dtld": dtld("train"), "atlas": at, "openlane": openlane("train")},
               "valid": {"dtld": dtld("valid"), "openlane": openlane("valid")}}
    if args.add_bdd:
        folders["train"]["bdd"] = bdd_train()
    if args.limit:
        folders = {k: {s: v[: args.limit] for s, v in f.items()} for k, f in folders.items()}
    folders = {k: [r for v in f.values() for r in v] for k, f in folders.items()}
    class_names = sorted({c for k in folders for r in folders[k] for _, c in r["boxes"]})
    print(f"{len(class_names)} classes")
    for k, recs in folders.items():
        images, anns = [], []
        for i, r in enumerate(recs):
            images.append({"id": i, "file_name": r["image"], "width": r["size"][0], "height": r["size"][1]})
            for (x1, y1, x2, y2), c in r["boxes"]:
                anns.append({"id": len(anns), "image_id": i, "category_id": class_names.index(c),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
        with open(os.path.join(args.out, k, "_annotations.coco.json"), "w") as f:
            json.dump({"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n, "supercategory": "light"} for i, n in enumerate(class_names)]}, f)
        by = collections.Counter(r["source"] for r in recs)
        boxes = collections.Counter(r["source"] for r in recs for _ in r["boxes"])
        print(f"{k}: {len(images)} images {dict(by)}, {len(anns)} boxes {dict(boxes)}")
        attr = collections.Counter()
        for r in recs:
            for _, c in r["boxes"]:
                p, j, col = c.split("|")
                attr["picto labelled"] += p != UNKNOWN
                attr["junction labelled"] += j != UNKNOWN
        print("   ", dict(attr))
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=0)


if __name__ == "__main__":
    main()
