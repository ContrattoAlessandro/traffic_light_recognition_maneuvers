"""Export the training set of the baseline M (Exp 5): joint "<moves>|<color>" light classes (``dtsu.moves``).

train/ = OpenLane-V2 A fit + B fit (kept frames) plus a random sample (seed 0) of the same number of kept D*_US fit
frames, so both domains weigh equally; valid/ = OpenLane-V2 A tune + B tune + D*_US tune (only RF-DETR's own per-epoch
log reads it). Classes are those of train/. Images are referenced by absolute path; nothing is copied. Ground-truth
boxes through ``dtsu.moves.assign_scored`` must reproduce every label of every folder (the oracle check, TSS 1).

Usage (after scripts/build_labels.py): python scripts/export_moves_set.py --out data/train_moves
"""
import argparse
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import labels_dtld, sources  # noqa: E402
from dtsu.evaluation import to_arrays  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.moves import assign_scored, box_class, parse_class  # noqa: E402


def openlane(part):
    sets = sources.openlane_sets()
    return [r for s in ("A", "B") for r in sources.load_records(*sets[f"{s}_{part}"])]


def dtld_us(part):
    return [r for r in sources.read_jsonl(labels_dtld.label_path("us", part)) if r["kept"]]


def oracle(records, names):
    colors = []
    for r in records:
        dets = [(m, c, 1.0) for m, c in (parse_class(box_class(b)) for b in r["boxes"] if box_class(b) in names) if m is not None]
        colors.append(assign_scored(dets, r["traffic_side"], r.get("near_turn", "dedicated")))
    return evaluate(*to_arrays(colors), *to_arrays([r["color"] for r in records]))["tss"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ol_fit = openlane("fit")
    folders = {"train": ol_fit + random.Random(0).sample(dtld_us("fit"), len(ol_fit)),
               "valid": openlane("tune") + dtld_us("tune")}
    names = sorted({box_class(b) for r in folders["train"] for b in r["boxes"]})
    print(f"{len(names)} classes")
    sizes = {}
    for k, recs in folders.items():
        images, anns = [], []
        for i, r in enumerate(recs):
            path = os.path.abspath(os.path.join(r.get("root", ""), r["image"]))
            root = r.get("root", "dtld")  # one image size per OpenLane-V2 subset, and one for DTLD
            if root not in sizes:
                with Image.open(path) as im:
                    sizes[root] = im.size
            w, h = sizes[root]
            images.append({"id": i, "file_name": path, "width": w, "height": h})
            for b in r["boxes"]:
                x1, y1, x2, y2 = b["box"]
                anns.append({"id": len(anns), "image_id": i, "category_id": names.index(box_class(b)),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
        with open(os.path.join(args.out, k, "_annotations.coco.json"), "w") as f:
            json.dump({"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n, "supercategory": "light"} for i, n in enumerate(names)]}, f)
        print(f"{k}: {len(images)} images, {len(anns)} boxes, oracle TSS {oracle(recs, names):.4f}", flush=True)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(names, f, indent=0)
    with open(os.path.join(args.out, "export.json"), "w") as f:
        json.dump({"kind": "moves", "train": len(folders["train"]), "valid": len(folders["valid"])}, f, indent=1)


if __name__ == "__main__":
    main()
