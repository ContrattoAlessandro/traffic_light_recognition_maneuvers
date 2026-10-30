"""Export kept frames as an RF-DETR (Roboflow COCO) dataset with "<moves>|<color>" light classes.

train/  = A+B train (the same frames the direction-aware model trains on)
valid/  = A+B tune split (a subset of train; only feeds RF-DETR's own per-epoch mAP log, never val)
With ``--fit`` (tuning runs, out dir data/rfdetr_coco_fit) train/ is the 90% fit split instead, so
valid/ is held out. ``--dataset dtld`` exports DTLD the same way (fit / tune / official test).
``--probe N`` also writes test/ = N random frames of train/: RF-DETR never evaluates it, but
``tools/train_rfdetr.py`` scores it each epoch exactly like valid/ to measure overfitting.
Each folder gets a records.jsonl with the label records of its images, in order.
The class list always comes from the full train split, so class indices match.
Images are referenced by absolute path, nothing is copied. Also checks that ground-truth boxes
through ``assign_scored`` reproduce every per-direction label (the "oracle" must score TSS = 1).
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
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, box_class, parse_class, to_arrays  # noqa: E402

L, R = "data/labels", "data/OpenLane-V2"
DTLD = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD_jpg")
OPENLANE = {
    "train": [(f"{L}/subset_A_train.jsonl", f"{R}/subset_A"), (f"{L}/subset_B_train.jsonl", f"{R}/subset_B")],
    "fit": [(f"{L}/tune/subset_A_train_fit.jsonl", f"{R}/subset_A"), (f"{L}/tune/subset_B_train_fit.jsonl", f"{R}/subset_B")],
    "valid": [(f"{L}/tune/subset_A_train_tune.jsonl", f"{R}/subset_A"), (f"{L}/tune/subset_B_train_tune.jsonl", f"{R}/subset_B")],
    "val_A": [(f"{L}/subset_A_val.jsonl", f"{R}/subset_A")],
    "val_B": [(f"{L}/subset_B_val.jsonl", f"{R}/subset_B")],
}
DTLD_SPLITS = {
    "train": [(f"{L}/dtld_train.jsonl", f"{DTLD}/train")],
    "fit": [(f"{L}/tune/dtld_train_fit.jsonl", f"{DTLD}/train")],
    "valid": [(f"{L}/tune/dtld_train_tune.jsonl", f"{DTLD}/train")],
    "test": [(f"{L}/dtld_test.jsonl", f"{DTLD}/test")],
}


def oracle(records, class_names):
    """TSS of ground-truth boxes pushed through the class encoding and the direction rules.

    Boxes whose class never occurs in train cannot be predicted and are left out, so on val this is
    the ceiling of the baseline rather than exactly 1.
    """
    colors = []
    for r in records:
        dets = []
        for b in r["boxes"]:
            if box_class(b) not in class_names:
                continue
            moves, color = parse_class(box_class(b))
            if moves is not None:
                dets.append((moves, color, 1.0))
        colors.append(assign_scored(dets, r["traffic_side"], r.get("near_turn", "dedicated")))
    pc, pp = to_arrays(colors)
    gc, gp = to_arrays([r["color"] for r in records])
    return evaluate(pc, pp, gc, gp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/rfdetr_coco")
    ap.add_argument("--fit", action="store_true", help="train/ = fit split (train minus tune)")
    ap.add_argument("--dataset", choices=("openlane", "dtld"), default="openlane")
    ap.add_argument("--probe", type=int, default=0, help="write test/ = this many random train/ frames (seed 0)")
    args = ap.parse_args()

    splits = DTLD_SPLITS if args.dataset == "dtld" else OPENLANE
    recs = {k: [r for lab, root in v for r in load_records(lab, root)] for k, v in splits.items()}
    class_names = sorted({box_class(b) for r in recs["train"] for b in r["boxes"]})
    unseen = collections.Counter(box_class(b) for k in recs if k not in ("train", "fit", "valid") for r in recs[k] for b in r["boxes"]
                                 if box_class(b) not in class_names)
    print(f"{len(class_names)} classes; val boxes with a class unseen in train: {dict(unseen)}")

    folders = {"train": recs["fit" if args.fit else "train"], "valid": recs["valid"]}
    if args.probe:
        folders["test"] = random.Random(0).sample(folders["train"], args.probe)
    sizes = {}
    for k, folder_recs in folders.items():
        images, anns = [], []
        for i, r in enumerate(folder_recs):
            path = os.path.abspath(os.path.join(r["root"], r["image"]))
            if r["root"] not in sizes:
                with Image.open(path) as im:
                    sizes[r["root"]] = im.size
            w, h = sizes[r["root"]]
            images.append({"id": i, "file_name": path, "width": w, "height": h})
            for b in r["boxes"]:
                x1, y1, x2, y2 = b["box"]
                anns.append({"id": len(anns), "image_id": i, "category_id": class_names.index(box_class(b)),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        cats = [{"id": i, "name": n, "supercategory": "light"} for i, n in enumerate(class_names)]
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
        with open(os.path.join(args.out, k, "_annotations.coco.json"), "w") as f:
            json.dump({"images": images, "annotations": anns, "categories": cats}, f)
        with open(os.path.join(args.out, k, "records.jsonl"), "w") as f:
            for r in folder_recs:
                f.write(json.dumps(r) + "\n")
        print(f"{k}: {len(images)} images, {len(anns)} boxes")
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=0)

    for k, rs in recs.items():
        m = oracle(rs, class_names)
        print(f"oracle {k}: {len(rs)} frames, TSS {m['tss']:.4f} EM {m['em']:.4f}")


if __name__ == "__main__":
    main()
