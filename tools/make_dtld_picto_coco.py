"""Export DTLD for Exp 6 (EXPERIMENT_LOG.md) as an RF-DETR COCO dataset with "<pictogram>|<J|N>|<color>" classes.

train/ = DTLD fit sessions, valid/ = DTLD tune sessions (the session split of data/labels/tune/dtld_train_*),
and dtld_test_records.jsonl for the official test split (not a COCO folder, so RF-DETR never evaluates it).
Every frame of those sessions is exported; boxes and labels come from dtsu.picto_rules.dtld_boxes. Both
arms (joint and factorized) train on this same export; the factorized one maps each class back to its
three attributes. Each folder also gets records.jsonl: per frame the image path, the boxes, the session,
and the rule's ground truth ("color", or "excluded" with the reason). Images are referenced, not copied.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.picto_rules import class_name, gt_status, dtld_boxes  # noqa: E402

DATASETS = os.environ.get("DATASETS", "../traffic-light-detection/datasets")
RAW, JPG = os.path.join(DATASETS, "DTLD", "v2.0"), os.path.join(DATASETS, "DTLD_jpg")
SIZE = (2048, 1024)


def records(split, sessions=None):
    out = []
    for im in json.load(open(os.path.join(RAW, f"DTLD_{split}.json")))["images"]:
        path = im["image_path"].replace("\\", "/")
        ses = "/".join(path.split("/")[-4:-1])
        if sessions is not None and ses not in sessions:
            continue
        boxes = dtld_boxes(im)
        gt, why = gt_status(boxes)
        out.append({"segment_id": ses, "image": os.path.abspath(os.path.join(JPG, split, os.path.basename(path).rsplit(".", 1)[0] + ".jpg")),
                    "boxes": boxes, "color": gt, "excluded": why})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--limit", type=int, help="first N frames per folder (smoke tests)")
    args = ap.parse_args()

    split_of = {}
    for part in ("fit", "tune"):
        for line in open(f"data/labels/tune/dtld_train_{part}.jsonl"):
            split_of[json.loads(line)["segment_id"].replace("\\", "/").split("DTLD_jpg/")[-1]] = part
    # segment ids there are "<City>/<CityN>/<date>" relative paths: keep the last three parts
    split_of = {"/".join(k.split("/")[-3:]): v for k, v in split_of.items()}
    train = records("train")
    missing = {r["segment_id"] for r in train} - set(split_of)
    assert not missing, f"sessions without a fit/tune split: {sorted(missing)[:5]}"
    folders = {"train": [r for r in train if split_of[r["segment_id"]] == "fit"],
               "valid": [r for r in train if split_of[r["segment_id"]] == "tune"]}
    test = records("test")
    if args.limit:
        folders = {k: v[: args.limit] for k, v in folders.items()}
        test = test[: args.limit]
    # classes from fit; a smoke export (--limit) takes them from every folder so valid/ has no unknown class
    class_names = sorted({class_name(b["picto"], b["junction"], b["color"])
                          for r in (folders["train"] + (folders["valid"] if args.limit else [])) for b in r["boxes"]})
    unseen = collections.Counter(class_name(b["picto"], b["junction"], b["color"]) for r in folders["valid"] + test for b in r["boxes"]
                                 if class_name(b["picto"], b["junction"], b["color"]) not in class_names)
    print(f"{len(class_names)} classes; valid/test boxes of a class unseen in fit: {dict(unseen)}")

    for k, recs in folders.items():
        images, anns = [], []
        for i, r in enumerate(recs):
            images.append({"id": i, "file_name": r["image"], "width": SIZE[0], "height": SIZE[1]})
            for b in r["boxes"]:
                x1, y1, x2, y2 = b["box"]
                anns.append({"id": len(anns), "image_id": i, "category_id": class_names.index(class_name(b["picto"], b["junction"], b["color"])),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
        with open(os.path.join(args.out, k, "_annotations.coco.json"), "w") as f:
            json.dump({"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n, "supercategory": "light"} for i, n in enumerate(class_names)]}, f)
        with open(os.path.join(args.out, k, "records.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    with open(os.path.join(args.out, "dtld_test_records.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in test)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=0)

    for k, recs in (*folders.items(), ("test", test)):
        ex = collections.Counter(r["excluded"] for r in recs)
        boxes = [b for r in recs for b in r["boxes"]]
        print(f"{k}: {len(recs)} frames, {len(boxes)} boxes ({sum(b['junction'] for b in boxes)} junction), "
              f"scored frames {ex[None]}, excluded {dict((a, n) for a, n in ex.items() if a)}")
        print("   pictograms", dict(collections.Counter(b["picto"] for b in boxes).most_common()))


if __name__ == "__main__":
    main()
