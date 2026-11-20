"""Export the training set as an RF-DETR COCO dataset with partial labels.

Every box gets the class ``"<pictogram>|<J|N|?>|<color>"`` (``dtsu.rules``); "?" marks an attribute the source does
not label, which the factorized loss ignores. Sources (``--sources``, default all four):
- dtld: DTLD fit sessions, every frame (data/labels/dtld_junction_fit.jsonl): pictogram, junction, color;
- atlas: ATLAS train minus the held-out sessions (splits/atlas_heldout_sessions.json): pictogram and color;
  "off" lamps are left out, red-yellow counts as red;
- openlane: OpenLane-V2 A fit + B fit, kept frames: junction (the label builder's "relevant") and color;
- bdd: BDD100K train frames with at least one light where every light has a color: color only.
valid/ (RF-DETR's own per-epoch log; model selection reads the tune label files directly) holds DTLD tune and
OpenLane-V2 A tune + B tune. Images are referenced by absolute path, not copied.

Usage: python scripts/export_training_set.py --out data/train_full [--sources dtld atlas openlane bdd]
"""
import argparse
import collections
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import sources  # noqa: E402
from dtsu.rules import PICTOGRAMS, STATE, UNKNOWN, class_name  # noqa: E402

SOURCES = ("dtld", "atlas", "openlane", "bdd")


def dtld(split):
    return [{"source": "dtld", "image": r["image"], "size": sources.DTLD_SIZE,
             "boxes": [(b["box"], class_name(b["picto"], b["junction"], b["color"])) for b in r["boxes"]]}
            for r in sources.read_jsonl(sources.dtld_junction_path(split))]


def atlas():
    held = sources.atlas_heldout_sessions()
    out = []
    for r in sources.atlas_train_frames():
        if r["session"] in held:
            continue
        boxes = [(lp["box"], class_name(lp["pictogram"], None, STATE[lp["state"]])) for lp in r["lamps"]
                 if lp["pictogram"] in PICTOGRAMS and lp["state"] in STATE]
        out.append({"source": "atlas", "image": r["image"], "size": r["size"], "boxes": boxes})
    return out


def openlane(split):
    out, sizes = [], {}
    for s in ("A", "B"):
        lab, root = sources.openlane_sets()[f"{s}_{split}"]
        for r in sources.load_records(lab, root):
            path = os.path.abspath(os.path.join(root, r["image"]))
            if root not in sizes:
                with Image.open(path) as im:
                    sizes[root] = im.size
            out.append({"source": f"openlane_{s}", "image": path, "size": sizes[root],
                        "boxes": [(b["box"], class_name(UNKNOWN, b["relevant"], b["color"])) for b in r["boxes"]]})
    return out


def bdd():
    return [{"source": "bdd", "image": r["image"], "size": r["size"],
             "boxes": [(lp["box"], class_name(UNKNOWN, None, STATE[lp["state"]])) for lp in r["lamps"]]}
            for r in sources.bdd_train_frames() if r["lamps"] and all(lp["state"] in STATE for lp in r["lamps"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--limit", type=int, help="first N frames per source and folder (smoke tests)")
    args = ap.parse_args()

    readers = {"dtld": lambda: dtld("fit"), "atlas": atlas, "openlane": lambda: openlane("fit"), "bdd": bdd}
    folders = {"train": {s: readers[s]() for s in SOURCES if s in args.sources},
               "valid": {"dtld": dtld("tune"), "openlane": openlane("tune")}}
    if args.limit:
        folders = {k: {s: v[: args.limit] for s, v in f.items()} for k, f in folders.items()}
    folders = {k: [r for v in f.values() for r in v] for k, f in folders.items()}
    class_names = sorted({c for k in folders for r in folders[k] for _, c in r["boxes"]})
    print(f"sources {args.sources}: {len(class_names)} classes")
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
        frames = collections.Counter(r["source"] for r in recs)
        boxes = collections.Counter(r["source"] for r in recs for _ in r["boxes"])
        labelled = collections.Counter()
        for r in recs:
            for _, c in r["boxes"]:
                p, j, _ = c.split("|")
                labelled["pictogram"] += p != UNKNOWN
                labelled["junction"] += j != UNKNOWN
        print(f"{k}: {len(images)} images {dict(frames)}, {len(anns)} boxes {dict(boxes)}, labelled {dict(labelled)}")
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=0)
    with open(os.path.join(args.out, "export.json"), "w") as f:
        json.dump({"sources": args.sources, "limit": args.limit}, f, indent=1)


if __name__ == "__main__":
    main()
