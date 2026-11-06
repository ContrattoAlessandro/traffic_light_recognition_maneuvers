"""Blinded pictogram-annotation pool for the OpenLane-V2 rule oracle (protocol: REPLICATION_PLAN.md, H1).

Strata (kept val frames of subset_A and subset_B, seed 0):
- D: every frame whose labels change when every relevant light is a circle ("dependent", see
     tools/audit_pictogram_rules.py), all of its relevant lights;
- N: a simple random sample of the other frames, proportional to subset size (--n-frames in total),
     all of their relevant lights;
- QC: --qc of the items again under new ids, to measure the annotator's self-agreement.
Near-identical boxes of one frame with the same color (IoU >= 0.8, OpenLane sometimes repeats a light)
are one item; its topology_moves are the union of theirs.

Writes to --out (under data/, which is git-ignored, as the crops are OpenLane-V2 images):
- items.jsonl     the key: id, stratum, subset, frame, box indices, color, topology moves (never shown);
- crops/<id>_t.jpg, crops/<id>_c.jpg   tight and context crops with the light outlined;
- annotator.html  open it in a browser: keyboard labelling, autosave, export of the labels as JSON.
The annotator sees only the crops and the light's annotated color, in shuffled order.
"""
import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_pictogram_rules import decode  # noqa: E402

ROOT = "data/OpenLane-V2"
CLASSES = ["circle", "arrow_left", "arrow_straight", "arrow_right", "arrow_straight_left", "arrow_straight_right",
           "arrow_uturn", "other", "unclear"]
OUTLINE = (255, 0, 255)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def light_groups(rec):
    """Relevant box indices, near-duplicates merged: list of lists of indices into rec['boxes']."""
    groups = []
    for j, b in enumerate(rec["boxes"]):
        if not b["relevant"]:
            continue
        for g in groups:
            first = rec["boxes"][g[0]]
            # same lamp only if the color matches too: a head with a red ball and a green arrow lit can be
            # two overlapping elements of different colors (15 such pairs in val)
            if first["color"] == b["color"] and iou(first["box"], b["box"]) >= 0.8:
                g.append(j)
                break
        else:
            groups.append([j])
    return groups


def crop(img, box, factor, min_side, out_side):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(factor * max(x2 - x1, y2 - y1), min_side)
    l, t = cx - side / 2, cy - side / 2
    c = img.crop((round(l), round(t), round(l + side), round(t + side))).resize((out_side, out_side), Image.BICUBIC)
    s = out_side / side
    pad = 3
    ImageDraw.Draw(c).rectangle([(x1 - l) * s - pad, (y1 - t) * s - pad, (x2 - l) * s + pad, (y2 - t) * s + pad],
                                outline=OUTLINE, width=2)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/annotation/pictogram_v1")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--qc", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    frames = {}
    for s in ("A", "B"):
        recs = [json.loads(l) for l in open(f"data/labels/subset_{s}_val.jsonl")]
        recs = [r for r in recs if r["kept"]]
        dep = [r for r in recs if decode(r, "circle") != r["color"]]
        frames[s] = {"D": dep, "rest": [r for r in recs if decode(r, "circle") == r["color"]]}
    n_rest = {s: len(f["rest"]) for s, f in frames.items()}
    quota = {"A": round(args.n_frames * n_rest["A"] / sum(n_rest.values()))}
    quota["B"] = args.n_frames - quota["A"]

    items = []
    for s in ("A", "B"):
        chosen = [("D", r) for r in frames[s]["D"]] + [("N", r) for r in rng.sample(frames[s]["rest"], quota[s])]
        for stratum, r in chosen:
            for g in light_groups(r):
                b = r["boxes"][g[0]]
                items.append({"stratum": stratum, "subset": s, "segment_id": r["segment_id"], "timestamp": r["timestamp"],
                              "image": r["image"], "box_indices": g, "box": b["box"], "color": b["color"],
                              "topology_moves": sorted({m for j in g for m in r["boxes"][j]["moves"]}), "qc_of": None})
    for it in rng.sample(items, round(args.qc * len(items))):
        items.append({**it, "_src": it})
    rng.shuffle(items)
    used = set()
    for it in items:
        while True:
            i = f"{rng.getrandbits(32):08x}"
            if i not in used:
                used.add(i)
                it["id"] = i
                break
    for it in items:
        if "_src" in it:
            it["qc_of"] = it.pop("_src")["id"]

    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    cache = {}
    for k, it in enumerate(items):
        path = os.path.join(ROOT, f"subset_{it['subset']}", it["image"])
        if path not in cache:
            cache = {path: Image.open(path).convert("RGB")}
        img = cache[path]
        crop(img, it["box"], 2.5, 32, 320).save(os.path.join(args.out, "crops", f"{it['id']}_t.jpg"), quality=92)
        crop(img, it["box"], 16, 220, 480).save(os.path.join(args.out, "crops", f"{it['id']}_c.jpg"), quality=88)
        if k % 200 == 0:
            print(f"crops {k}/{len(items)}", flush=True)
    with open(os.path.join(args.out, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pictogram_annotator.html"), encoding="utf-8").read()
    public = [{"id": it["id"], "color": it["color"]} for it in items]
    html = template.replace("/*ITEMS*/[]", json.dumps(public)).replace("/*CLASSES*/[]", json.dumps(CLASSES))
    with open(os.path.join(args.out, "annotator.html"), "w", encoding="utf-8") as f:
        f.write(html)

    strata = {(it["subset"], it["stratum"], "qc" if it["qc_of"] else "main") for it in items}
    for key in sorted(strata):
        n = sum((it["subset"], it["stratum"], "qc" if it["qc_of"] else "main") == key for it in items)
        print(key, n)
    print(f"{len(items)} items; frames: " + ", ".join(f"{s} D {len(frames[s]['D'])} N {quota[s]} (of {n_rest[s]})" for s in frames))


if __name__ == "__main__":
    main()
