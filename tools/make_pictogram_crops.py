"""Crops for the pictogram classifier P of the two-stage arm J+P (REPLICATION_PLAN.md, amendment 1).

Lamps come from the unified per-lamp files of the traffic-light-detection project
(``$TLD/datasets/lamps/*.jsonl``; boxes in source pixels, ``pictogram`` and ``state`` per lamp).
Every crop is stored once, square, centred on the box, of side STORE_MARGIN x the box's long side,
at STORE px, so the trainer can re-crop with scale / shift jitter; the classifier's own input is the
centre 1.25 x long side (see ``dtsu.pictogram``).

Sets (written to data/pictogram_crops/<set>.npz + <set>_img.npy):
- train: DTLD train fit sessions + ATLAS train (85% of sessions, seed 0);
- val:   DTLD train tune sessions + the other 15% of ATLAS train sessions (model selection only);
- atlas_test, lisa_test: ground-truth lamps of those test sets (reported, never used to choose);
- openlane: the hand-labelled OpenLane-V2 val lights (annotations/openlane_pictograms_v1.json).
Classes: dtsu.pictogram.CLASSES. DTLD pedestrian / bicycle / pedestrian_bicycle / tram -> other;
unknown or missing pictograms, and lamps without a red / yellow / green state, are left out. Per set and
source, circles and others are capped (``CAPS``) because DTLD and ATLAS are mostly circles.
"""
import collections
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.pictogram import CLASSES, COLORS, STORE, STORE_MARGIN  # noqa: E402

TLD = os.environ.get("TLD", "../traffic-light-detection")
LAMPS = os.path.join(TLD, "datasets", "lamps")
OUT = "data/pictogram_crops"
STATE = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}
OTHER = {"pedestrian", "bicycle", "pedestrian_bicycle", "tram"}
CAPS = {"circle": 15000, "other": 8000}  # per set and source, before the val sets are split off
MIN_SIDE = 6  # px, long side


def pictogram(p):
    if p in OTHER:
        return "other"
    return p if p in CLASSES else None


def read(name):
    return [json.loads(l) for l in open(os.path.join(LAMPS, f"{name}.jsonl"))]


def lamps_of(frames, dataset, facing_front=False):
    out = []
    for r in frames:
        for lp in r["lamps"]:
            cls, color = pictogram(lp["pictogram"]), STATE.get(lp["state"])
            x1, y1, x2, y2 = lp["box"]
            if cls is None or color is None or max(x2 - x1, y2 - y1) < MIN_SIDE:
                continue
            if facing_front and lp.get("facing") not in (None, "front"):
                continue
            out.append({"image": os.path.join(TLD, r["image"]), "box": lp["box"], "cls": cls, "color": color,
                        "dataset": dataset, "session": r["session"]})
    return out


def cap(lamps, rng):
    by = collections.defaultdict(list)
    for lp in lamps:
        by[lp["cls"]].append(lp)
    out = []
    for c, ls in by.items():
        out += rng.sample(ls, CAPS[c]) if c in CAPS and len(ls) > CAPS[c] else ls
    return out


def crop_image(args):
    path, boxes = args
    img = Image.open(path).convert("RGB")
    out = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, STORE_MARGIN * max(x2 - x1, y2 - y1)
        c = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))
        out.append(np.asarray(c.resize((STORE, STORE), Image.BILINEAR), dtype=np.uint8))
    return out


def write(name, lamps, workers):
    by_img = collections.defaultdict(list)
    for k, lp in enumerate(lamps):
        by_img[lp["image"]].append(k)
    arr = np.lib.format.open_memmap(os.path.join(OUT, f"{name}_img.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(lamps), STORE, STORE, 3))
    jobs = [(p, [lamps[k]["box"] for k in ks]) for p, ks in by_img.items()]
    with ProcessPoolExecutor(workers) as ex:
        for n, ((p, _), crops) in enumerate(zip(jobs, ex.map(crop_image, jobs, chunksize=16))):
            for k, c in zip(by_img[p], crops):
                arr[k] = c
            if n % 2000 == 0:
                print(f"{name}: {n}/{len(jobs)} images", flush=True)
    arr.flush()
    np.savez(os.path.join(OUT, f"{name}.npz"),
             label=np.array([CLASSES.index(lp["cls"]) for lp in lamps]),
             color=np.array([COLORS.index(lp["color"]) for lp in lamps]),
             dataset=np.array([lp["dataset"] for lp in lamps]), session=np.array([lp["session"] for lp in lamps]),
             box=np.array([lp["box"] for lp in lamps], dtype=np.float32), image=np.array([lp["image"] for lp in lamps]),
             item=np.array([lp.get("item", "") for lp in lamps]))
    print(f"{name}: {len(lamps)} crops {dict(collections.Counter((lp['dataset'], lp['cls']) for lp in lamps))}", flush=True)


def main():
    rng = random.Random(0)
    os.makedirs(OUT, exist_ok=True)
    workers = min(8, os.cpu_count() or 4)

    tune = {json.loads(l)["segment_id"] for l in open("data/labels/tune/dtld_train_tune.jsonl")}
    dtld = lamps_of(read("dtld_train"), "dtld", facing_front=True)
    atlas = lamps_of(read("atlas_full_train"), "atlas")
    sessions = sorted({lp["session"] for lp in atlas})
    held = set(random.Random(0).sample(sessions, round(0.15 * len(sessions))))
    sets = {
        "train": cap([lp for lp in dtld if lp["session"] not in tune], rng) + cap([lp for lp in atlas if lp["session"] not in held], rng),
        "val": cap([lp for lp in dtld if lp["session"] in tune], rng) + cap([lp for lp in atlas if lp["session"] in held], rng),
        "atlas_test": cap(lamps_of(read("atlas_full_test"), "atlas_test"), rng),
        "lisa_test": cap(lamps_of(read("lisa_test"), "lisa_test"), rng),
    }
    labels = json.load(open("annotations/openlane_pictograms_v1.json"))["labels"]
    ol = []
    for l in open("data/annotation/pictogram_v1/items.jsonl"):
        it = json.loads(l)
        lab = labels.get(it["id"])
        if it["qc_of"] or lab not in CLASSES:  # unclear and arrow_uturn (1 item) are not P classes
            continue
        ol.append({"image": os.path.join("data/OpenLane-V2", f"subset_{it['subset']}", it["image"]), "box": it["box"],
                   "cls": lab, "color": it["color"], "dataset": f"openlane_{it['subset']}",
                   "session": it["segment_id"], "item": it["id"]})
    sets["openlane"] = ol
    for name, lamps in sets.items():
        write(name, lamps, workers)


if __name__ == "__main__":
    main()
