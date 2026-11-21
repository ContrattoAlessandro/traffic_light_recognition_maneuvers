"""Readers for every dataset, from the files each dataset ships (paths in ``dtsu.paths``).

Training sources (``scripts/export_training_set.py``) and what each one labels:

| source | pictogram | junction | color |
|---|---|---|---|
| DTLD v2.0 (Germany) | yes | yes (route lamps + same-size vehicle lamps) | yes |
| ATLAS (Germany) | yes | no | yes |
| OpenLane-V2 subset A + B (US, Singapore) | no | yes (linked to a lane of the ego approach) | yes |
| BDD100K (US) | no | no | yes |

Evaluation sets: DTLD test, OpenLane-V2 val, ATLAS test, LISA test, S2TLD, BDD100K val and VZC-TLD.
Each evaluation reader returns per image a dict with ``path``, ``group``, ``boxes`` (xyxy pixels), ``color`` (None
= ignore box) and ``arrow`` (set of directions, None for round lights), plus dataset-specific keys.
"""
import csv
import glob
import json
import os
import statistics
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.ops import box_convert

from . import DIRECTIONS
from .paths import ANNOTATIONS, DATASETS, LABELS, OPENLANE, SPLITS
from .rules import ARROW_MOVES, PICTOGRAMS, STATE, ground_truth


# ------------------------------------------------------------------ OpenLane-V2 (labels from dtsu.labels)
def openlane_sets():
    """Name -> (label file, image root). *_fit / *_tune split the official train split by segment
    (``splits/openlane_tune_segments.json``); *_val is the official val split, the evaluation set."""
    out = {}
    for s in ("A", "B"):
        root = os.path.join(OPENLANE, f"subset_{s}")
        out[f"{s}_fit"] = (os.path.join(LABELS, f"subset_{s}_train_fit.jsonl"), root)
        out[f"{s}_tune"] = (os.path.join(LABELS, f"subset_{s}_train_tune.jsonl"), root)
        out[f"{s}_val"] = (os.path.join(LABELS, f"subset_{s}_val.jsonl"), root)
    return out


def load_records(path, image_root, kept_only=True):
    """Kept frames of a label file; each record remembers the image root."""
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    recs = [r for r in recs if r["kept"] or not kept_only]
    for r in recs:
        r["root"] = image_root
        for k in ("color", "pass"):  # older label files also carry a "uturn" entry
            if isinstance(r.get(k), dict):
                r[k] = {d: r[k][d] for d in DIRECTIONS}
    return recs


def openlane_val_pictograms():
    """(subset, segment, timestamp, box index) -> blind hand label of the pictogram (``annotations/``), first labels
    only (the 5% repeated for the self-agreement check are left out)."""
    out = {}
    for line in open(os.path.join(ANNOTATIONS, "openlane_val_pictograms.jsonl")):
        it = json.loads(line)
        if it["qc_of"] is None:
            for bi in it["box_indices"]:
                out[(it["subset"], it["segment_id"], str(it["timestamp"]), bi)] = it["pictogram"]
    return out


# ------------------------------------------------------------------ DTLD v2.0
DTLD_OTHER = {"pedestrian", "bicycle", "pedestrian_bicycle", "tram"}
DTLD_VEHICLE = set(ARROW_MOVES) | {"circle", "unknown"}
JUNCTION_RATIO = 0.6
DTLD_SIZE = (2048, 1024)


def dtld_boxes(im):
    """Front-facing lamps with a visible color: pictogram, junction flag and color.
    A lamp belongs to the junction when DTLD marks it relevant to the driven route, or when it is another vehicle
    lamp at least JUNCTION_RATIO x the median route-lamp height (a lamp of the same junction seen from the same
    distance). Pedestrian, bicycle and tram signals are "other" and never junction lamps."""
    front = [l for l in im["labels"] if l["attributes"]["direction"] == "front" and l["attributes"]["state"] in STATE]
    vehicle = [l for l in im["labels"] if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in DTLD_VEHICLE]
    route = [l for l in vehicle if l["attributes"]["relevance"] == "relevant"]
    h = statistics.median(l["h"] for l in route) if route else None
    out = []
    for l in front:
        a = l["attributes"]
        if a["pictogram"] not in DTLD_VEHICLE | DTLD_OTHER:
            continue
        picto = "other" if a["pictogram"] in DTLD_OTHER else a["pictogram"]
        junction = picto != "other" and route and (a["relevance"] == "relevant" or l["h"] >= JUNCTION_RATIO * h)
        out.append({"box": [l["x"], l["y"], l["x"] + l["w"], l["y"] + l["h"]], "picto": picto,
                    "junction": bool(junction), "color": STATE[a["state"]]})
    return out


def dtld_session(image_path):
    return "/".join(image_path.replace("\\", "/").split("/")[-4:-1])


def dtld_records(split):
    """Every frame of a DTLD split: image (JPEG, ``scripts/convert_dtld.py``), boxes, session, and the junction-rule
    ground truth ("color"), or the reason the frame is not scored ("excluded")."""
    raw, jpg = os.path.join(DATASETS, "DTLD", "v2.0"), os.path.join(DATASETS, "DTLD_jpg", split)
    out = []
    for im in json.load(open(os.path.join(raw, f"DTLD_{split}.json")))["images"]:
        path = im["image_path"].replace("\\", "/")
        boxes = dtld_boxes(im)
        gt, why = ground_truth(boxes)
        out.append({"segment_id": dtld_session(path),
                    "image": os.path.abspath(os.path.join(jpg, os.path.basename(path).rsplit(".", 1)[0] + ".jpg")),
                    "boxes": boxes, "color": gt, "excluded": why})
    return out


def dtld_tune_sessions():
    return set(json.load(open(os.path.join(SPLITS, "dtld_tune_sessions.json")))["tune"])


def dtld_junction_path(split):
    """Junction-rule records of a DTLD split: fit and tune (from DTLD train), test (``scripts/export_training_set.py``)."""
    return os.path.join(LABELS, f"dtld_junction_{split}.jsonl")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


# ------------------------------------------------------------------ ATLAS
ATLAS_CAMERAS = ("front_medium", "front_tele", "front_wide")
ATLAS_STATES = ("red_yellow", "red", "yellow", "green")  # longest suffix first


def _atlas_class(name):
    """"<pictogram>_<state>" -> (state, pictogram); "off" -> ("off", None)."""
    if name == "off":
        return "off", None
    for s in ATLAS_STATES:
        if name.endswith("_" + s):
            return s, name[: -len(s) - 1]
    raise ValueError(name)


def _atlas_stamp(stem):
    """Seconds from either file-name scheme: front_medium_<sec>-<ns>, or front_medium_<ns>_lat.._lon.."""
    parts = stem.split("_")[2]
    if "-" in parts:
        a, b = parts.split("-")
        return int(a) + int(b) / 1e9
    return int(parts) / 1e9


def atlas_train_frames():
    """ATLAS train, one dict per image: image, size, session (camera + 10-minute window), lamps (box, state, pictogram).
    Boxes are clipped to the image; each image's size is read from the file (some frames are cropped)."""
    base = Path(DATASETS) / "ATLAS_full" / "ATLAS"
    # BaseLoader keeps every value a string: safe_load would read the class name "off" as False
    classes = {int(k): v for k, v in yaml.load((base / "ATLAS_classes.yaml").read_text(), Loader=yaml.BaseLoader)["names"].items()}
    out = []
    for cam in ATLAS_CAMERAS:
        img_dir = base / "train" / cam / "images"
        for img in sorted(img_dir.iterdir()):
            with Image.open(img) as im:
                w, h = im.size
            lamps = []
            lab = base / "train" / cam / "labels" / f"{img.stem}.txt"
            for line in lab.read_text().split("\n") if lab.exists() else []:
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x1, y1, x2, y2 = max(cx - bw / 2, 0), max(cy - bh / 2, 0), min(cx + bw / 2, w), min(cy + bh / 2, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                state, picto = _atlas_class(classes[int(c)])
                lamps.append({"box": [float(x1), float(y1), float(x2), float(y2)], "state": state, "pictogram": picto})
            out.append({"image": str(img), "size": (w, h), "session": f"{cam}_{int(_atlas_stamp(img.stem) // 600)}", "lamps": lamps})
    return out


def atlas_heldout_sessions():
    return set(json.load(open(os.path.join(SPLITS, "atlas_heldout_sessions.json")))["held_out"])


def atlas_test():
    """ATLAS test images. Boxes use each camera's first image size and are not clipped (as in every reported result)."""
    root = os.path.join(DATASETS, "ATLAS_full", "ATLAS")
    names = yaml.safe_load(open(os.path.join(root, "ATLAS_classes.yaml")))["names"]

    def parse(name):
        if name in ("off", False):  # safe_load reads a bare off as False
            return None, None
        color = "red" if name.endswith("red_yellow") else name.rsplit("_", 1)[1]
        shape = name[: -len("_red_yellow")] if name.endswith("red_yellow") else name.rsplit("_", 1)[0]
        return color, (None if shape == "circle" else set(shape.split("_")[1:]))

    out = []
    for cam in ("front_wide", "front_medium", "front_tele"):
        paths = sorted(glob.glob(os.path.join(root, "test", cam, "images", "*")))
        with Image.open(paths[0]) as im:
            w, h = im.size
        for p in paths:
            lab = os.path.join(root, "test", cam, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
            rows = [l.split() for l in open(lab)] if os.path.exists(lab) else []
            parsed = [parse(names[int(r[0])]) for r in rows]
            cxcywh = torch.tensor([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4) * torch.tensor([w, h, w, h])
            out.append({"path": p, "group": cam, "boxes": box_convert(cxcywh, "cxcywh", "xyxy"),
                        "color": [c for c, _ in parsed], "arrow": [a for _, a in parsed]})
    return out


# ------------------------------------------------------------------ BDD100K
BDD_SIZE = (1280, 720)


def _bdd_json(split):
    return os.path.join(DATASETS, "BDD100K", "bdd100k_labels_release", "bdd100k", "labels", f"bdd100k_labels_images_{split}.json")


def _bdd_image(split, name):
    return os.path.join(DATASETS, "BDD100K", "bdd100k", "bdd100k", "images", "100k", split, name)


def bdd_train_frames():
    """BDD100K train images with their traffic lights (box clipped to the image, color or "unknown")."""
    w, h = BDD_SIZE
    state = {"green": "green", "red": "red", "yellow": "yellow", "none": "unknown"}
    out = []
    for im in json.load(open(_bdd_json("train"), encoding="utf-8")):
        lamps = []
        for l in im.get("labels") or []:
            if l["category"] != "traffic light" or "box2d" not in l:
                continue
            b = l["box2d"]
            x1, y1, x2, y2 = max(b["x1"], 0.0), max(b["y1"], 0.0), min(b["x2"], w - 1.0), min(b["y2"], h - 1.0)
            if x2 <= x1 or y2 <= y1:
                continue
            lamps.append({"box": [float(x1), float(y1), float(x2), float(y2)],
                          "state": state[l.get("attributes", {}).get("trafficLightColor", "none")]})
        out.append({"image": _bdd_image("train", im["name"]), "size": BDD_SIZE, "lamps": lamps})
    return out


def bdd_val():
    colors = {"red": "red", "yellow": "yellow", "green": "green"}  # "none" -> ignore
    out = []
    for im in json.load(open(_bdd_json("val"))):
        tl = [l for l in im["labels"] or [] if l["category"] == "traffic light"]
        out.append({"path": _bdd_image("val", im["name"]), "group": im["attributes"]["timeofday"],
                    "boxes": torch.tensor([[l["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for l in tl], dtype=torch.float32).reshape(-1, 4),
                    "color": [colors.get(l["attributes"]["trafficLightColor"]) for l in tl], "arrow": [None] * len(tl)})
    return out


# ------------------------------------------------------------------ LISA, S2TLD, VZC-TLD (evaluation only)
LISA_TAGS = {"go": ("green", None), "stop": ("red", None), "warning": ("yellow", None),
             "goLeft": ("green", "left"), "stopLeft": ("red", "left"), "warningLeft": ("yellow", "left"),
             "goForward": ("green", "straight")}
LISA_TEST = ("daySequence1", "daySequence2", "nightSequence1", "nightSequence2")


def lisa_test():
    root = os.path.join(DATASETS, "LISA")
    ann_root = os.path.join(root, "Annotations", "Annotations")
    out = []
    for seq in LISA_TEST:
        frames = {}
        with open(os.path.join(ann_root, seq, "frameAnnotationsBOX.csv")) as f:
            for row in csv.DictReader(f, delimiter=";"):
                box = [float(row[k]) for k in ("Upper left corner X", "Upper left corner Y", "Lower right corner X", "Lower right corner Y")]
                light = (row["Annotation tag"], box)
                name = os.path.basename(row["Filename"])
                if light not in frames.setdefault(name, []):  # a few rows are duplicated
                    frames[name].append(light)
        for name, lights in sorted(frames.items()):
            out.append({"path": os.path.join(root, f"{seq}/{seq}/frames/{name}"), "group": "night" if "night" in seq else "day",
                        "boxes": torch.tensor([b for _, b in lights], dtype=torch.float32).reshape(-1, 4),
                        "color": [LISA_TAGS[t][0] for t, _ in lights],
                        "arrow": [None if LISA_TAGS[t][1] is None else {LISA_TAGS[t][1]} for t, _ in lights]})
    return out


def s2tld():
    root = os.path.join(DATASETS, "S2TLD", "ds")
    colors = {"red": "red", "yellow": "yellow", "green": "green"}  # off / wait on -> ignore
    out = []
    for ann in sorted(glob.glob(os.path.join(root, "ann", "*.json"))):
        j = json.load(open(ann))
        objs = [o for o in j["objects"] if o["geometryType"] == "rectangle"]
        boxes = []
        for o in objs:
            (x1, y1), (x2, y2) = o["points"]["exterior"]
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        out.append({"path": os.path.join(root, "img", os.path.basename(ann)[: -len(".json")]),
                    "group": f"{j['size']['width']}x{j['size']['height']}",
                    "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    "color": [colors.get(o["classTitle"]) for o in objs], "arrow": [None] * len(objs)})
    return out


def vzc():
    """VZC-TLD test then train (no model trains on it): per light a box, a color ("unknown" -> ignore box) and a
    relevance flag; per image a session and the label RR / RG / NoR."""
    root = os.path.join(DATASETS, "VZC_TLD")
    out = []
    for split in ("test", "train"):
        coco = json.load(open(os.path.join(root, "labels", f"{split}_v1.json")))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for im in sorted(coco["images"], key=lambda i: i["file_name"]):
            w, h = int(im["width"]), int(im["height"])
            boxes, colors, relevant = [], [], []
            for a in sorted(by_image.get(im["id"], []), key=lambda a: a["id"]):
                rel, state = cats[a["category_id"]].rsplit("_", 1)
                x, y, bw, bh = map(float, a["bbox"])
                x1, y1, x2, y2 = max(x, 0.0), max(y, 0.0), min(x + bw, w - 1.0), min(y + bh, h - 1.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                colors.append(state if state in ("red", "yellow", "green") else None)
                relevant.append(rel == "relevant")
            rel_states = {c for c, r in zip(colors, relevant) if r}
            label = "RR" if rel_states & {"red", "yellow"} else "RG" if "green" in rel_states else "NoR"
            name = im["file_name"].split("/")[-1].rsplit(".", 1)[0]
            out.append({"path": os.path.join(root, im["file_name"]), "group": split, "session": name.rsplit("_", 1)[0],
                        "label": label, "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "color": colors,
                        "relevant": relevant, "arrow": [None] * len(boxes)})
    return out


PERCEPTION = {"atlas": atlas_test, "lisa": lisa_test, "s2tld": s2tld, "bdd100k": bdd_val, "vzc": vzc}
