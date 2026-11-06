"""Per-light OOD perception test of an RF-DETR run: detection, color and arrow direction.

The target datasets have no relevance or maneuver labels, so no TSS: only what each one labels natively.

| dataset | split | groups | color | arrow pictogram |
|---|---|---|---|---|
| ATLAS (Karlsruhe) | test | camera (wide / medium / tele) | yes (red_yellow = red) | left / right / straight / straight+left / straight+right |
| S2TLD (Shanghai) | all 5,786 images (none used in training) | image size | yes | no |
| BDD100K (US) | val, 10,000 images | time of day | yes | no |
| LISA (San Diego) | test sequences | day / night | yes | left (goLeft...) / straight (goForward) |

Lights without a usable color (ATLAS ``off``, S2TLD ``off`` / ``wait on``, BDD color ``none``) are
"ignore" boxes: not counted as misses, and detections on them are not false positives.

Box conventions differ: LISA and BDD100K boxes are about twice as wide as the lamp boxes of DTLD /
ATLAS / OpenLane-V2 (same height and center), which caps IoU near 0.5 for a perfect lamp detection. So
every metric is computed under two matching criteria (class-agnostic, greedy by score):
- ``iou0.5``: IoU >= 0.5 (strict, convention-sensitive);
- ``center``: each box's center lies inside the other box (convention-robust; ties broken by IoU).
Metrics per group and overall:
- AP: threshold-free average precision over all detections scoring > 0.05 (all-point interpolation);
- recall / precision at the run's own score threshold (picked on its source tune split), recall by box
  height in source pixels;
- color accuracy (and confusion) of matched lights at that threshold;
- arrows: for annotated arrows matched to a detection the run calls relevant, whether its predicted moves
  (minus "uturn" / "generic", which no pictogram shows) equal the arrow's directions (moves_acc), or
  include them (moves_contain_acc); gt_to_pred_moves counts each pair. Circles are skipped: which maneuvers a relevant ball controls depends on the junction, not on the pictogram.
Precision is a lower bound where a dataset leaves lights unannotated (LISA, some BDD).

Images are squeezed to the run's square input, as in training. Writes
``<run>/ood_perception/epoch<N>_<dataset>.json``.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, box_iou, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.rfdetr_baseline import detect, load_image, parse_class  # noqa: E402
from tools.eval_rfdetr import load_model  # noqa: E402

DATASETS = os.environ.get("DATASETS", "../traffic-light-detection/datasets")
COLORS = ("red", "yellow", "green")
HEIGHT_BINS = (0, 10, 20, 40, 1e9)
CRITERIA = ("iou0.5", "center")
MIN_SCORE = 0.05
NMS_IOU = 0.5  # class-agnostic, as in dtsu.rfdetr_baseline.frame_directions


# ---------------------------------------------------------------- dataset adapters
# Each returns a list of images: {"path", "group", "boxes" [n,4] xyxy pixels, "color" (None = ignore),
# "arrow" (set of directions, or None for circles / unknown shape)}.

def load_atlas():
    import yaml

    root = os.path.join(DATASETS, "ATLAS_full", "ATLAS")
    names = yaml.safe_load(open(os.path.join(root, "ATLAS_classes.yaml")))["names"]

    def parse(name):
        if name in ("off", False):  # yaml reads a bare off as False
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


def load_s2tld():
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


def load_bdd():
    root = os.path.join(DATASETS, "BDD100K")
    colors = {"red": "red", "yellow": "yellow", "green": "green"}  # none -> ignore
    out = []
    for im in json.load(open(os.path.join(root, "bdd100k_labels_release", "bdd100k", "labels",
                                          "bdd100k_labels_images_val.json"))):
        tl = [l for l in im["labels"] or [] if l["category"] == "traffic light"]
        out.append({"path": os.path.join(root, "bdd100k", "bdd100k", "images", "100k", "val", im["name"]),
                    "group": im["attributes"]["timeofday"],
                    "boxes": torch.tensor([[l["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for l in tl],
                                          dtype=torch.float32).reshape(-1, 4),
                    "color": [colors.get(l["attributes"]["trafficLightColor"]) for l in tl], "arrow": [None] * len(tl)})
    return out


def load_lisa():
    from dtsu.labels_lisa import TAGS, iter_frames

    root = os.path.join(DATASETS, "LISA")
    out = []
    for segment, _, image, lights in iter_frames(root, "test"):
        out.append({"path": os.path.join(root, image), "group": "night" if "night" in segment else "day",
                    "boxes": torch.tensor([b for _, b in lights], dtype=torch.float32).reshape(-1, 4),
                    "color": [TAGS[t][0] for t, _ in lights],
                    "arrow": [None if TAGS[t][1] == "generic" else {TAGS[t][1]} for t, _ in lights]})
    return out


LOADERS = {"atlas": load_atlas, "s2tld": load_s2tld, "bdd100k": load_bdd, "lisa": load_lisa}


# ---------------------------------------------------------------- inference and matching

class Images(Dataset):
    def __init__(self, items, resolution):
        self.items, self.resolution = items, resolution

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]["path"]
        with Image.open(p) as im:
            size = torch.tensor(im.size, dtype=torch.float32)
        return load_image(p, self.resolution), size


def predictions(model, items, class_names, resolution, workers, batch_size):
    """Per image: (class idx, score, xyxy pixel boxes) of detections > MIN_SCORE after class-agnostic NMS."""
    out = []
    for x, sizes in DataLoader(Images(items, resolution), batch_size=batch_size, num_workers=workers):
        cls, score, boxes = detect(model, x.cuda(non_blocking=True), class_names)
        for c, s, b, (w, h) in zip(cls, score, boxes, sizes):
            k = s > MIN_SCORE
            c, s, b = c[k], s[k], box_convert(b[k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            keep = nms(b, s, NMS_IOU) if len(s) else torch.zeros(0, dtype=torch.long)
            out.append((c[keep], s[keep], b[keep]))
    return out


def pair_ok(boxes, gt_boxes, criterion):
    """[n_pred, n_gt] IoU and whether each pair may match under ``criterion``."""
    iou = box_iou(boxes, gt_boxes)
    if criterion == "center":
        def inside(a, b):  # center of each a inside each b -> [len(a), len(b)]
            cx, cy = ((a[:, 0] + a[:, 2]) / 2)[:, None], ((a[:, 1] + a[:, 3]) / 2)[:, None]
            return (cx >= b[None, :, 0]) & (cx <= b[None, :, 2]) & (cy >= b[None, :, 1]) & (cy <= b[None, :, 3])
        return iou, inside(boxes, gt_boxes) & inside(gt_boxes, boxes).T
    return iou, iou >= float(criterion[len("iou"):])


def match(boxes, scores, gt_boxes, ignore, criterion):
    """Greedy by score. Returns per prediction: matched gt index, -1 = false positive, -2 = on an ignore box."""
    m = torch.full((len(scores),), -1, dtype=torch.long)
    if not len(scores) or not len(gt_boxes):
        return m
    iou, ok = pair_ok(boxes, gt_boxes, criterion)
    taken = set()
    for p in torch.argsort(scores, descending=True).tolist():
        cand = [(iou[p, g].item(), g) for g in range(len(gt_boxes)) if g not in taken and not ignore[g] and ok[p, g]]
        if cand:
            g = max(cand)[1]
            m[p] = g
            taken.add(g)
        elif ignore.any() and ok[p, ignore].any():
            m[p] = -2
    return m


def average_precision(scores, tp, n_gt):
    if not n_gt:
        return None
    order = np.argsort(-scores)
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def norm_moves(moves):
    return None if moves is None else set(moves) - {"uturn", "generic"}


def summarize(items, preds, class_names, threshold, criterion):
    all_scores, all_tp = [], []
    n_gt = n_det = n_fp = 0
    hit_h, gt_h = np.zeros(len(HEIGHT_BINS) - 1, int), np.zeros(len(HEIGHT_BINS) - 1, int)
    conf = np.zeros((3, 3), int)
    arrows = {"matched": 0, "called_relevant": 0, "moves_correct": 0, "moves_contain": 0}
    moves_conf = collections.Counter()  # "gt arrow -> predicted moves" for arrows called relevant
    for it, (c, s, b) in zip(items, preds):
        ignore = torch.tensor([col is None for col in it["color"]], dtype=torch.bool)
        m = match(b, s, it["boxes"], ignore, criterion)
        valid = m != -2
        all_scores.append(s[valid].numpy())
        all_tp.append((m[valid] >= 0).numpy().astype(float))
        n_gt += int((~ignore).sum())
        above = s > threshold
        n_det += int((above & valid).sum())
        n_fp += int((above & (m == -1)).sum())
        gt_to_pred = {int(g): p for p, g in enumerate(m.tolist()) if g >= 0 and above[p]}
        for g in range(len(it["color"])):
            if ignore[g]:
                continue
            hb = int(np.searchsorted(HEIGHT_BINS, float(it["boxes"][g, 3] - it["boxes"][g, 1]), side="right") - 1)
            gt_h[hb] += 1
            if g not in gt_to_pred:
                continue
            hit_h[hb] += 1
            moves, color = parse_class(class_names[int(c[gt_to_pred[g]])])
            conf[COLORS.index(it["color"][g]), COLORS.index(color)] += 1
            if it["arrow"][g] is not None:
                arrows["matched"] += 1
                if moves is not None:
                    arrows["called_relevant"] += 1
                    arrows["moves_correct"] += norm_moves(moves) == it["arrow"][g]
                    arrows["moves_contain"] += it["arrow"][g] <= norm_moves(moves)
                    moves_conf["+".join(sorted(it["arrow"][g])) + " -> " + "+".join(sorted(moves))] += 1
    scores, tp = np.concatenate(all_scores) if all_scores else np.zeros(0), np.concatenate(all_tp) if all_tp else np.zeros(0)
    hits = int(hit_h.sum())
    per_color_recall = {g: float(conf[a, a] / conf[a].sum()) if conf[a].sum() else None for a, g in enumerate(COLORS)}
    return {
        "images": len(items), "lights": n_gt, "AP": average_precision(scores, tp, n_gt),
        "detections": n_det, "recall": hits / max(n_gt, 1), "precision": (n_det - n_fp) / max(n_det, 1),
        "recall_by_height": {f"{HEIGHT_BINS[j]:g}-{HEIGHT_BINS[j + 1]:g}px": [int(hit_h[j]), int(gt_h[j])]
                             for j in range(len(gt_h))},
        "color_acc": float(np.trace(conf) / max(conf.sum(), 1)),
        "color_recall": per_color_recall,
        "color_confusion_gt_x_pred": {g: dict(zip(COLORS, map(int, conf[a]))) for a, g in enumerate(COLORS)},
        "arrows": {**arrows, "moves_acc": arrows["moves_correct"] / arrows["called_relevant"]
                   if arrows["called_relevant"] else None,
                   "moves_contain_acc": arrows["moves_contain"] / arrows["called_relevant"] if arrows["called_relevant"] else None,
                   "gt_to_pred_moves": dict(moves_conf.most_common())} if any(a is not None for it in items for a in it["arrow"]) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--class-names", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True, help="threshold picked on the run's own tune split")
    ap.add_argument("--datasets", nargs="+", choices=sorted(LOADERS), default=sorted(LOADERS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N images per dataset (smoke tests)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    class_names = json.load(open(args.class_names))
    model, res, _ = load_model(args.run, os.path.join(args.run, f"epoch_{args.epoch}_ema.pth"), len(class_names))
    out_dir = os.path.join(args.run, "ood_perception")
    os.makedirs(out_dir, exist_ok=True)
    for name in args.datasets:
        items = LOADERS[name]()[: args.limit]
        preds = predictions(model, items, class_names, res, args.workers, args.batch_size)
        out = {"run": args.run, "epoch": args.epoch, "threshold": args.threshold, "dataset": name}
        groups = sorted({it["group"] for it in items})
        for crit in CRITERIA:
            r = out[crit] = {"all": summarize(items, preds, class_names, args.threshold, crit)}
            for g in groups if len(groups) > 1 else ():
                sel = [i for i, it in enumerate(items) if it["group"] == g]
                r[g] = summarize([items[i] for i in sel], [preds[i] for i in sel], class_names, args.threshold, crit)
            for g, m in r.items():
                a = m["arrows"]
                print(f"{name} {crit} {g}: {m['images']} img {m['lights']} lights | AP {m['AP'] or 0:.3f} "
                      f"R {m['recall']:.3f} P {m['precision']:.3f} | color {m['color_acc']:.3f} "
                      f"{ {k: round(v, 3) if v is not None else None for k, v in m['color_recall'].items()} }"
                      + (f" | arrows {a['matched']} matched, {a['called_relevant']} relevant, moves acc "
                         f"{a['moves_acc'] if a['moves_acc'] is None else round(a['moves_acc'], 3)} {a['gt_to_pred_moves']}" if a else ""), flush=True)
        if args.limit is None:
            with open(os.path.join(out_dir, f"epoch{args.epoch}_{name}.json"), "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
