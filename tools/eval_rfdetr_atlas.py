"""Per-light test of an RF-DETR run on ATLAS (FZI, Karlsruhe): detection, color and arrow direction.

ATLAS has no relevance or maneuver labels, so there is no TSS; instead every annotated light is
matched to the run's detections (class-agnostic, IoU >= ``--iou``, greedy by score) per camera:
- recall / precision of light detection, recall by box height in pixels;
- color accuracy on matched lights (red_yellow counts as red, as in the DTLD labels);
- arrow direction: for ATLAS arrows matched to a detection the run calls relevant, whether its
  moves equal the arrow's (arrow_straight_left -> left+straight). Circles are skipped, since a
  relevant circle's moves depend on the junction, not on the pictogram.
"off" lights are not in the run's classes (DTLD labels drop them): they are not counted as misses,
and detections on them are not false positives.

Images are squeezed to the square as in training (2:1 for DTLD; ATLAS wide 2592x2048, medium and
tele 1920x1200). Writes ``<run>/ood_atlas/epoch<N>_iou<iou>.json``.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, box_iou, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.rfdetr_baseline import detect, load_image, parse_class  # noqa: E402
from tools.eval_rfdetr import load_model  # noqa: E402

ATLAS = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "ATLAS_full", "ATLAS")
CAMERAS = ("front_wide", "front_medium", "front_tele")
COLORS = ("red", "yellow", "green")
HEIGHT_BINS = (0, 10, 20, 40, 1e9)  # box height in source pixels


def parse_atlas(name):
    """'arrow_straight_left_red_yellow' -> ({'straight', 'left'}, 'red'); circles -> (None, color); off -> (None, None)."""
    if name in ("off", False):  # yaml reads a bare off as False
        return None, None
    color = "red" if name.endswith("red_yellow") else name.rsplit("_", 1)[1]
    shape = name[: -len("_red_yellow")] if name.endswith("red_yellow") else name.rsplit("_", 1)[0]
    return (None if shape == "circle" else set(shape.split("_")[1:])), color


class Images(Dataset):
    def __init__(self, paths, resolution):
        self.paths, self.resolution = paths, resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], self.resolution)


def load_camera(cam, names):
    """Image paths and per image: boxes (normalized cxcywh), arrow moves, colors (None = off)."""
    paths = sorted(glob.glob(os.path.join(ATLAS, "test", cam, "images", "*")))
    gts = []
    for p in paths:
        lab = os.path.join(ATLAS, "test", cam, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        rows = [l.split() for l in open(lab)] if os.path.exists(lab) else []
        parsed = [parse_atlas(names[int(r[0])]) for r in rows]
        gts.append({"boxes": torch.tensor([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4),
                    "moves": [m for m, _ in parsed], "color": [c for _, c in parsed]})
    return paths, gts


def match(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy class-agnostic matching by score; returns gt index -> pred index and IoU of preds vs gts."""
    if not len(pred_boxes) or not len(gt_boxes):
        return {}, torch.zeros(len(pred_boxes), len(gt_boxes))
    iou = box_iou(box_convert(pred_boxes, "cxcywh", "xyxy"), box_convert(gt_boxes, "cxcywh", "xyxy"))
    pairs = {}
    for p in torch.argsort(pred_scores, descending=True).tolist():
        cand = [(iou[p, g].item(), g) for g in range(len(gt_boxes)) if g not in pairs and iou[p, g] >= iou_thr]
        if cand:
            pairs[max(cand)[1]] = p
    return pairs, iou


def score_camera(model, paths, gts, class_names, resolution, threshold, iou_thr, workers, image_h):
    n_gt = n_pred = n_fp = 0
    hit_by_h = np.zeros(len(HEIGHT_BINS) - 1, int)
    gt_by_h = np.zeros(len(HEIGHT_BINS) - 1, int)
    confusion = np.zeros((len(COLORS), len(COLORS)), int)  # gt x pred
    arrows = {"matched": 0, "relevant": 0, "moves_correct": 0}
    i = 0
    for x in DataLoader(Images(paths, resolution), batch_size=8, num_workers=workers):
        cls, sc, bx = detect(model, x.cuda(non_blocking=True), class_names)
        for c, s, b in zip(cls, sc, bx):
            gt = gts[i]
            i += 1
            k = s > threshold
            c, s, b = c[k], s[k], b[k]
            if len(s):
                keep = nms(box_convert(b, "cxcywh", "xyxy"), s, 0.5)
                c, s, b = c[keep], s[keep], b[keep]
            pairs, iou = match(b, s, gt["boxes"], iou_thr)
            on = [j for j, col in enumerate(gt["color"]) if col is not None]
            off = [j for j, col in enumerate(gt["color"]) if col is None]
            matched_preds = {p for g, p in pairs.items() if g in on}
            for p in range(len(s)):
                if p in matched_preds:
                    continue
                if off and iou[p, off].max() >= iou_thr:  # on an off light: ignored
                    continue
                n_fp += 1
            n_pred += len(s) - sum(1 for g, p in pairs.items() if g in off)
            for j in on:
                n_gt += 1
                hb = np.searchsorted(HEIGHT_BINS, gt["boxes"][j, 3].item() * image_h, side="right") - 1
                gt_by_h[hb] += 1
                if j not in pairs:
                    continue
                hit_by_h[hb] += 1
                moves, color = parse_class(class_names[c[pairs[j]]])
                confusion[COLORS.index(gt["color"][j]), COLORS.index(color)] += 1
                if gt["moves"][j] is not None:
                    arrows["matched"] += 1
                    if moves is not None:
                        arrows["relevant"] += 1
                        arrows["moves_correct"] += moves == gt["moves"][j]
    hits = int(hit_by_h.sum())
    return {"images": len(paths), "lights": n_gt, "detections": n_pred,
            "recall": hits / max(n_gt, 1), "precision": (n_pred - n_fp) / max(n_pred, 1),
            "recall_by_height": {f"{HEIGHT_BINS[j]:g}-{HEIGHT_BINS[j + 1]:g}px": [int(hit_by_h[j]), int(gt_by_h[j])]
                                 for j in range(len(gt_by_h))},
            "color_acc": float(np.trace(confusion) / max(confusion.sum(), 1)),
            "color_confusion_gt_x_pred": {g: dict(zip(COLORS, map(int, confusion[a]))) for a, g in enumerate(COLORS)},
            "arrows": {**arrows, "moves_acc": arrows["moves_correct"] / max(arrows["relevant"], 1)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/rfdetr_m_dtld")
    ap.add_argument("--class-names", default="data/rfdetr_coco_dtld/class_names.json")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True, help="threshold picked on the run's own tune split")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N images per camera (smoke tests)")
    args = ap.parse_args()

    class_names = json.load(open(args.class_names))
    names = yaml.safe_load(open(os.path.join(ATLAS, "ATLAS_classes.yaml")))["names"]
    model, res, _ = load_model(args.run, os.path.join(args.run, f"epoch_{args.epoch}_ema.pth"), len(class_names))
    out = {"run": args.run, "epoch": args.epoch, "threshold": args.threshold, "iou": args.iou}
    for cam in CAMERAS:
        paths, gts = load_camera(cam, names)
        paths, gts = paths[: args.limit], gts[: args.limit]
        image_h = load_image.__globals__["Image"].open(paths[0]).size[1]
        out[cam] = m = score_camera(model, paths, gts, class_names, res, args.threshold, args.iou, args.workers, image_h)
        a = m["arrows"]
        print(f"{cam}: {m['images']} images, {m['lights']} lights | recall {m['recall']:.3f} precision {m['precision']:.3f} "
              f"| color acc {m['color_acc']:.3f} | arrows matched {a['matched']}, called relevant {a['relevant']}, "
              f"moves acc {a['moves_acc']:.3f} | recall by height {m['recall_by_height']}", flush=True)
    os.makedirs(os.path.join(args.run, "ood_atlas"), exist_ok=True)
    with open(os.path.join(args.run, "ood_atlas", f"epoch{args.epoch}_iou{args.iou:g}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
