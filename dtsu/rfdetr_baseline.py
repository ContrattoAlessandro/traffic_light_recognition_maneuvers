"""Structured baseline: RF-DETR light detector + the label builder's direction rules.

Each annotated light becomes one detection class "<moves>|<color>", where <moves> is the set of
maneuvers the light controls for the ego (e.g. "left+straight", "generic") or "irrelevant" for lights
of other lanes / junctions. Feeding ground-truth boxes through ``assign_scored`` reproduces every
label exactly (``python tools/make_rfdetr_coco.py`` checks this), so the baseline's errors are purely
detection and relevance errors.
"""
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance

from . import COLORS, DIRECTIONS, PASS, color_to_pass

IRRELEVANT = "irrelevant"


def box_class(b):
    return f"{'+'.join(b['moves']) if b['relevant'] else IRRELEVANT}|{b['color']}"


def parse_class(name):
    moves, color = name.split("|")
    return (None if moves == IRRELEVANT else set(moves.split("+"))), color


def assign_scored(dets, side="right"):
    """Per-direction colors from detected lights, mirroring ``labels.assign_directions``.

    dets: list of (moves set, color, score) for relevant lights. Among the most specific candidates
    the highest-scoring light wins; ``assign_directions`` instead marks color disagreement as a
    conflict (those frames are not in the benchmark, so ground truth never needs the tie-break).
    """
    far, near = ("left", "right") if side == "right" else ("right", "left")

    def pick(pred):
        cands = [(len(m), -s, c) for m, c, s in dets if pred(m)]
        return min(cands)[2] if cands else None

    out = {"straight": pick(lambda m: "straight" in m or "generic" in m)}
    out[far] = pick(lambda m: far in m) or out["straight"]
    out["uturn"] = pick(lambda m: "uturn" in m) or out[far]
    out[near] = pick(lambda m: m <= {near})
    return {d: out[d] or "none" for d in DIRECTIONS}


class Photometric:
    """Brightness/contrast jitter on the resized PIL image, as in ``dtsu.data`` (no hue, no flips)."""

    def __init__(self, low=0.75, high=1.25):
        self.low, self.high = low, high

    def __call__(self, img, target):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(self.low, self.high))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(self.low, self.high))
        return img, target


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path, resolution):
    """Square resize as RF-DETR's val transform does (torchvision bilinear + antialias), normalized."""
    import torchvision.transforms.v2.functional as TF

    img = TF.to_image(Image.open(path).convert("RGB"))
    img = TF.resize(img, [resolution, resolution], antialias=True)
    return (TF.to_dtype(img, torch.float32, scale=True) - MEAN) / STD


@torch.no_grad()
def detect(model, images, class_names, amp=True):
    """Per image: one (class index, score, box cxcywh normalized) per query, after per-query argmax."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(images)
    if isinstance(out, tuple):
        out = {"pred_boxes": out[0], "pred_logits": out[1]}
    probs = out["pred_logits"].float()[..., : len(class_names)].sigmoid()
    score, cls = probs.max(-1)
    return cls.cpu(), score.cpu(), out["pred_boxes"].float().cpu()


def frame_directions(cls, score, boxes, class_names, threshold, side="right", nms_iou=0.5):
    """Thresholded, class-agnostic-NMS'd detections of one image -> per-direction colors."""
    from torchvision.ops import box_convert, nms

    keep = score > threshold
    cls, score, boxes = cls[keep], score[keep], boxes[keep]
    if len(score):
        k = nms(box_convert(boxes, "cxcywh", "xyxy"), score, nms_iou)
        cls, score = cls[k], score[k]
    dets = []
    for c, s in zip(cls.tolist(), score.tolist()):
        moves, color = parse_class(class_names[c])
        if moves is not None:
            dets.append((moves, color, s))
    return assign_scored(dets, side)


def to_arrays(colors):
    """List of per-direction color dicts -> ([N,4] color idx, [N,4] pass idx)."""
    c = np.array([[COLORS.index(f[d]) for d in DIRECTIONS] for f in colors])
    p = np.array([[PASS.index(color_to_pass(f[d])) for d in DIRECTIONS] for f in colors])
    return c, p
