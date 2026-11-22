"""The baseline M: RF-DETR with joint "<moves>|<color>" light classes, decoded by the benchmark's own direction rule.

Each annotated light is one detection class: the set of ego maneuvers it controls according to the label builder
(``dtsu.labels`` on OpenLane-V2, ``dtsu.labels_dtld`` on DTLD), e.g. "left+straight" or "generic", or "irrelevant"
for lights of other lanes and junctions, joined with its color. M thus predicts the maneuvers a light governs
directly, where R reads the pictogram and the junction and lets a fixed rule assign the maneuvers.

``assign_scored`` mirrors ``dtsu.labels.assign_directions`` on scored detections: among the most specific candidates
the highest-scoring light wins. Ground-truth boxes through it reproduce every label of the benchmark, so all of M's
errors are detection, class and relevance errors.
"""
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

from . import DIRECTIONS
from .model import MIN_SCORE, NMS_IOU, _Frames, _Sized, run_args
from .rules import LAMP_COLORS, PICTOGRAMS

IRRELEVANT = "irrelevant"


def box_class(b):
    """A labelled light -> its class name."""
    return f"{'+'.join(b['moves']) if b['relevant'] else IRRELEVANT}|{b['color']}"


def parse_class(name):
    """Class name -> (set of moves, or None for an irrelevant light; color)."""
    moves, color = name.split("|")
    return (None if moves == IRRELEVANT else set(moves.split("+"))), color


def assign_scored(dets, side="right", near_turn="dedicated"):
    """Per-maneuver colors from detected relevant lights [(moves, color, score)].

    near_turn "dedicated" (OpenLane-V2, US rules): only a light controlling nothing but the near turn sets it;
    "signal" (DTLD, German rules): any light controlling the near turn does, the most specific first.
    """
    far, near = ("left", "right") if side == "right" else ("right", "left")

    def pick(pred):
        cands = [(len(m), -s, c) for m, c, s in dets if pred(m)]
        return min(cands)[2] if cands else None

    out = {"straight": pick(lambda m: "straight" in m or "generic" in m)}
    out[far] = pick(lambda m: far in m) or out["straight"]
    out[near] = pick((lambda m: m <= {near}) if near_turn == "dedicated" else (lambda m: near in m))
    return {d: out[d] or "none" for d in DIRECTIONS}


def class_names(run):
    """The class list of a run's training export (``scripts/export_moves_set.py``)."""
    return json.load(open(os.path.join(run_args(run)["dataset_dir"], "class_names.json")))


def is_moves_run(run):
    return not run_args(run).get("factorized", False)


def load_model(run, epoch, device="cuda"):
    """The EMA weights of one epoch of an M run, in eval mode."""
    import rfdetr

    args = run_args(run)
    n = len(class_names(run))
    net = getattr(rfdetr, args["variant"])(resolution=args["resolution"], num_classes=n, pretrain_weights=None).model.model
    net.load_state_dict(torch.load(os.path.join(run, f"epoch_{epoch}_ema.pth"), map_location="cpu", weights_only=False)["model"])
    return net.to(device).eval()


@torch.no_grad()
def detect(net, images, n_classes):
    """Per query: class index, score (top class probability) and normalized cxcywh box."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
        out = net(images)
    if isinstance(out, tuple):
        out = {"pred_boxes": out[0], "pred_logits": out[1]}
    score, cls = out["pred_logits"].float()[..., :n_classes].sigmoid().max(-1)
    return cls.cpu(), score.cpu(), out["pred_boxes"].float().cpu()


@torch.no_grad()
def frame_detections(net, paths, res, n_classes, workers=6, batch_size=8, device="cuda"):
    """Per image every query scoring above MIN_SCORE: (class, score, cxcywh box); thresholding and NMS at decoding."""
    out = []
    for x in DataLoader(_Frames(paths, res), batch_size=batch_size, num_workers=workers):
        cls, score, boxes = detect(net, x.to(device), n_classes)
        for c, s, b in zip(cls, score, boxes):
            k = s > MIN_SCORE
            out.append((c[k], s[k], b[k]))
    return out


def frame_colors(det, names, threshold, side="right", near_turn="dedicated"):
    """One frame's cached detections -> per-maneuver colors."""
    cls, score, boxes = det
    keep = score > threshold
    cls, score, boxes = cls[keep], score[keep], boxes[keep]
    if len(score):
        k = nms(box_convert(boxes, "cxcywh", "xyxy"), score, NMS_IOU)
        cls, score = cls[k], score[k]
    dets = []
    for c, s in zip(cls.tolist(), score.tolist()):
        moves, color = parse_class(names[c])
        if moves is not None:
            dets.append((moves, color, s))
    return assign_scored(dets, side, near_turn)


def attribute_codes(names):
    """Class index -> the attribute code of ``dtsu.model.box_detections`` (pictogram * 6 + junction * 3 + color), so
    M's detections go through the same perception metrics as R's. M has no pictogram; it is coded as a circle, and
    "junction" is M's relevance."""
    circle = PICTOGRAMS.index("circle")
    out = []
    for n in names:
        moves, color = parse_class(n)
        out.append(circle * 6 + (moves is not None) * 3 + LAMP_COLORS.index(color))
    return torch.tensor(out)


@torch.no_grad()
def box_detections(net, paths, res, names, workers=6, batch_size=8, device="cuda"):
    """Per image, after class-agnostic NMS: (attribute code, score, xyxy pixel box), as ``dtsu.model.box_detections``."""
    codes, out = attribute_codes(names), []
    for x, sizes in DataLoader(_Sized(paths, res), batch_size=batch_size, num_workers=workers):
        cls, score, boxes = detect(net, x.to(device), len(names))
        for i, (w, h) in enumerate(sizes):
            k = score[i] > MIN_SCORE
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            s = score[i][k]
            keep = nms(b, s, NMS_IOU) if len(s) else torch.zeros(0, dtype=torch.long)
            out.append((codes[cls[i][k][keep]], s[keep], b[keep]))
    return out
