"""Run an RF-DETR run on a video and write a copy with its predictions drawn (qualitative, no labels).

Every frame goes through the same path as the benchmark: the region of interest is squeezed to the
model's square input, per-query argmax, score threshold, class-agnostic NMS 0.5, then ``assign_scored``
with the chosen traffic side and near-turn rule. Nothing is smoothed over time.

Drawn on each frame:
- relevant lights: a box in the light's color, labelled "moves | color score";
- lights the model calls not relevant: a thin grey box;
- the region of interest given to the model (dashed white) when it is not the whole frame.
A side panel shows the per-direction colors and zooms on up to 6 relevant lights (highest score first).
Also writes <out>.jsonl: one line per frame with the detections and the per-direction colors.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.rfdetr_baseline import MEAN, STD, assign_scored, detect, parse_class  # noqa: E402
from tools.eval_rfdetr import load_model  # noqa: E402

BGR = {"red": (60, 60, 235), "yellow": (0, 200, 245), "green": (80, 200, 40), "none": (110, 110, 110)}
PANEL, ZOOM = 300, 132


def preprocess(frames, roi, res):
    x1, y1, x2, y2 = roi
    batch = []
    for f in frames:
        img = TF.to_image(np.ascontiguousarray(f[y1:y2, x1:x2, ::-1]))
        img = TF.resize(img, [res, res], antialias=True)
        batch.append((TF.to_dtype(img, torch.float32, scale=True) - MEAN) / STD)
    return torch.stack(batch)


def decode(c, s, b, class_names, roi, threshold, side, near_turn):
    x1, y1, x2, y2 = roi
    k = s > threshold
    c, s, b = c[k], s[k], b[k]
    b = box_convert(b, "cxcywh", "xyxy") * torch.tensor([x2 - x1, y2 - y1, x2 - x1, y2 - y1]) + torch.tensor([x1, y1, x1, y1])
    if len(s):
        keep = nms(b, s, 0.5)
        c, s, b = c[keep], s[keep], b[keep]
    dets = [(*parse_class(class_names[int(ci)]), float(si), [round(v, 1) for v in bi]) for ci, si, bi in zip(c, s, b.tolist())]
    dirs = assign_scored([(m, col, sc) for m, col, sc, _ in dets if m is not None], side, near_turn)
    return dets, dirs


def text(img, s, org, color, scale=0.5, thick=1):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, dets, dirs, roi, t, args):
    h, w = frame.shape[:2]
    img = frame.copy()
    if roi != (0, 0, w, h):
        x1, y1, x2, y2 = roi
        for x in range(x1, x2, 16):
            cv2.line(img, (x, y1), (min(x + 8, x2), y1), (230, 230, 230), 1)
            cv2.line(img, (x, y2 - 1), (min(x + 8, x2), y2 - 1), (230, 230, 230), 1)
    for moves, color, score, (x1, y1, x2, y2) in sorted(dets, key=lambda d: d[0] is not None):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if moves is None:
            cv2.rectangle(img, p1, p2, (170, 170, 170), 1)
            continue
        cv2.rectangle(img, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), BGR[color], 2)
        text(img, f"{'+'.join(sorted(moves))} | {color} {score:.2f}", (p1[0], max(p1[1] - 6, 14)), BGR[color], 0.42)
    panel = np.full((h, PANEL, 3), 24, np.uint8)
    cv2.putText(panel, f"t = {t:6.2f} s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{args.side}-hand traffic, near turn: {args.near_turn}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (160, 160, 160), 1, cv2.LINE_AA)
    for n, d in enumerate(DIRECTIONS):
        y = 66 + n * 38
        cv2.rectangle(panel, (12, y), (PANEL - 12, y + 30), BGR[dirs[d]], -1)
        cv2.putText(panel, f"{d}: {dirs[d]}", (22, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0) if dirs[d] in ("yellow", "green") else (255, 255, 255), 1, cv2.LINE_AA)
    rel = sorted([d for d in dets if d[0] is not None], key=lambda d: -d[2])[:6]
    n_rel, n_irr = sum(d[0] is not None for d in dets), sum(d[0] is None for d in dets)
    cv2.putText(panel, f"relevant {n_rel}, not relevant {n_irr}", (12, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                cv2.LINE_AA)
    for n, (moves, color, score, (x1, y1, x2, y2)) in enumerate(rel):
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 6) * 2.4
        a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        crop = frame[b:int(min(cy + side / 2, h)), a:int(min(cx + side / 2, w))]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (ZOOM, ZOOM), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (0, 0), (ZOOM - 1, ZOOM - 1), BGR[color], 3)
        px, py = 12 + (n % 2) * (ZOOM + 12), 250 + (n // 2) * (ZOOM + 24)
        if py + ZOOM + 20 > h:
            break
        panel[py:py + ZOOM, px:px + ZOOM] = crop
        cv2.putText(panel, "+".join(sorted(moves)), (px, py + ZOOM + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)
    return np.hstack([img, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--run", default="runs/rfdetr_m_mix")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--class-names", default="data/rfdetr_coco_fit_mix/class_names.json")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--side", choices=["left", "right"], default="right", help="traffic side")
    ap.add_argument("--near-turn", choices=["dedicated", "signal"], default="dedicated")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=[0, 0, 1, 1],
                    help="region given to the model, as fractions of the frame")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    class_names = json.load(open(args.class_names))
    model, res, _ = load_model(args.run, os.path.join(args.run, f"epoch_{args.epoch}_ema.pth"), len(class_names))
    cap = cv2.VideoCapture(args.video)
    fps, w, h, total = cap.get(5), int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    roi = (round(args.roi[0] * w), round(args.roi[1] * h), round(args.roi[2] * w), round(args.roi[3] * h))
    writer = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec), fps, (w + PANEL, h))
        if writer.isOpened():
            break
    print(f"{w}x{h} @ {fps} fps, {total} frames, roi {roi}, codec {codec}", flush=True)
    log = open(os.path.splitext(args.out)[0] + ".jsonl", "w")
    n = 0
    while True:
        frames = []
        while len(frames) < args.batch:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        if not frames:
            break
        cls, score, boxes = detect(model, preprocess(frames, roi, res).cuda(), class_names)
        for f, c, s, b in zip(frames, cls, score, boxes):
            dets, dirs = decode(c, s, b, class_names, roi, args.threshold, args.side, args.near_turn)
            writer.write(draw(f, dets, dirs, roi, n / fps, args))
            log.write(json.dumps({"frame": n, "t": round(n / fps, 3), "directions": dirs,
                                  "detections": [{"moves": sorted(m) if m is not None else None, "color": col, "score": round(sc, 3),
                                                  "box": bx} for m, col, sc, bx in dets]}) + "\n")
            n += 1
        if n % 600 < args.batch:
            print(f"{n}/{total}", flush=True)
    writer.release()
    log.close()
    print("wrote", args.out, n, "frames")


if __name__ == "__main__":
    main()
