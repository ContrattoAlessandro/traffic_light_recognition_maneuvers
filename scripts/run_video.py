"""Run a trained model on a video and write a copy with its predictions drawn (qualitative: videos have no labels).

Per frame, as in the evaluation: squeeze to the model's square input, score threshold, class-agnostic NMS 0.5, then
a traffic rule on the lights the model assigns to the junction (``--rule junction``, German convention, or ``us``).
Nothing is smoothed over time. Junction lights get a box in their color, labelled "pictogram | color score";
other lights (another junction, pedestrian / bicycle / tram signals) a thin grey box. A side panel shows the
per-maneuver colors and zooms on up to 6 junction lights. Also writes <out>.jsonl with every frame's detections.

Example: python scripts/run_video.py clip.mp4 --run runs/R --epoch 2 --threshold 0.35 --out runs/R/clip.mp4
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
from dtsu import DIRECTIONS, factorized  # noqa: E402
from dtsu.model import MEAN, NMS_IOU, STD, load_model, run_args  # noqa: E402
from dtsu.rules import LAMP_COLORS, PICTOGRAMS, junction_rule, us_rule  # noqa: E402

BGR = {"red": (60, 60, 235), "yellow": (0, 200, 245), "green": (80, 200, 40), "none": (110, 110, 110)}
SHORT = {"circle": "circle", "arrow_left": "left arrow", "arrow_right": "right arrow", "arrow_straight": "straight arrow",
         "arrow_straight_left": "straight+left arrow", "arrow_straight_right": "straight+right arrow", "unknown": "unknown",
         "other": "ped/bike/tram"}
PANEL, ZOOM = 300, 132


def preprocess(frames, res):
    batch = []
    for f in frames:
        img = TF.resize(TF.to_image(np.ascontiguousarray(f[:, :, ::-1])), [res, res], antialias=True)
        batch.append((TF.to_dtype(img, torch.float32, scale=True) - MEAN) / STD)
    return torch.stack(batch)


def text(img, s, org, color, scale=0.5, thick=1):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, dets, dirs, t, caption):
    """dets: (pictogram, junction, color, score, xyxy box)."""
    h, w = frame.shape[:2]
    img = frame.copy()
    for picto, junction, color, score, (x1, y1, x2, y2) in sorted(dets, key=lambda d: d[1]):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if not junction:
            cv2.rectangle(img, p1, p2, (170, 170, 170), 1)
            continue
        cv2.rectangle(img, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), BGR[color], 2)
        text(img, f"{SHORT[picto]} | {color} {score:.2f}", (p1[0], max(p1[1] - 6, 14)), BGR[color], 0.42)
    panel = np.full((h, PANEL, 3), 24, np.uint8)
    cv2.putText(panel, f"t = {t:6.2f} s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, caption, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    for n, d in enumerate(DIRECTIONS):
        y = 66 + n * 38
        cv2.rectangle(panel, (12, y), (PANEL - 12, y + 30), BGR[dirs[d]], -1)
        cv2.putText(panel, f"{d}: {dirs[d]}", (22, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0) if dirs[d] in ("yellow", "green") else (255, 255, 255), 1, cv2.LINE_AA)
    junction = sorted([d for d in dets if d[1]], key=lambda d: -d[3])[:6]
    cv2.putText(panel, f"junction {sum(d[1] for d in dets)}, other {sum(not d[1] for d in dets)}", (12, 236),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    for n, (picto, _, color, score, (x1, y1, x2, y2)) in enumerate(junction):
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
        cv2.putText(panel, SHORT[picto], (px, py + ZOOM + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return np.hstack([img, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--rule", choices=("junction", "us"), default="junction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    res = run_args(args.run)["resolution"]
    net = load_model(args.run, args.epoch)
    cap = cv2.VideoCapture(args.video)
    fps, w, h, total = cap.get(5), int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec), fps, (w + PANEL, h))
        if writer.isOpened():
            break
    caption = f"{os.path.basename(os.path.normpath(args.run))}, {args.rule} rule, thr {args.threshold}"
    print(f"{w}x{h} @ {fps} fps, {total} frames, {caption}, codec {codec}", flush=True)
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
        score, probs, boxes = factorized.detect(net, preprocess(frames, res).cuda())
        p, c, j = probs["picto"].argmax(-1), probs["color"].argmax(-1), probs["junction"] > 0.5
        for i, f in enumerate(frames):
            k = score[i] > args.threshold
            s = score[i][k]
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            keep = nms(b, s, NMS_IOU) if len(s) else torch.zeros(0, dtype=torch.long)
            dets = [(PICTOGRAMS[int(a)], bool(jj), LAMP_COLORS[int(cc)], float(ss), [round(v, 1) for v in bb])
                    for a, cc, jj, ss, bb in zip(p[i][k][keep], c[i][k][keep], j[i][k][keep], s[keep], b[keep].tolist())]
            lights = [(pi, ci, si) for pi, ji, ci, si, _ in dets if ji]
            dirs = junction_rule(lights) if args.rule == "junction" else us_rule(lights)
            writer.write(draw(f, dets, dirs, n / fps, caption))
            log.write(json.dumps({"frame": n, "t": round(n / fps, 3), "directions": dirs,
                                  "detections": [{"picto": pi, "junction": ji, "color": ci, "score": round(si, 3), "box": bb}
                                                 for pi, ji, ci, si, bb in dets]}) + "\n")
            n += 1
        if n % 600 < args.batch:
            print(f"{n}/{total}", flush=True)
    writer.release()
    log.close()
    print("wrote", args.out, n, "frames")


if __name__ == "__main__":
    main()
