"""Run an Exp 6 model (J or F), or the Exp 7 model (--run runs/exp7_multi_F), on a video and write a copy with its predictions drawn (qualitative, no labels).

Same drawing as tools/run_video.py. Per frame: squeeze to the model's input, score threshold (the arm's
DTLD-tune threshold unless given), class-agnostic NMS 0.5, then the junction rule (dtsu.picto_rules.rule)
on the lights the model assigns to the junction. Boxes of junction lights are labelled "pictogram | color
score"; lights of no junction (other junction, pedestrian / bike / tram) are thin grey boxes. No smoothing.
Also writes <out>.jsonl with the detections and per-maneuver colors of every frame.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import cv2
import torch
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import factorized  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS, rule  # noqa: E402
from dtsu.rfdetr_baseline import detect  # noqa: E402
from tools.eval_picto import load_model  # noqa: E402
from tools.make_picto_gallery import SHORT  # noqa: E402
from tools.run_video import PANEL, draw, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("J", "F"), default="J")
    ap.add_argument("--run", help="run folder (default runs/exp6_picto_<arm>); the Exp 7 and Exp 8 runs read their selection from exp7_eval/ and exp8_eval/")
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    run = args.run or f"runs/exp6_picto_{args.arm}"
    if os.path.exists(os.path.join(run, "exp8_eval", "result.json")):
        sel = json.load(open(os.path.join(run, "exp8_eval", "result.json")))["selection"]["F8"]
    elif os.path.exists(os.path.join(run, "exp7_eval", "result.json")):
        sel = json.load(open(os.path.join(run, "exp7_eval", "result.json")))["F7"]
    else:
        sel = json.load(open(os.path.join(run, "picto_eval", "result.json")))
    thr = args.threshold if args.threshold is not None else sel["threshold"]
    class_names = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    table = factorized.attribute_table(class_names)
    net, res = load_model(run, sel["epoch"], class_names, args.arm)
    cap = cv2.VideoCapture(args.video)
    fps, w, h, total = cap.get(5), int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    roi = (0, 0, w, h)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec), fps, (w + PANEL, h))
        if writer.isOpened():
            break
    print(f"{w}x{h} @ {fps} fps, {total} frames, {args.arm} epoch {sel['epoch']} threshold {thr}, codec {codec}", flush=True)
    panel_args = SimpleNamespace(side="junction rule", near_turn=f"{os.path.basename(run)}, thr {thr}")
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
        x = preprocess(frames, roi, res).cuda()
        if args.arm == "F":
            score, probs, boxes = factorized.detect(net, x)
            p, c, j = probs["picto"].argmax(-1), probs["color"].argmax(-1), probs["junction"] > 0.5
        else:
            cls, score, boxes = detect(net, x, class_names)
            p, c, j = table[cls][..., 0], table[cls][..., 1], table[cls][..., 2] > 0
        for i, f in enumerate(frames):
            k = score[i] > thr
            s = score[i][k]
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            keep = nms(b, s, 0.5) if len(s) else torch.zeros(0, dtype=torch.long)
            pi, ci, ji, s, b = p[i][k][keep], c[i][k][keep], j[i][k][keep], s[keep], b[keep]
            dets = [({SHORT[PICTOS[int(a)]]} if jj else None, COLORS[int(cc)], float(ss), [round(v, 1) for v in bb])
                    for a, cc, jj, ss, bb in zip(pi, ci, ji, s, b.tolist())]
            dirs = rule([(PICTOS[int(a)], COLORS[int(cc)], float(ss)) for a, cc, jj, ss in zip(pi, ci, ji, s) if jj])
            writer.write(draw(f, dets, dirs, roi, n / fps, panel_args))
            log.write(json.dumps({"frame": n, "t": round(n / fps, 3), "directions": dirs,
                                  "detections": [{"picto": PICTOS[int(a)], "junction": bool(jj), "color": COLORS[int(cc)],
                                                  "score": round(float(ss), 3), "box": bb}
                                                 for a, cc, jj, ss, bb in zip(pi, ci, ji, s, b.tolist())]}) + "\n")
            n += 1
        if n % 600 < args.batch:
            print(f"{n}/{total}", flush=True)
    writer.release()
    log.close()
    print("wrote", args.out, n, "frames")


if __name__ == "__main__":
    main()
