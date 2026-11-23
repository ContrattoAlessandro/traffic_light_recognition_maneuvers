"""Reference inference for the ONNX export of R: image -> per-light detections -> per-maneuver colors.

Needs only numpy, Pillow and onnxruntime (OpenCV optional, for --resize cv2-area). It does not import torch or the
training repository, so it can be copied to the Autoware machine as is. It is the specification an integration
(C++ / TensorRT / ROS 2) should reproduce, and ``--check`` verifies it against the golden outputs of the package.

    python reference_infer.py image.png [more images]          # print detections and maneuver colors as JSON
    python reference_infer.py --check golden                    # compare with the expected outputs
    python reference_infer.py --check-outputs mine.npz golden/awsim_0.npz   # compare raw outputs of another runtime

Pipeline (model_config.json holds every constant):
1. Preprocess: RGB, squeeze the whole image to a square (no crop, no padding) of side ``resolution``, scale to
   0-1, normalize with ImageNet mean and std, layout 1x3xHxW float32.
2. Network (ONNX): per query q of 300, ``score`` (light probability), ``boxes`` (cx, cy, w, h normalized to the
   input image), ``picto`` (8 softmax probabilities), ``color`` (3), ``junction`` (probability that the light
   belongs to the junction the ego vehicle approaches).
3. Post-process: keep score > threshold (0.35), class-agnostic NMS at IoU 0.5 in descending score order, then keep
   the lights with junction > 0.5. Pictogram and color are the argmax of their probabilities.
4. Junction rule (German convention, the one R is decoded with): for each maneuver (straight, left, right) the
   highest-scoring arrow pointing that way sets the color; otherwise the highest-scoring round light ("circle" or
   "unknown"); otherwise "none". "other" (pedestrian, bicycle, tram) never sets a maneuver.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DIRECTIONS = ("straight", "left", "right")
ARROW_MOVES = {"arrow_left": ("left",), "arrow_right": ("right",), "arrow_straight": ("straight",),
               "arrow_straight_left": ("straight", "left"), "arrow_straight_right": ("straight", "right")}
ROUND = ("circle", "unknown")
OUTPUTS = ("boxes", "score", "picto", "color", "junction")


def load_config(path=None):
    return json.load(open(path or os.path.join(HERE, "model_config.json")))


def preprocess(rgb, cfg, resize="pil"):
    """rgb: HxWx3 uint8 array -> 1x3xRxR float32. resize: "pil" (bilinear with antialias, the training transform's
    equivalent), "cv2-area" (cv2.INTER_AREA) or "cv2-linear" (cv2.INTER_LINEAR, no antialias); parity.json measures each."""
    r = cfg["resolution"]
    if resize == "pil":
        x = np.asarray(Image.fromarray(rgb).resize((r, r), Image.BILINEAR))
    elif resize in ("cv2-area", "cv2-linear"):
        import cv2
        x = cv2.resize(rgb, (r, r), interpolation=cv2.INTER_AREA if resize == "cv2-area" else cv2.INTER_LINEAR)
    else:
        raise ValueError(resize)
    x = (x.astype(np.float32) / 255.0 - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _iou(b, others):
    """b: [4] xyxy, others: [N, 4] -> [N]"""
    x1 = np.maximum(b[0], others[:, 0])
    y1 = np.maximum(b[1], others[:, 1])
    x2 = np.minimum(b[2], others[:, 2])
    y2 = np.minimum(b[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda t: (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])  # noqa: E731
    return inter / (area(b) + area(others) - inter + 1e-12)


def nms(xyxy, scores, iou):
    order, keep = np.argsort(-scores, kind="stable"), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][_iou(xyxy[i], xyxy[order[1:]]) <= iou]
    return np.array(keep, dtype=np.int64)


def postprocess(out, cfg, width, height):
    """Raw outputs of one image (dict of arrays with a leading batch dimension of 1) -> list of lights, all
    detections above the threshold after NMS, each with its pixel box in the original image."""
    s = out["score"][0]
    k = np.nonzero(s > cfg["threshold"])[0]
    cx, cy, w, h = out["boxes"][0][k].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * np.array([width, height, width, height], np.float32)
    keep = k[nms(xyxy, s[k], cfg["nms_iou"])] if len(k) else k
    lights = []
    for q in keep:
        b = xyxy[np.where(k == q)[0][0]]
        lights.append({"query": int(q), "score": float(s[q]), "box_xyxy": [round(float(v), 1) for v in b],
                       "pictogram": cfg["pictograms"][int(out["picto"][0][q].argmax())],
                       "color": cfg["colors"][int(out["color"][0][q].argmax())],
                       "junction": float(out["junction"][0][q])})
    return lights


def junction_rule(lights, junction_threshold=0.5):
    """lights -> {maneuver: "red" | "yellow" | "green" | "none"}."""
    jl = [lt for lt in lights if lt["junction"] > junction_threshold]
    arrows = {}
    for lt in jl:
        for m in ARROW_MOVES.get(lt["pictogram"], ()):
            if m not in arrows or lt["score"] > arrows[m]["score"]:
                arrows[m] = lt
    rounds = [lt for lt in jl if lt["pictogram"] in ROUND]
    circle = max(rounds, key=lambda lt: lt["score"]) if rounds else None
    return {d: (arrows.get(d) or circle or {"color": "none"})["color"] for d in DIRECTIONS}


class Model:
    def __init__(self, onnx_path=None, cfg=None, providers=None):
        import onnxruntime as ort

        self.cfg = cfg or load_config()
        path = onnx_path or os.path.join(HERE, self.cfg["onnx"])
        self.sess = ort.InferenceSession(path, providers=providers or ort.get_available_providers())

    def raw(self, x):
        return dict(zip(OUTPUTS, self.sess.run(list(OUTPUTS), {"image": x})))

    def __call__(self, rgb, resize="pil"):
        h, w = rgb.shape[:2]
        out = self.raw(preprocess(rgb, self.cfg, resize))
        lights = postprocess(out, self.cfg, w, h)
        return {"lights": lights, "maneuvers": junction_rule(lights, self.cfg["junction_threshold"])}, out


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def compare_raw(mine, ref, cfg):
    """Raw-output agreement between two runtimes on the same input tensor."""
    s1, s2 = mine["score"][0], ref["score"][0]
    keep = (s1 > 0.05) | (s2 > 0.05)
    res = {"max_abs_score_diff": float(np.abs(s1 - s2).max()),
           "max_abs_box_diff_on_scored_queries": float(np.abs(mine["boxes"][0][keep] - ref["boxes"][0][keep]).max()) if keep.any() else 0.0,
           "queries_above_threshold": [int((s1 > cfg["threshold"]).sum()), int((s2 > cfg["threshold"]).sum())]}
    for k in ("picto", "color"):
        res[f"{k}_argmax_agreement_on_scored_queries"] = float((mine[k][0][keep].argmax(-1) == ref[k][0][keep].argmax(-1)).mean()) if keep.any() else 1.0
    res["max_abs_junction_diff_on_scored_queries"] = float(np.abs(mine["junction"][0][keep] - ref["junction"][0][keep]).max()) if keep.any() else 0.0
    return res


def check_golden(folder, model):
    """Runs the reference pipeline on every golden frame and compares with the stored expectations."""
    ok = True
    for meta in sorted(glob.glob(os.path.join(folder, "*.json"))):
        exp = json.load(open(meta))
        rgb = read_rgb(os.path.join(folder, exp["image"]))
        x = preprocess(rgb, model.cfg, "pil")
        x_ref = np.load(os.path.join(folder, exp["input"]))["image"]
        pre_diff = float(np.abs(x - x_ref).max())
        res, out = model(rgb)
        raw = compare_raw(out, dict(np.load(os.path.join(folder, exp["outputs"]))), model.cfg)
        same = res["maneuvers"] == exp["maneuvers"]
        ok &= same and pre_diff < 1e-5 and raw["max_abs_score_diff"] < 1e-3
        print(f"{exp['image']}: maneuvers {res['maneuvers']} expected {exp['maneuvers']} -> {'OK' if same else 'DIFF'}; "
              f"preprocess max diff {pre_diff:.2e}; score max diff {raw['max_abs_score_diff']:.2e}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--onnx")
    ap.add_argument("--config")
    ap.add_argument("--resize", default="pil", choices=("pil", "cv2-area", "cv2-linear"))
    ap.add_argument("--check", metavar="GOLDEN_DIR", help="verify the pipeline against the golden outputs")
    ap.add_argument("--check-outputs", nargs=2, metavar=("MINE_NPZ", "GOLDEN_NPZ"),
                    help="compare raw outputs of another runtime (keys boxes, score, picto, color, junction)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.check_outputs:
        print(json.dumps(compare_raw(dict(np.load(args.check_outputs[0])), dict(np.load(args.check_outputs[1])), cfg), indent=1))
        return
    model = Model(args.onnx, cfg)
    if args.check:
        sys.exit(0 if check_golden(args.check, model) else 1)
    for p in args.images:
        res, _ = model(read_rgb(p), args.resize)
        print(json.dumps({"image": p, **res}, indent=1))


if __name__ == "__main__":
    main()
