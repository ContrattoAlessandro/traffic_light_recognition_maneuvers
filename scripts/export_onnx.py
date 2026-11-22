"""Export R to ONNX and build a self-contained hand-off package for Autoware / AWSIM.

    python scripts/export_onnx.py --out runs/onnx_R            # export, golden frames, parity, checksums
    python scripts/export_onnx.py --out runs/onnx_R --no-parity

The package (``--out``) holds:
- ``r_1280_opset17.onnx``: R (``runs/exp10_R``, the epoch of ``results/R.json``) in float32, static input
  1x3x1280x1280, opset 17 (TensorRT 8.6 and 10.x). Outputs per query, no threshold and no NMS:
  boxes [1,300,4] cxcywh normalized, score [1,300], picto [1,300,8], color [1,300,3], junction [1,300];
- ``model_config.json``: the I/O contract and every constant of the pre- and post-processing;
- ``reference_infer.py``, ``README.md``, ``AGENT_README.md`` (from ``deploy/``): the torch-free reference pipeline,
  the hand-off notes and how the model works;
- ``golden/``: frames with their exact input tensor, raw ONNX outputs and expected maneuver colors;
- ``parity.json``: ONNX + reference pipeline against the evaluated PyTorch model (bf16, the repository decoder) on
  AWSIM and DTLD test frames, and float32 raw-output differences on the golden frames;
- ``SHA256SUMS.txt``.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deploy"))
import reference_infer as ref  # noqa: E402
from dtsu import evaluation, factorized, rules  # noqa: E402
from dtsu.model import MEAN, NMS_IOU, STD, load_model  # noqa: E402

RUN = os.path.join(ROOT, "runs", "exp10_R")
RESULT = os.path.join(ROOT, "results", "R.json")
AWSIM = os.path.join(ROOT, "data", "awsim", "AWSIM_cam.mp4")
AWSIM_LOG = os.path.join(ROOT, "runs", "awsim", "AWSIM_cam_picto_R.jsonl")
JUNCTION_THRESHOLD = 0.5


class Export(nn.Module):
    """RF-DETR's export graph plus the factorized heads of the last decoder layer, as probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        boxes, logits = self.net.forward_export(x)[:2]
        attrs = self.net.class_embed.last
        self.net.class_embed.last = None
        last = lambda t: t[-1] if t.dim() == 4 else t  # noqa: E731  (all decoder layers or only the last one)
        boxes, logits = last(boxes), last(logits)
        return (boxes, logits[..., 0].sigmoid(), last(attrs["picto"]).softmax(-1), last(attrs["color"]).softmax(-1),
                last(attrs["junction"])[..., 0].sigmoid())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def torch_input(rgb, res):
    """The evaluation's preprocessing (dtsu.model.load_image) on an RGB array."""
    import torchvision.transforms.v2.functional as TF
    from PIL import Image

    img = TF.resize(TF.to_image(Image.fromarray(rgb)), [res, res], antialias=True)
    return ((TF.to_dtype(img, torch.float32, scale=True) - MEAN) / STD)[None]


@torch.no_grad()
def torch_maneuvers(net, x, thr):
    """The evaluated pipeline: bf16 PyTorch model, then the repository decoder (threshold, NMS, junction rule)."""
    score, probs, boxes = factorized.detect(net, x.cuda())
    d = {"score": score[0], "picto": probs["picto"][0].argmax(-1), "color": probs["color"][0].argmax(-1),
         "junction": probs["junction"][0], "box": boxes[0]}
    lights = evaluation.junction_lights(evaluation.kept(d, thr))
    return rules.junction_rule(lights), len(lights)


def export(net, res, path, opset):
    net = net.float().eval()
    net.export()
    model = Export(net).eval()
    x = torch.zeros(1, 3, res, res, device="cuda")
    with torch.no_grad():
        shapes = [tuple(t.shape) for t in model(x)]
    print("output shapes", shapes, flush=True)
    torch.onnx.export(model, (x,), path, input_names=["image"], output_names=list(ref.OUTPUTS), export_params=True,
                      do_constant_folding=True, opset_version=opset, dynamo=False)
    import onnx

    onnx.checker.check_model(path)
    return model, shapes


def read_video(path, frames):
    import cv2

    cap, out = cv2.VideoCapture(path), {}
    for i in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if ok:
            out[i] = np.ascontiguousarray(bgr[:, :, ::-1])
    return out


def golden_frames():
    """A few AWSIM frames (R's earlier video log picks red / yellow / green / arrow / nothing) plus DTLD and
    OpenLane-V2 frames: name -> RGB array."""
    log = [json.loads(line) for line in open(AWSIM_LOG)]
    want = {}
    for color in ("red", "yellow", "green"):
        want[f"awsim_straight_{color}"] = next(e["frame"] for e in log[200:] if e["directions"]["straight"] == color)
    want["awsim_arrow"] = next(e["frame"] for e in log if any(d["picto"].startswith("arrow") and d["junction"] for d in e["detections"]))
    want["awsim_none"] = next(e["frame"] for e in log[100:] if e["directions"]["straight"] == "none")
    vid = read_video(AWSIM, want.values())
    frames = {k: vid[i] for k, i in want.items()}
    dtld = [r for r in evaluation.sources.dtld_records("test") if r["color"]]
    frames["dtld_test"] = ref.read_rgb(dtld[len(dtld) // 2]["image"])
    lab, root = evaluation.sources.openlane_sets()["A_val"]
    rec = evaluation.sources.load_records(lab, root)[100]
    frames["openlane_A_val"] = ref.read_rgb(os.path.join(root, rec["image"]))
    return frames, want


def write_golden(folder, frames, model_t, sess_model, cfg):
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    raw = {}
    for name, rgb in frames.items():
        Image.fromarray(rgb).save(os.path.join(folder, name + ".png"))
        x = ref.preprocess(rgb, cfg, "pil")
        res, out = sess_model(rgb)
        np.savez_compressed(os.path.join(folder, name + ".input.npz"), image=x)
        np.savez_compressed(os.path.join(folder, name + ".outputs.npz"), **out)
        with torch.no_grad():
            t = [o.numpy() for o in model_t(torch.from_numpy(x))]
        raw[name] = ref.compare_raw(out, dict(zip(ref.OUTPUTS, t)), cfg)
        json.dump({"image": name + ".png", "input": name + ".input.npz", "outputs": name + ".outputs.npz",
                   "size_wh": [rgb.shape[1], rgb.shape[0]], **res}, open(os.path.join(folder, name + ".json"), "w"), indent=1)
        print(name, res["maneuvers"], "| float32 ONNX vs PyTorch:", raw[name], flush=True)
    return raw


def parity(net_bf16, sess_model, cfg, frames_by_set, thr):
    out = {}
    for set_name, frames in frames_by_set.items():
        agree = {m: {d: 0 for d in rules.DIRECTIONS} for m in ("pil", "cv2-area", "cv2-linear")}
        differ = {m: [] for m in agree}
        count_diff = {m: 0 for m in agree}
        for i, rgb in frames:
            want, n_t = torch_maneuvers(net_bf16, torch_input(rgb, cfg["resolution"]), thr)
            for m in agree:
                res, _ = sess_model(rgb, m)
                got = res["maneuvers"]
                n_o = sum(lt["junction"] > JUNCTION_THRESHOLD for lt in res["lights"])
                for d in rules.DIRECTIONS:
                    agree[m][d] += got[d] == want[d]
                if got != want:
                    differ[m].append(i)
                count_diff[m] += n_o != n_t
        n = len(frames)
        out[set_name] = {"frames": n, **{f"resize_{m}": {
            "agreement_per_maneuver": {d: agree[m][d] / n for d in rules.DIRECTIONS},
            "frames_with_any_maneuver_difference": len(differ[m]),
            "frames_with_different_junction_light_count": count_diff[m],
            "differing_frames_first_20": differ[m][:20]} for m in agree}}
        print(set_name, json.dumps(out[set_name]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "onnx_R"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--awsim-stride", type=int, default=16)
    ap.add_argument("--dtld-stride", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sel = json.load(open(RESULT))["selection"]
    res = ref_res = json.load(open(os.path.join(RUN, "args.json")))["resolution"]
    onnx_name = f"r_{res}_opset{args.opset}.onnx"
    onnx_path = os.path.join(args.out, onnx_name)
    model_t, shapes = export(load_model(RUN, sel["epoch"]), res, onnx_path, args.opset)
    ckpt = os.path.join(RUN, f"epoch_{sel['epoch']}_ema.pth")
    cfg = {"model": "R: factorized RF-DETR Medium, input 1280, trained on DTLD + ATLAS + OpenLane-V2 A/B + BDD100K",
           "onnx": onnx_name, "opset": args.opset, "precision": "float32",
           "source_run": "runs/exp10_R", "epoch": sel["epoch"], "checkpoint_sha256": sha256(ckpt),
           "resolution": ref_res, "input": {"name": "image", "shape": [1, 3, res, res], "dtype": "float32",
                                             "layout": "NCHW, RGB",
                                             "preprocessing": "squeeze the whole image to resolution x resolution (no crop, no "
                                                              "padding; bilinear with antialias), scale to 0-1, subtract mean, divide by std"},
           "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist(),
           "outputs": {"boxes": {"shape": list(shapes[0]), "meaning": "cx, cy, w, h normalized to the input image (0-1)"},
                       "score": {"shape": list(shapes[1]), "meaning": "light probability (sigmoid)"},
                       "picto": {"shape": list(shapes[2]), "meaning": "pictogram probabilities (softmax), order of 'pictograms'"},
                       "color": {"shape": list(shapes[3]), "meaning": "color probabilities (softmax), order of 'colors'"},
                       "junction": {"shape": list(shapes[4]), "meaning": "probability that the light belongs to the junction ahead (sigmoid)"}},
           "pictograms": rules.PICTOGRAMS, "colors": rules.LAMP_COLORS,
           "threshold": sel["threshold"], "nms_iou": NMS_IOU, "junction_threshold": JUNCTION_THRESHOLD,
           "maneuvers": list(rules.DIRECTIONS),
           "rule": "junction rule (German convention): per maneuver the highest-scoring junction arrow pointing that way, else "
                   "the highest-scoring junction round light (circle or unknown), else none; 'other' never sets a maneuver",
           "selection": "epoch and threshold chosen on the tune splits (results/R.json), never on a test set"}
    json.dump(cfg, open(os.path.join(args.out, "model_config.json"), "w"), indent=1)
    for f in ("reference_infer.py", "README.md", "AGENT_README.md"):
        shutil.copy(os.path.join(ROOT, "deploy", f), os.path.join(args.out, f))
    print("wrote", onnx_path, round(os.path.getsize(onnx_path) / 1e6, 1), "MB", flush=True)

    sess_model = ref.Model(onnx_path, cfg, ["CPUExecutionProvider"])
    # float32 PyTorch on the CPU: on the GPU, TF32 convolutions alone change scores by up to ~0.08 on these frames
    cpu = load_model(RUN, sel["epoch"], device="cpu").float().eval()
    cpu.export()
    frames, picked = golden_frames()
    raw = write_golden(os.path.join(args.out, "golden"), frames, Export(cpu).eval(), sess_model, cfg)
    report = {"golden_float32_onnx_vs_pytorch_cpu_same_input": raw, "golden_awsim_frames": picked}
    if not args.no_parity:
        net_bf16 = load_model(RUN, sel["epoch"])
        import cv2

        n = int(cv2.VideoCapture(AWSIM).get(cv2.CAP_PROP_FRAME_COUNT))
        awsim = read_video(AWSIM, range(0, n, args.awsim_stride))
        dtld = evaluation.sources.dtld_records("test")[:: args.dtld_stride]
        report["reference"] = "PyTorch R in bf16 (as evaluated), torchvision preprocessing, repository decoder"
        report["candidate"] = "ONNX float32 on onnxruntime CPU, reference_infer.py preprocessing and decoder"
        report["sets"] = parity(net_bf16, sess_model, cfg, {
            f"AWSIM_cam.mp4 (920x900), every {args.awsim_stride}th frame": sorted(awsim.items()),
            f"DTLD test (2048x1024), every {args.dtld_stride}th image": [(i, ref.read_rgb(r["image"])) for i, r in enumerate(dtld)]},
            sel["threshold"])
    json.dump(report, open(os.path.join(args.out, "parity.json"), "w"), indent=1)
    with open(os.path.join(args.out, "SHA256SUMS.txt"), "w", newline="\n") as f:  # LF, for sha256sum -c on Linux
        for d, _, files in sorted(os.walk(args.out)):
            for name in sorted(files):
                if name != "SHA256SUMS.txt":
                    p = os.path.join(d, name)
                    f.write(f"{sha256(p)}  {os.path.relpath(p, args.out).replace(os.sep, '/')}\n")
    print("done", args.out, flush=True)


if __name__ == "__main__":
    main()
