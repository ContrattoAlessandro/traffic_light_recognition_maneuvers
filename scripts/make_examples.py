"""Qualitative examples: R and the baseline M side by side on the same frames, in and out of distribution.

Frames are drawn at random (seed 0) from each set, without looking at either model's output:
- in distribution: OpenLane-V2 A val and B val (US rule), DTLD test (route labels, German rule);
- out of distribution: LISA test (day, night), S2TLD 1920x1080, VZC-TLD (US rule; no per-maneuver labels, VZC has
  an image label: relevant red/yellow = stop, relevant green = go).
Each row shows the same crop around the labelled lights twice: R (pictogram, junction flag and color per light, then
the traffic rule) and M (the maneuvers it assigns to each light, then its decoder). Thick boxes are lights the model
uses for the decision (R: junction lights; M: relevant lights), thin ones are lights it ignores; thin white boxes are
the labelled lights. Under each panel: the per-maneuver colors of the model and of the ground truth.

Runs on the CPU by default, so it can run while the GPU trains.
  python scripts/make_examples.py --out docs/examples
"""
import argparse
import os
import random
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS, factorized, labels_dtld, moves, sources  # noqa: E402
from dtsu.model import NMS_IOU, load_image, load_model, run_args  # noqa: E402
from dtsu.rules import LAMP_COLORS, PICTOGRAMS, junction_rule, us_rule  # noqa: E402

RGB = {"red": (230, 40, 40), "yellow": (250, 200, 0), "green": (40, 200, 70), "none": (120, 120, 120)}
PICTO_SHORT = {"circle": "O", "arrow_left": "←", "arrow_right": "→", "arrow_straight": "↑",
               "arrow_straight_left": "↑←", "arrow_straight_right": "↑→", "unknown": "?", "other": "ped"}
MOVE_SHORT = {"straight": "S", "left": "L", "right": "R", "generic": "G", "uturn": "U"}
PANEL_W = 720


def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ------------------------------------------------------------------ the frames
def pick_frames(seed=0):
    rng = random.Random(seed)
    out = []
    ol = sources.openlane_sets()
    for s in ("A", "B"):
        recs = sources.load_records(*ol[f"{s}_val"])
        for r in rng.sample(recs, 2):
            out.append({"set": f"OpenLane-V2 {s} val", "domain": "in distribution", "rule": "us", "side": r["traffic_side"],
                        "path": os.path.abspath(os.path.join(r["root"], r["image"])), "gt": r["color"],
                        "gt_boxes": [b["box"] for b in r["boxes"] if b["relevant"]], "id": f"{r['segment_id']} {r['timestamp']}"})
    route = sources.read_jsonl(labels_dtld.label_path("route", "test"))
    junction = sources.read_jsonl(sources.dtld_junction_path("test"))
    both = [r for r, j in zip(route, junction) if r["kept"] and j["excluded"] is None]
    for r in rng.sample(both, 2):
        out.append({"set": "DTLD test", "domain": "in distribution", "rule": "german", "side": "right", "path": r["image"],
                    "gt": r["color"], "gt_boxes": [b["box"] for b in r["boxes"] if b["relevant"]], "id": os.path.basename(r["image"])})
    lisa = sources.lisa_test()
    for group in ("day", "night"):
        it = rng.choice([it for it in lisa if it["group"] == group and any(it["color"])])
        out.append(_ood(it, f"LISA test ({group})"))
    out.append(_ood(rng.choice([it for it in sources.s2tld() if it["group"] == "1920x1080" and any(it["color"])]), "S2TLD 1920x1080"))
    vzc = rng.choice([it for it in sources.vzc() if it["label"] in ("RR", "RG")])
    out.append(_ood(vzc, "VZC-TLD", gt_text={"RR": "relevant light red / yellow: stop", "RG": "relevant light green: go"}[vzc["label"]]))
    return out


def _ood(it, name, gt_text="no per-maneuver labels"):
    boxes = [b for b, c in zip(it["boxes"].tolist(), it["color"]) if c]
    return {"set": name, "domain": "out of distribution", "rule": "us", "side": "right", "path": it["path"], "gt": None,
            "gt_text": gt_text, "gt_boxes": boxes, "id": os.path.basename(it["path"])}


# ------------------------------------------------------------------ the two models
class Model:
    def __init__(self, kind, run, epoch, threshold):
        self.kind, self.thr, self.res = kind, threshold, run_args(run)["resolution"]
        if kind == "R":
            self.net = load_model(run, epoch, "cpu")
        else:
            self.net, self.names = moves.load_model(run, epoch, "cpu"), moves.class_names(run)

    @torch.no_grad()
    def lights(self, path):
        """Detections above the threshold after NMS: (xyxy pixel box, color, label text, used for the decision)."""
        with Image.open(path) as im:
            w, h = im.size
        x = load_image(path, self.res)[None]
        if self.kind == "R":
            score, probs, boxes = factorized.detect(self.net, x, amp=False)
            score, boxes = score[0], boxes[0]
            picto, color, junc = probs["picto"][0].argmax(-1), probs["color"][0].argmax(-1), probs["junction"][0]
        else:
            cls, score, boxes = moves.detect(self.net, x, len(self.names))
            cls, score, boxes = cls[0], score[0], boxes[0]
        k = torch.nonzero(score > self.thr).flatten()
        b = box_convert(boxes[k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
        k = k[nms(b, score[k], NMS_IOU)] if len(k) else k
        out = []
        for q in k.tolist():
            bx = (box_convert(boxes[q][None], "cxcywh", "xyxy")[0] * torch.tensor([w, h, w, h])).tolist()
            if self.kind == "R":
                p, c, j = PICTOGRAMS[int(picto[q])], LAMP_COLORS[int(color[q])], float(junc[q]) > 0.5
                out.append({"box": bx, "color": c, "text": PICTO_SHORT.get(p, p), "used": j, "picto": p, "score": float(score[q])})
            else:
                m, c = moves.parse_class(self.names[int(cls[q])])
                text = "x" if m is None else "+".join(MOVE_SHORT.get(v, v) for v in sorted(m))
                out.append({"box": bx, "color": c, "text": text, "used": m is not None, "moves": m, "score": float(score[q])})
        return out

    def decide(self, lights, frame):
        if self.kind == "R":
            js = [(l["picto"], l["color"], l["score"]) for l in lights if l["used"]]
            return junction_rule(js) if frame["rule"] == "german" else us_rule(js, frame["side"])
        dets = [(l["moves"], l["color"], l["score"]) for l in lights if l["used"]]
        return moves.assign_scored(dets, frame["side"], "signal" if frame["rule"] == "german" else "dedicated")


# ------------------------------------------------------------------ drawing
def crop_box(frame, size):
    w, h = size
    bs = frame["gt_boxes"] or [[w * 0.4, h * 0.2, w * 0.6, h * 0.4]]
    x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
    x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
    cw = min(w, max((x2 - x1) * 2.5, 0.45 * w, 320))
    ch = min(h, cw * 0.62)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - cw / 2, 0), w - cw)
    top = min(max(cy - ch / 2, 0), h - ch)
    return left, top, left + cw, top + ch


def chips(draw, x, y, label, colors, f):
    draw.text((x, y), label, fill=(20, 20, 20), font=f)
    x += 56
    for d in DIRECTIONS:
        c = colors[d]
        draw.rectangle([x, y + 1, x + 104, y + 21], fill=RGB[c])
        draw.text((x + 5, y + 2), f"{d[0].upper()}: {c}", fill=(255, 255, 255) if c != "yellow" else (0, 0, 0), font=f)
        x += 110


def panel(img, frame, model_name, lights, decision, cb):
    left, top, right, bottom = cb
    s = PANEL_W / (right - left)
    crop = img.crop((int(left), int(top), int(right), int(bottom))).resize((PANEL_W, int((bottom - top) * s)))
    d = ImageDraw.Draw(crop)
    f = font(15)
    tr = lambda b: [(b[0] - left) * s, (b[1] - top) * s, (b[2] - left) * s, (b[3] - top) * s]
    for b in frame["gt_boxes"]:
        d.rectangle(tr(b), outline=(255, 255, 255), width=1)
    for l in lights:
        b = tr(l["box"])
        if b[2] < 0 or b[0] > PANEL_W or b[3] < 0 or b[1] > crop.height:
            continue
        d.rectangle([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], outline=RGB[l["color"]], width=4 if l["used"] else 1)
        tw = d.textlength(l["text"], font=f)
        d.rectangle([b[0] - 2, b[1] - 22, b[0] + tw + 4, b[1] - 3], fill=(0, 0, 0))
        d.text((b[0] + 1, b[1] - 21), l["text"], fill=RGB[l["color"]], font=f)
    strip = Image.new("RGB", (PANEL_W, 62), (245, 245, 245))
    ds = ImageDraw.Draw(strip)
    ds.text((8, 4), model_name, fill=(20, 20, 20), font=font(16))
    chips(ds, 60, 4, "model", decision, font(14))
    if frame["gt"] is not None:
        chips(ds, 60, 33, "truth", frame["gt"], font(14))
    else:
        ds.text((116, 35), "truth: " + frame["gt_text"], fill=(60, 60, 60), font=font(14))
    out = Image.new("RGB", (PANEL_W, crop.height + strip.height), (255, 255, 255))
    out.paste(crop, (0, 0))
    out.paste(strip, (0, crop.height))
    return out


def row(frame, panels):
    head = 34
    h = max(p.height for p in panels)
    out = Image.new("RGB", (PANEL_W * 2 + 12, h + head), (255, 255, 255))
    d = ImageDraw.Draw(out)
    tag = "IN" if frame["domain"] == "in distribution" else "OUT"
    d.rectangle([0, 0, out.width, head - 4], fill=(40, 60, 110) if tag == "IN" else (120, 50, 30))
    d.text((8, 6), f"{frame['set']}  |  {frame['domain']}  |  {frame['id']}", fill=(255, 255, 255), font=font(17))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL_W + 12), head))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r-run", default="runs/exp10_R")
    ap.add_argument("--r-epoch", type=int, default=2)
    ap.add_argument("--r-threshold", type=float, default=0.35)
    ap.add_argument("--m-run", default="runs/rfdetr_m_mix")
    ap.add_argument("--m-epoch", type=int, default=1)
    ap.add_argument("--m-threshold", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="docs/examples")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    frames = pick_frames(args.seed)
    models = [Model("R", args.r_run, args.r_epoch, args.r_threshold), Model("M", args.m_run, args.m_epoch, args.m_threshold)]
    rows = {"in": [], "out": []}
    for i, fr in enumerate(frames):
        img = Image.open(fr["path"]).convert("RGB")
        cb = crop_box(fr, img.size)
        panels = []
        for m in models:
            ls = m.lights(fr["path"])
            panels.append(panel(img, fr, m.kind, ls, m.decide(ls, fr), cb))
        r = row(fr, panels)
        r.save(os.path.join(args.out, f"{i:02d}.jpg"), quality=90)
        rows["in" if fr["domain"] == "in distribution" else "out"].append(r)
        print(f"{i:02d} {fr['set']}: {fr['id']}", flush=True)
    for k, rs in rows.items():
        sheet = Image.new("RGB", (rs[0].width, sum(r.height + 10 for r in rs)), (255, 255, 255))
        y = 0
        for r in rs:
            sheet.paste(r, (0, y))
            y += r.height + 10
        sheet.save(os.path.join(args.out, f"examples_{k}_distribution.jpg"), quality=88)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
