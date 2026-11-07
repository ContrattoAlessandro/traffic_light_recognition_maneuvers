"""Visual gallery of per-light OOD predictions of two RF-DETR runs (A+B and DTLD) on ATLAS, S2TLD, BDD100K, LISA.

For each dataset, SAMPLE random images with at least one annotated colored light (seed 0) are run through
both models (their own epoch and score threshold, class-agnostic NMS, as in
``tools/eval_rfdetr_perception.py``) and matched to the annotations with the center criterion. An image is
"correct" for a model when every annotated light is found with the right color and there is no unmatched
detection. Up to PER_KIND images are shown for each case: both correct, only A+B wrong, only DTLD wrong,
both wrong. Each panel shows, per model:
- the image with ground truth (white), correct detections (green), wrong color (orange), unmatched
  detections (red) and missed lights (magenta), each detection labelled with its class and score;
- the per-direction colors the model decodes from its relevant detections (A+B: US rule, right only by a
  dedicated signal; DTLD: its own German rule, the circle also governs right), to judge the target;
- zoomed crops of every light involved.
Writes data/ood_gallery/<dataset>_<n>.jpg and data/ood_gallery/index.html (git-ignored: dataset images).
"""
import json
import os
import random
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, parse_class  # noqa: E402
from tools.eval_hybrid import RUNS, best_epoch  # noqa: E402
from tools.eval_rfdetr import load_model  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS, match, predictions  # noqa: E402

OUT = "data/ood_gallery"
SAMPLE, PER_KIND, WIDTH, ZOOM = 160, 2, 1100, 110
MODELS = {"A+B": "dedicated", "DTLD": "signal"}
COL = {"gt": (255, 255, 255), "ok": (0, 220, 0), "color": (255, 150, 0), "fp": (255, 40, 40), "fn": (255, 0, 255)}
FILL = {"red": (220, 40, 40), "yellow": (240, 190, 0), "green": (30, 170, 60), "none": (120, 120, 120)}


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def analyse(it, pred, class_names, thr, near_turn):
    c, s, b = pred
    k = s > thr
    c, s, b = c[k], s[k], b[k]
    ignore = torch.tensor([col is None for col in it["color"]], dtype=torch.bool)
    m = match(b, s, it["boxes"], ignore, "center")
    dets, errors, rel = [], [], []
    for p in range(len(s)):
        moves, color = parse_class(class_names[int(c[p])])
        g = int(m[p])
        kind = "ok" if g >= 0 and it["color"][g] == color else "color" if g >= 0 else "fp" if g == -1 else "ign"
        dets.append({"box": b[p].tolist(), "label": f"{'+'.join(sorted(moves)) if moves else 'not relevant'} | {color} {float(s[p]):.2f}",
                     "kind": kind, "gt": g})
        if kind in ("color", "fp"):
            errors.append(kind)
        if moves is not None:
            rel.append((set(moves), color, float(s[p])))
    found = {d["gt"] for d in dets if d["gt"] >= 0}
    missed = [g for g in range(len(it["color"])) if not ignore[g] and g not in found]
    errors += ["fn"] * len(missed)
    return {"dets": dets, "missed": missed, "errors": errors, "directions": assign_scored(rel, "right", near_turn)}


def draw_row(img, it, a, name):
    scale = WIDTH / img.width
    im = img.resize((WIDTH, round(img.height * scale)))
    d = ImageDraw.Draw(im)
    f = font(13)
    for g, bx in enumerate(it["boxes"].tolist()):
        if it["color"][g] is not None:
            d.rectangle([v * scale for v in bx], outline=COL["fn"] if g in a["missed"] else COL["gt"], width=1 if g not in a["missed"] else 3)
    for det in a["dets"]:
        if det["kind"] == "ign":
            continue
        x1, y1, x2, y2 = [v * scale for v in det["box"]]
        d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=COL[det["kind"]], width=2)
        tw = d.textlength(det["label"], font=f)
        d.rectangle([x1, y1 - 18, x1 + tw + 4, y1 - 2], fill=(0, 0, 0))
        d.text((x1 + 2, y1 - 17), det["label"], fill=COL[det["kind"]], font=f)
    # header: model, errors, decoded directions
    head = Image.new("RGB", (WIDTH, 34), (25, 25, 25))
    hd = ImageDraw.Draw(head)
    err = {k: a["errors"].count(k) for k in ("fn", "fp", "color") if a["errors"].count(k)}
    status = "CORRECT" if not err else "WRONG: " + ", ".join(f"{n} {dict(fn='missed', fp='extra', color='wrong color')[k]}" for k, n in err.items())
    hd.text((8, 8), f"{name}  {status}", fill=(255, 255, 255), font=font(15))
    x = 600
    for dname in DIRECTIONS:
        col = a["directions"][dname]
        hd.rectangle([x, 6, x + 118, 28], fill=FILL[col])
        hd.text((x + 6, 9), f"{dname}: {col}", fill=(0, 0, 0) if col == "yellow" else (255, 255, 255), font=font(13))
        x += 124
    # zoom strip: every annotated light and every detection that is not ignored
    boxes = [(bx, "fn" if g in a["missed"] else "gt") for g, bx in enumerate(it["boxes"].tolist()) if it["color"][g] is not None]
    boxes += [(det["box"], det["kind"]) for det in a["dets"] if det["kind"] in ("fp", "color")]
    strip = Image.new("RGB", (WIDTH, ZOOM + 4 if boxes else 0), (40, 40, 40))
    for n, (bx, kind) in enumerate(boxes[: WIDTH // (ZOOM + 4)]):
        x1, y1, x2, y2 = bx
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 8) * 1.8
        crop = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))).resize((ZOOM, ZOOM))
        ImageDraw.Draw(crop).rectangle([0, 0, ZOOM - 1, ZOOM - 1], outline=COL[kind], width=3)
        strip.paste(crop, (n * (ZOOM + 4) + 2, 2))
    out = Image.new("RGB", (WIDTH, head.height + im.height + strip.height))
    out.paste(head, (0, 0))
    out.paste(im, (0, head.height))
    out.paste(strip, (0, head.height + im.height))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    models = {}
    for name in MODELS:
        run, cn, _ = RUNS[name]
        epoch, thr = best_epoch(run)
        class_names = json.load(open(cn))
        net, res, _ = load_model(run, os.path.join(run, f"epoch_{epoch}_ema.pth"), len(class_names))
        models[name] = (net, res, class_names, thr)
    html = ["<!doctype html><meta charset='utf-8'><title>OOD Gallery</title><style>body{background:#111;color:#eee;font:15px system-ui;"
            "max-width:1140px;margin:auto;padding:16px}img{width:100%;margin:6px 0 26px}h2{margin-top:40px}</style>",
            "<h1>OOD predictions: A+B vs DTLD</h1><p>White = annotated light, green = found with the right color, orange = found with "
            "the wrong color, red = detection with no annotated light, magenta = annotated light missed. The colored boxes at the top "
            "right are the per-direction signals each model decodes (A+B: US rule; DTLD: German rule). The strip at the bottom "
            "zooms on each light.</p>"]
    for ds in ("atlas", "s2tld", "bdd100k", "lisa"):
        items = [it for it in LOADERS[ds]() if any(c is not None for c in it["color"])]
        items = random.Random(0).sample(items, min(SAMPLE, len(items)))
        res = {}
        for name, (net, r, cn, thr) in models.items():
            preds = predictions(net, items, cn, r, 4, 8)
            res[name] = [analyse(it, p, cn, thr, MODELS[name]) for it, p in zip(items, preds)]
        kinds = {"both correct": (False, False), "only A+B wrong": (True, False), "only DTLD wrong": (False, True), "both wrong": (True, True)}
        html.append(f"<h2>{ds}</h2>")
        counts = {k: sum((bool(res['A+B'][i]['errors']), bool(res['DTLD'][i]['errors'])) == v for i in range(len(items))) for k, v in kinds.items()}
        html.append(f"<p>{len(items)} random images: " + ", ".join(f"{k} {n}" for k, n in counts.items()) + "</p>")
        for kind, want in kinds.items():
            idx = [i for i in range(len(items)) if (bool(res["A+B"][i]["errors"]), bool(res["DTLD"][i]["errors"])) == want][:PER_KIND]
            for i in idx:
                img = Image.open(items[i]["path"]).convert("RGB")
                rows = [draw_row(img, items[i], res[n][i], n) for n in MODELS]
                panel = Image.new("RGB", (WIDTH, sum(r.height for r in rows) + 6), (0, 0, 0))
                y = 0
                for r in rows:
                    panel.paste(r, (0, y))
                    y += r.height + 6
                fn = f"{ds}_{i}.jpg"
                panel.save(os.path.join(OUT, fn), quality=88)
                html.append(f"<h3>{kind} - {os.path.basename(items[i]['path'])} ({items[i]['group']})</h3><img src='{fn}' alt='{ds} {kind}'>")
        print(ds, counts, flush=True)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))


if __name__ == "__main__":
    main()
