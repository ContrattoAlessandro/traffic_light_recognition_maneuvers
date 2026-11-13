"""Manual review pages of the per-direction targets of the mixed model (Exp 5, runs/rfdetr_m_mix).

Part 1 (labelled test sets): 50 frames from A val, B val and D*_US test (15 / 15 / 20), half where the
model's per-direction prediction equals the label on every direction ("correct"), half where it does not
("wrong"), chosen at random (seed 0) among frames with at least one relevant labelled light. Correct /
wrong comes from runs/rfdetr_m_mix/val_preds_mix.npz (epoch 1, threshold 0.25). The page shows the
predicted and the labelled per-direction colors.

Part 2 (OOD, no target labels): 25 random images (seed 0) with at least one annotated colored light from
each of ATLAS, S2TLD, BDD100K and LISA, with the predicted per-direction colors only. Rules: ATLAS is
decoded with the German rule (the circle also governs right), the others with the US rule (right only by a
dedicated signal); S2TLD is Chinese, where right turns are not governed by the circle either.

Each image shows the detections the model calls relevant (box in the light's color, label "moves | color
score") and, thin and grey, those it calls not relevant; the strip below zooms on each relevant one.
Writes data/review_mix/*.jpg, items.json and index.html: a review page with a verdict per image
(correct / wrong / unsure), the wrong directions and a note, kept in the browser and exported as JSON.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import COLORS, DIRECTIONS  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, parse_class  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_rfdetr import load_model, run_detector  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS, predictions  # noqa: E402

RUN, EPOCH, THR = "runs/rfdetr_m_mix", 1, 0.25
CLASS_NAMES = "data/rfdetr_coco_fit_mix/class_names.json"
OUT = "data/review_mix"
TEST = {"A_val": ("A", 15), "B_val": ("B", 15), "DTLD_US_test": ("DTLD_US_test", 20)}  # set: (npz key, n)
OOD = {"atlas": "signal", "s2tld": "dedicated", "bdd100k": "dedicated", "lisa": "dedicated"}
N_OOD, WIDTH, ZOOM = 25, 1280, 96
RGB = {"red": (235, 50, 50), "yellow": (245, 200, 0), "green": (40, 200, 80), "none": (150, 150, 150)}


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def kept(c, s, b):
    """Threshold + class-agnostic NMS (as in decoding); b xyxy pixels."""
    k = s > THR
    c, s, b = c[k], s[k], b[k]
    if len(s):
        keep = nms(b, s, 0.5)
        c, s, b = c[keep], s[keep], b[keep]
    return c, s, b


def render(path, dets, out_path, show_score=True):
    """dets: list of (moves or None, color, score, xyxy pixel box)."""
    img = Image.open(path).convert("RGB")
    scale = WIDTH / img.width
    im = img.resize((WIDTH, round(img.height * scale)))
    d = ImageDraw.Draw(im)
    f = font(14)
    for moves, color, score, bx in sorted(dets, key=lambda x: x[0] is not None):
        x1, y1, x2, y2 = [v * scale for v in bx]
        if moves is None:
            d.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(170, 170, 170), width=1)
            continue
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=RGB[color], width=3)
        label = f"{'+'.join(sorted(moves))} | {color}" + (f" {score:.2f}" if show_score else "")
        tw = d.textlength(label, font=f)
        ty = y1 - 22 if y1 > 24 else y2 + 4
        d.rectangle([x1 - 3, ty, x1 + tw + 3, ty + 18], fill=(0, 0, 0))
        d.text((x1, ty + 1), label, fill=RGB[color], font=f)
    rel = [x for x in dets if x[0] is not None]
    per_row = WIDTH // (ZOOM + 4)
    rows = (len(rel) + per_row - 1) // per_row
    strip = Image.new("RGB", (WIDTH, rows * (ZOOM + 22)), (30, 30, 30))
    sd = ImageDraw.Draw(strip)
    for n, (moves, color, score, (x1, y1, x2, y2)) in enumerate(rel):
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 8) * 2.2
        crop = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))).resize((ZOOM, ZOOM))
        ImageDraw.Draw(crop).rectangle([0, 0, ZOOM - 1, ZOOM - 1], outline=RGB[color], width=3)
        x, y = (n % per_row) * (ZOOM + 4) + 2, (n // per_row) * (ZOOM + 22)
        strip.paste(crop, (x, y + 2))
        sd.text((x + 2, y + ZOOM + 4), "+".join(sorted(moves)), fill=(230, 230, 230), font=font(12))
    out = Image.new("RGB", (WIDTH, im.height + strip.height))
    out.paste(im, (0, 0))
    out.paste(strip, (0, im.height))
    out.save(out_path, quality=88)


def pick_opposite(gc, em, n):
    """Frames whose label has a red and a green direction, round-robin over the label patterns
    (straight, left, right, uturn colors) so rare combinations are shown too; random within a pattern."""
    rng = random.Random(0)
    groups = {}
    for i in np.flatnonzero((gc == COLORS.index("red")).any(1) & (gc == COLORS.index("green")).any(1)):
        groups.setdefault(tuple(gc[i]), []).append(int(i))
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    pick = []
    while len(pick) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(pick) < n:
                pick.append(groups[k].pop())
    return [(i, bool(em[i])) for i in pick]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--opposite", action="store_true",
                    help="labelled frames with a red and a green direction only (A val 25, B val 15, D*_US test 40), no OOD; "
                         "writes data/review_opposite")
    args = ap.parse_args()
    out = "data/review_opposite" if args.opposite else OUT
    test = {"A_val": ("A", 25), "B_val": ("B", 15), "DTLD_US_test": ("DTLD_US_test", 40)} if args.opposite else TEST
    os.makedirs(out, exist_ok=True)
    class_names = json.load(open(CLASS_NAMES))
    model, res, _ = load_model(RUN, os.path.join(RUN, f"epoch_{EPOCH}_ema.pth"), len(class_names))
    npz = np.load(os.path.join(RUN, "val_preds_mix.npz"))
    items = []

    for set_name, (key, n) in test.items():
        recs = load_records(*SETS[set_name])
        pc, pp, gc, gp = (npz[f"{key}_{x}"] for x in ("pc", "pp", "gc", "gp"))
        assert len(recs) == len(pc), (set_name, len(recs), len(pc))
        assert all(COLORS[gc[i, j]] == r["color"][dn] for i, r in enumerate(recs[:200]) for j, dn in enumerate(DIRECTIONS))
        em = ((pc == gc) & (pp == gp)).all(1)
        if args.opposite:
            pick = pick_opposite(gc, em, n)
        else:
            cand = [i for i, r in enumerate(recs) if r["n_relevant_lights"]]
            rng = random.Random(0)
            pick = [(i, True) for i in rng.sample([i for i in cand if em[i]], n // 2)]
            pick += [(i, False) for i in rng.sample([i for i in cand if not em[i]], n - n // 2)]
        sel = [recs[i] for i, _ in pick]
        dets = run_detector(model, sel, class_names, res, 4)
        for (i, ok), r, (c, s, b) in zip(pick, sel, dets):
            path = os.path.join(r["root"], r["image"])
            w, h = Image.open(path).size
            bx = box_convert(b, "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            c, s, bx = kept(c, s, bx)
            dl = [(*parse_class(class_names[int(ci)]), float(si), bi) for ci, si, bi in zip(c, s, bx.tolist())]
            pred = assign_scored([(m, col, sc) for m, col, sc, _ in dl if m is not None], r["traffic_side"], r.get("near_turn", "dedicated"))
            assert all(pred[dn] == COLORS[pc[i, j]] for j, dn in enumerate(DIRECTIONS)), (set_name, i, pred)
            fid = f"{set_name}_{i}"
            render(path, dl, os.path.join(out, f"{fid}.jpg"))
            items.append({"id": fid, "part": "test", "set": set_name, "auto": "correct" if ok else "wrong",
                          "rule": "US" if r.get("near_turn", "dedicated") == "dedicated" else "German",
                          "file": os.path.basename(path), "pred": pred, "gt": r["color"]})
        print(set_name, "done", flush=True)

    for ds, rule in ({} if args.opposite else OOD).items():
        pool = [it for it in LOADERS[ds]() if any(c is not None for c in it["color"])]
        sel = random.Random(0).sample(pool, N_OOD)
        preds = predictions(model, sel, class_names, res, 4, 8)
        for n, (it, (c, s, b)) in enumerate(zip(sel, preds)):
            c, s, b = kept(c, s, b)
            dl = [(*parse_class(class_names[int(ci)]), float(si), bi) for ci, si, bi in zip(c, s, b.tolist())]
            pred = assign_scored([(m, col, sc) for m, col, sc, _ in dl if m is not None], "right", rule)
            fid = f"{ds}_{n}"
            render(it["path"], dl, os.path.join(out, f"{fid}.jpg"))
            items.append({"id": fid, "part": "ood", "set": ds, "auto": None, "rule": "German" if rule == "signal" else "US",
                          "file": os.path.relpath(it["path"]).replace("\\", "/"), "group": it["group"], "pred": pred, "gt": None})
        print(ds, "done", flush=True)

    with open(os.path.join(out, "items.json"), "w") as f:
        json.dump(items, f, indent=1)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_gallery.html"), encoding="utf-8").read()
    name = os.path.basename(out)
    title = "opposite signals (a red and a green direction)" if args.opposite else "mixed model"
    page = page.replace("/*ITEMS*/[]", json.dumps(items)).replace("review_mix_v1", f"{name}_v1").replace("TITLE", title)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(len(items), "items ->", out)


if __name__ == "__main__":
    main()
