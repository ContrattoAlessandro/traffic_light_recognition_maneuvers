"""Manual review page of the Exp 8 model (runs/exp8_bdd_F, selected epoch and threshold) on 110 images.

- In domain (50): A val 13, B val 13 (US rule + lane topology mask), DTLD test 12 (junction rule), ATLAS test 12
  (junction rule, German convention). Sources the model trained on, but these frames were not trained on.
- OOD (50): LISA test day 17, S2TLD 17 (9 at 1080p, 8 at 720p), VZC-TLD 16 (US rule; VZC also shows its image label).
- Night (10): LISA test night 5 (OOD), BDD100K val night 5 (BDD100K train is in F8's training data).
Random (seed 0) among frames with at least one annotated colored light (OpenLane: one relevant light).
Each image: annotations thin white (magenta = missed by the model, center criterion), junction detections in their
color with "pictogram | color score", non-junction detections thin grey, a header with predicted (and labelled)
per-maneuver colors, and a zoom strip. Writes data/review_exp8/*.jpg, items.json and index.html.
Another model on the same frames: --run runs/exp10_R --result runs/exp10_eval/result.json --name R
--out data/review_exp10 (epoch and threshold are that model's selection in the result file).
"""
import argparse
import json
import os
import random
import sys

import cv2
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.exp7_checks as ck  # noqa: E402
import tools.eval_vzc as vzc  # noqa: E402
from dtsu import factorized  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS, rule  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_picto import load_model  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS, match  # noqa: E402
from tools.make_picto_gallery import SHORT  # noqa: E402
from tools.make_review_gallery import font  # noqa: E402
from tools.run_video import preprocess  # noqa: E402

RUN, DATA = "runs/exp8_bdd_F", "data/rfdetr_coco_exp8"
OUT = "data/review_exp8"
WIDTH, ZOOM = 1280, 96
RGB = {"red": (235, 50, 50), "yellow": (245, 200, 0), "green": (40, 200, 80), "none": (110, 110, 110)}
NAMES = {**SHORT, "arrow_straight_right": "straight+right arrow"}
MOVES = ("straight", "left", "right", "uturn")


def pick(pool, n, seed=0):
    return random.Random(seed).sample(pool, n)


def gather():
    """-> list of dict(id, part, set, rule, side, path, gt_boxes [n,4], gt_color, gt_dirs or None, vzc_label, topo)."""
    out = []
    for name, n in (("A_val", 13), ("B_val", 13)):
        recs = [r for r in load_records(*SETS[name]) if r["n_relevant_lights"]]
        sel = pick(recs, n)
        masks = ck.topo_masks(sel)
        for r, m in zip(sel, masks):
            out.append(dict(part="in domain", set=name, rule="US + topology", side=r["traffic_side"], path=os.path.join(r["root"], r["image"]),
                            gt_boxes=[b["box"] for b in r["boxes"]], gt_color=[b["color"] for b in r["boxes"]],
                            gt_rel=[b["relevant"] for b in r["boxes"]], gt_dirs=r["color"], topo=m))
    test = [json.loads(l) for l in open("data/rfdetr_coco_dtld_picto/dtld_test_records.jsonl")]
    test = [r for r in test if not r.get("excluded") and any(b["junction"] for b in r["boxes"])]
    for r in pick(test, 12):
        out.append(dict(part="in domain", set="DTLD_test", rule="junction (German)", side="right", path=r["image"],
                        gt_boxes=[b["box"] for b in r["boxes"]], gt_color=[b["color"] for b in r["boxes"]],
                        gt_rel=[b["junction"] for b in r["boxes"]], gt_dirs=r["color"]))
    loaded = {ds: [it for it in LOADERS[ds]() if any(c is not None for c in it["color"])] for ds in ("atlas", "lisa", "s2tld", "bdd100k")}
    loaded["vzc"] = [it for it in vzc.load_items() if any(c is not None for c in it["color"])]

    def add(part, ds, items, rule_name, tag=None):
        for it in items:
            out.append(dict(part=part, set=ds + (f" {tag}" if tag else ""), rule=rule_name, side="right", path=it["path"],
                            gt_boxes=it["boxes"].tolist(), gt_color=it["color"], gt_rel=it.get("relevant"),
                            vzc_label=it.get("label")))

    add("in domain", "ATLAS_test", pick(loaded["atlas"], 12), "junction (German)")
    add("OOD", "LISA", pick([i for i in loaded["lisa"] if i["group"] == "day"], 17), "US", "day")
    add("OOD", "S2TLD", pick([i for i in loaded["s2tld"] if i["group"] == "1920x1080"], 9), "US", "1080p")
    add("OOD", "S2TLD", pick([i for i in loaded["s2tld"] if i["group"] == "1280x720"], 8), "US", "720p")
    add("OOD", "VZC", pick(loaded["vzc"], 16), "US")
    add("night", "LISA", pick([i for i in loaded["lisa"] if i["group"] == "night"], 5), "US", "night (OOD)")
    add("night", "BDD100K val", pick([i for i in loaded["bdd100k"] if i["group"] == "night"], 5), "US", "night (in domain)")
    return out


def infer(net, res, path, thr):
    f = cv2.imread(path)
    h, w = f.shape[:2]
    score, probs, boxes = factorized.detect(net, preprocess([f], (0, 0, w, h), res).cuda())
    k = score[0] > thr
    s = score[0][k]
    b = box_convert(boxes[0][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
    keep = nms(b, s, 0.5) if len(s) else torch.zeros(0, dtype=torch.long)
    p, c, j = probs["picto"][0][k][keep].argmax(-1), probs["color"][0][k][keep].argmax(-1), probs["junction"][0][k][keep] > 0.5
    return [dict(picto=PICTOS[int(a)], color=COLORS[int(cc)], junction=bool(jj), score=float(ss), box=bb)
            for a, cc, jj, ss, bb in zip(p, c, j, s[keep], b[keep].tolist())]


def directions(it, dets):
    lights = [(d["picto"], d["color"], d["score"]) for d in dets if d["junction"]]
    if it["rule"].startswith("junction"):
        return rule(lights)
    f = ck.us_rule(lights, it["side"])
    return ck.apply_topo(f, it.get("topo"))


def image_decision(dets):
    """VZC-style image decision: the highest-scoring junction light; red / yellow = stop, green = go."""
    j = [d for d in dets if d["junction"]]
    if not j:
        return "none"
    return "go" if max(j, key=lambda d: d["score"])["color"] == "green" else "stop"


def render(it, dets, dirs, out_path):
    img = Image.open(it["path"]).convert("RGB")
    scale = WIDTH / img.width
    im = img.resize((WIDTH, round(img.height * scale)))
    d = ImageDraw.Draw(im)
    f = font(14)
    gtb = torch.tensor(it["gt_boxes"], dtype=torch.float32).reshape(-1, 4)
    ignore = torch.tensor([c is None for c in it["gt_color"]], dtype=torch.bool)
    m = match(torch.tensor([x["box"] for x in dets], dtype=torch.float32).reshape(-1, 4), torch.tensor([x["score"] for x in dets]), gtb, ignore, "center")
    found = set(m.tolist())
    missed = [g for g in range(len(gtb)) if not ignore[g] and g not in found]
    wrong_color = sum(1 for p, g in enumerate(m.tolist()) if g >= 0 and it["gt_color"][g] != dets[p]["color"])
    extra = int((m == -1).sum())
    for g, bx in enumerate(gtb.tolist()):
        x1, y1, x2, y2 = [v * scale for v in bx]
        if g in missed:
            d.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=(255, 0, 255), width=2)
        else:
            d.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(255, 255, 255), width=1)
    for x in sorted(dets, key=lambda x: x["junction"]):
        x1, y1, x2, y2 = [v * scale for v in x["box"]]
        if not x["junction"]:
            d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(170, 170, 170), width=1)
            continue
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=RGB[x["color"]], width=3)
        label = f"{NAMES[x['picto']]} | {x['color']} {x['score']:.2f}"
        tw = d.textlength(label, font=f)
        ty = y1 - 22 if y1 > 24 else y2 + 4
        d.rectangle([x1 - 3, ty, x1 + tw + 3, ty + 18], fill=(0, 0, 0))
        d.text((x1, ty + 1), label, fill=RGB[x["color"]], font=f)
    # header: predicted per-maneuver colors (and labelled ones when the set has them)
    rows = [("model", dirs)] + ([("label", it["gt_dirs"])] if it.get("gt_dirs") else [])
    head = Image.new("RGB", (WIDTH, 30 * len(rows) + 6), (25, 25, 25))
    hd = ImageDraw.Draw(head)
    for r, (who, dd) in enumerate(rows):
        y = 4 + 30 * r
        hd.text((8, y + 4), who, fill=(255, 255, 255), font=font(15))
        x = 80
        for mv in MOVES:
            col = dd.get(mv, "none")
            hd.rectangle([x, y, x + 150, y + 24], fill=RGB[col])
            hd.text((x + 6, y + 4), f"{mv}: {col}", fill=(0, 0, 0) if col == "yellow" else (255, 255, 255), font=font(13))
            x += 156
    # zoom strip on junction detections and missed lights
    zs = [(x["box"], RGB[x["color"]], NAMES[x["picto"]]) for x in dets if x["junction"]]
    zs += [(gtb[g].tolist(), (255, 0, 255), "missed") for g in missed]
    per_row = WIDTH // (ZOOM + 4)
    nrows = (len(zs) + per_row - 1) // per_row
    strip = Image.new("RGB", (WIDTH, nrows * (ZOOM + 22)), (30, 30, 30))
    sd = ImageDraw.Draw(strip)
    for n, ((x1, y1, x2, y2), col, txt) in enumerate(zs):
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 8) * 2.2
        crop = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))).resize((ZOOM, ZOOM))
        ImageDraw.Draw(crop).rectangle([0, 0, ZOOM - 1, ZOOM - 1], outline=col, width=3)
        x, y = (n % per_row) * (ZOOM + 4) + 2, (n // per_row) * (ZOOM + 22)
        strip.paste(crop, (x, y + 2))
        sd.text((x + 2, y + ZOOM + 4), txt, fill=(230, 230, 230), font=font(11))
    out = Image.new("RGB", (WIDTH, head.height + im.height + strip.height))
    out.paste(head, (0, 0))
    out.paste(im, (0, head.height))
    out.paste(strip, (0, head.height + im.height))
    out.save(out_path, quality=85)
    n_gt = int((~ignore).sum())
    return {"lights": n_gt, "found": n_gt - len(missed), "missed": len(missed), "wrong_color": wrong_color, "extra": extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--result", default=os.path.join(RUN, "exp8_eval", "result.json"))
    ap.add_argument("--name", default="F8", help="model key in the result file's selection")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    sel = json.load(open(args.result))["selection"][args.name]
    epoch, thr = sel["epoch"], sel["threshold"]
    classes = json.load(open(os.path.join(DATA, "class_names.json")))
    net, res = load_model(args.run, epoch, classes, "F")
    print(f"{args.name} epoch {epoch}, threshold {thr}, resolution {res}", flush=True)
    items = gather()
    print(len(items), "images", flush=True)
    page_items = []
    with torch.no_grad():
        for n, it in enumerate(items):
            dets = infer(net, res, it["path"], thr)
            dirs = directions(it, dets)
            fid = f"{n:03d}"
            stats = render(it, dets, dirs, os.path.join(out_dir, fid + ".jpg"))
            page_items.append({"id": fid, "part": it["part"], "set": it["set"], "rule": it["rule"], "file": os.path.basename(it["path"]),
                               "path": it["path"], "pred": dirs, "gt": it.get("gt_dirs"), "vzc_label": {"RR": "stop", "RG": "go", "NoR": "none"}.get(it.get("vzc_label")),
                               "decision": image_decision(dets), **stats})
            if n % 10 == 9:
                print(n + 1, flush=True)
    with open(os.path.join(out_dir, "items.json"), "w") as f:
        json.dump({"epoch": epoch, "threshold": thr, "items": page_items}, f, indent=1)
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_exp8.html"), encoding="utf-8").read()
    assert "/*ITEMS*/[]" in page
    page = page.replace("/*ITEMS*/[]", json.dumps(page_items)).replace("/*META*/", f"epoch {epoch}, threshold {thr}, resolution {res}")
    if args.name != "F8":
        stem = os.path.basename(out_dir.rstrip("/\\"))
        page = (page.replace("Exp 8 model (F8)", f"Model {args.name} ({args.run})").replace("<title>Exp 8 Review", f"<title>{args.name} Review")
                .replace("review_exp8", stem))
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(len(page_items), "items ->", out_dir)


if __name__ == "__main__":
    main()
