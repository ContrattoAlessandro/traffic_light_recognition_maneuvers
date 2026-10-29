"""Build the presentation deck: 100 val frames, ground truth vs the Swin and RF-DETR predictions.

Picks --per-subset frames at random (seeded) from A val and B val and reads both runs' per-frame
predictions from <run>/val_preds.npz (same frame order as the label files). Writes
<out>/index.html (from tools/boss_deck_template.html, data inlined), <out>/img/<id>.jpg (the frame
with the annotated lights outlined: relevant ones in their signal color, others thin grey) and
<out>/crops/<id>.jpg (zoomed tiles of the relevant lights).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import COLORS, DIRECTIONS, PASS  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
SETS = (("A", "data/labels/subset_A_val.jsonl", "data/OpenLane-V2/subset_A"),
        ("B", "data/labels/subset_B_val.jsonl", "data/OpenLane-V2/subset_B"))
OUTLINE = {"red": (255, 59, 48), "yellow": (255, 204, 0), "green": (52, 199, 89)}
TILE_W, TILE_H, MAX_TILES = 110, 180, 6
GROUPS = ("both", "rfdetr", "swin", "neither")  # who got every direction right


def draw_lights(img, boxes, s):
    d = ImageDraw.Draw(img)
    for b in sorted(boxes, key=lambda b: b["relevant"]):  # relevant ones on top
        x1, y1, x2, y2 = (v * s for v in b["box"])
        if b["relevant"]:
            d.rectangle((x1 - 5, y1 - 5, x2 + 5, y2 + 5), outline=OUTLINE.get(b["color"], (230, 230, 230)), width=3)
        else:
            d.rectangle((x1 - 3, y1 - 3, x2 + 3, y2 + 3), outline=(170, 170, 170), width=1)


def crop_tiles(img, boxes):
    tiles = Image.new("RGB", (TILE_W * max(len(boxes), 1), TILE_H), (24, 26, 30))
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        m = max(x2 - x1, y2 - y1) * 0.6 + 6
        c = img.crop((max(0, x1 - m), max(0, y1 - m), min(img.width, x2 + m), min(img.height, y2 + m)))
        f = min(TILE_W / c.width, TILE_H / c.height)
        c = c.resize((max(1, round(c.width * f)), max(1, round(c.height * f))), Image.LANCZOS)
        tiles.paste(c, (k * TILE_W + (TILE_W - c.width) // 2, (TILE_H - c.height) // 2))
    return tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--swin", default="runs/swin_t_AB")
    ap.add_argument("--rfdetr", default="runs/rfdetr_m_AB")
    ap.add_argument("--bootstrap", default="runs/bootstrap_swin_vs_rfdetr.json")
    ap.add_argument("--out", default="runs/boss_deck")
    ap.add_argument("--per-subset", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-long", type=int, default=1100)
    args = ap.parse_args()

    preds = {m: np.load(os.path.join(run, "val_preds.npz")) for m, run in (("swin", args.swin), ("rfdetr", args.rfdetr))}
    for sub in ("img", "crops"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    frames, picked_preds = [], {"swin": [], "rfdetr": [], "gt": []}
    for name, labels, root in SETS:
        recs = load_records(labels, root)
        for i in sorted(random.Random(args.seed).sample(range(len(recs)), args.per_subset)):
            rec = recs[i]
            gc, gp = preds["swin"][f"{name}_gc"][i], preds["swin"][f"{name}_gp"][i]
            assert [COLORS[c] for c in gc] == [rec["color"][d] for d in DIRECTIONS]
            out = {m: (preds[m][f"{name}_pc"][i], preds[m][f"{name}_pp"][i]) for m in preds}
            ok = {m: bool(((pc == gc) & (pp == gp)).all()) for m, (pc, pp) in out.items()}
            group = {(True, True): "both", (False, True): "rfdetr", (True, False): "swin",
                     (False, False): "neither"}[ok["swin"], ok["rfdetr"]]
            for m in out:
                picked_preds[m].append(out[m])
            picked_preds["gt"].append((gc, gp))

            fid = f"{name}-{i:04d}"
            img = Image.open(os.path.join(root, rec["image"])).convert("RGB")
            relevant = [b for b in rec["boxes"] if b["relevant"]][:MAX_TILES]
            crop_tiles(img, relevant).save(os.path.join(args.out, "crops", fid + ".jpg"), quality=88)
            s = args.img_long / max(img.size)
            small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            draw_lights(small, rec["boxes"], s)
            small.save(os.path.join(args.out, "img", fid + ".jpg"), quality=80)

            frames.append({
                "id": fid, "subset": name, "group": group, "size": list(small.size),
                "n_tiles": len(relevant), "n_lights": len(rec["boxes"]),
                "truth": [[COLORS[c], PASS[p]] for c, p in zip(gc, gp)],
                **{m: [[COLORS[c], PASS[p]] for c, p in zip(*out[m])] for m in out},
            })
        print(f"{name}: {args.per_subset} frames")

    frames.sort(key=lambda f: (GROUPS.index(f["group"]), f["id"]))
    gt = [np.stack(x) for x in zip(*picked_preds["gt"])]
    sample = {m: evaluate(*[np.stack(x) for x in zip(*picked_preds[m])], *gt) for m in ("swin", "rfdetr")}
    meta = {"directions": DIRECTIONS, "tile": [TILE_W, TILE_H], "sample": sample,
            "groups": {g: sum(f["group"] == g for f in frames) for g in GROUPS},
            "bootstrap": json.load(open(args.bootstrap))}
    page = open(os.path.join(TOOLS, "boss_deck_template.html"), encoding="utf8").read()
    data = json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf8") as f:
        f.write(page.replace("__DECK_DATA__", data))
    print(f"wrote {args.out}/index.html  groups {meta['groups']}  "
          f"sample TSS swin {sample['swin']['tss']:.4f} rfdetr {sample['rfdetr']['tss']:.4f}")


if __name__ == "__main__":
    main()
