"""Contact sheets of tricky label cases: full view + zoom on the traffic lights.

Each tile shows the four maneuver labels; relevant lights get a thick box tagged with
the movements their lanes lead into (S/L/R/U, G = generic), ignored lights a thin box.
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

BOX_RGB = {"red": (255, 40, 40), "green": (40, 230, 40), "yellow": (255, 200, 0)}
MOVE_ABBR = {"straight": "S", "left": "L", "right": "R", "uturn": "U", "generic": "G"}

CASES = {
    "turn_arrow": lambda r: r["kept"] and r["color"]["straight"] not in ("none", r["color"]["left"]),
    "straight_none": lambda r: r["kept"] and r["color"]["straight"] == "none",
    "dedicated_right": lambda r: r["kept"] and r["color"]["right"] != "none",
    "crossing": lambda r: r["kept"] and r.get("crossing"),
    "conflict": lambda r: bool(r["conflict"]),
}


def tile(rec, tile_w=520):
    img = Image.open(os.path.join(rec["root"], rec["image"])).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in rec["boxes"]:
        draw.rectangle(b["box"], outline=BOX_RGB[b["color"]], width=6 if b["relevant"] else 2)
    rel = [b["box"] for b in rec["boxes"] if b["relevant"]] or [b["box"] for b in rec["boxes"]]

    # zoom on the union of (relevant) light boxes, padded, at least 1/3 of the image wide
    x1, y1 = min(b[0] for b in rel), min(b[1] for b in rel)
    x2, y2 = max(b[2] for b in rel), max(b[3] for b in rel)
    cw = max(x2 - x1 + 160, img.width / 3)
    ch = cw * 0.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_box = [max(0, cx - cw / 2), max(0, cy - ch / 2), min(img.width, cx + cw / 2), min(img.height, cy + ch / 2)]
    zoom = img.crop([round(v) for v in crop_box])
    zoom = zoom.resize((tile_w, round(zoom.height * tile_w / zoom.width)))
    zd = ImageDraw.Draw(zoom)
    s = tile_w / (crop_box[2] - crop_box[0])
    for b in rec["boxes"]:
        if b["relevant"]:
            tag = "".join(MOVE_ABBR[m] for m in b["moves"])
            zd.text(((b["box"][0] - crop_box[0]) * s, (b["box"][3] - crop_box[1]) * s + 2), tag, fill=(255, 255, 0))

    full = img.resize((tile_w, round(img.height * tile_w / img.width)))
    head = Image.new("RGB", (tile_w, 30), (0, 0, 0))
    hd = ImageDraw.Draw(head)
    c = rec["color"]
    hd.text((4, 2), f"S:{c['straight']} L:{c['left']} R:{c['right']} U:{c['uturn']}"
            + (f"  CONFLICT {rec['conflict']}" if rec["conflict"] else "") + ("  [crossing]" if rec.get("crossing") else ""),
            fill=(255, 255, 255))
    hd.text((4, 16), f"{rec['subset']} {rec['segment_id']}/{rec['timestamp']}", fill=(170, 170, 170))
    out = Image.new("RGB", (tile_w, head.height + zoom.height + full.height), (0, 0, 0))
    out.paste(head, (0, 0)); out.paste(zoom, (0, head.height)); out.paste(full, (0, head.height + zoom.height))
    return out


def sheet(tiles, cols=3):
    rows = (len(tiles) + cols - 1) // cols
    w = tiles[0].width
    hs = [max(t.height for t in tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    out = Image.new("RGB", (w * cols, sum(hs)), (40, 40, 40))
    y = 0
    for r in range(rows):
        for k, t in enumerate(tiles[r * cols:(r + 1) * cols]):
            out.paste(t, (k * w, y))
        y += hs[r]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", nargs=3, action="append", required=True, metavar=("NAME", "LABELS", "ROOT"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-set", type=int, default=3, help="frames per case and per set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = []
    for name, labels, root in args.set:
        for l in open(labels):
            r = json.loads(l)
            r["root"], r["subset"] = root, name
            recs.append(r)
    rng = random.Random(args.seed)
    for case, pred in CASES.items():
        chosen = []
        for name, _, _ in args.set:
            pool = [r for r in recs if r["subset"] == name and pred(r) and r["boxes"]]
            chosen += rng.sample(pool, min(args.per_set, len(pool)))
        if chosen:
            path = os.path.join(args.out_dir, f"{case}.jpg")
            sheet([tile(r) for r in chosen]).save(path, quality=88)
            print(f"{case}: {len(chosen)} frames -> {path}")


if __name__ == "__main__":
    main()
