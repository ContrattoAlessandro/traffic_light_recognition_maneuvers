"""Crop sheet for tools/exp9_aval_false_go.py: one tile per light that gives REF its red where ARM makes a False-Go.
Each tile: the light (magenta) with context, and what REF and ARM read on it (pictogram, color, junction score).
Usage: exp9_aval_sheet.py [ARM [REF [SET]]] (defaults WF F8 A_val) -> out_path(ARM, REF, SET).jpg"""
import json, os, sys
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou, box_convert
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp8 as e8  # noqa: E402
from tools.exp9_aval_false_go import out_path, sel  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS  # noqa: E402

arm, ref, split = (sys.argv[1:] + ["WF", "F8", "A_val"][len(sys.argv) - 1:])[:3]
rows = [x for x in json.load(open(out_path(arm, ref, split) + ".json"))["rows"] if x["fails"] == arm and "box" in x]
_, abs_recs, _, _ = e8.openlane_set(split)
dets = {m: torch.load(sel(m, split)[0], weights_only=False) for m in (ref, arm)}
seen, tiles = set(), []
T = 256


def read(d, box):
    if not len(d["box"]):
        return "none"
    iou = box_iou(box_convert(box[None], "cxcywh", "xyxy"), box_convert(d["box"], "cxcywh", "xyxy"))[0]
    j = int(iou.argmax())
    if iou[j] < 0.3:
        return "none"
    return f"{PICTOS[int(d['picto'][j])].replace('arrow_', '')} {COLORS[int(d['color'][j])]} j{float(d['junction'][j]):.2f} s{float(d['score'][j]):.2f}"


for x in rows:
    key = (x["frame"], tuple(x["box"]))
    if key in seen:
        continue
    seen.add(key)
    im = Image.open(abs_recs[x["frame"]]["image"]).convert("RGB")
    W, H = im.size
    cx, cy, w, h = x["box"]
    cx, cy, w, h = (cx * W, cy * H, w * W, h * H) if cx <= 1 else (cx, cy, w, h)
    half = max(6 * h, 60)
    crop = im.crop((cx - half, cy - half, cx + half, cy + half)).resize((T, T))
    d = ImageDraw.Draw(crop)
    s = T / (2 * half)
    d.rectangle([(half - w / 2) * s, (half - h / 2) * s, (half + w / 2) * s, (half + h / 2) * s], outline=(255, 0, 255), width=2)
    b = torch.tensor(x["box"])
    tile = Image.new("RGB", (T, T + 48), (20, 20, 20))
    tile.paste(crop, (0, 0))
    t = ImageDraw.Draw(tile)
    t.text((4, T + 2), f"seg {x['segment']} f{x['frame']} {x['direction']}: {x['cause']}", fill=(255, 255, 0))
    t.text((4, T + 17), f"{ref} " + read(dets[ref][x["frame"]], b), fill=(200, 255, 200))
    t.text((4, T + 32), f"{arm} " + read(dets[arm][x["frame"]], b), fill=(255, 200, 200))
    tiles.append(tile)

cols = 6
sheet = Image.new("RGB", (cols * T, ((len(tiles) + cols - 1) // cols) * (T + 48)), (0, 0, 0))
for i, tl in enumerate(tiles):
    sheet.paste(tl, ((i % cols) * T, (i // cols) * (T + 48)))
out = out_path(arm, ref, split) + ".jpg"
sheet.save(out, quality=85)
print(len(tiles), "tiles ->", out)
