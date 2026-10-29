"""Build the manual-inspection page: N val frames with model predictions vs ground truth.

Picks frames at random (seeded) from A val and B val, runs the checkpoint on them and writes
<out>/index.html (from tools/inspect_template.html, data inlined), <out>/img/<id>.jpg (frame,
long side --img-long) and <out>/crops/<id>.jpg (a strip of zoomed crops, one tile per
annotated light, relevant lights first).
"""
import argparse
import json
import os
import random
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import COLORS, DIRECTIONS, PASS  # noqa: E402
from dtsu.data import LIGHT_CLASSES, DTSUDataset, collate, load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.model import DirectionAwareModel  # noqa: E402

TILE_W, TILE_H = 72, 120
TOOLS = os.path.dirname(os.path.abspath(__file__))


def crop_strip(img, boxes):
    """One tile per box: the box with margin, scaled to fit TILE_W x TILE_H."""
    strip = Image.new("RGB", (TILE_W * max(len(boxes), 1), TILE_H), (24, 26, 30))
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        w, h = x2 - x1, y2 - y1
        m = max(w, h) * 0.6 + 6
        c = img.crop((max(0, x1 - m), max(0, y1 - m), min(img.width, x2 + m), min(img.height, y2 + m)))
        s = min(TILE_W / c.width, TILE_H / c.height)
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
        strip.paste(c, (k * TILE_W + (TILE_W - c.width) // 2, (TILE_H - c.height) // 2))
    return strip


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/swin_t_AB")
    ap.add_argument("--out", default="runs/swin_t_AB/inspect")
    ap.add_argument("--per-subset", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-long", type=int, default=1280)
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    targs = json.load(open(os.path.join(args.run, "args.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionAwareModel((targs["long_side"],) * 2, pretrained=False,
                                backbone=targs.get("backbone", "swin_tiny_patch4_window7_224"),
                                top_k=targs["top_k"], n_queries=targs["n_queries"]).to(dev).eval()
    ckpt = torch.load(os.path.join(args.run, "last.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    frames = []
    for name, labels, root in targs["val_set"]:
        recs = load_records(labels, root)
        picked = sorted(random.Random(args.seed).sample(range(len(recs)), args.per_subset))
        ds = DTSUDataset([recs[i] for i in picked], targs["long_side"])
        for k in range(0, len(picked), 4):
            idx = list(range(k, min(k + 4, len(picked))))
            x, _ = collate([ds[i] for i in idx])
            with torch.no_grad(), torch.autocast(dev.type, enabled=targs["amp"]):
                out = model(x.to(dev))
            pcol = out["color_logits"].float().softmax(-1).cpu()
            ppas = out["pass_logits"].float().softmax(-1).cpu()
            det_p = out["det_logits"][-1].float().softmax(-1).cpu()
            det_b = out["det_boxes"][-1].float().cpu()
            for j, i in enumerate(idx):
                rec = ds.records[i]
                fid = f"{name}-{k + j:02d}"
                img = Image.open(os.path.join(rec["root"], rec["image"])).convert("RGB")
                scale, (H, W) = ds.canvas[i]
                s = args.img_long / max(img.size)
                img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).save(
                    os.path.join(args.out, "img", fid + ".jpg"), quality=82)
                gt_boxes = sorted(rec["boxes"], key=lambda b: not b["relevant"])[: args.max_crops]
                crop_strip(img, gt_boxes).save(os.path.join(args.out, "crops", fid + ".jpg"), quality=90)

                fg = det_p[j, :, :-1]
                score, cls = fg.max(-1)
                order = score.argsort(descending=True)[: targs["top_k"]]
                dets = []
                for q in order.tolist():  # canvas-normalized cxcywh -> image-normalized xyxy
                    cx, cy, w, h = det_b[j, q].tolist()
                    x1, y1 = (cx - w / 2) * W / scale / img.width, (cy - h / 2) * H / scale / img.height
                    x2, y2 = (cx + w / 2) * W / scale / img.width, (cy + h / 2) * H / scale / img.height
                    dets.append({"box": [round(v, 5) for v in (x1, y1, x2, y2)],
                                 "color": LIGHT_CLASSES[cls[q]], "score": round(score[q].item(), 3)})
                dirs = {}
                for d_i, d in enumerate(DIRECTIONS):
                    pc, pp = pcol[j, d_i], ppas[j, d_i]
                    dirs[d] = {"gt_color": rec["color"][d], "gt_pass": rec["pass"][d],
                               "pred_color": COLORS[pc.argmax()], "pred_pass": PASS[pp.argmax()],
                               "color_probs": [round(v, 3) for v in pc.tolist()],
                               "pass_prob": round(pp.max().item(), 3)}
                frames.append({
                    "id": fid, "subset": name, "segment": rec["segment_id"], "timestamp": rec["timestamp"],
                    "image": rec["image"], "size": [img.width, img.height], "crossing": rec["crossing"],
                    "directions": dirs,
                    "gt_boxes": [{"box": [round(b["box"][0] / img.width, 5), round(b["box"][1] / img.height, 5),
                                          round(b["box"][2] / img.width, 5), round(b["box"][3] / img.height, 5)],
                                  "color": b["color"], "relevant": b["relevant"], "moves": b["moves"]}
                                 for b in gt_boxes],
                    "dets": dets,
                })
        print(f"{name}: {len(picked)} frames")

    idx = lambda key: [[{"color": COLORS, "pass": PASS}[key[0]].index(f["directions"][d][key[1]]) for d in DIRECTIONS]
                       for f in frames]
    sample = evaluate(idx(("color", "pred_color")), idx(("pass", "pred_pass")),
                      idx(("color", "gt_color")), idx(("pass", "gt_pass")))
    meta = {"epoch": ckpt["epoch"], "run": args.run, "seed": args.seed, "per_subset": args.per_subset,
            "tile": [TILE_W, TILE_H], "sample_metrics": sample,
            "val_metrics": json.load(open(os.path.join(args.run, "final_metrics.json")))}
    page = open(os.path.join(TOOLS, "inspect_template.html"), encoding="utf8").read()
    data = json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf8") as f:
        f.write(page.replace("__INSPECT_DATA__", data))
    print(f"wrote {args.out}/index.html  sample TSS {sample['tss']:.4f}  EM {sample['em']:.4f}")


if __name__ == "__main__":
    main()
