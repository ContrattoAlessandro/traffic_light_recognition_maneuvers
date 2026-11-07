"""Exp 6 secondary check (REPLICATION_PLAN.md): arrow directions of J and F on ATLAS test and LISA test.

Each arm runs at its selected epoch and threshold (<run>/picto_eval/result.json), class-agnostic NMS 0.5.
For every annotated arrow with a color, matched (center criterion, as tools/eval_rfdetr_perception.py) to
a kept detection: "exact" = the predicted pictogram's moves equal the arrow's directions (a circle or
"other" has no moves, so it is wrong). Reported over all matched arrows and over those the arm calls
junction-relevant. F - J: bootstrap over images (10,000 draws, seed 0), resampling the same images for
both arms; images are correlated within a sequence, so the CI is too narrow (same deviation as amendment 1).
Writes runs/exp6_picto_F/arrows_vs_J.json.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import factorized  # noqa: E402
from dtsu.picto_rules import ARROW_MOVES, PICTOS  # noqa: E402
from dtsu.rfdetr_baseline import detect  # noqa: E402
from tools.eval_picto import load_model  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS, Images, match  # noqa: E402

RUNS = {"J": "runs/exp6_picto_J", "F": "runs/exp6_picto_F"}


def predictions(net, items, class_names, res, arm, thr, workers):
    """Per image: (picto idx, junction prob, score, xyxy pixel boxes) of kept detections."""
    table = factorized.attribute_table(class_names)
    out = []
    for x, sizes in DataLoader(Images(items, res), batch_size=8, num_workers=workers):
        x = x.cuda(non_blocking=True)
        if arm == "F":
            score, probs, boxes = factorized.detect(net, x)
            picto, junction = probs["picto"].argmax(-1), probs["junction"]
        else:
            cls, score, boxes = detect(net, x, class_names)
            picto, junction = table[cls][..., 0], table[cls][..., 2].float()
        for i, (w, h) in enumerate(sizes):
            k = score[i] > thr
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            s = score[i][k]
            keep = nms(b, s, 0.5) if len(s) else torch.zeros(0, dtype=torch.long)
            out.append((picto[i][k][keep], junction[i][k][keep], s[keep], b[keep]))
    return out


def rows(items, preds):
    """Per annotated arrow: [image, matched, exact, called junction, gt arrow key] (key as sorted directions)."""
    out, keys = [], []
    for i, (it, (p, j, s, b)) in enumerate(zip(items, preds)):
        ignore = torch.tensor([c is None for c in it["color"]], dtype=torch.bool)
        m = match(b, s, it["boxes"], ignore, "center")
        hit = {int(g): q for q, g in enumerate(m.tolist()) if g >= 0}
        for g, arrow in enumerate(it["arrow"]):
            if arrow is None or it["color"][g] is None:
                continue
            key = "_".join(sorted(arrow))
            if g in hit:
                q = hit[g]
                moves = ARROW_MOVES.get(PICTOS[int(p[q])], set())
                out.append([i, 1, int(moves == arrow), int(j[q] > 0.5)])
                keys.append((key, PICTOS[int(p[q])]))
            else:
                out.append([i, 0, 0, 0])
                keys.append((key, None))
    return np.array(out, dtype=np.int64).reshape(-1, 4), keys


def boot_diff(r, n_images, subset, draws=10000, seed=0):
    """Per arm exact rate on its rows passing ``subset``; F - J with the same image resample for both arms."""
    stats = {}
    for a, x in r.items():
        k = subset(x)
        stats[a] = (np.bincount(x[k, 0], x[k, 2], n_images), np.bincount(x[k, 0], minlength=n_images).astype(float))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws // 100):
        w = np.zeros((100, n_images))
        np.add.at(w, (np.arange(100)[:, None], rng.integers(0, n_images, (100, n_images))), 1)
        rate = {a: (w @ e) / np.maximum(w @ n, 1) for a, (e, n) in stats.items()}
        boots.append(rate["F"] - rate["J"])
    boots = np.concatenate(boots)
    out = {a: {"n": int(n.sum()), "exact": float(e.sum() / max(n.sum(), 1))} for a, (e, n) in stats.items()}
    out["F-J"] = out["F"]["exact"] - out["J"]["exact"]
    out["F-J_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["atlas", "lisa"])
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    class_names = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    sel = {a: json.load(open(os.path.join(run, "picto_eval", "result.json"))) for a, run in RUNS.items()}
    out = {a: {"epoch": s["epoch"], "threshold": s["threshold"]} for a, s in sel.items()}
    for ds in args.datasets:
        items = LOADERS[ds]()[: args.limit]
        r, conf = {}, {}
        for a, run in RUNS.items():
            cache = os.path.join(run, "picto_eval", f"arrows_e{sel[a]['epoch']}_{ds}.pt")
            if args.limit is None and os.path.exists(cache):
                preds = torch.load(cache, weights_only=False)
            else:
                net, res = load_model(run, sel[a]["epoch"], class_names, a)
                preds = predictions(net, items, class_names, res, a, sel[a]["threshold"], args.workers)
                del net
                torch.cuda.empty_cache()
                if args.limit is None:
                    torch.save(preds, cache)
            r[a], keys = rows(items, preds)
            conf[a] = {f"{g} -> {p}": n for (g, p), n in sorted(collections.Counter(keys).items(), key=lambda kv: -kv[1])}
        res = {"arrows": int(len(r["J"])),
               "recall": {a: float(x[:, 1].mean()) for a, x in r.items()},
               "matched": boot_diff(r, len(items), lambda x: x[:, 1] == 1),
               "matched_junction": boot_diff(r, len(items), lambda x: (x[:, 1] == 1) & (x[:, 3] == 1)),
               "gt_to_pred": conf}
        if ds == "lisa":
            for grp in ("day", "night"):
                img = np.array([it["group"] == grp for it in items])
                res[f"matched_{grp}"] = boot_diff(r, len(items), lambda x, img=img: (x[:, 1] == 1) & img[x[:, 0]])
        out[ds] = res
        print(ds, json.dumps({k: v for k, v in res.items() if k != "gt_to_pred"}, indent=1), flush=True)
        for a in RUNS:
            print(f"  {a} top confusions:", dict(list(conf[a].items())[:10]), flush=True)
    if args.limit is None:
        with open(os.path.join(RUNS["F"], "arrows_vs_J.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
