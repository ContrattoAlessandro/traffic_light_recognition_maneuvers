"""Per-light OOD perception of the Exp 6 arms (J, F) on ATLAS test and LISA test.

Same metrics as tools/eval_rfdetr_perception.py (its ``summarize``): AP under center and IoU >= 0.5
matching, recall / precision at the arm's selected threshold (picked on DTLD tune), color accuracy of
matched lights, and arrows. Each detection carries a pictogram, a junction flag and a color; for the arrow
columns a detection is "called relevant" when it is a junction light, and its moves are its pictogram's
(a junction circle has no direction, so an arrow read as a circle counts as wrong).
Writes <run>/ood_perception/epoch<N>_<dataset>.json and prints a J / F table.
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_rfdetr_perception as perc  # noqa: E402
from dtsu import factorized  # noqa: E402
from dtsu.picto_rules import ARROW_MOVES, COLORS, PICTOS, class_name, parse  # noqa: E402
from dtsu.rfdetr_baseline import detect  # noqa: E402
from tools.eval_picto import load_model  # noqa: E402

RUNS = {"J": "runs/exp6_picto_J", "F": "runs/exp6_picto_F"}
# virtual class list shared by both arms: index = picto * 6 + junction * 3 + color
VIRTUAL = [class_name(p, j, c) for p in PICTOS for j in (False, True) for c in COLORS]


def virtual_parse(name):
    p, j, c = parse(name)
    return (set(ARROW_MOVES.get(p, {p})) if j else None), c


perc.parse_class = virtual_parse  # summarize() reads moves / color through this


def predictions(net, items, class_names, res, arm, workers):
    """Per image: (virtual class, score, xyxy pixel boxes) of detections > 0.05 after class-agnostic NMS."""
    table = factorized.attribute_table(class_names)
    out = []
    for x, sizes in DataLoader(perc.Images(items, res), batch_size=8, num_workers=workers):
        x = x.cuda(non_blocking=True)
        if arm == "F":
            score, probs, boxes = factorized.detect(net, x)
            p, c, j = probs["picto"].argmax(-1), probs["color"].argmax(-1), (probs["junction"] > 0.5).long()
        else:
            cls, score, boxes = detect(net, x, class_names)
            p, c, j = table[cls][..., 0], table[cls][..., 1], table[cls][..., 2]
        v = p * 6 + j * 3 + c
        for i, (w, h) in enumerate(sizes):
            k = score[i] > perc.MIN_SCORE
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            s = score[i][k]
            keep = nms(b, s, perc.NMS_IOU) if len(s) else torch.zeros(0, dtype=torch.long)
            out.append((v[i][k][keep], s[keep], b[keep]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["atlas", "lisa"])
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    class_names = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    table = []
    for ds in args.datasets:
        items = perc.LOADERS[ds]()[: args.limit]
        groups = sorted({it["group"] for it in items})
        for arm, run in RUNS.items():
            sel = json.load(open(os.path.join(run, "picto_eval", "result.json")))
            epoch, thr = sel["epoch"], sel["threshold"]
            cache = os.path.join(run, "ood_perception", f"raw_e{epoch}_{ds}.pt")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if args.limit is None and os.path.exists(cache):
                preds = torch.load(cache, weights_only=False)
            else:
                net, res = load_model(run, epoch, class_names, arm)
                preds = predictions(net, items, class_names, res, arm, args.workers)
                del net
                torch.cuda.empty_cache()
                if args.limit is None:
                    torch.save(preds, cache)
            out = {"run": run, "epoch": epoch, "threshold": thr, "dataset": ds}
            for crit in perc.CRITERIA:
                r = out[crit] = {"all": perc.summarize(items, preds, VIRTUAL, thr, crit)}
                for g in groups if len(groups) > 1 else ():
                    k = [i for i, it in enumerate(items) if it["group"] == g]
                    r[g] = perc.summarize([items[i] for i in k], [preds[i] for i in k], VIRTUAL, thr, crit)
            for g, m in out["center"].items():
                a = m["arrows"] or {}
                table.append(f"{ds:6s} {g:12s} {arm}: AP center {m['AP'] or 0:.3f} (IoU {out['iou0.5'][g]['AP'] or 0:.3f}) "
                             f"R {m['recall']:.3f} P {m['precision']:.3f} | color {m['color_acc']:.3f} yellow R "
                             f"{m['color_recall']['yellow'] if m['color_recall']['yellow'] is None else round(m['color_recall']['yellow'], 3)}"
                             + (f" | arrows matched {a['matched']}, junction {a['called_relevant']}, exact "
                                f"{a['moves_acc'] if a['moves_acc'] is None else round(a['moves_acc'], 3)}" if a else ""))
                print(table[-1], flush=True)
            if args.limit is None:
                with open(os.path.join(run, "ood_perception", f"epoch{epoch}_{ds}.json"), "w") as f:
                    json.dump(out, f, indent=1)
    print("\n".join(sorted(table)))


if __name__ == "__main__":
    main()
