"""Score an Exp 6 run (EXPERIMENT_LOG.md): joint (J) or factorized (F) pictogram / junction / color lights on DTLD.

1. Every saved epoch (epoch_<n>_ema.pth) runs on the tune split (valid/records.jsonl); the epoch and score
   threshold (0.05-0.95) with the best rule TSS on tune's scored frames are selected.
2. That epoch runs on DTLD test (dtld_test_records.jsonl) and is scored at that threshold:
   - rule: junction lights -> per-maneuver colors (dtsu.picto_rules.rule) on the scored frames: TSS,
     False-Go and the other direction metrics;
   - per light: class-agnostic IoU >= 0.5 greedy matching (by score) of the kept detections to every
     labelled box; recall, precision, and on matched lights the pictogram (overall and per class),
     color and junction accuracy, junction F1, and all three right.
Detections are kept after the threshold and class-agnostic NMS 0.5. Raw detections are cached in
<run>/picto_eval/; writes <run>/picto_eval/result.json and <run>/picto_eval/test_frames.npz (per-frame and
per-light arrays for tools/compare_picto.py).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, box_iou, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import factorized  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS, parse, rule  # noqa: E402
from dtsu.rfdetr_baseline import detect, load_image  # noqa: E402
from tools.eval_hybrid import per_frame  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MIN_SCORE = 0.05
SIZE = (2048, 1024)


class Images(Dataset):
    def __init__(self, records, res):
        self.records, self.res = records, res

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return load_image(self.records[i]["image"], self.res)


def load_model(run, epoch, class_names, arm):
    import rfdetr

    targs = json.load(open(os.path.join(run, "args.json")))
    net = getattr(rfdetr, targs["variant"])(resolution=targs["resolution"], num_classes=len(class_names), pretrain_weights=None).model.model
    state = torch.load(os.path.join(run, f"epoch_{epoch}_ema.pth"), map_location="cpu", weights_only=False)["model"]
    if arm == "F":
        factorized.factorize(net, factorized.n_picto_of(state))
    net.load_state_dict(state)
    return net.cuda().eval(), targs["resolution"]


def raw_detections(net, records, class_names, res, arm, workers):
    """Per frame: dict of score, picto, color, junction (prob), box (cxcywh normalized) for queries > MIN_SCORE."""
    table = factorized.attribute_table(class_names)
    out = []
    for x in DataLoader(Images(records, res), batch_size=8, num_workers=workers):
        x = x.cuda(non_blocking=True)
        if arm == "F":
            score, probs, boxes = factorized.detect(net, x)
            picto, color, junction = probs["picto"].argmax(-1), probs["color"].argmax(-1), probs["junction"]
        else:
            cls, score, boxes = detect(net, x, class_names)
            attrs = table[cls]
            picto, color, junction = attrs[..., 0], attrs[..., 1], attrs[..., 2].float()
        for i in range(len(x)):
            k = score[i] > MIN_SCORE
            out.append({"score": score[i][k], "picto": picto[i][k], "color": color[i][k], "junction": junction[i][k], "box": boxes[i][k]})
    return out


def kept(d, thr):
    k = d["score"] > thr
    d = {a: v[k] for a, v in d.items()}
    if len(d["score"]):
        keep = nms(box_convert(d["box"], "cxcywh", "xyxy"), d["score"], 0.5)
        d = {a: v[keep] for a, v in d.items()}
    return d


def directions(d):
    lights = [(PICTOS[int(p)], COLORS[int(c)], float(s)) for p, c, j, s in zip(d["picto"], d["color"], d["junction"], d["score"]) if j > 0.5]
    return rule(lights)


def rule_metrics(records, dets, thr):
    idx = [i for i, r in enumerate(records) if r["excluded"] is None]
    preds = [directions(kept(dets[i], thr)) for i in idx]
    return per_frame(preds, [records[i] for i in idx]), idx


def light_rows(records, dets, thr):
    """Per labelled box: [frame, matched, picto ok, color ok, junction ok, all ok, gt picto, gt junction, pred junction]
    plus the number of kept detections per frame (for precision)."""
    rows, n_det = [], []
    for fi, (r, d) in enumerate(zip(records, dets)):
        d = kept(d, thr)
        n_det.append(len(d["score"]))
        if not r["boxes"]:
            continue
        gt = torch.tensor([b["box"] for b in r["boxes"]], dtype=torch.float)
        pred = box_convert(d["box"], "cxcywh", "xyxy") * torch.tensor([*SIZE, *SIZE])
        iou = box_iou(pred, gt) if len(pred) else torch.zeros((0, len(gt)))
        used, match = set(), {}
        for p in torch.argsort(d["score"], descending=True).tolist():
            if not len(gt):
                break
            cand = [(float(iou[p, g]), g) for g in range(len(gt)) if g not in used and iou[p, g] >= 0.5]
            if cand:
                g = max(cand)[1]
                used.add(g)
                match[g] = p
        for g, b in enumerate(r["boxes"]):
            gp, gj, gc = PICTOS.index(b["picto"]), int(b["junction"]), COLORS.index(b["color"])
            if g in match:
                p = match[g]
                pp, pj, pc = int(d["picto"][p]), int(d["junction"][p] > 0.5), int(d["color"][p])
                rows.append([fi, 1, pp == gp, pc == gc, pj == gj, pp == gp and pc == gc and pj == gj, gp, gj, pj])
            else:
                rows.append([fi, 0, 0, 0, 0, 0, gp, gj, -1])
    return np.array(rows, dtype=np.int64), np.array(n_det)


def light_summary(rows, n_det):
    m = rows[rows[:, 1] == 1]
    out = {"gt_lights": int(len(rows)), "detections": int(n_det.sum()), "recall": float(rows[:, 1].mean()),
           "precision": float(len(m) / max(n_det.sum(), 1))}
    for name, col in (("picto_acc", 2), ("color_acc", 3), ("junction_acc", 4), ("all_three_acc", 5)):
        out[name] = float(m[:, col].mean())
    tp = ((m[:, 7] == 1) & (m[:, 8] == 1)).sum()
    out["junction_f1"] = float(2 * tp / max((m[:, 7] == 1).sum() + (m[:, 8] == 1).sum(), 1))
    out["picto_recall_by_class"] = {PICTOS[c]: {"n": int((m[:, 6] == c).sum()), "acc": float(m[m[:, 6] == c, 2].mean())}
                                    for c in range(len(PICTOS)) if (m[:, 6] == c).any()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", choices=("J", "F"), required=True)
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N frames per set (smoke tests, no cache)")
    args = ap.parse_args()

    class_names = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    sets = {"tune": os.path.join(args.dataset_dir, "valid", "records.jsonl"),
            "test": os.path.join(args.dataset_dir, "dtld_test_records.jsonl")}
    recs = {k: [json.loads(l) for l in open(p)][: args.limit] for k, p in sets.items()}
    cache = os.path.join(args.run, "picto_eval")
    os.makedirs(cache, exist_ok=True)

    def get(epoch, k):
        path = os.path.join(cache, f"e{epoch}_{k}.pt")
        if args.limit is None and os.path.exists(path):
            return torch.load(path, weights_only=False)
        net, res = load_model(args.run, epoch, class_names, args.arm)
        d = raw_detections(net, recs[k], class_names, res, args.arm, args.workers)
        del net
        torch.cuda.empty_cache()
        if args.limit is None:
            torch.save(d, path)
        return d

    epochs = sorted(int(re.search(r"epoch_(\d+)_ema", p).group(1)) for p in glob.glob(os.path.join(args.run, "epoch_*_ema.pth")))
    grid = {}
    for e in epochs:
        d = get(e, "tune")
        grid[e] = {float(t): rule_metrics(recs["tune"], d, t)[0][3]["tss"] for t in THRESHOLDS}
        best_t = max(grid[e], key=grid[e].get)
        print(f"epoch {e}: tune TSS {grid[e][best_t]:.4f} @ {best_t}", flush=True)
    epoch, thr = max(((e, t) for e in grid for t in grid[e]), key=lambda et: (grid[et[0]][et[1]], -et[0]))
    print(f"selected epoch {epoch}, threshold {thr} (tune TSS {grid[epoch][thr]:.4f})", flush=True)

    d = get(epoch, "test")
    (s, fg_num, fg_den, m), idx = rule_metrics(recs["test"], d, thr)
    rows, n_det = light_rows(recs["test"], d, thr)
    lights = light_summary(rows, n_det)
    out = {"run": args.run, "arm": args.arm, "epoch": epoch, "threshold": thr, "tune_tss": grid[epoch][thr],
           "tune_grid": {str(e): g for e, g in grid.items()},
           "test": {"frames": len(recs["test"]), "scored_frames": len(idx), "rule": m, "lights": lights}}
    print(json.dumps(out["test"], indent=1), flush=True)
    if args.limit is None:
        with open(os.path.join(cache, "result.json"), "w") as f:
            json.dump(out, f, indent=1)
        np.savez(os.path.join(cache, "test_frames.npz"), frame=np.array(idx), tss=s, fg_num=fg_num, fg_den=fg_den,
                 session=np.array([recs["test"][i]["segment_id"] for i in idx]), light_rows=rows, n_det=n_det,
                 frame_session=np.array([r["segment_id"] for r in recs["test"]]))


if __name__ == "__main__":
    main()
