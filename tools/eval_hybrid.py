"""J vs J+P on the per-direction matrix (REPLICATION_PLAN.md, amendment 1).

For each J run (its epoch and score threshold: best source-tune TSS in its val_log.jsonl):
1. detect on its source tune split(s) and on A val, B val and D*_US test (squeeze, as in training);
2. after threshold and class-agnostic NMS, crop every detection J calls relevant from the source image
   and classify it with P (given J's color);
3. J+P decoding: a detection whose P top class is an arrow with probability >= tau takes the arrow's
   directions as moves; every other detection keeps J's. tau in TAUS is picked on the source tune split
   by TSS ("off" = J, so J is nested); ties go to the most conservative tau (largest, then off);
4. per test set: J and J+P metrics, and the paired J+P - J difference of TSS and False-Go with a
   segment-cluster bootstrap 95% CI (10,000 draws, seed 0).
Detections and P probabilities are cached in <run>/hybrid/. Writes <run>/hybrid/matrix.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS, PASS  # noqa: E402
from dtsu import pictogram as P  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, parse_class, to_arrays  # noqa: E402
from tools.eval_rfdetr import load_model, run_detector  # noqa: E402

L, R = "data/labels", "data/OpenLane-V2"
DTLD = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD_jpg")
SETS = {
    "A_tune": (f"{L}/tune/subset_A_train_tune.jsonl", f"{R}/subset_A"),
    "B_tune": (f"{L}/tune/subset_B_train_tune.jsonl", f"{R}/subset_B"),
    "DTLD_tune": (f"{L}/tune/dtld_train_tune.jsonl", f"{DTLD}/train"),
    "A_val": (f"{L}/subset_A_val.jsonl", f"{R}/subset_A"),
    "B_val": (f"{L}/subset_B_val.jsonl", f"{R}/subset_B"),
    "DTLD_US_test": (f"{L}/dtld_us_test.jsonl", f"{DTLD}/test"),
}
RUNS = {  # name: (run dir, class names, source tune sets)
    "A": ("runs/rfdetr_m_A", "data/rfdetr_coco_fit_A/class_names.json", ["A_tune"]),
    "B": ("runs/rfdetr_m_B", "data/rfdetr_coco_fit_B/class_names.json", ["B_tune"]),
    "A+B": ("runs/tune_rfdetr/r1_m_default", "data/rfdetr_coco_fit/class_names.json", ["A_tune", "B_tune"]),
    "DTLD": ("runs/rfdetr_m_dtld", "data/rfdetr_coco_dtld/class_names.json", ["DTLD_tune"]),
}
TESTS = ["A_val", "B_val", "DTLD_US_test"]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, None]  # None = off (J)
NMS_IOU = 0.5
NO_RIGHT = [DIRECTIONS.index(d) for d in DIRECTIONS if d != "right"]


def best_epoch(run):
    log = [json.loads(l) for l in open(os.path.join(run, "val_log.jsonl"))]
    e = max(log, key=lambda x: x["tune_tss"])
    return e["epoch"], e["threshold"]


def kept(dets, class_names, threshold):
    """One frame: thresholded + NMS'd detections J calls relevant -> list of (moves, color, score, cxcywh box)."""
    c, s, b = dets
    k = s > threshold
    c, s, b = c[k], s[k], b[k]
    if len(s):
        keep = nms(box_convert(b, "cxcywh", "xyxy"), s, NMS_IOU)
        c, s, b = c[keep], s[keep], b[keep]
    out = []
    for ci, si, bi in zip(c.tolist(), s.tolist(), b.tolist()):
        moves, color = parse_class(class_names[ci])
        if moves is not None:
            out.append((set(moves), color, si, bi))
    return out


class Crops(Dataset):
    def __init__(self, records, frames):
        self.items = [(os.path.join(r["root"], r["image"]), [d[3] for d in f]) for r, f in zip(records, frames) if f]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        return np.stack([P.store_crop(img, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
                         for cx, cy, bw, bh in boxes])


def pictogram_probs(net, records, frames, workers):
    """Per frame: [n_kept, n_classes] P probabilities (same order as ``frames``)."""
    crops = [c for c in DataLoader(Crops(records, frames), batch_size=None, num_workers=workers)]
    flat = np.concatenate(crops) if crops else np.zeros((0, P.STORE, P.STORE, 3), np.uint8)
    colors = [d[1] for f in frames for d in f]
    probs = P.classify(net, flat, colors)
    out, i = [], 0
    for f in frames:
        out.append(probs[i:i + len(f)])
        i += len(f)
    return out


def decode(rec, frame, probs, tau):
    dets = []
    for (moves, color, score, _), p in zip(frame, probs):
        top = int(np.argmax(p))
        name = P.CLASSES[top]
        if tau is not None and name in P.ARROW_MOVES and p[top] >= tau:
            moves = P.ARROW_MOVES[name]
        dets.append((moves, color, score))
    return assign_scored(dets, rec["traffic_side"], rec.get("near_turn", "dedicated"))


def per_frame(preds, records, cols=None):
    """Per-frame TSS contribution and False-Go numerator / denominator (decomposes the metrics)."""
    pc, pp = to_arrays(preds)
    gc, gp = to_arrays([r["color"] for r in records])
    if cols is not None:
        pc, pp, gc, gp = pc[:, cols], pp[:, cols], gc[:, cols], gp[:, cols]
    stop, go = PASS.index("stop"), PASS.index("go")
    c_ok, p_ok = pc == gc, pp == gp
    s = (c_ok.mean(1) + p_ok.mean(1) + (c_ok & p_ok).all(1)) / 3
    return s, ((gp == stop) & (pp == go)).sum(1), (gp == stop).sum(1), evaluate(pc, pp, gc, gp)


def paired_ci(records, j, jp, draws=10000, seed=0):
    segs = np.array([r["segment_id"] for r in records])
    uniq, inv = np.unique(segs, return_inverse=True)
    sums = lambda a: np.bincount(inv, weights=a, minlength=len(uniq))  # noqa: E731
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    ds, fj, fjp, den = sums(jp[0] - j[0]), sums(j[1]), sums(jp[1]), sums(j[2])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (draws, len(uniq)))
    w = np.zeros((draws, len(uniq)))
    np.add.at(w, (np.arange(draws)[:, None], idx), 1)
    d_tss = (w @ ds) / (w @ n)
    d_fg = (w @ fjp) / np.maximum(w @ den, 1) - (w @ fj) / np.maximum(w @ den, 1)
    point_fg = fjp.sum() / max(den.sum(), 1) - fj.sum() / max(den.sum(), 1)
    return {"d_tss": float(ds.sum() / n.sum()), "d_tss_ci95": [float(np.percentile(d_tss, 2.5)), float(np.percentile(d_tss, 97.5))],
            "d_false_go": float(point_fg), "d_false_go_ci95": [float(np.percentile(d_fg, 2.5)), float(np.percentile(d_fg, 97.5))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", choices=sorted(RUNS), default=list(RUNS))
    ap.add_argument("--pictogram", default="runs/pictogram_p/best.pth")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N frames per set (smoke tests, no cache)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    net = P.load(args.pictogram)
    for name in args.runs:
        run, cn_path, tune_sets = RUNS[name]
        class_names = json.load(open(cn_path))
        epoch, thr = best_epoch(run)
        cache_dir = os.path.join(run, "hybrid")
        os.makedirs(cache_dir, exist_ok=True)
        model = None
        recs, frames, probs = {}, {}, {}
        for k in tune_sets + TESTS:
            recs[k] = load_records(*SETS[k])[: args.limit]
            cache = os.path.join(cache_dir, f"e{epoch}_{k}.pt")
            if args.limit is None and os.path.exists(cache):
                frames[k], probs[k] = torch.load(cache, weights_only=False)
                continue
            if model is None:
                model, res, _ = load_model(run, os.path.join(run, f"epoch_{epoch}_ema.pth"), len(class_names))
            dets = run_detector(model, recs[k], class_names, res, args.workers)
            frames[k] = [kept(d, class_names, thr) for d in dets]
            probs[k] = pictogram_probs(net, recs[k], frames[k], args.workers)
            if args.limit is None:
                torch.save((frames[k], probs[k]), cache)
            print(f"[{name}] {k}: {len(recs[k])} frames, {sum(map(len, frames[k]))} relevant detections", flush=True)
        del model
        torch.cuda.empty_cache()

        tune_recs = [r for k in tune_sets for r in recs[k]]
        tune_tss = {}
        for tau in TAUS:
            preds = [decode(r, f, p, tau) for k in tune_sets for r, f, p in zip(recs[k], frames[k], probs[k])]
            tune_tss[str(tau)] = per_frame(preds, tune_recs)[3]["tss"]
        best = max(tune_tss.values())  # ties -> the most conservative tau (largest, then off)
        tau = [t for t in TAUS if tune_tss[str(t)] == best][-1]
        out = {"run": run, "epoch": epoch, "threshold": thr, "tau": tau, "tune_tss_by_tau": tune_tss, "tests": {}}
        print(f"[{name}] epoch {epoch} thr {thr} | tune TSS by tau {tune_tss} -> tau {tau}", flush=True)
        for k in TESTS:
            pj = [decode(r, f, p, None) for r, f, p in zip(recs[k], frames[k], probs[k])]
            pjp = [decode(r, f, p, tau) for r, f, p in zip(recs[k], frames[k], probs[k])]
            res_k = {"frames": len(recs[k]), "frames_changed": int(sum(a != b for a, b in zip(pj, pjp))),
                     "relevant_detections": sum(map(len, frames[k])),
                     "overridden": int(sum(tau is not None and P.CLASSES[int(np.argmax(p))] in P.ARROW_MOVES and p.max() >= tau
                                           for ps in probs[k] for p in ps))}
            for label, cols in (("overall", None), ("no_right", NO_RIGHT)):
                j, jp = per_frame(pj, recs[k], cols), per_frame(pjp, recs[k], cols)
                res_k[label] = {"J": j[3], "J+P": jp[3], **paired_ci(recs[k], j, jp)}
            out["tests"][k] = res_k
            o = res_k["overall"]
            print(f"[{name}] {k}: TSS J {o['J']['tss']:.4f} J+P {o['J+P']['tss']:.4f} d {o['d_tss']:+.4f} "
                  f"[{o['d_tss_ci95'][0]:+.4f}, {o['d_tss_ci95'][1]:+.4f}] | F-Go d {o['d_false_go']:+.4f} "
                  f"| frames changed {res_k['frames_changed']}, overridden {res_k['overridden']}/{res_k['relevant_detections']}", flush=True)
        if args.limit is None:
            with open(os.path.join(cache_dir, "matrix.json"), "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
