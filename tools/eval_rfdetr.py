"""Score RF-DETR epochs with the paper's metrics: detections -> direction rules -> TSS etc.

For every ``epoch_<n>_ema.pth`` in the run (or only ``--epochs``):
1. detect on the tune split (a held-out part of train) and on val A / val B;
2. pick the score threshold that maximizes combined tune TSS (val is never used for this);
3. write val metrics at that threshold to ``<run>/val_log.jsonl``, in the format of the
   direction-aware model's log, plus val TSS over the whole threshold grid (diagnostic only).
``--dataset dtld`` picks the threshold on the DTLD tune split and reports DTLD test.
With ``--tune-only`` (tuning runs trained on the fit split) only the tune split is loaded and scored,
and the log is ``<run>/tune_log.jsonl``.
``--per-direction`` also writes ``<run>/per_direction.json`` for the last scored epoch, as
``tools/eval_directions.py`` does for the direction-aware model, and its per-frame predictions
to ``<run>/val_preds.npz`` (for ``tools/bootstrap_compare.py``).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import detect, frame_directions, load_image, to_arrays  # noqa: E402
from tools.eval_directions import direction_stats, save_preds  # noqa: E402

L, R = "data/labels", "data/OpenLane-V2"
DTLD = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD_jpg")
# per dataset: (tune sets that pick the threshold, sets reported)
PRESETS = {
    "openlane": ({"A_tune": (f"{L}/tune/subset_A_train_tune.jsonl", f"{R}/subset_A"),
                  "B_tune": (f"{L}/tune/subset_B_train_tune.jsonl", f"{R}/subset_B")},
                 {"A": (f"{L}/subset_A_val.jsonl", f"{R}/subset_A"),
                  "B": (f"{L}/subset_B_val.jsonl", f"{R}/subset_B")}),
    "dtld": ({"DTLD_tune": (f"{L}/tune/dtld_train_tune.jsonl", f"{DTLD}/train")},
             {"DTLD_test": (f"{L}/dtld_test.jsonl", f"{DTLD}/test")}),
}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MIN_SCORE = 0.05  # detections kept per frame for the threshold sweep


class Images(Dataset):
    def __init__(self, records, resolution):
        self.records, self.resolution = records, resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return load_image(os.path.join(r["root"], r["image"]), self.resolution)


def run_detector(model, records, class_names, resolution, workers):
    dl = DataLoader(Images(records, resolution), batch_size=8, num_workers=workers)
    dets = []
    for x in dl:
        cls, score, boxes = detect(model, x.cuda(non_blocking=True), class_names)
        for c, s, b in zip(cls, score, boxes):
            k = s > MIN_SCORE
            dets.append((c[k], s[k], b[k]))
    return dets


def score(records, dets, class_names, threshold):
    pred = [frame_directions(*d, class_names, threshold, r["traffic_side"], r.get("near_turn", "dedicated"))
            for r, d in zip(records, dets)]
    pc, pp = to_arrays(pred)
    gc, gp = to_arrays([r["color"] for r in records])
    return pc, pp, gc, gp


def load_model(run, ckpt_path, num_classes):
    import rfdetr

    targs = json.load(open(os.path.join(run, "args.json")))
    wrapper = getattr(rfdetr, targs["variant"])(resolution=targs["resolution"], num_classes=num_classes,
                                                pretrain_weights=None)
    net = wrapper.model.model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    return net.cuda().eval(), targs["resolution"], ckpt["epoch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/rfdetr_m_AB")
    ap.add_argument("--class-names", default="data/rfdetr_coco/class_names.json")
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="openlane")
    ap.add_argument("--epochs", type=int, nargs="*", help="epochs to score (default: all not yet in val_log.jsonl)")
    ap.add_argument("--per-direction", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N frames per set (smoke tests)")
    ap.add_argument("--tune-only", action="store_true",
                    help="tuning runs: score the tune split only (val is never loaded), log to tune_log.jsonl")
    ap.add_argument("--threshold", type=float,
                    help="fixed score threshold (from tuning); needed when the run trained on the tune split")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    class_names = json.load(open(args.class_names))
    tune_sets, report_sets = PRESETS[args.dataset]
    sets = {**tune_sets, **report_sets}
    report = tuple(tune_sets) if args.tune_only else tuple(report_sets)
    pick_on = () if args.threshold is not None else tuple(tune_sets)
    recs = {k: load_records(*v)[: args.limit] for k, v in sets.items() if k in pick_on + report}
    log_path = os.path.join(args.run, "tune_log.jsonl" if args.tune_only else "val_log.jsonl")
    done = set()
    if os.path.exists(log_path):
        done = {json.loads(l)["epoch"] for l in open(log_path)}

    ckpts = {int(re.search(r"epoch_(\d+)_ema", p).group(1)): p
             for p in glob.glob(os.path.join(args.run, "epoch_*_ema.pth"))}
    todo = sorted(e for e in ckpts if (e in args.epochs if args.epochs else e not in done))
    for epoch in todo:
        model, resolution, _ = load_model(args.run, ckpts[epoch], len(class_names))
        dets = {k: run_detector(model, r, class_names, resolution, args.workers) for k, r in recs.items()}

        if args.threshold is None:
            tune_recs = [r for k in pick_on for r in recs[k]]
            tune_dets = [d for k in pick_on for d in dets[k]]
            tune_tss = [evaluate(*score(tune_recs, tune_dets, class_names, t))["tss"] for t in THRESHOLDS]
            thr, best_tune = float(THRESHOLDS[int(np.argmax(tune_tss))]), max(tune_tss)
        else:
            thr, best_tune = args.threshold, None

        preds = {k: score(recs[k], dets[k], class_names, thr) for k in report}
        preds["combined"] = [np.concatenate(x) for x in zip(*(preds[k] for k in report))]
        entry = {"epoch": epoch, "threshold": thr, "tune_tss": best_tune}
        entry.update({k: evaluate(*p) for k, p in preds.items()})
        entry["tss_by_threshold"] = {k: {str(t): evaluate(*score(recs[k], dets[k], class_names, t))["tss"]
                                         for t in THRESHOLDS} for k in report}
        if epoch not in done:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        print(f"epoch {epoch} thr {thr:.2f} " + (f"tune {best_tune:.4f} " if best_tune is not None else "fixed ") + "| " +
              " | ".join(f"{k} TSS {entry[k]['tss']:.4f} F-Go {entry[k]['false_go']:.4f} F-Stop {entry[k]['false_stop']:.4f}"
                         for k in report + ("combined",)), flush=True)

        if args.per_direction and epoch == todo[-1]:
            out = {"epoch": epoch, "threshold": thr}
            for name, (pc, pp, gc, gp) in preds.items():
                out[name] = {"overall": evaluate(pc, pp, gc, gp),
                             "directions": {d: direction_stats(pc[:, j], pp[:, j], gc[:, j], gp[:, j])
                                            for j, d in enumerate(DIRECTIONS)}}
            with open(os.path.join(args.run, "per_direction.json"), "w") as f:
                json.dump(out, f, indent=1)
            save_preds(os.path.join(args.run, "val_preds.npz"), recs, preds, report)


if __name__ == "__main__":
    main()
