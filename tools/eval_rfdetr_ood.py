"""Out-of-distribution test: an RF-DETR run trained on one dataset, scored on LISA.

The checks that make the comparison fair:
1. Label oracle: every LISA ground-truth light is mapped to the run's class names (a ball, LISA's
   "generic", becomes the trained "left+right+straight" class; arrows map directly) and pushed through
   ``assign_scored`` with LISA's US rule (near_turn "dedicated"). It must reproduce every LISA
   label, so any metric gap is the detector's.
2. Right turns follow the target country's rule (US: only a dedicated signal; LISA has none, so right
   is always "none"). Because that makes right trivial, metrics are also reported without it.
3. Geometry: the run's images were 2:1 and squeezed to a square; LISA is 4:3. ``--pad-aspect 2``
   pads LISA to 2:1 first so lights keep the shape the model saw in training. Both are reported.
4. Thresholds: the run's own (picked on its source tune split; the honest OOD number) and the best
   one on the LISA tune split (how much is only miscalibration).

Writes ``<run>/ood_lisa/epoch<N>_<geometry>.json`` with overall, no-right, per-direction and
day/night metrics at both thresholds.
"""
import argparse
import json
import os
import sys

import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, box_class, detect, frame_directions, load_image, parse_class, to_arrays  # noqa: E402
from tools.eval_directions import direction_stats  # noqa: E402
from tools.eval_rfdetr import MIN_SCORE, THRESHOLDS, load_model  # noqa: E402

L = "data/labels"
LISA = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "LISA")
SETS = {"LISA_tune": (f"{L}/tune/lisa_train_tune.jsonl", LISA), "LISA_test": (f"{L}/lisa_test.jsonl", LISA)}
NEAR_TURN = "dedicated"  # LISA = US rules
NO_RIGHT = [DIRECTIONS.index(d) for d in DIRECTIONS if d != "right"]


def to_trained_class(b, class_names):
    """LISA box -> the run's class name; balls ("generic") control every direction of the approach."""
    moves = ["left", "right", "straight"] if b["moves"] == ["generic"] else b["moves"]
    name = box_class({**b, "moves": moves})
    assert name in class_names, name
    return name


def oracle(records, class_names):
    """GT boxes through the class encoding and US rules; returns frames that differ from the labels."""
    bad = []
    for r in records:
        dets = []
        for b in r["boxes"]:
            moves, color = parse_class(to_trained_class(b, class_names))
            if moves is not None:
                dets.append((moves, color, 1.0))
        if assign_scored(dets, r["traffic_side"], NEAR_TURN) != r["color"]:
            bad.append(r["image"])
    return bad


class Images(Dataset):
    def __init__(self, records, resolution, pad_aspect):
        self.records, self.resolution, self.pad_aspect = records, resolution, pad_aspect

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return load_image(os.path.join(r["root"], r["image"]), self.resolution, self.pad_aspect)


def run_detector(model, records, class_names, resolution, pad_aspect, workers):
    dets = []
    for x in DataLoader(Images(records, resolution, pad_aspect), batch_size=8, num_workers=workers):
        cls, score, boxes = detect(model, x.cuda(non_blocking=True), class_names)
        for c, s, b in zip(cls, score, boxes):
            k = s > MIN_SCORE
            dets.append((c[k], s[k], b[k]))
    return dets


def score(records, dets, class_names, threshold):
    pred = [frame_directions(*d, class_names, threshold, r["traffic_side"], NEAR_TURN) for r, d in zip(records, dets)]
    return (*to_arrays(pred), *to_arrays([r["color"] for r in records]))


def report(records, arrays):
    pc, pp, gc, gp = arrays
    night = np.array(["night" in r["segment_id"] for r in records])
    out = {"overall": evaluate(*arrays),
           "no_right": evaluate(pc[:, NO_RIGHT], pp[:, NO_RIGHT], gc[:, NO_RIGHT], gp[:, NO_RIGHT]),
           "directions": {d: direction_stats(pc[:, j], pp[:, j], gc[:, j], gp[:, j]) for j, d in enumerate(DIRECTIONS)}}
    for name, m in (("day", ~night), ("night", night)):
        out[f"{name}_no_right"] = evaluate(pc[m][:, NO_RIGHT], pp[m][:, NO_RIGHT], gc[m][:, NO_RIGHT], gp[m][:, NO_RIGHT])
        out[f"{name}_frames"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/rfdetr_m_dtld")
    ap.add_argument("--class-names", default="data/rfdetr_coco_dtld/class_names.json")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--source-threshold", type=float, required=True, help="threshold picked on the run's own tune split")
    ap.add_argument("--pad-aspect", type=float, help="pad LISA to this width/height before the square resize")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N frames per set (smoke tests)")
    args = ap.parse_args()

    class_names = json.load(open(args.class_names))
    recs = {k: load_records(*v)[: args.limit] for k, v in SETS.items()}
    for k, rs in recs.items():
        bad = oracle(rs, class_names)
        print(f"oracle {k}: {len(rs)} frames, {len(bad)} differ from the labels {bad[:5]}")
        assert not bad

    model, res, _ = load_model(args.run, os.path.join(args.run, f"epoch_{args.epoch}_ema.pth"), len(class_names))
    dets = {k: run_detector(model, rs, class_names, res, args.pad_aspect, args.workers) for k, rs in recs.items()}
    tune_tss = [evaluate(*score(recs["LISA_tune"], dets["LISA_tune"], class_names, t))["tss"] for t in THRESHOLDS]
    thresholds = {"source": args.source_threshold, "lisa_tuned": float(THRESHOLDS[int(np.argmax(tune_tss))])}

    geometry = f"pad{args.pad_aspect:g}" if args.pad_aspect else "squeeze"
    out = {"run": args.run, "epoch": args.epoch, "geometry": geometry, "thresholds": thresholds,
           "frames": {k: len(v) for k, v in recs.items()}}
    for name, t in thresholds.items():
        out[name] = report(recs["LISA_test"], score(recs["LISA_test"], dets["LISA_test"], class_names, t))
        m, nr = out[name]["overall"], out[name]["no_right"]
        print(f"{geometry} {name} thr {t:.2f} | LISA_test TSS {m['tss']:.4f} EM {m['em']:.4f} | no right: TSS {nr['tss']:.4f} "
              f"Acc_color {nr['acc_color']:.4f} Acc_pass {nr['acc_pass']:.4f} EM {nr['em']:.4f} "
              f"F-Go {nr['false_go']:.4f} F-Stop {nr['false_stop']:.4f} | day TSS {out[name]['day_no_right']['tss']:.4f} "
              f"night TSS {out[name]['night_no_right']['tss']:.4f}", flush=True)
    os.makedirs(os.path.join(args.run, "ood_lisa"), exist_ok=True)
    with open(os.path.join(args.run, "ood_lisa", f"epoch{args.epoch}_{geometry}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
