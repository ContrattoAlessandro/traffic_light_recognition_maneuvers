"""Out-of-distribution test: an RF-DETR run trained on one dataset, scored on LISA or OpenLane-V2.

The checks that make the comparison fair:
1. Label oracle: every ground-truth light is mapped to the run's class names (a ball, "generic",
   becomes the trained "left+right+straight" class; arrows map directly; U-turn arrows, which DTLD
   has no class for, drop "uturn") and pushed through ``assign_scored`` with the target's US rule
   (near_turn "dedicated"). Frames it cannot reproduce are left out and counted (none on LISA, 6 of
   2,220 on OpenLane-V2 subset_A val), so any metric gap is the detector's.
2. Right turns follow the US rule (only a dedicated signal; LISA has none, so right is always
   "none"). Because that makes right trivial, metrics are also reported without it.
3. Geometry: the run's images were 2:1 and squeezed to a square; the targets are not (LISA 4:3,
   Argoverse 2 front camera 3:4 portrait, nuScenes 16:9). ``--pad-aspect 2`` pads to 2:1 first so
   lights keep the shape the model saw in training. Both are reported.
4. Thresholds: the run's own (picked on its source tune split; the honest OOD number) and the best
   one on the target's tune split (how much is only miscalibration).

Writes ``<run>/ood_<target>/epoch<N>_<geometry>.json`` with overall, no-right, per-direction (and on
LISA day/night) metrics per test set at both thresholds.
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

L, R = "data/labels", "data/OpenLane-V2"
LISA = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "LISA")
# per target: (tune sets that pick the "tuned" threshold, test sets reported)
TARGETS = {
    "lisa": ({"LISA_tune": (f"{L}/tune/lisa_train_tune.jsonl", LISA)},
             {"LISA_test": (f"{L}/lisa_test.jsonl", LISA)}),
    "openlane": ({"A_tune": (f"{L}/tune/subset_A_train_tune.jsonl", f"{R}/subset_A"),
                  "B_tune": (f"{L}/tune/subset_B_train_tune.jsonl", f"{R}/subset_B")},
                 {"A_val": (f"{L}/subset_A_val.jsonl", f"{R}/subset_A"),
                  "B_val": (f"{L}/subset_B_val.jsonl", f"{R}/subset_B")}),
}
NEAR_TURN = "dedicated"  # LISA and OpenLane-V2 labels use US rules
NO_RIGHT = [DIRECTIONS.index(d) for d in DIRECTIONS if d != "right"]


def to_trained_class(b, class_names):
    """Target box -> the run's class name; balls ("generic") control every direction of the approach."""
    moves = set(b["moves"]) - {"uturn"}
    if "generic" in moves:
        moves = (moves - {"generic"}) | {"left", "right", "straight"}
    name = box_class({**b, "moves": sorted(moves)})
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
    out = {"frames": len(records), "overall": evaluate(*arrays),
           "no_right": evaluate(pc[:, NO_RIGHT], pp[:, NO_RIGHT], gc[:, NO_RIGHT], gp[:, NO_RIGHT]),
           "directions": {d: direction_stats(pc[:, j], pp[:, j], gc[:, j], gp[:, j]) for j, d in enumerate(DIRECTIONS)}}
    for name, m in (("day", ~night), ("night", night)) if night.any() else ():
        out[f"{name}_no_right"] = evaluate(pc[m][:, NO_RIGHT], pp[m][:, NO_RIGHT], gc[m][:, NO_RIGHT], gp[m][:, NO_RIGHT])
        out[f"{name}_frames"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=sorted(TARGETS), default="lisa")
    ap.add_argument("--run", default="runs/rfdetr_m_dtld")
    ap.add_argument("--class-names", default="data/rfdetr_coco_dtld/class_names.json")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--source-threshold", type=float, required=True, help="threshold picked on the run's own tune split")
    ap.add_argument("--pad-aspect", type=float, help="pad images to this width/height before the square resize")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N frames per set (smoke tests)")
    args = ap.parse_args()

    class_names = json.load(open(args.class_names))
    tune_sets, test_sets = TARGETS[args.target]
    recs, dropped = {}, {}
    for k, v in {**tune_sets, **test_sets}.items():
        rs = load_records(*v)[: args.limit]
        bad = set(oracle(rs, class_names))
        print(f"oracle {k}: {len(rs)} frames, {len(bad)} not reproducible, left out {sorted(bad)[:5]}")
        recs[k], dropped[k] = [r for r in rs if r["image"] not in bad], len(bad)

    model, res, _ = load_model(args.run, os.path.join(args.run, f"epoch_{args.epoch}_ema.pth"), len(class_names))
    dets = {k: run_detector(model, rs, class_names, res, args.pad_aspect, args.workers) for k, rs in recs.items()}
    tune_recs = [r for k in tune_sets for r in recs[k]]
    tune_dets = [d for k in tune_sets for d in dets[k]]
    tune_tss = [evaluate(*score(tune_recs, tune_dets, class_names, t))["tss"] for t in THRESHOLDS]
    thresholds = {"source": args.source_threshold, "tuned": float(THRESHOLDS[int(np.argmax(tune_tss))])}

    geometry = f"pad{args.pad_aspect:g}" if args.pad_aspect else "squeeze"
    out = {"run": args.run, "epoch": args.epoch, "target": args.target, "geometry": geometry,
           "thresholds": thresholds, "dropped_by_oracle": dropped}
    for name, t in thresholds.items():
        out[name] = {}
        for k in test_sets:
            out[name][k] = rep = report(recs[k], score(recs[k], dets[k], class_names, t))
            m, nr = rep["overall"], rep["no_right"]
            dn = (f" | day TSS {rep['day_no_right']['tss']:.4f} night TSS {rep['night_no_right']['tss']:.4f}"
                  if "day_no_right" in rep else "")
            print(f"{geometry} {name} thr {t:.2f} | {k} TSS {m['tss']:.4f} Acc_color {m['acc_color']:.4f} "
                  f"Acc_pass {m['acc_pass']:.4f} EM {m['em']:.4f} F-Go {m['false_go']:.4f} F-Stop {m['false_stop']:.4f} "
                  f"| no right: TSS {nr['tss']:.4f} Acc_color {nr['acc_color']:.4f} Acc_pass {nr['acc_pass']:.4f} "
                  f"EM {nr['em']:.4f} F-Go {nr['false_go']:.4f}{dn}", flush=True)
    out_dir = os.path.join(args.run, f"ood_{args.target}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"epoch{args.epoch}_{geometry}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
