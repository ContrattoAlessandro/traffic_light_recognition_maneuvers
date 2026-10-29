"""Per-direction evaluation of a trained checkpoint on the val sets.

For each val set and direction: label mix, color/pass accuracy, False-Go / False-Stop, the
color confusion matrix, and a majority-class baseline (always predict that direction's most
frequent color) for context. Also re-computes the overall metrics as a check against
val_log.jsonl. Writes <run>/per_direction.json, and the per-frame predictions to
<run>/val_preds.npz (for tools/bootstrap_compare.py).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import COLORS, DIRECTIONS, PASS, color_to_pass  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.model import DirectionAwareModel  # noqa: E402
from dtsu.train import make_loader, predict  # noqa: E402

STOP, GO = PASS.index("stop"), PASS.index("go")


def direction_stats(pc, pp, gc, gp):
    n = len(gc)
    gt_mix = {c: float((gc == i).mean()) for i, c in enumerate(COLORS)}
    majority = np.bincount(gc, minlength=len(COLORS)).argmax()
    maj_pass = PASS.index(color_to_pass(COLORS[majority]))
    conf = np.zeros((len(COLORS), len(COLORS)), int)
    np.add.at(conf, (gc, pc), 1)
    return {
        "n": int(n),
        "gt_color_mix": gt_mix,
        "acc_color": float((pc == gc).mean()),
        "acc_pass": float((pp == gp).mean()),
        "false_go": float(((gp == STOP) & (pp == GO)).sum() / max((gp == STOP).sum(), 1)),
        "false_stop": float(((gp == GO) & (pp == STOP)).sum() / max((gp == GO).sum(), 1)),
        "n_stop": int((gp == STOP).sum()),
        "majority_color": COLORS[majority],
        "majority_acc_color": float((gc == majority).mean()),
        "majority_acc_pass": float((gp == maj_pass).mean()),
        "pass_consistent_with_color": float((pp == np.vectorize(lambda c: PASS.index(color_to_pass(COLORS[c])))(pc)).mean()),
        "color_confusion_rows_gt_cols_pred": conf.tolist(),
    }


def save_preds(path, records, preds, names):
    """Per-frame predictions and labels of each val set, with segment ids for a clustered bootstrap."""
    arrays = {}
    for name in names:
        arrays.update({f"{name}_{k}": a for k, a in zip(("pc", "pp", "gc", "gp"), preds[name])})
        arrays[f"{name}_segment"] = np.array([r["segment_id"] for r in records[name]])
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/swin_t_AB")
    ap.add_argument("--val-set", nargs=3, action="append", metavar=("NAME", "LABELS", "ROOT"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    val_sets = args.val_set or [
        ("A", "data/labels/subset_A_val.jsonl", "data/OpenLane-V2/subset_A"),
        ("B", "data/labels/subset_B_val.jsonl", "data/OpenLane-V2/subset_B"),
    ]
    targs = json.load(open(os.path.join(args.run, "args.json")))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionAwareModel((targs["long_side"], targs["long_side"]), pretrained=False,
                                backbone=targs.get("backbone", "swin_tiny_patch4_window7_224"),
                                top_k=targs["top_k"], n_queries=targs["n_queries"]).to(dev)
    ckpt = torch.load(os.path.join(args.run, "last.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])

    class LoaderArgs:
        long_side, batch_size, workers, seed = targs["long_side"], 4, args.workers, 0

    out, preds, records = {"epoch": ckpt["epoch"]}, {}, {}
    for name, labels, root in val_sets:
        records[name] = load_records(labels, root)
        dl = make_loader(records[name], LoaderArgs, train=False)
        preds[name] = predict(model, dl, dev, amp=targs["amp"])
    preds["combined"] = [np.concatenate(x) for x in zip(*preds.values())]
    for name, (pc, pp, gc, gp) in preds.items():
        out[name] = {"overall": evaluate(pc, pp, gc, gp),
                     "directions": {d: direction_stats(pc[:, j], pp[:, j], gc[:, j], gp[:, j])
                                    for j, d in enumerate(DIRECTIONS)}}
        print(f"== {name} (epoch {ckpt['epoch']}) TSS {out[name]['overall']['tss']:.4f}")
        print(f"{'dir':9s} {'none/red/yel/grn %':>22s} {'Acc_c':>6s} {'Acc_p':>6s} {'F-Go':>6s} {'F-Stop':>6s} "
              f"{'n_stop':>6s} {'maj_c':>6s} {'maj_p':>6s}")
        for d, s in out[name]["directions"].items():
            mix = "/".join(f"{100 * s['gt_color_mix'][c]:.1f}" for c in COLORS)
            print(f"{d:9s} {mix:>22s} {s['acc_color']:6.4f} {s['acc_pass']:6.4f} {s['false_go']:6.4f} "
                  f"{s['false_stop']:6.4f} {s['n_stop']:6d} {s['majority_acc_color']:6.4f} {s['majority_acc_pass']:6.4f}")
    with open(os.path.join(args.run, "per_direction.json"), "w") as f:
        json.dump(out, f, indent=1)
    save_preds(os.path.join(args.run, "val_preds.npz"), records, preds, records)


if __name__ == "__main__":
    main()
