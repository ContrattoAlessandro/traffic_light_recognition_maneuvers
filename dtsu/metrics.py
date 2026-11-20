"""Maneuver-level evaluation metrics (Zou et al., Eqs. 4-7) and per-direction statistics.

Inputs are integer arrays of shape [N, D] (frames x directions, D = len(DIRECTIONS)), with color
indices into ``COLORS`` and passability indices into ``PASS``.
"""
import numpy as np

from . import COLORS, PASS, color_to_pass

STOP = PASS.index("stop")
GO = PASS.index("go")


def evaluate(pred_color, pred_pass, gt_color, gt_pass):
    pc, pp, gc, gp = (np.asarray(a) for a in (pred_color, pred_pass, gt_color, gt_pass))
    assert pc.shape == pp.shape == gc.shape == gp.shape and pc.ndim == 2

    color_ok = pc == gc
    pass_ok = pp == gp
    acc_color = color_ok.mean()
    acc_pass = pass_ok.mean()
    em = (color_ok & pass_ok).all(axis=1).mean()

    n_stop = (gp == STOP).sum()
    n_go = (gp == GO).sum()
    false_go = ((gp == STOP) & (pp == GO)).sum() / max(n_stop, 1)
    false_stop = ((gp == GO) & (pp == STOP)).sum() / max(n_go, 1)

    return {
        "acc_color": float(acc_color),
        "acc_pass": float(acc_pass),
        "em": float(em),
        "tss": float((acc_color + acc_pass + em) / 3),
        "false_go": float(false_go),
        "false_stop": float(false_stop),
    }


def direction_stats(pc, pp, gc, gp):
    """Metrics of one direction (1-D arrays over frames): label mix, accuracies, False-Go / False-Stop,
    the color confusion matrix, and a majority-class baseline (always predict the most frequent color)."""
    n = len(gc)
    gt_mix = {c: float((gc == i).mean()) for i, c in enumerate(COLORS)}
    majority = np.bincount(gc, minlength=len(COLORS)).argmax()
    maj_pass = PASS.index(color_to_pass(COLORS[majority]))
    conf = np.zeros((len(COLORS), len(COLORS)), int)
    np.add.at(conf, (gc, pc), 1)
    rule_pass = np.array([PASS.index(color_to_pass(c)) for c in COLORS])[pc]
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
        "pass_consistent_with_color": float((pp == rule_pass).mean()),
        "color_confusion_rows_gt_cols_pred": conf.tolist(),
    }


def save_preds(path, records, preds, names):
    """Per-frame predictions and labels of each evaluation set, with segment ids for a clustered bootstrap
    (``tools/bootstrap_compare.py``). ``preds[name]`` is the tuple (pc, pp, gc, gp)."""
    arrays = {}
    for name in names:
        arrays.update({f"{name}_{k}": a for k, a in zip(("pc", "pp", "gc", "gp"), preds[name])})
        arrays[f"{name}_segment"] = np.array([r["segment_id"] for r in records[name]])
    np.savez_compressed(path, **arrays)


def format_metrics(m):
    keys = ("acc_color", "acc_pass", "em", "tss", "false_go", "false_stop")
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)


__all__ = ["evaluate", "direction_stats", "save_preds", "format_metrics", "COLORS", "PASS"]
