"""Evaluation protocol of the paper (Eqs. 4-7).

Inputs are integer arrays of shape [N, 4] (frames x directions), with color
indices into ``COLORS`` and passability indices into ``PASS``.
"""
import numpy as np

from . import COLORS, PASS

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


def format_metrics(m):
    keys = ("acc_color", "acc_pass", "em", "tss", "false_go", "false_stop")
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)


__all__ = ["evaluate", "format_metrics", "COLORS", "PASS"]
