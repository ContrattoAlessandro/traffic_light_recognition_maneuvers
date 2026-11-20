"""Maneuver-level metrics (Zou et al., Eqs. 4-7).

Inputs are integer arrays of shape [N, D] (frames x maneuvers), color indices into ``dtsu.COLORS`` and stop / go
indices into ``dtsu.PASS``.
- Acc_color, Acc_pass: accuracy over all frame x maneuver states;
- EM: a frame counts only if every color and every stop / go is right;
- TSS = (Acc_color + Acc_pass + EM) / 3;
- False-Go = #(label stop, prediction go) / #(label stop); False-Stop = #(label go, prediction stop) / #(label go),
  both pooled over maneuvers.
"""
import numpy as np

from . import PASS

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
    false_go = ((gp == STOP) & (pp == GO)).sum() / max((gp == STOP).sum(), 1)
    false_stop = ((gp == GO) & (pp == STOP)).sum() / max((gp == GO).sum(), 1)
    return {
        "acc_color": float(acc_color),
        "acc_pass": float(acc_pass),
        "em": float(em),
        "tss": float((acc_color + acc_pass + em) / 3),
        "false_go": float(false_go),
        "false_stop": float(false_stop),
    }
