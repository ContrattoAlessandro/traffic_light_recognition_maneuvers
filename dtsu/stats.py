"""Matching detections to labelled lights, average precision, and the cluster bootstraps behind every CI.

Frames of one drive are correlated, so every confidence interval resamples whole clusters: OpenLane-V2 segments,
DTLD sessions, VZC-TLD recordings, blocks of 100 consecutive frames on the video datasets (LISA, S2TLD), single
images where frames are independent (ATLAS, BDD100K).
"""
import warnings

import numpy as np
import torch
from torchvision.ops import box_iou


# ------------------------------------------------------------------ matching
def pair_ok(boxes, gt_boxes, criterion):
    """[n_pred, n_gt] IoU and whether each pair may match. "center": each box's center lies inside the other box
    (robust to the box conventions of different datasets); "iou0.5": IoU >= 0.5."""
    iou = box_iou(boxes, gt_boxes)
    if criterion == "center":
        def inside(a, b):
            cx, cy = ((a[:, 0] + a[:, 2]) / 2)[:, None], ((a[:, 1] + a[:, 3]) / 2)[:, None]
            return (cx >= b[None, :, 0]) & (cx <= b[None, :, 2]) & (cy >= b[None, :, 1]) & (cy <= b[None, :, 3])
        return iou, inside(boxes, gt_boxes) & inside(gt_boxes, boxes).T
    return iou, iou >= float(criterion[len("iou"):])


def match(boxes, scores, gt_boxes, ignore, criterion="center"):
    """Greedy by score; ties between labelled lights go to the higher IoU. Per prediction: the matched label index,
    -1 = false positive, -2 = on an ignore box (neither a hit nor a false positive)."""
    m = torch.full((len(scores),), -1, dtype=torch.long)
    if not len(scores) or not len(gt_boxes):
        return m
    iou, ok = pair_ok(boxes, gt_boxes, criterion)
    taken = set()
    for p in torch.argsort(scores, descending=True).tolist():
        cand = [(iou[p, g].item(), g) for g in range(len(gt_boxes)) if g not in taken and not ignore[g] and ok[p, g]]
        if cand:
            g = max(cand)[1]
            m[p] = g
            taken.add(g)
        elif ignore.any() and ok[p, ignore].any():
            m[p] = -2
    return m


def ignore_mask(item):
    return torch.tensor([c is None for c in item["color"]], dtype=torch.bool)


# ------------------------------------------------------------------ average precision (center matching)
def det_table(items, preds):
    """Every detection (score-sorted) with its image and whether it is a hit, plus the labelled lights per image."""
    im, sc, tp, n = [], [], [], np.zeros(len(items))
    for i, (it, (_, s, b)) in enumerate(zip(items, preds)):
        ign = ignore_mask(it)
        m = match(b, s, it["boxes"], ign, "center")
        v = m != -2
        im.append(np.full(int(v.sum()), i))
        sc.append(s[v].numpy())
        tp.append((m[v] >= 0).numpy().astype(float))
        n[i] = int((~ign).sum())
    im, sc, tp = np.concatenate(im), np.concatenate(sc), np.concatenate(tp)
    order = np.argsort(-sc, kind="stable")
    return im[order], tp[order], n


def weighted_ap(table, w):
    """All-point AP with per-image weights w (ones = the plain AP; bootstrap counts = one resample)."""
    im, tp, n = table
    ws = w[im]
    ctp, cfp = np.cumsum(ws * tp), np.cumsum(ws * (1 - tp))
    rec, prec = ctp / (w * n).sum(), ctp / np.maximum(ctp + cfp, 1e-9)
    env = np.maximum.accumulate(np.concatenate([prec, [0]])[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * env[:-1]))


def _resample_weights(n_clusters, draws, rng):
    w = np.zeros((draws, n_clusters))
    np.add.at(w, (np.arange(draws)[:, None], rng.integers(0, n_clusters, (draws, n_clusters))), 1)
    return w


# ------------------------------------------------------------------ bootstraps
def paired_frames(a, b, segments, draws=10000, seed=0):
    """Per-frame (TSS contribution, False-Go numerator, False-Go denominator) of models a and b on the same frames
    -> TSS and False-Go of each, and a - b with a segment-cluster bootstrap 95% CI."""
    uniq, inv = np.unique(segments, return_inverse=True)
    sums = lambda x: np.bincount(inv, weights=np.asarray(x, float), minlength=len(uniq))  # noqa: E731
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    ds, fa, fb, den = sums(a[0] - b[0]), sums(a[1]), sums(b[1]), sums(a[2])
    w = _resample_weights(len(uniq), draws, np.random.default_rng(seed))
    d_tss = (w @ ds) / (w @ n)
    d_fg = (w @ fa - w @ fb) / np.maximum(w @ den, 1)
    return {"a_tss": float(a[0].mean()), "b_tss": float(b[0].mean()), "d_tss": float(ds.sum() / n.sum()),
            "d_tss_ci95": [float(np.percentile(d_tss, 2.5)), float(np.percentile(d_tss, 97.5))],
            "a_false_go": float(a[1].sum() / max(a[2].sum(), 1)), "b_false_go": float(b[1].sum() / max(b[2].sum(), 1)),
            "d_false_go": float((fa.sum() - fb.sum()) / max(den.sum(), 1)),
            "d_false_go_ci95": [float(np.percentile(d_fg, 2.5)), float(np.percentile(d_fg, 97.5))], "frames": int(len(a[0]))}


def paired_rates(clusters, groups, ok_a, ok_b, draws=2000, seed=0):
    """Macro mean over groups of per-group success rates (a group with no row is left out) for models a and b on
    the same rows, and a - b with a cluster bootstrap 95% CI. groups: int per row, -1 = row not counted."""
    n_g = int(groups.max()) + 1 if len(groups) and groups.max() >= 0 else 0
    if n_g == 0:
        return None
    uniq, cl = np.unique(clusters, return_inverse=True)
    use = groups >= 0
    shape = (len(uniq), n_g)
    n, a, b = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(n, (cl[use], groups[use]), 1)
    np.add.at(a, (cl[use], groups[use]), ok_a[use])
    np.add.at(b, (cl[use], groups[use]), ok_b[use])

    def rate(w):
        nn, aa, bb = w @ n, w @ a, w @ b
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a resample can miss every row of a group
            ra, rb = aa / nn, bb / nn
            return np.nanmean(ra, -1), np.nanmean(rb, -1)

    va, vb = rate(np.ones(len(uniq)))
    ba, bb = rate(_resample_weights(len(uniq), draws, np.random.default_rng(seed)))
    d = ba - bb
    return {"a": float(va), "b": float(vb), "d": float(va - vb),
            "d_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "n": int(n.sum()), "n_by_group": n.sum(0).astype(int).tolist()}


def paired_ap(table_a, table_b, clusters, draws=2000, seed=0):
    """AP of a and b and a - b, resampling clusters (an int per image)."""
    nb = clusters.max() + 1
    rng = np.random.default_rng(seed)
    full = np.ones(len(clusters))
    diff = [weighted_ap(table_a, w) - weighted_ap(table_b, w)
            for w in (np.bincount(rng.integers(0, nb, nb), minlength=nb)[clusters].astype(float) for _ in range(draws))]
    a, b = weighted_ap(table_a, full), weighted_ap(table_b, full)
    return {"a_AP": a, "b_AP": b, "d_AP": a - b, "d_AP_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "clusters": int(nb)}
