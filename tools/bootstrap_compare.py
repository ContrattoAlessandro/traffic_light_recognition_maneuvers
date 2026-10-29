"""Paired bootstrap of the metric differences between two runs on the same val frames.

Reads ``<run>/val_preds.npz`` of both runs (written by ``tools/eval_directions.py`` /
``tools/eval_rfdetr.py --per-direction``). Frames of one driving segment are strongly correlated,
so whole segments are resampled (with replacement, separately per val set), and both runs are
scored on the same resample. Reports each metric for both runs, the difference (a - b), its 95%
percentile interval and a two-sided bootstrap p-value. Writes ``--out`` as JSON.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.metrics import evaluate  # noqa: E402

METRICS = ("tss", "acc_color", "acc_pass", "em", "false_go", "false_stop")


def load(path, sets):
    z = np.load(path)
    return {s: {k: z[f"{s}_{k}"] for k in ("pc", "pp", "gc", "gp", "segment")} for s in sets}


def metrics(pa, pb, idx):
    """Metrics of both runs on the frames ``idx`` (an index array per val set, concatenated)."""
    cat = lambda d, k: np.concatenate([d[s][k][i] for s, i in idx.items()])
    gc, gp = cat(pa, "gc"), cat(pa, "gp")
    return (evaluate(cat(pa, "pc"), cat(pa, "pp"), gc, gp), evaluate(cat(pb, "pc"), cat(pb, "pp"), gc, gp))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default="runs/swin_t_AB")
    ap.add_argument("--b", default="runs/rfdetr_m_AB")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bootstrap_swin_vs_rfdetr.json")
    args = ap.parse_args()

    sets = ("A", "B")
    pa, pb = load(os.path.join(args.a, "val_preds.npz"), sets), load(os.path.join(args.b, "val_preds.npz"), sets)
    for s in sets:  # same frames, same order, same labels
        for k in ("gc", "gp", "segment"):
            assert np.array_equal(pa[s][k], pb[s][k]), f"{s}_{k} differs between runs"

    groups = {s: [np.flatnonzero(pa[s]["segment"] == g) for g in np.unique(pa[s]["segment"])] for s in sets}
    rng = np.random.default_rng(args.seed)
    scopes = {"A": ("A",), "B": ("B",), "combined": sets}
    point = {sc: metrics(pa, pb, {s: np.arange(len(pa[s]["gc"])) for s in ss}) for sc, ss in scopes.items()}
    diffs = {sc: {m: [] for m in METRICS} for sc in scopes}
    for _ in range(args.n):
        idx = {s: np.concatenate([groups[s][j] for j in rng.integers(len(groups[s]), size=len(groups[s]))])
               for s in sets}
        for sc, ss in scopes.items():
            ma, mb = metrics(pa, pb, {s: idx[s] for s in ss})
            for m in METRICS:
                diffs[sc][m].append(ma[m] - mb[m])

    out = {"a": args.a, "b": args.b, "n_boot": args.n,
           "n_segments": {s: len(groups[s]) for s in sets}, "n_frames": {s: len(pa[s]["gc"]) for s in sets}}
    print(f"a = {args.a}, b = {args.b}; {args.n} segment-level resamples "
          f"({', '.join(f'{s}: {len(groups[s])} segments' for s in sets)})")
    for sc in scopes:
        out[sc] = {}
        print(f"== {sc}")
        print(f"{'metric':11s} {'a':>7s} {'b':>7s} {'a-b':>8s} {'95% CI':>19s} {'p':>6s}")
        for m in METRICS:
            d = np.array(diffs[sc][m])
            obs = point[sc][0][m] - point[sc][1][m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            out[sc][m] = {"a": point[sc][0][m], "b": point[sc][1][m], "diff": obs,
                          "ci95": [float(lo), float(hi)], "p": float(p)}
            print(f"{m:11s} {point[sc][0][m]:7.4f} {point[sc][1][m]:7.4f} {obs:+8.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {p:6.3f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
