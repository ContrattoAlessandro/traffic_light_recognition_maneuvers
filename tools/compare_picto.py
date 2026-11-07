"""Exp 6: paired F - J comparison on DTLD test from two tools/eval_picto.py outputs (session-cluster bootstrap).

Rule metrics are compared on the same scored frames. TSS is a per-frame mean; False-Go is a ratio of sums.
Per-light metrics are ratios of per-session sums (each arm keeps its own matches):
- recall: matched / labelled lights;
- precision: matched / kept detections;
- on matched lights: pictogram, color, junction and all-three accuracy, and pictogram accuracy per arrow class.
95% CI from 10,000 session resamples (seed 0). Writes <out>.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.picto_rules import PICTOS  # noqa: E402


def load(run):
    z = np.load(os.path.join(run, "picto_eval", "test_frames.npz"))
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--j", required=True)
    ap.add_argument("--f", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    J, F = load(args.j), load(args.f)
    assert (J["frame"] == F["frame"]).all()
    sessions = np.unique(J["frame_session"])
    sid = {s: i for i, s in enumerate(sessions)}
    n_s = len(sessions)
    rng = np.random.default_rng(0)
    w = np.zeros((args.draws, n_s))
    np.add.at(w, (np.arange(args.draws)[:, None], rng.integers(0, n_s, (args.draws, n_s))), 1)

    def sums(values, ses):
        return np.bincount([sid[s] for s in ses], weights=values, minlength=n_s)

    def ratio_ci(num_j, den_j, num_f, den_f):
        point = num_f.sum() / max(den_f.sum(), 1) - num_j.sum() / max(den_j.sum(), 1)
        boot = (w @ num_f) / np.maximum(w @ den_f, 1) - (w @ num_j) / np.maximum(w @ den_j, 1)
        return {"J": float(num_j.sum() / max(den_j.sum(), 1)), "F": float(num_f.sum() / max(den_f.sum(), 1)),
                "F_minus_J": float(point), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

    out = {"j": args.j, "f": args.f, "sessions": int(n_s), "scored_frames": int(len(J["frame"]))}
    ones = np.ones(len(J["frame"]))
    out["tss"] = ratio_ci(sums(J["tss"], J["session"]), sums(ones, J["session"]), sums(F["tss"], F["session"]), sums(ones, F["session"]))
    out["false_go"] = ratio_ci(sums(J["fg_num"], J["session"]), sums(J["fg_den"], J["session"]),
                               sums(F["fg_num"], F["session"]), sums(F["fg_den"], F["session"]))

    def light(arm, col, mask=None):
        rows = arm["light_rows"]
        ses = arm["frame_session"][rows[:, 0]]
        m = rows[:, 1] == 1 if mask is None else (rows[:, 1] == 1) & mask(rows)
        return sums(rows[:, col] * m, ses), sums(m.astype(float), ses)

    J_ses, F_ses = J["frame_session"], F["frame_session"]
    out["recall"] = ratio_ci(*[sums(a["light_rows"][:, 1], a["frame_session"][a["light_rows"][:, 0]]) for a in (J,)],
                             sums(np.ones(len(J["light_rows"])), J_ses[J["light_rows"][:, 0]]),
                             sums(F["light_rows"][:, 1], F_ses[F["light_rows"][:, 0]]),
                             sums(np.ones(len(F["light_rows"])), F_ses[F["light_rows"][:, 0]]))
    out["precision"] = ratio_ci(sums(J["light_rows"][:, 1], J_ses[J["light_rows"][:, 0]]), sums(J["n_det"], J_ses),
                                sums(F["light_rows"][:, 1], F_ses[F["light_rows"][:, 0]]), sums(F["n_det"], F_ses))
    for name, col in (("picto_acc", 2), ("color_acc", 3), ("junction_acc", 4), ("all_three_acc", 5)):
        out[name] = ratio_ci(*light(J, col), *light(F, col))
    for c, p in enumerate(PICTOS):
        if p.startswith("arrow") and ((J["light_rows"][:, 6] == c) & (J["light_rows"][:, 1] == 1)).sum() >= 20:
            out[f"picto_acc_{p}"] = ratio_ci(*light(J, 2, lambda r, c=c: r[:, 6] == c), *light(F, 2, lambda r, c=c: r[:, 6] == c))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:28s} J {v['J']:.4f}  F {v['F']:.4f}  F-J {v['F_minus_J']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
