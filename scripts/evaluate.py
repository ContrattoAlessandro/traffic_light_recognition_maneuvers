"""Select and evaluate a trained run, optionally against a reference run (``dtsu.evaluation``).

The run's epoch and score threshold are chosen on the tune splits unless given (``--epoch`` and ``--threshold``,
or ``--selection`` pointing at an earlier result file). With ``--reference`` every metric is paired with the
reference run: "a" is the run, "b" the reference, "d" = a - b with its 95% CI.

Examples:
  python scripts/evaluate.py --run runs/R --out results/R.json
  python scripts/evaluate.py --run runs/R --reference runs/R_1024 --out results/R_vs_R_1024.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import evaluation  # noqa: E402


def selection(run, sets, dtld_tune, given, epoch, threshold):
    if given:
        s = json.load(open(given))["selection"]
        return {"epoch": s["epoch"], "threshold": s["threshold"], "tune": s.get("tune"), "from": given}
    if epoch is not None:
        return {"epoch": epoch, "threshold": threshold}
    return evaluation.select(run, sets, dtld_tune)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", help="detection cache (default <run>/eval)")
    ap.add_argument("--selection", help="a result file whose selection to reuse")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--reference")
    ap.add_argument("--reference-cache")
    ap.add_argument("--reference-selection", help="the reference's result file (default: select it on the tune splits)")
    ap.add_argument("--reference-epoch", type=int)
    ap.add_argument("--reference-threshold", type=float)
    ap.add_argument("--no-detection", action="store_true", help="skip the detection cells (the longest part)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sets, dtld_tune, dtld_test = evaluation.evaluation_sets()
    ra = evaluation.Run("a", args.run, args.cache, args.workers)
    sa = selection(ra, sets, dtld_tune, args.selection, args.epoch, args.threshold)
    out = {"run": args.run, "resolution": ra.res, "selection": sa}
    rb = sb = None
    if args.reference:
        rb = evaluation.Run("b", args.reference, args.reference_cache, args.workers)
        sb = selection(rb, sets, dtld_tune, args.reference_selection, args.reference_epoch, args.reference_threshold)
        out.update(reference=args.reference, reference_resolution=rb.res, reference_selection=sb)
    out.update(evaluation.report(ra, sa, rb, sb, sets, dtld_tune, dtld_test, detection=not args.no_detection))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    m = out["maneuver_metrics"]["a"]
    print("TSS / False-Go:", {k: (round(v["tss"], 4), round(v["false_go"], 4)) for k, v in m.items()}, flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
