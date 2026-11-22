"""Select and evaluate the baseline M (``dtsu.moves_evaluation``), optionally paired with a factorized run such as R.

The epoch and score threshold are chosen on the tune splits (the Exp 5 rule) unless given with --epoch and --threshold
or --selection. With --reference (a factorized run) the publication cells are also paired frame by frame:
"a" = the reference, "b" = M, "d" = a - b with its segment-bootstrap 95% CI.

Examples:
  python scripts/evaluate_moves.py --run runs/M --out results/M.json
  python scripts/evaluate_moves.py --run runs/M --selection results/M.json --reference runs/R \
      --reference-selection results/R.json --no-detection --out results/R_vs_M.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import evaluation, moves_evaluation as me  # noqa: E402


def given_selection(path, epoch, threshold):
    if path:
        s = json.load(open(path))["selection"]
        return {"epoch": s["epoch"], "threshold": s["threshold"], "tune": s.get("tune"), "from": path}
    if epoch is not None:
        return {"epoch": epoch, "threshold": threshold}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", help="detection cache (default <run>/eval)")
    ap.add_argument("--selection", help="a result file whose selection to reuse")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--reference", help="a factorized run (R) to pair with M on the publication cells")
    ap.add_argument("--reference-cache")
    ap.add_argument("--reference-selection", help="the reference's result file (default: select it on the tune splits)")
    ap.add_argument("--no-detection", action="store_true", help="skip the detection cells (the longest part)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ol, dt, route_all = me.evaluation_sets()
    run = me.MovesRun("M", args.run, args.cache, args.workers, args.device)
    sel = given_selection(args.selection, args.epoch, args.threshold) or me.select(run, ol, dt)
    out = {"run": args.run, "resolution": run.res, "selection": sel}
    out.update(me.report(run, sel, ol, dt, detection=not args.no_detection))
    if args.reference:
        ref = evaluation.Run("a", args.reference, args.reference_cache, args.workers)
        sets, dtld_tune, dtld_test = evaluation.evaluation_sets()
        rsel = given_selection(args.reference_selection, None, None) or evaluation.select(ref, sets, dtld_tune)
        a, b, clusters = me.paired_cells(run, sel, ref, rsel, ol, dt, route_all, dtld_test)
        out.update(reference=args.reference, reference_selection=rsel, paired=me.compare(a, b, clusters),
                   paired_metrics={"a": {k: v[3] for k, v in a.items()}, "b": {k: v[3] for k, v in b.items()}})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("TSS / False-Go:", {k: (round(v["tss"], 4), round(v["false_go"], 4)) for k, v in out["maneuver_metrics"].items()}, flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
