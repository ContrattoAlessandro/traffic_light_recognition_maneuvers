"""R against M over several runs of each: the primary comparison of the publication protocol (docs/EXPERIMENT_LOG.md).

Each run enters through its result file, which gives the run folder and its selection (epoch and threshold, chosen on
the tune splits): ``scripts/evaluate.py`` for the runs of R, ``scripts/evaluate_moves.py`` for the runs of M. Per frame,
each model's scores are averaged over its runs, and R - M is paired on the same frames with a cluster bootstrap 95% CI
(10,000 draws, seed 0; segments on OpenLane-V2, sessions on DTLD):
- primary: A val and B val camera only (R: US rule; M: its own decoder) and DTLD test against the route labels
  (R: junction rule; M: German rule), on the frames both labelings score;
- secondary: A val and B val with the lane-topology mask.
The file also holds each model's mean and standard deviation over its runs and the protocol's claims.

By the protocol, the scores are read only when all runs exist: fewer than --min-runs runs per model is refused.

Example:
  python scripts/compare_runs.py --r results/R.json results/seeds/R_seed1.json results/seeds/R_seed2.json \
      --m results/M.json results/seeds/M_seed1.json results/seeds/M_seed2.json --out results/seeds/R_vs_M.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import evaluation, moves_evaluation as me  # noqa: E402


def runs_of(results, caches, kind):
    if caches and len(caches) != len(results):
        sys.exit(f"--{kind}-cache needs one folder per result file")
    out = []
    for i, path in enumerate(results):
        res = json.load(open(path))
        cache = caches[i] if caches else res.get("cache")
        sel = {"epoch": res["selection"]["epoch"], "threshold": res["selection"]["threshold"]}
        out.append({"result": path, "run": res["run"], "cache": cache, "selection": sel})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r", nargs="+", required=True, help="result files of the runs of R (scripts/evaluate.py)")
    ap.add_argument("--m", nargs="+", required=True, help="result files of the runs of M (scripts/evaluate_moves.py)")
    ap.add_argument("--r-cache", nargs="+", help="detection caches of the R runs (default: the result file's, else <run>/eval)")
    ap.add_argument("--m-cache", nargs="+", help="detection caches of the M runs")
    ap.add_argument("--min-runs", type=int, default=3, help="runs required per model (lower only for checks)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda", help="device of M's inference, if a cache is missing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r_runs, m_runs = runs_of(args.r, args.r_cache, "r"), runs_of(args.m, args.m_cache, "m")
    if min(len(r_runs), len(m_runs)) < args.min_runs:
        sys.exit(f"{len(r_runs)} runs of R and {len(m_runs)} of M: the protocol compares the models once every run exists "
                 f"(--min-runs {args.min_runs})")

    ol, _, route_all = me.evaluation_sets()
    dtld_test = evaluation.dtld_set("test")
    a = [me.factorized_cells(evaluation.Run(f"R{i}", r["run"], r["cache"], args.workers), r["selection"], ol, route_all, dtld_test)
         for i, r in enumerate(r_runs)]
    b = [me.moves_cells(me.MovesRun(f"M{i}", r["run"], r["cache"], args.workers, args.device), r["selection"], ol, route_all, dtld_test)
         for i, r in enumerate(m_runs)]
    cells = me.compare_runs(a, b, me.cell_clusters(ol, route_all, dtld_test))
    out = {"a": "R", "b": "M", "r_runs": r_runs, "m_runs": m_runs, "primary": list(me.PRIMARY), "secondary": list(me.SECONDARY),
           "cells": cells, "claims": me.claims(cells)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
