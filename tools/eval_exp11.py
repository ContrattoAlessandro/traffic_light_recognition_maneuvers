"""Exp 11 (REPLICATION_PLAN.md): R at 1536 after one epoch (arm S, runs/exp11_S) against R and WF at epoch 0.

Models: S (epoch 0), R0 / WF0 (R and WF at epoch 0), R / WF (the Exp 10 / Exp 9 selections).
Each epoch-0 model gets its threshold from the Exp 8 rule with the epoch fixed at 0 (tools/eval_exp9.select).
Primary pair S - R0: P1 / P2 and guards G1-G4 (tools/eval_exp10.decide), plus attributes and perception cells.
Reported pairs: S - WF0, R0 - WF0, S - R, S - WF (guards, attributes, P2).
Writes runs/exp11_eval/result.json (runs/exp11_eval_smoke with --limit).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp8 as e8  # noqa: E402
import tools.eval_exp9 as e9  # noqa: E402
import tools.eval_exp10 as e10  # noqa: E402

S_RUN, R_RUN, WF_RUN = "runs/exp11_S", "runs/exp10_R", "runs/exp9_WF"
OUT = "runs/exp11_eval"
PAIRS = (("S", "R0"), ("S", "WF0"), ("R0", "WF0"), ("S", "R"), ("S", "WF"))


def epoch0(m):
    m.epochs = [0]
    m.base.epochs = [0]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run", default=S_RUN, help="arm S's run folder (a smoke test can point at another run)")
    ap.add_argument("--limit", type=int, help="smoke test: evenly spaced N frames per set")
    ap.add_argument("--full", action="store_true",
                    help="S resumed to 5 epochs: S selected over all its epochs, against R and WF "
                         "(writes runs/exp11_eval_full; reuses the epoch-0 cache of runs/exp11_eval/S)")
    args = ap.parse_args()
    global OUT
    e8.WORKERS = e9.WORKERS = args.workers
    smoke = bool(args.limit)
    if args.full:
        OUT = "runs/exp11_eval_full"
    if smoke:
        e8.LIMIT = args.limit
        OUT = "runs/exp11_eval_smoke" + ("_full" if args.full else "")
    os.makedirs(OUT, exist_ok=True)
    cache = lambda name, real: os.path.join(OUT, name) if smoke else real  # noqa: E731
    if args.full:
        return full(args, cache, smoke)

    models = {"S": epoch0(e9.Fixed("S", args.run, e9.DATA, os.path.join(OUT, "S"))),
              "R0": epoch0(e9.Fixed("R0", R_RUN, e9.DATA, cache("R", "runs/exp10_eval/R"))),
              "WF0": epoch0(e9.Fixed("WF0", WF_RUN, e9.DATA, cache("WF", "runs/exp9_eval/WF"))),
              "R": e9.Fixed("R", R_RUN, e9.DATA, cache("R", "runs/exp10_eval/R")),
              "WF": e9.Fixed("WF", WF_RUN, e9.DATA, cache("WF", "runs/exp9_eval/WF"))}
    r9 = json.load(open("runs/exp9_eval/result.json"))["selection"]["WF"]
    r10 = json.load(open("runs/exp10_eval/result.json"))["selection"]["R"]
    sel = {"WF": {"epoch": r9["epoch"], "threshold": r9["threshold"], "tau": 0.0},
           "R": {"epoch": r10["epoch"], "threshold": r10["threshold"], "tau": 0.0}}

    sets = {n: e8.openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dtld_tune = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "valid", "records.jsonl"))])
    test = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "dtld_test_records.jsonl"))])
    asets = e9.attribute_sets()
    if smoke:
        asets = {k: (e8.sample(v[0]), *v[1:]) for k, v in asets.items()}

    for name in ("S", "R0", "WF0"):
        sel[name] = e9.select(name, models[name], sets, dtld_tune, (0.0,))
    res = {"selection": dict(sel)}
    for a, b in PAIRS:
        tag = f"{a}_vs_{b}"
        res[f"guards_{tag}"] = e9.guards(models, sel, sets, test, a, b)
        print(tag, "guards", {k: (round(v["d_tss"], 4), [round(x, 4) for x in v["d_tss_ci95"]], round(v["d_false_go"], 4))
                              for k, v in res[f"guards_{tag}"].items()}, flush=True)
        res[f"attributes_{tag}"] = e9.attributes(models, sel, asets, sets, test, a, b)
        res[f"P2_{tag}"] = e10.small_lamps(models, sel, asets, test, a, b)
        dt = res[f"attributes_{tag}"]["DTLD_test"]
        print(tag, "P1", {k: v for k, v in dt["picto_macro"].items() if k != "n_by_group"},
              "P2", {k: v for k, v in res[f"P2_{tag}"].items() if k != "n_by_group"}, flush=True)
        if not smoke and (a, b) == ("S", "R0"):
            res[f"detection_{tag}"] = e9.detection(models, sel, a, b)
    # decision on S - R0, with the Exp 10 rule (its key names are R vs WF)
    res["decision"] = {k.replace("R_replaces_WF", "S_continues"): v for k, v in e10.decide(
        {"guards_R_vs_WF": res["guards_S_vs_R0"], "attributes_R_vs_WF": res["attributes_S_vs_R0"],
         "P2_R_vs_WF": res["P2_S_vs_R0"]}).items()}
    print("decision", res["decision"], flush=True)
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)


def full(args, cache, smoke):
    """S over all its epochs (Exp 8 rule) against R and WF at their selections; decision on S - R."""
    models = {"S": e9.Fixed("S", args.run, e9.DATA, cache("S", "runs/exp11_eval/S")),
              "R": e9.Fixed("R", R_RUN, e9.DATA, cache("R", "runs/exp10_eval/R")),
              "WF": e9.Fixed("WF", WF_RUN, e9.DATA, cache("WF", "runs/exp9_eval/WF"))}
    print("S epochs", models["S"].epochs, flush=True)
    r9 = json.load(open("runs/exp9_eval/result.json"))["selection"]["WF"]
    r10 = json.load(open("runs/exp10_eval/result.json"))["selection"]["R"]
    sel = {"WF": {"epoch": r9["epoch"], "threshold": r9["threshold"], "tau": 0.0},
           "R": {"epoch": r10["epoch"], "threshold": r10["threshold"], "tau": 0.0}}
    sets = {n: e8.openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dtld_tune = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "valid", "records.jsonl"))])
    test = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "dtld_test_records.jsonl"))])
    asets = e9.attribute_sets()
    if smoke:
        asets = {k: (e8.sample(v[0]), *v[1:]) for k, v in asets.items()}
    sel["S"] = e9.select("S", models["S"], sets, dtld_tune, (0.0,))
    res = {"selection": dict(sel)}
    for a, b in (("S", "R"), ("S", "WF")):
        tag = f"{a}_vs_{b}"
        res[f"guards_{tag}"] = e9.guards(models, sel, sets, test, a, b)
        print(tag, "guards", {k: (round(v["d_tss"], 4), [round(x, 4) for x in v["d_tss_ci95"]], round(v["d_false_go"], 4))
                              for k, v in res[f"guards_{tag}"].items()}, flush=True)
        res[f"attributes_{tag}"] = e9.attributes(models, sel, asets, sets, test, a, b)
        res[f"P2_{tag}"] = e10.small_lamps(models, sel, asets, test, a, b)
        dt = res[f"attributes_{tag}"]["DTLD_test"]
        print(tag, "P1", {k: v for k, v in dt["picto_macro"].items() if k != "n_by_group"},
              "P2", {k: v for k, v in res[f"P2_{tag}"].items() if k != "n_by_group"}, flush=True)
        if not smoke:
            res[f"detection_{tag}"] = e9.detection(models, sel, a, b)
    # decision on S - R with the Exp 10 rule (its key names are R vs WF)
    res["decision"] = {k.replace("R_replaces_WF", "S_replaces_R"): v for k, v in e10.decide(
        {"guards_R_vs_WF": res["guards_S_vs_R"], "attributes_R_vs_WF": res["attributes_S_vs_R"],
         "P2_R_vs_WF": res["P2_S_vs_R"]}).items()}
    print("decision", res["decision"], flush=True)
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
