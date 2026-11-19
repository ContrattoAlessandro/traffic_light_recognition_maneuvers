"""Exp 10 (REPLICATION_PLAN.md): WF at 1280 (arm R, runs/exp10_R) against WF at 1024 (Exp 9) and F8.

Selection: the Exp 8 rule (tools/eval_exp9.select with tau 0). WF and F8 keep their Exp 9 / Exp 8 selections.
Each model is decoded at its own training resolution (tools/eval_exp8.Model reads args.json).
- guards G1 / G2 (R - WF, and R - F8 reported): ΔTSS and ΔFalse-Go on A val, B val, DTLD test;
- attributes on the lights both models match: P1 = DTLD test picto macro, P2 = the same on lamps under 20 px,
  G3 = DTLD color accuracy, G4 = false arrows on the hand-labelled OpenLane circles;
- the Exp 8 perception cells.
Writes runs/exp10_eval/result.json (runs/exp10_eval_smoke with --limit).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp8 as e8  # noqa: E402
import tools.eval_exp9 as e9  # noqa: E402

R_RUN, WF_RUN = "runs/exp10_R", "runs/exp9_WF"
OUT = "runs/exp10_eval"


def small_lamps(models, sel, asets, test, a, b, max_px=20):
    """P2: DTLD test picto macro on matched lamps under max_px tall."""
    items, kind, key, size = asets["DTLD_test"]
    rows = e9.light_rows(items, e9.preds_for(models[a], sel[a], kind, key, items, size, test), sel[a]["threshold"],
                         e9.preds_for(models[b], sel[b], kind, key, items, size, test), sel[b]["threshold"])
    sub = {k: v[rows["height"] < max_px] for k, v in rows.items()}
    return e9.metric(sub, "picto_macro", e9.P1_CLASSES)


def decide(res):
    g, at = res["guards_R_vs_WF"], res["attributes_R_vs_WF"]
    sets3 = [g[n] for n in ("A_val", "B_val", "DTLD_test")]
    lo = e9.lo
    d = {"G1_tss": all(c["d_tss_ci95"][0] > -0.01 for c in sets3),
         "G2_false_go": all(c["d_false_go"] <= 0.01 for c in sets3),
         "G3_color_acc": lo(at["DTLD_test"]["color_acc"]) > -0.005,
         "G4_false_arrows": at["OpenLane_hand_labelled"]["false_arrow_on_circles"]["d"] <= 0.03,
         "P1_picto_macro": lo(at["DTLD_test"]["picto_macro"]) > 0,
         "P2_picto_macro_under_20px": lo(res["P2_R_vs_WF"]) > 0}
    d["R_replaces_WF"] = (d["G1_tss"] and d["G2_false_go"] and d["G3_color_acc"] and d["G4_false_arrows"]
                          and (d["P1_picto_macro"] or d["P2_picto_macro_under_20px"]))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run", default=R_RUN, help="arm R's run folder (a smoke test can point at another run)")
    ap.add_argument("--limit", type=int, help="smoke test: evenly spaced N frames per set")
    args = ap.parse_args()
    global OUT
    e8.WORKERS = e9.WORKERS = args.workers
    if args.limit:
        e8.LIMIT = args.limit
        OUT = "runs/exp10_eval_smoke"
    os.makedirs(OUT, exist_ok=True)
    smoke = bool(args.limit)

    r8 = json.load(open(os.path.join(e9.F8_RUN, "exp8_eval", "result.json")))["selection"]["F8"]
    r9 = json.load(open("runs/exp9_eval/result.json"))["selection"]["WF"]
    sel = {"F8": {"epoch": r8["epoch"], "threshold": r8["threshold"], "tau": 0.0},
           "WF": {"epoch": r9["epoch"], "threshold": r9["threshold"], "tau": 0.0}}
    cache = lambda name, real: os.path.join(OUT, name) if smoke else real  # noqa: E731
    models = {"F8": e9.Fixed("F8", e9.F8_RUN, e9.DATA, cache("F8", os.path.join(e9.F8_RUN, "exp8_eval"))),
              "WF": e9.Fixed("WF", WF_RUN, e9.DATA, cache("WF", "runs/exp9_eval/WF")),
              "R": e9.Fixed("R", args.run, e9.DATA, os.path.join(OUT, "R"))}

    sets = {n: e8.openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dtld_tune = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "valid", "records.jsonl"))])
    test = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "dtld_test_records.jsonl"))])
    asets = e9.attribute_sets()
    if smoke:
        asets = {k: (e8.sample(v[0]), *v[1:]) for k, v in asets.items()}

    sel["R"] = e9.select("R", models["R"], sets, dtld_tune, (0.0,))
    res = {"selection": dict(sel)}
    for b in ("WF", "F8"):
        res[f"guards_R_vs_{b}"] = e9.guards(models, sel, sets, test, "R", b)
        print("R vs", b, "guards", {k: (round(v["d_tss"], 4), [round(x, 4) for x in v["d_tss_ci95"]], round(v["d_false_go"], 4))
                                    for k, v in res[f"guards_R_vs_{b}"].items()}, flush=True)
        res[f"attributes_R_vs_{b}"] = e9.attributes(models, sel, asets, sets, test, "R", b)
        res[f"P2_R_vs_{b}"] = small_lamps(models, sel, asets, test, "R", b)
        dt = res[f"attributes_R_vs_{b}"]["DTLD_test"]
        print("R vs", b, "P1", {k: v for k, v in dt["picto_macro"].items() if k != "n_by_group"},
              "P2", {k: v for k, v in res[f"P2_R_vs_{b}"].items() if k != "n_by_group"}, flush=True)
        if not smoke:
            res[f"detection_R_vs_{b}"] = e9.detection(models, sel, "R", b)
    res["decision"] = decide(res)
    print("decision", res["decision"], flush=True)
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
