"""Collect the three-direction re-scores (tools/rescore_three_dirs.sh) into one summary for the thesis.

Prints every number the thesis tables use and writes results/three_directions/summary.json.
"""
import json
import os

R = "runs"
OUT = "results/three_directions"
KEYS = ("acc_color", "acc_pass", "em", "tss", "false_go", "false_stop")


def load(p):
    return json.load(open(os.path.join(R, p)))


def jsonl(p):
    return [json.loads(l) for l in open(os.path.join(R, p))]


def row(m, keys=KEYS):
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)


def main():
    s = {}
    for name, run in (("swin", "swin_t_AB"), ("convnext", "convnextv2_t_AB")):
        d = load(f"{run}/per_direction_3dir.json")
        s[name] = {k: d[k]["overall"] for k in ("A", "B", "combined")}
        for k in ("A", "B", "combined"):
            print(f"{name:9s} {k:8s} {row(d[k]['overall'])}")
    rf = jsonl("rfdetr_m_AB/val_log_3dir.jsonl")
    s["rfdetr_epochs"] = [{"epoch": e["epoch"], **{k: e[k] for k in ("A", "B", "combined")}} for e in rf]
    for e in rf:
        print(f"rfdetr ep{e['epoch']} A {e['A']['tss']:.4f} B {e['B']['tss']:.4f} A+B {e['combined']['tss']:.4f} "
              f"FG {e['combined']['false_go']:.4f} FS {e['combined']['false_stop']:.4f} | A {row(e['A'])}")
    b = load("bootstrap_swin_vs_rfdetr_3dir.json")
    s["bootstrap"] = {k: b[k] for k in ("A", "B", "A+B")}
    for sc in ("A", "B", "A+B"):
        for m in ("tss", "em", "acc_pass", "false_go", "false_stop"):
            x = b[sc][m]
            print(f"boot {sc:4s} {m:10s} {x['a']:.4f} {x['b']:.4f} {x['diff']:+.4f} "
                  f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] p={x['p']:.2f}")
    s["dtld"] = {}
    for tag, ep in (("_3dir_e4", 4), ("_3dir", 14)):
        e = [x for x in jsonl(f"rfdetr_m_dtld/val_log{tag}.jsonl") if x["epoch"] == ep][-1]
        s["dtld"][ep] = {"threshold": e["threshold"], **e["DTLD_test"]}
        print(f"dtld ep{ep} thr {e['threshold']} {row(e['DTLD_test'])}")
    s["ood"] = {}
    for t in ("lisa", "openlane"):
        for g in ("squeeze", "pad2"):
            d = load(f"rfdetr_m_dtld/ood_{t}/epoch4_{g}_3dir.json")
            s["ood"][f"{t}_{g}"] = d
            print(f"ood {t} {g} thresholds {d['thresholds']} dropped {d.get('dropped_by_oracle')}")
            for th in ("source", "tuned"):
                for k, rep in d[th].items():
                    extra = ""
                    if "day_no_right" in rep:
                        extra = (f" | day {rep['day_no_right']['tss']:.4f} ({rep['day_frames']}) "
                                 f"night {rep['night_no_right']['tss']:.4f} ({rep['night_frames']})")
                    fg = " ".join(f"{dn} FG {v['false_go']:.3f}" for dn, v in rep["directions"].items())
                    print(f"   {th:6s} {k:10s} n={rep['frames']} {row(rep['overall'])}\n"
                          f"          no_right {row(rep['no_right'])}{extra}\n          {fg}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
