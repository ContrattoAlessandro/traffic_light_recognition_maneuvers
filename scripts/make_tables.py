"""Print the README / paper tables (Markdown) from the evaluation files in results/.

  python scripts/make_tables.py                      # main results of R
  python scripts/make_tables.py --resolution         # resolution ablation
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MANEUVER_ROWS = (("A_val", "OpenLane-V2 A val, US rule + lane-topology prior"), ("A_val_camera_only", "OpenLane-V2 A val, US rule, camera only"),
                 ("B_val", "OpenLane-V2 B val, US rule + lane-topology prior"), ("B_val_camera_only", "OpenLane-V2 B val, US rule, camera only"),
                 ("DTLD_test", "DTLD test, junction rule (German)"))
DETECTION = (("atlas_test", "ATLAS test"), ("lisa_day", "LISA test, day"), ("lisa_night", "LISA test, night"),
             ("s2tld_720p", "S2TLD 720p"), ("s2tld_1080p", "S2TLD 1080p"), ("s2tld_1080p_horizontal", "S2TLD 1080p, horizontal lights"),
             ("bdd_day", "BDD100K val, day"), ("bdd_night", "BDD100K val, night"), ("bdd_horizontal", "BDD100K val, horizontal lights"),
             ("vzc_all", "VZC-TLD"), ("vzc_horizontal", "VZC-TLD, horizontal lights"))
TRAINED_ON = {"atlas_test", "bdd_day", "bdd_night", "bdd_horizontal"}


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def a(cell):
    return None if cell is None else cell["a"]


def main_tables(r):
    print("**Per-maneuver signals** (straight, left, right). TSS = (Acc_color + Acc_pass + EM) / 3.\n")
    print("| Test set | Frames | TSS | Acc_color | EM | False-Go | False-Stop |")
    print("|---|---|---|---|---|---|---|")
    for key, label in MANEUVER_ROWS:
        m, n = r["maneuver_metrics"]["a"][key], r["maneuvers"][key]["frames"]
        print(f"| {label} | {n:,} | {f(m['tss'])} | {f(m['acc_color'])} | {f(m['em'])} | {f(m['false_go'])} | {f(m['false_stop'])} |")
    at = r["attributes"]
    d, atl, lisa = at["DTLD_test"], at["ATLAS_test"], at["LISA_test"]
    print("\n**Per-light attributes**, on the labelled lights the model detects (center matching).\n")
    print("| Attribute | DTLD test | ATLAS test | LISA test (out of domain) |")
    print("|---|---|---|---|")
    print(f"| Pictogram, macro recall | {f(a(d['picto_macro']))} | {f(a(atl['picto_macro']))} | {f(a(lisa['picto_macro']))} |")
    for c in ("circle", "arrow_left", "arrow_right", "arrow_straight"):
        row = [f(a(x["picto_recall"].get(c))) for x in (d, atl, lisa)]
        print(f"| Recall of {c.replace('_', ' ')} | " + " | ".join(row) + " |")
    print(f"| Color accuracy | {f(a(d['color_acc']))} | {f(a(atl['color_acc']))} | {f(a(lisa['color_acc']))} |")
    print(f"| Yellow recall | {f(a(d['yellow_recall']))} | {f(a(atl['yellow_recall']))} | {f(a(lisa['yellow_recall']))} |")
    print(f"| Junction accuracy | {f(a(d['junction_acc']))} | – | – |")
    hand = at["OpenLane_hand_labelled"]["false_arrow_on_circles"]
    print(f"\nOf the {at['OpenLane_hand_labelled']['lights']:,} hand-labelled OpenLane-V2 val lights the model detects, {f(a(hand))} of the "
          f"round ones are read as arrows. Pictogram macro recall on DTLD lamps under 20 px tall: {f(a(d['picto_macro_under_20px']))}.")
    print("\n**Detection** (center AP; ✱ = the training set includes this dataset's training split).\n")
    print("| Test set | Lights | AP | Recall | Precision | Color accuracy |")
    print("|---|---|---|---|---|---|")
    for key, label in DETECTION:
        c = r["detection"][key]
        mark = " ✱" if key in TRAINED_ON else ""
        print(f"| {label}{mark} | {c['lights']:,} | {f(c['a_AP'])} | {f(c['a']['recall'])} | {f(c['a']['precision'])} | {f(c['a']['color_acc'])} |")


def resolution_table():
    runs = {1024: load("ablation_resolution/1024.json"), 1280: load("R.json"), 1536: load("ablation_resolution/1536.json")}
    pairs = {1024: load("ablation_resolution/R_vs_1024.json"), 1536: load("ablation_resolution/R_vs_1536.json")}
    latency = {1024: 22, 1280: 29.5, 1536: 54}
    print("| Input | Selected epoch / threshold | TSS A val | TSS B val | TSS DTLD test | False-Go A / B / DTLD | DTLD pictogram macro | ... under 20 px | Latency (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for res, r in runs.items():
        m = r["maneuver_metrics"]["a"]
        d = r["attributes"]["DTLD_test"]
        fg = " / ".join(f(m[k]["false_go"]) for k in ("A_val", "B_val", "DTLD_test"))
        print(f"| {res}{' (R)' if res == 1280 else ''} | {r['selection']['epoch']} / {r['selection']['threshold']} | {f(m['A_val']['tss'])} | "
              f"{f(m['B_val']['tss'])} | {f(m['DTLD_test']['tss'])} | {fg} | {f(a(d['picto_macro']))} | {f(a(d['picto_macro_under_20px']))} | {latency[res]} |")
    print("\nR minus the other resolution, paired on the same frames / lights, with cluster-bootstrap 95% CIs:\n")
    print("| | ΔTSS A val | ΔTSS B val | ΔTSS DTLD test | ΔFalse-Go B val | ΔPictogram macro (DTLD) |")
    print("|---|---|---|---|---|---|")
    ci = lambda x, k="d_tss": f"{x[k]:+.3f} [{x[k + '_ci95'][0]:+.3f}, {x[k + '_ci95'][1]:+.3f}]"  # noqa: E731
    for res, p in pairs.items():
        mv, pm = p["maneuvers"], p["attributes"]["DTLD_test"]["picto_macro"]
        print(f"| 1280 − {res} | {ci(mv['A_val'])} | {ci(mv['B_val'])} | {ci(mv['DTLD_test'])} | {ci(mv['B_val'], 'd_false_go')} | "
              f"{pm['d']:+.3f} [{pm['d_ci95'][0]:+.3f}, {pm['d_ci95'][1]:+.3f}] |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolution", action="store_true")
    args = ap.parse_args()
    resolution_table() if args.resolution else main_tables(load("R.json"))


if __name__ == "__main__":
    main()
