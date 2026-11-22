"""Build every label file the training export and the evaluation read, under data/labels/.

1. OpenLane-V2 maneuver labels (``dtsu.labels``) for subset A and B, train and val, US rules everywhere (the
   convention that reproduces the label distribution of Zou et al.). Train is then split by segment into a fit
   split (training) and a tune split (model selection) with ``splits/openlane_tune_segments.json``.
2. DTLD junction-rule records (``dtsu.sources.dtld_records``): fit and tune from DTLD train, split by session with
   ``splits/dtld_tune_sessions.json``, and test.
3. For the baseline M: DTLD route labels under German rules and D*_US, the same frames under the OpenLane-V2 protocol
   (``dtsu.labels_dtld``), with the same fit / tune / test split.

Usage (from the repository root): python scripts/build_labels.py [--skip-openlane] [--skip-dtld]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import labels_dtld, sources  # noqa: E402
from dtsu.paths import LABELS, OPENLANE, SPLITS  # noqa: E402


def openlane():
    tune = json.load(open(os.path.join(SPLITS, "openlane_tune_segments.json")))
    for s in ("A", "B"):
        root = os.path.join(OPENLANE, f"subset_{s}")
        data_dict = os.path.join(root, f"data_dict_subset_{s}.json")
        for split in ("train", "val"):
            out = os.path.join(LABELS, f"subset_{s}_{split}.jsonl")
            cmd = [sys.executable, "-m", "dtsu.labels", "--root", root, "--split", split, "--out", out]
            if os.path.exists(data_dict):
                cmd += ["--data-dict", data_dict]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        held = set(tune[f"subset_{s}"])
        lines = open(os.path.join(LABELS, f"subset_{s}_train.jsonl")).read().splitlines()
        with open(os.path.join(LABELS, f"subset_{s}_train_fit.jsonl"), "w") as f_fit, \
                open(os.path.join(LABELS, f"subset_{s}_train_tune.jsonl"), "w") as f_tune:
            for line in lines:
                (f_tune if json.loads(line)["segment_id"] in held else f_fit).write(line + "\n")
        print(f"subset_{s}: {len(held)} tune segments", flush=True)


def dtld():
    tune = sources.dtld_tune_sessions()
    train = sources.dtld_records("train")
    fit_sessions = {r["segment_id"] for r in train} - tune
    fit = [r for r in train if r["segment_id"] not in tune]
    sources.write_jsonl(sources.dtld_junction_path("fit"), fit)
    sources.write_jsonl(sources.dtld_junction_path("tune"), [r for r in train if r["segment_id"] in tune])
    sources.write_jsonl(sources.dtld_junction_path("test"), sources.dtld_records("test"))
    print(f"DTLD: {len(fit)} fit frames ({len(fit_sessions)} sessions), {len(train) - len(fit)} tune frames ({len(tune)} sessions)", flush=True)


def dtld_routes():
    tune = sources.dtld_tune_sessions()
    train, test = labels_dtld.route_records("train"), labels_dtld.route_records("test")
    parts = {"fit": [r for r in train if r["segment_id"] not in tune], "tune": [r for r in train if r["segment_id"] in tune], "test": test}
    for part, recs in parts.items():
        sources.write_jsonl(labels_dtld.label_path("route", part), recs)
        us = [labels_dtld.to_openlane_protocol(r) for r in recs]
        sources.write_jsonl(labels_dtld.label_path("us", part), us)
        print(f"DTLD routes {part}: {len(recs)} frames, kept {sum(r['kept'] for r in recs)} (German rules), "
              f"{sum(r['kept'] for r in us)} (D*_US)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-openlane", action="store_true")
    ap.add_argument("--skip-dtld", action="store_true")
    ap.add_argument("--skip-dtld-routes", action="store_true", help="skip the route labels of the baseline M")
    args = ap.parse_args()
    os.makedirs(LABELS, exist_ok=True)
    if not args.skip_openlane:
        openlane()
    if not args.skip_dtld:
        dtld()
    if not args.skip_dtld_routes:
        dtld_routes()


if __name__ == "__main__":
    main()
