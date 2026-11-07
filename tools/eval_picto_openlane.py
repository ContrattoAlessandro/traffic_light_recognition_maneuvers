"""Exp 6 models (trained on DTLD only) on the OpenLane-V2 benchmark frames: A val and B val.

Each arm runs at its DTLD-tune epoch and threshold; detections -> class-agnostic NMS 0.5 -> the junction
rule on the lights the model assigns to the junction (dtsu.picto_rules.rule). U-turn follows the far turn:
left in right-hand traffic, right in left-hand traffic (Singapore, from the record's traffic_side).
Scored against the benchmark's per-direction labels (HD-map topology) two ways:
- all: the four directions as predicted. The rule gives every direction a color whenever a junction light
  is seen, while the label says "none" for a maneuver no light governs (e.g. not allowed from the ego
  lane), so this under-rates the rule;
- map-masked: directions labelled "none" are set to "none" in the prediction too, i.e. the HD map's
  allowed maneuvers are assumed known (as in the junction-rule audit). "none" in the label also covers
  a maneuver allowed but governed by no light, so the mask leaks a little label information.
Writes runs/exp6_picto_<arm>/openlane/result.json (+ cached raw detections).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.data import load_records  # noqa: E402
from tools.eval_hybrid import SETS, per_frame  # noqa: E402
from tools.eval_picto import directions, kept, load_model, raw_detections  # noqa: E402

RUNS = {"J": "runs/exp6_picto_J", "F": "runs/exp6_picto_F"}


def predict(d, thr, side):
    out = directions(kept(d, thr))
    out["uturn"] = out["left"] if side == "right" else out["right"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=["J", "F"])
    ap.add_argument("--sets", nargs="+", default=["A_val", "B_val"])
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco_dtld_picto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    class_names = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    for arm in args.arms:
        run = RUNS[arm]
        sel = json.load(open(os.path.join(run, "picto_eval", "result.json")))
        epoch, thr = sel["epoch"], sel["threshold"]
        out_dir = os.path.join(run, "openlane")
        os.makedirs(out_dir, exist_ok=True)
        out = {"run": run, "epoch": epoch, "threshold": thr}
        net = None
        for name in args.sets:
            recs = load_records(*SETS[name])[: args.limit]
            for r in recs:
                r["image"] = os.path.abspath(os.path.join(r["root"], r["image"]))
            cache = os.path.join(out_dir, f"raw_e{epoch}_{name}.pt")
            if args.limit is None and os.path.exists(cache):
                dets = torch.load(cache, weights_only=False)
            else:
                if net is None:
                    net, res = load_model(run, epoch, class_names, arm)
                dets = raw_detections(net, recs, class_names, res, arm, args.workers)
                if args.limit is None:
                    torch.save(dets, cache)
            preds = [predict(d, thr, r["traffic_side"]) for d, r in zip(dets, recs)]
            masked = [{k: ("none" if r["color"][k] == "none" else v) for k, v in p.items()} for p, r in zip(preds, recs)]
            res_set = {"frames": len(recs)}
            for kind, pr in (("all", preds), ("map_masked", masked)):
                res_set[kind] = per_frame(pr, recs)[3]
            for side in ("right", "left"):
                k = [i for i, r in enumerate(recs) if r["traffic_side"] == side]
                if k and len(k) < len(recs):
                    res_set[f"map_masked_{side}_traffic"] = per_frame([masked[i] for i in k], [recs[i] for i in k])[3] | {"frames": len(k)}
            out[name] = res_set
            print(arm, name, json.dumps({k: ({m: round(v, 4) for m, v in x.items() if isinstance(v, float)} if isinstance(x, dict) else x)
                                         for k, x in res_set.items()}), flush=True)
        if net is not None:
            del net
            torch.cuda.empty_cache()
        if args.limit is None:
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
