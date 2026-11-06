"""H1 of the pre-registered protocol (REPLICATION_PLAN.md): the pictogram + rules oracle on OpenLane-V2 val.

Written before any label existed. Reads the pool key (``make_pictogram_pool.py``) and the annotator's
export, and reports per subset:
- oracle TSS (and its parts): ground-truth relevance and color, each relevant light's moves from its
  annotated pictogram (circle -> generic, arrows -> their directions, other -> light dropped), US rules
  (``assign_scored``, near_turn dedicated). The estimate is exact on stratum D (every dependent frame)
  and uses the random N sample for the rest:  TSS = (sum_D s_i + n_rest * mean_N s_i) / n_frames, where
  s_i is a frame's (color acc + pass acc + EM) / 3 over its four directions (this equals the TSS
  definition of ``dtsu.metrics``). 95% interval: segment-cluster bootstrap of the N sample.
- primary analysis: unclear -> circle; sensitivity: frames with an unclear light left out;
- references on the same frames: every light a circle, and the "ideal" (turn-only lights = arrows);
- share of turn-only lights that are arrows, pictogram x topology-move-set table;
- oracle failures by type; QC self-agreement (raw and Cohen's kappa).
Writes ``--out`` (JSON).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_pictogram_rules import decode, turn_only  # noqa: E402
from dtsu import DIRECTIONS, color_to_pass  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored  # noqa: E402

PICTO_MOVES = {"circle": {"generic"}, "arrow_left": {"left"}, "arrow_straight": {"straight"}, "arrow_right": {"right"},
               "arrow_straight_left": {"straight", "left"}, "arrow_straight_right": {"straight", "right"},
               "arrow_uturn": {"uturn"}}


def frame_score(pred, gt):
    c = np.mean([pred[d] == gt[d] for d in DIRECTIONS])
    p = np.mean([color_to_pass(pred[d]) == color_to_pass(gt[d]) for d in DIRECTIONS])
    e = float(all(pred[d] == gt[d] and color_to_pass(pred[d]) == color_to_pass(gt[d]) for d in DIRECTIONS))
    return (c + p + e) / 3, c, p, e


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def oracle(rec, groups, unclear_as):
    dets = []
    for g in groups:
        lab = g["label"]
        if lab == "unclear":
            if unclear_as is None:
                return None
            lab = unclear_as
        if lab == "other":
            continue
        dets.append((PICTO_MOVES[lab], g["color"], 1.0))
    return assign_scored(dets, rec["traffic_side"], rec.get("near_turn", "dedicated"))


def failure_type(rec, groups, pred):
    if any(g["label"] == "circle" and turn_only(g["topology_moves"]) and set(g["topology_moves"]) <= {"left", "uturn"} for g in groups):
        return "circle-only head on a left-turn lane"
    if any(g["label"] == "circle" and set(g["topology_moves"]) == {"right"} for g in groups):
        return "circle-only head on a right-turn lane"
    if any(set(g["topology_moves"]) == {"left", "right"} for g in groups):
        return "T-junction (no straight) light"
    if any(g["label"].startswith("arrow") and not PICTO_MOVES[g["label"]] & (set(g["topology_moves"]) | {"generic"})
           and "generic" not in g["topology_moves"] for g in groups):
        return "arrow disagrees with topology"
    if any(g["label"] == "other" for g in groups):
        return "light labelled other"
    return "lane-specific heads / other"


def estimate(frames, n_total, n_rest, key, rng, draws):
    d = [f[key] for f in frames if f["stratum"] == "D" and f[key] is not None]
    n = [f for f in frames if f["stratum"] == "N" and f[key] is not None]
    if not n:
        return None
    sum_d = sum(x[0] for x in d)

    def est(sample):
        return (sum_d + n_rest * np.mean([f[key][0] for f in sample])) / n_total

    point = est(n)
    by_seg = collections.defaultdict(list)
    for f in n:
        by_seg[f["segment_id"]].append(f)
    segs = list(by_seg)
    boots = []
    for _ in range(draws):
        pick = rng.choice(len(segs), len(segs))
        boots.append(est([f for k in pick for f in by_seg[segs[k]]]))
    parts = {k: (sum(x[i] for x in d) + n_rest * np.mean([f[key][i] for f in n])) / n_total
             for i, k in ((1, "acc_color"), (2, "acc_pass"), (3, "em"))}
    return {"tss": float(point), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            **{k: float(v) for k, v in parts.items()}, "frames_D": len(d), "frames_N": len(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="data/annotation/pictogram_v1")
    ap.add_argument("--labels", default="annotations/openlane_pictograms_v1.json")
    ap.add_argument("--out", default="results/pictogram_oracle_v1.json")
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(args.pool, "items.jsonl"))]
    labels = json.load(open(args.labels))["labels"]
    main_items = [it for it in items if not it["qc_of"]]
    missing = [it["id"] for it in main_items if it["id"] not in labels]
    if missing:
        sys.exit(f"{len(missing)} of {len(main_items)} items have no label yet (first: {missing[:5]})")
    for it in main_items:
        it["label"] = labels[it["id"]]

    qc = [(labels[it["qc_of"]], labels[it["id"]]) for it in items if it["qc_of"] and it["id"] in labels]
    report = {"labels_file": args.labels, "items": len(main_items),
              "label_counts": dict(collections.Counter(it["label"] for it in main_items).most_common()),
              "qc": {"pairs": len(qc), "agreement": float(np.mean([a == b for a, b in qc])) if qc else None,
                     "kappa": kappa([a for a, _ in qc], [b for _, b in qc]) if qc else None}}

    rng = np.random.default_rng(0)
    for s in ("A", "B"):
        recs = [json.loads(l) for l in open(f"data/labels/subset_{s}_val.jsonl")]
        recs = {(r["segment_id"], r["timestamp"]): r for r in recs if r["kept"]}
        n_total = len(recs)
        n_dep = sum(decode(r, "circle") != r["color"] for r in recs.values())
        by_frame = collections.defaultdict(list)
        for it in main_items:
            if it["subset"] == s:
                by_frame[(it["segment_id"], it["timestamp"])].append(it)
        frames, failures = [], collections.Counter()
        for key, groups in by_frame.items():
            rec = recs[key]
            f = {"stratum": groups[0]["stratum"], "segment_id": rec["segment_id"]}
            for name, unclear_as in (("primary", "circle"), ("sensitivity", None)):
                pred = oracle(rec, groups, unclear_as)
                f[name] = None if pred is None else frame_score(pred, rec["color"])
                if name == "primary" and pred != rec["color"]:
                    failures[(f["stratum"], failure_type(rec, groups, pred))] += 1
            f["circle"] = frame_score(decode(rec, "circle"), rec["color"])
            f["ideal"] = frame_score(decode(rec, "ideal"), rec["color"])
            frames.append(f)
        n_rest = n_total - n_dep
        res = {"frames": n_total, "dependent_frames": n_dep,
               **{k: estimate(frames, n_total, n_rest, k, rng, args.draws) for k in ("primary", "sensitivity", "circle", "ideal")},
               "failures": {f"{st} | {t}": n for (st, t), n in failures.most_common()}}
        turn = [it for it in main_items if it["subset"] == s and turn_only(it["topology_moves"])]
        res["turn_only_lights"] = len(turn)
        res["turn_only_share_arrow"] = float(np.mean([it["label"].startswith("arrow") for it in turn])) if turn else None
        res["pictogram_by_topology_moves"] = {
            m: dict(collections.Counter(it["label"] for it in main_items if it["subset"] == s and "+".join(it["topology_moves"]) == m).most_common())
            for m in sorted({"+".join(it["topology_moves"]) for it in main_items if it["subset"] == s})}
        report[s] = res
        p = res["primary"]
        print(f"{s}: oracle TSS {p['tss']:.4f} [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] EM {p['em']:.4f} | "
              f"sensitivity {res['sensitivity']['tss']:.4f} | circle {res['circle']['tss']:.4f} ideal {res['ideal']['tss']:.4f} | "
              f"turn-only lights that are arrows {res['turn_only_share_arrow']}")
        print(f"   failures {res['failures']}")
    print(f"labels {report['label_counts']} | QC {report['qc']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
