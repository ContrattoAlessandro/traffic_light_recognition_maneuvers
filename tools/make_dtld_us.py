"""Build D*_US (DTLD relabelled with the OpenLane-V2 protocol) and audit it against the DTLD labels.

Reads data/labels/dtld_{train,test}.jsonl (German rules, ``dtsu.labels_dtld``) and writes
data/labels/dtld_us_{train,test}.jsonl plus the fit / tune split of train (same sessions as
data/labels/tune/dtld_train_{fit,tune}.jsonl), each record converted by
``labels_dtld.to_openlane_protocol``.

Audit, per split (printed and written to data/labels/dtld_us_audit.json):
- kept frames before / after and why frames enter or leave;
- per direction, old -> new color transitions on frames kept by both;
- per direction color mix of D*_US next to OpenLane-V2 A+B (same protocol, other country);
- oracle: ground-truth lamps through ``assign_scored`` with the US rule must give every D*_US label.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.labels_dtld import to_openlane_protocol  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored  # noqa: E402

L = "data/labels"


def read(path):
    return [json.loads(l) for l in open(path)]


def write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def mix(recs):
    out = {}
    for d in DIRECTIONS:
        c = collections.Counter(r["color"][d] for r in recs)
        out[d] = {k: round(c[k] / len(recs), 4) for k in ("none", "red", "yellow", "green")}
    return out


def why_dropped(r):
    if r["rule_ok"] is False:
        return "rule_mismatch"
    if r["n_relevant_lights"] == 0:
        return "no_relevant_light"
    if r["conflict"]:
        return "conflict"
    if r.get("unknown"):
        return "unknown_state"
    return "?"


def audit(old, new):
    both = [(o, n) for o, n in zip(old, new) if o["kept"] and n["kept"]]
    a = {
        "frames": len(old),
        "kept_dtld": sum(o["kept"] for o in old),
        "kept_us": sum(n["kept"] for n in new),
        "kept_both": len(both),
        "only_dtld (why dropped in US)": dict(collections.Counter(why_dropped(n) for o, n in zip(old, new) if o["kept"] and not n["kept"])),
        "only_us (why dropped in DTLD)": dict(collections.Counter(why_dropped(o) for o, n in zip(old, new) if n["kept"] and not o["kept"])),
        "us_kept_with_ignored_unknown_lamps": sum(n["kept"] and n["had_unknown"] for n in new),
        "changed_frames_kept_both": sum(o["color"] != n["color"] for o, n in both),
        "transitions_kept_both": {},
    }
    for d in DIRECTIONS:
        t = collections.Counter(f"{o['color'][d]}->{n['color'][d]}" for o, n in both if o["color"][d] != n["color"][d])
        a["transitions_kept_both"][d] = dict(t.most_common())
    kept_new = [n for n in new if n["kept"]]
    a["color_mix_us"] = mix(kept_new)
    a["color_mix_dtld"] = mix([o for o in old if o["kept"]])
    bad = 0
    for n in kept_new:
        dets = [(set(b["moves"]), b["color"], 1.0) for b in n["boxes"] if b["relevant"]]
        bad += assign_scored(dets, "right", "dedicated") != n["color"]
    a["oracle_mismatch"] = bad
    return a


def main():
    report = {}
    for split in ("train", "test"):
        old = read(f"{L}/dtld_{split}.jsonl")
        new = [to_openlane_protocol(r) for r in old]
        write(f"{L}/dtld_us_{split}.jsonl", new)
        report[split] = audit(old, new)
        if split == "train":
            by_img = {(r["segment_id"], r["image"]): r for r in new}
            for part in ("fit", "tune"):
                recs = [by_img[(r["segment_id"], r["image"])] for r in read(f"{L}/tune/dtld_train_{part}.jsonl")]
                write(f"{L}/tune/dtld_us_train_{part}.jsonl", recs)
                report[f"train_{part}_kept_us"] = sum(r["kept"] for r in recs)
    ol = [r for s in ("A_train", "B_train", "A_val", "B_val") for r in read(f"{L}/subset_{s}.jsonl") if r["kept"]]
    report["color_mix_openlane_AB"] = mix(ol)
    with open(f"{L}/dtld_us_audit.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
