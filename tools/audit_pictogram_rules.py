"""How much of the OpenLane-V2 direction labels a pictogram + rules method can recover, before any annotation.

OpenLane-V2 lights only carry a color; their moves come from the lanes they are linked to. A pictogram
method instead reads moves off the lamp: a circle governs the approach (straight, the far turn and the
U-turn by inheritance, never the near turn under US rules), an arrow governs its own direction(s).
Ground-truth relevance and color are used throughout; only each relevant light's moves are replaced:

- topo:   the topology moves (sanity check: must reproduce every label, TSS = 1);
- circle: every relevant light is a circle (moves = {"generic"}): no pictogram information at all;
- ideal:  lights whose topology moves have no straight ("turn-only": {left}, {left, uturn}, {right},
          {uturn}, ...) are arrows showing exactly those moves, every other light is a circle. This is
          the ceiling of a pictogram method, reached only if every turn-only light really is an arrow.

Frames where ``circle`` already gives the labels do not depend on the pictogram. The others
("dependent" frames) are right only if their turn-only lights are arrows, so the share of turn-only
lights that are circle-only dedicated heads (US) sets where between ``circle`` and ``ideal`` the method
lands. That share is what the annotation sample (``tools/make_pictogram_pool.py``) measures.

Writes data/labels/pictogram_audit.json and prints a summary.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import DIRECTIONS  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, to_arrays  # noqa: E402

L = "data/labels"
FILES = {f"{s}_{p}": f"{L}/subset_{s}_{p}.jsonl" for s in ("A", "B") for p in ("train", "val")}


def turn_only(moves):
    return not set(moves) & {"straight", "generic"}


def lights(rec):
    return [b for b in rec["boxes"] if b["relevant"]]


def decode(rec, scenario):
    dets = []
    for b in lights(rec):
        m = set(b["moves"])
        if scenario == "circle" or (scenario == "ideal" and not turn_only(m)):
            m = {"generic"}
        dets.append((m, b["color"], 1.0))
    return assign_scored(dets, rec["traffic_side"], rec.get("near_turn", "dedicated"))


def audit(recs):
    gc, gp = to_arrays([r["color"] for r in recs])
    out = {"frames": len(recs), "segments": len({r["segment_id"] for r in recs})}
    preds = {}
    for sc in ("topo", "circle", "ideal"):
        preds[sc] = [decode(r, sc) for r in recs]
        pc, pp = to_arrays(preds[sc])
        out[sc] = {k: round(v, 4) for k, v in evaluate(pc, pp, gc, gp).items()}
    dep = [i for i, r in enumerate(recs) if preds["circle"][i] != r["color"]]
    out["dependent_frames"] = len(dep)
    out["dependent_segments"] = len({recs[i]["segment_id"] for i in dep})
    out["dependent_not_fixed_by_ideal"] = sum(preds["ideal"][i] != recs[i]["color"] for i in dep)
    out["dependent_wrong_directions"] = dict(collections.Counter(
        d for i in dep for d in DIRECTIONS if preds["circle"][i][d] != recs[i]["color"][d]))
    rel = [b for r in recs for b in lights(r)]
    out["relevant_lights"] = len(rel)
    out["light_move_sets"] = dict(collections.Counter("+".join(b["moves"]) for b in rel).most_common())
    out["turn_only_lights"] = sum(turn_only(b["moves"]) for b in rel)
    out["dependent_by_turn_only_sets"] = dict(collections.Counter(
        " & ".join(sorted({"+".join(b["moves"]) for b in lights(recs[i]) if turn_only(b["moves"])})) or "(none)"
        for i in dep).most_common())
    out["turn_only_lights_in_dependent_frames"] = sum(turn_only(b["moves"]) for i in dep for b in lights(recs[i]))
    out["frames_with_turn_only_light"] = sum(any(turn_only(b["moves"]) for b in lights(r)) for r in recs)
    return out


def main():
    report = {}
    for name, path in FILES.items():
        recs = [json.loads(l) for l in open(path)]
        recs = [r for r in recs if r["kept"]]
        report[name] = a = audit(recs)
        print(f"{name}: {a['frames']} frames / {a['segments']} segments | TSS topo {a['topo']['tss']:.4f} "
              f"circle {a['circle']['tss']:.4f} (EM {a['circle']['em']:.4f}) ideal {a['ideal']['tss']:.4f} "
              f"(EM {a['ideal']['em']:.4f}) | dependent {a['dependent_frames']} frames / {a['dependent_segments']} "
              f"segments, not fixed by ideal {a['dependent_not_fixed_by_ideal']} | wrong dirs {a['dependent_wrong_directions']}")
        print(f"   relevant lights {a['relevant_lights']}, turn-only {a['turn_only_lights']} "
              f"(frames with one {a['frames_with_turn_only_light']}, in dependent frames "
              f"{a['turn_only_lights_in_dependent_frames']}) | dependent by turn-only sets {a['dependent_by_turn_only_sets']}")
        print(f"   move sets {a['light_move_sets']}")
    with open(f"{L}/pictogram_audit.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
