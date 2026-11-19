"""How often does a "junction lights + pictogram + color" rule fail on OpenLane-V2 and DTLD?

The rule under study (no lane-to-light association; the HD map gives the maneuvers the ego lane allows):
- take every vehicle signal head of the junction approach the ego is on (not only the ego lane's);
- an arrow sets the color of its own maneuvers;
- every maneuver without an arrow takes the color of the circles.
It needs, per frame, all circles to show one color and the arrows of one maneuver to agree.

Per frame (every frame with at least one colored junction light, kept by the benchmark or not):
- ok_single_color: all junction lights show one color (the rule works whatever the pictograms are);
- ok_arrows: several colors, separated by arrows, circles agree (and, on OpenLane, the answer matches
  the topology label on every labelled direction);
- fail_circles_disagree: circles of different colors (lane-specific heads, circle-only turn heads);
- fail_arrows_disagree: arrows for the same maneuver with different colors;
- fail_unreadable: a light whose pictogram cannot be read shows a color that differs from the others;
- fail_rule_vs_label (OpenLane): circles agree but the answer contradicts the topology label;
- multi_color_unlabelled (OpenLane): several colors but pictograms unknown (train splits, and val frames
  outside the annotated pool); estimated from the labelled val frames.

Junction lights:
- OpenLane-V2: the lights the label builder calls relevant, i.e. linked to any lane of the ego approach
  at the nearest junction (all lanes heading the ego's way, not just the ego lane). Pictograms come
  from the blind annotation of 1,231 val lights (annotations/openlane_pictograms_v1.json; other ->
  dropped, unclear -> unreadable). Subset B is split by traffic side (data/labels/variants/*_auto.jsonl):
  left = Singapore, right = Boston.
- DTLD: front-facing vehicle lamps of the ego intersection as in dtsu/labels_dtld.vehicle_group
  (relevant lamps plus non-relevant ones >= 0.6 x median relevant height), with DTLD's own pictogram
  ("unknown" -> unreadable) and state (red_yellow -> red; off / unknown -> no color). Pedestrian,
  bicycle and tram signals are left out. Also reported with the relevant lamps only.
Writes results/junction_rule_audit.json.
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DTLD_ROOT = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD/v2.0")
ARROWS = {"arrow_left": {"left"}, "arrow_right": {"right"}, "arrow_straight": {"straight"},
          "arrow_straight_left": {"straight", "left"}, "arrow_straight_right": {"straight", "right"}, "arrow_uturn": {"uturn"}}
STATE = {"red": "red", "red_yellow": "red", "yellow": "yellow", "green": "green"}
ORDER = ["ok_single_color", "ok_arrows", "fail_circles_disagree", "fail_arrows_disagree", "fail_unreadable",
         "fail_rule_vs_label", "multi_color_unlabelled"]


def classify(lights):
    """lights: (pictogram, color) with pictogram in circle / ARROWS / unreadable and color not None.
    Returns (category, per-maneuver colors or None)."""
    if len({c for _, c in lights}) == 1:
        return "ok_single_color", None
    circles = {c for p, c in lights if p == "circle"}
    if len(circles) > 1:
        return "fail_circles_disagree", None
    arrow = collections.defaultdict(set)
    for p, c in lights:
        for m in ARROWS.get(p, ()):
            arrow[m].add(c)
    if any(len(v) > 1 for v in arrow.values()):
        return "fail_arrows_disagree", None
    known = {c for p, c in lights if p != "unreadable"}
    if any(p == "unreadable" and (c not in known or len(known) > 1) for p, c in lights):
        return "fail_unreadable", None
    circle = circles.pop() if circles else None
    out = {m: (arrow[m].pop() if arrow.get(m) else circle) for m in ("straight", "left", "right")}
    return "ok_arrows", out


def openlane():
    picto = json.load(open("annotations/openlane_pictograms_v1.json"))["labels"]
    by_frame = collections.defaultdict(dict)  # (segment, timestamp) -> box index -> pictogram
    for line in open("data/annotation/pictogram_v1/items.jsonl"):
        it = json.loads(line)
        if it["qc_of"] is None and it["id"] in picto:
            for b in it["box_indices"]:
                by_frame[(it["segment_id"], it["timestamp"])][b] = picto[it["id"]]
    side = {}
    for split in ("train", "val"):
        for line in open(f"data/labels/variants/subset_B_{split}_auto.jsonl"):
            r = json.loads(line)
            side[(r["segment_id"], r["timestamp"])] = r["traffic_side"]

    res, examples = {}, collections.defaultdict(list)
    for subset in ("A", "B"):
        for split in ("train", "val"):
            for line in open(f"data/labels/subset_{subset}_{split}.jsonl"):
                r = json.loads(line)
                rel = [(i, b) for i, b in enumerate(r["boxes"]) if b["relevant"]]
                if not rel:
                    continue
                region = "A (US)" if subset == "A" else ("B Singapore" if side[(r["segment_id"], r["timestamp"])] == "left" else "B Boston")
                key = (r["segment_id"], r["timestamp"])
                labels = by_frame.get(key, {})
                colors = {b["color"] for _, b in rel}
                if len(colors) == 1:
                    cat = "ok_single_color"
                elif split == "val" and all(i in labels for i, _ in rel):
                    pic = {"other": None, "unclear": "unreadable"}
                    lights = [(pic.get(labels[i], labels[i]), b["color"]) for i, b in rel if pic.get(labels[i], labels[i])]
                    cat, out = classify(lights) if lights else ("ok_single_color", None)
                    if cat == "ok_arrows":
                        lab = r["color"]
                        if any(lab[d] != "none" and out[d] is not None and out[d] != lab[d] for d in lab):
                            cat = "fail_rule_vs_label"
                else:
                    cat = "multi_color_unlabelled"
                row = res.setdefault(region, {"frames_with_junction_light": 0, "benchmark_conflict": 0,
                                               "by_split": {}, **{c: 0 for c in ORDER}})
                row["frames_with_junction_light"] += 1
                row[cat] += 1
                row["benchmark_conflict"] += bool(r["conflict"])
                sp = row["by_split"].setdefault(split, {c: 0 for c in ORDER})
                sp[cat] += 1
                if cat.startswith("fail") and len(examples[(region, cat)]) < 5:
                    examples[(region, cat)].append(f"{subset}/{r['image']}")
    return res, {f"{k[0]} | {k[1]}": v for k, v in examples.items()}


def dtld():
    res, examples = {}, collections.defaultdict(list)
    for split in ("train", "test"):
        for im in json.load(open(os.path.join(DTLD_ROOT, f"DTLD_{split}.json")))["images"]:
            front = [l for l in im["labels"] if l["attributes"]["direction"] == "front"
                     and l["attributes"]["pictogram"] in set(ARROWS) | {"circle", "unknown"}]
            rel = [l for l in front if l["attributes"]["relevance"] == "relevant"]
            if not rel:
                continue
            h = statistics.median(l["h"] for l in rel)
            group = rel + [l for l in front if l["attributes"]["relevance"] != "relevant" and l["h"] >= 0.6 * h]
            city = im["image_path"].replace("\\", "/").split("/")
            for scope, lamps in (("junction", group), ("relevant only", rel)):
                lights = [("unreadable" if l["attributes"]["pictogram"] == "unknown" else l["attributes"]["pictogram"],
                           STATE[l["attributes"]["state"]]) for l in lamps if l["attributes"]["state"] in STATE]
                if not lights:
                    continue
                cat, _ = classify(lights)
                row = res.setdefault(scope, {"frames_with_junction_light": 0, "by_split": {}, **{c: 0 for c in ORDER}})
                row["frames_with_junction_light"] += 1
                row[cat] += 1
                sp = row["by_split"].setdefault(split, {c: 0 for c in ORDER})
                sp[cat] += 1
                if scope == "junction" and cat.startswith("fail") and len(examples[cat]) < 5:
                    examples[cat].append("/".join(city[-4:]))
    return res, dict(examples)


def show(name, res):
    print(f"\n== {name}")
    for region, row in res.items():
        n = row["frames_with_junction_light"]
        cells = ", ".join(f"{c} {row[c]} ({100 * row[c] / n:.1f}%)" for c in ORDER if row[c])
        print(f"{region}: {n} frames | {cells}" + (f" | benchmark conflict {row['benchmark_conflict']}" if "benchmark_conflict" in row else ""))


def main():
    ol, ol_ex = openlane()
    dt, dt_ex = dtld()
    show("OpenLane-V2", ol)
    show("DTLD", dt)
    os.makedirs("results", exist_ok=True)
    with open("results/junction_rule_audit.json", "w") as f:
        json.dump({"openlane": ol, "openlane_examples": ol_ex, "dtld": dt, "dtld_examples": dt_ex}, f, indent=1)


if __name__ == "__main__":
    main()
