"""Build maneuver-level traffic-signal labels from the LISA Traffic Light Dataset (San Diego).

LISA boxes carry only a state tag: go / stop / warning for ball lights, goLeft / stopLeft /
warningLeft for left arrows and goForward for straight arrows. There is no lane or relevance
information, so the lights of the ego intersection are approximated by size: a light counts if
its height is at least SIZE_RATIO times that of the tallest light of its kind (ball, left arrow,
straight arrow) in the frame. Lights of farther intersections or other approaches look smaller;
comparing per kind keeps far-side balls when a near-side left-arrow pole is much closer. These go
through ``dtsu.labels.assign_directions`` with right-hand
rules: balls are generic and arrows dedicated, so left inherits the ball unless a left arrow is
visible, U-turn inherits left, and right is always none (LISA has no right arrows). Disagreeing
lights of the same kind mark a conflict and drop the frame. Frames with only left arrows are
dropped too: their balls are usually out of view or unannotated, not absent, so straight = none
would be a guess.

Splits: train = dayTrain + nightTrain clips, test = day/nightSequence1-2 (the official test
sequences). Frames are consecutive video frames; ``--stride`` keeps every n-th one.
"""
import argparse
import csv
import glob
import json
import os

from .labels import assign_directions, stats

TAGS = {
    "go": ("green", "generic"), "stop": ("red", "generic"), "warning": ("yellow", "generic"),
    "goLeft": ("green", "left"), "stopLeft": ("red", "left"), "warningLeft": ("yellow", "left"),
    "goForward": ("green", "straight"),
}
SIZE_RATIO = 0.6  # a light belongs to the ego intersection if h >= ratio * tallest light's h
SPLITS = {
    "train": ["dayTrain/*", "nightTrain/*"],
    "test": ["daySequence1", "daySequence2", "nightSequence1", "nightSequence2"],
}


def build_frame(segment, frame, image, lights):
    """lights: list of (tag, [x1, y1, x2, y2])."""
    h_max = {}
    for t, b in lights:
        h_max[TAGS[t][1]] = max(h_max.get(TAGS[t][1], 0), b[3] - b[1])
    near = [j for j, (t, b) in enumerate(lights) if b[3] - b[1] >= SIZE_RATIO * h_max[TAGS[t][1]]]
    light_moves = {j: (TAGS[lights[j][0]][0], {TAGS[lights[j][0]][1]}) for j in near}
    rec = {
        "segment_id": segment,
        "timestamp": frame,
        "image": image,
        "crossing": False,
        "boxes": [{"box": b, "color": TAGS[t][0], "relevant": j in light_moves,
                   "moves": sorted(light_moves[j][1]) if j in light_moves else []}
                  for j, (t, b) in enumerate(lights)],
        "n_lights": len(lights),
        "n_relevant_lights": len(light_moves),
        **assign_directions(light_moves, "right"),
        "traffic_side": "right",
    }
    # LISA leaves some visible balls unannotated, so straight = none is not trusted
    rec["kept"] = rec["color"]["straight"] != "none" and not rec["conflict"]
    return rec


def iter_frames(root, split):
    """(segment, frame number, image path relative to root, lights) per annotated frame."""
    for pattern in SPLITS[split]:
        for ann_dir in sorted(glob.glob(os.path.join(root, "Annotations", "Annotations", pattern))):
            rel = os.path.relpath(ann_dir, os.path.join(root, "Annotations", "Annotations")).replace("\\", "/")
            top, segment = rel.split("/")[0], rel.split("/")[-1]
            frames = {}
            with open(os.path.join(ann_dir, "frameAnnotationsBOX.csv")) as f:
                for row in csv.DictReader(f, delimiter=";"):
                    name = os.path.basename(row["Filename"])
                    box = [float(row[k]) for k in ("Upper left corner X", "Upper left corner Y",
                                                   "Lower right corner X", "Lower right corner Y")]
                    light = (row["Annotation tag"], box)
                    if light not in frames.setdefault(name, []):  # a few rows are duplicated
                        frames[name].append(light)
            for name, lights in sorted(frames.items()):
                image = f"{top}/{top}/{segment}/frames/{name}" if top != segment else f"{top}/{top}/frames/{name}"
                yield segment, name.rsplit("--", 1)[1].split(".")[0], image, lights


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="LISA root (contains Annotations/, dayTrain/, daySequence1/, ...)")
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th video frame of each clip")
    ap.add_argument("--out", required=True, help="output .jsonl with all frames (kept flag included)")
    args = ap.parse_args()

    records = [build_frame(*f) for f in iter_frames(args.root, args.split) if int(f[1]) % args.stride == 0]
    missing = sum(not os.path.exists(os.path.join(args.root, r["image"])) for r in records)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in records if r["kept"]]
    print(f"frames={len(records)} kept={len(kept)} segments={len({r['segment_id'] for r in records})} "
          f"conflict={sum(bool(r['conflict']) for r in records)} missing_image={missing}")
    for d, c in stats(kept).items():
        print(f"  {d:8s} {c}")


if __name__ == "__main__":
    main()
