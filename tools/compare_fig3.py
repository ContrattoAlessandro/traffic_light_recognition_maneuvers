"""Compare kept-frame color distributions with the paper's Fig. 3 (train + val)."""
import argparse
import collections
import json

FIG3 = {
    "straight": {"none": 2044, "red": 11838, "green": 14443, "yellow": 726},
    "left": {"none": 2099, "red": 11985, "green": 14165, "yellow": 802},
    "right": {"none": 28145, "red": 76, "green": 780, "yellow": 50},
    "uturn": {"none": 2082, "red": 11987, "green": 14173, "yellow": 809},
}
COLORS = ("none", "red", "green", "yellow")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", nargs="+", help="label .jsonl files")
    args = ap.parse_args()

    frames = [json.loads(l) for path in args.labels for l in open(path)]
    kept = [r for r in frames if r["kept"]]
    print(f"frames={len(frames)} kept={len(kept)} ({len(kept) / len(frames):.1%})  paper kept=29051")
    reasons = collections.Counter(
        "conflict" if r["conflict"] else "no_relevant_light" for r in frames if not r["kept"]
    )
    print("dropped:", dict(reasons))

    print(f"{'dir':9s}" + "".join(f"{c:>18s}" for c in COLORS) + "   (ours% / paper%)")
    for d, ref in FIG3.items():
        ours = collections.Counter(r["color"][d] for r in kept)
        n_ref = sum(ref.values())
        cells = [f"{100 * ours[c] / max(len(kept), 1):7.1f} /{100 * ref[c] / n_ref:6.1f}" for c in COLORS]
        print(f"{d:9s}" + "".join(f"{s:>18s}" for s in cells))

    agree = lambda a, b: sum(r["color"][a] == r["color"][b] for r in kept) / max(len(kept), 1)
    print(f"left==straight {agree('left', 'straight'):.3f}   uturn==left {agree('uturn', 'left'):.3f}")


if __name__ == "__main__":
    main()
