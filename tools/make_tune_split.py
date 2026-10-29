"""Hold out a scene-level tuning split from a train label file.

Whole segments go to one side so consecutive (near-identical) frames never leak between
fit and tune. Writes <out-prefix>_fit.jsonl and <out-prefix>_tune.jsonl.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels")
    ap.add_argument("out_prefix")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lines = open(args.labels).read().splitlines()
    recs = [json.loads(l) for l in lines]
    segments = sorted({r["segment_id"] for r in recs})
    tune = set(random.Random(args.seed).sample(segments, round(len(segments) * args.fraction)))
    with open(f"{args.out_prefix}_fit.jsonl", "w") as f_fit, open(f"{args.out_prefix}_tune.jsonl", "w") as f_tune:
        for line, r in zip(lines, recs):
            (f_tune if r["segment_id"] in tune else f_fit).write(line + "\n")
    kept = lambda pred: sum(r["kept"] and pred(r["segment_id"]) for r in recs)
    print(f"{args.labels}: segments fit={len(segments) - len(tune)} tune={len(tune)}  "
          f"kept frames fit={kept(lambda s: s not in tune)} tune={kept(lambda s: s in tune)}")


if __name__ == "__main__":
    main()
