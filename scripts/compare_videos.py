"""Compare two models on the same video (outputs of ``scripts/run_video.py``): frame counts, where their straight color
disagrees, a sheet of sample frames side by side, and optionally a side-by-side video. Qualitative: videos have no labels.

Works for R (factorized, "junction" lights) and M ("relevant" lights) alike. Sample frames are the middles of the
longest runs of disagreement on the straight maneuver, plus one frame where both agree on a color.

Usage: python scripts/compare_videos.py runs/awsim/AWSIM_cam_picto_R runs/awsim/AWSIM_cam_M --names R M \
           --out runs/awsim/AWSIM_cam_R_vs_M [--video]
(each input is a path without extension: <path>.mp4 and <path>.jsonl)
"""
import argparse
import collections
import json

import cv2
import numpy as np

DIRS = ("straight", "left", "right")


def used(d):
    return d.get("junction", d.get("relevant", False))


def counts(rows):
    U = [[d for d in r["detections"] if used(d)] for r in rows]
    return {"frames": len(rows), "frames_with_any_detection": sum(bool(r["detections"]) for r in rows),
            "frames_with_used_light": sum(bool(u) for u in U), "detections": sum(len(r["detections"]) for r in rows),
            "used_detections": sum(map(len, U)), "used_by_color": dict(collections.Counter(d["color"] for u in U for d in u)),
            **{k: dict(collections.Counter(r["directions"][k] for r in rows)) for k in DIRS}}


def runs_of(ix):
    seg, cur = [], []
    for i in ix:
        if cur and i != cur[-1] + 1:
            seg.append(cur)
            cur = []
        cur.append(i)
    return seg + ([cur] if cur else [])


def frame(cap, i, label):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    assert ok, i
    cv2.rectangle(im, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=("A", "B"))
    ap.add_argument("--out", required=True, help="prefix of <out>_counts.json, <out>_sample.jpg and <out>.mp4")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="also write the two videos side by side")
    args = ap.parse_args()
    na, nb = args.names
    A, B = ([json.loads(x) for x in open(p + ".jsonl")] for p in (args.a, args.b))
    n = min(len(A), len(B))
    out = {na: counts(A[:n]), nb: counts(B[:n])}
    for k in DIRS:
        dis = [i for i in range(n) if A[i]["directions"][k] != B[i]["directions"][k]]
        out[f"{k}_disagree"] = len(dis)
        out[f"{k}_disagree_pairs ({na}->{nb})"] = {f"{a}->{b}": c for (a, b), c in collections.Counter(
            (A[i]["directions"][k], B[i]["directions"][k]) for i in dis).most_common()}
    dis = [i for i in range(n) if A[i]["directions"]["straight"] != B[i]["directions"]["straight"]]
    segs = sorted(runs_of(dis), key=len, reverse=True)
    picks = sorted(s[len(s) // 2] for s in segs[: args.samples - 1])
    agree = [i for i in range(n // 2, n) if A[i]["directions"]["straight"] == B[i]["directions"]["straight"] != "none"]
    picks += agree[:1]
    out["sample_frames"] = picks
    out["longest_disagreement_runs (frames)"] = [len(s) for s in segs[:10]]
    for k, v in out.items():
        print(k, json.dumps(v))
    caps = [cv2.VideoCapture(p + ".mp4") for p in (args.a, args.b)]
    rows = [np.hstack([frame(c, i, f"{nm}  frame {i}") for c, nm in zip(caps, (na, nb))]) for i in picks]
    width = min(r.shape[1] for r in rows)
    sheet = cv2.resize(np.vstack([r[:, :width] for r in rows]), None, fx=0.6, fy=0.6)
    cv2.imwrite(args.out + "_sample.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(out, open(args.out + "_counts.json", "w"), indent=1)
    if args.video:
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = None
        for i in range(n):
            ims = [c.read()[1] for c in caps]
            if any(im is None for im in ims):
                break
            h = min(im.shape[0] for im in ims)
            row = np.hstack([cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in ims])
            row = cv2.resize(row, (int(row.shape[1] * 0.75) // 2 * 2, int(row.shape[0] * 0.75) // 2 * 2))  # 3/4 size, even sides
            if writer is None:
                writer = cv2.VideoWriter(args.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (row.shape[1], row.shape[0]))
            writer.write(row)
        writer.release()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
