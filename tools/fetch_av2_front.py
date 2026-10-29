"""Fetch subset_A front images (ring_front_center) from the public Argoverse 2 bucket.

OpenLane-V2 subset_A frames are Argoverse 2 sensor-dataset frames: each info file
records the AV2 log id (meta_data.source_id), and the front image is named by the
same timestamp. The AV2 copy is pixel-identical up to JPEG re-encoding, so this
replaces downloading the ~121 GB all-camera image tars. Existing files are skipped.
"""
import argparse
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://s3.amazonaws.com/argoverse/datasets/av2/sensor"
AV2_SPLITS = ("train", "val", "test")


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="subset_A root with train/ val/ info files")
    ap.add_argument("--data-dict", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    dd = json.load(open(args.data_dict))
    jobs = {}  # segment -> (log_id, [image relative paths])
    for split in args.splits:
        for seg, frames in dd[split].items():
            info = json.load(open(os.path.join(args.root, split, seg, "info", frames[0])))
            paths = []
            for f in frames:
                rel = f"{split}/{seg}/image/ring_front_center/{f[:-5]}.jpg"
                if not os.path.exists(os.path.join(args.root, rel)):
                    paths.append(rel)
            if paths:
                jobs[seg] = (info["meta_data"]["source_id"], paths)
    n_todo = sum(len(p) for _, p in jobs.values())
    print(f"{len(jobs)} segments, {n_todo} images to fetch", flush=True)

    def do_segment(item):
        seg, (log_id, paths) = item
        av2_split = None
        for sp in AV2_SPLITS:  # find which AV2 split holds this log
            ts = os.path.basename(paths[0])
            if fetch(f"{BASE}/{sp}/{log_id}/sensors/cameras/ring_front_center/{ts}") is not None:
                av2_split = sp
                break
        if av2_split is None:
            return seg, 0, len(paths)
        ok = missing = 0
        for rel in paths:
            data = fetch(f"{BASE}/{av2_split}/{log_id}/sensors/cameras/ring_front_center/{os.path.basename(rel)}")
            if data is None:
                missing += 1
                continue
            dst = os.path.join(args.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            ok += 1
        return seg, ok, missing

    done = total_missing = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (seg, ok, missing) in enumerate(pool.map(do_segment, jobs.items()), 1):
            done += ok
            total_missing += missing
            if missing:
                print(f"  segment {seg}: {missing} images not found", flush=True)
            if k % 25 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)} segments] fetched {done}/{n_todo}, missing {total_missing}", flush=True)
    print(f"DONE fetched={done} missing={total_missing}")


if __name__ == "__main__":
    main()
