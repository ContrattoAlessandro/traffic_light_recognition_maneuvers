"""Copy the small result files of finished runs into a tracked folder, with a checksum manifest.

runs/ and data/ are git-ignored (large, and OpenLane-V2 is non-commercial), so a commit alone does not
keep the numbers behind a write-up. This copies every .json / .jsonl / .csv / .log under each run (no
checkpoints, no .npz) to ``<out>/<run path>/`` and writes ``<out>/MANIFEST.json`` with the git commit,
the sha256 of each run's scored checkpoint and of the label and class-name files it depends on, so a
later re-run can be checked against them.

    python tools/snapshot_results.py --out results/joint_classes_baseline \
        --run runs/rfdetr_m_A:2:data/rfdetr_coco_fit_A/class_names.json ...
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess

SMALL = (".json", ".jsonl", ".csv", ".log")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="<run dir>:<scored epoch>:<class_names.json>")
    ap.add_argument("--labels", nargs="*", default=sorted(glob.glob("data/labels/*.jsonl") + glob.glob("data/labels/tune/*.jsonl")),
                    help="label files to checksum (default: all of data/labels)")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    manifest = {"code_commit": commit, "code_dirty_at_snapshot": dirty, "runs": {}, "labels": {}}
    for spec in args.run:
        run, epoch, classes = spec.split(":")
        ckpt = os.path.join(run, f"epoch_{epoch}_ema.pth")
        files = [p for p in glob.glob(os.path.join(run, "**", "*"), recursive=True) if p.endswith(SMALL)]
        for p in files:
            dst = os.path.join(args.out, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
        manifest["runs"][run.replace("\\", "/")] = {
            "scored_epoch": int(epoch), "checkpoint": ckpt.replace("\\", "/"), "checkpoint_sha256": sha256(ckpt),
            "class_names": classes, "class_names_sha256": sha256(classes), "files_copied": len(files)}
        print(f"{run}: {len(files)} files, epoch {epoch}")
    for p in args.labels:
        manifest["labels"][p.replace("\\", "/")] = sha256(p)
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {len(manifest['labels'])} label files, commit {commit[:7]}{' (dirty)' if dirty else ''}")


if __name__ == "__main__":
    main()
