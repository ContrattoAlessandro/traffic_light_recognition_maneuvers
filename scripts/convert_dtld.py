"""Convert the DTLD v2.0 Bayer TIFF images to RGB JPEGs: <DATASETS>/DTLD_jpg/{train,test}/<name>.jpg.

Uses the official DTLD parser (https://github.com/julimueller/dtld_parsing, ``pip install`` it or put it on the
PYTHONPATH): ``DriveuDatabase(...).images[i].get_labeled_image()`` demosaics each image; it is saved with PIL's
default JPEG settings. Existing files are kept unless --overwrite.
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.paths import DATASETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=os.path.join(DATASETS, "DTLD"), help="DTLD root (the city folders)")
    ap.add_argument("--labels", default=os.path.join(DATASETS, "DTLD", "v2.0"), help="folder with DTLD_{train,test}.json")
    ap.add_argument("--out", default=os.path.join(DATASETS, "DTLD_jpg"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from dtld_parsing.driveu_dataset import DriveuDatabase

    for split in ("train", "test"):
        db = DriveuDatabase(os.path.join(args.labels, f"DTLD_{split}.json"))
        if not db.open(args.raw):
            raise RuntimeError(f"cannot open DTLD {split} under {args.raw}")
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        done = 0
        for frame in db.images:
            path = os.path.join(args.out, split, os.path.splitext(os.path.basename(frame.file_path))[0] + ".jpg")
            if os.path.exists(path) and not args.overwrite:
                continue
            bgr = frame.get_labeled_image()
            if bgr is None:
                raise RuntimeError(f"cannot decode {frame.file_path}")
            Image.fromarray(bgr[..., ::-1]).save(path)
            done += 1
        print(f"{split}: {done} images converted", flush=True)


if __name__ == "__main__":
    main()
