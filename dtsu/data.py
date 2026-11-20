"""Reading the label .jsonl files produced by ``dtsu.labels`` (and the DTLD / LISA / D*_US builders)."""
import json

from . import DIRECTIONS


def load_records(path, image_root, kept_only=True):
    """Kept frames of one label file; each record remembers the subset root its image lives under."""
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    recs = [r for r in recs if r["kept"] or not kept_only]
    for r in recs:
        r["root"] = image_root
        # label files built before U-turn was dropped still carry a "uturn" entry: keep DIRECTIONS only
        for k in ("color", "pass"):
            if isinstance(r.get(k), dict):
                r[k] = {d: r[k][d] for d in DIRECTIONS}
    return recs
