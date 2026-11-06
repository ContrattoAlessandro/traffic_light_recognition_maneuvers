"""J vs J+P arrow directions on ATLAS test and LISA test (REPLICATION_PLAN.md, amendment 1).

Same per-light test as ``tools/eval_rfdetr_perception.py`` (center matching, the run's own score
threshold, class-agnostic NMS): for every annotated arrow matched to a detection the run calls relevant,
whether the predicted moves (minus uturn / generic) equal the arrow's directions. J uses the run's class;
J+P replaces it with P's arrow when P's top class is an arrow with probability >= the run's tau
(picked on its source tune split by ``tools/eval_hybrid.py``, read from <run>/hybrid/matrix.json).
Paired 95% CI of the accuracy gain: bootstrap over images (10,000 draws, seed 0). ATLAS test has
3 cameras and LISA test 4 sequences, too few clusters for a cluster bootstrap (deviation from
amendment 1, recorded in REPLICATION_PLAN.md). Writes <run>/hybrid/arrows.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import pictogram as P  # noqa: E402
from dtsu.rfdetr_baseline import parse_class  # noqa: E402
from tools.eval_hybrid import RUNS, best_epoch  # noqa: E402
from tools.eval_rfdetr import load_model  # noqa: E402
from tools.eval_rfdetr_perception import LOADERS, match, norm_moves, predictions  # noqa: E402


def score(items, preds, probs, class_names, threshold, tau):
    rows = []  # per arrow called relevant: (image index, J correct, J+P correct, P overrode)
    for i, (it, (c, s, b), pr) in enumerate(zip(items, preds, probs)):
        above = s > threshold
        c, s, b = c[above], s[above], b[above]  # P probabilities exist for these only
        ignore = torch.tensor([col is None for col in it["color"]], dtype=torch.bool)
        m = match(b, s, it["boxes"], ignore, "center")
        for p, g in enumerate(m.tolist()):
            if g < 0 or it["arrow"][g] is None:
                continue
            moves, _ = parse_class(class_names[int(c[p])])
            if moves is None:
                continue
            j = norm_moves(moves)
            top = int(np.argmax(pr[p]))
            over = tau is not None and P.CLASSES[top] in P.ARROW_MOVES and pr[p][top] >= tau
            jp = set(P.ARROW_MOVES[P.CLASSES[top]]) if over else j
            rows.append((i, j == it["arrow"][g], jp == it["arrow"][g], over))
    return np.array(rows, dtype=float).reshape(-1, 4)


def paired(rows, n_images, draws=10000, seed=0):
    if not len(rows):
        return None
    img = rows[:, 0].astype(int)
    cnt = np.bincount(img, minlength=n_images).astype(float)
    dj = np.bincount(img, weights=rows[:, 1], minlength=n_images)
    djp = np.bincount(img, weights=rows[:, 2], minlength=n_images)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws // 100):  # 100 draws per block to bound memory
        w = np.zeros((100, n_images))
        idx = rng.integers(0, n_images, (100, n_images))
        np.add.at(w, (np.arange(100)[:, None], idx), 1)
        boots.append(((w @ djp) - (w @ dj)) / np.maximum(w @ cnt, 1))
    boots = np.concatenate(boots)
    return {"arrows": int(len(rows)), "J_exact": float(rows[:, 1].mean()), "J+P_exact": float(rows[:, 2].mean()),
            "d_exact": float(rows[:, 2].mean() - rows[:, 1].mean()),
            "d_exact_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "overridden": int(rows[:, 3].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", choices=sorted(RUNS), default=list(RUNS))
    ap.add_argument("--datasets", nargs="+", default=["atlas", "lisa"])
    ap.add_argument("--pictogram", default="runs/pictogram_p/best.pth")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tau", type=float, help="override the run's tau (smoke tests only)")
    args = ap.parse_args()

    net = P.load(args.pictogram)
    for name in args.runs:
        run, cn_path, _ = RUNS[name]
        class_names = json.load(open(cn_path))
        epoch, thr = best_epoch(run)
        tau = args.tau if args.tau is not None else json.load(open(os.path.join(run, "hybrid", "matrix.json")))["tau"]
        out = {"run": run, "epoch": epoch, "threshold": thr, "tau": tau}
        model = None
        for ds in args.datasets:
            items = LOADERS[ds]()[: args.limit]
            cache = os.path.join(run, "hybrid", f"arrows_e{epoch}_{ds}.pt")
            if args.limit is None and os.path.exists(cache):
                preds, probs = torch.load(cache, weights_only=False)
            else:
                if model is None:
                    model, res, _ = load_model(run, os.path.join(run, f"epoch_{epoch}_ema.pth"), len(class_names))
                preds = predictions(model, items, class_names, res, args.workers, 8)
                probs = []
                for it, (c, s, b) in zip(items, preds):
                    above = s > thr
                    c, b = c[above], b[above]
                    if not len(c):
                        probs.append(np.zeros((0, len(P.CLASSES)), np.float32))
                        continue
                    img = Image.open(it["path"]).convert("RGB")
                    crops = np.stack([P.store_crop(img, bb) for bb in b.tolist()])
                    colors = [parse_class(class_names[int(ci)])[1] for ci in c]
                    probs.append(P.classify(net, crops, colors))
                if args.limit is None:
                    torch.save((preds, probs), cache)
            out[ds] = r = paired(score(items, preds, probs, class_names, thr, tau), len(items))
            print(f"[{name}] {ds}: {r}", flush=True)
        if args.limit is None:
            with open(os.path.join(run, "hybrid", "arrows.json"), "w") as f:
                json.dump(out, f, indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
