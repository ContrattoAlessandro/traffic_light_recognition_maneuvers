"""Train the pictogram classifier P (REPLICATION_PLAN.md, amendment 1) on data/pictogram_crops.

Selection: the epoch with the best macro accuracy (mean per-class recall) on ``val`` (DTLD tune
sessions + held-out ATLAS train sessions). Only that checkpoint is then scored on atlas_test, lisa_test
and the hand-labelled OpenLane lights (reported, never used to choose anything).

Augmentation (on the GPU): the crop window is jittered per side in scale (0.8-1.5) and shifted by up to
15% of the box's long side (a detector's box is rarely the annotated one), brightness / contrast /
saturation +-25%, a horizontal flip with left <-> right labels swapped, and with probability 0.3 a
down-and-up resampling to 10-32 px (small, distant lights). Classes are drawn with weights ~ 1/sqrt(count).
Writes <out>/best.pth, <out>/log.jsonl and <out>/report.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.pictogram import CLASSES, FLIP_INDEX, INPUT, INPUT_MARGIN, STORE_MARGIN, PictogramNet, centre, normalize  # noqa: E402

D = "data/pictogram_crops"


def load_set(name):
    meta = dict(np.load(os.path.join(D, f"{name}.npz")))
    meta["img"] = np.load(os.path.join(D, f"{name}_img.npy"), mmap_mode="r")
    return meta


def augment(x_uint8, labels, gen):
    n, dev = len(labels), x_uint8.device
    x = normalize(x_uint8)
    base = INPUT_MARGIN / STORE_MARGIN
    sx = base * (0.8 + 0.7 * torch.rand(n, device=dev, generator=gen))
    sy = base * (0.8 + 0.7 * torch.rand(n, device=dev, generator=gen))
    t = (torch.rand(n, 2, device=dev, generator=gen) * 2 - 1) * 0.15 * (2 / STORE_MARGIN)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, :, 2] = sx, sy, t
    x = F.grid_sample(x, F.affine_grid(theta, (n, 3, INPUT, INPUT), align_corners=False), align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.5
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.5
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.5
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m * b
    g = x.mean(1, keepdim=True)
    x = (x - g) * s + g
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    x[flip] = x[flip].flip(-1)
    labels = torch.where(flip, torch.as_tensor(FLIP_INDEX, device=dev)[labels], labels)
    low = torch.rand(n, device=dev, generator=gen) < 0.3
    if low.any():
        r = int(torch.randint(10, 33, (1,), device=dev, generator=gen))
        x[low] = F.interpolate(F.interpolate(x[low], size=(r, r), mode="bilinear", align_corners=False, antialias=True),
                               size=(INPUT, INPUT), mode="bilinear", align_corners=False)
    return x, labels


@torch.no_grad()
def evaluate(net, s, batch=1024):
    net.eval()
    preds = []
    for i in range(0, len(s["label"]), batch):
        x = centre(torch.as_tensor(np.asarray(s["img"][i:i + batch])).cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(net(x, torch.as_tensor(s["color"][i:i + batch]).cuda()).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return summarize(s["label"], pred, s["dataset"])


def summarize(y, pred, datasets):
    def block(mask):
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        np.add.at(conf, (y[mask], pred[mask]), 1)
        present = conf.sum(1) > 0
        recall = np.diag(conf)[present] / conf.sum(1)[present]
        arrow_gt, arrow_pr = (y[mask] >= 1) & (y[mask] <= 5), (pred[mask] >= 1) & (pred[mask] <= 5)
        return {"n": int(mask.sum()), "acc": float((y[mask] == pred[mask]).mean()) if mask.any() else None,
                "macro_acc": float(recall.mean()) if present.any() else None,
                "arrow_recall": float((arrow_gt & arrow_pr).sum() / max(arrow_gt.sum(), 1)),
                "arrow_precision": float((arrow_gt & arrow_pr).sum() / max(arrow_pr.sum(), 1)),
                "exact_on_true_arrows": float((y[mask][arrow_gt] == pred[mask][arrow_gt]).mean()) if arrow_gt.any() else None,
                "confusion_gt_x_pred": {CLASSES[a]: {CLASSES[b]: int(conf[a, b]) for b in range(len(CLASSES)) if conf[a, b]}
                                        for a in range(len(CLASSES)) if conf[a].sum()}}

    out = {"all": block(np.ones(len(y), bool))}
    for d in sorted(set(datasets)):
        out[d] = block(datasets == d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/pictogram_p")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="steps per epoch (smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tr, va = load_set("train"), load_set("val")
    counts = np.bincount(tr["label"], minlength=len(CLASSES)).astype(float)
    w = (1 / np.sqrt(np.maximum(counts, 1)))[tr["label"]]
    w /= w.sum()
    steps = args.limit or len(tr["label"]) // args.batch
    print(f"train {len(tr['label'])} crops {dict(zip(CLASSES, counts.astype(int)))} | val {len(va['label'])} | {steps} steps/epoch", flush=True)

    net = PictogramNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    best = -1
    for epoch in range(args.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for step in range(steps):
            idx = np.sort(rng.choice(len(w), args.batch, p=w))
            x = torch.as_tensor(np.asarray(tr["img"][idx])).cuda(non_blocking=True)
            y = torch.as_tensor(tr["label"][idx]).cuda()
            col = torch.as_tensor(tr["color"][idx]).cuda()
            x, y = augment(x, y, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(x, col).float(), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(net, va)
        entry = {"epoch": epoch, "loss": tot / steps, "val_macro_acc": m["all"]["macro_acc"], "val_acc": m["all"]["acc"],
                 "val_by_dataset": {k: v["macro_acc"] for k, v in m.items() if k != "all"}, "sec": round(time.time() - t0)}
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if m["all"]["macro_acc"] > best:
            best = m["all"]["macro_acc"]
            torch.save({"model": net.state_dict(), "epoch": epoch, "val_macro_acc": best}, os.path.join(args.out, "best.pth"))

    ck = torch.load(os.path.join(args.out, "best.pth"), map_location="cpu")
    net.load_state_dict(ck["model"])
    report = {"selected_epoch": ck["epoch"], "val_macro_acc": ck["val_macro_acc"]}
    for name in ("val", "atlas_test", "lisa_test", "openlane"):
        report[name] = evaluate(net, load_set(name))
        a = report[name]["all"]
        print(f"{name}: acc {a['acc']:.3f} macro {a['macro_acc']:.3f} arrow R {a['arrow_recall']:.3f} P {a['arrow_precision']:.3f} "
              f"exact on arrows {a['exact_on_true_arrows']}", flush=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
