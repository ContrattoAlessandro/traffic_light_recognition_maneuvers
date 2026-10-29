"""Train / evaluate the direction-aware model (paper Eq. 13-15).

Train sets are concatenated; every val set is evaluated separately and all of them
together ("combined"). Val is the paper's evaluation split, so no checkpoint is
selected on it: the last epoch is the reported model.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CanvasBatchSampler, DTSUDataset, collate, load_records
from .detr import detr_loss
from .metrics import evaluate, format_metrics
from .model import DirectionAwareModel, param_groups


def compute_loss(out, targets, lam_c=1.0, lam_p=1.0, lam_d=1.0):
    gc = torch.stack([t["color"] for t in targets]).to(out["color_logits"].device)
    gp = torch.stack([t["pass"] for t in targets]).to(out["pass_logits"].device)
    # CE averaged over the four directions (Eq. 13, 14)
    l_color = F.cross_entropy(out["color_logits"].flatten(0, 1), gc.flatten())
    l_pass = F.cross_entropy(out["pass_logits"].flatten(0, 1), gp.flatten())
    l_det = detr_loss(out["det_logits"], out["det_boxes"], targets)
    total = lam_c * l_color + lam_p * l_pass + lam_d * l_det  # Eq. 15
    return total, {"color": l_color.item(), "pass": l_pass.item(), "det": l_det.item()}


def to_device(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


@torch.no_grad()
def predict(model, loader, device, amp=False):
    """Predicted and ground-truth [N, 4] color / pass indices."""
    model.eval()
    pc, pp, gc, gp = [], [], [], []
    for images, targets in loader:
        with torch.autocast(device.type, enabled=amp):
            out = model(images.to(device))
        pc.append(out["color_logits"].argmax(-1).cpu())
        pp.append(out["pass_logits"].argmax(-1).cpu())
        gc.append(torch.stack([t["color"] for t in targets]))
        gp.append(torch.stack([t["pass"] for t in targets]))
    return [torch.cat(x).numpy() for x in (pc, pp, gc, gp)]


def run_eval(model, val_loaders, device, amp=False):
    preds = {name: predict(model, dl, device, amp) for name, dl in val_loaders.items()}
    metrics = {name: evaluate(*p) for name, p in preds.items()}
    if len(preds) > 1:
        metrics["combined"] = evaluate(*(np.concatenate(xs) for xs in zip(*preds.values())))
    return metrics


def make_loader(records, args, train):
    ds = DTSUDataset(records, args.long_side, train=train)
    sampler = CanvasBatchSampler(ds, args.batch_size, shuffle=train, drop_last=train, seed=args.seed)
    return DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=args.workers > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-set", nargs=2, action="append", required=True, metavar=("LABELS", "ROOT"),
                    help="label .jsonl and the subset root its image paths are relative to (repeatable)")
    ap.add_argument("--val-set", nargs=3, action="append", required=True, metavar=("NAME", "LABELS", "ROOT"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--long-side", type=int, default=1024, help="images resized so their long side is this")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--last-epoch", type=int, default=-1,
                    help="stop after this 0-based epoch, keeping the --epochs lr schedule (default: run all)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--clip", type=float, default=0.1, help="gradient norm clip (DETR default 0.1)")
    ap.add_argument("--lam", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("C", "P", "D"))
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only N frames per train/val set (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224", help="timm model name")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt if present")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for the parts that stay fp32
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)

    def subsample(recs):
        if not args.limit:
            return recs
        return random.Random(args.seed).sample(recs, min(args.limit, len(recs)))

    train_recs = [r for labels, root in args.train_set for r in subsample(load_records(labels, root))]
    val_recs = {name: subsample(load_records(labels, root)) for name, labels, root in args.val_set}
    train_dl = make_loader(train_recs, args, train=True)
    val_dls = {name: make_loader(recs, args, train=False) for name, recs in val_recs.items()}

    model = DirectionAwareModel((args.long_side, args.long_side), backbone=args.backbone,
                                pretrained=not args.no_pretrained,
                                top_k=args.top_k, n_queries=args.n_queries).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train={len(train_recs)} "
          + " ".join(f"val_{n}={len(r)}" for n, r in val_recs.items()), flush=True)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr, args.weight_decay))
    total_iters = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(args.warmup_iters, 1))
                                              * 0.5 * (1 + math.cos(math.pi * it / max(total_iters, 1))))
    scaler = torch.amp.GradScaler(enabled=args.amp)

    it, start_epoch = 0, 0
    ckpt_path = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        it, start_epoch = ckpt["it"], ckpt["epoch"] + 1
        print(f"resumed from epoch {ckpt['epoch']} (it {it})", flush=True)

    bad_streak = 0
    end_epoch = args.epochs if args.last_epoch < 0 else min(args.epochs, args.last_epoch + 1)
    for epoch in range(start_epoch, end_epoch):
        train_dl.batch_sampler.rng.seed(args.seed + epoch)  # same batch order whether resumed or not
        model.train()
        t0 = time.time()
        for images, targets in train_dl:
            targets = to_device(targets, device)
            with torch.autocast(device.type, enabled=args.amp):
                out = model(images.to(device))
            loss, parts = compute_loss({k: v.float() for k, v in out.items()}, targets, *args.lam)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):  # skip the batch; give up if the weights themselves went bad
                bad_streak += 1
                print(f"ep {epoch} it {it} non-finite loss, batch skipped ({bad_streak} in a row)", flush=True)
                if bad_streak >= 20:
                    raise RuntimeError("20 non-finite losses in a row")
                it += 1
                sched.step()
                continue
            bad_streak = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if it % args.log_every == 0:
                print(f"ep {epoch} it {it}/{total_iters} loss {loss.item():.4f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in parts.items()) + f" {time.time() - t0:.0f}s", flush=True)
            it += 1

        metrics = run_eval(model, val_dls, device, args.amp)
        for name, m in metrics.items():
            print(f"[val {name}] epoch {epoch} {format_metrics(m)}", flush=True)
        with open(os.path.join(args.out_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "it": it}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
