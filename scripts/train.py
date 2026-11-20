"""Train the factorized RF-DETR (one light class + pictogram / color / junction heads, ``dtsu.factorized``).

RF-DETR's own recipe (optimizer, lr schedule, EMA, losses) is kept, except:
- no random crop and no multi-scale: a crop can cut out the light that sets a maneuver's label; one square input
  resolution (``--resolution``);
- brightness / contrast jitter (``Photometric``), inserted into RF-DETR's torchvision train pipeline;
- the horizontal flip is off by default because it turns left arrows into right arrows; ``--flip-swap`` enables a
  flip (p = 0.5) that also swaps the arrow labels (left <-> right, straight-left <-> straight-right);
- ``--class-weights POWER`` weights the pictogram and color cross-entropy by n_c ** -POWER, counted on train/
  (after the flip), scaled so the count-weighted mean weight is 1.
The EMA weights are saved after every epoch (``<out-dir>/epoch_<n>_ema.pth``); ``<out-dir>/last.ckpt`` holds the full
state for ``--resume``. ``<out-dir>/args.json`` records every setting, the class counts and the weights.

The main model R: python scripts/train.py --dataset-dir data/train_full --out-dir runs/R --resolution 1280 \
    --epochs 5 --batch-size 4 --grad-accum 4 --class-weights 0.5 --flip-swap --seed 0
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import torch
from PIL import ImageEnhance
from pytorch_lightning import Callback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu import factorized  # noqa: E402
from dtsu.rules import UNKNOWN, parse  # noqa: E402

VARIANTS = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")


class Photometric:
    """Brightness and contrast jitter on the resized PIL image (no hue shift)."""

    def __init__(self, low=0.75, high=1.25):
        self.low, self.high = low, high

    def __call__(self, img, target):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(self.low, self.high))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(self.low, self.high))
        return img, target


class FlipSwap:
    """Horizontal flip (probability p) of image and boxes that also mirrors the pictogram label through a class
    permutation (``dtsu.factorized.mirror_classes``)."""

    def __init__(self, perm, p=0.5):
        from rfdetr.datasets._torchvision import RandomHorizontalFlip

        self.flip = RandomHorizontalFlip(p=1.0)
        self.perm = torch.tensor(perm, dtype=torch.long)
        self.p = p

    def __call__(self, img, target):
        if torch.rand(()) >= self.p:
            return img, target
        img, target = self.flip(img, target)
        target = dict(target)
        target["labels"] = self.perm[target["labels"]]
        return img, target


class SaveEMAEachEpoch(Callback):
    """Saves the EMA weights after every epoch; with ``stop_after`` also ends training after that many epochs (the
    schedule still belongs to the full ``--epochs``, so a later ``--resume`` continues the same run)."""

    def __init__(self, out_dir, stop_after=None):
        self.out_dir, self.stop_after, self.t0 = out_dir, stop_after, None

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module

        print(f"[epoch {trainer.current_epoch}] {time.time() - self.t0:.0f}s incl. val, {trainer.num_training_batches} batches, "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        net = pl_module.model
        for cb in trainer.callbacks:
            inner = _get_ema_inner_module(cb)
            if inner is not None:
                net = inner.model
        if trainer.is_global_zero:
            torch.save({"model": net.state_dict(), "epoch": trainer.current_epoch},
                       os.path.join(self.out_dir, f"epoch_{trainer.current_epoch}_ema.pth"))
        if self.stop_after is not None and trainer.current_epoch + 1 >= self.stop_after:
            trainer.should_stop = True


class Factorize(Callback):
    """Swaps in the factorized head and loss before the optimizer and the EMA copy exist."""

    def __init__(self, class_names, class_weights=None):
        self.class_names, self.class_weights = class_names, class_weights

    def setup(self, trainer, pl_module, stage):
        net = getattr(pl_module.model, "_orig_mod", pl_module.model)
        factorized.factorize(net)
        net.to(next(net.backbone.parameters()).device)
        factorized.factorize_criterion(pl_module.criterion, self.class_names, class_weights=self.class_weights)
        print(f"factorized head: {type(net).__name__}, criterion {type(pl_module.criterion).__name__}, losses "
              f"{pl_module.criterion.losses}, {len(self.class_names)} attribute classes, class weights {self.class_weights}", flush=True)


def patch_rfdetr(out_dir, class_names, class_weights=None, flip_perm=None, stop_after=None):
    import rfdetr.datasets.coco as coco
    import rfdetr.training as training

    build_pipeline = coco._build_torchvision_pipeline

    def pipeline(image_set, *args, **kwargs):
        pipe = build_pipeline(image_set, *args, **kwargs)
        if image_set == "train":
            names = [type(t).__name__ for t in pipe.transforms]
            assert not any("Flip" in n or "Crop" in n for n in names), names
            pipe.transforms.insert(1, Photometric())  # after the resize, before ToImage
            if flip_perm is not None:
                pipe.transforms.insert(1, FlipSwap(flip_perm))
        return pipe

    coco._build_torchvision_pipeline = pipeline
    build_trainer = training.build_trainer

    def build_trainer_with_callbacks(*args, **kwargs):
        trainer = build_trainer(*args, **kwargs)
        trainer.callbacks.insert(0, Factorize(class_names, class_weights))
        trainer.callbacks.append(SaveEMAEachEpoch(out_dir, stop_after))
        return trainer

    training.build_trainer = build_trainer_with_callbacks


def train_counts(dataset_dir, classes, perm=None):
    """Pictogram and color box counts of train/ (labelled boxes only); with the flip permutation, the expected
    counts after a flip with p = 0.5 (half of each class moves to its mirror)."""
    coco_json = json.load(open(os.path.join(dataset_dir, "train", "_annotations.coco.json")))
    n = collections.Counter(a["category_id"] for a in coco_json["annotations"])
    if perm is not None:
        flipped = collections.Counter()
        for k, v in n.items():
            flipped[k] += v / 2
            flipped[perm[k]] += v / 2
        n = flipped
    out = {"picto": collections.Counter(), "color": collections.Counter()}
    for k, v in n.items():
        p, _, c = parse(classes[k])
        if p != UNKNOWN:
            out["picto"][p] += v
        if c != UNKNOWN:
            out["color"][c] += v
    return {h: dict(sorted(d.items(), key=lambda x: -x[1])) for h, d in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, help="a folder written by scripts/export_training_set.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="RFDETRMedium", choices=VARIANTS)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4, help="backbone lr (RF-DETR default)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    ap.add_argument("--stop-after", type=int, help="end after this many epochs (resumable to --epochs)")
    ap.add_argument("--resume", help="a full Lightning checkpoint, e.g. <out-dir>/last.ckpt")
    ap.add_argument("--class-weights", type=float, metavar="POWER", help="weight the pictogram / color cross-entropy by n_c ** -POWER")
    ap.add_argument("--flip-swap", action="store_true", help="horizontal flip p = 0.5 that swaps the left / right arrow labels")
    ap.add_argument("--seed", type=int, help="RF-DETR training seed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    classes = json.load(open(os.path.join(args.dataset_dir, "class_names.json")))
    weights, perm, info = None, None, {"factorized": True}
    if args.flip_swap:
        classes, perm = factorized.mirror_classes(classes)
        info["attribute_classes"] = classes
    if args.class_weights is not None:
        info["class_counts"] = train_counts(args.dataset_dir, classes, perm)
        weights = info["class_weights"] = factorized.class_weights(info["class_counts"], args.class_weights)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), **info}, f, indent=1)
    patch_rfdetr(args.out_dir, classes, weights, perm, args.stop_after)

    import rfdetr

    model = getattr(rfdetr, args.variant)(gradient_checkpointing=args.grad_ckpt)
    model.train(
        dataset_dir=args.dataset_dir, output_dir=args.out_dir, resolution=args.resolution, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum, lr=args.lr, lr_encoder=args.lr_encoder,
        num_workers=args.workers, aug_config={}, augmentation_backend="torchvision", scale_jitter=False,
        multi_scale=False, square_resize_div_64=True,
        checkpoint_interval=1000,  # last.ckpt (full state, for resuming) is still written every epoch
        progress_bar=None, tensorboard=False, resume=args.resume,
        **({"seed": args.seed} if args.seed is not None else {}),
    )


if __name__ == "__main__":
    main()
