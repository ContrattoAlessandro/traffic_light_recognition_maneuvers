"""Train RF-DETR on the "<moves>|<color>" light classes (structured baseline, see dtsu/rfdetr_baseline.py).

RF-DETR's own recipe (optimizer, lr, EMA, losses, schedule) is kept, except for what would corrupt
our labels or starve tiny lights:
- no horizontal flip (it swaps left/right arrows and lanes) and no random crop (it can cut out the
  light that sets a direction's label): ``aug_config={}`` and ``scale_jitter=False``;
- no multi-scale; one square resolution (default 1024, so lights keep at least the pixels the
  direction-aware model sees at 1024x800 / 576x1024);
- brightness/contrast jitter as in ``dtsu.data``, inserted into RF-DETR's torchvision train pipeline
  (the Albumentations route would resize train images with cv2 and val images with torchvision).
The EMA weights are saved after every epoch (``epoch_<n>_ema.pth``) for ``tools/eval_rfdetr.py``.
"""
import argparse
import json
import os
import sys
import time

import torch
from pytorch_lightning import Callback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.rfdetr_baseline import Photometric  # noqa: E402

VARIANTS = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")


class SaveEMAEachEpoch(Callback):
    """Saves the EMA weights after every epoch; with ``stop_after`` also ends training after that many epochs
    (the schedule still belongs to the full ``--epochs``, so a later ``--resume`` continues the same run)."""

    def __init__(self, out_dir, stop_after=None):
        self.out_dir = out_dir
        self.stop_after = stop_after
        self.t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"[epoch {trainer.current_epoch}] {time.time() - self.t0:.0f}s incl. val, "
              f"{trainer.num_training_batches} batches, peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)
        from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module

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


def patch_rfdetr(out_dir, stop_after=None):
    import rfdetr.datasets.coco as coco
    import rfdetr.training as training

    build_pipeline = coco._build_torchvision_pipeline

    def pipeline_with_photometric(image_set, *args, **kwargs):
        pipe = build_pipeline(image_set, *args, **kwargs)
        if image_set == "train":
            names = [type(t).__name__ for t in pipe.transforms]
            assert not any("Flip" in n or "Crop" in n for n in names), names
            pipe.transforms.insert(1, Photometric())  # after the resize, before ToImage
        return pipe

    coco._build_torchvision_pipeline = pipeline_with_photometric

    build_trainer = training.build_trainer

    def build_trainer_with_saver(*args, **kwargs):
        trainer = build_trainer(*args, **kwargs)
        trainer.callbacks.append(SaveEMAEachEpoch(out_dir, stop_after))
        return trainer

    training.build_trainer = build_trainer_with_saver


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", default="data/rfdetr_coco")
    ap.add_argument("--out-dir", default="runs/rfdetr_m_AB")
    ap.add_argument("--variant", default="RFDETRMedium", choices=VARIANTS)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4, help="backbone lr (RF-DETR default 1.5e-4)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    ap.add_argument("--stop-after", type=int, help="end after this many epochs (resumable to --epochs)")
    ap.add_argument("--resume", help="a full Lightning checkpoint, e.g. <out-dir>/last.ckpt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    patch_rfdetr(args.out_dir, args.stop_after)

    import rfdetr

    model = getattr(rfdetr, args.variant)(gradient_checkpointing=args.grad_ckpt)
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.out_dir,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        num_workers=args.workers,
        aug_config={},
        augmentation_backend="torchvision",
        scale_jitter=False,
        multi_scale=False,
        square_resize_div_64=True,
        checkpoint_interval=1000,  # last.ckpt (full state, for resuming) is still written every epoch
        progress_bar=None,
        tensorboard=False,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
