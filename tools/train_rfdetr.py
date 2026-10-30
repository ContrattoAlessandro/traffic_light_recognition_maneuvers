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
With ``--report`` (needs a dataset exported with ``--probe``) every epoch also prints an overfitting
report (``EpochReport``).
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


class EpochReport(Callback):
    """Per-epoch overfitting report, printed and appended to ``<out-dir>/epoch_report.jsonl``.

    RF-DETR's own train loss comes from the base weights in train mode over 13 query groups, its val
    loss from the EMA weights in eval mode, so the two are not comparable. Here both sides are scored
    the same way with the EMA weights in eval mode: valid/ (held out) and the train probe test/ (a fixed
    random part of train/, from ``make_rfdetr_coco.py --probe``). For each: the detection loss and the
    paper's direction metrics at the score threshold that maximizes valid TSS. RF-DETR's train-mode
    loss and valid mAP are reported too. Overfitting shows as valid TSS no longer improving (or valid
    loss rising) while the probe keeps improving. Creating ``<out-dir>/STOP`` ends training after the
    current epoch (its checkpoint is saved first).
    """

    def __init__(self, out_dir, dataset_dir, patience=3, workers=2):
        self.out_dir, self.dataset_dir, self.patience, self.workers = out_dir, dataset_dir, patience, workers
        self.path = os.path.join(out_dir, "epoch_report.jsonl")
        self.class_names = json.load(open(os.path.join(dataset_dir, "class_names.json")))
        self.records = {name: [json.loads(l) for l in open(os.path.join(dataset_dir, folder, "records.jsonl"))]
                        for name, folder in (("val", "valid"), ("probe", "test"))}
        self.history = [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []
        self.loaders = None
        self.train_losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:  # Lightning hands over the loss divided by the accumulation steps
            self.train_losses.append(float(loss.detach()) * trainer.accumulate_grad_batches)

    def _loss_loaders(self, trainer):
        """valid/ and test/ in RF-DETR's own eval pipeline, as its validation_step sees them."""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.datasets import build_dataset
        from torch.utils.data import DataLoader

        dm = trainer.datamodule
        ns = _namespace_from_configs(dm.model_config, dm.train_config)
        sets = {"val": dm._dataset_val, "probe": build_dataset("test", ns, dm.model_config.resolution)}
        return {k: DataLoader(ds, batch_size=dm._resolve_eval_batch_size(), shuffle=False, collate_fn=dm._collate_fn,
                              num_workers=self.workers, persistent_workers=True) for k, ds in sets.items()}

    @torch.no_grad()
    def _loss(self, trainer, pl_module, net, loader):
        dm, crit = trainer.datamodule, pl_module.criterion
        was_training = crit.training
        crit.eval()  # one query group, as in RF-DETR's validation_step
        total, n = 0.0, 0
        for batch in loader:
            batch = dm.on_after_batch_transfer(dm.transfer_batch_to_device(batch, pl_module.device, 0), 0)
            samples, targets = batch
            with trainer.precision_plugin.forward_context():
                outputs = net(samples)
                loss_dict = crit(outputs, targets)
            loss = sum(loss_dict[k] * crit.weight_dict[k] for k in loss_dict if k in crit.weight_dict)
            total, n = total + float(loss) * len(targets), n + len(targets)
        crit.train(was_training)
        return total / n

    def on_train_epoch_end(self, trainer, pl_module):
        import numpy as np
        from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module

        from dtsu.metrics import evaluate
        from tools.eval_rfdetr import THRESHOLDS, run_detector, score

        t0 = time.time()
        net = pl_module.model
        for cb in trainer.callbacks:
            inner = _get_ema_inner_module(cb)
            if inner is not None:
                net = inner.model
        was_training = net.training
        net.eval()
        if self.loaders is None:
            self.loaders = self._loss_loaders(trainer)
        losses = {k: self._loss(trainer, pl_module, net, dl) for k, dl in self.loaders.items()}
        res = trainer.datamodule.model_config.resolution
        dets = {k: run_detector(net, r, self.class_names, res, self.workers) for k, r in self.records.items()}
        net.train(was_training)

        val_tss = [evaluate(*score(self.records["val"], dets["val"], self.class_names, t))["tss"] for t in THRESHOLDS]
        thr = float(THRESHOLDS[int(np.argmax(val_tss))])
        m = {k: evaluate(*score(self.records[k], dets[k], self.class_names, thr)) for k in self.records}
        cm = trainer.callback_metrics
        entry = {
            "epoch": trainer.current_epoch,
            "train_loss_trainmode": float(np.mean(self.train_losses)) if self.train_losses else None,
            "probe_loss": losses["probe"], "val_loss": losses["val"],
            "val_mAP_50": float(cm["val/ema_mAP_50"]) if "val/ema_mAP_50" in cm else None,
            "val_mAP_50_95": float(cm["val/ema_mAP_50_95"]) if "val/ema_mAP_50_95" in cm else None,
            "threshold": thr, "val": m["val"], "probe": m["probe"],
        }
        hist = [h for h in self.history if h["epoch"] < entry["epoch"]] + [entry]
        best = max(hist, key=lambda h: h["val"]["tss"])
        since_best = entry["epoch"] - best["epoch"]
        entry["best_epoch"], entry["epochs_since_best"] = best["epoch"], since_best
        if since_best >= self.patience and entry["probe"]["tss"] > best["probe"]["tss"]:
            verdict = (f"OVERFITTING LIKELY: val TSS has not beaten epoch {best['epoch']} for {since_best} epochs "
                       f"while probe (train) TSS kept rising - consider stopping and using epoch {best['epoch']}")
        elif since_best >= self.patience:
            verdict = f"PLATEAU: no val TSS gain for {since_best} epochs, probe not improving either"
        elif len(hist) >= 3 and hist[-1]["val_loss"] > hist[-2]["val_loss"] > hist[-3]["val_loss"] \
                and hist[-1]["probe_loss"] < hist[-3]["probe_loss"]:
            verdict = "WATCH: val loss rose two epochs in a row while probe loss fell"
        else:
            verdict = "ok: still improving or within patience"
        entry["verdict"] = verdict
        self.history = hist
        if trainer.is_global_zero:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        prev = hist[-2] if len(hist) > 1 else None

        def delta(key, sub=None):
            if prev is None:
                return ""
            now, before = (entry[key][sub], prev[key][sub]) if sub else (entry[key], prev[key])
            return f" ({now - before:+.4f})"

        def row(name):
            v = m[name]
            return (f"TSS {v['tss']:.4f}{delta(name, 'tss')}  Acc_color {v['acc_color']:.4f}  Acc_pass {v['acc_pass']:.4f}  "
                    f"EM {v['em']:.4f}  False-Go {v['false_go']:.4f}  False-Stop {v['false_stop']:.4f}")

        mAP = lambda k: f"{entry[k]:.4f}" if entry[k] is not None else "n/a"
        train_loss = f"{entry['train_loss_trainmode']:.4f}" if entry["train_loss_trainmode"] is not None else "n/a"
        print("\n".join([
            f"===== epoch {entry['epoch']} report ({time.time() - t0:.0f}s) =====",
            f"loss (EMA, eval mode)  probe/train {entry['probe_loss']:.4f}{delta('probe_loss')}   "
            f"val {entry['val_loss']:.4f}{delta('val_loss')}   val - probe gap {entry['val_loss'] - entry['probe_loss']:+.4f}",
            f"loss (RF-DETR train mode, 13 query groups, not comparable to the above)  {train_loss}",
            f"val detection  mAP50 {mAP('val_mAP_50')}  mAP50-95 {mAP('val_mAP_50_95')}",
            f"directions @ threshold {thr:.2f} (picked on val)",
            f"  val          {row('val')}",
            f"  probe/train  {row('probe')}",
            f"  probe - val TSS gap {m['probe']['tss'] - m['val']['tss']:+.4f}",
            f"best val TSS {best['val']['tss']:.4f} at epoch {best['epoch']}, {since_best} epoch(s) since",
            f"verdict: {verdict}",
        ]), flush=True)

        if os.path.exists(os.path.join(self.out_dir, "STOP")):
            print(f"STOP file found: ending training after epoch {entry['epoch']}", flush=True)
            trainer.should_stop = True


def patch_rfdetr(out_dir, stop_after=None, report_dir=None):
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
        if report_dir:  # after the saver, so a STOP still keeps this epoch's weights
            trainer.callbacks.append(EpochReport(out_dir, report_dir))
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
    ap.add_argument("--report", action="store_true",
                    help="per-epoch overfitting report on valid/ and the train probe test/ (see EpochReport)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    patch_rfdetr(args.out_dir, args.stop_after, args.dataset_dir if args.report else None)

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
