"""Dataset over the label .jsonl files produced by ``dtsu.labels``."""
import json
import math
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Sampler

from . import COLORS, DIRECTIONS, PASS

LIGHT_CLASSES = ("red", "green", "yellow")  # detector classes (traffic-light color)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_records(path, image_root, kept_only=True):
    """Kept frames of one label file; each record remembers the subset root its image lives under."""
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    recs = [r for r in recs if r["kept"] or not kept_only]
    for r in recs:
        r["root"] = image_root
    return recs


def canvas_for(hw, long_side, multiple=32):
    """Resize scale and padded (H, W) canvas: long side -> long_side, aspect ratio kept."""
    h, w = hw
    scale = long_side / max(h, w)
    H, W = (math.ceil(round(v * scale) / multiple) * multiple for v in (h, w))
    return scale, (H, W)


class DTSUDataset(Dataset):
    """Front image resized so its long side is ``long_side`` (aspect kept), padded bottom/right.

    subset_A front images are portrait (2048x1550 -> 1024x800 canvas), subset_B landscape
    (900x1600 -> 576x1024). All images under one subset root share a size, read once per root.
    Boxes are returned as normalized (cx, cy, w, h) relative to the padded canvas.
    """

    def __init__(self, records, long_side=1024, train=False):
        self.records = records
        self.long_side = long_side
        self.train = train
        sizes = {}
        for r in records:
            if r["root"] not in sizes:
                with Image.open(os.path.join(r["root"], r["image"])) as im:
                    sizes[r["root"]] = (im.height, im.width)
        self.canvas = [canvas_for(sizes[r["root"]], long_side) for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        scale, (H, W) = self.canvas[idx]
        img = Image.open(os.path.join(rec["root"], rec["image"])).convert("RGB")
        nh, nw = round(img.height * scale), round(img.width * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        if self.train:
            # photometric only: no flips (they swap left/right) and no hue shift (it changes signal color)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))

        x = torch.zeros(3, H, W)
        pixels = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x[:, :nh, :nw] = (pixels - MEAN) / STD

        boxes, labels = [], []
        for b in rec["boxes"]:
            x1, y1, x2, y2 = (v * scale for v in b["box"])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
            labels.append(LIGHT_CLASSES.index(b["color"]))

        target = {
            "color": torch.tensor([COLORS.index(rec["color"][d]) for d in DIRECTIONS]),
            "pass": torch.tensor([PASS.index(rec["pass"][d]) for d in DIRECTIONS]),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return x, target


class CanvasBatchSampler(Sampler):
    """Batches whose images share one canvas shape (portrait A / landscape B never mix)."""

    def __init__(self, dataset, batch_size, shuffle, drop_last, seed=0):
        groups = {}
        for i, (_, hw) in enumerate(dataset.canvas):
            groups.setdefault(hw, []).append(i)
        self.groups = list(groups.values())
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.rng = random.Random(seed)

    def _batches(self):
        batches = []
        for idx in self.groups:
            idx = list(idx)
            if self.shuffle:
                self.rng.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                b = idx[k : k + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches.append(b)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        n = 0
        for idx in self.groups:
            n += len(idx) // self.batch_size if self.drop_last else math.ceil(len(idx) / self.batch_size)
        return n


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
