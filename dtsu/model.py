"""Loading a trained factorized RF-DETR run and running it on images.

Images are squeezed to the run's square input resolution (torchvision bilinear + antialias, as RF-DETR's own
validation transform), in bf16. Two detection formats are produced, both cached by ``dtsu.evaluation``:
- ``frame_detections``: per image every query scoring above MIN_SCORE, with its pictogram, color, junction probability
  and normalized cxcywh box; thresholding and NMS happen at decoding time;
- ``box_detections``: per image the queries above MIN_SCORE after class-agnostic NMS, as (attribute code, score,
  xyxy pixel box), with code = pictogram * 6 + junction * 3 + color.
"""
import glob
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, nms

from . import factorized

MIN_SCORE = 0.05
NMS_IOU = 0.5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path, resolution):
    import torchvision.transforms.v2.functional as TF

    img = TF.resize(TF.to_image(Image.open(path).convert("RGB")), [resolution, resolution], antialias=True)
    return (TF.to_dtype(img, torch.float32, scale=True) - MEAN) / STD


def run_args(run):
    return json.load(open(os.path.join(run, "args.json")))


def saved_epochs(run):
    return sorted(int(re.search(r"epoch_(\d+)_ema", p).group(1)) for p in glob.glob(os.path.join(run, "epoch_*_ema.pth")))


def load_model(run, epoch, device="cuda"):
    """The EMA weights of one epoch (``<run>/epoch_<n>_ema.pth``) in the factorized model, on ``device``, in eval mode."""
    import rfdetr

    args = run_args(run)
    net = getattr(rfdetr, args["variant"])(resolution=args["resolution"], pretrain_weights=None).model.model
    state = torch.load(os.path.join(run, f"epoch_{epoch}_ema.pth"), map_location="cpu", weights_only=False)["model"]
    factorized.factorize(net, factorized.n_picto_of(state))
    net.load_state_dict(state)
    return net.to(device).eval()


class _Frames(Dataset):
    def __init__(self, paths, res):
        self.paths, self.res = paths, res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], self.res)


class _Sized(_Frames):
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            size = torch.tensor(im.size, dtype=torch.float32)
        return load_image(self.paths[i], self.res), size


@torch.no_grad()
def frame_detections(net, paths, res, workers=6, batch_size=8):
    out = []
    for x in DataLoader(_Frames(paths, res), batch_size=batch_size, num_workers=workers):
        score, probs, boxes = factorized.detect(net, x.cuda(non_blocking=True))
        picto, color, junction = probs["picto"].argmax(-1), probs["color"].argmax(-1), probs["junction"]
        for i in range(len(x)):
            k = score[i] > MIN_SCORE
            out.append({"score": score[i][k], "picto": picto[i][k], "color": color[i][k], "junction": junction[i][k], "box": boxes[i][k]})
    return out


@torch.no_grad()
def box_detections(net, paths, res, workers=6, batch_size=8):
    out = []
    for x, sizes in DataLoader(_Sized(paths, res), batch_size=batch_size, num_workers=workers):
        score, probs, boxes = factorized.detect(net, x.cuda(non_blocking=True))
        code = probs["picto"].argmax(-1) * 6 + (probs["junction"] > 0.5).long() * 3 + probs["color"].argmax(-1)
        for i, (w, h) in enumerate(sizes):
            k = score[i] > MIN_SCORE
            b = box_convert(boxes[i][k], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            s = score[i][k]
            keep = nms(b, s, NMS_IOU) if len(s) else torch.zeros(0, dtype=torch.long)
            out.append((code[i][k][keep], s[keep], b[keep]))
    return out


def decode_code(code):
    """Attribute code of ``box_detections`` -> (pictogram index, color index, junction 0/1)."""
    return code // 6, code % 3, (code // 3) % 2
