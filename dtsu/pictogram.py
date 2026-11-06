"""Pictogram classifier P of the two-stage arm J+P (REPLICATION_PLAN.md, amendment 1).

A ResNet-18 (trained from scratch, small-input stem) reads a square crop around a light, together with
the light's color (one-hot, appended to the pooled features: it picks the lamp when a head has two lit),
and predicts its pictogram. Crops are stored at STORE px covering STORE_MARGIN x the box's long side
(``tools/make_pictogram_crops.py``); the classifier input is the centre INPUT_MARGIN x long side resized
to INPUT px. ``crop_light`` builds the same input from a full image, for decoding detections.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

CLASSES = ["circle", "arrow_left", "arrow_straight", "arrow_right", "arrow_straight_left", "arrow_straight_right", "other"]
COLORS = ["red", "yellow", "green"]
ARROW_MOVES = {"arrow_left": {"left"}, "arrow_straight": {"straight"}, "arrow_right": {"right"},
               "arrow_straight_left": {"straight", "left"}, "arrow_straight_right": {"straight", "right"}}
FLIP = {"arrow_left": "arrow_right", "arrow_right": "arrow_left",
        "arrow_straight_left": "arrow_straight_right", "arrow_straight_right": "arrow_straight_left"}
FLIP_INDEX = [CLASSES.index(FLIP.get(c, c)) for c in CLASSES]
STORE, STORE_MARGIN = 96, 2.0
INPUT, INPUT_MARGIN = 64, 1.25
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PictogramNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.backbone = net
        self.color = nn.Embedding(len(COLORS), 32)
        self.head = nn.Sequential(nn.Linear(512 + 32, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, len(CLASSES)))

    def forward(self, x, color):
        return self.head(torch.cat([self.backbone(x), self.color(color)], 1))


def normalize(x_uint8):
    """[N, H, W, 3] uint8 tensor -> normalized float [N, 3, H, W]."""
    x = x_uint8.permute(0, 3, 1, 2).float() / 255
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def centre(stored):
    """Stored crops [N, STORE, STORE, 3] uint8 -> classifier input [N, 3, INPUT, INPUT] (no jitter)."""
    x = normalize(stored)
    side = round(STORE * INPUT_MARGIN / STORE_MARGIN)
    o = (STORE - side) // 2
    return F.interpolate(x[:, :, o:o + side, o:o + side], size=(INPUT, INPUT), mode="bilinear", align_corners=False)


def store_crop(img, box):
    """PIL image, xyxy box in its pixels -> stored crop [STORE, STORE, 3] uint8 (as the training crops)."""
    x1, y1, x2, y2 = box
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, STORE_MARGIN * max(x2 - x1, y2 - y1, 1.0)
    c = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))
    return np.asarray(c.resize((STORE, STORE), Image.BILINEAR), dtype=np.uint8)


def load(path, device="cuda"):
    net = PictogramNet()
    net.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return net.to(device).eval()


@torch.no_grad()
def classify(net, stored, colors, device="cuda", batch=512):
    """Stored crops (uint8 array [N, STORE, STORE, 3]) and color names -> class probabilities [N, C] (numpy)."""
    if len(stored) == 0:
        return np.zeros((0, len(CLASSES)), np.float32)
    out = []
    for i in range(0, len(stored), batch):
        x = centre(torch.as_tensor(np.asarray(stored[i:i + batch])).to(device))
        c = torch.as_tensor([COLORS.index(k) for k in colors[i:i + batch]], device=device)
        out.append(net(x, c).softmax(1).float().cpu().numpy())
    return np.concatenate(out)
