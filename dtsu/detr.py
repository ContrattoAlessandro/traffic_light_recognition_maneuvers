"""DETR-style traffic-light detector pieces: transformer, Hungarian matcher, set loss."""
import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import box_convert, generalized_box_iou


def sine_pos_embed(h, w, dim, device):
    """2D sine positional encoding [h*w, dim] (DETR)."""
    npf = dim // 2
    y = torch.arange(h, device=device, dtype=torch.float32).add_(0.5).div_(h) * 2 * math.pi
    x = torch.arange(w, device=device, dtype=torch.float32).add_(0.5).div_(w) * 2 * math.pi
    t = 10000 ** (2 * (torch.arange(npf // 2, device=device) // 1) / npf)
    py = y[:, None] / t  # [h, npf/2]
    px = x[:, None] / t
    py = torch.cat([py.sin(), py.cos()], -1)[:, None, :].expand(h, w, npf)
    px = torch.cat([px.sin(), px.cos()], -1)[None, :, :].expand(h, w, npf)
    return torch.cat([py, px], -1).reshape(h * w, dim)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers):
        super().__init__()
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class DETRLite(nn.Module):
    """Encoder over the coarsest FPN level; decoder queries attend to fine + encoded levels.

    Small traffic lights need fine features, so the decoder memory also contains the
    stride-8 and stride-16 levels (without encoder self-attention, which would be costly).
    """

    def __init__(self, d=256, n_queries=50, n_classes=3, enc_layers=6, dec_layers=6, heads=8, ffn=2048):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.1, batch_first=True), enc_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, ffn, dropout=0.1, batch_first=True), dec_layers,
            norm=nn.LayerNorm(d),
        )
        self.query_embed = nn.Embedding(n_queries, d)
        self.level_embed = nn.Embedding(3, d)
        self.class_head = nn.Linear(d, n_classes + 1)  # last class = no object
        self.box_head = MLP(d, d, 4, 3)
        self.d = d

    def forward(self, p3, p4, p5):
        B = p5.shape[0]
        mem = []
        for lvl, f in enumerate((p3, p4, p5)):
            h, w = f.shape[-2:]
            pe = sine_pos_embed(h, w, self.d, f.device) + self.level_embed.weight[lvl]
            tokens = f.flatten(2).transpose(1, 2)  # [B, hw, d]
            if lvl == 2:
                tokens = self.encoder(tokens + pe)
            mem.append(tokens + pe)
        memory = torch.cat(mem, 1)

        # run decoder layer by layer to get auxiliary outputs
        out, hs = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1), []
        for layer in self.decoder.layers:
            out = layer(out, memory)
            hs.append(self.decoder.norm(out))
        hs = torch.stack(hs)  # [L, B, Q, d]
        logits = self.class_head(hs)
        boxes = self.box_head(hs).sigmoid()  # normalized cxcywh
        return logits, boxes, hs[-1]


@torch.no_grad()
def hungarian_match(logits, boxes, targets, w_cls=2.0, w_l1=5.0, w_giou=2.0):
    """Per-image (query_idx, target_idx) assignment as in DETR."""
    out = []
    prob = logits.softmax(-1)
    for b, t in enumerate(targets):
        if len(t["labels"]) == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        cost = (
            -w_cls * prob[b][:, t["labels"]]
            + w_l1 * torch.cdist(boxes[b], t["boxes"], p=1)
            - w_giou * generalized_box_iou(box_convert(boxes[b], "cxcywh", "xyxy"), box_convert(t["boxes"], "cxcywh", "xyxy"))
        )
        qi, ti = linear_sum_assignment(cost.nan_to_num(1e4, 1e4, -1e4).cpu().numpy())
        out.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(ti, dtype=torch.long)))
    return out


def detr_loss(all_logits, all_boxes, targets, n_classes=3, eos_coef=0.1, w_cls=1.0, w_l1=5.0, w_giou=2.0):
    """Set loss summed over decoder layers (auxiliary losses included)."""
    device = all_logits.device
    class_w = torch.ones(n_classes + 1, device=device)
    class_w[-1] = eos_coef
    n_boxes = max(sum(len(t["labels"]) for t in targets), 1)
    total = all_logits.new_zeros(())
    for logits, boxes in zip(all_logits, all_boxes):
        match = hungarian_match(logits, boxes, targets)
        tgt_cls = torch.full(logits.shape[:2], n_classes, dtype=torch.long, device=device)
        src_b, tgt_b = [], []
        for b, (qi, ti) in enumerate(match):
            tgt_cls[b, qi] = targets[b]["labels"][ti]
            src_b.append(boxes[b, qi])
            tgt_b.append(targets[b]["boxes"][ti])
        loss = w_cls * F.cross_entropy(logits.transpose(1, 2), tgt_cls, class_w)
        src_b, tgt_b = torch.cat(src_b), torch.cat(tgt_b)
        if len(src_b):
            loss = loss + w_l1 * F.l1_loss(src_b, tgt_b, reduction="sum") / n_boxes
            giou = generalized_box_iou(box_convert(src_b, "cxcywh", "xyxy"), box_convert(tgt_b, "cxcywh", "xyxy"))
            loss = loss + w_giou * (1 - giou.diag()).sum() / n_boxes
        total = total + loss
    return total
