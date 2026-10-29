"""Direction-aware baseline (paper Fig. 4, Eqs. 8-12).

Swin-T (or ConvNeXtV2-T) + FPN -> global token g and K traffic-light ROI tokens t_j
built from a DETR-style detector's boxes; four learnable direction queries attend to
T = [g, t_1..t_K] and feed color / passability heads.
"""
import timm
import torch
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from torch import nn
from torchvision.ops import box_convert, roi_align

from . import COLORS, DIRECTIONS, PASS
from .detr import MLP, DETRLite


class FPN(nn.Module):
    def __init__(self, in_channels, d=256):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, d, 1) for c in in_channels)
        self.output = nn.ModuleList(nn.Conv2d(d, d, 3, padding=1) for _ in in_channels)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        return [o(x) for o, x in zip(self.output, lat)]


class DirectionAwareModel(nn.Module):
    def __init__(self, img_size=(1024, 800), backbone="swin_tiny_patch4_window7_224", pretrained=True,
                 d=256, n_queries=50, top_k=16, roi_size=7, enc_layers=6, dec_layers=6):
        super().__init__()
        self.nhwc = backbone.startswith("swin")  # timm Swin returns NHWC feature maps
        # Swin needs the input size; ConvNeXt is fully convolutional and takes no img_size.
        # drop_path 0.1 is timm's Swin default, set explicitly so ConvNeXt gets the same
        size_kw = dict(img_size=img_size, strict_img_size=False) if self.nhwc else {}
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
            drop_path_rate=0.1, **size_kw,
        )
        channels = self.backbone.feature_info.channels()
        # one LayerNorm per output stage, as in the usual Swin / ConvNeXt detection setups; without
        # them the feature scale drifted to ~1e5 and training went unstable (swin_t_AB v1)
        self.out_norms = nn.ModuleList(nn.LayerNorm(c) if self.nhwc else LayerNorm2d(c) for c in channels)
        self.fpn = FPN(channels, d)
        self.detector = DETRLite(d, n_queries, enc_layers=enc_layers, dec_layers=dec_layers)
        self.top_k, self.roi_size = top_k, roi_size

        self.global_proj = nn.Linear(d, d)
        self.roi_proj = nn.Linear(d * roi_size * roi_size, d)  # phi_r
        self.box_proj = nn.Linear(4, d)  # phi_b
        self.type_embed = nn.Embedding(2, d)  # 0 = global token, 1 = ROI token

        self.dir_queries = nn.Embedding(len(DIRECTIONS), d)
        self.dir_attn = nn.MultiheadAttention(d, 8, dropout=0.1, batch_first=True)
        self.dir_norm1, self.dir_norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dir_ffn = MLP(d, 4 * d, d, 2)
        self.color_head = MLP(d, d, len(COLORS), 2)
        self.pass_head = MLP(d, d, len(PASS), 2)

    def forward(self, images):
        B, _, H, W = images.shape
        feats = self.backbone(images)
        if self.nhwc:
            feats = [n(f).permute(0, 3, 1, 2).contiguous() for n, f in zip(self.out_norms, feats)]
        else:
            feats = [n(f) for n, f in zip(self.out_norms, feats)]
        p2, p3, p4, p5 = self.fpn(feats)

        # global token: pooled feature pyramid (Eq. 8-10)
        g = torch.stack([p.mean((2, 3)) for p in (p2, p3, p4, p5)]).mean(0)
        g = self.global_proj(g) + self.type_embed.weight[0]

        # traffic-light boxes from the detector; ROI tokens from the top-K most confident (Eq. 9)
        # under fp16 the decoder's cross-attention overflowed (NaN in t1 and in swin_t_AB v1
        # epochs 14-15): with AMP the detector runs in bf16, which has fp32's range
        dev = images.device.type
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=torch.is_autocast_enabled(dev)):
            logits, boxes, _ = self.detector(p3, p4, p5)
        # the ROI/direction branch is cheap and overflowed too: fp32
        with torch.autocast(dev, enabled=False):
            return self._directions(p2.float(), g.float(), logits.float(), boxes.float(), B, H, W)

    def _directions(self, p2, g, logits, boxes, B, H, W):
        last_logits, last_boxes = logits[-1], boxes[-1].detach()
        score = last_logits.softmax(-1)[..., :-1].max(-1).values  # foreground confidence
        k = min(self.top_k, score.shape[1])
        idx = score.topk(k, dim=1).indices
        sel = torch.gather(last_boxes, 1, idx[..., None].expand(-1, -1, 4))  # [B, K, 4] cxcywh
        xyxy = box_convert(sel, "cxcywh", "xyxy") * sel.new_tensor([W, H, W, H])
        rois = torch.cat([torch.arange(B, device=p2.device).repeat_interleave(k)[:, None].float(),
                          xyxy.reshape(-1, 4)], 1)
        # traffic lights are tiny: pool from the stride-4 level
        app = roi_align(p2, rois, self.roi_size, spatial_scale=p2.shape[-1] / W, sampling_ratio=2, aligned=True)
        t = self.roi_proj(app.flatten(1)).view(B, k, -1) + self.box_proj(sel) + self.type_embed.weight[1]

        T = torch.cat([g[:, None], t], 1)  # [B, 1+K, d] (Eq. 10)

        # direction queries attend to the token sequence (Eq. 11)
        q = self.dir_queries.weight.unsqueeze(0).expand(B, -1, -1)
        h = self.dir_norm1(q + self.dir_attn(q, T, T, need_weights=False)[0])
        h = self.dir_norm2(h + self.dir_ffn(h))
        return {
            "color_logits": self.color_head(h),  # [B, 4, |COLORS|] (Eq. 12)
            "pass_logits": self.pass_head(h),  # [B, 4, |PASS|]
            "det_logits": logits,  # [L, B, Q, 4]
            "det_boxes": boxes,  # [L, B, Q, 4]
        }


def param_groups(model, lr, backbone_lr, weight_decay):
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return [{"params": bb, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay}]
