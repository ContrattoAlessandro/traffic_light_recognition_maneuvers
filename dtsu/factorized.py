"""Factorized head for RF-DETR (Exp 6, arm F): one "light" class plus pictogram, color and junction heads.

The training data is the joint-class export (``<pictogram>|<J|N>|<color>``, dtsu/picto_rules.py); F maps each
target class back to its three attributes. Nothing in the rfdetr package is edited:
- ``factorize(model, class_names)`` resizes the detection head to one class (``reinitialize_detection_head``,
  which also resizes the two-stage encoder heads) and replaces ``class_embed`` with ``FactorizedClassEmbed``:
  the same light logit plus three linear heads on the same decoder features. The model's class becomes
  ``FactorizedLWDETR``, whose forward adds ``pred_attrs`` to the final and auxiliary decoder outputs.
- ``factorize_criterion(criterion, class_names)`` turns the SetCriterion into ``FactorizedCriterion``: targets
  get label 0 (light) for matching and the detection losses, and an extra "attrs" loss (cross-entropy for
  pictogram and color, binary cross-entropy for junction, weight 1 each) on the Hungarian-matched queries of
  every decoder layer. The encoder proposals carry no attributes.
Training applies both in a Lightning callback's ``setup`` (before the optimizer and the EMA copy exist);
inference rebuilds the same structure with ``factorize`` before loading the weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .picto_rules import COLORS, PICTOS, UNKNOWN, class_name, parse

HEADS = {"picto": len(PICTOS), "color": len(COLORS), "junction": 1}
IGNORE = -1  # Exp 7: attribute not labelled by the source dataset -> no loss


def heads(n_picto=None):
    """Head sizes; Exp 6 models have 7 pictogram outputs (before arrow_straight_right was appended)."""
    return {**HEADS, "picto": n_picto or HEADS["picto"]}


def n_picto_of(state_dict):
    w = [v for k, v in state_dict.items() if k.endswith("attrs.picto.weight")]
    return w[0].shape[0] if w else None


class FactorizedClassEmbed(nn.Module):
    def __init__(self, light: nn.Linear, n_picto=None):
        super().__init__()
        self.light = light
        self.attrs = nn.ModuleDict({k: nn.Linear(light.in_features, n) for k, n in heads(n_picto).items()})
        self.last = None

    def forward(self, hs):
        self.last = {k: m(hs) for k, m in self.attrs.items()}
        return self.light(hs)

    @property
    def out_features(self):
        return self.light.out_features


def attribute_table(class_names):
    """[n_classes, 3] long tensor: pictogram index, color index, junction (0/1) per joint class; IGNORE where unlabelled."""
    rows = []
    for n in class_names:
        p, j, c = parse(n)
        rows.append([IGNORE if p == UNKNOWN else PICTOS.index(p), IGNORE if c == UNKNOWN else COLORS.index(c),
                     IGNORE if j is None else int(j)])
    return torch.tensor(rows, dtype=torch.long)


def _make_factorized_model_class(base):
    class FactorizedLWDETR(base):
        def forward(self, samples, targets=None):
            out = super().forward(samples, targets)
            attrs = self.class_embed.last
            self.class_embed.last = None
            if attrs is not None and "pred_logits" in out:
                out["pred_attrs"] = {k: v[-1] for k, v in attrs.items()}
                for i, aux in enumerate(out.get("aux_outputs", [])):
                    aux["pred_attrs"] = {k: v[i] for k, v in attrs.items()}
            return out

    FactorizedLWDETR.__name__ = "FactorizedLWDETR"
    return FactorizedLWDETR


COCO_TRAFFIC_LIGHT = 10  # row of the COCO-pretrained class head (91 COCO ids) for "traffic light"


def factorize(model, n_picto=None):
    """In place: one light class + attribute heads. Returns the model.

    The light logit (decoder and two-stage encoder heads) starts from the pretrained COCO "traffic light"
    row, so F starts from a pretrained light detector; a plain resize would keep row 0, an unused COCO id
    with a very negative bias. The attribute heads start from PyTorch's default init."""
    if isinstance(model.class_embed, FactorizedClassEmbed):
        return model
    row = COCO_TRAFFIC_LIGHT if model.class_embed.out_features > COCO_TRAFFIC_LIGHT else 0
    heads = [model.class_embed] + (list(model.transformer.enc_out_class_embed) if model.two_stage else [])
    rows = [(h.weight.detach()[row].clone(), h.bias.detach()[row].clone()) for h in heads]
    model.reinitialize_detection_head(1)
    heads = [model.class_embed] + (list(model.transformer.enc_out_class_embed) if model.two_stage else [])
    with torch.no_grad():
        for h, (w, b) in zip(heads, rows):
            h.weight[0].copy_(w)
            h.bias[0].copy_(b)
    model.class_embed = FactorizedClassEmbed(model.class_embed, n_picto)
    model.__class__ = _make_factorized_model_class(type(model))
    return model


def _make_factorized_criterion_class(base):
    class FactorizedCriterion(base):
        def forward(self, outputs, targets, num_boxes=None):
            table = self.attr_table.to(targets[0]["labels"].device) if targets else self.attr_table
            new = []
            for t in targets:
                t = dict(t)
                t["attrs"] = table[t["labels"]]
                t["labels"] = torch.zeros_like(t["labels"])
                new.append(t)
            return super().forward(outputs, new, num_boxes)

        def get_loss(self, loss, outputs, targets, indices, num_boxes, matched_targets=None, **kwargs):
            if loss != "attrs":
                return super().get_loss(loss, outputs, targets, indices, num_boxes, matched_targets=matched_targets, **kwargs)
            if "pred_attrs" not in outputs:
                return {}
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for src, _ in indices])
            tgt = torch.cat([t["attrs"][j] for t, (_, j) in zip(targets, indices)])
            pa = outputs["pred_attrs"]
            if len(tgt) == 0:
                zero = pa["picto"].sum() * 0
                return {"loss_picto": zero, "loss_color": zero, "loss_junction": zero}
            out = {}
            for name, col in (("picto", 0), ("color", 1)):
                ok = tgt[:, col] != IGNORE
                logits = pa[name][batch_idx, src_idx].float()
                w = self.attr_weights.get(name)
                w = None if w is None else w.to(logits.device)[: logits.shape[-1]]
                out[f"loss_{name}"] = F.cross_entropy(logits[ok], tgt[ok, col], weight=w) if ok.any() else logits.sum() * 0
            ok = tgt[:, 2] != IGNORE
            logit = pa["junction"][batch_idx, src_idx, 0].float()
            out["loss_junction"] = (F.binary_cross_entropy_with_logits(logit[ok], tgt[ok, 2].float()) if ok.any()
                                    else logit.sum() * 0)
            return out

    FactorizedCriterion.__name__ = "FactorizedCriterion"
    return FactorizedCriterion


def factorize_criterion(criterion, class_names, weight=1.0, class_weights=None):
    """class_weights (Exp 9): optional {"picto": [len(PICTOS)], "color": [len(COLORS)]} per-class weights of the
    cross-entropy (PyTorch's weighted mean, so the loss is normalised by the weights of the targets in the batch)."""
    if getattr(criterion, "attr_table", None) is not None:
        return criterion
    criterion.attr_table = attribute_table(class_names)
    criterion.attr_weights = {k: torch.tensor(v, dtype=torch.float32) for k, v in (class_weights or {}).items()}
    criterion.num_classes = 1
    criterion.__class__ = _make_factorized_criterion_class(type(criterion))
    criterion.losses = list(criterion.losses) + ["attrs"]
    n_aux = max((int(k.rsplit("_", 1)[1]) for k in criterion.weight_dict if k.startswith("loss_bbox_") and k.rsplit("_", 1)[1].isdigit()),
                default=-1) + 1
    for name in ("loss_picto", "loss_color", "loss_junction"):
        criterion.weight_dict[name] = weight
        for i in range(n_aux):
            criterion.weight_dict[f"{name}_{i}"] = weight
    return criterion


SWAP = {"arrow_left": "arrow_right", "arrow_straight_left": "arrow_straight_right"}
SWAP.update({v: k for k, v in SWAP.items()})


def class_weights(counts, power=0.5):
    """Exp 9: w_c proportional to n_c ** -power, scaled so the count-weighted mean is 1; classes with no count get 1.
    counts: {"picto": {name: n}, "color": {name: n}} -> {"picto": [len(PICTOS)], "color": [len(COLORS)]}."""
    out = {}
    for head, names in (("picto", PICTOS), ("color", COLORS)):
        n = {k: float(v) for k, v in counts[head].items() if v}
        raw = {k: v ** -power for k, v in n.items()}
        scale = sum(n.values()) / sum(n[k] * raw[k] for k in n)
        out[head] = [round(raw[k] * scale, 6) if k in raw else 1.0 for k in names]
    return out


def mirror_classes(class_names):
    """Exp 9: class_names plus the mirrored classes the dataset lacks (e.g. arrow_straight_right|J|green: ATLAS has
    straight-right arrows but no junction label), and the permutation index -> index of the mirrored light.
    The extra classes exist only in the criterion's attribute table: the factorized detection head has one class."""
    ext = list(class_names)
    for name in class_names:
        p, j, c = parse(name)
        m = class_name(SWAP.get(p, p), j, c)
        if m not in ext:
            ext.append(m)
    perm = []
    for name in ext:
        p, j, c = parse(name)
        perm.append(ext.index(class_name(SWAP.get(p, p), j, c)))
    return ext, perm


@torch.no_grad()
def detect(model, images, amp=True):
    """Per image: (light score [Q], attribute probabilities {picto [Q,7], color [Q,3], junction [Q]}, boxes cxcywh [Q,4])."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(images)
    score = out["pred_logits"].float()[..., 0].sigmoid().cpu()
    pa = out["pred_attrs"]
    probs = {"picto": pa["picto"].float().softmax(-1).cpu(), "color": pa["color"].float().softmax(-1).cpu(),
             "junction": pa["junction"].float()[..., 0].sigmoid().cpu()}
    return score, probs, out["pred_boxes"].float().cpu()
