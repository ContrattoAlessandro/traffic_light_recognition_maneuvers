"""Test every RF-DETR model on VZC-TLD (Trinci et al.), a US dataset with native per-light relevance. No model trained on it.

Each model runs at the epoch and score threshold picked on its own tune split; nothing is tuned on VZC.
Native labels only: box, color (red / yellow / green / unknown) and relevance per light, and the image label
RR (relevant red or yellow) / RG (relevant green) / NoR (no relevant colored light).
- detection: center AP, recall, precision and color of matched lights (tools/eval_rfdetr_perception.summarize;
  "unknown" lights are ignore boxes);
- relevance per light, center matching against every box: on matched lights, accuracy and F1 of "relevant";
  end to end, precision / recall / F1 of detections called relevant against the relevant lights;
- image decision: "top": the highest-scoring light the model calls relevant; "straight": the decoder of the
  model for the straight maneuver. Red or yellow -> stop, green -> go, nothing -> no relevant light (acts as go).
  3-way accuracy, False-Go = P(not stop | RR), False-Stop = P(stop | RG).
Relevance differs by model: lane topology (A, B, A+B, mix), route (DTLD), junction (Exp 6, Exp 7). VZC defines it
as applying to the current direction of the vehicle, so it is closest to route and stricter than junction.
Paired session bootstrap (10,000 draws, seed 0) against the mixed moves model. Caches detections in runs/vzc_eval/.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_rfdetr_perception as perc  # noqa: E402
import tools.eval_picto_perception as pp_  # noqa: E402  (patches perc.parse_class; set per model below)
from dtsu.picto_rules import rule  # noqa: E402
from dtsu.rfdetr_baseline import assign_scored, parse_class  # noqa: E402
from tools.eval_picto import load_model as load_picto  # noqa: E402
from tools.eval_rfdetr import load_model as load_moves  # noqa: E402

TLD = os.path.abspath(os.environ.get("TLD", "../traffic-light-detection"))
OUT = "runs/vzc_eval"
REFERENCE = "mix"
PICTO_DTLD, PICTO_MULTI = "data/rfdetr_coco_dtld_picto/class_names.json", "data/rfdetr_coco_multi_picto/class_names.json"
# name: (kind, run, epoch, threshold, class names, arm)
MODELS = {
    "A": ("moves", "runs/rfdetr_m_A", 2, 0.20, "data/rfdetr_coco_fit_A/class_names.json", None),
    "B": ("moves", "runs/rfdetr_m_B", 2, 0.15, "data/rfdetr_coco_fit_B/class_names.json", None),
    "A+B": ("moves", "runs/tune_rfdetr/r1_m_default", 1, 0.20, "data/rfdetr_coco_fit/class_names.json", None),
    "DTLD": ("moves", "runs/rfdetr_m_dtld", 4, 0.25, "data/rfdetr_coco_dtld/class_names.json", None),
    "mix": ("moves", "runs/rfdetr_m_mix", 1, 0.25, "data/rfdetr_coco_fit_mix/class_names.json", None),
}
for _name, _run, _cls, _sel, _arm in (("Exp6 J", "runs/exp6_picto_J", PICTO_DTLD, "picto_eval/result.json", "J"),
                                      ("Exp6 F", "runs/exp6_picto_F", PICTO_DTLD, "picto_eval/result.json", "F"),
                                      ("Exp7 F", "runs/exp7_multi_F", PICTO_MULTI, "exp7_eval/result.json", "F")):
    _s = json.load(open(os.path.join(_run, _sel)))
    _s = _s.get("F7", _s)
    MODELS[_name] = ("picto", _run, _s["epoch"], _s["threshold"], _cls, _arm)


def load_items():
    items = []
    for split in ("test", "train"):
        for line in open(os.path.join(TLD, "datasets", "lamps", f"vzc_{split}.jsonl")):
            r = json.loads(line)
            items.append({"path": os.path.join(TLD, r["image"]), "group": split, "session": r["session"], "label": r["image_label"],
                          "boxes": torch.tensor([l["box"] for l in r["lamps"]], dtype=torch.float32).reshape(-1, 4),
                          "color": [l["state"] if l["state"] in perc.COLORS else None for l in r["lamps"]],
                          "relevant": [bool(l["relevant"]) for l in r["lamps"]], "arrow": [None] * len(r["lamps"])})
    return items


def run_model(name, items, workers):
    kind, run, epoch, _, cls_file, arm = MODELS[name]
    cache = os.path.join(OUT, name.replace(" ", "_").replace("+", "p") + ".pt")
    if os.path.exists(cache):
        preds = torch.load(cache, weights_only=False)
        if len(preds) == len(items):
            return preds
    names = json.load(open(cls_file))
    if kind == "moves":
        net, res, _ = load_moves(run, os.path.join(run, f"epoch_{epoch}_ema.pth"), len(names))
        preds = perc.predictions(net, items, names, res, workers, 8)
    else:
        net, res = load_picto(run, epoch, names, arm)
        preds = pp_.predictions(net, items, names, res, arm, workers)
    del net
    torch.cuda.empty_cache()
    torch.save(preds, cache)
    return preds


def decoder(name):
    """-> (class names for summarize, parse function, class idx -> (relevant, what the rule needs, color))."""
    kind, _, _, _, cls_file, _ = MODELS[name]
    if kind == "moves":
        names = json.load(open(cls_file))

        def d(c):
            moves, color = parse_class(names[c])
            return moves is not None, moves, color
        return names, parse_class, d

    def d(c):
        p, j, color = pp_.VIRTUAL[c].split("|")
        return j == "J", p, color
    return pp_.VIRTUAL, pp_.virtual_parse, d


def score_model(name, items, preds):
    kind, _, _, thr, _, _ = MODELS[name]
    _, _, decode = decoder(name)
    rows, light = [], np.zeros(7, int)  # matched tp fp fn tn | end to end: called relevant, of them right, relevant lights
    for it, (c, s, b) in zip(items, preds):
        k = s > thr
        c, s, b = c[k], s[k], b[k]
        dec = [decode(int(x)) for x in c]
        rel_dets = [(what, color, float(sc)) for (r, what, color), sc in zip(dec, s) if r]
        m = perc.match(b, s, it["boxes"], torch.zeros(len(it["boxes"]), dtype=torch.bool), "center")
        for p, g in enumerate(m.tolist()):
            if g >= 0:
                t, q = it["relevant"][g], dec[p][0]
                light[0 if t and q else 1 if q else 2 if t else 3] += 1
            if dec[p][0]:
                light[4] += 1
                light[5] += int(g >= 0 and it["relevant"][g])
        light[6] += sum(it["relevant"])
        top = max(rel_dets, key=lambda x: x[2])[1] if rel_dets else None
        if kind == "moves":
            straight = assign_scored(rel_dets, "right", "dedicated")["straight"]
        else:
            straight = rule(rel_dets)["straight"]
        straight = None if straight == "none" else straight
        rows.append([{"RR": 0, "RG": 1, "NoR": 2}[it["label"]], *(2 if x is None else 1 if x == "green" else 0 for x in (top, straight))])
    return np.array(rows), light


def ratios(gt, pred):
    return {"acc": ((pred == gt) * 1.0, np.ones(len(gt))), "false_go": ((gt == 0) & (pred != 0), gt == 0), "false_stop": ((gt == 1) & (pred == 0), gt == 1)}


def boot(sessions, draws, seed=0):
    uniq, inv = np.unique(sessions, return_inverse=True)
    rng = np.random.default_rng(seed)
    w = np.zeros((draws, len(uniq)), dtype=np.float32)
    np.add.at(w, (np.arange(draws)[:, None], rng.integers(0, len(uniq), (draws, len(uniq)))), 1)
    return lambda x: w @ np.bincount(inv, weights=np.asarray(x, float), minlength=len(uniq)).astype(np.float32)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N images, no result file (smoke test)")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    if args.limit:
        OUT = os.path.join(OUT, "smoke")
    os.makedirs(OUT, exist_ok=True)
    items = load_items()[: args.limit]
    print(len(items), "images,", sum(it["group"] == "test" for it in items), "official test,", len({it["session"] for it in items}), "sessions", flush=True)
    rows, out = {}, {}
    splits = {"all": np.arange(len(items)), "test": np.array([i for i, it in enumerate(items) if it["group"] == "test"])}
    for name in args.models:
        preds = run_model(name, items, args.workers)
        names, parse, _ = decoder(name)
        perc.parse_class = parse
        rows[name], out[name] = {}, {"run": MODELS[name][1], "epoch": MODELS[name][2], "threshold": MODELS[name][3]}
        for sp, idx in splits.items():
            sub_i, sub_p = [items[i] for i in idx], [preds[i] for i in idx]
            s = perc.summarize(sub_i, sub_p, names, MODELS[name][3], "center")
            r, l = score_model(name, sub_i, sub_p)
            rows[name][sp] = r
            tp, fp, fn, tn, n_rel_det, tp_e2e, n_rel_gt = (int(v) for v in l)
            prec, rec = tp_e2e / max(n_rel_det, 1), tp_e2e / max(n_rel_gt, 1)
            out[name][sp] = {
                "images": len(idx),
                "detection": {k: s[k] for k in ("lights", "AP", "recall", "precision", "color_acc", "color_recall")},
                "relevance_on_matched": {"n": tp + fp + fn + tn, "acc": (tp + tn) / max(tp + fp + fn + tn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)},
                "relevance_end_to_end": {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}}
            for col, dec in ((1, "top"), (2, "straight")):
                out[name][sp][dec] = {m: float(a.sum() / max(b.sum(), 1)) for m, (a, b) in ratios(r[:, 0], r[:, col]).items()}
            o = out[name][sp]
            print(f"{name:7s} {sp:4s} AP {s['AP'] or 0:.3f} R {s['recall']:.3f} P {s['precision']:.3f} col {s['color_acc']:.3f} | rel matched acc "
                  f"{o['relevance_on_matched']['acc']:.3f} F1 {o['relevance_on_matched']['f1']:.3f} | e2e F1 {o['relevance_end_to_end']['f1']:.3f} | top {o['top']} | straight {o['straight']}", flush=True)
    if REFERENCE in rows:
        for sp, idx in splits.items():
            if not len(idx):
                continue
            b = boot(np.array([items[i]["session"] for i in idx]), 200 if args.limit else 10000)
            for name in rows:
                for col, dec in ((1, "top"), (2, "straight")):
                    rat = ratios(rows[name][sp][:, 0], rows[name][sp][:, col])
                    ref = ratios(rows[REFERENCE][sp][:, 0], rows[REFERENCE][sp][:, col])
                    for m in rat:
                        den = np.maximum(b(rat[m][1]), 1)
                        va = b(rat[m][0]) / den
                        out[name][sp][dec][m + "_ci95"] = [float(np.percentile(va, 2.5)), float(np.percentile(va, 97.5))]
                        if name != REFERENCE:
                            diff = va - b(ref[m][0]) / den
                            out[name][sp][dec][m + "_minus_mix_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    if not args.limit:
        with open(os.path.join(OUT, "result.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", os.path.join(OUT, "result.json"))


if __name__ == "__main__":
    main()
