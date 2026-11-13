"""Exp 9 (REPLICATION_PLAN.md): class imbalance of the pictogram and color heads. Arms LA, W, WF against F8.

- LA: F8 with post-hoc logit adjustment, argmax of log p - tau * log(prior) for the pictogram and the color, prior =
  the F8 training counts (tools/train_rfdetr.train_counts). "unknown" has no training box and takes the circle prior
  (the rule treats it as a circle). Needs F8's probabilities, so F8 is run again with them kept (cached in
  runs/exp9_eval/LA/); tau = 0 must reproduce F8.
- W, WF: runs/exp9_W, runs/exp9_WF (tools/run_exp9.sh), decoded exactly as F8 (tools/eval_exp8.Model).
Selection, for every arm: the Exp 8 rule (epoch, threshold and, for LA, tau maximising the mean of DTLD tune TSS and
A / B tune TSS under the US rule + topology mask).
Guards G1 / G2: ΔTSS and ΔFalse-Go against F8 on A val, B val (US rule + topology mask) and DTLD test (junction rule),
segment bootstrap, 10,000 draws. Attribute endpoints (P1, P2, G3, G4 and the reported ones) on the lights both models
of a pair match above their own thresholds (center matching), cluster bootstrap, 2,000 draws, seed 0.
F7 (Exp 8's selection) is reported beside, from its Exp 8 caches. Writes runs/exp9_eval/result.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, nms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp7 as e7  # noqa: E402
import tools.eval_exp8 as e8  # noqa: E402
import tools.eval_rfdetr_perception as perc  # noqa: E402
import tools.exp8_attributes as att  # noqa: E402
from dtsu import factorized  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.picto_rules import COLORS, PICTOS  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_picto import THRESHOLDS, rule_metrics  # noqa: E402
from tools.train_rfdetr import train_counts  # noqa: E402

OUT = "runs/exp9_eval"
DATA = "data/rfdetr_coco_exp8"
F8_RUN, F7_RUN = "runs/exp8_bdd_F", "runs/exp7_multi_F"
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)
DRAWS, SEED = 2000, 0
P1_CLASSES = ("circle", "arrow_left", "arrow_right", "arrow_straight", "other")
REPORT_CLASSES = {"ATLAS_test": ("circle", "arrow_left", "arrow_right", "arrow_straight"),
                  "LISA_test": ("circle", "arrow_left", "arrow_straight")}
ARROWS = {p for p in PICTOS if p.startswith("arrow")}
WORKERS = 6


# ------------------------------------------------------------------ the LA arm: F8 with its probabilities kept
class LA:
    """F8 re-run keeping log-probabilities; frames / boxes decode them with a logit adjustment tau."""

    def __init__(self, workers):
        self.base = e8.Model("F8", F8_RUN, DATA, os.path.join(OUT, "LA"))
        self.epochs = self.base.epochs
        counts = train_counts(DATA, self.base.classes)
        prior = {h: np.array([counts[h].get(n, 0) for n in names], float) for h, names in (("picto", PICTOS), ("color", COLORS))}
        prior["picto"][PICTOS.index("unknown")] = prior["picto"][PICTOS.index("circle")]
        self.log_prior = {h: torch.tensor(np.log(v / v.sum()), dtype=torch.float32) for h, v in prior.items()}
        self.workers = workers
        self.res = self.base.res

    def rich(self, epoch, tag, paths):
        def run():
            net = self.base.net(epoch)
            out = []
            loader = DataLoader(perc.Images([{"path": p} for p in paths], self.res), batch_size=8, num_workers=self.workers)
            for x, sizes in loader:
                score, probs, boxes = factorized.detect(net, x.cuda(non_blocking=True))
                for i in range(len(x)):
                    k = score[i] > perc.MIN_SCORE
                    out.append({"score": score[i][k], "lp": probs["picto"][i][k].clamp_min(1e-12).log(),
                                "lc": probs["color"][i][k].clamp_min(1e-12).log(), "junction": probs["junction"][i][k],
                                "box": boxes[i][k], "size": sizes[i]})
            return out
        return self.base.cached(f"rich_e{epoch}_{tag}", len(paths), run)

    def decode(self, d, tau):
        lp = d["lp"][:, : len(PICTOS)] - tau * self.log_prior["picto"][: d["lp"].shape[1]]
        lc = d["lc"] - tau * self.log_prior["color"]
        return lp.argmax(-1), lc.argmax(-1)

    def frames(self, epoch, tag, records, tau):
        out = []
        for d in self.rich(epoch, tag, [r["image"] for r in records]):
            p, c = self.decode(d, tau)
            out.append({"score": d["score"], "picto": p, "color": c, "junction": d["junction"], "box": d["box"]})
        return out

    def boxes(self, epoch, ds, items, tau):
        out = []
        for d in self.rich(epoch, ds, [it["path"] for it in items]):
            w, h = d["size"].tolist()
            b = box_convert(d["box"], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            keep = nms(b, d["score"], perc.NMS_IOU) if len(d["score"]) else torch.zeros(0, dtype=torch.long)
            p, c = self.decode(d, tau)
            v = p * 6 + (d["junction"] > 0.5).long() * 3 + c
            out.append((v[keep], d["score"][keep], b[keep]))
        return out


class Fixed:
    """A run decoded as F8 (eval_exp8.Model), with the same call signature as LA (tau ignored, must be 0)."""

    def __init__(self, name, run, data, cache):
        self.base = e8.Model(name, run, data, cache)
        self.epochs = self.base.epochs

    def frames(self, epoch, tag, records, tau=0.0):
        assert tau == 0.0
        return self.base.frames(epoch, tag, records)

    def boxes(self, epoch, ds, items, tau=0.0):
        assert tau == 0.0
        return self.base.boxes(epoch, ds, items)


# ------------------------------------------------------------------ selection (the Exp 8 rule, plus tau for LA)
def select(name, m, sets, dtld_tune, taus):
    grid = {}
    for e in m.epochs:
        for tau in taus:
            d_dt = m.frames(e, "dtld_tune", dtld_tune, tau)
            ol = {n: m.frames(e, n, sets[n][1], tau) for n in ("A_tune", "B_tune")}
            for t in THRESHOLDS:
                v = {"DTLD_tune": rule_metrics(dtld_tune, d_dt, t)[0][3]["tss"]}
                for n, d in ol.items():
                    recs, _, masks, (gc, gp) = sets[n]
                    v[n] = evaluate(*e8.us_topo(recs, d, masks, t), gc, gp)["tss"]
                v["mean"] = float(np.mean(list(v.values())))
                grid[(e, tau, float(t))] = v
        print(f"{name} epoch {e}: {max(v['mean'] for k, v in grid.items() if k[0] == e):.4f}", flush=True)
    # ties: lower epoch, then smaller tau (closer to F8), then lower threshold
    best = max(grid, key=lambda k: (grid[k]["mean"], -k[0], -k[1], -k[2]))
    print(f"{name} selected epoch {best[0]}, tau {best[1]}, threshold {best[2]}: {grid[best]}", flush=True)
    return {"epoch": best[0], "tau": best[1], "threshold": best[2], "tune": grid[best],
            "grid": {f"e{k[0]}_tau{k[1]}_t{k[2]}": v["mean"] for k, v in grid.items()}}


# ------------------------------------------------------------------ guards G1 / G2
def renamed(res, a, b):
    """eval_exp7.paired names its two models F7 and M."""
    return {(a + k[2:] if k.startswith("F7_") else b + k[1:] if k.startswith("M_") else k): v for k, v in res.items()}


def guards(models, sel, sets, test, a, b):
    out = {}
    ma, mb, sa, sb = models[a], models[b], sel[a], sel[b]
    for name in ("A_val", "B_val"):
        recs, abs_recs, masks, (gc, gp) = sets[name]
        fa = e7.frame_stats(*e8.us_topo(recs, ma.frames(sa["epoch"], name, abs_recs, sa["tau"]), masks, sa["threshold"]), gc, gp)
        fb = e7.frame_stats(*e8.us_topo(recs, mb.frames(sb["epoch"], name, abs_recs, sb["tau"]), masks, sb["threshold"]), gc, gp)
        out[name] = renamed(e7.paired(fa[:3], fb[:3], np.array([r["segment_id"] for r in recs])), a, b)
    (s1, n1, d1, _), idx = rule_metrics(test, ma.frames(sa["epoch"], "dtld_test", test, sa["tau"]), sa["threshold"])
    (s2, n2, d2, _), _ = rule_metrics(test, mb.frames(sb["epoch"], "dtld_test", test, sb["tau"]), sb["threshold"])
    out["DTLD_test"] = renamed(e7.paired((s1, n1, d1), (s2, n2, d2), np.array([test[i]["segment_id"] for i in idx])), a, b)
    return out


# ------------------------------------------------------------------ attribute sets and paired light rows
def hand_labelled(name):
    """OpenLane val items whose pictogram is the blind hand label where one exists (others None)."""
    items, size = att.openlane_items(name)
    labels = json.load(open("annotations/openlane_pictograms_v1.json"))["labels"]
    sub = name[0]
    want = {}
    for line in open("data/annotation/pictogram_v1/items.jsonl"):
        it = json.loads(line)
        if it["qc_of"] is None and it["subset"] == sub and labels.get(it["id"]) in PICTOS:
            for bi in it["box_indices"]:
                want[(it["segment_id"], str(it["timestamp"]), bi)] = labels[it["id"]]
    for it, r in zip(items, load_records(*SETS[name])):
        it["picto"] = [want.get((r["segment_id"], str(r["timestamp"]), bi)) for bi in range(len(it["color"]))]
    return items, size


def attribute_sets():
    """name -> (items, kind, key, size): kind "frame" (key = cache tag, size for pixels) or "perc" (key = dataset)."""
    out = {"DTLD_test": (att.dtld_items(), "frame", "dtld_test", att.DTLD_SIZE)}
    for n in ("A_val", "B_val"):
        items, size = hand_labelled(n)
        out[n] = (items, "frame", n, size)
    for name, ds in (("ATLAS_test", "atlas"), ("LISA_test", "lisa"), ("BDD_val", "bdd100k"), ("S2TLD", "s2tld"), ("VZC", "vzc")):
        out[name] = (att.perception_items(ds), "perc", ds, None)
    return out


def preds_for(m, s, kind, key, items, size, records):
    if kind == "frame":
        return att.frame_preds(m.frames(s["epoch"], key, records, s["tau"]), size)
    return att.virtual_preds(m.boxes(s["epoch"], key, items, s["tau"]))


def light_rows(items, pa, ta, pb, tb):
    """Lights matched by both models above their thresholds -> dict of arrays."""
    keys = ("cluster", "group", "gt_p", "gt_c", "gt_j", "a_p", "a_c", "a_j", "b_p", "b_c", "b_j", "height")
    rows = {k: [] for k in keys}
    for it, A, B in zip(items, pa, pb):
        ign = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
        hit = []
        for (s, p, c, j, b), t in ((A, ta), (B, tb)):
            m = perc.match(b, s, it["boxes"], ign, "center")
            hit.append({int(m[q]): q for q in torch.nonzero((s > t) & (m >= 0)).flatten().tolist()})
        for g in sorted(hit[0].keys() & hit[1].keys()):
            qa, qb = hit[0][g], hit[1][g]
            gp = it["picto"][g]
            rows["cluster"].append(str(it["cluster"]))
            rows["group"].append(it.get("group", ""))
            rows["gt_p"].append(gp if gp in PICTOS else "")
            rows["gt_c"].append(it["color"][g])
            rows["gt_j"].append(-1 if it["junction"][g] is None else int(it["junction"][g]))
            for side, (s, p, c, j, b), q in (("a", A, qa), ("b", B, qb)):
                rows[f"{side}_p"].append(PICTOS[int(p[q])])
                rows[f"{side}_c"].append(COLORS[int(c[q])])
                rows[f"{side}_j"].append(int(j[q]))
            rows["height"].append(float(it["boxes"][g, 3] - it["boxes"][g, 1]))
    return {k: np.array(v) for k, v in rows.items()}


def boot(rows, groups, ok_a, ok_b):
    """Macro mean over groups of per-group rates (a group with no row is left out), each model and the difference;
    cluster bootstrap. groups: int array per row (-1 = not counted)."""
    n_g = int(groups.max()) + 1 if len(groups) and groups.max() >= 0 else 0
    if n_g == 0:
        return None
    uniq, cl = np.unique(rows["cluster"], return_inverse=True)
    use = groups >= 0
    shape = (len(uniq), n_g)
    n, a, b = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(n, (cl[use], groups[use]), 1)
    np.add.at(a, (cl[use], groups[use]), ok_a[use])
    np.add.at(b, (cl[use], groups[use]), ok_b[use])

    def rate(w):
        nn, aa, bb = w @ n, w @ a, w @ b
        with np.errstate(invalid="ignore", divide="ignore"):
            ra, rb = aa / nn, bb / nn
        return np.nanmean(ra, -1), np.nanmean(rb, -1)

    va, vb = rate(np.ones(len(uniq)))
    rng = np.random.default_rng(SEED)
    w = np.zeros((DRAWS, len(uniq)))
    np.add.at(w, (np.arange(DRAWS)[:, None], rng.integers(0, len(uniq), (DRAWS, len(uniq)))), 1)
    ba, bb = rate(w)
    d = ba - bb
    return {"a": float(va), "b": float(vb), "d": float(va - vb), "d_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "n": int(n.sum()), "n_by_group": n.sum(0).astype(int).tolist()}


def metric(rows, kind, classes=None):
    gp, gc, gj = rows["gt_p"], rows["gt_c"], rows["gt_j"]
    if not len(gp):
        return None
    if kind == "picto_macro":
        g = np.array([classes.index(x) if x in classes else -1 for x in gp])
        return boot(rows, g, rows["a_p"] == gp, rows["b_p"] == gp)
    if kind.startswith("recall_"):
        cls = kind[7:]
        return boot(rows, np.where(gp == cls, 0, -1), rows["a_p"] == gp, rows["b_p"] == gp)
    if kind == "yellow_recall":
        return boot(rows, np.where(gc == "yellow", 0, -1), rows["a_c"] == gc, rows["b_c"] == gc)
    if kind == "color_acc":
        return boot(rows, np.zeros(len(gc), int), rows["a_c"] == gc, rows["b_c"] == gc)
    if kind == "junction_acc":
        return boot(rows, np.where(gj >= 0, 0, -1), rows["a_j"] == gj, rows["b_j"] == gj)
    if kind == "false_arrow_on_circles":
        return boot(rows, np.where(gp == "circle", 0, -1), np.isin(rows["a_p"], list(ARROWS)), np.isin(rows["b_p"], list(ARROWS)))
    raise ValueError(kind)


def attributes(models, sel, asets, sets, test, a, b):
    out = {}
    ma, mb, sa, sb = models[a], models[b], sel[a], sel[b]
    frame_recs = {"dtld_test": test, "A_val": sets["A_val"][1], "B_val": sets["B_val"][1]}
    for name, (items, kind, key, size) in asets.items():
        recs = frame_recs.get(key)
        rows = light_rows(items, preds_for(ma, sa, kind, key, items, size, recs), sa["threshold"],
                          preds_for(mb, sb, kind, key, items, size, recs), sb["threshold"])
        cell = {"lights_matched_by_both": int(len(rows["gt_c"])), "color_acc": metric(rows, "color_acc"),
                "yellow_recall": metric(rows, "yellow_recall")}
        if name == "DTLD_test":
            cell["picto_macro"] = metric(rows, "picto_macro", P1_CLASSES)
            heights = {}
            for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 1e9)):
                sub = {k: v[(rows["height"] >= lo) & (rows["height"] < hi)] for k, v in rows.items()}
                heights[f"{lo:g}-{hi:g}px"] = metric(sub, "picto_macro", P1_CLASSES)
            cell["picto_macro_by_height"] = heights
        if name in REPORT_CLASSES:
            cell["picto_macro"] = metric(rows, "picto_macro", REPORT_CLASSES[name])
        if name in ("DTLD_test", "ATLAS_test", "LISA_test"):
            cell["picto_recall"] = {c: metric(rows, "recall_" + c) for c in P1_CLASSES + ("arrow_straight_left", "arrow_straight_right")
                                    if (rows["gt_p"] == c).any()}
            cell["false_arrow_on_circles"] = metric(rows, "false_arrow_on_circles")
        if name in ("A_val", "B_val"):
            cell["hand_labelled_lights"] = int((rows["gt_p"] != "").sum())
            cell["false_arrow_on_circles"] = metric(rows, "false_arrow_on_circles")
            cell["picto_recall"] = {c: metric(rows, "recall_" + c) for c in ("circle", "arrow_left", "arrow_straight") if (rows["gt_p"] == c).any()}
        if name in ("DTLD_test", "A_val", "B_val", "VZC"):
            cell["junction_acc"] = metric(rows, "junction_acc")
        if name in ("LISA_test", "BDD_val", "S2TLD"):
            for grp in sorted(set(rows["group"].tolist())):
                sub = {k: v[rows["group"] == grp] for k, v in rows.items()}
                cell[f"yellow_recall_{grp}"] = metric(sub, "yellow_recall")
                cell[f"color_acc_{grp}"] = metric(sub, "color_acc")
                if name == "LISA_test":
                    cell[f"recall_arrow_left_{grp}"] = metric(sub, "recall_arrow_left")
        out[name] = cell
    # G4: false arrows on the hand-labelled OpenLane circles, A val + B val together
    both = [light_rows(asets[n][0], preds_for(ma, sa, "frame", n, *asets[n][::3], frame_recs[n]), sa["threshold"],
                       preds_for(mb, sb, "frame", n, *asets[n][::3], frame_recs[n]), sb["threshold"]) for n in ("A_val", "B_val")]
    rows = {k: np.concatenate([r[k] for r in both]) for k in both[0]}
    out["OpenLane_hand_labelled"] = {"false_arrow_on_circles": metric(rows, "false_arrow_on_circles"),
                                     "lights": int((rows["gt_p"] != "").sum())}
    return out


def detection(models, sel, a, b):
    """The Exp 8 perception cells, a vs b (AP with CI, recall / precision / color at each model's threshold)."""
    out = {}
    raw = {ds: (perc.LOADERS[ds]() if ds != "vzc" else e8.vzc.load_items()) for ds in sorted({c[1] for c in e8.CELLS})}
    for name, ds, group, wide in e8.CELLS:
        items = raw[ds]
        pa = models[a].boxes(sel[a]["epoch"], ds, items, sel[a]["tau"])
        pb = models[b].boxes(sel[b]["epoch"], ds, items, sel[b]["tau"])
        idx = [i for i, it in enumerate(items) if group is None or it["group"] == group]
        sub = [items[i] for i in idx]
        perc.parse_class = e8.pp_.virtual_parse
        r = e8.cell(ds, e8.horizontal_only(sub) if wide else sub, [pa[i] for i in idx], [pb[i] for i in idx],
                    sel[a]["threshold"], sel[b]["threshold"])
        out[name] = None if r is None else {k.replace("F8", a, 1).replace("F7", b, 1) if k[:2] in ("F8", "F7") else k: v for k, v in r.items()}
    return out


def lo(x, key="d_ci95"):
    return None if x is None else x[key][0]


def decide(res, arms):
    dec = {}
    for arm in arms:
        g, at = res[f"guards_{arm}_vs_F8"], res[f"attributes_{arm}_vs_F8"]
        sets3 = [g[n] for n in ("A_val", "B_val", "DTLD_test")]
        d = {"G1_tss": all(c["d_tss_ci95"][0] > -0.01 for c in sets3),
             "G2_false_go": all(c["d_false_go"] <= 0.01 for c in sets3),
             "G3_color_acc": lo(at["DTLD_test"]["color_acc"]) > -0.005,
             "G4_false_arrows": at["OpenLane_hand_labelled"]["false_arrow_on_circles"]["d"] <= 0.03,
             "P1_picto_macro": lo(at["DTLD_test"]["picto_macro"]) > 0,
             "P2_yellow": lo(at["DTLD_test"]["yellow_recall"]) > 0}
        d["success"] = d["G1_tss"] and d["G2_false_go"] and d["G3_color_acc"] and d["G4_false_arrows"] and (d["P1_picto_macro"] or d["P2_yellow"])
        dec[arm] = d
    adopted = None
    trained = [a for a in ("W", "WF") if a in dec and dec[a]["success"]]
    if dec.get("LA", {}).get("success"):
        beats = [a for a in trained if f"attributes_{a}_vs_LA" in res
                 and lo(res[f"attributes_{a}_vs_LA"]["DTLD_test"]["picto_macro"]) > 0
                 and lo(res[f"attributes_{a}_vs_LA"]["DTLD_test"]["yellow_recall"]) > 0]
        adopted = "LA" if not beats else None
        trained = beats if beats else trained
    if adopted is None and trained:
        if "WF" in trained and "W" in trained:
            adopted = "WF" if lo(res["attributes_WF_vs_W"]["DTLD_test"]["picto_macro"]) > 0 else "W"
        else:
            adopted = trained[0]
    dec["adopted"] = adopted
    dec["complete"] = all(a in dec for a in ("LA", "W", "WF"))
    return dec


def main():
    global WORKERS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=["LA", "W", "WF"], choices=["LA", "W", "WF"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="smoke test: evenly spaced N frames per set (caches under runs/exp9_eval_smoke)")
    args = ap.parse_args()
    global OUT
    WORKERS = args.workers
    if args.limit:
        e8.LIMIT = args.limit
        OUT = "runs/exp9_eval_smoke"
    os.makedirs(OUT, exist_ok=True)
    r8 = json.load(open(os.path.join(F8_RUN, "exp8_eval", "result.json")))["selection"]
    sel = {"F8": {**{k: r8["F8"][k] for k in ("epoch", "threshold")}, "tau": 0.0},
           "F7": {**{k: r8["F7"][k] for k in ("epoch", "threshold")}, "tau": 0.0}}
    smoke = bool(args.limit)
    models = {"F8": Fixed("F8", F8_RUN, DATA, os.path.join(OUT, "F8") if smoke else os.path.join(F8_RUN, "exp8_eval")),
              "F7": Fixed("F7", F7_RUN, e8.F7_DATA, os.path.join(OUT, "F7") if smoke else os.path.join(F7_RUN, "exp7_eval"))}
    for arm in args.arms:
        if arm == "LA":
            models["LA"] = LA(WORKERS)
        else:
            run = f"runs/exp9_{arm}"
            models[arm] = Fixed(arm, run, DATA, os.path.join(OUT, arm))

    sets = {n: e8.openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dtld_tune = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "valid", "records.jsonl"))])
    test = e8.sample([json.loads(l) for l in open(os.path.join(e8.EXP6, "dtld_test_records.jsonl"))])
    asets = attribute_sets()
    if smoke:  # the same evenly spaced frames as the frame sets (eval_exp8.sample)
        asets = {k: (e8.sample(v[0]), *v[1:]) for k, v in asets.items()}

    res = {"selection": {}}
    for arm in args.arms:
        sel[arm] = select(arm, models[arm], sets, dtld_tune, TAUS if arm == "LA" else (0.0,))
        res["selection"][arm] = sel[arm]
    res["selection"]["F8"], res["selection"]["F7"] = sel["F8"], sel["F7"]

    pairs = [(arm, "F8") for arm in args.arms] + [("F7", "F8")]
    if "LA" in args.arms:
        pairs += [(a, "LA") for a in ("W", "WF") if a in args.arms]
    if "W" in args.arms and "WF" in args.arms:
        pairs.append(("WF", "W"))
    for a, b in pairs:
        if b == "F8" and a != "F7":
            res[f"guards_{a}_vs_F8"] = guards(models, sel, sets, test, a, b)
            print(a, "guards", {k: (round(v["d_tss"], 4), [round(x, 4) for x in v["d_tss_ci95"]], round(v["d_false_go"], 4))
                                for k, v in res[f"guards_{a}_vs_F8"].items()}, flush=True)
        res[f"attributes_{a}_vs_{b}"] = attributes(models, sel, asets, sets, test, a, b)
        dt = res[f"attributes_{a}_vs_{b}"]["DTLD_test"]
        print(a, "vs", b, "DTLD picto macro", {k: round(v, 4) if isinstance(v, float) else v for k, v in dt["picto_macro"].items() if k != "n_by_group"},
              "yellow", {k: round(v, 4) if isinstance(v, float) else v for k, v in dt["yellow_recall"].items() if k != "n_by_group"}, flush=True)
        if b == "F8" and not smoke:
            res[f"detection_{a}_vs_F8"] = detection(models, sel, a, b)
    res["decision"] = decide(res, [a for a in args.arms])
    print("decision", res["decision"], flush=True)
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
