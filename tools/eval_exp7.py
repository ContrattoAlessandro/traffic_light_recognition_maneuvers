"""Exp 7 (EXPERIMENT_LOG.md): select and score the multi-dataset factorized model F7, score the moves model M
(runs/rfdetr_m_mix, epoch 1, threshold 0.25) on the same cells, compare and apply the decision rule.

1. F7 selection: every saved epoch on DTLD tune (junction rule), A tune and B tune (map-masked); the epoch and
   threshold (0.05-0.95) with the best mean of the three TSS values.
2. In domain (paired, segment / session bootstrap 95% CI, 10,000 draws, seed 0):
   - A val, B val: map-masked TSS for both (directions labelled "none" -> "none"); M's native TSS as reference
     (M's per-direction predictions come from runs/rfdetr_m_mix/val_preds_mix.npz);
   - DTLD test, junction-rule ground truth (the Exp 6 scored frames): F7 with the junction rule, M with its own
     decoder under the German rule ("signal").
3. Transfer: LISA test, S2TLD, BDD100K val (and ATLAS test, reported only):
   - center AP, recall / precision, color accuracy (tools/eval_rfdetr_perception.summarize);
   - arrows: exact moves on matched (center) arrows the model calls relevant (M) / junction (F7);
     F7 - M with an image bootstrap (too narrow: few sequences).
Caches raw detections in runs/exp7_multi_F/exp7_eval/. Writes runs/exp7_multi_F/exp7_eval/result.json.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_rfdetr_perception as perc  # noqa: E402
from dtsu import COLORS as DCOLORS, PASS  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import detect, frame_directions, parse_class, to_arrays  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_picto import THRESHOLDS, Images, load_model, raw_detections, rule_metrics  # noqa: E402
from tools.eval_picto_openlane import predict  # noqa: E402
import tools.eval_picto_perception as pp_  # noqa: E402  (patches perc.parse_class; reset below)
from tools.eval_rfdetr import load_model as load_moves  # noqa: E402

perc.parse_class = parse_class
RUN, DATA = "runs/exp7_multi_F", "data/rfdetr_coco_multi_picto"
EXP6 = "data/rfdetr_coco_dtld_picto"
M_RUN, M_EPOCH, M_THR, M_CLASSES = "runs/rfdetr_m_mix", 1, 0.25, "data/rfdetr_coco_fit_mix/class_names.json"
OOD = ("lisa", "s2tld", "bdd100k", "atlas")
NONE, GO, STOP = DCOLORS.index("none"), PASS.index("go"), PASS.index("stop")


def frame_stats(pc, pp, gc, gp):
    """Per-frame TSS contribution, False-Go numerator / denominator, metrics (as tools.eval_hybrid.per_frame)."""
    c_ok, p_ok = pc == gc, pp == gp
    s = (c_ok.mean(1) + p_ok.mean(1) + (c_ok & p_ok).all(1)) / 3
    return s, ((gp == STOP) & (pp == GO)).sum(1), (gp == STOP).sum(1), evaluate(pc, pp, gc, gp)


def mask(pc, pp, gc):
    return np.where(gc == NONE, NONE, pc), np.where(gc == NONE, GO, pp)


LIMIT = None


def openlane(name, workers, net=None, epoch=None, res=None, cache=None):
    recs = load_records(*SETS[name])[:LIMIT]
    for r in recs:
        r["image"] = os.path.abspath(os.path.join(r["root"], r["image"]))
    if cache and os.path.exists(cache):
        return recs, torch.load(cache, weights_only=False)
    d = raw_detections(net(), recs, CLASS_NAMES, res, "F", workers)
    if cache:
        torch.save(d, cache)
    return recs, d


def f7_openlane_arrays(recs, dets, thr):
    pc, pp = to_arrays([predict(d, thr, r["traffic_side"]) for d, r in zip(dets, recs)])
    gc, gp = to_arrays([r["color"] for r in recs])
    return (*mask(pc, pp, gc), gc, gp)


def paired(a, b, segments, draws=10000, seed=0):
    """a, b: (s, fg_num, fg_den) per frame for F7 and M on the same frames. F7 - M, cluster bootstrap."""
    uniq, inv = np.unique(segments, return_inverse=True)
    sums = lambda x: np.bincount(inv, weights=np.asarray(x, float), minlength=len(uniq))  # noqa: E731
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    ds, fa, fb, den = sums(a[0] - b[0]), sums(a[1]), sums(b[1]), sums(a[2])
    rng = np.random.default_rng(seed)
    w = np.zeros((draws, len(uniq)))
    np.add.at(w, (np.arange(draws)[:, None], rng.integers(0, len(uniq), (draws, len(uniq)))), 1)
    d_tss = (w @ ds) / (w @ n)
    d_fg = (w @ fa - w @ fb) / np.maximum(w @ den, 1)
    return {"F7_tss": float(a[0].mean()), "M_tss": float(b[0].mean()), "d_tss": float(ds.sum() / n.sum()),
            "d_tss_ci95": [float(np.percentile(d_tss, 2.5)), float(np.percentile(d_tss, 97.5))],
            "F7_false_go": float(a[1].sum() / max(a[2].sum(), 1)), "M_false_go": float(b[1].sum() / max(b[2].sum(), 1)),
            "d_false_go": float((fa.sum() - fb.sum()) / max(den.sum(), 1)),
            "d_false_go_ci95": [float(np.percentile(d_fg, 2.5)), float(np.percentile(d_fg, 97.5))], "frames": int(len(a[0]))}


def arrow_rows(items, preds, thr, moves_of):
    """Per arrow matched (center) to a kept detection called relevant: [image, exact]."""
    rows = []
    for i, (it, (c, s, b)) in enumerate(zip(items, preds)):
        k = s > thr
        c, s, b = c[k], s[k], b[k]
        ignore = torch.tensor([col is None for col in it["color"]], dtype=torch.bool)
        m = perc.match(b, s, it["boxes"], ignore, "center")
        for p, g in enumerate(m.tolist()):
            if g < 0 or it["arrow"][g] is None:
                continue
            mv = moves_of(int(c[p]))
            if mv is not None:
                rows.append((i, float(mv == it["arrow"][g])))
    return np.array(rows, float).reshape(-1, 2)


def arrow_diff(rf, rm, n_images, draws=10000, seed=0):
    ef, nf = np.bincount(rf[:, 0].astype(int), rf[:, 1], n_images), np.bincount(rf[:, 0].astype(int), minlength=n_images)
    em, nm = np.bincount(rm[:, 0].astype(int), rm[:, 1], n_images), np.bincount(rm[:, 0].astype(int), minlength=n_images)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws // 100):
        w = np.zeros((100, n_images))
        np.add.at(w, (np.arange(100)[:, None], rng.integers(0, n_images, (100, n_images))), 1)
        boots.append((w @ ef) / np.maximum(w @ nf, 1) - (w @ em) / np.maximum(w @ nm, 1))
    boots = np.concatenate(boots)
    f, m = ef.sum() / max(nf.sum(), 1), em.sum() / max(nm.sum(), 1)
    return {"F7_exact": float(f), "F7_n": int(nf.sum()), "M_exact": float(m), "M_n": int(nm.sum()), "d_exact": float(f - m),
            "d_exact_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    global CLASS_NAMES, LIMIT, RUN
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--limit", type=int, help="first N frames / images per set; writes to --out-dir (smoke tests)")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    RUN, LIMIT = args.run, args.limit
    CLASS_NAMES = json.load(open(os.path.join(args.data_dir, "class_names.json")))
    out_dir = args.out_dir or os.path.join(RUN, "exp7_eval")
    os.makedirs(out_dir, exist_ok=True)
    res = json.load(open(os.path.join(RUN, "args.json")))["resolution"]
    nets = {}

    def f7(epoch):
        if epoch not in nets:
            nets.clear()
            torch.cuda.empty_cache()
            nets[epoch] = load_model(RUN, epoch, CLASS_NAMES, "F")[0]
        return nets[epoch]

    # 1. selection
    dtld_tune = [json.loads(l) for l in open(os.path.join(EXP6, "valid", "records.jsonl"))][:LIMIT]
    epochs = sorted(int(re.search(r"epoch_(\d+)_ema", p).group(1)) for p in glob.glob(os.path.join(RUN, "epoch_*_ema.pth")))
    grid = {}
    for e in epochs:
        c = os.path.join(out_dir, f"e{e}_dtld_tune.pt")
        if os.path.exists(c):
            d_dt = torch.load(c, weights_only=False)
        else:
            d_dt = raw_detections(f7(e), dtld_tune, CLASS_NAMES, res, "F", args.workers)
            torch.save(d_dt, c)
        ol = {n: openlane(n, args.workers, lambda: f7(e), e, res, os.path.join(out_dir, f"e{e}_{n}.pt")) for n in ("A_tune", "B_tune")}
        grid[e] = {}
        for t in THRESHOLDS:
            v = {"DTLD_tune": rule_metrics(dtld_tune, d_dt, t)[0][3]["tss"]}
            for n, (recs, d) in ol.items():
                v[n] = evaluate(*f7_openlane_arrays(recs, d, t))["tss"]
            v["mean"] = float(np.mean(list(v.values())))
            grid[e][float(t)] = v
        bt = max(grid[e], key=lambda t: grid[e][t]["mean"])
        print(f"epoch {e}: best mean tune TSS {grid[e][bt]['mean']:.4f} @ {bt} {grid[e][bt]}", flush=True)
    epoch, thr = max(((e, t) for e in grid for t in grid[e]), key=lambda et: (grid[et[0]][et[1]]["mean"], -et[0]))
    print(f"selected epoch {epoch}, threshold {thr}: {grid[epoch][thr]}", flush=True)
    out = {"F7": {"epoch": epoch, "threshold": thr, "tune": grid[epoch][thr], "grid": {str(e): g for e, g in grid.items()}},
           "M": {"run": M_RUN, "epoch": M_EPOCH, "threshold": M_THR}}

    # 2. in domain
    npz = np.load(os.path.join(M_RUN, "val_preds_mix.npz"))
    m_classes = json.load(open(M_CLASSES))
    m_net = None
    for name, key in (("A_val", "A"), ("B_val", "B")):
        recs, d = openlane(name, args.workers, lambda: f7(epoch), epoch, res, os.path.join(out_dir, f"e{epoch}_{name}.pt"))
        a = frame_stats(*f7_openlane_arrays(recs, d, thr))
        mpc, mpp, mgc, mgp = (npz[f"{key}_{x}"][: len(recs)] for x in ("pc", "pp", "gc", "gp"))
        gc, gp = to_arrays([r["color"] for r in recs])
        assert (gc == mgc).all() and (gp == mgp).all(), name
        b = frame_stats(*mask(mpc, mpp, mgc), mgc, mgp)
        out[name] = paired(a[:3], b[:3], np.array([r["segment_id"] for r in recs]))
        out[name]["M_native_tss"] = evaluate(mpc, mpp, mgc, mgp)["tss"]
        out[name]["F7_metrics"], out[name]["M_masked_metrics"] = a[3], b[3]
        print(name, {k: v for k, v in out[name].items() if "metrics" not in k}, flush=True)

    test = [json.loads(l) for l in open(os.path.join(EXP6, "dtld_test_records.jsonl"))][:LIMIT]
    c = os.path.join(out_dir, f"e{epoch}_dtld_test.pt")
    if os.path.exists(c):
        d = torch.load(c, weights_only=False)
    else:
        d = raw_detections(f7(epoch), test, CLASS_NAMES, res, "F", args.workers)
        torch.save(d, c)
    (s, fgn, fgd, mf), idx = rule_metrics(test, d, thr)
    c = os.path.join(out_dir, "M_dtld_test.pt")
    if os.path.exists(c):
        md = torch.load(c, weights_only=False)
    else:
        m_net, m_res, _ = load_moves(M_RUN, os.path.join(M_RUN, f"epoch_{M_EPOCH}_ema.pth"), len(m_classes))
        md = []
        for x in DataLoader(Images([test[i] for i in idx], m_res), batch_size=8, num_workers=args.workers):
            cls, score, boxes = detect(m_net, x.cuda(non_blocking=True), m_classes)
            md += [(cc[ss > 0.05], ss[ss > 0.05], bb[ss > 0.05]) for cc, ss, bb in zip(cls, score, boxes)]
        torch.save(md, c)
    mpc, mpp = to_arrays([frame_directions(*x, m_classes, M_THR, "right", "signal") for x in md])
    gc, gp = to_arrays([test[i]["color"] for i in idx])
    b = frame_stats(mpc, mpp, gc, gp)
    out["DTLD_test"] = paired((s, fgn, fgd), b[:3], np.array([test[i]["segment_id"] for i in idx]))
    out["DTLD_test"]["F7_metrics"], out["DTLD_test"]["M_metrics"] = mf, b[3]
    print("DTLD_test", {k: v for k, v in out["DTLD_test"].items() if "metrics" not in k}, flush=True)

    # 3. transfer
    nets.clear()
    torch.cuda.empty_cache()
    for ds in OOD:
        items = perc.LOADERS[ds]()[:LIMIT]
        c = os.path.join(out_dir, f"e{epoch}_{ds}.pt")
        if os.path.exists(c):
            pf = torch.load(c, weights_only=False)
        else:
            pf = pp_.predictions(f7(epoch), items, CLASS_NAMES, res, "F", args.workers)
            torch.save(pf, c)
        nets.clear()
        torch.cuda.empty_cache()
        c = os.path.join(out_dir, f"M_{ds}.pt")
        if os.path.exists(c):
            pm = torch.load(c, weights_only=False)
        else:
            if m_net is None:
                m_net, m_res, _ = load_moves(M_RUN, os.path.join(M_RUN, f"epoch_{M_EPOCH}_ema.pth"), len(m_classes))
            pm = perc.predictions(m_net, items, m_classes, m_res, args.workers, 8)
            torch.save(pm, c)
        perc.parse_class = pp_.virtual_parse
        sf = perc.summarize(items, pf, pp_.VIRTUAL, thr, "center")
        rf = arrow_rows(items, pf, thr, lambda v: pp_.virtual_parse(pp_.VIRTUAL[v])[0])
        perc.parse_class = parse_class
        sm = perc.summarize(items, pm, m_classes, M_THR, "center")
        rm = arrow_rows(items, pm, M_THR, lambda v: perc.norm_moves(parse_class(m_classes[v])[0]))
        keys = ("AP", "recall", "precision", "color_acc", "color_recall")
        out[ds] = {"F7": {k: sf[k] for k in keys}, "M": {k: sm[k] for k in keys},
                   "arrows": arrow_diff(rf, rm, len(items)) if len(rf) and len(rm) else None}
        print(ds, json.dumps(out[ds]), flush=True)

    # decision rule
    ind = [out[n] for n in ("A_val", "B_val", "DTLD_test")]
    ap_f = np.mean([out[d]["F7"]["AP"] for d in ("lisa", "s2tld", "bdd100k")])
    ap_m = np.mean([out[d]["M"]["AP"] for d in ("lisa", "s2tld", "bdd100k")])
    out["decision"] = {
        "a_in_domain_noninferior": all(c["d_tss_ci95"][0] > -0.01 for c in ind),
        "b_false_go": all(c["d_false_go"] <= 0.01 for c in ind),
        "c1_lisa_arrows": bool(out["lisa"]["arrows"] and out["lisa"]["arrows"]["d_exact_ci95"][0] > 0),
        "c2_ood_ap": bool(ap_f >= ap_m - 0.02), "mean_ood_ap": {"F7": float(ap_f), "M": float(ap_m)}}
    out["decision"]["F7_stronger"] = all(v for k, v in out["decision"].items() if k != "mean_ood_ap")
    print("decision", out["decision"], flush=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
