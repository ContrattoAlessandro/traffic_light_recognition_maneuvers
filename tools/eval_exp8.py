"""Exp 8 (EXPERIMENT_LOG.md): select and score F8 (Exp 7 data + BDD100K train) against F7 under the pre-registered rule.

1. Selection, the same for both models: every saved epoch and threshold, best mean of DTLD tune (junction rule),
   A tune and B tune (US rule + lane topology mask; the Exp 7 label mask is not used).
2. In-domain guards, F8 - F7, segment / session bootstrap: TSS and False-Go on A val, B val (US rule + topology
   mask) and DTLD test (junction labels).
3. Perception, F8 - F7: center AP with a bootstrap CI (blocks of 100 frames on LISA and S2TLD, sessions on VZC,
   images on BDD100K and ATLAS), and recall / precision / color at each model's threshold. Horizontal cells
   (width > 1.2 x height) score only horizontal lights; the other lights are ignore boxes. S2TLD is split by
   resolution: only the 1080p group holds horizontal lights in number.
4. The decision rule. LISA arrows are reported as a check.
Caches detections in <run>/exp8_eval/ (F8) and reuses F7's in runs/exp7_multi_F/exp7_eval/ and runs/vzc_eval/.
Writes <run>/exp8_eval/result.json.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp7 as e7  # noqa: E402
import tools.exp7_checks as ck  # noqa: E402
import tools.eval_vzc as vzc  # noqa: E402
import tools.eval_rfdetr_perception as perc  # noqa: E402
import tools.eval_picto_perception as pp_  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.rfdetr_baseline import to_arrays  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_picto import THRESHOLDS, load_model, raw_detections, rule_metrics  # noqa: E402

F7_RUN, F7_DATA = "runs/exp7_multi_F", "data/rfdetr_coco_multi_picto"
EXP6 = "data/rfdetr_coco_dtld_picto"
HORIZONTAL = 1.2
BLOCK, DRAWS = 100, 2000
LIMIT = None
WORKERS = 6


class Model:
    """One factorized run: its class names, cache folder and network (loaded on first use, one at a time)."""
    loaded = {}

    def __init__(self, name, run, data, cache):
        self.name, self.run, self.cache = name, run, cache
        self.classes = json.load(open(os.path.join(data, "class_names.json")))
        self.res = json.load(open(os.path.join(run, "args.json")))["resolution"]
        self.epochs = sorted(int(re.search(r"epoch_(\d+)_ema", p).group(1)) for p in glob.glob(os.path.join(run, "epoch_*_ema.pth")))
        os.makedirs(cache, exist_ok=True)

    def net(self, epoch):
        key = (self.run, epoch)
        if key not in Model.loaded:
            Model.loaded.clear()
            torch.cuda.empty_cache()
            Model.loaded[key] = load_model(self.run, epoch, self.classes, "F")[0]
        return Model.loaded[key]

    def cached(self, name, n, fn):
        path = os.path.join(self.cache, name + ".pt")
        if os.path.exists(path):
            x = torch.load(path, weights_only=False)
            if len(x) == n:
                return x
        x = fn()
        torch.save(x, path)
        return x

    def frames(self, epoch, tag, records):
        return self.cached(f"e{epoch}_{tag}", len(records), lambda: raw_detections(self.net(epoch), records, self.classes, self.res, "F", WORKERS))

    def boxes(self, epoch, ds, items):
        if self.name == "F7" and ds == "vzc" and epoch == 4 and LIMIT is None:
            x = torch.load("runs/vzc_eval/Exp7_F.pt", weights_only=False)
            if len(x) == len(items):
                return x
        return self.cached(f"e{epoch}_{ds}", len(items), lambda: pp_.predictions(self.net(epoch), items, self.classes, self.res, "F", WORKERS))


def sample(x):
    """Smoke tests: evenly spaced, so every group of a dataset is present."""
    return x if LIMIT is None or len(x) <= LIMIT else [x[i] for i in np.linspace(0, len(x) - 1, LIMIT).astype(int)]


# ------------------------------------------------------------------ frame-level sets
def openlane_set(name):
    """-> (records with relative paths, the same with absolute image paths, topology masks, gt arrays)."""
    recs = sample(load_records(*SETS[name]))
    return recs, [dict(r, image=os.path.abspath(os.path.join(r["root"], r["image"]))) for r in recs], ck.topo_masks(recs), to_arrays([r["color"] for r in recs])


def us_topo(recs, dets, masks, thr):
    return to_arrays([ck.apply_topo(ck.f7_us(d, float(thr), r["traffic_side"]), m) for d, r, m in zip(dets, recs, masks)])


def select(m, sets, dtld_tune):
    grid = {}
    for e in m.epochs:
        d_dt = m.frames(e, "dtld_tune", dtld_tune)
        ol = {n: m.frames(e, n, sets[n][1]) for n in ("A_tune", "B_tune")}
        grid[e] = {}
        for t in THRESHOLDS:
            v = {"DTLD_tune": rule_metrics(dtld_tune, d_dt, t)[0][3]["tss"]}
            for n, d in ol.items():
                recs, _, masks, (gc, gp) = sets[n]
                v[n] = evaluate(*us_topo(recs, d, masks, t), gc, gp)["tss"]
            v["mean"] = float(np.mean(list(v.values())))
            grid[e][float(t)] = v
        print(f"{m.name} epoch {e}: {max(v['mean'] for v in grid[e].values()):.4f}", flush=True)
    epoch, thr = max(((e, t) for e in grid for t in grid[e]), key=lambda et: (grid[et[0]][et[1]]["mean"], -et[0]))
    print(f"{m.name} selected epoch {epoch}, threshold {thr}: {grid[epoch][thr]}", flush=True)
    return {"epoch": epoch, "threshold": thr, "tune": grid[epoch][thr], "grid": {str(e): g for e, g in grid.items()}}


def relabel(res):
    """e7.paired names its two models F7 and M; here they are F8 and F7."""
    return {k.replace("F7_", "F8_", 1) if k.startswith("F7_") else k.replace("M_", "F7_", 1) if k.startswith("M_") else k: v for k, v in res.items()}


# ------------------------------------------------------------------ perception cells
def horizontal_only(items):
    out = []
    for it in items:
        b = it["boxes"]
        wide = (b[:, 2] - b[:, 0]) > HORIZONTAL * (b[:, 3] - b[:, 1])
        out.append(dict(it, color=[c if w else None for c, w in zip(it["color"], wide.tolist())]))
    return out


def blocks(ds, items):
    n = len(items)
    if ds in ("lisa", "s2tld"):
        return np.arange(n) // BLOCK
    if ds == "vzc":
        return np.unique([it["session"] for it in items], return_inverse=True)[1]
    return np.arange(n)


def cell(ds, items, p8, p7, t8, t7):
    n_lights = sum(c is not None for it in items for c in it["color"])
    if not n_lights:
        return None
    a, b = ck.det_table(items, p8), ck.det_table(items, p7)
    full = np.ones(len(items))
    block = blocks(ds, items)
    nb = block.max() + 1
    rng = np.random.default_rng(0)
    diff = [ck.ap_w(a, w) - ck.ap_w(b, w) for w in (np.bincount(rng.integers(0, nb, nb), minlength=nb)[block].astype(float) for _ in range(DRAWS))]
    ap8, ap7 = ck.ap_w(a, full), ck.ap_w(b, full)
    keys = ("recall", "precision", "color_acc", "color_recall")
    s8, s7 = perc.summarize(items, p8, pp_.VIRTUAL, t8, "center"), perc.summarize(items, p7, pp_.VIRTUAL, t7, "center")
    return {"images": len(items), "lights": n_lights, "clusters": int(nb), "F8_AP": ap8, "F7_AP": ap7, "d_AP": ap8 - ap7,
            "d_AP_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "F8": {k: s8[k] for k in keys}, "F7": {k: s7[k] for k in keys}}


# (name, dataset, group or None, horizontal only)
CELLS = (("lisa_night", "lisa", "night", False), ("lisa_day", "lisa", "day", False),
         ("bdd_night", "bdd100k", "night", False), ("bdd_day", "bdd100k", "daytime", False), ("bdd_all", "bdd100k", None, False),
         ("s2tld_1080p_horizontal", "s2tld", "1920x1080", True), ("s2tld_1080p", "s2tld", "1920x1080", False), ("s2tld_720p", "s2tld", "1280x720", False),
         ("bdd_horizontal", "bdd100k", None, True), ("vzc_horizontal", "vzc", None, True), ("vzc_all", "vzc", None, False), ("atlas_test", "atlas", None, False))


def main():
    global LIMIT, WORKERS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/exp8_bdd_F")
    ap.add_argument("--data-dir", default="data/rfdetr_coco_exp8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="evenly spaced N frames per set, caches under --out-dir (smoke tests)")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    LIMIT, WORKERS = args.limit, args.workers
    out_dir = args.out_dir or os.path.join(args.run, "exp8_eval")
    f8 = Model("F8", args.run, args.data_dir, out_dir)
    f7 = Model("F7", F7_RUN, F7_DATA, os.path.join(out_dir, "F7") if LIMIT else os.path.join(F7_RUN, "exp7_eval"))

    sets = {n: openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dtld_tune = sample([json.loads(l) for l in open(os.path.join(EXP6, "valid", "records.jsonl"))])
    sel = {m.name: select(m, sets, dtld_tune) for m in (f8, f7)}
    e8, t8, e7_, t7 = sel["F8"]["epoch"], sel["F8"]["threshold"], sel["F7"]["epoch"], sel["F7"]["threshold"]
    out = {"selection": sel}

    for name in ("A_val", "B_val"):
        recs, abs_recs, masks, (gc, gp) = sets[name]
        a = e7.frame_stats(*us_topo(recs, f8.frames(e8, name, abs_recs), masks, t8), gc, gp)
        b = e7.frame_stats(*us_topo(recs, f7.frames(e7_, name, abs_recs), masks, t7), gc, gp)
        out[name] = {**relabel(e7.paired(a[:3], b[:3], np.array([r["segment_id"] for r in recs]))), "F8_metrics": a[3], "F7_metrics": b[3]}
        print(name, {k: v for k, v in out[name].items() if "metrics" not in k}, flush=True)

    test = sample([json.loads(l) for l in open(os.path.join(EXP6, "dtld_test_records.jsonl"))])
    (s8, fn8, fd8, m8), idx = rule_metrics(test, f8.frames(e8, "dtld_test", test), t8)
    (s7, fn7, fd7, m7), _ = rule_metrics(test, f7.frames(e7_, "dtld_test", test), t7)
    out["DTLD_test"] = {**relabel(e7.paired((s8, fn8, fd8), (s7, fn7, fd7), np.array([test[i]["segment_id"] for i in idx]))), "F8_metrics": m8, "F7_metrics": m7}
    print("DTLD_test", {k: v for k, v in out["DTLD_test"].items() if "metrics" not in k}, flush=True)

    perc.parse_class = pp_.virtual_parse
    data = {}
    for ds in sorted({c[1] for c in CELLS}):
        items = sample(vzc.load_items() if ds == "vzc" else perc.LOADERS[ds]())
        data[ds] = (items, f8.boxes(e8, ds, items), f7.boxes(e7_, ds, items))
    for name, ds, group, wide in CELLS:
        items, p8, p7 = data[ds]
        idx = [i for i, it in enumerate(items) if group is None or it["group"] == group]
        sub = [items[i] for i in idx]
        out[name] = cell(ds, horizontal_only(sub) if wide else sub, [p8[i] for i in idx], [p7[i] for i in idx], t8, t7)
        if out[name]:
            print(name, {k: v for k, v in out[name].items() if k not in ("F8", "F7")}, "recall", out[name]["F8"]["recall"], out[name]["F7"]["recall"], flush=True)

    items, p8, p7 = data["lisa"]
    moves = lambda v: pp_.virtual_parse(pp_.VIRTUAL[v])[0]  # noqa: E731
    r8, r7 = e7.arrow_rows(items, p8, t8, moves), e7.arrow_rows(items, p7, t7, moves)
    out["lisa_arrows"] = relabel(e7.arrow_diff(r8, r7, len(items))) if len(r8) and len(r7) else None

    guards = [out[n] for n in ("A_val", "B_val", "DTLD_test")]
    ok = lambda c, k, lim: bool(out[c] and out[c][k][0] > lim)  # noqa: E731
    d = {"a_tss_noninferior": all(c["d_tss_ci95"][0] > -0.01 for c in guards), "b_false_go": all(c["d_false_go"] <= 0.01 for c in guards),
         "c_lisa_night_ap": ok("lisa_night", "d_AP_ci95", 0.0), "d_s2tld_horizontal_ap": ok("s2tld_1080p_horizontal", "d_AP_ci95", 0.0),
         "e_lisa_day_ap": ok("lisa_day", "d_AP_ci95", -0.02)}
    d["F8_replaces_F7"] = all(d.values())
    out["decision"] = d
    print("decision", d, flush=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
