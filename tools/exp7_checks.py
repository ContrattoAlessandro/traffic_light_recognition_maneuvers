"""Follow-up checks on Exp 7 that need no training (exploratory, decided after the Exp 7 results were known).

F7 = runs/exp7_multi_F (epoch 4, threshold 0.20), M = runs/rfdetr_m_mix (epoch 1, threshold 0.25). Reuses cached detections.
- arrows: LISA test arrows under three denominators, day / night, F7 - M with a block bootstrap (blocks of 100 consecutive frames);
- ap: bootstrap CIs of the center AP difference on LISA (blocks of 100 frames), S2TLD and BDD100K (images), and of the mean over the three;
- topo: A val and B val without using the labels. F7 is decoded with the US rule (right only from a right arrow, left falls back
  on straight, as M's decoder), alone and with a mask built only from lane topology (no traffic light is used): a maneuver is
  blanked when no lane of the ego approach at the nearest junction leads to it, keeping the benchmark's inheritance (left and
  U-turn are blanked only if neither left nor straight exist). Compared with M native and M with the same mask;
- route: DTLD test against the route labels (data/labels/dtld_test.jsonl, German rule), the target M trained on; frames in
  both the route labels and the junction-scored set;
- fg: every False-Go direction (label red, prediction not red) attributed, on A val, B val and DTLD test (junction labels), to
  the first that applies: the red light was found, called relevant and red (decoder chose another light); found and relevant,
  wrong color; found but called not relevant; not found;
- latency: network and end-to-end time at batch size 1 on an idle GPU.
Writes runs/exp7_multi_F/exp7_eval/checks_<name>.json.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.eval_exp7 as e7  # noqa: E402
import tools.eval_rfdetr_perception as perc  # noqa: E402
import tools.eval_picto_perception as pp_  # noqa: E402
from dtsu import factorized  # noqa: E402
from dtsu.data import load_records  # noqa: E402
from dtsu.labels import ALIGNED_DEG, AHEAD_X, SAME_JUNCTION_M, _heading, _wrap, lane_movements  # noqa: E402
from dtsu.metrics import evaluate  # noqa: E402
from dtsu.picto_rules import ARROW_MOVES, COLORS, PICTOS  # noqa: E402
from dtsu.rfdetr_baseline import detect, frame_directions, load_image, parse_class, to_arrays  # noqa: E402
from tools.eval_hybrid import SETS  # noqa: E402
from tools.eval_picto import Images, directions, kept, load_model, THRESHOLDS  # noqa: E402
from tools.eval_rfdetr import load_model as load_moves  # noqa: E402

D = "runs/exp7_multi_F/exp7_eval"
EPOCH, THR = 4, 0.20
M_THR = e7.M_THR
M_CLASSES = json.load(open(e7.M_CLASSES))
F7_CLASSES = json.load(open("data/rfdetr_coco_multi_picto/class_names.json"))
BLOCK = 100
EXP6 = "data/rfdetr_coco_dtld_picto"


def save(name, out):
    with open(os.path.join(D, f"checks_{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(name, json.dumps(out)[:3000], flush=True)


def m_moves(c):
    return perc.norm_moves(parse_class(M_CLASSES[c])[0])


def load_ood(ds):
    items = perc.LOADERS[ds]()
    return items, torch.load(f"{D}/e{EPOCH}_{ds}.pt", weights_only=False), torch.load(f"{D}/M_{ds}.pt", weights_only=False)


def blocks_of(ds, n):
    return np.arange(n) // BLOCK if ds == "lisa" else np.arange(n)


# ------------------------------------------------------------------ arrows
def arrows():
    items, pf, pm = load_ood("lisa")
    n = len(items)
    keys = ("gt", "matched", "called", "exact", "shape")
    counts = {m: {k: np.zeros(n) for k in keys} for m in ("F7", "M")}
    for i, it in enumerate(items):
        ign = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
        for name, (c, s, b), thr in (("F7", pf[i], THR), ("M", pm[i], M_THR)):
            k = s > thr
            c, s, b = c[k], s[k], b[k]
            o = counts[name]
            o["gt"][i] = sum(a is not None and col is not None for a, col in zip(it["arrow"], it["color"]))
            for p, g in enumerate(perc.match(b, s, it["boxes"], ign, "center").tolist()):
                if g < 0 or it["arrow"][g] is None:
                    continue
                o["matched"][i] += 1
                v = int(c[p])
                if name == "F7":
                    pic, j, _ = pp_.VIRTUAL[v].split("|")
                    moves = set(ARROW_MOVES.get(pic, {pic}))
                    called, shape = j == "J", moves == it["arrow"][g]
                    exact = called and shape
                else:
                    mv = m_moves(v)
                    called = mv is not None
                    exact = shape = called and mv == it["arrow"][g]
                o["called"][i] += called
                o["exact"][i] += exact
                o["shape"][i] += shape
    block = blocks_of("lisa", n)
    groups = {"all": np.ones(n, bool), "day": np.array([it["group"] == "day" for it in items]), "night": np.array([it["group"] == "night" for it in items])}
    metrics = {"exact_of_called": ("exact", "called"), "exact_of_matched": ("exact", "matched"), "shape_of_matched": ("shape", "matched"), "exact_of_all_arrows": ("exact", "gt")}
    rng = np.random.default_rng(0)
    out = {}
    for g, sel in groups.items():
        ub, inv = np.unique(block[sel], return_inverse=True)
        w = np.zeros((10000, len(ub)), dtype=np.float32)
        np.add.at(w, (np.arange(10000)[:, None], rng.integers(0, len(ub), (10000, len(ub)))), 1)
        sums = {m: {k: w @ np.bincount(inv, counts[m][k][sel], len(ub)).astype(np.float32) for k in keys} for m in counts}
        out[g] = {"images": int(sel.sum()), "F7": {k: int(counts["F7"][k][sel].sum()) for k in keys}, "M": {k: int(counts["M"][k][sel].sum()) for k in keys}}
        for mname, (a, b) in metrics.items():
            pt = {m: counts[m][a][sel].sum() / max(counts[m][b][sel].sum(), 1) for m in counts}
            diff = sums["F7"][a] / np.maximum(sums["F7"][b], 1) - sums["M"][a] / np.maximum(sums["M"][b], 1)
            out[g][mname] = {"F7": float(pt["F7"]), "M": float(pt["M"]), "d": float(pt["F7"] - pt["M"]), "d_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    save("arrows", out)


# ------------------------------------------------------------------ AP
def det_table(items, preds):
    im, sc, tp, n = [], [], [], np.zeros(len(items))
    for i, (it, (c, s, b)) in enumerate(zip(items, preds)):
        ign = torch.tensor([x is None for x in it["color"]], dtype=torch.bool)
        m = perc.match(b, s, it["boxes"], ign, "center")
        v = m != -2
        im.append(np.full(int(v.sum()), i))
        sc.append(s[v].numpy())
        tp.append((m[v] >= 0).numpy().astype(float))
        n[i] = int((~ign).sum())
    im, sc, tp = np.concatenate(im), np.concatenate(sc), np.concatenate(tp)
    order = np.argsort(-sc, kind="stable")
    return im[order], tp[order], n


def ap_w(table, w):
    im, tp, n = table
    ws = w[im]
    ctp, cfp = np.cumsum(ws * tp), np.cumsum(ws * (1 - tp))
    rec, prec = ctp / (w * n).sum(), ctp / np.maximum(ctp + cfp, 1e-9)
    env = np.maximum.accumulate(np.concatenate([prec, [0]])[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * env[:-1]))


def ap():
    draws, out, diffs = 2000, {}, {}
    rng = np.random.default_rng(0)
    for ds in ("lisa", "s2tld", "bdd100k"):
        items, pf, pm = load_ood(ds)
        tf, tm = det_table(items, pf), det_table(items, pm)
        n = len(items)
        block = blocks_of(ds, n)
        nb = block.max() + 1
        full = np.ones(n)
        pt = {"F7": ap_w(tf, full), "M": ap_w(tm, full)}
        d = []
        for _ in range(draws):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[block].astype(float)
            d.append(ap_w(tf, w) - ap_w(tm, w))
        diffs[ds] = np.array(d)
        out[ds] = {**pt, "d": pt["F7"] - pt["M"], "d_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "clusters": int(nb), "draws": draws}
        print(ds, out[ds], flush=True)
    m = np.mean(list(diffs.values()), axis=0)
    out["mean_three"] = {"F7": float(np.mean([out[k]["F7"] for k in diffs])), "M": float(np.mean([out[k]["M"] for k in diffs])),
                         "d": float(np.mean([out[k]["d"] for k in diffs])), "d_ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    save("ap", out)


# ------------------------------------------------------------------ topology mask
def info_path(r):
    p = r["image"]
    for cam in ("ring_front_center", "CAM_FRONT"):
        p = p.replace(f"/image/{cam}/", "/info/")
    return os.path.join(r["root"], os.path.splitext(p)[0] + ".json")


def approach_moves(path):
    """Maneuvers reachable from the ego approach at the nearest junction, from lane topology only; None if no junction ahead."""
    ann = json.load(open(path))["annotation"]
    lanes = [{"pts": np.asarray(l["points"], dtype=np.float64), "connector": bool(l.get("is_intersection_or_connector", False))} for l in ann["lane_centerline"]]
    n = len(lanes)
    lclc = np.asarray(ann["topology_lclc"], dtype=np.int64).reshape(n, n) if n else np.zeros((0, 0), int)
    cands = []
    for i, l in enumerate(lanes):
        if abs(_wrap(_heading(l["pts"], at_end=not l["connector"]))) < ALIGNED_DEG and l["pts"][-1, 0] > AHEAD_X:
            moves, stop = lane_movements(i, lanes, lclc)
            if moves:
                cands.append((stop, moves))
    if not cands:
        return None
    near = min(s for s, _ in cands)
    return set().union(*(m for s, m in cands if s <= near + SAME_JUNCTION_M))


def topo_masks(recs):
    cache = {}
    out = []
    for r in recs:
        k = info_path(r)
        cache.setdefault(k, approach_moves(k))
        out.append(cache[k])
    return out


def us_rule(lights, side):
    """(picto, color, score) junction lights -> colors under the US rule, the convention M's decoder uses."""
    best, circle = {}, max(((s, c) for p, c, s in lights if p in ("circle", "unknown")), default=None)
    for p, c, s in lights:
        for m in ARROW_MOVES.get(p, ()):
            if m not in best or s > best[m][0]:
                best[m] = (s, c)
    far, near = ("left", "right") if side == "right" else ("right", "left")
    straight = best.get("straight") or circle
    f = best.get(far) or straight
    n = best.get(near)
    res = {"straight": straight, far: f, near: n, "uturn": f}
    return {d: v[1] if v else "none" for d, v in res.items()}


def f7_us(d, thr, side):
    d = kept(d, thr)
    return us_rule([(PICTOS[int(p)], COLORS[int(c)], float(s)) for p, c, j, s in zip(d["picto"], d["color"], d["junction"], d["score"]) if j > 0.5], side)


def apply_topo(f, moves):
    if moves is None:
        return f
    f = dict(f)
    if "straight" not in moves:
        f["straight"] = "none"
    if not moves & {"left", "straight"}:
        f["left"] = f["uturn"] = "none"
    if "right" not in moves:
        f["right"] = "none"
    return f


def m_topo_arrays(mpc, recs, masks):
    from dtsu import COLORS as DC, DIRECTIONS, color_to_pass, PASS
    out = []
    for row, moves in zip(mpc, masks):
        out.append(apply_topo({d: DC[int(v)] for d, v in zip(DIRECTIONS, row)}, moves))
    return to_arrays(out)


def topo():
    npz = np.load(os.path.join(e7.M_RUN, "val_preds_mix.npz"))
    tune = {}
    for name in ("A_tune", "B_tune"):
        recs = load_records(*SETS[name])
        tune[name] = (recs, torch.load(f"{D}/e{EPOCH}_{name}.pt", weights_only=False), topo_masks(recs))
    grid = {}
    for t in THRESHOLDS:
        v = []
        for recs, dets, masks in tune.values():
            pc, pp = to_arrays([apply_topo(f7_us(d, float(t), r["traffic_side"]), m) for d, r, m in zip(dets, recs, masks)])
            gc, gp = to_arrays([r["color"] for r in recs])
            v.append(evaluate(pc, pp, gc, gp)["tss"])
        grid[float(t)] = float(np.mean(v))
    best = max(grid, key=grid.get)
    print("tune grid under the US rule + topology mask: best", best, grid[best], "at 0.20:", grid[0.2], flush=True)
    out = {"tune_mean_tss_by_threshold": grid, "tune_best_threshold": best}
    for name, key in (("A_val", "A"), ("B_val", "B")):
        recs = load_records(*SETS[name])
        dets = torch.load(f"{D}/e{EPOCH}_{name}.pt", weights_only=False)
        masks = topo_masks(recs)
        gc, gp = to_arrays([r["color"] for r in recs])
        segs = np.array([r["segment_id"] for r in recs])
        mpc, mpp = (npz[f"{key}_{x}"][: len(recs)] for x in ("pc", "pp"))
        mtc, mtp = m_topo_arrays(mpc, recs, masks)
        m_native, m_topo = e7.frame_stats(mpc, mpp, gc, gp), e7.frame_stats(mtc, mtp, gc, gp)
        res = {"frames": len(recs), "no_junction_ahead": sum(m is None for m in masks), "M_native_metrics": m_native[3], "M_topology_metrics": m_topo[3]}
        for label, thr in (("thr_0.20", 0.20), ("thr_tune_best", best)):
            pu = [f7_us(d, thr, r["traffic_side"]) for d, r in zip(dets, recs)]
            pt = [apply_topo(f, m) for f, m in zip(pu, masks)]
            fu, ft = (e7.frame_stats(*to_arrays(x), gc, gp) for x in (pu, pt))
            res[label] = {"threshold": thr, "F7_us_rule_metrics": fu[3], "F7_us_topology_metrics": ft[3],
                          "F7_us_rule_vs_M_native": e7.paired(fu[:3], m_native[:3], segs),
                          "F7_us_topology_vs_M_native": e7.paired(ft[:3], m_native[:3], segs),
                          "F7_us_topology_vs_M_topology": e7.paired(ft[:3], m_topo[:3], segs)}
        out[name] = res
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if "metrics" not in kk} if isinstance(v, dict) else v for k, v in res.items() if k.startswith("thr")})[:1600], flush=True)
    save("topo", out)


# ------------------------------------------------------------------ route labels
def route():
    from dtsu.data import load_records as lr
    root = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD_jpg", "test")
    labels = lr("data/labels/dtld_test.jsonl", root)
    test = [json.loads(l) for l in open(os.path.join(EXP6, "dtld_test_records.jsonl"))]
    dets = torch.load(f"{D}/e{EPOCH}_dtld_test.pt", weights_only=False)
    md = torch.load(f"{D}/M_dtld_test.pt", weights_only=False)
    idx = [i for i, r in enumerate(test) if r["excluded"] is None]
    assert len(idx) == len(md)
    by_name = {os.path.basename(test[i]["image"]): (i, m) for m, i in enumerate(idx)}
    kept_recs = [r for r in labels if os.path.basename(r["image"]) in by_name]
    f, m = [], []
    for r in kept_recs:
        i, j = by_name[os.path.basename(r["image"])]
        f.append(directions(kept(dets[i], THR)))
        m.append(frame_directions(*md[j], M_CLASSES, M_THR, "right", "signal"))
    gc, gp = to_arrays([r["color"] for r in kept_recs])
    a, b = e7.frame_stats(*to_arrays(f), gc, gp), e7.frame_stats(*to_arrays(m), gc, gp)
    res = e7.paired(a[:3], b[:3], np.array([r["segment_id"] for r in kept_recs]))
    res.update({"route_label_frames": len(labels), "frames_in_both": len(kept_recs), "F7_metrics": a[3], "M_metrics": b[3]})
    save("route", res)


# ------------------------------------------------------------------ False-Go causes
def m_openlane(name, recs):
    c = f"{D}/M_{name}.pt"
    if os.path.exists(c):
        return torch.load(c, weights_only=False)
    net, res, _ = load_moves(e7.M_RUN, os.path.join(e7.M_RUN, f"epoch_{e7.M_EPOCH}_ema.pth"), len(M_CLASSES))
    out = []
    for x in DataLoader(Images(recs, res), batch_size=8, num_workers=6):
        cls, score, boxes = detect(net, x.cuda(non_blocking=True), M_CLASSES)
        out += [(cc[ss > 0.05], ss[ss > 0.05], bb[ss > 0.05]) for cc, ss, bb in zip(cls, score, boxes)]
    del net
    torch.cuda.empty_cache()
    torch.save(out, c)
    return out


def kept_m(x):
    from torchvision.ops import nms
    c, s, b = x
    k = s > M_THR
    c, s, b = c[k], s[k], b[k]
    if len(s):
        keep = nms(box_convert(b, "cxcywh", "xyxy"), s, 0.5)
        c, s, b = c[keep], s[keep], b[keep]
    return c, s, b


def attribute(gt_boxes, gt_red, boxes, scores, rel, red):
    """Cause for a frame's False-Go directions: relevant red lights, against this model's kept detections."""
    if not len(gt_boxes) or not any(gt_red):
        return "no_red_label_box"
    m = perc.match(boxes, scores, torch.tensor(gt_boxes, dtype=torch.float32), torch.zeros(len(gt_boxes), dtype=torch.bool), "center")
    hit = {g: p for p, g in enumerate(m.tolist()) if g >= 0 and gt_red[g]}
    if not hit:
        return "not_found"
    if any(rel[p] and red[p] for p in hit.values()):
        return "decoder_chose_another"
    if any(rel[p] for p in hit.values()):
        return "wrong_color"
    return "called_not_relevant"


def fg():
    out = {}
    for name in ("A_val", "B_val", "DTLD_test"):
        if name == "DTLD_test":
            all_recs = [json.loads(l) for l in open(os.path.join(EXP6, "dtld_test_records.jsonl"))]
            idx = [i for i, r in enumerate(all_recs) if r["excluded"] is None]
            recs = [all_recs[i] for i in idx]
            dets = torch.load(f"{D}/e{EPOCH}_dtld_test.pt", weights_only=False)
            dets = [dets[i] for i in idx]
            md = torch.load(f"{D}/M_dtld_test.pt", weights_only=False)
            size = [(2048, 1024)] * len(recs)
            gt = [([b["box"] for b in r["boxes"]], [b["junction"] and b["color"] == "red" for b in r["boxes"]]) for r in recs]
            f_pred = [directions(kept(d, THR)) for d in dets]
            m_pred = [frame_directions(*x, M_CLASSES, M_THR, "right", "signal") for x in md]
        else:
            recs = load_records(*SETS[name])
            for r in recs:
                r["image"] = os.path.abspath(os.path.join(r["root"], r["image"]))
            dets = torch.load(f"{D}/e{EPOCH}_{name}.pt", weights_only=False)
            md = m_openlane(name, recs)
            with Image.open(recs[0]["image"]) as im:
                size = [im.size] * len(recs)
            gt = [([b["box"] for b in r["boxes"]], [b["relevant"] and b["color"] == "red" for b in r["boxes"]]) for r in recs]
            f_pred = [e7.predict(d, THR, r["traffic_side"]) for d, r in zip(dets, recs)]
            m_pred = [frame_directions(*x, M_CLASSES, M_THR, r["traffic_side"], "dedicated") for x, r in zip(md, recs)]
            npz = np.load(os.path.join(e7.M_RUN, "val_preds_mix.npz"))
            assert (to_arrays(m_pred)[0] == npz[f"{name[0]}_pc"][: len(recs)]).all(), "M detections do not reproduce the stored predictions"
        res = {"F7": collections.Counter(), "M": collections.Counter()}
        n_red = 0
        for i, r in enumerate(recs):
            reds = [d for d, c in r["color"].items() if c == "red"]
            n_red += len(reds)
            w, h = size[i]
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            d = kept(dets[i], THR)
            fb = box_convert(d["box"], "cxcywh", "xyxy") * scale
            cls, s, b = kept_m(md[i])
            mb = box_convert(b, "cxcywh", "xyxy") * scale
            parsed = [parse_class(M_CLASSES[int(c)]) for c in cls]
            for label, pred, boxes, score, rel, red in (
                    ("F7", f_pred[i], fb, d["score"], (d["junction"] > 0.5).tolist(), [COLORS[int(c)] == "red" for c in d["color"]]),
                    ("M", m_pred[i], mb, s, [p[0] is not None for p in parsed], [p[1] == "red" for p in parsed])):
                for direction in reds:
                    if pred[direction] != "red":
                        res[label]["all_false_go"] += 1
                        res[label][attribute(*gt[i], boxes, score, rel, red)] += 1
                        res[label]["predicted_" + pred[direction]] += 1
        out[name] = {"red_directions": n_red, "F7": dict(res["F7"]), "M": dict(res["M"])}
        print(name, out[name], flush=True)
    save("fg", out)


# ------------------------------------------------------------------ latency
def latency():
    test = [json.loads(l) for l in open(os.path.join(EXP6, "dtld_test_records.jsonl"))][:320]
    out = {}
    torch.backends.cuda.matmul.allow_tf32 = True
    for name in ("F7", "M"):
        if name == "F7":
            net, res = load_model("runs/exp7_multi_F", EPOCH, F7_CLASSES, "F")
            fwd = lambda x: factorized.detect(net, x)  # noqa: E731
        else:
            net, res, _ = load_moves(e7.M_RUN, os.path.join(e7.M_RUN, f"epoch_{e7.M_EPOCH}_ema.pth"), len(M_CLASSES))
            fwd = lambda x: detect(net, x, M_CLASSES)  # noqa: E731
        t_net, t_all = [], []
        for i, r in enumerate(test):
            t0 = time.perf_counter()
            x = load_image(r["image"], res)[None].cuda()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd(x)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if i >= 20:
                t_net.append(t2 - t1)
                t_all.append(t2 - t0)
        out[name] = {"frames": len(t_net), "network_ms_median": float(np.median(t_net) * 1000), "network_ms_p95": float(np.percentile(t_net, 95) * 1000),
                     "end_to_end_ms_median": float(np.median(t_all) * 1000), "params_M": sum(p.numel() for p in net.parameters()) / 1e6}
        del net
        torch.cuda.empty_cache()
    save("latency", out)


def main():
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("checks", nargs="+", choices=("arrows", "ap", "topo", "route", "fg", "latency"))
    for c in ap_.parse_args().checks:
        print("==", c, flush=True)
        globals()[c]()


if __name__ == "__main__":
    main()
