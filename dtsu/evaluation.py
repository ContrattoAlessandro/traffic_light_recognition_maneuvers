"""Model selection and evaluation of a trained run, alone or against a reference run.

Selection (tune splits only, never a test set): the epoch and score threshold that maximise the mean of
- DTLD tune TSS, junction rule (``dtsu.rules.junction_rule``);
- OpenLane-V2 A tune TSS and B tune TSS, US rule plus the lane-topology mask (``dtsu.rules.us_rule``, ``dtsu.topology``).
Ties go to the lower epoch, then the lower threshold.

Evaluation at the selected epoch and threshold:
- maneuvers: TSS and False-Go on OpenLane-V2 A val and B val (US rule, with the topology mask and camera only) and
  on DTLD test (junction rule, on the frames whose ground truth the rule settles);
- attributes, on the labelled lights a model matches (center criterion): pictogram recall (macro and per class, also
  by lamp height), color accuracy and yellow recall, junction accuracy, and false arrows on round lights;
- detection: center AP, recall, precision and color accuracy on ATLAS test, LISA test (day / night), S2TLD (720p,
  1080p, horizontal lights), BDD100K val (day / night / horizontal) and VZC-TLD.
Compared with a reference, attribute metrics use the lights both models match, and every difference carries a
cluster-bootstrap 95% CI (``dtsu.stats``).

Detections are cached per run in ``<cache>/e<epoch>_<set>.pt``, so a run is inferred once per set.
"""
import os

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert, nms

from . import COLORS as DIRECTION_COLORS, DIRECTIONS, PASS, color_to_pass, sources
from .metrics import evaluate
from .model import NMS_IOU, box_detections, frame_detections, load_model, run_args, saved_epochs
from .rules import LAMP_COLORS, PICTOGRAMS, apply_topology, junction_rule, us_rule
from .stats import det_table, ignore_mask, match, paired_ap, paired_frames, paired_rates
from .topology import topology_masks

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
STOP, GO = PASS.index("stop"), PASS.index("go")
ARROWS = {p for p in PICTOGRAMS if p.startswith("arrow")}
PICTO_MACRO = ("circle", "arrow_left", "arrow_right", "arrow_straight", "other")  # DTLD test; one straight-left lamp
PICTO_MACRO_OOD = {"ATLAS_test": ("circle", "arrow_left", "arrow_right", "arrow_straight"),
                   "LISA_test": ("circle", "arrow_left", "arrow_straight")}
HEIGHT_BINS = ((0, 15), (15, 20), (20, 30), (30, 1e9))
HORIZONTAL = 1.2  # a light is horizontal when its width exceeds 1.2 x its height
BLOCK = 100
# perception cells: (name, dataset, group or None, horizontal lights only)
CELLS = (("lisa_night", "lisa", "night", False), ("lisa_day", "lisa", "day", False),
         ("bdd_night", "bdd100k", "night", False), ("bdd_day", "bdd100k", "daytime", False), ("bdd_all", "bdd100k", None, False),
         ("s2tld_1080p_horizontal", "s2tld", "1920x1080", True), ("s2tld_1080p", "s2tld", "1920x1080", False),
         ("s2tld_720p", "s2tld", "1280x720", False), ("bdd_horizontal", "bdd100k", None, True),
         ("vzc_horizontal", "vzc", None, True), ("vzc_all", "vzc", None, False), ("atlas_test", "atlas", None, False))


# ------------------------------------------------------------------ a run and its cached detections
class Run:
    loaded = {}

    def __init__(self, name, run, cache=None, workers=6):
        self.name, self.run, self.workers = name, run, workers
        self.cache = cache or os.path.join(run, "eval")
        self.res = run_args(run)["resolution"]
        self.epochs = saved_epochs(run)
        os.makedirs(self.cache, exist_ok=True)

    def net(self, epoch):
        key = (self.run, epoch)
        if key not in Run.loaded:
            Run.loaded.clear()
            torch.cuda.empty_cache()
            Run.loaded[key] = load_model(self.run, epoch)
        return Run.loaded[key]

    def _cached(self, name, n, fn):
        path = os.path.join(self.cache, name + ".pt")
        if os.path.exists(path):
            x = torch.load(path, weights_only=False)
            if len(x) == n:
                return x
        x = fn()
        torch.save(x, path)
        return x

    def frames(self, epoch, tag, records):
        paths = [r["image"] for r in records]
        return self._cached(f"e{epoch}_{tag}", len(paths), lambda: frame_detections(self.net(epoch), paths, self.res, self.workers))

    def boxes(self, epoch, ds, items):
        paths = [it["path"] for it in items]
        return self._cached(f"e{epoch}_{ds}", len(paths), lambda: box_detections(self.net(epoch), paths, self.res, self.workers))


# ------------------------------------------------------------------ decoding frames into maneuvers
def kept(d, thr):
    """Detections above the threshold after class-agnostic NMS."""
    k = d["score"] > thr
    d = {a: v[k] for a, v in d.items()}
    if len(d["score"]):
        keep = nms(box_convert(d["box"], "cxcywh", "xyxy"), d["score"], NMS_IOU)
        d = {a: v[keep] for a, v in d.items()}
    return d


def junction_lights(d):
    return [(PICTOGRAMS[int(p)], LAMP_COLORS[int(c)], float(s)) for p, c, j, s in zip(d["picto"], d["color"], d["junction"], d["score"]) if j > 0.5]


def to_arrays(colors):
    """Per-frame {maneuver: color} -> ([N, D] color index, [N, D] stop / go index)."""
    c = np.array([[DIRECTION_COLORS.index(f[d]) for d in DIRECTIONS] for f in colors]).reshape(-1, len(DIRECTIONS))
    p = np.array([[PASS.index(color_to_pass(f[d])) for d in DIRECTIONS] for f in colors]).reshape(-1, len(DIRECTIONS))
    return c, p


def frame_stats(pc, pp, gc, gp):
    """Per-frame TSS contribution, False-Go numerator and denominator, and the pooled metrics."""
    c_ok, p_ok = pc == gc, pp == gp
    s = (c_ok.mean(1) + p_ok.mean(1) + (c_ok & p_ok).all(1)) / 3
    return s, ((gp == STOP) & (pp == GO)).sum(1), (gp == STOP).sum(1), evaluate(pc, pp, gc, gp)


def openlane_set(name):
    """-> (records, topology masks, ground-truth arrays) of an OpenLane-V2 set (``sources.openlane_sets``)."""
    recs = sources.load_records(*sources.openlane_sets()[name])
    masks = topology_masks(recs)
    recs = [dict(r, image=os.path.abspath(os.path.join(r["root"], r["image"]))) for r in recs]
    return recs, masks, to_arrays([r["color"] for r in recs])


def openlane_arrays(recs, dets, masks, thr, topology=True):
    """US rule per frame (mirrored in left-hand traffic), with or without the lane-topology mask."""
    out = []
    for d, r, m in zip(dets, recs, masks):
        colors = us_rule(junction_lights(kept(d, float(thr))), r["traffic_side"])
        out.append(apply_topology(colors, m) if topology else colors)
    return to_arrays(out)


def dtld_stats(records, dets, thr):
    """Junction rule on the DTLD frames the ground truth settles -> (frame_stats, indices of those frames)."""
    idx = [i for i, r in enumerate(records) if r["excluded"] is None]
    pc, pp = to_arrays([junction_rule(junction_lights(kept(dets[i], thr))) for i in idx])
    gc, gp = to_arrays([records[i]["color"] for i in idx])
    return frame_stats(pc, pp, gc, gp), idx


def dtld_set(split):
    """DTLD junction-rule records of the tune or test split (written by scripts/export_training_set.py)."""
    return sources.read_jsonl(sources.dtld_junction_path(split))


# ------------------------------------------------------------------ selection
def select(run, sets, dtld_tune):
    grid = {}
    for e in run.epochs:
        d_dt = run.frames(e, "dtld_tune", dtld_tune)
        ol = {n: run.frames(e, n, sets[n][0]) for n in ("A_tune", "B_tune")}
        for t in THRESHOLDS:
            v = {"DTLD_tune": dtld_stats(dtld_tune, d_dt, t)[0][3]["tss"]}
            for n, d in ol.items():
                recs, masks, (gc, gp) = sets[n]
                v[n] = evaluate(*openlane_arrays(recs, d, masks, t), gc, gp)["tss"]
            v["mean"] = float(np.mean(list(v.values())))
            grid[(e, float(t))] = v
        print(f"{run.name} epoch {e}: best tune mean {max(v['mean'] for k, v in grid.items() if k[0] == e):.4f}", flush=True)
    best = max(grid, key=lambda k: (grid[k]["mean"], -k[0], -k[1]))
    print(f"{run.name} selected epoch {best[0]}, threshold {best[1]}: {grid[best]}", flush=True)
    return {"epoch": best[0], "threshold": best[1], "tune": grid[best], "grid": {f"e{k[0]}_t{k[1]}": v["mean"] for k, v in grid.items()}}


# ------------------------------------------------------------------ maneuvers
def maneuver_stats(run, sel, sets, dtld_test):
    """Per set: frame_stats of the run at its selection, and the cluster ids of the frames."""
    e, t = sel["epoch"], sel["threshold"]
    out = {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = sets[name]
        dets = run.frames(e, name, recs)
        seg = np.array([r["segment_id"] for r in recs])
        out[name] = (frame_stats(*openlane_arrays(recs, dets, masks, t), gc, gp), seg)
        out[name + "_camera_only"] = (frame_stats(*openlane_arrays(recs, dets, masks, t, topology=False), gc, gp), seg)
    st, idx = dtld_stats(dtld_test, run.frames(e, "dtld_test", dtld_test), t)
    out["DTLD_test"] = (st, np.array([dtld_test[i]["segment_id"] for i in idx]))
    return out


def compare_maneuvers(a, b):
    return {k: paired_frames(a[k][0][:3], b[k][0][:3], a[k][1]) for k in a}


# ------------------------------------------------------------------ attributes of matched lights
SHAPE = {None: "circle", ("left",): "arrow_left", ("right",): "arrow_right", ("straight",): "arrow_straight",
         ("left", "straight"): "arrow_straight_left", ("right", "straight"): "arrow_straight_right"}


def _image_size(path):
    with Image.open(path) as im:
        return im.size


def attribute_sets(sets, dtld_test):
    """name -> (items, kind, key, size, records). kind "frame": detections in the frame cache (key = its tag),
    "box": detections in the box cache (key = dataset)."""
    out = {"DTLD_test": ([{"boxes": torch.tensor([b["box"] for b in r["boxes"]], dtype=torch.float32).reshape(-1, 4),
                           "color": [b["color"] for b in r["boxes"]],
                           "picto": [None if b["picto"] == "unknown" else b["picto"] for b in r["boxes"]],
                           "junction": [bool(b["junction"]) for b in r["boxes"]], "cluster": r["segment_id"]} for r in dtld_test],
                         "frame", "dtld_test", sources.DTLD_SIZE, dtld_test)}
    hand = sources.openlane_val_pictograms()
    for name in ("A_val", "B_val"):
        recs = sets[name][0]
        items = []
        for r in recs:
            picto = [hand.get((name[0], r["segment_id"], str(r["timestamp"]), bi)) for bi in range(len(r["boxes"]))]
            items.append({"boxes": torch.tensor([b["box"] for b in r["boxes"]], dtype=torch.float32).reshape(-1, 4),
                          "color": [b["color"] if b["color"] in LAMP_COLORS else None for b in r["boxes"]],
                          "picto": [p if p in PICTOGRAMS else None for p in picto],
                          "junction": [bool(b["relevant"]) for b in r["boxes"]], "cluster": r["segment_id"]})
        out[name] = (items, "frame", name, _image_size(recs[0]["image"]), recs)
    for name, ds in (("ATLAS_test", "atlas"), ("LISA_test", "lisa"), ("BDD_val", "bdd100k"), ("S2TLD", "s2tld"), ("VZC", "vzc")):
        items = []
        for i, it in enumerate(sources.PERCEPTION[ds]()):
            n = len(it["color"])
            items.append({**it, "cluster": it["session"] if ds == "vzc" else i // BLOCK if ds in ("lisa", "s2tld") else i,
                          "picto": [SHAPE.get(None if a is None else tuple(sorted(a)), "?") for a in it["arrow"]] if ds in ("atlas", "lisa") else [None] * n,
                          "junction": it["relevant"] if ds == "vzc" else [None] * n})
        out[name] = (items, "box", ds, None, None)
    return out


def _frame_lights(dets, size):
    w, h = size
    out = []
    for d in dets:
        b = box_convert(d["box"], "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
        keep = nms(b, d["score"], NMS_IOU) if len(d["score"]) else torch.zeros(0, dtype=torch.long)
        out.append((d["score"][keep], d["picto"][keep], d["color"][keep], (d["junction"][keep] > 0.5).long(), b[keep]))
    return out


def lights(run, sel, aset):
    """Per image (score, pictogram, color, junction, xyxy box) of every detection after NMS, below threshold too."""
    items, kind, key, size, records = aset
    if kind == "frame":
        return _frame_lights(run.frames(sel["epoch"], key, records), size)
    return [(s, c // 6, c % 3, (c // 3) % 2, b) for c, s, b in run.boxes(sel["epoch"], key, items)]


def light_rows(items, la, ta, lb, tb):
    """The labelled lights both models match above their thresholds -> arrays of labels and of both predictions."""
    keys = ("cluster", "group", "gt_p", "gt_c", "gt_j", "a_p", "a_c", "a_j", "b_p", "b_c", "b_j", "height")
    rows = {k: [] for k in keys}
    for it, A, B in zip(items, la, lb):
        ign = ignore_mask(it)
        hit = []
        for (s, p, c, j, b), t in ((A, ta), (B, tb)):
            m = match(b, s, it["boxes"], ign, "center")
            hit.append({int(m[q]): q for q in torch.nonzero((s > t) & (m >= 0)).flatten().tolist()})
        for g in sorted(hit[0].keys() & hit[1].keys()):
            gp = it["picto"][g]
            rows["cluster"].append(str(it["cluster"]))
            rows["group"].append(it.get("group", ""))
            rows["gt_p"].append(gp if gp in PICTOGRAMS else "")
            rows["gt_c"].append(it["color"][g])
            rows["gt_j"].append(-1 if it["junction"][g] is None else int(it["junction"][g]))
            for side, (s, p, c, j, b), q in (("a", A, hit[0][g]), ("b", B, hit[1][g])):
                rows[f"{side}_p"].append(PICTOGRAMS[int(p[q])])
                rows[f"{side}_c"].append(LAMP_COLORS[int(c[q])])
                rows[f"{side}_j"].append(int(j[q]))
            rows["height"].append(float(it["boxes"][g, 3] - it["boxes"][g, 1]))
    return {k: np.array(v) for k, v in rows.items()}


def rate(rows, kind, classes=None):
    """One attribute metric on the rows, both models, with the bootstrap CI of the difference."""
    gp, gc, gj = rows["gt_p"], rows["gt_c"], rows["gt_j"]
    if not len(gp):
        return None
    cl = rows["cluster"]
    if kind == "picto_macro":
        return paired_rates(cl, np.array([classes.index(x) if x in classes else -1 for x in gp]), rows["a_p"] == gp, rows["b_p"] == gp)
    if kind.startswith("recall_"):
        return paired_rates(cl, np.where(gp == kind[7:], 0, -1), rows["a_p"] == gp, rows["b_p"] == gp)
    if kind == "yellow_recall":
        return paired_rates(cl, np.where(gc == "yellow", 0, -1), rows["a_c"] == gc, rows["b_c"] == gc)
    if kind == "color_acc":
        return paired_rates(cl, np.zeros(len(gc), int), rows["a_c"] == gc, rows["b_c"] == gc)
    if kind == "junction_acc":
        return paired_rates(cl, np.where(gj >= 0, 0, -1), rows["a_j"] == gj, rows["b_j"] == gj)
    if kind == "false_arrow_on_circles":
        return paired_rates(cl, np.where(gp == "circle", 0, -1), np.isin(rows["a_p"], list(ARROWS)), np.isin(rows["b_p"], list(ARROWS)))
    raise ValueError(kind)


def _subset(rows, keep):
    return {k: v[keep] for k, v in rows.items()}


def attribute_cell(name, rows):
    cell = {"lights_matched": int(len(rows["gt_c"])), "color_acc": rate(rows, "color_acc"), "yellow_recall": rate(rows, "yellow_recall")}
    if name == "DTLD_test":
        cell["picto_macro"] = rate(rows, "picto_macro", PICTO_MACRO)
        cell["picto_macro_by_height"] = {f"{lo:g}-{hi:g}px": rate(_subset(rows, (rows["height"] >= lo) & (rows["height"] < hi)), "picto_macro", PICTO_MACRO)
                                         for lo, hi in HEIGHT_BINS}
        cell["picto_macro_under_20px"] = rate(_subset(rows, rows["height"] < 20), "picto_macro", PICTO_MACRO)
    if name in PICTO_MACRO_OOD:
        cell["picto_macro"] = rate(rows, "picto_macro", PICTO_MACRO_OOD[name])
    if name in ("DTLD_test", "ATLAS_test", "LISA_test"):
        cell["picto_recall"] = {c: rate(rows, "recall_" + c) for c in PICTO_MACRO + ("arrow_straight_left", "arrow_straight_right") if (rows["gt_p"] == c).any()}
        cell["false_arrow_on_circles"] = rate(rows, "false_arrow_on_circles")
    if name in ("A_val", "B_val"):
        cell["hand_labelled_lights"] = int((rows["gt_p"] != "").sum())
        cell["false_arrow_on_circles"] = rate(rows, "false_arrow_on_circles")
        cell["picto_recall"] = {c: rate(rows, "recall_" + c) for c in ("circle", "arrow_left", "arrow_straight") if (rows["gt_p"] == c).any()}
    if name in ("DTLD_test", "A_val", "B_val", "VZC"):
        cell["junction_acc"] = rate(rows, "junction_acc")
    if name in ("LISA_test", "BDD_val", "S2TLD"):
        for grp in sorted(set(rows["group"].tolist())):
            sub = _subset(rows, rows["group"] == grp)
            cell[f"yellow_recall_{grp}"] = rate(sub, "yellow_recall")
            cell[f"color_acc_{grp}"] = rate(sub, "color_acc")
            if name == "LISA_test":
                cell[f"recall_arrow_left_{grp}"] = rate(sub, "recall_arrow_left")
    return cell


def compare_attributes(ra, sa, rb, sb, asets):
    out, hand = {}, []
    for name, aset in asets.items():
        rows = light_rows(aset[0], lights(ra, sa, aset), sa["threshold"], lights(rb, sb, aset), sb["threshold"])
        out[name] = attribute_cell(name, rows)
        if name in ("A_val", "B_val"):
            hand.append(rows)
    rows = {k: np.concatenate([r[k] for r in hand]) for k in hand[0]}
    out["OpenLane_hand_labelled"] = {"false_arrow_on_circles": rate(rows, "false_arrow_on_circles"), "lights": int((rows["gt_p"] != "").sum())}
    return out


# ------------------------------------------------------------------ detection
def horizontal_only(items):
    out = []
    for it in items:
        b = it["boxes"]
        wide = (b[:, 2] - b[:, 0]) > HORIZONTAL * (b[:, 3] - b[:, 1])
        out.append(dict(it, color=[c if w else None for c, w in zip(it["color"], wide.tolist())]))
    return out


def clusters_of(ds, items):
    if ds in ("lisa", "s2tld"):
        return np.arange(len(items)) // BLOCK
    if ds == "vzc":
        return np.unique([it["session"] for it in items], return_inverse=True)[1]
    return np.arange(len(items))


def operating_point(items, preds, thr):
    """Recall, precision (a lower bound where lights are unlabelled) and color accuracy at a threshold, center matching."""
    n_gt = n_det = n_fp = hits = 0
    conf = np.zeros((3, 3), int)
    for it, (c, s, b) in zip(items, preds):
        ign = ignore_mask(it)
        m = match(b, s, it["boxes"], ign, "center")
        n_gt += int((~ign).sum())
        above = s > thr
        n_det += int((above & (m != -2)).sum())
        n_fp += int((above & (m == -1)).sum())
        gt_to_pred = {int(g): p for p, g in enumerate(m.tolist()) if g >= 0 and above[p]}
        for g in range(len(it["color"])):
            if not ign[g] and g in gt_to_pred:
                hits += 1
                conf[LAMP_COLORS.index(it["color"][g]), int(c[gt_to_pred[g]]) % 3] += 1
    return {"recall": hits / max(n_gt, 1), "precision": (n_det - n_fp) / max(n_det, 1), "color_acc": float(np.trace(conf) / max(conf.sum(), 1)),
            "color_recall": {g: float(conf[a, a] / conf[a].sum()) if conf[a].sum() else None for a, g in enumerate(LAMP_COLORS)}}


def compare_detection(ra, sa, rb, sb):
    out = {}
    raw = {ds: sources.PERCEPTION[ds]() for ds in sorted({c[1] for c in CELLS})}
    for name, ds, group, wide in CELLS:
        items = raw[ds]
        pa, pb = ra.boxes(sa["epoch"], ds, items), rb.boxes(sb["epoch"], ds, items)
        idx = [i for i, it in enumerate(items) if group is None or it["group"] == group]
        sub = [items[i] for i in idx]
        sub = horizontal_only(sub) if wide else sub
        n_lights = sum(c is not None for it in sub for c in it["color"])
        if not n_lights:
            out[name] = None
            continue
        pa, pb = [pa[i] for i in idx], [pb[i] for i in idx]
        out[name] = {"images": len(sub), "lights": n_lights,
                     **paired_ap(det_table(sub, pa), det_table(sub, pb), clusters_of(ds, sub)),
                     "a": operating_point(sub, pa, sa["threshold"]), "b": operating_point(sub, pb, sb["threshold"])}
    return out


# ------------------------------------------------------------------ everything
def evaluation_sets():
    sets = {n: openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    return sets, dtld_set("tune"), dtld_set("test")


def report(ra, sa, rb=None, sb=None, sets=None, dtld_tune=None, dtld_test=None, detection=True):
    """Metrics of run ``ra`` at selection ``sa``; with a reference (rb, sb), every metric is paired with it
    ("a" = ra, "b" = rb). Without a reference the run is paired with itself, so "a" holds its metrics."""
    rb, sb = (ra, sa) if rb is None else (rb, sb)
    if sets is None:
        sets, dtld_tune, dtld_test = evaluation_sets()
    ma, mb = maneuver_stats(ra, sa, sets, dtld_test), maneuver_stats(rb, sb, sets, dtld_test)
    out = {"maneuvers": compare_maneuvers(ma, mb),
           "maneuver_metrics": {"a": {k: v[0][3] for k, v in ma.items()}, "b": {k: v[0][3] for k, v in mb.items()}}}
    out["attributes"] = compare_attributes(ra, sa, rb, sb, attribute_sets(sets, dtld_test))
    if detection:
        out["detection"] = compare_detection(ra, sa, rb, sb)
    return out
