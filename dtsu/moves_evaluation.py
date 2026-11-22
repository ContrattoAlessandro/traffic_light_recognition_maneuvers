"""Selection and evaluation of the baseline M (``dtsu.moves``), alone or paired with a factorized run such as R.

Selection (tune splits only; the rule of Exp 5, over three maneuvers): the epoch and score threshold that maximise
the equal-weight mean of OpenLane-V2 tune TSS (A tune and B tune pooled) and D*_US tune TSS, both with M's own decoder
and the US rule. Ties go to the lower epoch, then the lower threshold.

Evaluation at the selection:
- maneuvers, M's decoder: OpenLane-V2 A val and B val (US rule; camera only, M's native output, and with the
  lane-topology mask), DTLD test with the route labels (German rule) and D*_US test (US rule);
- detection: the perception cells of ``dtsu.evaluation`` (center AP, recall, precision, color accuracy).
Paired with a factorized run (``reference``), the publication cells pair frame by frame: A val and B val camera only
(the factorized run with ``rules.us_rule``) and the DTLD route-label cell, on the test frames that both the route labels
and the junction rule score (the factorized run with ``rules.junction_rule``, against the route labels).
"""
import os

import numpy as np
import torch

from . import labels_dtld, sources
from .evaluation import (THRESHOLDS, compare_detection, frame_stats, junction_lights, kept, openlane_set, to_arrays)
from .metrics import evaluate
from .model import run_args, saved_epochs
from .moves import box_detections, class_names, frame_colors, frame_detections, load_model
from .rules import apply_topology, junction_rule, us_rule
from .stats import paired_frames


class MovesRun:
    """An M run and its cached detections (``<cache>/e<epoch>_<set>.pt``), with the interface of ``evaluation.Run``."""
    loaded = {}

    def __init__(self, name, run, cache=None, workers=6, device="cuda"):
        self.name, self.run, self.workers, self.device = name, run, workers, device
        self.cache = cache or os.path.join(run, "eval")
        self.res = run_args(run)["resolution"]
        self.epochs = saved_epochs(run)
        self.names = class_names(run)
        os.makedirs(self.cache, exist_ok=True)

    def net(self, epoch):
        key = (self.run, epoch)
        if key not in MovesRun.loaded:
            MovesRun.loaded.clear()
            MovesRun.loaded[key] = load_model(self.run, epoch, self.device)
        return MovesRun.loaded[key]

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
        return self._cached(f"e{epoch}_{tag}", len(paths), lambda: frame_detections(
            self.net(epoch), paths, self.res, len(self.names), self.workers, device=self.device))

    def boxes(self, epoch, ds, items):
        paths = [it["path"] for it in items]
        return self._cached(f"e{epoch}_{ds}", len(paths), lambda: box_detections(
            self.net(epoch), paths, self.res, self.names, self.workers, device=self.device))


# ------------------------------------------------------------------ sets
def dtld_sets():
    """D*_US tune and test (kept frames), DTLD route test (kept frames, German rules), and DTLD test as route records
    in DTLD's own order (all frames: the order of the junction-rule test records of ``dtsu.evaluation``)."""
    read = lambda kind, part: sources.read_jsonl(labels_dtld.label_path(kind, part))
    route_test = read("route", "test")
    return {"DTLD_US_tune": [r for r in read("us", "tune") if r["kept"]],
            "DTLD_US_test": [r for r in read("us", "test") if r["kept"]],
            "DTLD_route_test": [r for r in route_test if r["kept"]]}, route_test


def evaluation_sets():
    ol = {n: openlane_set(n) for n in ("A_tune", "B_tune", "A_val", "B_val")}
    dt, route_all = dtld_sets()
    return ol, dt, route_all


def colors_of(run, epoch, tag, recs, thr):
    names = run.names
    return [frame_colors(d, names, thr, r["traffic_side"], r.get("near_turn", "dedicated"))
            for d, r in zip(run.frames(epoch, tag, recs), recs)]


def tss(colors, recs):
    return evaluate(*to_arrays(colors), *to_arrays([r["color"] for r in recs]))["tss"]


# ------------------------------------------------------------------ selection
def select(run, ol, dt):
    grid = {}
    ol_recs = ol["A_tune"][0] + ol["B_tune"][0]
    for e in run.epochs:
        dets = {"A_tune": run.frames(e, "A_tune", ol["A_tune"][0]), "B_tune": run.frames(e, "B_tune", ol["B_tune"][0]),
                "DTLD_US_tune": run.frames(e, "DTLD_US_tune", dt["DTLD_US_tune"])}
        for t in THRESHOLDS:
            c_ol = [frame_colors(d, run.names, t, r["traffic_side"]) for n in ("A_tune", "B_tune") for d, r in zip(dets[n], ol[n][0])]
            c_us = [frame_colors(d, run.names, t, r["traffic_side"], r["near_turn"]) for d, r in zip(dets["DTLD_US_tune"], dt["DTLD_US_tune"])]
            v = {"OpenLane_tune": tss(c_ol, ol_recs), "DTLD_US_tune": tss(c_us, dt["DTLD_US_tune"])}
            v["mean"] = float(np.mean(list(v.values())))
            grid[(e, float(t))] = v
        print(f"{run.name} epoch {e}: best tune mean {max(v['mean'] for k, v in grid.items() if k[0] == e):.4f}", flush=True)
    best = max(grid, key=lambda k: (grid[k]["mean"], -k[0], -k[1]))
    print(f"{run.name} selected epoch {best[0]}, threshold {best[1]}: {grid[best]}", flush=True)
    return {"epoch": best[0], "threshold": best[1], "tune": grid[best], "grid": {f"e{k[0]}_t{k[1]}": v["mean"] for k, v in grid.items()}}


# ------------------------------------------------------------------ maneuvers
def maneuver_stats(run, sel, ol, dt):
    """Per set: (frame_stats, cluster ids) of M at its selection."""
    e, t = sel["epoch"], sel["threshold"]
    out = {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = ol[name]
        colors = colors_of(run, e, name, recs, t)
        seg = np.array([r["segment_id"] for r in recs])
        out[name + "_camera_only"] = (frame_stats(*to_arrays(colors), gc, gp), seg)
        out[name] = (frame_stats(*to_arrays([apply_topology(c, m) for c, m in zip(colors, masks)]), gc, gp), seg)
    for name in ("DTLD_route_test", "DTLD_US_test"):
        recs = dt[name]
        colors = colors_of(run, e, name, recs, t)
        out[name] = (frame_stats(*to_arrays(colors), *to_arrays([r["color"] for r in recs])), np.array([r["segment_id"] for r in recs]))
    return out


def paired_cells(m_run, m_sel, f_run, f_sel, ol, dt, route_all, dtld_junction_test):
    """The publication cells, M ("b") against a factorized run ("a"), frame by frame. Returns (a stats, b stats)."""
    a, b = {}, {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = ol[name]
        fd = f_run.frames(f_sel["epoch"], name, recs)
        a[name + "_camera_only"] = frame_stats(*to_arrays([us_rule(junction_lights(kept(d, f_sel["threshold"])), r["traffic_side"])
                                                          for d, r in zip(fd, recs)]), gc, gp)
        b[name + "_camera_only"] = frame_stats(*to_arrays(colors_of(m_run, m_sel["epoch"], name, recs, m_sel["threshold"])), gc, gp)
    # DTLD test: route-label ground truth on the frames both labelings score; the factorized run's detections come from
    # its junction-record cache (same frames, same order as route_all)
    assert len(route_all) == len(dtld_junction_test)
    idx = [i for i, (r, j) in enumerate(zip(route_all, dtld_junction_test)) if r["kept"] and j["excluded"] is None]
    assert all(os.path.basename(route_all[i]["image"]) == os.path.basename(dtld_junction_test[i]["image"]) for i in idx)
    fd = f_run.frames(f_sel["epoch"], "dtld_test", dtld_junction_test)
    recs = [route_all[i] for i in idx]
    gc, gp = to_arrays([r["color"] for r in recs])
    a["DTLD_route_test_common"] = frame_stats(*to_arrays([junction_rule(junction_lights(kept(fd[i], f_sel["threshold"]))) for i in idx]), gc, gp)
    md = m_run.frames(m_sel["epoch"], "DTLD_route_test_all", route_all)
    b["DTLD_route_test_common"] = frame_stats(*to_arrays([frame_colors(md[i], m_run.names, m_sel["threshold"], "right", "signal") for i in idx]), gc, gp)
    seg = {n: np.array([r["segment_id"] for r in ol[n][0]]) for n in ("A_val", "B_val")}
    seg["DTLD"] = np.array([r["segment_id"] for r in recs])
    clusters = {"A_val_camera_only": seg["A_val"], "B_val_camera_only": seg["B_val"], "DTLD_route_test_common": seg["DTLD"]}
    return a, b, clusters


def report(run, sel, ol, dt, detection=True):
    st = maneuver_stats(run, sel, ol, dt)
    out = {"maneuver_metrics": {k: v[0][3] for k, v in st.items()},
           "frames": {k: int(len(v[1])) for k, v in st.items()}}
    if detection:
        out["detection"] = compare_detection(run, sel, run, sel)
    return out


def compare(a_stats, b_stats, clusters):
    return {k: paired_frames(a_stats[k][:3], b_stats[k][:3], clusters[k]) for k in a_stats}
