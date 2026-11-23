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
def maneuver_stats(run, sel, ol, dt, route_all):
    """Per set: (frame_stats, cluster ids) of M at its selection."""
    e, t = sel["epoch"], sel["threshold"]
    out = {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = ol[name]
        colors = colors_of(run, e, name, recs, t)
        seg = np.array([r["segment_id"] for r in recs])
        out[name + "_camera_only"] = (frame_stats(*to_arrays(colors), gc, gp), seg)
        out[name] = (frame_stats(*to_arrays([apply_topology(c, m) for c, m in zip(colors, masks)]), gc, gp), seg)
    recs = dt["DTLD_US_test"]
    colors = colors_of(run, e, "DTLD_US_test", recs, t)
    out["DTLD_US_test"] = (frame_stats(*to_arrays(colors), *to_arrays([r["color"] for r in recs])), np.array([r["segment_id"] for r in recs]))
    # route test: the kept frames, decoded from the cache of all route test frames (shared with the paired cells)
    md = run.frames(e, "DTLD_route_test_all", route_all)
    idx = [i for i, r in enumerate(route_all) if r["kept"]]
    recs = [route_all[i] for i in idx]
    colors = [frame_colors(md[i], run.names, t, "right", "signal") for i in idx]
    out["DTLD_route_test"] = (frame_stats(*to_arrays(colors), *to_arrays([r["color"] for r in recs])), np.array([r["segment_id"] for r in recs]))
    return out


def common_dtld_frames(route_all, dtld_junction_test):
    """Indices of the DTLD test frames that both the route labels (kept) and the junction rule (not excluded) score.
    ``route_all`` and ``dtld_junction_test`` hold every DTLD test frame in the same order."""
    assert len(route_all) == len(dtld_junction_test)
    idx = [i for i, (r, j) in enumerate(zip(route_all, dtld_junction_test)) if r["kept"] and j["excluded"] is None]
    assert all(os.path.basename(route_all[i]["image"]) == os.path.basename(dtld_junction_test[i]["image"]) for i in idx)
    return idx


def factorized_cells(f_run, f_sel, ol, route_all, dtld_junction_test):
    """Publication cells of a factorized run (R) at its selection -> {cell: frame_stats}.
    A val and B val with the US rule, camera only (primary) and with the lane-topology mask (secondary); DTLD test with
    the junction rule against the route labels, on the common frames. The DTLD detections come from the run's
    junction-record cache (same frames, same order as ``route_all``)."""
    e, t = f_sel["epoch"], f_sel["threshold"]
    out = {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = ol[name]
        colors = [us_rule(junction_lights(kept(d, t)), r["traffic_side"]) for d, r in zip(f_run.frames(e, name, recs), recs)]
        out[name + "_camera_only"] = frame_stats(*to_arrays(colors), gc, gp)
        out[name] = frame_stats(*to_arrays([apply_topology(c, m) for c, m in zip(colors, masks)]), gc, gp)
    idx = common_dtld_frames(route_all, dtld_junction_test)
    fd = f_run.frames(e, "dtld_test", dtld_junction_test)
    gc, gp = to_arrays([route_all[i]["color"] for i in idx])
    out["DTLD_route_test_common"] = frame_stats(*to_arrays([junction_rule(junction_lights(kept(fd[i], t))) for i in idx]), gc, gp)
    return out


def moves_cells(m_run, m_sel, ol, route_all, dtld_junction_test):
    """The same cells for an M run, with its own decoder (US rule on OpenLane-V2, German rule on DTLD)."""
    e, t = m_sel["epoch"], m_sel["threshold"]
    out = {}
    for name in ("A_val", "B_val"):
        recs, masks, (gc, gp) = ol[name]
        colors = colors_of(m_run, e, name, recs, t)
        out[name + "_camera_only"] = frame_stats(*to_arrays(colors), gc, gp)
        out[name] = frame_stats(*to_arrays([apply_topology(c, m) for c, m in zip(colors, masks)]), gc, gp)
    idx = common_dtld_frames(route_all, dtld_junction_test)
    md = m_run.frames(e, "DTLD_route_test_all", route_all)
    gc, gp = to_arrays([route_all[i]["color"] for i in idx])
    out["DTLD_route_test_common"] = frame_stats(*to_arrays([frame_colors(md[i], m_run.names, t, "right", "signal") for i in idx]), gc, gp)
    return out


def cell_clusters(ol, route_all, dtld_junction_test):
    """Cluster ids of the publication cells: OpenLane-V2 segments, DTLD sessions."""
    seg = {n: np.array([r["segment_id"] for r in ol[n][0]]) for n in ("A_val", "B_val")}
    dtld = np.array([route_all[i]["segment_id"] for i in common_dtld_frames(route_all, dtld_junction_test)])
    return {"A_val_camera_only": seg["A_val"], "B_val_camera_only": seg["B_val"], "A_val": seg["A_val"],
            "B_val": seg["B_val"], "DTLD_route_test_common": dtld}


def paired_cells(m_run, m_sel, f_run, f_sel, ol, dt, route_all, dtld_junction_test):
    """The publication cells, M ("b") against a factorized run ("a"), frame by frame. Returns (a stats, b stats, clusters)."""
    a = factorized_cells(f_run, f_sel, ol, route_all, dtld_junction_test)
    b = moves_cells(m_run, m_sel, ol, route_all, dtld_junction_test)
    return a, b, cell_clusters(ol, route_all, dtld_junction_test)


def report(run, sel, ol, dt, route_all, detection=True):
    st = maneuver_stats(run, sel, ol, dt, route_all)
    out = {"maneuver_metrics": {k: v[0][3] for k, v in st.items()},
           "frames": {k: int(len(v[1])) for k, v in st.items()}}
    if detection:
        out["detection"] = compare_detection(run, sel, run, sel)
    return out


def compare(a_stats, b_stats, clusters):
    return {k: paired_frames(a_stats[k][:3], b_stats[k][:3], clusters[k]) for k in a_stats}


# ------------------------------------------------------------------ several runs of each model (publication protocol)
PRIMARY = ("A_val_camera_only", "B_val_camera_only", "DTLD_route_test_common")
SECONDARY = ("A_val", "B_val")  # with the lane-topology mask ("with map prior")
METRICS = ("tss", "false_go", "acc_color", "acc_pass", "em", "false_stop")


def average_runs(stats):
    """``frame_stats`` of several runs of one model on the same frames -> those of the run-averaged model: per frame
    the mean TSS contribution and the mean False-Go numerator (the denominator is the ground truth's, shared by all
    runs), and the mean of each pooled metric. Every metric is a mean over frames or over stop / go states, so the
    mean over runs is the run-averaged model's value."""
    den = stats[0][2]
    assert all(np.array_equal(x[2], den) for x in stats), "the runs are not scored on the same frames"
    return (np.mean([x[0] for x in stats], 0), np.mean([x[1] for x in stats], 0), den,
            {k: float(np.mean([x[3][k] for x in stats])) for k in stats[0][3]})


def spread(stats):
    """Per metric: each run's value, their mean and their sample standard deviation (ddof 1; 0 with a single run)."""
    out = {}
    for k in METRICS:
        v = [float(x[3][k]) for x in stats]
        out[k] = {"runs": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    return out


def compare_runs(a_runs, b_runs, clusters, draws=10000, seed=0):
    """a_runs, b_runs: per run of each model, {cell: frame_stats}. Per cell: the run-averaged models paired frame by
    frame (a - b with a cluster-bootstrap 95% CI, ``stats.paired_frames``), the averaged models' metrics and the
    spread over runs. The runs are fixed: the CI covers the sampling of frames, not the choice of seeds."""
    out = {}
    for cell in a_runs[0]:
        a, b = average_runs([r[cell] for r in a_runs]), average_runs([r[cell] for r in b_runs])
        assert np.array_equal(a[2], b[2]), cell
        out[cell] = {"paired": paired_frames(a[:3], b[:3], clusters[cell], draws, seed), "a_metrics": a[3], "b_metrics": b[3],
                     "a_runs": spread([r[cell] for r in a_runs]), "b_runs": spread([r[cell] for r in b_runs])}
    return out


def claims(cells):
    """The claims of the publication protocol, from the paired cells ("a" = R, "b" = M)."""
    p = {c: cells[c]["paired"] for c in PRIMARY}
    return {"non_inferior_in_domain": all(p[c]["d_tss_ci95"][0] > -0.01 for c in PRIMARY),
            "better_in_domain": {c: p[c]["d_tss_ci95"][0] > 0 for c in PRIMARY},
            "lower_false_go": {c: p[c]["d_false_go_ci95"][1] < 0 for c in PRIMARY}}
