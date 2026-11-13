"""Thesis figures from the run outputs in runs/ (run from the repository root).

python thesis/make_figures.py  ->  thesis/figures/*.pdf
"""
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "thesis/figures"
SWIN, CNX, RFD = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SIGNAL = {"none": "#a8a7a2", "red": "#d93a3a", "yellow": "#eda100", "green": "#1f9d55"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4,
})


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def label_distribution():
    """A+B kept frames, ours vs the paper's Fig. 3 (REPORT.md section 4.3)."""
    dirs = ["straight", "left", "right", "U-turn"]
    ours = {"none": [6.5, 0.9, 96.6, 0.8], "red": [36.9, 40.8, 1.8, 40.9],
            "yellow": [2.7, 2.9, 0.1, 2.9], "green": [53.9, 55.5, 1.5, 55.4]}
    paper = {"none": [7.0, 7.2, 96.9, 7.2], "red": [40.7, 41.3, 0.3, 41.3],
             "yellow": [2.5, 2.8, 0.2, 2.8], "green": [49.7, 48.8, 2.7, 48.8]}
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    w = 0.36
    x = np.arange(len(dirs))
    for off, data, tag in ((-w / 2 - 0.01, ours, "ours"), (w / 2 + 0.01, paper, "paper")):
        bottom = np.zeros(len(dirs))
        for c in ("none", "red", "yellow", "green"):
            v = np.array(data[c])
            ax.bar(x + off, v, w, bottom=bottom, color=SIGNAL[c], edgecolor="white", linewidth=1,
                   label=c if tag == "ours" else None)
            bottom += v
        for xi in x:
            ax.text(xi + off, 102, tag, ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, dirs)
    ax.set_ylabel("share of kept frames (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "label_distribution.pdf")


def tss_per_epoch():
    swin = jsonl("runs/swin_t_AB/val_log.jsonl")
    cnx = jsonl("runs/convnextv2_t_AB/val_log.jsonl")
    rfd = jsonl("runs/rfdetr_m_AB/val_log.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, key, paper, title in ((axes[0], "tss", 0.8802, "TSS (higher is better)"),
                                  (axes[1], "false_go", 0.1372, "False-Go (lower is better)")):
        for log, col, name in ((swin, SWIN, "Swin-T"), (cnx, CNX, "ConvNeXtV2-T"), (rfd, RFD, "RF-DETR-M")):
            ax.plot([e["epoch"] for e in log], [e["A"][key] for e in log], "-o", color=col, label=name)
        ax.axhline(paper, color=MUTED, linestyle="--", linewidth=1)
        ax.text(15.0, paper, "paper", va="center", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("epoch")
        ax.set_xlim(-0.5, 17)
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "tss_per_epoch.pdf")


def dtld_overfitting():
    rep = jsonl("runs/rfdetr_m_dtld/epoch_report.jsonl")
    ep = [e["epoch"] for e in rep]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axes[0].plot(ep, [e["probe_loss"] for e in rep], "-o", color=SWIN, label="train probe")
    axes[0].plot(ep, [e["val_loss"] for e in rep], "-o", color=CNX, label="held-out val")
    axes[0].set_title("detection loss (EMA, eval mode)", fontsize=9)
    axes[1].plot(ep, [e["probe"]["tss"] for e in rep], "-o", color=SWIN, label="train probe")
    axes[1].plot(ep, [e["val"]["tss"] for e in rep], "-o", color=CNX, label="held-out val")
    axes[1].set_title("TSS", fontsize=9)
    for ax in axes:
        ax.axvline(4, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("epoch")
    axes[1].text(4.2, 0.962, "best val", fontsize=7, color=MUTED)
    axes[0].legend(fontsize=7)
    save(fig, "dtld_overfitting.pdf")


def atlas_recall():
    d = json.load(open("runs/rfdetr_m_dtld/ood_atlas/epoch4_iou0.5.json"))
    cams = (("front_wide", SWIN, "wide"), ("front_medium", CNX, "medium"), ("front_tele", RFD, "tele"))
    bins = list(d["front_wide"]["recall_by_height"])
    labels = ["<10", "10-20", "20-40", ">40"]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    w = 0.26
    for i, (cam, col, name) in enumerate(cams):
        rec = [h / max(n, 1) for h, n in (d[cam]["recall_by_height"][b] for b in bins)]
        ax.bar(np.arange(4) + (i - 1) * (w + 0.02), rec, w, color=col, label=name)
    ax.set_xticks(np.arange(4), labels)
    ax.set_xlabel("light height in the source image (px)")
    ax.set_ylabel("detection recall (IoU 0.5)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(title="ATLAS camera", fontsize=7, title_fontsize=7, loc="upper left")
    save(fig, "atlas_recall.pdf")


def per_direction():
    """subset_A val color accuracy per direction: both models vs the majority-color baseline."""
    swin = json.load(open("runs/swin_t_AB/per_direction.json"))
    rfd = json.load(open("runs/rfdetr_m_AB/per_direction.json"))
    sw = swin["A"]["directions"] if "directions" in swin.get("A", {}) else swin["A"]
    rf = rfd["A"]["directions"]
    dirs = ["straight", "left", "right", "uturn"]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    w = 0.26
    x = np.arange(4)
    ax.bar(x - w - 0.02, [rf[d]["majority_acc_color"] for d in dirs], w, color="#c9c8c2", label="majority color")
    ax.bar(x, [sw[d]["acc_color"] for d in dirs], w, color=SWIN, label="Swin-T (direction-aware)")
    ax.bar(x + w + 0.02, [rf[d]["acc_color"] for d in dirs], w, color=RFD, label="RF-DETR-M (structured)")
    ax.set_xticks(x, ["straight", "left", "right", "U-turn"])
    ax.set_ylabel(r"Acc$_\mathrm{color}$, subset_A val")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "per_direction.pdf")


def examples():
    for name in ("A-0004.jpg", "B-0028.jpg"):
        shutil.copy(os.path.join("runs/boss_deck/img", name), os.path.join(OUT, "example_" + name))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (label_distribution, tss_per_epoch, dtld_overfitting, atlas_recall, per_direction, examples):
        f()
        print("done", f.__name__)
