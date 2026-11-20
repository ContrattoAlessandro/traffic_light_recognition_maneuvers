"""Matching, average precision and the paired bootstraps."""
import numpy as np
import torch

from dtsu.stats import det_table, match, paired_frames, paired_rates, weighted_ap


def test_match_greedy_by_score_with_ignore_boxes():
    gt = torch.tensor([[0, 0, 10, 20], [50, 0, 60, 20], [100, 0, 110, 20]], dtype=torch.float32)
    ignore = torch.tensor([False, False, True])
    pred = torch.tensor([[1, 1, 9, 19], [0, 0, 11, 21], [101, 1, 109, 19], [200, 0, 210, 20]], dtype=torch.float32)
    score = torch.tensor([0.5, 0.9, 0.8, 0.7])
    # the higher score takes gt 0; the other duplicate is a false positive; the ignore box is neither
    assert match(pred, score, gt, ignore, "center").tolist() == [-1, 0, -2, -1]


def test_center_matching_accepts_wider_boxes():
    gt = torch.tensor([[0, 0, 20, 20]], dtype=torch.float32)
    pred = torch.tensor([[5, 0, 15, 20]], dtype=torch.float32)  # IoU 0.5 but centers inside each other
    assert match(pred, torch.tensor([0.9]), gt, torch.tensor([False]), "center").tolist() == [0]


def test_ap_is_one_for_perfect_detections():
    items = [{"boxes": torch.tensor([[0, 0, 10, 20]], dtype=torch.float32), "color": ["red"]} for _ in range(3)]
    preds = [(None, torch.tensor([0.9]), torch.tensor([[0, 0, 10, 20]], dtype=torch.float32)) for _ in range(3)]
    assert weighted_ap(det_table(items, preds), np.ones(3)) == 1.0


def test_paired_bootstrap_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    s = rng.random(50)
    a = (s, rng.integers(0, 2, 50), np.ones(50))
    r = paired_frames(a, a, np.repeat(np.arange(10), 5), draws=200)
    assert r["d_tss"] == 0 and r["d_tss_ci95"] == [0.0, 0.0] and r["a_tss"] == r["b_tss"]
    ok = rng.random(50) > 0.5
    r = paired_rates(np.repeat(np.arange(10), 5), np.zeros(50, int), ok, ok, draws=200)
    assert r["d"] == 0 and r["a"] == ok.mean()
