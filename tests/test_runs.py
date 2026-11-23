"""The multi-run comparison of the publication protocol: run averaging, spread, pairing and claims."""
import numpy as np
import pytest

from dtsu.evaluation import frame_stats
from dtsu.moves_evaluation import PRIMARY, average_runs, claims, compare_runs, spread
from dtsu.stats import paired_frames

N, D = 60, 3
RNG = np.random.default_rng(0)
GC = RNG.integers(0, 4, (N, D))  # colors: none / red / yellow / green
GP = (GC != 1).astype(int)  # stop (0) only on red
SEG = np.repeat(np.arange(12), N // 12)


def model(flip):
    """frame_stats of a prediction that gets a fraction ``flip`` of the states wrong."""
    rng = np.random.default_rng(int(flip * 1000))
    pc = np.where(rng.random((N, D)) < flip, rng.integers(0, 4, (N, D)), GC)
    return frame_stats(pc, (pc != 1).astype(int), GC, GP)


def test_one_run_is_the_single_run_comparison():
    a, b = model(0.1), model(0.2)
    out = compare_runs([{"c": a}], [{"c": b}], {"c": SEG})["c"]
    assert out["paired"] == paired_frames(a[:3], b[:3], SEG)
    assert out["a_metrics"] == a[3] and out["b_runs"]["tss"] == {"runs": [b[3]["tss"]], "mean": b[3]["tss"], "std": 0.0}


def test_averaged_model_is_the_mean_of_the_runs():
    runs = [model(f) for f in (0.05, 0.1, 0.3)]
    s, fg, den, m = average_runs(runs)
    for k in m:
        assert m[k] == pytest.approx(np.mean([r[3][k] for r in runs]))
    assert s.mean() == pytest.approx(m["tss"])
    assert fg.sum() / den.sum() == pytest.approx(m["false_go"])


def test_paired_difference_is_the_mean_of_the_run_differences():
    a_runs, b_runs = [model(f) for f in (0.05, 0.1, 0.15)], [model(f) for f in (0.2, 0.25, 0.3)]
    out = compare_runs([{"c": r} for r in a_runs], [{"c": r} for r in b_runs], {"c": SEG})["c"]["paired"]
    assert out["d_tss"] == pytest.approx(np.mean([a[3]["tss"] - b[3]["tss"] for a, b in zip(a_runs, b_runs)]))
    assert out["d_false_go"] == pytest.approx(np.mean([a[3]["false_go"] - b[3]["false_go"] for a, b in zip(a_runs, b_runs)]))
    assert out["d_tss_ci95"][0] <= out["d_tss"] <= out["d_tss_ci95"][1]


def test_spread_uses_the_sample_standard_deviation():
    runs = [model(f) for f in (0.05, 0.1, 0.3)]
    v = [r[3]["em"] for r in runs]
    assert spread(runs)["em"]["std"] == pytest.approx(np.std(v, ddof=1))


def test_runs_on_different_frames_are_refused():
    a = model(0.1)
    other = (a[0], a[1], a[2] + 1, a[3])
    with pytest.raises(AssertionError):
        average_runs([a, other])


def test_claims_follow_the_protocol_bounds():
    def cell(tss_ci, fg_ci):
        return {"paired": {"d_tss_ci95": tss_ci, "d_false_go_ci95": fg_ci}}
    cells = {c: cell([-0.005, 0.02], [-0.03, -0.001]) for c in PRIMARY}
    cells[PRIMARY[1]] = cell([0.001, 0.03], [-0.02, 0.01])
    out = claims(cells)
    assert out["non_inferior_in_domain"]
    assert out["better_in_domain"] == {PRIMARY[0]: False, PRIMARY[1]: True, PRIMARY[2]: False}
    assert out["lower_false_go"] == {PRIMARY[0]: True, PRIMARY[1]: False, PRIMARY[2]: True}
    cells[PRIMARY[2]] = cell([-0.011, 0.02], [-0.03, -0.001])
    assert not claims(cells)["non_inferior_in_domain"]
