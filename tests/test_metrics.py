import numpy as np
import pytest

from dtsu import COLORS, PASS, color_to_pass
from dtsu.metrics import evaluate

R, G, Y, N = (COLORS.index(c) for c in ("red", "green", "yellow", "none"))
STOP, GO = PASS.index("stop"), PASS.index("go")


def to_pass(colors):
    return np.vectorize(lambda c: PASS.index(color_to_pass(COLORS[c])))(colors)


def test_perfect():
    gc = np.array([[R, R, N, R], [G, G, N, G]])
    m = evaluate(gc, to_pass(gc), gc, to_pass(gc))
    assert m["acc_color"] == m["acc_pass"] == m["em"] == m["tss"] == 1.0
    assert m["false_go"] == m["false_stop"] == 0.0


def test_hand_computed():
    gc = np.array([[R, R, N, R], [G, G, N, G]])
    # frame 0: straight red predicted green (false go); frame 1: right none predicted yellow
    pc = np.array([[G, R, N, R], [G, G, Y, G]])
    m = evaluate(pc, to_pass(pc), gc, to_pass(gc))
    assert m["acc_color"] == pytest.approx(6 / 8)
    assert m["acc_pass"] == pytest.approx(7 / 8)  # yellow vs none are both go
    assert m["em"] == 0.0
    assert m["tss"] == pytest.approx((6 / 8 + 7 / 8 + 0) / 3)
    assert m["false_go"] == pytest.approx(1 / 3)  # 3 gt-stop states, one predicted go
    assert m["false_stop"] == 0.0


def test_false_stop():
    gc = np.array([[G, G, N, G]])
    pc = np.array([[R, G, N, G]])
    m = evaluate(pc, to_pass(pc), gc, to_pass(gc))
    assert m["false_stop"] == pytest.approx(1 / 4)
    assert m["false_go"] == 0.0
