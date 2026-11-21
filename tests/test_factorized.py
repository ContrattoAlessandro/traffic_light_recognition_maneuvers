"""Partial labels, class weights and the label-swapping flip of the factorized head."""
import numpy as np

from dtsu.factorized import IGNORE, attribute_table, class_weights, mirror_classes
from dtsu.rules import LAMP_COLORS, PICTOGRAMS, class_name, parse


def test_unlabelled_attributes_are_ignored():
    t = attribute_table(["?|?|red", "arrow_left|J|green", "circle|N|yellow"]).tolist()
    assert t[0] == [IGNORE, LAMP_COLORS.index("red"), IGNORE]
    assert t[1] == [PICTOGRAMS.index("arrow_left"), LAMP_COLORS.index("green"), 1]
    assert t[2] == [PICTOGRAMS.index("circle"), LAMP_COLORS.index("yellow"), 0]


def test_class_weights_keep_the_count_weighted_mean_at_one():
    counts = {"picto": {"circle": 1000, "arrow_left": 100, "arrow_right": 10}, "color": {"red": 500, "green": 400, "yellow": 50}}
    w = class_weights(counts, 0.5)
    for head, names in (("picto", PICTOGRAMS), ("color", LAMP_COLORS)):
        n = {k: v for k, v in counts[head].items()}
        mean = sum(n[k] * w[head][names.index(k)] for k in n) / sum(n.values())
        assert abs(mean - 1) < 1e-4
    assert w["picto"][PICTOGRAMS.index("arrow_right")] > w["picto"][PICTOGRAMS.index("circle")]
    assert w["picto"][PICTOGRAMS.index("other")] == 1.0  # no count -> weight 1


def test_flip_permutation_swaps_arrows_and_is_an_involution():
    names = ["circle|J|red", "arrow_left|J|green", "arrow_straight_left|J|red", "arrow_right|N|red"]
    ext, perm = mirror_classes(names)
    assert "arrow_straight_right|J|red" in ext and "arrow_right|J|green" in ext and "arrow_left|N|red" in ext
    assert all(perm[perm[i]] == i for i in range(len(ext)))
    for i, n in enumerate(ext):
        p, j, c = parse(n)
        q, k, d = parse(ext[perm[i]])
        assert (j, c) == (k, d)
        assert q == {"arrow_left": "arrow_right", "arrow_right": "arrow_left", "arrow_straight_left": "arrow_straight_right",
                     "arrow_straight_right": "arrow_straight_left"}.get(p, p)
    assert ext[perm[ext.index(class_name("circle", True, "red"))]] == "circle|J|red"
    assert np.array_equal(sorted(perm), np.arange(len(ext)))
