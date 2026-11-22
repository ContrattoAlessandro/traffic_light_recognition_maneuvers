"""The torch-free reference pipeline of the ONNX package (deploy/reference_infer.py) must decode exactly as the
repository does: same junction rule, same NMS."""
import os
import random
import sys

import numpy as np
import torch
from torchvision.ops import nms as tv_nms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))
import reference_infer as ref  # noqa: E402
from dtsu import rules  # noqa: E402


def test_constants_match_repository():
    assert ref.DIRECTIONS == tuple(rules.DIRECTIONS)
    assert ref.ROUND == tuple(rules.ROUND)
    assert {k: set(v) for k, v in ref.ARROW_MOVES.items()} == rules.ARROW_MOVES


def test_junction_rule_matches_repository():
    rng = random.Random(0)
    for _ in range(2000):
        lights = [{"pictogram": rng.choice(rules.PICTOGRAMS), "color": rng.choice(rules.LAMP_COLORS),
                   "score": rng.random(), "junction": rng.random()} for _ in range(rng.randint(0, 6))]
        repo = rules.junction_rule([(lt["pictogram"], lt["color"], lt["score"]) for lt in lights if lt["junction"] > 0.5])
        assert ref.junction_rule(lights) == repo


def test_nms_matches_torchvision():
    g = np.random.default_rng(0)
    for _ in range(200):
        n = int(g.integers(1, 40))
        xy = g.uniform(0, 100, (n, 2))
        boxes = np.concatenate([xy, xy + g.uniform(2, 30, (n, 2))], 1).astype(np.float32)
        scores = g.uniform(0, 1, n).astype(np.float32)
        mine = ref.nms(boxes, scores, 0.5)
        theirs = tv_nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5).numpy()
        assert mine.tolist() == theirs.tolist()
