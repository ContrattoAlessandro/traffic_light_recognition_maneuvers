#!/usr/bin/env python3
"""Run the node's inference on golden/*.png and compare the maneuvers with golden/*.json (written by
scripts/export_onnx.py on the export machine, the same frames as the R hand-off package)."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HOME, "ros2_ws", "src", "tl_maneuver_perception"))
from tl_maneuver_perception.inference import ManeuverModel, directions  # noqa: E402

model = ManeuverModel(os.path.join(HOME, "model", "r_1280_opset17.onnx"), os.path.join(HOME, "model", "model_config.json"))
print("execution provider:", model.provider)
if model.provider != "CUDAExecutionProvider":
    print("WARNING: no GPU. Check nvidia-smi: the node would be too slow for closed-loop driving.")
cases = sorted(glob.glob(os.path.join(HOME, "golden", "*.json")))
if not cases:
    sys.exit("no golden/*.json found")
wrong = 0
for path in cases:
    case = json.load(open(path))
    rgb = np.asarray(Image.open(os.path.join(HOME, "golden", case["image"])).convert("RGB"))
    got = {m: v[0] for m, v in directions(model.detect(rgb)).items()}
    ok = got == case["maneuvers"]
    wrong += not ok
    print(f"{case['image']}: {'ok' if ok else 'DIFFERENT, expected ' + str(case['maneuvers'])} {got}")
print("self test", "passed" if not wrong else f"FAILED on {wrong} of {len(cases)} images")
sys.exit(1 if wrong else 0)
