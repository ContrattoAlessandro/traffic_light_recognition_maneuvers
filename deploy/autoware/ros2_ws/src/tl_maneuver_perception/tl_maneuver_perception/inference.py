"""ONNX inference for the model R and the junction rule. No ROS import, so it can be tested alone.

Same steps as deploy/reference_infer.py in the training repository: square resize without padding (bilinear with
antialiasing), score threshold, class-agnostic NMS, then the junction rule on the lights the model assigns to the junction.
"""
import json
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from PIL import Image

ARROW_MOVES = {
    "arrow_left": {"left"},
    "arrow_right": {"right"},
    "arrow_straight": {"straight"},
    "arrow_straight_left": {"straight", "left"},
    "arrow_straight_right": {"straight", "right"},
}
NONE = "none"


class Detection(NamedTuple):
    picto: str
    color: str
    junction: bool
    score: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes; indices of the kept boxes, highest score first."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (area[i] + area[rest] - inter + 1e-12)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def rule(lights: Sequence[Tuple[str, str, float]]) -> Dict[str, Tuple[str, float]]:
    """(picto, color, score) of the junction lights -> maneuver: (color, score of the deciding light).

    An arrow sets its own maneuvers (the highest-scoring arrow wins); every other maneuver takes the color of the
    highest-scoring circle, unknown pictograms counting as circles. "none" when nothing governs it.
    R has no U-turn output (Lanelet2 turn_direction has only straight, left and right).
    """
    circle = max(((s, c) for p, c, s in lights if p in ("circle", "unknown")), default=None)
    best: Dict[str, Tuple[float, str]] = {}
    for p, c, s in lights:
        for move in ARROW_MOVES.get(p, ()):
            if move not in best or s > best[move][0]:
                best[move] = (s, c)
    out = {}
    for d in ("straight", "left", "right"):
        src = best.get(d) or circle
        out[d] = (src[1], src[0]) if src else (NONE, 0.0)
    return out


class ManeuverModel:
    def __init__(self, model_path: str, config_path: str, use_gpu: bool = True, gpu_id: int = 0, threshold: float = -1.0):
        import onnxruntime as ort

        self.cfg = json.load(open(config_path))
        self.res = int(self.cfg["resolution"])
        self.threshold = float(self.cfg["threshold"]) if threshold < 0 else float(threshold)
        self.mean = np.array(self.cfg["mean"], dtype=np.float32)[None, None]
        self.std = np.array(self.cfg["std"], dtype=np.float32)[None, None]
        providers: List = ["CPUExecutionProvider"]
        if use_gpu:
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": gpu_id}))
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.provider = self.session.get_providers()[0]

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(rgb).resize((self.res, self.res), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """Kept lights of one RGB uint8 image (H, W, 3), highest score first."""
        h, w = rgb.shape[:2]
        boxes, score, picto, color, junction = self.session.run(None, {"image": self.preprocess(rgb)})
        boxes, score, picto, color, junction = boxes[0], score[0], picto[0], color[0], junction[0]
        k = score > self.threshold
        if not k.any():
            return []
        cx, cy, bw, bh = boxes[k].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
        s, p, c, j = score[k], picto[k].argmax(-1), color[k].argmax(-1), junction[k] > self.cfg["junction_threshold"]
        pictos, colors = self.cfg["pictograms"], self.cfg["colors"]
        return [Detection(pictos[int(p[i])], colors[int(c[i])], bool(j[i]), float(s[i]), tuple(float(v) for v in xyxy[i]))
                for i in nms(xyxy, s, float(self.cfg["nms_iou"]))]


def directions(detections: Sequence[Detection]) -> Dict[str, Tuple[str, float]]:
    return rule([(d.picto, d.color, d.score) for d in detections if d.junction])
