"""Where the data lives. Override with the environment variables DATASETS and OPENLANE.

- DATASETS: the source datasets (DTLD, DTLD_jpg, ATLAS_full, BDD100K, LISA, S2TLD, VZC_TLD); default ``datasets/``.
- OPENLANE: the OpenLane-V2 root holding ``subset_A`` and ``subset_B``; default ``data/OpenLane-V2``.
Everything the code writes goes under ``data/`` (labels, training exports) and ``runs/`` (checkpoints, evaluations).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.abspath(os.environ.get("DATASETS", os.path.join(REPO, "datasets")))
OPENLANE = os.path.abspath(os.environ.get("OPENLANE", os.path.join(REPO, "data", "OpenLane-V2")))
LABELS = os.path.join(REPO, "data", "labels")
SPLITS = os.path.join(REPO, "splits")
ANNOTATIONS = os.path.join(REPO, "annotations")
