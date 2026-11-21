#!/usr/bin/env bash
# Rebuild the labels and the training set, train the main model R and evaluate it.
# Needs the datasets under $DATASETS and OpenLane-V2 under $OPENLANE (see README.md). About 16 h of training and
# 3 h of evaluation on one 12 GB GPU.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}

$PY scripts/build_labels.py
$PY scripts/export_training_set.py --out data/train_full
$PY scripts/train.py --dataset-dir data/train_full --out-dir runs/R --variant RFDETRMedium --resolution 1280 \
  --epochs 5 --batch-size 4 --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 \
  --class-weights 0.5 --flip-swap
$PY scripts/evaluate.py --run runs/R --out results/R.json
