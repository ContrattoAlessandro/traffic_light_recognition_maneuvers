#!/usr/bin/env bash
# Maneuver labels for the extra datasets DTLD and LISA (dtsu/labels_dtld.py, dtsu/labels_lisa.py).
# Splits: train / val / test. test = the official test split. val = whole sessions (DTLD) or clips
# (LISA) held out from the official train split via tools/make_tune_split.py, written next to the
# OpenLane-V2 tune splits as data/labels/tune/<set>_train_{fit,tune}.jsonl.
# DATASETS defaults to the traffic-light-detection project's dataset folder.
set -eu
cd "$(dirname "$0")/.."
DATASETS=${DATASETS:-../traffic-light-detection/datasets}
L=data/labels
PY=./.venv/Scripts/python
mkdir -p $L/tune

for s in train test; do
  $PY -m dtsu.labels_dtld --labels $DATASETS/DTLD/v2.0/DTLD_$s.json \
    --route-frames $DATASETS/DTLD_route/route_frames.json --out $L/dtld_$s.jsonl | tee $L/dtld_$s.log
  $PY -m dtsu.labels_lisa --root $DATASETS/LISA --split $s --out $L/lisa_$s.jsonl | tee $L/lisa_$s.log
done
$PY tools/make_tune_split.py $L/dtld_train.jsonl $L/tune/dtld_train --fraction 0.1
# LISA train has only 18 clips (13 day, 5 night); seed 0 holds out dayClip9, dayClip10 and nightClip1
$PY tools/make_tune_split.py $L/lisa_train.jsonl $L/tune/lisa_train --fraction 0.15 --seed 0
