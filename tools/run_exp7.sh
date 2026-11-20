#!/usr/bin/env bash
# Exp 7 (EXPERIMENT_LOG.md): one factorized model on DTLD + ATLAS + OpenLane-V2 with partial labels.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_multi_picto
OUT=runs/exp7_multi_F
[ -f $D/class_names.json ] || $PY tools/make_multi_picto_coco.py --out $D
mkdir -p $OUT
echo "train $(date +%T)"
$PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1024 --epochs 5 --batch-size 4 \
  --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized --out-dir $OUT >> $OUT/train.log 2>&1
echo "train exit $? $(date +%T)"
