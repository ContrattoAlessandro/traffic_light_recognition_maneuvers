#!/usr/bin/env bash
# Exp 8 (EXPERIMENT_LOG.md): the Exp 7 data plus BDD100K train clean frames, for night and horizontal lights.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_exp8
OUT=runs/exp8_bdd_F
[ -f $D/class_names.json ] || $PY tools/make_multi_picto_coco.py --add-bdd --out $D
mkdir -p $OUT
echo "train $(date +%T)"
$PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1024 --epochs 5 --batch-size 4 \
  --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized --out-dir $OUT >> $OUT/train.log 2>&1
echo "train exit $? $(date +%T)"
$PY -W ignore -u tools/eval_exp8.py --run $OUT --data-dir $D >> $OUT/eval.log 2>&1
echo "eval exit $? $(date +%T)"
