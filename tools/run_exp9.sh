#!/usr/bin/env bash
# Exp 9 (EXPERIMENT_LOG.md): class imbalance. LA (no training) first, then W, then WF, then the evaluation of all arms.
# Each arm is the F8 recipe (tools/run_exp8.sh) plus one change.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_exp8
mkdir -p runs/exp9_eval
echo "LA eval $(date +%T)"
$PY -W ignore -u tools/eval_exp9.py --arms LA >> runs/exp9_eval/eval_LA.log 2>&1
echo "LA exit $? $(date +%T)"
for ARM in W WF; do
  OUT=runs/exp9_$ARM
  EXTRA="--class-weights 0.5"
  [ $ARM = WF ] && EXTRA="$EXTRA --flip-swap"
  mkdir -p $OUT
  echo "train $ARM $(date +%T)"
  $PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1024 --epochs 5 --batch-size 4 \
    --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized $EXTRA --out-dir $OUT >> $OUT/train.log 2>&1
  echo "train $ARM exit $? $(date +%T)"
done
$PY -W ignore -u tools/eval_exp9.py --arms LA W WF >> runs/exp9_eval/eval.log 2>&1
echo "eval exit $? $(date +%T)"
