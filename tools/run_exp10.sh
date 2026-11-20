#!/usr/bin/env bash
# Exp 10 (EXPERIMENT_LOG.md): the WF recipe of Exp 9 (tools/run_exp9.sh) at resolution 1280, then the evaluation.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_exp8
OUT=runs/exp10_R
mkdir -p $OUT runs/exp10_eval
echo "train R $(date +%T)"
$PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1280 --epochs 5 --batch-size 4 \
  --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized --class-weights 0.5 --flip-swap \
  --out-dir $OUT >> $OUT/train.log 2>&1
echo "train R exit $? $(date +%T)"
$PY -W ignore -u tools/eval_exp10.py >> runs/exp10_eval/eval.log 2>&1
echo "eval exit $? $(date +%T)"
