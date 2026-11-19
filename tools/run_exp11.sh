#!/usr/bin/env bash
# Exp 11 (REPLICATION_PLAN.md): the R recipe of Exp 10 at resolution 1536 (batch 2 x accum 8: batch 4 does not fit),
# first epoch only (5-epoch schedule, resumable from last.ckpt), then the evaluation.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_exp8
OUT=runs/exp11_S
mkdir -p $OUT runs/exp11_eval
echo "train S $(date +%T)"
$PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1536 --epochs 5 --stop-after 1 \
  --batch-size 2 --grad-accum 8 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized --class-weights 0.5 \
  --flip-swap --out-dir $OUT >> $OUT/train.log 2>&1
echo "train S exit $? $(date +%T)"
$PY -W ignore -u tools/eval_exp11.py >> runs/exp11_eval/eval.log 2>&1
echo "eval exit $? $(date +%T)"
