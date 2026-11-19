#!/usr/bin/env bash
# Exp 11, continued at the user's request (2026-11-17): arm S resumed from last.ckpt (end of epoch 0)
# to the full 5-epoch schedule, then S selected over all epochs and evaluated against R and WF.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_exp8
OUT=runs/exp11_S
CKPT=${1:-$OUT/epoch0_backup/last.ckpt}  # checkpoint to resume from (default: end of epoch 0)
mkdir -p $OUT/epoch0_backup runs/exp11_eval_full
cp -n $OUT/args.json $OUT/metrics.csv $OUT/last.ckpt $OUT/training_config.json $OUT/epoch0_backup/
echo "resume S from $CKPT $(date +%T)"
$PY -W ignore -u tools/train_rfdetr.py --dataset-dir $D --variant RFDETRMedium --resolution 1536 --epochs 5 \
  --batch-size 2 --grad-accum 8 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0 --factorized --class-weights 0.5 \
  --flip-swap --out-dir $OUT --resume $CKPT >> $OUT/train.log 2>&1
echo "resume S exit $? $(date +%T)"
$PY -W ignore -u tools/eval_exp11.py --full >> runs/exp11_eval_full/eval.log 2>&1
echo "eval exit $? $(date +%T)"
