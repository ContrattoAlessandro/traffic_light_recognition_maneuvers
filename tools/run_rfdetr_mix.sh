#!/usr/bin/env bash
# Exp 5 (REPLICATION_PLAN.md): mixed A+B+D*_US training, scored on A val, B val, D*_US test (US rules) and
# DTLD test (German rules), then compared with A+B (r1) and the DTLD model on the same frames.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"
OUT=runs/rfdetr_m_mix
CN=data/rfdetr_coco_fit_mix/class_names.json
mkdir -p $OUT
if [ ! -f $OUT/epoch_3_ema.pth ]; then
  resume=(); [ -f $OUT/last.ckpt ] && resume=(--resume $OUT/last.ckpt)
  echo "train $(date +%T)"
  $PY tools/train_rfdetr.py --dataset-dir data/rfdetr_coco_fit_mix --out-dir $OUT --variant RFDETRMedium --resolution 1024 \
    --epochs 4 --batch-size 4 --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 "${resume[@]}" >> $OUT/train.log 2>&1
  echo "train exit $? $(date +%T)"
fi
best() { $PY -c "
import json
log = [json.loads(l) for l in open('$1')]
b = max(log, key=lambda e: e['tune_tss'])
print(b['epoch'], b['threshold'])"; }
echo "score mix $(date +%T)"
$PY tools/eval_rfdetr.py --run $OUT --dataset mixed --class-names $CN >> $OUT/eval.log 2>&1
set -- $(best $OUT/val_log.jsonl)
echo "mix best epoch $1 threshold $2"
$PY tools/eval_rfdetr.py --run $OUT --dataset mixed --class-names $CN --epochs $1 --threshold $2 --per-direction --tag _mix >> $OUT/eval.log 2>&1
for entry in "runs/tune_rfdetr/r1_m_default data/rfdetr_coco_fit/class_names.json" "runs/rfdetr_m_dtld data/rfdetr_coco_dtld/class_names.json"; do
  set -- $entry
  run=$1 cn=$2
  set -- $(best $run/val_log.jsonl)
  echo "[$run] epoch $1 threshold $2 on the mixed sets $(date +%T)"
  $PY tools/eval_rfdetr.py --run $run --dataset mixed --class-names $cn --epochs $1 --threshold $2 --per-direction --tag _mix >> $run/eval_mix.log 2>&1
done
for b in runs/tune_rfdetr/r1_m_default runs/rfdetr_m_dtld; do
  $PY tools/bootstrap_compare.py --a $OUT --b $b --a-preds val_preds_mix.npz --b-preds val_preds_mix.npz \
    --sets A B DTLD_US_test DTLD_test --out $OUT/bootstrap_vs_$(basename $b).json >> $OUT/bootstrap.log 2>&1
done
echo "ALL DONE $(date +%T)"
