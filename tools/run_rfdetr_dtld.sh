#!/usr/bin/env bash
# Structured baseline on DTLD: RF-DETR Medium at 1024 on "<moves>|<color>" light classes + the label rules
# (German right turns: near_turn = "signal"). Recipe of tools/run_rfdetr.sh (default lr, which RF-DETR keeps
# constant), but 15 epochs with a per-epoch overfitting report (train_rfdetr.py --report): loss and direction
# metrics on the held-out tune split vs a fixed 1,000-frame train probe -> runs/rfdetr_m_dtld/epoch_report.jsonl.
# `touch runs/rfdetr_m_dtld/STOP` ends training after the current epoch.
# Trains on the DTLD fit split (data/rfdetr_coco_dtld); the tune split picks the score threshold and
# DTLD test is only reported, for the best-val epoch and the last one.
# Re-running resumes training from runs/rfdetr_m_dtld/last.ckpt; scoring skips epochs already logged.
set -u
cd "$(dirname "$0")/.."
OUT=runs/rfdetr_m_dtld
D=data/rfdetr_coco_dtld
mkdir -p $OUT
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
[ -f $D/test/records.jsonl ] || ./.venv/Scripts/python tools/make_rfdetr_coco.py --dataset dtld --fit --probe 1000 --out $D
RESUME=()
[ -f $OUT/last.ckpt ] && RESUME=(--resume $OUT/last.ckpt)
./.venv/Scripts/python -W ignore -u tools/train_rfdetr.py --dataset-dir $D --out-dir $OUT --variant RFDETRMedium \
  --resolution 1024 --epochs 15 --batch-size 4 --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --report \
  "${RESUME[@]}" >> $OUT/train.log 2>&1
echo "train exit $?"
EPOCHS=$(./.venv/Scripts/python -c "
import json
h = [json.loads(l) for l in open('$OUT/epoch_report.jsonl')]
print(*sorted({h[-1]['best_epoch'], h[-1]['epoch']}))")
./.venv/Scripts/python -W ignore -u tools/eval_rfdetr.py --run $OUT --dataset dtld --class-names $D/class_names.json \
  --epochs $EPOCHS --per-direction >> $OUT/eval.log 2>&1
echo "eval exit $?"
