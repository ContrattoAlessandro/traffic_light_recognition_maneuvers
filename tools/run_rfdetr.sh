#!/usr/bin/env bash
# Structured baseline, final run: RF-DETR Medium at 1024 on "<moves>|<color>" light classes + the label rules.
# Config from tools/run_rfdetr_tuning.sh (r1_m_default: default lr won; tune TSS plateaued after 2 epochs,
# so 4 epochs on the 13% larger full train set). Trains on A+B train (data/rfdetr_coco), then scores every
# epoch's EMA weights on A/B val at the threshold picked during tuning (0.20): the tune split is inside
# this run's training data, so it can no longer pick the threshold.
# Re-running resumes training from runs/rfdetr_m_AB/last.ckpt; scoring skips epochs already logged.
set -u
cd "$(dirname "$0")/.."
OUT=runs/rfdetr_m_AB
mkdir -p $OUT
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
RESUME=()
[ -f $OUT/last.ckpt ] && RESUME=(--resume $OUT/last.ckpt)
./.venv/Scripts/python -W ignore -u tools/train_rfdetr.py --out-dir $OUT --variant RFDETRMedium \
  --resolution 1024 --epochs 4 --batch-size 4 --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 \
  "${RESUME[@]}" >> $OUT/train.log 2>&1
echo "train exit $?"
./.venv/Scripts/python -W ignore -u tools/eval_rfdetr.py --run $OUT --threshold 0.20 --per-direction >> $OUT/eval.log 2>&1
echo "eval exit $?"
