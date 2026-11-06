#!/usr/bin/env bash
# Cross-subset / cross-dataset protocol (constant for every run):
#   train on the source's fit split, RF-DETR Medium at 1024, default lr, 4 epochs (recipe of r1_m_default);
#   per epoch, the score threshold is picked on the source's own tune split; the reported epoch is the one
#   with the best source-tune TSS. Targets: OpenLane-V2 A val, B val, and D*_US test (DTLD relabelled
#   with the OpenLane-V2 protocol, tools/make_dtld_us.py), all at the source threshold.
#   A   -> runs/rfdetr_m_A   (data/rfdetr_coco_fit_A)
#   B   -> runs/rfdetr_m_B   (data/rfdetr_coco_fit_B)
#   A+B -> runs/tune_rfdetr/r1_m_default (already trained on the A+B fit split with this recipe)
# Re-running resumes training and skips epochs already scored.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"

train() {  # out, dataset dir
  local out=$1 d=$2
  if [ -f $out/epoch_3_ema.pth ]; then echo "[$out] trained"; return; fi
  mkdir -p $out
  local resume=()
  [ -f $out/last.ckpt ] && resume=(--resume $out/last.ckpt)
  echo "[$out] train $(date +%T)"
  $PY tools/train_rfdetr.py --dataset-dir $d --out-dir $out --variant RFDETRMedium --resolution 1024 --epochs 4 \
    --batch-size 4 --grad-accum 4 --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 "${resume[@]}" >> $out/train.log 2>&1
  echo "[$out] train exit $? $(date +%T)"
}

score() {  # run, preset, class names
  local run=$1 preset=$2 cn=$3
  echo "[$run] score $(date +%T)"
  $PY tools/eval_rfdetr.py --run $run --dataset $preset --class-names $cn >> $run/eval.log 2>&1
  local pick
  pick=$($PY -c "
import json
log = [json.loads(l) for l in open('$run/val_log.jsonl')]
best = max(log, key=lambda e: e['tune_tss'])
print(best['epoch'], best['threshold'])")
  set -- $run $preset $cn $pick
  echo "[$run] best source-tune epoch $4 threshold $5"
  [ -f $run/per_direction.json ] && grep -q "\"epoch\": $4," $run/per_direction.json || \
    $PY tools/eval_rfdetr.py --run $run --dataset $preset --class-names $cn --epochs $4 --per-direction >> $run/eval.log 2>&1
  [ -f $run/ood_dtld_us/epoch$4_squeeze.json ] || \
    $PY tools/eval_rfdetr_ood.py --target dtld_us --keep-all --run $run --class-names $cn --epoch $4 \
      --source-threshold $5 >> $run/eval.log 2>&1
  echo "[$run] done $(date +%T)"
}

train runs/rfdetr_m_A data/rfdetr_coco_fit_A
train runs/rfdetr_m_B data/rfdetr_coco_fit_B
score runs/rfdetr_m_A openlane_A data/rfdetr_coco_fit_A/class_names.json
score runs/rfdetr_m_B openlane_B data/rfdetr_coco_fit_B/class_names.json
score runs/tune_rfdetr/r1_m_default openlane data/rfdetr_coco_fit/class_names.json
echo "ALL DONE $(date +%T)"
