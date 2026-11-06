#!/usr/bin/env bash
# Per-light OOD perception test (tools/eval_rfdetr_perception.py) of every RF-DETR run on ATLAS, S2TLD,
# BDD100K and LISA. Each run is scored at the epoch with the best source-tune TSS and that epoch's
# source-tune threshold (from its val_log.jsonl, written by tools/eval_rfdetr.py). Skips finished files.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"
RUNS=(
  "runs/rfdetr_m_A data/rfdetr_coco_fit_A/class_names.json"
  "runs/rfdetr_m_B data/rfdetr_coco_fit_B/class_names.json"
  "runs/tune_rfdetr/r1_m_default data/rfdetr_coco_fit/class_names.json"
  "runs/rfdetr_m_dtld data/rfdetr_coco_dtld/class_names.json"
)
for entry in "${RUNS[@]}"; do
  set -- $entry
  run=$1 cn=$2
  set -- $($PY -c "
import json
log = [json.loads(l) for l in open('$run/val_log.jsonl')]
best = max(log, key=lambda e: e['tune_tss'])
print(best['epoch'], best['threshold'])")
  todo=()
  for d in atlas s2tld bdd100k lisa; do [ -f $run/ood_perception/epoch$1_$d.json ] || todo+=($d); done
  [ ${#todo[@]} -eq 0 ] && { echo "[$run] done"; continue; }
  echo "[$run] epoch $1 threshold $2: ${todo[*]} $(date +%T)"
  $PY tools/eval_rfdetr_perception.py --run $run --class-names $cn --epoch $1 --threshold $2 --datasets "${todo[@]}" \
    >> $run/perception.log 2>&1
  echo "[$run] exit $? $(date +%T)"
done
echo "ALL DONE $(date +%T)"
