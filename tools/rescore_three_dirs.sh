#!/usr/bin/env bash
# Re-score the thesis results after U-turn was dropped from DIRECTIONS (straight / left / right only).
# Model selection stays as it was made (epochs, thresholds); only the reported evaluations are re-run,
# with outputs tagged _3dir next to the original four-direction files.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"
DC=data/rfdetr_coco_dtld/class_names.json
step() { echo "== $* $(date +%T)"; }
step rfdetr_AB; $PY tools/eval_rfdetr.py --run runs/rfdetr_m_AB --threshold 0.20 --epochs 0 1 2 3 --per-direction --tag _3dir
step dtld_e4;   $PY tools/eval_rfdetr.py --run runs/rfdetr_m_dtld --dataset dtld --class-names $DC --threshold 0.25 --epochs 4 --per-direction --tag _3dir_e4
step dtld_e14;  $PY tools/eval_rfdetr.py --run runs/rfdetr_m_dtld --dataset dtld --class-names $DC --threshold 0.30 --epochs 14 --per-direction --tag _3dir
for t in lisa openlane; do
  step ood_$t;  $PY tools/eval_rfdetr_ood.py --target $t --epoch 4 --source-threshold 0.25 --tag _3dir
  step ood_${t}_pad2; $PY tools/eval_rfdetr_ood.py --target $t --epoch 4 --source-threshold 0.25 --pad-aspect 2 --tag _3dir
done
step done
