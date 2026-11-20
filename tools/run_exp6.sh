#!/usr/bin/env bash
# Exp 6 (EXPERIMENT_LOG.md): DTLD pictogram / junction / color lights, joint classes (J) vs factorized head (F).
# Same export, recipe, epochs and seed; only the head differs. Then both are scored (tools/eval_picto.py:
# epoch and threshold on tune, rule TSS and per-light metrics on DTLD test) and compared (tools/compare_picto.py).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=data/rfdetr_coco_dtld_picto
[ -f $D/dtld_test_records.jsonl ] || $PY tools/make_dtld_picto_coco.py --out $D
COMMON=(--dataset-dir $D --variant RFDETRMedium --resolution 1024 --epochs 6 --batch-size 4 --grad-accum 4
        --lr 1e-4 --lr-encoder 1.5e-4 --workers 6 --seed 0)
for arm in J F; do
  OUT=runs/exp6_picto_$arm
  mkdir -p $OUT
  EXTRA=(); [ $arm = F ] && EXTRA=(--factorized)
  echo "train $arm $(date +%T)"
  $PY -W ignore -u tools/train_rfdetr.py "${COMMON[@]}" --out-dir $OUT "${EXTRA[@]}" >> $OUT/train.log 2>&1
  echo "train $arm exit $? $(date +%T)"
done
for arm in J F; do
  echo "eval $arm $(date +%T)"
  $PY -W ignore -u tools/eval_picto.py --run runs/exp6_picto_$arm --arm $arm --dataset-dir $D > runs/exp6_picto_$arm/eval.log 2>&1
  echo "eval $arm exit $?"
done
$PY tools/compare_picto.py --j runs/exp6_picto_J --f runs/exp6_picto_F --out runs/exp6_picto_F/compare_vs_J.json \
  > runs/exp6_picto_F/compare_vs_J.log 2>&1
echo "compare exit $? $(date +%T)"
echo "ALL DONE"
