#!/usr/bin/env bash
# RF-DETR tuning on the 90% fit split, scored on the 10% tune split (never on val), like run_tuning.sh.
# Stage 1: every config trains 2 epochs of a 4-epoch run. Stage 2: the config with the best tune TSS
# after stage 1 resumes to 4 epochs. RF-DETR's lr is constant (step drop at epoch 100), so stopping
# after 2 epochs and resuming gives the same run as 4 straight epochs.
# Re-running the script skips finished stages and resumes from each run's last.ckpt.
set -u
cd "$(dirname "$0")/.."
D=data/rfdetr_coco_fit
OUT=runs/tune_rfdetr
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"
mkdir -p $OUT

train() {  # name, epochs to reach, extra args
  local name=$1 reach=$2; shift 2
  local dir=$OUT/$name
  if [ -f $dir/epoch_$((reach - 1))_ema.pth ]; then echo "[$name] has $reach epochs"; return; fi
  local resume=()
  [ -f $dir/last.ckpt ] && resume=(--resume $dir/last.ckpt)
  echo "[$name] train to epoch $reach $(date +%T)"
  $PY tools/train_rfdetr.py --dataset-dir $D --out-dir $dir --epochs 4 --stop-after $reach \
    --resolution 1024 --batch-size 4 --grad-accum 4 --workers 6 "${resume[@]}" "$@" >> $dir.log 2>&1
  echo "[$name] train exit $? $(date +%T)"
  $PY tools/eval_rfdetr.py --run $dir --tune-only >> $dir.eval.log 2>&1
  echo "[$name] eval exit $? $(date +%T)"
  tail -n 2 $dir.eval.log
}

declare -A ARGS=(
  [r1_m_default]="--variant RFDETRMedium --lr 1e-4 --lr-encoder 1.5e-4"
  [r2_m_lr_half]="--variant RFDETRMedium --lr 5e-5 --lr-encoder 7.5e-5"
  [r3_l_default]="--variant RFDETRLarge --lr 1e-4 --lr-encoder 1.5e-4"
)
NAMES=(r1_m_default r2_m_lr_half r3_l_default)

for n in "${NAMES[@]}"; do train $n 2 ${ARGS[$n]}; done

LEADER=$($PY - "${NAMES[@]}" <<'EOF'
import json, sys
best = {}
for n in sys.argv[1:]:
    log = [json.loads(l) for l in open(f"runs/tune_rfdetr/{n}/tune_log.jsonl")]
    best[n] = next(e["tune_tss"] for e in log if e["epoch"] == 1)
    print(f"  {n}: tune TSS after 2 epochs {best[n]:.4f}", file=sys.stderr)
print(max(best, key=best.get))
EOF
)
echo "leader: $LEADER"
train $LEADER 4 ${ARGS[$LEADER]}
echo "ALL DONE $(date +%T)"
