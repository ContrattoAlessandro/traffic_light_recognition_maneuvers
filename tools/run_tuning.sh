#!/usr/bin/env bash
# Short tuning runs: train on the 90% fit split, score on the 10% tune split (never on val).
# Each run resumes from its last checkpoint, so re-running the script continues where it stopped.
set -u
cd "$(dirname "$0")/.."
T=data/labels/tune
R=data/OpenLane-V2
EPOCHS=${EPOCHS:-4}
COMMON="--train-set $T/subset_A_train_fit.jsonl $R/subset_A --train-set $T/subset_B_train_fit.jsonl $R/subset_B
  --val-set A_tune $T/subset_A_train_tune.jsonl $R/subset_A --val-set B_tune $T/subset_B_train_tune.jsonl $R/subset_B
  --epochs $EPOCHS --batch-size 4 --workers 6 --amp --log-every 200 --resume"

run() {  # name, extra args
  local name=$1; shift
  if [ -f "runs/tune/$name/final_metrics.json" ]; then echo "[$name] already finished"; return; fi
  echo "[$name] start $(date +%T)"
  ./.venv/Scripts/python -W ignore -u -m dtsu.train $COMMON --out-dir "runs/tune/$name" "$@" \
    > "runs/tune/$name.log" 2>&1
  echo "[$name] exit $? $(date +%T)"
}

mkdir -p runs/tune
run t1_default --lam 1 1 1 --clip 0.1
run t2_clip1 --lam 1 1 1 --clip 1.0
run t3_det0.2 --lam 1 1 0.2 --clip 0.1
run t4_det0.2_norm --lam 1 1 0.2 --clip 0.1  # + per-stage LayerNorm, fp32 detector
echo "ALL DONE $(date +%T)"
