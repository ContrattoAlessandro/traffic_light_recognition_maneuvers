#!/usr/bin/env bash
# Final 24-epoch run on A+B train, evaluated on A val, B val and combined.
# Config picked on the tune split (tools/run_tuning.sh): lambda 1/1/0.2, clip 0.1, with per-stage
# Swin LayerNorm and fp32 detector (t4_det0.2_norm; v1 without them is in runs/swin_t_AB_v1_nan).
# Re-running the script resumes from runs/swin_t_AB/last.pt.
set -u
cd "$(dirname "$0")/.."
L=data/labels
R=data/OpenLane-V2
mkdir -p runs/swin_t_AB
./.venv/Scripts/python -W ignore -u -m dtsu.train \
  --train-set $L/subset_A_train.jsonl $R/subset_A --train-set $L/subset_B_train.jsonl $R/subset_B \
  --val-set A $L/subset_A_val.jsonl $R/subset_A --val-set B $L/subset_B_val.jsonl $R/subset_B \
  --out-dir runs/swin_t_AB --epochs 24 --batch-size 4 --workers 6 --amp --log-every 200 --resume \
  --lam 1 1 0.2 --clip 0.1 >> runs/swin_t_AB/train.log 2>&1
echo "exit $?"
