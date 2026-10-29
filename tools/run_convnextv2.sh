#!/usr/bin/env bash
# Backbone ablation: same run as tools/run_full.sh with only the backbone swapped, Swin-T ->
# ConvNeXtV2-T (FCMAE pretrain + ImageNet-1k fine-tune, same size and pretraining data as Swin-T
# ms_in1k). Same 24-epoch cosine schedule, stopped after epoch 14 like runs/swin_t_AB, so the two
# are compared at the same point. Re-running the script resumes from runs/convnextv2_t_AB/last.pt.
set -u
cd "$(dirname "$0")/.."
L=data/labels
R=data/OpenLane-V2
mkdir -p runs/convnextv2_t_AB
./.venv/Scripts/python -W ignore -u -m dtsu.train \
  --train-set $L/subset_A_train.jsonl $R/subset_A --train-set $L/subset_B_train.jsonl $R/subset_B \
  --val-set A $L/subset_A_val.jsonl $R/subset_A --val-set B $L/subset_B_val.jsonl $R/subset_B \
  --out-dir runs/convnextv2_t_AB --epochs 24 --last-epoch 14 --batch-size 4 --workers 6 --amp --log-every 200 \
  --resume --lam 1 1 0.2 --clip 0.1 --backbone convnextv2_tiny.fcmae_ft_in1k >> runs/convnextv2_t_AB/train.log 2>&1
echo "exit $?"
