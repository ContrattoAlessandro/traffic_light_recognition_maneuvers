#!/usr/bin/env bash
# Amendment 1 (REPLICATION_PLAN.md): J vs J+P on the per-direction matrix, then on ATLAS / LISA arrows.
# Needs runs/pictogram_p/best.pth (tools/train_pictogram.py). Cached detections make re-runs cheap.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="./.venv/Scripts/python -W ignore -u"
echo "matrix $(date +%T)"
$PY tools/eval_hybrid.py > runs/hybrid_matrix.log 2>&1
echo "matrix exit $? $(date +%T)"
$PY tools/eval_hybrid_arrows.py > runs/hybrid_arrows.log 2>&1
echo "arrows exit $? $(date +%T)"
echo "ALL DONE $(date +%T)"
