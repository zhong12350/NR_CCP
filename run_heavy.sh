#!/bin/bash
set -e
cd ~/Desktop/NR_CCP
source .venv/bin/activate
export MPLBACKEND=Agg
mkdir -p outputs/logs

echo "===== [1/2] delta_sweep START $(date) ====="
python3 main.py delta_sweep 2>&1 | tee outputs/logs/delta_sweep.log

echo "===== [2/2] batch START $(date) ====="
python3 main.py batch 2>&1 | tee outputs/logs/batch.log

echo "===== ALL HEAVY JOBS DONE $(date) ====="
