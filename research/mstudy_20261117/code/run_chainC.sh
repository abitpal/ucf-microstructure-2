#!/bin/bash
# BTC perp fair models after the BTC15M chain finishes
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
L=/mnt/scratch/mstudy/logs
step(){ echo "[$(date -u +%H:%M:%S)] START $1" >> $L/pipeline.log; shift; "$@" >> $L/pipeline.out 2>&1; rc=$?; echo "[$(date -u +%H:%M:%S)] END rc=$rc $*" >> $L/pipeline.log; $PY push_results.py >> $L/pipeline.out 2>&1; }
until grep -q CHAIN_A_DONE $L/pipeline.log 2>/dev/null; do sleep 30; done
NTHREADS=6 step perp_baselines $PY fair_models.py BTCPERP baselines
NTHREADS=6 BUDGET=1500 step perp_transformer $PY fair_models.py BTCPERP transformer
NTHREADS=6 BUDGET=1200 step perp_gru $PY fair_models.py BTCPERP gru
echo CHAIN_C_DONE >> $L/pipeline.log
