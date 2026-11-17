#!/bin/bash
# orchestrates all training/analysis after features are ready
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
L=/mnt/scratch/mstudy/logs
F=/mnt/scratch/mstudy/data/feat
step(){ echo "[$(date -u +%H:%M:%S)] START $1" >> $L/pipeline.log; shift; "$@" >> $L/pipeline.out 2>&1; rc=$?; echo "[$(date -u +%H:%M:%S)] END rc=$rc $*" >> $L/pipeline.log; $PY push_results.py >> $L/pipeline.out 2>&1; }
# --- chain A: fair models need only crypto features (all 6 days)
(
  for d in 2026-09-18 2026-09-19 2026-09-20 2026-09-21 2026-09-22 2026-09-23; do
    while [ ! -f $F/done_crypto_${d}_all ]; do sleep 30; done
  done
  NTHREADS=6 step fair_baselines $PY fair_models.py BTC15M baselines
  NTHREADS=6 BUDGET=2400 step fair_transformer $PY fair_models.py BTC15M transformer
  NTHREADS=6 BUDGET=1800 step fair_gru $PY fair_models.py BTC15M gru
  echo CHAIN_A_DONE >> $L/pipeline.log
) &
# --- chain B: analysis needs sports (Sat+Sun at minimum) and perps
while [ ! -f $F/done_football_2026-09-19_all ] || [ ! -f $F/done_football_2026-09-20_all ] || [ ! -f $F/done_perps_2026-09-23_all ]; do sleep 30; done
step classical $PY analysis_classical.py
NTHREADS=6 step classify $PY classify.py
NTHREADS=6 AE_BUDGET=900 CLF_BUDGET=600 step ae $PY ae.py
NTHREADS=6 MT_BUDGET=360 step trade $PY trade_model.py
NTHREADS=6 AE_BUDGET=600 STRICT_AE=1 step ae_strict $PY ae.py
wait
echo PIPELINE_DONE >> $L/pipeline.log
$PY push_results.py >> $L/pipeline.out 2>&1
