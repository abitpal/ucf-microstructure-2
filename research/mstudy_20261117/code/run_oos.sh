#!/bin/bash
# out-of-sample: extract + fair features for unseen days (crypto & perps streams in parallel), eval as days land
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
L=/mnt/scratch/mstudy/logs; F=/mnt/scratch/mstudy/data/feat
DAYS="2026-11-17 2026-09-24 2026-09-25 2026-09-26 2026-09-27 2026-09-28 2026-09-29 2026-09-30 2026-10-01 2026-10-02 2026-10-03 2026-10-04 2026-10-05 2026-10-06 2026-10-07 2026-10-08 2026-10-09 2026-09-17 2026-09-16 2026-09-15"
stream() { r=$1; for d in $DAYS; do
    $PY extract.py $r $d 10 >> $L/oos_extract.out 2>&1
    [ -f /mnt/scratch/mstudy/data/events/$r/$d/_meta.json ] && $PY features.py $r $d 10 fair >> $L/oos_features.out 2>&1
    touch $F/oos_ready_${r}_$d; echo "[$(date +%T)] ready $r $d" >> $L/oos.log
  done; }
echo "[$(date +%T)] START oos" >> $L/oos.log
stream crypto & stream perps &
NTHREADS=10 $PY eval_fair.py BTC15M 2026-09-22 2026-09-23 >> $L/oos_eval.out 2>&1
NTHREADS=10 $PY eval_fair.py BTCPERP 2026-09-22 2026-09-23 >> $L/oos_eval.out 2>&1
for d in $DAYS; do
  while [ ! -f $F/oos_ready_crypto_$d ] || [ ! -f $F/oos_ready_perps_$d ]; do sleep 30; done
  NTHREADS=10 $PY eval_fair.py BTC15M $d >> $L/oos_eval.out 2>&1
  NTHREADS=10 $PY eval_fair.py BTCPERP $d >> $L/oos_eval.out 2>&1
done
wait
echo "[$(date +%T)] END oos" >> $L/oos.log
