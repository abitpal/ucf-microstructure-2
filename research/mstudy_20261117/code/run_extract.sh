#!/bin/bash
# sequential extraction queue
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
rm -rf /mnt/scratch/mstudy/data/events/perps/2026-09-19
for d in 2026-09-19 2026-09-20 2026-09-21 2026-09-22 2026-09-18 2026-09-23; do
  $PY extract.py perps $d 11
  $PY extract.py crypto $d 11
done
for d in 2026-09-19 2026-09-20 2026-09-21; do
  $PY extract.py football $d 11
done
echo ALLDONE >> /mnt/scratch/mstudy/logs/extract.log
