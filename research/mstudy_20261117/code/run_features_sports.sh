#!/bin/bash
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
for d in 2026-09-19 2026-09-20 2026-09-21; do
  while [ ! -f /mnt/scratch/mstudy/data/events/football/$d/_meta.json ]; do sleep 20; done
  $PY features.py football $d 8 >> /mnt/scratch/mstudy/logs/features.out 2>&1
done
echo FEATDONE >> /mnt/scratch/mstudy/logs/features.log
