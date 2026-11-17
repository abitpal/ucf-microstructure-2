#!/bin/bash
cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code
PY=/opt/kalshi/venv/bin/python
JOBS="perps:2026-09-19 crypto:2026-09-19 perps:2026-09-20 crypto:2026-09-20 perps:2026-09-21 crypto:2026-09-21 perps:2026-09-22 crypto:2026-09-22 perps:2026-09-18 crypto:2026-09-18 perps:2026-09-23 crypto:2026-09-23 football:2026-09-19 football:2026-09-20 football:2026-09-21"
for j in $JOBS; do
  r=${j%%:*}; d=${j##*:}
  while [ ! -f /mnt/scratch/mstudy/data/events/$r/$d/_meta.json ]; do sleep 20; done
  $PY features.py $r $d 6 >> /mnt/scratch/mstudy/logs/features.out 2>&1
done
echo FEATDONE >> /mnt/scratch/mstudy/logs/features.log
