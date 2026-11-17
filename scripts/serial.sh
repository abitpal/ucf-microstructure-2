#!/bin/bash
# ONE AT A TIME, FROM A PERSISTENT DIRECTORY.
#
# Two lessons are baked in here. Running five studies at once OOM-killed six
# jobs, including a run that had completed 3h36m of CPU work and wrote nothing
# until the end. And the scripts previously lived in /tmp, which a reboot
# wipes -- so a VM restart would have left this queue pointing at files that no
# longer existed. Everything now lives under /var/lib/kalshi-study/scripts.
set -u
PY=/opt/kalshi/venv/bin/python
D=/var/lib/kalshi-study/scripts
LOG=/var/lib/kalshi-study/SERIAL.log
run() {
  local n="$1"; shift
  echo "=== $(date -u +%H:%M:%S) START $n" | tee -a "$LOG"
  if PYTHONPATH=/opt/kalshi KALSHI_BLOB_ACCOUNT=kalshiobfdfaed53 \
     KALSHI_BLOB_CONTAINER=tape KALSHI_CACHE=/opt/kalshi/data/cache \
     SCALE_FREE_STRICT=1 "$@" >>"/var/lib/kalshi-study/${n}.out" 2>&1; then
    echo "=== $(date -u +%H:%M:%S) OK    $n" | tee -a "$LOG"
  else
    echo "=== $(date -u +%H:%M:%S) FAIL  $n rc=$?" | tee -a "$LOG"
  fi
}
# cheap studies FIRST: they answer open questions in minutes, and putting the
# 5-hour replay first meant a shutdown cost everything behind it.
run volstab  $PY -u $D/vs.py
run xhorizon $PY -u $D/xh.py
run incroute $PY -u $D/ir.py
run seedens2 $PY -u $D/se.py
run featexp  $PY -u $D/fe.py
run richbuild $PY -u -m tools.panel_parallel --coin BTC \
  --dates 2026-09-15 2026-09-16 2026-09-17 2026-09-18 2026-09-19 2026-09-20 \
          2026-09-21 2026-09-22 2026-09-23 2026-09-24 2026-09-25 2026-09-26 \
          2026-09-27 2026-09-28 \
  --tier rich --perp-book --every 25 --workers 12 --resume \
  --out /var/lib/kalshi-study/panel_rich
echo "=== $(date -u +%H:%M:%S) QUEUE DONE" | tee -a "$LOG"
