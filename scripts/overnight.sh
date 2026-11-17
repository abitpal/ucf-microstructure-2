#!/bin/bash
# 1) finish the 219-feature frames, 2) tradeability PnL by horizon.
set -u
PY=/opt/kalshi/venv/bin/python
D=/var/lib/kalshi-study/scripts
LOG=/var/lib/kalshi-study/OVERNIGHT.log
export PYTHONPATH=/opt/kalshi
export KALSHI_BLOB_ACCOUNT=kalshiobfdfaed53 KALSHI_BLOB_CONTAINER=tape
export KALSHI_CACHE=/opt/kalshi/data/cache SCALE_FREE_STRICT=1
run() {
  local n="$1"; shift
  echo "=== $(date -u +%m-%dT%H:%M) START $n" | tee -a "$LOG"
  if "$@" >>"/var/lib/kalshi-study/${n}.out" 2>&1; then
    echo "=== $(date -u +%m-%dT%H:%M) OK    $n" | tee -a "$LOG"
  else echo "=== $(date -u +%m-%dT%H:%M) FAIL  $n" | tee -a "$LOG"; fi
}
mountpoint -q /mnt/data || sudo mount /mnt/data
mountpoint -q /mnt/scratch || sudo /opt/kalshi/infra/provision_scratch.sh
run finalgrid  $PY -u $D/fg.py     # 219 feats incl. 110 perp/cross, all coins
run tradecheck $PY -u $D/tc.py     # trade iff edge > half-spread + taker fee
echo "=== $(date -u +%m-%dT%H:%M) QUEUE DONE" | tee -a "$LOG"
