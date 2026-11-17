#!/bin/bash
set -euo pipefail
RUN=/var/lib/kalshi-study/reproduction-btc-20261117
exec >>"$RUN/parallel-middle-resume.log" 2>&1
export PYTHONPATH=/opt/kalshi KALSHI_BLOB_ACCOUNT=kalshiobfdfaed53 KALSHI_BLOB_CONTAINER=tape KALSHI_CACHE=/mnt/scratch/reproduction/cache SCALE_FREE_STRICT=1 PYTHONWARNINGS=ignore
while ! grep -Eq '^(COMPLETE|FAIL)' "$RUN/provisional-status.txt" 2>/dev/null; do sleep 30; done
# Interrupted workers may leave partial files only in our isolated secondary
# directory. Completed files were linked atomically to the primary directory.
/opt/kalshi/venv/bin/python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
root=Path('/mnt/scratch/reproduction/accelerated-middle')
for p in root.glob('panel_KXBTC15M-*.parquet'):
 try:pq.ParquetFile(p)
 except Exception:
  print('Removing interrupted secondary output',p,flush=True);p.unlink()
PY
exec /opt/kalshi/venv/bin/python -u "$RUN/parallel_middle_replay.py"
