#!/bin/bash
# Build a study frame from panel_rich, in (coin,date) chunks, ON SCRATCH.
#
# make_study_frame reads a whole panel into pandas, so its peak scales with the
# input: it was OOM-killed at 24.8GB on 578 markets and panel_rich holds 1,205.
# One chunk per date is ~96 markets, the size that has always worked, and the
# chunk frames are then stream-concatenated.
#
# Output goes to /mnt/scratch (1.7T, striped local NVMe) rather than the 247G OS
# disk, which hit 100% full and killed the previous merge with ENOSPC after the
# 4.7-hour replay had already succeeded.
set -u
P=/var/lib/kalshi-study/panel_rich
W=/mnt/scratch/frames/richchunks
OUT=/var/lib/kalshi-study/frame_rich.parquet
PY=/opt/kalshi/venv/bin/python
mkdir -p "$W"
built=0; skipped=0
for d in $(ls "$P" | sed -n "s/^panel_KXBTC15M-\([0-9A-Z]\{7\}\)[0-9]\{4\}-.*/\1/p" | sort -u); do
  out="$W/frame_BTC_${d}.parquet"
  [ -s "$out" ] && { skipped=$((skipped+1)); continue; }
  n=$(ls "$P"/panel_KXBTC15M-${d}*.parquet 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && continue
  tmp="$W/_m_${d}.parquet"
  $PY - "$tmp" "$P" "$d" <<'PYEOF' || { echo "MERGE FAIL $d"; continue; }
import sys, glob
import pyarrow as pa, pyarrow.parquet as pq
tmp, P, d = sys.argv[1:4]
paths = sorted(glob.glob(f"{P}/panel_KXBTC15M-{d}*.parquet"))
target = pa.unify_schemas([pq.ParquetFile(p).schema_arrow for p in paths])
w = None
try:
    for p in paths:
        t = pq.read_table(p)
        for f in target:
            if f.name not in t.schema.names:
                t = t.append_column(f, pa.nulls(t.num_rows, type=f.type))
        t = t.select([f.name for f in target]).cast(target)
        if w is None: w = pq.ParquetWriter(tmp, target, compression="zstd")
        w.write_table(t); del t
finally:
    if w is not None: w.close()
PYEOF
  if SCALE_FREE_STRICT=1 PYTHONPATH=/opt/kalshi $PY -u -m tools.make_study_frame \
       --panel "$tmp" --out "$out" >>"$W/_build.log" 2>&1; then
    built=$((built+1)); echo "built $d ($n markets)"
  else
    echo "FRAME FAIL $d -- see $W/_build.log"; tail -3 "$W/_build.log"; rm -f "$out"
  fi
  rm -f "$tmp"
done
echo "chunks built=$built skipped=$skipped total=$(ls $W/frame_BTC_*.parquet 2>/dev/null|grep -v _exec|wc -l)"
$PY -u /var/lib/kalshi-study/scripts/cf.py "$W" "$OUT"
