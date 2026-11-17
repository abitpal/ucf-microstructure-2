"""Stream-concatenate the chunk frames into one, one chunk resident at a time."""
import sys, glob
import pyarrow as pa, pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]
# make_study_frame writes a `*_exec.parquet` execution companion next to each
# frame, and it matches frame_*.parquet too. Globbing naively would concatenate
# 56 frames AND their 56 companions into one file -- different schemas, rows
# silently doubled. Exclude them explicitly.
paths = sorted(p for p in glob.glob(f"{src}/frame_*.parquet")
               if not p.endswith("_exec.parquet"))
print(f"{len(paths)} chunk frames -> {out}", flush=True)
target = pa.unify_schemas([pq.ParquetFile(p).schema_arrow for p in paths])
w = None; total = 0
try:
    for i, p in enumerate(paths, 1):
        t = pq.read_table(p)
        for f in target:
            if f.name not in t.schema.names:
                t = t.append_column(f, pa.nulls(t.num_rows, type=f.type))
        t = t.select([f.name for f in target]).cast(target)
        if w is None: w = pq.ParquetWriter(out, target, compression="zstd")
        w.write_table(t); total += t.num_rows; del t
        if i % 10 == 0: print(f"  {i}/{len(paths)}  {total:,} rows", flush=True)
finally:
    if w is not None: w.close()
print(f"wrote {out}  {total:,} rows  {len(target)} cols", flush=True)
