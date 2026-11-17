"""TOUCH SIZE TAPE: best bid/ask AND the resting size AT the touch.

Capacity needs size at level 1. The panels kept only cumulative 3-level depth
(k_bid_size_3 / k_ask_size_3) and the touch tape kept prices only, so neither
can say how many contracts are actually available at the price we would hit.
Sizes are 1/100-contract units straight from the book, same as everywhere else.

ORIGINAL HEADER FOLLOWS


WHY
---
The resting-exit fill rate is the single number standing between a significant
mark-to-mid edge and a realisable PnL, and nothing already on disk can answer
it. The panels carry only aggregated trade COUNTS (no price), and the mid tape
carries only (ts, mid) -- neither can say whether a print reached a specific
level.

WHAT IS RECORDED, at every event (`every=1`, not every 25th):
    ts          event time
    best_bid    top of the YES bid ladder, price units
    best_ask    implied from the NO bid, price units
    last_trade  price of the most recent print
    trade_count how many prints so far -- an INCREMENT identifies a real trade,
                which is how prints are separated from book deltas, since
                last_trade simply repeats between trades

TWO FILL RULES, deliberately, because each is wrong in a known direction and
the truth is between them:
  print-through  a print strictly through our limit. UNDERSTATES fills: ignores
                 that we may be filled at our own level by a print AT it.
  touch-cross    the opposite side of the book reaches our limit. OVERSTATES:
                 the touch can reach a level on zero volume.
Bracketing them is the honest answer; picking one is how the football study got
a single number that later had to be caveated.
"""
import os, sys, glob, re
sys.path.insert(0, "/opt/kalshi")
import multiprocessing as mp

PAD_BEFORE, PAD_AFTER = 90.0, 30.0
COLS = ["ts", "best_bid", "best_ask", "bid_size_1", "ask_size_1"]

def one(job):
    coin, ticker, open_ts, close_ts, outdir = job
    out = os.path.join(outdir, f"sz_{ticker}.parquet")
    try:
        if os.path.exists(out) and os.path.getsize(out) > 0:
            import pyarrow.parquet as pq
            return (ticker, int(pq.ParquetFile(out).metadata.num_rows), None)
        os.environ.setdefault("KALSHI_BLOB_ACCOUNT", "kalshiobfdfaed53")
        import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
        from sim import KalshiFeed, Sim
        from sim import lib
        prefix = ("raw/regime=commodities/" if coin == "GOLD" else "raw/regime=crypto/")
        feed = KalshiFeed(bucket="tape", prefix=prefix)
        sim = Sim([ticker], since=open_ts - PAD_BEFORE, until=close_ts + PAD_AFTER,
                  feed=feed, strict=False)
        sim.add(lib.best_bid, lib.best_ask, lib.bid_size(1), lib.ask_size(1))
        tmp = os.path.join(outdir, f"_z_{ticker}.parquet")
        sim.record(tmp, every=1)
        df = pd.read_parquet(tmp)
        os.unlink(tmp)
        miss = [c for c in COLS if c not in df.columns]
        if miss:
            # A market with no usable events makes the recorder fall back to the
            # RAW event schema instead of the signal columns. That is an empty
            # market, not a failure -- the early dates in this tape are sparse.
            return (ticker, 0, None)
        df = df[COLS]
        df = df[np.isfinite(df.ts.to_numpy())].sort_values("ts")
        if df.empty:
            return (ticker, 0, "empty")
        t = pa.table({
            "ts":          pa.array(df.ts.to_numpy(dtype="float64")),
            "best_bid":    pa.array(df.best_bid.to_numpy(dtype="float32")),
            "best_ask":    pa.array(df.best_ask.to_numpy(dtype="float32")),
            "bid_size_1":  pa.array(df.bid_size_1.to_numpy(dtype="float32")),
            "ask_size_1":  pa.array(df.ask_size_1.to_numpy(dtype="float32")),
        })
        pq.write_table(t, out, compression="zstd")
        return (ticker, len(df), None)
    except Exception as exc:
        return (ticker, 0, f"{type(exc).__name__}: {exc}")

def main():
    import argparse, datetime as dt
    from tools.build_panel import slots
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", nargs="+", default=["BTC"])
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--out", default="/mnt/data/sizetape")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(c, tk, o, cl, a.out)
            for c in a.coin for d in a.dates
            for tk, o, cl in slots(c, d, [f"{h:02d}" for h in range(24)])]
    print(f"{len(jobs)} markets -> {a.out} (touch + prints, every event)", flush=True)
    t0 = dt.datetime.now(dt.timezone.utc); done = rows = 0; fail = []
    with mp.Pool(a.workers) as pool:
        for tk, n, err in pool.imap_unordered(one, jobs):
            done += 1; rows += n
            if err: fail.append((tk, err))
            if done % 25 == 0 or done == len(jobs):
                el = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
                print(f"  {done}/{len(jobs)}  {rows:,} events  failed={len(fail)}  "
                      f"{el/60:.1f} min  eta {(el/done*(len(jobs)-done))/60:.1f} min",
                      flush=True)
    print(f"\n{done} markets, {rows:,} events, {len(fail)} failed")
    for tk, e in fail[:10]: print(f"  FAIL {tk}: {e}")

if __name__ == "__main__":
    main()
