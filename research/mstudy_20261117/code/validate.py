import sys, json, glob
import numpy as np, pandas as pd
regime, day = sys.argv[1], sys.argv[2]
d = f'/mnt/scratch/mstudy/data/events/{regime}/{day}'
meta = json.load(open(f'{d}/_meta.json'))
meta = sorted(meta, key=lambda m: -m['n'])
lim = int(sys.argv[3]) if len(sys.argv) > 3 else 6
for m in meta[:lim]:
    df = pd.read_parquet(f"{d}/{m['ticker']}.parquet")
    v = df[df.valid == 1]
    spr = (v.ap0 - v.bp0) / m['tick']
    tr = df[df.ty == 1].copy()
    # book state BEFORE the trade = previous row's state
    prev = df.shift(1)
    tb = prev.loc[tr.index]
    mid = (tb.ap0 + tb.bp0) / 2
    sign_book = np.sign(tr.px - mid)
    agree = (sign_book[sign_book != 0] == tr.sd[sign_book != 0]).mean()
    crossed = ((df.ap0 <= df.bp0) & df.ap0.notna() & df.bp0.notna()).mean()
    print(f"{m['ticker'][:34]:34s} {m['fine']:12s} n={m['n']:>9,} tr={m['n_trades']:>7,} valid={m['valid_frac']:.3f} "
          f"crossed={crossed:.4f} spr_ticks med={spr.median():.1f} p90={spr.quantile(.9):.1f} tick={m['tick']:.4g} "
          f"tradeSign~book={agree:.3f} hrs={(m['t1']-m['t0'])/3.6e12:.1f}")
print(len(meta), 'markets;', pd.Series([m['fine'] for m in meta]).value_counts().to_dict())
