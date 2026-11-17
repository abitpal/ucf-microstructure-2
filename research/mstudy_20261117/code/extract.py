"""Stage A: stream raw tape for (regime, day) -> per-market event arrays -> L2 book rebuild -> parquet.

usage: python extract.py REGIME DAY [workers]
  REGIME in crypto|football|perps ; DAY like 2026-09-19

Output: DATA/events/{regime}/{day}/{ticker}.parquet with one row per book-changing event or trade, and the
book state (L1-L5 price/size both sides) *after* the event.
"""
import sys, os, gzip, zlib, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import orjson
import multiprocessing as mp
from common import *

PREFIX = {'crypto': 'raw/regime=crypto', 'football': 'raw/regime=football', 'perps': 'perps/regime=perps'}
T_DELTA, T_TRADE, T_SNAP, T_SNAPSTART = 0, 1, 2, 3
MIN_EVENTS = 1500


def parse_file(args):
    name, regime = args
    raw = get_blob(name)
    try:
        data = zlib.decompress(raw, 31)
    except zlib.error:
        # truncated tape part: keep every complete line we can recover
        d = zlib.decompressobj(31)
        try:
            data = d.decompress(raw)
        except zlib.error:
            data = b''
        data = data[:data.rfind(b'\n') + 1]
        log(f'  WARN truncated gzip {name}: recovered {len(data)} bytes', 'extract.log')
    tk, rn, ty, sd, px, qx, ts = [], [], [], [], [], [], []
    tmap = {}
    keep_cache = {}
    is_perp = regime == 'perps'
    for line in data.split(b'\n'):
        if not line:
            continue
        # cheap prefilter on type
        if b'"orderbook_delta"' in line:
            t = T_DELTA
        elif b'"type":"trade"' in line:
            t = T_TRADE
        elif b'"orderbook_snapshot"' in line:
            t = T_SNAP
        else:
            continue
        try:
            o = orjson.loads(line)
            m = o['msg']['msg']
        except Exception:
            continue
        tick = m.get('market_ticker')
        if tick is None:
            continue
        k = keep_cache.get(tick)
        if k is None:
            k = keep_series(regime, tick.split('-')[0])
            keep_cache[tick] = k
        if not k:
            continue
        code = tmap.setdefault(tick, len(tmap))
        r = o['recv_ns']
        tsm = m.get('ts_ms') or (r // 1000000)
        if t == T_DELTA:
            if is_perp:
                p = float(m['price']); q = float(m['delta']); s = 0 if m['side'] == 'bid' else 1
            else:
                p = float(m['price_dollars']); q = float(m['delta_fp']); s = 0 if m['side'] == 'yes' else 1
            tk.append(code); rn.append(r); ty.append(T_DELTA); sd.append(s); px.append(p); qx.append(q); ts.append(tsm)
        elif t == T_TRADE:
            if is_perp:
                p = float(m['price']); q = float(m['count']); s = 1 if m['taker_side'] == 'bid' else -1
            else:
                p = float(m['yes_price_dollars']); q = float(m['count_fp']); s = 1 if m['taker_side'] == 'yes' else -1
            tk.append(code); rn.append(r); ty.append(T_TRADE); sd.append(s); px.append(p); qx.append(q); ts.append(tsm)
        else:
            tk.append(code); rn.append(r); ty.append(T_SNAPSTART); sd.append(0); px.append(0.0); qx.append(0.0); ts.append(tsm)
            if is_perp:
                sides = (('bid', 0), ('ask', 1))
            else:
                sides = (('yes_dollars_fp', 0), ('no_dollars_fp', 1))
            for key, s in sides:
                for lv in (m.get(key) or ()):
                    tk.append(code); rn.append(r); ty.append(T_SNAP); sd.append(s)
                    px.append(float(lv[0])); qx.append(float(lv[1])); ts.append(tsm)
    names = np.array(list(tmap.keys()), dtype=object)
    return (names, np.array(tk, np.int32), np.array(rn, np.int64), np.array(ty, np.int8),
            np.array(sd, np.int8), np.array(px, np.float64), np.array(qx, np.float64), np.array(ts, np.int64))


# --------------------------------------------------------------------------- book rebuild
import numba as nb

K = 5


@nb.njit(cache=True)
def rebuild(ty, sd, gi, qx, G, scan_lim):
    """gi: grid index of event price on the unified YES/bid-ask grid. ask side for binaries already flipped.
    returns arrays of book state after each event."""
    n = len(ty)
    B = np.zeros(G, np.float64)
    A = np.zeros(G, np.float64)
    bb = -1; ba = G
    bpx = np.full((n, K), -1, np.int32); bsz = np.zeros((n, K), np.float32)
    apx = np.full((n, K), -1, np.int32); asz = np.zeros((n, K), np.float32)
    lvl = np.zeros(n, np.int32)  # event price distance from same-side touch before event (>0 deeper)
    valid = np.zeros(n, np.int8)
    have = False
    for i in range(n):
        t = ty[i]
        if t == 3:  # snapshot start -> clear
            B[:] = 0.0; A[:] = 0.0; bb = -1; ba = G; have = True
        elif t == 2 or t == 0:
            g = gi[i]
            if g < 0 or g >= G:
                pass
            elif sd[i] == 0:
                lvl[i] = (bb - g) if bb >= 0 else 0
                if t == 2:
                    B[g] = qx[i]
                else:
                    B[g] += qx[i]
                if B[g] < 1e-9:
                    B[g] = 0.0
                if B[g] > 0 and g > bb:
                    bb = g
                if B[g] == 0 and g == bb:
                    j = bb - 1; c = 0
                    while j >= 0 and B[j] == 0 and c < scan_lim * 50:
                        j -= 1; c += 1
                    bb = j if (j >= 0 and B[j] > 0) else -1
            else:
                lvl[i] = (g - ba) if ba < G else 0
                if t == 2:
                    A[g] = qx[i]
                else:
                    A[g] += qx[i]
                if A[g] < 1e-9:
                    A[g] = 0.0
                if A[g] > 0 and g < ba:
                    ba = g
                if A[g] == 0 and g == ba:
                    j = ba + 1; c = 0
                    while j < G and A[j] == 0 and c < scan_lim * 50:
                        j += 1; c += 1
                    ba = j if (j < G and A[j] > 0) else G
        # record state
        if have:
            c = 0; j = bb; s = 0
            while j >= 0 and c < K and s < scan_lim:
                if B[j] > 0:
                    bpx[i, c] = j; bsz[i, c] = B[j]; c += 1
                j -= 1; s += 1
            c = 0; j = ba; s = 0
            while j < G and c < K and s < scan_lim:
                if A[j] > 0:
                    apx[i, c] = j; asz[i, c] = A[j]; c += 1
                j += 1; s += 1
            valid[i] = 1 if (bb >= 0 and ba < G and bb < ba) else 0
    return bpx, bsz, apx, asz, lvl, valid


def build_market(name, regime, tk_rn, tk_ty, tk_sd, tk_px, tk_qx, tk_ts, outdir):
    import pandas as pd
    is_perp = regime == 'perps'
    if is_perp:
        pos = tk_px[(tk_ty == T_DELTA) | (tk_ty == T_SNAP) | (tk_ty == T_TRADE)]
        pos = pos[pos > 0]
        if len(pos) < 100:
            return None
        ints = np.round(pos * 1e4).astype(np.int64)
        lo = ints.min()
        # tick = gcd of distinct differences (sampled)
        u = np.unique(ints)
        d = np.diff(u)
        d = d[d > 0]
        tick = int(np.gcd.reduce(d[:200000])) if len(d) else 1
        tick = max(tick, 1)
        # restrict grid to central 99.9% range +- margin to bound memory
        q0, q1 = np.quantile(ints, [0.0005, 0.9995])
        lo = int(q0 - 0.02 * (q1 - q0)); hi = int(q1 + 0.02 * (q1 - q0))
        lo = lo - (lo % tick)
        G = int((hi - lo) // tick) + 2
        if G > 3_000_000:
            tick = int(tick * np.ceil(G / 3_000_000)); G = int((hi - lo) // tick) + 2
        gi = ((np.round(tk_px * 1e4).astype(np.int64) - lo) // tick).astype(np.int64)
        gi[(tk_ty == T_SNAPSTART)] = 0
        scale = tick / 1e4; offset = lo / 1e4
        econ_tick = scale
    else:
        # binary: grid in 0.001 $ ; NO-side prices flipped to YES ask
        G = 1001
        g = np.round(tk_px * 1000).astype(np.int64)
        flip = ((tk_ty == T_DELTA) | (tk_ty == T_SNAP)) & (tk_sd == 1)
        g[flip] = 1000 - g[flip]
        gi = g; scale = 0.001; offset = 0.0
        econ_tick = 0.01
    bpx, bsz, apx, asz, lvl, valid = rebuild(tk_ty, tk_sd, gi, tk_qx, G, 2000 if is_perp else 1001)
    keep = (tk_ty != T_SNAP)  # drop individual snapshot level rows; keep start marker (state after full snap = last level row)
    # the state after the snapshot is on the last level row; move it onto the start marker row index:
    # simplest: keep the last row of each snapshot block instead of the marker.
    is_snap = tk_ty == T_SNAP
    nxt_not_snap = np.ones(len(tk_ty), bool)
    nxt_not_snap[:-1] = ~is_snap[1:]
    keep = (tk_ty == T_DELTA) | (tk_ty == T_TRADE) | (is_snap & nxt_not_snap)
    # snapshot-start markers for empty books
    keep |= (tk_ty == T_SNAPSTART) & nxt_not_snap
    idx = np.nonzero(keep)[0]
    tyk = tk_ty[idx].copy(); tyk[(tyk == T_SNAP) | (tyk == T_SNAPSTART)] = T_SNAP
    df = pd.DataFrame({
        'rn': tk_rn[idx], 'ts': tk_ts[idx], 'ty': tyk, 'sd': tk_sd[idx], 'px': tk_px[idx].astype(np.float64),
        'qx': tk_qx[idx].astype(np.float32), 'lvl': lvl[idx], 'valid': valid[idx],
    })
    for k in range(K):
        bp = bpx[idx, k].astype(np.float64); ap = apx[idx, k].astype(np.float64)
        df[f'bp{k}'] = np.where(bp >= 0, offset + bp * scale, np.nan)
        df[f'ap{k}'] = np.where(ap >= 0, offset + ap * scale, np.nan)
        df[f'bs{k}'] = bsz[idx, k]; df[f'as{k}'] = asz[idx, k]
    nev = int(((df.ty == 0) | (df.ty == 1)).sum())
    if nev < MIN_EVENTS:
        return None
    coarse, fine = classify(name)
    meta = dict(ticker=name, regime=regime, coarse=coarse, fine=fine, tick=econ_tick, grid=scale, n=len(df),
                n_trades=int((df.ty == 1).sum()), t0=int(df.rn.iloc[0]), t1=int(df.rn.iloc[-1]),
                valid_frac=float(df.valid.mean()))
    safe = name.replace('/', '_')
    df.to_parquet(f'{outdir}/{safe}.parquet', index=False)
    return meta


def _build_one(args):
    name, regime, slices, outdir = args
    try:
        arrs = [np.concatenate([s[j] for s in slices]) for j in range(6)]
        o = np.argsort(arrs[0], kind='stable')
        arrs = [a[o] for a in arrs]
        return build_market(name, regime, *arrs, outdir)
    except Exception as e:
        import traceback
        return {'ticker': name, 'error': traceback.format_exc()[-600:]}


def main(regime, day, workers=10):
    outdir = f'{DATA}/events/{regime}/{day}'
    tmp = f'{DATA}/tmp/{regime}/{day}'
    os.makedirs(outdir, exist_ok=True); os.makedirs(tmp, exist_ok=True)
    if os.path.exists(f'{outdir}/_meta.json'):
        log(f'skip {regime} {day}: done', 'extract.log'); return
    files = sorted(n for n, s in list_blobs(f'{PREFIX[regime]}/dt={day}/') if n.endswith('.jsonl.gz'))
    hours = sorted(set(f.split('/hh=')[1][:2] for f in files))
    log(f'{regime} {day}: {len(files)} files, {len(hours)} hours', 'extract.log')
    t0 = time.time()
    # ---- pass 1: per hour, parse + sort by (ticker, recv) and dump to npy
    counts = {}
    with mp.Pool(workers) as pool:
        for h in hours:
            if os.path.exists(f'{tmp}/{h}_done'):
                for nm, c in orjson.loads(open(f'{tmp}/{h}_names.json','rb').read()).items():
                    counts[nm] = counts.get(nm, 0) + c[2]
                continue
            hf = [f for f in files if f'/hh={h}/' in f]
            parts = list(pool.imap_unordered(parse_file, [(f, regime) for f in hf], chunksize=1))
            allnames = {}
            for p in parts:
                for nm in p[0]:
                    allnames.setdefault(nm, len(allnames))
            cols = [[] for _ in range(7)]
            for p in parts:
                names = p[0]
                remap = np.array([allnames[n] for n in names], np.int32) if len(names) else np.zeros(0, np.int32)
                cols[0].append(remap[p[1]] if len(p[1]) else p[1])
                for j in range(2, 8):
                    cols[j - 1].append(p[j])
            del parts
            TK = np.concatenate(cols[0]); RN, TY, SD, PX, QX, TS = [np.concatenate(c) for c in cols[1:]]
            del cols
            o = np.lexsort((np.arange(len(TK)), RN, TK))
            arrs = [a[o] for a in (RN, TY, SD, PX, QX, TS)]
            TK = TK[o]
            bounds = np.searchsorted(TK, np.arange(len(allnames) + 1))
            for j, nmj in enumerate(('rn', 'ty', 'sd', 'px', 'qx', 'ts')):
                np.save(f'{tmp}/{h}_{nmj}.npy', arrs[j])
            info = {}
            for nm, c in allnames.items():
                a, b = int(bounds[c]), int(bounds[c + 1])
                ne = int(((arrs[1][a:b] == 0) | (arrs[1][a:b] == 1)).sum())
                info[nm] = (a, b, ne)
                counts[nm] = counts.get(nm, 0) + ne
            open(f'{tmp}/{h}_names.json', 'wb').write(orjson.dumps(info))
            open(f'{tmp}/{h}_done', 'w').write('1')
            log(f'  hour {h}: {len(TK):,} rows {len(allnames)} tickers  {time.time()-t0:.0f}s', 'extract.log')
            del arrs, TK
    # ---- pass 2: per market, gather slices across hours and rebuild
    keep = [nm for nm, c in counts.items() if c >= MIN_EVENTS]
    log(f'  building {len(keep)} markets (of {len(counts)})', 'extract.log')
    hinfo = {h: orjson.loads(open(f'{tmp}/{h}_names.json','rb').read()) for h in hours}
    mm = {h: [np.load(f'{tmp}/{h}_{nmj}.npy', mmap_mode='r') for nmj in ('rn', 'ty', 'sd', 'px', 'qx', 'ts')] for h in hours}

    def jobs():
        for nm in keep:
            sl = []
            for h in hours:
                if nm in hinfo[h]:
                    a, b, _ = hinfo[h][nm]
                    sl.append([np.array(x[a:b]) for x in mm[h]])
            yield (nm, regime, sl, outdir)
    metas = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(_build_one, jobs(), chunksize=1):
            if r is not None:
                metas.append(r)
    errs = [m for m in metas if 'error' in m]
    if errs:
        log(f'  {len(errs)} build errors, first: {errs[0]}', 'extract.log')
    metas = [m for m in metas if 'error' not in m]
    with open(f'{outdir}/_meta.json', 'wb') as f:
        f.write(orjson.dumps(metas))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    log(f'DONE {regime} {day}: {len(metas)} markets, {time.time()-t0:.0f}s', 'extract.log')


if __name__ == '__main__':
    regime, day = sys.argv[1], sys.argv[2]
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    main(regime, day, w)
